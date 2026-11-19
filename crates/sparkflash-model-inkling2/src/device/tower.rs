// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The tower's order of operations follows Hugging Face Transformers' Inkling-2 implementation
// (src/transformers/models/inkling2/modeling_inkling2.py; Copyright 2026 the HuggingFace Team,
// Apache License 2.0), as the host reference in crate::vision does. No Transformers source is
// copied; see NOTICE.

//! Inkling-2-Small's vision tower on one device: the host reference ([crate::vision]) as launches.
//!
//! The ViT is 1024 wide with 16 heads of 64, DeepSeek's geometry, so its GEMM and attention are
//! DeepSeek's tower kernels (kernels/opt_vision_sm121.cu) unchanged, and the weight arena is the
//! one [sparkflash_kernels::vision_tower::VisionTower] lays out: every tensor at an aligned offset
//! of one allocation, a row zero-padded where the GEMM's K step needs it. The four stages
//! Inkling's ViT does not share - the patch cut, the LayerNorm with a bias, the interleaved axial
//! rope and the layer scale - are kernels/inkling_vision_sm121.cu.
//!
//! One canvas runs as:
//!
//! 1. the class row and the resampled position grid are made on the host (the reference's own
//!    [interpolate_positions]) and copied in as the residual stream's first value;
//! 2. the canvas is cut into merge-block-ordered patches on the device and the patch embedding
//!    is added onto the positions (the GEMM's residual epilogue); then the pre-norm;
//! 3. each block: LayerNorm, the fused q|k|v projection with its bias, the rope, attention, the
//!    output projection, the layer scale's residual; LayerNorm, the fc1 projection with its bias
//!    and the erf GELU, fc2 with its bias, the layer scale's residual;
//! 4. the merger reads each cell's four consecutive patch rows as one 4096-wide row (the class
//!    row left out), through proj_in, the GELU and proj_out, and the rows are widened to f32.
//!
//! Every value between the stages is bf16 and every stage computes in f32, rounding once where
//! the released bf16 ops round. The tower is held to the f32 host reference by a test rather
//! than to bit equality: a bf16 walk and an f32 one cannot agree to the last bit.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Module};
use sparkflash_kernels::images::{Image, inkling_cubin};
use sparkflash_ops::plan::Precision;
use sparkflash_weights::quant::f32_to_bf16_bits;
use sparkflash_weights::{DType, WeightSource};

use super::Bound;

use crate::config::InklingVisionConfig;
use crate::vision::{
    axial_rope_tables, interpolate_positions, mlp_width, patch_values, plan_cells,
};

/// The tower width the kernels are written for (OV_DIM).
const WIDTH: usize = 1024;
/// The heads and head width opt_vision_attn and inkling_vision_rope assume.
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
/// The patch side inkling_vision_patchify cuts (IV_PATCH) and the merge its order assumes.
const PATCH: usize = 20;
const MERGE: usize = 2;
/// A patch's features padded to the GEMM's K step (IV_KP).
const PATCH_STRIDE: usize = 1216;
const QKV: usize = 3 * WIDTH;
const BF16: usize = 2;
const F32: usize = 4;
const ALIGN: usize = 256;
/// opt_vision_gemm's epilogue flags.
const GEMM_RESIDUAL: u64 = 1;
const GEMM_GELU: u64 = 2;
/// The block every tower launch but the GEMM and the attention runs.
const BLOCK: u32 = 256;

/// The tower's kernels, from the Inkling image (which carries DeepSeek's tower kernels too).
#[derive(Clone, Copy)]
struct Kernels {
    patchify: Bound,
    layernorm: Bound,
    rope: Bound,
    scale_add: Bound,
    gemm: Bound,
    attention: Bound,
    widen: Bound,
}

/// One block's tensors in the arena.
#[derive(Clone, Copy)]
struct Block {
    ln_1: [Deviceptr; 2],
    ln_2: [Deviceptr; 2],
    ls_1: Deviceptr,
    ls_2: Deviceptr,
    /// q, k and v's weights as one [3072][1024] matrix, and their biases as one [3072] row.
    qkv: Deviceptr,
    qkv_bias: Deviceptr,
    out: Deviceptr,
    out_bias: Deviceptr,
    fc1: Deviceptr,
    fc1_bias: Deviceptr,
    fc2: Deviceptr,
    fc2_bias: Deviceptr,
}

/// One canvas's scratch, carved from one allocation.
///
/// The patches, and later the merger's output, share the MLP's space; the merger's hidden rows
/// and a block's projection outputs share the q|k|v space. Each is dead by the time the next
/// stage writes over it.
#[derive(Clone, Copy)]
struct Layout {
    canvas: usize,
    x: usize,
    h: usize,
    qkv: usize,
    mlp: usize,
    cos: usize,
    sin: usize,
    bytes: usize,
}

impl Layout {
    fn of(
        rows: usize,
        canvas_bytes: usize,
        rope_positions: usize,
        tokens: usize,
        config: &InklingVisionConfig,
    ) -> Self {
        let mut at = 0;
        let mut take = |bytes: usize| {
            let offset = at;
            at += bytes.max(1).next_multiple_of(ALIGN);
            offset
        };
        let canvas = take(canvas_bytes);
        let x = take(rows * WIDTH * BF16);
        let h = take(rows * WIDTH * BF16);
        let qkv = take((rows * QKV).max(tokens * config.projection_hidden_dim) * BF16);
        let mlp = take(
            (rows * mlp_width(WIDTH))
                .max(rows * PATCH_STRIDE)
                .max(tokens * config.out_hidden_size)
                * BF16,
        );
        let table = rope_positions * (HEAD_DIM / 4) * F32;
        let cos = take(table);
        let sin = take(table);
        Self {
            canvas,
            x,
            h,
            qkv,
            mlp,
            cos,
            sin,
            bytes: at,
        }
    }
}

/// One tensor's place in the arena: its checkpoint name, shape and padded row stride.
struct Placed {
    name: String,
    shape: Vec<usize>,
    stride: usize,
    offset: usize,
}

/// The arena's plan: every tensor's place, and the bytes they take.
#[derive(Default)]
struct Arena {
    placed: Vec<Placed>,
    at: usize,
}

impl Arena {
    /// Places one tensor of `shape`, its rows `stride` values apart; answers its offset.
    fn place(&mut self, name: String, shape: &[usize], stride: usize) -> usize {
        let rows: usize = shape[..shape.len() - 1].iter().product();
        let offset = self.at;
        self.at += (rows * stride * BF16).next_multiple_of(ALIGN);
        self.placed.push(Placed {
            name,
            shape: shape.to_vec(),
            stride,
            offset,
        });
        offset
    }

    /// Places a LayerNorm's weight and bias.
    fn norm(&mut self, name: &str) -> [usize; 2] {
        [
            self.place(format!("{name}.weight"), &[WIDTH], WIDTH),
            self.place(format!("{name}.bias"), &[WIDTH], WIDTH),
        ]
    }
}

/// The tower on one device; see the module documentation.
pub struct InklingVisionTower {
    config: InklingVisionConfig,
    kernels: Kernels,
    _modules: Vec<Module>,
    arena: Deviceptr,
    arena_bytes: usize,
    patch: Deviceptr,
    ln_pre: [Deviceptr; 2],
    blocks: Vec<Block>,
    proj_in: Deviceptr,
    proj_out: Deviceptr,
    /// The class row, class_embedding + cls_position_embedding, in f32.
    class_row: Vec<f32>,
    /// The learned position grid, `[side * side][1024]` f32, resampled per canvas.
    position: Vec<f32>,
    scratch: Option<(Deviceptr, usize)>,
}

// SAFETY: the kernel handles and device pointers are driver handles. Every launch and copy goes
// to the default stream of the context the caller holds current, and the owner serialises use
// behind a lock, which is the argument the text arm's own kernel table makes.
unsafe impl Send for InklingVisionTower {}
// SAFETY: as above; no method takes `&self` and mutates device state.
unsafe impl Sync for InklingVisionTower {}

impl std::fmt::Debug for InklingVisionTower {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingVisionTower")
            .field("depth", &self.blocks.len())
            .field("device_bytes", &self.device_bytes())
            .finish_non_exhaustive()
    }
}

impl InklingVisionTower {
    /// Loads the kernels and uploads the tower's tensors from `source` (the checkpoint).
    ///
    /// # Errors
    /// A geometry the kernels are not written for, a build without the Inkling image, a missing
    /// symbol or tensor, a tensor whose dtype or shape is not the release's, or the driver's
    /// failure.
    pub fn load(
        cuda: &Cuda,
        config: &InklingVisionConfig,
        source: &dyn WeightSource,
    ) -> Result<Self, String> {
        let geometry = [
            ("hidden_size", config.hidden_size, WIDTH),
            ("heads", config.heads, HEADS),
            ("patch_size", config.patch_size, PATCH),
            ("adapter_merge_size", config.adapter_merge_size, MERGE),
        ];
        // No temporal_patch_size rule: the release folds the temporal axis into the stored
        // patch projection, so its [width, 3 x patch x patch] shape (checked on upload, the host
        // reference's shape) is what a still image needs, whatever the config says.
        for (name, value, expected) in geometry {
            if value != expected {
                return Err(format!(
                    "the device tower is written for a vision {name} of {expected}, and this \
                     checkpoint's is {value}"
                ));
            }
        }
        if !config.projection_hidden_dim.is_multiple_of(32)
            || !config.out_hidden_size.is_multiple_of(8)
        {
            return Err(format!(
                "the merger's widths {} and {} are not the GEMM's multiples of 32 and 8",
                config.projection_hidden_dim, config.out_hidden_size
            ));
        }
        let cubin = inkling_cubin();
        if cubin.is_empty() {
            return Err(
                "this build has no Inkling cubin: the device tower's kernels are \
                        kernels/inkling_vision_sm121.cu"
                    .to_owned(),
            );
        }
        let mut modules = Vec::new();
        let mut kernel = |name: &'static str, precision: Precision| -> Result<Bound, String> {
            let (module, bound) = Bound::new(cuda, Image::Inkling, cubin, name, precision)?;
            modules.push(module);
            Ok(bound)
        };
        let kernels = Kernels {
            patchify: kernel("inkling_vision_patchify", Precision::F32)?,
            layernorm: kernel("inkling_vision_layernorm", Precision::F32)?,
            rope: kernel("inkling_vision_rope", Precision::F32)?,
            scale_add: kernel("inkling_vision_scale_add", Precision::F32)?,
            gemm: kernel("opt_vision_gemm", Precision::Bf16)?,
            attention: kernel("opt_vision_attn", Precision::Bf16)?,
            widen: kernel("inkling_widen_f32", Precision::F32)?,
        };

        let prefix = "model.visual";
        let embeddings = format!("{prefix}.backbone.embeddings");
        let intermediate = mlp_width(WIDTH);
        let mut arena = Arena::default();
        let patch = arena.place(
            format!("{embeddings}.patch_embedding.weight"),
            &[WIDTH, patch_values(config)],
            PATCH_STRIDE,
        );
        let ln_pre = arena.norm(&format!("{prefix}.backbone.ln_pre"));
        let mut blocks = Vec::with_capacity(config.depth);
        for layer in 0..config.depth {
            let p = format!("{prefix}.backbone.layers.{layer}");
            let ln_1 = arena.norm(&format!("{p}.ln_1"));
            let ln_2 = arena.norm(&format!("{p}.ln_2"));
            let ls_1 = arena.place(format!("{p}.ls_1.gamma"), &[WIDTH], WIDTH);
            let ls_2 = arena.place(format!("{p}.ls_2.gamma"), &[WIDTH], WIDTH);
            // A [1024][1024] matrix and a [1024] row are whole multiples of the alignment, so
            // q, k and v placed in turn are one [3072][1024] matrix and one [3072] bias.
            let attention = format!("{p}.self_attn");
            let [qkv, ..] = ["q_proj", "k_proj", "v_proj"].map(|name| {
                arena.place(format!("{attention}.{name}.weight"), &[WIDTH, WIDTH], WIDTH)
            });
            let [qkv_bias, ..] = ["q_proj", "k_proj", "v_proj"]
                .map(|name| arena.place(format!("{attention}.{name}.bias"), &[WIDTH], WIDTH));
            let out = arena.place(
                format!("{attention}.out_proj.weight"),
                &[WIDTH, WIDTH],
                WIDTH,
            );
            let out_bias = arena.place(format!("{attention}.out_proj.bias"), &[WIDTH], WIDTH);
            let fc1 = arena.place(
                format!("{p}.mlp.c_fc.weight"),
                &[intermediate, WIDTH],
                WIDTH,
            );
            let fc1_bias = arena.place(format!("{p}.mlp.c_fc.bias"), &[intermediate], intermediate);
            let fc2 = arena.place(
                format!("{p}.mlp.c_proj.weight"),
                &[WIDTH, intermediate],
                intermediate,
            );
            let fc2_bias = arena.place(format!("{p}.mlp.c_proj.bias"), &[WIDTH], WIDTH);
            blocks.push([
                ln_1[0], ln_1[1], ln_2[0], ln_2[1], ls_1, ls_2, qkv, qkv_bias, out, out_bias, fc1,
                fc1_bias, fc2, fc2_bias,
            ]);
        }
        let cell = WIDTH * MERGE * MERGE;
        let proj_in = arena.place(
            format!("{prefix}.proj_in.weight"),
            &[config.projection_hidden_dim, cell],
            cell,
        );
        let proj_out = arena.place(
            format!("{prefix}.proj_out.weight"),
            &[config.out_hidden_size, config.projection_hidden_dim],
            config.projection_hidden_dim,
        );
        let Arena { placed, at } = arena;

        let read = |name: &str, shape: &[usize]| -> Result<Vec<u8>, String> {
            let spec = source
                .spec(name)
                .map_err(|error| format!("{name}: {error}"))?;
            if spec.dtype != DType::BF16 || spec.shape != shape {
                return Err(format!(
                    "{name}: {:?} {:?}, expected BF16 {shape:?}",
                    spec.dtype, spec.shape
                ));
            }
            source
                .read(name)
                .map_err(|error| format!("{name}: {error}"))
        };
        let widen = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes(*pair)))
                .collect()
        };
        let side = config.posemb_grid_size;
        let position = widen(&read(
            &format!("{embeddings}.position_embedding"),
            &[side * side, WIDTH],
        )?);
        let class = widen(&read(&format!("{embeddings}.class_embedding"), &[WIDTH])?);
        let class_position = widen(&read(
            &format!("{embeddings}.cls_position_embedding"),
            &[WIDTH],
        )?);
        let class_row: Vec<f32> = class
            .iter()
            .zip(&class_position)
            .map(|(a, b)| a + b)
            .collect();

        let mut total = 0usize;
        let arena = cuda.alloc(at, &mut total)?;
        let upload = || -> Result<(), String> {
            for item in &placed {
                let bytes = read(&item.name, &item.shape)?;
                let width = item.shape[item.shape.len() - 1];
                let destination = arena.offset(item.offset);
                if item.stride == width {
                    cuda.copy_htod(&destination, &bytes)?;
                } else {
                    let mut padded = vec![0u8; bytes.len() / width * item.stride];
                    for (row, values) in bytes.chunks_exact(width * BF16).enumerate() {
                        let start = row * item.stride * BF16;
                        padded[start..start + values.len()].copy_from_slice(values);
                    }
                    cuda.copy_htod(&destination, &padded)?;
                }
            }
            Ok(())
        };
        if let Err(error) = upload() {
            cuda.free_buffers(&[&arena]);
            return Err(error);
        }
        let pointer = |offset: usize| arena.offset(offset);
        Ok(Self {
            config: config.clone(),
            kernels,
            _modules: modules,
            arena,
            arena_bytes: at,
            patch: pointer(patch),
            ln_pre: ln_pre.map(pointer),
            blocks: blocks
                .into_iter()
                .map(
                    |[
                        ln_1w,
                        ln_1b,
                        ln_2w,
                        ln_2b,
                        ls_1,
                        ls_2,
                        qkv,
                        qkv_bias,
                        out,
                        out_bias,
                        fc1,
                        fc1_bias,
                        fc2,
                        fc2_bias,
                    ]| Block {
                        ln_1: [pointer(ln_1w), pointer(ln_1b)],
                        ln_2: [pointer(ln_2w), pointer(ln_2b)],
                        ls_1: pointer(ls_1),
                        ls_2: pointer(ls_2),
                        qkv: pointer(qkv),
                        qkv_bias: pointer(qkv_bias),
                        out: pointer(out),
                        out_bias: pointer(out_bias),
                        fc1: pointer(fc1),
                        fc1_bias: pointer(fc1_bias),
                        fc2: pointer(fc2),
                        fc2_bias: pointer(fc2_bias),
                    },
                )
                .collect(),
            proj_in: pointer(proj_in),
            proj_out: pointer(proj_out),
            class_row,
            position,
            scratch: None,
        })
    }

    /// The tower's geometry.
    #[must_use]
    pub fn config(&self) -> &InklingVisionConfig {
        &self.config
    }

    /// Device bytes the weights and the current scratch hold.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.arena_bytes + self.scratch.map_or(0, |(_, bytes)| bytes)
    }

    /// The tokens a canvas of `height` by `width` pixels makes: [plan_cells]'s rows times columns.
    ///
    /// # Errors
    /// As [plan_cells].
    pub fn tokens(&self, height: usize, width: usize) -> Result<usize, String> {
        let (rows, columns) = plan_cells(height, width, &self.config)?;
        Ok(rows * columns)
    }

    /// Runs the tower over one RGB canvas (`height * width * 3` bytes, rows top to bottom) and
    /// writes its image tokens to `out`: [Self::tokens] rows of `out_hidden_size` f32, one a cell
    /// in raster order. Returns the token count. The writes are queued on the default stream.
    ///
    /// # Errors
    /// A canvas of the wrong length, one [plan_cells] refuses, or the driver's failure.
    pub fn encode(
        &mut self,
        cuda: &Cuda,
        canvas: &[u8],
        height: usize,
        width: usize,
        out: Deviceptr,
    ) -> Result<usize, String> {
        if canvas.len() != height * width * 3 {
            return Err(format!(
                "a {width}x{height} canvas is {} bytes, not {}",
                canvas.len(),
                height * width * 3
            ));
        }
        let (cells_h, cells_w) = plan_cells(height, width, &self.config)?;
        let tokens = cells_h * cells_w;
        let (grid_h, grid_w) = (cells_h * MERGE, cells_w * MERGE);
        let patches = grid_h * grid_w;
        let rows = patches + 1;
        let rope_positions = grid_h.max(grid_w) + 1;
        let layout = Layout::of(rows, canvas.len(), rope_positions, tokens, &self.config);
        let base = self.scratch_for(cuda, layout.bytes)?;
        let at = |offset: usize| base.offset(offset);
        let (canvas_d, x, h, qkv, mlp) = (
            at(layout.canvas),
            at(layout.x),
            at(layout.h),
            at(layout.qkv),
            at(layout.mlp),
        );
        let k = self.kernels;
        let row_bytes = WIDTH * BF16;

        // The residual stream starts as the class row and the resampled positions; the patch
        // embedding is the GEMM's residual onto them.
        let positions = interpolate_positions(
            &self.position,
            self.config.posemb_grid_size,
            WIDTH,
            grid_h,
            grid_w,
            MERGE,
        );
        let mut stream = Vec::with_capacity(rows * row_bytes);
        for value in self.class_row.iter().chain(&positions) {
            stream.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
        }
        cuda.copy_htod(&x, &stream)?;
        let (cos, sin) = axial_rope_tables(rope_positions, HEAD_DIM);
        let (cos_d, sin_d) = (at(layout.cos), at(layout.sin));
        cuda.copy_htod(&cos_d, &f32_bytes(&cos))?;
        cuda.copy_htod(&sin_d, &f32_bytes(&sin))?;
        cuda.copy_htod(&canvas_d, canvas)?;
        cuda.launch(
            k.patchify.kernel,
            (blocks(patches * PATCH_STRIDE), 1, 1),
            (BLOCK, 1, 1),
            &[
                canvas_d.0,
                mlp.0,
                height as u64,
                width as u64,
                cells_w as u64,
                patches as u64,
            ],
        )?;
        self.gemm(
            cuda,
            mlp,
            self.patch,
            None,
            x.offset(row_bytes),
            patches,
            WIDTH,
            PATCH_STRIDE,
            PATCH_STRIDE,
            WIDTH,
            GEMM_RESIDUAL,
        )?;
        let epsilon = self.config.layer_norm_eps as f32;
        self.layernorm(cuda, x, self.ln_pre, x, rows, epsilon)?;

        for block in &self.blocks {
            // x = x + ls_1 * out(attention(rope(qkv(ln_1(x)))))
            self.layernorm(cuda, x, block.ln_1, h, rows, epsilon)?;
            self.gemm(
                cuda,
                h,
                block.qkv,
                Some(block.qkv_bias),
                qkv,
                rows,
                QKV,
                WIDTH,
                WIDTH,
                QKV,
                0,
            )?;
            cuda.launch(
                k.rope.kernel,
                (blocks(rows * 2 * HEADS * (HEAD_DIM / 2)), 1, 1),
                (BLOCK, 1, 1),
                &[qkv.0, cos_d.0, sin_d.0, rows as u64, cells_w as u64],
            )?;
            cuda.launch(
                k.attention.kernel,
                (rows.div_ceil(64) as u32, HEADS as u32, 1),
                (128, 1, 1),
                &[qkv.0, h.0, rows as u64],
            )?;
            // The q|k|v rows are dead once the attention has read them: the projection lands there.
            self.gemm(
                cuda,
                h,
                block.out,
                Some(block.out_bias),
                qkv,
                rows,
                WIDTH,
                WIDTH,
                WIDTH,
                WIDTH,
                0,
            )?;
            self.scale_add(cuda, x, block.ls_1, qkv, rows)?;
            // x = x + ls_2 * fc2(gelu(fc1(ln_2(x))))
            self.layernorm(cuda, x, block.ln_2, h, rows, epsilon)?;
            let intermediate = mlp_width(WIDTH);
            self.gemm(
                cuda,
                h,
                block.fc1,
                Some(block.fc1_bias),
                mlp,
                rows,
                intermediate,
                WIDTH,
                WIDTH,
                intermediate,
                GEMM_GELU,
            )?;
            self.gemm(
                cuda,
                mlp,
                block.fc2,
                Some(block.fc2_bias),
                qkv,
                rows,
                WIDTH,
                intermediate,
                intermediate,
                WIDTH,
                0,
            )?;
            self.scale_add(cuda, x, block.ls_2, qkv, rows)?;
        }

        // The merger: each cell's four patch rows are consecutive, so the rows after the class
        // row are already the [tokens][4096] input.
        let cell = WIDTH * MERGE * MERGE;
        let hidden = self.config.projection_hidden_dim;
        let out_hidden = self.config.out_hidden_size;
        self.gemm(
            cuda,
            x.offset(row_bytes),
            self.proj_in,
            None,
            qkv,
            tokens,
            hidden,
            cell,
            cell,
            hidden,
            GEMM_GELU,
        )?;
        self.gemm(
            cuda,
            qkv,
            self.proj_out,
            None,
            mlp,
            tokens,
            out_hidden,
            hidden,
            hidden,
            out_hidden,
            0,
        )?;
        let count = tokens * out_hidden;
        cuda.launch(
            k.widen.kernel,
            (blocks(count).min(65_535), 1, 1),
            (BLOCK, 1, 1),
            &[mlp.0, out.0, count as u64],
        )?;
        Ok(tokens)
    }

    /// Frees the weights and the scratch.
    pub fn free(&mut self, cuda: &Cuda) {
        match self.scratch.take() {
            Some((scratch, _)) => cuda.free_buffers(&[&self.arena, &scratch]),
            None => cuda.free_buffers(&[&self.arena]),
        }
        self.arena_bytes = 0;
    }

    /// The scratch, grown (never shrunk) to `bytes`.
    fn scratch_for(&mut self, cuda: &Cuda, bytes: usize) -> Result<Deviceptr, String> {
        if let Some((base, held)) = self.scratch
            && held >= bytes
        {
            return Ok(base);
        }
        if let Some((base, _)) = self.scratch.take() {
            // The previous canvas's launches may still read it.
            cuda.synchronize()?;
            cuda.free_buffers(&[&base]);
        }
        let mut total = 0usize;
        let base = cuda.alloc(bytes, &mut total)?;
        self.scratch = Some((base, bytes));
        Ok(base)
    }

    fn layernorm(
        &self,
        cuda: &Cuda,
        x: Deviceptr,
        norm: [Deviceptr; 2],
        out: Deviceptr,
        m: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.layernorm.kernel,
            (m.div_ceil(8) as u32, 1, 1),
            (BLOCK, 1, 1),
            &[
                x.0,
                norm[0].0,
                norm[1].0,
                out.0,
                m as u64,
                u64::from(epsilon.to_bits()),
            ],
        )
    }

    fn scale_add(
        &self,
        cuda: &Cuda,
        x: Deviceptr,
        gamma: Deviceptr,
        h: Deviceptr,
        m: usize,
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.scale_add.kernel,
            (blocks(m * WIDTH / 2), 1, 1),
            (BLOCK, 1, 1),
            &[x.0, gamma.0, h.0, m as u64],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        cuda: &Cuda,
        a: Deviceptr,
        w: Deviceptr,
        bias: Option<Deviceptr>,
        out: Deviceptr,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldo: usize,
        flags: u64,
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.gemm.kernel,
            (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
            (BLOCK, 1, 1),
            &[
                a.0,
                w.0,
                bias.map_or(0, |bias| bias.0),
                out.0,
                m as u64,
                n as u64,
                k as u64,
                lda as u64,
                ldo as u64,
                flags,
            ],
        )
    }
}

/// The blocks of [BLOCK] threads that cover `threads`, at least one.
fn blocks(threads: usize) -> u32 {
    u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).unwrap_or(u32::MAX)
}

/// f32 values as the little-endian bytes the driver copies.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
