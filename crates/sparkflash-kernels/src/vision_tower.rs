//! The vision tower on the device (DECISIONS.md D59).
//!
//! [VisionTower] holds the checkpoint's ViT (`vision.*`) and aligner (`aligner.*`) weights on one
//! device and runs `kernels/opt_vision_sm121.cu` over a padded RGB canvas, producing the 5120-wide
//! rows an image span's IMAGE positions take, in reading order. It also holds the three delimiter
//! rows (`image_start`, `image_newline`, `image_end`), so every row of a span comes from device
//! memory this module owns.
//!
//! The launch order is the reference `vision.py`'s: the patch embedding, 32 pre-norm blocks
//! (RMSNorm, qkv, 2D RoPE, bidirectional attention, wo plus the residual; RMSNorm, w1, SwiGLU, w2
//! plus the residual), the final RMSNorm, then the aligner's 3 x 3 unfold, w1 with GELU and w2.
//! The kernel file states where each result is rounded to bf16; the GEMMs and the attention sum
//! in their own order, so the rows match torch to accumulation-order rounding, not bit for bit.
//!
//! Scratch is sized per canvas and only grows: about 240 MB at the largest canvas the recipe plans
//! (9216 patches). The weights take 0.97 GB.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
use sparkflash_engine::{DType, WeightSource};

/// ViT width (`vision_dim`).
pub const DIM: usize = 1024;
/// ViT blocks (`vision_n_layers`).
pub const LAYERS: usize = 32;
/// SwiGLU width (`vision_inter_dim`).
pub const INTER: usize = 2816;
/// Patch side in pixels (`vision_patch_size`).
pub const PATCH: usize = 14;
/// Aligner downsample per side (`vision_downsample_ratio`).
pub const DOWNSAMPLE: usize = 3;
/// Features per patch: three channels of [PATCH] x [PATCH].
pub const PATCH_FEATURES: usize = 3 * PATCH * PATCH;
/// [PATCH_FEATURES] padded to the GEMM's K step; the patch rows and the embedding weight use it.
pub const PATCH_STRIDE: usize = 608;
/// The LLM's hidden width: one aligner or delimiter row, in bf16 values.
pub const ROW: usize = 5120;

const QKV: usize = 3 * DIM;
const UNFOLD: usize = DOWNSAMPLE * DOWNSAMPLE * DIM;
const BF16: usize = 2;
const ALIGN: usize = 256;

const GEMM_RESIDUAL: u64 = 1;
const GEMM_GELU: u64 = 2;

/// The patch grid and token grid of one canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionShape {
    /// Patch rows.
    pub n_vit_h: usize,
    /// Patch columns.
    pub n_vit_w: usize,
    /// Aligner rows: patch rows over [DOWNSAMPLE], rounded up.
    pub n_llm_h: usize,
    /// Aligner columns.
    pub n_llm_w: usize,
}

impl VisionShape {
    /// The grids of a `height x width` canvas.
    ///
    /// # Errors
    /// A side that is zero or not a multiple of [PATCH].
    pub fn of(height: usize, width: usize) -> Result<Self, String> {
        if height == 0
            || width == 0
            || !height.is_multiple_of(PATCH)
            || !width.is_multiple_of(PATCH)
        {
            return Err(format!(
                "a {width} x {height} canvas is not a positive multiple of the {PATCH}-pixel patch"
            ));
        }
        let (n_vit_h, n_vit_w) = (height / PATCH, width / PATCH);
        Ok(Self {
            n_vit_h,
            n_vit_w,
            n_llm_h: n_vit_h.div_ceil(DOWNSAMPLE),
            n_llm_w: n_vit_w.div_ceil(DOWNSAMPLE),
        })
    }

    /// ViT patches, the attention's sequence length.
    #[must_use]
    pub const fn patches(&self) -> usize {
        self.n_vit_h * self.n_vit_w
    }

    /// Aligner rows: the span's IMAGE positions.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.n_llm_h * self.n_llm_w
    }
}

/// A delimiter row of the image span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    /// `image_start`, the span's first row.
    Start,
    /// `image_newline`, after each row of the token grid.
    NewLine,
    /// `image_end`, the span's last row.
    End,
}

/// An intermediate [VisionTower::run] leaves in scratch, for checking against a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// The patches, bf16 `[patches][PATCH_STRIDE]`.
    Patches,
    /// The residual stream after the blocks the run stopped at, bf16 `[patches][DIM]`.
    Stream,
    /// The ViT output (after the final norm), bf16 `[patches][DIM]`; set when every block ran.
    Vit,
}

#[derive(Clone, Copy)]
struct Kernels {
    patchify: Kernel,
    gemm: Kernel,
    rmsnorm: Kernel,
    rope: Kernel,
    attn: Kernel,
    swiglu: Kernel,
    unfold: Kernel,
    span_rows: Kernel,
}

#[derive(Clone, Copy)]
struct Block {
    norm1: Deviceptr,
    wqkv: Deviceptr,
    bqkv: Deviceptr,
    wo: Deviceptr,
    bo: Deviceptr,
    norm2: Deviceptr,
    w1: Deviceptr,
    w2: Deviceptr,
}

/// One canvas's scratch, carved from one allocation. The patches share the SwiGLU input's space:
/// the embedding GEMM consumes them before the first block writes it.
#[derive(Clone, Copy)]
struct Layout {
    canvas: usize,
    x: usize,
    h: usize,
    qkv: usize,
    g: usize,
    unfold: usize,
    a1: usize,
    bytes: usize,
}

impl Layout {
    fn of(shape: &VisionShape, canvas_bytes: usize) -> Self {
        let n = shape.patches();
        let rows = shape.rows();
        let mut at = 0;
        let mut take = |bytes: usize| {
            let offset = at;
            at += bytes.next_multiple_of(ALIGN);
            offset
        };
        let canvas = take(canvas_bytes);
        let x = take(n * DIM * BF16);
        let h = take(n * DIM * BF16);
        let qkv = take(n * QKV * BF16);
        let g = take(n * (2 * INTER).max(PATCH_STRIDE) * BF16);
        let unfold = take(rows * UNFOLD * BF16);
        let a1 = take(rows * ROW * BF16);
        Self {
            canvas,
            x,
            h,
            qkv,
            g,
            unfold,
            a1,
            bytes: at,
        }
    }
}

/// The ViT and aligner on one device; see the module documentation.
pub struct VisionTower {
    kernels: Kernels,
    _modules: Vec<Module>,
    arena: Deviceptr,
    arena_bytes: usize,
    patch_w: Deviceptr,
    patch_b: Deviceptr,
    blocks: Vec<Block>,
    norm: Deviceptr,
    aligner_w1: Deviceptr,
    aligner_b1: Deviceptr,
    aligner_w2: Deviceptr,
    aligner_b2: Deviceptr,
    delimiters: [Deviceptr; 3],
    scratch: Option<(Deviceptr, usize)>,
    layout: Option<Layout>,
}

/// One weight's place in the arena: its checkpoint name, shape, and the row stride it is stored
/// with (the shape's last dimension, or wider for a zero-padded row).
struct Placed {
    name: String,
    shape: Vec<usize>,
    stride: usize,
    offset: usize,
}

impl VisionTower {
    /// Loads the kernels from `cubin` (`opt_vision_sm121.cubin`) and uploads every vision weight
    /// `source` holds.
    ///
    /// # Errors
    /// A missing symbol, a missing tensor or one whose dtype or shape is not the checkpoint's, or
    /// the driver's allocation or copy failure.
    pub fn load(cuda: &Cuda, cubin: &[u8], source: &dyn WeightSource) -> Result<Self, String> {
        let mut modules = Vec::new();
        let mut kernel = |name: &'static str| -> Result<Kernel, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(kernel)
        };
        let kernels = Kernels {
            patchify: kernel("opt_vision_patchify")?,
            gemm: kernel("opt_vision_gemm")?,
            rmsnorm: kernel("opt_vision_rmsnorm")?,
            rope: kernel("opt_vision_rope")?,
            attn: kernel("opt_vision_attn")?,
            swiglu: kernel("opt_vision_swiglu")?,
            unfold: kernel("opt_vision_unfold")?,
            span_rows: kernel("opt_vision_span_rows")?,
        };

        let mut placed: Vec<Placed> = Vec::new();
        let mut at = 0usize;
        let mut place = |name: String, shape: &[usize], stride: usize| -> usize {
            let rows: usize = shape[..shape.len() - 1].iter().product();
            let offset = at;
            at += (rows * stride * BF16).next_multiple_of(ALIGN);
            placed.push(Placed {
                name,
                shape: shape.to_vec(),
                stride,
                offset,
            });
            offset
        };
        let patch_w = place(
            "vision.patch_embed.proj.weight".into(),
            &[DIM, PATCH_FEATURES],
            PATCH_STRIDE,
        );
        let patch_b = place("vision.patch_embed.proj.bias".into(), &[DIM], DIM);
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let p = format!("vision.blocks.{layer}");
            blocks.push([
                place(format!("{p}.norm1.weight"), &[DIM], DIM),
                place(format!("{p}.attn.wqkv.weight"), &[QKV, DIM], DIM),
                place(format!("{p}.attn.wqkv.bias"), &[QKV], QKV),
                place(format!("{p}.attn.wo.weight"), &[DIM, DIM], DIM),
                place(format!("{p}.attn.wo.bias"), &[DIM], DIM),
                place(format!("{p}.norm2.weight"), &[DIM], DIM),
                place(format!("{p}.mlp.w1.weight"), &[2 * INTER, DIM], DIM),
                place(format!("{p}.mlp.w2.weight"), &[DIM, INTER], INTER),
            ]);
        }
        let norm = place("vision.norm.weight".into(), &[DIM], DIM);
        let aligner_w1 = place("aligner.w1.weight".into(), &[ROW, UNFOLD], UNFOLD);
        let aligner_b1 = place("aligner.w1.bias".into(), &[ROW], ROW);
        let aligner_w2 = place("aligner.w2.weight".into(), &[ROW, ROW], ROW);
        let aligner_b2 = place("aligner.w2.bias".into(), &[ROW], ROW);
        let delimiters = [
            place("image_start".into(), &[ROW], ROW),
            place("image_newline".into(), &[ROW], ROW),
            place("image_end".into(), &[ROW], ROW),
        ];

        let mut total = 0usize;
        let arena = cuda.alloc(at, &mut total)?;
        let upload = || -> Result<(), String> {
            for item in &placed {
                let spec = source
                    .spec(&item.name)
                    .map_err(|error| format!("{}: {error}", item.name))?;
                if spec.dtype != DType::BF16 || spec.shape != item.shape {
                    return Err(format!(
                        "{}: {:?} {:?}, expected BF16 {:?}",
                        item.name, spec.dtype, spec.shape, item.shape
                    ));
                }
                let bytes = source
                    .read(&item.name)
                    .map_err(|error| format!("{}: {error}", item.name))?;
                let width = item.shape[item.shape.len() - 1];
                let dst = arena.offset(item.offset);
                if item.stride == width {
                    cuda.copy_htod(&dst, &bytes)?;
                } else {
                    let mut padded = vec![0u8; bytes.len() / width * item.stride];
                    for (row, src) in bytes.chunks_exact(width * BF16).enumerate() {
                        let start = row * item.stride * BF16;
                        padded[start..start + src.len()].copy_from_slice(src);
                    }
                    cuda.copy_htod(&dst, &padded)?;
                }
            }
            Ok(())
        };
        if let Err(error) = upload() {
            cuda.free_buffers(&[&arena]);
            return Err(error);
        }
        let ptr = |offset: usize| arena.offset(offset);
        Ok(Self {
            kernels,
            _modules: modules,
            arena,
            arena_bytes: at,
            patch_w: ptr(patch_w),
            patch_b: ptr(patch_b),
            blocks: blocks
                .into_iter()
                .map(|[norm1, wqkv, bqkv, wo, bo, norm2, w1, w2]| Block {
                    norm1: ptr(norm1),
                    wqkv: ptr(wqkv),
                    bqkv: ptr(bqkv),
                    wo: ptr(wo),
                    bo: ptr(bo),
                    norm2: ptr(norm2),
                    w1: ptr(w1),
                    w2: ptr(w2),
                })
                .collect(),
            norm: ptr(norm),
            aligner_w1: ptr(aligner_w1),
            aligner_b1: ptr(aligner_b1),
            aligner_w2: ptr(aligner_w2),
            aligner_b2: ptr(aligner_b2),
            delimiters: delimiters.map(ptr),
            scratch: None,
            layout: None,
        })
    }

    /// Device bytes the weights and the current scratch hold.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.arena_bytes + self.scratch.map_or(0, |(_, bytes)| bytes)
    }

    /// The device address of a delimiter row, [ROW] bf16.
    #[must_use]
    pub fn delimiter(&self, which: Delimiter) -> Deviceptr {
        self.delimiters[match which {
            Delimiter::Start => 0,
            Delimiter::NewLine => 1,
            Delimiter::End => 2,
        }]
    }

    /// Runs the tower over one canvas (`height * width * 3` RGB bytes, rows top to bottom) and
    /// writes its [VisionShape::rows] aligner rows to `out`, [ROW] bf16 each, in reading order.
    ///
    /// # Errors
    /// See [VisionTower::run].
    pub fn encode(
        &mut self,
        cuda: &Cuda,
        canvas: &[u8],
        height: usize,
        width: usize,
        out: Deviceptr,
    ) -> Result<VisionShape, String> {
        self.run(cuda, canvas, height, width, LAYERS, Some(out))
    }

    /// The tower over one canvas, stopped after `layers` blocks: with every block the final norm
    /// runs too, and with `out` the aligner then writes the rows there. [VisionTower::probe] reads
    /// what the run left in scratch.
    ///
    /// # Errors
    /// A canvas whose size is not `height * width * 3` or whose sides are not multiples of
    /// [PATCH], more than [LAYERS] blocks, rows asked for without every block, or the driver's
    /// failure.
    pub fn run(
        &mut self,
        cuda: &Cuda,
        canvas: &[u8],
        height: usize,
        width: usize,
        layers: usize,
        out: Option<Deviceptr>,
    ) -> Result<VisionShape, String> {
        let shape = VisionShape::of(height, width)?;
        if canvas.len() != height * width * 3 {
            return Err(format!(
                "a {width} x {height} canvas is {} bytes, not {}",
                canvas.len(),
                height * width * 3
            ));
        }
        if layers > LAYERS || (out.is_some() && layers != LAYERS) {
            return Err(format!(
                "{layers} blocks of {LAYERS}, rows wanted: {}",
                out.is_some()
            ));
        }
        let layout = Layout::of(&shape, canvas.len());
        let base = self.scratch_for(cuda, layout.bytes)?;
        self.layout = Some(layout);
        let at = |offset: usize| base.offset(offset);
        let (canvas_d, x, h, qkv, g) = (
            at(layout.canvas),
            at(layout.x),
            at(layout.h),
            at(layout.qkv),
            at(layout.g),
        );
        let n = shape.patches();
        let k = &self.kernels;

        cuda.copy_htod(&canvas_d, canvas)?;
        cuda.launch(
            k.patchify,
            (blocks_for(n * PATCH_STRIDE, 256), 1, 1),
            (256, 1, 1),
            &[
                canvas_d.0,
                g.0,
                shape.n_vit_w as u64,
                width as u64,
                n as u64,
            ],
        )?;
        self.gemm(
            cuda,
            g,
            self.patch_w,
            Some(self.patch_b),
            x,
            n,
            DIM,
            PATCH_STRIDE,
            PATCH_STRIDE,
            DIM,
            0,
        )?;
        for block in &self.blocks[..layers] {
            self.rmsnorm(cuda, x, block.norm1, h, n)?;
            self.gemm(
                cuda,
                h,
                block.wqkv,
                Some(block.bqkv),
                qkv,
                n,
                QKV,
                DIM,
                DIM,
                QKV,
                0,
            )?;
            cuda.launch(
                k.rope,
                (blocks_for(n * 1024, 256), 1, 1),
                (256, 1, 1),
                &[qkv.0, n as u64, shape.n_vit_w as u64],
            )?;
            cuda.launch(
                k.attn,
                (n.div_ceil(64) as u32, (DIM / 64) as u32, 1),
                (128, 1, 1),
                &[qkv.0, h.0, n as u64],
            )?;
            self.gemm(
                cuda,
                h,
                block.wo,
                Some(block.bo),
                x,
                n,
                DIM,
                DIM,
                DIM,
                DIM,
                GEMM_RESIDUAL,
            )?;
            self.rmsnorm(cuda, x, block.norm2, h, n)?;
            self.gemm(
                cuda,
                h,
                block.w1,
                None,
                g,
                n,
                2 * INTER,
                DIM,
                DIM,
                2 * INTER,
                0,
            )?;
            cuda.launch(
                k.swiglu,
                (blocks_for(n * INTER / 2, 256), 1, 1),
                (256, 1, 1),
                &[g.0, qkv.0, n as u64],
            )?;
            self.gemm(
                cuda,
                qkv,
                block.w2,
                None,
                x,
                n,
                DIM,
                INTER,
                INTER,
                DIM,
                GEMM_RESIDUAL,
            )?;
        }
        if layers == LAYERS {
            self.rmsnorm(cuda, x, self.norm, h, n)?;
        }
        if let Some(out) = out {
            let rows = shape.rows();
            let (unfold, a1) = (at(layout.unfold), at(layout.a1));
            cuda.launch(
                k.unfold,
                (blocks_for(rows * UNFOLD, 256), 1, 1),
                (256, 1, 1),
                &[
                    h.0,
                    unfold.0,
                    shape.n_vit_h as u64,
                    shape.n_vit_w as u64,
                    shape.n_llm_w as u64,
                    rows as u64,
                ],
            )?;
            self.gemm(
                cuda,
                unfold,
                self.aligner_w1,
                Some(self.aligner_b1),
                a1,
                rows,
                ROW,
                UNFOLD,
                UNFOLD,
                ROW,
                GEMM_GELU,
            )?;
            self.gemm(
                cuda,
                a1,
                self.aligner_w2,
                Some(self.aligner_b2),
                out,
                rows,
                ROW,
                ROW,
                ROW,
                ROW,
                0,
            )?;
        }
        Ok(shape)
    }

    /// Where the last [VisionTower::run] left an intermediate; see [Probe] for each layout.
    #[must_use]
    pub fn probe(&self, which: Probe) -> Option<Deviceptr> {
        let (base, _) = self.scratch?;
        let layout = self.layout?;
        Some(base.offset(match which {
            Probe::Patches => layout.g,
            Probe::Stream => layout.x,
            Probe::Vit => layout.h,
        }))
    }

    /// Copies `count` rows of [ROW] bf16 into `dst`: entry `i` of `table`, a device array of `u64`
    /// pairs, is (destination row, source address).
    ///
    /// # Errors
    /// The driver's launch failure.
    pub fn copy_rows(
        &self,
        cuda: &Cuda,
        table: Deviceptr,
        dst: Deviceptr,
        count: usize,
    ) -> Result<(), String> {
        if count == 0 {
            return Ok(());
        }
        cuda.launch(
            self.kernels.span_rows,
            (count as u32, 1, 1),
            (256, 1, 1),
            &[table.0, dst.0],
        )
    }

    /// Frees the weights and the scratch.
    pub fn free(self, cuda: &Cuda) {
        match self.scratch {
            Some((scratch, _)) => cuda.free_buffers(&[&self.arena, &scratch]),
            None => cuda.free_buffers(&[&self.arena]),
        }
    }

    fn scratch_for(&mut self, cuda: &Cuda, bytes: usize) -> Result<Deviceptr, String> {
        if let Some((base, held)) = self.scratch
            && held >= bytes
        {
            return Ok(base);
        }
        if let Some((base, _)) = self.scratch.take() {
            cuda.free_buffers(&[&base]);
        }
        let mut total = 0usize;
        let base = cuda.alloc(bytes, &mut total)?;
        self.scratch = Some((base, bytes));
        Ok(base)
    }

    fn rmsnorm(
        &self,
        cuda: &Cuda,
        x: Deviceptr,
        weight: Deviceptr,
        out: Deviceptr,
        m: usize,
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.rmsnorm,
            (m.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
            &[x.0, weight.0, out.0, m as u64],
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
            self.kernels.gemm,
            (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                a.0,
                w.0,
                bias.map_or(0, |b| b.0),
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

fn blocks_for(threads: usize, block: usize) -> u32 {
    threads.div_ceil(block) as u32
}
