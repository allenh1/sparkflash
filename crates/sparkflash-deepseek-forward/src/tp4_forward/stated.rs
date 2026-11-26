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

//! A stated block on the rows path ([crate::plan::BlockSteps::Stated]): a stack whose layout
//! states its own wiring (GLM-5.3-Flash's), each block walked on the shared role bodies at the
//! producer's types, as the host's whole-model reference walks it
//! (`sparkflash-model-deepseek-v41` `forward::stated`).
//!
//! Every sublayer is the manifold mix's entry with its own mix and the stated norm
//! ([ManifoldMixer::enter]), the token mixer or the FFN, the base's reduce on the BF16 wire,
//! and the mix's leave in the producer's expression ([ManifoldMixer::leave]). Every projection
//! is the tiled body's ([TiledProjection]) and is held in BF16, as the producer's modules return
//! their outputs; an FP8 linear quantises its input on the stated 128-wide tiles. The router's
//! logits stay f32, as the producer computes them.

use super::rows::{LaneSegment, RecurrentBody, RowsCx, RowsState, TiledLinear, span};
use super::*;
use crate::plan::{StatedBlock, StatedFfn, StatedMixer};
use sparkflash_kernels::images::Image;
use sparkflash_ops::calls;
use sparkflash_ops::indexer::MAX_SELECTED_TOKENS;
use sparkflash_ops::plan::Precision;
use sparkflash_ops::roles::{
    AbsorbedGeometry, AbsorbedMla, AbsorbedRows, AbsorbedWeights, Bound, KdaLayer, KernelTable,
    Launcher, ManifoldMixer, ManifoldRows, ManifoldWeights, PooledGeometry, PooledIndexer,
    PooledRows, PooledWeights, Router, RouterRule, RouterRows,
};
use sparkflash_ops::tiled::{TiledMatrix, TiledProjection, TiledScratch};

/// The rows one pass of the absorbed attention and the FFN's expert bodies takes at once: their
/// per-row scratch (the logits over every selectable position, the experts' rows) is carved for
/// this many rows, and a longer pass walks them a chunk at a time.
const CHUNK: usize = 64;

/// The tiles an FP8 linear's input is quantised on: the tiled body's E4M3 block.
pub(super) const ACTIVATION_TILE: usize = 128;

/// The expert GEMV's tokens a launch: a token picks an expert once, so a launch of this many
/// tokens gives no expert more pairs than the body batches (`Q4E_MAX_BATCH`).
const EXPERT_TOKENS: usize = 8;

/// The expert GEMV's threads a block (`Q4E_TG_THREADS`).
const EXPERT_THREADS: u32 = 256;

/// The rows a tile of the expert GEMV's outputs covers (`Q4E_TG_TILE`).
const EXPERT_TILE: usize = 32;

/// One sublayer's manifold mix: the `fn` projection, its f32 scale and base, and the sublayer
/// norm's BF16 gains.
#[derive(Debug, Clone, Copy)]
struct Mix {
    projection: TiledMatrix,
    scale: Deviceptr,
    base: Deviceptr,
    norm: Deviceptr,
}

/// A Kimi Delta Attention block's tensors.
#[derive(Debug, Clone, Copy)]
struct KdaTensors {
    qkv: [TiledMatrix; 3],
    forget: (TiledMatrix, TiledMatrix),
    gate: (TiledMatrix, TiledMatrix),
    b: TiledMatrix,
    out: TiledMatrix,
    conv: [Deviceptr; 3],
    decay: (Deviceptr, Deviceptr),
    norm: Deviceptr,
}

/// A pooled indexer's tensors.
#[derive(Debug, Clone, Copy)]
pub(super) struct IndexerTensors {
    pub(super) keys: TiledMatrix,
    pub(super) gates: TiledMatrix,
    pub(super) head_weights: TiledMatrix,
    pub(super) queries: TiledMatrix,
    pub(super) key_norm: (Deviceptr, Deviceptr),
    pub(super) ape: Deviceptr,
}

/// An absorbed MLA block's tensors, its own indexer's when it runs one.
#[derive(Debug, Clone, Copy)]
pub(super) struct AbsorbedTensors {
    pub(super) q_a: TiledMatrix,
    pub(super) kv_a: TiledMatrix,
    pub(super) q_b: TiledMatrix,
    pub(super) out: TiledMatrix,
    pub(super) norms: (Deviceptr, Deviceptr),
    pub(super) kv_b: Deviceptr,
    /// Optional raw BF16 `kv_b` and transposed key rows for A4 bulk prefill MMA.
    pub(super) kv_b_bf16: Option<(Deviceptr, Deviceptr)>,
    pub(super) indexer: Option<IndexerTensors>,
}

#[derive(Debug, Clone, Copy)]
enum MixerTensors {
    Kda(KdaTensors),
    Absorbed(AbsorbedTensors),
}

/// One gated MLP: gate, up and down.
#[derive(Debug, Clone, Copy)]
pub(super) struct Mlp {
    pub(super) gate: TiledMatrix,
    pub(super) up: TiledMatrix,
    pub(super) down: TiledMatrix,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FfnTensors {
    Dense(Mlp),
    Moe {
        router: TiledMatrix,
        bias: Deviceptr,
        /// The routed experts' `[experts][2]` (weight, scales) tables: gate, up and down.
        tables: [Deviceptr; 3],
        shared: Option<Mlp>,
    },
}

#[derive(Debug, Clone, Copy)]
struct StatedLayer {
    block: StatedBlock,
    mixes: [Mix; 2],
    mixer: MixerTensors,
    ffn: FfnTensors,
}

/// The widths the rows a stated walk carves are sized by, on this rank.
#[derive(Debug, Clone, Copy)]
pub(super) struct Widths {
    /// The widest input a projection quantises.
    pub(super) columns: usize,
    /// A KDA projection's outputs (the stage its row copy places them from).
    pub(super) kda_part: usize,
    pub(super) experts: usize,
    pub(super) top_k: usize,
    /// A routed expert's intermediate rows on this rank.
    pub(super) routed_inter: usize,
    /// The widest MLP body that is not a routed expert's (dense or shared), on this rank.
    pub(super) mlp: usize,
}

/// A stated stack's device state: the shared bodies, each block's tensors in the arena (with
/// the few the bodies read in another type built here), and the widths its rows are carved by.
pub(super) struct StatedDevice {
    pub(super) mixer: ManifoldMixer,
    pub(super) tiled: TiledProjection,
    pub(super) absorbed: Option<AbsorbedMla>,
    pub(super) indexer: Option<PooledIndexer>,
    pub(super) router: Router,
    /// `q4e_expert_gemv_e4m3_block`.
    pub(super) expert: Bound,
    /// `q4e_copy_rows`, `q4e_swiglu_held`, `q4e_combine_held` and `q4e_narrow_bf16`.
    pub(super) token: [Bound; 4],
    layers: Vec<StatedLayer>,
    pub(super) widths: Widths,
    pub(super) quantise: bool,
    /// Whether the absorbed MLA latent cache stores E4M3 codes with 128-wide f32 block scales
    /// (the serving default; `attention.latent_cache = "f32"` selects f32).
    pub(super) e4m3_latent: bool,
    allocations: Vec<Deviceptr>,
}

impl StatedDevice {
    /// Frees what the attach built (the arena's own tensors are the arena's).
    pub(super) fn release(self, cuda: &Cuda) {
        let refs: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&refs);
    }
}

/// Whether the stated absorbed MLA latent cache stores E4M3 codes with f32 block scales
/// (`true`, the serving default) or unquantised f32 (`false`, the parity form). Chosen by the
/// profile's `attention.latent_cache` (`fp8` / `auto` / unset => E4M3; `f32` => f32), never as a
/// model fact in the layout.
pub(super) fn latent_cache_fp8_for(
    mode: Option<sparkflash_model::profile::LatentCacheMode>,
) -> bool {
    !matches!(mode, Some(sparkflash_model::profile::LatentCacheMode::F32))
}

pub(super) fn latent_cache_fp8() -> bool {
    latent_cache_fp8_for(sparkflash_model::active_profile().attention.latent_cache)
}

/// One absorbed MLA block's lane region geometry, in bytes per position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AbsorbedRegionPlan {
    pub(super) layer: usize,
    /// Bytes a position for the latent cache (`kv_lora_rank` for E4M3, `kv_lora_rank * 4` for f32).
    pub(super) latent_row_bytes: usize,
    /// Bytes a position for the E4M3 block scales, or `None` when the latent cache is f32.
    pub(super) scale_row_bytes: Option<usize>,
    /// `(key_row_bytes, pool)` when the layer owns a pooled indexer; `None` for a shared-indexer
    /// layer, which holds no keys, gates or pooled keys of its own.
    pub(super) indexer: Option<(usize, usize)>,
}

/// The lane region geometry of every absorbed MLA block in `description`, for E4M3 (`e4m3_latent`)
/// or f32 latent caching. Empty for a stack that states no absorbed block.
///
/// # Errors
/// A block that cannot say how it runs, or a stated stack with no absorbed latent.
pub(super) fn absorbed_regions_for(
    description: &crate::plan::Description,
    e4m3_latent: bool,
) -> Result<Vec<AbsorbedRegionPlan>, String> {
    if !description.values().stated().bf16_stream {
        return Ok(Vec::new());
    }
    let values = description.values();
    let pool = values.stated().index_pool.max(1);
    let mut out = Vec::new();
    for layer in 0..values.n_layers {
        let steps = description
            .block_steps(layer)
            .map_err(|error| format!("lanes: {error}"))?;
        let crate::plan::BlockSteps::Stated(StatedBlock {
            mixer: StatedMixer::Absorbed { own_indexer },
            ..
        }) = steps
        else {
            continue;
        };
        let latent = values
            .absorbed_latent
            .ok_or_else(|| {
                format!("lanes: absorbed block {layer} in a stack that states no absorbed latent")
            })?
            .kv_lora_rank;
        let latent_row_bytes = if e4m3_latent { latent } else { latent * 4 };
        let scale_row_bytes = e4m3_latent.then(|| latent.div_ceil(ACTIVATION_TILE) * 4);
        let indexer = own_indexer.then(|| (values.index_head_dim * 4, pool));
        out.push(AbsorbedRegionPlan {
            layer,
            latent_row_bytes,
            scale_row_bytes,
            indexer,
        });
    }
    Ok(out)
}

/// The active stack's absorbed MLA lane regions for the active profile's `attention.latent_cache`.
///
/// # Errors
/// As [absorbed_regions_for].
pub(super) fn absorbed_regions() -> Result<Vec<AbsorbedRegionPlan>, String> {
    if !crate::plan::stated_stack() {
        return Ok(Vec::new());
    }
    absorbed_regions_for(crate::plan::description(), latent_cache_fp8())
}

/// A tensor's arena slice by its claimed name.
pub(super) fn slice<'s>(
    arena: &'s sparkflash_device::arena::ResidentArena,
    name: &str,
) -> Result<&'s sparkflash_device::arena::ArenaSlice, String> {
    arena
        .slice(name)
        .ok_or_else(|| format!("stated: the arena holds no {name}"))
}

/// The rank's `[rows, columns]` matrix `name`: block-scaled E4M3 when the arena holds its
/// `.scale` (f32, one a 128 x 128 block), else BF16.
pub(super) fn matrix(
    arena: &sparkflash_device::arena::ResidentArena,
    name: &str,
    (rows, columns): (usize, usize),
) -> Result<TiledMatrix, String> {
    let weight = slice(arena, name)?;
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    match arena.slice(&format!("{stem}.scale")) {
        Some(scales) => {
            let blocks = rows.div_ceil(ACTIVATION_TILE) * columns.div_ceil(ACTIVATION_TILE);
            if weight.bytes != rows * columns || scales.bytes != blocks * 4 {
                return Err(format!(
                    "stated: {name} holds {} bytes and {} scale bytes, not a [{rows}, {columns}] E4M3 matrix with f32 128 x 128 block scales",
                    weight.bytes, scales.bytes
                ));
            }
            Ok(TiledMatrix::whole(
                weight.ptr,
                Some(scales.ptr),
                rows,
                columns,
            ))
        }
        None => {
            if weight.bytes != rows * columns * 2 {
                return Err(format!(
                    "stated: {name} holds {} bytes, not a [{rows}, {columns}] BF16 matrix",
                    weight.bytes
                ));
            }
            Ok(TiledMatrix::whole(weight.ptr, None, rows, columns))
        }
    }
}

/// A layer's claimed tensor name by its tail, refused by name when the block claims none.
fn claim(layer: usize, tail: &str) -> Result<&'static str, String> {
    crate::plan::layer_claim_if(layer, tail)
        .ok_or_else(|| format!("stated: layer {layer} claims no tensor ending .{tail}"))
}

impl DeviceLayer<'_> {
    /// A vector of `count` values in the type its body reads: the arena's own slice when it
    /// holds that type; a BF16 tensor a body reads as f32 widened exactly here (and kept).
    /// Narrowing would change a stored value, so an f32 tensor a body reads as BF16 is refused.
    pub(super) fn stated_vector(
        &self,
        arena: &sparkflash_device::arena::ResidentArena,
        name: &str,
        (count, wide): (usize, bool),
        allocations: &mut Vec<Deviceptr>,
    ) -> Result<Deviceptr, String> {
        let tensor = slice(arena, name)?;
        let element = if wide { 4 } else { 2 };
        if tensor.bytes == count * element {
            return Ok(tensor.ptr);
        }
        if wide && tensor.bytes == count * 2 {
            let mut bits = vec![0u8; count * 2];
            self.cuda.copy_dtoh(&mut bits, &tensor.ptr)?;
            let widened: Vec<u8> = bits
                .chunks_exact(2)
                .flat_map(|word| {
                    sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes([word[0], word[1]]))
                        .to_le_bytes()
                })
                .collect();
            return self.stated_upload(&widened, allocations);
        }
        Err(format!(
            "stated: {name} holds {} bytes, not {count} {} values",
            tensor.bytes,
            if wide { "f32 (or BF16 to widen)" } else { "BF16" }
        ))
    }

    pub(super) fn stated_upload(
        &self,
        bytes: &[u8],
        allocations: &mut Vec<Deviceptr>,
    ) -> Result<Deviceptr, String> {
        let ptr = self.cuda.alloc(bytes.len().max(4), &mut 0)?;
        allocations.push(ptr);
        self.cuda.copy_htod(&ptr, bytes)?;
        Ok(ptr)
    }

    /// The rank's `kv_b` (`[heads (nope + value), latent]`, BF16 or block-scaled E4M3) widened
    /// to f32 in the absorbed attention's layout: a head's key rows as stored, then its value
    /// rows transposed to `[latent][value]` ([AbsorbedWeights::kv_b]).
    pub(super) fn stated_kv_b(
        &self,
        arena: &sparkflash_device::arena::ResidentArena,
        name: &str,
        geometry: AbsorbedGeometry,
        allocations: &mut Vec<Deviceptr>,
    ) -> Result<(Deviceptr, Option<(Deviceptr, Deviceptr)>), String> {
        let AbsorbedGeometry {
            heads,
            nope,
            value,
            latent,
            ..
        } = geometry;
        let rows = heads * (nope + value);
        let stored = matrix(arena, name, (rows, latent))?;
        let mut bf16_pair = None;
        let decoded: Vec<f32> = match stored.scales {
            None => {
                let mut bits = vec![0u8; rows * latent * 2];
                self.cuda.copy_dtoh(&mut bits, &stored.weight)?;
                // A4: transpose each head's [nope][latent] BF16 key rows to [latent][nope] BF16
                // for glm_prefill_heads_bf16_mma when latent == 512 and value is 128-aligned.
                if latent == 512 && value.is_multiple_of(128) {
                    let mut key_t = vec![0u8; heads * latent * nope * 2];
                    for head in 0..heads {
                        let first = head * (nope + value);
                        for k in 0..nope {
                            for col in 0..latent {
                                let src = ((first + k) * latent + col) * 2;
                                let dst = ((head * latent + col) * nope + k) * 2;
                                key_t[dst] = bits[src];
                                key_t[dst + 1] = bits[src + 1];
                            }
                        }
                    }
                    let key_t_dev = self.stated_upload(&key_t, allocations)?;
                    bf16_pair = Some((stored.weight, key_t_dev));
                }
                bits.chunks_exact(2)
                    .map(|word| {
                        sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes([
                            word[0], word[1],
                        ]))
                    })
                    .collect()
            }
            Some(scales) => {
                let mut codes = vec![0u8; rows * latent];
                self.cuda.copy_dtoh(&mut codes, &stored.weight)?;
                let blocks = latent.div_ceil(ACTIVATION_TILE);
                let mut scale_bytes = vec![0u8; rows.div_ceil(ACTIVATION_TILE) * blocks * 4];
                self.cuda.copy_dtoh(&mut scale_bytes, &scales)?;
                let scale = |row: usize, column: usize| {
                    let at = ((row / ACTIVATION_TILE) * blocks + column / ACTIVATION_TILE) * 4;
                    f32::from_le_bytes([
                        scale_bytes[at],
                        scale_bytes[at + 1],
                        scale_bytes[at + 2],
                        scale_bytes[at + 3],
                    ])
                };
                codes
                    .iter()
                    .enumerate()
                    .map(|(at, &code)| {
                        sparkflash_weights::quant::f8e4m3_to_f32(code)
                            * scale(at / latent, at % latent)
                    })
                    .collect()
            }
        };
        let mut laid = Vec::with_capacity(rows * latent);
        for head in 0..heads {
            let first = head * (nope + value);
            laid.extend_from_slice(&decoded[first * latent..(first + nope) * latent]);
            for column in 0..latent {
                for row in 0..value {
                    laid.push(decoded[(first + nope + row) * latent + column]);
                }
            }
        }
        let bytes: Vec<u8> = laid.iter().flat_map(|value| value.to_le_bytes()).collect();
        Ok((self.stated_upload(&bytes, allocations)?, bf16_pair))
    }

    /// Opens a stated stack's device state from the arena: the shared bodies at the producer's
    /// types and each block's tensors. The routed experts are the bind's rank slices
    /// ([Self::attach_routed]), which must be attached first.
    ///
    /// # Errors
    /// A wiring the walk does not run, a tensor the arena does not hold in the shape and type
    /// the bodies read, or a kernel that does not load, each by name.
    pub(super) fn attach_stated(
        &mut self,
        arena: &sparkflash_device::arena::ResidentArena,
    ) -> Result<(), String> {
        if self.stated.is_some() {
            return Err("stated: attached twice".to_owned());
        }
        let description = crate::plan::description();
        let values = description.values();
        let wiring = values.stated();
        for (on, what) in [
            (wiring.own_mix, "each sublayer collapses with its own mix (own_mix)"),
            (wiring.mean_fold, "the head folds the copies by their mean (mean_fold)"),
            (wiring.bf16_stream, "the stream holds BF16 values (bf16_stream)"),
        ] {
            if !on {
                return Err(format!(
                    "stated: the device walk runs a stack where {what}, and this layout states otherwise"
                ));
            }
        }
        if wiring.activation_tile != 0 && wiring.activation_tile != ACTIVATION_TILE {
            return Err(format!(
                "stated: activations quantised on {}-wide tiles; the tiled body quantises {ACTIVATION_TILE}-wide ones",
                wiring.activation_tile
            ));
        }
        let quantise = wiring.activation_tile != 0;
        let cuda = self.cuda;
        let world = self.routed_geometry.world;
        let dim = self.dim;
        let hc = values.hc_mult;
        let mixer = ManifoldMixer::open(
            cuda,
            sparkflash_weights::mhc::HcGeometry {
                hc_mult: hc,
                dim,
                hc_sinkhorn_iters: values.hc_sinkhorn_iters as _,
                hc_eps: values.hc_eps as f32,
                norm_eps: values.norm_eps as f32,
            },
            Precision::Bf16,
        )?;
        let tiled = TiledProjection::open(
            cuda,
            quantise,
            TiledProjection::default_activation(quantise),
        )?;
        let split = |what: &str, count: usize| {
            if count.is_multiple_of(world) {
                Ok(count / world)
            } else {
                Err(format!("stated: {count} {what} do not split over {world} ranks"))
            }
        };
        let absorbed_geometry = match values.absorbed_latent {
            Some(latent) => Some(AbsorbedGeometry {
                heads: split("query heads", values.n_heads)?,
                nope: values.head_dim,
                value: latent.v_head_dim,
                latent: latent.kv_lora_rank,
                q_lora: values.q_lora_rank,
            }),
            None => None,
        };
        let pooled_geometry = PooledGeometry {
            heads: values.index_n_heads,
            head_dim: values.index_head_dim,
            pool: wiring.index_pool,
            topk: values.index_topk,
            always_select_tail: wiring.index_tail,
        };
        let [router] = KernelTable {
            image: Image::DECODE,
            symbols: [("q4e_noaux_router", Precision::F32)],
        }
        .load(cuda)?;
        let [expert] = KernelTable {
            image: Image::DECODE,
            symbols: [("q4e_expert_gemv_e4m3_block", Precision::Fp8E4M3)],
        }
        .load(cuda)?;
        let token = KernelTable {
            image: Image::TOKEN,
            symbols: [
                ("q4e_copy_rows", Precision::F32),
                ("q4e_swiglu_held", Precision::Bf16),
                ("q4e_combine_held", Precision::Bf16),
                ("q4e_narrow_bf16", Precision::Bf16),
            ],
        }
        .load(cuda)?;
        let geometry = self.routed_geometry;
        let shared_inter = split(
            "shared-expert rows",
            values.moe_inter_dim * values.n_shared_experts,
        )?;
        let dense_inter = split("dense-MLP rows", values.dense_inter_dim)?;
        let kda = super::rows::recurrent_geometry()?;
        let mut allocations = Vec::new();
        let mut layers = Vec::with_capacity(values.n_layers);
        let mut any_absorbed = false;
        let mut any_indexer = false;
        let mut columns = (hc * dim).max(dim);
        for layer in 0..values.n_layers {
            let what = |error: String| format!("stated layer {layer}: {error}");
            let block = match description.block_steps(layer).map_err(what)? {
                crate::plan::BlockSteps::Stated(block) => *block,
                other => {
                    return Err(format!(
                        "stated layer {layer}: a stated stack's block runs {other:?}"
                    ));
                }
            };
            let mixes = (2 + hc) * hc;
            let mix = |sub: &str, norm: &str, allocations: &mut Vec<Deviceptr>| {
                Ok::<_, String>(Mix {
                    projection: matrix(arena, claim(layer, &format!("hc_{sub}_fn"))?, (mixes, hc * dim))?,
                    scale: self.stated_vector(
                        arena,
                        claim(layer, &format!("hc_{sub}_scale"))?,
                        (3, true),
                        allocations,
                    )?,
                    base: self.stated_vector(
                        arena,
                        claim(layer, &format!("hc_{sub}_base"))?,
                        (mixes, true),
                        allocations,
                    )?,
                    norm: self.stated_vector(arena, claim(layer, norm)?, (dim, false), allocations)?,
                })
            };
            let mixes = [
                mix("attn", "attn_norm.weight", &mut allocations).map_err(what)?,
                mix("ffn", "ffn_norm.weight", &mut allocations).map_err(what)?,
            ];
            let mixer = match block.mixer {
                StatedMixer::Kda => {
                    let geometry = kda.ok_or_else(|| {
                        format!("stated layer {layer}: a KDA block in a stack with no KDA geometry")
                    })?;
                    let names = sparkflash_engine::tensor::KDA_TENSORS;
                    let (width, part, heads, low, taps) = (
                        geometry.value_width(),
                        geometry.key_width(),
                        geometry.heads,
                        geometry.key_dim,
                        geometry.taps,
                    );
                    let m = |tail: &str, shape: (usize, usize)| {
                        matrix(arena, claim(layer, tail)?, shape).map_err(what)
                    };
                    let vector = |tail: &str, shape: (usize, bool), allocations: &mut Vec<Deviceptr>| {
                        self.stated_vector(arena, claim(layer, tail)?, shape, allocations)
                            .map_err(what)
                    };
                    columns = columns.max(width).max(low);
                    MixerTensors::Kda(KdaTensors {
                        qkv: [
                            m(names.qkv[0], (part, dim))?,
                            m(names.qkv[1], (part, dim))?,
                            m(names.qkv[2], (width, dim))?,
                        ],
                        forget: (m(names.forget[0], (low, dim))?, m(names.forget[1], (width, low))?),
                        gate: (m(names.gate[0], (low, dim))?, m(names.gate[1], (width, low))?),
                        b: m(names.b, (heads, dim))?,
                        out: m(names.out, (dim, width))?,
                        conv: [
                            vector(names.conv[0], (part * taps, false), &mut allocations)?,
                            vector(names.conv[1], (part * taps, false), &mut allocations)?,
                            vector(names.conv[2], (width * taps, false), &mut allocations)?,
                        ],
                        decay: (
                            vector(names.a_log, (heads, true), &mut allocations)?,
                            vector(names.dt_bias, (width, true), &mut allocations)?,
                        ),
                        norm: vector(names.norm, (geometry.value_dim, false), &mut allocations)?,
                    })
                }
                StatedMixer::Absorbed { own_indexer } => {
                    any_absorbed = true;
                    let geometry = absorbed_geometry.ok_or_else(|| {
                        format!(
                            "stated layer {layer}: an absorbed block in a stack that states no absorbed latent"
                        )
                    })?;
                    let AbsorbedGeometry {
                        heads,
                        nope,
                        value,
                        latent,
                        q_lora,
                    } = geometry;
                    let m = |tail: &str, shape: (usize, usize)| {
                        matrix(arena, claim(layer, tail)?, shape).map_err(what)
                    };
                    columns = columns.max(q_lora).max(heads * value);
                    let indexer = if own_indexer {
                        any_indexer = true;
                        let PooledGeometry { heads, head_dim, pool, .. } = pooled_geometry;
                        let vector = |tail: &str, shape: (usize, bool), allocations: &mut Vec<Deviceptr>| {
                            self.stated_vector(arena, claim(layer, tail)?, shape, allocations)
                                .map_err(what)
                        };
                        Some(IndexerTensors {
                            keys: m("attn.indexer.wk.weight", (head_dim, dim))?,
                            gates: m("attn.indexer.kpool_gate", (head_dim, dim))?,
                            head_weights: m("attn.indexer.weights_proj.weight", (heads, dim))?,
                            queries: m("attn.indexer.wq_b.weight", (heads * head_dim, q_lora))?,
                            key_norm: (
                                vector("attn.indexer.k_norm.weight", (head_dim, false), &mut allocations)?,
                                vector("attn.indexer.k_norm.bias", (head_dim, false), &mut allocations)?,
                            ),
                            ape: vector("attn.indexer.kpool_ape", (pool * head_dim, false), &mut allocations)?,
                        })
                    } else {
                        None
                    };
                    let (kv_b, kv_b_bf16) = self
                        .stated_kv_b(
                            arena,
                            claim(layer, "attn.wkv_b.weight")?,
                            geometry,
                            &mut allocations,
                        )
                        .map_err(what)?;
                    MixerTensors::Absorbed(AbsorbedTensors {
                        q_a: m("attn.wq_a.weight", (q_lora, dim))?,
                        kv_a: m("attn.wkv.weight", (latent, dim))?,
                        q_b: m("attn.wq_b.weight", (heads * nope, q_lora))?,
                        out: m("attn.wo.weight", (dim, heads * value))?,
                        norms: (
                            self.stated_vector(arena, claim(layer, "attn.q_norm.weight")?, (q_lora, false), &mut allocations)
                                .map_err(what)?,
                            self.stated_vector(arena, claim(layer, "attn.kv_norm.weight")?, (latent, false), &mut allocations)
                                .map_err(what)?,
                        ),
                        kv_b,
                        kv_b_bf16,
                        indexer,
                    })
                }
            };
            let m = |tail: &str, shape: (usize, usize)| {
                matrix(arena, claim(layer, tail)?, shape).map_err(what)
            };
            let mlp = |segment: &str, inter: usize| {
                Ok::<_, String>(Mlp {
                    gate: m(&format!("{segment}.w1.weight"), (inter, dim))?,
                    up: m(&format!("{segment}.w3.weight"), (inter, dim))?,
                    down: m(&format!("{segment}.w2.weight"), (dim, inter))?,
                })
            };
            let ffn = match block.ffn {
                StatedFfn::Dense => {
                    columns = columns.max(dense_inter);
                    FfnTensors::Dense(mlp("ffn", dense_inter)?)
                }
                StatedFfn::Moe => {
                    columns = columns.max(shared_inter).max(geometry.local_inter);
                    let experts = self
                        .routed
                        .as_ref()
                        .and_then(|routed| routed.get(layer))
                        .ok_or_else(|| {
                            format!("stated layer {layer}: no routed experts were attached")
                        })?;
                    let mut tables = [Deviceptr(0); 3];
                    for (index, table) in tables.iter_mut().enumerate() {
                        let words: Vec<u8> = experts
                            .iter()
                            .flat_map(|pair| {
                                let bind = pair[index];
                                [bind.weight.0, bind.scales.0]
                            })
                            .flat_map(u64::to_le_bytes)
                            .collect();
                        *table = self.stated_upload(&words, &mut allocations)?;
                    }
                    FfnTensors::Moe {
                        router: m("ffn.gate.weight", (geometry.experts, dim))?,
                        bias: self
                            .stated_vector(arena, claim(layer, "ffn.gate.bias")?, (geometry.experts, true), &mut allocations)
                            .map_err(what)?,
                        tables,
                        shared: if shared_inter == 0 {
                            None
                        } else {
                            Some(mlp("ffn.shared_experts", shared_inter)?)
                        },
                    }
                }
            };
            layers.push(StatedLayer {
                block,
                mixes,
                mixer,
                ffn,
            });
        }
        let absorbed = match absorbed_geometry.filter(|_| any_absorbed) {
            Some(geometry) => Some(AbsorbedMla::open(cuda, geometry, Precision::Bf16)?),
            None => None,
        };
        let indexer = if any_indexer {
            Some(PooledIndexer::open(cuda, pooled_geometry, Precision::Bf16)?)
        } else {
            None
        };
        self.stated = Some(StatedDevice {
            mixer,
            tiled,
            absorbed,
            indexer,
            router: Router::new(router)?,
            expert,
            token,
            layers,
            widths: Widths {
                columns,
                kda_part: kda.map_or(0, |geometry| geometry.key_width().max(geometry.value_width())),
                experts: geometry.experts,
                top_k: geometry.top_k,
                routed_inter: geometry.local_inter,
                mlp: shared_inter.max(dense_inter),
            },
            quantise,
            e4m3_latent: latent_cache_fp8(),
            allocations,
        });
        Ok(())
    }
}

/// A stated walk's rows, carved from a rows state's allocator: every row's mix, branch and
/// routing, and a [CHUNK]'s attention and expert scratch.
pub(super) struct StatedRows {
    mix: ManifoldRows,
    pub(super) scratch: TiledScratch,
    /// The sublayer's f32 output before it narrows to the BF16 wire, `[rows][hidden]`.
    pub(super) branch: Deviceptr,
    pub(super) absorbed: Option<AbsorbedRows>,
    pub(super) pooled: Option<PooledRows>,
    pub(super) router: RouterRows,
    /// The bodies' gate, up and activation rows, a chunk's.
    pub(super) gate: Deviceptr,
    pub(super) up: Deviceptr,
    pub(super) act: Deviceptr,
    /// The routed experts' held outputs, a chunk's `[rows][top_k][hidden]`.
    pub(super) expert_out: Deviceptr,
    /// The shared expert's held output, `[rows][hidden]` (zero where a stack has none).
    pub(super) shared: Deviceptr,
    /// The stage a KDA projection's packed rows are placed from.
    pub(super) stage: Deviceptr,
    /// The head's folded and normed rows, f32.
    pub(super) head: Deviceptr,
}

impl StatedRows {
    /// The rows of a `d`-row state.
    ///
    /// # Errors
    /// No stated device was attached, or an allocation failed.
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        d: usize,
        alloc: &mut impl FnMut(usize) -> Result<u64, String>,
    ) -> Result<Self, String> {
        let stated = device
            .stated
            .as_ref()
            .ok_or("rows: a stated stack's rows state opened before its device state")?;
        let widths = stated.widths;
        let dim = device.dim;
        let chunk = d.min(CHUNK);
        let mut f32s = |count: usize| alloc(count * 4).map(Deviceptr);
        let (partials, counters) = TiledScratch::gemv_bytes();
        let values = d * widths.columns;
        let (code_bytes, code_scale_bytes) = TiledScratch::code_bytes(values.next_multiple_of(ACTIVATION_TILE));
        let scratch = TiledScratch {
            partials: f32s(partials.div_ceil(4))?,
            counters: f32s(counters.div_ceil(4))?,
            codes: f32s(code_bytes.div_ceil(4))?,
            code_scales: f32s(code_scale_bytes.div_ceil(4))?,
            code_capacity: values,
        };
        let mix = ManifoldRows {
            raw: f32s(d * stated.mixer.mixes())?,
            coeffs: f32s(d * stated.mixer.mixes())?,
        };
        let absorbed = match stated.absorbed {
            Some(body) => {
                let AbsorbedGeometry {
                    heads,
                    nope,
                    value,
                    latent,
                    q_lora,
                } = body.geometry_of();
                Some(AbsorbedRows {
                    q_latent: f32s(chunk * q_lora)?,
                    latent: f32s(chunk * latent)?,
                    query: f32s(chunk * heads * nope)?,
                    absorbed: f32s(chunk * heads * latent)?,
                    logits: f32s(chunk * heads * MAX_SELECTED_TOKENS)?,
                    total: f32s(chunk * heads)?,
                    mixed: f32s(chunk * heads * latent)?,
                    heads: f32s(chunk * heads * value)?,
                })
            }
            None => None,
        };
        let pooled = match stated.indexer {
            Some(body) => {
                let PooledGeometry {
                    heads, head_dim, ..
                } = body.geometry_of();
                Some(PooledRows {
                    keys: f32s(chunk * head_dim)?,
                    gates: f32s(chunk * head_dim)?,
                    head_weights: f32s(chunk * heads)?,
                    queries: f32s(chunk * heads * head_dim)?,
                    scores: f32s(chunk * body.score_stride(device.max_positions))?,
                    // The selection outlives the chunk: the next block without an indexer reads
                    // every row's.
                    selected: f32s(d * MAX_SELECTED_TOKENS)?,
                    count: f32s(d)?,
                })
            }
            None => None,
        };
        let body = chunk * (widths.top_k * widths.routed_inter).max(widths.mlp);
        Ok(Self {
            mix,
            scratch,
            branch: f32s(d * dim)?,
            absorbed,
            pooled,
            router: RouterRows {
                logits: f32s(d * widths.experts.max(1))?,
                indices: f32s(d * widths.top_k.max(1))?,
                weights: f32s(d * widths.top_k.max(1))?,
            },
            gate: f32s(body)?,
            up: f32s(body)?,
            act: f32s(body)?,
            expert_out: f32s(chunk * widths.top_k.max(1) * dim)?,
            shared: f32s(chunk * dim)?,
            stage: f32s(d * widths.kda_part.max(1))?,
            head: f32s(d * dim)?,
        })
    }
}

/// The rows `first..first + rows` of a pass's segments, each piece re-based to that chunk.
fn chunk_segments(segments: &[LaneSegment], first: usize, rows: usize) -> Vec<LaneSegment> {
    segments
        .iter()
        .filter_map(|segment| {
            let start = segment.first.max(first);
            let end = (segment.first + segment.rows).min(first + rows);
            (start < end).then(|| LaneSegment {
                first: start - first,
                rows: end - start,
                position: segment.position + (start - segment.first),
                regions: segment.regions.clone(),
                snapshots: Vec::new(),
            })
        })
        .collect()
}

impl DeviceLayer<'_> {
    pub(super) fn stated_share(&self) -> Result<Option<sparkflash_model::TpPlan>, String> {
        let world = self.routed_geometry.world;
        (world > 1)
            .then(|| sparkflash_model::TpPlan::new(world, self.tp_rank))
            .transpose()
            .map_err(|error| error.to_string())
    }

    /// The absorbed block's lane regions for every segment of rows p0..p0+n: the attention's
    /// (latent cache, scales) and the indexer's (keys, gates, pooled keys).
    fn absorbed_segments(
        &self,
        layer: usize,
        (p0, n): (usize, usize),
    ) -> Result<(Vec<LaneSegment>, Vec<LaneSegment>), String> {
        let lanes = self.lanes.as_ref().ok_or_else(|| {
            format!("rows layer {layer}: an absorbed block keeps its caches in the lanes, and none are attached")
        })?;
        let mut attention = Vec::new();
        let mut indexer = Vec::new();
        for seg in self.step_segs(p0, n) {
            let lane = if seg.lane == super::lanes::NO_LANE {
                lanes.selected
            } else {
                seg.lane
            };
            let regions = lanes
                .pointers(lane)
                .absorbed
                .get(layer)
                .copied()
                .flatten()
                .ok_or_else(|| format!("rows layer {layer}: lane {lane} holds no absorbed caches"))?
                .map(Deviceptr);
            let piece = |regions: &[Deviceptr]| LaneSegment {
                first: seg.row0,
                rows: seg.n,
                position: seg.p0,
                regions: regions.to_vec(),
                snapshots: Vec::new(),
            };
            attention.push(piece(&regions[..2]));
            indexer.push(piece(&regions[2..]));
        }
        Ok((attention, indexer))
    }

    /// The head's input in a stated stack: the first `n` rows' streams folded by their mean and
    /// normed with the producer's two rounds ([ManifoldMixer::head]), narrowed to the BF16 rows
    /// the LM head reads (exact: the norm's output is held).
    pub(super) fn stated_head(
        &self,
        rows: &RowsState,
        norm_weight: Deviceptr,
        eps: f32,
        n: usize,
    ) -> Result<(), String> {
        let stated = self.stated.as_ref().ok_or("rows head: no stated device")?;
        let state = rows.stated.as_ref().ok_or("rows head: no stated rows")?;
        let b = rows.b;
        let cx = RowsCx::<TiledLinear<'_>>::new(self.cuda, n, None);
        stated
            .mixer
            .head(&cx, (Deviceptr(b.stream), norm_weight, state.head), (eps, true))
            .map_err(|error| format!("rows head: {error}"))?;
        cx.call(
            stated.token[3],
            calls::narrow(b.lm_normed, state.head.0, n * self.dim),
        )
        .map_err(|error| format!("rows head: {error}"))
    }

    /// Layer `layer`'s stated block over rows p0..p0+n: the attention sublayer, then the FFN's,
    /// each between its mix's entry and leave with the base's reduce between.
    pub(super) fn run_block_stated(
        &mut self,
        rows: &RowsState,
        layer: usize,
        block: StatedBlock,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        if self.sp_rows(rows, n).is_some() {
            return Err(format!(
                "rows layer {layer}: a split chunk reached a stated block; the walk projects whole rows (enable_prefill_sp)"
            ));
        }
        let stated = self
            .stated
            .as_ref()
            .ok_or_else(|| format!("rows layer {layer}: no stated device was attached"))?;
        let state = rows
            .stated
            .as_ref()
            .ok_or_else(|| format!("rows layer {layer}: the rows state carved no stated rows"))?;
        let tensors = *stated
            .layers
            .get(layer)
            .ok_or_else(|| format!("rows layer {layer}: past the stated layers"))?;
        if tensors.block != block {
            return Err(format!(
                "rows layer {layer}: the walk was asked for {block:?} and attached {:?}",
                tensors.block
            ));
        }
        let what = |error: sparkflash_model::backend::BackendError| {
            format!("rows layer {layer}: {error}")
        };
        let share = self.stated_share()?;
        let b = rows.b;
        let dim = self.dim;
        let eps = crate::plan::description().values().norm_eps as f32;
        let linear = |matrix: TiledMatrix| TiledLinear {
            body: &stated.tiled,
            scratch: &state.scratch,
            matrix,
            exact: true,
            held: true,
            stage: None,
        };
        let cx = RowsCx::<TiledLinear<'_>>::new(self.cuda, n, share);
        for (sublayer, mix) in tensors.mixes.iter().enumerate() {
            let projection = TiledLinear {
                held: false,
                ..linear(mix.projection)
            };
            stated
                .mixer
                .enter(
                    &cx,
                    &ManifoldWeights {
                        projection: &projection,
                        scale: mix.scale,
                        base: mix.base,
                    },
                    state.mix,
                    (Deviceptr(b.stream), mix.norm, Deviceptr(b.normed)),
                    (eps, true),
                )
                .map_err(what)?;
            if sublayer == 0 {
                self.stated_mixer(rows, (stated, state), &tensors, layer, (p0, n))?;
                cx.call(
                    stated.token[3],
                    calls::narrow(b.attn_out, state.branch.0, n * dim),
                )
                .map_err(what)?;
            } else {
                self.stated_ffn((stated, state), &tensors.ffn, b, layer, n, share)?;
            }
            self.reduce_layer_rows(
                rows,
                "all_reduce",
                span(b.attn_out, n * dim * 2),
                span(b.reduced, n * dim * 2),
                n,
            )?;
            stated
                .mixer
                .leave(
                    &cx,
                    (Deviceptr(b.stream), Deviceptr(b.reduced), state.mix.coeffs),
                    (true, true),
                )
                .map_err(what)?;
        }
        if let Some((kernel, features, ref targets, feature_width)) = rows.dflash_tap
            && let Some(slot) = targets.iter().position(|&id| id == layer)
        {
            self.cuda.launch(
                kernel,
                (dim.div_ceil(256) as u32, n as u32, 1),
                (256, 1, 1),
                &[
                    b.stream,
                    features.0,
                    dim as u64,
                    self.hc_mult as u64,
                    feature_width as u64,
                    slot as u64,
                ],
            )?;
        }
        Ok(())
    }

    /// The token mixer over the normed rows into the f32 branch.
    fn stated_mixer(
        &self,
        rows: &RowsState,
        (stated, state): (&StatedDevice, &StatedRows),
        tensors: &StatedLayer,
        layer: usize,
        (p0, n): (usize, usize),
    ) -> Result<(), String> {
        let what = |error: sparkflash_model::backend::BackendError| {
            format!("rows layer {layer}: {error}")
        };
        let share = self.stated_share()?;
        let b = rows.b;
        let eps = crate::plan::description().values().norm_eps as f32;
        let linear = |matrix: TiledMatrix| TiledLinear {
            body: &stated.tiled,
            scratch: &state.scratch,
            matrix,
            exact: true,
            held: true,
            stage: None,
        };
        let normed = Deviceptr(b.normed);
        match (&tensors.mixer, tensors.block.mixer) {
            (MixerTensors::Kda(kda), StatedMixer::Kda) => {
                let (recurrent, segments) =
                    self.recurrent_segments(rows, layer, (p0, n), "Kimi Delta Attention")?;
                let RecurrentBody::Kda {
                    chunk,
                    rows: scratch,
                    conv: joined,
                    lower_bound,
                } = &recurrent.body
                else {
                    return Err(format!(
                        "rows layer {layer}: a Kimi Delta Attention block in a stack whose recurrent rows are a Gated DeltaNet's"
                    ));
                };
                let geometry = recurrent.geometry;
                let mut at = 0;
                for (conv, rows) in kda.conv.iter().zip([
                    geometry.key_width(),
                    geometry.key_width(),
                    geometry.value_width(),
                ]) {
                    let bytes = rows * geometry.taps * 2;
                    self.cuda.copy_dtod_async(&joined.offset(at), conv, bytes)?;
                    at += bytes;
                }
                let staged = |matrix: TiledMatrix| TiledLinear {
                    stage: Some((stated.token[0], state.stage)),
                    ..linear(matrix)
                };
                let qkv = kda.qkv.map(staged);
                let forget = (linear(kda.forget.0), linear(kda.forget.1));
                let gate = (linear(kda.gate.0), linear(kda.gate.1));
                let (write, out) = (linear(kda.b), linear(kda.out));
                KdaLayer {
                    qkv: [&qkv[0], &qkv[1], &qkv[2]],
                    forget: (&forget.0, &forget.1),
                    gate: (&gate.0, &gate.1),
                    b: &write,
                    out: &out,
                    conv: *joined,
                    decay: kda.decay,
                    norm: kda.norm,
                    hold: Some(stated.tiled.rounding()),
                }
                .run(
                    &RowsCx::new(self.cuda, n, share).with_segments(&segments),
                    (chunk, recurrent.scan),
                    &geometry,
                    scratch,
                    (normed, state.branch),
                    (eps, *lower_bound),
                )
                .map_err(what)
            }
            (MixerTensors::Absorbed(absorbed), StatedMixer::Absorbed { own_indexer }) => {
                if own_indexer != absorbed.indexer.is_some() {
                    return Err(format!(
                        "rows layer {layer}: the block's indexer and its attached tensors disagree"
                    ));
                }
                let (attention, index) = self.absorbed_segments(layer, (p0, n))?;
                self.stated_absorbed_exec(
                    (stated, state),
                    absorbed,
                    (&attention, &index),
                    normed,
                    n,
                    &format!("rows layer {layer}"),
                    true,
                )
            }
            (_, mixer) => Err(format!(
                "rows layer {layer}: the walk was asked for {mixer:?} and attached another mixer's tensors"
            )),
        }
    }

    /// Runs absorbed MLA cache, optional pooled indexer cache+pick, and optional attend over
    /// `normed` (`[n][dim]` f32), shared by backbone MLA layers and GLM's MTP draft stack.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stated_absorbed_exec(
        &self,
        (stated, state): (&StatedDevice, &StatedRows),
        absorbed: &AbsorbedTensors,
        (attention, index): (&[LaneSegment], &[LaneSegment]),
        normed: Deviceptr,
        n: usize,
        label: &str,
        attend: bool,
    ) -> Result<(), String> {
        let what = |error: sparkflash_model::backend::BackendError| format!("{label}: {error}");
        let share = self.stated_share()?;
        let dim = self.dim;
        let eps = crate::plan::description().values().norm_eps as f32;
        let linear = |matrix: TiledMatrix| TiledLinear {
            body: &stated.tiled,
            scratch: &state.scratch,
            matrix,
            exact: true,
            held: true,
            stage: None,
        };
        let body = stated
            .absorbed
            .ok_or_else(|| format!("{label}: no absorbed attention was opened"))?;
        let scratch = state
            .absorbed
            .ok_or_else(|| format!("{label}: no absorbed rows were carved"))?;
        let selection = state
            .pooled
            .ok_or_else(|| format!("{label}: an absorbed block with no indexer rows to select by"))?;
        let (q_a, kv_a, q_b, out) = (
            linear(absorbed.q_a),
            linear(absorbed.kv_a),
            linear(absorbed.q_b),
            linear(absorbed.out),
        );
        let weights = AbsorbedWeights {
            q_a: &q_a,
            kv_a: &kv_a,
            q_b: &q_b,
            out: &out,
            norms: absorbed.norms,
            kv_b: absorbed.kv_b,
            kv_b_bf16: absorbed.kv_b_bf16,
        };
        let indexer = absorbed.indexer.map(|tensors| {
            (
                linear(tensors.keys),
                linear(tensors.gates),
                linear(tensors.head_weights),
                linear(tensors.queries),
                tensors,
            )
        });
        let key_eps = crate::plan::description().values().index_key_norm_eps as f32;
        for first in (0..n).step_by(CHUNK) {
            let count = CHUNK.min(n - first);
            let input = normed.offset(first * dim * 4);
            let selected = selection.selected.offset(first * MAX_SELECTED_TOKENS * 4);
            let picked = selection.count.offset(first * 4);
            let attention = chunk_segments(attention, first, count);
            let cx = RowsCx::new(self.cuda, count, share).with_segments(&attention);
            body.cache(&cx, &weights, &scratch, input, (eps, stated.e4m3_latent, true))
                .map_err(what)?;
            if let Some((keys, gates, head_weights, queries, tensors)) = indexer.as_ref() {
                let picker = stated
                    .indexer
                    .ok_or_else(|| format!("{label}: no pooled indexer was opened"))?;
                let index = chunk_segments(index, first, count);
                let cx = RowsCx::new(self.cuda, count, share).with_segments(&index);
                let pooled_weights = PooledWeights {
                    keys,
                    gates,
                    head_weights,
                    queries,
                    key_norm: tensors.key_norm,
                    ape: tensors.ape,
                };
                let pooled_rows = PooledRows {
                    selected,
                    count: picked,
                    ..selection
                };
                picker
                    .cache(&cx, &pooled_weights, &pooled_rows, input, (key_eps, true))
                    .map_err(what)?;
                if attend {
                    picker
                        .pick(
                            &cx,
                            &pooled_weights,
                            &pooled_rows,
                            scratch.q_latent,
                            (
                                picker.score_stride(self.max_positions),
                                usize::MAX,
                                true,
                            ),
                        )
                        .map_err(what)?;
                }
            }
            if attend {
                body.attend(
                    &cx,
                    &weights,
                    &scratch,
                    (selected, picked),
                    state.branch.offset(first * dim * 4),
                    (stated.e4m3_latent, true),
                )
                .map_err(what)?;
            }
        }
        Ok(())
    }

    /// The FFN over the normed rows into the BF16 rows the reduce sums (`attn_out`).
    pub(super) fn stated_ffn<'s>(
        &self,
        (stated, state): (&'s StatedDevice, &'s StatedRows),
        ffn: &FfnTensors,
        b: super::rows::RowsBuffers,
        layer: usize,
        n: usize,
        share: Option<sparkflash_model::TpPlan>,
    ) -> Result<(), String> {
        let what = |error: sparkflash_model::backend::BackendError| {
            format!("rows layer {layer}: {error}")
        };
        let dim = self.dim;
        let limit = crate::plan::description().values().swiglu_limit as f32;
        let [_, swiglu, combine, narrow] = stated.token;
        let linear = |matrix: TiledMatrix, held: bool| TiledLinear::<'s> {
            body: &stated.tiled,
            scratch: &state.scratch,
            matrix,
            exact: true,
            held,
            stage: None,
        };
        // One gated MLP over `count` rows from `input` into `output` (f32, held).
        let mlp = |cx: &RowsCx<'_, TiledLinear<'s>>,
                   body: &Mlp,
                   input: Deviceptr,
                   output: Deviceptr,
                   count: usize| {
            use sparkflash_ops::roles::DeviceCx;
            let (gate, up, down) = (
                linear(body.gate, true),
                linear(body.up, true),
                linear(body.down, true),
            );
            cx.project(&gate, input, state.gate, count)?;
            cx.project(&up, input, state.up, count)?;
            cx.call(
                swiglu,
                calls::swiglu_held(
                    state.gate.0,
                    state.up.0,
                    state.act.0,
                    count * body.gate.rows,
                    limit,
                ),
            )?;
            cx.project(&down, state.act, output, count)
        };
        let normed = Deviceptr(b.normed);
        match ffn {
            FfnTensors::Dense(body) => {
                for first in (0..n).step_by(CHUNK) {
                    let count = CHUNK.min(n - first);
                    let cx = RowsCx::new(self.cuda, count, share);
                    mlp(
                        &cx,
                        body,
                        normed.offset(first * dim * 4),
                        state.branch.offset(first * dim * 4),
                        count,
                    )
                    .map_err(what)?;
                }
                RowsCx::<TiledLinear<'_>>::new(self.cuda, n, share)
                    .call(narrow, calls::narrow(b.attn_out, state.branch.0, n * dim))
                    .map_err(what)
            }
            FfnTensors::Moe {
                router,
                bias,
                tables,
                shared,
            } => {
                let widths = stated.widths;
                let (experts, top_k, inter) = (widths.experts, widths.top_k, widths.routed_inter);
                let values = crate::plan::description().values();
                let cx = RowsCx::new(self.cuda, n, share);
                {
                    use sparkflash_ops::roles::DeviceCx;
                    cx.project(&linear(*router, false), normed, state.router.logits, n)
                        .map_err(what)?;
                }
                stated
                    .router
                    .run(
                        &cx,
                        RouterRule::NoauxAscending {
                            bias: *bias,
                            renormalise: values.norm_topk_prob,
                            scaling: values.route_scale as f32,
                        },
                        &state.router,
                        (experts, top_k),
                        n,
                    )
                    .map_err(what)?;
                let rounding = stated.tiled.rounding();
                let quantise = u64::from(stated.quantise);
                for first in (0..n).step_by(CHUNK) {
                    let count = CHUNK.min(n - first);
                    let cx = RowsCx::new(self.cuda, count, share);
                    let input = normed.offset(first * dim * 4);
                    let indices = state.router.indices.offset(first * top_k * 4);
                    // gate and up on the token's input, down on the pair's activation; each
                    // output held, as the producer's FP8 experts return BF16.
                    let expert = |table: Deviceptr,
                                  (x, y): (Deviceptr, Deviceptr),
                                  (outputs, inputs, per_slot): (usize, usize, u64)| {
                        for group in (0..count).step_by(EXPERT_TOKENS) {
                            let tokens = EXPERT_TOKENS.min(count - group);
                            let pairs = tokens * top_k;
                            let x_row = if per_slot == 0 {
                                group * inputs
                            } else {
                                group * top_k * inputs
                            };
                            cx.launch(
                                stated.expert,
                                (
                                    (outputs.div_ceil(EXPERT_TILE) as u32, 1, pairs as u32),
                                    (EXPERT_THREADS, 1, 1),
                                ),
                                &[
                                    table.0,
                                    indices.offset(group * top_k * 4).0,
                                    0,
                                    x.offset(x_row * 4).0,
                                    y.offset(group * top_k * outputs * 4).0,
                                    outputs as u64,
                                    inputs as u64,
                                    top_k as u64,
                                    per_slot,
                                    quantise,
                                ],
                            )?;
                        }
                        sparkflash_ops::roles::round_bf16(&cx, rounding, y, count * top_k * outputs)
                    };
                    expert(tables[0], (input, state.gate), (inter, dim, 0)).map_err(what)?;
                    expert(tables[1], (input, state.up), (inter, dim, 0)).map_err(what)?;
                    cx.call(
                        swiglu,
                        calls::swiglu_held(
                            state.gate.0,
                            state.up.0,
                            state.act.0,
                            count * top_k * inter,
                            limit,
                        ),
                    )
                    .map_err(what)?;
                    expert(tables[2], (state.act, state.expert_out), (dim, inter, 1))
                        .map_err(what)?;
                    if let Some(body) = shared {
                        mlp(&cx, body, input, state.shared, count).map_err(what)?;
                    }
                    cx.call(
                        combine,
                        calls::combine_held(
                            (
                                state.expert_out.0,
                                state.router.weights.offset(first * top_k * 4).0,
                                state.shared.0,
                            ),
                            b.attn_out + (first * dim * 2) as u64,
                            (count, top_k, dim),
                        ),
                    )
                    .map_err(what)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_kernels::checkpoint::Checkpoint;
    use sparkflash_kernels::lane_memory::{ChunkDevice, ChunkPool, LaneSpan, Layout};
    use std::cell::RefCell;

    const GLM_CONFIG: &str = include_str!("../../../../models/configs/glm53_flash.hf.json");
    const GRANULARITY: usize = 2 << 20;
    const ROW_QUANTUM: usize = 2048;
    const ONE_MILLION: usize = 1_048_576;
    const DEFAULT_LANE_BUDGET: usize = 24 << 30;

    #[derive(Default)]
    struct MockDevice {
        next: RefCell<u64>,
        copies: RefCell<Vec<(u64, u64, usize)>>,
        unmaps: RefCell<Vec<u64>>,
    }

    impl ChunkDevice for MockDevice {
        fn create(&self, _bytes: usize) -> Result<u64, String> {
            let mut next = self.next.borrow_mut();
            *next += 1;
            Ok(*next)
        }
        fn release(&self, _handle: u64) {}
        fn map(&self, _at: u64, _bytes: usize, _handle: u64) -> Result<(), String> {
            Ok(())
        }
        fn unmap(&self, at: u64, _bytes: usize) -> Result<(), String> {
            self.unmaps.borrow_mut().push(at);
            Ok(())
        }
        fn zero(&self, _at: u64, _bytes: usize) -> Result<(), String> {
            Ok(())
        }
        fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
            self.copies.borrow_mut().push((dst, src, bytes));
            Ok(())
        }
    }

    fn glm_description(world: usize) -> crate::plan::Description {
        let plan = sparkflash_ops::engine_plan::EnginePlan::from_json(
            GLM_CONFIG,
            sparkflash_model::Target::Sm121,
            world,
            0,
        )
        .expect("the real GLM-5.3-Flash plan builds");
        crate::plan::Description::new(plan).expect("the real GLM-5.3-Flash description builds")
    }

    fn build_glm_lane_shape(
        description: &crate::plan::Description,
        max_positions: usize,
        e4m3_latent: bool,
        world: usize,
    ) -> (
        sparkflash_kernels::lane_memory::LaneShape,
        Vec<(usize, [Option<usize>; super::super::lanes::ABSORBED_REGIONS])>,
        usize,
    ) {
        let mut layout = Layout::default();
        let plans = absorbed_regions_for(description, e4m3_latent).expect("absorbed regions");
        let mut offsets = Vec::with_capacity(plans.len());
        for plan in plans {
            let reg = super::super::lanes::plan_absorbed_regions(&mut layout, max_positions, plan);
            offsets.push((plan.layer, reg));
        }
        let values = description.values();
        let kda_layers = values.kda_layers.len();
        let kda = values.kda.expect("GLM states KDA geometry");
        let heads = kda.heads / world;
        let conv_width = heads * kda.head_dim * 3;
        let history_bytes = conv_width * kda.conv.saturating_sub(1) * 4;
        let state_bytes = heads * kda.head_dim * kda.head_dim * 4;
        let mut recurrent_sizes = Vec::with_capacity(kda_layers * 2);
        for _ in 0..kda_layers {
            recurrent_sizes.push(history_bytes);
            recurrent_sizes.push(state_bytes);
            layout.state(history_bytes);
            layout.state(state_bytes);
        }
        let checkpoint_bytes = Checkpoint::arena_bytes(&recurrent_sizes);
        layout.state(checkpoint_bytes);
        let recurrent_total = recurrent_sizes.iter().sum::<usize>() + checkpoint_bytes;
        (
            layout.finish(GRANULARITY, ROW_QUANTUM),
            offsets,
            recurrent_total,
        )
    }

    #[test]
    fn a2_e4m3_latent_cache_fits_1m_window_with_slide_and_prefix_adoption() {
        let description = glm_description(4);
        let plans_fp8 = absorbed_regions_for(&description, true).expect("fp8 plans");
        let plans_f32 = absorbed_regions_for(&description, false).expect("f32 plans");
        assert_eq!(plans_fp8.len(), 11, "GLM-5.3-Flash has 11 sparse MLA layers");
        assert_eq!(
            plans_fp8.iter().filter(|p| p.indexer.is_some()).count(),
            11,
            "all 11 sparse language layers in GLM-5.3-Flash own a pooled indexer (MTP shares the last)"
        );
        for (fp8, f32_plan) in plans_fp8.iter().zip(&plans_f32) {
            assert_eq!(fp8.latent_row_bytes, 512);
            assert_eq!(fp8.scale_row_bytes, Some(16));
            assert_eq!(f32_plan.latent_row_bytes, 2048);
            assert_eq!(f32_plan.scale_row_bytes, None);
        }

        let (shape_fp8, offsets, recurrent_bytes) =
            build_glm_lane_shape(&description, ONE_MILLION, true, 4);
        let (shape_f32, _, _) = build_glm_lane_shape(&description, ONE_MILLION, false, 4);

        let floor_1m = super::super::lanes::slide_floor(ONE_MILLION);
        let fp8_1m_bytes = shape_fp8.chunks_for(ONE_MILLION, floor_1m) * GRANULARITY;
        let f32_1m_bytes = shape_f32.chunks_for(ONE_MILLION, floor_1m) * GRANULARITY;
        eprintln!(
            "GLM-5.3-Flash TP4 lane @ 1M (floor {floor_1m}): E4M3={:.2} GiB, F32={:.2} GiB, 34-KDA state+ckpt={:.2} MiB, stride_fp8={:.2} GiB",
            fp8_1m_bytes as f64 / (1u64 << 30) as f64,
            f32_1m_bytes as f64 / (1u64 << 30) as f64,
            recurrent_bytes as f64 / (1u64 << 20) as f64,
            shape_fp8.stride as f64 / (1u64 << 30) as f64,
        );
        assert!(
            fp8_1m_bytes < 8 << 30,
            "1M E4M3 window should stay under 8 GiB, got {} MiB (recurrent {} MiB)",
            fp8_1m_bytes >> 20,
            recurrent_bytes >> 20,
        );
        assert!(fp8_1m_bytes < DEFAULT_LANE_BUDGET / 3);
        assert!(
            f32_1m_bytes > fp8_1m_bytes * 3,
            "f32 latent should be >3x E4M3 ({} MiB vs {} MiB)",
            f32_1m_bytes >> 20,
            fp8_1m_bytes >> 20,
        );

        let device = MockDevice::default();
        let mut pool = ChunkPool::new(GRANULARITY, DEFAULT_LANE_BUDGET);
        let mut lane0 = LaneSpan::new(0x1000_0000, &shape_fp8);
        let mut lane1 = LaneSpan::new(0x2000_0000, &shape_fp8);

        let rows = 65_536;
        assert!(lane0.ensure(&device, &shape_fp8, &mut pool, rows).expect("grow"));
        let mapped_before = lane0.mapped();
        let floor = super::super::lanes::slide_floor(rows);
        assert!(floor > 0);
        let reclaimed = lane0
            .slide(&device, &shape_fp8, &mut pool, floor)
            .expect("slide");
        assert!(
            reclaimed > 0 && lane0.mapped() < mapped_before,
            "sliding raw indexer keys/gates must reclaim physical chunks (reclaimed {reclaimed})"
        );
        assert_eq!(device.unmaps.borrow().len(), reclaimed);

        device.copies.borrow_mut().clear();
        assert!(
            lane1
                .copy_prefix(&device, &shape_fp8, &mut pool, &lane0, rows, floor)
                .expect("copy_prefix")
        );
        let copies = device.copies.borrow();
        let first_layer_regions = offsets[0].1;
        let latent_off = first_layer_regions[0].expect("latent");
        let scales_off = first_layer_regions[1].expect("scales");
        assert!(
            copies
                .iter()
                .any(|&(dst, _, len)| dst == lane1.base() + latent_off as u64 && len == rows * 512),
            "prefix adoption must copy the E4M3 latent codes"
        );
        assert!(
            copies
                .iter()
                .any(|&(dst, _, len)| dst == lane1.base() + scales_off as u64 && len == rows * 16),
            "prefix adoption must copy the E4M3 block scales"
        );
        assert_eq!(copies.len(), 11 * 2 + 11 * 3);
        lane0.release(&device, &shape_fp8, &mut pool).expect("release 0");
        lane1.release(&device, &shape_fp8, &mut pool).expect("release 1");
    }

    #[test]
    fn a2_shared_indexer_layer_allocates_no_indexer_regions() {
        const TINY: &str = include_str!("../../../sparkflash-model-deepseek-v41/tests/fixtures/glm-tiny/config.json");
        let plan = sparkflash_ops::engine_plan::EnginePlan::from_json(
            TINY,
            sparkflash_model::Target::Sm121,
            1,
            0,
        )
        .expect("glm-tiny plan");
        let description = crate::plan::Description::new(plan).expect("glm-tiny description");
        let plans = absorbed_regions_for(&description, true).expect("tiny absorbed plans");
        assert_eq!(plans.len(), 3);
        assert!(plans[0].indexer.is_some(), "L3 owns an indexer");
        assert!(plans[1].indexer.is_some(), "L6 owns an indexer");
        assert!(plans[2].indexer.is_none(), "L7 shares L6's selection");
    }

    #[test]
    fn a2_profile_switch_selects_e4m3_by_default_and_f32_when_asked() {
        assert!(latent_cache_fp8_for(None));
        assert!(latent_cache_fp8_for(Some(sparkflash_model::profile::LatentCacheMode::Auto)));
        assert!(latent_cache_fp8_for(Some(sparkflash_model::profile::LatentCacheMode::Fp8)));
        assert!(!latent_cache_fp8_for(Some(sparkflash_model::profile::LatentCacheMode::F32)));
    }

    #[test]
    fn k4_kda_interleaves_prompt_chunks_while_gdn_stays_refused() {
        assert!(crate::plan::interleaves(sparkflash_model::Part::KdaScan));
        assert!(!crate::plan::interleaves(sparkflash_model::Part::GdnScan));
        assert!(!crate::plan::interleaves(sparkflash_model::Part::GqaAttention));

        let description = glm_description(4);
        let non_interleaving = description
            .plan()
            .stacks()
            .iter()
            .filter(|stack| stack.kind == sparkflash_model::StackKind::Language)
            .flat_map(|stack| &stack.blocks)
            .find_map(|block| {
                block
                    .parts
                    .iter()
                    .find(|instance| !crate::plan::interleaves(instance.part))
                    .map(|instance| (block.index, instance.part))
            });
        assert_eq!(non_interleaving, None, "every GLM language block interleaves");
    }
}
