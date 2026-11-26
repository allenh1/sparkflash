//! Stated-stack drafting on the base (docs/engine/08-glm-on-the-base.md D1–D3):
//!
//! - D1: GLM's MTP layer (`mtp.0.*`) run as a stated draft stack using the base's own
//!   `AbsorbedMla`, `PooledIndexer`, `Router`, and `stated_ffn` bodies.
//! - D2: DFlash 2 drafter (`sparkflash_ops::dflash::GpuDflash2`) attached when
//!   `active_profile().speculation.drafter` (`SPARKFLASH_GLM_DFLASH2`) names a drafter directory.
//! - D3: E4M3 copy of this rank's BF16 LM head slice (`GpuMatE4m3::quantize`) shared by DFlash 2
//!   and (when `drafter.head_fp8()` is on) GLM's MTP layer.

use super::draft::{DraftBlock, Drafts};
use super::rows::{LaneSegment, RowsCx, RowsState, TiledLinear, span};
use super::stated::{
    ACTIVATION_TILE, AbsorbedTensors, FfnTensors, IndexerTensors, Mlp, matrix, slice,
};
use super::*;
use sparkflash_kernels::cuda_driver::Kernel;
use sparkflash_kernels::images::Image;
use sparkflash_model::backend::BackendError;
use sparkflash_model::dflash2::DraftTarget;
use sparkflash_ops::calls;
use sparkflash_ops::dflash::{DraftDevice, DraftScratch, GpuDflash2};
use sparkflash_ops::gemv::{
    GemvDevice, GpuMat, GpuMatE4m3, TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS, gemv_e4m3,
};
use sparkflash_ops::plan::Precision;
use sparkflash_ops::roles::{AbsorbedGeometry, KernelTable, Launcher};
use sparkflash_ops::tiled::TiledMatrix;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::Path;

/// Quantises this rank's BF16 `lm_head.weight` slice in `arena` to a row-scaled E4M3 matrix (D3).
fn quantize_lm_head_e4m3(
    cuda: &Cuda,
    arena: &sparkflash_device::arena::ResidentArena,
    quantize_kernel: Kernel,
    local_vocab: usize,
    dim: usize,
    allocations: &mut Vec<Deviceptr>,
) -> Result<GpuMatE4m3, String> {
    let lm_head = slice(arena, "head.weight")?;
    let expected = local_vocab * dim * 2;
    if lm_head.bytes != expected {
        return Err(format!(
            "stated draft head: head.weight holds {} bytes, not [{local_vocab}, {dim}] BF16",
            lm_head.bytes
        ));
    }
    let mat = GpuMat {
        pointer: lm_head.ptr,
        rows: local_vocab,
        columns: dim,
    };
    let mut bytes = 0usize;
    let head = GpuMatE4m3::quantize(cuda, quantize_kernel, &mat, &mut bytes)
        .map_err(|error| format!("stated draft head: {error}"))?;
    allocations.push(head.weight);
    allocations.push(head.scales);
    Ok(head)
}

/// Resolves the full `[vocab, dim]` BF16 embedding table on device: either directly from the
/// arena when the layout replicates `embed.weight` (`split = "whole"`), or from
/// `device.embed_table` when replicated during open.
fn resolve_embed_table(
    device: &DeviceLayer<'_>,
    arena: &sparkflash_device::arena::ResidentArena,
) -> Result<Deviceptr, String> {
    let values = crate::plan::description().values();
    let whole_bytes = values.vocab_size * values.dim * 2;
    if let Some(s) = arena.slice("embed.weight")
        && s.bytes == whole_bytes
    {
        return Ok(s.ptr);
    }
    device
        .embed_table
        .map(Deviceptr)
        .ok_or_else(|| "stated draft: full embed.weight table is not resident on device".to_string())
}

/// All-gathers a `count`-element slice of 32-bit values across `world` ranks through `transport`.
fn all_gather_u32_words(
    transport: &dyn sparkflash_comm::transport::Transport,
    local: &[u32],
    world: usize,
    seq: u64,
) -> Result<Vec<u32>, String> {
    if world <= 1 {
        return Ok(local.to_vec());
    }
    let mut local_u16 = Vec::with_capacity(local.len() * 2);
    for &w in local {
        local_u16.push((w & 0xFFFF) as u16);
        local_u16.push((w >> 16) as u16);
    }
    let mut out_u16 = vec![0u16; local_u16.len() * world];
    transport
        .all_gather_bf16(&local_u16, &mut out_u16, seq)
        .map_err(|error| format!("stated draft gather: {error}"))?;
    let mut out = Vec::with_capacity(local.len() * world);
    for pair in out_u16.chunks_exact(2) {
        out.push(u32::from(pair[0]) | (u32::from(pair[1]) << 16));
    }
    Ok(out)
}

/// Lightweight `GemvDevice` / `DraftDevice` adapter over `DeviceLayer` + `RowsState`.
struct StatedDraftDevice<'a, 'b> {
    device: &'a DeviceLayer<'b>,
    rows: &'a RowsState,
    partials: Deviceptr,
    counters: Deviceptr,
    reduce_in: Deviceptr,
    reduce_out: Deviceptr,
    reduce_f32: Deviceptr,
    embed_kernel: Option<Kernel>,
    add_kernel: Option<Kernel>,
}

impl Launcher for StatedDraftDevice<'_, '_> {
    fn launch(
        &self,
        kernel: impl Into<Kernel>,
        (grid, block): ((u32, u32, u32), (u32, u32, u32)),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.device
            .cuda
            .launch(kernel.into(), grid, block, args)
            .map_err(|error| BackendError::new("stated_draft", error))
    }
}

impl GemvDevice for StatedDraftDevice<'_, '_> {
    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        let stated = self
            .device
            .stated
            .as_ref()
            .ok_or_else(|| BackendError::new("stated_draft", "no stated device"))?;
        let state = self
            .rows
            .stated
            .as_ref()
            .ok_or_else(|| BackendError::new("stated_draft", "no stated rows"))?;
        let cx = RowsCx::<TiledLinear<'_>>::new(self.device.cuda, rows, None);
        let tiled_mat = TiledMatrix::whole(matrix.pointer, None, matrix.rows, matrix.columns);
        stated
            .tiled
            .project(&cx, &state.scratch, &tiled_mat, (input, output), rows, (true, false))
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.partials, self.counters)
    }
}

impl DraftDevice for StatedDraftDevice<'_, '_> {
    fn reduce_add(
        &self,
        partial: Deviceptr,
        residual: Deviceptr,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        let stated = self
            .device
            .stated
            .as_ref()
            .ok_or_else(|| BackendError::new("stated_draft", "no stated device"))?;
        let count = rows * width;
        let cx = RowsCx::<TiledLinear<'_>>::new(self.device.cuda, rows, None);
        cx.call(stated.token[3], calls::narrow(self.reduce_in.0, partial.0, count))?;
        self.device
            .reduce_layer_rows(
                self.rows,
                "dflash_reduce",
                span(self.reduce_in.0, count * 2),
                span(self.reduce_out.0, count * 2),
                rows,
            )
            .map_err(|error| BackendError::new("stated_draft", error))?;
        let embed_kernel = self
            .embed_kernel
            .ok_or_else(|| BackendError::new("stated_draft", "missing q4e_embed"))?;
        let add_kernel = self
            .add_kernel
            .ok_or_else(|| BackendError::new("stated_draft", "missing q4e_add"))?;
        // Convert reduced BF16 back to f32 and accumulate into residual.
        self.launch(
            embed_kernel,
            ((count.div_ceil(256) as u32, 1, 1), (256, 1, 1)),
            &[self.reduce_out.0, self.reduce_f32.0, 0, count as u64, count as u64, 0],
        )?;
        self.launch(
            add_kernel,
            ((count.div_ceil(256) as u32, 1, 1), (256, 1, 1)),
            &[residual.0, self.reduce_f32.0, count as u64],
        )
    }
}

/// D2: DFlash 2 drafter state on the base.
pub(super) struct DFlash2Draft {
    dflash: GpuDflash2,
    embed_table: Deviceptr,
    capture_kernel: Kernel,
    capture_features: Deviceptr,
    max_rows: usize,
    partials: Deviceptr,
    counters: Deviceptr,
    local_cand_v: Deviceptr,
    local_cand_i: Deviceptr,
    gathered_cand_v: Deviceptr,
    gathered_cand_i: Deviceptr,
    reduce_in: Deviceptr,
    reduce_out: Deviceptr,
    reduce_f32: Deviceptr,
    embed_kernel: Kernel,
    add_kernel: Kernel,
    scratches: RefCell<BTreeMap<usize, DraftScratch>>,
    scratch_allocs: RefCell<Vec<Deviceptr>>,
    gather_seq: Cell<u64>,
    allocations: Vec<Deviceptr>,
}

impl DFlash2Draft {
    fn open(
        device: &DeviceLayer<'_>,
        arena: &sparkflash_device::arena::ResidentArena,
        dir: &Path,
        max_rows: usize,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let values = crate::plan::description().values();
        let world = device.routed_geometry.world;
        let rank = device.tp_rank;
        let local_vocab = crate::plan::vocab_per_rank();
        let dim = device.dim;
        let embed_table = resolve_embed_table(device, arena)?;
        let mut allocations = Vec::new();
        let [quantize_e4m3, embed_bound, add_bound] = KernelTable {
            image: Image::TOKEN,
            symbols: [
                ("q4e_quantize_e4m3_rows", Precision::Fp8E4M3),
                ("q4e_embed", Precision::Bf16),
                ("q4e_add", Precision::F32),
            ],
        }
        .load(cuda)?;
        let head = quantize_lm_head_e4m3(
            cuda,
            arena,
            quantize_e4m3.kernel(),
            local_vocab,
            dim,
            &mut allocations,
        )?;
        let target = DraftTarget {
            hidden_size: values.dim,
            num_hidden_layers: values.n_layers,
            vocab_size: values.vocab_size,
        };
        let dflash = GpuDflash2::load(cuda, dir, target, &head, (rank, world))
            .map_err(|error| format!("stated dflash2: {error}"))?;
        let [capture_bound] = KernelTable {
            image: Image::Dflash2,
            symbols: [("q4d_capture", Precision::F32)],
        }
        .load(cuda)?;
        let capture_kernel = capture_bound.kernel();
        let embed_kernel = embed_bound.kernel();
        let add_kernel = add_bound.kernel();
        let mut alloc = |bytes: usize| -> Result<Deviceptr, String> {
            let ptr = cuda.alloc(bytes.max(4), &mut 0)?;
            allocations.push(ptr);
            Ok(ptr)
        };
        let block = dflash.config.block_size;
        let top_k = dflash.config.selector_top_k;
        let cand_elems = block * top_k;
        let capture_features = alloc(max_rows * dflash.config.feature_width() * 4)?;
        let partials = alloc(TILED_GEMV_PARTIALS * 4)?;
        let counters = alloc(TILED_GEMV_BLOCKS * 4)?;
        cuda.memset_d8_async(counters, 0, TILED_GEMV_BLOCKS * 4)?;
        let local_cand_v = alloc(cand_elems * 4)?;
        let local_cand_i = alloc(cand_elems * 4)?;
        let gathered_cand_v = alloc(world * cand_elems * 4)?;
        let gathered_cand_i = alloc(world * cand_elems * 4)?;
        let reduce_in = alloc(block * dim * 2)?;
        let reduce_out = alloc(block * dim * 2)?;
        let reduce_f32 = alloc(block * dim * 4)?;
        Ok(Self {
            dflash,
            embed_table,
            capture_kernel,
            capture_features,
            max_rows,
            partials,
            counters,
            local_cand_v,
            local_cand_i,
            gathered_cand_v,
            gathered_cand_i,
            reduce_in,
            reduce_out,
            reduce_f32,
            embed_kernel,
            add_kernel,
            scratches: RefCell::new(BTreeMap::new()),
            scratch_allocs: RefCell::new(Vec::new()),
            gather_seq: Cell::new(0x5000_0000),
            allocations,
        })
    }

    fn block_size(&self) -> usize {
        self.dflash.config.block_size.saturating_sub(1)
    }

    fn tap_info(&self) -> (Kernel, Deviceptr, Vec<usize>, usize) {
        (
            self.capture_kernel,
            self.capture_features,
            self.dflash.config.target_layer_ids.clone(),
            self.dflash.config.feature_width(),
        )
    }

    fn ensure_scratch(&self, cuda: &Cuda, lane: usize, reset_rings: bool) -> Result<DraftScratch, String> {
        let mut scratches = self.scratches.borrow_mut();
        let scratch = match scratches.get(&lane) {
            Some(&scratch) => scratch,
            None => {
                let mut extra = self.scratch_allocs.borrow_mut();
                let scratch = DraftScratch::allocate(&self.dflash, self.max_rows, |floats| {
                    let ptr = cuda.alloc(floats * 4, &mut 0)?;
                    extra.push(ptr);
                    Ok(ptr)
                })?;
                let (rings, floats) = scratch.rings_region(&self.dflash.config);
                cuda.memset_d8_async(rings, 0, floats * 4)?;
                scratches.insert(lane, scratch);
                scratch
            }
        };
        if reset_rings {
            let (rings, floats) = scratch.rings_region(&self.dflash.config);
            cuda.memset_d8_async(rings, 0, floats * 4)?;
        }
        Ok(scratch)
    }

    fn seed(&self, device: &DeviceLayer<'_>, rows: &RowsState, p0: usize, n: usize) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let dev = StatedDraftDevice {
            device,
            rows,
            partials: self.partials,
            counters: self.counters,
            reduce_in: self.reduce_in,
            reduce_out: self.reduce_out,
            reduce_f32: self.reduce_f32,
            embed_kernel: Some(self.embed_kernel),
            add_kernel: Some(self.add_kernel),
        };
        for seg in device.step_segs(p0, n) {
            let lane = if seg.lane == super::lanes::NO_LANE {
                device.lanes.as_ref().map_or(0, |l| l.selected)
            } else {
                seg.lane
            };
            let scratch = self.ensure_scratch(device.cuda, lane, seg.p0 == 0)?;
            let bound = scratch.with_features(self.capture_features);
            self.dflash
                .commit(&dev, &bound, seg.row0, seg.n, seg.p0)
                .map_err(|error| format!("stated dflash2 commit: {error}"))?;
        }
        Ok(())
    }

    fn draft_blocks(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        transport: &dyn sparkflash_comm::transport::Transport,
        blocks: &[DraftBlock],
        k: usize,
    ) -> Result<Vec<Drafts>, String> {
        let embed = self.embed_table.0;
        let world = device.routed_geometry.world;
        let rank = device.tp_rank;
        let first_id = rank * crate::plan::vocab_per_rank();
        let block_size = self.dflash.config.block_size;
        let top_k = self.dflash.config.selector_top_k;
        let cand_elems = (block_size - 1) * top_k;
        let dev = StatedDraftDevice {
            device,
            rows,
            partials: self.partials,
            counters: self.counters,
            reduce_in: self.reduce_in,
            reduce_out: self.reduce_out,
            reduce_f32: self.reduce_f32,
            embed_kernel: Some(self.embed_kernel),
            add_kernel: Some(self.add_kernel),
        };
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            let scratch = self.ensure_scratch(device.cuda, block.lane, false)?;
            self.dflash
                .score(
                    &dev,
                    &scratch,
                    Deviceptr(embed),
                    block.anchor,
                    block.d,
                    first_id,
                    (self.local_cand_v, self.local_cand_i),
                )
                .map_err(|error| format!("stated dflash2 score: {error}"))?;
            if world > 1 {
                device.cuda.synchronize()?;
                let mut lv_bytes = vec![0u8; cand_elems * 4];
                let mut li_bytes = vec![0u8; cand_elems * 4];
                device.cuda.copy_dtoh(&mut lv_bytes, &self.local_cand_v)?;
                device.cuda.copy_dtoh(&mut li_bytes, &self.local_cand_i)?;
                let lv: Vec<u32> = lv_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let li: Vec<u32> = li_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let seq = self.gather_seq.get();
                self.gather_seq.set(seq.wrapping_add(2));
                let gv = all_gather_u32_words(transport, &lv, world, seq)?;
                let gi = all_gather_u32_words(transport, &li, world, seq.wrapping_add(1))?;
                let gv_bytes: Vec<u8> = gv.iter().flat_map(|w| w.to_le_bytes()).collect();
                let gi_bytes: Vec<u8> = gi.iter().flat_map(|w| w.to_le_bytes()).collect();
                device.cuda.copy_htod(&self.gathered_cand_v, &gv_bytes)?;
                device.cuda.copy_htod(&self.gathered_cand_i, &gi_bytes)?;
                self.dflash
                    .merge(
                        &dev,
                        &scratch,
                        (self.gathered_cand_v, self.gathered_cand_i),
                        world,
                        cand_elems,
                    )
                    .map_err(|error| format!("stated dflash2 merge: {error}"))?;
            } else {
                self.dflash
                    .merge(
                        &dev,
                        &scratch,
                        (self.local_cand_v, self.local_cand_i),
                        1,
                        cand_elems,
                    )
                    .map_err(|error| format!("stated dflash2 merge: {error}"))?;
            }
            self.dflash
                .select(&dev, &scratch, block.anchor)
                .map_err(|error| format!("stated dflash2 select: {error}"))?;
            device.cuda.synchronize()?;
            let mut raw = vec![0u8; 2 * block_size * 4];
            device.cuda.copy_dtoh(&mut raw, &scratch.drafts())?;
            let word = |i: usize| {
                u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
            };
            let ids: Vec<u32> = (1..=k).map(word).collect();
            let probs: Vec<f32> = (1..=k).map(|i| f32::from_bits(word(block_size + i))).collect();
            out.push((ids, probs));
        }
        Ok(out)
    }

    fn release(self, cuda: &Cuda) {
        self.dflash.free(cuda);
        let extra = self.scratch_allocs.into_inner();
        let borrowed_extra: Vec<&Deviceptr> = extra.iter().collect();
        cuda.free_buffers(&borrowed_extra);
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

/// Per-lane caches for GLM's MTP layer (`mtp.0.*`).
#[derive(Clone, Copy)]
struct MtpLane {
    /// `[R_MAX][dim]` f32: backbone normed hidden states for positions `recent_p0..recent_p0+recent_n`.
    /// Holding the whole verify pass's hidden states lets MTP pick the exact accepted row
    /// `block.d - 1` even when a verify step rejects some drafts.
    recent_hiddens: Deviceptr,
    recent_p0: usize,
    recent_n: usize,
    last_hidden: Deviceptr,
    latent: Deviceptr,
    latent_scales: Deviceptr,
    idx_keys: Deviceptr,
    idx_gates: Deviceptr,
    idx_pooled: Deviceptr,
}

/// D1 + D3: GLM's MTP layer (`mtp.0.*`) run as a stated draft stack on the base.
pub(super) struct MtpDraft {
    embed_table: Deviceptr,
    tail_norm: Deviceptr,
    enorm: Deviceptr,
    hnorm: Deviceptr,
    attn_norm: Deviceptr,
    ffn_norm: Deviceptr,
    head_norm: Deviceptr,
    eh_proj: TiledMatrix,
    absorbed: AbsorbedTensors,
    ffn: FfnTensors,
    /// D3: E4M3 copy of this rank's BF16 LM head slice (None when `drafter.head_fp8()` is off).
    head_e4m3: Option<GpuMatE4m3>,
    gemv_e4m3_kernel: Kernel,
    embed_kernel: Kernel,
    rmsnorm_kernel: Kernel,
    add_kernel: Kernel,
    argmax_prob_kernel: Kernel,
    embed_f32: Deviceptr,
    eh_in: Deviceptr,
    mtp_x: Deviceptr,
    cur_hidden: Deviceptr,
    branch_f32: Deviceptr,
    head_logits_local: Deviceptr,
    head_logits_gathered: Deviceptr,
    pick_tok: Deviceptr,
    pick_prob: Deviceptr,
    partials: Deviceptr,
    counters: Deviceptr,
    lanes: RefCell<BTreeMap<usize, MtpLane>>,
    lane_allocs: RefCell<Vec<Deviceptr>>,
    gather_seq: Cell<u64>,
    allocations: Vec<Deviceptr>,
}

impl MtpDraft {
    fn open(
        device: &DeviceLayer<'_>,
        arena: &sparkflash_device::arena::ResidentArena,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let values = crate::plan::description().values();
        let world = device.routed_geometry.world;
        let dim = device.dim;
        let split = |what: &str, count: usize| -> Result<usize, String> {
            if count.is_multiple_of(world) {
                Ok(count / world)
            } else {
                Err(format!("stated mtp: {count} {what} do not split over {world} ranks"))
            }
        };
        let latent_cfg = values
            .absorbed_latent
            .ok_or("stated mtp: layout states no absorbed_latent geometry")?;
        let heads = split("query heads", values.n_heads)?;
        let nope = values.head_dim;
        let value = latent_cfg.v_head_dim;
        let latent = latent_cfg.kv_lora_rank;
        let q_lora = values.q_lora_rank;
        let geom = AbsorbedGeometry {
            heads,
            nope,
            value,
            latent,
            q_lora,
        };
        let mut allocations = Vec::new();
        let vec_bf16 = |name: &str, allocs: &mut Vec<Deviceptr>| {
            device.stated_vector(arena, name, (dim, false), allocs)
        };
        let embed_table = resolve_embed_table(device, arena)?;
        let tail_norm = vec_bf16("norm.weight", &mut allocations)?;
        let enorm = vec_bf16("mtp.0.enorm.weight", &mut allocations)?;
        let hnorm = vec_bf16("mtp.0.hnorm.weight", &mut allocations)?;
        let attn_norm = vec_bf16("mtp.0.attn_norm.weight", &mut allocations)?;
        let ffn_norm = vec_bf16("mtp.0.ffn_norm.weight", &mut allocations)?;
        let head_norm = vec_bf16("mtp.0.norm.weight", &mut allocations)?;
        let eh_proj = matrix(arena, "mtp.0.eh_proj.weight", (dim, 2 * dim))?;
        let (kv_b, kv_b_bf16) = device.stated_kv_b(
            arena,
            "mtp.0.attn.wkv_b.weight",
            geom,
            &mut allocations,
        )?;
        let indexer = if arena.slice("mtp.0.attn.indexer.wk.weight").is_some() {
            let index_dim = values.index_head_dim;
            let index_heads = values.index_n_heads;
            let pool = values.stated().index_pool.max(1);
            Some(IndexerTensors {
                keys: matrix(arena, "mtp.0.attn.indexer.wk.weight", (index_dim, dim))?,
                gates: matrix(arena, "mtp.0.attn.indexer.kpool_gate", (index_dim, dim))?,
                head_weights: matrix(
                    arena,
                    "mtp.0.attn.indexer.weights_proj.weight",
                    (index_heads, dim),
                )?,
                queries: matrix(
                    arena,
                    "mtp.0.attn.indexer.wq_b.weight",
                    (index_heads * index_dim, q_lora),
                )?,
                key_norm: (
                    device.stated_vector(
                        arena,
                        "mtp.0.attn.indexer.k_norm.weight",
                        (index_dim, true),
                        &mut allocations,
                    )?,
                    device.stated_vector(
                        arena,
                        "mtp.0.attn.indexer.k_norm.bias",
                        (index_dim, true),
                        &mut allocations,
                    )?,
                ),
                ape: device.stated_vector(
                    arena,
                    "mtp.0.attn.indexer.kpool_ape",
                    (pool * index_dim, false),
                    &mut allocations,
                )?,
            })
        } else {
            None
        };
        let absorbed = AbsorbedTensors {
            q_a: matrix(arena, "mtp.0.attn.wq_a.weight", (q_lora, dim))?,
            kv_a: matrix(arena, "mtp.0.attn.wkv.weight", (latent, dim))?,
            q_b: matrix(arena, "mtp.0.attn.wq_b.weight", (heads * nope, q_lora))?,
            out: matrix(arena, "mtp.0.attn.wo.weight", (dim, heads * value))?,
            norms: (
                device.stated_vector(
                    arena,
                    "mtp.0.attn.q_norm.weight",
                    (q_lora, false),
                    &mut allocations,
                )?,
                device.stated_vector(
                    arena,
                    "mtp.0.attn.kv_norm.weight",
                    (latent, false),
                    &mut allocations,
                )?,
            ),
            kv_b,
            kv_b_bf16,
            indexer,
        };
        let experts = device.routed_geometry.experts;
        let shared_inter = split(
            "shared intermediate rows",
            values.moe_inter_dim * values.n_shared_experts,
        )?;
        let mut tables = [Deviceptr(0); 3];
        let stems = ["w1", "w3", "w2"];
        for (index, table) in tables.iter_mut().enumerate() {
            let mut words = Vec::with_capacity(experts * 16);
            for e in 0..experts {
                let wname = format!("mtp.0.ffn.experts.{e}.{}.weight", stems[index]);
                let sname = format!("mtp.0.ffn.experts.{e}.{}.scale", stems[index]);
                let w = slice(arena, &wname)?;
                let s = slice(arena, &sname)?;
                words.extend_from_slice(&w.ptr.0.to_le_bytes());
                words.extend_from_slice(&s.ptr.0.to_le_bytes());
            }
            *table = device.stated_upload(&words, &mut allocations)?;
        }
        let shared = if shared_inter == 0 {
            None
        } else {
            Some(Mlp {
                gate: matrix(arena, "mtp.0.ffn.shared_experts.w1.weight", (shared_inter, dim))?,
                up: matrix(arena, "mtp.0.ffn.shared_experts.w3.weight", (shared_inter, dim))?,
                down: matrix(arena, "mtp.0.ffn.shared_experts.w2.weight", (dim, shared_inter))?,
            })
        };
        let ffn = FfnTensors::Moe {
            router: matrix(arena, "mtp.0.ffn.gate.weight", (experts, dim))?,
            bias: device.stated_vector(
                arena,
                "mtp.0.ffn.gate.bias",
                (experts, true),
                &mut allocations,
            )?,
            tables,
            shared,
        };
        let [q_b, g_b, e_b, r_b, a_b, p_b] = KernelTable {
            image: Image::TOKEN,
            symbols: [
                ("q4e_quantize_e4m3_rows", Precision::Fp8E4M3),
                ("q4e_gemv_tiled_e4m3", Precision::Fp8E4M3),
                ("q4e_embed", Precision::Bf16),
                ("q4e_rmsnorm", Precision::Bf16),
                ("q4e_add", Precision::F32),
                ("q4e_argmax_prob_slices", Precision::F32),
            ],
        }
        .load(cuda)?;
        let quantize_e4m3 = q_b.kernel();
        let gemv_e4m3_kernel = g_b.kernel();
        let embed_kernel = e_b.kernel();
        let rmsnorm_kernel = r_b.kernel();
        let add_kernel = a_b.kernel();
        let argmax_prob_kernel = p_b.kernel();
        let local_vocab = crate::plan::vocab_per_rank();
        let head_e4m3 = if sparkflash_model::active_profile().drafter.head_fp8() {
            Some(quantize_lm_head_e4m3(
                cuda,
                arena,
                quantize_e4m3,
                local_vocab,
                dim,
                &mut allocations,
            )?)
        } else {
            None
        };
        let mut alloc = |bytes: usize| -> Result<Deviceptr, String> {
            let ptr = cuda.alloc(bytes.max(4), &mut 0)?;
            allocations.push(ptr);
            Ok(ptr)
        };
        let embed_f32 = alloc(dim * 4)?;
        let eh_in = alloc(2 * dim * 4)?;
        let mtp_x = alloc(dim * 4)?;
        let cur_hidden = alloc(dim * 4)?;
        let branch_f32 = alloc(dim * 4)?;
        let head_logits_local = alloc(local_vocab * 4)?;
        let head_logits_gathered = alloc(world * local_vocab * 4)?;
        let pick_tok = alloc(4)?;
        let pick_prob = alloc(4)?;
        let partials = alloc(TILED_GEMV_PARTIALS * 4)?;
        let counters = alloc(TILED_GEMV_BLOCKS * 4)?;
        cuda.memset_d8_async(counters, 0, TILED_GEMV_BLOCKS * 4)?;
        Ok(Self {
            embed_table,
            tail_norm,
            enorm,
            hnorm,
            attn_norm,
            ffn_norm,
            head_norm,
            eh_proj,
            absorbed,
            ffn,
            head_e4m3,
            gemv_e4m3_kernel,
            embed_kernel,
            rmsnorm_kernel,
            add_kernel,
            argmax_prob_kernel,
            embed_f32,
            eh_in,
            mtp_x,
            cur_hidden,
            branch_f32,
            head_logits_local,
            head_logits_gathered,
            pick_tok,
            pick_prob,
            partials,
            counters,
            lanes: RefCell::new(BTreeMap::new()),
            lane_allocs: RefCell::new(Vec::new()),
            gather_seq: Cell::new(0x6000_0000),
            allocations,
        })
    }

    fn ensure_lane(&self, device: &DeviceLayer<'_>, lane: usize) -> Result<MtpLane, String> {
        let mut lanes = self.lanes.borrow_mut();
        if let Some(&entry) = lanes.get(&lane) {
            return Ok(entry);
        }
        let stated = device
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated device")?;
        let values = crate::plan::description().values();
        let latent = values
            .absorbed_latent
            .map_or(512, |cfg| cfg.kv_lora_rank);
        let cap = device.max_positions;
        let mut extra = self.lane_allocs.borrow_mut();
        let mut alloc = |bytes: usize| -> Result<Deviceptr, String> {
            let ptr = device.cuda.alloc(bytes.max(4), &mut 0)?;
            extra.push(ptr);
            Ok(ptr)
        };
        let recent_hiddens = alloc(super::rows::R_MAX * device.dim * 4)?;
        device
            .cuda
            .memset_d8_async(recent_hiddens, 0, super::rows::R_MAX * device.dim * 4)?;
        let last_hidden = alloc(device.dim * 4)?;
        device.cuda.memset_d8_async(last_hidden, 0, device.dim * 4)?;
        let latent_row = if stated.e4m3_latent { latent } else { latent * 4 };
        let latent_buf = alloc(cap * latent_row)?;
        let scale_row = if stated.e4m3_latent {
            latent.div_ceil(ACTIVATION_TILE) * 4
        } else {
            0
        };
        let latent_scales = if scale_row > 0 {
            alloc(cap * scale_row)?
        } else {
            Deviceptr(0)
        };
        let (idx_keys, idx_gates, idx_pooled) = if self.absorbed.indexer.is_some() {
            let key_row = values.index_head_dim * 4;
            let pool = values.stated().index_pool.max(1);
            let pooled_rows = cap.div_ceil(pool);
            (
                alloc(cap * key_row)?,
                alloc(cap * key_row)?,
                alloc(pooled_rows * key_row)?,
            )
        } else {
            (Deviceptr(0), Deviceptr(0), Deviceptr(0))
        };
        let entry = MtpLane {
            recent_hiddens,
            recent_p0: 0,
            recent_n: 0,
            last_hidden,
            latent: latent_buf,
            latent_scales,
            idx_keys,
            idx_gates,
            idx_pooled,
        };
        lanes.insert(lane, entry);
        Ok(entry)
    }

    fn rmsnorm(&self, cuda: &Cuda, x: Deviceptr, w: Deviceptr, out: Deviceptr, dim: usize, eps: f32) -> Result<(), String> {
        cuda.launch(
            self.rmsnorm_kernel,
            (1, 1, 1),
            (256, 1, 1),
            &[x.0, w.0, out.0, dim as u64, dim as u64, u64::from(eps.to_bits()), 0, 1],
        )
    }

    fn run_eh_and_attn(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        lane: MtpLane,
        tok: u32,
        hidden: Deviceptr,
        mtp_pos: usize,
        attend: bool,
    ) -> Result<(), String> {
        let stated = device
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated device")?;
        let state = rows
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated rows")?;
        let dim = device.dim;
        let eps = crate::plan::description().values().norm_eps as f32;
        let embed = self.embed_table.0;
        device.cuda.launch(
            self.embed_kernel,
            (dim.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[embed, self.embed_f32.0, u64::from(tok), dim as u64, dim as u64, 0],
        )?;
        self.rmsnorm(device.cuda, self.embed_f32, self.enorm, self.eh_in, dim, eps)?;
        self.rmsnorm(
            device.cuda,
            hidden,
            self.hnorm,
            self.eh_in.offset(dim * 4),
            dim,
            eps,
        )?;
        let cx = RowsCx::<TiledLinear<'_>>::new(device.cuda, 1, None);
        stated
            .tiled
            .project(&cx, &state.scratch, &self.eh_proj, (self.eh_in, self.mtp_x), 1, (true, true))
            .map_err(|error| format!("stated mtp eh_proj: {error}"))?;
        self.rmsnorm(
            device.cuda,
            self.mtp_x,
            self.attn_norm,
            Deviceptr(rows.b.normed),
            dim,
            eps,
        )?;
        let attn_seg = vec![LaneSegment {
            first: 0,
            rows: 1,
            position: mtp_pos,
            regions: vec![lane.latent, lane.latent_scales],
            snapshots: Vec::new(),
        }];
        let idx_seg = vec![LaneSegment {
            first: 0,
            rows: 1,
            position: mtp_pos,
            regions: vec![lane.idx_keys, lane.idx_gates, lane.idx_pooled],
            snapshots: Vec::new(),
        }];
        device.stated_absorbed_exec(
            (stated, state),
            &self.absorbed,
            (&attn_seg, &idx_seg),
            Deviceptr(rows.b.normed),
            1,
            "stated mtp",
            attend,
        )?;
        Ok(())
    }

    fn seed(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let norm_weight = self.tail_norm;
        if n == 0 {
            return Ok(());
        }
        let stated = device
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated device")?;
        let state = rows
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated rows")?;
        let dim = device.dim;
        let eps = crate::plan::description().values().norm_eps as f32;
        let cx = RowsCx::<TiledLinear<'_>>::new(device.cuda, n, None);
        stated
            .mixer
            .head(&cx, (Deviceptr(rows.b.stream), norm_weight, state.head), (eps, true))
            .map_err(|error| format!("stated mtp seed head: {error}"))?;
        // Read back the pass's token IDs so MTP can populate its own KV cache for positions < last.
        let mut tok_bytes = vec![0u8; n * 4];
        device.cuda.copy_dtoh(&mut tok_bytes, &Deviceptr(rows.b.row_ids))?;
        let toks: Vec<u32> = tok_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for seg in device.step_segs(p0, n) {
            let lane_id = if seg.lane == super::lanes::NO_LANE {
                device.lanes.as_ref().map_or(0, |l| l.selected)
            } else {
                seg.lane
            };
            let lane = self.ensure_lane(device, lane_id)?;
            for t in 0..seg.n {
                let abs_pos = seg.p0 + t;
                if abs_pos == 0 {
                    continue;
                }
                let prev_h = if t == 0 {
                    lane.last_hidden
                } else {
                    state.head.offset((seg.row0 + t - 1) * dim * 4)
                };
                self.run_eh_and_attn(
                    device,
                    rows,
                    lane,
                    toks[seg.row0 + t],
                    prev_h,
                    abs_pos - 1,
                    false,
                )?;
            }
            let last_h = state.head.offset((seg.row0 + seg.n - 1) * dim * 4);
            device.cuda.copy_dtod_async(&lane.last_hidden, &last_h, dim * 4)?;
            let keep = seg.n.min(super::rows::R_MAX);
            let skip = seg.n - keep;
            let src = state.head.offset((seg.row0 + skip) * dim * 4);
            device
                .cuda
                .copy_dtod_async(&lane.recent_hiddens, &src, keep * dim * 4)?;
            if let Some(entry) = self.lanes.borrow_mut().get_mut(&lane_id) {
                entry.recent_p0 = seg.p0 + skip;
                entry.recent_n = keep;
            }
        }
        Ok(())
    }

    fn draft_blocks(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        blocks: &[DraftBlock],
        k: usize,
    ) -> Result<Vec<Drafts>, String> {
        let stated = device
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated device")?;
        let state = rows
            .stated
            .as_ref()
            .ok_or("stated mtp: no stated rows")?;
        let dim = device.dim;
        let world = device.routed_geometry.world;
        let local_vocab = crate::plan::vocab_per_rank();
        let eps = crate::plan::description().values().norm_eps as f32;
        let share = device.stated_share()?;
        let cx1 = RowsCx::<TiledLinear<'_>>::new(device.cuda, 1, share);
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            let lane = self.ensure_lane(device, block.lane)?;
            let target_pos = block.d.saturating_sub(1);
            let src_hidden = if lane.recent_n > 0
                && target_pos >= lane.recent_p0
                && target_pos < lane.recent_p0 + lane.recent_n
            {
                lane.recent_hiddens
                    .offset((target_pos - lane.recent_p0) * dim * 4)
            } else {
                lane.last_hidden
            };
            device.cuda.copy_dtod_async(&self.cur_hidden, &src_hidden, dim * 4)?;
            let mut cur_tok = block.anchor;
            let mut mtp_pos = block.d.saturating_sub(1);
            let mut ids = Vec::with_capacity(k);
            let mut probs = Vec::with_capacity(k);
            for _step in 0..k {
                // 1. eh_proj + absorbed MLA attention
                self.run_eh_and_attn(
                    device,
                    rows,
                    lane,
                    cur_tok,
                    self.cur_hidden,
                    mtp_pos,
                    true,
                )?;
                if world > 1 {
                    cx1.call(
                        stated.token[3],
                        calls::narrow(rows.b.attn_out, state.branch.0, dim),
                    )
                    .map_err(|error| format!("stated mtp narrow attn: {error}"))?;
                    device.reduce_layer_rows(
                        rows,
                        "mtp_attn_ar",
                        span(rows.b.attn_out, dim * 2),
                        span(rows.b.reduced, dim * 2),
                        1,
                    )?;
                    // Widen reduced BF16 row into branch_f32 via q4e_embed(table=reduced, tok=0).
                    device.cuda.launch(
                        self.embed_kernel,
                        (dim.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        &[rows.b.reduced, self.branch_f32.0, 0, dim as u64, dim as u64, 0],
                    )?;
                    device.cuda.launch(
                        self.add_kernel,
                        (dim.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        &[self.mtp_x.0, self.branch_f32.0, dim as u64],
                    )?;
                } else {
                    device.cuda.launch(
                        self.add_kernel,
                        (dim.div_ceil(256) as u32, 1, 1),
                        (256, 1, 1),
                        &[self.mtp_x.0, state.branch.0, dim as u64],
                    )?;
                }
                // 2. MoE FFN (reusing stated_ffn directly)
                self.rmsnorm(
                    device.cuda,
                    self.mtp_x,
                    self.ffn_norm,
                    Deviceptr(rows.b.normed),
                    dim,
                    eps,
                )?;
                device.stated_ffn((stated, state), &self.ffn, rows.b, 0, 1, share)?;
                device.reduce_layer_rows(
                    rows,
                    "mtp_ffn_ar",
                    span(rows.b.attn_out, dim * 2),
                    span(rows.b.reduced, dim * 2),
                    1,
                )?;
                device.cuda.launch(
                    self.embed_kernel,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &[rows.b.reduced, self.branch_f32.0, 0, dim as u64, dim as u64, 0],
                )?;
                device.cuda.launch(
                    self.add_kernel,
                    (dim.div_ceil(256) as u32, 1, 1),
                    (256, 1, 1),
                    &[self.mtp_x.0, self.branch_f32.0, dim as u64],
                )?;
                // 3. Shared head norm -> next cur_hidden & head scoring
                self.rmsnorm(
                    device.cuda,
                    self.mtp_x,
                    self.head_norm,
                    self.cur_hidden,
                    dim,
                    eps,
                )?;
                let (tok, prob) = if let Some(head) = self.head_e4m3.as_ref() {
                    // D3: E4M3 draft head scoring
                    let dev = StatedDraftDevice {
                        device,
                        rows,
                        partials: self.partials,
                        counters: self.counters,
                        reduce_in: Deviceptr(0),
                        reduce_out: Deviceptr(0),
                        reduce_f32: Deviceptr(0),
                        embed_kernel: None,
                        add_kernel: None,
                    };
                    gemv_e4m3(
                        &dev,
                        self.gemv_e4m3_kernel,
                        head,
                        self.cur_hidden,
                        self.head_logits_local,
                        1,
                    )
                    .map_err(|error| format!("stated mtp head_e4m3: {error}"))?;
                    let logits_ptr = if world > 1 {
                        device.cuda.synchronize()?;
                        let mut lb = vec![0u8; local_vocab * 4];
                        device.cuda.copy_dtoh(&mut lb, &self.head_logits_local)?;
                        let lw: Vec<u32> = lb
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let seq = self.gather_seq.get();
                        self.gather_seq.set(seq.wrapping_add(1));
                        let gw = all_gather_u32_words(transport, &lw, world, seq)?;
                        let gb: Vec<u8> = gw.iter().flat_map(|w| w.to_le_bytes()).collect();
                        device.cuda.copy_htod(&self.head_logits_gathered, &gb)?;
                        self.head_logits_gathered
                    } else {
                        self.head_logits_local
                    };
                    device.cuda.launch(
                        self.argmax_prob_kernel,
                        (1, 1, 1),
                        (256, 1, 1),
                        &[
                            logits_ptr.0,
                            self.pick_tok.0,
                            self.pick_prob.0,
                            world as u64,
                            1,
                            local_vocab as u64,
                        ],
                    )?;
                    device.cuda.synchronize()?;
                    let mut tb = [0u8; 4];
                    let mut pb = [0u8; 4];
                    device.cuda.copy_dtoh(&mut tb, &self.pick_tok)?;
                    device.cuda.copy_dtoh(&mut pb, &self.pick_prob)?;
                    (u32::from_le_bytes(tb), f32::from_le_bytes(pb))
                } else {
                    // BF16 head fallback (`SPARKFLASH_DSPARK_HEAD_FP8=0`)
                    cx1.call(
                        stated.token[3],
                        calls::narrow(rows.b.lm_normed, self.cur_hidden.0, dim),
                    )
                    .map_err(|error| format!("stated mtp narrow head: {error}"))?;
                    let picked = device.head_rows(rows, tail, transport, 1)?;
                    (picked[0].0, 1.0f32)
                };
                ids.push(tok);
                probs.push(prob);
                cur_tok = tok;
                mtp_pos += 1;
            }
            out.push((ids, probs));
        }
        Ok(out)
    }

    fn release(self, cuda: &Cuda) {
        let extra = self.lane_allocs.into_inner();
        let borrowed_extra: Vec<&Deviceptr> = extra.iter().collect();
        cuda.free_buffers(&borrowed_extra);
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

/// Active stated-stack drafter: either DFlash 2 (D2) when configured or GLM's MTP layer (D1).
pub(super) enum StatedDraftKind {
    Mtp(Box<MtpDraft>),
    DFlash2(Box<DFlash2Draft>),
}

/// Stated-stack drafter state held inside `DraftState`.
pub(super) struct StatedDraftState {
    kind: StatedDraftKind,
    /// `[block_size + 1]` u32 on device for `DraftState::ids()` (used by single-sequence verify).
    ids: Deviceptr,
    block_size: usize,
}

impl StatedDraftState {
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        arena: &sparkflash_device::arena::ResidentArena,
        max_rows: usize,
    ) -> Result<Self, String> {
        let drafter_dir = sparkflash_model::active_profile()
            .speculation
            .drafter
            .clone();
        let (kind, block_size) = if let Some(dir) = drafter_dir {
            let dflash = DFlash2Draft::open(device, arena, Path::new(&dir), max_rows)?;
            let block = dflash.block_size().min(super::rows::R_MAX - 1).max(1);
            (StatedDraftKind::DFlash2(Box::new(dflash)), block)
        } else {
            let mtp = MtpDraft::open(device, arena)?;
            // MTP runs 1-step (or up to stated k <= 4) autoregressively; default block = 1.
            let block = sparkflash_model::active_profile()
                .speculation
                .dspark_k
                .unwrap_or(1)
                .clamp(1, super::rows::R_MAX - 1);
            (StatedDraftKind::Mtp(Box::new(mtp)), block)
        };
        crate::plan::set_stated_draft_block(block_size);
        let ids = device.cuda.alloc((block_size + 1) * 4, &mut 0)?;
        Ok(Self {
            kind,
            ids,
            block_size,
        })
    }

    pub(super) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(super) fn ids(&self) -> u64 {
        self.ids.0
    }

    pub(super) fn dflash_tap(&self) -> Option<(Kernel, Deviceptr, Vec<usize>, usize)> {
        match &self.kind {
            StatedDraftKind::DFlash2(d) => Some(d.tap_info()),
            StatedDraftKind::Mtp(_) => None,
        }
    }

    pub(super) fn seed(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        match &self.kind {
            StatedDraftKind::DFlash2(d) => d.seed(device, rows, p0, n),
            StatedDraftKind::Mtp(m) => m.seed(device, rows, p0, n),
        }
    }

    pub(super) fn draft_blocks(
        &self,
        device: &DeviceLayer<'_>,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        blocks: &[DraftBlock],
        k: usize,
    ) -> Result<Vec<Drafts>, String> {
        match &self.kind {
            StatedDraftKind::DFlash2(d) => d.draft_blocks(device, rows, transport, blocks, k),
            StatedDraftKind::Mtp(m) => m.draft_blocks(device, rows, tail, transport, blocks, k),
        }
    }

    pub(super) fn release(self, cuda: &Cuda) {
        cuda.free_buffers(&[&self.ids]);
        match self.kind {
            StatedDraftKind::DFlash2(d) => d.release(cuda),
            StatedDraftKind::Mtp(m) => m.release(cuda),
        }
    }
}
