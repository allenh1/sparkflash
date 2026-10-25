//! The bulk prefill: a prompt chunk of more than `R_MAX` rows through the rows step, with the
//! GEMVs on the tensor cores (kernels/opt_mma_sm121.cu, through `launch_opt`) and the stages the
//! rows step runs per row replaced by chunk-wide launches: the compressor (every completed group of
//! the chunk at once), the indexer (scores on the FP4 tensor cores, one select launch over the
//! rows), the router and the routed experts (grouped by expert, so each expert's slices are read
//! about once per chunk).
//!
//! The results are not bit-identical to the M = 1 step (the GEMMs accumulate in another order and
//! the routed partials round through bf16), so the prefill driver only takes this path for a long
//! uncached prompt and SPARKFLASH_PREFILL_EXACT=1 keeps every prompt on the exact rows step.

use super::rows::RowsState;
use super::*;
use sparkflash_kernels::cuda_driver::{HostMapped, Kernel, Module};
use sparkflash_kernels::hostw_gemv_launch::Mode;
use sparkflash_kernels::moe_routed_device::MoeBulk;
use sparkflash_kernels::mxfp8_gemv_launch::{OptArgs, OptKind, launch_opt};

/// The indexer scores buffer, in floats: a chunk whose rows times compressed positions exceed it is
/// scored and selected in row slices.
const SCORE_FLOATS: usize = 32 << 20;
/// opt_select_rows' blocks.
const SELECT_BLOCKS: usize = 96;
/// opt_select_sm121.cu's FAST_MAX_POS: the most compressed positions the select serves.
const SELECT_MAX_POSITIONS: usize = 1 << 21;

pub(super) struct BulkKernels {
    _modules: Vec<Module>,
    pack: Kernel,
    scores: Kernel,
    /// opt_mma_hostw_{bf16w, bf16w_f32o, f32w_f32o}, by hostw_gemv_launch::Mode.
    hostw: [Kernel; 3],
    router_dots: Kernel,
    router_topk: Kernel,
    /// opt_mma_fp8_rows_bf16o: the Engram wkv (row-major E4M3 weights).
    engram_wkv: Kernel,
    /// opt_bulk_attn: the sparse attention with the 16 heads on the tensor cores.
    pub(super) attn: Kernel,
    /// opt_bulk_mhc_{partials, tail}: the mHC seam over the chunk; opt_bulk_mhc_post_partials:
    /// the post before a seam fused with that seam's partials.
    pub(super) seam_partials: Kernel,
    pub(super) seam_tail: Kernel,
    pub(super) post_partials: Kernel,
    /// opt_bulk_engram_{embed, post}: the Engram stage around the wkv over the chunk.
    engram_embed: Kernel,
    engram_post: Kernel,
    select_rows: Kernel,
    pool_rows: Kernel,
    k_norm_rows: Kernel,
    rope: Kernel,
}

/// The bulk chunk's kernels and its chunk-wide scratch (the row buffers are the [`RowsState`]'s,
/// carved at the chunk capacity).
pub(super) struct BulkState {
    pub(super) k: BulkKernels,
    pub(super) moe: MoeBulk,
    /// The compressor rows of the chunk, `[cap + 16][head_dim]` f32: row i is position
    /// p0 - lead + i, the first `lead` rows the open group's rows copied from the ring.
    wkv: u64,
    wgate: u64,
    /// The chunk's completed groups: pooled f32, narrowed bf16, normed (then roped) bf16, f32.
    pooled: u64,
    latent_bf16: u64,
    latent_normed: u64,
    latent_f32: u64,
    /// The groups' index keys: f32, then normed and roped bf16.
    wk_f32: u64,
    wk_bf16: u64,
    /// The rows' index queries as E2M1 codes (one per byte) and ue8m0 scales.
    q_codes: u64,
    q_scales: u64,
    /// The owner's index keys as E2M1 codes and scales, `[max_positions][index_head_dim]`.
    k_codes: u64,
    k_scales: u64,
    scores: u64,
    /// opt_select_rows' scratch, `select_words` u32 per block: a row's keys and chosen flags
    /// past the shared-memory table (n + n / 4 words for n compressed positions).
    select_scratch: u64,
    select_words: usize,
    /// The router's raw dots, `[cap][experts]` f32.
    dots: u64,
    /// The chunk's DSpark taps, `[cap][3][dim]` f32 (the rows state's tap sink).
    pub(super) taps: u64,
    /// With Engram: the two pinned buffers a chunk's table reads land in (a chunk's reads run
    /// under the chunk before it), and the live flags `[slots][positions]` the post kernel reads
    /// through the mapping (host-written as each slot's layer is queued; by position, so a chunk
    /// never rewrites a flag a queued launch reads).
    pub(super) engram_staging: Option<[HostMapped; 2]>,
    engram_live: Option<HostMapped>,
    allocations: Vec<Deviceptr>,
}

impl BulkState {
    /// Loads the bulk kernels (installing the tensor-core GEMMs `launch_opt` runs above 8 tokens)
    /// and carves the chunk scratch for `cap` rows.
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        kernel_dir: &Path,
        cap: usize,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let mut modules = Vec::new();
        let (mma, _) = cached_cubin(&kernel_dir.join("opt_mma_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        modules.push(sparkflash_kernels::mxfp8_gemv_launch::install_mma(
            cuda, &mma,
        )?);
        let (select, _) = cached_cubin(&kernel_dir.join("opt_select_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let (misc, _) = cached_cubin(&kernel_dir.join("opt_misc_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let (rows_cubin, _) = cached_cubin(&kernel_dir.join("opt_rows_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let mut load = |cubin: &[u8], name: &'static str| -> Result<Kernel, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(kernel)
        };
        let moe_kernels = [
            load(&mma, "opt_mma_moe_plan")?,
            load(&mma, "opt_mma_moe_gate_up")?,
            load(&mma, "opt_mma_moe_down")?,
            load(&mma, "opt_mma_moe_combine")?,
        ];
        let k = BulkKernels {
            pack: load(&mma, "opt_bulk_fp4_pack")?,
            scores: load(&mma, "opt_mma_index_scores")?,
            hostw: [
                load(&mma, "opt_mma_hostw_bf16w")?,
                load(&mma, "opt_mma_hostw_bf16w_f32o")?,
                load(&mma, "opt_mma_hostw_f32w_f32o")?,
            ],
            router_dots: load(&mma, "opt_mma_router_dots")?,
            router_topk: load(&mma, "opt_mma_router_topk")?,
            engram_wkv: load(&mma, "opt_mma_fp8_rows_bf16o")?,
            attn: load(&mma, "opt_bulk_attn")?,
            seam_partials: load(&rows_cubin, "opt_bulk_mhc_partials")?,
            seam_tail: load(&rows_cubin, "opt_bulk_mhc_tail")?,
            post_partials: load(&rows_cubin, "opt_bulk_mhc_post_partials")?,
            engram_embed: load(&misc, "opt_bulk_engram_embed")?,
            engram_post: load(&misc, "opt_bulk_engram_post")?,
            select_rows: load(&select, "opt_select_rows")?,
            pool_rows: load(&misc, "opt_pool_rows")?,
            k_norm_rows: load(&misc, "opt_k_norm_rows")?,
            rope: load(&misc, "opt_rope_heads_bf16")?,
            _modules: modules,
        };
        let head_dim = device.head_width as usize;
        let index_dim = device.index_head_dim;
        let index_cols = device.index_n_heads * index_dim;
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize| -> Result<u64, String> {
            let ptr = cuda.alloc(bytes.max(16), &mut total)?;
            allocations.push(ptr);
            Ok(ptr.0)
        };
        let [sorted, items, items_count, codes, scales, slot] = MoeBulk::buffer_bytes(cap);
        let moe = MoeBulk {
            plan: moe_kernels[0],
            gate_up: moe_kernels[1],
            down: moe_kernels[2],
            combine: moe_kernels[3],
            sorted: Deviceptr(alloc(sorted)?),
            items: Deviceptr(alloc(items)?),
            items_count: Deviceptr(alloc(items_count)?),
            codes: Deviceptr(alloc(codes)?),
            scales: Deviceptr(alloc(scales)?),
            slot: Deviceptr(alloc(slot)?),
            max_tokens: cap,
        };
        let ring_rows = cap + 16;
        // A row reaches at most max_positions compressed positions (ratio 1).
        let select_words = device.max_positions + device.max_positions.div_ceil(4);
        let engram_staging = match device.engram_staging(cap)? {
            Some(first) => match device.engram_staging(cap) {
                Ok(Some(second)) => Some([first, second]),
                other => {
                    cuda.free_host_mapped(&first);
                    other?;
                    None
                }
            },
            None => None,
        };
        let engram_live = match device.engram.as_ref() {
            Some(engram) => {
                Some(cuda.alloc_host_mapped(engram.layer_ids.len() * engram.positions, &mut 0)?)
            }
            None => None,
        };
        let state = Self {
            k,
            moe,
            wkv: alloc(ring_rows * head_dim * 4)?,
            wgate: alloc(ring_rows * head_dim * 4)?,
            pooled: alloc(cap * head_dim * 4)?,
            latent_bf16: alloc(cap * head_dim * 2)?,
            latent_normed: alloc(cap * head_dim * 2)?,
            latent_f32: alloc(cap * head_dim * 4)?,
            wk_f32: alloc(cap * index_dim * 4)?,
            wk_bf16: alloc(cap * index_dim * 2)?,
            q_codes: alloc(cap * index_cols)?,
            q_scales: alloc(cap * index_cols / 32)?,
            k_codes: alloc(device.max_positions * index_dim)?,
            k_scales: alloc(device.max_positions * index_dim / 32)?,
            scores: alloc(SCORE_FLOATS * 4)?,
            select_scratch: alloc(SELECT_BLOCKS * select_words * 4)?,
            select_words,
            dots: alloc(cap * device.routed_geometry.experts * 4)?,
            taps: alloc(cap * 3 * device.dim * 4)?,
            engram_staging,
            engram_live,
            allocations,
        };
        Ok(state)
    }

    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
        for pinned in self
            .engram_staging
            .iter()
            .flatten()
            .chain(&self.engram_live)
        {
            cuda.free_host_mapped(pinned);
        }
    }

    fn hostw_kernel(&self, mode: Mode) -> Kernel {
        match mode {
            Mode::Bf16Weight => self.k.hostw[0],
            Mode::Bf16WeightF32Stores => self.k.hostw[1],
            Mode::F32WeightF32Stores => self.k.hostw[2],
        }
    }
}

impl DeviceLayer<'_> {
    /// A host-weight projection of `tokens` f32 rows (`[tokens][inner]`) on the tensor cores.
    #[allow(clippy::too_many_arguments)]
    fn bulk_hostw(
        &self,
        kernel: Kernel,
        a: u64,
        w: u64,
        out: u64,
        cols: usize,
        inner: usize,
        tokens: usize,
    ) -> Result<(), String> {
        self.cuda.launch(
            kernel,
            (cols.div_ceil(128) as u32, tokens.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                a,
                w,
                out,
                cols as u64,
                inner as u64,
                tokens as u64,
                inner as u64,
                cols as u64,
            ],
        )
    }

    /// opt_rope_heads_bf16 over `tokens` rows of `heads` heads, token t at table position
    /// `first + t * step`.
    #[allow(clippy::too_many_arguments)]
    fn bulk_rope(
        &self,
        bulk: &BulkState,
        rows: u64,
        width: usize,
        heads: usize,
        tokens: usize,
        first: usize,
        step: usize,
    ) -> Result<(), String> {
        let pairs = self.rope_pairs as usize;
        let table_bytes = (self.max_positions * pairs * 4) as u64;
        let tables = self.buffers.workspace.rope_tables_compressed.device;
        let at = (first * pairs * 4) as u64;
        self.cuda.launch(
            bulk.k.rope,
            (heads as u32, tokens as u32, 1),
            (32, 1, 1),
            &[
                rows,
                tables + at,
                tables + table_bytes + at,
                width as u64,
                u64::from(self.rope_head_dim),
                heads as u64,
                (pairs * step) as u64,
            ],
        )
    }

    /// The compressor and the indexer select of the chunk's rows p0..p0+n (the bulk twin of
    /// `rows_seam_select`).
    pub(super) fn bulk_seam_select(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let ratio = self.compress_ratio.get(layer).copied().unwrap_or(0);
        if ratio == 0 {
            return Ok(());
        }
        if self.is_kv_source(layer) {
            self.bulk_compress(rows, bulk, layer, ratio, p0, n)?;
        }
        if self.is_index_source(layer) {
            self.bulk_index(rows, bulk, layer, ratio, p0, n)?;
        }
        Ok(())
    }

    /// wkv / wgate over the rows, then every group the chunk completes: the pool, the latent norm,
    /// the index key (wk, k_norm, rope, FP4 round trip) and the roped latent's store rows. The last
    /// rows go back to the ring, so the decode steps after the chunk pool their open group as the
    /// M = 1 path would have left it.
    fn bulk_compress(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        ratio: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::engram_convert_launch::{
            self as convert, EngramConvertParams, EngramConvertSpans,
        };
        let refusal = |what: &str| format!("bulk compressor: layer {layer}: {what}");
        let (Some(staged), Some(ring)) = (self.seam_staged, self.seam_ring) else {
            return Err(refusal("no staged compressor"));
        };
        let owner = self
            .owner_index_for(layer)
            .ok_or_else(|| refusal("no kv-source owner"))?;
        let (wkv, wkv_mode) = staged.wkv.ok_or_else(|| refusal("no staged wkv"))?;
        let hd = ring.head_dim;
        let row_bytes = hd * 4;
        let lead = p0 % ratio;
        let owner_base = owner * ring.rows * row_bytes;
        let ring_row = |position: usize| (owner_base + (position % ring.rows) * row_bytes) as u64;
        let cuda = self.cuda;
        let b = rows.b;
        // Copies scratch rows [from, from + count) to or from the ring rows of positions
        // p0 - lead + from.., one contiguous run at a time.
        let ring_copy = |scratch: u64, ring_base: u64, from: usize, count: usize, to_ring: bool| {
            let mut i = from;
            while i < from + count {
                let position = p0 - lead + i;
                let slot = position % ring.rows;
                let run = (ring.rows - slot).min(from + count - i);
                let s = Deviceptr(scratch + (i * row_bytes) as u64);
                let r = Deviceptr(ring_base + ring_row(position));
                let (dst, src) = if to_ring { (r, s) } else { (s, r) };
                cuda.copy_dtod_async(&dst, &src, run * row_bytes)?;
                i += run;
            }
            Ok::<(), String>(())
        };
        let mut projections = vec![(wkv, wkv_mode, bulk.wkv, ring.wkv.0)];
        if let Some((wgate, mode)) = staged.wgate {
            projections.push((wgate, mode, bulk.wgate, ring.wgate.0));
        }
        for &(weight, mode, scratch, ring_base) in &projections {
            ring_copy(scratch, ring_base, 0, lead, false)?;
            self.bulk_hostw(
                bulk.hostw_kernel(mode),
                b.normed,
                weight.0,
                scratch + (lead * row_bytes) as u64,
                hd,
                self.dim,
                n,
            )?;
        }
        let g0 = p0 / ratio;
        let groups = (p0 + n) / ratio - g0;
        if groups > 0 {
            let unnormed = if ratio == 1 {
                bulk.wkv
            } else {
                if staged.wgate.is_none() {
                    return Err(refusal("a pooled ratio without wgate"));
                }
                cuda.launch(
                    bulk.k.pool_rows,
                    (hd.div_ceil(128) as u32, groups as u32, 1),
                    (128, 1, 1),
                    &[bulk.wkv, bulk.wgate, bulk.pooled, ratio as u64, hd as u64],
                )?;
                bulk.pooled
            };
            let count = groups * hd;
            let narrow = self.seam_narrow.ok_or_else(|| refusal("no narrow"))?;
            let widen = self.seam_widen.ok_or_else(|| refusal("no widen"))?;
            let norm = self.seam_norm.ok_or_else(|| refusal("no norm"))?;
            let to_bf16 = |src: u64, dst: u64| {
                convert::launch_narrow(
                    cuda,
                    narrow,
                    &EngramConvertParams {
                        src: Deviceptr(src),
                        dst: Deviceptr(dst),
                        count: count as u32,
                    },
                    &EngramConvertSpans {
                        src_bytes: count * 4,
                        dst_bytes: count * 2,
                    },
                )
            };
            let to_f32 = |src: u64, dst: u64| {
                convert::launch_widen(
                    cuda,
                    widen,
                    &EngramConvertParams {
                        src: Deviceptr(src),
                        dst: Deviceptr(dst),
                        count: count as u32,
                    },
                    &EngramConvertSpans {
                        src_bytes: count * 2,
                        dst_bytes: count * 4,
                    },
                )
            };
            to_bf16(unnormed, bulk.latent_bf16)?;
            let (norm_weight, norm_bytes) = staged
                .compressor_norm
                .ok_or_else(|| refusal("no compressor norm"))?;
            if norm_bytes != hd * 2 {
                return Err(refusal("the compressor norm is not one bf16 row"));
            }
            sparkflash_kernels::rmsnorm_launch::launch(
                cuda,
                norm,
                &sparkflash_kernels::rmsnorm_launch::RmsNormLaunchParams {
                    x: Deviceptr(bulk.latent_bf16),
                    weight: norm_weight,
                    out: Deviceptr(bulk.latent_normed),
                    rows: groups as u32,
                    cols: hd as u32,
                    eps: self.eps,
                    trace_ring: 0,
                    layer_id: layer as u32,
                    step_seq: p0 as u64,
                },
            )?;
            // The index keys, from the pre-rope normed latent.
            let wk = staged.index_wk.ok_or_else(|| refusal("no indexer wk"))?;
            let (k_norm, _) = staged
                .index_k_norm
                .ok_or_else(|| refusal("no indexer k_norm"))?;
            let fp4_rt = self
                .seam_fp4_round_trip
                .ok_or_else(|| refusal("no FP4 round trip"))?;
            let index_dim = self.index_head_dim;
            to_f32(bulk.latent_normed, bulk.latent_f32)?;
            self.bulk_hostw(
                bulk.k.hostw[1],
                bulk.latent_f32,
                wk.0,
                bulk.wk_f32,
                index_dim,
                hd,
                groups,
            )?;
            cuda.launch(
                bulk.k.k_norm_rows,
                ((groups as u32).div_ceil(128), 1, 1),
                (128, 1, 1),
                &[
                    bulk.wk_f32,
                    k_norm.0,
                    bulk.wk_bf16,
                    index_dim as u64,
                    u64::from(self.eps.to_bits()),
                    groups as u64,
                ],
            )?;
            self.bulk_rope(bulk, bulk.wk_bf16, index_dim, 1, groups, g0 * ratio, ratio)?;
            sparkflash_kernels::pool_group_launch::launch_fp4_round_trip(
                cuda,
                fp4_rt,
                Deviceptr(bulk.wk_bf16),
                Deviceptr(
                    ring.index_keys.0 + ((owner * self.max_positions + g0) * index_dim * 4) as u64,
                ),
                (groups * index_dim / 32) as u32,
            )?;
            // The compressed store rows, from the roped latent.
            self.bulk_rope(bulk, bulk.latent_normed, hd, 1, groups, g0 * ratio, ratio)?;
            to_f32(bulk.latent_normed, bulk.latent_f32)?;
            let append = self.seam_append.ok_or_else(|| refusal("no append"))?;
            let (code_bytes, scale_bytes) = seam_store_owner_bytes(self.max_positions)?;
            let params = sparkflash_kernels::compressed_append_launch::AppendParams {
                latent: bulk.latent_f32,
                codes: ring.store_codes.0 + (owner * code_bytes) as u64,
                scales: ring.store_scales.0 + (owner * scale_bytes) as u64,
                first_row: g0 as u32,
                row_count: groups as u32,
                ratio: ratio as u32,
            };
            cuda.launch(
                append,
                (groups as u32, 1, 1),
                sparkflash_kernels::compressed_append_launch::BLOCK,
                &params.words(),
            )?;
        }
        let keep = n.min(ring.rows);
        for &(_, _, scratch, ring_base) in &projections {
            ring_copy(scratch, ring_base, lead + n - keep, keep, true)?;
        }
        Ok(())
    }

    /// The indexer over the chunk's rows: the head weights, the roped and FP4 round-tripped
    /// queries, then the scores of every row against the owner's keys on the FP4 tensor cores and
    /// one select launch, into the rows' selected lists (and the candidate masks).
    fn bulk_index(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        ratio: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let refusal = |what: &str| format!("bulk indexer: layer {layer}: {what}");
        let n_max = (p0 + n) / ratio;
        if n_max == 0 {
            return Ok(());
        }
        if n_max > SELECT_MAX_POSITIONS {
            return Err(refusal("more compressed positions than the select holds"));
        }
        let (Some(staged), Some(ring)) = (self.seam_staged, self.seam_ring) else {
            return Err(refusal("no staged indexer"));
        };
        let owner = self
            .owner_index_for(layer)
            .ok_or_else(|| refusal("no kv-source owner"))?;
        let weights_proj = staged
            .weights_proj
            .ok_or_else(|| refusal("no weights_proj"))?;
        let (wq_b, wq_b_scales) = staged.indexer_wq_b.ok_or_else(|| refusal("no wq_b"))?;
        let scale_weights = self
            .seam_scale_weights
            .ok_or_else(|| refusal("no weight scale"))?;
        let fp4_rt = self
            .seam_fp4_round_trip
            .ok_or_else(|| refusal("no FP4 round trip"))?;
        let cuda = self.cuda;
        let b = rows.b;
        let (heads, index_dim) = (self.index_n_heads, self.index_head_dim);
        let index_cols = heads * index_dim;
        self.bulk_hostw(
            bulk.k.hostw[0],
            b.normed,
            weights_proj.0,
            b.idx_weights_bf16,
            heads,
            self.dim,
            n,
        )?;
        let factor = ((index_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
        sparkflash_kernels::pool_group_launch::launch_scale_index_weights(
            cuda,
            scale_weights,
            Deviceptr(b.idx_weights_bf16),
            Deviceptr(b.idx_weights_f32),
            factor,
            (n * heads) as u32,
        )?;
        let lora = self.projections[0].cols as usize;
        launch_opt(
            cuda,
            OptKind::Fp8aBf16o,
            &OptArgs {
                a: b.q_codes,
                a_scales: b.q_scales,
                w: wq_b.0,
                ws: wq_b_scales.0,
                out: b.idx_query,
                cols: index_cols as u32,
                inner: lora as u32,
                tokens: n as u32,
                a_tok_stride: lora as u64,
                out_tok_stride: index_cols as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| refusal(&format!("wq_b: {error}")))?;
        self.bulk_rope(bulk, b.idx_query, index_dim, heads, n, p0, 1)?;
        sparkflash_kernels::pool_group_launch::launch_fp4_round_trip(
            cuda,
            fp4_rt,
            Deviceptr(b.idx_query),
            Deviceptr(b.idx_q_f32),
            (n * index_cols / 32) as u32,
        )?;
        let pack = |src: u64, codes: u64, scales: u64, blocks: usize| {
            cuda.launch(
                bulk.k.pack,
                ((blocks as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                &[src, codes, scales, blocks as u64],
            )
        };
        pack(
            b.idx_q_f32,
            bulk.q_codes,
            bulk.q_scales,
            n * index_cols / 32,
        )?;
        pack(
            ring.index_keys.0 + (owner * self.max_positions * index_dim * 4) as u64,
            bulk.k_codes,
            bulk.k_scales,
            n_max * index_dim / 32,
        )?;
        let mask_mode = match layer.cmp(&self.candidate_source_layer) {
            std::cmp::Ordering::Less => 0u64,
            std::cmp::Ordering::Equal => 2,
            std::cmp::Ordering::Greater => 1,
        };
        let mask_bytes = rows.mask_bytes;
        let stride = n_max.next_multiple_of(4);
        let slice = ((SCORE_FLOATS / stride) / 4 * 4).max(4);
        let mut y0 = 0;
        while y0 < n {
            let m = slice.min(n - y0);
            let keys = (p0 + y0 + m) / ratio;
            if keys > 0 {
                cuda.launch(
                    bulk.k.scores,
                    ((keys as u32).div_ceil(128), (m as u32).div_ceil(4), 1),
                    (256, 1, 1),
                    &[
                        bulk.q_codes + (y0 * index_cols) as u64,
                        bulk.q_scales + (y0 * index_cols / 32) as u64,
                        bulk.k_codes,
                        bulk.k_scales,
                        b.idx_weights_f32 + (y0 * heads * 4) as u64,
                        bulk.scores,
                        m as u64,
                        (p0 + y0) as u64,
                        ratio as u64,
                        stride as u64,
                    ],
                )?;
                let masks = b.masks + (y0 * mask_bytes) as u64;
                cuda.launch(
                    bulk.k.select_rows,
                    (SELECT_BLOCKS as u32, 1, 1),
                    (1024, 1, 1),
                    &[
                        bulk.scores,
                        b.selected + (y0 * SELECTED_STRIDE * 4) as u64,
                        if mask_mode == 1 { masks } else { 0 },
                        if mask_mode == 2 { masks } else { 0 },
                        bulk.select_scratch,
                        m as u64,
                        (p0 + y0) as u64,
                        ratio as u64,
                        self.index_topk as u64,
                        mask_mode,
                        self.candidate_topk_blocks as u64,
                        self.candidate_block_size as u64,
                        stride as u64,
                        SELECTED_STRIDE as u64,
                        (mask_bytes / 4) as u64,
                        bulk.select_words as u64,
                    ],
                )?;
            }
            y0 += m;
        }
        Ok(())
    }

    /// The Engram wkv of the chunk's rows into `engram_kv` (`[rows][out_width]` bf16), one
    /// tensor-core GEMM against the row-major weights.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bulk_engram_wkv(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        weight: u64,
        scales: u64,
        out_width: usize,
        inner: usize,
        n: usize,
    ) -> Result<(), String> {
        if !out_width.is_multiple_of(32) || !inner.is_multiple_of(32) {
            return Err(format!(
                "bulk engram wkv: {out_width} x {inner} is not in 32 x 32 blocks"
            ));
        }
        let b = rows.b;
        self.cuda.launch(
            bulk.k.engram_wkv,
            (n.div_ceil(128) as u32, out_width.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                b.engram_codes,
                b.engram_scales,
                weight,
                scales,
                b.engram_kv,
                out_width as u64,
                inner as u64,
                n as u64,
                inner as u64,
                out_width as u64,
            ],
        )
    }

    /// The chunk's Engram stage at `slot`: the embed of every row's arena entry, the wkv, then the
    /// gates and the stream write of the rows `live` marks (three launches for the chunk).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bulk_engram(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        slot: usize,
        live: &[bool],
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let engram = self
            .engram
            .as_ref()
            .ok_or("bulk engram: the arena was not attached")?;
        let mask = bulk
            .engram_live
            .as_ref()
            .ok_or("bulk engram: no live flags")?;
        let at = slot * engram.positions + engram.base(p0);
        if live.len() != n || at + n > mask.len() {
            return Err(format!(
                "bulk engram: rows {p0}..{} are outside the flags",
                p0 + n
            ));
        }
        // SAFETY: inside the mapping (checked), and no queued launch reads these positions' flags.
        let flags =
            unsafe { std::slice::from_raw_parts_mut(mask.host_ptr().cast::<u8>().add(at), n) };
        for (flag, &on) in flags.iter_mut().zip(live) {
            *flag = u8::from(on);
        }
        let layout = engram.layout;
        let align = crate::engram_device_state::ARENA_ALIGN;
        let stride = layout.stride();
        let b = rows.b;
        self.cuda.launch(
            bulk.k.engram_embed,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                engram.arena.0 + (at * stride) as u64,
                stride as u64,
                (layout.codes_len().div_ceil(align) * align) as u64,
                b.engram_codes,
                b.engram_scales,
                engram.inner as u64,
            ],
        )?;
        self.bulk_engram_wkv(
            rows,
            bulk,
            engram.wkv_weight[slot].0,
            engram.wkv_scales[slot].0,
            engram.out_width,
            engram.inner,
            n,
        )?;
        self.cuda.launch(
            bulk.k.engram_post,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                b.stream,
                b.engram_kv,
                engram.product[slot].0,
                mask.device().0 + at as u64,
                engram.dim as u64,
                engram.hc_mult as u64,
                u64::from(engram.eps.to_bits()),
                engram.out_width as u64,
            ],
        )
    }

    /// The router over the chunk's rows: the raw gate dots on the tensor cores, then the
    /// per-token scoring and top-k into moe_ids / moe_weights.
    pub(super) fn bulk_router(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        gate: u64,
        bias: u64,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let experts = self.routed_geometry.experts;
        if experts != 384 || self.routed_geometry.top_k != 6 {
            return Err("bulk router: the kernels hold 384 experts, top 6".to_owned());
        }
        self.cuda.launch(
            bulk.k.router_dots,
            ((experts / 128) as u32, (n as u32).div_ceil(128), 1),
            (256, 1, 1),
            &[
                b.ffn_x,
                gate,
                bulk.dots,
                experts as u64,
                self.dim as u64,
                n as u64,
            ],
        )?;
        self.cuda.launch(
            bulk.k.router_topk,
            ((n as u32).div_ceil(8), 1, 1),
            (256, 1, 1),
            &[bulk.dots, bias, b.moe_ids, b.moe_weights, n as u64],
        )
    }
}

/// The selected list's row stride in u32 ids (as the rows step's).
const SELECTED_STRIDE: usize = 512;
