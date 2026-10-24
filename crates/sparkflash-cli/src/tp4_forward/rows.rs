//! The multi-row (speculative verify) decode step: up to [`R_MAX`] consecutive positions through
//! every layer with one pass over the weights. Row r is position p0 + r.
//!
//! Every stage is the M = 1 stage's arithmetic per row: the batched kernels are
//! kernels/opt_rows_sm121.cu and the t8 GEMV instantiations, written expression for expression
//! against the single-row kernels, and the stages without a batched form (the compressor seam, the
//! indexer select, the Engram gather and gate) run the M = 1 code once per row with the layer's
//! buffer views pointed at that row. The two collectives are one NCCL group of per-row reduces, so
//! each element is summed in the single-row order.
//!
//! Only the opt fast path is served (opt seams, GEMV, rope, attention, router, fused MoE with the
//! rank-split shared expert); [`RowsState::open`] refuses any other configuration.
//!
//! A row past the accepted prefix leaves state behind (its KV window row, its compressor ring row,
//! its compressed store row): every one of those is addressed by position, so the next step that
//! decodes that position overwrites it before anything reads it. [`DeviceLayer::rewind_rows`] only
//! moves the position counters.

use super::*;
use crate::layer_device::DeviceSpan;
use sparkflash_kernels::cuda_driver::{Kernel, Module};
use sparkflash_kernels::mxfp8_gemv_launch::{OptArgs, OptKind, launch_opt};

/// The widest verify step: the t8 GEMV, the router and the head kernels hold eight rows.
pub(super) const R_MAX: usize = 8;

/// The mix table's coefficient rows (pre 4, post 4, comb 16).
const MIX_ROWS: usize = 24;
/// opt_rows_mhc_seam's per-row scratch: 80 blocks x 32 partial words.
const SEAM_PARTIAL_FLOATS: usize = 80 * 32;
/// The transport sequence range the rows head's argmax gather uses.
const ROWS_GATHER_SEQ: u64 = u64::MAX - 0xC_0000;
/// The most ranks the device gathers carve a slot for.
pub(super) const GATHER_WORLD: usize = 8;

pub(super) struct RowsKernels {
    _modules: Vec<Module>,
    pub(super) seam: Kernel,
    pub(super) post: Kernel,
    pub(super) kv: Kernel,
    attn: Kernel,
    pub(super) inv_rope: Kernel,
    lm_head: Kernel,
    pub(super) fold: Kernel,
    /// opt_rows_argmax_pick and opt_rows_store_u32: the device side of the argmax gathers.
    pub(super) pick: Kernel,
    pub(super) store: Kernel,
    /// opt_rows_embed_own and opt_rows_embed_stream: the rows' embedding streams from device ids.
    embed_own: Kernel,
    embed_stream: Kernel,
    engram_wkv: Kernel,
    rank_permute: Kernel,
    /// opt_rows_hostw_{bf16w, bf16w_f32o, f32w_f32o}, by hostw_gemv_launch::Mode.
    hostw: [Kernel; 3],
    prefetch_l2: Kernel,
}

/// The per-row device buffers, each `[R_MAX][natural width]`.
#[derive(Clone, Copy)]
pub(super) struct RowsBuffers {
    pub(super) stream: u64,
    pub(super) hc_rows: u64,
    pub(super) normed: u64,
    pub(super) pre_mix: u64,
    pub(super) attn_mix: u64,
    pub(super) ffn_mix: u64,
    pub(super) act_codes: u64,
    pub(super) act_scales: u64,
    pub(super) q_lora: u64,
    pub(super) q_normed: u64,
    pub(super) q_codes: u64,
    pub(super) q_scales: u64,
    pub(super) q: u64,
    pub(super) kv_latent: u64,
    pub(super) kv_rotated: u64,
    pub(super) attn_body: u64,
    pub(super) attn_lo: u64,
    pub(super) o_codes: u64,
    pub(super) o_scales: u64,
    pub(super) attn_out: u64,
    pub(super) reduced: u64,
    pub(super) seam_row: u64,
    pub(super) ffn_codes: u64,
    pub(super) ffn_scales: u64,
    pub(super) ffn_x: u64,
    pub(super) moe_ids: u64,
    pub(super) moe_weights: u64,
    pub(super) scores: u64,
    pub(super) delta_f32: u64,
    pub(super) delta_bf16: u64,
    pub(super) moe_reduced: u64,
    pub(super) shared_gate: u64,
    pub(super) shared_swiglu: u64,
    pub(super) shared_out: u64,
    pub(super) selected: u64,
    pub(super) masks: u64,
    pub(super) seam_partials: u64,
    pub(super) seam_row_buf: u64,
    pub(super) seam_tickets: u64,
    pub(super) engram_codes: u64,
    pub(super) engram_scales: u64,
    pub(super) engram_kv: u64,
    /// The split Engram wkv's per-rank slices, `[world][rows][out_width / world]`.
    pub(super) engram_slices: u64,
    /// The indexer's per-row weights (bf16, then scaled f32) and query (bf16, then the f32 round
    /// trip), for ref_select to read row by row.
    pub(super) idx_weights_bf16: u64,
    pub(super) idx_weights_f32: u64,
    pub(super) idx_query: u64,
    pub(super) idx_q_f32: u64,
    pub(super) lm_hidden: u64,
    pub(super) lm_normed: u64,
    pub(super) lm_block_max: u64,
    pub(super) lm_block_arg: u64,
    pub(super) lm_value: u64,
    pub(super) lm_index: u64,
    /// `[GATHER_WORLD][values R_MAX | indices R_MAX]`: each rank's argmax fold, exchanged in place.
    pub(super) argmax_gather: u64,
    /// The verify head's picks, `[ids rows | values rows]`.
    pub(super) lm_picks: u64,
    /// `[GATHER_WORLD][R_MAX][dim]` bf16: each rank's owned embedding rows, exchanged in place.
    embed_gather: u64,
}

/// The run-scope state of the multi-row step: its kernels and row buffers.
pub(super) struct RowsState {
    pub(super) kernels: RowsKernels,
    pub(super) b: RowsBuffers,
    allocations: Vec<Deviceptr>,
    /// Widths the buffers were carved for.
    q_cols: usize,
    lora_cols: usize,
    kv_cols: usize,
    lo_cols: usize,
    pub(super) shared_local: usize,
    mask_bytes: usize,
    engram_inner: usize,
    engram_out: usize,
    gathers: Cell<u64>,
    /// The DSpark tap capture (spec mode): opt_draft_taps, the `[R_MAX][3][dim]` f32 tap buffer,
    /// and the target layers in tap order. Before each target layer the rows' streams are
    /// hc-averaged into their tap slot.
    pub(super) taps: Option<(Kernel, u64, Vec<usize>)>,
    /// The side stream the all-reduce windows' L2 prefetches run on (SPARKFLASH_L2_PREFETCH=0: none).
    side: Option<sparkflash_kernels::cuda_driver::SideStream>,
}

impl RowsState {
    /// Loads opt_rows_sm121.cubin and carves the row buffers for `device`'s geometry.
    ///
    /// # Errors
    /// Refuses a layer whose fast path is not the opt one, or a geometry the row kernels do not
    /// hold, and returns any load or allocation error.
    pub(super) fn open(device: &DeviceLayer<'_>, kernel_dir: &Path) -> Result<Self, String> {
        let refuse = |what: &str| Err(format!("rows: {what}"));
        if device.dim != 5120 || device.hc_mult != 4 {
            return refuse("the row kernels hold dim 5120 and hc 4 only");
        }
        if !device.opt_seams_active()
            || device.opt_rope.is_none()
            || device.opt_router.is_none()
            || device.opt_attn.is_none()
            || !device.shared_tp_active()
            || !sparkflash_kernels::mxfp8_gemv_launch::opt_installed()
        {
            return refuse(
                "the multi-row step needs the opt seams, rope, router, attention, GEMV and the \
                 rank-split shared expert",
            );
        }
        let cuda = device.cuda;
        let (cubin, _sha) = cached_cubin(&kernel_dir.join("opt_rows_sm121.cubin"))
            .map_err(|error| format!("rows: {error}"))?;
        let mut modules = Vec::new();
        let mut load = |name: &'static str| -> Result<Kernel, String> {
            let (module, kernel) = cuda.load_named(&cubin, name)?;
            modules.push(module);
            Ok(kernel)
        };
        let seam = load("opt_rows_mhc_seam")?;
        let post = load("opt_rows_mhc_post")?;
        let kv = load("opt_rows_kv")?;
        let attn = load("opt_rows_attn")?;
        let inv_rope = load("opt_rows_inv_rope")?;
        let lm_head = load("opt_rows_lm_head")?;
        let fold = load("opt_rows_argmax_fold")?;
        let pick = load("opt_rows_argmax_pick")?;
        let store = load("opt_rows_store_u32")?;
        let embed_own = load("opt_rows_embed_own")?;
        let embed_stream = load("opt_rows_embed_stream")?;
        let engram_wkv = load("opt_rows_engram_wkv")?;
        let rank_permute = load("opt_rows_rank_permute")?;
        let hostw = [
            load("opt_rows_hostw_bf16w")?,
            load("opt_rows_hostw_bf16w_f32o")?,
            load("opt_rows_hostw_f32w_f32o")?,
        ];
        let prefetch_l2 = load("opt_rows_prefetch_l2")?;
        let kernels = RowsKernels {
            _modules: modules,
            seam,
            post,
            kv,
            attn,
            inv_rope,
            lm_head,
            fold,
            pick,
            store,
            embed_own,
            embed_stream,
            engram_wkv,
            rank_permute,
            hostw,
            prefetch_l2,
        };

        let dim = device.dim;
        let hc_width = device.hc_mult * dim;
        let lora_cols = device.projections[0].cols as usize;
        let kv_cols = device.projections[1].cols as usize;
        let q_cols = device.projections[2].cols as usize;
        let lo_cols = device.out_geometry.groups * device.out_geometry.o_lora_rank;
        let shared = device
            .shared
            .as_ref()
            .ok_or("rows: no shared expert operands")?;
        let shared_local = shared.inter / device.routed_geometry.world;
        let top_k = device.routed_geometry.top_k;
        let experts = device.routed_geometry.experts;
        let mask_bytes = sparkflash_kernels::ref_select_launch::MASK_WORDS * 4;
        let (engram_inner, engram_out) = device
            .engram
            .as_ref()
            .map_or((32, 8), |engram| (engram.inner, engram.out_width));
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        let index_heads = device.index_n_heads;
        let index_cols = index_heads * device.index_head_dim;
        let r = R_MAX;
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize| -> Result<u64, String> {
            let ptr = cuda.alloc(bytes.max(4), &mut total)?;
            cuda.copy_htod(&ptr, &vec![0u8; bytes.max(4)])?;
            allocations.push(ptr);
            Ok(ptr.0)
        };
        let b = RowsBuffers {
            stream: alloc(r * hc_width * 4)?,
            hc_rows: alloc(r * hc_width * 4)?,
            normed: alloc(r * dim * 4)?,
            pre_mix: alloc(r * device.hc_mult * 4)?,
            attn_mix: alloc(r * MIX_ROWS * 4)?,
            ffn_mix: alloc(r * MIX_ROWS * 4)?,
            act_codes: alloc(r * dim)?,
            act_scales: alloc(r * dim / 32)?,
            q_lora: alloc(r * lora_cols * 2)?,
            q_normed: alloc(r * lora_cols * 2)?,
            q_codes: alloc(r * lora_cols)?,
            q_scales: alloc(r * lora_cols / 32)?,
            q: alloc(r * q_cols * 2)?,
            kv_latent: alloc(r * kv_cols * 2)?,
            kv_rotated: alloc(r * kv_cols * 2)?,
            attn_body: alloc(r * q_cols * 2)?,
            attn_lo: alloc(r * lo_cols * 2)?,
            o_codes: alloc(r * lo_cols)?,
            o_scales: alloc(r * lo_cols / 32)?,
            attn_out: alloc(r * dim * 2)?,
            reduced: alloc(r * dim * 2)?,
            seam_row: alloc(r * dim * 4)?,
            ffn_codes: alloc(r * dim)?,
            ffn_scales: alloc(r * dim / 32)?,
            ffn_x: alloc(r * dim * 2)?,
            moe_ids: alloc(r * top_k * 4)?,
            moe_weights: alloc(r * top_k * 4)?,
            scores: alloc(r * experts * 4)?,
            delta_f32: alloc(r * dim * 4)?,
            delta_bf16: alloc(r * dim * 2)?,
            moe_reduced: alloc(r * dim * 2)?,
            shared_gate: alloc(2 * r * shared_local * 4)?,
            shared_swiglu: alloc(r * shared_local * 4)?,
            shared_out: alloc(r * dim * 4)?,
            selected: alloc(r * SELECTED_STRIDE * 4)?,
            masks: alloc(r * mask_bytes)?,
            seam_partials: alloc(r * SEAM_PARTIAL_FLOATS * 4)?,
            seam_row_buf: alloc(r * dim * 4)?,
            seam_tickets: alloc(r * 4)?,
            engram_codes: alloc(r * engram_inner)?,
            engram_scales: alloc(r * engram_inner / 32)?,
            engram_kv: alloc(r * engram_out * 2)?,
            engram_slices: alloc(r * engram_out * 2)?,
            idx_weights_bf16: alloc(r * index_heads * 2)?,
            idx_weights_f32: alloc(r * index_heads * 4)?,
            idx_query: alloc(r * index_cols * 2)?,
            idx_q_f32: alloc(r * index_cols * 4)?,
            lm_hidden: alloc(r * dim * 2)?,
            lm_normed: alloc(r * dim * 2)?,
            lm_block_max: alloc(r * chunks * 4)?,
            lm_block_arg: alloc(r * chunks * 4)?,
            lm_value: alloc(r * 4)?,
            lm_index: alloc(r * 4)?,
            argmax_gather: alloc(GATHER_WORLD * r * 8)?,
            lm_picks: alloc(r * 8)?,
            embed_gather: alloc(GATHER_WORLD * r * dim * 2)?,
        };
        Ok(Self {
            kernels,
            b,
            allocations,
            q_cols,
            lora_cols,
            kv_cols,
            lo_cols,
            shared_local,
            mask_bytes,
            engram_inner,
            engram_out,
            gathers: Cell::new(0),
            taps: None,
            side: match std::env::var("SPARKFLASH_L2_PREFETCH").as_deref() {
                Ok("0") => None,
                _ => Some(cuda.side_stream()?),
            },
        })
    }

    /// Frees the row buffers (the modules drop with the state).
    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

/// The selected list's row stride in u32 ids (`workspace.selected` is 2,048 B).
const SELECTED_STRIDE: usize = 512;

fn span(device: u64, len: usize) -> DeviceSpan {
    DeviceSpan::new(device, len)
}

impl DeviceLayer<'_> {
    /// Loads `n` rows' embeddings (`hidden` is `[n][dim]` f32) as rows p0..p0+n: each row's stream
    /// is the hc copies of its embedding and its pre mix the identity seed, as `begin_position`.
    ///
    /// # Errors
    /// Refuses a row count outside 1..=R_MAX or a first row other than the next position.
    pub(super) fn begin_rows(
        &mut self,
        rows: &RowsState,
        p0: usize,
        hidden: &[f32],
    ) -> Result<usize, String> {
        let dim = self.dim;
        let n = hidden.len() / dim;
        if hidden.len() != n * dim {
            return Err(format!(
                "rows: {} values are not 1..={R_MAX} rows of {dim}",
                hidden.len()
            ));
        }
        self.check_begin(p0, n)?;
        let mut stream = Vec::with_capacity(n * self.hc_mult * dim * 4);
        let mut seed = Vec::with_capacity(n * self.hc_mult * 4);
        for row in hidden.chunks_exact(dim) {
            for _ in 0..self.hc_mult {
                for value in row {
                    stream.extend_from_slice(&value.to_le_bytes());
                }
            }
            for copy in 0..self.hc_mult {
                seed.extend_from_slice(&(if copy == 0 { 1.0f32 } else { 0.0 }).to_le_bytes());
            }
        }
        self.cuda.copy_htod(&Deviceptr(rows.b.stream), &stream)?;
        self.cuda.copy_htod(&Deviceptr(rows.b.pre_mix), &seed)?;
        self.positions_begun = p0 + n;
        self.attn_window = AttentionWindow::default();
        Ok(n)
    }

    /// [Self::begin_rows] for the `n` token ids at `ids` (device u32): the rows are gathered from
    /// the owners' shards of `embed` over the exchange, with no host copy.
    pub(super) fn begin_rows_device(
        &mut self,
        rows: &RowsState,
        p0: usize,
        embed: u64,
        ids: u64,
        n: usize,
        exchange: (usize, usize),
    ) -> Result<usize, String> {
        self.check_begin(p0, n)?;
        self.embed_rows_on_device(rows, embed, ids, n, n, 0, exchange)?;
        self.positions_begun = p0 + n;
        self.attn_window = AttentionWindow::default();
        Ok(n)
    }

    fn check_begin(&self, p0: usize, n: usize) -> Result<(), String> {
        if n == 0 || n > R_MAX {
            return Err(format!("rows: {n} rows are not 1..={R_MAX}"));
        }
        if p0 != self.positions_begun {
            return Err(format!(
                "rows: first row at position {p0} but {} positions have begun",
                self.positions_begun
            ));
        }
        if p0 + n > self.max_positions {
            return Err(format!(
                "rows: positions {p0}..{} pass the {}-position run",
                p0 + n,
                self.max_positions
            ));
        }
        Ok(())
    }

    /// Writes `total` rows' streams and identity pre mixes: rows below `n` are the embeddings of
    /// the token ids at `ids` (device u32), rows from `n` on are `noise` (device f32 `[dim]`). Each
    /// owner copies its rows into its slot of the embedding gather and the exchange fills the
    /// peers' slots, so the rows never pass through the host.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn embed_rows_on_device(
        &self,
        rows: &RowsState,
        embed: u64,
        ids: u64,
        n: usize,
        total: usize,
        noise: u64,
        (_world, rank): (usize, usize),
    ) -> Result<(), String> {
        if self.dim != HIDDEN || self.hc_mult != 4 || n == 0 || n > total || total > R_MAX {
            return Err(format!(
                "rows: the device embedding gather takes 1..={R_MAX} rows of {} x 4, not {n} of {total} rows of {} x {}",
                HIDDEN, self.dim, self.hc_mult
            ));
        }
        let gather = rows.b.embed_gather;
        let count = n * self.dim;
        self.cuda.launch(
            rows.kernels.embed_own,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                embed,
                ids,
                rank as u64,
                crate::tp4_tail::VOCAB_PER_RANK as u64,
                gather + (rank * count * 2) as u64,
            ],
        )?;
        self.exchange_slots(gather, count, rank, "embedding gather")?;
        self.cuda.launch(
            rows.kernels.embed_stream,
            (total as u32, 1, 1),
            (256, 1, 1),
            &[
                gather,
                ids,
                n as u64,
                crate::tp4_tail::VOCAB_PER_RANK as u64,
                noise,
                rows.b.stream,
                rows.b.pre_mix,
            ],
        )
    }

    /// Moves the position counters back to `next` after a verify step accepted rows below it: every
    /// state a later row wrote is addressed by position and is rewritten before it is read.
    pub(super) fn rewind_rows(&mut self, next: usize) {
        self.positions_begun = next;
        for count in &mut self.appended_rows {
            *count = (*count).min(next);
        }
    }

    /// One layer over rows p0..p0+n.
    pub(super) fn run_layer_rows(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        if let Some((kernel, taps, layers)) = rows.taps.as_ref()
            && let Some(tap) = layers.iter().position(|&target| target == layer)
        {
            self.cuda.launch(
                *kernel,
                (20, n as u32, 1),
                (256, 1, 1),
                &[b.stream, *taps, tap as u64],
            )?;
        }
        self.rows_engram(rows, layer, p0, n)?;
        let weights = self.seam_layer(layer, "rows entry")?;
        self.rows_seam(
            rows,
            n,
            b.stream,
            b.pre_mix,
            self.hc_mult as u32,
            [
                weights.attn_fn.0,
                weights.attn_base.0,
                weights.attn_scale.0,
                weights.attn_norm.0,
            ],
            b.attn_mix,
            0,
            b.hc_rows,
            b.normed,
            b.act_codes,
            b.act_scales,
            0,
            1,
        )?;
        self.rows_query_kv(rows, layer, p0, n)?;
        self.rows_seam_select(rows, layer, p0, n)?;
        self.rows_attention(rows, layer, p0, n)?;
        self.rows_output(rows, layer, n)?;
        self.prefetch_ffn(rows, layer, &weights)?;
        self.reduce_rows(
            "all_reduce",
            span(b.attn_out, n * dim * 2),
            span(b.reduced, n * dim * 2),
            n,
        )?;
        self.rows_post(rows, n, b.hc_rows, b.attn_mix, b.reduced, b.stream)?;
        self.rows_seam(
            rows,
            n,
            b.stream,
            b.attn_mix,
            MIX_ROWS as u32,
            [
                weights.ffn_fn.0,
                weights.ffn_base.0,
                weights.ffn_scale.0,
                weights.ffn_norm.0,
            ],
            b.ffn_mix,
            b.pre_mix,
            0,
            b.seam_row,
            b.ffn_codes,
            b.ffn_scales,
            b.ffn_x,
            0,
        )?;
        self.rows_moe(rows, layer, p0, n)?;
        self.prefetch_attention(rows, layer + 1)?;
        self.reduce_rows(
            "all_reduce_moe",
            span(b.delta_bf16, n * dim * 2),
            span(b.moe_reduced, n * dim * 2),
            n,
        )?;
        self.rows_post(rows, n, b.stream, b.ffn_mix, b.moe_reduced, b.stream)
    }

    /// Queues opt_rows_prefetch_l2 over `spans` on the side stream (none without one): it starts
    /// once everything queued so far is done, i.e. as the run's stream enters the all-reduce.
    fn prefetch_l2(&self, rows: &RowsState, spans: [(u64, usize); 4]) -> Result<(), String> {
        let Some(side) = rows.side.as_ref() else {
            return Ok(());
        };
        let args: Vec<u64> = spans
            .iter()
            .flat_map(|&(address, bytes)| [address, bytes as u64])
            .collect();
        self.cuda
            .side_launch(side, rows.kernels.prefetch_l2, (1, 1, 1), (32, 1, 1), &args)
    }

    /// During the attention all-reduce: what the FFN half reads first, the FFN seam's mix weights,
    /// the router's gate and the rank's shared gate/up slices.
    fn prefetch_ffn(
        &self,
        rows: &RowsState,
        layer: usize,
        weights: &SeamLayerWeights,
    ) -> Result<(), String> {
        let hc_width = self.hc_mult * self.dim;
        let gate = self
            .router
            .as_ref()
            .and_then(|router| router.get(layer).copied())
            .map_or((0, 0), |(gate, _)| {
                (gate.0, self.routed_geometry.experts * self.dim * 2)
            });
        let local = rows.shared_local;
        let (w1, w3) = self
            .shared
            .as_ref()
            .and_then(|shared| shared.layers.get(layer).copied())
            .map_or(((0, 0), (0, 0)), |ops| {
                let offset =
                    self.tp_rank as u64 * (local / 32) as u64 * (self.dim / 32) as u64 * 1024;
                (
                    (ops.w1.0 + offset, local * self.dim),
                    (ops.w3.0 + offset, local * self.dim),
                )
            });
        self.prefetch_l2(
            rows,
            [(weights.ffn_fn.0, MIX_ROWS * hc_width * 4), gate, w1, w3],
        )
    }

    /// During the MoE all-reduce: what layer `layer` (the next one) reads first, its entry seam's
    /// mix weights, wq_a and wkv.
    fn prefetch_attention(&self, rows: &RowsState, layer: usize) -> Result<(), String> {
        let Ok(weights) = self.seam_layer(layer, "prefetch") else {
            return Ok(());
        };
        let hc_width = self.hc_mult * self.dim;
        let matrix = |name: &str, cols: usize| {
            self.weights
                .pair(layer, name)
                .map_or((0, 0), |(w, _)| (w.0, cols * self.dim))
        };
        self.prefetch_l2(
            rows,
            [
                (weights.attn_fn.0, MIX_ROWS * hc_width * 4),
                matrix("wq_a", rows.lora_cols),
                matrix("wkv", rows.kv_cols),
                (0, 0),
            ],
        )
    }

    /// opt_rows_mhc_seam: the entry (hc_rows set, narrowed codes) or the FFN seam (pre_mix_out,
    /// codes and x_bf16 set). `in_pre_stride` is 4 for the pre mix rows, 24 for the attention mix.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_seam(
        &self,
        rows: &RowsState,
        n: usize,
        stream: u64,
        in_pre: u64,
        in_pre_stride: u32,
        w: [u64; 4],
        mix: u64,
        pre_mix_out: u64,
        hc_rows: u64,
        normed: u64,
        codes: u64,
        scales: u64,
        x_bf16: u64,
        quant_narrow: u32,
    ) -> Result<(), String> {
        let b = rows.b;
        self.cuda.launch(
            rows.kernels.seam,
            (80, n as u32, 1),
            (256, 1, 1),
            &[
                stream,
                in_pre,
                w[0],
                w[1],
                w[2],
                w[3],
                mix,
                pre_mix_out,
                hc_rows,
                normed,
                codes,
                scales,
                x_bf16,
                b.seam_partials,
                b.seam_row_buf,
                b.seam_tickets,
                u64::from(SEAM_NORM_BF16),
                u64::from(self.native_bf16),
                u64::from(quant_narrow),
                u64::from(self.sinkhorn_iters),
                u64::from(self.eps.to_bits()),
                u64::from(self.hc_eps.to_bits()),
                u64::from(in_pre_stride),
                MIX_ROWS as u64,
            ],
        )
    }

    pub(super) fn rows_post(
        &self,
        rows: &RowsState,
        n: usize,
        residual: u64,
        mix: u64,
        delta: u64,
        out: u64,
    ) -> Result<(), String> {
        self.cuda.launch(
            rows.kernels.post,
            (40, n as u32, 1),
            (128, 1, 1),
            &[
                residual,
                mix,
                delta,
                0,
                out,
                u64::from(self.native_bf16),
                MIX_ROWS as u64,
            ],
        )
    }

    /// The Engram write per row: the gather per row, one batched wkv over the rows, then the gate
    /// and the stream write per row (the M = 1 launches over the row's buffers).
    fn rows_engram(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::{
            engram_convert_launch::{self, EngramConvertParams, EngramConvertSpans},
            engram_embed_launch::{self, EngramEmbedLaunchParams},
            engram_gate_write_launch::{self, EngramGateWriteLaunchParams},
            engram_write_launch::{self, EngramWriteParams, EngramWriteSpans},
        };
        if !self.engram_rows.contains(&layer) {
            return Ok(());
        }
        let engram = self
            .engram
            .as_ref()
            .ok_or("rows engram: the arena was not attached")?;
        if engram.inner != rows.engram_inner || engram.out_width != rows.engram_out {
            return Err("rows engram: the row buffers were carved for another geometry".to_owned());
        }
        let slot = engram.layer_slot(layer)?;
        let mut live = Vec::with_capacity(n);
        for row in 0..n {
            live.push(!engram.is_skipped(slot, p0 + row)?);
        }
        if !live.iter().any(|&on| on) {
            return Ok(());
        }
        let b = rows.b;
        let layout = engram.layout;
        let align = crate::engram_device_state::ARENA_ALIGN;
        let (hc, dim) = (engram.hc_mult, engram.dim);
        let layer_id = layer as u32;
        for row in 0..n {
            let entry = slot * engram.positions + p0 + row;
            let base = engram.arena.0 + (entry * layout.stride()) as u64;
            engram_embed_launch::launch(
                self.cuda,
                engram.kernels.embed,
                &EngramEmbedLaunchParams {
                    row_codes: Deviceptr(base),
                    row_scales: Deviceptr(
                        base + (layout.codes_len().div_ceil(align) * align) as u64,
                    ),
                    fetched: engram.fetched,
                    act_codes: Deviceptr(b.engram_codes + (row * engram.inner) as u64),
                    act_scales: Deviceptr(b.engram_scales + (row * engram.inner / 32) as u64),
                    rows_cols: layout.columns() as u32,
                    head_dim: layout.head_dim() as u32,
                    trace_ring: 0,
                    layer_id,
                    step_seq: (p0 + row) as u64,
                },
            )?;
        }
        self.rows_engram_wkv(
            rows,
            engram.wkv_weight[slot].0,
            engram.wkv_scales[slot].0,
            engram.out_width,
            engram.inner,
            n,
        )?;
        let stream_bytes = hc * dim * 4;
        for (row, &on) in live.iter().enumerate() {
            if !on {
                continue;
            }
            let stream = b.stream + (row * stream_bytes) as u64;
            let kv = b.engram_kv + (row * engram.out_width * 2) as u64;
            let step_seq = (p0 + row) as u64;
            engram_convert_launch::launch_narrow(
                self.cuda,
                engram.kernels.narrow,
                &EngramConvertParams {
                    src: Deviceptr(stream),
                    dst: engram.hs_bf16,
                    count: (hc * dim) as u32,
                },
                &EngramConvertSpans {
                    src_bytes: stream_bytes,
                    dst_bytes: hc * dim * 2,
                },
            )?;
            engram_gate_write_launch::launch(
                self.cuda,
                engram.kernels.gate,
                &EngramGateWriteLaunchParams {
                    kv: Deviceptr(kv),
                    stream: engram.hs_bf16,
                    product: engram.product[slot],
                    out: engram.hs_bf16,
                    gates: engram.gates,
                    dim: dim as u32,
                    hc: hc as u32,
                    eps: engram.eps,
                    trace_ring: 0,
                    layer_id,
                    step_seq,
                },
            )?;
            engram_convert_launch::launch_widen(
                self.cuda,
                engram.kernels.widen,
                &EngramConvertParams {
                    src: Deviceptr(kv + (hc * dim * 2) as u64),
                    dst: engram.value_f32,
                    count: dim as u32,
                },
                &EngramConvertSpans {
                    src_bytes: dim * 2,
                    dst_bytes: dim * 4,
                },
            )?;
            engram_write_launch::launch(
                self.cuda,
                engram.kernels.write,
                &EngramWriteParams {
                    stream: Deviceptr(stream),
                    gate: engram.gates,
                    value: engram.value_f32,
                    stream_out: Deviceptr(stream),
                    dim: dim as u32,
                    copies: hc as u32,
                    trace_ring: 0,
                    layer_id,
                    step_seq,
                },
                &EngramWriteSpans {
                    stream_bytes,
                    gate_bytes: hc * 4,
                    value_bytes: dim * 4,
                    stream_out_bytes: stream_bytes,
                },
            )?;
        }
        Ok(())
    }

    /// The Engram wkv of the rows into `engram_kv` (`[rows][out_width]` bf16). With the one-shot
    /// exchange, each rank computes its quarter of the outputs and the slices are gathered and
    /// put back in row order (SPARKFLASH_ENGRAM_SPLIT=0 computes every output on every rank).
    fn rows_engram_wkv(
        &self,
        rows: &RowsState,
        weight: u64,
        scales: u64,
        out_width: usize,
        inner: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let wkv = |row_base: usize, row_end: usize, out: u64, stride: usize| {
            self.cuda.launch(
                rows.kernels.engram_wkv,
                (((row_end - row_base) as u32).div_ceil(8), 1, 1),
                (256, 1, 1),
                &[
                    b.engram_codes,
                    b.engram_scales,
                    weight,
                    scales,
                    out,
                    out_width as u64,
                    inner as u64,
                    n as u64,
                    row_base as u64,
                    row_end as u64,
                    stride as u64,
                ],
            )
        };
        let split = self.reduce.and_then(|reduce| {
            let handle = reduce.0.try_borrow().ok()?;
            let (world, rank) = handle.world_rank();
            (handle.has_exchange()
                && world > 1
                && out_width.is_multiple_of(world * 8)
                && std::env::var("SPARKFLASH_ENGRAM_SPLIT").as_deref() != Ok("0"))
            .then_some((world, rank))
        });
        let Some((world, rank)) = split else {
            return wkv(0, out_width, b.engram_kv, out_width);
        };
        let chunk = out_width / world;
        let mine = b.engram_slices + (rank * n * chunk * 2) as u64;
        wkv(rank * chunk, (rank + 1) * chunk, mine, chunk)?;
        let reduce = self.reduce.ok_or("rows engram: no reduce handle")?;
        let mut handle = reduce
            .0
            .try_borrow_mut()
            .map_err(|_| "rows engram: the reduce handle is already borrowed".to_owned())?;
        // SAFETY: the slices buffer holds world * n * chunk bf16 values (R_MAX rows carved), this
        // rank's slice is the send buffer and every peer's lands in its own slot; stream 0 is the
        // stream every launch of this run uses.
        unsafe { handle.exchange_bf16(mine, b.engram_slices, n * chunk, 0) }
            .map_err(|error| format!("rows engram: slice exchange: {error}"))?;
        drop(handle);
        self.collectives_in_flight
            .set(self.collectives_in_flight.get().saturating_add(1));
        let vecs = (chunk * 2 / 16) as u32;
        self.cuda.launch(
            rows.kernels.rank_permute,
            ((world as u32 * n as u32 * vecs).div_ceil(256), 1, 1),
            (256, 1, 1),
            &[
                b.engram_slices,
                b.engram_kv,
                n as u64,
                u64::from(vecs),
                world as u64,
            ],
        )
    }

    /// The rope table pair (cos base, sin base) of `layer`'s class.
    pub(super) fn rows_rope_tables(&self, layer: usize) -> Result<(u64, u64), String> {
        let ratio = self
            .compress_ratio
            .get(layer)
            .copied()
            .ok_or("rows: no rope class")?;
        let span = if ratio > 0 {
            self.buffers.workspace.rope_tables_compressed
        } else {
            self.buffers.workspace.rope_tables_plain
        };
        let table_bytes = self.max_positions * self.rope_pairs as usize * 4;
        Ok((span.device, span.device + table_bytes as u64))
    }

    /// wq_a, wkv, q_norm, the latent quantiser, wq_b and the q rope; then the kv norm, rope, window
    /// row and cache append.
    fn rows_query_kv(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::layer_seams_launch as seams;
        let b = rows.b;
        let dim = self.dim;
        let tokens = n as u32;
        let gemv = |name: &str,
                    a: u64,
                    a_scales: u64,
                    out: u64,
                    cols: usize,
                    inner: usize|
         -> Result<(), String> {
            let (w, ws) = self
                .weights
                .pair(layer, name)
                .map_err(|error| format!("rows {name}: {error}"))?;
            launch_opt(
                self.cuda,
                OptKind::Fp8aBf16o,
                &OptArgs {
                    a,
                    a_scales,
                    w: w.0,
                    ws: ws.0,
                    out,
                    cols: cols as u32,
                    inner: inner as u32,
                    tokens,
                    a_tok_stride: inner as u64,
                    out_tok_stride: cols as u64,
                    groups: 1,
                    a_grp_stride: 0,
                    w_grp_stride: 0,
                    s_grp_stride: 0,
                    out_grp_stride: 0,
                    k_stride_blocks: 0,
                },
            )
            .map_err(|error| format!("rows {name}: {error}"))
        };
        let lora = rows.lora_cols;
        gemv("wq_a", b.act_codes, b.act_scales, b.q_lora, lora, dim)?;
        gemv(
            "wkv",
            b.act_codes,
            b.act_scales,
            b.kv_latent,
            rows.kv_cols,
            dim,
        )?;
        let q_norm = self
            .weights
            .get(layer, "q_norm")
            .map_err(|error| format!("rows q_norm: {error}"))?;
        self.row_norm(
            span(b.q_lora, n * lora * 2),
            span(q_norm.0, lora * 2),
            span(b.q_normed, n * lora * 2),
            tokens,
            lora as u32,
            layer as u32,
            p0 as u64,
        )?;
        seams::launch_prologue_bf16(
            self.cuda,
            self.kernels.prologue_bf16,
            &seams::PrologueBf16Params {
                normed: Deviceptr(b.q_normed),
                codes: Deviceptr(b.q_codes),
                scales: Deviceptr(b.q_scales),
                cols: lora as u32,
                rows: tokens,
                trace_ring: 0,
                layer_id: layer as u32,
                step_seq: p0 as u64,
            },
        )?;
        gemv("wq_b", b.q_codes, b.q_scales, b.q, rows.q_cols, lora)?;
        let (cos, sin) = self.rows_rope_tables(layer)?;
        let pair_bytes = (self.rope_pairs as usize * 4 * p0) as u64;
        let (_, rope) = self.opt_rope.ok_or("rows: the opt rope is not loaded")?;
        self.cuda.launch(
            rope,
            (self.heads_per_rank, tokens, 1),
            (32, 1, 1),
            &[
                b.q,
                cos + pair_bytes,
                sin + pair_bytes,
                u64::from(self.head_width),
                u64::from(self.rope_head_dim),
                u64::from(self.heads_per_rank),
                u64::from(self.rope_pairs),
            ],
        )?;
        // The kv tail: kv_norm, rope, the window row's round trip, the cache append, per row.
        let kv_norm = self
            .weights
            .get(layer, "kv_norm")
            .map_err(|error| format!("rows kv_norm: {error}"))?;
        let ring_layer = self.kv_ring_base + self.kv_ring.layer_offset(layer) as u64;
        let ring_codes = ring_layer + self.kv_ring.codes_offset(0)? as u64;
        let ring_scales = ring_layer + self.kv_ring.scales_offset(0)? as u64;
        let cache = self.kv_cache.layer_span(layer)?;
        let appended = self
            .appended_rows
            .get(layer)
            .copied()
            .ok_or("rows: no cache count")?;
        if appended != p0 {
            return Err(format!(
                "rows kv: layer {layer}'s cache holds {appended} rows, the step starts at {p0}"
            ));
        }
        self.cuda.launch(
            rows.kernels.kv,
            (tokens, 1, 1),
            (256, 1, 1),
            &[
                b.kv_latent,
                kv_norm.0,
                u64::from(self.eps.to_bits()),
                cos,
                sin,
                b.kv_rotated,
                ring_codes,
                ring_scales,
                cache.device,
                self.capacity_rows as u64,
                self.kv_ring.rows() as u64,
                u64::from(self.head_width),
                u64::from(self.rope_head_dim),
                p0 as u64,
            ],
        )?;
        if let Some(count) = self.appended_rows.get_mut(layer) {
            *count = p0 + n;
        }
        Ok(())
    }

    /// The compressor seam and the indexer select, per row through the M = 1 stages with the
    /// layer's views on the row: normed (the compressor input), q_normed (the index query input),
    /// selected (the row's list, kept across layers) and the candidate mask.
    fn rows_seam_select(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let saved = self.buffers;
        let saved_ring = self.seam_ring;
        let dim = self.dim;
        let lora = rows.lora_cols;
        let prepared = self.rows_seam_prepare(rows, layer, p0, n)?;
        let heads = self.index_n_heads;
        let index_cols = heads * self.index_head_dim;
        let mut outcome = Ok(());
        for row in 0..n {
            self.buffers.residual.normed = span(b.normed + (row * dim * 4) as u64, dim * 4);
            self.buffers.workspace.q_normed = span(b.q_normed + (row * lora * 2) as u64, lora * 2);
            self.buffers.workspace.selected = span(
                b.selected + (row * SELECTED_STRIDE * 4) as u64,
                SELECTED_STRIDE * 4,
            );
            if let (Some(ring), Some(saved)) = (self.seam_ring.as_mut(), saved_ring) {
                ring.candidate_mask = Deviceptr(b.masks + (row * rows.mask_bytes) as u64);
                if prepared {
                    ring.weights_f32 = Deviceptr(b.idx_weights_f32 + (row * heads * 4) as u64);
                    ring.q_index_f32 = Deviceptr(b.idx_q_f32 + (row * index_cols * 4) as u64);
                }
                let _ = saved;
            }
            self.seam_rows_prepared = prepared;
            outcome = self
                .reference_seam(layer, p0 + row)
                .and_then(|()| self.ref_select(layer, p0 + row));
            if outcome.is_err() {
                break;
            }
        }
        self.seam_rows_prepared = false;
        self.buffers = saved;
        self.seam_ring = saved_ring;
        outcome
    }

    /// The row-local GEMVs of the compressor seam (wkv, wgate into the rows' ring slots) and of the
    /// indexer select (weights_proj and its scale; wq_b on the rows' q codes, rope, the FP4 round
    /// trip), each over all `n` rows at once with every row's chain the M = 1 kernel's. The pool,
    /// key and selection stages stay per row in reference_seam / ref_select, which skip these while
    /// [DeviceLayer::seam_rows_prepared] is set. False when the layer has none, or when a later
    /// row's ring slot would overwrite a group an earlier row still pools.
    fn rows_seam_prepare(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<bool, String> {
        use sparkflash_kernels::hostw_gemv_launch::Mode;
        let ratio = self.compress_ratio.get(layer).copied().unwrap_or(0);
        let (Some(staged), Some(ring)) = (self.seam_staged, self.seam_ring) else {
            return Ok(false);
        };
        let kv_source = self.is_kv_source(layer);
        let index = self.is_index_source(layer) && (p0 + n) / ratio.max(1) > 0;
        if ratio == 0
            || (!kv_source && !index)
            || n + ratio > ring.rows + 1
            || std::env::var("SPARKFLASH_ROWS_INDEXER").as_deref() == Ok("0")
        {
            return Ok(false);
        }
        let b = rows.b;
        let dim = self.dim;
        let cuda = self.cuda;
        let hostw = |mode: Mode, a: u64, w: Deviceptr, out: u64, cols: usize, count: usize| {
            let kernel = match mode {
                Mode::Bf16Weight => rows.kernels.hostw[0],
                Mode::Bf16WeightF32Stores => rows.kernels.hostw[1],
                Mode::F32WeightF32Stores => rows.kernels.hostw[2],
            };
            cuda.launch(
                kernel,
                ((cols / 32) as u32, count as u32, 1),
                (32, 1, 1),
                &[
                    a,
                    w.0,
                    out,
                    cols as u64,
                    dim as u64,
                    dim as u64,
                    cols as u64,
                ],
            )
        };
        if kv_source {
            let owner = self
                .owner_index_for(layer)
                .ok_or("rows seam: the layer has no kv-source owner")?;
            let (wkv, wkv_mode) = staged.wkv.ok_or("rows seam: no staged compressor wkv")?;
            let row_bytes = ring.head_dim * 4;
            let owner_base = owner * ring.rows * row_bytes;
            let mut row = 0;
            while row < n {
                let slot = (p0 + row) % ring.rows;
                let run = (ring.rows - slot).min(n - row);
                let a = b.normed + (row * dim * 4) as u64;
                let at = (owner_base + slot * row_bytes) as u64;
                hostw(wkv_mode, a, wkv, ring.wkv.0 + at, ring.head_dim, run)?;
                if let Some((wgate, wgate_mode)) = staged.wgate {
                    hostw(wgate_mode, a, wgate, ring.wgate.0 + at, ring.head_dim, run)?;
                }
                row += run;
            }
        }
        if index {
            let refusal = || format!("rows seam: layer {layer} has no staged indexer");
            let weights_proj = staged.weights_proj.ok_or_else(refusal)?;
            let (wq_b, wq_b_scales) = staged.indexer_wq_b.ok_or_else(refusal)?;
            let scale_weights = self.seam_scale_weights.ok_or_else(refusal)?;
            let fp4_round_trip = self.seam_fp4_round_trip.ok_or_else(refusal)?;
            let (heads, head_dim) = (self.index_n_heads, self.index_head_dim);
            let index_cols = heads * head_dim;
            hostw(
                Mode::Bf16Weight,
                b.normed,
                weights_proj,
                b.idx_weights_bf16,
                heads,
                n,
            )?;
            let factor = ((head_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
            sparkflash_kernels::pool_group_launch::launch_scale_index_weights(
                cuda,
                scale_weights,
                Deviceptr(b.idx_weights_bf16),
                Deviceptr(b.idx_weights_f32),
                factor,
                (n * heads) as u32,
            )?;
            // ref_select quantises q_normed exactly as rows_query_kv already did into q_codes.
            let lora = rows.lora_cols;
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
            .map_err(|error| format!("rows indexer wq_b: {error}"))?;
            let pairs = self.rope_pairs as usize;
            let table_bytes = self.max_positions * pairs * 4;
            let tables = self.buffers.workspace.rope_tables_compressed.device;
            for row in 0..n {
                let rope_offset = ((p0 + row) * pairs * 4) as u64;
                crate::layer_attention_stages::launch_forward_rope(
                    cuda,
                    self.kernels.rope_rows,
                    Deviceptr(b.idx_query + (row * index_cols * 2) as u64),
                    Deviceptr(tables + rope_offset),
                    Deviceptr(tables + table_bytes as u64 + rope_offset),
                    heads as u32,
                    head_dim as u32,
                    self.rope_head_dim,
                    self.seam_trace_ring(),
                    layer as u32,
                    (p0 + row) as u64,
                )?;
            }
            sparkflash_kernels::pool_group_launch::launch_fp4_round_trip(
                cuda,
                fp4_round_trip,
                Deviceptr(b.idx_query),
                Deviceptr(b.idx_q_f32),
                (n * index_cols / 32) as u32,
            )?;
        }
        Ok(true)
    }

    /// The attention body over the rows (window and selected count from each row's position),
    /// then the inverse rope.
    fn rows_attention(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let ratio = self
            .compress_ratio
            .get(layer)
            .copied()
            .ok_or("rows: no compress ratio")?;
        let ring_layer = self.kv_ring_base + self.kv_ring.layer_offset(layer) as u64;
        let ring_codes = ring_layer + self.kv_ring.codes_offset(0)? as u64;
        let ring_scales = ring_layer + self.kv_ring.scales_offset(0)? as u64;
        let sink_stride = self.out_geometry.heads_per_rank * 4;
        if (layer + 1) * sink_stride > self.sink_pool_bytes {
            return Err(format!(
                "rows attention: layer {layer}'s sink row is outside the pool"
            ));
        }
        let sink = self.sink_pool.0 + (layer * sink_stride) as u64;
        let (kv_fp4, kv_scales) = if ratio > 0 {
            let owner = self
                .owner_index_for(layer)
                .ok_or_else(|| format!("rows attention: layer {layer} has no kv-source owner"))?;
            let ring = self.seam_ring.ok_or("rows attention: no seam ring")?;
            let (code_bytes, scale_bytes) = seam_store_owner_bytes(self.max_positions)?;
            (
                ring.store_codes.0 + (owner * code_bytes) as u64,
                ring.store_scales.0 + (owner * scale_bytes) as u64,
            )
        } else {
            (
                self.buffers.workspace.store_codes.device,
                self.buffers.workspace.store_scales.device,
            )
        };
        self.cuda.launch(
            rows.kernels.attn,
            (self.heads_per_rank, n as u32, 1),
            (256, 1, 1),
            &[
                b.q,
                kv_fp4,
                kv_scales,
                ring_codes,
                ring_scales,
                b.selected,
                sink,
                b.attn_body,
                p0 as u64,
                ratio as u64,
                self.index_topk as u64,
                SELECTED_STRIDE as u64,
                self.kv_ring.rows() as u64,
            ],
        )?;
        let (cos, sin) = self.rows_rope_tables(layer)?;
        self.cuda.launch(
            rows.kernels.inv_rope,
            (self.heads_per_rank, n as u32, 1),
            (32, 1, 1),
            &[
                b.attn_body,
                cos,
                sin,
                u64::from(self.head_width),
                u64::from(self.rope_head_dim),
                p0 as u64,
            ],
        )
    }

    /// wo_a over both groups, the bf16 quantiser, wo_b.
    fn rows_output(&mut self, rows: &RowsState, layer: usize, n: usize) -> Result<(), String> {
        use sparkflash_kernels::layer_seams_launch as seams;
        let b = rows.b;
        let bind = self.attention.ok_or("rows output: no attention bind")?;
        let lookup = |name: String| -> Result<(Deviceptr, Deviceptr), String> {
            bind.pointers
                .get(&name)
                .copied()
                .ok_or_else(|| format!("rows output: the bind has no {name}"))
        };
        let wo_a = lookup(format!("layers.{layer}.attn.wo_a.weight"))?;
        let wo_b = lookup(format!("layers.{layer}.attn.wo_b.weight"))?;
        let geometry = self.out_geometry;
        let groups = geometry.groups as u64;
        let (cols, inner) = (geometry.o_lora_rank as u64, geometry.group_inner as u64);
        let tokens = n as u32;
        launch_opt(
            self.cuda,
            OptKind::Bf16aBf16o,
            &OptArgs {
                a: b.attn_body,
                a_scales: 0,
                w: wo_a.0.0,
                ws: wo_a.1.0,
                out: b.attn_lo,
                cols: cols as u32,
                inner: inner as u32,
                tokens,
                a_tok_stride: inner * groups,
                out_tok_stride: cols * groups,
                groups: groups as u32,
                a_grp_stride: inner,
                w_grp_stride: cols * inner,
                s_grp_stride: (cols / 32) * (inner / 32),
                out_grp_stride: cols,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("rows wo_a: {error}"))?;
        let lo = rows.lo_cols;
        seams::launch_prologue_bf16(
            self.cuda,
            self.kernels.prologue_bf16,
            &seams::PrologueBf16Params {
                normed: Deviceptr(b.attn_lo),
                codes: Deviceptr(b.o_codes),
                scales: Deviceptr(b.o_scales),
                cols: lo as u32,
                rows: tokens,
                trace_ring: 0,
                layer_id: layer as u32,
                step_seq: 0,
            },
        )?;
        launch_opt(
            self.cuda,
            OptKind::Fp8aBf16o,
            &OptArgs {
                a: b.o_codes,
                a_scales: b.o_scales,
                w: wo_b.0.0,
                ws: wo_b.1.0,
                out: b.attn_out,
                cols: geometry.hidden as u32,
                inner: lo as u32,
                tokens,
                a_tok_stride: lo as u64,
                out_tok_stride: geometry.hidden as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("rows wo_b: {error}"))
    }

    /// The router, the rank-split shared expert and the fused routed experts over the rows.
    fn rows_moe(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::layer_seams_launch as seams;
        use sparkflash_kernels::moe_routed_device::{DeviceSpan as MoeSpan, MoeRoutedOperands};
        let b = rows.b;
        let dim = self.dim;
        let tokens = n as u32;
        let (gate, bias) = self
            .router
            .as_ref()
            .and_then(|router| router.get(layer).copied())
            .ok_or_else(|| format!("rows router: layer {layer} has no router pointers"))?;
        let (_, router_kernel, ticket) = self.opt_router.ok_or("rows: no opt router")?;
        self.cuda.launch(
            router_kernel,
            (48, 1, 1),
            (256, 1, 1),
            &[
                b.ffn_x,
                gate.0,
                bias.0,
                b.moe_ids,
                b.moe_weights,
                b.scores,
                n as u64,
                ticket.0,
            ],
        )?;
        // The shared expert's rank slice: gate/up as [group][row][local], SwiGLU, down.
        let shared = self
            .shared
            .as_ref()
            .ok_or("rows: no shared expert operands")?;
        let ops = shared
            .layers
            .get(layer)
            .copied()
            .ok_or_else(|| format!("rows shared: layer {layer} has no gathered operands"))?;
        let local = rows.shared_local;
        let inter = shared.inter;
        let rank = self.tp_rank as u64;
        let dim_blocks = (dim / 32) as u64;
        let local_tiles = (local / 32) as u64;
        launch_opt(
            self.cuda,
            OptKind::F32aF32o,
            &OptArgs {
                a: b.seam_row,
                a_scales: 0,
                w: ops.w1.0 + rank * local_tiles * dim_blocks * 1024,
                ws: ops.s1.0 + rank * local_tiles * dim_blocks,
                out: b.shared_gate,
                cols: local as u32,
                inner: dim as u32,
                tokens,
                a_tok_stride: dim as u64,
                out_tok_stride: local as u64,
                groups: 2,
                a_grp_stride: 0,
                w_grp_stride: ops.w3.0.wrapping_sub(ops.w1.0),
                s_grp_stride: ops.s3.0.wrapping_sub(ops.s1.0),
                out_grp_stride: (n * local) as u64,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("rows shared gate/up: {error}"))?;
        seams::launch_shared_swiglu(
            self.cuda,
            self.kernels.shared_swiglu,
            &seams::SharedSwigluParams {
                gate: Deviceptr(b.shared_gate),
                up: Deviceptr(b.shared_gate + (n * local * 4) as u64),
                out: Deviceptr(b.shared_swiglu),
                limit: self.swiglu_limit,
                inter: (n * local) as u32,
            },
        )
        .map_err(|error| format!("rows shared swiglu: {error}"))?;
        let kb0 = rank * local_tiles;
        launch_opt(
            self.cuda,
            OptKind::F32aF32o,
            &OptArgs {
                a: b.shared_swiglu,
                a_scales: 0,
                w: ops.w2.0 + kb0 * 1024,
                ws: ops.s2.0 + kb0,
                out: b.shared_out,
                cols: dim as u32,
                inner: local as u32,
                tokens,
                a_tok_stride: local as u64,
                out_tok_stride: dim as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: (inter / 32) as u32,
            },
        )
        .map_err(|error| format!("rows shared down: {error}"))?;
        let device = self
            .moe_routed_device
            .as_ref()
            .ok_or("rows: no routed device")?;
        let top_k = self.routed_geometry.top_k;
        let operand = |device: u64, len: usize| MoeSpan::new(Deviceptr(device), len);
        let operands = MoeRoutedOperands {
            act_codes: operand(b.ffn_codes, n * dim),
            act_scales: operand(b.ffn_scales, n * dim / 32),
            ids: operand(b.moe_ids, n * top_k * 4),
            weights: operand(b.moe_weights, n * top_k * 4),
            delta_f32: operand(b.delta_f32, n * dim * 4),
            delta_bf16: operand(b.delta_bf16, n * dim * 2),
        };
        device.set_addend(b.shared_out);
        device.run(self.cuda, layer, p0, n, &operands)
    }

    /// (world, rank) when the run's reduce handle carries the one-shot exchange: the argmax and
    /// embedding gathers then stay on the device stream instead of a host round trip each.
    /// SPARKFLASH_DEVICE_GATHER=0 keeps them on the host transport.
    pub(super) fn device_exchange(&self) -> Option<(usize, usize)> {
        if std::env::var("SPARKFLASH_DEVICE_GATHER").as_deref() == Ok("0") {
            return None;
        }
        let handle = self.reduce?.0.try_borrow().ok()?;
        let (world, rank) = handle.world_rank();
        (handle.has_exchange() && world <= GATHER_WORLD).then_some((world, rank))
    }

    /// Enqueues the one-shot exchange of this rank's `count` bf16 values at `gather + rank * count`
    /// into every peer's slot of `gather` (rank-major).
    pub(super) fn exchange_slots(
        &self,
        gather: u64,
        count: usize,
        rank: usize,
        what: &str,
    ) -> Result<(), String> {
        let reduce = self
            .reduce
            .ok_or_else(|| format!("{what}: no reduce handle"))?;
        let mut handle = reduce
            .0
            .try_borrow_mut()
            .map_err(|_| format!("{what}: the reduce handle is already borrowed"))?;
        let mine = gather + (rank * count * 2) as u64;
        // SAFETY: the caller's gather buffer holds world * count bf16 values; this rank's slot is the
        // send buffer and every peer's lands in its own slot; stream 0 is the stream every launch of
        // this run uses.
        unsafe { handle.exchange_bf16(mine, gather, count, 0) }
            .map_err(|error| format!("{what}: exchange: {error}"))?;
        drop(handle);
        self.collectives_in_flight
            .set(self.collectives_in_flight.get().saturating_add(1));
        Ok(())
    }

    /// The cross-rank argmax of `n` rows' chunk maxima on the device: the per-row fold into this
    /// rank's slot of the argmax gather, the exchange, then the pick (the host gather's rule) into
    /// `ids_out` / `values_out` (0 for none). Returns `ids_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn argmax_on_device(
        &self,
        rows: &RowsState,
        block_max: u64,
        block_arg: u64,
        n: usize,
        (world, rank): (usize, usize),
        ids_out: u64,
        values_out: u64,
    ) -> Result<u64, String> {
        let gather = rows.b.argmax_gather;
        let slot = gather + (rank * n * 8) as u64;
        self.cuda.launch(
            rows.kernels.fold,
            (n as u32, 1, 1),
            (32, 1, 1),
            &[
                block_max,
                block_arg,
                crate::tp4_tail::CHUNKS_PER_RANK as u64,
                slot,
                slot + (n * 4) as u64,
            ],
        )?;
        self.exchange_slots(gather, n * 4, rank, "argmax gather")?;
        self.cuda.launch(
            rows.kernels.pick,
            (1, 1, 1),
            (32, 1, 1),
            &[
                gather,
                n as u64,
                world as u64,
                crate::tp4_tail::VOCAB_PER_RANK as u64,
                ids_out,
                values_out,
            ],
        )?;
        Ok(ids_out)
    }

    /// The head over the rows: hc collapse per row, the final norm, one pass of the vocabulary
    /// shard for every row, the per-row fold, then the cross-rank argmax. Returns (token, value)
    /// per row.
    pub(super) fn head_rows(
        &self,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        n: usize,
    ) -> Result<Vec<(u32, f32)>, String> {
        let b = rows.b;
        let dim = self.dim;
        let kernel = self
            .seam_hc_pre_collapse
            .ok_or("rows head: hc_pre_collapse_bf16 is not loaded")?;
        for row in 0..n {
            sparkflash_kernels::pool_group_launch::launch_hc_pre_collapse(
                self.cuda,
                kernel,
                Deviceptr(b.pre_mix + (row * self.hc_mult * 4) as u64),
                Deviceptr(b.stream + (row * self.hc_mult * dim * 4) as u64),
                Deviceptr(b.lm_hidden + (row * dim * 2) as u64),
                tail.buffers.certified_count,
                self.hc_mult as u32,
                dim as u32,
            )?;
        }
        let norm = sparkflash_kernels::rmsnorm_launch::RmsNormLaunchParams {
            x: Deviceptr(b.lm_hidden),
            weight: tail.norm_weight,
            out: Deviceptr(b.lm_normed),
            rows: n as u32,
            cols: dim as u32,
            eps: tail.eps,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        sparkflash_kernels::rmsnorm_launch::launch(self.cuda, self.row_norm.1, &norm)?;
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        self.cuda.launch(
            rows.kernels.lm_head,
            (chunks as u32, 1, 1),
            (512, 1, 1),
            &[
                b.lm_normed,
                tail.head_weight.0,
                b.lm_block_max,
                b.lm_block_arg,
                crate::tp4_tail::VOCAB_PER_RANK as u64,
                n as u64,
            ],
        )?;
        if let Some((world, rank)) = self.device_exchange() {
            let picks = self.argmax_on_device(
                rows,
                b.lm_block_max,
                b.lm_block_arg,
                n,
                (world, rank),
                b.lm_picks,
                b.lm_picks + (n * 4) as u64,
            )?;
            let mut words = vec![0u8; n * 8];
            self.cuda.copy_dtoh(&mut words, &Deviceptr(picks))?;
            let word = |at: usize| {
                u32::from_le_bytes([words[at], words[at + 1], words[at + 2], words[at + 3]])
            };
            return Ok((0..n)
                .map(|row| (word(row * 4), f32::from_bits(word((n + row) * 4))))
                .collect());
        }
        self.cuda.launch(
            rows.kernels.fold,
            (n as u32, 1, 1),
            (32, 1, 1),
            &[
                b.lm_block_max,
                b.lm_block_arg,
                chunks as u64,
                b.lm_value,
                b.lm_index,
            ],
        )?;
        let mut values = vec![0u8; n * 4];
        let mut indices = vec![0u8; n * 4];
        self.cuda.copy_dtoh(&mut values, &Deviceptr(b.lm_value))?;
        self.cuda.copy_dtoh(&mut indices, &Deviceptr(b.lm_index))?;
        let mut words = Vec::with_capacity(n * 4);
        for row in 0..n {
            let value =
                f32::from_le_bytes(values[row * 4..row * 4 + 4].try_into().unwrap_or([0; 4]));
            let index =
                u32::from_le_bytes(indices[row * 4..row * 4 + 4].try_into().unwrap_or([0; 4]));
            let bits = value.to_bits();
            words.extend_from_slice(&[
                (bits >> 16) as u16,
                bits as u16,
                (index >> 16) as u16,
                index as u16,
            ]);
        }
        let world = transport.world_size();
        let mut gathered = vec![0u16; world * n * 4];
        let call = rows.gathers.get();
        rows.gathers.set(call.wrapping_add(1));
        transport
            .all_gather_bf16(
                &words,
                &mut gathered,
                ROWS_GATHER_SEQ.wrapping_sub(call % 0x1_0000),
            )
            .map_err(|error| format!("rows argmax all-gather: {error}"))?;
        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            let mut winner = (f32::NEG_INFINITY, 0u32, false);
            for rank in 0..world {
                let w = &gathered[(rank * n + row) * 4..(rank * n + row) * 4 + 4];
                let value = f32::from_bits((u32::from(w[0]) << 16) | u32::from(w[1]));
                let index = ((u32::from(w[2]) << 16) | u32::from(w[3]))
                    + (rank * crate::tp4_tail::VOCAB_PER_RANK) as u32;
                if !winner.2 || value > winner.0 {
                    winner = (value, index, true);
                }
            }
            out.push((winner.1, winner.0));
        }
        Ok(out)
    }
}
