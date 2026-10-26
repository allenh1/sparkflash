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
use sparkflash_kernels::mxfp8_gemv_launch::{OptArgs, OptKind, launch_opt, launch_opt_split};

/// The widest verify step. The t8 GEMV takes the rows eight to a block, the router and the Engram
/// wkv eight to a weight pass, and the head all sixteen in one launch ([HEAD_ROWS]); every row's
/// arithmetic is the single-row kernel's.
pub(super) const R_MAX: usize = 16;

/// The rows one opt_rows_lm_head launch serves (its OR_LM_MAXR).
const HEAD_ROWS: usize = 16;

/// [launch_opt] with up to `gemv_rows` tokens kept on the t8 GEMV (each token's bits the
/// single-row kernel's); more are a bulk chunk, for the tensor-core GEMMs.
pub(super) fn launch_opt_rows(
    cuda: &Cuda,
    kind: OptKind,
    args: &OptArgs,
    gemv_rows: usize,
) -> Result<(), String> {
    if args.tokens as usize <= gemv_rows {
        launch_opt_split(cuda, kind, args)
    } else {
        launch_opt(cuda, kind, args)
    }
}

/// The mix table's coefficient rows (pre 4, post 4, comb 16).
const MIX_ROWS: usize = 24;
/// opt_rows_mhc_seam's per-row scratch: 80 blocks x 32 partial words.
const SEAM_PARTIAL_FLOATS: usize = 80 * 32;
/// The transport sequence range the rows head's argmax gather uses.
const ROWS_GATHER_SEQ: u64 = u64::MAX - 0xC_0000;
/// The most ranks the device gathers carve a slot for.
pub(super) const GATHER_WORLD: usize = 8;
/// The DSpark tap slots a row's tap buffer holds (`[rows][SP_TAPS][dim]` f32).
const SP_TAPS: usize = 3;

/// A split bulk chunk's rows on this rank ([DeviceLayer::sp_rows]): it posts and seams rows
/// `first .. first + count`, block `rank` of the chunk padded to `world * part` rows.
#[derive(Clone, Copy)]
pub(super) struct SpRows {
    pub(super) first: usize,
    pub(super) count: usize,
    pub(super) part: usize,
}

pub(super) struct RowsKernels {
    _modules: Vec<Module>,
    pub(super) seam: Kernel,
    /// opt_rows_mhc_partials_rb{1, 2, 4} and opt_rows_mhc_tail: opt_rows_mhc_seam in two launches,
    /// bit-identical to it, the partials sharing each fn_w read across 1, 2 or 4 rows and every
    /// row's tail on its own 1024-thread block (None under SPARKFLASH_SEAM_SPLIT=0).
    seam_split: Option<([Kernel; 3], Kernel)>,
    pub(super) post: Kernel,
    pub(super) kv: Kernel,
    attn: Kernel,
    /// opt_rows_attn_{scores, softmax, values}: opt_rows_attn in three kernels for 16 heads a
    /// rank, bit-identical to it (None under SPARKFLASH_ATTN_SPLIT=0).
    attn_split: Option<[Kernel; 3]>,
    pub(super) inv_rope: Kernel,
    lm_head: Kernel,
    pub(super) fold: Kernel,
    /// opt_rows_argmax_pick and opt_rows_store_u32: the device side of the argmax gathers.
    pub(super) pick: Kernel,
    pub(super) store: Kernel,
    /// opt_rows_topc: a row's slice statistics for the truncated sampler.
    topc: Kernel,
    /// opt_rows_embed_own and opt_rows_embed_stream: the rows' embedding streams from device ids;
    /// opt_rows_embed_table: the same from the whole table on this rank.
    pub(super) embed_own: Kernel,
    embed_stream: Kernel,
    embed_table: Kernel,
    engram_wkv: Kernel,
    pub(super) rank_permute: Kernel,
    /// opt_rows_hostw_{bf16w, bf16w_f32o, f32w_f32o}, by hostw_gemv_launch::Mode.
    hostw: [Kernel; 3],
    prefetch_l2: Kernel,
    /// opt_rows_prologue_bf16 and opt_rows_shared_swiglu: layer_seams' activation quantiser and
    /// shared-expert SwiGLU with the PDL entry ([RowsState::quantise_bf16], [RowsState::swiglu]).
    prologue: Kernel,
    swiglu: Kernel,
    /// opt_rows_hc_pre_collapse: hc_pre_collapse_bf16 over a grid of rows in one launch.
    pub(super) hc_collapse: Kernel,
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
    /// The split attention's scores `[R_MAX][16][640]` f32, probabilities `[R_MAX][640][16]` f32
    /// and sums `[R_MAX][16]` f32.
    attn_scores: u64,
    attn_probs: u64,
    attn_ssum: u64,
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
    /// `[R_MAX]` u32: a prompt chunk's token ids, for the device embedding gather.
    pub(super) row_ids: u64,
    /// `[R_MAX][2]` u64: a step over several lanes' rows, row r's lane offset and position
    /// ([DeviceLayer::upload_row_table]).
    row_tab: u64,
    /// The head's raw logits `[R_MAX][vocab slice]` f32, for the truncated sampler.
    logits: u64,
    /// `[GATHER_WORLD][R_MAX][TOPC_WORDS]` u32: each rank's opt_rows_topc slots, exchanged in place.
    topc_gather: u64,
    /// `[GATHER_WORLD][R_MAX][dim]` bf16: each rank's owned embedding rows, exchanged in place.
    pub(super) embed_gather: u64,
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
    pub(super) mask_bytes: usize,
    engram_inner: usize,
    engram_out: usize,
    gathers: Cell<u64>,
    /// A bulk chunk's FFN post of this layer, left for the next layer's attention seam to fuse
    /// (opt_bulk_mhc_post_partials): the stream rows of the chunk are stale until it runs.
    post_pending: Cell<Option<usize>>,
    /// The segments whose row table `b.row_tab` holds (empty: none).
    row_tab_for: std::cell::RefCell<Vec<super::lanes::Seg>>,
    /// The DSpark tap capture (spec mode): opt_draft_taps, the `[R_MAX][3][dim]` f32 tap buffer,
    /// and the target layers in tap order. Before each target layer the rows' streams are
    /// hc-averaged into their tap slot.
    pub(super) taps: Option<(Kernel, u64, Vec<usize>)>,
    /// The side stream the all-reduce windows' L2 prefetches run on (SPARKFLASH_L2_PREFETCH=0: none).
    side: Option<sparkflash_kernels::cuda_driver::SideStream>,
    /// The most rows a step takes: R_MAX, or a bulk prefill chunk's capacity.
    pub(super) cap: usize,
    /// The bulk prefill's kernels and chunk scratch (a state opened above R_MAX rows).
    pub(super) bulk: Option<super::bulk::BulkState>,
    /// Paired bulk chunks (DeviceLayer::prefill_layers_paired): this chunk's all-reduces run on
    /// the device's collective stream and mark `ar_done`, and the run's stream waits on it only at
    /// the result's first reader (`ar_pending`), so the other chunk's launches queued between run
    /// under the collective. `post_after_ar`: the FFN post the last all-reduce feeds when no seam
    /// fuses it, left for after that wait.
    pub(super) overlap: Cell<bool>,
    ar_done: Option<sparkflash_kernels::cuda_driver::StreamEvent>,
    ar_pending: Cell<bool>,
    post_after_ar: Cell<bool>,
    /// A bulk state's place in the routed-expert rows the interleaved states share (None: its own).
    pub(super) moe_way: Option<super::bulk::MoeWay>,
    /// The head's per-row noise when one step's rows sample under several requests' samplers:
    /// `[R_MAX][4]` u32 for opt_rows_lm_head, then `[R_MAX][4]` for opt_rows_topc.
    pub(super) head_params: sparkflash_kernels::cuda_driver::HostMapped,
}

impl RowsState {
    /// opt_rows_hc_pre_collapse of `groups` x `k` rows: output row g k + r from row g stride + r of
    /// pre_mix and stream, into `out` (bf16), zeroing *certified_count.
    pub(super) fn hc_collapse(
        &self,
        cuda: &Cuda,
        pre_mix: u64,
        stream: u64,
        out: u64,
        certified_count: u64,
        (groups, k, stride): (usize, usize, usize),
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.hc_collapse,
            (5120 / 256, k as u32, groups as u32),
            (256, 1, 1),
            &[pre_mix, stream, out, certified_count, stride as u64],
        )
    }

    /// Loads opt_rows_sm121.cubin and carves the row buffers for `device`'s geometry.
    ///
    /// # Errors
    /// Refuses a layer whose fast path is not the opt one, or a geometry the row kernels do not
    /// hold, and returns any load or allocation error.
    pub(super) fn open(device: &DeviceLayer<'_>, kernel_dir: &Path) -> Result<Self, String> {
        Self::open_with(device, kernel_dir, R_MAX)
    }

    /// [Self::open] with the row buffers carved for `cap` rows; above R_MAX the state is a bulk
    /// prefill state (super::bulk), whose head buffers stay R_MAX rows (the head reads one row).
    pub(super) fn open_with(
        device: &DeviceLayer<'_>,
        kernel_dir: &Path,
        cap: usize,
    ) -> Result<Self, String> {
        Self::open_shared(device, kernel_dir, cap, None)
    }

    /// [Self::open_with] for a bulk state that is way `way` of `share`: its routed-expert rows and
    /// scratch are the share's.
    pub(super) fn open_shared(
        device: &DeviceLayer<'_>,
        kernel_dir: &Path,
        cap: usize,
        share: Option<(&super::bulk::MoeShare, usize)>,
    ) -> Result<Self, String> {
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
        let seam_split = if std::env::var("SPARKFLASH_SEAM_SPLIT").as_deref() == Ok("0") {
            None
        } else {
            Some((
                [
                    load("opt_rows_mhc_partials_rb1")?,
                    load("opt_rows_mhc_partials_rb2")?,
                    load("opt_rows_mhc_partials_rb4")?,
                ],
                load("opt_rows_mhc_tail")?,
            ))
        };
        let post = load("opt_rows_mhc_post")?;
        let kv = load("opt_rows_kv")?;
        let attn = load("opt_rows_attn")?;
        let attn_split = if std::env::var("SPARKFLASH_ATTN_SPLIT").as_deref() == Ok("0") {
            None
        } else {
            Some([
                load("opt_rows_attn_scores")?,
                load("opt_rows_attn_softmax")?,
                load("opt_rows_attn_values")?,
            ])
        };
        let inv_rope = load("opt_rows_inv_rope")?;
        let lm_head = load("opt_rows_lm_head")?;
        let fold = load("opt_rows_argmax_fold")?;
        let pick = load("opt_rows_argmax_pick")?;
        let store = load("opt_rows_store_u32")?;
        let topc = load("opt_rows_topc")?;
        let embed_own = load("opt_rows_embed_own")?;
        let embed_stream = load("opt_rows_embed_stream")?;
        let embed_table = load("opt_rows_embed_table")?;
        let engram_wkv = load("opt_rows_engram_wkv")?;
        let rank_permute = load("opt_rows_rank_permute")?;
        let hostw = [
            load("opt_rows_hostw_bf16w")?,
            load("opt_rows_hostw_bf16w_f32o")?,
            load("opt_rows_hostw_f32w_f32o")?,
        ];
        let prefetch_l2 = load("opt_rows_prefetch_l2")?;
        let prologue = load("opt_rows_prologue_bf16")?;
        let swiglu = load("opt_rows_shared_swiglu")?;
        let hc_collapse = load("opt_rows_hc_pre_collapse")?;
        let kernels = RowsKernels {
            _modules: modules,
            seam,
            seam_split,
            post,
            kv,
            attn,
            attn_split,
            inv_rope,
            lm_head,
            fold,
            pick,
            store,
            topc,
            embed_own,
            embed_stream,
            embed_table,
            engram_wkv,
            rank_permute,
            hostw,
            prefetch_l2,
            prologue,
            swiglu,
            hc_collapse,
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
        let mask_bytes = candidate_mask_words(device.max_positions) * 4;
        let (engram_inner, engram_out) = device
            .engram
            .as_ref()
            .map_or((32, 8), |engram| (engram.inner, engram.out_width));
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        let index_heads = device.index_n_heads;
        let index_cols = index_heads * device.index_head_dim;
        let r = cap.max(R_MAX);
        // The drafter runs several sequences' blocks through the row buffers at once: they hold
        // that many rows even where a step holds R_MAX.
        let d = r.max(super::draft::DRAFT_ROWS);
        let moe_way = match share {
            Some((share, _)) if share.stride != d || r <= R_MAX => {
                return refuse("a bulk share's ways hold the rows of a bulk state");
            }
            Some((share, way)) => Some(share.way(way)?),
            None => None,
        };
        let shared = moe_way.map(|way| way.rows(dim, top_k));
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize| -> Result<u64, String> {
            let ptr = cuda.alloc(bytes.max(4), &mut total)?;
            cuda.copy_htod(&ptr, &vec![0u8; bytes.max(4)])?;
            allocations.push(ptr);
            Ok(ptr.0)
        };
        let b = RowsBuffers {
            stream: alloc(d * hc_width * 4)?,
            hc_rows: alloc(d * hc_width * 4)?,
            normed: alloc(d * dim * 4)?,
            pre_mix: alloc(d * device.hc_mult * 4)?,
            attn_mix: alloc(d * MIX_ROWS * 4)?,
            ffn_mix: alloc(d * MIX_ROWS * 4)?,
            act_codes: alloc(d * dim)?,
            act_scales: alloc(d * dim / 32)?,
            q_lora: alloc(d * lora_cols * 2)?,
            q_normed: alloc(d * lora_cols * 2)?,
            q_codes: alloc(d * lora_cols)?,
            q_scales: alloc(d * lora_cols / 32)?,
            q: alloc(d * q_cols * 2)?,
            kv_latent: alloc(d * kv_cols * 2)?,
            kv_rotated: alloc(d * kv_cols * 2)?,
            attn_body: alloc(d * q_cols * 2)?,
            attn_lo: alloc(d * lo_cols * 2)?,
            o_codes: alloc(d * lo_cols)?,
            o_scales: alloc(d * lo_cols / 32)?,
            attn_out: alloc(d * dim * 2)?,
            reduced: alloc(d * dim * 2)?,
            seam_row: alloc(d * dim * 4)?,
            ffn_codes: match shared {
                Some(rows) => rows.act_codes,
                None => alloc(d * dim)?,
            },
            ffn_scales: match shared {
                Some(rows) => rows.act_scales,
                None => alloc(d * dim / 32)?,
            },
            ffn_x: alloc(d * dim * 2)?,
            moe_ids: match shared {
                Some(rows) => rows.ids,
                None => alloc(d * top_k * 4)?,
            },
            moe_weights: match shared {
                Some(rows) => rows.weights,
                None => alloc(d * top_k * 4)?,
            },
            scores: alloc(d * experts * 4)?,
            delta_f32: alloc(d * dim * 4)?,
            delta_bf16: alloc(d * dim * 2)?,
            moe_reduced: alloc(d * dim * 2)?,
            shared_gate: alloc(2 * d * shared_local * 4)?,
            shared_swiglu: alloc(d * shared_local * 4)?,
            shared_out: alloc(d * dim * 4)?,
            selected: alloc(r * SELECTED_STRIDE * 4)?,
            attn_scores: alloc(R_MAX * ATTN_HEADS * ATTN_MAX_ATTENDED * 4)?,
            attn_probs: alloc(R_MAX * ATTN_MAX_ATTENDED * ATTN_HEADS * 4)?,
            attn_ssum: alloc(R_MAX * ATTN_HEADS * 4)?,
            masks: alloc(r * mask_bytes)?,
            seam_partials: alloc(d * SEAM_PARTIAL_FLOATS * 4)?,
            seam_row_buf: alloc(d * dim * 4)?,
            seam_tickets: alloc(d * 4)?,
            engram_codes: alloc(r * engram_inner)?,
            engram_scales: alloc(r * engram_inner / 32)?,
            engram_kv: alloc(r * engram_out * 2)?,
            engram_slices: alloc(r * engram_out * 2)?,
            idx_weights_bf16: alloc(r * index_heads * 2)?,
            idx_weights_f32: alloc(r * index_heads * 4)?,
            idx_query: alloc(r * index_cols * 2)?,
            idx_q_f32: alloc(r * index_cols * 4)?,
            lm_hidden: alloc(d * dim * 2)?,
            lm_normed: alloc(d * dim * 2)?,
            lm_block_max: alloc(R_MAX * chunks * 4)?,
            lm_block_arg: alloc(R_MAX * chunks * 4)?,
            lm_value: alloc(R_MAX * 4)?,
            lm_index: alloc(R_MAX * 4)?,
            argmax_gather: alloc(GATHER_WORLD * R_MAX * 8)?,
            lm_picks: alloc(R_MAX * 8)?,
            row_ids: alloc(r * 4)?,
            row_tab: alloc(R_MAX * 16)?,
            logits: alloc(R_MAX * crate::tp4_tail::VOCAB_PER_RANK * 4)?,
            topc_gather: alloc(GATHER_WORLD * R_MAX * super::sample::TOPC_WORDS * 4)?,
            embed_gather: alloc(GATHER_WORLD * r * dim * 2)?,
        };
        let head_params = cuda.alloc_host_mapped(R_MAX * 8 * 4, &mut 0)?;
        Ok(Self {
            kernels,
            b,
            allocations,
            head_params,
            q_cols,
            lora_cols,
            kv_cols,
            lo_cols,
            shared_local,
            mask_bytes,
            engram_inner,
            engram_out,
            gathers: Cell::new(0),
            post_pending: Cell::new(None),
            row_tab_for: std::cell::RefCell::new(Vec::new()),
            taps: None,
            // A bulk chunk's stages read far more than the L2 holds, so it prefetches nothing.
            side: match std::env::var("SPARKFLASH_L2_PREFETCH").as_deref() {
                Ok("0") => None,
                _ if r > R_MAX => None,
                _ => Some(cuda.side_stream()?),
            },
            cap: r,
            bulk: if r > R_MAX {
                Some(super::bulk::BulkState::open(
                    device,
                    kernel_dir,
                    r,
                    share.map(|(share, _)| share),
                )?)
            } else {
                None
            },
            overlap: Cell::new(false),
            ar_done: if r > R_MAX {
                Some(cuda.stream_event()?)
            } else {
                None
            },
            ar_pending: Cell::new(false),
            post_after_ar: Cell::new(false),
            moe_way,
        })
    }

    /// Ends a paired chunk's overlap; after a failure also drops the wait and the post its last
    /// collective left behind (the next chunk begins from fresh rows).
    pub(super) fn end_overlap(&self, failed: bool) {
        self.overlap.set(false);
        if failed {
            self.ar_pending.set(false);
            self.post_after_ar.set(false);
        }
    }

    /// The MX quantiser of `n` rows of `cols` bf16 values into E4M3 codes and ue8m0 scales
    /// (w8a8_prologue_bf16_e4m3's arithmetic, a warp per 32-value block).
    pub(super) fn quantise_bf16(
        &self,
        cuda: &Cuda,
        normed: u64,
        codes: u64,
        scales: u64,
        cols: usize,
        n: usize,
    ) -> Result<(), String> {
        let blocks = (n * cols / 32).div_ceil(8);
        cuda.launch(
            self.kernels.prologue,
            (blocks as u32, 1, 1),
            (256, 1, 1),
            &[normed, codes, scales, cols as u64, n as u64],
        )
    }

    /// The shared expert's SwiGLU over `inter` f32 elements (shared_swiglu_f32's arithmetic).
    pub(super) fn swiglu(
        &self,
        cuda: &Cuda,
        gate: u64,
        up: u64,
        out: u64,
        limit: f32,
        inter: usize,
    ) -> Result<(), String> {
        cuda.launch(
            self.kernels.swiglu,
            ((inter as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            &[gate, up, out, u64::from(limit.to_bits()), inter as u64],
        )
    }

    /// Frees the row buffers (the modules drop with the state).
    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
        cuda.free_host_mapped(&self.head_params);
        if let Some(bulk) = self.bulk {
            bulk.release(cuda);
        }
    }
}

/// The selected list's row stride in u32 ids (`workspace.selected` is 2,048 B).
const SELECTED_STRIDE: usize = 512;

/// The split attention's heads a rank and slots a row (OR_A3_HEADS, OR_A_MAX_ATTENDED), and the
/// slots a scores block takes (OR_A3_CHUNK).
const ATTN_HEADS: usize = 16;
const ATTN_MAX_ATTENDED: usize = 640;
const ATTN_SCORE_CHUNK: usize = 32;

/// opt_select_scores_rows' positions a tile (SR_TILE), and the blocks a batched select's scores
/// spread over (each strides over its row's tiles).
const SELECT_SCORE_TILE: usize = 16;
const SELECT_SCORE_BLOCKS: usize = 384;

/// opt_select_scores_seg's heads, positions a tile and rows (SG_HEADS, SG_TILE, SG_ROWS), and the
/// blocks its tiles spread over (SPARKFLASH_SEG_BLOCKS overrides).
const SELECT_SEG_HEADS: usize = 32;
const SELECT_SEG_TILE: usize = 32;
const SELECT_SEG_ROWS: usize = 8;
const SELECT_SEG_BLOCKS: usize = 96;

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
        self.check_begin(p0, n, rows.cap)?;
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
        self.check_begin(p0, n, rows.cap)?;
        self.embed_rows_on_device(rows, embed, ids, n, n, 0, exchange, true)?;
        self.positions_begun = p0 + n;
        self.attn_window = AttentionWindow::default();
        Ok(n)
    }

    /// [Self::begin_rows_device] for rows whose embeddings are already queued (the draft's
    /// `verify` rows, checked with [Self::check_begin] before they were): the bookkeeping.
    pub(super) fn begin_rows_queued(
        &mut self,
        rows: &RowsState,
        p0: usize,
        n: usize,
    ) -> Result<usize, String> {
        self.check_begin(p0, n, rows.cap)?;
        self.positions_begun = p0 + n;
        self.attn_window = AttentionWindow::default();
        Ok(n)
    }

    pub(super) fn check_begin(&self, p0: usize, n: usize, cap: usize) -> Result<(), String> {
        self.check_state()?;
        if n == 0 || n > cap {
            return Err(format!("rows: {n} rows are not 1..={cap}"));
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
        // Outside the rows its lane backs, a step would write (or its window read) unmapped
        // memory: a device fault on every rank that names no cause.
        if let Some(lanes) = self.lanes.as_ref() {
            let lane = lanes.selected;
            let (backed, floor) = (lanes.backed(lane), lanes.floor(lane));
            if p0 + n > backed {
                return Err(format!(
                    "rows: positions {p0}..{} pass the {backed} rows lane {lane} backs",
                    p0 + n
                ));
            }
            if floor > 0 && p0 < floor + sparkflash_kernels::flash_mla_csa2_seq_launch::SWA_WINDOW {
                return Err(format!(
                    "rows: position {p0}'s window reads below lane {lane}'s floor {floor}"
                ));
            }
        }
        Ok(())
    }

    /// Writes `total` rows' streams and identity pre mixes: rows below `n` are the embeddings of
    /// the token ids at `ids` (device u32), rows from `n` on are `noise` (device f32 `[dim]`). Each
    /// owner copies its rows into its slot of the embedding gather and the exchange fills the
    /// peers' slots, so the rows never pass through the host. With `spans` the rows are a prefill
    /// chunk's: the tower rank overwrites its image span rows in its slot before the exchange
    /// (super::vision).
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
        spans: bool,
    ) -> Result<(), String> {
        if self.dim != HIDDEN || self.hc_mult != 4 || n == 0 || n > total || total > rows.cap {
            return Err(format!(
                "rows: the device embedding gather takes 1..={} rows of {} x 4, not {n} of {total} rows of {} x {}",
                rows.cap, HIDDEN, self.dim, self.hc_mult
            ));
        }
        if let Some(table) = self.embed_table
            && !(spans && self.vision_rows())
        {
            return self.cuda.launch(
                rows.kernels.embed_table,
                (total as u32, 1, 1),
                (256, 1, 1),
                &[table, ids, n as u64, noise, rows.b.stream, rows.b.pre_mix],
            );
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
        if spans {
            self.vision_inject(gather + (rank * count * 2) as u64)?;
        }
        if n > R_MAX {
            self.gather_slots(gather, count, rank, "embedding gather")?;
        } else {
            self.exchange_slots(gather, count, rank, "embedding gather")?;
        }
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

    /// Gathers every rank's embedding shard into a whole table on this rank (one AllGather at boot,
    /// vocab x hidden bf16, 1.3 GB): a step's embedding rows are then read here instead of
    /// exchanged with their owners, so the steps' first collectives after a host wait go.
    /// SPARKFLASH_EMBED_REPLICA=0 keeps the shard alone.
    ///
    /// # Errors
    /// The allocation, the copy or the gather failed.
    pub(super) fn replicate_embedding(&mut self, shard: u64) -> Result<(), String> {
        if std::env::var("SPARKFLASH_EMBED_REPLICA").as_deref() == Ok("0") {
            return Ok(());
        }
        let Some((world, rank)) = self.device_exchange() else {
            return Ok(());
        };
        let count = crate::tp4_tail::VOCAB_PER_RANK * self.dim;
        let table = self.cuda.alloc(world * count * 2, &mut 0)?;
        self.cuda.copy_dtod(
            &Deviceptr(table.0 + (rank * count * 2) as u64),
            &Deviceptr(shard),
            count * 2,
        )?;
        self.gather_slots(table.0, count, rank, "embedding replica")?;
        self.wait_collectives()?;
        self.embed_table = Some(table.0);
        Ok(())
    }

    /// Moves the position counters back to `next` after a verify step accepted rows below it: every
    /// state a later row wrote is addressed by position and is rewritten before it is read.
    pub(super) fn rewind_rows(&mut self, next: usize) {
        self.positions_begun = next;
        for count in &mut self.appended_rows {
            *count = (*count).min(next);
        }
    }

    /// The latest position at or before `next` the rows can resume from over the state a longer
    /// sequence left: the compressor ring holds only the last SEAM_RING_ROWS positions' rows, so a
    /// later position may have overwritten a row of the group still open at `next`. A multiple of
    /// every compress ratio leaves no group open.
    pub(super) fn resume_boundary(&self, next: usize) -> usize {
        fn gcd(a: usize, b: usize) -> usize {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        let step = self
            .compress_ratio
            .iter()
            .filter(|&&ratio| ratio > 0)
            .fold(1, |step, &ratio| step / gcd(step, ratio) * ratio);
        next - next % step
    }

    /// One layer over rows p0..p0+n.
    pub(super) fn run_layer_rows(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        self.run_layer_attn(rows, layer, p0, n)?;
        self.run_layer_ffn(rows, layer, p0, n)
    }

    /// The layer's attention half: the previous layer's post (fused into the entry seam when it
    /// was deferred), the taps and Engram, the entry seam, q/kv, the select, the attention, the
    /// output projection, through the queued attention all-reduce.
    pub(super) fn run_layer_attn(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        self.run_layer_attn_seam(rows, layer, p0, n)?;
        self.run_layer_attn_rest(rows, layer, p0, n)
    }

    /// The attention half up to its seam: the previous layer's post, the taps and Engram and the
    /// entry seam (of the rank's own rows of a split chunk, then their AllGather).
    pub(super) fn run_layer_attn_seam(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        self.rows_ar_wait(rows, n)?;
        let b = rows.b;
        let dim = self.dim;
        let bulk = rows.bulk.is_some() && n > R_MAX;
        let sp = self.sp_rows(rows, n);
        let (first, count) = sp.map_or((0, n), |sp| (sp.first, sp.count));
        let own = |base: u64, row_bytes: usize| base + (first * row_bytes) as u64;
        let (stream_bytes, pre_bytes, mix_bytes) =
            (self.hc_mult * dim * 4, self.hc_mult * 4, MIX_ROWS * 4);
        // The previous layer's FFN post, when it was left to this seam (a step that failed midway
        // can leave one behind; the next step's layer 0 begins from fresh rows and drops it).
        let pending = match rows.post_pending.take() {
            Some(previous) if layer > 0 && previous + 1 != layer => {
                return Err(format!(
                    "rows layer {layer}: layer {previous}'s deferred post was never run"
                ));
            }
            Some(_) => layer > 0,
            None => false,
        };
        if let Some((kernel, taps, layers)) = rows.taps.as_ref()
            && let Some(tap) = layers.iter().position(|&target| target == layer)
        {
            self.cuda.launch(
                *kernel,
                (20, count as u32, 1),
                (256, 1, 1),
                &[
                    own(b.stream, stream_bytes),
                    own(*taps, SP_TAPS * dim * 4),
                    tap as u64,
                ],
            )?;
        }
        self.rows_engram(rows, layer, p0, n, (first, count))?;
        let weights = self.seam_layer(layer, "rows entry")?;
        let attn_w = [
            weights.attn_fn.0,
            weights.attn_base.0,
            weights.attn_scale.0,
            weights.attn_norm.0,
        ];
        // A split chunk's seam writes its rows' bf16 normed rows (into ffn_x, free until the FFN
        // seam) for the AllGather, and the derive the codes of every row after it; the f32 rows'
        // readers take the gathered bf16 rows instead.
        let attn_out = match sp {
            Some(_) => [
                own(b.attn_mix, mix_bytes),
                0,
                0,
                0,
                0,
                own(b.ffn_x, dim * 2),
            ],
            None => [b.attn_mix, 0, b.normed, b.act_codes, b.act_scales, 0],
        };
        if pending {
            self.rows_post_seam(
                rows,
                count,
                [
                    own(b.stream, stream_bytes),
                    own(b.ffn_mix, mix_bytes),
                    own(b.moe_reduced, dim * 2),
                ],
                own(b.stream, stream_bytes),
                own(b.pre_mix, pre_bytes),
                self.hc_mult as u32,
                attn_w,
                attn_out[0],
                attn_out[1],
                attn_out[2],
                attn_out[3],
                attn_out[4],
                attn_out[5],
                1,
            )?;
        } else {
            // A bulk chunk's attention post reads the stream in place (nothing writes it before
            // the post), so its seam keeps no copy of it.
            self.rows_seam(
                rows,
                count,
                own(b.stream, stream_bytes),
                own(b.pre_mix, pre_bytes),
                self.hc_mult as u32,
                attn_w,
                attn_out[0],
                attn_out[1],
                if bulk { 0 } else { b.hc_rows },
                attn_out[2],
                attn_out[3],
                attn_out[4],
                attn_out[5],
                1,
            )?;
        }
        if let Some(sp) = sp {
            self.sp_collective(
                rows,
                "all_gather",
                true,
                own(b.ffn_x, dim * 2),
                b.ffn_x,
                sp.part * dim,
            )?;
        }
        Ok(())
    }

    /// The attention half past its seam: (a split chunk's codes of every row from the gathered
    /// rows,) q/kv, the select, the attention and the output projection, through the queued
    /// attention all-reduce (a split chunk's ReduceScatter into its own rows).
    pub(super) fn run_layer_attn_rest(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        let sp = self.sp_rows(rows, n);
        if sp.is_some() {
            self.rows_ar_wait(rows, n)?;
            self.sp_derive(rows, b.ffn_x, 0, b.act_codes, b.act_scales, n)?;
        }
        let weights = self.seam_layer(layer, "rows entry")?;
        self.rows_query_kv(rows, layer, p0, n)?;
        match rows.bulk.as_ref().filter(|_| n > R_MAX) {
            Some(bulk) => self.bulk_seam_select(rows, bulk, layer, p0, n)?,
            None => self.rows_seam_select(rows, layer, p0, n)?,
        }
        self.rows_attention(rows, layer, p0, n)?;
        self.rows_output(rows, layer, n)?;
        self.prefetch_ffn(rows, layer, &weights)?;
        match sp {
            Some(sp) => self.sp_collective(
                rows,
                "reduce_scatter",
                false,
                b.attn_out,
                b.reduced + (sp.first * dim * 2) as u64,
                sp.part * dim,
            ),
            None => self.reduce_layer_rows(
                rows,
                "all_reduce",
                span(b.attn_out, n * dim * 2),
                span(b.reduced, n * dim * 2),
                n,
            ),
        }
    }

    /// The layer's FFN half: the post of the attention and the FFN seam, the MoE, through the
    /// queued MoE all-reduce (its post deferred to the next layer's seam where that can fuse it).
    pub(super) fn run_layer_ffn(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        self.run_layer_ffn_seam(rows, layer, n)?;
        self.run_layer_ffn_rest(rows, layer, p0, n)
    }

    /// The FFN half up to its seam: the attention's post and the FFN seam (of the rank's own rows
    /// of a split chunk, then their AllGather).
    pub(super) fn run_layer_ffn_seam(
        &mut self,
        rows: &RowsState,
        layer: usize,
        n: usize,
    ) -> Result<(), String> {
        self.rows_ar_wait(rows, n)?;
        let b = rows.b;
        let dim = self.dim;
        let bulk = rows.bulk.is_some() && n > R_MAX;
        let sp = self.sp_rows(rows, n);
        let (first, count) = sp.map_or((0, n), |sp| (sp.first, sp.count));
        let own = |base: u64, row_bytes: usize| base + (first * row_bytes) as u64;
        let (stream_bytes, pre_bytes, mix_bytes) =
            (self.hc_mult * dim * 4, self.hc_mult * 4, MIX_ROWS * 4);
        let weights = self.seam_layer(layer, "rows ffn")?;
        let (seam_row, codes, scales) = match sp {
            Some(_) => (0, 0, 0),
            None => (b.seam_row, b.ffn_codes, b.ffn_scales),
        };
        self.rows_post_seam(
            rows,
            count,
            [
                if bulk {
                    own(b.stream, stream_bytes)
                } else {
                    b.hc_rows
                },
                own(b.attn_mix, mix_bytes),
                own(b.reduced, dim * 2),
            ],
            own(b.stream, stream_bytes),
            own(b.attn_mix, mix_bytes),
            MIX_ROWS as u32,
            [
                weights.ffn_fn.0,
                weights.ffn_base.0,
                weights.ffn_scale.0,
                weights.ffn_norm.0,
            ],
            own(b.ffn_mix, mix_bytes),
            own(b.pre_mix, pre_bytes),
            if seam_row == 0 {
                0
            } else {
                own(seam_row, dim * 4)
            },
            codes,
            scales,
            own(b.ffn_x, dim * 2),
            0,
        )?;
        if let Some(sp) = sp {
            self.sp_collective(
                rows,
                "all_gather",
                true,
                own(b.ffn_x, dim * 2),
                b.ffn_x,
                sp.part * dim,
            )?;
        }
        Ok(())
    }

    /// The FFN half past its seam: (a split chunk's codes of every row from the gathered rows,) the MoE, through the queued MoE all-reduce (its post deferred to the next
    /// layer's seam where that can fuse it). A split chunk's last layer posts its own rows and
    /// gathers what the head and the drafter read ([Self::sp_finish]).
    pub(super) fn run_layer_ffn_rest(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        self.ffn_rest_inputs(rows, layer, n)?;
        self.rows_moe_routed(rows, layer, p0, n)?;
        self.ffn_rest_tail(rows, layer, n)
    }

    /// [Self::run_layer_ffn_rest] up to the routed experts: (a split chunk's codes of every row,)
    /// the router and the shared expert.
    pub(super) fn ffn_rest_inputs(
        &mut self,
        rows: &RowsState,
        layer: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        if self.sp_rows(rows, n).is_some() {
            self.rows_ar_wait(rows, n)?;
            self.sp_derive(rows, b.ffn_x, 0, b.ffn_codes, b.ffn_scales, n)?;
        }
        self.rows_moe_inputs(rows, layer, n)
    }

    /// The routed experts of several bulk chunks' rows (`(state, rows)` of ways of one
    /// [super::bulk::MoeShare]) as one pass over the expert weights, then each chunk's combine
    /// over its shared expert. Every row's values are the ones its chunk's own pass gives.
    pub(super) fn rows_moe_merged(
        &mut self,
        chunks: &[(&RowsState, usize)],
        layer: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::moe_routed_device::MAX_BULK_SEGMENTS;
        let device = self
            .moe_routed_device
            .as_ref()
            .ok_or("rows: no routed device")?;
        let (first, _) = chunks.first().ok_or("rows merged moe: no chunks")?;
        let share = first
            .moe_way
            .ok_or("rows merged moe: not a shared bulk state")?;
        let bulk = first
            .bulk
            .as_ref()
            .ok_or("rows merged moe: not a bulk state")?;
        let mut counts = [0usize; MAX_BULK_SEGMENTS];
        let mut used = 0;
        for &(state, n) in chunks {
            let way = state
                .moe_way
                .filter(|way| way.base.ids == share.base.ids && way.way < MAX_BULK_SEGMENTS)
                .ok_or("rows merged moe: a chunk outside the share")?;
            if n <= R_MAX || counts[way.way] != 0 {
                return Err(format!(
                    "rows merged moe: way {} twice or {n} rows",
                    way.way
                ));
            }
            counts[way.way] = n;
            used = used.max(way.way + 1);
        }
        device.run_bulk_rows(
            self.cuda,
            &bulk.moe,
            layer,
            share.stride,
            &counts[..used],
            &share.base,
        )?;
        for &(state, n) in chunks {
            let b = state.b;
            let way = state.moe_way.map_or(0, |way| way.way);
            device.set_addend(b.shared_out);
            device.combine_bulk(
                self.cuda,
                &bulk.moe,
                way * share.stride,
                n,
                b.moe_ids,
                b.delta_f32,
                b.delta_bf16,
            )?;
        }
        Ok(())
    }

    /// [Self::run_layer_ffn_rest] past the routed experts: the queued MoE reduce and the post.
    pub(super) fn ffn_rest_tail(
        &mut self,
        rows: &RowsState,
        layer: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        let bulk = rows.bulk.is_some() && n > R_MAX;
        let sp = self.sp_rows(rows, n);
        self.prefetch_attention(rows, layer + 1)?;
        match sp {
            Some(sp) => self.sp_collective(
                rows,
                "reduce_scatter_moe",
                false,
                b.delta_bf16,
                b.moe_reduced + (sp.first * dim * 2) as u64,
                sp.part * dim,
            )?,
            None => self.reduce_layer_rows(
                rows,
                "all_reduce_moe",
                span(b.delta_bf16, n * dim * 2),
                span(b.moe_reduced, n * dim * 2),
                n,
            )?,
        }
        // A bulk chunk leaves the post to the next layer's seam, unless that layer reads the stream
        // before its seam (a DSpark tap, an Engram write) or this is the last layer.
        let next = layer + 1;
        let last = self.seam_layer(next, "rows post").is_err();
        let defer = bulk
            && !last
            && !self.engram_rows.contains(&next)
            && !rows
                .taps
                .as_ref()
                .is_some_and(|(_, _, layers)| layers.contains(&next));
        if defer {
            rows.post_pending.set(Some(layer));
            return Ok(());
        }
        if let Some(sp) = sp.filter(|_| last) {
            self.rows_ar_wait(rows, n)?;
            self.sp_post(rows, sp)?;
            return self.sp_finish(rows, sp, n);
        }
        if rows.ar_pending.get() {
            rows.post_after_ar.set(true);
            return Ok(());
        }
        match sp {
            Some(sp) => self.sp_post(rows, sp),
            None => self.rows_post(rows, n, b.stream, b.ffn_mix, b.moe_reduced, b.stream),
        }
    }

    /// The rows of a bulk chunk of `n` rows this rank posts and seams when the device splits them
    /// ([DeviceLayer::enable_prefill_sp]); None when it does not, or when a rank's share would
    /// be a rows step's (the same answer on every rank: it depends on `n` alone).
    pub(super) fn sp_rows(&self, rows: &RowsState, n: usize) -> Option<SpRows> {
        let world = self.prefill_sp?;
        if rows.bulk.is_none() || rows.ar_done.is_none() || n <= R_MAX {
            return None;
        }
        let part = n.div_ceil(world);
        if part * world > rows.cap || n <= (world - 1) * part + R_MAX {
            return None;
        }
        let first = self.tp_rank * part;
        Some(SpRows {
            first,
            count: part.min(n - first),
            part,
        })
    }

    /// A split chunk's collective on the collective stream ([DeviceLayer::bulk_collective]):
    /// paired, the chunk's next reader waits for it (`ar_pending`); alone, the run's stream at once.
    fn sp_collective(
        &self,
        rows: &RowsState,
        stage: &'static str,
        gather: bool,
        input: u64,
        output: u64,
        count: usize,
    ) -> Result<(), String> {
        let done = rows
            .ar_done
            .as_ref()
            .ok_or_else(|| format!("rows {stage}: the chunk has no collective event"))?;
        self.bulk_collective(stage, gather, input, output, count, done)?;
        if rows.overlap.get() {
            rows.ar_pending.set(true);
            Ok(())
        } else {
            self.cuda.run_wait(done)
        }
    }

    /// opt_bulk_seam_derive over the chunk's `n` gathered bf16 rows `x`: their FP8 codes and
    /// scales into `codes` and `scales` (and their f32 rows into a non-null `normed`).
    fn sp_derive(
        &self,
        rows: &RowsState,
        x: u64,
        normed: u64,
        codes: u64,
        scales: u64,
        n: usize,
    ) -> Result<(), String> {
        let bulk = rows.bulk.as_ref().ok_or("rows derive: not a bulk chunk")?;
        self.cuda.launch(
            bulk.k.seam_derive,
            ((self.dim / 1024) as u32, n as u32, 1),
            (128, 1, 1),
            &[x, normed, codes, scales],
        )
    }

    /// A split chunk's FFN post of its own rows.
    fn sp_post(&self, rows: &RowsState, sp: SpRows) -> Result<(), String> {
        let b = rows.b;
        let own = |base: u64, row_bytes: usize| base + (sp.first * row_bytes) as u64;
        let stream = own(b.stream, self.hc_mult * self.dim * 4);
        self.rows_post(
            rows,
            sp.count,
            stream,
            own(b.ffn_mix, MIX_ROWS * 4),
            own(b.moe_reduced, self.dim * 2),
            stream,
        )
    }

    /// A split chunk's drafter seed ([DeviceLayer::draft_seed]): the in-place AllGather of
    /// `count` bf16 words a rank from `mine` into `chunk`, which the run's stream waits for.
    pub(super) fn sp_seed_gather(
        &self,
        rows: &RowsState,
        mine: u64,
        chunk: u64,
        count: usize,
    ) -> Result<(), String> {
        let overlap = rows.overlap.replace(false);
        let result = self.sp_collective(rows, "all_gather_seed", true, mine, chunk, count);
        rows.overlap.set(overlap);
        result
    }

    /// After a split chunk's last post: the last row's stream and pre mix (the head's, which rank
    /// `(n - 1) / part` posted) into that row on every rank, through each rank's slot of
    /// `sp_last`; the run's stream waits for it. (The drafter seeds each rank's own rows from
    /// their taps and gathers the seeded ring entries.)
    fn sp_finish(&self, rows: &RowsState, sp: SpRows, n: usize) -> Result<(), String> {
        let b = rows.b;
        let bulk = rows.bulk.as_ref().ok_or("rows finish: not a bulk chunk")?;
        let overlap = rows.overlap.replace(false);
        let result = (|| {
            let (stream_bytes, pre_bytes) = (self.hc_mult * self.dim * 4, self.hc_mult * 4);
            let slot_bytes = stream_bytes + pre_bytes;
            let mine = bulk.sp_last + (self.tp_rank * slot_bytes) as u64;
            let row = (sp.first + sp.count - 1) as u64;
            let cuda = self.cuda;
            cuda.copy_dtod_async(
                &Deviceptr(mine),
                &Deviceptr(b.stream + row * stream_bytes as u64),
                stream_bytes,
            )?;
            cuda.copy_dtod_async(
                &Deviceptr(mine + stream_bytes as u64),
                &Deviceptr(b.pre_mix + row * pre_bytes as u64),
                pre_bytes,
            )?;
            self.sp_collective(
                rows,
                "all_gather_last",
                true,
                mine,
                bulk.sp_last,
                slot_bytes / 2,
            )?;
            let owner = bulk.sp_last + ((n - 1) / sp.part * slot_bytes) as u64;
            let last = (n - 1) as u64;
            cuda.copy_dtod_async(
                &Deviceptr(b.stream + last * stream_bytes as u64),
                &Deviceptr(owner),
                stream_bytes,
            )?;
            cuda.copy_dtod_async(
                &Deviceptr(b.pre_mix + last * pre_bytes as u64),
                &Deviceptr(owner + stream_bytes as u64),
                pre_bytes,
            )
        })();
        rows.overlap.set(overlap);
        result
    }

    /// A layer's all-reduce of `n` rows: on the collective stream when `rows` is a paired bulk
    /// chunk ([RowsState::overlap]), else on the run's stream.
    fn reduce_layer_rows(
        &self,
        rows: &RowsState,
        stage: &'static str,
        input: crate::layer_device::DeviceSpan,
        output: crate::layer_device::DeviceSpan,
        n: usize,
    ) -> Result<(), String> {
        match rows
            .ar_done
            .as_ref()
            .filter(|_| rows.overlap.get() && n > R_MAX)
        {
            Some(done) => {
                self.reduce_rows_overlapped(stage, input, output, n, done)?;
                rows.ar_pending.set(true);
                Ok(())
            }
            None => self.reduce_rows(stage, input, output, n),
        }
    }

    /// A paired chunk's next reader of its last all-reduce: the run's stream waits for the
    /// collective, then the FFN post it fed runs when no seam fuses it. Nothing without overlap.
    pub(super) fn rows_ar_wait(&self, rows: &RowsState, n: usize) -> Result<(), String> {
        if rows.ar_pending.replace(false)
            && let Some(done) = rows.ar_done.as_ref()
        {
            self.cuda.run_wait(done)?;
        }
        if rows.post_after_ar.replace(false) {
            let b = rows.b;
            match self.sp_rows(rows, n) {
                Some(sp) => self.sp_post(rows, sp)?,
                None => self.rows_post(rows, n, b.stream, b.ffn_mix, b.moe_reduced, b.stream)?,
            }
        }
        Ok(())
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
    /// mix weights, wq_a and wkv, then wq_b (off under SPARKFLASH_PREFETCH_WQB=0).
    fn prefetch_attention(&self, rows: &RowsState, layer: usize) -> Result<(), String> {
        static WQB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wqb =
            *WQB.get_or_init(|| std::env::var("SPARKFLASH_PREFETCH_WQB").as_deref() != Ok("0"));
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
                self.weights
                    .pair(layer, "wq_b")
                    .ok()
                    .filter(|_| wqb)
                    .map_or((0, 0), |(w, _)| (w.0, rows.q_cols * rows.lora_cols)),
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
        if let Some(bulk) = rows.bulk.as_ref().filter(|_| n > R_MAX) {
            // A bulk chunk: the partials with each fn_w column held across 32 rows, then the
            // tails one block a row (no tickets).
            self.cuda.launch(
                bulk.k.seam_partials,
                (40, n.div_ceil(32) as u32, 1),
                (128, 1, 1),
                &[
                    stream,
                    in_pre,
                    w[0],
                    hc_rows,
                    b.seam_partials,
                    b.seam_row_buf,
                    n as u64,
                    u64::from(self.native_bf16),
                    u64::from(in_pre_stride),
                ],
            )?;
            return self.bulk_seam_tail(
                rows,
                bulk,
                n,
                w,
                [mix, pre_mix_out, normed, codes, scales, x_bf16],
                quant_narrow,
            );
        }
        if let Some((partials, tail)) = rows.kernels.seam_split.as_ref() {
            // Rows a block in the partials: 4 from 8 rows, 2 from 2 (the bench's fastest).
            let (kernel, per) = match n {
                0..=1 => (partials[0], 1),
                2..=7 => (partials[1], 2),
                _ => (partials[2], 4),
            };
            self.cuda.launch(
                kernel,
                (80, n.div_ceil(per) as u32, 1),
                (256, 1, 1),
                &[
                    stream,
                    in_pre,
                    w[0],
                    hc_rows,
                    b.seam_partials,
                    b.seam_row_buf,
                    u64::from(self.native_bf16),
                    u64::from(in_pre_stride),
                    n as u64,
                ],
            )?;
            return self.cuda.launch(
                *tail,
                (n as u32, 1, 1),
                (1024, 1, 1),
                &[
                    w[1],
                    w[2],
                    w[3],
                    mix,
                    pre_mix_out,
                    normed,
                    codes,
                    scales,
                    x_bf16,
                    b.seam_partials,
                    b.seam_row_buf,
                    u64::from(SEAM_NORM_BF16),
                    u64::from(self.native_bf16),
                    u64::from(quant_narrow),
                    u64::from(self.sinkhorn_iters),
                    u64::from(self.eps.to_bits()),
                    u64::from(self.hc_eps.to_bits()),
                    MIX_ROWS as u64,
                ],
            );
        }
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

    /// rows_post of `post` (residual, mix, delta) into `stream`, then the seam that reads it
    /// ([`Self::rows_seam`]'s arguments, no hc_rows copy). A bulk chunk computes the post's rows and
    /// the seam's partials in one pass over the stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_post_seam(
        &self,
        rows: &RowsState,
        n: usize,
        post: [u64; 3],
        stream: u64,
        in_pre: u64,
        in_pre_stride: u32,
        w: [u64; 4],
        mix: u64,
        pre_mix_out: u64,
        normed: u64,
        codes: u64,
        scales: u64,
        x_bf16: u64,
        quant_narrow: u32,
    ) -> Result<(), String> {
        let Some(bulk) = rows.bulk.as_ref().filter(|_| n > R_MAX) else {
            self.rows_post(rows, n, post[0], post[1], post[2], stream)?;
            return self.rows_seam(
                rows,
                n,
                stream,
                in_pre,
                in_pre_stride,
                w,
                mix,
                pre_mix_out,
                0,
                normed,
                codes,
                scales,
                x_bf16,
                quant_narrow,
            );
        };
        let b = rows.b;
        self.cuda.launch(
            bulk.k.post_partials,
            (40, n.div_ceil(32) as u32, 1),
            (128, 1, 1),
            &[
                post[0],
                post[1],
                post[2],
                stream,
                MIX_ROWS as u64,
                in_pre,
                w[0],
                b.seam_partials,
                b.seam_row_buf,
                n as u64,
                u64::from(self.native_bf16),
                u64::from(in_pre_stride),
            ],
        )?;
        self.bulk_seam_tail(
            rows,
            bulk,
            n,
            w,
            [mix, pre_mix_out, normed, codes, scales, x_bf16],
            quant_narrow,
        )
    }

    /// opt_bulk_mhc_tail over the chunk's rows once the partials are in: `out` is (mix,
    /// pre_mix_out, normed, codes, scales, x_bf16).
    fn bulk_seam_tail(
        &self,
        rows: &RowsState,
        bulk: &super::bulk::BulkState,
        n: usize,
        w: [u64; 4],
        out: [u64; 6],
        quant_narrow: u32,
    ) -> Result<(), String> {
        let b = rows.b;
        let [mix, pre_mix_out, normed, codes, scales, x_bf16] = out;
        self.cuda.launch(
            bulk.k.seam_tail,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                w[1],
                w[2],
                w[3],
                mix,
                pre_mix_out,
                normed,
                codes,
                scales,
                x_bf16,
                b.seam_partials,
                b.seam_row_buf,
                u64::from(SEAM_NORM_BF16),
                u64::from(self.native_bf16),
                u64::from(quant_narrow),
                u64::from(self.sinkhorn_iters),
                u64::from(self.eps.to_bits()),
                u64::from(self.hc_eps.to_bits()),
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
        (first, count): (usize, usize),
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
        let base = engram.base(p0);
        for row in 0..n {
            live.push(!engram.is_skipped(slot, base + row)?);
        }
        if !live.iter().any(|&on| on) {
            return Ok(());
        }
        if let Some(bulk) = rows.bulk.as_ref().filter(|_| n > R_MAX) {
            if engram.hc_mult > 4 || !engram.dim.is_multiple_of(2) {
                return Err(
                    "rows engram: the bulk post takes hc <= 4 copies of an even width".into(),
                );
            }
            return self.bulk_engram(rows, bulk, slot, &live[first..first + count], p0, first);
        }
        let b = rows.b;
        let layout = engram.layout;
        let align = crate::engram_device_state::ARENA_ALIGN;
        let (hc, dim) = (engram.hc_mult, engram.dim);
        let layer_id = layer as u32;
        for row in 0..n {
            let entry = slot * engram.positions + base + row;
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
        if let Some(bulk) = rows.bulk.as_ref().filter(|_| n > R_MAX) {
            return self.bulk_engram_wkv(rows, bulk, weight, scales, out_width, inner, n);
        }
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

    /// Writes the row table of a step over several lanes' rows: row r's lane offset from the
    /// first segment's lane (the same for its window ring and its compressed store) and its
    /// position. The q rope, the kv tail and the split attention then run once over all the rows
    /// instead of once per lane. Without lanes, or with the bf16 cache (one per run, not per lane),
    /// or with SPARKFLASH_ROW_TABLE=0, they stay per segment.
    pub(super) fn upload_row_table(
        &self,
        rows: &RowsState,
        segs: &[super::lanes::Seg],
    ) -> Result<(), String> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        rows.row_tab_for.borrow_mut().clear();
        let off = *OFF.get_or_init(|| std::env::var("SPARKFLASH_ROW_TABLE").as_deref() == Ok("0"));
        let Some(lanes) = self.lanes.as_ref() else {
            return Ok(());
        };
        if off || segs.len() < 2 || !self.kv_cache_off || rows.cap > R_MAX {
            return Ok(());
        }
        let first = lanes.pointers(segs[0].lane);
        let mut words = Vec::with_capacity(2 * R_MAX);
        for seg in segs {
            let lane = lanes.pointers(seg.lane);
            let offset = lane.kv_ring.wrapping_sub(first.kv_ring);
            if lane.store_codes.wrapping_sub(first.store_codes) != offset
                || lane.store_scales.wrapping_sub(first.store_scales) != offset
            {
                return Ok(());
            }
            for row in 0..seg.n {
                words.push(offset);
                words.push((seg.p0 + row) as u64);
            }
        }
        if words.len() > 2 * R_MAX {
            return Ok(());
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.cuda.copy_htod(&Deviceptr(rows.b.row_tab), &bytes)?;
        rows.row_tab_for.borrow_mut().extend_from_slice(segs);
        Ok(())
    }

    /// The row table when the step's rows are several lanes' and [Self::upload_row_table] wrote
    /// it for them.
    fn row_table(&self, rows: &RowsState, segs: &[super::lanes::Seg]) -> Option<u64> {
        (segs.len() > 1 && rows.row_tab_for.borrow().as_slice() == segs).then_some(rows.b.row_tab)
    }

    /// The launches of the step's segments: one over all the rows with the row table, else one
    /// per segment.
    fn row_units(segs: &[super::lanes::Seg], table: Option<u64>) -> Vec<super::lanes::Seg> {
        match (table, segs.first()) {
            (Some(_), Some(&first)) => vec![super::lanes::Seg {
                n: segs.iter().map(|seg| seg.n).sum(),
                row0: 0,
                ..first
            }],
            _ => segs.to_vec(),
        }
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
            launch_opt_rows(
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
                R_MAX,
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
        rows.quantise_bf16(self.cuda, b.q_normed, b.q_codes, b.q_scales, lora, n)?;
        gemv("wq_b", b.q_codes, b.q_scales, b.q, rows.q_cols, lora)?;
        let (cos, sin) = self.rows_rope_tables(layer)?;
        let (_, rope) = self.opt_rope.ok_or("rows: the opt rope is not loaded")?;
        let kv_norm = self
            .weights
            .get(layer, "kv_norm")
            .map_err(|error| format!("rows kv_norm: {error}"))?;
        // The q rope and the kv tail (kv_norm, rope, the window row's round trip, the cache append)
        // run per sequence: each reads its rows' positions and writes its own lane's ring. With
        // the row table one launch serves every lane's rows.
        let segs = self.step_segs(p0, n);
        let table = self.row_table(rows, &segs);
        for seg in &segs {
            self.enter_seg(seg)?;
            let appended = self
                .appended_rows
                .get(layer)
                .copied()
                .ok_or("rows: no cache count")?;
            if appended != seg.p0 {
                return Err(format!(
                    "rows kv: layer {layer}'s cache holds {appended} rows, the step starts at {}",
                    seg.p0
                ));
            }
        }
        for seg in Self::row_units(&segs, table) {
            self.enter_seg(&seg)?;
            let (row0, count) = (seg.row0 as u64, seg.n as u32);
            let pair_bytes = match table {
                Some(_) => 0,
                None => (self.rope_pairs as usize * 4 * seg.p0) as u64,
            };
            self.cuda.launch(
                rope,
                (self.heads_per_rank, count, 1),
                (32, 1, 1),
                &[
                    b.q + row0 * (rows.q_cols * 2) as u64,
                    cos + pair_bytes,
                    sin + pair_bytes,
                    u64::from(self.head_width),
                    u64::from(self.rope_head_dim),
                    u64::from(self.heads_per_rank),
                    u64::from(self.rope_pairs),
                    table.unwrap_or(0),
                ],
            )?;
            let ring_layer = self.kv_ring_base + self.kv_ring.layer_offset(layer) as u64;
            let ring_codes = ring_layer + self.kv_ring.codes_offset(0)? as u64;
            let ring_scales = ring_layer + self.kv_ring.scales_offset(0)? as u64;
            let cache = self.kv_cache.layer_span(layer)?;
            let kv_row = row0 * (rows.kv_cols * 2) as u64;
            self.cuda.launch(
                rows.kernels.kv,
                (count, 1, 1),
                (256, 1, 1),
                &[
                    b.kv_latent + kv_row,
                    kv_norm.0,
                    u64::from(self.eps.to_bits()),
                    cos,
                    sin,
                    b.kv_rotated + kv_row,
                    ring_codes,
                    ring_scales,
                    if self.kv_cache_off { 0 } else { cache.device },
                    self.capacity_rows as u64,
                    self.kv_ring.rows() as u64,
                    u64::from(self.head_width),
                    u64::from(self.rope_head_dim),
                    seg.p0 as u64,
                    table.unwrap_or(0),
                ],
            )?;
        }
        for seg in &segs {
            self.enter_seg(seg)?;
            if let Some(count) = self.appended_rows.get_mut(layer) {
                *count = seg.p0 + seg.n;
            }
        }
        Ok(())
    }

    /// The compressor seam and the indexer select, per row through the M = 1 stages with the
    /// layer's views on the row: normed (the compressor input), q_normed (the index query input),
    /// selected (the row's list, kept across layers) and the candidate mask. When the rows'
    /// indexer GEMVs ran together (rows_seam_prepare), an index-source layer's select runs once
    /// per segment after its rows' seams instead ([Self::rows_index_select]).
    fn rows_seam_select(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let saved = self.buffers;
        // The row views replace the ring's three scratch pointers only: its other pointers are the
        // selected lane's, which the rows below may change.
        let saved_scratch = self
            .seam_ring
            .map(|ring| (ring.candidate_mask, ring.weights_f32, ring.q_index_f32));
        let dim = self.dim;
        let lora = rows.lora_cols;
        let prepared = self.rows_seam_prepare(rows, layer, p0, n)?;
        let heads = self.index_n_heads;
        let index_cols = heads * self.index_head_dim;
        let segs = self.step_segs(p0, n);
        let batched = prepared && self.rows_select_batched(layer, &segs);
        let mut outcome = Ok(());
        'segs: for seg in &segs {
            if let Err(error) = self.enter_seg(seg) {
                outcome = Err(error);
                break;
            }
            for row in seg.row0..seg.row0 + seg.n {
                let position = seg.p0 + row - seg.row0;
                self.buffers.residual.normed = span(b.normed + (row * dim * 4) as u64, dim * 4);
                self.buffers.workspace.q_normed =
                    span(b.q_normed + (row * lora * 2) as u64, lora * 2);
                self.buffers.workspace.selected = span(
                    b.selected + (row * SELECTED_STRIDE * 4) as u64,
                    SELECTED_STRIDE * 4,
                );
                if let Some(ring) = self.seam_ring.as_mut() {
                    ring.candidate_mask = Deviceptr(b.masks + (row * rows.mask_bytes) as u64);
                    if prepared {
                        ring.weights_f32 = Deviceptr(b.idx_weights_f32 + (row * heads * 4) as u64);
                        ring.q_index_f32 = Deviceptr(b.idx_q_f32 + (row * index_cols * 4) as u64);
                    }
                }
                self.seam_rows_prepared = prepared;
                outcome = self.reference_seam(layer, position);
                if outcome.is_ok() && !batched {
                    outcome = self.ref_select(layer, position);
                }
                if outcome.is_err() {
                    break 'segs;
                }
            }
            if batched {
                outcome = self.rows_index_select(rows, layer, seg);
                if outcome.is_err() {
                    break;
                }
            }
        }
        self.seam_rows_prepared = false;
        self.buffers = saved;
        if let (Some(ring), Some((mask, weights, query))) = (self.seam_ring.as_mut(), saved_scratch)
        {
            ring.candidate_mask = mask;
            ring.weights_f32 = weights;
            ring.q_index_f32 = query;
        }
        outcome
    }

    /// Whether an index-source layer's select runs batched over each segment's rows: the opt
    /// select's row kernels are loaded (not the rank form, not SPARKFLASH_ROWS_SELECT=0), the
    /// indexer geometry is theirs, and every segment's scores and scratch fit.
    fn rows_select_batched(&self, layer: usize, segs: &[super::lanes::Seg]) -> bool {
        let ratio = self.compress_ratio.get(layer).copied().unwrap_or(0);
        let Some(select) = self.opt_select.as_ref() else {
            return false;
        };
        if ratio == 0
            || select.rows.is_none()
            || !self.is_index_source(layer)
            || self.index_n_heads > 64
            || self.index_head_dim != 128
            || std::env::var("SPARKFLASH_ROWS_SELECT").as_deref() == Ok("0")
        {
            return false;
        }
        segs.iter().all(|seg| {
            let reach = (seg.p0 + seg.n) / ratio;
            let stride = reach.next_multiple_of(4);
            reach <= select.capacity
                && seg.n * stride <= select.scores_floats
                && seg.n * (reach + reach.div_ceil(4)) <= select.scratch_words
        })
    }

    /// One segment's index-source select, batched: opt_select_scores_rows scores every row's
    /// reachable compressed positions, then opt_select_rows takes each row's top-k into its
    /// selected list as expanded store rows (what ref_select's select and fill write) and writes
    /// or reads its candidate mask. Every row's chain is the M = 1 select's.
    fn rows_index_select(
        &mut self,
        rows: &RowsState,
        layer: usize,
        seg: &super::lanes::Seg,
    ) -> Result<(), String> {
        let ratio = self.compress_ratio.get(layer).copied().unwrap_or(0);
        let reach = (seg.p0 + seg.n) / ratio.max(1);
        if reach == 0 {
            return Ok(());
        }
        let owner = self
            .owner_index_for(layer)
            .ok_or_else(|| format!("rows select: layer {layer} has no kv-source owner"))?;
        let (Some(ring), Some(select)) = (self.seam_ring, self.opt_select.as_ref()) else {
            return Err(Self::refusal("rows select"));
        };
        let Some((scores_rows, select_rows, scores_seg)) = select.rows else {
            return Err(Self::refusal("rows select"));
        };
        let b = rows.b;
        let heads = self.index_n_heads;
        let index_cols = heads * self.index_head_dim;
        let row0 = seg.row0;
        let keys =
            ring.index_keys.0 + (owner * self.max_positions * self.index_head_dim * 4) as u64;
        let stride = reach.next_multiple_of(4);
        let mask_mode = match layer.cmp(&self.candidate_source_layer) {
            std::cmp::Ordering::Less => 0u64,
            std::cmp::Ordering::Equal => 2,
            std::cmp::Ordering::Greater => 1,
        };
        let masks = b.masks + (row0 * rows.mask_bytes) as u64;
        if heads == SELECT_SEG_HEADS && seg.n <= SELECT_SEG_ROWS {
            // The segment's rows share each key tile; a consumed mask skips what no row needs.
            let tiles = reach.div_ceil(SELECT_SEG_TILE);
            let cap = std::env::var("SPARKFLASH_SEG_BLOCKS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(SELECT_SEG_BLOCKS)
                .max(1);
            let slices = (4 / seg.n).max(1);
            self.cuda.launch(
                scores_seg,
                (tiles.min(cap) as u32, 1, 1),
                ((64 * seg.n * slices) as u32, 1, 1),
                &[
                    b.idx_q_f32 + (row0 * index_cols * 4) as u64,
                    keys,
                    b.idx_weights_f32 + (row0 * heads * 4) as u64,
                    select.scores_out.0,
                    if mask_mode == 1 { masks } else { 0 },
                    seg.p0 as u64,
                    ratio as u64,
                    stride as u64,
                    (rows.mask_bytes / 4) as u64,
                    seg.n as u64,
                ],
            )?;
        } else {
            let tiles = reach.div_ceil(SELECT_SCORE_TILE);
            let blocks = tiles.min((SELECT_SCORE_BLOCKS / seg.n).max(1));
            self.cuda.launch(
                scores_rows,
                (blocks as u32, seg.n as u32, 1),
                (256, 1, 1),
                &[
                    b.idx_q_f32 + (row0 * index_cols * 4) as u64,
                    keys,
                    b.idx_weights_f32 + (row0 * heads * 4) as u64,
                    select.scores_out.0,
                    seg.p0 as u64,
                    ratio as u64,
                    stride as u64,
                    heads as u64,
                ],
            )?;
        }
        self.cuda.launch(
            select_rows,
            (seg.n as u32, 1, 1),
            (1024, 1, 1),
            &[
                select.scores_out.0,
                b.selected + (row0 * SELECTED_STRIDE * 4) as u64,
                if mask_mode == 1 { masks } else { 0 },
                if mask_mode == 2 { masks } else { 0 },
                select.scratch.0,
                seg.n as u64,
                seg.p0 as u64,
                ratio as u64,
                self.index_topk as u64,
                mask_mode,
                self.candidate_topk_blocks as u64,
                self.candidate_block_size as u64,
                stride as u64,
                SELECTED_STRIDE as u64,
                (rows.mask_bytes / 4) as u64,
                (reach + reach.div_ceil(4)) as u64,
            ],
        )?;
        self.attn_window.selected_count = self.index_topk.min((seg.p0 + seg.n) / ratio);
        Ok(())
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
        let segs = self.step_segs(p0, n);
        let kv_source = self.is_kv_source(layer);
        let index = self.is_index_source(layer)
            && segs.iter().any(|seg| (seg.p0 + seg.n) / ratio.max(1) > 0);
        if ratio == 0
            || (!kv_source && !index)
            || segs.iter().any(|seg| seg.n + ratio > ring.rows + 1)
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
            // Each sequence's rows land in its own lane's compressor ring.
            for seg in &segs {
                self.enter_seg(seg)?;
                let ring = self.seam_ring.ok_or("rows seam: no compressor ring")?;
                let mut row = 0;
                while row < seg.n {
                    let slot = (seg.p0 + row) % ring.rows;
                    let run = (ring.rows - slot).min(seg.n - row);
                    let a = b.normed + ((seg.row0 + row) * dim * 4) as u64;
                    let at = (owner_base + slot * row_bytes) as u64;
                    hostw(wkv_mode, a, wkv, ring.wkv.0 + at, ring.head_dim, run)?;
                    if let Some((wgate, wgate_mode)) = staged.wgate {
                        hostw(wgate_mode, a, wgate, ring.wgate.0 + at, ring.head_dim, run)?;
                    }
                    row += run;
                }
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
            launch_opt_rows(
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
                R_MAX,
            )
            .map_err(|error| format!("rows indexer wq_b: {error}"))?;
            let pairs = self.rope_pairs as usize;
            let table_bytes = self.max_positions * pairs * 4;
            let tables = self.buffers.workspace.rope_tables_compressed.device;
            for row in 0..n {
                let (_, position) = Self::row_seg(&segs, row)?;
                let rope_offset = (position * pairs * 4) as u64;
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
                    position as u64,
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
        let sink_stride = self.out_geometry.heads_per_rank * 4;
        if (layer + 1) * sink_stride > self.sink_pool_bytes {
            return Err(format!(
                "rows attention: layer {layer}'s sink row is outside the pool"
            ));
        }
        let sink = self.sink_pool.0 + (layer * sink_stride) as u64;
        // A bulk chunk runs the tensor-core attention (16 heads as the MMA's M), one block a row.
        let bulk_attn = rows
            .bulk
            .as_ref()
            .filter(|_| n > R_MAX && self.heads_per_rank == 16)
            .map(|bulk| bulk.k.attn);
        let (cos, sin) = self.rows_rope_tables(layer)?;
        let q_row = (rows.q_cols * 2) as u64;
        // Each sequence attends its own lane's window ring and compressed store: the split
        // attention runs every lane's rows in one launch with the row table.
        let segs = self.step_segs(p0, n);
        let table = self.row_table(rows, &segs).filter(|_| {
            bulk_attn.is_none() && rows.kernels.attn_split.is_some() && self.heads_per_rank == 16
        });
        let last = segs.last().copied();
        for seg in Self::row_units(&segs, table) {
            self.enter_seg(&seg)?;
            let ring_layer = self.kv_ring_base + self.kv_ring.layer_offset(layer) as u64;
            let ring_codes = ring_layer + self.kv_ring.codes_offset(0)? as u64;
            let ring_scales = ring_layer + self.kv_ring.scales_offset(0)? as u64;
            let (kv_fp4, kv_scales) = if ratio > 0 {
                let owner = self.owner_index_for(layer).ok_or_else(|| {
                    format!("rows attention: layer {layer} has no kv-source owner")
                })?;
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
            let (row0, count) = (seg.row0 as u64, seg.n as u32);
            let selected = b.selected + row0 * (SELECTED_STRIDE * 4) as u64;
            let attn_body = b.attn_body + row0 * q_row;
            let geometry = [seg.p0 as u64, ratio as u64, self.index_topk as u64];
            let ring = [SELECTED_STRIDE as u64, self.kv_ring.rows() as u64];
            match (bulk_attn, rows.kernels.attn_split) {
                (None, Some([scores, softmax, values])) if self.heads_per_rank == 16 => {
                    let attn_scores =
                        b.attn_scores + row0 * (ATTN_HEADS * ATTN_MAX_ATTENDED * 4) as u64;
                    let attn_probs =
                        b.attn_probs + row0 * (ATTN_MAX_ATTENDED * ATTN_HEADS * 4) as u64;
                    let attn_ssum = b.attn_ssum + row0 * (ATTN_HEADS * 4) as u64;
                    let chunks = ATTN_MAX_ATTENDED.div_ceil(ATTN_SCORE_CHUNK) as u32;
                    let mut args = vec![
                        b.q + row0 * q_row,
                        kv_fp4,
                        kv_scales,
                        ring_codes,
                        ring_scales,
                        selected,
                        attn_scores,
                    ];
                    args.extend(geometry);
                    args.extend(ring);
                    args.push(table.unwrap_or(0));
                    self.cuda
                        .launch(scores, (chunks, count, 1), (256, 1, 1), &args)?;
                    let mut args = vec![attn_scores, sink, attn_probs, attn_ssum];
                    args.extend(geometry);
                    args.push(table.unwrap_or(0));
                    self.cuda
                        .launch(softmax, (ATTN_HEADS as u32, count, 1), (256, 1, 1), &args)?;
                    let mut args = vec![
                        kv_fp4,
                        kv_scales,
                        ring_codes,
                        ring_scales,
                        selected,
                        attn_probs,
                        attn_ssum,
                        attn_body,
                    ];
                    args.extend(geometry);
                    args.extend(ring);
                    args.push(table.unwrap_or(0));
                    self.cuda
                        .launch(values, (8, count, 1), (256, 1, 1), &args)?;
                }
                _ => {
                    let (attn, grid) = match bulk_attn {
                        Some(kernel) => (kernel, (count, 1, 1)),
                        None => (rows.kernels.attn, (self.heads_per_rank, count, 1)),
                    };
                    let mut args = vec![
                        b.q + row0 * q_row,
                        kv_fp4,
                        kv_scales,
                        ring_codes,
                        ring_scales,
                        selected,
                        sink,
                        attn_body,
                    ];
                    args.extend(geometry);
                    args.extend(ring);
                    self.cuda.launch(attn, grid, (256, 1, 1), &args)?;
                }
            }
            self.cuda.launch(
                rows.kernels.inv_rope,
                (self.heads_per_rank, count, 1),
                (32, 1, 1),
                &[
                    b.attn_body + row0 * q_row,
                    cos,
                    sin,
                    u64::from(self.head_width),
                    u64::from(self.rope_head_dim),
                    seg.p0 as u64,
                    table.unwrap_or(0),
                ],
            )?;
        }
        // The last segment's lane stays selected, as after a per-segment loop.
        if let Some(seg) = last {
            self.enter_seg(&seg)?;
        }
        Ok(())
    }

    /// wo_a over both groups, the bf16 quantiser, wo_b.
    fn rows_output(&mut self, rows: &RowsState, layer: usize, n: usize) -> Result<(), String> {
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
        launch_opt_rows(
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
            R_MAX,
        )
        .map_err(|error| format!("rows wo_a: {error}"))?;
        let lo = rows.lo_cols;
        rows.quantise_bf16(self.cuda, b.attn_lo, b.o_codes, b.o_scales, lo, n)?;
        launch_opt_rows(
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
            R_MAX,
        )
        .map_err(|error| format!("rows wo_b: {error}"))
    }

    /// The router and the rank-split shared expert over the rows (the fused routed experts are
    /// [Self::rows_moe_routed]).
    fn rows_moe_inputs(&mut self, rows: &RowsState, layer: usize, n: usize) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        let tokens = n as u32;
        let (gate, bias) = self
            .router
            .as_ref()
            .and_then(|router| router.get(layer).copied())
            .ok_or_else(|| format!("rows router: layer {layer} has no router pointers"))?;
        let (_, router_kernel, ticket) = self.opt_router.ok_or("rows: no opt router")?;
        let bulk = rows.bulk.as_ref().filter(|_| n > R_MAX);
        if let Some(bulk) = bulk {
            self.bulk_router(rows, bulk, gate.0, bias.0, n)?;
            self.vision_route(layer, bulk.dots, true, b.moe_ids, b.moe_weights)?;
        } else {
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
            self.vision_route(layer, b.scores, false, b.moe_ids, b.moe_weights)?;
        }
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
        let gate_up = |a: u64| OptArgs {
            a,
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
        };
        // A split chunk's FFN rows are its gathered bf16 rows (no f32 seam rows).
        match bulk.filter(|_| self.sp_rows(rows, n).is_some()) {
            Some(bulk) => bulk.mma(self.cuda, bulk.k.bf16a_f32o, &gate_up(b.ffn_x)),
            None => launch_opt_rows(self.cuda, OptKind::F32aF32o, &gate_up(b.seam_row), R_MAX),
        }
        .map_err(|error| format!("rows shared gate/up: {error}"))?;
        rows.swiglu(
            self.cuda,
            b.shared_gate,
            b.shared_gate + (n * local * 4) as u64,
            b.shared_swiglu,
            self.swiglu_limit,
            n * local,
        )
        .map_err(|error| format!("rows shared swiglu: {error}"))?;
        let kb0 = rank * local_tiles;
        launch_opt_rows(
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
            R_MAX,
        )
        .map_err(|error| format!("rows shared down: {error}"))
    }

    /// The fused routed experts over the rows, adding the shared expert's rows.
    fn rows_moe_routed(
        &mut self,
        rows: &RowsState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::moe_routed_device::{DeviceSpan as MoeSpan, MoeRoutedOperands};
        let b = rows.b;
        let dim = self.dim;
        let bulk = rows.bulk.as_ref().filter(|_| n > R_MAX);
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
        match bulk {
            Some(bulk) => device.run_bulk(self.cuda, &bulk.moe, layer, n, &operands),
            None => device.run(self.cuda, layer, p0, n, &operands),
        }
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

    /// [Self::exchange_slots] as one ring AllGather (the same bits in the same slots), for a bulk
    /// chunk's slots: the point-to-point sends of a 2K-row embedding slot take ~110 ms.
    fn gather_slots(
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
        // SAFETY: as exchange_slots; this rank's slot is the in-place AllGather's send buffer.
        unsafe { handle.gather_bf16(mine, gather, count, 0) }
            .map_err(|error| format!("{what}: gather: {error}"))?;
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
    /// shard for every [HEAD_ROWS] rows, the per-row fold, then the cross-rank argmax. Returns
    /// (token, value) per row.
    pub(super) fn head_rows(
        &self,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        n: usize,
    ) -> Result<Vec<(u32, f32)>, String> {
        self.head_rows_sampled(rows, tail, transport, n, &super::sample::Sampler::GREEDY, 0)
    }

    /// [`Self::head_rows`] under `sampler`: row y's token lands at sequence index `index0 + y`
    /// (the noise's position). Greedy is the argmax; temperature-only adds the noise in the head
    /// kernel; a truncated sampler cuts and samples on the host from the exchanged opt_rows_topc
    /// slots (the device exchange only). A sampled row's value is 0.
    pub(super) fn head_rows_sampled(
        &self,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        n: usize,
        sampler: &super::sample::Sampler,
        index0: usize,
    ) -> Result<Vec<(u32, f32)>, String> {
        let samplers: Vec<_> = (0..n).map(|row| (*sampler, index0 + row)).collect();
        self.head_rows_multi(rows, tail, transport, &samplers)
    }

    /// The head over rows of several requests: row y samples under `samplers[y].0` with its token
    /// at sequence index `samplers[y].1`. Rows that share one sampler at consecutive indices take
    /// the head kernel's scalar noise (the single request's path); otherwise each row's noise comes
    /// from the per-row parameters, and a truncated row is picked on the host while the others take
    /// the device argmax.
    pub(super) fn head_rows_multi(
        &self,
        rows: &RowsState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        samplers: &[(super::sample::Sampler, usize)],
    ) -> Result<Vec<(u32, f32)>, String> {
        let n = samplers.len();
        let Some(&(sampler, index0)) = samplers.first() else {
            return Ok(Vec::new());
        };
        if n > R_MAX {
            return Err(format!("rows head: {n} rows are more than {R_MAX}"));
        }
        // Greedy rows draw no noise, so an all-greedy step is one sampler's too.
        let uniform = samplers
            .iter()
            .enumerate()
            .all(|(row, &(other, index))| other == sampler && index == index0 + row)
            || samplers.iter().all(|(other, _)| other.is_greedy());
        let b = rows.b;
        let dim = self.dim;
        rows.hc_collapse(
            self.cuda,
            b.pre_mix,
            b.stream,
            b.lm_hidden,
            tail.buffers.certified_count.0,
            (1, n, 0),
        )?;
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
        let vocab = crate::tp4_tail::VOCAB_PER_RANK;
        let truncated: Vec<bool> = samplers.iter().map(|(s, _)| s.truncated()).collect();
        let any_truncated = truncated.iter().any(|&cut| cut);
        let sampled = samplers.iter().any(|(s, _)| !s.is_greedy());
        let (inv_t, seed) = if uniform {
            sampler.head_noise()
        } else {
            (0, 0)
        };
        // The per-row noise ({inv_t bits, position, seed lo, seed hi}, then the topc temperatures
        // at R_MAX * 4 words), written before the launch reads them (the last step's head is done:
        // its picks were read back).
        let params = if uniform {
            0
        } else {
            let mut words = [0u32; R_MAX * 8];
            for (row, (sampler, index)) in samplers.iter().enumerate() {
                let (inv_t, seed) = sampler.head_noise();
                words[row * 4] = inv_t as u32;
                words[row * 4 + 1] = *index as u32;
                words[row * 4 + 2] = seed as u32;
                words[row * 4 + 3] = (seed >> 32) as u32;
                words[R_MAX * 4 + row * 4] = sampler.inv_t().to_bits();
            }
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            rows.head_params.write_at(0, &bytes)?;
            rows.head_params.device().0
        };
        let id_base = self.device_exchange().map_or(0, |(_, rank)| rank * vocab);
        // HEAD_ROWS rows a pass over the vocabulary shard: each piece's chunk maxima, logits and
        // noise are its rows of the step's.
        for row0 in (0..n).step_by(HEAD_ROWS) {
            self.cuda.launch(
                rows.kernels.lm_head,
                (chunks as u32, 1, 1),
                (512, 1, 1),
                &[
                    b.lm_normed + (row0 * dim * 2) as u64,
                    tail.head_weight.0,
                    b.lm_block_max + (row0 * chunks * 4) as u64,
                    b.lm_block_arg + (row0 * chunks * 4) as u64,
                    vocab as u64,
                    HEAD_ROWS.min(n - row0) as u64,
                    if any_truncated {
                        b.logits + (row0 * vocab * 4) as u64
                    } else {
                        0
                    },
                    inv_t,
                    seed,
                    (index0 + row0) as u64,
                    id_base as u64,
                    if params == 0 {
                        0
                    } else {
                        params + (row0 * 16) as u64
                    },
                ],
            )?;
        }
        let mut picked: Vec<Option<u32>> = vec![None; n];
        if any_truncated {
            let Some((world, rank)) = self.device_exchange() else {
                return Err(
                    "rows head: top-k / top-p sampling needs the device exchange (SPARKFLASH_DEVICE_GATHER)"
                        .to_owned(),
                );
            };
            let words = super::sample::TOPC_WORDS;
            self.cuda.launch(
                rows.kernels.topc,
                (n as u32, 1, 1),
                (1024, 1, 1),
                &[
                    b.logits,
                    vocab as u64,
                    u64::from(sampler.inv_t().to_bits()),
                    id_base as u64,
                    b.topc_gather + (rank * n * words * 4) as u64,
                    if uniform {
                        0
                    } else {
                        params + (R_MAX * 16) as u64
                    },
                ],
            )?;
            self.exchange_slots(b.topc_gather, n * words * 2, rank, "top-p gather")?;
            let mut bytes = vec![0u8; world * n * words * 4];
            self.cuda.copy_dtoh(&mut bytes, &Deviceptr(b.topc_gather))?;
            let slots: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            for (row, (sampler, index)) in samplers.iter().enumerate() {
                if truncated[row] {
                    picked[row] =
                        Some(sampler.pick_truncated(&slots, n, row, world, *index as u32));
                }
            }
            if truncated.iter().all(|&cut| cut) {
                return Ok(picked
                    .into_iter()
                    .map(|id| (id.unwrap_or(0), 0.0))
                    .collect());
            }
        }
        if sampled && self.device_exchange().is_none() {
            return Err(
                "rows head: sampling needs the device exchange (SPARKFLASH_DEVICE_GATHER)"
                    .to_owned(),
            );
        }
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
                .map(|row| match picked[row] {
                    Some(id) => (id, 0.0),
                    None => (word(row * 4), f32::from_bits(word((n + row) * 4))),
                })
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
