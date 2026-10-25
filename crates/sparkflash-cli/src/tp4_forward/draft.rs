//! The on-device DSpark drafter: the three `mtp.{s}` stages over a block of [`BLOCK`] rows, the
//! shared head and the Markov head, all on the device and tensor-parallel like the backbone.
//!
//! Each stage is a decoder block with the backbone's tensor names and geometry, so it runs on the
//! multi-row step's kernels and buffers ([`RowsState`]): the same seams, GEMVs, rope, KV kernel,
//! output projection, rank-split shared expert and one-shot reduces. What differs is the drafter's
//! own (kernels/opt_draft_sm121.cu): the attention reads the stage's linear ring, which holds the
//! main rows' KV at their positions and the block's own rows at `d..d + BLOCK`; the router and the
//! fused MoE are the top-3-of-128 instantiations; and the head adds the Markov bias one draft at a
//! time, each step's argmax feeding the next step's bias (forward/mtp.rs dspark_forward_observed).
//!
//! The drafter only has to draft well: its arithmetic follows the reference's operations but is not
//! held to the verify step's bit-exactness. The verify step alone decides what is emitted.

use super::rows::{R_MAX, RowsState};
use super::*;
use crate::layer_device::DeviceSpan;
use sparkflash_kernels::cuda_driver::{Kernel, Module};
use sparkflash_kernels::mxfp8_gemv_launch::{OptArgs, OptKind, launch_opt};

/// The draft block: the anchor and four noise rows (`dspark_block_size`).
pub(super) const BLOCK: usize = 5;
/// The DSpark stages (`n_mtp_layers`).
const STAGES: usize = 3;
/// `dspark_n_routed_experts` / `dspark_n_activated_experts`.
const EXPERTS: usize = 128;
const TOP_K: usize = 3;
/// The tap width: three target layers of `dim`.
const TAPS: usize = 3;
/// The head logits' blocks: one per GB10 SM (each strides over the vocabulary slice), so every SM
/// streams the head; the 32 chunks of the slice left a third of them idle.
const LOGITS_BLOCKS: u32 = 48;
/// The Markov head's rank.
const MARKOV_RANK: usize = 256;
/// The expert intermediate rows on this rank (2304 / 4).
const LOCAL_INTER: usize = 576;
/// The transport sequence range the draft argmax gathers use.
const DRAFT_GATHER_SEQ: u64 = u64::MAX - 0xD_0000;

struct DraftKernels {
    _modules: Vec<Module>,
    taps: Kernel,
    norm_quant: Kernel,
    attn: Kernel,
    logits: Kernel,
    head_quant: Kernel,
    logits_fp8: Kernel,
    markov: Kernel,
    fold_prob: Kernel,
    pick_prob: Kernel,
    router: Kernel,
    gate_up: Kernel,
    down: Kernel,
}

/// One stage's resident operands: arena pointers (B12x weights and their scale grids), its expert
/// pointer table and its ring.
struct Stage {
    attn_seam: [u64; 4],
    ffn_seam: [u64; 4],
    wq_a: (u64, u64),
    wkv: (u64, u64),
    q_norm: u64,
    wq_b: (u64, u64),
    kv_norm: u64,
    sink: u64,
    wo_a: (u64, u64),
    wo_b: (u64, u64),
    gate: u64,
    bias: u64,
    shared: [(u64, u64); 3],
    /// `[gate, up, down][EXPERTS][weight, scales]`.
    table: u64,
    ring_codes: u64,
    ring_scales: u64,
}

/// Where the draft block's anchor row comes from.
pub(super) enum AnchorRow<'a> {
    /// Its f32 embedding, gathered on the host.
    Host(&'a [f32]),
    /// Gathered on the device from the owners' shards of `embed` over the exchange.
    Device {
        embed: u64,
        exchange: (usize, usize),
    },
}

/// The drafter's run-scope state.
pub(super) struct DraftState {
    kernels: DraftKernels,
    stages: Vec<Stage>,
    main_proj: (u64, u64),
    main_norm: u64,
    head_norm: u64,
    markov_head: u64,
    markov_embed: u64,
    /// `[R_MAX][TAPS][dim]` f32: the rows' taps, written by the rows step's tap hook.
    pub(super) taps: u64,
    main_x: u64,
    main_codes: u64,
    main_scales: u64,
    h_codes: u64,
    h_scales: u64,
    logits: u64,
    /// `[BLOCK + 1]` u32 ids (the anchor then the drafts), then `[BLOCK + 1]` f32: the draft's
    /// softmax probability of each draft (slot 0 unused).
    ids: u64,
    block_max: u64,
    block_arg: u64,
    block_sum: u64,
    value: u64,
    index: u64,
    ticket: u64,
    /// The head slice as E4M3 codes and ue8m0 scales for the draft logits (half the bf16 head's
    /// bytes), filled from the verify head at the first draft; None under
    /// SPARKFLASH_DSPARK_HEAD_FP8=0. The verify keeps the bf16 head.
    head_fp8: Option<(u64, u64)>,
    head_fp8_ready: Cell<bool>,
    /// The noise token's embedding row (f32), gathered once at open, and its device copy.
    noise_row: Vec<f32>,
    noise: u64,
    ring_rows: usize,
    allocations: Vec<Deviceptr>,
    gathers: Cell<u64>,
}

impl DraftState {
    /// Resolves the three stages' operands from the arena, retiles main_proj for the GEMV, carves
    /// the drafter's buffers and the stage rings, and loads opt_draft / the top-3 router and MoE.
    ///
    /// # Errors
    /// A missing tensor or kernel, or a load or allocation failure.
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        arena: &sparkflash_device::arena::ResidentArena,
        kernel_dir: &Path,
        noise_row: Vec<f32>,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let dim = device.dim;
        let mut modules = Vec::new();
        let mut load = |file: &str, name: &'static str| -> Result<Kernel, String> {
            let (cubin, _sha) =
                cached_cubin(&kernel_dir.join(file)).map_err(|error| format!("draft: {error}"))?;
            let (module, kernel) = cuda.load_named(&cubin, name)?;
            modules.push(module);
            Ok(kernel)
        };
        let taps = load("opt_draft_sm121.cubin", "opt_draft_taps")?;
        let norm_quant = load("opt_draft_sm121.cubin", "opt_draft_norm_quant")?;
        let attn = load("opt_draft_sm121.cubin", "opt_draft_attn")?;
        let logits = load("opt_draft_sm121.cubin", "opt_draft_logits")?;
        let head_quant = load("opt_draft_sm121.cubin", "opt_draft_head_fp8")?;
        let logits_fp8 = load("opt_draft_sm121.cubin", "opt_draft_logits_fp8")?;
        let markov = load("opt_draft_sm121.cubin", "opt_draft_markov")?;
        let fold_prob = load("opt_draft_sm121.cubin", "opt_draft_fold_prob")?;
        let pick_prob = load("opt_draft_sm121.cubin", "opt_draft_pick_prob")?;
        let router = load("opt_router_sm121.cubin", "opt_router_e128k3_sm121")?;
        let gate_up = load("opt_moe_sm121.cubin", "opt_moe_gate_up_k3_sm121")?;
        let down = load("opt_moe_sm121.cubin", "opt_moe_down_combine_k3_sm121")?;
        let kernels = DraftKernels {
            _modules: modules,
            taps,
            norm_quant,
            attn,
            logits,
            head_quant,
            logits_fp8,
            markov,
            fold_prob,
            pick_prob,
            router,
            gate_up,
            down,
        };

        let ptr = |name: String| -> Result<u64, String> {
            arena
                .device_ptr(&name)
                .map(|pointer| pointer.0)
                .ok_or_else(|| format!("draft: the arena holds no {name}"))
        };
        let pair = |base: String| -> Result<(u64, u64), String> {
            Ok((
                ptr(format!("{base}.weight"))?,
                ptr(format!("{base}.scale"))?,
            ))
        };
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize| -> Result<u64, String> {
            let pointer = cuda.alloc(bytes.max(4), &mut total)?;
            cuda.copy_htod(&pointer, &vec![0u8; bytes.max(4)])?;
            allocations.push(pointer);
            Ok(pointer.0)
        };
        // The stage rings' rows (what a lane lays out), and the rows of this state's own rings:
        // fewer when lanes hold every sequence's rings instead.
        let ring_rows = device.max_positions + BLOCK + 1;
        let own_rows = device.sequence_rows + BLOCK + 1;
        let latent = device.head_width as usize;
        let mut stages = Vec::with_capacity(STAGES);
        for stage in 0..STAGES {
            let p = format!("mtp.{stage}");
            let mut words = Vec::with_capacity(3 * EXPERTS * 2);
            for leaf in ["w1", "w3", "w2"] {
                for expert in 0..EXPERTS {
                    let (weight, scales) = pair(format!("{p}.ffn.experts.{expert}.{leaf}"))?;
                    words.push(weight);
                    words.push(scales);
                }
            }
            let table_bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let table = alloc(table_bytes.len())?;
            cuda.copy_htod(&Deviceptr(table), &table_bytes)?;
            stages.push(Stage {
                attn_seam: [
                    ptr(format!("{p}.hc_attn_fn"))?,
                    ptr(format!("{p}.hc_attn_base"))?,
                    ptr(format!("{p}.hc_attn_scale"))?,
                    ptr(format!("{p}.attn_norm.weight"))?,
                ],
                ffn_seam: [
                    ptr(format!("{p}.hc_ffn_fn"))?,
                    ptr(format!("{p}.hc_ffn_base"))?,
                    ptr(format!("{p}.hc_ffn_scale"))?,
                    ptr(format!("{p}.ffn_norm.weight"))?,
                ],
                wq_a: pair(format!("{p}.attn.wq_a"))?,
                wkv: pair(format!("{p}.attn.wkv"))?,
                q_norm: ptr(format!("{p}.attn.q_norm.weight"))?,
                wq_b: pair(format!("{p}.attn.wq_b"))?,
                kv_norm: ptr(format!("{p}.attn.kv_norm.weight"))?,
                sink: ptr(format!("{p}.attn.attn_sink"))?,
                wo_a: pair(format!("{p}.attn.wo_a"))?,
                wo_b: pair(format!("{p}.attn.wo_b"))?,
                gate: ptr(format!("{p}.ffn.gate.weight"))?,
                bias: ptr(format!("{p}.ffn.gate.bias"))?,
                shared: [
                    pair(format!("{p}.ffn.shared_experts.w1"))?,
                    pair(format!("{p}.ffn.shared_experts.w3"))?,
                    pair(format!("{p}.ffn.shared_experts.w2"))?,
                ],
                table,
                ring_codes: alloc(own_rows * latent)?,
                ring_scales: alloc(own_rows * latent / 32)?,
            });
        }

        // main_proj is stored row-major (it is replicated, not a B12x shard): tile it once here.
        let (proj_codes, proj_scales) = pair("mtp.0.main_proj".to_owned())?;
        let inner = TAPS * dim;
        let mut row_major = vec![0u8; dim * inner];
        cuda.copy_dtoh(&mut row_major, &Deviceptr(proj_codes))?;
        let tiled = sparkflash_device::arena::tile_b12x(&row_major, dim, inner)
            .map_err(|error| format!("draft: main_proj tiling: {error}"))?;
        drop(row_major);
        let main_proj_tiled = alloc(tiled.len())?;
        cuda.copy_htod(&Deviceptr(main_proj_tiled), &tiled)?;

        let vocab = crate::tp4_tail::VOCAB_PER_RANK;
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        let markov_head = ptr("mtp.2.markov_head.head.weight".to_owned())?
            + (device.tp_rank * vocab * MARKOV_RANK * 2) as u64;
        let head_fp8 = if std::env::var("SPARKFLASH_DSPARK_HEAD_FP8").as_deref() == Ok("0") {
            None
        } else {
            Some((alloc(vocab * dim)?, alloc(vocab * dim / 32)?))
        };
        let r = R_MAX;
        Ok(Self {
            kernels,
            stages,
            main_proj: (main_proj_tiled, proj_scales),
            main_norm: ptr("mtp.0.main_norm.weight".to_owned())?,
            head_norm: ptr("mtp.2.norm.weight".to_owned())?,
            markov_head,
            markov_embed: ptr("mtp.2.markov_head.embed.weight".to_owned())?,
            taps: alloc(r * TAPS * dim * 4)?,
            main_x: alloc(r * dim * 4)?,
            main_codes: alloc(r * dim)?,
            main_scales: alloc(r * dim / 32)?,
            h_codes: alloc(r * TOP_K * LOCAL_INTER)?,
            h_scales: alloc(r * TOP_K * LOCAL_INTER / 32)?,
            logits: alloc(BLOCK * vocab * 4)?,
            ids: alloc((BLOCK + 1) * 8)?,
            block_max: alloc(chunks * 4)?,
            block_arg: alloc(chunks * 4)?,
            block_sum: alloc(chunks * 4)?,
            value: alloc(4)?,
            index: alloc(4)?,
            ticket: alloc(4)?,
            noise: {
                let bytes: Vec<u8> = noise_row.iter().flat_map(|v| v.to_le_bytes()).collect();
                let noise = alloc(bytes.len())?;
                cuda.copy_htod(&Deviceptr(noise), &bytes)?;
                noise
            },
            noise_row,
            ring_rows,
            allocations,
            head_fp8,
            head_fp8_ready: Cell::new(false),
            gathers: Cell::new(0),
        })
    }

    /// The stage rings' rows (slot = position): the lanes lay theirs out with as many.
    pub(super) fn ring_rows(&self) -> usize {
        self.ring_rows
    }

    /// `[BLOCK + 1]` u32: the draft's ids, the anchor then the k drafts.
    pub(super) fn ids(&self) -> u64 {
        self.ids
    }

    /// The tap kernel and buffer the rows step's hook writes.
    pub(super) fn tap_sink(&self) -> (Kernel, u64) {
        (self.kernels.taps, self.taps)
    }

    /// Frees the drafter's buffers (the modules drop with the state).
    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

fn span(device: u64, len: usize) -> DeviceSpan {
    DeviceSpan::new(device, len)
}

impl DeviceLayer<'_> {
    /// One Fp8a/Bf16o GEMV over `tokens` rows of a B12x weight.
    #[allow(clippy::too_many_arguments)]
    fn draft_gemv(
        &self,
        kind: OptKind,
        a: u64,
        a_scales: u64,
        w: (u64, u64),
        out: u64,
        cols: usize,
        inner: usize,
        tokens: usize,
    ) -> Result<(), String> {
        launch_opt(
            self.cuda,
            kind,
            &OptArgs {
                a,
                a_scales,
                w: w.0,
                ws: w.1,
                out,
                cols: cols as u32,
                inner: inner as u32,
                tokens: tokens as u32,
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
    }

    /// The plain rope table pair (the MTP stages are window-only, compress ratio 0).
    fn draft_rope(&self) -> (u64, u64) {
        let span = self.buffers.workspace.rope_tables_plain;
        let table_bytes = self.max_positions * self.rope_pairs as usize * 4;
        (span.device, span.device + table_bytes as u64)
    }

    /// Writes rows p0..p0+n's window KV into every stage's ring: main_x = main_norm(main_proj(taps))
    /// (model.py:1128-1135), then each stage's wkv, kv_norm, rope and window round trip at the
    /// rows' own positions (dspark_seed_ring).
    pub(super) fn draft_seed(
        &self,
        rows: &RowsState,
        draft: &DraftState,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        if n == 0 || n > rows.cap || p0 + n > self.max_positions {
            return Err(format!(
                "draft seed: rows {p0}..{} are outside the run",
                p0 + n
            ));
        }
        // The taps are the rows state's (a bulk state has its own), and a bulk chunk's main rows
        // use that state's per-layer scratch (free once the layers ran).
        let taps = rows.taps.as_ref().map_or(draft.taps, |(_, taps, _)| *taps);
        let (main_x, main_codes, main_scales) = if n > R_MAX {
            (b.normed, b.act_codes, b.act_scales)
        } else {
            (draft.main_x, draft.main_codes, draft.main_scales)
        };
        launch_opt(
            self.cuda,
            OptKind::F32aF32o,
            &OptArgs {
                a: taps,
                a_scales: 0,
                w: draft.main_proj.0,
                ws: draft.main_proj.1,
                out: main_x,
                cols: dim as u32,
                inner: (TAPS * dim) as u32,
                tokens: n as u32,
                a_tok_stride: (TAPS * dim) as u64,
                out_tok_stride: dim as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("draft main_proj: {error}"))?;
        self.cuda.launch(
            draft.kernels.norm_quant,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                main_x,
                draft.main_norm,
                u64::from(self.eps.to_bits()),
                main_codes,
                main_scales,
            ],
        )?;
        let (cos, sin) = self.draft_rope();
        let latent = self.head_width as usize;
        for (index, stage) in draft.stages.iter().enumerate() {
            self.draft_gemv(
                OptKind::Fp8aBf16o,
                main_codes,
                main_scales,
                stage.wkv,
                b.kv_latent,
                latent,
                dim,
                n,
            )
            .map_err(|error| format!("draft seed wkv: {error}"))?;
            for seg in self.step_segs(p0, n) {
                let ring = self.seg_draft_ring(stage, index, &seg);
                self.draft_kv(rows, draft, stage, ring, cos, sin, seg.row0, seg.p0, seg.n)?;
            }
        }
        Ok(())
    }

    /// Stage `index`'s ring (codes, scales): the selected lane's when the run has lanes, else the
    /// drafter's own.
    fn stage_ring(&self, stage: &Stage, index: usize) -> (u64, u64) {
        self.draft_rings
            .and_then(|rings| rings.get(index).copied())
            .unwrap_or((stage.ring_codes, stage.ring_scales))
    }

    /// Stage `index`'s ring for the rows of `seg`: its lane's, or [Self::stage_ring] outside a
    /// multi-lane step.
    fn seg_draft_ring(&self, stage: &Stage, index: usize, seg: &super::lanes::Seg) -> (u64, u64) {
        match self.lanes.as_ref() {
            Some(lanes) if seg.lane != super::lanes::NO_LANE => {
                lanes.pointers(seg.lane).draft[index]
            }
            _ => self.stage_ring(stage, index),
        }
    }

    /// kv_norm, rope at p0.., the window round trip into `ring` (slot = position), over the step
    /// rows row0..row0+n.
    #[allow(clippy::too_many_arguments)]
    fn draft_kv(
        &self,
        rows: &RowsState,
        draft: &DraftState,
        stage: &Stage,
        (ring_codes, ring_scales): (u64, u64),
        cos: u64,
        sin: u64,
        row0: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let row = (row0 * self.head_width as usize * 2) as u64;
        self.cuda.launch(
            rows.kernels.kv,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                b.kv_latent + row,
                stage.kv_norm,
                u64::from(self.eps.to_bits()),
                cos,
                sin,
                b.kv_rotated + row,
                ring_codes,
                ring_scales,
                0,
                1,
                draft.ring_rows as u64,
                u64::from(self.head_width),
                u64::from(self.rope_head_dim),
                p0 as u64,
            ],
        )
    }

    /// One draft from the anchor `anchor` at position `d` (the row after the last seeded one):
    /// returns ids[1..=k], the drafts for positions d+1..=d+k, and the draft's softmax probability
    /// of each (1.0 on the host-gather path, which does not compute them). Under a sampling
    /// `sampler` each draft is a sample under the noise the verify head uses at its index.
    /// With `verify` rows (on the device path) the stream also embeds the verify's first `verify`
    /// rows, [anchor, drafts], right after the chain and before the host reads the drafts, so
    /// that exchange runs while the ranks are still in step rather than after the host's planning
    /// gap; the caller begins those rows with [Self::begin_rows_queued].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_block(
        &self,
        rows: &RowsState,
        draft: &DraftState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        d: usize,
        anchor: u32,
        anchor_row: AnchorRow<'_>,
        k: usize,
        verify: usize,
        sampler: &super::sample::Sampler,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let b = rows.b;
        let dim = self.dim;
        let hc = self.hc_mult;
        let n = BLOCK;
        if d + n > self.max_positions || k == 0 || k > BLOCK {
            return Err(format!(
                "draft: block at {d} with {k} drafts is outside the run"
            ));
        }
        let verify = match anchor_row {
            AnchorRow::Device { embed, .. } if verify > 0 => Some((embed, verify)),
            _ => None,
        };
        // The block's streams: [anchor, noise x 4], each copied over the hc axis; identity pre mix.
        match anchor_row {
            AnchorRow::Device { embed, exchange } => {
                self.cuda.launch(
                    rows.kernels.store,
                    (1, 1, 1),
                    (32, 1, 1),
                    &[draft.ids, u64::from(anchor)],
                )?;
                self.embed_rows_on_device(rows, embed, draft.ids, 1, n, draft.noise, exchange)?;
            }
            AnchorRow::Host(anchor_row) => {
                let mut stream = Vec::with_capacity(n * hc * dim * 4);
                let mut seed = Vec::with_capacity(n * hc * 4);
                for row in 0..n {
                    let source = if row == 0 {
                        anchor_row
                    } else {
                        &draft.noise_row
                    };
                    for _ in 0..hc {
                        for value in source {
                            stream.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                    for copy in 0..hc {
                        seed.extend_from_slice(
                            &(if copy == 0 { 1.0f32 } else { 0.0 }).to_le_bytes(),
                        );
                    }
                }
                self.cuda.copy_htod(&Deviceptr(b.stream), &stream)?;
                self.cuda.copy_htod(&Deviceptr(b.pre_mix), &seed)?;
            }
        }
        let (cos, sin) = self.draft_rope();
        let lora = rows_lora(self);
        let latent = self.head_width as usize;
        let q_cols = self.heads_per_rank as usize * latent;
        for (index, stage) in draft.stages.iter().enumerate() {
            self.rows_seam(
                rows,
                n,
                b.stream,
                b.pre_mix,
                hc as u32,
                stage.attn_seam,
                b.attn_mix,
                0,
                b.hc_rows,
                b.normed,
                b.act_codes,
                b.act_scales,
                0,
                1,
            )?;
            self.draft_gemv(
                OptKind::Fp8aBf16o,
                b.act_codes,
                b.act_scales,
                stage.wq_a,
                b.q_lora,
                lora,
                dim,
                n,
            )
            .map_err(|error| format!("draft wq_a: {error}"))?;
            self.draft_gemv(
                OptKind::Fp8aBf16o,
                b.act_codes,
                b.act_scales,
                stage.wkv,
                b.kv_latent,
                latent,
                dim,
                n,
            )
            .map_err(|error| format!("draft wkv: {error}"))?;
            self.row_norm(
                span(b.q_lora, n * lora * 2),
                span(stage.q_norm, lora * 2),
                span(b.q_normed, n * lora * 2),
                n as u32,
                lora as u32,
                0,
                d as u64,
            )?;
            rows.quantise_bf16(self.cuda, b.q_normed, b.q_codes, b.q_scales, lora, n)?;
            self.draft_gemv(
                OptKind::Fp8aBf16o,
                b.q_codes,
                b.q_scales,
                stage.wq_b,
                b.q,
                q_cols,
                lora,
                n,
            )
            .map_err(|error| format!("draft wq_b: {error}"))?;
            let pair_bytes = (self.rope_pairs as usize * 4 * d) as u64;
            let (_, rope) = self.opt_rope.ok_or("draft: the opt rope is not loaded")?;
            self.cuda.launch(
                rope,
                (self.heads_per_rank, n as u32, 1),
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
            let ring = self.stage_ring(stage, index);
            self.draft_kv(rows, draft, stage, ring, cos, sin, 0, d, n)?;
            let (ring_codes, ring_scales) = self.stage_ring(stage, index);
            self.cuda.launch(
                draft.kernels.attn,
                (self.heads_per_rank, n as u32, 1),
                (256, 1, 1),
                &[
                    b.q,
                    ring_codes,
                    ring_scales,
                    stage.sink,
                    b.attn_body,
                    d as u64,
                ],
            )?;
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
                    d as u64,
                ],
            )?;
            self.draft_output(rows, stage, n)?;
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
                24,
                stage.ffn_seam,
                b.ffn_mix,
                b.pre_mix,
                0,
                b.seam_row,
                b.ffn_codes,
                b.ffn_scales,
                b.ffn_x,
                0,
            )?;
            self.draft_moe(rows, draft, stage, n)?;
            self.reduce_rows(
                "all_reduce_moe",
                span(b.delta_bf16, n * dim * 2),
                span(b.moe_reduced, n * dim * 2),
                n,
            )?;
            self.rows_post(rows, n, b.stream, b.ffn_mix, b.moe_reduced, b.stream)?;
        }
        self.draft_head(
            rows,
            draft,
            tail,
            transport,
            anchor,
            n,
            k,
            verify,
            (sampler, d + 1),
        )
    }

    /// wo_a over both groups, the bf16 quantiser, wo_b.
    fn draft_output(&self, rows: &RowsState, stage: &Stage, n: usize) -> Result<(), String> {
        let b = rows.b;
        let geometry = self.out_geometry;
        let groups = geometry.groups as u64;
        let (cols, inner) = (geometry.o_lora_rank as u64, geometry.group_inner as u64);
        launch_opt(
            self.cuda,
            OptKind::Bf16aBf16o,
            &OptArgs {
                a: b.attn_body,
                a_scales: 0,
                w: stage.wo_a.0,
                ws: stage.wo_a.1,
                out: b.attn_lo,
                cols: cols as u32,
                inner: inner as u32,
                tokens: n as u32,
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
        .map_err(|error| format!("draft wo_a: {error}"))?;
        let lo = (cols * groups) as usize;
        rows.quantise_bf16(self.cuda, b.attn_lo, b.o_codes, b.o_scales, lo, n)?;
        self.draft_gemv(
            OptKind::Fp8aBf16o,
            b.o_codes,
            b.o_scales,
            stage.wo_b,
            b.attn_out,
            geometry.hidden,
            lo,
            n,
        )
        .map_err(|error| format!("draft wo_b: {error}"))
    }

    /// The top-3-of-128 router, the stage's rank-split shared expert and the fused routed experts.
    fn draft_moe(
        &self,
        rows: &RowsState,
        draft: &DraftState,
        stage: &Stage,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let dim = self.dim;
        self.cuda.launch(
            draft.kernels.router,
            ((EXPERTS / 8) as u32, 1, 1),
            (256, 1, 1),
            &[
                b.ffn_x,
                stage.gate,
                stage.bias,
                b.moe_ids,
                b.moe_weights,
                b.scores,
                n as u64,
                draft.ticket,
            ],
        )?;
        let local = rows.shared_local;
        let [(w1, s1), (w3, s3), (w2, s2)] = stage.shared;
        launch_opt(
            self.cuda,
            OptKind::F32aF32o,
            &OptArgs {
                a: b.seam_row,
                a_scales: 0,
                w: w1,
                ws: s1,
                out: b.shared_gate,
                cols: local as u32,
                inner: dim as u32,
                tokens: n as u32,
                a_tok_stride: dim as u64,
                out_tok_stride: local as u64,
                groups: 2,
                a_grp_stride: 0,
                w_grp_stride: w3.wrapping_sub(w1),
                s_grp_stride: s3.wrapping_sub(s1),
                out_grp_stride: (n * local) as u64,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("draft shared gate/up: {error}"))?;
        rows.swiglu(
            self.cuda,
            b.shared_gate,
            b.shared_gate + (n * local * 4) as u64,
            b.shared_swiglu,
            self.swiglu_limit,
            n * local,
        )
        .map_err(|error| format!("draft shared swiglu: {error}"))?;
        launch_opt(
            self.cuda,
            OptKind::F32aF32o,
            &OptArgs {
                a: b.shared_swiglu,
                a_scales: 0,
                w: w2,
                ws: s2,
                out: b.shared_out,
                cols: dim as u32,
                inner: local as u32,
                tokens: n as u32,
                a_tok_stride: local as u64,
                out_tok_stride: dim as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| format!("draft shared down: {error}"))?;
        let matrix = (EXPERTS * 2 * 8) as u64;
        self.cuda.launch(
            draft.kernels.gate_up,
            ((LOCAL_INTER / 32) as u32, (n * TOP_K) as u32, 1),
            (128, 1, 1),
            &[
                stage.table,
                stage.table + matrix,
                b.moe_ids,
                b.moe_weights,
                b.ffn_codes,
                b.ffn_scales,
                draft.h_codes,
                draft.h_scales,
                (n * TOP_K) as u64,
                EXPERTS as u64,
                u64::from(self.swiglu_limit.to_bits()),
            ],
        )?;
        self.cuda.launch(
            draft.kernels.down,
            ((dim / 32) as u32, n as u32, 1),
            (256, 1, 1),
            &[
                stage.table + 2 * matrix,
                b.moe_ids,
                draft.h_codes,
                draft.h_scales,
                b.delta_f32,
                b.delta_bf16,
                n as u64,
                EXPERTS as u64,
                b.shared_out,
            ],
        )
    }

    /// forward_head (model.py:1137-1156): hc_pre with the last stage's FFN pre mix, mtp.2.norm, the
    /// shared head over the rank's vocabulary slice, then the Markov steps: draft i is the argmax of
    /// row i's logits plus the bias of ids[i], gathered across the ranks. Under a sampling sampler
    /// step i's value carries the noise at sequence index `index0 + i`. `verify` (embed, rows):
    /// see [Self::draft_block].
    #[allow(clippy::too_many_arguments)]
    fn draft_head(
        &self,
        rows: &RowsState,
        draft: &DraftState,
        tail: &crate::tp4_tail::TailStage,
        transport: &dyn sparkflash_comm::transport::Transport,
        anchor: u32,
        n: usize,
        k: usize,
        verify: Option<(u64, usize)>,
        (sampler, index0): (&super::sample::Sampler, usize),
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let b = rows.b;
        let dim = self.dim;
        let kernel = self
            .seam_hc_pre_collapse
            .ok_or("draft head: hc_pre_collapse_bf16 is not loaded")?;
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
            weight: Deviceptr(draft.head_norm),
            out: Deviceptr(b.lm_normed),
            rows: n as u32,
            cols: dim as u32,
            eps: tail.eps,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        sparkflash_kernels::rmsnorm_launch::launch(self.cuda, self.row_norm.1, &norm)?;
        let vocab = crate::tp4_tail::VOCAB_PER_RANK;
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        if let Some((codes, scales)) = draft.head_fp8 {
            if !draft.head_fp8_ready.get() {
                self.cuda.launch(
                    draft.kernels.head_quant,
                    (vocab as u32, 1, 1),
                    (256, 1, 1),
                    &[tail.head_weight.0, codes, scales],
                )?;
                draft.head_fp8_ready.set(true);
            }
            self.cuda.launch(
                draft.kernels.logits_fp8,
                (LOGITS_BLOCKS, 1, 1),
                (512, 1, 1),
                &[
                    b.lm_normed,
                    codes,
                    scales,
                    draft.logits,
                    vocab as u64,
                    n as u64,
                ],
            )?;
        } else {
            self.cuda.launch(
                draft.kernels.logits,
                (LOGITS_BLOCKS, 1, 1),
                (512, 1, 1),
                &[
                    b.lm_normed,
                    tail.head_weight.0,
                    draft.logits,
                    vocab as u64,
                    n as u64,
                ],
            )?;
        }
        let (inv_t, seed) = sampler.draft_noise();
        if let Some((world, rank)) = self.device_exchange() {
            // The whole Markov chain stays on the stream: each step's pick writes ids[step + 1],
            // which the next step's bias reads, and its probability; the host reads the k drafts
            // and their probabilities once at the end.
            self.cuda.launch(
                rows.kernels.store,
                (1, 1, 1),
                (32, 1, 1),
                &[draft.ids, u64::from(anchor)],
            )?;
            let probs = draft.ids + ((BLOCK + 1) * 4) as u64;
            // [value, index, mass] per rank: 6 bf16 of the exchange.
            let gather = rows.b.argmax_gather;
            for step in 0..k {
                self.cuda.launch(
                    draft.kernels.markov,
                    (chunks as u32, 1, 1),
                    (512, 1, 1),
                    &[
                        draft.logits,
                        draft.ids,
                        step as u64,
                        draft.markov_head,
                        draft.markov_embed,
                        draft.block_max,
                        draft.block_arg,
                        vocab as u64,
                        draft.block_sum,
                        inv_t,
                        seed,
                        index0 as u64,
                        (rank * vocab) as u64,
                    ],
                )?;
                self.cuda.launch(
                    draft.kernels.fold_prob,
                    (1, 1, 1),
                    (32, 1, 1),
                    &[
                        draft.block_max,
                        draft.block_arg,
                        draft.block_sum,
                        chunks as u64,
                        gather + (rank * 12) as u64,
                    ],
                )?;
                self.exchange_slots(gather, 6, rank, "draft argmax gather")?;
                self.cuda.launch(
                    draft.kernels.pick_prob,
                    (1, 1, 1),
                    (32, 1, 1),
                    &[
                        gather,
                        world as u64,
                        vocab as u64,
                        draft.ids + ((step + 1) * 4) as u64,
                        probs + ((step + 1) * 4) as u64,
                    ],
                )?;
            }
            if let Some((embed, verify)) = verify {
                self.embed_rows_on_device(
                    rows,
                    embed,
                    draft.ids,
                    verify,
                    verify,
                    0,
                    (world, rank),
                )?;
            }
            let mut words = vec![0u8; (BLOCK + 1) * 8];
            self.cuda.copy_dtoh(&mut words, &Deviceptr(draft.ids))?;
            let word = |i: usize| {
                u32::from_le_bytes([
                    words[4 * i],
                    words[4 * i + 1],
                    words[4 * i + 2],
                    words[4 * i + 3],
                ])
            };
            return Ok((
                (1..=k).map(word).collect(),
                (1..=k)
                    .map(|i| f32::from_bits(word(BLOCK + 1 + i)))
                    .collect(),
            ));
        }
        if !sampler.is_greedy() {
            return Err(
                "draft head: sampling needs the device exchange (SPARKFLASH_DEVICE_GATHER)"
                    .to_owned(),
            );
        }
        let mut ids = vec![anchor];
        let world = transport.world_size();
        for step in 0..k {
            let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            self.cuda.copy_htod(&Deviceptr(draft.ids), &words)?;
            self.cuda.launch(
                draft.kernels.markov,
                (chunks as u32, 1, 1),
                (512, 1, 1),
                &[
                    draft.logits,
                    draft.ids,
                    step as u64,
                    draft.markov_head,
                    draft.markov_embed,
                    draft.block_max,
                    draft.block_arg,
                    vocab as u64,
                    draft.block_sum,
                    0,
                    0,
                    0,
                    0,
                ],
            )?;
            self.cuda.launch(
                rows.kernels.fold,
                (1, 1, 1),
                (32, 1, 1),
                &[
                    draft.block_max,
                    draft.block_arg,
                    chunks as u64,
                    draft.value,
                    draft.index,
                ],
            )?;
            let mut value = [0u8; 4];
            let mut index = [0u8; 4];
            self.cuda.copy_dtoh(&mut value, &Deviceptr(draft.value))?;
            self.cuda.copy_dtoh(&mut index, &Deviceptr(draft.index))?;
            let bits = f32::from_le_bytes(value).to_bits();
            let index = u32::from_le_bytes(index);
            let local = [
                (bits >> 16) as u16,
                bits as u16,
                (index >> 16) as u16,
                index as u16,
            ];
            let mut gathered = vec![0u16; world * 4];
            let call = draft.gathers.get();
            draft.gathers.set(call.wrapping_add(1));
            transport
                .all_gather_bf16(
                    &local,
                    &mut gathered,
                    DRAFT_GATHER_SEQ.wrapping_sub(call % 0x1_0000),
                )
                .map_err(|error| format!("draft argmax all-gather: {error}"))?;
            let mut winner = (f32::NEG_INFINITY, 0u32, false);
            for rank in 0..world {
                let w = &gathered[rank * 4..rank * 4 + 4];
                let value = f32::from_bits((u32::from(w[0]) << 16) | u32::from(w[1]));
                let index = ((u32::from(w[2]) << 16) | u32::from(w[3])) + (rank * vocab) as u32;
                if !winner.2 || value > winner.0 {
                    winner = (value, index, true);
                }
            }
            ids.push(winner.1);
        }
        Ok((ids[1..].to_vec(), vec![1.0; k]))
    }
}

/// The q_lora width of the layer's projections.
fn rows_lora(device: &DeviceLayer<'_>) -> usize {
    device.projections[0].cols as usize
}
