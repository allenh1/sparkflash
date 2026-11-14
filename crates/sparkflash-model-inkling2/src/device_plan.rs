// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small device schedule: every launch of one pass, in order, at one rank.
//!
//! The f32 host reference ([crate::forward]) decides *what* the model computes. This module
//! decides *in what order the device computes it* and *which image and entry* each launch uses,
//! and it does so on a CPU host so the schedule can be checked without a GPU. The executor in
//! `device` walks these steps; it owns no ordering of its own.
//!
//! The schedule is derived from two typed inputs, never from a family name:
//!
//! * [Inkling2Config] and a [TpPlan], which give every rank-local width;
//! * [InklingTensorPlan::served], which gives every weight its rank-local shape and its index in
//!   the converted rank file.
//!
//! Three properties are checked by the tests and are the point of the module:
//!
//! 1. **Every served weight is read exactly once.** A tensor the converter wrote and the schedule
//!    never names is a weight paid for and not used; a tensor named twice is a step that will
//!    double-count a residual. The tiling test in `convert` proves four rank files hold the
//!    checkpoint; this proves one pass reads each of those slots.
//! 2. **Every named kernel exists.** A step that names a [Part] must resolve to a catalogue entry
//!    for the same image and symbol; a step with no [Part] is arm-private glue and must name a
//!    symbol the Inkling image exports.
//! 3. **The buffers and the state are accounted for.** The KV rows, the four convolution
//!    histories a layer, the block-summary stack and the expert band are all sized here from the
//!    admitted context, so the memory claim is a computation rather than a restatement.
//!
//! # The passes
//!
//! [Pass::Decode] is one token: every projection is an M=1 GEMV and the attention walks the cache.
//! [Pass::Prefill] is a chunk of rows: the projections take the tensor-core tiled entry, the
//! attention reads a prefix of cache rows, and the short convolution reads its taps across rows.
//! Both schedules have the same steps in the same order; only the named entries differ, which is
//! what lets one walk serve both.
//!
//! # Tensor-parallel geometry
//!
//! At TP4 a rank holds 12 query heads of 128, 2 key/value heads, 96 of the 384 routed experts
//! whole, and one quarter of the vocabulary. Routed experts are **expert-parallel**: the router
//! is replicated, so every rank selects the same global top-8 and then computes only the experts
//! it owns. Each rank therefore produces a partial sum of the same eight weighted expert outputs,
//! and one all-reduce over the residual width completes it. The schedule records that as
//! [Collective::ExpertDelta], which is why the expert band is followed by a reduce and then the
//! residual add rather than by a local combine.

use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::{Part, TpPlan};
use sparkflash_ops::images::Image;

use crate::config::Inkling2Config;
use crate::tensor::InklingTensorPlan;

/// The rank-local vocabulary alignment the released vocabulary has.
///
/// 201,024 = 64 * 3,141: the vocabulary is a multiple of 64 and of nothing coarser. The released
/// arms' 128-row rule therefore does not apply, and the plan uses the exact width.
pub const VOCAB_ALIGNMENT: usize = 64;

/// Which pass a schedule describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// One token, cache-resident attention, M=1 projections.
    Decode,
    /// A chunk of rows against a cache prefix.
    Prefill,
}

impl Pass {
    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

/// One collective the pass performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective {
    /// The last prompt token's embedding row, gathered from the rank that owns its vocabulary
    /// row. The embedding is vocabulary-parallel, so exactly one rank holds the row.
    EmbeddingRow,
    /// The routed experts' partial hidden deltas, summed across ranks. Expert parallelism leaves
    /// each rank holding a subset of the selected experts, and the mix is a sum, so an all-reduce
    /// is the whole exchange.
    ExpertDelta,
    /// The logit slices, gathered so the sampler sees the released 200,058 entries. The head is
    /// vocabulary-parallel; the softcap is elementwise and is applied before the gather.
    LogitSlice,
}

impl Collective {
    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmbeddingRow => "embedding_row",
            Self::ExpertDelta => "expert_delta",
            Self::LogitSlice => "logit_slice",
        }
    }
}

/// What a scheduled launch does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Widen one embedding row (or a chunk's rows) to f32 hidden.
    Embed,
    /// An RMSNorm with a BF16 gain.
    RmsNorm,
    /// The block-attention residual: a softmax mix over the block-summary stack.
    BlockResidual,
    /// A dense projection over the rank-local weight.
    Gemv,
    /// A static depthwise short convolution with decode history.
    ShortConv,
    /// A dynamic short convolution: the static taps plus a per-token projection.
    ShortConvDynamic,
    /// The relative-attention body.
    Attention,
    /// Write the f32 key/value projections into the BF16 cache at one position.
    KvStore,
    /// The sigmoid router: scores, the learned bias, top-k, route scale.
    Router,
    /// One expert's gate/up projection.
    ExpertGateUp,
    /// The softcapped SwiGLU over the expert band.
    ExpertActivation,
    /// One expert's down projection.
    ExpertDown,
    /// Sum the selected experts' weighted partials into the layer's MLP output. Every expert body
    /// folds the router weight into its own output - NVFP4 through its per-expert scale_2, MXFP8
    /// through the scale argument - so the mix is a plain sum over the band's slots.
    ExpertMix,
    /// The elementwise residual add, in place.
    ResidualAdd,
    /// The final logit softcap and the unpadded slice.
    LogitSoftcap,
    /// A collective.
    Collective(Collective),
}

impl Op {
    /// The stable name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::RmsNorm => "rms_norm",
            Self::BlockResidual => "block_residual",
            Self::Gemv => "gemv",
            Self::ShortConv => "short_conv",
            Self::ShortConvDynamic => "short_conv_dynamic",
            Self::Attention => "attention",
            Self::KvStore => "kv_store",
            Self::Router => "router",
            Self::ExpertGateUp => "expert_gate_up",
            Self::ExpertActivation => "expert_activation",
            Self::ExpertDown => "expert_down",
            Self::ExpertMix => "expert_mix",
            Self::ResidualAdd => "residual_add",
            Self::LogitSoftcap => "logit_softcap",
            Self::Collective(collective) => collective.name(),
        }
    }
}

/// The layout of a short convolution's static taps, which the checkpoint stores two ways.
///
/// The sliding layers store `[channels, 1, kernel]`, contiguous as `[channels][kernel]`, and the
/// full layers store `[kernel, channels]`. The two decode entries read those two layouts
/// respectively, so the uploader must not transpose either: a transposed tap table is in bounds,
/// the same size and wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapLayout {
    /// `taps[channel * kernel + tap]`: the static depthwise form.
    ChannelMajor,
    /// `taps[tap * channels + channel]`: the dynamic form's static half.
    TapMajor,
}

/// Which learned log-scaling factor a layer's attention uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TauRule {
    /// The sliding layers use no scaling.
    One,
    /// The full layers scale by `1 + alpha * ln(max(position + 1, n_floor) / n_floor)`.
    LogScaling {
        /// The released `log_scaling_n_floor`.
        n_floor: usize,
        /// The released `log_scaling_alpha`.
        alpha: f64,
    },
}

impl TauRule {
    /// The factor the kernel is given at `position`, or `1.0` when the layer does not scale.
    ///
    /// The device passes this as the `tau` argument; the host's [crate::reference::attention]
    /// computes the same expression, which is what makes the two agree on a full layer deep into
    /// a context.
    #[must_use]
    pub fn at(self, position: usize) -> f32 {
        match self {
            Self::One => 1.0,
            Self::LogScaling { n_floor, alpha } => {
                if n_floor == 0 {
                    return 1.0;
                }
                let ratio = ((position + 1) as f64 / n_floor as f64).max(1.0);
                (1.0 + alpha * ratio.ln()) as f32
            }
        }
    }
}

/// One scheduled launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// What the launch does.
    pub op: Op,
    /// The shared part it implements, when the vocabulary names one. Arm-private glue (the
    /// residual add and the cache write) has none: inventing a part for them would claim a
    /// catalogue entry this tree does not carry.
    pub part: Option<Part>,
    /// The image that carries the entry.
    pub image: Option<Image>,
    /// The entry the executor loads by name.
    pub entry: Option<&'static str>,
    /// Served-plan indices the step reads, in launch order.
    pub weights: Vec<usize>,
    /// The pass's row count for this step's outputs (1 in decode, the chunk in prefill).
    pub rows: usize,
    /// The rank-local output width (or reduction width, for a reduction step).
    pub columns: usize,
    /// The static-tap layout, on a convolution step.
    pub taps: Option<TapLayout>,
}

impl Step {
    /// Whether the step reads any weight.
    #[must_use]
    pub fn reads_weights(&self) -> bool {
        !self.weights.is_empty()
    }
}

/// One decoder layer's schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSchedule {
    /// The checkpoint layer index.
    pub index: usize,
    /// Whether the layer is a sliding-window layer.
    pub sliding: bool,
    /// How many learned queries the block-attention residual reads (1 or 3).
    pub block_queries: usize,
    /// The convolution's static-tap layout.
    pub taps: TapLayout,
    /// The attention's log-scaling rule.
    pub tau: TauRule,
    /// The steps, in order.
    pub steps: Vec<Step>,
}

/// The bytes one session needs beside the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferPlan {
    /// `[rows][hidden]` f32: the residual stream.
    pub hidden: usize,
    /// `[rows][3][hidden]` f32: the block-residual sublayer inputs (three queries at a time).
    pub block_inputs: usize,
    /// `[rows][heads][head_dim]` f32: the query projection.
    pub query: usize,
    /// `[rows][heads][head_dim]` f32: the output-gate projection.
    pub gate: usize,
    /// `[rows][heads][d_rel]` f32: the relative-profile projection.
    pub relative: usize,
    /// `[heads]` f32: the attention sink.
    pub sink: usize,
    /// `[rows][kv_width]` f32: the key and value projections before the cache write.
    pub kv: usize,
    /// `[rows][hidden]` f32: a sublayer output before its residual add.
    pub delta: usize,
    /// `[top_k][2 * intermediate]` f32: the expert band.
    pub expert_band: usize,
    /// `[rows][intermediate]` f32: one expert's activation.
    pub expert_activation: usize,
    /// `[rows][rank_vocab]` f32: the rank's logit slice.
    pub logits: usize,
    /// Four `[channels][kernel - 1]` f32 histories a layer, all layers.
    pub conv_state: usize,
    /// The key and value caches, all layers, BF16.
    pub kv_cache: usize,
    /// The block-summary stack, `block_count * hidden` f32.
    pub block_summaries: usize,
    /// Every term above, summed.
    pub total: usize,
}

/// The device schedule of one rank.
#[derive(Debug, Clone)]
pub struct InklingDevicePlan {
    config: Inkling2Config,
    tp: TpPlan,
    pass: Pass,
    rows: usize,
    context: usize,
    served: InklingTensorPlan,
    globals: Vec<Step>,
    layers: Vec<LayerSchedule>,
    buffers: BufferPlan,
}

impl InklingDevicePlan {
    /// Build the schedule of one rank and one pass.
    ///
    /// # Errors
    /// A world the model's heads, experts or vocabulary do not divide, a zero row count, or a
    /// context outside the model window.
    pub fn build(
        config: &Inkling2Config,
        tp: TpPlan,
        pass: Pass,
        rows: usize,
        context: usize,
    ) -> Result<Self, String> {
        if rows == 0 {
            return Err("a pass needs at least one row".to_owned());
        }
        if pass == Pass::Decode && rows != 1 {
            return Err(format!("a decode pass is one row, not {rows}"));
        }
        let window = config.context_tokens();
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for the released model"
            ));
        }
        let served = InklingTensorPlan::served(config, tp)?;
        let mut plan = Self {
            config: config.clone(),
            tp,
            pass,
            rows,
            context,
            served,
            globals: Vec::new(),
            layers: Vec::new(),
            buffers: BufferPlan {
                hidden: 0,
                block_inputs: 0,
                query: 0,
                gate: 0,
                relative: 0,
                sink: 0,
                kv: 0,
                delta: 0,
                expert_band: 0,
                expert_activation: 0,
                logits: 0,
                conv_state: 0,
                kv_cache: 0,
                block_summaries: 0,
                total: 0,
            },
        };
        plan.globals = plan.global_steps()?;
        for layer in 0..config.num_hidden_layers {
            plan.layers.push(plan.layer_schedule(layer)?);
        }
        plan.buffers = plan.buffer_plan();
        Ok(plan)
    }

    /// The typed configuration.
    #[must_use]
    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    /// The tensor-parallel plan.
    #[must_use]
    pub const fn tp(&self) -> TpPlan {
        self.tp
    }

    /// The pass this schedule describes.
    #[must_use]
    pub const fn pass(&self) -> Pass {
        self.pass
    }

    /// The rows one launch of this pass covers.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The admitted context.
    #[must_use]
    pub const fn context(&self) -> usize {
        self.context
    }

    /// The served tensor plan the step indices address.
    #[must_use]
    pub fn served(&self) -> &InklingTensorPlan {
        &self.served
    }

    /// The steps outside the decoder layers.
    #[must_use]
    pub fn globals(&self) -> &[Step] {
        &self.globals
    }

    /// The decoder layers in forward order.
    #[must_use]
    pub fn layers(&self) -> &[LayerSchedule] {
        &self.layers
    }

    /// Every step, globals first, then layers in forward order.
    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.globals
            .iter()
            .chain(self.layers.iter().flat_map(|layer| layer.steps.iter()))
    }

    /// The buffer and state accounting.
    #[must_use]
    pub const fn buffers(&self) -> &BufferPlan {
        &self.buffers
    }

    /// The number of launches one pass issues.
    #[must_use]
    pub fn launch_count(&self) -> usize {
        self.steps().count()
    }

    /// The rank-local vocabulary rows the head produces.
    #[must_use]
    pub fn rank_vocab(&self) -> usize {
        self.served
            .index_of("model.llm.unembed.weight")
            .map_or(0, |index| self.served.rank_spec_at(index).shape[0])
    }

    /// The rank-local expert count.
    #[must_use]
    pub fn rank_experts(&self) -> usize {
        self.served
            .index_of("model.llm.layers.0.mlp.experts.w2_weight")
            .map_or(0, |index| self.served.rank_spec_at(index).shape[0])
    }

    /// The rank-local query-head count.
    #[must_use]
    pub fn rank_heads(&self) -> usize {
        self.config.num_attention_heads / self.tp.world().max(1)
    }

    /// The rank-local key/value head count.
    #[must_use]
    pub fn rank_kv_heads(&self) -> usize {
        self.config.num_key_value_heads / self.tp.world().max(1)
    }

    /// A one-line boot summary: pass, rows, layers, launches and the state byte total.
    #[must_use]
    pub fn boot_summary(&self) -> String {
        format!(
            "inkling2 {} rows={} TP{}/{}: {} layers, {} launches, heads {}/{}, experts {}/{}, \
             vocab {}, state {:.1} MiB",
            self.pass.name(),
            self.rows,
            self.tp.rank(),
            self.tp.world(),
            self.layers.len(),
            self.launch_count(),
            self.rank_heads(),
            self.rank_kv_heads(),
            self.rank_experts(),
            self.config.n_routed_experts,
            self.rank_vocab(),
            self.buffers.total as f64 / (1024.0 * 1024.0),
        )
    }

    fn index(&self, name: &str) -> Result<usize, String> {
        self.served
            .index_of(name)
            .ok_or_else(|| format!("{name} is not a served tensor of this plan"))
    }

    /// A weight-reading step, with the entry the pass uses.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        op: Op,
        part: Option<Part>,
        image: Option<Image>,
        entry: Option<&'static str>,
        weights: &[&str],
        rows: usize,
        columns: usize,
    ) -> Result<Step, String> {
        let mut indices = Vec::with_capacity(weights.len());
        for name in weights {
            indices.push(self.index(name)?);
        }
        Ok(Step {
            op,
            part,
            image,
            entry,
            weights: indices,
            rows,
            columns,
            taps: None,
        })
    }

    /// The residual add: the arm's own seam kernel, with no shared part identity.
    fn add(rows: usize, columns: usize) -> Step {
        Step {
            op: Op::ResidualAdd,
            part: None,
            image: Some(Image::Inkling),
            entry: Some("inkling_residual_add"),
            weights: Vec::new(),
            rows,
            columns,
            taps: None,
        }
    }

    /// A launch with no weight: a collective or an arithmetic step.
    fn plain(op: Op, part: Option<Part>, rows: usize, columns: usize) -> Step {
        Step {
            op,
            part,
            image: None,
            entry: None,
            weights: Vec::new(),
            rows,
            columns,
            taps: None,
        }
    }

    /// A shared launch that reads no weight of this plan's: the combine over the expert band.
    fn shared(
        op: Op,
        part: Part,
        image: Image,
        entry: &'static str,
        rows: usize,
        columns: usize,
    ) -> Step {
        Step {
            op,
            part: Some(part),
            image: Some(image),
            entry: Some(entry),
            weights: Vec::new(),
            rows,
            columns,
            taps: None,
        }
    }

    /// The entries of a pass. The two passes differ only in the named kernel, never in the order.
    fn entry_for(&self, decode: &'static str, prefill: &'static str) -> &'static str {
        match self.pass {
            Pass::Decode => decode,
            Pass::Prefill => prefill,
        }
    }

    /// The dense BF16 projection of a pass: an M=1 GEMV in decode, a tiled GEMM in prefill.
    ///
    /// The decode body is the shared host-weight GEMV with f32 accumulators, which is the one
    /// the checkpoint's BF16 matrices want: a BF16 store would narrow the query and the relative
    /// profile before the L2 clip, and the clip is on the f32 vector. The prefill body is the
    /// token image's BF16 GEMM, which the Qwen arm already uses for the same matrices.
    fn gemv_launch(&self) -> (Image, &'static str) {
        match self.pass {
            Pass::Decode => (Image::Deepseek, "b12x_hostw_gemv_m1_bf16w_f32o_sm121"),
            Pass::Prefill => (Image::TOKEN, "q4e_gemm_bf16"),
        }
    }

    fn global_steps(&self) -> Result<Vec<Step>, String> {
        let config = &self.config;
        let hidden = config.hidden_size;
        let rows = self.rows;
        let mut steps = Vec::new();
        steps.push(self.step(
            Op::Embed,
            Some(Part::Embedding),
            Some(Image::TOKEN),
            Some("q4e_embed"),
            &["model.llm.embed.weight"],
            rows,
            hidden,
        )?);
        steps.push(Self::plain(
            Op::Collective(Collective::EmbeddingRow),
            Some(Part::Collective),
            rows,
            hidden,
        ));
        steps.push(self.step(
            Op::RmsNorm,
            Some(Part::RmsNorm),
            Some(Image::TOKEN),
            Some(self.entry_for("q4e_rmsnorm", "q4e_rmsnorm_rows")),
            &["model.llm.embed_norm.weight"],
            rows,
            hidden,
        )?);
        // The output query is the one learned query of the final block residual.
        steps.push(self.step(
            Op::BlockResidual,
            Some(Part::BlockAttentionResidual),
            Some(Image::Inkling),
            Some("inkling_block_residual"),
            &["model.llm.attn_res.output_query"],
            1,
            hidden,
        )?);
        steps.push(self.step(
            Op::RmsNorm,
            Some(Part::RmsNorm),
            Some(Image::TOKEN),
            Some(self.entry_for("q4e_rmsnorm", "q4e_rmsnorm_rows")),
            &["model.llm.norm.weight"],
            rows,
            hidden,
        )?);
        // The vocabulary projection is the shared dense GEMV: the released head is an untied
        // BF16 matrix and the shared body already serves an M=1 row, so the arm does not carry a
        // head kernel of its own. The LmHead part's own entries are fused reduces that Inkling
        // does not use, because the slice gather is the collective below.
        steps.push(self.step(
            Op::Gemv,
            Some(Part::DenseGemm),
            Some(self.gemv_launch().0),
            Some(self.gemv_launch().1),
            &["model.llm.unembed.weight"],
            self.rank_vocab(),
            hidden,
        )?);
        steps.push(self.step(
            Op::LogitSoftcap,
            Some(Part::SoftcappedActivation),
            Some(Image::Inkling),
            Some("inkling_logit_softcap"),
            &[],
            1,
            config.unpadded_vocab_size,
        )?);
        steps.push(Self::plain(
            Op::Collective(Collective::LogitSlice),
            Some(Part::Collective),
            1,
            config.unpadded_vocab_size,
        ));
        Ok(steps)
    }

    fn layer_schedule(&self, layer: usize) -> Result<LayerSchedule, String> {
        let config = &self.config;
        let hidden = config.hidden_size;
        let heads = self.rank_heads();
        let kv_heads = self.rank_kv_heads();
        let head_dim = config.head_dim;
        let kv_width = kv_heads * head_dim;
        let rows = self.rows;
        let prefix = format!("model.llm.layers.{layer}");
        let attn = format!("{prefix}.attn");
        let sliding = config.is_sliding(layer);
        let full = !sliding;
        let taps = if full {
            TapLayout::TapMajor
        } else {
            TapLayout::ChannelMajor
        };
        // The block-attention residual reads the bank addressed by the layer *before* this one;
        // layer 0 has no bank and consumes the raw embedding, which is the host's rule too.
        let bank_step = if layer > 0 {
            let bank = format!("model.llm.attn_res.attn_queries.{}", layer - 1);
            Some(self.step(
                Op::BlockResidual,
                Some(Part::BlockAttentionResidual),
                Some(Image::Inkling),
                Some("inkling_block_residual"),
                &[&bank],
                if full { 3 } else { 1 },
                hidden,
            )?)
        } else {
            None
        };
        // The largest layer is 28 launches, so the vector is sized from the shape rather than
        // grown, and the capacity is a statement about the schedule's shape.
        let mut steps = Vec::with_capacity(32);
        steps.extend(bank_step);

        // The attention input norm: one shared norm on a sliding layer, three on a full one.
        if sliding {
            steps.push(self.step(
                Op::RmsNorm,
                Some(Part::RmsNorm),
                Some(Image::TOKEN),
                Some(self.entry_for("q4e_rmsnorm", "q4e_rmsnorm_rows")),
                &[&format!("{prefix}.attn_norm.weight")],
                rows,
                hidden,
            )?);
        } else {
            for name in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                steps.push(self.step(
                    Op::RmsNorm,
                    Some(Part::RmsNorm),
                    Some(Image::TOKEN),
                    Some(self.entry_for("q4e_rmsnorm", "q4e_rmsnorm_rows")),
                    &[&format!("{prefix}.{name}.weight")],
                    rows,
                    hidden,
                )?);
            }
        }

        // The six attention projections, in the host's order.
        for (name, width) in [
            ("wq_du", heads * head_dim),
            ("wk_dv", kv_width),
            ("wv_dv", kv_width),
            ("wr_du", heads * config.d_rel),
            ("gate_attn", heads * head_dim),
            ("attn_sink_proj", heads),
        ] {
            steps.push(self.step(
                Op::Gemv,
                Some(Part::DenseGemm),
                Some(self.gemv_launch().0),
                Some(self.gemv_launch().1),
                &[&format!("{attn}.{name}.weight")],
                width,
                hidden,
            )?);
        }

        // The key and value convolutions. A full layer's form reads the static taps tap-major and
        // adds a per-token projection; a sliding layer's reads them channel-major.
        for name in ["k_sconv", "v_sconv"] {
            if full {
                steps.push(self.step(
                    Op::ShortConvDynamic,
                    Some(Part::ShortConv),
                    Some(Image::Inkling),
                    Some(self.entry_for(
                        "inkling_short_conv_dynamic_decode",
                        "inkling_short_conv_dynamic",
                    )),
                    &[
                        &format!("{attn}.{name}.static_w"),
                        &format!("{attn}.{name}.weight_proj"),
                    ],
                    kv_width,
                    config.sconv_kernel_size,
                )?);
            } else {
                steps.push(self.step(
                    Op::ShortConv,
                    Some(Part::ShortConv),
                    Some(Image::Inkling),
                    Some(self.entry_for("inkling_short_conv_decode", "inkling_short_conv")),
                    &[&format!("{attn}.{name}.weight")],
                    kv_width,
                    config.sconv_kernel_size,
                )?);
            }
        }

        // The two f32 projections become the BF16 cache rows the attention reads.
        steps.push(self.step(
            Op::KvStore,
            None,
            Some(Image::Inkling),
            Some("inkling_kv_store"),
            &[],
            rows,
            kv_width,
        )?);

        steps.push(self.step(
            Op::Attention,
            Some(Part::RelativeAttention),
            Some(Image::Inkling),
            Some(self.entry_for("inkling_attn_decode", "inkling_attn_prefill")),
            &[
                &format!("{attn}.rel_logits_proj.proj"),
                &format!("{attn}.q_l2_clip.scale"),
                &format!("{attn}.rel_l2_clip.scale"),
            ],
            rows,
            heads * head_dim,
        )?);

        // The attention output projection and its short convolution.
        steps.push(self.step(
            Op::Gemv,
            Some(Part::DenseGemm),
            Some(self.gemv_launch().0),
            Some(self.gemv_launch().1),
            &[&format!("{attn}.wo_ud.weight")],
            hidden,
            heads * head_dim,
        )?);
        steps.push(self.step(
            Op::ShortConv,
            Some(Part::ShortConv),
            Some(Image::Inkling),
            Some(self.entry_for("inkling_short_conv_decode", "inkling_short_conv")),
            &[&format!("{prefix}.attn_sconv.weight")],
            hidden,
            config.sconv_kernel_size,
        )?);

        // The residual. At a block boundary the input *is* the sublayer's output and the summary
        // stack grows instead: layer % n_layers_per_attn_res_block == 0 returns the attention
        // output alone and appends the pre-attention hidden state as a new summary.
        let boundary = layer.is_multiple_of(config.n_layers_per_attn_res_block.max(1));
        if !boundary {
            steps.push(Self::add(rows, hidden));
        }

        // The MLP input is a second block-residual read, one learned query a layer.
        steps.push(self.step(
            Op::BlockResidual,
            Some(Part::BlockAttentionResidual),
            Some(Image::Inkling),
            Some("inkling_block_residual"),
            &[&format!("model.llm.attn_res.mlp_queries.{layer}")],
            1,
            hidden,
        )?);
        steps.push(self.step(
            Op::RmsNorm,
            Some(Part::RmsNorm),
            Some(Image::TOKEN),
            Some(self.entry_for("q4e_rmsnorm", "q4e_rmsnorm_rows")),
            &[&format!("{prefix}.mlp_norm.weight")],
            rows,
            hidden,
        )?);

        // The router, then the routed band.
        steps.push(self.step(
            Op::Gemv,
            Some(Part::DenseGemm),
            Some(self.gemv_launch().0),
            Some(self.gemv_launch().1),
            &[&format!("{prefix}.mlp.gate.weight")],
            config.n_routed_experts,
            hidden,
        )?);
        steps.push(self.step(
            Op::Router,
            Some(Part::Moe),
            Some(Image::Inkling),
            Some("inkling_router_topk"),
            &[&format!("{prefix}.mlp.gate.bias")],
            1,
            config.num_experts_per_tok,
        )?);

        // One expert's three launches, repeated over the local experts the router selected. The
        // band is one set of steps, not one per expert: the executor loops the selected experts
        // through it, so the schedule does not grow with the top-k count.
        let intermediate = config.moe_intermediate_size;
        // The band's entry and the tensors it reads are one fact: the stored format. An NVFP4
        // expert carries a per-expert amax and a per-tensor outer-scale denominator beside its
        // packed codes and block scales; an MXFP8 expert carries a ue8m0 scale a 32 columns and
        // no per-expert factor. Naming only the codes would leave the scales - a third of an
        // NVFP4 expert's bytes - loaded and never read.
        //
        // NVFP4 applies its per-expert scale_2 to the expert's own output, and the router weight
        // is another factor of the same product, so the executor passes scale_2 * weight and the
        // sum over selected experts is already the mix. The MXFP8 body has no per-expert factor,
        // so the arm applies that weight when it accumulates the band.
        let (entry, gate_up_weights, down_weights) = match config.layer_format(layer) {
            Some(WeightFormat::NvFp4 { .. }) => (
                "inkling_nvfp4_expert_matvec",
                vec![
                    format!("{prefix}.mlp.experts.w13_weight"),
                    format!("{prefix}.mlp.experts.w13_weight_scale"),
                    format!("{prefix}.mlp.experts.w13_weight_amax"),
                    format!("{prefix}.mlp.experts.w13_weight_outer_scale_denominator"),
                ],
                vec![
                    format!("{prefix}.mlp.experts.w2_weight"),
                    format!("{prefix}.mlp.experts.w2_weight_scale"),
                    format!("{prefix}.mlp.experts.w2_weight_amax"),
                    format!("{prefix}.mlp.experts.w2_weight_outer_scale_denominator"),
                ],
            ),
            Some(WeightFormat::MxFp8 { .. }) => (
                "inkling_mxfp8_expert_matvec",
                vec![
                    format!("{prefix}.mlp.experts.w13_weight"),
                    format!("{prefix}.mlp.experts.w13_weight_scale"),
                ],
                vec![
                    format!("{prefix}.mlp.experts.w2_weight"),
                    format!("{prefix}.mlp.experts.w2_weight_scale"),
                ],
            ),
            other => {
                return Err(format!(
                    "layer {layer} stores its experts as {other:?}, which has no Inkling expert                      entry (the arm serves the released NVFP4 and MXFP8 bands)"
                ));
            }
        };
        let gate_up: Vec<&str> = gate_up_weights.iter().map(String::as_str).collect();
        let down: Vec<&str> = down_weights.iter().map(String::as_str).collect();
        steps.push(self.step(
            Op::ExpertGateUp,
            Some(Part::Moe),
            Some(Image::Inkling),
            Some(entry),
            &gate_up,
            2 * intermediate,
            hidden,
        )?);
        steps.push(self.step(
            Op::ExpertActivation,
            Some(Part::SoftcappedActivation),
            Some(Image::Inkling),
            Some("inkling_softcapped_swiglu"),
            &[],
            intermediate,
            2 * intermediate,
        )?);
        steps.push(self.step(
            Op::ExpertDown,
            Some(Part::Moe),
            Some(Image::Inkling),
            Some(entry),
            &down,
            hidden,
            intermediate,
        )?);
        steps.push(Self::plain(
            Op::Collective(Collective::ExpertDelta),
            Some(Part::Collective),
            rows,
            hidden,
        ));
        // The combine sums the band's slots. The router weight is already inside each partial,
        // so this is the shared plain sum and not a second weighting.
        steps.push(Self::shared(
            Op::ExpertMix,
            Part::MoeCombine,
            Image::ROUTED_EXPERTS,
            "q4e_moe_combine",
            rows,
            hidden,
        ));

        // The MLP output's short convolution, then the residual add that closes the layer.
        steps.push(self.step(
            Op::ShortConv,
            Some(Part::ShortConv),
            Some(Image::Inkling),
            Some(self.entry_for("inkling_short_conv_decode", "inkling_short_conv")),
            &[&format!("{prefix}.mlp_sconv.weight")],
            hidden,
            config.sconv_kernel_size,
        )?);
        steps.push(Self::add(rows, hidden));

        Ok(LayerSchedule {
            index: layer,
            sliding,
            block_queries: if full { 3 } else { 1 },
            taps,
            tau: if sliding {
                TauRule::One
            } else {
                TauRule::LogScaling {
                    n_floor: config.log_scaling_n_floor,
                    alpha: config.log_scaling_alpha,
                }
            },
            steps,
        })
    }

    fn buffer_plan(&self) -> BufferPlan {
        let config = &self.config;
        let hidden = config.hidden_size;
        let heads = self.rank_heads();
        let kv_heads = self.rank_kv_heads();
        let head_dim = config.head_dim;
        let rows = self.rows;
        let activation = config.moe_intermediate_size;
        let history = config.conv_history();
        let mut blocks = 0usize;
        let mut kv_cache = 0usize;
        for layer in 0..config.num_hidden_layers {
            if layer.is_multiple_of(config.n_layers_per_attn_res_block.max(1)) {
                blocks += 1;
            }
            // A sliding layer addresses at most its window; a full layer addresses the context.
            let cache_rows = if config.is_sliding(layer) {
                config.sliding_window_size.min(self.context)
            } else {
                self.context
            };
            // Two caches (key and value) of BF16 rows.
            kv_cache = kv_cache.saturating_add(cache_rows * kv_heads * head_dim * 2 * 2);
        }
        let conv_state = config.num_hidden_layers
            * (2 * kv_heads * head_dim + 2 * hidden)
            * history
            * std::mem::size_of::<f32>();
        let mut plan = BufferPlan {
            hidden: rows * hidden * 4,
            block_inputs: rows * 3 * hidden * 4,
            query: rows * heads * head_dim * 4,
            gate: rows * heads * head_dim * 4,
            relative: rows * heads * config.d_rel * 4,
            sink: heads * 4,
            kv: rows * kv_heads * head_dim * 4,
            delta: rows * hidden * 4,
            expert_band: config.num_experts_per_tok * 2 * activation * 4,
            expert_activation: rows * activation * 4,
            logits: rows * self.rank_vocab() * 4,
            conv_state,
            kv_cache,
            block_summaries: blocks * hidden * 4,
            total: 0,
        };
        plan.total = plan.hidden
            + plan.block_inputs
            + plan.query
            + plan.gate
            + plan.relative
            + plan.sink
            + plan.kv
            + plan.delta
            + plan.expert_band
            + plan.expert_activation
            + plan.logits
            + plan.conv_state
            + plan.kv_cache
            + plan.block_summaries;
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::Target;

    const CONFIG: &str = include_str!("../config/inkling2_small.hf.json");
    const QUANT: &str = include_str!("../config/inkling2_small.hf_quant.json");

    fn released() -> Inkling2Config {
        Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses")
    }

    fn plan(
        world: usize,
        rank: usize,
        pass: Pass,
        rows: usize,
        context: usize,
    ) -> InklingDevicePlan {
        let tp = TpPlan::new(world, rank).expect("a TP plan");
        InklingDevicePlan::build(&released(), tp, pass, rows, context).expect("the schedule")
    }

    #[test]
    fn the_tp4_geometry_is_the_released_split() {
        let plan = plan(4, 2, Pass::Decode, 1, 262_144);
        assert_eq!(plan.rank_heads(), 12);
        assert_eq!(plan.rank_kv_heads(), 2);
        assert_eq!(plan.rank_experts(), 96);
        assert_eq!(plan.rank_vocab(), 50_256);
        // The expert's intermediate width is not sharded as well: whole experts, not tiles.
        let experts = plan
            .served()
            .index_of("model.llm.layers.0.mlp.experts.w13_weight")
            .expect("the expert tensor");
        assert_eq!(plan.served().rank_spec_at(experts).shape[1], 2 * 2304);
    }

    #[test]
    fn every_served_weight_is_read_exactly_once() {
        for (world, rank) in [(1, 0), (4, 0), (4, 3)] {
            let plan = plan(world, rank, Pass::Decode, 1, 4096);
            let served = plan.served();
            let mut reads = vec![0usize; served.plan().len()];
            for step in plan.steps() {
                for index in &step.weights {
                    reads[*index] += 1;
                }
            }
            for (index, count) in reads.iter().enumerate() {
                let name = &served.plan().specs()[index].name;
                assert_eq!(
                    *count, 1,
                    "TP{world}/{rank}: {name} is read {count} times, not once"
                );
            }
        }
    }

    #[test]
    fn every_named_kernel_is_carried_by_its_image() {
        use sparkflash_ops::parts::kernels;
        for (pass, rows) in [(Pass::Decode, 1), (Pass::Prefill, 8)] {
            let plan = plan(4, 0, pass, rows, 4096);
            for step in plan.steps() {
                let Some(image) = step.image else {
                    assert!(
                        matches!(step.op, Op::Collective(_)),
                        "{} has no image and no kernel",
                        step.op.name()
                    );
                    continue;
                };
                let entry = step.entry.expect("a step with an image names its entry");
                assert!(
                    image.exported_symbols().contains(&entry),
                    "{} names {entry}, which {} does not export",
                    step.op.name(),
                    image.name()
                );
                if let Some(part) = step.part {
                    let found = kernels(part, Target::Sm121).any(|kernel| {
                        kernel.image == image
                            && kernel.symbols.iter().any(|symbol| symbol.name == entry)
                    });
                    assert!(
                        found,
                        "{} names {entry} under {part} on {}, which the catalogue does not carry",
                        step.op.name(),
                        image.name()
                    );
                }
            }
        }
    }

    #[test]
    fn the_block_boundary_layers_are_every_sixth() {
        let plan = plan(4, 0, Pass::Decode, 1, 4096);
        for layer in plan.layers() {
            let boundary = layer.index.is_multiple_of(6);
            // A block residual appears once per layer for the MLP input, and once more at the
            // attention input from layer 1 on (layer 0 has no bank: it consumes the embedding).
            let banks = layer
                .steps
                .iter()
                .filter(|step| step.op == Op::BlockResidual)
                .count();
            assert_eq!(
                banks,
                if layer.index == 0 { 1 } else { 2 },
                "layer {}",
                layer.index
            );
            // Two residual adds a layer, less one at a block boundary where the attention output
            // *is* the stream.
            let adds = layer
                .steps
                .iter()
                .filter(|step| step.op == Op::ResidualAdd)
                .count();
            assert_eq!(adds, if boundary { 1 } else { 2 }, "layer {}", layer.index);
            assert_eq!(layer.tau == TauRule::One, layer.sliding);
            assert_eq!(layer.block_queries, if layer.sliding { 1 } else { 3 });
        }
    }

    #[test]
    fn the_convolution_taps_keep_the_stored_layout() {
        let plan = plan(4, 0, Pass::Decode, 1, 4096);
        for layer in plan.layers() {
            let expected = if layer.sliding {
                TapLayout::ChannelMajor
            } else {
                TapLayout::TapMajor
            };
            assert_eq!(layer.taps, expected, "layer {}", layer.index);
            let convolutions = layer
                .steps
                .iter()
                .filter(|step| matches!(step.op, Op::ShortConv | Op::ShortConvDynamic))
                .count();
            // Two K/V convolutions, the attention output and the MLP output.
            assert_eq!(convolutions, 4, "layer {}", layer.index);
        }
        // The full layers' K/V form is the dynamic one, the other two are static.
        let full = plan
            .layers()
            .iter()
            .find(|layer| !layer.sliding)
            .expect("a full layer");
        assert_eq!(
            full.steps
                .iter()
                .filter(|step| step.op == Op::ShortConvDynamic)
                .count(),
            2
        );
    }

    #[test]
    fn the_kv_cache_is_the_per_layer_window_allocation() {
        let plan = plan(4, 0, Pass::Decode, 1, 1_048_576);
        let width = plan.rank_kv_heads() * plan.config().head_dim;
        // 8 full layers address the whole context; 40 sliding layers address a 1,024-row window.
        // A row is 256 BF16 fields a cache, and there are two caches.
        let full = 8 * 1_048_576 * width * 2 * 2;
        let sliding = 40 * 1024 * width * 2 * 2;
        assert_eq!(plan.buffers().kv_cache, full + sliding);
        // 8 GiB at 1M: 1,024 bytes a row over 8 full layers. The same plan without the 1,024-row
        // window on the 40 sliding layers would be 25 GiB, which is what the window buys.
        assert_eq!(full + sliding, 8 * 1024 * 1024 * 1024 + 40 * 1024 * 1024);
    }

    #[test]
    fn the_log_scaling_factor_grows_with_the_position() {
        let rule = TauRule::LogScaling {
            n_floor: 128_000,
            alpha: 0.1,
        };
        assert_eq!(rule.at(0), 1.0);
        // The released floor is a position count: position + 1 <= n_floor is the flat region.
        assert_eq!(rule.at(127_999), 1.0);
        assert!(rule.at(128_000) > 1.0);
        let deep = rule.at(1_048_576);
        assert!((deep - 1.2099).abs() < 1e-3, "tau at 1M is {deep}");
        assert_eq!(TauRule::One.at(999_999), 1.0);
    }

    #[test]
    fn the_buffer_total_covers_every_term() {
        let plan = plan(4, 1, Pass::Decode, 1, 65_536);
        let buffers = plan.buffers();
        let sum = buffers.hidden
            + buffers.block_inputs
            + buffers.query
            + buffers.gate
            + buffers.relative
            + buffers.sink
            + buffers.kv
            + buffers.delta
            + buffers.expert_band
            + buffers.expert_activation
            + buffers.logits
            + buffers.conv_state
            + buffers.kv_cache
            + buffers.block_summaries;
        assert_eq!(buffers.total, sum);
        // Eight block summaries, one a boundary layer.
        assert_eq!(buffers.block_summaries, 8 * 4608 * 4);
        // Four histories a layer of kernel - 1 = 3 samples.
        assert_eq!(buffers.conv_state, 48 * (2 * 256 + 2 * 4608) * 3 * 4);
    }

    #[test]
    fn a_decode_pass_is_one_row_and_a_prefill_chunk_is_many() {
        let tp = TpPlan::new(4, 0).expect("a TP plan");
        let config = released();
        assert!(InklingDevicePlan::build(&config, tp, Pass::Decode, 4, 4096).is_err());
        let chunk = InklingDevicePlan::build(&config, tp, Pass::Prefill, 8, 4096).expect("chunk");
        assert_eq!(chunk.rows(), 8);
        assert_eq!(
            chunk.layers()[0].steps.len(),
            plan(4, 0, Pass::Decode, 1, 4096).layers()[0].steps.len()
        );
        assert!(
            InklingDevicePlan::build(&config, tp, Pass::Decode, 1, 2_097_152).is_err(),
            "a context past the checkpoint's window"
        );
    }

    #[test]
    fn the_schedule_does_not_grow_with_the_top_k() {
        let plan = plan(4, 0, Pass::Decode, 1, 4096);
        for layer in plan.layers() {
            let band: Vec<&Step> = layer
                .steps
                .iter()
                .filter(|step| {
                    matches!(
                        step.op,
                        Op::ExpertGateUp | Op::ExpertActivation | Op::ExpertDown
                    )
                })
                .collect();
            assert_eq!(band.len(), 3);
            // gate/up produces 2 x 2,304 rows over 4,608 columns of hidden; the activation reads
            // the pair and writes 2,304; the down projection reduces 2,304 into 4,608.
            assert_eq!((band[0].rows, band[0].columns), (4608, 4608));
            assert_eq!((band[1].rows, band[1].columns), (2304, 4608));
            assert_eq!((band[2].rows, band[2].columns), (4608, 2304));
        }
    }

    #[test]
    fn the_launch_count_is_the_layers_plus_the_globals() {
        let plan = plan(4, 0, Pass::Decode, 1, 4096);
        // A sliding layer is 27 launches and a full layer 29: the full layer's third input norm
        // and its two dynamic K/V convolutions replace the sliding layer's single norm and two
        // static convolutions. Layer 0 is one fewer because it has no attention block-residual
        // bank, and the eight global launches are the embed, its norm, the output block residual,
        // the final norm, the head, its softcap and the three collectives' two device steps.
        let mut expected = plan.globals().len();
        for layer in plan.layers() {
            // A sliding layer is 26 launches and a full layer 28: the full layer's three input
            // norms and its two dynamic K/V convolutions replace the sliding layer's single norm
            // and two static convolutions. Layer 0 has no attention block-residual bank, and a
            // block-boundary layer has no attention residual add.
            let mut steps = if layer.sliding { 26 } else { 28 };
            if layer.index == 0 {
                steps -= 1;
            }
            if layer.index.is_multiple_of(6) {
                steps -= 1;
            }
            assert_eq!(layer.steps.len(), steps, "layer {}", layer.index);
            expected += steps;
        }
        assert_eq!(plan.launch_count(), expected);
        assert_eq!(plan.launch_count(), 1_263);
    }

    #[test]
    fn the_attention_part_reads_the_relative_bank_and_both_clips() {
        let plan = plan(4, 0, Pass::Decode, 1, 4096);
        for layer in plan.layers() {
            let attention = layer
                .steps
                .iter()
                .find(|step| step.op == Op::Attention)
                .expect("the attention step");
            let names: Vec<&str> = attention
                .weights
                .iter()
                .map(|index| plan.served().plan().specs()[*index].name.as_str())
                .collect();
            assert_eq!(names.len(), 3, "layer {}", layer.index);
            assert!(names[0].ends_with("rel_logits_proj.proj"));
            assert!(names[1].ends_with("q_l2_clip.scale"));
            assert!(names[2].ends_with("rel_l2_clip.scale"));
        }
    }
}
