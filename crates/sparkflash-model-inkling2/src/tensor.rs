// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small tensor plan, logical and at one TP rank.
//!
//! The released backbone spells every tensor under `model.llm.*` and `model.visual.*`.
//! [InklingTensorPlan::checkpoint] reproduces those names and stored shapes from the typed
//! configuration, so the plan's byte total is a check against the released index rather than a
//! second transcription of it. The drafter's `mtp.safetensors` uses bare `layers.N` names;
//! models/layouts/inkling2.toml describes that file and device/draft.rs reads it directly.

use sparkflash_model::architecture::WeightFormat;
use sparkflash_model::{Split, TpAxis, TpPlan};
use sparkflash_weights::{DType, TensorPlan, TensorSpec};

use crate::config::Inkling2Config;

/// What part of the model a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Token embedding.
    Embedding,
    /// Untied LM head.
    LmHead,
    /// A norm.
    Norm,
    /// The block-attention residual queries.
    BlockResidual,
    /// A dense projection (attention Q/K/V/O/gate/sink/relative).
    Attention,
    /// A short convolution.
    ShortConv,
    /// The L2 clip scales.
    Clip,
    /// The router and its bias.
    Router,
    /// One routed expert class.
    Experts,
    /// The MTP stage.
    Mtp,
    /// The vision tower.
    Vision,
    /// The loop's message path: its controllers and the cross-pass query banks.
    Loop,
}

/// How a rank file divides the routed experts among the ranks.
///
/// Not a released field: the converter picks it, the rank file records it, and a load takes the
/// file's word for it. A file written before the choice existed records nothing and reads as
/// [ExpertSplit::Experts], which is what it holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertSplit {
    /// Expert parallelism: a rank holds a contiguous run of whole experts, 96 of 384 at TP4.
    /// The router's eight picks land on the ranks unevenly, so a decode step waits on the rank
    /// that drew the most of them: three and a half experts on average against the mean of two.
    #[default]
    Experts,
    /// Every rank holds every expert's quarter of the intermediate axis, 576 of 2,304 columns at
    /// TP4: a contiguous run of w13's interleaved gate/up rows and the same columns of w2, with
    /// the per-expert amax vectors whole. Each rank computes all eight picks at a quarter width,
    /// so the ranks finish together; the down projection's sum over the intermediate axis is
    /// then split across ranks and finished by the all-reduce that already follows the band,
    /// which is a different rounding from a rank summing all 2,304 columns itself.
    Intermediate,
    /// Every rank holds every expert's quarter of w13's interleaved gate/up rows (576
    /// intermediate columns at TP4, as [Self::Intermediate]) and a quarter of w2's *output* rows
    /// (1,152 of the 4,608 hidden rows) with all 2,304 of its contraction columns. A rank
    /// computes its quarter of the eight picks' activations, the ranks gather them, and each rank
    /// then finishes its own hidden rows of every pick with the whole sum: every element is the
    /// expert-parallel walk's arithmetic, term for term and tree for tree, and the rank rebuilds
    /// the expert-parallel ranks' mixes and their fold for its rows. So the bits are
    /// [Self::Experts]' while every rank computes the same quarter of the picks.
    Rows,
}

impl ExpertSplit {
    /// The name the converter takes and the rank file records.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Experts => "experts",
            Self::Intermediate => "intermediate",
            Self::Rows => "rows",
        }
    }

    /// Whether every rank holds every expert (a slice of each), so a rank's run is all of them.
    #[must_use]
    pub const fn every_expert(self) -> bool {
        matches!(self, Self::Intermediate | Self::Rows)
    }

    /// Parses [Self::name].
    ///
    /// # Errors
    /// Any other spelling.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "experts" => Ok(Self::Experts),
            "intermediate" => Ok(Self::Intermediate),
            "rows" => Ok(Self::Rows),
            other => Err(format!(
                "{other:?} is not an expert split (experts, intermediate or rows)"
            )),
        }
    }
}

/// One rank's slice of a tensor: the logical axis and the run on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// The axis the run applies to.
    pub axis: usize,
    /// The split on that axis.
    pub split: Split,
}

/// The planned tensors and a rank's slices.
#[derive(Debug, Clone)]
pub struct InklingTensorPlan {
    plan: TensorPlan<Role>,
    rank: Vec<TensorSpec>,
    runs: Vec<Vec<Slice>>,
    tp: TpPlan,
}

impl InklingTensorPlan {
    /// The served backbone at one rank.
    ///
    /// # Errors
    /// A world the model's heads, experts or vocabulary do not divide.
    pub fn served(config: &Inkling2Config, tp: TpPlan) -> Result<Self, String> {
        Self::build(config, tp, false)
    }

    /// Backbone and vision.
    ///
    /// The MTP stage is a separate file with its own bare layers.N spelling. Its description
    /// lives in models/layouts/inkling2.toml and its bytes are read by the device drafter
    /// (device/draft.rs), not by this plan.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn checkpoint(config: &Inkling2Config, tp: TpPlan) -> Result<Self, String> {
        Self::build(config, tp, true)
    }

    fn build(config: &Inkling2Config, tp: TpPlan, vision: bool) -> Result<Self, String> {
        let mut plan = TensorPlan::default();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        for (axis, count, name) in [
            (TpAxis::Heads, heads, "attention heads"),
            (TpAxis::KvHeads, kv_heads, "key/value heads"),
            (TpAxis::Vocab, vocab, "vocabulary rows"),
        ] {
            tp.split(count, axis, 1)
                .map_err(|error| format!("{name}: {}", error.detail))?;
        }
        match config.expert_split {
            ExpertSplit::Experts => {
                tp.split(config.n_routed_experts, TpAxis::Experts, 1)
                    .map_err(|error| format!("routed experts: {}", error.detail))?;
            }
            ExpertSplit::Intermediate => {
                // A rank's columns must be whole scale blocks, or one block's scale would cover
                // columns on two ranks.
                let block = (0..config.num_hidden_layers)
                    .map(|layer| match config.layer_format(layer) {
                        Some(WeightFormat::NvFp4 { group_size, .. }) => group_size,
                        Some(WeightFormat::MxFp8Row { block }) => block,
                        _ => 1,
                    })
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let width = config.moe_intermediate_size;
                if !width.is_multiple_of(tp.world() * block) {
                    return Err(format!(
                        "an intermediate expert split needs the {width}-column expert width to \
                         be whole {block}-column scale blocks on each of {} ranks",
                        tp.world()
                    ));
                }
            }
            ExpertSplit::Rows => {
                // The ranks gather equal pieces in place (the activation's columns, then the
                // mix's hidden rows), so both axes divide exactly; no scale block is cut, since
                // neither slice touches a contraction axis.
                let (width, world) = (config.moe_intermediate_size, tp.world());
                if !width.is_multiple_of(world) || !hidden.is_multiple_of(world) {
                    return Err(format!(
                        "a row expert split needs the {width}-column expert width and the \
                         {hidden}-wide residual to divide evenly over {world} ranks"
                    ));
                }
            }
        }

        plan.push(
            "model.llm.embed.weight",
            DType::BF16,
            vec![vocab, hidden],
            Role::Embedding,
        );
        plan.push(
            "model.llm.embed_norm.weight",
            DType::BF16,
            vec![hidden],
            Role::Norm,
        );
        plan.push(
            "model.llm.norm.weight",
            DType::BF16,
            vec![hidden],
            Role::Norm,
        );
        plan.push(
            "model.llm.unembed.weight",
            DType::BF16,
            vec![vocab, hidden],
            Role::LmHead,
        );
        plan.push(
            "model.llm.attn_res.output_query",
            DType::F32,
            vec![hidden],
            Role::BlockResidual,
        );
        // The bank is addressed by the layer before the one that consumes it: bank 4 is the
        // three-query attention input of full layer 5, and bank 5 is the one-query input of
        // sliding layer 6. Layer 0 has no input bank, so the names are 0..=46 for layers
        // 1..=47.
        for bank in 0..config.num_hidden_layers.saturating_sub(1) {
            let layer = bank + 1;
            let query_width = if config.full_layer_ids.contains(&layer) {
                3
            } else {
                1
            };
            plan.push(
                &format!("model.llm.attn_res.attn_queries.{bank}"),
                DType::F32,
                if query_width == 1 {
                    vec![hidden]
                } else {
                    vec![query_width, hidden]
                },
                Role::BlockResidual,
            );
        }
        for layer in 0..config.num_hidden_layers {
            plan.push(
                &format!("model.llm.attn_res.mlp_queries.{layer}"),
                DType::F32,
                vec![hidden],
                Role::BlockResidual,
            );
        }
        if let Some(loop_plan) = config.loop_plan {
            // The loop's message path: four gates a core layer (query, key, value and MLP; the
            // checkpoint spells the query stream `attn`) and the two cross-pass query banks.
            // modeling_inkling2_loop.py:81-99 and :147-176.
            for layer in loop_plan.core_start..loop_plan.core_end {
                for stream in ["attn", "k", "v", "mlp"] {
                    let prefix = format!("model.llm.layers.{layer}.controllers.{stream}");
                    plan.push(
                        &format!("{prefix}.down_weight"),
                        DType::BF16,
                        vec![loop_plan.gate_rank, 2 * hidden],
                        Role::Loop,
                    );
                    plan.push(
                        &format!("{prefix}.forget_weight"),
                        DType::BF16,
                        vec![hidden, loop_plan.gate_rank],
                        Role::Loop,
                    );
                    plan.push(
                        &format!("{prefix}.gain_weight"),
                        DType::BF16,
                        vec![hidden, loop_plan.gate_rank],
                        Role::Loop,
                    );
                }
            }
            for bank in [
                "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries",
            ] {
                plan.push(
                    bank,
                    DType::F32,
                    vec![2 * loop_plan.core_len(), hidden],
                    Role::Loop,
                );
            }
        }

        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.llm.layers.{layer}");
            let attn = format!("{prefix}.attn");
            plan.push(
                &format!("{prefix}.mlp_norm.weight"),
                DType::BF16,
                vec![hidden],
                Role::Norm,
            );
            if config.full_layer_ids.contains(&layer) {
                for name in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                    plan.push(
                        &format!("{prefix}.{name}.weight"),
                        DType::BF16,
                        vec![hidden],
                        Role::Norm,
                    );
                }
            } else {
                plan.push(
                    &format!("{prefix}.attn_norm.weight"),
                    DType::BF16,
                    vec![hidden],
                    Role::Norm,
                );
            }
            plan.push(
                &format!("{prefix}.attn_sconv.weight"),
                DType::BF16,
                vec![hidden, 1, config.sconv_kernel_size],
                Role::ShortConv,
            );
            plan.push(
                &format!("{prefix}.mlp_sconv.weight"),
                DType::BF16,
                vec![hidden, 1, config.sconv_kernel_size],
                Role::ShortConv,
            );
            plan.push(
                &format!("{attn}.wq_du.weight"),
                DType::BF16,
                vec![heads * head_dim, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.wk_dv.weight"),
                DType::BF16,
                vec![kv_heads * head_dim, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.wv_dv.weight"),
                DType::BF16,
                vec![kv_heads * head_dim, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.wo_ud.weight"),
                DType::BF16,
                vec![hidden, heads * head_dim],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.wr_du.weight"),
                DType::BF16,
                vec![heads * config.d_rel, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.gate_attn.weight"),
                DType::BF16,
                vec![heads * head_dim, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.attn_sink_proj.weight"),
                DType::BF16,
                vec![heads, hidden],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.rel_logits_proj.proj"),
                DType::BF16,
                vec![config.d_rel, config.rel_extent],
                Role::Attention,
            );
            plan.push(
                &format!("{attn}.q_l2_clip.scale"),
                DType::BF16,
                vec![heads, head_dim],
                Role::Clip,
            );
            plan.push(
                &format!("{attn}.rel_l2_clip.scale"),
                DType::BF16,
                vec![heads, config.d_rel],
                Role::Clip,
            );
            if config.full_layer_ids.contains(&layer) {
                // A full layer's dynamic short convolution stores the static taps as
                // `static_w` and a per-token projection from the hidden state.
                for name in ["k_sconv", "v_sconv"] {
                    plan.push(
                        &format!("{attn}.{name}.static_w"),
                        DType::BF16,
                        vec![config.sconv_kernel_size, kv_heads * head_dim],
                        Role::ShortConv,
                    );
                    plan.push(
                        &format!("{attn}.{name}.weight_proj"),
                        DType::BF16,
                        vec![kv_heads * config.sconv_kernel_size, hidden],
                        Role::ShortConv,
                    );
                }
            } else {
                for name in ["k_sconv", "v_sconv"] {
                    plan.push(
                        &format!("{attn}.{name}.weight"),
                        DType::BF16,
                        vec![kv_heads * head_dim, 1, config.sconv_kernel_size],
                        Role::ShortConv,
                    );
                }
            }
            plan.push(
                &format!("{prefix}.mlp.gate.weight"),
                DType::BF16,
                vec![config.n_routed_experts, hidden],
                Role::Router,
            );
            plan.push(
                &format!("{prefix}.mlp.gate.bias"),
                DType::F32,
                vec![config.n_routed_experts],
                Role::Router,
            );
            match config.layer_format(layer).unwrap_or(WeightFormat::Bf16) {
                WeightFormat::NvFp4 { group_size, .. } => {
                    let w13_out = 2 * config.moe_intermediate_size;
                    let w13_cols = hidden.div_ceil(group_size);
                    let w2_cols = config.moe_intermediate_size.div_ceil(group_size);
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight"),
                        DType::U8,
                        vec![config.n_routed_experts, w13_out, hidden / 2],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight"),
                        DType::U8,
                        vec![
                            config.n_routed_experts,
                            hidden,
                            config.moe_intermediate_size / 2,
                        ],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight_scale"),
                        DType::F8E4M3,
                        vec![config.n_routed_experts, w13_out, w13_cols],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight_scale"),
                        DType::F8E4M3,
                        vec![config.n_routed_experts, hidden, w2_cols],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight_amax"),
                        DType::F32,
                        vec![config.n_routed_experts],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight_amax"),
                        DType::F32,
                        vec![config.n_routed_experts],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight_outer_scale_denominator"),
                        DType::F32,
                        vec![1],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight_outer_scale_denominator"),
                        DType::F32,
                        vec![1],
                        Role::Experts,
                    );
                }
                WeightFormat::MxFp8Row { block } => {
                    let w13_out = 2 * config.moe_intermediate_size;
                    let w13_cols = hidden.div_ceil(block);
                    let w2_cols = config.moe_intermediate_size.div_ceil(block);
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight"),
                        DType::F8E4M3,
                        vec![config.n_routed_experts, w13_out, hidden],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight"),
                        DType::F8E4M3,
                        vec![
                            config.n_routed_experts,
                            hidden,
                            config.moe_intermediate_size,
                        ],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight_scale"),
                        DType::U8,
                        vec![config.n_routed_experts, w13_out, w13_cols],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight_scale"),
                        DType::U8,
                        vec![config.n_routed_experts, hidden, w2_cols],
                        Role::Experts,
                    );
                }
                WeightFormat::Bf16 => {
                    // An unquantised checkpoint (and the tiny fixture) stores each expert dense:
                    // w13 is [experts, 2 * intermediate, hidden] and w2 the reverse, with no
                    // scale or amax beside them.
                    plan.push(
                        &format!("{prefix}.mlp.experts.w13_weight"),
                        DType::BF16,
                        vec![
                            config.n_routed_experts,
                            2 * config.moe_intermediate_size,
                            hidden,
                        ],
                        Role::Experts,
                    );
                    plan.push(
                        &format!("{prefix}.mlp.experts.w2_weight"),
                        DType::BF16,
                        vec![
                            config.n_routed_experts,
                            hidden,
                            config.moe_intermediate_size,
                        ],
                        Role::Experts,
                    );
                }
                other => {
                    return Err(format!(
                        "layer {layer}: the Inkling arm reads BF16, NVFP4 and row-scaled MXFP8 experts, and this layer states {}",
                        other.name()
                    ));
                }
            }
        }

        if vision && let Some(vision) = &config.vision {
            let width = vision.hidden_size;
            // The release folds the temporal axis before the patch projection; the stored
            // input width is three channels times the square patch.
            let patch = 3 * vision.patch_size * vision.patch_size;
            let intermediate = ((width as f64) * 4.0).round() as usize;
            let prefix = "model.visual";
            plan.push(
                &format!("{prefix}.backbone.embeddings.patch_embedding.weight"),
                DType::BF16,
                vec![width, patch],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.backbone.embeddings.position_embedding"),
                DType::BF16,
                vec![vision.posemb_grid_size * vision.posemb_grid_size, width],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.backbone.embeddings.class_embedding"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.backbone.embeddings.cls_position_embedding"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.backbone.ln_pre.weight"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.backbone.ln_pre.bias"),
                DType::BF16,
                vec![width],
                Role::Vision,
            );
            for layer in 0..vision.depth {
                let p = format!("{prefix}.backbone.layers.{layer}");
                for name in ["ln_1", "ln_2"] {
                    plan.push(
                        &format!("{p}.{name}.weight"),
                        DType::BF16,
                        vec![width],
                        Role::Vision,
                    );
                    plan.push(
                        &format!("{p}.{name}.bias"),
                        DType::BF16,
                        vec![width],
                        Role::Vision,
                    );
                }
                for name in ["ls_1", "ls_2"] {
                    plan.push(
                        &format!("{p}.{name}.gamma"),
                        DType::BF16,
                        vec![width],
                        Role::Vision,
                    );
                }
                for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    plan.push(
                        &format!("{p}.self_attn.{name}.weight"),
                        DType::BF16,
                        vec![width, width],
                        Role::Vision,
                    );
                    plan.push(
                        &format!("{p}.self_attn.{name}.bias"),
                        DType::BF16,
                        vec![width],
                        Role::Vision,
                    );
                }
                plan.push(
                    &format!("{p}.mlp.c_fc.weight"),
                    DType::BF16,
                    vec![intermediate, width],
                    Role::Vision,
                );
                plan.push(
                    &format!("{p}.mlp.c_fc.bias"),
                    DType::BF16,
                    vec![intermediate],
                    Role::Vision,
                );
                plan.push(
                    &format!("{p}.mlp.c_proj.weight"),
                    DType::BF16,
                    vec![width, intermediate],
                    Role::Vision,
                );
                plan.push(
                    &format!("{p}.mlp.c_proj.bias"),
                    DType::BF16,
                    vec![width],
                    Role::Vision,
                );
            }
            plan.push(
                &format!("{prefix}.proj_in.weight"),
                DType::BF16,
                vec![
                    vision.projection_hidden_dim,
                    width * vision.adapter_merge_size * vision.adapter_merge_size,
                ],
                Role::Vision,
            );
            plan.push(
                &format!("{prefix}.proj_out.weight"),
                DType::BF16,
                vec![vision.out_hidden_size, vision.projection_hidden_dim],
                Role::Vision,
            );
        }

        let runs = plan
            .specs()
            .iter()
            .map(|spec| split_of(config, tp, &spec.name, &spec.shape))
            .collect::<Result<Vec<_>, _>>()?;
        let rank = plan
            .specs()
            .iter()
            .zip(&runs)
            .map(|(spec, run)| rank_spec(spec, run))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            rank,
            runs,
            tp,
        })
    }

    /// The logical plan.
    #[must_use]
    pub fn plan(&self) -> &TensorPlan<Role> {
        &self.plan
    }

    /// The tensor-parallel plan.
    #[must_use]
    pub const fn tp(&self) -> TpPlan {
        self.tp
    }

    /// The rank-local spec at an index.
    #[must_use]
    pub fn rank_spec_at(&self, index: usize) -> &TensorSpec {
        &self.rank[index]
    }

    /// The slice of the logical axis this rank holds, when sharded.
    #[must_use]
    pub fn rank_run_at(&self, index: usize) -> Option<Slice> {
        self.runs[index].first().copied()
    }

    /// Every slice of a rank-local tensor. An expert tensor is split on two axes: the expert
    /// axis (whole experts) and the intermediate axis (576 columns at TP4).
    #[must_use]
    pub fn rank_slices_at(&self, index: usize) -> &[Slice] {
        &self.runs[index]
    }

    /// Look a tensor up by released name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.plan.index_of(name)
    }

    /// The exact stored byte total of the whole model, at any TP world.
    #[must_use]
    pub fn storage_bytes(&self) -> u128 {
        self.plan.storage_bytes()
    }

    /// The exact byte total this rank holds: the same number [Self::storage_bytes] gives at TP1,
    /// and a quarter of it at TP4.
    #[must_use]
    pub fn rank_storage_bytes(&self) -> u128 {
        self.rank.iter().map(TensorSpec::storage_bytes).sum()
    }
}

fn split_of(
    config: &Inkling2Config,
    tp: TpPlan,
    name: &str,
    shape: &[usize],
) -> Result<Vec<Slice>, String> {
    let slice = |axis: usize, count: usize, tp_axis: TpAxis| -> Result<Slice, String> {
        tp.split(count, tp_axis, 1)
            .map(|split| Slice { axis, split })
            .map_err(|error| format!("{name}: {}", error.detail))
    };
    // A short convolution's stored axis is *channel rows*, and one head is a run of them: a
    // key/value tap table is [head][channel][tap], not [channel][tap] in a flat list. Dividing the
    // rows directly would cut a head in quarters at TP4 instead of giving each rank its own - and
    // would disagree with the shared key/value head rule, where two heads over four ranks are one
    // head each, read twice. This splits the head axis and scales to rows, so the rank's table
    // starts at the same head its key/value projection produced.
    let head_block = |axis: usize, heads: usize, block: usize| -> Result<Slice, String> {
        let split = tp
            .split(heads, TpAxis::KvHeads, 1)
            .map_err(|error| format!("{name}: {}", error.detail))?;
        let per_rank = split.per_rank * block;
        Ok(Slice {
            axis,
            split: Split {
                total: heads * block,
                per_rank,
                offset: split.offset * block,
                padded: per_rank,
                padding: 0,
                rank: split.rank,
                world: split.world,
            },
        })
    };
    if name == "model.llm.embed.weight" || name == "model.llm.unembed.weight" {
        return Ok(vec![slice(0, shape[0], TpAxis::Vocab)?]);
    }
    if name.ends_with("_outer_scale_denominator") {
        return Ok(Vec::new());
    }
    if name.contains(".mlp.experts.") {
        // Expert parallelism: each rank holds whole experts, [96, 2*2304, H] and [96, H, 2304].
        //
        // The expert axis alone, not the expert axis *and* the intermediate axis. Splitting both
        // gives a rank 96 experts of 576 intermediate columns, a sixteenth of the expert weights,
        // and four such files hold a quarter of the model. The router picks 8 of 384 experts, so
        // an expert is the natural unit: a rank computes its own experts whole and the collective
        // exchanges tokens, while the intermediate width stays the shape the kernels are built
        // for. The tiling test in `convert` is the check that four files hold the served plan.
        //
        // [ExpertSplit::Intermediate] is the other quarter: every expert at a quarter of the
        // intermediate width (not both splits at once). w13's rows interleave gate and up, row
        // 2c the gate of column c and 2c + 1 its up, so a contiguous quarter of the rows is a
        // quarter of the columns with both halves; w2 and its block scales split their last
        // axis, which is the same columns. The amax vectors are per expert and stay whole.
        // [ExpertSplit::Rows] takes w13 the same way and w2 (codes and block scales alike) on its
        // output-row axis, the residual width, so no row loses a contraction column.
        let every = config.expert_split.every_expert();
        return match config.expert_split {
            ExpertSplit::Experts => Ok(vec![slice(0, shape[0], TpAxis::Experts)?]),
            _ if every && name.ends_with("_amax") => Ok(Vec::new()),
            _ if every && name.contains(".w13_weight") => {
                let run = slice(1, shape[1], TpAxis::Intermediate)?;
                if !run.split.per_rank.is_multiple_of(2) {
                    return Err(format!(
                        "{name}: {} gate/up rows a rank would split a column's pair",
                        run.split.per_rank
                    ));
                }
                Ok(vec![run])
            }
            ExpertSplit::Intermediate if name.contains(".w2_weight") => {
                Ok(vec![slice(2, shape[2], TpAxis::Intermediate)?])
            }
            ExpertSplit::Rows if name.contains(".w2_weight") => {
                Ok(vec![slice(1, shape[1], TpAxis::Hidden)?])
            }
            split => Err(format!(
                "{name}: an expert tensor the {} split has no rule for",
                split.name()
            )),
        };
    }
    if name.ends_with(".wo_ud.weight") {
        return Ok(vec![slice(1, shape[1], TpAxis::Heads)?]);
    }
    if name.ends_with(".wr_du.weight")
        || name.ends_with(".rel_l2_clip.scale")
        || name.ends_with(".attn_sink_proj.weight")
        || name.ends_with(".q_l2_clip.scale")
        || name.ends_with(".wq_du.weight")
        || name.ends_with(".gate_attn.weight")
    {
        return Ok(vec![slice(0, shape[0], TpAxis::Heads)?]);
    }
    // A key/value projection is one head a rank, not a quarter of a head. Splitting its rows
    // directly divides the *dimensions* as soon as the head count is not a multiple of the world
    // (two heads over four ranks are four dimensions each, not one head each), and the kernel
    // then mixes half a head with another head's other half.
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let kernel = config.sconv_kernel_size;
    if name.ends_with(".wk_dv.weight") || name.ends_with(".wv_dv.weight") {
        return Ok(vec![head_block(0, kv_heads, head_dim)?]);
    }
    // The key/value short convolutions are per head, in the same head order the projections
    // produce, so they follow the same split at their own granularity.
    if name.ends_with(".k_sconv.weight") || name.ends_with(".v_sconv.weight") {
        return Ok(vec![head_block(0, kv_heads, head_dim)?]);
    }
    if name.ends_with(".k_sconv.static_w") || name.ends_with(".v_sconv.static_w") {
        return Ok(vec![head_block(1, kv_heads, head_dim)?]);
    }
    if name.ends_with(".k_sconv.weight_proj") || name.ends_with(".v_sconv.weight_proj") {
        return Ok(vec![head_block(0, kv_heads, kernel)?]);
    }
    Ok(Vec::new())
}

fn rank_spec(spec: &TensorSpec, runs: &[Slice]) -> Result<TensorSpec, String> {
    let mut shape = spec.shape.clone();
    for run in runs {
        let axis = run.axis;
        if axis >= shape.len() {
            return Err(format!(
                "{}: slice axis {axis} is outside the shape",
                spec.name
            ));
        }
        if shape[axis] != run.split.total {
            return Err(format!(
                "{}: axis {axis} is {} wide, the split covers {}",
                spec.name, shape[axis], run.split.total
            ));
        }
        shape[axis] = run.split.per_rank;
    }
    Ok(TensorSpec {
        name: spec.name.clone(),
        dtype: spec.dtype,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Inkling2Config;
    use sparkflash_model::TpPlan;

    const CONFIG: &str = include_str!("../config/inkling2_small.hf.json");
    const QUANT: &str = include_str!("../config/inkling2_small.hf_quant.json");

    #[test]
    fn the_checkpoint_plan_matches_the_released_byte_total() {
        let config = Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        let tp = TpPlan::new(4, 0).expect("a TP plan");
        let plan = InklingTensorPlan::checkpoint(&config, tp).expect("the plan");
        // The backbone index's total_size is 390,056,448,320 bytes; the plan also carries the
        // MTP names (its separate 3,423,429,800-byte file), so compare the backbone role sum.
        let backbone: u128 = plan
            .plan()
            .specs()
            .iter()
            .zip((0..plan.plan().len()).map(|index| plan.plan().role(index)))
            .filter(|(_, role)| !matches!(role, Role::Mtp))
            .map(|(spec, _)| spec.storage_bytes())
            .sum();
        eprintln!("backbone bytes {backbone}, tensors {}", plan.plan().len());
        assert_eq!(backbone, 390_056_448_320);
    }

    #[test]
    fn the_rank_slices_divide_the_released_axes() {
        let config = Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        let tp = TpPlan::new(4, 3).expect("a TP plan");
        let plan = InklingTensorPlan::served(&config, tp).expect("the plan");
        let experts = plan
            .index_of("model.llm.layers.0.mlp.experts.w13_weight")
            .unwrap();
        assert_eq!(plan.rank_spec_at(experts).shape[0], 96);
        // Whole experts: the intermediate width is not sharded as well, or a rank would hold a
        // sixteenth of the expert weights instead of a quarter.
        assert_eq!(plan.rank_spec_at(experts).shape[1], 2 * 2304);
        let down = plan
            .index_of("model.llm.layers.0.mlp.experts.w2_weight")
            .unwrap();
        // The packed w2 keeps half its intermediate width in bytes, so the whole 2,304-element
        // width is stored as 1,152 bytes per row and is not split any further.
        assert_eq!(plan.rank_spec_at(down).shape[2], 1152);
        let embed = plan.index_of("model.llm.embed.weight").unwrap();
        assert_eq!(plan.rank_spec_at(embed).shape[0], 50_256);
    }

    #[test]
    fn an_intermediate_split_gives_every_rank_a_quarter_of_every_expert() {
        let mut config =
            Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        let whole = InklingTensorPlan::served(&config, TpPlan::new(4, 2).expect("TP4"))
            .expect("the expert split")
            .rank_storage_bytes();
        config.expert_split = ExpertSplit::Intermediate;
        let plan = InklingTensorPlan::served(&config, TpPlan::new(4, 2).expect("TP4"))
            .expect("the intermediate split");
        let shape = |name: &str| {
            let index = plan.index_of(name).unwrap_or_else(|| panic!("{name}"));
            plan.rank_spec_at(index).shape.clone()
        };
        // NVFP4 (layer 0): 576 columns are 1,152 interleaved gate/up rows, 288 packed w2 bytes
        // and 36 of w2's 16-column block scales; the amax vectors stay whole.
        let nvfp4 = "model.llm.layers.0.mlp.experts";
        assert_eq!(shape(&format!("{nvfp4}.w13_weight")), vec![384, 1152, 2304]);
        assert_eq!(
            shape(&format!("{nvfp4}.w13_weight_scale")),
            vec![384, 1152, 288]
        );
        assert_eq!(shape(&format!("{nvfp4}.w2_weight")), vec![384, 4608, 288]);
        assert_eq!(
            shape(&format!("{nvfp4}.w2_weight_scale")),
            vec![384, 4608, 36]
        );
        assert_eq!(shape(&format!("{nvfp4}.w13_weight_amax")), vec![384]);
        assert_eq!(shape(&format!("{nvfp4}.w2_weight_amax")), vec![384]);
        // MXFP8 (layer 47): a byte a column and a ue8m0 scale a 32.
        let mxfp8 = "model.llm.layers.47.mlp.experts";
        assert_eq!(shape(&format!("{mxfp8}.w13_weight")), vec![384, 1152, 4608]);
        assert_eq!(
            shape(&format!("{mxfp8}.w13_weight_scale")),
            vec![384, 1152, 144]
        );
        assert_eq!(shape(&format!("{mxfp8}.w2_weight")), vec![384, 4608, 576]);
        assert_eq!(
            shape(&format!("{mxfp8}.w2_weight_scale")),
            vec![384, 4608, 18]
        );
        // The same quarter of the expert bytes, plus three ranks' worth of the 1.5 KB amax
        // vectors a layer that every rank now holds whole.
        let amax = 40 * 2 * 384 * 4 * 3 / 4;
        assert_eq!(plan.rank_storage_bytes(), whole + amax as u128);
    }

    #[test]
    fn an_intermediate_split_refuses_a_world_that_cuts_a_scale_block() {
        let mut config =
            Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        config.expert_split = ExpertSplit::Intermediate;
        // 2,304 columns over 8 ranks are 288 a rank, nine MXFP8 blocks of 32; over 16 they are
        // 144, four and a half.
        assert!(InklingTensorPlan::served(&config, TpPlan::new(8, 0).expect("TP8")).is_ok());
        let refused = InklingTensorPlan::served(&config, TpPlan::new(16, 0).expect("TP16"))
            .expect_err("a ragged block");
        assert!(refused.contains("scale blocks"), "{refused}");
        assert_eq!(
            ExpertSplit::parse("intermediate"),
            Ok(ExpertSplit::Intermediate)
        );
        assert_eq!(ExpertSplit::parse("rows"), Ok(ExpertSplit::Rows));
        assert!(ExpertSplit::parse("columns").is_err());
    }

    #[test]
    fn a_row_split_gives_every_rank_a_quarter_of_every_experts_rows() {
        let mut config =
            Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the release parses");
        let whole = InklingTensorPlan::served(&config, TpPlan::new(4, 2).expect("TP4"))
            .expect("the expert split")
            .rank_storage_bytes();
        config.expert_split = ExpertSplit::Rows;
        let plan = InklingTensorPlan::served(&config, TpPlan::new(4, 2).expect("TP4"))
            .expect("the row split");
        let shape = |name: &str| {
            let index = plan.index_of(name).unwrap_or_else(|| panic!("{name}"));
            plan.rank_spec_at(index).shape.clone()
        };
        // NVFP4 (layer 0): w13 as the intermediate split's, w2 a quarter of its 4,608 rows with
        // all 2,304 columns (1,152 packed bytes, 144 block scales).
        let nvfp4 = "model.llm.layers.0.mlp.experts";
        assert_eq!(shape(&format!("{nvfp4}.w13_weight")), vec![384, 1152, 2304]);
        assert_eq!(
            shape(&format!("{nvfp4}.w13_weight_scale")),
            vec![384, 1152, 288]
        );
        assert_eq!(shape(&format!("{nvfp4}.w2_weight")), vec![384, 1152, 1152]);
        assert_eq!(
            shape(&format!("{nvfp4}.w2_weight_scale")),
            vec![384, 1152, 144]
        );
        assert_eq!(shape(&format!("{nvfp4}.w2_weight_amax")), vec![384]);
        let mxfp8 = "model.llm.layers.47.mlp.experts";
        assert_eq!(shape(&format!("{mxfp8}.w2_weight")), vec![384, 1152, 2304]);
        assert_eq!(
            shape(&format!("{mxfp8}.w2_weight_scale")),
            vec![384, 1152, 72]
        );
        // The same bytes as the intermediate split's.
        let amax = 40 * 2 * 384 * 4 * 3 / 4;
        assert_eq!(plan.rank_storage_bytes(), whole + amax as u128);
    }
}
