// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small tensor plan, logical and at one TP rank.
//!
//! The released checkpoint spells every tensor under `model.llm.*`, `mtp.*` and
//! `model.visual.*`. [InklingTensorPlan::checkpoint] reproduces those names and stored shapes
//! from the typed configuration, so the plan's byte total is a check against the released index
//! rather than a second transcription of it.

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
        Self::build(config, tp, false, false)
    }

    /// Backbone, MTP and vision.
    ///
    /// # Errors
    /// As [Self::served].
    pub fn checkpoint(config: &Inkling2Config, tp: TpPlan) -> Result<Self, String> {
        Self::build(config, tp, true, true)
    }

    fn build(
        config: &Inkling2Config,
        tp: TpPlan,
        mtp: bool,
        vision: bool,
    ) -> Result<Self, String> {
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
        tp.split(config.n_routed_experts, TpAxis::Experts, 1)
            .map_err(|error| format!("routed experts: {}", error.detail))?;

        plan.push("model.llm.embed.weight", DType::BF16, vec![vocab, hidden], Role::Embedding);
        plan.push("model.llm.embed_norm.weight", DType::BF16, vec![hidden], Role::Norm);
        plan.push("model.llm.norm.weight", DType::BF16, vec![hidden], Role::Norm);
        plan.push("model.llm.unembed.weight", DType::BF16, vec![vocab, hidden], Role::LmHead);
        plan.push("model.llm.attn_res.output_query", DType::F32, vec![hidden], Role::BlockResidual);
        // The bank is addressed by the layer before the one that consumes it: bank 4 is the
        // three-query attention input of full layer 5, and bank 5 is the one-query input of
        // sliding layer 6. Layer 0 has no input bank, so the names are 0..=46 for layers
        // 1..=47.
        for bank in 0..config.num_hidden_layers.saturating_sub(1) {
            let layer = bank + 1;
            let query_width = if config.full_layer_ids.contains(&layer) { 3 } else { 1 };
            plan.push(
                &format!("model.llm.attn_res.attn_queries.{bank}"),
                DType::F32,
                if query_width == 1 { vec![hidden] } else { vec![query_width, hidden] },
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

        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.llm.layers.{layer}");
            let attn = format!("{prefix}.attn");
            plan.push(&format!("{prefix}.mlp_norm.weight"), DType::BF16, vec![hidden], Role::Norm);
            if config.full_layer_ids.contains(&layer) {
                for name in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                    plan.push(&format!("{prefix}.{name}.weight"), DType::BF16, vec![hidden], Role::Norm);
                }
            } else {
                plan.push(&format!("{prefix}.attn_norm.weight"), DType::BF16, vec![hidden], Role::Norm);
            }
            plan.push(&format!("{prefix}.attn_sconv.weight"), DType::BF16, vec![hidden, 1, config.sconv_kernel_size], Role::ShortConv);
            plan.push(&format!("{prefix}.mlp_sconv.weight"), DType::BF16, vec![hidden, 1, config.sconv_kernel_size], Role::ShortConv);
            plan.push(&format!("{attn}.wq_du.weight"), DType::BF16, vec![heads * head_dim, hidden], Role::Attention);
            plan.push(&format!("{attn}.wk_dv.weight"), DType::BF16, vec![kv_heads * head_dim, hidden], Role::Attention);
            plan.push(&format!("{attn}.wv_dv.weight"), DType::BF16, vec![kv_heads * head_dim, hidden], Role::Attention);
            plan.push(&format!("{attn}.wo_ud.weight"), DType::BF16, vec![hidden, heads * head_dim], Role::Attention);
            plan.push(&format!("{attn}.wr_du.weight"), DType::BF16, vec![heads * config.d_rel, hidden], Role::Attention);
            plan.push(&format!("{attn}.gate_attn.weight"), DType::BF16, vec![heads * head_dim, hidden], Role::Attention);
            plan.push(&format!("{attn}.attn_sink_proj.weight"), DType::BF16, vec![heads, hidden], Role::Attention);
            plan.push(&format!("{attn}.rel_logits_proj.proj"), DType::BF16, vec![config.d_rel, config.rel_extent], Role::Attention);
            plan.push(&format!("{attn}.q_l2_clip.scale"), DType::BF16, vec![heads, head_dim], Role::Clip);
            plan.push(&format!("{attn}.rel_l2_clip.scale"), DType::BF16, vec![heads, config.d_rel], Role::Clip);
            if config.full_layer_ids.contains(&layer) {
                // A full layer's dynamic short convolution stores the static taps as
                // `static_w` and a per-token projection from the hidden state.
                for name in ["k_sconv", "v_sconv"] {
                    plan.push(&format!("{attn}.{name}.static_w"), DType::BF16, vec![config.sconv_kernel_size, kv_heads * head_dim], Role::ShortConv);
                    plan.push(&format!("{attn}.{name}.weight_proj"), DType::BF16, vec![kv_heads * config.sconv_kernel_size, hidden], Role::ShortConv);
                }
            } else {
                for name in ["k_sconv", "v_sconv"] {
                    plan.push(&format!("{attn}.{name}.weight"), DType::BF16, vec![kv_heads * head_dim, 1, config.sconv_kernel_size], Role::ShortConv);
                }
            }
            plan.push(&format!("{prefix}.mlp.gate.weight"), DType::BF16, vec![config.n_routed_experts, hidden], Role::Router);
            plan.push(&format!("{prefix}.mlp.gate.bias"), DType::F32, vec![config.n_routed_experts], Role::Router);
            match config.layer_format(layer).unwrap_or(WeightFormat::Bf16) {
                WeightFormat::NvFp4 { group_size } => {
                    let w13_out = 2 * config.moe_intermediate_size;
                    let w13_cols = hidden.div_ceil(group_size);
                    let w2_cols = config.moe_intermediate_size.div_ceil(group_size);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight"), DType::U8, vec![config.n_routed_experts, w13_out, hidden / 2], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight"), DType::U8, vec![config.n_routed_experts, hidden, config.moe_intermediate_size / 2], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight_scale"), DType::F8E4M3, vec![config.n_routed_experts, w13_out, w13_cols], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight_scale"), DType::F8E4M3, vec![config.n_routed_experts, hidden, w2_cols], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight_amax"), DType::F32, vec![config.n_routed_experts], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight_amax"), DType::F32, vec![config.n_routed_experts], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight_outer_scale_denominator"), DType::F32, vec![1], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight_outer_scale_denominator"), DType::F32, vec![1], Role::Experts);
                }
                WeightFormat::MxFp8 { block } => {
                    let w13_out = 2 * config.moe_intermediate_size;
                    let w13_cols = hidden.div_ceil(block);
                    let w2_cols = config.moe_intermediate_size.div_ceil(block);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight"), DType::F8E4M3, vec![config.n_routed_experts, w13_out, hidden], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight"), DType::F8E4M3, vec![config.n_routed_experts, hidden, config.moe_intermediate_size], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w13_weight_scale"), DType::U8, vec![config.n_routed_experts, w13_out, w13_cols], Role::Experts);
                    plan.push(&format!("{prefix}.mlp.experts.w2_weight_scale"), DType::U8, vec![config.n_routed_experts, hidden, w2_cols], Role::Experts);
                }
                _ => return Err(format!("layer {layer} has no expert format")),
            }
        }

        if vision {
            if let Some(vision) = &config.vision {
                let width = vision.hidden_size;
                // The release folds the temporal axis before the patch projection; the stored
                // input width is three channels times the square patch.
                let patch = 3 * vision.patch_size * vision.patch_size;
                let intermediate = ((width as f64) * 4.0).round() as usize;
                let prefix = "model.visual";
                plan.push(&format!("{prefix}.backbone.embeddings.patch_embedding.weight"), DType::BF16, vec![width, patch], Role::Vision);
                plan.push(&format!("{prefix}.backbone.embeddings.position_embedding"), DType::BF16, vec![vision.posemb_grid_size * vision.posemb_grid_size, width], Role::Vision);
                plan.push(&format!("{prefix}.backbone.embeddings.class_embedding"), DType::BF16, vec![width], Role::Vision);
                plan.push(&format!("{prefix}.backbone.embeddings.cls_position_embedding"), DType::BF16, vec![width], Role::Vision);
                plan.push(&format!("{prefix}.backbone.ln_pre.weight"), DType::BF16, vec![width], Role::Vision);
                plan.push(&format!("{prefix}.backbone.ln_pre.bias"), DType::BF16, vec![width], Role::Vision);
                for layer in 0..vision.depth {
                    let p = format!("{prefix}.backbone.layers.{layer}");
                    for name in ["ln_1", "ln_2"] {
                        plan.push(&format!("{p}.{name}.weight"), DType::BF16, vec![width], Role::Vision);
                        plan.push(&format!("{p}.{name}.bias"), DType::BF16, vec![width], Role::Vision);
                    }
                    for name in ["ls_1", "ls_2"] {
                        plan.push(&format!("{p}.{name}.gamma"), DType::BF16, vec![width], Role::Vision);
                    }
                    for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                        plan.push(&format!("{p}.self_attn.{name}.weight"), DType::BF16, vec![width, width], Role::Vision);
                        plan.push(&format!("{p}.self_attn.{name}.bias"), DType::BF16, vec![width], Role::Vision);
                    }
                    plan.push(&format!("{p}.mlp.c_fc.weight"), DType::BF16, vec![intermediate, width], Role::Vision);
                    plan.push(&format!("{p}.mlp.c_fc.bias"), DType::BF16, vec![intermediate], Role::Vision);
                    plan.push(&format!("{p}.mlp.c_proj.weight"), DType::BF16, vec![width, intermediate], Role::Vision);
                    plan.push(&format!("{p}.mlp.c_proj.bias"), DType::BF16, vec![width], Role::Vision);
                }
                plan.push(&format!("{prefix}.proj_in.weight"), DType::BF16, vec![vision.projection_hidden_dim, width * vision.adapter_merge_size * vision.adapter_merge_size], Role::Vision);
                plan.push(&format!("{prefix}.proj_out.weight"), DType::BF16, vec![vision.out_hidden_size, vision.projection_hidden_dim], Role::Vision);
            }
        }

        if mtp {
            // The MTP checkpoint is a separate file; the plan names its stage tensors for the
            // converter and the drafter gate. The names follow the released module.
            if let Some(mtp) = &config.mtp {
                let hidden = mtp.hidden_size;
                let prefix = "mtp";
                plan.push(&format!("{prefix}.fuse_weight"), DType::BF16, vec![hidden, 2 * hidden], Role::Mtp);
                for layer in 0..mtp.num_hidden_layers {
                    let p = format!("{prefix}.layers.{layer}");
                    for name in ["input_layernorm", "post_attention_layernorm", "dflare_norm"] {
                        plan.push(&format!("{p}.{name}.weight"), DType::BF16, vec![hidden], Role::Mtp);
                    }
                    for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] {
                        let (rows, cols) = match name {
                            "q_proj" => (mtp.num_attention_heads * mtp.head_dim, hidden),
                            "k_proj" | "v_proj" => (mtp.num_key_value_heads * mtp.head_dim, hidden),
                            "o_proj" => (hidden, mtp.num_attention_heads * mtp.head_dim),
                            "gate_proj" | "up_proj" => (mtp.intermediate_size, hidden),
                            "down_proj" => (hidden, mtp.intermediate_size),
                            _ => unreachable!(),
                        };
                        plan.push(&format!("{p}.self_attn.{name}.weight"), DType::BF16, vec![rows, cols], Role::Mtp);
                    }
                }
            }
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
        Ok(Self { plan, rank, runs, tp })
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
    _config: &Inkling2Config,
    tp: TpPlan,
    name: &str,
    shape: &[usize],
) -> Result<Vec<Slice>, String> {
    let slice = |axis: usize, count: usize, tp_axis: TpAxis| -> Result<Slice, String> {
        tp.split(count, tp_axis, 1)
            .map(|split| Slice { axis, split })
            .map_err(|error| format!("{name}: {}", error.detail))
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
        return Ok(vec![slice(0, shape[0], TpAxis::Experts)?]);
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
    if name.ends_with(".wk_dv.weight") || name.ends_with(".wv_dv.weight") {
        return Ok(vec![slice(0, shape[0], TpAxis::KvHeads)?]);
    }
    Ok(Vec::new())
}

fn rank_spec(spec: &TensorSpec, runs: &[Slice]) -> Result<TensorSpec, String> {
    let mut shape = spec.shape.clone();
    for run in runs {
        let axis = run.axis;
        if axis >= shape.len() {
            return Err(format!("{}: slice axis {axis} is outside the shape", spec.name));
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
    use sparkflash_model::{TpPlan, Target};

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
        let experts = plan.index_of("model.llm.layers.0.mlp.experts.w13_weight").unwrap();
        assert_eq!(plan.rank_spec_at(experts).shape[0], 96);
        // Whole experts: the intermediate width is not sharded as well, or a rank would hold a
        // sixteenth of the expert weights instead of a quarter.
        assert_eq!(plan.rank_spec_at(experts).shape[1], 2 * 2304);
        let down = plan.index_of("model.llm.layers.0.mlp.experts.w2_weight").unwrap();
        // The packed w2 keeps half its intermediate width in bytes, so the whole 2,304-element
        // width is stored as 1,152 bytes per row and is not split any further.
        assert_eq!(plan.rank_spec_at(down).shape[2], 1152);
        let embed = plan.index_of("model.llm.embed.weight").unwrap();
        assert_eq!(plan.rank_spec_at(embed).shape[0], 50_256);
    }
}
