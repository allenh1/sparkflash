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

//! The released Inkling-2-Small geometry and its TP4 rank-local tensor slices.

use sparkflash_model::architecture::{AttentionKind, Ffn, Residual, WeightFormat};
use sparkflash_model::{Target, TpPlan, built};
use sparkflash_model_inkling2::architecture::architecture_from_config;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::tensor::InklingTensorPlan;

fn released() -> Inkling2Config {
    Inkling2Config::from_json(
        include_str!("../config/inkling2_small.hf.json"),
        Some(include_str!("../config/inkling2_small.hf_quant.json")),
    )
    .expect("the released Inkling document parses")
}

#[test]
fn the_released_architecture_names_its_schedule() {
    let config = released();
    let architecture = architecture_from_config(&config);
    assert_eq!(architecture.layers.len(), 48);
    assert_eq!(config.local_layer_ids.len(), 40);
    assert_eq!(config.full_layer_ids, vec![5, 11, 17, 23, 29, 35, 41, 47]);
    assert!(architecture.layers.iter().all(|layer| {
        matches!(layer.attention.kind, AttentionKind::Inkling { kv_heads: 8 })
            && layer.attention.rope.rotated_dims == 0
            && matches!(layer.residual, Residual::BlockAttention)
    }));
    for layer in &architecture.layers {
        let Ffn::Moe(moe) = &layer.ffn else {
            panic!("layer {} is not MoE", layer.index);
        };
        assert_eq!(moe.routed_experts, 384);
        assert_eq!(moe.active_experts, 8);
        assert_eq!(moe.intermediate, 2304);
        assert_eq!(moe.score, sparkflash_model::architecture::Score::Sigmoid);
        assert_eq!(moe.routed_scaling, 8.0);
        assert!(moe.selection_bias);
    }
    assert_eq!(config.quant.nvfp4_layers, (0..40).collect::<Vec<_>>());
    assert_eq!(config.quant.mxfp8_layers, (40..48).collect::<Vec<_>>());
    assert!(
        config.quant.layer_expert_formats[..40]
            .iter()
            .all(|format| matches!(format, WeightFormat::NvFp4 { group_size: 16, .. }))
    );
    assert!(
        config.quant.layer_expert_formats[40..]
            .iter()
            .all(|format| matches!(format, WeightFormat::MxFp8 { block: 32 }))
    );
    assert_eq!(config.vision.as_ref().map(|vision| vision.depth), Some(23));
    assert_eq!(
        config.mtp.as_ref().map(|mtp| mtp.num_hidden_layers),
        Some(5)
    );
    assert_eq!(config.mtp.as_ref().map(|mtp| mtp.block_size), Some(8));
    assert!(architecture.missing_on(Target::Sm121).is_empty());
    assert_eq!(
        built(Target::Sm121).len(),
        sparkflash_model::Part::ALL.len()
    );
}

#[test]
fn the_tp4_rank_slices_divide_heads_experts_and_vocabulary() {
    let config = released();
    let plan = InklingTensorPlan::served(&config, TpPlan::new(4, 3).expect("TP4 rank 3"))
        .expect("the plan builds");
    let spec = |name: &str| {
        let index = plan.index_of(name).unwrap_or_else(|| panic!("{name}"));
        plan.rank_spec_at(index).shape.clone()
    };
    assert_eq!(
        spec("model.llm.layers.0.attn.wq_du.weight"),
        vec![12 * 128, 4608]
    );
    assert_eq!(
        spec("model.llm.layers.0.attn.wk_dv.weight"),
        vec![2 * 128, 4608]
    );
    assert_eq!(
        spec("model.llm.layers.0.attn.wr_du.weight"),
        vec![12 * 16, 4608]
    );
    assert_eq!(
        spec("model.llm.layers.0.attn.wo_ud.weight"),
        vec![4608, 12 * 128]
    );
    // Experts are split by expert, not by intermediate width as well: a rank holds 96 whole
    // experts, so the intermediate stays 2,304 and four ranks hold all 384 experts once.
    assert_eq!(
        spec("model.llm.layers.0.mlp.experts.w13_weight"),
        vec![96, 2 * 2304, 4608 / 2]
    );
    assert_eq!(
        spec("model.llm.layers.0.mlp.experts.w2_weight"),
        vec![96, 4608, 2304 / 2]
    );
    assert_eq!(
        spec("model.llm.layers.40.mlp.experts.w13_weight"),
        vec![96, 2 * 2304, 4608]
    );
    assert_eq!(
        spec("model.llm.layers.40.mlp.experts.w2_weight"),
        vec![96, 4608, 2304]
    );
    assert_eq!(spec("model.llm.embed.weight"), vec![50_256, 4608]);
    assert_eq!(spec("model.llm.unembed.weight"), vec![50_256, 4608]);
    // Norms, the hidden-width post convolutions, relative banks and block-residual queries
    // are replicated, not sliced.
    assert_eq!(spec("model.llm.norm.weight"), vec![4608]);
    assert_eq!(spec("model.llm.layers.0.mlp_norm.weight"), vec![4608]);
    assert_eq!(
        spec("model.llm.layers.0.attn_sconv.weight"),
        vec![4608, 1, 4]
    );
    assert_eq!(
        spec("model.llm.layers.0.attn.rel_logits_proj.proj"),
        vec![16, 1024]
    );
    assert_eq!(spec("model.llm.attn_res.attn_queries.4"), vec![3, 4608]);
    assert_eq!(spec("model.llm.attn_res.mlp_queries.7"), vec![4608]);
}
