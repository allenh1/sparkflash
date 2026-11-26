// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The released Inkling-2-Small document parses into the shared architecture.

use sparkflash_model::architecture::{AttentionKind, Ffn, Residual, Score, WeightFormat};
use sparkflash_model::{Architecture, Part};

const CONFIG: &str = include_str!("data/inkling2_config.json");
const QUANT: &str = include_str!("data/inkling2_quant.json");

fn architecture() -> Architecture {
    let mut architecture = Architecture::from_json(CONFIG).expect("the released document parses");
    let quant: serde_json::Value = serde_json::from_str(QUANT).expect("the quant fixture parses");
    architecture
        .apply_quant_config(&quant)
        .expect("the quant document applies");
    architecture
}

#[test]
fn the_released_document_parses_its_hybrid_schedule_and_geometry() {
    let architecture = architecture();
    assert_eq!(architecture.family, "inkling2");
    assert_eq!(architecture.max_position_embeddings, 1_048_576);
    assert_eq!(architecture.hidden_size, 4608);
    assert_eq!(architecture.vocab_size, 201_024);
    assert_eq!(architecture.layers.len(), 48);
    let sliding: Vec<usize> = architecture
        .layers
        .iter()
        .filter(|layer| layer.attention.window == Some(1024))
        .map(|layer| layer.index)
        .collect();
    assert_eq!(sliding.len(), 40);
    assert!(!sliding.contains(&5));
    assert!(!sliding.contains(&47));
    for layer in &architecture.layers {
        assert!(matches!(
            layer.attention.kind,
            AttentionKind::Inkling { kv_heads: 8 }
        ));
        assert_eq!(layer.attention.rope.rotated_dims, 0);
        assert!(matches!(layer.residual, Residual::BlockAttention));
        let Ffn::Moe(moe) = &layer.ffn else {
            panic!("layer {} is not MoE", layer.index);
        };
        assert_eq!(moe.routed_experts, 384);
        assert_eq!(moe.active_experts, 8);
        assert_eq!(moe.intermediate, 2304);
        assert_eq!(moe.shared_experts, 0);
        assert_eq!(moe.score, Score::Sigmoid);
        assert_eq!(moe.routed_scaling, 8.0);
        assert!(moe.selection_bias);
    }
    let inkling = architecture.inkling.as_ref().expect("Inkling geometry");
    assert_eq!(inkling.d_rel, 16);
    assert_eq!(inkling.rel_extent, 1024);
    assert_eq!(inkling.short_conv, 4);
    assert_eq!(inkling.unpadded_vocab_size, 200_058);
    assert_eq!(inkling.final_logit_softcap, Some(30.0));
    assert_eq!(inkling.softcaps.gate, 4.0);
    assert_eq!(inkling.softcaps.up, 25.0);
    assert_eq!(
        inkling.layer_expert_formats[0],
        WeightFormat::NvFp4 {
            group_size: 16,
            wide: sparkflash_model::format::NvFp4Wide::Inkling
        }
    );
    assert_eq!(
        inkling.layer_expert_formats[40],
        WeightFormat::MxFp8Row { block: 32 }
    );
}

#[test]
fn the_cache_plan_charges_sliding_layers_once_and_full_layers_per_token() {
    let architecture = architecture();
    let plan = architecture.kv_plan(4, false).expect("the cache plan");
    // 8 full layers x 2 KV heads per rank x 128 values x 2 (K and V) x 2 bytes.
    assert_eq!(plan.per_token(), 8 * 2 * 128 * 2 * 2);
    // 40 sliding layers x 1023 retained positions at the same per-position width.
    assert_eq!(plan.fixed(), 40 * 1023 * 2 * 128 * 2 * 2);
}

#[test]
fn the_part_set_names_every_new_inkling_mechanism() {
    let parts = architecture().parts();
    for part in [
        // Inkling's text attention is the relative form; GqaAttention is another arm's body.
        Part::RelativeAttention,
        Part::ShortConv,
        Part::BlockAttentionResidual,
        Part::SoftcappedActivation,
        Part::Moe,
        Part::VisionPatchEmbed,
        Part::VisionPosition,
        Part::VisionNorm,
        Part::VisionAttention,
        Part::VisionMlp,
        Part::VisionMerger,
        Part::Mtp,
    ] {
        assert!(parts.contains(part), "{part} is missing");
    }
}

const MINI: &str = include_str!("data/inkling2_mini_config.json");
const MINI_QUANT: &str = include_str!("data/inkling2_mini_quant.json");

#[test]
fn the_looped_document_parses_its_core_and_its_message_parts() {
    let mut architecture = Architecture::from_json(MINI).expect("the looped document parses");
    let quant: serde_json::Value =
        serde_json::from_str(MINI_QUANT).expect("the quant fixture parses");
    architecture
        .apply_quant_config(&quant)
        .expect("the quant document applies");
    assert_eq!(architecture.family, "inkling2_loop");
    assert_eq!(architecture.hidden_size, 2304);
    assert_eq!(architecture.layers.len(), 24);
    let inkling = architecture.inkling.as_ref().expect("Inkling geometry");
    let plan = inkling.loop_plan.expect("the loop schedule");
    assert_eq!(plan.core_start, 6);
    assert_eq!(plan.core_end, 18);
    assert_eq!(plan.passes, 3);
    assert_eq!(plan.gate_rank, 64);
    assert_eq!(plan.streams, 4);
    // The released document's expert split: NVFP4 through layer 19, MXFP8 from 20.
    assert_eq!(
        inkling.layer_expert_formats[19],
        WeightFormat::NvFp4 {
            group_size: 16,
            wide: sparkflash_model::format::NvFp4Wide::Inkling
        }
    );
    assert_eq!(
        inkling.layer_expert_formats[20],
        WeightFormat::MxFp8Row { block: 32 }
    );
    let sliding: Vec<usize> = architecture
        .layers
        .iter()
        .filter(|layer| layer.attention.window == Some(1024))
        .map(|layer| layer.index)
        .collect();
    assert_eq!(sliding.len(), 20);
    assert!(sliding.contains(&0));
    assert!(!sliding.contains(&5));
    // The message path is present exactly on the twelve core layers, ahead of the attention.
    for step in architecture.steps() {
        let in_core = (6..18).contains(&step.index);
        assert_eq!(step.message.contains(&Part::CrossPassAttention), in_core);
        assert_eq!(step.message.contains(&Part::LoopMessageGate), in_core);
        if in_core {
            let part = Part::LoopMessageGate;
            let gate = architecture
                .stacks()
                .into_iter()
                .flat_map(|stack| stack.blocks)
                .find(|block| block.index == step.index)
                .and_then(|block| {
                    block
                        .parts
                        .into_iter()
                        .find(|instance| instance.part == part)
                })
                .expect("the core layer's message gate");
            assert_eq!(
                gate.facts.get("loop_streams"),
                Some(&sparkflash_model::stack::FactValue::Integer(4))
            );
        }
    }
    // The layout applied: the forward walk sees the inherited block geometry and the loop facts.
    let block = inkling
        .block_residual
        .expect("the block-attention residual");
    assert_eq!(block.layers_per_block, 6);
}
