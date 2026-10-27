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

//! M3-a chain-contract tests: the CED schedule derivation, the stage map, the planned seam counts,
//! the ledger self-check (missing **and** unplanned seams), and the labels a plan report must carry.
//!
//! These tests run on a CPU-only host: no driver is opened, no kernel is launched and no token is
//! produced. They pin the numbers the design states so a later wiring cannot quietly change them.

use sparkflash_device::chain::ChainError;
use sparkflash_device::chain::{
    ChainLabels, ChainOps, ChainSchedule, ChainSeamPlan, ChainStage, FIRST_CHAIN_LABEL,
    FUTURE_LABEL_PREFIX, HEAD_LABEL, HYBRID_TOKEN_LABEL, LAYER_STAGE_ORDER, MILESTONE_DECODE_STEPS,
    MILESTONE_PROMPT_TOKENS, PLAN_ONLY_LABEL, STEP_STAGE_ORDER, StageCall, TAIL_STAGE_ORDER,
    run_skeleton,
};
use sparkflash_kernels::sequence_chain::{SeamKind, SeamLedger, SeamRecord, SequencePlan};
use sparkflash_kv::config::{ENCODER_LAST_LAYER, NUM_HIDDEN_LAYERS, SWA_WINDOW_SIZE};

#[test]
fn the_milestone_schedule_is_5_482_token_layer_executions_derived_from_ced() {
    let schedule = ChainSchedule::milestone().expect("milestone schedule");
    assert_eq!(schedule.prompt_tokens, MILESTONE_PROMPT_TOKENS);
    assert_eq!(schedule.decode_steps, MILESTONE_DECODE_STEPS);
    assert_eq!(schedule.layers.len(), NUM_HIDDEN_LAYERS);
    // 21 layers x 130 prompt tokens + 19 layers x 128 window tokens.
    assert_eq!(
        schedule.prefill_token_layer_executions(),
        21 * 130 + 19 * 128
    );
    assert_eq!(schedule.prefill_token_layer_executions(), 5_162);
    assert_eq!(schedule.decode_token_layer_executions(), 8 * 40);
    assert_eq!(schedule.decode_token_layer_executions(), 320);
    assert_eq!(schedule.token_layer_executions(), 5_482);

    // The plain 130 x 40 + 8 x 40 product differs by exactly the positions the 19 decoder layers do
    // not run: they start at position 2, so each of the 8 decode steps still covers 40 layers but
    // the prefill is 21 x 130 + 19 x 128, not 130 x 40.
    let naive =
        MILESTONE_PROMPT_TOKENS * NUM_HIDDEN_LAYERS + MILESTONE_DECODE_STEPS * NUM_HIDDEN_LAYERS;
    assert_eq!(naive, 5_520);
    assert_eq!(naive - schedule.token_layer_executions(), 19 * (130 - 128));
    assert_eq!(naive - schedule.token_layer_executions(), 38);

    // Every extent agrees with the sequence plan the chain will follow.
    let plan = SequencePlan::ced(
        MILESTONE_PROMPT_TOKENS,
        NUM_HIDDEN_LAYERS,
        ENCODER_LAST_LAYER + 1,
        SWA_WINDOW_SIZE,
    )
    .expect("sequence plan");
    for (layer, row) in schedule.layers.iter().enumerate() {
        let extent = plan.extents[layer];
        assert_eq!(row.layer, layer);
        assert_eq!(
            row.first_position, extent.first_position,
            "layer {layer} first"
        );
        assert_eq!(row.tokens, extent.tokens, "layer {layer} tokens");
    }
    for row in &schedule.layers[..=ENCODER_LAST_LAYER] {
        assert_eq!(row.first_position, 0);
        assert_eq!(row.tokens, 130);
    }
    for row in &schedule.layers[ENCODER_LAST_LAYER + 1..] {
        assert_eq!(row.first_position, 2);
        assert_eq!(row.tokens, 128);
    }
}

#[test]
fn the_planned_seam_counts_are_derived_from_the_extents_and_layers() {
    let schedule = ChainSchedule::milestone().expect("schedule");
    let plan = ChainSeamPlan::for_schedule(&schedule);
    let count = |name: &str| {
        plan.entry(name)
            .unwrap_or_else(|| panic!("{name} missing from the plan"))
            .expected
    };
    assert_eq!(plan.entries.len(), 18);
    // Seams that run once per token-layer execution.
    for name in [
        "dispatch",
        "attn_norm",
        "attn_rope",
        "window_kv_roundtrip",
        "moe_routing_weight",
        "moe_swiglu_quant",
        "shared_expert",
        "ffn_epilogue",
    ] {
        assert_eq!(
            count(name),
            5_482,
            "{name} runs every token-layer execution"
        );
    }
    // Per-token-step seams: 130 prefill embeddings + 8 decode embeddings; the final norm and head
    // run once per output step (1 prefill + 8 decode).
    assert_eq!(count("embedding_gather"), 138);
    assert_eq!(count("final_norm"), 9);
    assert_eq!(count("lm_head_argmax"), 9);
    // Engram layers 1 and 14: (130 + 130) + 8 x 2.
    assert_eq!(count("engram_rw"), 276);
    // 38 ratio>0 layers: 19 x 130 + 19 x 128 + 8 x 38.
    assert_eq!(count("compressor_cache"), 5_206);
    // The 8 Full/Reindex layers: 4 x 130 + 4 x 128 + 8 x 8. The Top-512 merge runs on the same
    // layers (Reuse layers reuse the indices), and the expanded compressed-store view runs on every
    // ratio>0 layer, which is the compressor's set.
    assert_eq!(count("index_key_path"), 1_096);
    assert_eq!(count("indexer_weights"), 1_096);
    assert_eq!(count("top512_merge"), 1_096);
    assert_eq!(count("expanded_view"), 5_206);
    // Layer 20 plus the four Reindex layers: 130 + 4 x 128 + 8 x 5.
    assert_eq!(count("hsi_scoring"), 682);
    // Every planned seam record is host work with a phase-3 removal plan.
    for entry in &plan.entries {
        assert!(
            !entry.removal.is_empty(),
            "{} needs a removal plan",
            entry.name
        );
    }
    let total: usize = plan.entries.iter().map(|entry| entry.expected).sum();
    assert_eq!(total, 58_670, "sum of the planned seam records");

    // The plan/ledger self-check passes on a complete ledger and fails on a missing seam.
    let full = ledger_for(&plan, None);
    plan.validate(&full).expect("complete ledger validates");
    let missing = ledger_for(&plan, Some("hsi_scoring"));
    assert!(matches!(
        plan.validate(&missing).expect_err("missing seam"),
        ChainError::SeamCount {
            name: "hsi_scoring",
            expected: 682,
            found: 0,
            ..
        }
    ));
    let mut short = ledger_for(&plan, None);
    short = remove_one(short, "ffn_epilogue");
    assert!(matches!(
        plan.validate(&short).expect_err("short seam"),
        ChainError::SeamCount {
            name: "ffn_epilogue",
            found: 5_481,
            ..
        }
    ));
    let mut unplanned = ledger_for(&plan, None);
    unplanned.record(seam("not_in_the_plan", SeamKind::Other, true));
    assert!(matches!(
        plan.validate(&unplanned).expect_err("unplanned seam"),
        ChainError::UnplannedSeam {
            name: "not_in_the_plan",
            ..
        }
    ));
    let mut wrong_kind = ledger_for(&plan, None);
    wrong_kind = remove_one(wrong_kind, "engram_rw");
    wrong_kind.record(seam("engram_rw", SeamKind::Other, true));
    assert!(matches!(
        plan.validate(&wrong_kind).expect_err("kind"),
        ChainError::SeamKind {
            name: "engram_rw",
            ..
        }
    ));
    let mut no_removal = ledger_for(&plan, None);
    no_removal = remove_one(no_removal, "attn_norm");
    no_removal.record(seam("attn_norm", SeamKind::Other, false));
    assert!(matches!(
        plan.validate(&no_removal).expect_err("no removal"),
        ChainError::SeamWithoutRemoval { name: "attn_norm" }
    ));
}

#[test]
fn the_stage_map_labels_gate_up_and_the_router_as_device_measured_not_wired() {
    // Lead adjustment 1: gate/up is merged and measured in isolation (m10), not a host seam and not
    // wired into the chain; the router is device-proven one launch; the other device stages are
    // named but not launched.
    for stage in [ChainStage::MoeGateUp, ChainStage::MoeRouter] {
        assert!(
            stage.seam_name().is_none(),
            "{stage:?} must not be a host seam"
        );
        assert!(stage.kernel().is_some(), "{stage:?} names a device kernel");
    }
    assert_eq!(ChainStage::MoeGateUp.kernel(), Some("routed_gate_up_sm121"));
    assert_eq!(ChainStage::MoeRouter.kernel(), Some("moe_router_sm121"));
    assert_eq!(ChainStage::Attention.kernel(), Some("flash_mla_csa2_sm121"));
    assert_eq!(
        ChainStage::AttentionOutputProjection.kernel(),
        Some("b12x_mxfp8_wo_a_sm121")
    );
    assert_eq!(ChainStage::MhcEntry.kernel(), Some("mega_mhc_sm121"));
    assert_eq!(
        ChainStage::LmHeadScreen.kernel(),
        Some("cert_lm_head_sm121")
    );
    // Host seams that must stay host until their device stage exists.
    for stage in [
        ChainStage::Embedding,
        ChainStage::Engram,
        ChainStage::AttnNorm,
        ChainStage::AttnRope,
        ChainStage::WindowKv,
        ChainStage::CompressorCache,
        ChainStage::IndexKeyPath,
        ChainStage::IndexerWeights,
        ChainStage::HsiScoring,
        ChainStage::Top512Merge,
        ChainStage::ExpandedView,
        ChainStage::MoeRoutingWeight,
        ChainStage::MoeSwiGLUQuant,
        ChainStage::SharedExpertSwiGLUQuant,
        ChainStage::FfnEpilogue,
        ChainStage::FinalNorm,
        ChainStage::LmHeadArgmax,
    ] {
        assert!(stage.seam_name().is_some(), "{stage:?} is a host seam");
    }
    // The full literal order is the design 2.2 order, not just a first/last presence check: a
    // swap of any two adjacent stages must fail here.
    assert_eq!(
        LAYER_STAGE_ORDER,
        &[
            ChainStage::Engram,
            ChainStage::Dispatch,
            ChainStage::MhcEntry,
            ChainStage::AttnNorm,
            ChainStage::AttentionProjections,
            ChainStage::AttnRope,
            ChainStage::WindowKv,
            ChainStage::CompressorCache,
            ChainStage::IndexKeyPath,
            ChainStage::IndexerWeights,
            ChainStage::HsiScoring,
            ChainStage::Top512Merge,
            ChainStage::ExpandedView,
            ChainStage::Attention,
            ChainStage::AttentionOutputProjection,
            ChainStage::MhcPost,
            ChainStage::MoeRouter,
            ChainStage::MoeRoutingWeight,
            ChainStage::MoeGateUp,
            ChainStage::MoeSwiGLUQuant,
            ChainStage::MoeDownCombine,
            ChainStage::SharedExpertGate,
            ChainStage::SharedExpertSwiGLUQuant,
            ChainStage::FfnEpilogue,
        ]
    );
    // The head of the map cites the design 2.2 items it implements: mHC entry (item 2) comes
    // before the attention norm (items 2-3), projections and RoPE (item 3).
    let items: Vec<&str> = LAYER_STAGE_ORDER
        .iter()
        .map(|stage| stage.design_item())
        .collect();
    assert_eq!(
        &items[..6],
        [
            "2.2 item 1",
            "2.2 item 1",
            "2.2 item 2",
            "2.2 items 2-3",
            "2.2 item 3",
            "2.2 item 3",
        ]
    );
    assert_eq!(STEP_STAGE_ORDER, &[ChainStage::Embedding]);
    assert_eq!(
        TAIL_STAGE_ORDER,
        &[
            ChainStage::FinalNorm,
            ChainStage::LmHeadScreen,
            ChainStage::LmHeadArgmax,
        ]
    );
}

#[test]
fn the_skeleton_records_every_planned_seam_in_stage_order_and_no_execution() {
    let schedule = ChainSchedule::milestone().expect("schedule");
    let mut recording = Recording::default();
    let report = run_skeleton(&schedule, &mut recording).expect("skeleton");
    assert!(!report.chain_execution);
    assert!(!report.device_execution);
    assert!(!report.token_produced);
    assert_eq!(
        recording.ledger.records().len(),
        58_670,
        "every planned seam record was produced"
    );
    // The layer body runs only the stages applicable to that layer; the embedding runs once per
    // token step and the three tail stages once per output step. Derive the call count from the
    // same predicates the executor uses instead of hard-coding a total.
    let applicable = |layer: &sparkflash_device::chain::ChainLayer| {
        LAYER_STAGE_ORDER
            .iter()
            .filter(|stage| stage.runs_at(layer))
            .count()
    };
    let prefill_calls: usize = schedule
        .layers
        .iter()
        .map(|layer| layer.tokens * applicable(layer))
        .sum();
    let decode_calls: usize =
        schedule.decode_steps * schedule.layers.iter().map(applicable).sum::<usize>();
    let step_calls = 138 + 9 * 3;
    assert_eq!(
        recording.calls.len(),
        prefill_calls + decode_calls + step_calls
    );
    // 101,556 prefill + 6,296 decode layer calls + 138 embedding + 27 tail = 108,017. The
    // attention output projection (wo_a) is one more all-layer stage than the original map.
    assert_eq!(prefill_calls, 101_556);
    assert_eq!(decode_calls, 6_296);
    assert_eq!(recording.calls.len(), 108_017);
    // The first prompt token is embedded before any layer body; layer 0's first body stage is the
    // dispatch seam and layer 1 has the first Engram path. Decoder layers start at position 2.
    assert_eq!(recording.calls[0].0, 0);
    assert_eq!(recording.calls[0].1, 0);
    assert_eq!(recording.calls[0].2, ChainStage::Embedding);
    assert!(
        recording
            .calls
            .iter()
            .any(|(layer, position, stage)| *layer == 0
                && *position == 0
                && *stage == ChainStage::Dispatch)
    );
    assert!(
        recording
            .calls
            .iter()
            .any(|(layer, position, stage)| *layer == 1
                && *position == 0
                && *stage == ChainStage::Engram)
    );
    let decoder_first = recording
        .calls
        .iter()
        .find(|(layer, _, _)| *layer == 21)
        .expect("decoder layer 21");
    assert_eq!(decoder_first.1, 2);
    let last = recording.calls.last().expect("last call");
    assert_eq!(last.2, ChainStage::LmHeadArgmax);
    assert_eq!(last.1, 137);
    // The report prints the ledger summary before the labels and makes no execution claim. The
    // only label stated as a state is the plan-only label; the run labels are future-only text.
    let lines = report.report_lines();
    let label_index = lines
        .iter()
        .position(|line| line == &format!("{FUTURE_LABEL_PREFIX} {HYBRID_TOKEN_LABEL}"))
        .expect("future label");
    let ledger_index = lines
        .iter()
        .position(|line| line.contains("seam ledger total"))
        .expect("ledger");
    assert!(ledger_index < label_index);
    assert!(
        lines
            .iter()
            .any(|line| line == &format!("label: {PLAN_ONLY_LABEL}"))
    );
    assert!(lines.iter().all(|line| !line.starts_with("label: hybrid:")));
    assert!(
        lines
            .iter()
            .any(|line| line.contains("no comparison was made"))
    );
    report
        .seam_plan
        .validate(&report.ledger)
        .expect("self-check");
}

#[test]
fn the_report_labels_are_the_design_wording_verbatim() {
    assert_eq!(
        HYBRID_TOKEN_LABEL,
        "hybrid: token produced by host math over device logits"
    );
    assert_eq!(FIRST_CHAIN_LABEL, "first chain execution, unverified");
    assert_eq!(
        HEAD_LABEL,
        "NOT YET DEVICE-EXECUTED AS A PRUNING PATH: one stage-5 launch per rank, 0 certified, never compared against the chain reference"
    );
    assert!(PLAN_ONLY_LABEL.contains("no chain execution"));
    let run_labels = ChainLabels::run_lines();
    assert!(run_labels.contains(&HYBRID_TOKEN_LABEL));
    assert!(run_labels.contains(&FIRST_CHAIN_LABEL));
    assert!(run_labels.contains(&HEAD_LABEL));
    assert!(!run_labels.contains(&PLAN_ONLY_LABEL));
    let plan_labels = ChainLabels::plan_lines();
    assert_eq!(plan_labels[0], format!("label: {PLAN_ONLY_LABEL}"));
    for (line, label) in
        plan_labels[1..]
            .iter()
            .zip([HYBRID_TOKEN_LABEL, FIRST_CHAIN_LABEL, HEAD_LABEL])
    {
        assert_eq!(line, &format!("{FUTURE_LABEL_PREFIX} {label}"));
    }
    assert!(
        plan_labels
            .iter()
            .all(|line| !line.contains("label: hybrid:"))
    );
}

#[test]
fn the_skeleton_refuses_a_missing_or_extra_stage_seam() {
    let schedule = ChainSchedule::milestone().expect("schedule");
    let mut missing = Recording {
        skip_seam: Some("ffn_epilogue"),
        ..Recording::default()
    };
    assert!(matches!(
        run_skeleton(&schedule, &mut missing).expect_err("missing"),
        ChainError::MissingStageSeam {
            stage: ChainStage::FfnEpilogue,
            ..
        }
    ));
    let mut extra = Recording {
        extra_seam: true,
        ..Recording::default()
    };
    assert!(matches!(
        run_skeleton(&schedule, &mut extra).expect_err("extra"),
        ChainError::NonSeamRecord {
            stage: ChainStage::Attention,
            ..
        }
    ));
    let mut wrong_kind = Recording {
        wrong_kind: true,
        ..Recording::default()
    };
    assert!(matches!(
        run_skeleton(&schedule, &mut wrong_kind).expect_err("kind"),
        ChainError::SeamKind {
            name: "attn_norm",
            ..
        }
    ));
    let mut swapped = Recording {
        swap_names: true,
        ..Recording::default()
    };
    assert!(matches!(
        run_skeleton(&schedule, &mut swapped).expect_err("swapped seam name"),
        ChainError::SeamName {
            stage: ChainStage::AttnNorm,
            expected: "attn_norm",
            found: "attn_rope",
            ..
        }
    ));
    let mut no_removal = Recording {
        no_removal: true,
        ..Recording::default()
    };
    assert!(matches!(
        run_skeleton(&schedule, &mut no_removal).expect_err("removal"),
        ChainError::SeamWithoutRemoval { name: "attn_norm" }
    ));
}

fn seam(name: &'static str, kind: SeamKind, phase3_followup: bool) -> SeamRecord {
    SeamRecord {
        name,
        kind,
        host_bytes_in: 1,
        host_bytes_out: 0,
        transfer_bytes: 0,
        phase3_followup,
        note: "test seam",
    }
}

fn kind_of(name: &str) -> SeamKind {
    match name {
        "compressor_cache" => SeamKind::Compressor,
        "top512_merge" => SeamKind::IndexerSelection,
        "engram_rw" => SeamKind::Engram,
        _ => SeamKind::Other,
    }
}

fn ledger_for(plan: &ChainSeamPlan, skip: Option<&str>) -> SeamLedger {
    let mut ledger = SeamLedger::default();
    for entry in &plan.entries {
        if skip == Some(entry.name) {
            continue;
        }
        for _ in 0..entry.expected {
            ledger.record(seam(entry.name, entry.kind, true));
        }
    }
    ledger
}

fn remove_one(ledger: SeamLedger, name: &str) -> SeamLedger {
    let mut kept = SeamLedger::default();
    let mut removed = false;
    for record in ledger.records() {
        if !removed && record.name == name {
            removed = true;
            continue;
        }
        kept.record(record.clone());
    }
    assert!(removed, "removed one {name}");
    kept
}

#[derive(Default)]
struct Recording {
    calls: Vec<(usize, usize, ChainStage)>,
    ledger: SeamLedger,
    skip_seam: Option<&'static str>,
    extra_seam: bool,
    wrong_kind: bool,
    no_removal: bool,
    swap_names: bool,
}

impl ChainOps for Recording {
    fn stage(&mut self, call: StageCall) -> Result<Option<SeamRecord>, ChainError> {
        self.calls.push((call.layer, call.position, call.stage));
        if self.extra_seam && call.stage == ChainStage::Attention {
            return Ok(Some(seam("extra_device_seam", SeamKind::Other, true)));
        }
        let Some(name) = call.stage.seam_name() else {
            return Ok(None);
        };
        if self.skip_seam == Some(name) {
            return Ok(None);
        }
        // A swap keeps both planned seam counts equal, so only the stage-to-name check can catch
        // it. This is the mutation the plan-vs-ledger self-check alone misses.
        let record_name = if self.swap_names {
            match name {
                "attn_norm" => "attn_rope",
                "attn_rope" => "attn_norm",
                other => other,
            }
        } else {
            name
        };
        let kind = if self.wrong_kind && record_name == "attn_norm" {
            SeamKind::Engram
        } else {
            kind_of(record_name)
        };
        let phase3 = !(self.no_removal && record_name == "attn_norm");
        let record = seam(record_name, kind, phase3);
        self.ledger.record(record.clone());
        Ok(Some(record))
    }
}
