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

//! Milestone 2A: the host sequence fixture and the comparison contract.
//!
//! These tests are weight-free and CUDA-free. They are the 2A acceptance: the committed CI
//! fixture reproduces byte-identically from its recorded command, the compared set is exactly the
//! last prefill position plus the decode positions, widening it fails and names D7, and the
//! engine's layer plan agrees with the kv CED schedule on all 40 backbone layers.

use std::path::PathBuf;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::layer_plan::{LayerMode, LayerPlan};
use sparkflash_engine::sequence_fixture::{
    CompareError, DeviceStep, FixtureWeightKind, FixtureWeights, LogitsCapture, SequenceFixture,
    StepKind, compare_logits, compare_steps, compared_positions, exclusion_boundary,
    generate_fixture,
};
use sparkflash_kv::{Csa2Mode, ced_schedule};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sequence-130-tiny.json")
}

fn load_fixture() -> (String, SequenceFixture) {
    let text = std::fs::read_to_string(fixture_path()).expect("the committed fixture is readable");
    let fixture: SequenceFixture =
        serde_json::from_str(&text).expect("the committed fixture parses");
    (text, fixture)
}

fn device_from(fixture: &SequenceFixture) -> Vec<DeviceStep> {
    fixture
        .steps
        .iter()
        .map(|step| DeviceStep {
            position: step.position,
            next_token: step.next_token,
            logits: step.logits.clone(),
            logits_sha256: Some(step.logits_sha256.clone()),
        })
        .collect()
}

#[test]
fn committed_ci_fixture_regenerates_byte_identically() {
    let (text, fixture) = load_fixture();
    let regenerated = generate_fixture(
        &fixture.config,
        &fixture.config_label,
        FixtureWeights::FiniteFp32 {
            seed: fixture.source.seed,
        },
        &fixture.prompt_tokens,
        fixture.decode_steps,
        LogitsCapture::All,
        fixture.provenance.clone(),
    )
    .expect("the CI fixture regenerates");
    let mut encoded =
        serde_json::to_string_pretty(&regenerated).expect("the regenerated fixture serialises");
    encoded.push('\n');
    assert_eq!(
        encoded, text,
        "the committed fixture must be exactly what its recorded command generates"
    );
}

#[test]
fn committed_ci_fixture_is_byte_identical_under_one_or_several_reference_threads() {
    // The parallel helpers split independent output work only, so the fixture must not depend on
    // how many threads the reference pass uses. One thread is the serial pass; two, four and eight
    // additionally take the threaded arms wherever the tiny configuration's sizes cross a helper's
    // threshold - here that is the MoE routed-expert and shared-expert arms, plus whatever
    // decoders/matmuls/norms the fixture's call graph reaches. The per-routine threaded arms
    // (decoders, matmul, apply_scales, rms_norm) are proven bit-identical by their own unit tests;
    // this test is the end-to-end byte check, not a per-routine coverage claim.
    let (text, fixture) = load_fixture();
    let generate = |threads: usize| {
        sparkflash_engine::parallel::with_reference_threads(threads, || {
            let regenerated = generate_fixture(
                &fixture.config,
                &fixture.config_label,
                FixtureWeights::FiniteFp32 {
                    seed: fixture.source.seed,
                },
                &fixture.prompt_tokens,
                fixture.decode_steps,
                LogitsCapture::All,
                fixture.provenance.clone(),
            )
            .expect("the CI fixture regenerates");
            let mut encoded =
                serde_json::to_string_pretty(&regenerated).expect("the fixture serialises");
            encoded.push('\n');
            encoded
        })
    };
    let serial = generate(1);
    assert_eq!(
        serial, text,
        "the serial regeneration must reproduce the fixture"
    );
    for threads in [2, 4, 8] {
        assert_eq!(
            generate(threads),
            serial,
            "the regeneration at {threads} reference threads must reproduce the serial bytes"
        );
    }
}

#[test]
fn the_ci_fixture_contract_is_pinned() {
    let (_, fixture) = load_fixture();
    assert_eq!(
        fixture.schema_version,
        sparkflash_engine::sequence_fixture::FIXTURE_SCHEMA_VERSION
    );
    assert_eq!(fixture.config_label, "tiny");
    assert_eq!(fixture.source.kind, FixtureWeightKind::FiniteFp32);
    assert_eq!(fixture.source.checkpoint, None);
    assert_eq!(fixture.prompt_tokens.len(), 130);
    assert_eq!(fixture.decode_steps, 8);
    assert_eq!(fixture.config, ModelConfig::tiny());
    assert_eq!(fixture.window, fixture.config.window_size);
    assert_eq!(fixture.steps.len(), 9);
    assert_eq!(fixture.steps[0].kind, StepKind::Prefill);
    assert!(
        fixture.steps[1..]
            .iter()
            .all(|step| step.kind == StepKind::Decode)
    );
    assert_eq!(
        fixture.compared_positions,
        vec![129, 130, 131, 132, 133, 134, 135, 136, 137]
    );
    assert!(
        fixture
            .prompt_tokens
            .iter()
            .all(|&token| (token as usize) < fixture.config.vocab_size),
        "every prompt token is inside the vocabulary"
    );
    for step in &fixture.steps {
        assert_eq!(step.logits_sha256.len(), 64, "a lowercase hex sha256");
        assert!(step.next_token.is_some(), "every step selected a token");
        assert!(
            step.logits
                .as_ref()
                .is_some_and(|logits| logits.iter().all(|value| value.is_finite())),
            "the CI fixture records full finite logits"
        );
    }
    assert!(fixture.kv_rows.is_empty(), "2C populates the KV digests");
    fixture
        .validate()
        .expect("the fixture is internally consistent");
}

#[test]
fn the_committed_fixture_passes_its_own_comparison() {
    let (_, fixture) = load_fixture();
    let device = device_from(&fixture);
    compare_steps(&fixture, &device).expect("the fixture's tokens match themselves");
    compare_logits(&fixture, &device, 0.0).expect("the fixture's logits match themselves exactly");
}

#[test]
fn widening_the_compared_set_fails_the_exclusion_test_and_names_d7() {
    let (_, fixture) = load_fixture();
    // Every position older than the last prefill position is refused, including positions inside
    // the sliding window: the engine's prefill returns logits only for its last position, and the
    // device CED path runs layers 21..=39 over the last window only (D7).
    for older in [0usize, 64, 122, 128] {
        let mut device = device_from(&fixture);
        device.push(DeviceStep::tokens(older, None));
        let error =
            compare_steps(&fixture, &device).expect_err("an older position is not comparable");
        assert!(
            matches!(error, CompareError::UnexpectedPosition { position } if position == older),
            "the rejection names the position"
        );
        let text = error.to_string();
        assert!(text.contains("D7"), "the rejection names D7: {text}");
        assert!(
            text.contains(&older.to_string()),
            "the rejection names the position: {text}"
        );
    }
    // The un-widened set passes, so the test is about the widening and not a broken fixture.
    compare_steps(&fixture, &device_from(&fixture)).expect("the exact set passes");
}

#[test]
fn the_released_window_boundary_is_two_positions_for_a_130_token_prompt() {
    let schedule = ced_schedule(130).expect("a 130-token prompt schedules");
    assert_eq!(schedule.window, 128);
    assert_eq!(schedule.exclusion_boundary(), 2);
    assert_eq!(exclusion_boundary(130, 128), 2);
    let positions = compared_positions(130, 8);
    assert_eq!(positions.len(), 9);
    assert_eq!(positions[0], 129);
    assert_eq!(positions[8], 137);
    assert!(
        positions.iter().all(|&position| position >= 129),
        "no compared position is older than the last prefill position"
    );
    // The decoder runs over the last 128 tokens (positions 2..=129 of the prompt), which is why
    // the two oldest prompt tokens have no decoder output at all.
    assert_eq!(schedule.decoder_tokens, 128);
    assert_eq!(schedule.encoder_tokens, 130);
}

#[test]
fn the_engine_layer_plan_agrees_with_the_kv_schedule_on_all_40_layers() {
    let config = ModelConfig::v41_flash().expect("the released config is valid");
    let plan = LayerPlan::from_config(&config).expect("the released layer plan");
    let schedule = ced_schedule(130).expect("a 130-token prompt schedules");
    assert_eq!(
        plan.layers().len(),
        43,
        "40 backbone layers plus 3 MTP stages"
    );
    for row in &schedule.layers {
        let spec = plan
            .layers()
            .iter()
            .find(|spec| spec.layer == row.layer)
            .expect("one spec per backbone layer");
        let expected_mode = match row.mode {
            Csa2Mode::SwaOnly => LayerMode::SwaOnly,
            Csa2Mode::Full => LayerMode::Full,
            Csa2Mode::Reindex => LayerMode::Reindex,
            Csa2Mode::Reuse => LayerMode::Reuse,
        };
        assert_eq!(spec.mode, expected_mode, "layer {} mode", row.layer);
        assert_eq!(
            spec.owns_kv, row.owns_main_kv,
            "layer {} owns_kv",
            row.layer
        );
        assert_eq!(
            spec.owns_index, row.owns_index,
            "layer {} owns_index",
            row.layer
        );
        assert_eq!(
            spec.kv_source, row.kv_source,
            "layer {} kv_source",
            row.layer
        );
        assert_eq!(
            spec.index_source, row.index_source,
            "layer {} index_source",
            row.layer
        );
        assert_eq!(
            spec.candidate_source, row.candidate_source,
            "layer {} candidate_source",
            row.layer
        );
        assert_eq!(
            spec.publishes_topk(),
            row.owns_index,
            "layer {} publishes_topk",
            row.layer
        );
    }
    for spec in plan.layers().iter().filter(|spec| spec.layer >= 40) {
        assert_eq!(
            spec.mode,
            LayerMode::Mtp,
            "MTP stage {} is out of scope",
            spec.layer
        );
    }
}

#[test]
fn the_fixture_schema_pins_the_two_weight_kinds() {
    // The node golden fixture must be distinguishable from the CI fixture by its serialised kind
    // alone, so a reader never mistakes one for the other.
    assert_eq!(
        serde_json::to_string(&FixtureWeightKind::FiniteFp32).expect("serialises"),
        "\"finite_fp32\""
    );
    assert_eq!(
        serde_json::to_string(&FixtureWeightKind::ReferenceCheckpoint).expect("serialises"),
        "\"reference_checkpoint\""
    );
}
