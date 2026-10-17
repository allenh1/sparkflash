//! The Phase 1 one-token report's contract: forty rows, the five seams, the sink line, the
//! tensor audit, run determinism, and the logits class. A report that loses any of them fails here.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::report::{SEAM_LABELS, SMALL_SHAPE_LIMIT, WeightClass, one_token_report};
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::TensorPlan;

/// The released layer counts at small shapes: forty backbone layers and three MTP stages, so the
/// row rule is checked at the shape the roadmap states instead of at [ModelConfig::tiny].
fn released_depth() -> ModelConfig {
    ModelConfig {
        n_layers: 40,
        n_mtp_layers: 3,
        compress_ratios: vec![0; 43],
        kv_source_layers: Vec::new(),
        index_source_layers: Vec::new(),
        candidate_source_layer: 0,
        engram_layer_ids: Vec::new(),
        engram_num_embeddings: Vec::new(),
        dspark_target_layer_ids: vec![37, 38, 39],
        ..ModelConfig::tiny()
    }
}

#[test]
fn the_one_token_report_has_forty_rows_seams_sink_audit_and_determinism() {
    let config = released_depth();
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 71);
    let report = one_token_report(
        &config,
        Box::new(source),
        SMALL_SHAPE_LIMIT,
        3,
        WeightClass::Bytes,
    )
    .expect("the one-token report runs at the released layer counts");

    assert!(
        report.contains("rows: 40 (a first step records n_layers=40 backbone rows)"),
        "the row count line is the report's first contract:\n{report}"
    );
    let rows: Vec<&str> = report
        .lines()
        .filter(|line| line.starts_with("row "))
        .collect();
    assert_eq!(rows.len(), 40, "one row per backbone layer:\n{report}");
    for (expected, line) in rows.iter().enumerate() {
        assert!(
            line.starts_with(&format!("row {expected}: tensors=18/18")),
            "row {expected} is in order with the documented tensor set: {line}"
        );
    }

    assert!(
        report.contains("determinism: two runs byte-identical"),
        "two runs of the same token are bit-identical:\n{report}"
    );
    assert!(
        report.contains("logits: n=128"),
        "the logits line is printed"
    );
    assert!(report.contains("argmax="), "the argmax is printed");
    assert!(
        report.contains("audit: plan="),
        "the tensor audit is printed"
    );
    assert!(
        report.contains("expected-unread=[unselected-expert"),
        "the audit classifies what it did not read:\n{report}"
    );
    assert!(
        report.contains("logits class: the weight source is bytes, not numbers"),
        "a byte source says the logits are not a prediction:\n{report}"
    );

    for seam in SEAM_LABELS {
        assert!(
            report.contains(seam),
            "the run labels every seam; missing: {seam}\n{report}"
        );
    }
    assert_eq!(SEAM_LABELS.len(), 6, "five seams plus the sink line");
    let sink = SEAM_LABELS
        .iter()
        .find(|line| line.starts_with("sink:"))
        .expect("the sink line is part of the labels");
    assert!(
        sink.contains("consumed on both sides"),
        "the sink line says what the kernel and the reference do: {sink}"
    );
}

#[test]
fn the_one_token_report_refuses_a_configuration_without_a_boundary_table() {
    let mut config = released_depth();
    config.dim = 100; // not a multiple of the 32-element quantisation block
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 71);
    let error = one_token_report(
        &config,
        Box::new(source),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Bytes,
    )
    .expect_err("dim 100 does not divide into 32-element blocks");
    assert!(
        error.to_string().contains("boundary capture"),
        "the refusal names the boundary capture: {error}"
    );
}

#[test]
fn a_numbers_class_run_refuses_non_finite_logits_from_a_byte_source() {
    let config = released_depth();
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 71);
    let error = one_token_report(
        &config,
        Box::new(source),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Numbers,
    )
    .expect_err("synthetic bytes are not numbers, so a Numbers-class run must refuse");
    assert!(
        error.to_string().contains("not finite"),
        "the refusal names the non-finite logits: {error}"
    );
}
