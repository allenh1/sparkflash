//! The Phase 1 one-token report's contract: forty rows, the five seams, the sink line, the
//! tensor audit, run determinism, and the logits class. A report that loses any of them fails here.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::report::{SEAM_LABELS, SMALL_SHAPE_LIMIT, WeightClass, one_token_report};
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

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

/// The numbers on the audit line, parsed so a test can check the partition rather than a prefix:
/// plan, read, unread, then the four class counts in line order (experts, stages, text-path,
/// indexer). The line's shape is part of the report's contract, so a missing token is a failure.
fn audit_numbers(report: &str) -> (usize, usize, usize, [usize; 4]) {
    let line = report
        .lines()
        .find(|line| line.starts_with("audit:"))
        .expect("the report has an audit line");
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let number = |key: &str| -> usize {
        let token = tokens
            .iter()
            .find(|token| token.starts_with(key))
            .unwrap_or_else(|| panic!("no {key} in the audit line: {line}"));
        token
            .strip_prefix(key)
            .expect("the prefix was matched")
            .trim_end_matches(',')
            .parse()
            .unwrap_or_else(|_| panic!("{key} is not a number in: {line}"))
    };
    let after = |label: &str| -> usize {
        let at = tokens
            .iter()
            .position(|token| *token == label)
            .unwrap_or_else(|| panic!("no {label} in the audit line: {line}"));
        tokens[at + 1]
            .trim_end_matches([',', ']'])
            .parse()
            .unwrap_or_else(|_| panic!("{label} is not followed by a number in: {line}"))
    };
    (
        number("plan="),
        number("read="),
        number("unread="),
        [
            after("expected-unread=[unselected-expert"),
            after("DSpark-stage"),
            after("text-path"),
            after("indexer"),
        ],
    )
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
    // At the released layer counts and tiny shapes the partition is exact and pinned: 40 layers x 2
    // unselected experts x 6 tensors = 480 expert tensors, 130 DSpark-stage tensors, no text-path
    // tensor and no indexer (the harness config clears compression), against a 2,253-tensor plan of
    // which 1,643 are read.
    let (plan, read, unread, classes) = audit_numbers(&report);
    assert_eq!((plan, read, unread), (2253, 1643, 610));
    assert_eq!(
        classes,
        [480, 130, 0, 0],
        "the four classes at released depth"
    );
    assert_eq!(
        classes.iter().sum::<usize>(),
        unread,
        "the printed classes partition the unread set:\n{report}"
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

/// A ratio-2 configuration skips its indexers at compress_len 0, so the audit has a real indexer
/// class to name there rather than a class that only exists on paper. The tiny configuration is
/// small enough to run the report end to end.
#[test]
fn the_audit_names_the_indexer_a_first_step_could_not_consult() {
    let config = ModelConfig::tiny();
    assert!(
        config.compress_ratios.iter().any(|&ratio| ratio > 0),
        "the tiny configuration has a compression layer"
    );
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 73);
    let report = one_token_report(
        &config,
        Box::new(source),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Bytes,
    )
    .expect("the tiny report runs and its unread indexer is classified");
    let (_, _, unread, classes) = audit_numbers(&report);
    assert_eq!(
        classes[3], 8,
        "the tiny plan's eight indexer tensors are unread"
    );
    assert_eq!(
        classes.iter().sum::<usize>(),
        unread,
        "the tiny configuration's classes partition its unread set:\n{report}"
    );
}

/// A vision-enabled configuration plans three more gate.bias_vl tensors, one per MTP stage, and
/// every mtp.* tensor is unread on a first step. The text-path class must count the backbone's
/// tensors only, or the printed classes stop partitioning the unread set in exactly the
/// configuration criterion 5 targets (the released plan has 43 bias_vl: 40 backbone + 3 stages).
#[test]
fn a_vision_enabled_configuration_keeps_the_audit_classes_disjoint() {
    let config = ModelConfig {
        vision_n_layers: Some(32),
        ..released_depth()
    };
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 79);
    let report = one_token_report(
        &config,
        Box::new(source),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Bytes,
    )
    .expect("the vision-enabled report runs");
    let (plan, read, unread, classes) = audit_numbers(&report);
    assert_eq!(
        (plan, read, unread),
        (2296, 1643, 653),
        "the vision plan's counts"
    );
    assert_eq!(classes, [480, 133, 40, 0], "text-path is backbone-only");
    assert_eq!(
        classes.iter().sum::<usize>(),
        unread,
        "the stage tensors are not double-counted as text-path:\n{report}"
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

/// A source that serves the Engram tables by range, like the CLI's loader, and refuses them whole.
/// The wrapped synthetic source is the only one a host without the checkpoint can use, so this is
/// the end-to-end check that the paged path and the report's audit agree with the whole-table path.
struct PagedSynthetic {
    inner: SyntheticWeights,
}

impl WeightSource for PagedSynthetic {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if name.ends_with(".engram.embed.weight") || name.ends_with(".engram.embed.scale") {
            return Err(WeightError::Refused {
                name: name.to_owned(),
                detail: "the paged test source serves these tables by range only".to_owned(),
            });
        }
        self.inner.read(name)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        let table = self
            .inner
            .read(name)
            .expect("the synthetic source holds the table");
        let mut out = Vec::new();
        for &(offset, len) in ranges {
            let start = offset as usize;
            out.extend_from_slice(&table[start..start + len]);
        }
        Some(Ok(out))
    }
}

#[test]
fn a_paged_engram_run_reports_exactly_what_the_whole_table_run_reports() {
    let mut config = released_depth();
    config.engram_layer_ids = vec![1, 14];
    config.engram_num_embeddings = vec![1024, 1024];
    let plan = TensorPlan::language_model(&config);
    let whole = one_token_report(
        &config,
        Box::new(SyntheticWeights::from_plan(&plan, 83)),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Bytes,
    )
    .expect("the whole-table engram run reports");
    let paged = one_token_report(
        &config,
        Box::new(PagedSynthetic {
            inner: SyntheticWeights::from_plan(&plan, 83),
        }),
        SMALL_SHAPE_LIMIT,
        0,
        WeightClass::Bytes,
    )
    .expect("the paged engram run reports; the table is refused whole, so a fallback fails here");
    assert_eq!(
        whole, paged,
        "the two table reads must produce the same rows, seams and audit"
    );
}
