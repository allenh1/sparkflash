//! The layer plan, built from the released configuration and checked against independent evidence.
//!
//! Most tests parse `config/deepseek_v41_flash.released.json` - the checkpoint's own
//! `config.json`, vendored and sha256-pinned when the released reader landed - and check the
//! plan against the arrays in that document: the ratios, the four id lists and the window. One test,
//! `the_pinned_table_covers_all_43_rows`, pins all 43 rows as literals instead: a table
//! re-derived by the same rule as [`LayerPlan`] would agree with a wrong plan, so the pinned rows are
//! the independent transcription that breaks the symmetry.

use std::collections::BTreeMap;

use serde_json::Value;
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::layer_plan::{CompressionKind, LayerMode, LayerPlan, LayerSpec, PlanError};

/// The released checkpoint's `config.json`, byte-for-byte (sha256 in `config/README.md`).
const RELEASED: &str = include_str!("../config/deepseek_v41_flash.released.json");

fn document() -> Value {
    serde_json::from_str(RELEASED).expect("the released fixture is valid JSON")
}

fn integer(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("an unsigned integer")).expect("a usize")
}

fn integers(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(integer)
        .collect()
}

fn released_config() -> ModelConfig {
    ModelConfig::from_released_json(RELEASED).expect("the released configuration must parse")
}

/// The released plan's rows, for tests that break one field and hand the rows back to
/// [`LayerPlan::from_specs`].
fn released_specs() -> Vec<LayerSpec> {
    LayerPlan::from_config(&released_config())
        .expect("the released configuration must plan")
        .layers()
        .to_vec()
}

/// A configuration-level error must be refused by both constructors: a caller who hands in the
/// released rows cannot launder a broken configuration through `from_specs`.
fn refused_by_both(config: &ModelConfig, specs: &[LayerSpec]) -> PlanError {
    let from_config = LayerPlan::from_config(config).expect_err("the configuration must not plan");
    let from_specs = LayerPlan::from_specs(config, specs.to_vec())
        .expect_err("the specs must not plan under this configuration");
    assert_eq!(
        from_config, from_specs,
        "both constructors must report the same error"
    );
    from_config
}

/// One literal row of the released table:
/// `(mode, ratio, kv_source, index_source, candidate_source, engram, dspark_target)`.
///
/// The values are transcribed from the released document and the architecture docs, not computed by
/// the rule under test. Rows 40, 41 and 42 are the three MTP stages.
type PinnedRow = (
    LayerMode,
    usize,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    bool,
    bool,
);

const PINNED: [PinnedRow; 43] = [
    (LayerMode::SwaOnly, 0, None, None, None, false, false),
    (LayerMode::SwaOnly, 0, None, None, None, true, false),
    (LayerMode::Full, 2, None, None, None, false, false),
    (LayerMode::Reuse, 2, Some(2), Some(2), None, false, false),
    (LayerMode::Reuse, 2, Some(2), Some(2), None, false, false),
    (LayerMode::Reuse, 2, Some(2), Some(2), None, false, false),
    (LayerMode::Reuse, 2, Some(2), Some(2), None, false, false),
    (LayerMode::Reuse, 2, Some(2), Some(2), None, false, false),
    (LayerMode::Full, 2, None, None, None, false, false),
    (LayerMode::Reuse, 2, Some(8), Some(8), None, false, false),
    (LayerMode::Reuse, 2, Some(8), Some(8), None, false, false),
    (LayerMode::Reuse, 2, Some(8), Some(8), None, false, false),
    (LayerMode::Reuse, 2, Some(8), Some(8), None, false, false),
    (LayerMode::Reuse, 2, Some(8), Some(8), None, false, false),
    (LayerMode::Full, 2, None, None, None, true, false),
    (LayerMode::Reuse, 2, Some(14), Some(14), None, false, false),
    (LayerMode::Reuse, 2, Some(14), Some(14), None, false, false),
    (LayerMode::Reuse, 2, Some(14), Some(14), None, false, false),
    (LayerMode::Reuse, 2, Some(14), Some(14), None, false, false),
    (LayerMode::Reuse, 2, Some(14), Some(14), None, false, false),
    (LayerMode::Full, 1, None, None, Some(20), false, false),
    (LayerMode::Reuse, 1, Some(20), Some(20), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(20), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(20), None, false, false),
    (
        LayerMode::Reindex,
        1,
        Some(20),
        None,
        Some(20),
        false,
        false,
    ),
    (LayerMode::Reuse, 1, Some(20), Some(24), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(24), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(24), None, false, false),
    (
        LayerMode::Reindex,
        1,
        Some(20),
        None,
        Some(20),
        false,
        false,
    ),
    (LayerMode::Reuse, 1, Some(20), Some(28), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(28), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(28), None, false, false),
    (
        LayerMode::Reindex,
        1,
        Some(20),
        None,
        Some(20),
        false,
        false,
    ),
    (LayerMode::Reuse, 1, Some(20), Some(32), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(32), None, false, false),
    (LayerMode::Reuse, 1, Some(20), Some(32), None, false, false),
    (
        LayerMode::Reindex,
        1,
        Some(20),
        None,
        Some(20),
        false,
        false,
    ),
    (LayerMode::Reuse, 1, Some(20), Some(36), None, false, true),
    (LayerMode::Reuse, 1, Some(20), Some(36), None, false, true),
    (LayerMode::Reuse, 1, Some(20), Some(36), None, false, true),
    (LayerMode::Mtp, 0, None, None, None, false, false),
    (LayerMode::Mtp, 0, None, None, None, false, false),
    (LayerMode::Mtp, 0, None, None, None, false, false),
];

/// Every per-row fact, checked against the released document's own arrays.
#[test]
fn the_plan_matches_the_released_document_layer_by_layer() {
    let document = document();
    let text = &document["text_config"];
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");

    let n_layers = integer(&text["num_hidden_layers"]);
    let n_mtp_layers = integer(&text["num_nextn_predict_layers"]);
    let window = integer(&text["sliding_window"]);
    let ratios = integers(&text["compress_ratios"]);
    let kv_sources = integers(&text["kv_source_layer_ids"]);
    let index_sources = integers(&text["index_source_layer_ids"]);
    let candidate = integer(&text["candidate_source_layer_id"]);
    let engrams = integers(&text["engram_layer_ids"]);
    let targets = integers(&text["dspark_target_layer_ids"]);

    // The released facts this plan is built against. These are read from the released document
    // itself, so a stale or hand-edited fixture cannot certify the plan by agreeing with itself.
    assert_eq!(n_layers, 40);
    assert_eq!(n_mtp_layers, 3);
    assert_eq!(window, 128);
    assert_eq!(integer(&text["index_topk"]), 512);
    assert_eq!(integer(&text["num_attention_heads"]), 64);
    assert_eq!(integer(&text["num_key_value_heads"]), 1);
    assert_eq!(integer(&text["head_dim"]), 512);
    assert_eq!(integer(&text["qk_rope_head_dim"]), 64);
    assert_eq!(ratios.len(), n_layers + n_mtp_layers);
    assert_eq!(ratios[0], 0, "layer 0 is uncompressed");
    assert_eq!(ratios[1], 0, "layer 1 is uncompressed");
    assert_eq!(&ratios[2..20], &[2; 18], "layers 2..19 pool two tokens");
    assert_eq!(
        &ratios[20..40],
        &[1; 20],
        "layers 20..39 compress one-to-one"
    );
    assert_eq!(
        &ratios[40..],
        &[0; 3],
        "the three MTP stages are window-only"
    );
    assert_eq!(kv_sources, vec![2, 8, 14, 20]);
    assert_eq!(index_sources, vec![2, 8, 14, 20, 24, 28, 32, 36]);
    assert_eq!(candidate, 20);
    assert_eq!(engrams, vec![1, 14]);
    assert_eq!(targets, vec![37, 38, 39]);

    assert_eq!(plan.len(), n_layers + n_mtp_layers);
    assert_eq!(
        plan.order(),
        (0..n_layers + n_mtp_layers).collect::<Vec<_>>()
    );
    assert_eq!(plan.candidate_source(), Some(candidate));

    let geometry = plan.attention();
    assert_eq!(geometry.n_heads, integer(&text["num_attention_heads"]));
    assert_eq!(
        geometry.n_kv_heads,
        integer(&text["num_key_value_heads"]),
        "the released reader checks the MLA has one KV head"
    );
    assert_eq!(geometry.head_dim, integer(&text["head_dim"]));
    assert_eq!(geometry.rope_head_dim, integer(&text["qk_rope_head_dim"]));
    assert_eq!(geometry.window, window);
    assert_eq!(geometry.index_topk, integer(&text["index_topk"]));

    for spec in plan.layers() {
        let layer = spec.layer;
        let stage = layer >= n_layers;
        let ratio = ratios[layer];
        assert_eq!(spec.compress_ratio, ratio, "layer {layer} ratio");
        let expected_kind = match ratio {
            0 => CompressionKind::Uncompressed,
            1 => CompressionKind::Direct,
            _ => CompressionKind::Pooled,
        };
        assert_eq!(
            spec.compression, expected_kind,
            "layer {layer} compression kind"
        );
        assert_eq!(
            spec.engram,
            engrams.contains(&layer),
            "layer {layer} Engram"
        );
        assert_eq!(
            spec.dspark_target,
            targets.contains(&layer),
            "layer {layer} DSpark target"
        );

        let owns_kv = !stage && kv_sources.contains(&layer);
        let owns_index = !stage && index_sources.contains(&layer);
        let expected_mode = if stage {
            LayerMode::Mtp
        } else if ratio == 0 {
            LayerMode::SwaOnly
        } else if owns_kv {
            LayerMode::Full
        } else if owns_index {
            LayerMode::Reindex
        } else {
            LayerMode::Reuse
        };
        assert_eq!(spec.mode, expected_mode, "layer {layer} mode");
        assert_eq!(spec.owns_kv, owns_kv, "layer {layer} owns KV");
        assert_eq!(spec.owns_index, owns_index, "layer {layer} owns index");
        assert_eq!(
            spec.publishes_topk(),
            owns_index,
            "layer {layer} publishes Top-K"
        );

        let expected_kv = if stage || ratio == 0 || owns_kv {
            None
        } else {
            kv_sources
                .iter()
                .copied()
                .filter(|&source| source < layer)
                .max()
        };
        assert_eq!(spec.kv_source, expected_kv, "layer {layer} KV source");

        let expected_index = if stage || ratio == 0 || owns_index {
            None
        } else {
            index_sources
                .iter()
                .copied()
                .filter(|&source| source < layer)
                .max()
        };
        assert_eq!(
            spec.index_source, expected_index,
            "layer {layer} index source"
        );

        let expected_candidate = (!stage && owns_index && layer >= candidate).then_some(candidate);
        assert_eq!(
            spec.candidate_source, expected_candidate,
            "layer {layer} candidate source"
        );

        let expected_offset = (ratio > 0).then_some(window);
        assert_eq!(spec.kv_offset, expected_offset, "layer {layer} KV offset");
        if ratio > 0 {
            assert_eq!(
                spec.kv_offset_at(0, 1),
                Some(1),
                "layer {layer} prefill offset is the chunk length"
            );
            assert_eq!(
                spec.kv_offset_at(7, 1),
                Some(window),
                "layer {layer} decode offset is the window"
            );
        } else {
            assert_eq!(
                spec.kv_offset_at(7, 1),
                None,
                "layer {layer} has no compressed KV"
            );
        }
    }

    let ids = |mode: LayerMode| {
        plan.layers()
            .iter()
            .filter(|spec| spec.mode == mode)
            .map(|spec| spec.layer)
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(LayerMode::SwaOnly), vec![0, 1], "layers 0 and 1");
    assert_eq!(
        ids(LayerMode::Full),
        kv_sources,
        "the Full layers are exactly the KV sources"
    );
    assert_eq!(
        ids(LayerMode::Reindex),
        index_sources
            .iter()
            .copied()
            .filter(|layer| !kv_sources.contains(layer))
            .collect::<Vec<_>>(),
        "the Reindex layers are the index sources that do not own KV"
    );
    assert_eq!(
        ids(LayerMode::Mtp),
        (n_layers..n_layers + n_mtp_layers).collect::<Vec<_>>(),
        "the stages are rows 40, 41 and 42"
    );
    assert_eq!(ids(LayerMode::SwaOnly).len(), 2);
    assert_eq!(ids(LayerMode::Full).len(), 4);
    assert_eq!(ids(LayerMode::Reindex).len(), 4);
    assert_eq!(ids(LayerMode::Reuse).len(), 30);
    assert_eq!(ids(LayerMode::Mtp).len(), 3);
}

/// The whole 43-row table, pinned literally and then checked row by row.
///
/// This is deliberately not derived from the released arrays by the plan's own rule: the pinned
/// values are an independent transcription, so a wrong source rule, mode rule or candidate rule in
/// `layer_plan.rs` shows up as a diff against this table instead of agreeing with a copy of itself.
#[test]
fn the_pinned_table_covers_all_43_rows() {
    let document = document();
    let text = &document["text_config"];
    let n_layers = integer(&text["num_hidden_layers"]);
    let n_mtp_layers = integer(&text["num_nextn_predict_layers"]);
    let ratios = integers(&text["compress_ratios"]);
    let kv_sources = integers(&text["kv_source_layer_ids"]);
    let index_sources = integers(&text["index_source_layer_ids"]);
    let engrams = integers(&text["engram_layer_ids"]);
    let targets = integers(&text["dspark_target_layer_ids"]);

    assert_eq!(
        PINNED.len(),
        n_layers + n_mtp_layers,
        "the pinned table is the whole plan"
    );
    let plan =
        LayerPlan::from_config(&released_config()).expect("the released configuration must plan");
    assert_eq!(plan.len(), PINNED.len());

    for (layer, expected) in PINNED.iter().enumerate() {
        let (mode, ratio, kv_source, index_source, candidate_source, engram, dspark_target) =
            *expected;
        let spec = plan.get(layer).expect("every pinned row is planned");
        assert_eq!(spec.layer, layer, "layer {layer} row index");
        assert_eq!(spec.mode, mode, "layer {layer} mode");
        assert_eq!(spec.compress_ratio, ratio, "layer {layer} ratio");
        assert_eq!(
            spec.compress_ratio, ratios[layer],
            "layer {layer} released ratio"
        );
        assert_eq!(spec.kv_source, kv_source, "layer {layer} KV source");
        assert_eq!(
            spec.index_source, index_source,
            "layer {layer} index source"
        );
        assert_eq!(
            spec.candidate_source, candidate_source,
            "layer {layer} candidate source"
        );
        assert_eq!(spec.engram, engram, "layer {layer} Engram");
        assert_eq!(
            spec.dspark_target, dspark_target,
            "layer {layer} DSpark tap"
        );

        // The independent ownership fields must follow the released id lists, not the mode: in the
        // released model every Full row also owns an indexer, a Reindex row owns only the index, and
        // a stage owns neither. The custom-configuration tests below exercise the fourth
        // combination, (owns_kv, owns_index) = (true, false), which the released lists do not use.
        let stage = layer >= n_layers;
        assert_eq!(
            spec.owns_kv,
            !stage && kv_sources.contains(&layer),
            "layer {layer} owns KV"
        );
        assert_eq!(
            spec.owns_index,
            !stage && index_sources.contains(&layer),
            "layer {layer} owns index"
        );
        assert_eq!(
            spec.owns_kv,
            mode == LayerMode::Full,
            "layer {layer} Full owns KV"
        );
        assert_eq!(
            spec.publishes_topk(),
            matches!(mode, LayerMode::Full | LayerMode::Reindex),
            "layer {layer} publishes Top-K"
        );
        assert_eq!(
            spec.engram,
            engrams.contains(&layer),
            "layer {layer} released Engram"
        );
        assert_eq!(
            spec.dspark_target,
            targets.contains(&layer),
            "layer {layer} released DSpark target"
        );
        assert_eq!(
            spec.compression,
            CompressionKind::for_ratio(ratio),
            "layer {layer} compression kind"
        );
        let expected_offset = (ratio > 0).then_some(integer(&text["sliding_window"]));
        assert_eq!(spec.kv_offset, expected_offset, "layer {layer} KV offset");
    }
}

/// A reader's KV and index sources must come earlier in the execution order.
#[test]
fn every_kv_and_index_source_precedes_the_layer_that_reads_it() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    assert!(
        plan.is_topologically_ordered(),
        "the plan's own order must satisfy every edge"
    );

    let position: BTreeMap<usize, usize> = plan
        .order()
        .iter()
        .copied()
        .enumerate()
        .map(|(position, layer)| (layer, position))
        .collect();
    assert_eq!(position.len(), plan.len(), "the order names each row once");

    for spec in plan.layers() {
        let reader = position[&spec.layer];
        if let Some(source) = spec.kv_source {
            let producer = position[&source];
            assert!(
                producer < reader,
                "layer {} reads KV from layer {source}, but the execution order places the source at {producer} and the reader at {reader}",
                spec.layer
            );
            let source_spec = plan.get(source).expect("a source is planned");
            assert!(source_spec.owns_kv, "layer {source} must own KV");
            assert_eq!(
                source_spec.compress_ratio, spec.compress_ratio,
                "layer {} and its KV source {source} must share a ratio",
                spec.layer
            );
        }
        if let Some(source) = spec.index_source {
            let producer = position[&source];
            assert!(
                producer < reader,
                "layer {} reads indices from layer {source}, but the execution order places the source at {producer} and the reader at {reader}",
                spec.layer
            );
            let source_spec = plan.get(source).expect("a source is planned");
            assert!(
                source_spec.publishes_topk(),
                "layer {source} must publish Top-K"
            );
            assert_eq!(
                source_spec.compress_ratio, spec.compress_ratio,
                "layer {} and its index source {source} must share a ratio",
                spec.layer
            );
        }
    }

    // The two source kinds are not the same relation: after layer 20 the nearest index source is
    // the group's Reindex layer, while the KV still comes from layer 20. A plan that collapsed
    // them would pass a naive per-layer check and compute the wrong indices.
    let spec = plan.get(25).expect("layer 25 is planned");
    assert_eq!(spec.kv_source, Some(20));
    assert_eq!(spec.index_source, Some(24));
    let spec = plan.get(37).expect("layer 37 is planned");
    assert_eq!(spec.kv_source, Some(20));
    assert_eq!(spec.index_source, Some(36));
}

/// The plan is the whole model: forty backbone rows and the three MTP stages.
#[test]
fn the_three_mtp_stages_are_planned_rows_with_no_sources() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    assert_eq!(plan.len(), config.n_layers + config.n_mtp_layers);
    assert_eq!(
        plan.order(),
        (0..config.n_layers + config.n_mtp_layers).collect::<Vec<_>>()
    );
    assert!(
        plan.get(config.n_layers + config.n_mtp_layers).is_none(),
        "there is no row after the last stage"
    );

    let stages: Vec<&LayerSpec> = plan
        .layers()
        .iter()
        .filter(|spec| spec.mode == LayerMode::Mtp)
        .collect();
    assert_eq!(
        stages.iter().map(|spec| spec.layer).collect::<Vec<_>>(),
        vec![40, 41, 42]
    );
    for spec in stages {
        assert_eq!(spec.compress_ratio, 0, "a stage is window-only");
        assert_eq!(spec.compression, CompressionKind::Uncompressed);
        assert!(!spec.owns_kv, "a stage owns no KV");
        assert!(!spec.owns_index, "a stage owns no indexer");
        assert_eq!(spec.kv_source, None);
        assert_eq!(spec.index_source, None);
        assert_eq!(spec.candidate_source, None);
        assert_eq!(spec.kv_offset, None);
        assert!(!spec.engram, "a stage carries no Engram table");
        assert!(!spec.dspark_target, "a stage is not a tap");
    }
    assert!(
        config.compress_ratios[config.n_layers..]
            .iter()
            .all(|&ratio| ratio == 0),
        "the MTP stages are ratio 0"
    );

    // The DSpark ids are taps in the backbone, not the stages.
    assert_eq!(
        plan.dspark_targets()
            .map(|spec| spec.layer)
            .collect::<Vec<_>>(),
        config.dspark_target_layer_ids
    );
    assert!(plan.is_topologically_ordered());
}

/// The plan is 43 rows, not 40: a truncated or over-long plan is refused.
#[test]
fn a_plan_with_the_wrong_number_of_stage_rows_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let expected = config.n_layers + config.n_mtp_layers;

    let mut truncated = plan.layers().to_vec();
    truncated.pop();
    let error = LayerPlan::from_specs(&config, truncated)
        .expect_err("the forty backbone rows alone are not the plan");
    assert!(
        matches!(
            error,
            PlanError::LayerCount {
                expected: found_expected,
                found: 42
            } if found_expected == expected
        ),
        "{error}"
    );

    let mut extended = plan.layers().to_vec();
    extended.push(plan.layers()[42].clone());
    let error =
        LayerPlan::from_specs(&config, extended).expect_err("a fourth stage is not in the plan");
    assert!(
        matches!(
            error,
            PlanError::LayerCount {
                expected: found_expected,
                found: 44
            } if found_expected == expected
        ),
        "{error}"
    );
}

/// A stage has no cache of its own, so a stage row naming one is refused.
#[test]
fn a_stage_row_with_a_kv_or_index_source_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");

    let mut specs = plan.layers().to_vec();
    specs[40].kv_source = Some(20);
    let error =
        LayerPlan::from_specs(&config, specs).expect_err("a stage reads no backbone KV cache");
    assert!(
        matches!(
            error,
            PlanError::KvSourceUnexpected {
                layer: 40,
                from_layer: 20
            }
        ),
        "{error}"
    );

    let mut specs = plan.layers().to_vec();
    specs[41].index_source = Some(36);
    let error =
        LayerPlan::from_specs(&config, specs).expect_err("a stage reuses no backbone Top-K");
    assert!(
        matches!(
            error,
            PlanError::IndexSourceUnexpected {
                layer: 41,
                from_layer: 36
            }
        ),
        "{error}"
    );

    let mut specs = plan.layers().to_vec();
    specs[42].owns_index = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("a stage owns no indexer");
    assert!(
        matches!(
            error,
            PlanError::OwnsIndexMismatch {
                layer: 42,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );
}

/// A stage row is a row after the backbone, not a label a backbone row can borrow.
#[test]
fn an_mtp_stage_placed_inside_the_backbone_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let mut specs = plan.layers().to_vec();
    specs[20].mode = LayerMode::Mtp;
    let error = LayerPlan::from_specs(&config, specs)
        .expect_err("layer 20 is a backbone layer, not a stage");
    assert!(
        matches!(
            error,
            PlanError::StageInBackbone {
                layer: 20,
                n_layers: 40
            }
        ),
        "{error}"
    );
}

/// A stage row past the last stage has no ratio and no place in the plan.
#[test]
fn an_unknown_stage_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let mut specs = plan.layers().to_vec();
    specs[42].layer = config.n_layers + config.n_mtp_layers;
    let error = LayerPlan::from_specs(&config, specs).expect_err("there is no fourth MTP stage");
    assert!(
        matches!(
            error,
            PlanError::UnknownStage {
                layer: 43,
                n_layers: 40,
                n_mtp_layers: 3
            }
        ),
        "{error}"
    );
}

/// A stage is not interchangeable with a window-only backbone layer, even at the same ratio.
#[test]
fn a_stage_row_must_carry_the_mtp_mode() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let mut specs = plan.layers().to_vec();
    specs[40].mode = LayerMode::SwaOnly;
    let error =
        LayerPlan::from_specs(&config, specs).expect_err("a stage dispatches to the DSpark block");
    assert!(
        matches!(
            error,
            PlanError::ModeMismatch {
                layer: 40,
                found: LayerMode::SwaOnly,
                expected: LayerMode::Mtp
            }
        ),
        "{error}"
    );
}

/// A stage is window-only; the released reader gives it ratio 0, and the plan enforces it.
#[test]
fn a_compressed_mtp_stage_is_refused() {
    let mut config = released_config();
    config.compress_ratios[config.n_layers] = 1;
    let error = LayerPlan::from_config(&config)
        .expect_err("a stage with a compressor contradicts the DSpark block");
    assert!(
        matches!(
            error,
            PlanError::CompressedStage {
                layer: 40,
                ratio: 1
            }
        ),
        "{error}"
    );
}

/// KV ownership does not imply index ownership. The reference computes `is_kv_source` and
/// `is_index_source` separately (model.py:654-655), lets a KV source without an indexer reuse
/// `shared_attn.topk_idxs` (model.py:725-726) while still publishing its own `compress_kv`
/// (model.py:746-748).
///
/// The released configuration lists every KV source as an index source, so the fourth combination
/// the plan can construct - `(owns_kv, owns_index) = (true, false)` - needs a custom configuration.
/// Removing layer 8 from `index_source_layers` keeps its compressor and compressed KV and makes it
/// read layer 2's Top-K; the mode stays Full because Full means "owns KV", and `publishes_topk()`
/// follows the flag rather than the mode.
#[test]
fn a_kv_source_without_an_indexer_owns_kv_and_reuses_the_preceding_index_source() {
    let mut config = released_config();
    config.index_source_layers.retain(|&layer| layer != 8);
    let plan = LayerPlan::from_config(&config).expect("a KV source may be a non-index source");

    let spec = plan.get(8).expect("layer 8 is planned");
    assert_eq!(
        spec.mode,
        LayerMode::Full,
        "Full is KV ownership, not index ownership"
    );
    assert!(spec.owns_kv, "layer 8 still owns the compressed KV");
    assert!(!spec.owns_index, "layer 8 no longer runs an indexer");
    assert!(
        !spec.publishes_topk(),
        "publishes_topk is owns_index, not the mode"
    );
    assert_eq!(
        spec.kv_source, None,
        "a KV source publishes rather than reads"
    );
    assert_eq!(
        spec.index_source,
        Some(2),
        "it reuses the preceding index source"
    );
    assert_eq!(spec.compress_ratio, 2, "its own compressor is unchanged");
    assert_eq!(spec.compression, CompressionKind::Pooled);
    assert_eq!(spec.kv_offset, Some(128));
    assert_eq!(spec.candidate_source, None, "it precedes the pool builder");

    // The reader path and the Full row use the same index source: layer 9 reads layer 8's KV and
    // layer 2's Top-K, exactly as the reference's shared slots would.
    let reader = plan.get(9).expect("layer 9 is planned");
    assert_eq!(reader.kv_source, Some(8));
    assert_eq!(reader.index_source, Some(2));

    // The configuration, not the mode, decides who publishes: the derived specs are accepted.
    let rebuilt = LayerPlan::from_specs(&config, plan.layers().to_vec())
        .expect("the derived specs must validate under their configuration");
    assert_eq!(rebuilt, plan);

    // All four ownership combinations are constructible across the released plan and this one:
    // (true, true) on a Full index source, (true, false) on layer 8, (false, true) on a Reindex
    // layer, (false, false) on a Reuse or window-only row.
    let released = LayerPlan::from_config(&released_config()).expect("the released plan");
    let mut combinations: Vec<(bool, bool)> = released
        .layers()
        .iter()
        .chain(plan.layers())
        .map(|spec| (spec.owns_kv, spec.owns_index))
        .collect();
    combinations.sort_unstable();
    combinations.dedup();
    assert_eq!(
        combinations,
        vec![(false, false), (false, true), (true, false), (true, true)],
        "the independent flags must reach all four combinations"
    );
    assert_eq!(plan.candidate_source(), Some(20));
}

/// The HSI pool is built inside the indexer (`select_candidate_blocks`, model.py:569-571), so the
/// builder must be an index source; `is_candidate_source` does not consult `owns_k`
/// (model.py:502-503), so the builder need not own KV. Layer 24 is a Reindex layer in the released
/// index list and owns no compressor.
#[test]
fn a_reindex_layer_can_build_the_hsi_pool() {
    let mut config = released_config();
    config.candidate_source_layer = 24;
    let plan = LayerPlan::from_config(&config).expect("a Reindex layer can build the pool");
    assert_eq!(plan.candidate_source(), Some(24));

    let builder = plan.get(24).expect("layer 24 is planned");
    assert_eq!(builder.mode, LayerMode::Reindex);
    assert!(!builder.owns_kv, "the builder owns no compressed KV");
    assert!(
        builder.owns_index,
        "the indexer that builds the pool is its own"
    );
    assert_eq!(
        builder.kv_source,
        Some(20),
        "it reads the KV source's cache"
    );
    assert_eq!(
        builder.index_source, None,
        "it publishes rather than reuses"
    );
    assert_eq!(builder.candidate_source, Some(24), "it builds the pool");

    // The pool starts at the builder: earlier index sources do not have one, later index sources
    // and their readers are involved with it.
    assert_eq!(plan.get(20).expect("layer 20").candidate_source, None);
    for layer in [28, 32, 36] {
        assert_eq!(
            plan.get(layer)
                .expect("a later index source")
                .candidate_source,
            Some(24),
            "layer {layer} uses the pool layer 24 built"
        );
    }
    for layer in [25, 37] {
        assert_eq!(
            plan.get(layer).expect("a reuse row").candidate_source,
            None,
            "layer {layer} runs no indexer and does not use candidates"
        );
    }

    let rebuilt = LayerPlan::from_specs(&config, plan.layers().to_vec())
        .expect("the derived specs must validate under their configuration");
    assert_eq!(rebuilt, plan);
}

/// A Reuse layer runs no indexer, so it cannot build the pool even though it is an ordinary
/// compressed backbone row.
#[test]
fn a_candidate_source_that_is_not_an_index_source_is_refused() {
    let mut config = released_config();
    config.candidate_source_layer = 21;
    let specs = released_specs();
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::CandidateSourceNotIndexSource { from_layer: 21 }
        ),
        "{error}"
    );
}

/// The invariant is a check, not a comment: a deliberately wrong edge must be refused.
#[test]
fn a_kv_source_that_does_not_precede_its_reader_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let mut specs = plan.layers().to_vec();
    let reader = specs
        .iter_mut()
        .find(|spec| spec.layer == 3)
        .expect("layer 3 exists");
    reader.kv_source = Some(20);
    let error = LayerPlan::from_specs(&config, specs)
        .expect_err("a source after its reader must be refused");
    assert!(
        matches!(
            error,
            PlanError::KvSourceNotBefore {
                layer: 3,
                from_layer: 20
            }
        ),
        "{error}"
    );
}

/// A compressed reader with no source at all is refused, so an id list that omits the first
/// source cannot silently plan a layer that reads an empty cache.
#[test]
fn a_reader_with_no_preceding_kv_source_is_refused() {
    // Drop layer 2 from both source lists: it is the first compressed layer, so it has no
    // preceding source, while the HSI builder (layer 20) is still both a KV and an index source.
    let mut config = released_config();
    config.kv_source_layers.retain(|&layer| layer != 2);
    config.index_source_layers.retain(|&layer| layer != 2);
    let error = LayerPlan::from_config(&config)
        .expect_err("a ratio-2 reader with no KV source must be refused");
    assert!(
        matches!(error, PlanError::MissingKvSource { layer: 2, ratio: 2 }),
        "{error}"
    );
}

/// The DSpark target ids are taps *inside* the base forward, not the MTP stages.
#[test]
fn the_dspark_targets_are_taps_inside_the_base_forward() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");
    let targets: Vec<usize> = plan.dspark_targets().map(|spec| spec.layer).collect();
    assert_eq!(targets, config.dspark_target_layer_ids);
    assert_eq!(targets, vec![37, 38, 39]);

    for layer in targets {
        let spec = plan
            .get(layer)
            .expect("a target is a backbone layer, so it is planned");
        assert!(layer < config.n_layers);
        assert_eq!(spec.mode, LayerMode::Reuse);
        assert_eq!(spec.compress_ratio, 1);
        assert_eq!(spec.kv_source, Some(20));
        assert_eq!(spec.index_source, Some(36));
        assert!(!spec.engram);
    }

    // The three stages are planned, but they are not taps.
    for layer in config.n_layers..config.n_layers + config.n_mtp_layers {
        let spec = plan.get(layer).expect("the stages are planned");
        assert_eq!(spec.mode, LayerMode::Mtp);
        assert!(!spec.dspark_target);
    }
}

/// The ownership flags are independent of the mode, so a row cannot claim an ownership the released
/// lists do not give it.
#[test]
fn a_row_that_mislabels_its_ownership_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");

    let mut specs = plan.layers().to_vec();
    specs[3].owns_kv = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 3 owns no KV");
    assert!(
        matches!(
            error,
            PlanError::OwnsKvMismatch {
                layer: 3,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );

    let mut specs = plan.layers().to_vec();
    specs[24].owns_index = false;
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 24 runs its own indexer");
    assert!(
        matches!(
            error,
            PlanError::OwnsIndexMismatch {
                layer: 24,
                found: false,
                expected: true
            }
        ),
        "{error}"
    );
}

/// The plan is a function of the configuration, not a transcription of one document.
#[test]
fn the_plan_is_derived_from_the_configuration() {
    let released = released_config();
    let runtime = ModelConfig::v41_flash().expect("the shipped runtime configuration must parse");
    assert_eq!(
        released, runtime,
        "the released and runtime documents must describe the same model"
    );
    assert_eq!(
        LayerPlan::from_config(&released).expect("the released plan"),
        LayerPlan::from_config(&runtime).expect("the runtime plan"),
        "the same configuration must derive the same plan"
    );

    // The tiny configuration has a different shape but the same relations: one KV source, two
    // index sources (the KV source and one Reindex layer), one Engram layer and one MTP stage. Its
    // stage ratio is 0, like the released stages, so it is a valid 43-row-shaped plan in miniature.
    let tiny = ModelConfig::tiny();
    let tiny_plan = LayerPlan::from_config(&tiny).expect("the tiny configuration must plan");
    assert_eq!(tiny_plan.len(), tiny.n_layers + tiny.n_mtp_layers);
    assert!(tiny_plan.is_topologically_ordered());
    assert!(tiny_plan.get(tiny.n_layers + tiny.n_mtp_layers).is_none());
    let stage = tiny_plan
        .get(tiny.n_layers)
        .expect("the tiny stage row is planned");
    assert_eq!(stage.mode, LayerMode::Mtp);
    assert_eq!(stage.compress_ratio, 0);
    assert_eq!(
        tiny_plan.get(1).expect("layer 1 is planned").mode,
        LayerMode::Reindex
    );
}

/// A hand-built spec that is internally consistent but contradicts the configuration still fails.
#[test]
fn a_plan_that_contradicts_its_configuration_is_refused() {
    let config = released_config();
    let plan = LayerPlan::from_config(&config).expect("the released configuration must plan");

    // The wrong compression label for a ratio.
    let mut specs: Vec<LayerSpec> = plan.layers().to_vec();
    specs[2].compression = CompressionKind::Direct;
    let error = LayerPlan::from_specs(&config, specs).expect_err("the label must match the ratio");
    assert!(
        matches!(
            error,
            PlanError::CompressionMismatch {
                layer: 2,
                kind: CompressionKind::Direct,
                ratio: 2
            }
        ),
        "{error}"
    );

    // A Reuse layer given an indexer's role.
    let mut specs = plan.layers().to_vec();
    specs[25].mode = LayerMode::Reindex;
    let error = LayerPlan::from_specs(&config, specs).expect_err("the mode must follow the ids");
    assert!(
        matches!(error, PlanError::ModeMismatch { layer: 25, .. }),
        "{error}"
    );

    // A Reindex layer pointing at a preceding index source that is not its own group's.
    let mut specs = plan.layers().to_vec();
    specs[25].index_source = Some(20);
    let error =
        LayerPlan::from_specs(&config, specs).expect_err("the nearest source is the right one");
    assert!(
        matches!(
            error,
            PlanError::IndexSourceMismatch {
                layer: 25,
                found: 20,
                expected: 24
            }
        ),
        "{error}"
    );
}
/// A stage is not a backbone row: an id list that names one is refused with a typed error rather
/// than accepted because the row exists. Without this check a stage id reaches the per-row
/// comparison and is reported as a row mismatch far from the list that caused it.
#[test]
fn a_stage_id_in_any_configuration_list_is_refused() {
    let released = released_config();
    let specs = released_specs();

    let mut config = released.clone();
    config.kv_source_layers.push(41);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "kv_source_layers",
                layer: 41,
                n_layers: 40
            }
        ),
        "{error}"
    );

    let mut config = released.clone();
    config.index_source_layers.push(41);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "index_source_layers",
                layer: 41,
                ..
            }
        ),
        "{error}"
    );

    let mut config = released.clone();
    config.engram_layer_ids.push(41);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "engram_layer_ids",
                layer: 41,
                ..
            }
        ),
        "{error}"
    );

    let mut config = released;
    config.dspark_target_layer_ids.push(41);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "dspark_target_layer_ids",
                layer: 41,
                ..
            }
        ),
        "{error}"
    );
}

/// The pool builder is a backbone layer. A stage id in `candidate_source_layer` is refused before
/// the plan can compare it with a row that has no candidate.
#[test]
fn a_stage_candidate_source_is_refused() {
    let released = released_config();
    let specs = released_specs();

    let mut config = released.clone();
    config.candidate_source_layer = 41;
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "candidate_source_layer",
                layer: 41,
                n_layers: 40
            }
        ),
        "{error}"
    );

    // The critic's exact construction: the stage in both source lists and in the candidate field.
    let mut config = released;
    config.kv_source_layers.push(41);
    config.index_source_layers.push(41);
    config.candidate_source_layer = 41;
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "kv_source_layers",
                layer: 41,
                ..
            }
        ),
        "{error}"
    );
}

/// An id past every row is not a layer at all, in any list.
#[test]
fn an_out_of_range_configuration_id_is_refused() {
    let released = released_config();
    let specs = released_specs();

    let mut config = released.clone();
    config.kv_source_layers.push(43);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "kv_source_layers",
                layer: 43,
                n_layers: 40
            }
        ),
        "{error}"
    );

    let mut config = released;
    let out_of_range = config.n_layers + config.n_mtp_layers;
    config.engram_layer_ids.push(out_of_range);
    let error = refused_by_both(&config, &specs);
    assert!(
        matches!(
            error,
            PlanError::LayerOutsideBackbone {
                field: "engram_layer_ids",
                layer: 43,
                ..
            }
        ),
        "{error}"
    );
}

/// The checks above preserve a property for every accepted plan: every source, pool, Engram and
/// DSpark id names a backbone row, never a stage and never an id outside the plan.
#[test]
fn every_accepted_plan_names_only_backbone_rows() {
    for config in [released_config(), ModelConfig::tiny()] {
        let plan = LayerPlan::from_config(&config).expect("the configuration must plan");
        let n_layers = config.n_layers;
        for spec in plan.layers() {
            if let Some(source) = spec.kv_source {
                assert!(
                    source < n_layers,
                    "layer {} names KV source {source}",
                    spec.layer
                );
                assert!(plan.get(source).expect("a source is planned").owns_kv);
            }
            if let Some(source) = spec.index_source {
                assert!(
                    source < n_layers,
                    "layer {} names index source {source}",
                    spec.layer
                );
                assert!(
                    plan.get(source)
                        .expect("a source is planned")
                        .publishes_topk()
                );
            }
            if let Some(source) = spec.candidate_source {
                assert!(source < n_layers, "the pool builder is a backbone row");
            }
            if spec.engram || spec.dspark_target {
                assert!(
                    spec.layer < n_layers,
                    "layer {} is a backbone row",
                    spec.layer
                );
            }
            if spec.owns_kv || spec.owns_index {
                assert!(
                    spec.layer < n_layers,
                    "layer {} is a backbone row",
                    spec.layer
                );
            }
        }
        assert!(
            plan.candidate_source()
                .is_none_or(|source| source < n_layers)
        );
    }
}

/// A window-only configuration has no HSI pool: `candidate_source_layer` still names a layer, but
/// no row builds the pool. The plan records `None` instead of refusing a model that is valid.
#[test]
fn a_window_only_configuration_plans_with_no_hsi_pool() {
    let mut config = ModelConfig::tiny();
    let rows = config.n_layers + config.n_mtp_layers;
    config.compress_ratios = vec![0; rows];
    config.kv_source_layers.clear();
    config.index_source_layers.clear();
    config.candidate_source_layer = 0;
    config
        .validate()
        .expect("a window-only configuration is a valid model");

    let plan = LayerPlan::from_config(&config).expect("a window-only configuration must plan");
    assert_eq!(plan.candidate_source(), None);
    for spec in plan.layers() {
        assert_eq!(
            spec.candidate_source, None,
            "layer {} has no pool",
            spec.layer
        );
        assert!(!spec.owns_kv && !spec.owns_index, "layer {}", spec.layer);
        assert_eq!(spec.kv_source, None);
        assert_eq!(spec.index_source, None);
        assert_eq!(spec.compress_ratio, 0);
    }
    assert_eq!(plan.get(0).expect("layer 0").mode, LayerMode::SwaOnly);
    assert_eq!(plan.get(1).expect("layer 1").mode, LayerMode::SwaOnly);
    assert_eq!(plan.get(2).expect("the stage").mode, LayerMode::Mtp);
    assert!(plan.is_topologically_ordered());
}

/// The `--synthetic` serve path uses [`ModelConfig::tiny`] (crates/sparkflash-cli/src/serve.rs:177)
/// and its stage ratio is 0 rather than 1. That is a behavior change in a production test
/// configuration, and both reasons are in the reference: the DSpark block asserts an uncompressed
/// stage (`assert self.compress_ratio == 0`, model.py:1034), and the ratio selects the RoPE table,
/// so part of the forward reads a different table at ratio 0 than at ratio 1.
#[test]
fn the_synthetic_serve_configuration_plans_a_ratio_zero_mtp_stage() {
    let config = ModelConfig::tiny();
    config
        .validate()
        .expect("the synthetic configuration must be valid");
    let plan = LayerPlan::from_config(&config).expect("the synthetic configuration must plan");
    let stage = plan.get(config.n_layers).expect("the stage row is planned");
    assert_eq!(
        stage.mode,
        LayerMode::Mtp,
        "the forward dispatches the DSpark block"
    );
    assert_eq!(
        stage.compress_ratio, 0,
        "DSpark asserts ratio 0 (model.py:1034)"
    );
    assert_eq!(
        stage.compression,
        CompressionKind::Uncompressed,
        "the stage must use the uncompressed RoPE table"
    );
    assert_eq!(stage.kv_offset, None);
    assert_eq!(stage.kv_source, None);
    assert_eq!(stage.index_source, None);
    assert_eq!(stage.candidate_source, None);
}

/// The plan's rows are dense and ascending; a duplicate or reordered row is refused.
#[test]
fn a_plan_that_is_not_dense_or_ascending_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[1].layer = 0;
    let error = LayerPlan::from_specs(&config, specs).expect_err("a duplicated row is not dense");
    assert!(
        matches!(
            error,
            PlanError::NotDenseOrAscending {
                position: 1,
                layer: 0,
                rows: 43
            }
        ),
        "{error}"
    );
}

/// A row the configuration has no ratio for is refused rather than indexed out of bounds.
#[test]
fn a_missing_ratio_is_refused() {
    let mut config = released_config();
    let rows = config.n_layers + config.n_mtp_layers;
    config.compress_ratios.truncate(rows - 1);
    let error =
        LayerPlan::from_specs(&config, released_specs()).expect_err("the last row has no ratio");
    assert!(
        matches!(error, PlanError::MissingRatio { layer: 42 }),
        "{error}"
    );
}

/// The row's ratio is the configuration's, not the plan's.
#[test]
fn a_ratio_that_disagrees_with_the_configuration_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[2].compress_ratio = 1;
    let error = LayerPlan::from_specs(&config, specs).expect_err("the ratio must match");
    assert!(
        matches!(
            error,
            PlanError::RatioMismatch {
                layer: 2,
                found: 1,
                expected: 2
            }
        ),
        "{error}"
    );
}

/// A ratio-0 layer has no compressor, so a source list that names one contradicts the model.
#[test]
fn a_ratio_zero_layer_named_as_a_source_is_refused() {
    let mut config = released_config();
    config.kv_source_layers.push(0);
    let error = refused_by_both(&config, &released_specs());
    assert!(
        matches!(error, PlanError::SourceWithoutCompression { layer: 0 }),
        "{error}"
    );
}

/// A reader's KV source must be in the KV-source list, precede the reader and have its ratio.
#[test]
fn a_kv_source_that_is_not_a_configuration_source_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[3].kv_source = Some(1);
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 1 owns no KV");
    assert!(
        matches!(
            error,
            PlanError::KvSourceNotSource {
                layer: 3,
                from_layer: 1
            }
        ),
        "{error}"
    );
}

/// A ratio-2 reader cannot read a ratio-1 cache: the compressor layouts differ.
#[test]
fn a_kv_ratio_that_disagrees_with_its_source_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[21].kv_source = Some(14);
    let error = LayerPlan::from_specs(&config, specs).expect_err("the ratios differ");
    assert!(
        matches!(
            error,
            PlanError::KvRatioMismatch {
                layer: 21,
                from_layer: 14,
                ratio: 1,
                source_ratio: 2
            }
        ),
        "{error}"
    );
}

/// The nearest preceding KV source is the cache the reference publishes; an earlier one is wrong
/// even when it exists and has the right ratio.
#[test]
fn a_kv_source_that_is_not_the_nearest_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[9].kv_source = Some(2);
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 8 is nearer");
    assert!(
        matches!(
            error,
            PlanError::KvSourceMismatch {
                layer: 9,
                found: 2,
                expected: 8
            }
        ),
        "{error}"
    );
}

/// An index source after its reader, or one that is not an index source, is refused before the
/// plan can use it.
#[test]
fn an_index_source_that_does_not_precede_or_is_not_a_source_is_refused() {
    let config = released_config();

    let mut specs = released_specs();
    specs[25].index_source = Some(36);
    let error = LayerPlan::from_specs(&config, specs).expect_err("36 is after 25");
    assert!(
        matches!(
            error,
            PlanError::IndexSourceNotBefore {
                layer: 25,
                from_layer: 36
            }
        ),
        "{error}"
    );

    let mut specs = released_specs();
    specs[25].index_source = Some(21);
    let error = LayerPlan::from_specs(&config, specs).expect_err("21 runs no indexer");
    assert!(
        matches!(
            error,
            PlanError::IndexSourceNotSource {
                layer: 25,
                from_layer: 21
            }
        ),
        "{error}"
    );
}

/// A ratio-1 reader cannot reuse a ratio-2 index source's Top-K.
#[test]
fn an_index_ratio_that_disagrees_with_its_source_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[21].index_source = Some(14);
    let error = LayerPlan::from_specs(&config, specs).expect_err("the ratios differ");
    assert!(
        matches!(
            error,
            PlanError::IndexRatioMismatch {
                layer: 21,
                from_layer: 14,
                ratio: 1,
                source_ratio: 2
            }
        ),
        "{error}"
    );
}

/// A compressed reader whose configuration has no preceding index source cannot reuse anything,
/// whether it is a Reuse row or a Full row that lost its indexer.
#[test]
fn a_reader_with_no_preceding_index_source_is_refused() {
    let mut config = released_config();
    config.index_source_layers.retain(|&layer| layer != 2);
    let error = LayerPlan::from_config(&config)
        .expect_err("layer 2 is the first compressed layer and has no index source");
    assert!(
        matches!(error, PlanError::MissingIndexSource { layer: 2, ratio: 2 }),
        "{error}"
    );

    // `from_specs` applies the same rule to a row that agrees with the configuration's ownership.
    let mut specs = released_specs();
    specs[2].owns_index = false;
    specs[2].index_source = None;
    let error = LayerPlan::from_specs(&config, specs).expect_err("there is nothing to reuse");
    assert!(
        matches!(error, PlanError::MissingIndexSource { layer: 2, ratio: 2 }),
        "{error}"
    );
}

/// The decode offset is the window the plan carries, not any value a spec happens to hold.
#[test]
fn a_kv_offset_that_disagrees_with_the_window_rule_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[3].kv_offset = Some(1);
    let error = LayerPlan::from_specs(&config, specs).expect_err("the window is 128");
    assert!(
        matches!(
            error,
            PlanError::KvOffsetMismatch {
                layer: 3,
                found: Some(1),
                expected: Some(128)
            }
        ),
        "{error}"
    );
}

/// The candidate marker is derived from the builder and the row's index ownership; a row cannot
/// claim a pool it does not use.
#[test]
fn a_candidate_marker_that_disagrees_with_the_pool_builder_is_refused() {
    let config = released_config();
    let mut specs = released_specs();
    specs[21].candidate_source = Some(20);
    let error = LayerPlan::from_specs(&config, specs).expect_err("21 runs no indexer");
    assert!(
        matches!(
            error,
            PlanError::CandidateSourceMismatch {
                layer: 21,
                found: Some(20),
                expected: None
            }
        ),
        "{error}"
    );
}

/// Engram and DSpark membership are the configuration's lists, and a stage carries neither.
#[test]
fn an_engram_or_dspark_flag_that_disagrees_with_the_ids_is_refused() {
    let config = released_config();

    let mut specs = released_specs();
    specs[2].engram = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 2 carries no table");
    assert!(
        matches!(
            error,
            PlanError::EngramMismatch {
                layer: 2,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );

    let mut specs = released_specs();
    specs[3].dspark_target = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("layer 3 is not a tap");
    assert!(
        matches!(
            error,
            PlanError::DsparkTargetMismatch {
                layer: 3,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );

    // A stage is rejected by the stage branch itself, not by the general list comparison.
    let mut specs = released_specs();
    specs[41].engram = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("a stage carries no table");
    assert!(
        matches!(
            error,
            PlanError::EngramMismatch {
                layer: 41,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );

    let mut specs = released_specs();
    specs[42].dspark_target = true;
    let error = LayerPlan::from_specs(&config, specs).expect_err("a stage is not a tap");
    assert!(
        matches!(
            error,
            PlanError::DsparkTargetMismatch {
                layer: 42,
                found: true,
                expected: false
            }
        ),
        "{error}"
    );
}
