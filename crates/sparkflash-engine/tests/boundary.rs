//! The Phase 1 per-layer boundary table, and the perturbation test that makes the diff able to fail.
//!
//! Spec: `docs/build/phase1-boundary-table.md` (the format) and `docs/build/roadmap-to-serving.md`
//! section 4.5 criteria 1 and 2 (the table and the perturbation test are Phase 1's exit criteria).
//!
//! What is asserted here, in the terms the roadmap uses:
//!
//! 1. One step of the reference forward, captured through the observed entry points, holds one row
//!    per backbone layer with the full documented tensor set, in the documented order and shapes,
//!    and the observed step returns exactly what the unobserved step returns - the capture only
//!    reads.
//! 2. A one-code change in layer 17's `wq_b` weight flags layer 17's row and nothing else, and the
//!    table is asserted non-empty and the perturbed run is asserted to differ, so a capture that
//!    returned nothing, or a perturbation that did nothing, fails the test rather than passing it.
//! 3. The MTP stages record their own rows when they run. This engine skips the MTP pass on a first
//!    step; the reference returns no draft there (`forward_spec` returns nothing at
//!    `start_pos == 0`, model.py:1123-1125, 1274-1282) although its stage attention still seeds the
//!    window cache (docs/engine/01-forward-order.md section 3.9), so the row count is the capture's
//!    contract. The released-depth test pins the 40-row first step and the 43-row decode step.
//! 4. A middle-layer change propagates through every later row of a composed run: the diff's first
//!    divergence is the changed layer, which is the format document's default attribution rule, not
//!    a proof of where the fault is - localising it needs the document's layer replay.
//!
//! The weight source is [BoundarySource]: the plan's own names and shapes served as finite f32
//! values from a seed, with an optional one-code change in one element of one named tensor. The
//! stored checkpoint carries layer 17's `wq_b` as FP8 E4M3, but this fixture is f32, so the change
//! is a code-space stand-in rather than a byte reproduction of the checkpoint: re-encode the clean
//! f32 value to E4M3, move the code by one, and decode it back to the f32 that code stands for. It
//! is not claimed to be the stored tensor's decoded value; the format document calls the source a
//! test fixture for exactly this reason.
//!
//! No device is involved and no device number is claimed: this is the host reference half.

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::forward::boundary::{self, BoundaryTable, LayerBoundary, TENSOR_ORDER};
use sparkflash_engine::forward::{InferenceEngine, Sampling, SequenceState};
use sparkflash_engine::quant;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// The backbone depth of the perturbation configuration.
///
/// Eighteen backbone layers is the smallest configuration in which layer 17 is a real backbone
/// layer, and the perturbation criterion asks for a layer-17 change to flag layer 17 and no other
/// row. Layer 17 is therefore the last row of this table, which is the only end-to-end shape in
/// which "no other row" can hold: a change in a middle layer changes the residual stream every
/// later layer receives, and the test `a_middle_layer_change_propagates...` pins that difference
/// so the attribution rule is not a surprise. The one-layer replay protocol that localises a
/// middle-layer fault needs the device half of Phase 1 (format document section 7).
const BACKBONE_LAYERS: usize = 18;

/// The layer the perturbation targets.
const PERTURBED_LAYER: usize = 17;

/// A finite, plan-shaped source with an optional one-code change.
///
/// Values are the plan's names and shapes served as little-endian f32 - every tensor is reported as
/// [DType::F32] by [WeightSource::spec], exactly what [WeightSource::read] writes - positive,
/// deterministic, and varying with the seed, the tensor's name and the element.
///
/// The magnitude is `0.05 + 0.1 * byte`, deliberately larger than the `0.01`-scale source of
/// `tests/forward.rs`: this is a test-design choice made so the one-code change survives into the
/// perturbed row instead of being lost below the source's own rounding. The committed assertions
/// are the row's measured diff counts and deltas (the perturbation test and format document section
/// 6), not a score or ulp bar; this comment records why the source was chosen, not a measurement the
/// suite checks.
struct BoundarySource {
    specs: Vec<TensorSpec>,
    seed: u64,
    change: Option<OneCodeChange>,
}

/// One element of one tensor moved by one E4M3 code.
struct OneCodeChange {
    tensor: String,
    element: usize,
}

impl BoundarySource {
    fn from_plan(plan: &TensorPlan, seed: u64, change: Option<OneCodeChange>) -> Self {
        Self {
            specs: plan.specs().to_vec(),
            seed,
            change,
        }
    }
}

/// A stable per-name offset, so two tensors of the same shape are not identical.
fn name_salt(name: &str) -> u64 {
    name.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u64::from(byte))
    })
}

impl WeightSource for BoundarySource {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(TensorSpec {
                dtype: DType::F32,
                ..spec.clone()
            }),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let elements = usize::try_from(self.spec(name)?.elements()).expect("a test tensor fits");
        let salt = name_salt(name);
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            // A per-element mix, not a per-row one: with the high bits of the index the pattern
            // would barely change across a row, two rows could produce equal dot products, and the
            // argmax would then be a tie that hides a mis-indexed read.
            let mixed = (index as u64)
                .wrapping_add(salt)
                .wrapping_add(self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let hashed = (mixed ^ (mixed >> 13)).wrapping_mul(0x2545_f491_4f6c_dd1d);
            let byte = (hashed >> 56) as f32;
            let value = 0.05 + 0.1 * (byte / 255.0);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(change) = &self.change
            && change.tensor == name
        {
            let at = change.element * 4;
            let clean = quant::f32_from_le_bytes(
                bytes[at..at + 4].try_into().expect("four bytes of an f32"),
            );
            // The stored form is FP8 E4M3 (docs/build/device-path-design.md section 2): a
            // one-code change takes the clean value's code and moves it one step.
            let code = quant::e4m3_encode(clean);
            let changed = quant::f8e4m3_to_f32(code ^ 1);
            assert_ne!(
                changed, clean,
                "a one-code change must move the value it is injected into"
            );
            bytes[at..at + 4].copy_from_slice(&changed.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// The eighteen-layer, small-shape configuration the perturbation test runs.
///
/// Not the released shapes and not [ModelConfig::tiny] (two backbone layers): the attention on every
/// layer is sliding-window only (`compress_ratios` all zero), so the test isolates the boundary
/// capture and the one-code change from the cross-layer compressor/indexer modes, which other tests
/// cover. The released model's layer variety is not what this test is for.
fn eighteen_layers() -> ModelConfig {
    ModelConfig {
        n_layers: BACKBONE_LAYERS,
        n_mtp_layers: 1,
        compress_ratios: vec![0; BACKBONE_LAYERS + 1],
        kv_source_layers: Vec::new(),
        index_source_layers: Vec::new(),
        candidate_source_layer: 0,
        engram_layer_ids: Vec::new(),
        engram_num_embeddings: Vec::new(),
        dspark_target_layer_ids: vec![PERTURBED_LAYER],
        ..ModelConfig::tiny()
    }
}

/// An engine over `config`'s plan from [BoundarySource], with the optional one-code change.
fn engine(config: &ModelConfig, seed: u64, change: Option<OneCodeChange>) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = BoundarySource::from_plan(&plan, seed, change);
    InferenceEngine::new(config.clone(), Box::new(source), 1 << 22)
        .expect("the tested configuration is usable")
}

/// Asserts one row is the complete documented tensor set, in the documented order and shapes.
fn assert_row_is_documented(config: &ModelConfig, row: &LayerBoundary, positions: usize) {
    assert_eq!(
        row.tensors().len(),
        TENSOR_ORDER.len(),
        "layer {}: every documented tensor is recorded",
        row.layer()
    );
    for (tensor, expected) in row.tensors().iter().zip(TENSOR_ORDER) {
        assert_eq!(
            tensor.name(),
            expected,
            "layer {}: tensors are in the documented order",
            row.layer()
        );
    }
    let dim = config.dim;
    let hc = config.hc_mult;
    let shape = |name: &str| {
        row.tensor(name)
            .unwrap_or_else(|| panic!("layer {}: no tensor {name}", row.layer()))
            .shape()
            .to_vec()
    };
    assert_eq!(shape(boundary::RESIDUAL_IN), vec![positions, hc, dim]);
    assert_eq!(shape(boundary::HC_IN_PRE), vec![positions, hc]);
    assert_eq!(shape(boundary::HC_ATTN_PRE), vec![positions, hc]);
    assert_eq!(shape(boundary::HC_ATTN_POST), vec![positions, hc]);
    assert_eq!(shape(boundary::HC_ATTN_COMB), vec![positions, hc, hc]);
    assert_eq!(shape(boundary::PRE_ATTENTION_HIDDEN), vec![positions, dim]);
    assert_eq!(shape(boundary::PRE_ATTENTION_XHAT), vec![positions, dim]);
    assert_eq!(
        shape(boundary::PRE_ATTENTION_SCALES),
        vec![positions, dim / 32]
    );
    assert_eq!(shape(boundary::ATTENTION_OUT), vec![positions, dim]);
    assert_eq!(
        shape(boundary::POST_ATTENTION_HIDDEN),
        vec![positions, hc, dim]
    );
    assert_eq!(shape(boundary::HC_FFN_PRE), vec![positions, hc]);
    assert_eq!(shape(boundary::HC_FFN_POST), vec![positions, hc]);
    assert_eq!(shape(boundary::HC_FFN_COMB), vec![positions, hc, hc]);
    assert_eq!(shape(boundary::PRE_FFN_HIDDEN), vec![positions, dim]);
    assert_eq!(shape(boundary::PRE_FFN_XHAT), vec![positions, dim]);
    assert_eq!(shape(boundary::PRE_FFN_SCALES), vec![positions, dim / 32]);
    assert_eq!(shape(boundary::FFN_OUT), vec![positions, dim]);
    assert_eq!(shape(boundary::RESIDUAL_OUT), vec![positions, hc, dim]);

    // The two formats: streams and sublayer outputs are f32, the quantised block input is raw E4M3
    // code bytes and E8M0 scale bytes (docs/build/phase1-boundary-table.md section 4).
    for f32_name in [
        boundary::RESIDUAL_IN,
        boundary::HC_IN_PRE,
        boundary::HC_ATTN_PRE,
        boundary::HC_ATTN_POST,
        boundary::HC_ATTN_COMB,
        boundary::PRE_ATTENTION_HIDDEN,
        boundary::ATTENTION_OUT,
        boundary::POST_ATTENTION_HIDDEN,
        boundary::HC_FFN_PRE,
        boundary::HC_FFN_POST,
        boundary::HC_FFN_COMB,
        boundary::PRE_FFN_HIDDEN,
        boundary::FFN_OUT,
        boundary::RESIDUAL_OUT,
    ] {
        let tensor = row.tensor(f32_name).expect("checked above");
        assert!(
            tensor.data().as_f32().is_some(),
            "layer {}: {f32_name} is an f32 tensor",
            row.layer()
        );
    }
    for code_name in [
        boundary::PRE_ATTENTION_XHAT,
        boundary::PRE_ATTENTION_SCALES,
        boundary::PRE_FFN_XHAT,
        boundary::PRE_FFN_SCALES,
    ] {
        let tensor = row.tensor(code_name).expect("checked above");
        let codes = tensor
            .data()
            .as_u8()
            .unwrap_or_else(|| panic!("layer {}: {code_name} is a code tensor", row.layer()));
        if code_name.ends_with("xhat") {
            assert!(
                codes.iter().all(|&code| code != 0x7f && code != 0xff),
                "layer {}: an E4M3 code is never the NaN encoding",
                row.layer()
            );
        } else {
            assert!(
                codes.iter().all(|&code| code != quant::E8M0_NAN),
                "layer {}: an E8M0 scale is never the NaN encoding",
                row.layer()
            );
        }
    }

    // A row of default-initialised buffers is not a capture. Every f32 tensor of a real run is
    // non-zero somewhere in this fixture, and the code tensors carry non-zero codes, so a row that
    // was never filled in fails here instead of passing the shape-only checks above.
    assert!(
        row.tensors()
            .iter()
            .filter_map(|tensor| tensor.data().as_f32())
            .any(|values| values.iter().any(|value| *value != 0.0)),
        "layer {}: the row's f32 tensors are not all zero",
        row.layer()
    );
    assert!(
        row.tensors()
            .iter()
            .filter_map(|tensor| tensor.data().as_u8())
            .any(|codes| codes.iter().any(|code| *code != 0)),
        "layer {}: the row's code tensors are not all zero",
        row.layer()
    );
}

/// Runs one observed prefill of `tokens` over `config` at `seed`.
fn observed_prefill_tokens(
    config: &ModelConfig,
    seed: u64,
    change: Option<OneCodeChange>,
    tokens: &[u32],
) -> sparkflash_engine::forward::ObservedStep {
    let configured = engine(config, seed, change);
    let mut state = SequenceState::new(config);
    configured
        .prefill_observed(&mut state, tokens)
        .expect("the observed reference prefill runs")
}

/// Runs one observed prefill of one token, [7], over `config` at `seed`.
fn observed_prefill(
    config: &ModelConfig,
    seed: u64,
    change: Option<OneCodeChange>,
) -> sparkflash_engine::forward::ObservedStep {
    observed_prefill_tokens(config, seed, change, &[7])
}

#[test]
fn a_one_code_change_in_layer_17s_wq_b_flags_layer_17_and_no_other_layer() {
    let config = eighteen_layers();
    let clean = observed_prefill(&config, 23, None);
    let perturbed = observed_prefill(
        &config,
        23,
        Some(OneCodeChange {
            tensor: format!("layers.{PERTURBED_LAYER}.attn.wq_b.weight"),
            element: 0,
        }),
    );

    // The capture is populated - a capture that silently returned nothing fails here - and every
    // backbone layer recorded the full documented row.
    assert_eq!(
        clean.boundary.len(),
        config.n_layers,
        "one row per backbone layer"
    );
    assert_eq!(perturbed.boundary.len(), config.n_layers);
    assert_eq!(clean.boundary.start_position(), 0);
    assert_eq!(clean.boundary.tokens(), &[7]);
    for (index, row) in clean.boundary.rows().iter().enumerate() {
        assert_eq!(
            row.layer(),
            index,
            "rows are recorded in ascending layer order"
        );
        assert_row_is_documented(&config, row, 1);
    }
    for (index, row) in perturbed.boundary.rows().iter().enumerate() {
        assert_eq!(row.layer(), index);
        assert_row_is_documented(&config, row, 1);
    }

    // The perturbation reached the model: the perturbed run differs somewhere, which the layer-17
    // row below agrees with. Without this, a source that quietly ignored the change would leave
    // every row bit-identical and the localisation assertion would pass for the wrong reason.
    assert!(
        clean
            .output
            .logits
            .iter()
            .zip(perturbed.output.logits.iter())
            .any(|(clean, perturbed)| clean.to_bits() != perturbed.to_bits()),
        "the one-code change must change the perturbed run's logits"
    );

    // The per-layer diff: layer 17's row differs, every other row is bit-identical.
    let diff = clean
        .boundary
        .diff(&perturbed.boundary)
        .expect("two captures of the same step and layout are comparable");
    assert_eq!(
        diff.differing_layers(),
        vec![PERTURBED_LAYER],
        "the diff flags exactly the perturbed layer"
    );
    for row in diff.rows() {
        if row.layer() == PERTURBED_LAYER {
            continue;
        }
        assert!(
            row.is_identical(),
            "layer {}: a one-code change at layer {PERTURBED_LAYER} must not move this row",
            row.layer()
        );
        assert_eq!(row.max_abs_delta(), 0.0, "layer {}", row.layer());
        assert_eq!(row.differing(), 0, "layer {}", row.layer());
    }

    // Inside the flagged row, the boundary is localised to what the changed weight feeds: the
    // entry stream, the input mix, both coefficient sets above the change and the attention block
    // input are computed before `wq_b` is read and are bit-identical; the attention output and
    // everything downstream of it differ.
    let flagged = diff
        .rows()
        .iter()
        .find(|row| row.layer() == PERTURBED_LAYER)
        .expect("the flagged row is in the table");
    assert!(!diff.is_identical(), "the perturbed run differs somewhere");
    assert!(!flagged.is_identical());
    assert!(
        flagged.max_abs_delta() > 0.0,
        "the diff is numeric, not a flag"
    );
    for name in [
        boundary::RESIDUAL_IN,
        boundary::HC_IN_PRE,
        boundary::HC_ATTN_PRE,
        boundary::HC_ATTN_POST,
        boundary::HC_ATTN_COMB,
        boundary::PRE_ATTENTION_HIDDEN,
        boundary::PRE_ATTENTION_XHAT,
        boundary::PRE_ATTENTION_SCALES,
    ] {
        assert_eq!(
            flagged.tensor(name).expect("documented tensor").differing(),
            0,
            "{name} is computed before the changed wq_b weight"
        );
    }
    // The changed query reaches the attention output and then the stream every later tensor of
    // the row is computed from. A one-code weight change is small: it survives in some values of
    // the row and rounds away in others (the quantised block input can snap to the same E4M3 code,
    // for one), which is why the format document gives every boundary its own comparison class
    // (docs/build/phase1-boundary-table.md section 5) instead of one bar for the whole row.
    // The row's full measured result, pinned: these five tensors differ and the other thirteen are
    // bit-identical. The counts and deltas are deterministic for this configuration and seed and
    // are the numbers format document section 6 quotes; comparing the deltas by bits keeps the
    // test and the document from drifting apart. The row maximum is now 1.49e-8 on three tensors
    // (`post_attention_hidden`, `pre_ffn_hidden`, `residual_out`); `hc_ffn_comb` used to carry the
    // 5.96e-8 maximum and now rounds the one-code change away entirely, because the host reference
    // rounds the block's bf16 module outputs the way native does (task-10, audit F5).
    let measured = [
        (boundary::ATTENTION_OUT, 64usize, 1.396_983_9e-9f32),
        (boundary::POST_ATTENTION_HIDDEN, 5, 1.490_116_1e-8),
        (boundary::PRE_FFN_HIDDEN, 4, 1.490_116_1e-8),
        (boundary::FFN_OUT, 8, 2.328_306_4e-10),
        (boundary::RESIDUAL_OUT, 6, 1.490_116_1e-8),
    ];
    for (name, count, max) in measured {
        let tensor = flagged
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} is a documented tensor"));
        assert_eq!(tensor.differing(), count, "{name}: differing count");
        assert_eq!(
            tensor.max_abs_delta().to_bits(),
            max.to_bits(),
            "{name}: max abs delta {} against the documented {max}",
            tensor.max_abs_delta()
        );
    }
    for tensor in flagged.tensors() {
        if !measured.iter().any(|(name, _, _)| *name == tensor.name()) {
            assert_eq!(
                tensor.differing(),
                0,
                "{} is computed before wq_b or rounds the change away",
                tensor.name()
            );
        }
    }
    assert_eq!(
        flagged.differing(),
        87,
        "the row's measured total, across five tensors"
    );
    assert_eq!(
        flagged.max_abs_delta().to_bits(),
        1.490_116_1e-8f32.to_bits(),
        "the row maximum is the 1.49e-8 shared by three tensors"
    );

    // The boundary chain has no gap: every backbone layer's output stream is the next layer's
    // entry stream, so the rows' stream wiring is pinned. This is about the wiring, not about a
    // row's replay inputs: only a first-step backbone row is self-contained, and the single-bits
    // comparison below is what the format document's "byte for byte" wording means (the derived
    // `PartialEq` would call -0.0 equal to 0.0). A decode or MTP row also needs cache/main_x state
    // the table does not record (format document section 7.3).
    for layer in 0..config.n_layers - 1 {
        let out = clean
            .boundary
            .row(layer)
            .unwrap_or_else(|| panic!("layer {layer} ran"))
            .tensor(boundary::RESIDUAL_OUT)
            .expect("documented");
        let next_in = clean
            .boundary
            .row(layer + 1)
            .unwrap_or_else(|| panic!("layer {} ran", layer + 1))
            .tensor(boundary::RESIDUAL_IN)
            .expect("documented");
        assert!(
            out.data().bit_eq(next_in.data()),
            "layer {layer}'s output stream is layer {}'s entry stream, bit for bit",
            layer + 1
        );
    }
}

#[test]
fn the_observed_step_is_the_plain_step_and_two_runs_capture_identically() {
    let config = eighteen_layers();
    let captured = observed_prefill(&config, 31, None);

    // The observed entry point returns exactly the unobserved step's output: asking for a table
    // does not change the forward.
    let configured = engine(&config, 31, None);
    let mut plain_state = SequenceState::new(&config);
    let plain = configured
        .prefill(&mut plain_state, &[7])
        .expect("the plain prefill runs");
    assert_eq!(
        plain.logits, captured.output.logits,
        "the capture only reads: same logits"
    );
    assert_eq!(plain.next_token, captured.output.next_token);
    assert_eq!(plain.consumed, captured.output.consumed);

    // A fixed seed is a fixed capture: two runs of the same weights give bit-identical tables, so
    // the non-zero diff in the perturbation test means the table changed, not that capture noise
    // exists.
    let repeated = observed_prefill(&config, 31, None);
    let diff = captured
        .boundary
        .diff(&repeated.boundary)
        .expect("two captures of the same step");
    assert!(diff.is_identical(), "the capture is deterministic");
    assert_eq!(diff.first_differing(), None);

    // A different seed is a different set of weights and moves rows (the capture is not a constant).
    let other_seed = observed_prefill(&config, 32, None);
    let diff = captured
        .boundary
        .diff(&other_seed.boundary)
        .expect("two captures of the same step");
    assert!(
        !diff.is_identical(),
        "a different seed is a different table"
    );
    assert_eq!(diff.first_differing(), Some(0));
}

#[test]
fn the_mtp_stages_record_their_own_rows_when_they_run() {
    let config = ModelConfig::tiny();
    assert_eq!(config.n_mtp_layers, 1);
    let configured = engine(&config, 5, None);
    let mut state = SequenceState::new(&config);

    // A first step starts at position 0: the reference returns no draft there, so only the backbone
    // rows exist (model.py:1123-1125, 1274-1282).
    let prefill = configured
        .prefill_observed(&mut state, &[1, 2, 3])
        .expect("the observed prefill runs");
    assert_eq!(prefill.boundary.len(), config.n_layers);
    assert!(
        prefill
            .boundary
            .rows()
            .iter()
            .all(|row| row.layer() < config.n_layers),
        "a prefill step has no MTP rows"
    );
    assert_eq!(prefill.boundary.start_position(), 0);
    assert_eq!(prefill.boundary.tokens(), &[1, 2, 3]);
    assert!(
        prefill.boundary.consumed_prefix().is_empty(),
        "a first step consumed no prefix"
    );
    assert_eq!(
        prefill.boundary.mtp_anchor(),
        None,
        "a step with no MTP row has no anchor"
    );

    // A later step drafts: the MTP stages run and append one row each, with layer ids
    // n_layers..n_layers+n_mtp_layers.
    let decode = configured
        .decode_observed(&mut state, 4)
        .expect("the observed decode runs");
    assert_eq!(
        decode.boundary.len(),
        config.n_layers + config.n_mtp_layers,
        "the backbone rows and one row per MTP stage"
    );
    for (index, row) in decode.boundary.rows().iter().enumerate() {
        assert_eq!(row.layer(), index, "backbone rows first, then MTP stages");
    }
    let stage = config.n_layers;
    let row = decode
        .boundary
        .row(stage)
        .expect("the MTP stage recorded its row");
    assert_eq!(row.layer(), stage);
    assert_row_is_documented(&config, row, config.dspark_block_size);

    // The stage is a real pass over the draft block, and the capture says so: the block's rows are
    // present and the MTP row is not a copy of a backbone row's step.
    assert_eq!(decode.boundary.start_position(), 3);
    assert_eq!(decode.boundary.tokens(), &[4]);
    assert_eq!(
        decode.boundary.consumed_prefix(),
        &[1, 2, 3],
        "the decode's header names the history its caches were built from"
    );
    assert_eq!(
        decode.boundary.mtp_anchor(),
        decode.output.next_token,
        "the MTP stages consume the backbone's sampled anchor"
    );
    assert!(
        decode.boundary.mtp_anchor().is_some(),
        "a step with an MTP row records its anchor"
    );
    assert!(
        decode.output.draft.len() > 1,
        "the decode step produced a draft, which is what makes the stages run"
    );
}

#[test]
fn the_header_records_the_consumed_prefix_and_the_mtp_anchor() {
    let config = ModelConfig::tiny();
    let configured = engine(&config, 5, None);
    let mut state = SequenceState::new(&config);

    let prefill = configured
        .prefill_observed(&mut state, &[1, 2, 3])
        .expect("the observed prefill runs");
    assert_eq!(prefill.boundary.start_position(), 0);
    assert_eq!(prefill.boundary.tokens(), &[1, 2, 3]);
    assert!(
        prefill.boundary.consumed_prefix().is_empty(),
        "a first step consumed no prefix"
    );
    assert_eq!(prefill.boundary.mtp_anchor(), None);

    let decode = configured
        .decode_observed(&mut state, 4)
        .expect("the observed decode runs");
    assert_eq!(decode.boundary.start_position(), 3);
    assert_eq!(decode.boundary.tokens(), &[4]);
    assert_eq!(
        decode.boundary.consumed_prefix(),
        &[1, 2, 3],
        "the decode's header names the history its caches were built from"
    );
    assert_eq!(
        decode.boundary.mtp_anchor(),
        decode.output.next_token,
        "the MTP stages consume the backbone's sampled anchor"
    );
    assert!(decode.boundary.mtp_anchor().is_some());
}

#[test]
fn a_diff_of_the_same_decode_header_over_a_different_prefix_is_refused() {
    // The review's repro: prefill [7] then decode 8 and prefill [6] then decode 8 both have
    // start_position 1 and tokens [8], and both run 19 rows. They are different steps because the
    // window/compressed caches were built from different prefixes; a header that omits the prefix
    // compares them and reports 18 differing backbone rows instead of refusing.
    let config = eighteen_layers();
    let decode_after = |prefix: &[u32], token: u32| {
        let configured = engine(&config, 41, None);
        let mut state = SequenceState::new(&config);
        configured
            .prefill_observed(&mut state, prefix)
            .expect("the observed prefill runs");
        configured
            .decode_observed(&mut state, token)
            .expect("the observed decode runs")
    };
    let left = decode_after(&[7], 8);
    let right = decode_after(&[6], 8);

    assert_eq!(left.boundary.start_position(), 1);
    assert_eq!(
        left.boundary.start_position(),
        right.boundary.start_position()
    );
    assert_eq!(left.boundary.tokens(), &[8]);
    assert_eq!(left.boundary.tokens(), right.boundary.tokens());
    assert_eq!(left.boundary.len(), config.n_layers + config.n_mtp_layers);
    assert_eq!(left.boundary.len(), right.boundary.len());
    assert_eq!(left.boundary.consumed_prefix(), &[7]);
    assert_eq!(right.boundary.consumed_prefix(), &[6]);

    let refused = left
        .boundary
        .diff(&right.boundary)
        .expect_err("different histories are different steps");
    assert_eq!(
        refused.to_string(),
        "boundary diff: the tables describe different steps: consumed prefix [7] against prefix [6]"
    );
}

#[test]
fn a_diff_of_the_same_header_over_a_different_mtp_anchor_is_refused() {
    // The same weights and the same step header, but a different sampler picks a different anchor
    // for the MTP stages. The anchor is an input to every MTP row (model.py:1131-1132), so the two
    // tables are not comparable even though the backbone rows are bit-identical.
    let config = ModelConfig::tiny();
    let decode_with = |sampling: Sampling| {
        let configured = engine(&config, 5, None).with_sampling(sampling);
        let mut state = SequenceState::new(&config);
        configured
            .prefill_observed(&mut state, &[7])
            .expect("the observed prefill runs");
        configured
            .decode_observed(&mut state, 4)
            .expect("the observed decode runs")
    };
    let first = decode_with(Sampling {
        temperature: 1.0,
        seed: 1,
    });
    let first_anchor = first.boundary.mtp_anchor().expect("the MTP row ran");
    let mut other = None;
    for seed in 2..64 {
        let candidate = decode_with(Sampling {
            temperature: 1.0,
            seed,
        });
        if candidate.boundary.mtp_anchor() != Some(first_anchor) {
            other = Some(candidate);
            break;
        }
    }
    let other = other.expect("a sampler seed in 2..64 picks a different anchor");

    assert_eq!(
        first.boundary.start_position(),
        other.boundary.start_position()
    );
    assert_eq!(first.boundary.tokens(), other.boundary.tokens());
    assert_eq!(
        first.boundary.consumed_prefix(),
        other.boundary.consumed_prefix()
    );
    assert_eq!(first.boundary.len(), other.boundary.len());
    assert_ne!(
        first.boundary.mtp_anchor(),
        other.boundary.mtp_anchor(),
        "the two runs must disagree on the anchor for this test to measure anything"
    );

    let expected = format!(
        "boundary diff: the tables describe different steps: MTP anchor {:?} against anchor {:?}",
        first.boundary.mtp_anchor(),
        other.boundary.mtp_anchor()
    );
    let refused = first
        .boundary
        .diff(&other.boundary)
        .expect_err("a different anchor is a different step");
    assert_eq!(refused.to_string(), expected);
}

#[test]
fn a_dim_that_does_not_divide_into_32_element_blocks_is_refused() {
    // kernel.py:108 asserts the quantised row divides into whole 32-element blocks; the observed
    // entry point refuses before a weight is read, so the state stays untouched.
    let config = ModelConfig {
        dim: 48,
        ..ModelConfig::tiny()
    };
    let configured = engine(&config, 1, None);
    let mut state = SequenceState::new(&config);
    let refused = configured
        .prefill_observed(&mut state, &[7])
        .expect_err("48 is not a multiple of 32");
    assert_eq!(
        refused.to_string(),
        "boundary capture: dim 48 is not a multiple of the 32 element quantisation block"
    );
    assert!(state.is_empty(), "the refusal happens before the step runs");
}

#[test]
fn a_diff_of_a_different_token_at_the_same_start_is_refused() {
    let config = ModelConfig::tiny();
    let left = observed_prefill_tokens(&config, 9, None, &[7]).boundary;
    let right = observed_prefill_tokens(&config, 9, None, &[8]).boundary;
    assert_eq!(left.start_position(), right.start_position());
    assert_eq!(left.start_position(), 0);
    assert_eq!(left.consumed_prefix(), right.consumed_prefix());
    assert_eq!(left.len(), right.len());
    let refused = left.diff(&right).expect_err("different tokens");
    assert_eq!(
        refused.to_string(),
        "boundary diff: the tables describe different steps: tokens [7] against tokens [8]"
    );
}

#[test]
fn a_diff_of_the_same_token_at_a_different_start_is_refused() {
    // The review's B4 gap: the two tables hold the same token [7], so the token check cannot catch
    // them; the start position is the only header field that differs (the unit test in boundary.rs
    // even holds the prefix equal, which this engine-generated pair cannot).
    let config = ModelConfig::tiny();
    let first = observed_prefill_tokens(&config, 9, None, &[7]).boundary;
    let configured = engine(&config, 9, None);
    let mut state = SequenceState::new(&config);
    configured
        .prefill_observed(&mut state, &[6])
        .expect("the first observed prefill runs");
    let second = configured
        .prefill_observed(&mut state, &[7])
        .expect("the second observed prefill runs")
        .boundary;
    assert_eq!(first.start_position(), 0);
    assert_eq!(second.start_position(), 1);
    assert_eq!(first.tokens(), second.tokens());
    assert_eq!(first.tokens(), &[7]);
    assert!(first.consumed_prefix().is_empty());
    assert_eq!(second.consumed_prefix(), &[6]);
    let refused = first.diff(&second).expect_err("different start positions");
    assert_eq!(
        refused.to_string(),
        "boundary diff: the tables describe different steps: position 0 against position 1"
    );
}

#[test]
fn an_mtp_row_at_hc_mult_one_records_a_comb_matrix() {
    // `hc_mult == 1` is the shape in which a size-chosen comb would collapse to 2-D; the MTP row
    // must go through `push_comb` just as the backbone rows do.
    let config = ModelConfig {
        hc_mult: 1,
        ..ModelConfig::tiny()
    };
    let configured = engine(&config, 71, None);
    let mut state = SequenceState::new(&config);
    configured
        .prefill_observed(&mut state, &[7])
        .expect("the observed prefill runs");
    let decode = configured
        .decode_observed(&mut state, 8)
        .expect("the observed decode runs");
    let stage = config.n_layers;
    let row = decode
        .boundary
        .row(stage)
        .expect("the MTP stage recorded its row");
    assert_row_is_documented(&config, row, config.dspark_block_size);
    assert_eq!(
        row.tensor(boundary::HC_ATTN_COMB)
            .expect("documented tensor")
            .shape(),
        [config.dspark_block_size, 1, 1],
        "an hc_mult == 1 MTP row still records a [positions, 1, 1] comb"
    );
    assert_eq!(
        row.tensor(boundary::HC_FFN_COMB)
            .expect("documented tensor")
            .shape(),
        [config.dspark_block_size, 1, 1]
    );
}

#[test]
fn a_plain_decode_is_the_observed_decode() {
    // The prefill half of the plain/observed equality is in
    // `the_observed_step_is_the_plain_step_and_two_runs_capture_identically`; this covers the
    // decode half, where the MTP stages also run.
    let config = ModelConfig::tiny();
    let configured = engine(&config, 31, None);
    let mut plain_state = SequenceState::new(&config);
    let mut observed_state = SequenceState::new(&config);
    configured
        .prefill(&mut plain_state, &[7])
        .expect("the plain prefill runs");
    configured
        .prefill_observed(&mut observed_state, &[7])
        .expect("the observed prefill runs");
    let plain = configured
        .decode(&mut plain_state, 8)
        .expect("the plain decode runs");
    let observed = configured
        .decode_observed(&mut observed_state, 8)
        .expect("the observed decode runs");
    assert_eq!(plain.logits, observed.output.logits, "same logits");
    assert_eq!(plain.next_token, observed.output.next_token);
    assert_eq!(plain.draft, observed.output.draft, "same drafted tokens");
    assert_eq!(plain.confidence, observed.output.confidence);
    assert_eq!(plain.consumed, observed.output.consumed);
    assert_eq!(
        observed.boundary.len(),
        config.n_layers + config.n_mtp_layers,
        "the observed decode is the one that also records the stage rows"
    );
}

/// The released layer counts at small shapes: 40 backbone layers and three MTP stages, with the
/// released `dspark_target_layer_ids` 37/38/39 (model.py:1111-1113), so the row-count rule can be
/// checked at the shape the roadmap states instead of only at the two-layer tiny configuration.
fn released_depth() -> ModelConfig {
    const LAYERS: usize = 40;
    const STAGES: usize = 3;
    ModelConfig {
        n_layers: LAYERS,
        n_mtp_layers: STAGES,
        compress_ratios: vec![0; LAYERS + STAGES],
        kv_source_layers: Vec::new(),
        index_source_layers: Vec::new(),
        candidate_source_layer: 0,
        engram_layer_ids: Vec::new(),
        engram_num_embeddings: Vec::new(),
        dspark_target_layer_ids: vec![37, 38, 39],
        ..ModelConfig::tiny()
    }
}

/// The same 43-row decode rule at a geometry whose heads are wider than the residual stream
/// (8 heads of 32 = 256 values against dim 64, four times wider; the released geometry is 6.4
/// times wider). The backbone scratch and the MTP stage scratch both write the head buffer before
/// the projection narrows it; the
/// released config cannot run yet (the Engram tables need a paging source), so this is the host
/// shape that keeps both buffers honest.
#[test]
fn a_wide_head_decode_records_all_forty_three_rows() {
    use sparkflash_engine::forward::boundary::ATTENTION_OUT;

    let config = ModelConfig {
        n_heads: 8,
        head_dim: 32,
        o_groups: 2,
        ..released_depth()
    };
    assert!(config.n_heads * config.head_dim > config.dim);
    let configured = engine(&config, 63, None);
    let mut state = SequenceState::new(&config);
    configured
        .prefill_observed(&mut state, &[1, 2])
        .expect("the wide-head prefill runs");
    let decode = configured
        .decode_observed(&mut state, 3)
        .expect("the wide-head decode runs the MTP stages");
    assert_eq!(
        decode.boundary.len(),
        config.n_layers + config.n_mtp_layers,
        "40 backbone rows and 3 stage rows"
    );
    for row in decode.boundary.rows() {
        let elements = row
            .tensor(ATTENTION_OUT)
            .expect("every row records its attention output")
            .data()
            .elements();
        if row.layer() < config.n_layers {
            assert_eq!(
                elements,
                config.dim,
                "backbone layer {} records one dim-wide row for the decode token",
                row.layer()
            );
        } else {
            assert_eq!(
                elements,
                config.dim * config.dspark_block_size,
                "stage {} records one dim-wide row per draft position",
                row.layer()
            );
        }
    }
}

/// The roadmap's row-count rule at the released layer counts: a first step has exactly
/// `n_layers` rows (40), and a decode step after a prefill has `n_layers + n_mtp_layers` (43), with
/// the stages at layer ids `n_layers + stage`.
#[test]
fn the_released_row_count_rules_hold_at_forty_layers_and_three_stages() {
    let config = released_depth();
    assert_eq!(config.n_layers, 40);
    assert_eq!(config.n_mtp_layers, 3);
    let configured = engine(&config, 61, None);
    let mut state = SequenceState::new(&config);

    let prefill = configured
        .prefill_observed(&mut state, &[1, 2])
        .expect("the observed prefill runs");
    assert_eq!(
        prefill.boundary.len(),
        config.n_layers,
        "a prefill/first step has exactly n_layers rows"
    );
    assert_eq!(prefill.boundary.start_position(), 0);
    for (index, row) in prefill.boundary.rows().iter().enumerate() {
        assert_eq!(row.layer(), index, "backbone rows only, in layer order");
    }

    let decode = configured
        .decode_observed(&mut state, 3)
        .expect("the observed decode runs");
    assert_eq!(
        decode.boundary.len(),
        config.n_layers + config.n_mtp_layers,
        "a decode step after a prefill has n_layers + n_mtp_layers rows"
    );
    assert_eq!(decode.boundary.start_position(), 2);
    assert_eq!(decode.boundary.tokens(), &[3]);
    for (index, row) in decode.boundary.rows().iter().enumerate() {
        assert_eq!(row.layer(), index, "backbone rows first, then the stages");
    }
    for stage in 0..config.n_mtp_layers {
        let layer = config.n_layers + stage;
        let row = decode
            .boundary
            .row(layer)
            .unwrap_or_else(|| panic!("stage {stage} recorded a row"));
        assert_eq!(row.layer(), layer);
        assert_row_is_documented(&config, row, config.dspark_block_size);
    }
}

/// `hc_mult == 1` is a valid configuration (validation refuses only 0), and its per-token comb
/// size is 1 - the same count as a pre/post vector - so a shape chosen from the value count would
/// record a 2-D comb. The comb is always a matrix per token.
#[test]
fn one_mhc_copy_still_records_a_comb_matrix() {
    let config = ModelConfig {
        hc_mult: 1,
        ..ModelConfig::tiny()
    };
    let observed = observed_prefill(&config, 71, None);
    assert_eq!(observed.boundary.len(), config.n_layers);
    let row = observed.boundary.row(0).expect("layer 0 ran");
    assert_eq!(
        row.tensor(boundary::HC_ATTN_COMB)
            .expect("documented tensor")
            .shape(),
        [1, 1, 1],
        "one mHC copy still records a [positions, 1, 1] comb"
    );
    assert_eq!(
        row.tensor(boundary::HC_FFN_COMB)
            .expect("documented tensor")
            .shape(),
        [1, 1, 1]
    );
    assert_row_is_documented(&config, row, 1);
}

#[test]
fn a_middle_layer_change_propagates_and_the_first_divergence_is_its_layer() {
    // The composed-run fact the format document states: a change in a middle layer is visible in
    // every later row too, because those layers receive a different residual stream. The diff
    // therefore attributes a composed run by its FIRST diverging row; localising the fault to the
    // changed layer needs the layer replay protocol of format document section 7.
    let config = eighteen_layers();
    let clean = observed_prefill(&config, 37, None);
    let changed_layer = 5;
    let perturbed = observed_prefill(
        &config,
        37,
        Some(OneCodeChange {
            tensor: format!("layers.{changed_layer}.attn.wq_b.weight"),
            element: 3,
        }),
    );
    let diff = clean
        .boundary
        .diff(&perturbed.boundary)
        .expect("two captures of the same step");
    assert_eq!(
        diff.first_differing(),
        Some(changed_layer),
        "the first diverging row is the changed layer"
    );
    for row in diff.rows().iter().filter(|row| row.layer() < changed_layer) {
        assert!(
            row.is_identical(),
            "layer {}: nothing before the changed layer can move",
            row.layer()
        );
    }
    for row in diff.rows().iter().filter(|row| row.layer() > changed_layer) {
        assert!(
            !row.is_identical(),
            "layer {}: the changed residual stream reaches every later layer, so an \
             end-to-end diff cannot be zero there",
            row.layer()
        );
    }
}

#[test]
fn a_diff_between_different_steps_or_layer_sets_is_refused() {
    let config = eighteen_layers();
    let table = observed_prefill(&config, 41, None).boundary;
    // Same tokens and start position, but a different configuration: 2 rows against 18 rows. A diff
    // across them is a wiring error, not a difference, so it is refused by name.
    let tiny = ModelConfig::tiny();
    let tiny_table = observed_prefill(&tiny, 41, None).boundary;
    let refused = table.diff(&tiny_table).expect_err("different layer sets");
    assert_eq!(
        refused.to_string(),
        "boundary diff: 18 rows against 2 rows",
        "the row-count refusal names both counts"
    );

    // Different tokens at the same start position: a one-token and a two-token prefill. The header
    // check fires before any row is compared, and the exact string is asserted so a weakened or
    // removed token check cannot pass behind a generic "different steps" substring.
    let two_tokens = observed_prefill_tokens(&config, 41, None, &[7, 8]).boundary;
    let refused = table.diff(&two_tokens).expect_err("different tokens");
    assert_eq!(
        refused.to_string(),
        "boundary diff: the tables describe different steps: tokens [7] against tokens [7, 8]"
    );

    // Different start positions: a prefill at position 0 against a decode step at position 1. The
    // start-position check runs first, so the exact position string is the one reported even though
    // the tokens and the consumed prefix differ too.
    let configured = engine(&config, 41, None);
    let mut state = SequenceState::new(&config);
    configured
        .prefill(&mut state, &[7])
        .expect("the plain prefill runs");
    let decode = configured
        .decode_observed(&mut state, 8)
        .expect("the observed decode runs");
    assert_eq!(decode.boundary.start_position(), 1);
    let refused = table.diff(&decode.boundary).expect_err("different steps");
    assert_eq!(
        refused.to_string(),
        "boundary diff: the tables describe different steps: position 0 against position 1"
    );

    // A table is not empty by construction either: this is the assertion a silently disabled
    // capture would trip, alongside the row assertions of the perturbation test.
    assert!(!table.is_empty());
    assert_eq!(table.len(), config.n_layers);
    let empty_ish = BoundaryTable::diff(&table, &table).expect("a table compares with itself");
    assert!(empty_ish.is_identical());
}
