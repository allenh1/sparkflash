//! The hybrid chain's host-control contract: when every MhcEntry handoff is the reference host
//! arithmetic over finite f32 weights, the 40-layer chain must reproduce the reference prefill bit
//! for bit. This proves the chain wiring, the stage order and the tail without a driver; the device
//! arm only substitutes the handoff values.
//!
//! The source is the shared finite f32 source, not SyntheticWeights: raw synthetic bytes can decode
//! to NaN or inf, and a bit comparison over NaNs is vacuous.

mod common;

use common::FiniteSource;
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource};
use sparkflash_engine::forward::mhc::{self, Mixes};
use sparkflash_engine::forward::{HybridEntry, InferenceEngine, SequenceState};
use sparkflash_engine::linalg::{self, EngineError};
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;

fn host_entry(
    config: &ModelConfig,
    source: &dyn WeightSource,
    layer: usize,
    hidden: &[f32],
    pre_mix: &[f32],
) -> Result<HybridEntry, EngineError> {
    let prefix = format!("layers.{layer}");
    let dim = config.dim;
    let limit = usize::MAX;
    let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_fn"), limit)?;
    let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_scale"), limit)?;
    let base = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_base"), limit)?;
    let mut mixes = Mixes::for_config(config);
    mhc::hc_mixes(config, hidden, &fn_w, &scale, &base, &mut mixes)?;
    let mut attn_in = vec![0.0f32; dim];
    mhc::hc_pre(config, pre_mix, hidden, &mut attn_in)?;
    let norm = linalg::decode_by_name(source, &format!("{prefix}.attn_norm.weight"), limit)?;
    let mut attention_hidden = attn_in;
    linalg::rms_norm(&mut attention_hidden, &norm, dim, config.norm_eps as f32)?;
    Ok(HybridEntry {
        entry: LayerZeroEntry {
            pre_attention_hidden: attention_hidden,
            pre: mixes.pre,
            post: mixes.post,
            comb: mixes.comb,
            source: LayerZeroEntrySource::Host,
        },
        x: hidden.to_vec(),
    })
}

fn assert_finite(label: &str, values: &[f32]) {
    assert!(!values.is_empty(), "{label} must not be empty");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} must be finite so a bit compare is meaningful"
    );
}

fn embedding_for(source: &dyn WeightSource, config: &ModelConfig) -> Vec<f32> {
    let table = linalg::decode_by_name(source, "embed.weight", usize::MAX).expect("embed");
    table[..config.dim].to_vec()
}

#[test]
fn a_host_controlled_hybrid_chain_is_bit_identical_to_reference_prefill() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(FiniteSource::from_plan(&plan, 0x5eed)),
        usize::MAX,
    )
    .expect("engine");
    let provider_source = FiniteSource::from_plan(&plan, 0x5eed);
    let embedding = embedding_for(&provider_source, &config);

    let mut reference_state = SequenceState::new(&config);
    let reference = engine
        .prefill(&mut reference_state, &[0])
        .expect("reference prefill");

    let mut provider = |layer: usize, hidden: &[f32], pre_mix: &[f32]| {
        host_entry(&config, &provider_source, layer, hidden, pre_mix)
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let hybrid = engine
        .run_hybrid_chain(0, &embedding, &mut provider, &mut progress)
        .expect("hybrid chain");

    assert_eq!(hybrid.layers.len(), config.n_layers);
    assert_eq!(reference.next_token, hybrid.token);
    assert_eq!(reference.head_input.len(), hybrid.head_input.len());
    assert_eq!(reference.logits.len(), hybrid.logits.len());
    assert_finite("reference head_input", &reference.head_input);
    assert_finite("hybrid head_input", &hybrid.head_input);
    assert_finite("reference logits", &reference.logits);
    assert_finite("hybrid logits", &hybrid.logits);
    for (index, (left, right)) in reference
        .head_input
        .iter()
        .zip(hybrid.head_input.iter())
        .enumerate()
    {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "head input element {index} differs"
        );
    }
    for (index, (left, right)) in reference
        .logits
        .iter()
        .zip(hybrid.logits.iter())
        .enumerate()
    {
        assert_eq!(left.to_bits(), right.to_bits(), "logit {index} differs");
    }
}

#[test]
fn a_perturbed_entry_capture_changes_the_hybrid_output() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(FiniteSource::from_plan(&plan, 0x5eed)),
        usize::MAX,
    )
    .expect("engine");
    let provider_source = FiniteSource::from_plan(&plan, 0x5eed);
    let embedding = embedding_for(&provider_source, &config);

    let mut progress = |_layer: usize, _host_ms: u64| {};
    let mut plain = |layer: usize, hidden: &[f32], pre_mix: &[f32]| {
        host_entry(&config, &provider_source, layer, hidden, pre_mix)
    };
    let baseline = engine
        .run_hybrid_chain(0, &embedding, &mut plain, &mut progress)
        .expect("baseline hybrid chain");
    let mut perturbed = |layer: usize, hidden: &[f32], pre_mix: &[f32]| {
        let mut entry = host_entry(&config, &provider_source, layer, hidden, pre_mix)?;
        entry.entry.pre_attention_hidden[0] += 1.0;
        Ok(entry)
    };
    let changed = engine
        .run_hybrid_chain(0, &embedding, &mut perturbed, &mut progress)
        .expect("perturbed hybrid chain");

    assert_finite("baseline head_input", &baseline.head_input);
    assert_finite("perturbed head_input", &changed.head_input);
    assert!(
        baseline
            .head_input
            .iter()
            .zip(&changed.head_input)
            .any(|(left, right)| left.to_bits() != right.to_bits()),
        "a block that recomputes the capture instead of consuming the handoff would ignore this perturbation"
    );
}
