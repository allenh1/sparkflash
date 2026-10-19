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
use sparkflash_engine::forward::attention::{
    AttentionProjectionBackend, HostAttentionProjectionBackend,
};
use sparkflash_engine::forward::engram;
use sparkflash_engine::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource};
use sparkflash_engine::forward::mhc::{self, Mixes};
use sparkflash_engine::forward::moe::MoeDeviceBackend;
use sparkflash_engine::forward::{
    DeferredHandoff, HybridEntry, HybridPost, HybridTokenOutput, InferenceEngine, SequenceState,
};

/// How many FFN epilogue additions the deferred chain staged, and how many are host-side. A mutation
/// that turns the deferral off flips the first count to zero and the second to forty.
fn deferred_epilogue_counts(output: &HybridTokenOutput) -> (usize, usize) {
    let mut deferred = 0;
    let mut host = 0;
    for layer in &output.layers {
        for stage in &layer.stages {
            if stage.stage == "ffn_epilogue" {
                if stage.host_bytes_in == 0 {
                    deferred += 1;
                } else {
                    host += 1;
                }
            }
        }
    }
    (deferred, host)
}
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

fn host_post(
    config: &ModelConfig,
    source: &dyn WeightSource,
    layer: usize,
    residual: &[f32],
    attention_out: &[f32],
    attn_mixes: &Mixes,
) -> Result<HybridPost, EngineError> {
    let prefix = format!("layers.{layer}");
    let dim = config.dim;
    let limit = usize::MAX;
    let mut x = vec![0.0f32; residual.len()];
    mhc::hc_post(
        config,
        attention_out,
        residual,
        &attn_mixes.post,
        &attn_mixes.comb,
        &mut x,
    )?;
    let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_fn"), limit)?;
    let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_scale"), limit)?;
    let base = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_base"), limit)?;
    let mut ffn_mixes = Mixes::for_config(config);
    mhc::hc_mixes(config, &x, &fn_w, &scale, &base, &mut ffn_mixes)?;
    let mut ffn_in = vec![0.0f32; dim];
    mhc::hc_pre(config, &attn_mixes.pre, &x, &mut ffn_in)?;
    let ffn_norm = linalg::decode_by_name(source, &format!("{prefix}.ffn_norm.weight"), limit)?;
    linalg::rms_norm(&mut ffn_in, &ffn_norm, dim, config.norm_eps as f32)?;
    Ok(HybridPost {
        x,
        ffn_in,
        ffn_mixes,
        source: LayerZeroEntrySource::Host,
    })
}

fn host_deferred_entry(
    config: &ModelConfig,
    source: &dyn WeightSource,
    layer: usize,
    hidden: &[f32],
    pre_mix: &[f32],
    deferred: Option<&DeferredHandoff>,
) -> Result<HybridEntry, EngineError> {
    let mut x = if let Some(deferred) = deferred {
        let mut x = vec![0.0f32; hidden.len()];
        mhc::hc_post(
            config,
            &deferred.y,
            &deferred.x,
            &deferred.mixes.post,
            &deferred.mixes.comb,
            &mut x,
        )?;
        x
    } else {
        hidden.to_vec()
    };
    if config.engram_rows(layer).is_some() {
        let mut hashes = Vec::new();
        engram::ngram_hashes(config, &[0], 0, &mut hashes)?;
        engram::engram_forward(config, layer, &mut x, &hashes, source, usize::MAX)?;
    }
    host_entry(config, source, layer, &x, pre_mix)
}

#[test]
fn a_host_controlled_deferred_chain_is_bit_identical_to_reference_prefill() {
    // Forty layers with no compression: the deferral accounting must be 39 deferred + 1 host, and
    // layer 39 (not tiny's layer 1) is the layer with no next entry. Engram stays at layers 1 and 14.
    let mut config = ModelConfig::tiny();
    config.n_layers = 40;
    config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
    config.engram_layer_ids = vec![1, 14];
    config.engram_num_embeddings = vec![1024; 2];
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
    let mut deferred_seen = vec![false; config.n_layers];
    let mut engram_layers = Vec::new();
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            deferred_seen[layer] = deferred.is_some();
            if config.engram_rows(layer).is_some() {
                engram_layers.push(layer);
            }
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let hybrid = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            None,
            None,
            &mut progress,
        )
        .expect("deferred hybrid chain");
    assert_eq!(hybrid.layers.len(), config.n_layers);
    assert_eq!(reference.next_token, hybrid.token);
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
            "head input {index} differs"
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
    let last = hybrid.layers.last().expect("one trace per layer");
    assert_eq!(last.layer, 39, "layer 39 is the layer with no next entry");
    assert_eq!(
        engram_layers,
        vec![1, 14],
        "Engram re-entry is at layers 1 and 14"
    );
    assert!(
        last.stages
            .iter()
            .any(|event| event.stage == "ffn_epilogue" && event.host_bytes_in > 0),
        "layer 39 keeps a host FFN epilogue when there is no next entry"
    );
    assert_eq!(
        deferred_epilogue_counts(&hybrid),
        (config.n_layers - 1, 1),
        "the deferral count must be 39 deferred additions and 1 host addition"
    );
    assert!(
        deferred_seen.iter().skip(1).all(|seen| *seen),
        "every layer after 0 must receive a DeferredHandoff; a layer that silently loses it fails here"
    );
    let first = hybrid.layers.first().expect("layer 0");
    assert!(
        first
            .stages
            .iter()
            .any(|event| event.stage == "mhc_post" && event.note.contains("MhcPost handoff")),
        "layer 0 uses the deferred post hook instead of the host hc_post"
    );
}

struct RecordingAttentionBackend {
    inner: HostAttentionProjectionBackend,
    calls: usize,
    perturb: bool,
    body: bool,
    body_calls: usize,
    body_strides: Vec<usize>,
}

impl Default for RecordingAttentionBackend {
    fn default() -> Self {
        Self {
            inner: HostAttentionProjectionBackend,
            calls: 0,
            perturb: false,
            body: false,
            body_calls: 0,
            body_strides: Vec::new(),
        }
    }
}

/// A host-control routed-MoE backend: answers `true` once per layer and counts the 17 launches the
/// real c4b backend would issue (1 router + 8 gate/up + 4 weighted + 4 combine). Returning `true`
/// must make the engine skip the host routed block and run only the host shared-expert seam.
#[derive(Default)]
struct RecordingMoeBackend {
    calls: usize,
    launches: usize,
}

impl MoeDeviceBackend for RecordingMoeBackend {
    fn ffn(
        &mut self,
        _config: &ModelConfig,
        _layer: usize,
        _prefix: &str,
        x: &[f32],
        _source: &dyn WeightSource,
        _limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        self.calls += 1;
        self.launches += 17;
        out.fill(0.0);
        let _ = x;
        Ok(true)
    }
}

#[derive(Default)]
struct RecordingC4cBackend {
    routed_calls: usize,
    shared_calls: usize,
    norm_calls: usize,
    head_calls: usize,
    launches: usize,
}

impl MoeDeviceBackend for RecordingC4cBackend {
    fn ffn(
        &mut self,
        _config: &ModelConfig,
        _layer: usize,
        _prefix: &str,
        _x: &[f32],
        _source: &dyn WeightSource,
        _limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        self.routed_calls += 1;
        self.launches += 17;
        out.fill(0.0);
        Ok(true)
    }

    fn shared_expert(
        &mut self,
        _config: &ModelConfig,
        _layer: usize,
        _prefix: &str,
        _x: &[f32],
        _source: &dyn WeightSource,
        _limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        self.shared_calls += 1;
        self.launches += 4;
        out.fill(0.0);
        Ok(true)
    }

    fn final_norm(
        &mut self,
        config: &ModelConfig,
        x: &[f32],
        _source: &dyn WeightSource,
        _limit: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, EngineError> {
        let rows = x.len() / config.dim;
        self.norm_calls += 1;
        self.launches += 1;
        out.clear();
        out.resize(rows * config.dim * 2, 0);
        Ok(true)
    }

    fn lm_head(
        &mut self,
        _config: &ModelConfig,
        _hidden: &[f32],
        _source: &dyn WeightSource,
        _limit: usize,
        logits: &mut [f32],
    ) -> Result<Option<u32>, EngineError> {
        self.head_calls += 1;
        self.launches += 4;
        logits.fill(0.0);
        Ok(Some(0))
    }
}

impl AttentionProjectionBackend for RecordingAttentionBackend {
    fn wq_a(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        self.calls += 1;
        self.inner.wq_a(config, layer, prefix, x, src, limit)
    }
    fn wq_b(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        self.calls += 1;
        self.inner.wq_b(config, layer, prefix, latent, src, limit)
    }
    fn wkv(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        self.calls += 1;
        self.inner.wkv(config, layer, prefix, x, src, limit)
    }
    fn wo_a(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        self.calls += 1;
        self.inner.wo_a(config, prefix, wide, src, limit)
    }
    fn wo_b(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        grouped: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        self.calls += 1;
        let mut out = self.inner.wo_b(config, prefix, grouped, src, limit)?;
        if self.perturb {
            for value in &mut out {
                *value += 0.25;
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        _layer: usize,
        _q: &[f32],
        _kv: &[f32],
        _sink: &[f32],
        _combined: &[usize],
        _heads: usize,
        _head_dim: usize,
        _seqlen: usize,
        _window_width: usize,
        compressed_stride: usize,
        _scale: f32,
        _start_pos: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        if !self.body {
            return Ok(false);
        }
        self.body_calls += 1;
        self.body_strides.push(compressed_stride);
        for value in out.iter_mut() {
            *value = 0.0;
        }
        Ok(true)
    }
}

#[test]
fn a_host_controlled_attention_backend_seam_is_bit_identical_to_reference_prefill() {
    // The seam must be exact at the model's full depth, not only at tiny()'s two layers.
    let mut config = ModelConfig::tiny();
    config.n_layers = 40;
    config.compress_ratios = {
        let mut ratios = vec![1; config.n_layers];
        ratios.push(0);
        ratios
    };
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
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let mut backend = RecordingAttentionBackend::default();
    let hybrid = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            Some(&mut backend),
            None,
            &mut progress,
        )
        .expect("deferred hybrid chain with the attention backend seam");
    assert_eq!(
        backend.calls,
        5 * config.n_layers,
        "the seam must replace exactly the five projection GEMMs per layer"
    );
    assert_eq!(reference.next_token, hybrid.token);
    for (index, (left, right)) in reference
        .head_input
        .iter()
        .zip(hybrid.head_input.iter())
        .enumerate()
    {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "head input {index} differs"
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

    // Perturbing the seam's final projection must move the chain, proving the values are consumed
    // rather than recorded and ignored.
    let mut perturbed = RecordingAttentionBackend {
        perturb: true,
        ..RecordingAttentionBackend::default()
    };
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let changed = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            Some(&mut perturbed),
            None,
            &mut progress,
        )
        .expect("perturbed attention backend chain");
    assert!(
        hybrid
            .head_input
            .iter()
            .zip(changed.head_input.iter())
            .any(|(left, right)| left.to_bits() != right.to_bits()),
        "a perturbed attention projection must change the chain output"
    );
}

#[test]
fn the_attention_body_seam_is_invoked_once_per_layer_and_changes_the_chain() {
    let mut config = ModelConfig::tiny();
    // Forty layers, every one a ratio-1 compressed layer: the body seam must fire forty times, and
    // each ratio-1 layer must reach the backend with the one-candidate stride 1, not the stride 0
    // that made b32 reject twenty ratio-1 layers and fall back to the host loop.
    config.n_layers = 40;
    config.compress_ratios = {
        let mut ratios = vec![1; config.n_layers];
        ratios.push(0);
        ratios
    };
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
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let mut backend = RecordingAttentionBackend {
        body: true,
        ..RecordingAttentionBackend::default()
    };
    let hybrid = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            Some(&mut backend),
            None,
            &mut progress,
        )
        .expect("deferred hybrid chain with a device attention body");
    assert_eq!(
        backend.body_calls, config.n_layers,
        "the attention body must be called once per layer"
    );
    assert_eq!(
        backend.calls,
        5 * config.n_layers,
        "the projection seam still sees its five calls per layer"
    );
    assert_eq!(
        backend.body_strides,
        vec![1; config.n_layers],
        "every ratio-1 layer must reach the body with stride 1; stride 0 is the b32 rejection"
    );
    assert!(
        reference
            .head_input
            .iter()
            .zip(hybrid.head_input.iter())
            .any(|(left, right)| left.to_bits() != right.to_bits()),
        "a backend that replaces the attention body must change the chain output"
    );
}

#[test]
fn the_routed_moe_seam_is_invoked_once_per_layer_and_issues_680_launches() {
    let mut config = ModelConfig::tiny();
    config.n_layers = 40;
    config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(FiniteSource::from_plan(&plan, 0x5eed)),
        usize::MAX,
    )
    .expect("engine");
    let provider_source = FiniteSource::from_plan(&plan, 0x5eed);
    let embedding = embedding_for(&provider_source, &config);
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let mut moe = RecordingMoeBackend::default();
    engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            None,
            Some(&mut moe),
            &mut progress,
        )
        .expect("deferred hybrid chain with a routed-MoE backend");
    assert_eq!(
        moe.calls, config.n_layers,
        "the routed MoE seam must be called once per layer"
    );
    assert_eq!(
        moe.launches,
        17 * config.n_layers,
        "forty layers of routing must still account for 17 launches per layer"
    );
}

#[test]
fn a_perturbed_post_coefficient_changes_the_deferred_hybrid_output() {
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
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut plain_post =
        |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
            host_post(
                &config,
                &provider_source,
                layer,
                residual,
                attention_out,
                attn_mixes,
            )
        };
    let baseline = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut plain_post,
            None,
            None,
            &mut progress,
        )
        .expect("baseline deferred chain");
    let mut changed_post =
        |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
            let mut post = host_post(
                &config,
                &provider_source,
                layer,
                residual,
                attention_out,
                attn_mixes,
            )?;
            if layer == 1 {
                post.ffn_mixes.pre[0] += 1.0;
            }
            Ok(post)
        };
    let changed = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut changed_post,
            None,
            None,
            &mut progress,
        )
        .expect("perturbed deferred chain");
    assert_finite("baseline deferred head_input", &baseline.head_input);
    assert_finite("perturbed deferred head_input", &changed.head_input);
    assert!(
        baseline
            .head_input
            .iter()
            .zip(&changed.head_input)
            .any(|(left, right)| left.to_bits() != right.to_bits()),
        "a block that recomputes the post coefficient instead of consuming the device post handoff would ignore this perturbation"
    );
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

#[test]
fn the_engine_seam_calls_forty_shared_one_norm_and_one_head() {
    // The test's name is the requirement: forty layers, not tiny()'s two.
    let mut config = ModelConfig::tiny();
    config.n_layers = 40;
    config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(FiniteSource::from_plan(&plan, 0x5eed)),
        usize::MAX,
    )
    .expect("engine");
    let provider_source = FiniteSource::from_plan(&plan, 0x5eed);
    let embedding = embedding_for(&provider_source, &config);
    let mut provider =
        |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
            host_deferred_entry(&config, &provider_source, layer, hidden, pre_mix, deferred)
        };
    let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], attn_mixes: &Mixes| {
        host_post(
            &config,
            &provider_source,
            layer,
            residual,
            attention_out,
            attn_mixes,
        )
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let mut backend = RecordingC4cBackend::default();
    let _ = engine
        .run_hybrid_chain_deferred(
            0,
            &embedding,
            &mut provider,
            &mut post,
            None,
            Some(&mut backend),
            &mut progress,
        )
        .expect("deferred chain with the c4c MoE/norm/head seam");
    assert_eq!(
        backend.routed_calls, config.n_layers,
        "one routed MoE call/layer"
    );
    assert_eq!(
        backend.shared_calls, config.n_layers,
        "one shared-expert call/layer"
    );
    assert_eq!(backend.norm_calls, 1, "one final norm");
    assert_eq!(backend.head_calls, 1, "one head call");
    assert_eq!(
        backend.launches,
        21 * config.n_layers + 1 + 4,
        "17 routed + 4 shared per layer, plus norm and head"
    );
}
