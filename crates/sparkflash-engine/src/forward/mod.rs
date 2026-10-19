//! The reference forward pass: DeepSeek-V4.1-Flash, run on `f32` over a [crate::weights::WeightSource].
//!
//! This is a *reference*, and it is deliberately the smallest thing that can carry the architecture's
//! shape flow: one sequence, one chunk at a time, no batching, no CUDA, no quantised kernels.
//! `docs/engine/01-forward-order.md` is its specification - every step below carries the `model.py`
//! line range that document cites, and each module it calls carries the detail.
//!
//! # Why it exists
//!
//! The engine has to be testable on a host with no weights, no GPU and no cluster (510 GB of
//! checkpoint, 98 GB per Engram table). [crate::synthetic::SyntheticWeights] produces every tensor of
//! a [crate::tensor::TensorPlan] from a seed, so a test can run the whole architecture at a
//! configuration small enough to fit in a test process and assert the properties that do not depend
//! on the numbers: the shape flow, causality, the KV cache growing by exactly one token per decode
//! step, attention never reaching past the window in an SWA layer, and determinism for a fixed seed.
//!
//! What it is NOT is a performance path, and nothing here claims a speed: each weight is read through
//! [crate::linalg::decode_by_name], which refuses a tensor above an element limit, so a released
//! configuration stops with an explicit refusal instead of quietly trying to allocate 98 GB.
//!
//! # The per-layer boundary table
//!
//! [InferenceEngine::prefill_observed] and [InferenceEngine::decode_observed] run this exact pass
//! and also return the per-layer [boundary::BoundaryTable] Phase 1 diffs the device against
//! (`docs/build/phase1-boundary-table.md`, `docs/build/roadmap-to-serving.md` section 4.5). The two
//! entry points delegate to the same `run` with a table instead of `None`, so the default path is
//! the same code and keeps its determinism; [boundary] documents what is recorded and in what order.

pub mod attention;
pub mod attention_case;
pub mod boundary;
pub mod engram;
pub mod layer_zero;
pub mod mhc;
pub mod moe;
pub mod mtp;

use crate::config::ModelConfig;
use crate::forward::boundary::{BoundaryTable, LayerBoundary};
use crate::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource, StageEvent};
use crate::forward::mhc::Mixes;
use crate::linalg::{self, EngineError};
use crate::tensor::TensorPlan;
use crate::weights::WeightSource;

/// Per-sequence state: the tokens consumed and every cache the attention layers own.
///
/// Each backbone layer and each MTP stage has its own window ring and compressed KV, so the caches
/// are indexed by layer. Nothing here is shared between sequences.
pub struct SequenceState {
    tokens: Vec<u32>,
    attention: Vec<attention::AttnState>,
}

impl SequenceState {
    /// Fresh state for one sequence of `config`.
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            tokens: Vec::new(),
            attention: (0..config.n_layers + config.n_mtp_layers)
                .map(|_| attention::AttnState::for_config(config))
                .collect(),
        }
    }

    /// The tokens this sequence has consumed, in order.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// How many tokens the sequence has consumed: the position the next step writes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the sequence has consumed nothing yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The window row a backbone layer holds for `position`, for tests that assert what the cache
    /// kept. `None` when the layer has not written that position.
    #[must_use]
    pub fn window_row(&self, layer: usize, position: usize) -> Option<&[f32]> {
        self.attention.get(layer)?.window.row(position)
    }

    /// The attention state of one layer or MTP stage, for tests.
    #[must_use]
    pub fn attention(&self, layer: usize) -> Option<&attention::AttnState> {
        self.attention.get(layer)
    }
}

/// How the step chooses its next token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0 (the default) is greedy; above it the reference's Gumbel-max draw at that temperature.
    pub temperature: f32,
    /// The sampler's seed. The same logits and seed always give the same token.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 0,
        }
    }
}

/// What one step produced.
#[derive(Debug)]
pub struct StepOutput {
    /// The logits of the last position, `vocab_size` long.
    pub logits: Vec<f32>,
    /// The vector the head multiplied: the last position's collapsed residual after the final RMS
    /// norm, `dim` long. It is the device head harness's input (criterion 3 compares the device
    /// `cert_lm_head_sm121` logits against [Self::logits] on this same vector), and it is captured
    /// rather than recomputed because the collapse and norm are the reference's own.
    pub head_input: Vec<f32>,
    /// The chosen next token, or `None` when the step produced no logits.
    pub next_token: Option<u32>,
    /// The tokens the DSpark stages drafted after it (empty when the tap did not fire).
    pub draft: Vec<u32>,
    /// One confidence per drafted token.
    pub confidence: Vec<f32>,
    /// How many tokens the step consumed.
    pub consumed: usize,
}

/// What one observed step produced: the step's output and the boundary table recorded for it.
#[derive(Debug)]
pub struct ObservedStep {
    /// What the step returned: logits, the sampled token and the draft, exactly as [StepOutput].
    pub output: StepOutput,
    /// The per-layer boundary table, one row per layer that ran, in run order.
    pub boundary: BoundaryTable,
}

/// The reference engine: a configuration, a plan, a weight source and a sampler.
///
/// It owns no per-sequence state - that lives in [SequenceState] - so one engine can drive several
/// sequences, exactly as the reference does.
pub struct InferenceEngine {
    config: ModelConfig,
    plan: TensorPlan,
    source: Box<dyn WeightSource>,
    limit: usize,
    sampling: Sampling,
}

impl InferenceEngine {
    /// An engine over a source, with the plan that source is read against.
    ///
    /// `limit` caps the elements of any single tensor the pass materialises; use
    /// [linalg::REFERENCE_TENSOR_LIMIT] unless a test wants a smaller one.
    ///
    /// # Errors
    ///
    /// [EngineError::Config] when the configuration is not usable.
    pub fn new(
        config: ModelConfig,
        source: Box<dyn WeightSource>,
        limit: usize,
    ) -> Result<Self, EngineError> {
        config.validate()?;
        let plan = TensorPlan::language_model(&config);
        Ok(Self {
            config,
            plan,
            source,
            limit,
            sampling: Sampling::default(),
        })
    }

    /// The same engine with a different sampler.
    #[must_use]
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// The configuration this engine runs.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The plan the weights are read against.
    #[must_use]
    pub fn plan(&self) -> &TensorPlan {
        &self.plan
    }

    /// The element limit in force.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Consumes `tokens` as one chunk and returns the logits of the last position.
    ///
    /// # Errors
    ///
    /// [EngineError] from any weight read, decode or shape check.
    pub fn prefill(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
    ) -> Result<StepOutput, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Shape {
                what: "prefill".to_owned(),
                detail: "no tokens to prefill".to_owned(),
            });
        }
        self.run(state, tokens, None)
    }

    /// Consumes one token and returns the logits for the position after it.
    ///
    /// # Errors
    ///
    /// [EngineError] from any weight read, decode or shape check.
    pub fn decode(&self, state: &mut SequenceState, token: u32) -> Result<StepOutput, EngineError> {
        self.run(state, &[token], None)
    }

    /// [Self::prefill] with the per-layer boundary table recorded
    /// (`docs/build/phase1-boundary-table.md`).
    ///
    /// The forward is the one [Self::prefill] runs and the returned [ObservedStep::output] is what
    /// [Self::prefill] would have returned; the capture only reads.
    ///
    /// # Errors
    ///
    /// As [Self::prefill], plus [EngineError::Shape] when the configuration's `dim` does not divide
    /// into the reference's 32-element activation blocks: the quantised block input the table
    /// carries is one E4M3 code per element with one E8M0 scale per 32 (kernel.py:108), so a
    /// boundary table cannot describe a row the reference itself refuses to quantise.
    pub fn prefill_observed(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
    ) -> Result<ObservedStep, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Shape {
                what: "prefill".to_owned(),
                detail: "no tokens to prefill".to_owned(),
            });
        }
        self.run_observed(state, tokens)
    }

    /// [Self::decode] with the per-layer boundary table recorded, including the DSpark stages when
    /// they run (`docs/build/phase1-boundary-table.md`).
    ///
    /// # Errors
    ///
    /// As [Self::prefill_observed].
    pub fn decode_observed(
        &self,
        state: &mut SequenceState,
        token: u32,
    ) -> Result<ObservedStep, EngineError> {
        self.run_observed(state, &[token])
    }

    /// The observed form of one step: build a table for this step, run the shared pass with it, and
    /// return both. Checked up front because the capture's quantised block input is defined only for
    /// a row of whole 32-element blocks (kernel.py:108).
    fn run_observed(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
    ) -> Result<ObservedStep, EngineError> {
        if !self.config.dim.is_multiple_of(crate::dtype::SCALE_BLOCK) {
            return Err(EngineError::Shape {
                what: "boundary capture".to_owned(),
                detail: format!(
                    // kernel.py:108 asserts the row divides into 32-element blocks.
                    "dim {} is not a multiple of the {} element quantisation block",
                    self.config.dim,
                    crate::dtype::SCALE_BLOCK
                ),
            });
        }
        // The header records the history the step starts from, not only where it starts: a decode
        // table depends on the consumed prefix through the window and compressed/index caches, so
        // the prefix is captured before `run` extends the sequence (BoundaryTable::diff refuses a
        // prefix mismatch even when the start position and chunk tokens agree).
        let mut boundary = BoundaryTable::new(state.len(), tokens, state.tokens());
        let output = self.run(state, tokens, Some(&mut boundary))?;
        Ok(ObservedStep { output, boundary })
    }
}

/// The buffers one step needs, sized once per chunk from the configuration and reused by every
/// layer. Nothing here is model-sized: the weight tensors themselves are never retained.
struct Scratch {
    attn_mixes: Mixes,
    ffn_mixes: Mixes,
    attn_in: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_in: Vec<f32>,
    ffn_out: Vec<f32>,
    residual: Vec<f32>,
    moe: moe::MoeScratch,
}

impl Scratch {
    fn for_chunk(config: &ModelConfig, positions: usize) -> Self {
        let width = positions * config.dim;
        Self {
            attn_mixes: Mixes::for_tokens(config, positions),
            ffn_mixes: Mixes::for_tokens(config, positions),
            attn_in: vec![0.0; width],
            attn_out: vec![0.0; width],
            ffn_in: vec![0.0; width],
            ffn_out: vec![0.0; width],
            residual: vec![0.0; positions * config.hc_width()],
            moe: moe::MoeScratch::for_config(config),
        }
    }
}

/// The per-layer provider the hybrid chain calls: the layer, the previous FFN residual and the
/// previous FFN pre-mix, returning this layer's entry and the residual the host continues from.
pub type HybridProvider<'a> =
    dyn FnMut(usize, &[f32], &[f32]) -> Result<HybridEntry, EngineError> + 'a;

/// One device post handoff: the post-attention residual the device wrote (X), the normalized FFN
/// block input and the 24 FFN coefficients the device published. The host FFN continues from these
/// device values; it does not recompute the mHC coefficient set.
#[derive(Debug, Clone)]
pub struct HybridPost {
    /// The post-attention residual stream the kernel left in X, `hc_mult * dim` f32 values.
    pub x: Vec<f32>,
    /// The normalized FFN block input the kernel captured, `dim` f32 values.
    pub ffn_in: Vec<f32>,
    /// The 24 FFN coefficients the kernel published (pre/post/comb in Mixes form).
    pub ffn_mixes: Mixes,
    /// Where the coefficients and capture came from.
    pub source: LayerZeroEntrySource,
}

/// The previous layer's FFN epilogue, deferred into the next device entry.
#[derive(Debug, Clone)]
pub struct DeferredHandoff {
    /// The post-attention residual X the device left in its X buffer, `hc_mult * dim` f32 values.
    pub x: Vec<f32>,
    /// The previous layer's FFN block output Y, `dim` f32 values.
    pub y: Vec<f32>,
    /// The previous layer's 24 FFN coefficients in Mixes form.
    pub mixes: Mixes,
}

/// The c3b provider: like [HybridProvider], plus the previous layer's deferred FFN epilogue.
pub type HybridDeferredProvider<'a> = dyn FnMut(usize, &[f32], &[f32], Option<&DeferredHandoff>) -> Result<HybridEntry, EngineError>
    + 'a;

/// The device post callback: residual input, attention output, and the attention coefficients.
pub type HybridPostProvider<'a> =
    dyn FnMut(usize, &[f32], &[f32], &Mixes) -> Result<HybridPost, EngineError> + 'a;

/// One hybrid MhcEntry handoff: the engine's entry values plus the residual stream the device left
/// in X. The X is the BF16-rounded stream the host must continue from; it is not the f32 stream it
/// handed in.
#[derive(Debug, Clone)]
pub struct HybridEntry {
    /// The device (or host-control) MhcEntry coefficients and normalised attention input.
    pub entry: LayerZeroEntry,
    /// The residual stream read back from the kernel's X buffer, `hc_mult * dim` f32 values.
    pub x: Vec<f32>,
}

/// One layer of a hybrid chain: the host stages that ran and the residual stream they produced.
#[derive(Debug, Clone)]
pub struct HybridLayerTrace {
    /// The backbone layer.
    pub layer: usize,
    /// The trace stages in [crate::forward::layer_zero::LAYER_ZERO_HOST_STAGES] order, plus Engram
    /// first on the layers that own it.
    pub stages: Vec<StageEvent>,
    /// The residual stream after the host mhc_post/ffn_epilogue, `hc_mult * dim` values. With a
    /// deferred FFN epilogue this is the post-attention X for the 39 deferred layers; only layer 39
    /// keeps the host epilogue add.
    pub residual_out: Vec<f32>,
    /// Host continuation wall time in milliseconds, from the end of the MhcEntry handoff to the end
    /// of the FFN epilogue.
    pub host_ms: u64,
}

/// The result of a one-token hybrid chain: the per-layer traces, the tail stages and the token.
#[derive(Debug, Clone)]
pub struct HybridTokenOutput {
    /// One trace per backbone layer, in order.
    pub layers: Vec<HybridLayerTrace>,
    /// The tail stages: final norm, head screen and argmax.
    pub tail_stages: Vec<StageEvent>,
    /// The collapsed and final-normed vector the head multiplied.
    pub head_input: Vec<f32>,
    /// The raw logits, `vocab_size` values.
    pub logits: Vec<f32>,
    /// The greedy token the host argmax selected.
    pub token: Option<u32>,
}

fn u64_bytes(values: usize) -> u64 {
    u64::try_from(values).unwrap_or(u64::MAX).saturating_mul(4)
}

fn record_stage(
    stages: &mut Option<&mut Vec<StageEvent>>,
    stage: &'static str,
    host_bytes_in: u64,
    host_bytes_out: u64,
    note: &'static str,
) {
    if let Some(stages) = stages.as_deref_mut() {
        stages.push(StageEvent {
            stage,
            host_bytes_in,
            host_bytes_out,
            note,
        });
    }
}

/// Samples one token from `logits`: greedy at temperature 0, Gumbel-max above it.
///
/// The reference samples `argmax(logits / temperature + gumbel_noise)` (model.py:1285-1292), which
/// is a draw from the temperature-scaled softmax. The noise here comes from splitmix64 seeded by
/// `seed`, so the same logits and seed always give the same token. The GPU path's sync-free philox
/// generator is a different milestone: this is the CPU reference and it says so.
#[must_use]
pub fn sample_gumbel(logits: &[f32], temperature: f32, seed: u64) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return linalg::argmax(logits).map(|index| index as u32);
    }
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut best: Option<(u32, f32)> = None;
    for (index, &logit) in logits.iter().enumerate() {
        state = splitmix64(state);
        // A uniform in (0, 1): the top 53 bits, nudged off zero so the logarithm below is finite.
        let uniform = (((state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64);
        let gumbel = -(-(uniform.ln())).ln();
        let score = logit / temperature + gumbel as f32;
        if best.is_none_or(|(_, held)| score > held) {
            best = Some((index as u32, score));
        }
    }
    best.map(|(index, _)| index)
}

/// One splitmix64 step: the generator this file's reference sampler uses.
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl InferenceEngine {
    /// The step [InferenceEngine::prefill] and [InferenceEngine::decode] both run: the whole-model
    /// order of `Transformer.forward` (model.py:1241-1272, section 3.1 of the forward-order doc),
    /// with `seqlen` positions at a time and `start_pos` already in the caches.
    fn run(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        mut capture: Option<&mut BoundaryTable>,
    ) -> Result<StepOutput, EngineError> {
        let config = &self.config;
        let seqlen = tokens.len();
        let start_pos = state.tokens.len();
        let dim = config.dim;
        let hc = config.hc_mult;
        let width = config.hc_width();
        let source: &dyn WeightSource = self.source.as_ref();

        // One hashing pass for the whole chunk, before any layer runs, shared by every Engram layer
        // (model.py:1252). The pass needs the sequence, not just the chunk: an n-gram looks back
        // across the chunk boundary, and [engram::ngram_hashes] walks the positions
        // `start_pos..tokens.len()` of the tokens it is handed, so handing it the chunk alone
        // hashes nothing on a decode step and pads every chunked-prefill row. The reference keeps
        // the token history in one tensor, which is what this concatenation reproduces; it is a
        // reference-only cost (one copy per step) and it is the caller's, not the hasher's, because
        // only the engine holds the history.
        let mut history = Vec::with_capacity(start_pos + seqlen);
        history.extend_from_slice(&state.tokens);
        history.extend_from_slice(tokens);
        let mut hashes = Vec::new();
        engram::ngram_hashes(config, &history, start_pos, &mut hashes)?;

        // The embedding lookup, one row per token (model.py:1253-1255).
        let embed = linalg::decode_by_name(source, "embed.weight", self.limit)?;
        if embed.len() < config.vocab_size * dim {
            return Err(EngineError::Shape {
                what: "embed.weight".to_owned(),
                detail: format!(
                    "a {}x{} table has {} values, so it cannot serve vocab_size {}",
                    config.vocab_size,
                    dim,
                    embed.len(),
                    config.vocab_size
                ),
            });
        }

        // The stream becomes [seqlen, hc, dim]: every copy starts as the same embedding, and the
        // first input mix selects copy 0 (model.py:1258-1260).
        let mut hidden = vec![0.0f32; seqlen * width];
        for (position, &token) in tokens.iter().enumerate() {
            let row = (token as usize) * dim;
            let Some(embedding) = embed.get(row..row + dim) else {
                return Err(EngineError::Shape {
                    what: "embed.weight".to_owned(),
                    detail: format!(
                        "token {token} is outside the {} row table",
                        config.vocab_size
                    ),
                });
            };
            for copy in 0..hc {
                hidden[(position * hc + copy) * dim..(position * hc + copy + 1) * dim]
                    .copy_from_slice(embedding);
            }
        }
        let mut pre_mix = vec![0.0f32; seqlen * hc];
        for position in 0..seqlen {
            pre_mix[position * hc] = 1.0;
        }

        // The layers, ascending, with the Engram write before the block it belongs to and the DSpark
        // tap reading the block INPUT of the three target layers (model.py:1261-1267).
        let mut scratch = Scratch::for_chunk(config, seqlen);
        // The DSpark tap reads the block INPUT of the target layers, averaged over the hc copies.
        // The reference concatenates the taps PER POSITION (model.py:1264-1266), so main_hidden holds
        // `taps * dim` values for each token with the taps in tap order - not one tap's worth of
        // tokens after another's.
        let tap_layers: Vec<usize> = (0..config.n_layers)
            .filter(|layer| config.dspark_target_layer_ids.contains(layer))
            .collect();
        let taps = tap_layers.len();
        let mut main_hidden = vec![0.0f32; seqlen * taps * dim];
        for layer in 0..config.n_layers {
            if config.engram_rows(layer).is_some() {
                engram::engram_forward(config, layer, &mut hidden, &hashes, source, self.limit)?;
            }
            if let Some(tap) = tap_layers.iter().position(|&target| target == layer) {
                let mut averaged = vec![0.0f32; dim];
                for position in 0..seqlen {
                    let base = position * width;
                    for (column, value) in averaged.iter_mut().enumerate() {
                        let mut sum = 0.0f32;
                        for copy in 0..hc {
                            sum += hidden[base + copy * dim + column];
                        }
                        *value = sum / hc as f32;
                    }
                    write_tap(&mut main_hidden, position, tap, taps, dim, &averaged);
                }
            }
            let mut row = capture.as_deref_mut().map(|_| LayerBoundary::new(layer));
            self.block(
                state,
                layer,
                &mut hidden,
                &mut pre_mix,
                &mut scratch,
                start_pos,
                None,
                None,
                row.as_mut(),
                None,
                None,
                None,
            )?;
            if let (Some(table), Some(row)) = (capture.as_deref_mut(), row) {
                table.push_row(row)?;
            }
        }

        // Collapse the copies with the LAST block's FFN coefficient, then the final norm and the LM
        // head (model.py:1268-1269).
        let mut collapsed = vec![0.0f32; seqlen * dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut collapsed)?;
        let norm = linalg::decode_by_name(source, "norm.weight", self.limit)?;
        let mut tail = collapsed[(seqlen - 1) * dim..].to_vec();
        linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        let head_input = tail.clone();
        let head = linalg::decode_by_name(source, "head.weight", self.limit)?;
        let mut logits = vec![0.0f32; config.vocab_size];
        linalg::matmul(&mut logits, &tail, &head, config.vocab_size, dim, 1)?;

        let next_token = sample_gumbel(&logits, self.sampling.temperature, self.sampling.seed);

        // The DSpark stages draft after this token when the tap fired (model.py:1274-1282).
        // The reference drafts nothing for the chunk at position 0: `forward_spec` returns None there
        // (model.py:1123-1125, 1274-1282), so a prefill that starts a sequence has no draft, while every
        // later step does. Refusing the step would be wrong in the other direction - the draft is
        // optional, the logits are not.
        let (draft, confidence) = if main_hidden.is_empty() || start_pos == 0 {
            (Vec::new(), Vec::new())
        } else {
            // The MTP stages own their own window rings, held after the backbone layers.
            let anchor = next_token.unwrap_or(0);
            // The anchor is the MTP block's first draft token (model.py:1131-1132), so it is part
            // of the step's header whenever an MTP row is recorded, not only of the output.
            if let Some(table) = capture.as_deref_mut() {
                table.set_mtp_anchor(anchor);
            }
            let stages = &mut state.attention[config.n_layers..];
            let output = mtp::dspark_forward_observed(
                config,
                tokens,
                start_pos,
                &main_hidden,
                anchor,
                stages,
                source,
                self.limit,
                capture,
            )?;
            (output.draft, output.confidence)
        };

        state.tokens.extend_from_slice(tokens);
        Ok(StepOutput {
            logits,
            head_input,
            next_token,
            draft,
            confidence,
            consumed: seqlen,
        })
    }

    /// One block, in the order `Block.forward` runs it (model.py:968-994, section 3.2).
    ///
    /// The shift is the point of single-pass mHC: the attention sublayer mixes the copies with the
    /// previous sublayer's coefficient and leaves this one's behind, which the FFN sublayer then uses
    /// (model.py:972-994).
    #[allow(clippy::too_many_arguments)] // the reference's own block signature plus the optional hybrid entry and stage recorder; bundling the pass state would hide which buffer a call got wrong.
    fn block(
        &self,
        state: &mut SequenceState,
        layer: usize,
        hidden: &mut [f32],
        pre_mix: &mut [f32],
        scratch: &mut Scratch,
        start_pos: usize,
        entry: Option<&LayerZeroEntry>,
        mut stages: Option<&mut Vec<StageEvent>>,
        mut boundary: Option<&mut LayerBoundary>,
        mut post: Option<&mut HybridPostProvider<'_>>,
        attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        deferred: Option<&mut Option<DeferredHandoff>>,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let prefix = format!("layers.{layer}");
        let source: &dyn WeightSource = self.source.as_ref();

        // The entry boundaries: the residual stream this block received and the input mix it will
        // consume (docs/build/phase1-boundary-table.md section 2).
        if let Some(row) = boundary.as_deref_mut() {
            row.push_stream(config, boundary::RESIDUAL_IN, hidden.to_vec())?;
            row.push_mixes(config, boundary::HC_IN_PRE, pre_mix.to_vec())?;
        }
        record_stage(
            &mut stages,
            "dispatch",
            0,
            0,
            "host dispatch of the block; no arithmetic",
        );

        // The attention sublayer (model.py:972-980). In a hybrid chain the device MhcEntry has
        // already supplied the normalised attention block input and the 24 attention coefficients;
        // the host must not recompute either. Otherwise this is the reference entry exactly.
        let attn_mixes = if let Some(entry) = entry {
            let mixes = Mixes {
                pre: entry.pre.clone(),
                post: entry.post.clone(),
                comb: entry.comb.clone(),
            };
            if let Some(row) = boundary.as_deref_mut() {
                row.push_mixes(config, boundary::HC_ATTN_PRE, mixes.pre.clone())?;
                row.push_mixes(config, boundary::HC_ATTN_POST, mixes.post.clone())?;
                row.push_comb(config, boundary::HC_ATTN_COMB, mixes.comb.clone())?;
            }
            record_stage(
                &mut stages,
                "mhc_entry",
                u64_bytes(hidden.len() + mixes.pre.len() + mixes.post.len() + mixes.comb.len()),
                u64_bytes(entry.pre_attention_hidden.len()),
                match entry.source {
                    LayerZeroEntrySource::Device => {
                        "device MhcEntry handoff: identity step-1 X write (B=I, C=0, Y=0), device H(X_l), capture, codes/scales and 1/rms; host does not recompute the entry"
                    }
                    LayerZeroEntrySource::Host => {
                        "caller-supplied host MhcEntry handoff; host does not recompute the entry"
                    }
                },
            );
            record_stage(
                &mut stages,
                "attn_norm",
                u64_bytes(entry.pre_attention_hidden.len()),
                0,
                "attn_norm is part of the device MhcEntry capture; no separate host normalisation",
            );
            scratch.attn_in.copy_from_slice(&entry.pre_attention_hidden);
            mixes
        } else {
            let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_fn"), self.limit)?;
            let scale =
                linalg::decode_by_name(source, &format!("{prefix}.hc_attn_scale"), self.limit)?;
            let base =
                linalg::decode_by_name(source, &format!("{prefix}.hc_attn_base"), self.limit)?;
            mhc::hc_mixes(
                config,
                hidden,
                &fn_w,
                &scale,
                &base,
                &mut scratch.attn_mixes,
            )?;
            if let Some(row) = boundary.as_deref_mut() {
                row.push_mixes(
                    config,
                    boundary::HC_ATTN_PRE,
                    scratch.attn_mixes.pre.clone(),
                )?;
                row.push_mixes(
                    config,
                    boundary::HC_ATTN_POST,
                    scratch.attn_mixes.post.clone(),
                )?;
                row.push_comb(
                    config,
                    boundary::HC_ATTN_COMB,
                    scratch.attn_mixes.comb.clone(),
                )?;
            }
            mhc::hc_pre(config, pre_mix, hidden, &mut scratch.attn_in)?;
            let attn_norm =
                linalg::decode_by_name(source, &format!("{prefix}.attn_norm.weight"), self.limit)?;
            linalg::rms_norm(
                &mut scratch.attn_in,
                &attn_norm,
                config.dim,
                config.norm_eps as f32,
            )?;
            record_stage(
                &mut stages,
                "mhc_entry",
                u64_bytes(hidden.len()),
                u64_bytes(
                    scratch.attn_in.len()
                        + scratch.attn_mixes.pre.len()
                        + scratch.attn_mixes.post.len()
                        + scratch.attn_mixes.comb.len(),
                ),
                "hc_mixes + hc_pre over the hc residual copies",
            );
            record_stage(
                &mut stages,
                "attn_norm",
                u64_bytes(scratch.attn_in.len() * 2),
                u64_bytes(scratch.attn_in.len()),
                "RMSNorm of the attention block input with the staged attn_norm weight",
            );
            scratch.attn_mixes.clone()
        };
        if let Some(row) = boundary.as_deref_mut() {
            row.push_block_input(
                config,
                boundary::PRE_ATTENTION_HIDDEN,
                boundary::PRE_ATTENTION_XHAT,
                boundary::PRE_ATTENTION_SCALES,
                &scratch.attn_in,
            )?;
        }
        // The attention writes one concatenated head vector per query before the output
        // projection narrows it to dim, so its buffer is the wider of the two widths. The
        // released geometry is wider (64 heads of 512 = 32,768 values against dim = 5,120); a
        // buffer sized only positions * dim panics there on the first layer, which is why the
        // width is derived from the geometry rather than from the residual stream.
        let attention_width =
            config
                .n_heads
                .checked_mul(config.head_dim)
                .ok_or_else(|| EngineError::Shape {
                    what: "attention scratch".to_owned(),
                    detail: format!(
                        "{} heads of {} do not fit a buffer width",
                        config.n_heads, config.head_dim
                    ),
                })?;
        let positions = hidden.len() / config.hc_width();
        let wide = positions.saturating_mul(attention_width);
        if scratch.attn_out.len() < wide {
            scratch.attn_out.resize(wide, 0.0);
        }
        scratch.attn_mixes = attn_mixes;
        record_stage(
            &mut stages,
            "attention_projections",
            u64_bytes(scratch.attn_in.len()),
            u64_bytes(wide),
            "q/kv projections over the normalised attention input",
        );
        record_stage(
            &mut stages,
            "attn_rope",
            u64_bytes(wide),
            u64_bytes(wide),
            "RoPE on the projected q/kv rows",
        );
        record_stage(
            &mut stages,
            "window_kv",
            u64_bytes(wide),
            u64_bytes(wide),
            "write and read the block's sliding-window KV ring",
        );
        record_stage(
            &mut stages,
            "attention",
            u64_bytes(scratch.attn_in.len()),
            u64_bytes(wide),
            "reference attention arithmetic over the window KV",
        );
        match attention {
            Some(backend) => attention::attention_forward_with_backend(
                config,
                layer,
                &prefix,
                &scratch.attn_in,
                start_pos,
                &mut state.attention[layer],
                source,
                &mut scratch.attn_out,
                self.limit,
                backend,
            )?,
            None => attention::attention_forward(
                config,
                layer,
                &prefix,
                &scratch.attn_in,
                start_pos,
                &mut state.attention[layer],
                source,
                &mut scratch.attn_out,
                self.limit,
            )?,
        }
        // The projection left exactly one residual row per position; the head buffer beyond that is
        // not a dense row and must not be recorded or mixed as one.
        scratch.attn_out.truncate(positions * config.dim);
        if let Some(row) = boundary.as_deref_mut() {
            row.push_dense(config, boundary::ATTENTION_OUT, scratch.attn_out.clone())?;
        }
        record_stage(
            &mut stages,
            "attention_output_projection",
            u64_bytes(wide),
            u64_bytes(scratch.attn_out.len()),
            "wo_a + wo_b projection of the concatenated head rows",
        );
        if let Some(post_hook) = post.as_deref_mut() {
            let post = post_hook(layer, hidden, &scratch.attn_out, &scratch.attn_mixes)?;
            if post.x.len() != hidden.len() || post.ffn_in.len() != config.dim {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} device post"),
                    detail: format!(
                        "post X is {} values (expected {}) and ffn_in is {} values (expected {})",
                        post.x.len(),
                        hidden.len(),
                        post.ffn_in.len(),
                        config.dim
                    ),
                });
            }
            hidden.copy_from_slice(&post.x);
            scratch.ffn_mixes = post.ffn_mixes;
            scratch.ffn_in.copy_from_slice(&post.ffn_in);
            if let Some(row) = boundary.as_deref_mut() {
                row.push_stream(config, boundary::POST_ATTENTION_HIDDEN, hidden.to_vec())?;
            }
            record_stage(
                &mut stages,
                "mhc_post",
                u64_bytes(
                    hidden.len()
                        + scratch.attn_out.len()
                        + scratch.attn_mixes.pre.len()
                        + scratch.attn_mixes.post.len()
                        + scratch.attn_mixes.comb.len(),
                ),
                u64_bytes(hidden.len() + scratch.ffn_in.len()),
                match post.source {
                    LayerZeroEntrySource::Device => {
                        "device MhcPost: attention hc_post write and the 24 FFN coefficients plus the normalized FFN input capture; the host does not recompute the merge"
                    }
                    LayerZeroEntrySource::Host => {
                        "caller-supplied host MhcPost handoff; host does not recompute the merge"
                    }
                },
            );
        } else {
            scratch.residual.copy_from_slice(hidden);
            mhc::hc_post(
                config,
                &scratch.attn_out,
                &scratch.residual,
                &scratch.attn_mixes.post,
                &scratch.attn_mixes.comb,
                hidden,
            )?;
            if let Some(row) = boundary.as_deref_mut() {
                row.push_stream(config, boundary::POST_ATTENTION_HIDDEN, hidden.to_vec())?;
            }
            record_stage(
                &mut stages,
                "mhc_post",
                u64_bytes(
                    scratch.attn_out.len()
                        + scratch.residual.len()
                        + scratch.attn_mixes.post.len()
                        + scratch.attn_mixes.comb.len(),
                ),
                u64_bytes(hidden.len()),
                "hc_post: attention output mixed back onto the hc residual",
            );
        }

        // The FFN sublayer (model.py:982-994), whose input mix is the attention sublayer's.
        if post.is_none() {
            let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_fn"), self.limit)?;
            let scale =
                linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_scale"), self.limit)?;
            let base =
                linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_base"), self.limit)?;
            mhc::hc_mixes(config, hidden, &fn_w, &scale, &base, &mut scratch.ffn_mixes)?;
            if let Some(row) = boundary.as_deref_mut() {
                row.push_mixes(config, boundary::HC_FFN_PRE, scratch.ffn_mixes.pre.clone())?;
                row.push_mixes(
                    config,
                    boundary::HC_FFN_POST,
                    scratch.ffn_mixes.post.clone(),
                )?;
                row.push_comb(
                    config,
                    boundary::HC_FFN_COMB,
                    scratch.ffn_mixes.comb.clone(),
                )?;
            }
            mhc::hc_pre(config, &scratch.attn_mixes.pre, hidden, &mut scratch.ffn_in)?;
            let ffn_norm =
                linalg::decode_by_name(source, &format!("{prefix}.ffn_norm.weight"), self.limit)?;
            linalg::rms_norm(
                &mut scratch.ffn_in,
                &ffn_norm,
                config.dim,
                config.norm_eps as f32,
            )?;
            if let Some(row) = boundary.as_deref_mut() {
                row.push_block_input(
                    config,
                    boundary::PRE_FFN_HIDDEN,
                    boundary::PRE_FFN_XHAT,
                    boundary::PRE_FFN_SCALES,
                    &scratch.ffn_in,
                )?;
            }
        } else if let Some(row) = boundary.as_deref_mut() {
            row.push_mixes(config, boundary::HC_FFN_PRE, scratch.ffn_mixes.pre.clone())?;
            row.push_mixes(
                config,
                boundary::HC_FFN_POST,
                scratch.ffn_mixes.post.clone(),
            )?;
            row.push_comb(
                config,
                boundary::HC_FFN_COMB,
                scratch.ffn_mixes.comb.clone(),
            )?;
            row.push_block_input(
                config,
                boundary::PRE_FFN_HIDDEN,
                boundary::PRE_FFN_XHAT,
                boundary::PRE_FFN_SCALES,
                &scratch.ffn_in,
            )?;
        }
        let topk = config.n_activated_experts;
        let shared_inter = config
            .moe_inter_dim
            .saturating_mul(config.n_shared_experts.max(1));
        record_stage(
            &mut stages,
            "moe_router",
            u64_bytes(scratch.ffn_in.len()),
            u64_bytes(topk.saturating_mul(8)),
            "route the normalised FFN input through the reference router",
        );
        record_stage(
            &mut stages,
            "moe_routing_weight",
            u64_bytes(topk.saturating_mul(8)),
            u64_bytes(topk.saturating_mul(12)),
            "apply the selected experts' routing weights",
        );
        record_stage(
            &mut stages,
            "moe_gate_up",
            u64_bytes(scratch.ffn_in.len()),
            u64_bytes(config.moe_inter_dim.saturating_mul(2)),
            "routed gate/up branch for the selected experts",
        );
        record_stage(
            &mut stages,
            "moe_swiglu_quant",
            u64_bytes(config.moe_inter_dim.saturating_mul(2)),
            u64_bytes(config.moe_inter_dim),
            "SwiGLU and activation quantisation of the routed branch",
        );
        record_stage(
            &mut stages,
            "moe_down_combine",
            u64_bytes(config.moe_inter_dim.saturating_mul(topk)),
            u64_bytes(config.dim),
            "routed down projection and combine",
        );
        record_stage(
            &mut stages,
            "shared_expert_gate",
            u64_bytes(scratch.ffn_in.len()),
            u64_bytes(shared_inter.saturating_mul(2)),
            "dense shared-expert gate/up GEMM",
        );
        record_stage(
            &mut stages,
            "shared_expert_swiglu_quant",
            u64_bytes(shared_inter.saturating_mul(2)),
            u64_bytes(shared_inter),
            "shared-expert SwiGLU and activation quantisation",
        );
        moe::moe_forward(
            config,
            &prefix,
            &scratch.ffn_in,
            source,
            self.limit,
            &mut scratch.moe,
            &mut scratch.ffn_out,
        )?;
        if let Some(row) = boundary.as_deref_mut() {
            row.push_dense(config, boundary::FFN_OUT, scratch.ffn_out.clone())?;
        }
        if let Some(slot) = deferred {
            *slot = Some(DeferredHandoff {
                x: hidden.to_vec(),
                y: scratch.ffn_out.clone(),
                mixes: scratch.ffn_mixes.clone(),
            });
            record_stage(
                &mut stages,
                "ffn_epilogue",
                0,
                0,
                "deferred into the next device MhcEntry step-1 (B/C/Y from this layer's device post); the host does not execute the FFN epilogue add here",
            );
        } else {
            scratch.residual.copy_from_slice(hidden);
            mhc::hc_post(
                config,
                &scratch.ffn_out,
                &scratch.residual,
                &scratch.ffn_mixes.post,
                &scratch.ffn_mixes.comb,
                hidden,
            )?;
            if let Some(row) = boundary {
                row.push_stream(config, boundary::RESIDUAL_OUT, hidden.to_vec())?;
            }
            record_stage(
                &mut stages,
                "ffn_epilogue",
                u64_bytes(
                    scratch.ffn_out.len()
                        + scratch.residual.len()
                        + scratch.ffn_mixes.post.len()
                        + scratch.ffn_mixes.comb.len(),
                ),
                u64_bytes(hidden.len()),
                "hc_post of the FFN output and the FFN epilogue add",
            );
        }

        // The next block's input mix is THIS block's FFN coefficient (model.py:994).
        pre_mix.copy_from_slice(&scratch.ffn_mixes.pre);
        Ok(())
    }

    /// Runs one prompt token through the 40-layer reference block order, replacing each layer's
    /// MhcEntry with the caller's handoff. The caller is responsible for the device launch and for
    /// returning the exact residual stream its X buffer holds; the engine does not touch a driver.
    ///
    /// This is the chain-wiring path: every other stage is the reference host arithmetic. It is not
    /// a device-chain claim and it does not make the host stand-ins device-executed.
    ///
    /// # Errors
    ///
    /// [EngineError] from a weight read, a shape check, or a handoff whose X length is not the
    /// configured `hc_mult * dim`.
    #[allow(clippy::too_many_arguments)] // the chain's own inputs plus the two callbacks; bundling them would hide which one is the device seam.
    pub fn run_hybrid_chain(
        &self,
        token: u32,
        embedding: &[f32],
        provider: &mut HybridProvider<'_>,
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridTokenOutput, EngineError> {
        let config = &self.config;
        let dim = config.dim;
        let hc = config.hc_mult;
        let width = config.hc_width();
        if dim == 0 || hc == 0 || width == 0 {
            return Err(EngineError::Shape {
                what: "hybrid chain".to_owned(),
                detail: "dim and hc_mult must both be positive".to_owned(),
            });
        }
        if token as usize >= config.vocab_size {
            return Err(EngineError::Shape {
                what: "hybrid chain embedding".to_owned(),
                detail: format!(
                    "token {token} is outside the {} row table",
                    config.vocab_size
                ),
            });
        }
        if embedding.len() != dim {
            return Err(EngineError::Shape {
                what: "hybrid chain embedding".to_owned(),
                detail: format!("expected {dim} values, got {}", embedding.len()),
            });
        }
        let mut hashes = Vec::new();
        engram::ngram_hashes(config, &[token], 0, &mut hashes)?;
        let mut hidden = vec![0.0f32; width];
        for copy in 0..hc {
            hidden[copy * dim..(copy + 1) * dim].copy_from_slice(embedding);
        }
        let mut pre_mix = vec![0.0f32; hc];
        pre_mix[0] = 1.0;
        let mut state = SequenceState::new(config);
        let mut scratch = Scratch::for_chunk(config, 1);
        let mut layers = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let mut layer_stages = Vec::new();
            if config.engram_rows(layer).is_some() {
                let wrote = engram::engram_forward(
                    config,
                    layer,
                    &mut hidden,
                    &hashes,
                    self.source.as_ref(),
                    self.limit,
                )?;
                if wrote {
                    layer_stages.push(StageEvent {
                        stage: "engram",
                        host_bytes_in: u64_bytes(width),
                        host_bytes_out: u64_bytes(width),
                        note: "Engram read/write at its layers (host seam)",
                    });
                }
            }
            let handoff = provider(layer, &hidden, &pre_mix)?;
            if handoff.x.len() != width {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} MhcEntry X"),
                    detail: format!("expected {width} values, got {}", handoff.x.len()),
                });
            }
            hidden.copy_from_slice(&handoff.x);
            let started = std::time::Instant::now();
            self.block(
                &mut state,
                layer,
                &mut hidden,
                &mut pre_mix,
                &mut scratch,
                0,
                Some(&handoff.entry),
                Some(&mut layer_stages),
                None,
                None,
                None,
                None,
            )?;
            let host_ms = started.elapsed().as_millis() as u64;
            layers.push(HybridLayerTrace {
                layer,
                stages: layer_stages,
                residual_out: hidden.clone(),
                host_ms,
            });
            progress(layer, host_ms);
        }
        let mut collapsed = vec![0.0f32; dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut collapsed)?;
        let norm = linalg::decode_by_name(self.source.as_ref(), "norm.weight", self.limit)?;
        let mut tail = collapsed;
        linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        let head_input = tail.clone();
        let head = linalg::decode_by_name(self.source.as_ref(), "head.weight", self.limit)?;
        let mut logits = vec![0.0f32; config.vocab_size];
        linalg::matmul(&mut logits, &tail, &head, config.vocab_size, dim, 1)?;
        let token = linalg::argmax(&logits).map(|index| index as u32);
        let tail_stages = vec![
            StageEvent {
                stage: "final_norm",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(dim),
                note: "host final RMS norm; class-(b) seam, not device-executed",
            },
            StageEvent {
                stage: "lm_head_screen",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(config.vocab_size),
                note: "host head matmul; the device cert_lm_head_sm121 screen is a host stand-in in this run",
            },
            StageEvent {
                stage: "lm_head_argmax",
                host_bytes_in: u64_bytes(config.vocab_size),
                host_bytes_out: 4,
                note: "host greedy argmax; class-(b) seam, not device-executed",
            },
        ];
        Ok(HybridTokenOutput {
            layers,
            tail_stages,
            head_input,
            logits,
            token,
        })
    }

    /// Runs one prompt token through the c3b order: each layer's FFN epilogue is deferred into the
    /// next layer's device MhcEntry step 1, and the device post publishes the FFN coefficients and
    /// normalized FFN input the host FFN consumes. Layer 0 keeps the identity entry; layer 39's FFN
    /// epilogue stays host because there is no next entry to carry it.
    ///
    /// This remains a chain-wiring path with host stand-ins; it is not a device-token claim.
    ///
    /// # Errors
    /// [EngineError] from a weight read, a shape check, a provider result, or a deferred post slot
    /// that the block did not fill.
    #[allow(clippy::too_many_arguments)] // the chain's own inputs plus the two callbacks; bundling them would hide which one is the device seam.
    pub fn run_hybrid_chain_deferred(
        &self,
        token: u32,
        embedding: &[f32],
        provider: &mut HybridDeferredProvider<'_>,
        post: &mut HybridPostProvider<'_>,
        mut attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridTokenOutput, EngineError> {
        let config = &self.config;
        let dim = config.dim;
        let hc = config.hc_mult;
        let width = config.hc_width();
        if dim == 0 || hc == 0 || width == 0 {
            return Err(EngineError::Shape {
                what: "deferred hybrid chain".to_owned(),
                detail: "dim and hc_mult must both be positive".to_owned(),
            });
        }
        if token as usize >= config.vocab_size {
            return Err(EngineError::Shape {
                what: "deferred hybrid chain embedding".to_owned(),
                detail: format!(
                    "token {token} is outside the {} row table",
                    config.vocab_size
                ),
            });
        }
        if embedding.len() != dim {
            return Err(EngineError::Shape {
                what: "deferred hybrid chain embedding".to_owned(),
                detail: format!("expected {dim} values, got {}", embedding.len()),
            });
        }
        let mut hidden = vec![0.0f32; width];
        for copy in 0..hc {
            hidden[copy * dim..(copy + 1) * dim].copy_from_slice(embedding);
        }
        let mut pre_mix = vec![0.0f32; hc];
        pre_mix[0] = 1.0;
        let mut state = SequenceState::new(config);
        let mut scratch = Scratch::for_chunk(config, 1);
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut deferred: Option<DeferredHandoff> = None;
        for layer in 0..config.n_layers {
            let mut layer_stages = Vec::new();
            // The c3b provider applies Engram itself, after it materialises the deferred FFN
            // epilogue into H_l and before it computes H_l's entry. Applying Engram here would
            // run it on A_{l-1}, before that epilogue, which is the wrong operand.
            let handoff = provider(layer, &hidden, &pre_mix, deferred.as_ref())?;
            if handoff.x.len() != width {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} MhcEntry X"),
                    detail: format!("expected {width} values, got {}", handoff.x.len()),
                });
            }
            hidden.copy_from_slice(&handoff.x);
            let started = std::time::Instant::now();
            let mut next_deferred = None;
            let defer_epilogue = layer + 1 < config.n_layers;
            self.block(
                &mut state,
                layer,
                &mut hidden,
                &mut pre_mix,
                &mut scratch,
                0,
                Some(&handoff.entry),
                Some(&mut layer_stages),
                None,
                Some(&mut *post),
                attention.as_mut().map(|backend| {
                    &mut **backend as &mut dyn attention::AttentionProjectionBackend
                }),
                if defer_epilogue {
                    Some(&mut next_deferred)
                } else {
                    None
                },
            )?;
            if defer_epilogue {
                deferred = Some(next_deferred.ok_or_else(|| EngineError::Shape {
                    what: format!("layer {layer} deferred FFN epilogue"),
                    detail: "the block did not return the deferred handoff".to_owned(),
                })?);
            } else {
                deferred = None;
            }
            let host_ms = started.elapsed().as_millis() as u64;
            layers.push(HybridLayerTrace {
                layer,
                stages: layer_stages,
                residual_out: hidden.clone(),
                host_ms,
            });
            progress(layer, host_ms);
        }
        let mut collapsed = vec![0.0f32; dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut collapsed)?;
        let norm = linalg::decode_by_name(self.source.as_ref(), "norm.weight", self.limit)?;
        let mut tail = collapsed;
        linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        let head_input = tail.clone();
        let head = linalg::decode_by_name(self.source.as_ref(), "head.weight", self.limit)?;
        let mut logits = vec![0.0f32; config.vocab_size];
        linalg::matmul(&mut logits, &tail, &head, config.vocab_size, dim, 1)?;
        let token = linalg::argmax(&logits).map(|index| index as u32);
        let tail_stages = vec![
            StageEvent {
                stage: "final_norm",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(dim),
                note: "host final RMS norm; class-(b) seam, not device-executed",
            },
            StageEvent {
                stage: "lm_head_screen",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(config.vocab_size),
                note: "host head matmul; the device cert_lm_head_sm121 screen is a host stand-in in this run",
            },
            StageEvent {
                stage: "lm_head_argmax",
                host_bytes_in: u64_bytes(config.vocab_size),
                host_bytes_out: 4,
                note: "host greedy argmax; class-(b) seam, not device-executed",
            },
        ];
        Ok(HybridTokenOutput {
            layers,
            tail_stages,
            head_input,
            logits,
            token,
        })
    }
}

/// Writes one tap of one position into the DSpark tap buffer, which holds `taps * dim` values per
/// token with the taps in tap order (model.py:1264-1266).
fn write_tap(
    out: &mut [f32],
    position: usize,
    tap: usize,
    taps: usize,
    dim: usize,
    values: &[f32],
) {
    let start = (position * taps + tap) * dim;
    out[start..start + dim].copy_from_slice(values);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::tensor::TensorSpec;
    use crate::weights::WeightError;

    /// The tap buffer is position-major: all of a token's taps sit together, in tap order. The
    /// reference concatenates the taps per position (model.py:1264-1266). A layer-major layout
    /// passes every end-to-end test at this scale, because the draft is insensitive to the
    /// permutation, so the axis is pinned here rather than through the forward pass.
    #[test]
    fn the_dspark_taps_are_written_per_position_in_tap_order() {
        let mut out = vec![-1.0f32; 3 * 2 * 2];
        write_tap(&mut out, 0, 0, 2, 2, &[1.0, 2.0]);
        write_tap(&mut out, 0, 1, 2, 2, &[3.0, 4.0]);
        write_tap(&mut out, 2, 1, 2, 2, &[1.0, 2.0]);
        assert_eq!(
            out[0..4],
            [1.0, 2.0, 3.0, 4.0],
            "position 0 holds both of its taps, in tap order"
        );
        assert_eq!(out[4..8], [-1.0f32; 4], "position 1 was not written");
        assert_eq!(out[8..10], [-1.0f32; 2], "position 2 holds its first tap");
        assert_eq!(
            out[10..12],
            [1.0, 2.0],
            "position 2's second tap follows its first"
        );
    }

    /// The default sampler is greedy, and a temperature that is not above zero is greedy too rather
    /// than a division by zero.
    #[test]
    fn the_reference_sampler_is_greedy_at_zero_temperature() {
        let logits = [0.5f32, 2.0, -1.0];
        assert_eq!(sample_gumbel(&logits, 0.0, 0), Some(1));
        assert_eq!(sample_gumbel(&logits, -1.0, 7), Some(1));
        assert_eq!(sample_gumbel(&[], 1.0, 1), None);
    }

    /// A draw is a function of the logits, the temperature and the seed, and the seeds do not all
    /// collapse onto one token.
    #[test]
    fn a_seeded_draw_is_a_function_of_its_inputs() {
        let logits = [0.0f32, 0.1, 0.2, 0.3];
        let first = sample_gumbel(&logits, 1.0, 99);
        assert_eq!(first, sample_gumbel(&logits, 1.0, 99));
        let draws: Vec<u32> = (0..32)
            .filter_map(|seed| sample_gumbel(&logits, 1.0, seed))
            .collect();
        assert_eq!(draws.len(), 32);
        assert!(
            draws.iter().any(|token| Some(*token) != first),
            "32 seeds do not all agree on one token: {draws:?}"
        );
        assert!(draws.iter().all(|token| (*token as usize) < logits.len()));
    }

    /// At a temperature above zero the noise can outvote a small logit gap, which is the whole point
    /// of the temperature.
    #[test]
    fn a_temperature_above_zero_can_pick_what_greedy_would_not() {
        let logits = [1.0f32, 0.999_9];
        assert_eq!(sample_gumbel(&logits, 0.0, 0), Some(0));
        assert!(
            (0..64).any(|seed| sample_gumbel(&logits, 1.0, seed) == Some(1)),
            "a 1e-4 logit gap is well inside Gumbel noise at temperature 1"
        );
    }

    /// A plan-shaped source of finite numbers for tests that drive one block instead of the whole
    /// pass. Every tensor keeps the plan's shape but is served as `f32` with the same small positive
    /// pattern mtp.rs's finite hand uses, so nothing here is NaN or inf.
    struct Hand {
        specs: Vec<TensorSpec>,
    }

    impl Hand {
        fn for_config(config: &ModelConfig) -> Self {
            Self {
                specs: TensorPlan::language_model(config).specs().to_vec(),
            }
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.specs.iter().map(|spec| spec.name.clone()).collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.specs
                .iter()
                .find(|spec| spec.name == name)
                .map(|spec| TensorSpec {
                    dtype: DType::F32,
                    ..spec.clone()
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            let elements = usize::try_from(self.spec(name)?.elements()).unwrap_or(usize::MAX);
            let mut bytes = Vec::with_capacity(elements.saturating_mul(4));
            for index in 0..elements {
                let value = 0.02 + 0.01 * (index % 5) as f32;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        }
    }

    /// The FFN block's RMSNorm reads `config.norm_eps`, not `hc_eps` (M7), the half the window-row
    /// test in tests/forward.rs cannot reach.
    ///
    /// That test's observable is `attn_in`, the ATTENTION norm's output; by the time the FFN norm's
    /// scale reaches a stored window row, the next layer's attention norm and the hc mixing have
    /// rescaled it, so substituting hc_eps there leaves the whole suite green. The FFN norm is visible
    /// here instead: `block` normalises `scratch.ffn_in` in place, so immediately after the call that
    /// buffer IS the FFN norm's own output. Two configurations identical except norm_eps - 1.0 against
    /// tiny's 1e-20, hc_eps held equal at 1e-6 - must give ffn_in RMS values a wide factor apart, and
    /// they do because the hidden values are small enough that the epsilon dominates the denominator.
    ///
    /// Measured on this configuration, ffn_in RMS after one block of layer 0:
    ///   norm_eps 1e-20 (tiny), correct engine          4.2069e-2
    ///   norm_eps 1.0,          correct engine          8.6641e-5  -> 486x
    ///   the FFN norm reading hc_eps instead (1e-6)     1.008x     (3.8161e-2 against 3.7844e-2)
    /// The threshold is 100x, so the engine as written passes with about 5x of room and the substitution
    /// fails it by two orders of magnitude. Replacing the epsilon in the ATTENTION norm instead leaves
    /// this ratio at 486x, which is the point: this test and the window-row test in tests/forward.rs
    /// cover the two block norms between them, and neither one alone covers both.
    #[test]
    fn the_ffn_block_rms_norm_reads_the_configured_epsilon() {
        let sharp = ModelConfig {
            norm_eps: 1.0,
            ..ModelConfig::tiny()
        };
        let plain = ModelConfig::tiny();
        assert!(
            sharp.norm_eps > plain.norm_eps * 1e15,
            "the two epsilons must be far apart for this to measure anything"
        );
        assert_eq!(
            sharp.hc_eps, plain.hc_eps,
            "only norm_eps differs between the two runs"
        );

        let ffn_row = |config: &ModelConfig| -> f32 {
            let engine =
                InferenceEngine::new(config.clone(), Box::new(Hand::for_config(config)), 1 << 22)
                    .expect("a consistent configuration");
            let mut state = SequenceState::new(config);
            let mut scratch = Scratch::for_chunk(config, 1);
            // Nonzero and small: mean(x^2) near 1e-6 leaves the epsilon in charge of the denominator.
            let mut hidden: Vec<f32> = (0..config.hc_width())
                .map(|index| 1e-3 * (1 + (index % 3)) as f32)
                .collect();
            let mut pre_mix = vec![0.0f32; scratch.attn_mixes.pre.len()];
            engine
                .block(
                    &mut state,
                    0,
                    &mut hidden,
                    &mut pre_mix,
                    &mut scratch,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("one block over a finite source");
            let mean_square: f32 = scratch
                .ffn_in
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / scratch.ffn_in.len() as f32;
            mean_square.sqrt()
        };

        let small = ffn_row(&sharp);
        let large = ffn_row(&plain);
        assert!(
            large > small * 100.0,
            "norm_eps 1.0 must shrink the FFN norm's output far below the 1e-20 run, got {small} against {large}"
        );
    }
}
