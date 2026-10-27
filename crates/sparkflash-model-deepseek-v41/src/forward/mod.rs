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
//! (`docs/archive/build/phase1-boundary-table.md`, `docs/archive/build/roadmap-to-serving.md` section 4.5). The two
//! entry points delegate to the same `run` with a table instead of `None`, so the default path is
//! the same code and keeps its determinism; [boundary] documents what is recorded and in what order.

pub mod attention;
pub mod attention_case;
pub mod batch;
pub mod boundary;
pub mod engram;
pub mod layer_zero;
pub mod mhc;
pub mod moe;
pub mod mtp;
mod weight_view;

use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::forward::boundary::{BoundaryTable, LayerBoundary};
use crate::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource, StageEvent};
use crate::forward::mhc::Mixes;
use crate::forward::weight_view::WeightView;
use crate::linalg::{self, EngineError};
use crate::tensor::TensorPlan;
use crate::weights::WeightSource;

/// Per-sequence state: the tokens consumed and every cache the attention layers own.
///
/// Each backbone layer and each MTP stage has its own window ring and compressed KV, so the caches
/// are indexed by layer. Nothing here is shared between sequences.
#[derive(Clone)]
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

    /// An independent copy of this state whose caches hold the same consumed prefix.
    ///
    /// This is the reference-level prefix-cache primitive: a caller that reaches a cached
    /// prefix through the KV layer can fork the state that already computed it and prefill only
    /// the uncached suffix. The copy owns its caches, so either side can continue without
    /// touching the other's KV; the arithmetic is the whole-prompt arithmetic because the
    /// windows, compressed rows and index keys are the ones the prefix forward wrote. The
    /// device pool's hash-block reuse is the paged cache's job; this is its host-reference
    /// equivalent and is tested against one whole-prompt forward in tests/batched_prefill.rs.
    #[must_use]
    pub fn fork(&self) -> Self {
        self.clone()
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

/// The generation-changing sampler options a request can set.
///
/// Kept apart from [Sampling] so that the temperature/seed call sites and their tests stay exactly
/// as they were: the default value below is the disabled configuration, and a step whose options
/// are default calls [`sample_gumbel`] verbatim.
///
/// vLLM applies the penalties to the raw logits before temperature scaling
/// (`sampling_params.py`: `repetition_penalty` over the prompt and the generated text so far,
/// lines 248-253; `presence_penalty`/`frequency_penalty` over the generated text so far, lines
/// 240-246), then temperature, then the `top_k`/`top_p` candidate filters (lines 256-262). The
/// reference's Gumbel-max draw is distributionally the same as a draw from the filtered softmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerOptions {
    /// Nucleus threshold over the temperature-scaled probabilities. 1.0 disables it. 0.0 keeps
    /// only the most likely token: our documented extension, because vLLM refuses it (its range
    /// is (0, 1], `sampling_params.py` lines 583-585).
    pub top_p: f32,
    /// Keep this many most likely tokens. 0 and -1 disable it; vLLM refuses anything below
    /// -1 (`sampling_params.py` lines 583-599: "top_k must be 0 (disable), or at least 1").
    pub top_k: i32,
    /// Divide the logits of already-seen tokens above zero and multiply those below it by this
    /// value (`sampling_params.py` lines 248-253). 1.0 disables it; values above 1 discourage
    /// repetition and values below 1 encourage it.
    pub repetition_penalty: f32,
    /// Subtracted once from the logit of every token that appears in the generated text
    /// (`sampling_params.py` lines 240-243). 0.0 disables it; positive values discourage tokens
    /// already generated.
    pub presence_penalty: f32,
    /// Subtracted from the logit of every generated token once per occurrence
    /// (`sampling_params.py` lines 244-246). 0.0 disables it.
    pub frequency_penalty: f32,
    /// How many of the consumed tokens are the prompt. Everything at or after this index is
    /// generated text, which is the history the presence/frequency penalties count; the
    /// repetition penalty sees the prompt too.
    pub prompt_len: usize,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            prompt_len: 0,
        }
    }
}

impl SamplerOptions {
    /// True when every option that changes generation is disabled, so [`sample`] can take the
    /// untouched [`sample_gumbel`] path.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.top_p == 1.0
            && self.top_k <= 0
            && self.repetition_penalty == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
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

/// One prefill chunk's output together with the LM-head rows of selected positions.
///
/// This is the one-pass prompt-logprobs seam: every selected position's next-token distribution is
/// read from the same forward that consumes the chunk, instead of re-forwarding the prefix once per
/// scored position. The rows are the same completed function of the position - the collapsed mHC
/// residual after the final RMS norm and the LM head - so selecting a position changes what is
/// returned, never the arithmetic that produced it.
#[derive(Debug)]
pub struct PositionLogits {
    /// What [`InferenceEngine::prefill`] returns for the same chunk: the last position's logits,
    /// head input, sampled token and draft.
    pub step: StepOutput,
    /// One logits row per requested position, in request order, each `vocab_size` long.
    pub rows: Vec<Vec<f32>>,
}

/// What one observed step produced: the step's output and the boundary table recorded for it.
#[derive(Debug)]
pub struct ObservedStep {
    /// What the step returned: logits, the sampled token and the draft, exactly as [StepOutput].
    pub output: StepOutput,
    /// The per-layer boundary table, one row per layer that ran, in run order.
    pub boundary: BoundaryTable,
}

/// One rank's LM-head logits without the gather, for the TP=4 prompt-logprob contract.
#[derive(Debug, Clone)]
pub struct RankHeadLogits {
    /// The rank-local rows, `rows * local_vocab` values in row-major order.
    pub logits: Vec<f32>,
    /// The rank's shard width in vocabulary entries.
    pub local_vocab: usize,
    /// The rank the gather seam reported.
    pub rank: usize,
    /// The world size the gather seam reported.
    pub world: usize,
    /// The first global token id in [Self::logits] (0 for a replicated head).
    pub offset: usize,
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
    /// The generation-changing sampler options: penalties and candidate filters. The default is
    /// the disabled configuration, so the default path calls [`sample_gumbel`] verbatim.
    sampler: SamplerOptions,
    /// The TP seam state when this engine runs one rank of a sharded model.
    tp: Option<crate::parallel_tp::TpRuntime>,
    /// The exact f32 gather for row-sharded attention/head weights (batch 18). `None` means the
    /// source serves whole tensors and every projection runs as it always has.
    gather: Option<crate::parallel_tp::BitGatherHandle>,
}

/// The layer window one sequence step runs. The end is inclusive; the default is the whole model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepLayers {
    /// First layer in the window (inclusive).
    pub start: usize,
    /// Last layer in the window (inclusive).
    pub end: usize,
}

impl StepLayers {
    /// A window covering every layer of `n_layers`.
    #[must_use]
    pub fn all(n_layers: usize) -> Self {
        Self {
            start: 0,
            end: n_layers.saturating_sub(1),
        }
    }

    /// A window from layer 0 through `stop_after_layer`, clamped to the model.
    #[must_use]
    pub fn through(stop_after_layer: usize, n_layers: usize) -> Self {
        Self {
            start: 0,
            end: stop_after_layer.min(n_layers.saturating_sub(1)),
        }
    }
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
            sampler: SamplerOptions::default(),
            tp: None,
            gather: None,
        })
    }

    /// An engine that runs ONE RANK of the TP reference forward over its local shards.
    ///
    /// The source answers global names with this rank's local specs: routed-expert
    /// `w1`/`w3` have `moe_inter_dim/world` rows and routed `w2` has
    /// `moe_inter_dim/world` input columns; every other tensor is replicated and keeps its global
    /// shape. The engine reads each of the plan's NON-ENGRAM tensors once, at construction, into a
    /// [crate::parallel_tp::ResidentSource], so a per-token forward issues no whole-tensor read
    /// against the caller's source. The two Engram table tensors per Engram layer are left in the
    /// caller's source and paged per token through [WeightSource::read_ranges]; a source that cannot
    /// page them falls back to a whole-tensor read under the element limit, which refuses a
    /// released-size table.
    ///
    /// The one seam is the routed FFN output: after the local expert bodies the engine reduces it
    /// once per layer through `collective`, then adds the replicated shared expert host-f32. With
    /// `tp.world_size == 1` the adapter is a strict no-op and this engine is bit-identical to
    /// [InferenceEngine::new].
    ///
    /// # Errors
    ///
    /// [EngineError::Config] when the configuration is not usable, [EngineError::Shape] when the
    /// layout does not split the routed intermediate or disagrees with the adapter, and
    /// [crate::weights::WeightError] when a planned tensor cannot be read.
    pub fn new_tp(
        config: ModelConfig,
        source: Box<dyn WeightSource>,
        limit: usize,
        tp: crate::parallel_tp::TpLayout,
        collective: Box<dyn crate::parallel_tp::TpCollective>,
    ) -> Result<Self, EngineError> {
        config.validate()?;
        let plan = TensorPlan::language_model(&config);
        let runtime = crate::parallel_tp::TpRuntime::new(tp, config.moe_inter_dim, collective)?;
        let resident = crate::parallel_tp::ResidentSource::load(source, &plan)?;
        Ok(Self {
            config,
            plan,
            source: Box::new(resident),
            limit,
            sampling: Sampling::default(),
            sampler: SamplerOptions::default(),
            tp: Some(runtime),
            gather: None,
        })
    }

    /// The TP seam state, when this engine runs one rank of a sharded model.
    #[must_use]
    pub fn tp(&self) -> Option<&crate::parallel_tp::TpRuntime> {
        self.tp.as_ref()
    }

    /// The exact f32 gather seam for a source that serves row-sharded attention/head weights.
    ///
    /// The source must answer `head.weight` (stages 1-2: `wq_b` and `wo_a` too; stage 3 adds
    /// `wo_b`) with this rank's row slice, and the batch-18 shard plan must give every rank a
    /// contiguous row block in rank order. The engine multiplies its local rows and gathers the
    /// pieces in ascending rank order with [crate::parallel_tp::BitGather], which is a byte copy:
    /// no value is rounded, so the result is bit-identical to the replicated computation. A source
    /// that serves a shard with no seam configured is refused rather than silently multiplying a
    /// partial weight.
    ///
    /// The MTP/DSpark stage (`prefill_with_taps`/`decode_with_taps` and the dspark block) reads its
    /// own attention weights whole and is not wired to this seam, so a sharded source must not be
    /// used on a path that runs those taps: the TP=4 host driver never does.
    #[must_use]
    pub fn with_bit_gather(mut self, gather: Box<dyn crate::parallel_tp::BitGather>) -> Self {
        self.gather = Some(crate::parallel_tp::BitGatherHandle::new(gather));
        self
    }

    /// The exact f32 gather seam, when this engine reads row-sharded weights.
    #[must_use]
    pub fn bit_gather(&self) -> Option<&crate::parallel_tp::BitGatherHandle> {
        self.gather.as_ref()
    }

    /// The same engine with a different sampler.
    #[must_use]
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Replaces how the next steps choose tokens without consuming the engine.
    ///
    /// [Self::with_sampling] is the builder form for a caller that can move the engine. A driver
    /// that owns the engine across requests (serve's TP rank) cannot move it out of its own field, so
    /// it applies one request's [Sampling] here before the request's first [Self::prefill]. The
    /// resident weights and the TP seam state are untouched; only the sampler changes.
    pub fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    /// The same engine with different generation-changing sampler options.
    #[must_use]
    pub fn with_sampler_options(mut self, options: SamplerOptions) -> Self {
        self.sampler = options;
        self
    }

    /// Replaces the generation-changing sampler options without consuming the engine.
    ///
    /// This is the per-request hook beside [Self::set_sampling]: a driver that owns its engine
    /// across requests applies one request's penalties and candidate filters here before the
    /// request's first [Self::prefill]. The weights and the TP seam state are untouched.
    pub fn set_sampler_options(&mut self, options: SamplerOptions) {
        self.sampler = options;
    }

    /// The sampler options in force.
    #[must_use]
    pub fn sampler_options(&self) -> SamplerOptions {
        self.sampler
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

    /// The LM-head logits for `rows` tail positions, gathering row-sharded weights exactly.
    ///
    /// With a replicated `head.weight` this is the old read and matmul, unchanged. When the source
    /// serves `vocab_size / world` rows (the batch-18 head row shard), each rank multiplies its own
    /// rows and every rank's segment is copied into the full logit vector in ascending rank order:
    /// a byte copy, so the gathered vector is bit-identical to the replicated matmul and the same
    /// sample/argmax rule then runs over the same values.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the source serves a head slice but no gather seam is configured (a
    /// partial weight is never used as if it were whole), or when the slice and the world do not
    /// multiply to `vocab_size`; any weight read error otherwise.
    fn head_logits(
        &self,
        source: &dyn WeightSource,
        tail: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let dim = self.config.dim;
        let vocab = self.config.vocab_size;
        let spec = source.spec("head.weight")?;
        let local_vocab = spec
            .shape
            .first()
            .copied()
            .filter(|rows| *rows > 0)
            .unwrap_or(vocab);
        if local_vocab == vocab {
            let head = WeightView::read_plain(source, "head.weight", vocab, dim, self.limit)?;
            let mut logits = vec![0.0f32; rows * vocab];
            head.matmul(&mut logits, tail, vocab, dim, rows)?;
            return Ok(logits);
        }
        let Some(gather) = self.gather.as_ref() else {
            return Err(EngineError::Shape {
                what: "lm head".to_owned(),
                detail: format!(
                    "the source serves {local_vocab} of {vocab} head rows but the engine has no rank-order gather seam"
                ),
            });
        };
        let world = gather.world();
        if local_vocab.checked_mul(world) != Some(vocab) {
            return Err(EngineError::Shape {
                what: "lm head".to_owned(),
                detail: format!(
                    "{local_vocab} local head rows over {world} ranks are not the {vocab} vocab"
                ),
            });
        }
        let head = WeightView::read_plain(source, "head.weight", local_vocab, dim, self.limit)?;
        let mut local = vec![0.0f32; rows * local_vocab];
        head.matmul(&mut local, tail, local_vocab, dim, rows)?;
        let mut gathered = vec![0.0f32; rows * vocab];
        gather.gather_bits(&local, &mut gathered)?;
        if rows == 1 {
            return Ok(gathered);
        }
        // Rank-major [rank][row][local] to row-major [row][vocab].
        let mut logits = vec![0.0f32; rows * vocab];
        for rank in 0..world {
            for row in 0..rows {
                let from = (rank * rows + row) * local_vocab;
                logits[row * vocab + rank * local_vocab..][..local_vocab]
                    .copy_from_slice(&gathered[from..from + local_vocab]);
            }
        }
        Ok(logits)
    }

    /// The rank-local LM-head logits for `rows` tail positions, without the exact gather.
    ///
    /// This is the same matmul [Self::head_logits] performs before it gathers: when the source
    /// serves a row shard, the result is this rank's segment of the vocabulary only, which is what
    /// the TP=4 prompt-logprob contract screens and exchanges. A replicated head returns the whole
    /// vector and rank 0, and every rank computes the same values.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] for a shard without a matching gather seam or a mismatched tail shape;
    /// any weight read error otherwise.
    pub fn head_rank_logits(
        &self,
        tail: &[f32],
        rows: usize,
    ) -> Result<RankHeadLogits, EngineError> {
        let dim = self.config.dim;
        let vocab = self.config.vocab_size;
        if dim == 0 || tail.len() != rows.saturating_mul(dim) {
            return Err(EngineError::Shape {
                what: "rank lm head".to_owned(),
                detail: format!("{} tail values are not {rows} rows of {dim}", tail.len()),
            });
        }
        let spec = self.source.spec("head.weight")?;
        let local_vocab = spec
            .shape
            .first()
            .copied()
            .filter(|rows| *rows > 0)
            .unwrap_or(vocab);
        if local_vocab == vocab {
            let head = WeightView::read_plain(
                self.source.as_ref(),
                "head.weight",
                vocab,
                dim,
                self.limit,
            )?;
            let mut logits = vec![0.0f32; rows * vocab];
            head.matmul(&mut logits, tail, vocab, dim, rows)?;
            let world = self.gather.as_ref().map_or(1, |gather| gather.world());
            return Ok(RankHeadLogits {
                logits,
                local_vocab,
                rank: 0,
                world,
                offset: 0,
            });
        }
        let Some(gather) = self.gather.as_ref() else {
            return Err(EngineError::Shape {
                what: "rank lm head".to_owned(),
                detail: format!(
                    "the source serves {local_vocab} of {vocab} head rows but the engine has no rank-order gather seam"
                ),
            });
        };
        let world = gather.world();
        if local_vocab.checked_mul(world) != Some(vocab) {
            return Err(EngineError::Shape {
                what: "rank lm head".to_owned(),
                detail: format!(
                    "{local_vocab} local head rows over {world} ranks are not the {vocab} vocab"
                ),
            });
        }
        let rank = gather.rank();
        let head = WeightView::read_plain(
            self.source.as_ref(),
            "head.weight",
            local_vocab,
            dim,
            self.limit,
        )?;
        let mut logits = vec![0.0f32; rows * local_vocab];
        head.matmul(&mut logits, tail, local_vocab, dim, rows)?;
        Ok(RankHeadLogits {
            logits,
            local_vocab,
            rank,
            world,
            offset: rank * local_vocab,
        })
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

    /// Consumes `tokens` as one chunk and returns [`Self::prefill`]'s output together with the
    /// LM-head logits of the requested positions (0-based, inside this chunk).
    ///
    /// This is the one-pass form of prompt logprobs: the per-layer work runs once over the chunk,
    /// and only the selected positions go through the final norm and the head. A caller scoring a
    /// prompt feeds it in contiguous spans and asks each span for the distribution rows that score
    /// the tokens after them; the sum of the spans is one pass over the prompt, where the older
    /// token-at-a-time path re-forwarded a growing prefix per position.
    ///
    /// The rows are [`StepOutput::logits`] for the selected positions, computed by the same
    /// collapse, final norm and head; the existing bit-for-bit chunked-prefill invariant means the
    /// span boundaries do not change a value (see `tests/chunked_prefill.rs`).
    ///
    /// # Errors
    ///
    /// [`EngineError::Shape`] when the chunk is empty or a position is outside it, and any weight
    /// read error [`Self::prefill`] returns.
    pub fn prefill_with_position_logits(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<PositionLogits, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::Shape {
                what: "prefill".to_owned(),
                detail: "no tokens to prefill".to_owned(),
            });
        }
        if let Some(&position) = positions.iter().find(|&&position| position >= tokens.len()) {
            return Err(EngineError::Shape {
                what: "prompt logprobs".to_owned(),
                detail: format!(
                    "position {position} is outside the {} token prefill chunk; a position's logits require the token at that position to be in the same pass",
                    tokens.len()
                ),
            });
        }
        let (step, rows) = self.run_selected(state, tokens, None, Some(positions))?;
        Ok(PositionLogits { step, rows })
    }

    /// [Self::prefill] with the per-layer boundary table recorded
    /// (`docs/archive/build/phase1-boundary-table.md`).
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
    /// they run (`docs/archive/build/phase1-boundary-table.md`).
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
        let (output, _) = self.run_selected(state, tokens, Some(&mut boundary), None)?;
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

/// The chunked c3b provider: the deferred provider plus the token's index inside the chunk, so a
/// caller can apply that token's own Engram hashes and deferred epilogue.
pub type HybridChunkDeferredProvider<'a> = dyn FnMut(
        usize,
        usize,
        &[f32],
        &[f32],
        Option<&DeferredHandoff>,
    ) -> Result<HybridEntry, EngineError>
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
    /// The normalised attention block input the layer consumed, dim values. Diagnostic only: it is
    /// the device-vs-host comparison point before the five projection GEMMs.
    pub pre_attention_hidden: Vec<f32>,
    /// The residual stream the provider handed this layer, hc_mult * dim values. For layer l > 0 in
    /// the deferred chain this is the previous layer's epilogue-applied residual.
    pub entry_input_x: Vec<f32>,
    /// The post-attention residual X the device left, hc_mult * dim values; empty when the layer is
    /// the last one and the host epilogue ran in place.
    pub post_attention_x: Vec<f32>,
    /// The host FFN output Y for this layer, dim values; empty when the layer is the last one.
    pub ffn_out: Vec<f32>,
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
    /// One certified byte per vocabulary entry, when the backend ran a certified screen.
    pub certified_mask: Option<Vec<u8>>,
}

/// One chunked hybrid step's output: the tail of the last position, plus the token it selected.
///
/// [InferenceEngine::run_hybrid_sequence_chunk] runs the same hybrid order as
/// [InferenceEngine::run_hybrid_sequence_step] but over a whole chunk. The per-layer trace is not
/// returned: a chunk's trace would repeat the chunk-sized stage table once per layer for no
/// consumer, and the runner's progress callback already reports the per-layer host time.
#[derive(Debug, Clone)]
pub struct HybridChunkOutput {
    /// The collapsed and final-normed vector the head multiplied (last position only).
    pub head_input: Vec<f32>,
    /// The raw logits (last position only), `vocab_size` values.
    pub logits: Vec<f32>,
    /// The greedy token the host argmax selected for the next position.
    pub token: Option<u32>,
    /// One certified byte per vocabulary entry, when the backend ran a certified screen.
    pub certified_mask: Option<Vec<u8>>,
    /// One final-normed head input per requested position, in request order (empty when the caller
    /// asked for no rows). Each row is `dim` values: the collapsed residual after the final RMS
    /// norm, the same vector [Self::head_input] carries for the last position. It is the input a
    /// rank-local LM-head shard multiplies, which is what the TP=4 prompt-logprob contract needs.
    pub position_head_inputs: Vec<Vec<f32>>,
}

fn u64_bytes(values: usize) -> u64 {
    u64::try_from(values).unwrap_or(u64::MAX).saturating_mul(4)
}

/// One embedding row per token, widened to f32.
///
/// The release path reads the BF16 table as stored and widens only the rows the chunk uses (the
/// widening is exact), so a decode step no longer decodes the whole 1.3 GB table. An f32 source
/// (every fixture) takes the same whole-table decode as before, byte for byte.
fn embedding_rows(
    source: &dyn WeightSource,
    tokens: &[u32],
    dim: usize,
    vocab_size: usize,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let spec = source.spec("embed.weight")?;
    let elements = spec.elements();
    if elements > limit as u128 {
        return Err(EngineError::TooLarge {
            name: "embed.weight".to_owned(),
            elements: usize::try_from(elements).unwrap_or(usize::MAX),
            limit,
        });
    }
    let needed = vocab_size.saturating_mul(dim);
    if elements < needed as u128 {
        return Err(EngineError::Shape {
            what: "embed.weight".to_owned(),
            detail: format!(
                "a {}x{} table has {} values, so it cannot serve vocab_size {}",
                vocab_size, dim, elements, vocab_size
            ),
        });
    }
    let mut rows = Vec::with_capacity(tokens.len() * dim);
    match spec.dtype {
        DType::F32 => {
            let table = linalg::decode_by_name(source, "embed.weight", limit)?;
            for &token in tokens {
                let row = (token as usize).saturating_mul(dim);
                let Some(embedding) = table.get(row..row + dim) else {
                    return Err(EngineError::Shape {
                        what: "embed.weight".to_owned(),
                        detail: format!("token {token} is outside the {vocab_size} row table"),
                    });
                };
                rows.extend_from_slice(embedding);
            }
        }
        DType::BF16 => {
            let borrowed = source.read_view("embed.weight")?;
            let owned;
            let bytes: &[u8] = match borrowed {
                Some(bytes) => bytes,
                None => {
                    owned = source.read("embed.weight")?;
                    &owned
                }
            };
            let bytes_needed = needed.checked_mul(2).ok_or_else(|| EngineError::Shape {
                what: "embed.weight".to_owned(),
                detail: "the table's byte count overflows".to_owned(),
            })?;
            if bytes.len() < bytes_needed {
                return Err(EngineError::Shape {
                    what: "embed.weight".to_owned(),
                    detail: format!(
                        "a {vocab_size}x{dim} bf16 table needs {bytes_needed} bytes, got {}",
                        bytes.len()
                    ),
                });
            }
            for &token in tokens {
                let row = (token as usize).saturating_mul(dim);
                let start = row * 2;
                let Some(slice) = bytes.get(start..start + dim * 2) else {
                    return Err(EngineError::Shape {
                        what: "embed.weight".to_owned(),
                        detail: format!("token {token} is outside the {vocab_size} row table"),
                    });
                };
                for pair in slice.as_chunks::<2>().0 {
                    rows.push(crate::quant::bf16_to_f32(u16::from_le_bytes(*pair)));
                }
            }
        }
        _ => {
            let table = linalg::decode_by_name(source, "embed.weight", limit)?;
            for &token in tokens {
                let row = (token as usize).saturating_mul(dim);
                let Some(embedding) = table.get(row..row + dim) else {
                    return Err(EngineError::Shape {
                        what: "embed.weight".to_owned(),
                        detail: format!("token {token} is outside the {vocab_size} row table"),
                    });
                };
                rows.extend_from_slice(embedding);
            }
        }
    }
    Ok(rows)
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

/// Samples one token with vLLM's penalties and candidate filters applied.
///
/// `tokens` is every token the sequence has consumed, in order; `options.prompt_len` splits it into
/// the prompt (seen by the repetition penalty only) and the generated text (counted by the
/// presence/frequency penalties and by the repetition penalty too). When
/// [SamplerOptions::is_default] holds this is [`sample_gumbel`] unchanged, so the disabled
/// configuration cannot move a token.
///
/// The greedy rule is vLLM's: a temperature that is not above zero means greedy, and vLLM resets
/// `top_p`/`top_k` at that point (`sampling_params.py` lines 530-535), so greedy is the argmax of
/// the penalized logits and the candidate filters are ignored.
#[must_use]
pub fn sample(
    logits: &[f32],
    tokens: &[u32],
    sampling: Sampling,
    options: SamplerOptions,
) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if options.is_default() {
        return sample_gumbel(logits, sampling.temperature, sampling.seed);
    }
    let mut shaped = logits.to_vec();
    apply_penalties(&mut shaped, tokens, options);
    if !sampling.temperature.is_finite() || sampling.temperature <= 0.0 {
        return linalg::argmax(&shaped).map(|index| index as u32);
    }
    let candidates = candidate_filter(&shaped, sampling.temperature, options);
    keep_candidates(&mut shaped, &candidates);
    sample_gumbel(&shaped, sampling.temperature, sampling.seed)
}

/// Applies the three vLLM penalties to `logits` in place.
///
/// `repetition_penalty` divides positive logits and multiplies negative ones for every token that
/// appears in the prompt or the generated text (`sampling_params.py` lines 248-253);
/// `presence_penalty` is subtracted once and `frequency_penalty` once per occurrence only for
/// tokens in the generated text (lines 240-246). Tokens outside the vocabulary cannot be produced
/// by the sampler and are ignored rather than indexing past the row.
fn apply_penalties(logits: &mut [f32], tokens: &[u32], options: SamplerOptions) {
    let vocab = logits.len();
    if options.repetition_penalty > 0.0 && options.repetition_penalty != 1.0 {
        let penalty = options.repetition_penalty;
        let mut seen = vec![false; vocab];
        for &token in tokens {
            if let Some(flag) = seen.get_mut(token as usize) {
                *flag = true;
            }
        }
        for (index, &flag) in seen.iter().enumerate() {
            if !flag {
                continue;
            }
            let value = logits[index];
            logits[index] = if value > 0.0 {
                value / penalty
            } else if value < 0.0 {
                value * penalty
            } else {
                value
            };
        }
    }
    if options.presence_penalty != 0.0 || options.frequency_penalty != 0.0 {
        let start = options.prompt_len.min(tokens.len());
        let mut occurrences = vec![0u32; vocab];
        for &token in &tokens[start..] {
            if let Some(count) = occurrences.get_mut(token as usize) {
                *count += 1;
            }
        }
        for (index, &count) in occurrences.iter().enumerate() {
            if count == 0 {
                continue;
            }
            logits[index] -= options.presence_penalty + options.frequency_penalty * count as f32;
        }
    }
}

/// The candidate rows the filters keep, most likely first.
///
/// `top_k` above zero keeps that many; `top_p` below one keeps the shortest prefix of the
/// temperature-scaled probability mass whose cumulative probability reaches it, always at least the
/// most likely token, so 0.0 degenerates to that token (our documented extension: vLLM refuses 0.0).
fn candidate_filter(logits: &[f32], temperature: f32, options: SamplerOptions) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len())
        .filter(|&index| logits[index].is_finite())
        .collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    if options.top_k > 0 {
        let keep = usize::try_from(options.top_k).unwrap_or(usize::MAX);
        order.truncate(keep.max(1));
    }
    if options.top_p < 1.0 && !order.is_empty() {
        let temperature = f64::from(temperature);
        let max = f64::from(logits[order[0]]) / temperature;
        let mut total = 0.0f64;
        let mut weights = Vec::with_capacity(order.len());
        for &index in &order {
            let scaled = f64::from(logits[index]) / temperature;
            let weight = (scaled - max).exp();
            total += weight;
            weights.push(weight);
        }
        let mut cumulative = 0.0f64;
        let mut keep = 1usize;
        for (position, weight) in weights.iter().enumerate() {
            cumulative += weight / total;
            keep = position + 1;
            if cumulative >= f64::from(options.top_p) {
                break;
            }
        }
        order.truncate(keep);
    }
    order
}

/// Masks every row outside `candidates` so the Gumbel draw cannot pick it.
fn keep_candidates(logits: &mut [f32], candidates: &[usize]) {
    let mut keep = vec![false; logits.len()];
    for &index in candidates {
        if let Some(flag) = keep.get_mut(index) {
            *flag = true;
        }
    }
    for (index, value) in logits.iter_mut().enumerate() {
        if !keep[index] {
            *value = f32::NEG_INFINITY;
        }
    }
}

impl InferenceEngine {
    /// The step [InferenceEngine::prefill] and [InferenceEngine::decode] both run: the whole-model
    /// order of `Transformer.forward` (model.py:1241-1272, section 3.1 of the forward-order doc),
    /// with `seqlen` positions at a time and `start_pos` already in the caches.
    /// One step with no selected positions: the plain prefill/decode output.
    fn run(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        capture: Option<&mut BoundaryTable>,
    ) -> Result<StepOutput, EngineError> {
        self.run_selected(state, tokens, capture, None)
            .map(|(step, _)| step)
    }

    /// One step, optionally reading the LM-head logits of the requested positions from the same
    /// forward.
    ///
    /// The positions are 0-based inside the chunk. Callers validate them before the pass (so an
    /// invalid request fails before it costs a pass); this function debug-asserts that and never
    /// indexes outside the collapse.
    fn run_selected(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        mut capture: Option<&mut BoundaryTable>,
        positions: Option<&[usize]>,
    ) -> Result<(StepOutput, Vec<Vec<f32>>), EngineError> {
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

        // The embedding lookup, one row per token (model.py:1253-1255). The rows are widened from
        // the stored BF16 table on the release path, so a decode step touches only its own rows.
        let embeddings = embedding_rows(source, tokens, dim, config.vocab_size, self.limit)?;

        // The stream becomes [seqlen, hc, dim]: every copy starts as the same embedding, and the
        // first input mix selects copy 0 (model.py:1258-1260).
        let mut hidden = vec![0.0f32; seqlen * width];
        for position in 0..seqlen {
            let embedding = &embeddings[position * dim..][..dim];
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
        // The selected positions read the same collapse, final norm and head that produce the
        // tail below: the per-layer pass is shared and no prefix is re-forwarded.
        let selected_rows = match positions {
            None => Vec::new(),
            Some([]) => Vec::new(),
            Some(positions) => {
                let mut selected = Vec::with_capacity(positions.len().saturating_mul(dim));
                for &position in positions {
                    debug_assert!(position < seqlen);
                    selected.extend_from_slice(&collapsed[position * dim..][..dim]);
                }
                for row in selected.chunks_mut(dim) {
                    linalg::rms_norm(row, &norm, dim, config.norm_eps as f32)?;
                }
                let selected_logits = self.head_logits(source, &selected, positions.len())?;
                selected_logits
                    .chunks_exact(config.vocab_size)
                    .map(<[f32]>::to_vec)
                    .collect()
            }
        };
        let mut tail = collapsed[(seqlen - 1) * dim..].to_vec();
        linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        let head_input = tail.clone();
        let logits = self.head_logits(source, &tail, 1)?;

        // The sampler options see the tokens consumed so far: `history` is the state before this
        // chunk plus the chunk itself, built above for the Engram hashes. The default options take
        // the untouched `sample_gumbel` path, so a request that sets none of them keeps producing
        // exactly the tokens the driver produced before the options existed.
        let next_token = if self.sampler.is_default() {
            sample_gumbel(&logits, self.sampling.temperature, self.sampling.seed)
        } else {
            sample(&logits, &history, self.sampling, self.sampler)
        };

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
        Ok((
            StepOutput {
                logits,
                head_input,
                next_token,
                draft,
                confidence,
                consumed: seqlen,
            },
            selected_rows,
        ))
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
        mut moe: Option<&mut dyn moe::MoeDeviceBackend>,
        deferred: Option<&mut Option<DeferredHandoff>>,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let prefix = format!("layers.{layer}");
        let source: &dyn WeightSource = self.source.as_ref();
        if self.tp.is_some() && (attention.is_some() || moe.is_some()) {
            return Err(EngineError::Shape {
                what: "the TP reference block".to_owned(),
                detail: "the sharded reference path has no device backend seam; pass none"
                    .to_owned(),
            });
        }

        // The entry boundaries: the residual stream this block received and the input mix it will
        // consume (docs/archive/build/phase1-boundary-table.md section 2).
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
        // Batch-18: an engine that has the exact gather seam and a rank-local attention source
        // runs the Option-A body (one rank's head block, wo_a rows and o_lora slice). A
        // whole-tensor source answers None and keeps the replicated path, which is also the
        // default whenever no gather is installed.
        let shard_geometry = match self.gather.as_ref() {
            Some(gather) => {
                attention::attention_shard_geometry(config, &prefix, source, gather.world())?
            }
            None => None,
        };
        if let Some((heads, groups)) = shard_geometry {
            if attention.is_some() {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} attention"),
                    detail: "the batch-18 attention shard is host-only; a device projection backend cannot run it".to_owned(),
                });
            }
            let gather = self
                .gather
                .as_ref()
                .expect("a shard geometry implies the gather");
            let shard = attention::AttentionShard {
                gather,
                rank: gather.rank(),
                heads,
                groups,
            };
            attention::attention_forward_sharded(
                config,
                layer,
                &prefix,
                &scratch.attn_in,
                start_pos,
                &mut state.attention[layer],
                source,
                &mut scratch.attn_out,
                self.limit,
                &shard,
            )?;
        } else {
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
            if post.x.len() != hidden.len() || post.ffn_in.len() != scratch.ffn_in.len() {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} device post"),
                    detail: format!(
                        "post X is {} values (expected {}) and ffn_in is {} values (expected {})",
                        post.x.len(),
                        hidden.len(),
                        post.ffn_in.len(),
                        scratch.ffn_in.len()
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
        // A hybrid chunk asks the backend once for the whole chunk; a single-token step keeps the
        // original one-token seam so every existing backend sees the call it always saw. A backend
        // that returns `false` for a chunk is refused below rather than silently falling back to the
        // host block: on the sharded path that fallback would compute the global experts.
        let chunk_positions = scratch.ffn_in.len() / config.dim;
        let device_moe = match moe.as_deref_mut() {
            Some(backend) if chunk_positions > 1 => backend.ffn_chunk(
                config,
                layer,
                &prefix,
                &scratch.ffn_in,
                source,
                self.limit,
                chunk_positions,
                &mut scratch.ffn_out,
            )?,
            Some(backend) => backend.ffn(
                config,
                layer,
                &prefix,
                &scratch.ffn_in,
                source,
                self.limit,
                &mut scratch.ffn_out,
            )?,
            None => false,
        };
        if let Some(tp) = &self.tp {
            // The one Phase-1 TP seam: the routed experts are this rank's shards, the shared
            // expert is replicated, and the routed sum crosses one all-reduce per layer.
            moe::routed_forward_sharded(
                config,
                &prefix,
                &scratch.ffn_in,
                source,
                self.limit,
                &mut scratch.moe,
                &mut scratch.ffn_out,
                tp.local_moe_inter(),
            )?;
            tp.all_reduce(&mut scratch.ffn_out)?;
            moe::add_shared_expert(
                config,
                &prefix,
                &scratch.ffn_in,
                source,
                self.limit,
                &mut scratch.moe,
                &mut scratch.ffn_out,
            )?;
        } else if !device_moe {
            if chunk_positions > 1 && moe.is_some() {
                return Err(EngineError::Shape {
                    what: format!("layer {layer} chunked hybrid FFN"),
                    detail: "the device backend did not accept the chunk".to_owned(),
                });
            }
            moe::moe_forward(
                config,
                &prefix,
                &scratch.ffn_in,
                source,
                self.limit,
                &mut scratch.moe,
                &mut scratch.ffn_out,
            )?;
        } else {
            let device_shared = match moe.as_mut() {
                Some(backend) if chunk_positions > 1 => backend.shared_expert_chunk(
                    config,
                    layer,
                    &prefix,
                    &scratch.ffn_in,
                    source,
                    self.limit,
                    chunk_positions,
                    &mut scratch.ffn_out,
                )?,
                Some(backend) => backend.shared_expert(
                    config,
                    layer,
                    &prefix,
                    &scratch.ffn_in,
                    source,
                    self.limit,
                    &mut scratch.ffn_out,
                )?,
                None => false,
            };
            if !device_shared {
                moe::add_shared_expert(
                    config,
                    &prefix,
                    &scratch.ffn_in,
                    source,
                    self.limit,
                    &mut scratch.moe,
                    &mut scratch.ffn_out,
                )?;
            }
        }
        record_stage(
            &mut stages,
            "shared_expert_down",
            u64_bytes(scratch.ffn_in.len()),
            u64_bytes(config.dim),
            "shared-expert w2 projection and the shared add into the routed row",
        );
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
                None,
            )?;
            let host_ms = started.elapsed().as_millis() as u64;
            layers.push(HybridLayerTrace {
                layer,
                stages: layer_stages,
                residual_out: hidden.clone(),
                pre_attention_hidden: handoff.entry.pre_attention_hidden.clone(),
                entry_input_x: handoff.x.clone(),
                post_attention_x: Vec::new(),
                ffn_out: Vec::new(),
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
        let logits = self.head_logits(self.source.as_ref(), &tail, 1)?;
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
            certified_mask: None,
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
        attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        moe_backend: Option<&mut dyn moe::MoeDeviceBackend>,
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridTokenOutput, EngineError> {
        let mut state = SequenceState::new(&self.config);
        self.run_hybrid_sequence_step(
            &mut state,
            token,
            embedding,
            0,
            provider,
            post,
            attention,
            moe_backend,
            progress,
        )
    }

    /// One deferred hybrid step at an explicit position in a persistent [SequenceState].
    ///
    /// This is the sequence-aware form of [Self::run_hybrid_chain_deferred]: the provider and the
    /// device attention/MoE backends see the caller's accumulated KV/attention state, and `start_pos`
    /// is the position this one-token step writes. The one-token method above is a fresh-state
    /// wrapper, so the b40/b41 single-token evidence keeps the same code path.
    /// The full-step form: every layer, no behaviour change for existing callers.
    #[allow(clippy::too_many_arguments)]
    pub fn run_hybrid_sequence_step(
        &self,
        state: &mut SequenceState,
        token: u32,
        embedding: &[f32],
        start_pos: usize,
        provider: &mut HybridDeferredProvider<'_>,
        post: &mut HybridPostProvider<'_>,
        attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        moe_backend: Option<&mut dyn moe::MoeDeviceBackend>,
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridTokenOutput, EngineError> {
        let layer_window = StepLayers::all(self.config.n_layers);
        self.run_hybrid_sequence_step_ranged(
            state,
            token,
            embedding,
            start_pos,
            provider,
            post,
            attention,
            moe_backend,
            layer_window,
            progress,
        )
    }

    /// One deferred hybrid step over the requested layer window. Diagnostic bisection uses a window
    /// below n_layers - 1; the numbers it produces are partial-model, not logits of the full
    /// checkpoint, and callers must label them that way.
    #[allow(clippy::too_many_arguments)]
    pub fn run_hybrid_sequence_step_ranged(
        &self,
        state: &mut SequenceState,
        token: u32,
        embedding: &[f32],
        start_pos: usize,
        provider: &mut HybridDeferredProvider<'_>,
        post: &mut HybridPostProvider<'_>,
        mut attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        mut moe_backend: Option<&mut dyn moe::MoeDeviceBackend>,
        layer_window: StepLayers,
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridTokenOutput, EngineError> {
        let input_token = token;
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
        if state.tokens.len() != start_pos {
            return Err(EngineError::Shape {
                what: "deferred hybrid sequence step".to_owned(),
                detail: format!(
                    "start position {start_pos} does not match the state's {} consumed tokens",
                    state.tokens.len()
                ),
            });
        }
        let mut scratch = Scratch::for_chunk(config, 1);
        let mut layers = Vec::with_capacity(config.n_layers);
        let mut deferred: Option<DeferredHandoff> = None;
        if layer_window.start > layer_window.end || layer_window.end >= config.n_layers {
            return Err(EngineError::Shape {
                what: "deferred hybrid layer window".to_owned(),
                detail: format!(
                    "layers {}..={} are outside the {} model layers",
                    layer_window.start, layer_window.end, config.n_layers
                ),
            });
        }
        for layer in layer_window.start..=layer_window.end {
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
                state,
                layer,
                &mut hidden,
                &mut pre_mix,
                &mut scratch,
                start_pos,
                Some(&handoff.entry),
                Some(&mut layer_stages),
                None,
                Some(&mut *post),
                attention.as_mut().map(|backend| {
                    &mut **backend as &mut dyn attention::AttentionProjectionBackend
                }),
                moe_backend
                    .as_deref_mut()
                    .map(|backend| &mut *backend as &mut dyn moe::MoeDeviceBackend),
                if defer_epilogue {
                    Some(&mut next_deferred)
                } else {
                    None
                },
            )?;
            let deferred_post_x = next_deferred
                .as_ref()
                .map(|handoff| handoff.x.clone())
                .unwrap_or_default();
            let deferred_ffn = next_deferred
                .as_ref()
                .map(|handoff| handoff.y.clone())
                .unwrap_or_default();
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
                pre_attention_hidden: handoff.entry.pre_attention_hidden.clone(),
                entry_input_x: handoff.x.clone(),
                post_attention_x: deferred_post_x,
                ffn_out: deferred_ffn,
                host_ms,
            });
            progress(layer, host_ms);
        }
        let mut collapsed = vec![0.0f32; dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut collapsed)?;
        let mut tail = collapsed;
        let mut device_norm = false;
        if let Some(backend) = moe_backend.as_mut() {
            let mut norm_bf16 = Vec::new();
            if backend.final_norm(
                config,
                &tail,
                self.source.as_ref(),
                self.limit,
                &mut norm_bf16,
            )? {
                tail = norm_bf16
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| {
                        let bits = u16::from_le_bytes(*pair);
                        f32::from_bits(u32::from(bits) << 16)
                    })
                    .collect();
                device_norm = true;
            }
        }
        if !device_norm {
            let norm = linalg::decode_by_name(self.source.as_ref(), "norm.weight", self.limit)?;
            linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        }
        let head_input = tail.clone();
        let mut logits = vec![0.0f32; config.vocab_size];
        let device_token = match moe_backend.as_mut() {
            Some(backend) => {
                backend.lm_head(config, &tail, self.source.as_ref(), self.limit, &mut logits)?
            }
            None => None,
        };
        let token = if let Some(token) = device_token {
            Some(token)
        } else {
            logits = self.head_logits(self.source.as_ref(), &tail, 1)?;
            linalg::argmax(&logits).map(|index| index as u32)
        };
        let certified_mask = moe_backend
            .as_ref()
            .and_then(|backend| backend.head_certified_mask().map(<[u8]>::to_vec));
        let head_note = if device_token.is_some() {
            "device cert_lm_head_sm121 screen and in-kernel argmax over four rank shards; host block reduction"
        } else {
            "host head matmul; the device cert_lm_head_sm121 screen is a host stand-in in this run"
        };
        let argmax_note = if device_token.is_some() {
            "device block argmax; host four-shard assembly"
        } else {
            "host greedy argmax; class-(b) seam, not device-executed"
        };
        let tail_stages = vec![
            StageEvent {
                stage: "final_norm",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(dim),
                note: if device_norm {
                    "device rmsnorm_sm121 on the collapsed tail; BF16 out, host readback into the certified head"
                } else {
                    "host final RMS norm; device rmsnorm_sm121 was not run in this invocation"
                },
            },
            StageEvent {
                stage: "lm_head_screen",
                host_bytes_in: u64_bytes(dim),
                host_bytes_out: u64_bytes(config.vocab_size),
                note: head_note,
            },
            StageEvent {
                stage: "lm_head_argmax",
                host_bytes_in: u64_bytes(config.vocab_size),
                host_bytes_out: 4,
                note: argmax_note,
            },
        ];
        state.tokens.push(input_token);
        Ok(HybridTokenOutput {
            layers,
            tail_stages,
            head_input,
            logits,
            token,
            certified_mask,
        })
    }

    /// One chunked hybrid step: [Self::run_hybrid_sequence_step] over `tokens.len()` positions in a
    /// single layer-major pass.
    ///
    /// The provider is called once per (layer, token) with that token's own hidden row and its own
    /// deferred handoff, so every token's entry arithmetic is the same call it makes on a one-token
    /// step. The block then runs once for the whole chunk: one attention pass, one device-post call,
    /// one MoE backend call and (on the sharded path) one collective per layer.
    ///
    /// The post callback receives the chunk's `hidden` and `attention_out` and must return one `x`
    /// row and one normalized `ffn_in` row per position, with `mixes` in [Mixes::for_tokens] layout.
    /// `embeddings` is `tokens.len() * dim` values in token order. The returned logits and
    /// certified mask are the last position's; `state` advances by `tokens.len()` positions.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] for an empty chunk, a start position that does not match the state, a
    /// token outside the vocabulary, an embedding block of the wrong length, a provider/post/backend
    /// handoff of the wrong shape, a backend that declines a chunk, or a device attention backend on
    /// a multi-token chunk (that backend has no chunk seam), plus any weight-read or arithmetic
    /// error the per-token hybrid step can return.
    #[allow(clippy::too_many_arguments)] // the per-token step's own inputs plus the chunk; bundling them would hide which one is the chunk.
    pub fn run_hybrid_sequence_chunk(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        embeddings: &[f32],
        start_pos: usize,
        provider: &mut HybridChunkDeferredProvider<'_>,
        post: &mut HybridPostProvider<'_>,
        mut attention: Option<&mut dyn attention::AttentionProjectionBackend>,
        mut moe_backend: Option<&mut dyn moe::MoeDeviceBackend>,
        position_logits: &[usize],
        progress: &mut dyn FnMut(usize, u64),
    ) -> Result<HybridChunkOutput, EngineError> {
        let config = &self.config;
        let dim = config.dim;
        let hc = config.hc_mult;
        let width = config.hc_width();
        let seqlen = tokens.len();
        if dim == 0 || hc == 0 || width == 0 {
            return Err(EngineError::Shape {
                what: "chunked hybrid chain".to_owned(),
                detail: "dim and hc_mult must both be positive".to_owned(),
            });
        }
        if seqlen == 0 {
            return Err(EngineError::Shape {
                what: "chunked hybrid chain".to_owned(),
                detail: "no tokens to run".to_owned(),
            });
        }
        if attention.is_some() && seqlen > 1 {
            return Err(EngineError::Shape {
                what: "chunked hybrid chain".to_owned(),
                detail: "the device attention backend has no chunk seam".to_owned(),
            });
        }
        if embeddings.len() != seqlen * dim {
            return Err(EngineError::Shape {
                what: "chunked hybrid chain embedding".to_owned(),
                detail: format!("expected {} values, got {}", seqlen * dim, embeddings.len()),
            });
        }
        for &token in tokens {
            if token as usize >= config.vocab_size {
                return Err(EngineError::Shape {
                    what: "chunked hybrid chain embedding".to_owned(),
                    detail: format!(
                        "token {token} is outside the {} row table",
                        config.vocab_size
                    ),
                });
            }
        }
        if state.tokens.len() != start_pos {
            return Err(EngineError::Shape {
                what: "chunked hybrid sequence step".to_owned(),
                detail: format!(
                    "start position {start_pos} does not match the state's {} consumed tokens",
                    state.tokens.len()
                ),
            });
        }

        let mut hidden = vec![0.0f32; seqlen * width];
        for position in 0..seqlen {
            let embedding = &embeddings[position * dim..(position + 1) * dim];
            for copy in 0..hc {
                hidden[(position * width) + copy * dim..(position * width) + (copy + 1) * dim]
                    .copy_from_slice(embedding);
            }
        }
        let mut pre_mix = vec![0.0f32; seqlen * hc];
        for position in 0..seqlen {
            pre_mix[position * hc] = 1.0;
        }
        let mut scratch = Scratch::for_chunk(config, seqlen);
        let mut deferred: Option<DeferredHandoff> = None;
        for layer in 0..config.n_layers {
            let started = std::time::Instant::now();
            // One entry per token, concatenated in token order: the block consumes the chunk entry
            // exactly as the single-token block consumes one token's entry.
            let mut entry = LayerZeroEntry {
                pre_attention_hidden: vec![0.0f32; seqlen * dim],
                pre: vec![0.0f32; seqlen * hc],
                post: vec![0.0f32; seqlen * hc],
                comb: vec![0.0f32; seqlen * hc * hc],
                source: LayerZeroEntrySource::Host,
            };
            for position in 0..seqlen {
                let token_deferred = deferred.as_ref().map(|handoff| DeferredHandoff {
                    x: handoff.x[position * width..(position + 1) * width].to_vec(),
                    y: handoff.y[position * dim..(position + 1) * dim].to_vec(),
                    mixes: Mixes {
                        pre: handoff.mixes.pre[position * hc..(position + 1) * hc].to_vec(),
                        post: handoff.mixes.post[position * hc..(position + 1) * hc].to_vec(),
                        comb: handoff.mixes.comb[position * hc * hc..(position + 1) * hc * hc]
                            .to_vec(),
                    },
                });
                let handoff = {
                    let hidden_row = &hidden[position * width..(position + 1) * width];
                    let pre_row = &pre_mix[position * hc..(position + 1) * hc];
                    provider(
                        layer,
                        position,
                        hidden_row,
                        pre_row,
                        token_deferred.as_ref(),
                    )?
                };
                if handoff.x.len() != width {
                    return Err(EngineError::Shape {
                        what: format!("layer {layer} MhcEntry X"),
                        detail: format!("expected {width} values, got {}", handoff.x.len()),
                    });
                }
                hidden[position * width..(position + 1) * width].copy_from_slice(&handoff.x);
                entry.pre_attention_hidden[position * dim..(position + 1) * dim]
                    .copy_from_slice(&handoff.entry.pre_attention_hidden);
                entry.pre[position * hc..(position + 1) * hc].copy_from_slice(&handoff.entry.pre);
                entry.post[position * hc..(position + 1) * hc].copy_from_slice(&handoff.entry.post);
                entry.comb[position * hc * hc..(position + 1) * hc * hc]
                    .copy_from_slice(&handoff.entry.comb);
                entry.source = handoff.entry.source;
            }
            let mut next_deferred = None;
            let defer_epilogue = layer + 1 < config.n_layers;
            self.block(
                state,
                layer,
                &mut hidden,
                &mut pre_mix,
                &mut scratch,
                start_pos,
                Some(&entry),
                None,
                None,
                Some(&mut *post),
                attention.as_mut().map(|backend| {
                    &mut **backend as &mut dyn attention::AttentionProjectionBackend
                }),
                moe_backend
                    .as_deref_mut()
                    .map(|backend| &mut *backend as &mut dyn moe::MoeDeviceBackend),
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
            progress(layer, started.elapsed().as_millis() as u64);
        }

        // The tail runs on the last position: the collapsed copies with the last block's FFN
        // coefficient, then the final norm and the LM head (model.py:1268-1269).
        let mut collapsed = vec![0.0f32; seqlen * dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut collapsed)?;
        let mut tail = collapsed[(seqlen - 1) * dim..].to_vec();
        let mut device_norm = false;
        if let Some(backend) = moe_backend.as_mut() {
            let mut norm_bf16 = Vec::new();
            if backend.final_norm(
                config,
                &tail,
                self.source.as_ref(),
                self.limit,
                &mut norm_bf16,
            )? {
                tail = norm_bf16
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| {
                        let bits = u16::from_le_bytes(*pair);
                        f32::from_bits(u32::from(bits) << 16)
                    })
                    .collect();
                device_norm = true;
            }
        }
        if !device_norm {
            let norm = linalg::decode_by_name(self.source.as_ref(), "norm.weight", self.limit)?;
            linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)?;
        }
        let head_input = tail.clone();
        // The one-pass prompt-logprob seam for the hybrid chunk: the same collapsed rows the last
        // position's head input comes from, final-normed for every requested position. The device
        // final-norm backend has no per-position seam yet, so a device backend refuses instead of
        // returning rows it did not norm.
        let mut position_head_inputs = Vec::new();
        if !position_logits.is_empty() {
            if device_norm {
                return Err(EngineError::Shape {
                    what: "chunked hybrid all-position rows".to_owned(),
                    detail: "the device final-norm backend has no all-position seam; run the host norm path to select rows"
                        .to_owned(),
                });
            }
            let norm = linalg::decode_by_name(self.source.as_ref(), "norm.weight", self.limit)?;
            for &position in position_logits {
                if position >= seqlen {
                    return Err(EngineError::Shape {
                        what: "chunked hybrid all-position rows".to_owned(),
                        detail: format!("position {position} is outside the {seqlen} token chunk"),
                    });
                }
                let mut row = collapsed[position * dim..(position + 1) * dim].to_vec();
                linalg::rms_norm(&mut row, &norm, dim, config.norm_eps as f32)?;
                position_head_inputs.push(row);
            }
        }
        let mut logits = vec![0.0f32; config.vocab_size];
        let device_token = match moe_backend.as_mut() {
            Some(backend) => {
                backend.lm_head(config, &tail, self.source.as_ref(), self.limit, &mut logits)?
            }
            None => None,
        };
        let token = if let Some(token) = device_token {
            Some(token)
        } else {
            logits = self.head_logits(self.source.as_ref(), &tail, 1)?;
            linalg::argmax(&logits).map(|index| index as u32)
        };
        let certified_mask = moe_backend
            .as_ref()
            .and_then(|backend| backend.head_certified_mask().map(<[u8]>::to_vec));
        state.tokens.extend_from_slice(tokens);
        Ok(HybridChunkOutput {
            head_input,
            logits,
            token,
            certified_mask,
            position_head_inputs,
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

    /// The disabled options are the pre-existing sampler: the new entry point and `sample_gumbel`
    /// return the same token for every temperature, seed and logits row. This is the default-off
    /// equivalence that lets the options exist without moving a byte of output.
    #[test]
    fn default_sampler_options_are_the_untouched_reference_sampler() {
        let cases: [&[f32]; 5] = [
            &[0.5, 2.0, -1.0],
            &[0.0, 0.1, 0.2, 0.3],
            &[1e-6, 1e-6, 1e-6],
            &[10.0, -10.0, 0.0],
            &[1.0, 0.999_9],
        ];
        for logits in cases {
            for temperature in [0.0, 0.7, 1.5] {
                for seed in 0..64 {
                    let sampling = Sampling { temperature, seed };
                    assert_eq!(
                        sample(logits, &[0, 1], sampling, SamplerOptions::default()),
                        sample_gumbel(logits, temperature, seed),
                        "logits {logits:?} temperature {temperature} seed {seed}"
                    );
                }
            }
        }
    }

    /// `top_k` 1 is the argmax at every seed: one candidate cannot be outvoted by the noise.
    #[test]
    fn top_k_one_is_the_argmax_at_every_seed() {
        let logits = [0.2f32, 0.9, -0.4, 2.5, 1.1];
        let expected = linalg::argmax(&logits).expect("non-empty") as u32;
        for seed in 0..256 {
            let options = SamplerOptions {
                top_k: 1,
                ..SamplerOptions::default()
            };
            assert_eq!(
                sample(
                    &logits,
                    &[1, 2],
                    Sampling {
                        temperature: 1.0,
                        seed
                    },
                    options
                ),
                Some(expected)
            );
        }
    }

    /// `top_k` keeps exactly the most likely rows, and 0/-1 disable it (the untouched sampler).
    #[test]
    fn top_k_keeps_exactly_the_most_likely_rows() {
        let logits = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        for seed in 0..256 {
            let options = SamplerOptions {
                top_k: 2,
                ..SamplerOptions::default()
            };
            let picked = sample(
                &logits,
                &[],
                Sampling {
                    temperature: 1.0,
                    seed,
                },
                options,
            );
            assert!(
                matches!(picked, Some(4) | Some(3)),
                "top_k 2 picked {picked:?} at seed {seed}"
            );
        }
        for seed in 0..64 {
            let sampling = Sampling {
                temperature: 1.0,
                seed,
            };
            for disabled in [0, -1] {
                let options = SamplerOptions {
                    top_k: disabled,
                    ..SamplerOptions::default()
                };
                assert_eq!(
                    sample(&logits, &[], sampling, options),
                    sample_gumbel(&logits, 1.0, seed)
                );
            }
        }
    }

    /// `top_p` 0.0 keeps only the most likely token: our documented extension, because vLLM
    /// refuses that value. It is the argmax at every seed.
    #[test]
    fn top_p_zero_is_the_argmax_at_every_seed() {
        let logits = [0.2f32, 0.9, -0.4, 2.5, 1.1];
        let expected = linalg::argmax(&logits).expect("non-empty") as u32;
        for seed in 0..256 {
            let options = SamplerOptions {
                top_p: 0.0,
                ..SamplerOptions::default()
            };
            assert_eq!(
                sample(
                    &logits,
                    &[],
                    Sampling {
                        temperature: 1.0,
                        seed
                    },
                    options
                ),
                Some(expected)
            );
        }
    }

    /// `top_p` 1.0 is disabled, so it is the untouched sampler.
    #[test]
    fn top_p_one_is_the_untouched_sampler() {
        let logits = [0.2f32, 0.9, -0.4, 2.5, 1.1];
        for seed in 0..64 {
            let sampling = Sampling {
                temperature: 1.0,
                seed,
            };
            assert_eq!(
                sample(
                    &logits,
                    &[],
                    sampling,
                    SamplerOptions {
                        top_p: 1.0,
                        ..SamplerOptions::default()
                    }
                ),
                sample_gumbel(&logits, 1.0, seed)
            );
        }
    }

    /// The nucleus keeps the shortest probability prefix that reaches the threshold. With logits
    /// ln(1), ln(2), ln(4) the probabilities are 1/7, 2/7 and 4/7: 0.5 keeps the single 4/7 token,
    /// 0.6 keeps the top two, and only seeds inside the kept set can be drawn.
    #[test]
    fn a_nucleus_keeps_only_the_probability_mass_it_names() {
        let logits = [0.0f32, 2.0f32.ln(), 4.0f32.ln()];
        for seed in 0..256 {
            let one = sample(
                &logits,
                &[],
                Sampling {
                    temperature: 1.0,
                    seed,
                },
                SamplerOptions {
                    top_p: 0.5,
                    ..SamplerOptions::default()
                },
            );
            assert_eq!(one, Some(2), "0.5 keeps the 4/7 token only");
            let two = sample(
                &logits,
                &[],
                Sampling {
                    temperature: 1.0,
                    seed,
                },
                SamplerOptions {
                    top_p: 0.6,
                    ..SamplerOptions::default()
                },
            );
            assert!(
                matches!(two, Some(1) | Some(2)),
                "0.6 keeps 1 and 2: {two:?}"
            );
        }
    }

    /// `repetition_penalty` above 1 divides a positive seen logit and multiplies a negative one,
    /// so the repeated token loses on both sides of zero (vLLM's rule).
    #[test]
    fn repetition_penalty_discounts_a_seen_token_above_one() {
        let greedy = Sampling {
            temperature: 0.0,
            seed: 0,
        };
        let logits = [10.0f32, 9.0];
        let seen = SamplerOptions {
            repetition_penalty: 100.0,
            ..SamplerOptions::default()
        };
        assert_eq!(
            sample(&logits, &[], greedy, SamplerOptions::default()),
            Some(0)
        );
        // 10 / 100 = 0.1 falls below 9.
        assert_eq!(sample(&logits, &[0], greedy, seen), Some(1));
        // The negative branch: -1 * 100 = -100 falls below -2.
        let negative = [-1.0f32, -2.0];
        assert_eq!(sample(&negative, &[0], greedy, seen), Some(1));
    }

    /// `repetition_penalty` below 1 encourages the seen token: 1 / 0.4 = 2.5 beats 2.
    #[test]
    fn repetition_penalty_below_one_encourages_a_seen_token() {
        let greedy = Sampling {
            temperature: 0.0,
            seed: 0,
        };
        let logits = [1.0f32, 2.0];
        let seen = SamplerOptions {
            repetition_penalty: 0.4,
            ..SamplerOptions::default()
        };
        assert_eq!(
            sample(&logits, &[], greedy, SamplerOptions::default()),
            Some(1)
        );
        assert_eq!(sample(&logits, &[0], greedy, seen), Some(0));
    }

    /// `presence_penalty` lowers a generated token once, and only generated tokens: everything
    /// before `prompt_len` is prompt and is not penalized.
    #[test]
    fn presence_penalty_lowers_a_generated_token_once() {
        let greedy = Sampling {
            temperature: 0.0,
            seed: 0,
        };
        let generated = SamplerOptions {
            presence_penalty: 2.0,
            prompt_len: 1,
            ..SamplerOptions::default()
        };
        // Token 2 is generated: 0.3 - 2 = -1.7, below token 0's 0.5.
        assert_eq!(
            sample(&[0.5f32, 0.4, 0.3], &[9, 2], greedy, generated),
            Some(0)
        );
        // Token 2 is the prompt's last token: not penalized, so it stays the argmax.
        let prompt_only = SamplerOptions {
            presence_penalty: 2.0,
            prompt_len: 2,
            ..SamplerOptions::default()
        };
        assert_eq!(
            sample(&[0.5f32, 0.4, 1.0], &[9, 2, 5], greedy, prompt_only),
            Some(2)
        );
    }

    /// `frequency_penalty` counts occurrences: one occurrence of 5 with penalty 2 leaves 3 and
    /// still wins, two occurrences leave 1 and lose to 2.
    #[test]
    fn frequency_penalty_counts_each_occurrence() {
        let greedy = Sampling {
            temperature: 0.0,
            seed: 0,
        };
        let options = SamplerOptions {
            frequency_penalty: 2.0,
            ..SamplerOptions::default()
        };
        let logits = [2.0f32, 5.0];
        assert_eq!(sample(&logits, &[1], greedy, options), Some(1));
        assert_eq!(sample(&logits, &[1, 1], greedy, options), Some(0));
    }

    /// The penalties are applied before the candidate filters: a repeated token that loses to the
    /// penalty is no longer the top-p candidate either.
    #[test]
    fn penalties_apply_before_the_candidate_filters() {
        let greedy = Sampling {
            temperature: 0.0,
            seed: 0,
        };
        let options = SamplerOptions {
            repetition_penalty: 100.0,
            ..SamplerOptions::default()
        };
        // 10 becomes 0.1, so the nucleus's most likely token is 9, not 10.
        let logits = [10.0f32, 9.0];
        assert_eq!(sample(&logits, &[0], greedy, options), Some(1));
        let with_nucleus = SamplerOptions {
            top_p: 0.0,
            ..options
        };
        assert_eq!(sample(&logits, &[0], greedy, with_nucleus), Some(1));
    }

    /// The seeded draw stays a pure function of its inputs with every knob on.
    #[test]
    fn seeded_draws_are_reproducible_with_penalties_and_filters() {
        let options = SamplerOptions {
            top_p: 0.9,
            top_k: 3,
            repetition_penalty: 1.5,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            prompt_len: 1,
        };
        let logits = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let tokens = [2u32, 3, 4];
        for seed in 0..8 {
            let sampling = Sampling {
                temperature: 0.8,
                seed,
            };
            let first = sample(&logits, &tokens, sampling, options);
            assert_eq!(first, sample(&logits, &tokens, sampling, options));
            assert!(first.is_some());
        }
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
                    // f32 for every tensor except the Engram wkv projection, whose checkpoint
                    // storage format is the only way to recover the codes (audit F4).
                    dtype: crate::tensor::checkpoint_storage_dtype(name).unwrap_or(DType::F32),
                    ..spec.clone()
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            let spec = self.spec(name)?;
            let elements = usize::try_from(spec.elements()).unwrap_or(usize::MAX);
            let value_at = |index: usize| 0.02 + 0.01 * (index % 5) as f32;
            let mut bytes = Vec::with_capacity(elements.saturating_mul(4));
            match spec.dtype {
                DType::F8E4M3 => {
                    for index in 0..elements {
                        bytes.push(crate::quant::e4m3_encode(value_at(index)));
                    }
                }
                DType::F8E8M0 => bytes.resize(elements, 127u8),
                _ => {
                    for index in 0..elements {
                        bytes.extend_from_slice(&value_at(index).to_le_bytes());
                    }
                }
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
    /// F8: the reference's norms call `torch.rsqrt` (model.py:292 for RMSNorm, 953 for hc_mixes),
    /// which is the reciprocal of the f32 square root. This host computes `1.0 / sqrt` and nothing
    /// else, so the test pins the bits: a later change cannot quietly substitute an rsqrt
    /// approximation. The device ambiguity is declared, not aligned: CUDA's `torch.rsqrt` lowers to
    /// a ~2 ulp intrinsic and no host arithmetic reproduces it bit for bit (native-arithmetic-audit
    /// F8).
    #[test]
    fn the_norm_rsqrt_is_the_correctly_rounded_reciprocal_sqrt() {
        let source = [3.0f32, 4.0, 0.5, 1.25];
        let weight = [1.5f32, 1.0, 0.5, 2.0];
        let mut values = source;
        linalg::rms_norm(&mut values, &weight, 4, 1e-20).expect("one row of four");
        let mean = source.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inverse = 1.0 / (mean + 1e-20f32).sqrt();
        for ((value, source), weight) in values.iter().zip(source).zip(weight) {
            assert_eq!(
                *value,
                source * inverse * weight,
                "the norm must scale by the correctly rounded reciprocal square root"
            );
        }
    }
}
