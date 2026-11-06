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

//! The serving backend: the device rank behind [ModelBackend], one session a request.
//!
//! The rank is the resident half - its weights, kernels and scratch - and every session owns only
//! a [DeviceSequence] (its KV state) plus its sampler. A rank's forward takes `&mut self` because a
//! step writes the rank's scratch, so sessions share the rank through a mutex: one forward at a
//! time, which is also what the tensor-parallel group requires (every rank runs the same step).
//!
//! The endpoint's sampling runs on the host over the row a greedy step leaves on the device
//! ([DeviceRank::last_logits]), so temperature, top-k, top-p and the penalties cost one vocabulary
//! copy a step rather than a second device kernel.
//!
//! A backend stepped together ([Glm5NextDeviceBackend::enable_fusion]) hands out handles on one
//! engine thread instead ([GlmFusion]): the sessions of a step are one batch over the rank, so the
//! weights are read once for every member's row. Fusion and drafting are two trades on the same
//! step and this arm runs one of them.
//!
//! A session drafts once [Glm5NextDeviceBackend::enable_speculation] has been called: a decode
//! drafts a chain after the token it was handed ([DeviceRank::draft]), verifies the chain and the
//! token in one step ([DeviceRank::verify_batch]), keeps the leading drafts the model would have
//! picked ([sparkflash_model::verdict]) and hands back every token the step produced, one per
//! `decode` call ([sparkflash_model::Queue]). The draft depth and its accounting are
//! [sparkflash_model::Speculator]'s, so the same rule and the same counters serve every drafting
//! arm. A verify runs the head over every row, so each token carries the row that chose it and a
//! logprob request scores a drafted answer exactly as it scores a plain one.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use sparkflash_model::{
    BackendError, Capabilities, DepthPolicy, FusedArm, FusedEngine, ModelBackend, ModelDescriptor,
    ModelSession, PackIdentity, Queue, Sampling, Speculation, Speculator, StepOutput, verdict,
};
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_weights::tp::{BitGatherHandle, SingleRank, TpLayout, TpRuntime};

use crate::config::Glm5NextConfig;
use crate::device::{CudaHandle, DeviceRank, DeviceSequence, LatentCache};

/// How a session drafts, once the backend has been told to.
#[derive(Debug, Clone)]
struct SpecConfig {
    /// The most drafts a step verifies.
    drafts: usize,
    /// The depth policy, when the drafter says how sure it is of each draft (the block drafter).
    policy: Option<DepthPolicy>,
}

/// The backend's rank and the collectives every session's step runs through.
struct Shared {
    rank: Mutex<DeviceRank>,
    tp: TpRuntime,
    gather: BitGatherHandle,
    eos: Vec<u32>,
    /// Set once, before the first session opens: a session's sequence is a drafting one or it is
    /// not, so the choice cannot change under a live session.
    spec: OnceLock<SpecConfig>,
}

/// The arm that steps GLM slots together, behind [sparkflash_model::FusedEngine].
///
/// A slot is a whole [Glm5NextSession] minus its rank: the sequence, the sampler and the history.
/// The engine's thread is the only stepper, so the rank lock is taken once a step, and a batch of
/// members is one [DeviceRank::forward_batch_greedy] call: the wide stages read the weights once
/// for every member's row.
pub struct GlmFusion {
    shared: Arc<Shared>,
    context_limit: usize,
}

/// The TP=1 device backend: one GLM-5.3-Flash rank on this host's GPU.
pub struct Glm5NextDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
    context_limit: usize,
    /// Set by [Glm5NextDeviceBackend::enable_fusion]: sessions become handles on this engine.
    fused: Option<Arc<FusedEngine<GlmFusion>>>,
}

impl GlmFusion {
    fn slot(&self, spec: Option<&SpecConfig>) -> Result<Glm5NextSession, BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let sequence = match spec {
            Some(_) if rank.has_block_drafter() => {
                rank.new_block_drafting_sequence(self.context_limit)?
            }
            Some(_) => rank.new_drafting_sequence(self.context_limit)?,
            None => rank.new_sequence(self.context_limit)?,
        };
        Ok(Glm5NextSession {
            shared: Arc::clone(&self.shared),
            sequence,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            spec: spec.map(|spec| Speculator::new(spec.drafts, spec.policy.clone())),
            pending: Queue::new(),
            confidence: Vec::new(),
            consumed: Vec::new(),
            prefill_cached: 0,
        })
    }
}

impl FusedArm for GlmFusion {
    type Slot = Glm5NextSession;

    fn open_slot(&self) -> Result<Glm5NextSession, BackendError> {
        // A fused batch and a drafter are different trades on the same step, and this arm runs
        // one of them: [Glm5NextDeviceBackend::enable_fusion] refuses to combine them.
        self.slot(None)
    }

    fn holds_request(&self, slot: &Glm5NextSession) -> bool {
        !slot.consumed.is_empty() || slot.sequence.position() != 0
    }

    fn renew(&self, slot: &mut Glm5NextSession) -> Result<(), BackendError> {
        *slot = self.slot(None)?;
        Ok(())
    }

    fn set_sampling(&self, slot: &mut Glm5NextSession, sampling: Sampling) {
        slot.set_sampling(sampling);
    }

    fn prefill(
        &self,
        slot: &mut Glm5NextSession,
        tokens: &[u32],
    ) -> Result<StepOutput, BackendError> {
        slot.prefill(tokens)
    }

    fn step(
        &self,
        members: &mut [(Glm5NextSession, u32)],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let rows: Vec<[u32; 1]> = members.iter().map(|(_, token)| [*token]).collect();
        {
            let mut batch: Vec<(&mut DeviceSequence, &[u32])> = members
                .iter_mut()
                .zip(&rows)
                .map(|((slot, _), row)| (&mut slot.sequence, row.as_slice()))
                .collect();
            rank.forward_batch_greedy(&self.shared.tp, &self.shared.gather, &mut batch)?;
        }
        let mut steps = Vec::with_capacity(members.len());
        for ((slot, token), row) in members.iter_mut().zip(&rows) {
            let _ = row;
            slot.sampler.record(*token);
            slot.consumed.push(*token);
            let logits = rank.last_logits(&slot.sequence)?;
            let chosen = slot.sampler.choose(&logits, &slot.sampling);
            let next_token = if self.shared.eos.contains(&chosen) {
                None
            } else {
                Some(chosen)
            };
            steps.push(StepOutput {
                logits,
                next_token,
                draft: Vec::new(),
            });
        }
        Ok(steps)
    }

    fn speculation(&self, slot: &Glm5NextSession) -> Speculation {
        slot.speculation()
    }

    fn max_batch(&self) -> usize {
        // A step is one row a member and the rank's batch is `MAX_ROWS` rows wide.
        crate::device::MAX_ROWS
    }
}

impl std::fmt::Debug for Glm5NextDeviceBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Glm5NextDeviceBackend")
            .field("descriptor", &self.descriptor.id)
            .field("context_limit", &self.context_limit)
            .finish_non_exhaustive()
    }
}

impl Glm5NextDeviceBackend {
    /// Open one TP=1 rank from a checkpoint and upload every resident weight.
    ///
    /// # Errors
    /// When the context is outside the model's window, the checkpoint cannot be read, the
    /// configuration is one the kernels refuse, or the device refuses an allocation or a copy.
    pub fn open_single_rank(
        cuda: Arc<CudaHandle>,
        config: Glm5NextConfig,
        checkpoint: &Path,
        context_limit: usize,
    ) -> Result<Self, BackendError> {
        let layout = TpLayout {
            rank: 0,
            world_size: 1,
        };
        let tp = TpRuntime::new(layout, Box::new(SingleRank)).map_err(|error| {
            BackendError::new("glm5_device", format!("single-rank collective: {error}"))
        })?;
        let gather = BitGatherHandle::new(Box::new(SingleRank));
        Self::open_with_cache(
            cuda,
            config,
            checkpoint,
            tp,
            gather,
            context_limit,
            crate::single_rank_latent(),
        )
    }

    /// [Self::open_single_rank] with an explicit latent-cache representation.
    ///
    /// The single-node arm always passes [crate::single_rank_latent] (the F32 latent), which
    /// is the representation [crate::single_rank_kv_plan] admits. A caller that wants another
    /// cache takes responsibility for admitting it.
    ///
    /// # Errors
    /// As [Self::open_single_rank].
    pub fn open_single_rank_with_cache(
        cuda: Arc<CudaHandle>,
        config: Glm5NextConfig,
        checkpoint: &Path,
        context_limit: usize,
        cache: LatentCache,
    ) -> Result<Self, BackendError> {
        let layout = TpLayout {
            rank: 0,
            world_size: 1,
        };
        let tp = TpRuntime::new(layout, Box::new(SingleRank)).map_err(|error| {
            BackendError::new("glm5_device", format!("single-rank collective: {error}"))
        })?;
        let gather = BitGatherHandle::new(Box::new(SingleRank));
        Self::open_with_cache(cuda, config, checkpoint, tp, gather, context_limit, cache)
    }

    /// [Self::open_single_rank] from an already-built execution plan.
    ///
    /// The plan's architecture replaces the typed configuration's layer schedule, so the weight
    /// walk reads the same layer selection the boot resolver checked. The typed config still owns
    /// the tensor shapes and weight names.
    ///
    /// # Errors
    /// As [Self::open_single_rank], and a plan of another family or layer count.
    pub fn open_single_rank_with_plan(
        cuda: Arc<CudaHandle>,
        mut config: Glm5NextConfig,
        checkpoint: &Path,
        context_limit: usize,
        plan: &EnginePlan,
    ) -> Result<Self, BackendError> {
        if plan.family() != "glm5_next" {
            return Err(BackendError::new(
                "glm5_plan",
                format!(
                    "the execution plan is {}, the GLM arm is glm5_next",
                    plan.family()
                ),
            ));
        }
        if plan.layers().len() != config.num_hidden_layers {
            return Err(BackendError::new(
                "glm5_plan",
                format!(
                    "the execution plan has {} layers, the GLM configuration has {}",
                    plan.layers().len(),
                    config.num_hidden_layers
                ),
            ));
        }
        config.architecture = plan.architecture().clone();
        Self::open_single_rank(cuda, config, checkpoint, context_limit)
    }

    /// Open one rank from a checkpoint over a caller-owned tensor-parallel group.
    ///
    /// `layout` is this rank's place in the group; `tp` and `gather` are the collectives that
    /// place it there (the CLI's TCP seams, or [SingleRank] for one host).
    ///
    /// # Errors
    /// As [Self::open_single_rank], plus a collective whose rank or world disagrees with `layout`.
    pub fn open(
        cuda: Arc<CudaHandle>,
        config: Glm5NextConfig,
        checkpoint: &Path,
        layout: TpLayout,
        tp: TpRuntime,
        gather: BitGatherHandle,
        context_limit: usize,
    ) -> Result<Self, BackendError> {
        if layout.rank != tp.rank() || layout.world_size != tp.world() {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the collective is rank {} of {}, but the layout is rank {} of {}",
                    tp.rank(),
                    tp.world(),
                    layout.rank,
                    layout.world_size
                ),
            ));
        }
        Self::open_with_cache(
            cuda,
            config,
            checkpoint,
            tp,
            gather,
            context_limit,
            LatentCache::Auto,
        )
    }

    /// [Self::open] with an explicit latent-cache representation.
    ///
    /// The rank layout is the collective's: [Self::from_rank] refuses a mismatch.
    fn open_with_cache(
        cuda: Arc<CudaHandle>,
        config: Glm5NextConfig,
        checkpoint: &Path,
        tp: TpRuntime,
        gather: BitGatherHandle,
        context_limit: usize,
        cache: LatentCache,
    ) -> Result<Self, BackendError> {
        let source = SafeTensorsSource::open(checkpoint, &checkpoint.join(INDEX_NAME))
            .map_err(|error| BackendError::new("glm5_weights", error.to_string()))?;
        let layout = TpLayout {
            rank: tp.rank(),
            world_size: tp.world(),
        };
        let rank = DeviceRank::load_with_cache(cuda, &config, &source, layout, cache)?;
        Self::from_rank(config, rank, tp, gather, context_limit)
    }

    /// [Self::open] over a converted rank file instead of the checkpoint.
    ///
    /// # Errors
    /// As [Self::open].
    pub fn open_converted(
        cuda: Arc<CudaHandle>,
        config: Glm5NextConfig,
        rank_file: &Path,
        layout: TpLayout,
        tp: TpRuntime,
        gather: BitGatherHandle,
        context_limit: usize,
    ) -> Result<Self, BackendError> {
        let rank = DeviceRank::load_converted(cuda, &config, rank_file, layout)?;
        Self::from_rank(config, rank, tp, gather, context_limit)
    }

    fn from_rank(
        config: Glm5NextConfig,
        rank: DeviceRank,
        tp: TpRuntime,
        gather: BitGatherHandle,
        context_limit: usize,
    ) -> Result<Self, BackendError> {
        if context_limit == 0 || context_limit > config.max_position_embeddings {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "context limit {context_limit} is outside 1..={}",
                    config.max_position_embeddings
                ),
            ));
        }
        if tp.rank() != rank.layout().rank || tp.world() != rank.layout().world_size {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the collective is rank {} of {}, but the rank loaded rank {} of {}",
                    tp.rank(),
                    tp.world(),
                    rank.layout().rank,
                    rank.layout().world_size
                ),
            ));
        }
        let descriptor = descriptor_for(&config, context_limit);
        Ok(Self {
            descriptor,
            shared: Arc::new(Shared {
                rank: Mutex::new(rank),
                tp,
                gather,
                eos: config.eos_token_ids.clone(),
                spec: OnceLock::new(),
            }),
            context_limit,
            fused: None,
        })
    }

    /// Step this rank's sessions together, one row each, as one pass over the weights.
    ///
    /// The host driver gives every request its own session, so without this N workers read the
    /// rank's 82 GB N times for N rows. With it each session is a handle on one engine thread,
    /// which batches whatever is waiting into a single `forward_batch_greedy`. Call it before
    /// the first session opens, and before any session drafts: this arm does one or the other.
    ///
    /// # Errors
    /// A slot the rank cannot allocate, a thread that cannot start, or a rank already told to
    /// draft.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        if self.shared.spec.get().is_some() {
            return Err(BackendError::new(
                "glm5_device",
                "a fused batch verifies no drafts; choose fusion or speculation",
            ));
        }
        let arm = Arc::new(GlmFusion {
            shared: Arc::clone(&self.shared),
            context_limit: self.context_limit,
        });
        self.fused = Some(FusedEngine::start(arm, capacity)?);
        Ok(())
    }

    /// Whether this backend steps its sessions together.
    #[must_use]
    pub fn fused(&self) -> bool {
        self.fused.is_some()
    }

    /// Draft with this rank's MTP layer, verifying at most `drafts` tokens a step.
    ///
    /// The device arm has drafted since it was written ([DeviceRank::draft]); this is the seam
    /// that lets a *served* request use it, which is what turns a 34 ms step into two or three
    /// tokens. Call it before the first session opens: a sequence is made a drafting one at
    /// [ModelBackend::open_session].
    ///
    /// # Errors
    /// When the rank has no MTP layer.
    pub fn enable_speculation(
        &mut self,
        drafts: usize,
        policy: Option<DepthPolicy>,
    ) -> Result<(), BackendError> {
        if drafts == 0 {
            return Ok(());
        }
        if self.fused.is_some() {
            return Err(BackendError::new(
                "glm5_device",
                "a fused batch verifies no drafts; choose fusion or speculation",
            ));
        }
        let has_mtp = {
            let rank = self
                .shared
                .rank
                .lock()
                .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
            rank.has_block_drafter() || rank.has_mtp()
        };
        if !has_mtp {
            return Err(BackendError::new(
                "glm5_device",
                "drafting needs a drafter: this rank has neither an MTP layer nor a loaded block \
                 drafter",
            ));
        }
        self.shared
            .spec
            .set(SpecConfig { drafts, policy })
            .map_err(|_| {
                BackendError::new(
                    "glm5_device",
                    "speculation was already configured for this rank",
                )
            })?;
        Ok(())
    }

    /// Load the DFlash 2 block drafter in `directory`, for [Self::enable_speculation].
    ///
    /// # Errors
    /// As [DeviceRank::load_block_drafter].
    pub fn load_block_drafter(&mut self, directory: &Path) -> Result<(), BackendError> {
        self.shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?
            .load_block_drafter(directory)
    }

    /// The context window this backend allocates for one sequence.
    #[must_use]
    pub fn context_limit(&self) -> usize {
        self.context_limit
    }
}

impl ModelBackend for Glm5NextDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> Capabilities {
        // The text arm renders the tool catalogue, the `<think>` block and the reasoning-effort
        // instruction; the vision tower and the drafters are not loaded.
        Capabilities::ALL
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        if let Some(engine) = &self.fused {
            return engine
                .open()
                .map(|session| Box::new(session) as Box<dyn ModelSession>);
        }
        let spec = self.shared.spec.get().cloned();
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let sequence = match spec.as_ref() {
            Some(_) if rank.has_block_drafter() => {
                rank.new_block_drafting_sequence(self.context_limit)?
            }
            Some(_) => rank.new_drafting_sequence(self.context_limit)?,
            None => rank.new_sequence(self.context_limit)?,
        };
        Ok(Box::new(Glm5NextSession {
            shared: Arc::clone(&self.shared),
            sequence,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            spec: spec
                .as_ref()
                .map(|spec| Speculator::new(spec.drafts, spec.policy.clone())),
            pending: Queue::new(),
            confidence: Vec::new(),
            consumed: Vec::new(),
            prefill_cached: 0,
        }))
    }
}

/// One request's device state.
///
/// The sequence is the rank's cache and position; this type is the request around it. A serving
/// adapter keeps one session per worker between requests and hands it the next prompt, which may
/// extend the tokens the sequence already forwarded ([ModelSession::reusable_prefix]). The
/// history it was given is kept beside the caches, not inside them, so the arm can report what it
/// holds without the device arm knowing about serving.
pub struct Glm5NextSession {
    shared: Arc<Shared>,
    sequence: DeviceSequence,
    sampler: Sampler,
    sampling: Sampling,
    /// The draft depth and its accounting, when this sequence drafts.
    spec: Option<Speculator>,
    /// Tokens a verify already produced, oldest first, each with the token that follows it. The
    /// adapter feeds them back one at a time; the model has consumed them already.
    pending: Queue,
    /// The drafter's confidence in each of the last round's drafts.
    confidence: Vec<f32>,
    /// Every token this sequence has forwarded, oldest first: every prefill's fresh tokens and
    /// every decode's. A request that extends this list forwards only what it adds.
    consumed: Vec<u32>,
    /// How many of the last prefill's tokens its caches already held, for the request's usage.
    prefill_cached: usize,
}

impl std::fmt::Debug for Glm5NextSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Glm5NextSession")
            .field("position", &self.sequence.position())
            .field("temperature", &self.sampling.temperature)
            .finish_non_exhaustive()
    }
}

impl Glm5NextSession {
    /// Run `rows` through this rank and sample the token after the last of them.
    ///
    /// The greedy path leaves each row's pick on the device and the last kept row's logits behind
    /// it, so a step reads back one vocabulary row, not one a row.
    fn step(&mut self, rows: &[u32]) -> Result<StepOutput, BackendError> {
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let mut batch = [(&mut self.sequence, rows)];
        rank.forward_batch_greedy(&self.shared.tp, &self.shared.gather, &mut batch)?;
        let logits = rank.last_logits(&self.sequence)?;
        let token = self.sampler.choose(&logits, &self.sampling);
        let next_token = if self.shared.eos.contains(&token) {
            None
        } else {
            Some(token)
        };
        Ok(StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        })
    }

    /// One speculative round: draft after `anchor`, verify the chain, keep what the model chose.
    ///
    /// This is the arm's whole speculative path. The rule that decides how much of the chain
    /// survives is [sparkflash_model::verdict] and the depth and the counters are
    /// [Speculator]'s, so this function only sequences the three device calls
    /// ([DeviceRank::draft], [DeviceRank::verify_batch], [DeviceRank::commit]) and hands back the
    /// rows the later `decode` calls will return.
    ///
    /// A verify runs the head over every row, so each row's logits are already on the host: a
    /// token the round produces carries the row that chose it, and a logprob request scores the
    /// drafted answer exactly as it would a plain one.
    fn round(&mut self, anchor: u32) -> Result<StepOutput, BackendError> {
        if self.spec.is_none() {
            return self.step(&[anchor]);
        }
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        // A draft writes the MTP caches past what the sequence keeps, so a chain longer than the
        // room left would draft past the window.
        let room = self
            .sequence
            .capacity()
            .saturating_sub(self.sequence.position() + 1);
        let confidence = self.confidence.clone();
        let depth = self
            .spec
            .as_ref()
            .map_or(0, |spec| spec.depth(&confidence, room));
        let drafts = if depth == 0 {
            Vec::new()
        } else {
            let drafted = {
                let mut batch = [(&mut self.sequence, anchor)];
                rank.draft(&self.shared.tp, &self.shared.gather, &mut batch, depth)?
            };
            drafted.into_iter().next().unwrap_or_default()
        };
        self.confidence = self.sequence.draft_confidence().to_vec();
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        let rows = {
            let mut batch = [(&mut self.sequence, chunk.as_slice())];
            rank.verify_batch(&self.shared.tp, &self.shared.gather, &mut batch)?
        };
        let rows = rows.into_iter().next().unwrap_or_default();
        let picks: Vec<u32> = rows
            .iter()
            .map(|row| self.sampler.choose(row, &self.sampling))
            .collect();
        let kept = verdict(&chunk, &picks);
        if chunk.len() > 1 {
            // Give back the rows the model did not keep, before this rank's next step.
            rank.commit(&mut self.sequence, kept)?;
        }
        if let Some(spec) = self.spec.as_mut() {
            spec.observe(&confidence, chunk.len() - 1, kept);
        }
        let eos = &self.shared.eos;
        self.pending.extend_round(&chunk, &picks, &rows, kept, eos);
        let next_token = picks.first().copied().filter(|token| !eos.contains(token));
        Ok(StepOutput {
            logits: rows.first().cloned().unwrap_or_default(),
            next_token,
            draft: Vec::new(),
        })
    }
}

impl ModelSession for Glm5NextSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row this step
        // already copies back, so a non-greedy request is served rather than approximated.
        self.sampling = sampling;
        self.sampler.reset();
        // The adapter calls this once per request, and a kept sequence carries the last one's
        // counts: a request reports its own speculation, not the session's lifetime.
        if let Some(spec) = self.spec.as_mut() {
            spec.reset();
        }
        // A non-greedy request runs a token a step: its answer is a draw, and the drafts were
        // scored by the same rule the verify uses - but the sampler's history is what a draw
        // depends on, and a verify advances it a row at a time. Quiet, exact, and one token a
        // step is what the arm documented before it drafted.
        self.confidence.clear();
        self.pending.clear();
    }

    /// The tokens this sequence has forwarded, while its caches end exactly where they end.
    ///
    /// A sequence whose caches are ahead of the tokens the caller knows cannot honour a prefix it
    /// would have to rewind to, so it reports nothing and the next request opens a fresh one.
    fn reusable_prefix(&self) -> Option<&[u32]> {
        (self.pending.is_empty() && self.sequence.position() == self.consumed.len())
            .then_some(self.consumed.as_slice())
    }

    fn cached_tokens(&self) -> usize {
        self.prefill_cached
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "glm5_device",
                "an empty prompt has no logits",
            ));
        }
        // A sequence the serving adapter kept may already hold a prefix of this prompt, in which
        // case only the tokens past it are forwarded. Its sampler history is the whole prompt
        // either way: the penalties count every token the request reads.
        let reused = self
            .consumed
            .iter()
            .zip(tokens)
            .take_while(|(held, given)| held == given)
            .count();
        if reused != self.consumed.len() || reused >= tokens.len() {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "a prefill must extend the {} token(s) this sequence holds by at least one; \
                     this one does not",
                    self.consumed.len()
                ),
            ));
        }
        if self.sequence.position() != self.consumed.len() {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the caches hold {} position(s) but the history has {} token(s)",
                    self.sequence.position(),
                    self.consumed.len()
                ),
            ));
        }
        self.prefill_cached = reused;
        // A new prompt's tokens replace any chain the last one left queued.
        self.pending.clear();
        self.confidence.clear();
        self.consumed.extend_from_slice(&tokens[reused..]);
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let logits = rank.forward_prefill(
            &self.shared.tp,
            &self.shared.gather,
            &mut self.sequence,
            &tokens[reused..],
        )?;
        let token = self.sampler.choose(&logits, &self.sampling);
        let next_token = if self.shared.eos.contains(&token) {
            None
        } else {
            Some(token)
        };
        Ok(StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.sampler.record(token);
        self.consumed.push(token);
        // A verify already consumed this token and produced the answer: hand back the row it
        // sampled, exactly as the one-token step it stands in for would have.
        if let Some(front) = self.pending.front() {
            let expected = front.token;
            let Some(next) = self.pending.pop(token) else {
                return Err(BackendError::new(
                    "glm5_device",
                    format!(
                        "the engine fed token {token} where the verify produced {expected}; a \
                         check that did not accept the drafts must be re-drafted"
                    ),
                ));
            };
            return Ok(StepOutput {
                logits: next.logits,
                next_token: next.next_token,
                draft: Vec::new(),
            });
        }
        self.round(token)
    }

    fn speculation(&self) -> Speculation {
        self.spec
            .as_ref()
            .map_or_else(Speculation::default, Speculator::stats)
    }
}

/// This pack's identity in the registry.
pub const IDENTITY: PackIdentity = PackIdentity {
    model_id: crate::MODEL_ID,
    family: crate::FAMILY,
    model_type: "glm5_next",
    label: "GLM",
    weights_id: crate::WEIGHTS_ID,
    max_output_tokens: crate::MAX_OUTPUT_TOKENS,
};

/// What this arm can serve of `config` once loaded.
///
/// The text channels are the codec's: the tool catalogue, the ` thinking` block and the
/// reasoning-effort instruction are all rendered. The vision tower is in the checkpoint but is not
/// loaded, so the arm does not claim the channel ([sparkflash_model::Capabilities] has no way to
/// over-promise by omission).
#[must_use]
pub fn capabilities(_config: &Glm5NextConfig) -> Capabilities {
    Capabilities::ALL
}

/// The descriptor one rank serves: the pack's id, this build's output ceiling, the context the
/// sequence actually allocates, and the vocabulary and window the released document names
/// ([sparkflash_model::PackIdentity]).
pub fn descriptor_for(config: &Glm5NextConfig, context: usize) -> ModelDescriptor {
    IDENTITY.descriptor(&config.architecture, context, capabilities(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The backend is a factory the serving layer shares, and a session moves between workers.
    #[test]
    fn the_backend_and_a_session_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_send<T: Send>() {}
        assert_send_sync::<Glm5NextDeviceBackend>();
        assert_send::<Glm5NextSession>();
    }

    #[test]
    fn the_descriptor_reports_the_served_context_and_vocabulary() {
        let config = Glm5NextConfig::tiny();
        let descriptor = descriptor_for(&config, 64);
        assert_eq!(descriptor.id, crate::MODEL_ID);
        assert_eq!(descriptor.weights_id, crate::WEIGHTS_ID);
        assert_eq!(descriptor.context_tokens, 64);
        assert_eq!(descriptor.vocab_size, config.vocab_size);
        assert_eq!(descriptor.max_output_tokens, crate::MAX_OUTPUT_TOKENS);
        assert!(descriptor.capabilities.tools);
        assert!(!descriptor.capabilities.vision);
    }
}
