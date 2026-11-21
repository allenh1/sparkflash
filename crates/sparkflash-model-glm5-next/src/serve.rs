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

//! The serving backend: the device rank behind [ModelBackend], one fused session a request.
//!
//! The rank is the resident half - its weights, kernels and scratch. The shared runtime owns
//! everything around it: the lanes, the prompt parts, the history, resume and rewind, the ledger
//! and the round path. This file opens the rank and states what the runtime serves it by
//! ([WalkArm]: [crate::walk::GlmWalk] and the group's wire); the runtime starts the tick engine,
//! alone or as rank 0 of a tensor-parallel group ([sparkflash_runtime::serve]). A session is a handle on one engine thread, where the sessions of a step are one batch over the
//! rank and the weights are read once for every member's row.
//!
//! A session drafts once [Glm5NextDeviceBackend::enable_speculation] has been called:
//! [crate::walk::GlmDrafter] proposes a chain after the token it was handed
//! ([DeviceRank::draft]), verifies the chain and the token in one step
//! ([DeviceRank::verify_batch]), keeps the leading drafts the model would have picked
//! ([sparkflash_model::verdict]) and queues every token the step produced for the calls that feed
//! them. The draft depth and its accounting are [sparkflash_model::Speculator]'s, so the same
//! rule and the same counters serve every drafting arm. A verify runs the head over every row, so
//! each token carries the row that chose it and a logprob request scores a drafted answer exactly
//! as it scores a plain one.

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker};
use sparkflash_model::{
    BackendError, Capabilities, DepthPolicy, ModelBackend, ModelDescriptor, ModelSession,
    PackIdentity,
};
use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_runtime::{GroupEngine, GroupWire, Place, Serving, TextWire, Walk, WalkArm};
use sparkflash_weights::WeightSource;
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_weights::tp::{BitGatherHandle, SingleRank, TpLayout, TpRuntime};

use crate::config::Glm5NextConfig;
use crate::device::{CudaHandle, DeviceRank, LatentCache};
use crate::walk::{GlmWalk, Shared, SpecConfig};

/// The device backend: one GLM-5.3-Flash rank on this host's GPU, alone or one of a
/// tensor-parallel group.
pub struct Glm5NextDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
    context_limit: usize,
    /// Set by [Glm5NextDeviceBackend::enable_fusion] or [Glm5NextDeviceBackend::enable_group_fusion]:
    /// sessions become handles on this engine.
    fused: Option<Serving>,
}

/// What the runtime serves this backend by ([sparkflash_runtime::serve] and its group forms):
/// the tick engine, DeepSeek's scheduler, over the GLM walk. It is rank 0 of a tensor-parallel
/// group when the backend was opened as one, and alone otherwise; the frames differ, the engine
/// does not.
impl WalkArm for Glm5NextDeviceBackend {
    fn walk(&self, _capacity: usize) -> Result<impl Walk<Seq: Send> + Send + Sync, BackendError> {
        GlmWalk::new(Arc::clone(&self.shared), self.context_limit)
    }

    /// The arm serves text.
    fn wire(&self) -> impl GroupWire {
        TextWire::new("glm", "GLM", self.context_limit)
    }

    /// The backend holds a collective alone and in a group, so it refuses neither place.
    fn admit(&self, _place: Place) -> Result<(), BackendError> {
        Ok(())
    }

    fn group_engine(&self) -> GroupEngine {
        GroupEngine::Tick
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

    /// The backend over a rank the caller has already loaded, with the collectives that place it
    /// in its group: the four-rank serve path loads its rank once (converted file, latent cache,
    /// drafter) and serves it through this backend's sessions.
    ///
    /// # Errors
    /// A context outside the model's window, or a collective whose rank or world disagrees with
    /// the rank's layout.
    pub fn from_rank(
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

    /// Step this rank's sessions together, one row each, as one pass over the weights
    /// ([sparkflash_runtime::serve]).
    ///
    /// The host driver gives every request its own session, so without this N workers read the
    /// rank's 82 GB N times for N rows. With it each session is a handle on the tick engine,
    /// whose config is GLM's profile: the 4,096-row prompt chunk, the verify width as the member
    /// cap, no decoder-layer skip and no row exchange. Call it before the first session opens
    /// and after [Self::enable_speculation], if the sessions draft: then their rounds verify
    /// together.
    ///
    /// # Errors
    /// A slot the rank cannot allocate, a thread that cannot start, or a
    /// [sparkflash_runtime::JOINT_ROUNDS_ENV] that is neither on nor off.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        self.fused = Some(sparkflash_runtime::serve(self, capacity)?);
        Ok(())
    }

    /// [Self::enable_fusion] on rank 0 of a group ([sparkflash_runtime::serve_group]): rank 0's
    /// decisions are broadcast through `hub` and each worker replays them
    /// ([Self::mirror_group]). The row exchange is on for the group, so a prompt takes the
    /// 4,096-row bulk chunks its profile states.
    ///
    /// # Errors
    /// As [Self::enable_fusion], or a watch that cannot start.
    pub fn enable_group_fusion(
        &mut self,
        capacity: usize,
        hub: Tp4ControlHub,
    ) -> Result<(), BackendError> {
        self.fused = Some(sparkflash_runtime::serve_group(self, capacity, hub)?);
        Ok(())
    }

    /// A worker of a group: replays rank 0's tick frames over `capacity` lanes of this rank's own
    /// until the group can no longer serve, then ends the process ([sparkflash_runtime::mirror]).
    /// Configure speculation first, as rank 0 does.
    ///
    /// # Errors
    /// The arm's configuration, before any frame.
    pub fn mirror_group(
        &self,
        control: &mut Tp4ControlWorker,
        capacity: usize,
    ) -> Result<std::convert::Infallible, String> {
        sparkflash_runtime::mirror(self, control, capacity)
    }

    /// The lanes a group opens when its operator gave no `--concurrency`: the member width this
    /// rank's walk states ([sparkflash_runtime::lane_limit]).
    ///
    /// # Errors
    /// The runtime refuses this rank's walk.
    pub fn lane_limit(&self) -> Result<usize, BackendError> {
        sparkflash_runtime::lane_limit(self)
    }

    /// The same rank as a bare walk, for the generated conformance checks: they open the
    /// runtime themselves so a check can drive a prompt, a continued turn and a rewind.
    ///
    /// # Errors
    /// A poisoned rank lock, or a rank whose cache plan cannot be read.
    #[cfg(test)]
    pub(crate) fn walk_arm(&self) -> Result<GlmWalk, BackendError> {
        GlmWalk::new(Arc::clone(&self.shared), self.context_limit)
    }

    /// Whether this backend steps its sessions together.
    #[must_use]
    pub fn fused(&self) -> bool {
        self.fused.is_some()
    }

    /// Whether this rank drafts with a block drafter, and the most drafts it proposes after a
    /// token ([DeviceRank::max_drafts]).
    ///
    /// # Errors
    /// A poisoned rank lock.
    pub fn drafting_limits(&self) -> Result<(bool, usize), BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        Ok((rank.has_block_drafter(), rank.max_drafts()))
    }

    /// Draft with this rank's MTP layer, verifying at most `drafts` tokens a step.
    ///
    /// The device arm has drafted since it was written ([DeviceRank::draft]); this is the seam
    /// that lets a *served* request use it, which is what turns a 34 ms step into two or three
    /// tokens. Call it before the first session opens: a sequence is made a drafting one at
    /// [ModelBackend::open_session].
    ///
    /// # Errors
    /// When the rank has no MTP layer, or its sessions are already fused.
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
                "speculation is configured before fusion: the fused slots are already open",
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

impl sparkflash_ops::EngineBuilder for Glm5NextDeviceBackend {
    type Backend = sparkflash_ops::PartsDeviceBackend<Self>;

    /// Open the GLM arm from the shared execution plan.
    ///
    /// The plan's own target and world gate the open ([EnginePlan::single_rank_context]), the
    /// shared rank view then validates the checkpoint and proves both passes before a weight is
    /// uploaded, and the arm opens the same plan for its numeric forward. The TP4 group opens
    /// its ranks through the pack's `OPEN_RANK` instead.
    ///
    /// # Errors
    /// A plan this build's single-rank arm cannot open, a weight the plan names and the source
    /// lacks, a part no catalogue kernel serves in either pass, a checkpoint that does not
    /// parse, or a device that does not open.
    fn open(
        plan: &EnginePlan,
        weights: &dyn WeightSource,
        context: usize,
    ) -> Result<Self::Backend, sparkflash_ops::PlanError> {
        let plan_context = plan.single_rank_context()?;
        // The serve path's admitted `--context`, capped by the checkpoint's own window. The arm
        // must allocate its caches and advertise its descriptor at this width; the plan's
        // `max_position_embeddings` is the checkpoint ceiling, not the operator's request.
        let context = plan.admitted_context(context)?;
        let rank = sparkflash_ops::PartsDeviceRank::open(plan, weights, plan_context)?;
        let directory = weights.directory().ok_or_else(|| {
            sparkflash_ops::PlanError::Invalid(
                "the GLM arm needs the checkpoint directory beside its weight source".to_owned(),
            )
        })?;
        let config = Glm5NextConfig::from_checkpoint_dir(directory)
            .map_err(|error| sparkflash_ops::PlanError::Invalid(error.to_string()))?;
        let cuda = sparkflash_kernels::cuda_driver::Cuda::load()
            .map_err(|error| sparkflash_ops::PlanError::Invalid(format!("CUDA: {error}")))?;
        let arm = Self::open_single_rank_with_plan(
            Arc::new(CudaHandle::new(cuda)),
            config,
            directory,
            context,
            plan,
        )
        .map_err(|error| sparkflash_ops::PlanError::Invalid(error.to_string()))?;
        Ok(sparkflash_ops::PartsDeviceBackend::new(rank, arm))
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

    /// A session on the fused runtime. The arm has no second session shape: the runtime owns
    /// the prompt parts, the history and the round, so a backend that has not been fused has no
    /// serving path (the CLI always fuses before it starts its driver).
    ///
    /// # Errors
    /// [Self::enable_fusion] was not called.
    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        self.fused.as_ref().map_or_else(
            || {
                Err(BackendError::new(
                    "glm5_device",
                    "the GLM arm serves on the shared runtime: call enable_fusion before opening                      a session",
                ))
            },
            Serving::open,
        )
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

    /// The backend is a factory the serving layer shares, and the runtime it fuses moves between
    /// workers.
    #[test]
    fn the_backend_and_the_walk_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Glm5NextDeviceBackend>();
        assert_send_sync::<sparkflash_runtime::Runtime<GlmWalk>>();
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
