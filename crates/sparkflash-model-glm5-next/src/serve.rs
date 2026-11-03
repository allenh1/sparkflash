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
//! copy a step rather than a second device kernel. Drafting is not wired into the session yet: the
//! MTP and DFlash 2 drafters remain available through [DeviceRank] and the `glm-tp4` harness, and
//! [ModelSession::speculation] reports the zero default until a session drafts.

use std::path::Path;
use std::sync::{Arc, Mutex};

use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, Sampling, Speculation,
    StepOutput,
};
use sparkflash_model_qwen4_exp::sampling::Sampler;
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_weights::tp::{BitGatherHandle, SingleRank, TpLayout, TpRuntime};

use crate::config::Glm5NextConfig;
use crate::device::{CudaHandle, DeviceRank, DeviceSequence};

/// The backend's rank and the collectives every session's step runs through.
struct Shared {
    rank: Mutex<DeviceRank>,
    tp: TpRuntime,
    gather: BitGatherHandle,
    eos: Vec<u32>,
}

/// The TP=1 device backend: one GLM-5.3-Flash rank on this host's GPU.
pub struct Glm5NextDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<Shared>,
    context_limit: usize,
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
        Self::open(cuda, config, checkpoint, layout, tp, gather, context_limit)
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
        let source = SafeTensorsSource::open(checkpoint, &checkpoint.join(INDEX_NAME))
            .map_err(|error| BackendError::new("glm5_weights", error.to_string()))?;
        let rank = DeviceRank::load(cuda, &config, &source, layout)?;
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
            }),
            context_limit,
        })
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
        Capabilities {
            tools: true,
            thinking: true,
            reasoning_effort: true,
            vision: false,
            constrained: false,
        }
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        let sequence = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?
            .new_sequence(self.context_limit)?;
        Ok(Box::new(Glm5NextSession {
            shared: Arc::clone(&self.shared),
            sequence,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            speculation: Speculation::default(),
        }))
    }
}

/// One request's device state.
pub struct Glm5NextSession {
    shared: Arc<Shared>,
    sequence: DeviceSequence,
    sampler: Sampler,
    sampling: Sampling,
    speculation: Speculation,
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
}

impl ModelSession for Glm5NextSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row this step
        // already copies back, so a non-greedy request is served rather than approximated.
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "glm5_device",
                "an empty prompt has no logits",
            ));
        }
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
            tokens,
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
        self.step(&[token])
    }

    fn speculation(&self) -> Speculation {
        self.speculation
    }
}

/// The descriptor one rank serves: the pack's id, this build's output ceiling, and the context the
/// sequence actually allocates.
fn descriptor_for(config: &Glm5NextConfig, context: usize) -> ModelDescriptor {
    ModelDescriptor {
        id: crate::MODEL_ID.to_owned(),
        family: crate::FAMILY.to_owned(),
        weights_id: crate::WEIGHTS_ID.to_owned(),
        context_tokens: context.min(config.max_position_embeddings),
        max_output_tokens: crate::MAX_OUTPUT_TOKENS,
        vocab_size: config.vocab_size,
        capabilities: Capabilities {
            tools: true,
            thinking: true,
            reasoning_effort: true,
            vision: false,
            constrained: false,
        },
    }
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
