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

//! The Dory host backend: the reference forward behind the serving driver.
//!
//! This is the CPU path a checkpoint uses when this build has no Dory/Qwen cubin. It loads the
//! BF16 checkpoint once through [sparkflash_weights::WeightSource], then runs one token at a time
//! with the looped schedule and the normalized residual arithmetic in [crate::forward].

use std::sync::Arc;

use sparkflash_model::ModelDescriptor;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::WeightSource;

use crate::config::DoryConfig;
use crate::forward::{DoryState, DoryWeights};

/// A Dory model running on the host reference.
pub struct DoryHostBackend {
    descriptor: ModelDescriptor,
    weights: Arc<DoryWeights>,
    context: usize,
}

impl DoryHostBackend {
    /// Load the checkpoint and open the backend.
    ///
    /// # Errors
    /// A tensor the configuration needs is missing or malformed, or the context is zero.
    pub fn open(
        mut descriptor: ModelDescriptor,
        config: DoryConfig,
        source: &dyn WeightSource,
        context: usize,
    ) -> Result<Self, String> {
        if context == 0 {
            return Err("the served context must be positive".to_owned());
        }
        if context > config.max_position_embeddings {
            return Err(format!(
                "context {context} is past the checkpoint window {}",
                config.max_position_embeddings
            ));
        }
        crate::device::check_profile()?;
        let weights = DoryWeights::load(&config, source)?;
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
            weights: Arc::new(weights),
            context,
        })
    }
}

impl ModelBackend for DoryHostBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(DorySession {
            weights: Arc::clone(&self.weights),
            state: DoryState::new(&self.weights.config),
            context: self.context,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        }))
    }
}

struct DorySession {
    weights: Arc<DoryWeights>,
    state: DoryState,
    context: usize,
    sampler: Sampler,
    sampling: Sampling,
}

impl DorySession {
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        if self.state.position >= self.context {
            return Err(BackendError::new(
                "dory_context",
                format!("the sequence is past the served context {}", self.context),
            ));
        }
        let position = self.state.position;
        let logits = self
            .weights
            .forward_token(token, position, &mut self.state)
            .map_err(|error| BackendError::new("dory_host", error))?;
        Ok(logits)
    }
}

impl ModelSession for DorySession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p are host arithmetic over the row the
        // forward already produced, so the host reference serves the same request as the device
        // arm rather than silently returning a greedy token.
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "dory_host",
                "the host path needs at least one prompt token",
            ));
        }
        let mut logits = Vec::new();
        for token in tokens {
            logits = self.forward(*token)?;
        }
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: self.weights.config.next_token(chosen),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let logits = self.forward(token)?;
        self.sampler.record(token);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: self.weights.config.next_token(chosen),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        // The host path forwards every prompt from a fresh state; it does not implement a
        // prefix-continuation seam yet.
        None
    }
}
