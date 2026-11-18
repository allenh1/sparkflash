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

//! The SenseNova-Loop host backend: the reference forward behind the serving driver.
//!
//! The CPU path loads the text backbone once and runs one token at a time through the released
//! 56-visit plan with per-visit caches. It is the correctness path; the device arm is the
//! performance path.

use std::sync::Arc;

use sparkflash_model::ModelDescriptor;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::WeightSource;

use crate::config::SenseNovaConfig;
use crate::forward::{SenseNovaState, SenseNovaWeights};

/// A SenseNova-Loop model running on the host reference.
pub struct SenseNovaHostBackend {
    descriptor: ModelDescriptor,
    weights: Arc<SenseNovaWeights>,
    context: usize,
}

impl SenseNovaHostBackend {
    /// Load the checkpoint and open the backend.
    ///
    /// # Errors
    /// A tensor the configuration needs is missing or malformed, or the context is invalid.
    pub fn open(
        mut descriptor: ModelDescriptor,
        config: SenseNovaConfig,
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
        let weights = SenseNovaWeights::load(&config, source)?;
        descriptor.context_tokens = context;
        Ok(Self {
            descriptor,
            weights: Arc::new(weights),
            context,
        })
    }

    /// The loaded weights, for a parity harness.
    #[must_use]
    pub fn weights(&self) -> &SenseNovaWeights {
        &self.weights
    }
}

impl ModelBackend for SenseNovaHostBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(SenseNovaSession {
            weights: Arc::clone(&self.weights),
            state: SenseNovaState::new(&self.weights.config),
            context: self.context,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        }))
    }
}

struct SenseNovaSession {
    weights: Arc<SenseNovaWeights>,
    state: SenseNovaState,
    context: usize,
    sampler: Sampler,
    sampling: Sampling,
}

impl SenseNovaSession {
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        if self.state.position >= self.context {
            return Err(BackendError::new(
                "sensenova_context",
                format!("the sequence is past the served context {}", self.context),
            ));
        }
        let position = self.state.position;
        self.weights
            .forward_token(token, position, &mut self.state)
            .map_err(|error| BackendError::new("sensenova_host", error))
    }
}

impl ModelSession for SenseNovaSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "sensenova_host",
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
            next_token: Some(chosen as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let logits = self.forward(token)?;
        self.sampler.record(token);
        let chosen = self.sampler.choose(&logits, &self.sampling);
        Ok(StepOutput {
            next_token: Some(chosen as u32),
            logits,
            draft: Vec::new(),
        })
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        None
    }
}
