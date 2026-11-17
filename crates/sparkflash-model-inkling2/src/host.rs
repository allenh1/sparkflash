// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small host backend.
//!
//! The f32 reference is the first serving path: a checkpoint whose device cubin has not been
//! built still answers, layer by layer, through the same arithmetic. It is intentionally not a
//! throughput claim.

use std::sync::Arc;

use sparkflash_model::gumbel::GumbelSampler;
use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, Sampling, StepOutput,
};
use sparkflash_ops::sampler::Sampler;
use sparkflash_weights::WeightSource;

use crate::config::Inkling2Config;
use crate::forward::{Inkling2State, Inkling2Weights};

/// A CPU reference backend over one Inkling checkpoint.
pub struct Inkling2HostBackend {
    descriptor: ModelDescriptor,
    config: Arc<Inkling2Config>,
    weights: Arc<Inkling2Weights>,
    context: usize,
}

impl std::fmt::Debug for Inkling2HostBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Inkling2HostBackend")
            .field("descriptor", &self.descriptor.id)
            .field("context", &self.context)
            .finish()
    }
}

impl Inkling2HostBackend {
    /// Open the reference over a checkpoint source.
    ///
    /// # Errors
    /// A missing tensor, a shape disagreement, or a context outside the model window.
    pub fn open(
        descriptor: ModelDescriptor,
        config: Inkling2Config,
        source: Arc<dyn WeightSource>,
        context: usize,
    ) -> Result<Self, String> {
        let window = config.context_tokens();
        if context == 0 || (window != 0 && context > window) {
            return Err(format!(
                "context {context} is outside 1..={window} for {}",
                descriptor.id
            ));
        }
        let weights = Inkling2Weights::load(config.clone(), source)?;
        Ok(Self {
            descriptor,
            config: Arc::new(config),
            weights: Arc::new(weights),
            context,
        })
    }

    /// The configuration this backend serves.
    #[must_use]
    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    /// The allocated context.
    #[must_use]
    pub const fn context(&self) -> usize {
        self.context
    }
}

impl ModelBackend for Inkling2HostBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> Capabilities {
        // The reference runs the text backbone. The tower and drafter are in the checkpoint but
        // the first host path does not execute either.
        Capabilities {
            vision: false,
            ..self.descriptor.capabilities
        }
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(Inkling2HostSession {
            weights: Arc::clone(&self.weights),
            state: Inkling2State::new(&self.config),
            context: self.context,
            history: Vec::new(),
            last_logits: Vec::new(),
            cached: 0,
            sampling: Sampling::default(),
            sampler: Sampler::new(),
        }))
    }
}

struct Inkling2HostSession {
    weights: Arc<Inkling2Weights>,
    state: Inkling2State,
    context: usize,
    history: Vec<u32>,
    last_logits: Vec<f32>,
    cached: usize,
    sampling: Sampling,
    sampler: Sampler,
}

impl ModelSession for Inkling2HostSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
        self.sampler.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new("request", "the prompt is empty"));
        }
        // Continue this session only when the prompt extends exactly the history it holds; a
        // different prompt starts a fresh state. This is what makes reusable_prefix honest.
        let start = if tokens.starts_with(&self.history) {
            self.history.len()
        } else {
            self.state = Inkling2State::new(self.weights.config());
            self.history.clear();
            self.last_logits.clear();
            0
        };
        self.cached = start;
        let remaining = tokens.len().saturating_sub(start);
        if self.state.position() + remaining > self.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {remaining} positions, {} remain",
                    self.context.saturating_sub(self.state.position())
                ),
            ));
        }
        // The penalties count the whole prompt, the reused prefix included: it is one request.
        self.sampler.reset();
        self.sampler.record_prompt(tokens);
        if remaining == 0 && !self.last_logits.is_empty() {
            return Ok(self.output(self.last_logits.clone()));
        }
        if crate::forward::trace_on() {
            eprintln!(
                "[inkling] prefill tokens={} reused={start} ids={:?}",
                tokens.len(),
                &tokens[start..]
            );
        }
        let prefill = std::time::Instant::now();
        let mut logits = Vec::new();
        for token in &tokens[start..] {
            logits = self.step(*token)?;
            self.history.push(*token);
        }
        if crate::forward::trace_on() {
            eprintln!(
                "[inkling] prefill done in {:.1} ms",
                prefill.elapsed().as_secs_f64() * 1e3
            );
        }
        self.last_logits = logits.clone();
        Ok(self.output(logits))
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        if self.state.position() >= self.context {
            return Err(BackendError::new("context", "the context is exhausted"));
        }
        let logits = self.step(token)?;
        self.history.push(token);
        self.sampler.record(token);
        self.last_logits = logits.clone();
        self.cached = 0;
        Ok(self.output(logits))
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        Some(&self.history)
    }

    fn cached_tokens(&self) -> usize {
        self.cached
    }
}

impl Inkling2HostSession {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        self.weights
            .forward_token(token, &mut self.state)
            .map_err(|message| BackendError::new("forward", message))
    }

    fn output(&mut self, logits: Vec<f32>) -> StepOutput {
        let next_token = choose(
            self.weights.config(),
            &mut self.sampler,
            &logits,
            &self.sampling,
        );
        StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        }
    }
}

/// The next token under one request's sampling, or `None` at an end-of-generation id.
///
/// The penalties are the shared served sampler's (the one Dory and the DeepSeek host group use),
/// and a greedy request takes its argmax. A request with a temperature is sampled by Gumbel-max
/// with counter-based noise ([sparkflash_model::gumbel], DeepSeek's device sampler): the token at
/// sequence index `t` is the argmax of `logit / T + G(seed, t, id)` over the request's top-k and
/// top-p cut, an exact sample of the cut distribution that is a function of the row, the seed and
/// the index alone. A verify's row and a plain step's row at the same index therefore give the
/// same token, and a drafter that perturbs with the same noise proposes what the target will
/// sample. The device arm calls the same function, so the two backends answer a request
/// identically.
pub(crate) fn choose(
    config: &Inkling2Config,
    sampler: &mut Sampler,
    logits: &[f32],
    sampling: &Sampling,
) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    let chosen = match gumbel_sampler(sampling) {
        None => sampler.choose(logits, sampling),
        Some(gumbel) => {
            let scores = sampler.penalized(logits, sampling);
            // The token lands after every id the sequence has consumed.
            let position = u32::try_from(sampler.history().len()).unwrap_or(u32::MAX);
            gumbel.pick_row(scores.as_deref().unwrap_or(logits), position)
        }
    };
    config.next_token(chosen)
}

/// The Gumbel parameters of a request that samples, `None` for a greedy one (temperature zero,
/// below it or NaN). A top-k at or past the vocabulary, or at or below zero, keeps every token.
pub(crate) fn gumbel_sampler(sampling: &Sampling) -> Option<GumbelSampler> {
    let temperature = sampling.temperature;
    if temperature.is_nan() || temperature <= 0.0 {
        return None;
    }
    Some(GumbelSampler {
        temperature,
        top_p: if sampling.top_p.is_nan() {
            1.0
        } else {
            sampling.top_p
        },
        top_k: u32::try_from(sampling.top_k).unwrap_or(0),
        seed: sampling.seed,
    })
}
