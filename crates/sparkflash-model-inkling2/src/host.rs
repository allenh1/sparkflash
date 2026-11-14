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

use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, Sampling, StepOutput,
};
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
}

impl ModelSession for Inkling2HostSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
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

    fn output(&self, logits: Vec<f32>) -> StepOutput {
        let next_token = greedy(&logits, &self.sampling);
        StepOutput {
            logits,
            next_token,
            draft: Vec::new(),
        }
    }
}

/// The next token under one sampling request: greedy, or a deterministic temperature draw.
///
/// The device arm calls the same function so the two backends answer a request identically.
pub(crate) fn greedy(logits: &[f32], sampling: &Sampling) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if sampling.temperature <= 0.0 || sampling.top_k == 1 {
        let mut best = 0usize;
        for (index, value) in logits.iter().enumerate() {
            if *value > logits[best] {
                best = index;
            }
        }
        return u32::try_from(best).ok();
    }
    // A deterministic temperature sample: enough for the host path, and reproducible per seed.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits
        .iter()
        .map(|value| ((value - max) / sampling.temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut state = sampling.seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    let draw = (state >> 11) as f32 / (1u64 << 53) as f32 * total;
    let mut running = 0.0f32;
    for (index, weight) in weights.iter().enumerate() {
        running += weight;
        if running >= draw {
            return u32::try_from(index).ok();
        }
    }
    u32::try_from(logits.len() - 1).ok()
}
