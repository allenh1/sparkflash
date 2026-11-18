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

//! The backend contract: how a model definition runs on one host.
//!
//! A model pack owns the architecture and the weights. A backend owns the state and the
//! arithmetic that turns a prompt into logits. This module names the seam between them so the
//! serving layer can drive a model without knowing whether the execution is the CPU reference,
//! a CUDA session, or a test double.
//!
//! The contract is deliberately one step at a time. Prefill consumes a prompt and returns the
//! last position's logits plus the token the backend chose; decode consumes one token and returns
//! the next step. A serving adapter dispatches at the step boundary, which is once per token, not
//! once per tensor or kernel.

use crate::capabilities::Capabilities;
use crate::descriptor::ModelDescriptor;
use crate::vision::{ImageSpan, VisionImage};

/// The error code for a session whose state did not fit on the device.
///
/// It is the one code the serving driver reads rather than reports ([BackendError::is_out_of_memory]):
/// a fresh sequence can be given back and opened again after a kept one releases its memory, and
/// a second failure is the honest answer.
pub const OUT_OF_MEMORY: &str = "out_of_memory";

/// The smallest cache a request-sized session reserves: 2K tokens.
///
/// A request-sized session does not reserve the served context, because a 59-token title call
/// that reserved 128K tokens would hold the memory a whole conversation needs. The floor is what a
/// small request can still pay: at 28 KiB a token it is a few tens of MB.
pub const REQUEST_CAPACITY_FLOOR: usize = 2 * 1024;

/// The headroom a request-sized session reserves past what the request itself reaches.
///
/// About one chat turn and the answer to it, so a conversation that keeps growing outgrows its
/// allocation only every few turns. A session's cache is fixed when it opens, so the turn that
/// does grow pays one re-prefill; the headroom is what keeps that rare.
pub const REQUEST_CAPACITY_SLACK: usize = 1536;

/// The capacity a request-sized session reserves when the request can reach `reach` tokens.
///
/// The reach is the prompt plus the request's generation cap: everything the sequence must hold
/// before its answer is complete. It is rounded up to a power of two from
/// [REQUEST_CAPACITY_FLOOR], so the cache a conversation holds is one of a few stable sizes
/// rather than a request's exact length, and capped at the served context so no request reserves
/// more than the model advertises.
#[must_use]
pub fn request_capacity(configured: usize, reach: usize) -> usize {
    let wanted = reach.saturating_add(REQUEST_CAPACITY_SLACK);
    let mut capacity = REQUEST_CAPACITY_FLOOR;
    while capacity < wanted && capacity < configured {
        capacity = capacity.saturating_mul(2);
    }
    capacity.min(configured)
}

/// A backend failure.
///
/// The message is written for an operator; the code is the stable category a serving layer can
/// expose. Backend implementations decide which category their failure belongs to.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct BackendError {
    /// The stable machine-readable category.
    code: &'static str,
    /// The operator-facing explanation.
    message: String,
}

impl BackendError {
    /// A failure with a caller-supplied category.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// The stable machine-readable category.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.code
    }

    /// The operator-facing explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the session could not allocate its state on the device.
    ///
    /// A serving driver may give back a kept sequence's memory and try the open again: two long
    /// conversations can exceed the card even when each was sized for its own request.
    #[must_use]
    pub fn is_out_of_memory(&self) -> bool {
        self.code == OUT_OF_MEMORY
    }
}

/// What one prefill or decode step produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepOutput {
    /// The logits of the last position. Empty when the backend does not expose them.
    pub logits: Vec<f32>,
    /// The token the backend selected for the next position, or None when generation is done.
    pub next_token: Option<u32>,
    /// Draft tokens after the anchor, for a backend with speculative decoding. Empty when the
    /// backend drafts nothing; a generational adapter may use it as evidence, but must not treat
    /// a draft as an emitted token.
    pub draft: Vec<u32>,
}

/// What a sequence's speculative decoder has done so far.
///
/// The serving layer reports these per request; a backend that drafts nothing keeps the zero
/// default, which reads as "no speculation" in the stats block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Speculation {
    /// Forward passes that verified at least one draft token.
    pub verify_steps: usize,
    /// Draft tokens those passes offered.
    pub drafts_proposed: usize,
    /// Draft tokens the model's own choice agreed with.
    pub drafts_accepted: usize,
}

/// The sampling configuration for one request.
///
/// This mirrors the protocol's sampling fields so a model backend can apply them with the
/// arithmetic and random source it owns. The serving layer sets it on a fresh session before
/// prefill; a backend that has no sampling stage may ignore it and return a deterministic token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0.0 is greedy; above it the backend's seeded distribution.
    pub temperature: f32,
    /// Nucleus threshold; 1.0 disables it.
    pub top_p: f32,
    /// Keep this many most likely tokens; 0 and -1 disable it.
    pub top_k: i32,
    /// Repetition penalty over the prompt and the generated tokens; 1.0 disables it.
    pub repetition_penalty: f32,
    /// Presence penalty over generated tokens; 0.0 disables it.
    pub presence_penalty: f32,
    /// Frequency penalty over generated tokens; 0.0 disables it.
    pub frequency_penalty: f32,
    /// The deterministic seed when the request named one.
    pub seed: u64,
    /// How many of the consumed tokens are the prompt.
    pub prompt_len: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            prompt_len: 0,
        }
    }
}

/// One sequence's backend state.
///
/// A session serves one request at a time. It is `Send`, so a serving adapter may move it between
/// worker threads and keep it between requests, but it is not `Sync`: exactly one thread calls
/// [`ModelSession::prefill`] or [`ModelSession::decode`] at a time. A backend whose sessions share
/// mutable engine state must serialize that state itself; see [`ModelBackend`].
pub trait ModelSession: Send {
    /// Applies this request's sampler before any step.
    ///
    /// The default keeps the backend's own deterministic choice, which is the right behavior for
    /// a test double and for a backend that has not implemented sampling yet.
    fn set_sampling(&mut self, sampling: Sampling) {
        let _ = sampling;
    }

    /// Consumes a prompt and returns the first step.
    ///
    /// # Errors
    /// A backend failure, such as a shape mismatch or an exhausted context.
    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError>;

    /// Consumes a prompt whose image spans take `images`' features, and returns the first step.
    ///
    /// The images are in prompt order and their spans lie inside `tokens`. Without images this is
    /// [`Self::prefill`]. The default refuses any image: a session that cannot encode one must
    /// say so rather than read the placeholders as text.
    ///
    /// # Errors
    /// As [`Self::prefill`], and a session without a vision path refuses images.
    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if images.is_empty() {
            return self.prefill(tokens);
        }
        Err(BackendError::new(
            "vision",
            format!(
                "this session has no vision path for the request's {} images",
                images.len()
            ),
        ))
    }

    /// Consumes one generated token and returns the next step.
    ///
    /// # Errors
    /// A backend failure, such as a shape mismatch or an exhausted context.
    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError>;

    /// The speculative decoder's counts for this sequence so far.
    ///
    /// The default is zero, for a backend that drafts nothing.
    fn speculation(&self) -> Speculation {
        Speculation::default()
    }

    /// The tokens whose state this sequence holds, when it can continue from them.
    ///
    /// A serving adapter may keep a session between requests and hand it a prompt that starts
    /// with these tokens; the session then forwards only the rest. `None` - the default - says
    /// the session cannot serve another request, and the adapter opens a fresh one. A history
    /// counts only tokens the sequence has actually forwarded and still holds: a state that is
    /// ahead of them (an un-consumed speculative tail, say) must report `None` rather than a
    /// prefix it cannot honour.
    fn reusable_prefix(&self) -> Option<&[u32]> {
        None
    }

    /// A shorter prefix of [`Self::reusable_prefix`] the sequence can go back to, when it can.
    ///
    /// A session that kept its state at an earlier point (the end of its last prompt, say) may
    /// take a prompt that starts with these tokens and parts from the rest of its history: it
    /// rewinds to them and forwards the rest, and reports them as the hit. An adapter offers it
    /// such a prompt only when the prompt goes past them. `None` - the default - says the session
    /// has no such point.
    fn rewind_prefix(&self) -> Option<&[u32]> {
        None
    }

    /// The images [`Self::reusable_prefix`] holds: their spans as they were fed, content keys
    /// included.
    ///
    /// A history is ids, and an image's placeholders are the same ids whatever the image, so an
    /// adapter continues a history that holds images only with a prompt that carries these same
    /// images at these same spans. The ones that start before [`Self::rewind_prefix`]'s end are
    /// the rewind point's. The default - none - suits a session that keeps no image with its
    /// history: a prompt with an image inside the held tokens then starts a fresh sequence.
    fn held_images(&self) -> &[ImageSpan] {
        &[]
    }

    /// How many of the last [`Self::prefill`]'s tokens the sequence already held.
    ///
    /// This is what the request reports as its cache hit. The default is zero, for a session
    /// that forwards every prompt token.
    fn cached_tokens(&self) -> usize {
        0
    }

    /// Says the request is over and the adapter keeps the session for a later one: whatever the
    /// sequence holds may be given up for another sequence that needs the memory, after which
    /// [`Self::reusable_prefix`] no longer names it. The default does nothing, for a session that
    /// holds its memory until it is dropped.
    fn park(&mut self) {}

    /// How many tokens this sequence's state can hold, prompt and generation together.
    ///
    /// The serving driver continues a kept sequence only with a request that fits: a session's
    /// cache is fixed when it opens, so one that cannot hold the prompt plus the generation cap
    /// would run past its own context mid-answer. Such a sequence is dropped and a fresh one is
    /// opened at the request's size ([ModelBackend::open_session_for]) instead of grown.
    ///
    /// `usize::MAX` - the default - says the session does not size its state against a request,
    /// so every request fits.
    fn capacity_tokens(&self) -> usize {
        usize::MAX
    }
}

/// How a model runs on one host.
///
/// A backend is a factory for sessions, not a session itself: it can be shared, while each
/// request gets its own state. [`ModelBackend::open_session`] may be called from several worker
/// threads at once, and those sessions may run concurrently, so a backend whose sessions share
/// mutable engine state must serialize that state itself. Read-only weights and handles may be
/// shared; per-session streams and scratch allocations are the backend's choice. The descriptor
/// is the identity the serving layer reports; the capabilities are what this backend can actually
/// execute, which may be narrower than what the model family supports when loaded fully.
pub trait ModelBackend: Send + Sync {
    /// The model this backend executes.
    fn descriptor(&self) -> &ModelDescriptor;

    /// What this backend can serve.
    ///
    /// The default takes the descriptor's capabilities. A backend that does not load an optional
    /// tower narrows the answer rather than advertising a channel it cannot run.
    fn capabilities(&self) -> Capabilities {
        self.descriptor().capabilities
    }

    /// The interconnect this backend's ranks exchange over, when it serves a group.
    ///
    /// A single-process backend leaves this `None`, and the serving layer reports that there is no
    /// cluster behind it. A backend that is one rank of a group names the transport honestly, so
    /// the model card and the cluster stats do not claim "no cluster" for a four-host server.
    fn fabric_mode(&self) -> Option<&'static str> {
        None
    }

    /// Opens a fresh sequence.
    ///
    /// # Errors
    /// A backend failure, such as unavailable weights or an unsupported configuration.
    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError>;

    /// Whether [`Self::open_session_for`] makes a session smaller than the served context.
    ///
    /// The serving driver keeps one sequence more than it has workers for such a backend: a
    /// second request-sized sequence costs only what its request reaches, and the spare slot is
    /// what lets a client interleave an unrelated request (a title call) without evicting the
    /// conversation the next turn continues. A backend whose sessions reserve the whole served
    /// context keeps the worker count instead, because two of those may not fit on one card.
    fn sizes_sessions(&self) -> bool {
        false
    }

    /// Opens a fresh sequence sized for a request that can reach `tokens` positions.
    ///
    /// `tokens` is the request's prompt plus its generation cap: what the sequence must hold
    /// before the answer is complete. A backend that reserves its cache when the session opens
    /// uses it to reserve only what the request can reach ([request_capacity]), instead of the
    /// whole served context for a request that will never get there.
    ///
    /// The default ignores the size and opens the backend's configured session, which is right
    /// for a backend whose state grows as it consumes tokens: it has nothing to reserve up front.
    ///
    /// # Errors
    /// A backend failure, such as unavailable weights, or a device that cannot fit the state.
    /// [BackendError::is_out_of_memory] distinguishes the second.
    fn open_session_for(&self, tokens: usize) -> Result<Box<dyn ModelSession>, BackendError> {
        let _ = tokens;
        self.open_session()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_error_keeps_its_code_and_message() {
        let error = BackendError::new("weights", "the shard is missing");
        assert_eq!(error.code(), "weights");
        assert_eq!(error.message(), "the shard is missing");
        assert_eq!(error.to_string(), "the shard is missing");
    }

    #[test]
    fn a_request_capacity_rounds_up_to_a_power_of_two_and_stops_at_the_context() {
        // A title call reaches 67 tokens: it reserves the 2K floor, not the served context.
        assert_eq!(request_capacity(131_072, 59 + 8), REQUEST_CAPACITY_FLOOR);
        // One token past the floor's reach moves the reservation to the next power of two.
        assert_eq!(
            request_capacity(131_072, REQUEST_CAPACITY_FLOOR - REQUEST_CAPACITY_SLACK + 1),
            2 * REQUEST_CAPACITY_FLOOR
        );
        assert_eq!(
            request_capacity(131_072, REQUEST_CAPACITY_FLOOR - REQUEST_CAPACITY_SLACK),
            REQUEST_CAPACITY_FLOOR
        );
        // A 120K prompt at an 8-token cap reserves the whole 128K window it can reach.
        assert_eq!(request_capacity(131_072, 120 * 1024 + 8), 131_072);
        // The served context caps it: no request reserves more than the model advertises.
        assert_eq!(request_capacity(32_768, 120 * 1024 + 8), 32_768);
        // A context below the floor is reserved whole rather than rounded past.
        assert_eq!(request_capacity(1024, 59), 1024);
    }

    #[test]
    fn only_the_out_of_memory_code_reads_as_an_allocation_failure() {
        assert!(BackendError::new(OUT_OF_MEMORY, "cuMemAlloc failed").is_out_of_memory());
        assert!(!BackendError::new("weights", "cuMemAlloc failed").is_out_of_memory());
    }

    #[test]
    fn default_sampling_is_the_disabled_configuration() {
        let sampling = Sampling::default();
        assert_eq!(sampling.temperature, 0.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
    }
}
