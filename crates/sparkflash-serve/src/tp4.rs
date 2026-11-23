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

//! The tensor-parallel group's rank seam: what one process of a serve group is told, and what it
//! answers the process that drives it.
//!
//! A rank opener is per-arm construction code and lives in the arm's crate. What travels between
//! the driver and an opener is this module: the checkpoint, the group's geometry and the
//! deadlines, in one struct the arm reads and the driver fills, and the step adapter's contract
//! ([Tp4RankApi]) together with the shapes a step answers ([Tp4Sampling], [Tp4Step],
//! [Tp4RankMetrics]). Nothing here owns arithmetic or a transport; an adapter maps an arm's own
//! step onto [Tp4Step] and performs no forward of its own.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sparkflash_model::{MergedPosition, Sampling, Speculation};

/// What rank 0 passes to the rank factory: the checkpoint and the group geometry, not a tokenizer.
///
/// Token ids are the contract between serve and the rank; the tokenizer stays on rank 0 and never
/// reaches a worker.
#[derive(Debug, Clone)]
pub struct Tp4RankConfig {
    /// This process's rank.
    pub rank: usize,
    /// The number of ranks in the group.
    pub world_size: usize,
    /// Every rank's address, in rank order.
    pub peers: Vec<IpAddr>,
    /// Rank R binds the engine collective on base_port + R.
    pub base_port: u16,
    /// Rank 0's serve control port.
    pub control_port: u16,
    /// The released sharded checkpoint directory or index.
    pub checkpoint: PathBuf,
    /// The element cap for one tensor.
    pub element_limit: usize,
    /// The context one packed-model sequence allocates. The host reference rank sizes its own
    /// sequence state from the checkpoint, so only a packed arm reads this.
    pub context: usize,
    /// The startup and control deadline.
    pub timeout: Duration,
}

impl Tp4RankConfig {
    /// The control channel address.
    ///
    /// Rank 0's address is the hub. A one-rank group names no peer, so its hub is loopback: the
    /// control channel is serve's, not the engine collective, and a single rank has no other host
    /// to reach.
    #[must_use]
    pub fn control_address(&self) -> SocketAddr {
        let hub = self
            .peers
            .first()
            .copied()
            .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        SocketAddr::new(hub, self.control_port)
    }
}

/// The directory a checkpoint path names: the path itself, or the directory holding the file.
///
/// A rank opens the same checkpoint whether the operator named its directory or its index file, so
/// the rule is the seam's rather than each opener's.
#[must_use]
pub fn checkpoint_directory(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.to_path_buf()
    } else {
        checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf()
    }
}

/// The prefix every prompt-logprob contract refusal carries.
///
/// The serve control plane uses it to tell a deterministic, request-scoped refusal (the whole
/// group made the same decision after the same collectives, so it can keep serving) from a
/// group-fatal forward or transport failure. An arm's step adapter builds its "not scored yet"
/// refusal with this prefix too, so the driver classifies every rank's refusal the same way.
pub const PROMPT_LOGPROBS_REFUSED_PREFIX: &str = "TP=4 prompt logprobs refused:";

/// One forward step's output, reduced to what serve needs.
///
/// A resident TP=4 rank's step adapter maps the engine's step output into this; serve reads only
/// the logits and the rank-0 sampled token. Workers ignore next_token (it is computed identically
/// on every rank, but it must not drive the token stream).
#[derive(Debug, Clone, Default)]
pub struct Tp4Step {
    /// The raw logits for the position just fed; empty from a rank that was not asked for them
    /// ([Tp4Sampling::logits]) and took its greedy pick on the device.
    pub logits: Vec<f32>,
    /// The greedy next token, when the rank produced one.
    pub next_token: Option<u32>,
    /// The tokens the step decoded before `next_token`: a speculative step's kept drafts, which
    /// its verify already ran, so they are emitted and not fed back. Empty for a one-token step.
    pub accepted: Vec<u32>,
    /// The drafts the step's drafter offered before the depth policy shortened the chain.
    pub proposed: usize,
    /// The drafts the depth policy left unverified: `proposed - accepted - (1 if the chain ran)`.
    pub gated: usize,
    /// Prompt tokens this prefill found already in the rank's window, from the previous request.
    /// Zero for every step but a prefill, and zero when the request could not reuse the window.
    pub cached_tokens: usize,
    /// What the rank's own speculator reported for this step, when the session keeps its own
    /// accounting (`ModelSession::speculation`). A packed arm verifies a chain inside one session
    /// call and hands the extra tokens back through later calls, so the fields above cannot
    /// describe the step; this is the step's share of the session's counters instead.
    pub speculation: Option<Speculation>,
}

/// The request sampler a rank applies before its first step.
///
/// The reference driver consumes its engine with with_sampling once per request. A resident TP=4
/// rank holds one engine for the process lifetime, so the same choice arrives through
/// begin_request instead. Sampling changes only which token rank 0 chooses; it does not change the
/// forward arithmetic, and workers keep the greedy default because they ignore next_token.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tp4Sampling {
    /// The request's whole sampler: temperature, top-p, top-k, the three penalties and the seed.
    /// A rank honours all of it or refuses the request.
    pub sampling: Sampling,
    /// Whether each step's logits row is read (a request for logprobs). Without it a rank whose
    /// pick is the row's argmax may take it on the device and leave the row there; the token is
    /// the same.
    pub logits: bool,
    /// Whether the rank may decode the request speculatively: rank 0 sets it for a request whose
    /// tokens are each its row's argmax and whose rows are never read (no temperature, no
    /// penalty, no logprobs), and the Request frame carries it to the workers. A rank that drafts nothing
    /// decodes a token a step.
    pub speculative: bool,
}

/// The evidence counters the rank's extraction exposes.
#[derive(Debug, Clone, Default)]
pub struct Tp4RankMetrics {
    /// Startup load time.
    pub load_ms: u64,
    /// Resident raw-cache bytes.
    pub resident_cache_bytes: u64,
    /// Resident raw-cache tensors.
    pub resident_cache_tensors: usize,
    /// Collective calls made by this rank.
    pub reduce_calls: u64,
    /// Collective elements reduced by this rank.
    pub reduce_elements: u64,
    /// Forward steps run by this rank.
    pub steps: u64,
    /// Wall milliseconds of the most recent step.
    pub last_step_ms: u64,
}

/// The serve-side seam over tp-dev's Tp4Rank extraction.
///
/// This is deliberately a trait rather than a direct dependency: the serve branch compiles against
/// a stub until the extraction lands, and the real adapter is one impl on top of the extracted
/// Tp4Rank. The trait contains no forward arithmetic; Tp4Rank owns the lockstep loop and the
/// resident cache.
///
/// The adapters that implement it live with the arm whose rank they map: a pack's crate holds its
/// own step adapter and answers a boxed [Tp4RankApi] from its opener, so the CLI keeps the entry
/// points and the construction stays with the arm.
pub trait Tp4RankApi: Send {
    /// Resets the per-request SequenceState and applies the request sampler.
    ///
    /// This runs exactly once per request on every rank, before that rank feeds the Request
    /// frame's prompt ids: replace the rank's SequenceState with a fresh
    /// SequenceState::new(global_config) and assert it is empty, then apply the sampler. It must
    /// not read weights or touch the collective; the engine and collective are opened once in
    /// open_tp4_rank and stay owned by Tp4Rank. A rank that holds InferenceEngine by value applies
    /// the sampler with the non-consuming InferenceEngine::set_sampling(Sampling { temperature,
    /// seed }) (task/engine-resident 74e9ff2); it must not rebuild the engine or re-read weights.
    /// A rank that cannot apply a non-greedy sampler must return an error rather than silently run
    /// greedy.
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String>;

    /// The tokens of `prompt` this rank's window already holds and can keep.
    ///
    /// Rank 0 asks this before it broadcasts the Request, so the count every rank prefill uses is
    /// read off one rank rather than inferred four times: two ranks that disagreed would run
    /// different collectives in the same step.
    fn plan_prefill(&mut self, _prompt: &[u32]) -> usize {
        0
    }

    /// Runs the prompt through this rank's forward and returns the last position's step.
    ///
    /// `cached` is the group's decision, from [Self::plan_prefill] on rank 0: those leading tokens
    /// are already in every rank's window and only `prompt[cached..]` is forwarded.
    fn prefill(&mut self, prompt: &[u32], cached: usize) -> Result<Tp4Step, String>;

    /// Runs the prompt through this rank's forward and returns the last position's step together
    /// with every scored prompt position's merged logprob contract.
    ///
    /// Selection is the OpenAI-style top count: `None` scores only the owning token and its
    /// competition rank, `Some(n)` also returns the n best alternatives. A contract refusal (a
    /// certified rank entry, an unprovable f32 order, a malformed shard layout) comes back as a
    /// string beginning with the refusal prefix; every rank makes the same decision after the
    /// same collectives, so the group can keep serving.
    fn prefill_prompt_logprobs(
        &mut self,
        prompt: &[u32],
        selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String>;

    /// Runs one sampled token through this rank's forward.
    fn decode(&mut self, token: u32) -> Result<Tp4Step, String>;

    /// The evidence counters this rank collected.
    fn metrics(&self) -> Tp4RankMetrics;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trait's default is the one the group relies on: a rank that does not track a window
    /// reuses nothing, so rank 0 broadcasts a full prefill and no rank is asked to honour a count
    /// it never computed.
    #[test]
    fn a_rank_that_does_not_track_a_window_plans_no_reuse() {
        struct NoWindow;
        impl Tp4RankApi for NoWindow {
            fn begin_request(&mut self, _sampling: Tp4Sampling) -> Result<(), String> {
                Ok(())
            }
            fn prefill(&mut self, _prompt: &[u32], _cached: usize) -> Result<Tp4Step, String> {
                Ok(Tp4Step::default())
            }
            fn prefill_prompt_logprobs(
                &mut self,
                _prompt: &[u32],
                _selection: Option<usize>,
            ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
                Ok((Tp4Step::default(), Vec::new()))
            }
            fn decode(&mut self, _token: u32) -> Result<Tp4Step, String> {
                Ok(Tp4Step::default())
            }
            fn metrics(&self) -> Tp4RankMetrics {
                Tp4RankMetrics::default()
            }
        }
        let mut rank = NoWindow;
        assert_eq!(rank.plan_prefill(&[1, 2, 3]), 0);
        // The default step shape: no drafts, no rows, no reuse.
        let step = rank.prefill(&[1, 2, 3], 0).expect("a step");
        assert!(step.logits.is_empty() && step.accepted.is_empty());
        assert_eq!(step.proposed, 0);
        assert_eq!(step.cached_tokens, 0);
    }
}
