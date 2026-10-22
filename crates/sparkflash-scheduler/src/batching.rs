//! Continuous batching and the default batched/chunked prefill policy.
//!
//! This module is the scheduler half of the serving batching work item (WS-F): it plans a
//! scheduling *step* over several requests, with the batched chunked-prefill policy as the
//! default and the pre-batch-17 per-token decode loop kept as a compatibility mode.
//!
//! # What a step is
//!
//! One [ContinuousBatcher::plan_step] returns a [BatchStep]: a list of [ScheduledOp]s, in
//! execution order, plus the token-budget accounting for the step. The executor (the engine
//! driver) then runs each forward-bearing op and calls [ContinuousBatcher::accept_step] with
//! the token each op produced, which is what records the new history and frees the KV of a
//! finished sequence. Splitting plan from accept is deliberate: preemption has to replay a
//! victim's real token history, so the batcher owns the tokens (prompt plus generated) rather
//! than only their counts.
//!
//! # The prefill policy
//!
//! [PrefillMode::Batched] is the default: a prompt is forwarded in contiguous token-budget
//! chunks of at most [BatchConfig::prefill_chunk_tokens], one forward per chunk, and a
//! request advances at most one chunk per step. Prefill tokens never become decode ops, which
//! is the property the old path broke: one decode-shaped forward per prompt token. The
//! compatibility mode [PrefillMode::TokenLoop] keeps exactly that old shape (chunk size 1)
//! for a bisect or an exotic case, and is never selected by [BatchConfig::default].
//!
//! The device forward is out of this crate. The reference engine executes one forward per
//! forward-bearing op, so the BatchStep host_forwards count is the host call count; the device
//! path is expected to fuse a step into one launch, which is why device_forward_launches is
//! one per non-empty step. Neither number is a measurement.
//!
//! # Continuous batching
//!
//! A step mixes decodes and prefill chunks: running sequences in the decode phase each receive
//! one token (the decode-first policy, [BatchConfig::decode_first]), and the remaining token
//! budget is filled with at most one prefill chunk per prefilling sequence, rotating the
//! starting sequence so a request that missed the budget gets the first turn next step. That
//! rotation is the fairness property the tests pin: under a budget smaller than the demand, no
//! request starves.
//!
//! # Preemption
//!
//! The KV pool is the paged cache from the sparkflash-kv crate, so capacity is real (64-token
//! pages with hash-block sharing). When a step needs a page it does not have, a victim is
//! preempted: its pages are returned and the request is queued as [RequestPhase::Swapped]. The
//! two policies are:
//!
//! * [PreemptionPolicy::Recompute] (default): on resume the retained prefix is reloaded
//!   (complete blocks the paged cache still holds) and the rest of the consumed history is
//!   forwarded again as prefill chunks.
//! * [PreemptionPolicy::Swap]: on resume the consumed history is reattached and the request
//!   returns to decode with no re-forwarding. The *bytes* of a swap are not moved here - this
//!   crate plans and accounts, it does not touch KV - so [RequestSnapshot::swap_bytes] is the
//!   size the device/tier swap would carry, and the device window owns the copy.
//!
//! # Prefix caching
//!
//! The paged cache shares a complete 64-token block whose hash is already cached, and keeps an
//! unreferenced block's data on the free list while its hash survives. This module reports the
//! consequence per request ([RequestSnapshot::prefix_hit_tokens], the
//! [ScheduledOp::PrefixReuse] op) and plans the prefill from the end of the reusable prefix,
//! capped one token short of the prompt so the final logits still come from a forward. The
//! [PrefillCostEstimate] type turns a cached-token count into an explicitly-labelled TTFT
//! estimate built from measured per-token costs; it is an estimate, not a measurement.
//!
//! # What is not here
//!
//! The device launch sequence, the per-step fused forward, the swap byte copy and any TTFT or
//! throughput number. The scheduler is the decision layer: everything here is host arithmetic
//! and is tested on synthetic token fixtures with no weights and no GPU.

use std::collections::VecDeque;
use std::ops::Range;

use sparkflash_kv::{
    GLOBAL_KV_BYTES_PER_TOKEN, KV_PAGE_TOKENS, PagedGeometry, PagedKvCache, PagedStats,
    SWA_TAIL_BYTES,
};

use crate::RequestId;
use crate::config::{MAX_CONTEXT_TOKENS, PREFILL_CHUNK_CHOICES};
use crate::error::SchedulerError;

/// Requests a step admits by default: the CUDA graph's batch bucket ceiling (spec 5.6.3,
/// CUDA_GRAPH_MAX_BATCH = 16).
pub const DEFAULT_MAX_NUM_SEQS: usize = 16;

/// Tokens a step may batch by default; also the ceiling a configured prefill chunk is bounded
/// by.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;

/// Default prefill chunk size: the smaller of the two chunk sizes the specification allows
/// (spec 5.6.1, 4096).
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = PREFILL_CHUNK_CHOICES[0];

/// Default paged-KV pool size in 64-token pages (65,536 tokens).
pub const DEFAULT_KV_PAGES: u32 = 1024;

/// Default SWA tail slots: one per sequence that may run, so [DEFAULT_MAX_NUM_SEQS].
pub const DEFAULT_KV_TAILS: u32 = 16;

/// Chunk size the compatibility per-token path uses: one prompt token per forward.
pub const TOKEN_LOOP_CHUNK_TOKENS: usize = 1;

/// Measured cost of one decode-shaped forward on the four-rank TP=4 host path, in milliseconds.
///
/// Evidence: docs/build/tp4-tokens-per-second.md line 38 - the pre-fix census TTFT of
/// 39.134645546 s is 37 prompt tokens at ~0.945 s/token plus ~2 s HTTP/tokenizer overhead. This
/// is a measured number for the *token-loop* path; it is used only by [PrefillCostEstimate],
/// which labels its output an estimate.
pub const MEASURED_TOKEN_LOOP_MS_PER_PROMPT_TOKEN: f64 = 945.0;

/// Measured cost of the batch-17 whole-prompt chunk prefill in the *bench* harness, in
/// milliseconds per prompt token: 24.6 s for 37 tokens (evidence:
/// docs/build/tp4-tokens-per-second.md line 371-373). Bench only; the serve endpoint measured
/// this path *slower* than the token loop (43.62 s at 37 tokens, same document line 46-51).
pub const MEASURED_BENCH_CHUNK_MS_PER_PROMPT_TOKEN: f64 = 664.9;

/// Measured cost of the batch-17 whole-prompt chunk prefill at the *serve endpoint*, in
/// milliseconds per prompt token: 43.624741849 s for 37 tokens (evidence:
/// docs/build/tp4-tokens-per-second.md line 46-51). Slower than the token loop; the overhead is
/// the open batch-18 question and is why the CLI default stayed on the token loop.
pub const MEASURED_ENDPOINT_CHUNK_MS_PER_PROMPT_TOKEN: f64 = 1179.0;

/// How the scheduler prefills a prompt.
///
/// The default is [PrefillMode::Batched]. [PrefillMode::TokenLoop] exists so the pre-batch-17
/// execution shape can still be selected for a bisect or a case the batched path cannot serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillMode {
    /// Default: contiguous token-budget chunks, one forward per chunk.
    #[default]
    Batched,
    /// Compatibility: one decode-shaped forward per prompt token (chunk size 1).
    TokenLoop,
}

impl PrefillMode {
    /// Parses the environment/CLI spelling of a prefill mode.
    ///
    /// Accepted values: batched, chunk, chunked for [Self::Batched]; token-loop, token_loop,
    /// loop, token for [Self::TokenLoop]. Anything else returns None so the caller can refuse it
    /// rather than silently keeping the default.
    #[must_use]
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "batched" | "chunk" | "chunked" => Some(Self::Batched),
            "token-loop" | "token_loop" | "loop" | "token" => Some(Self::TokenLoop),
            _ => None,
        }
    }

    /// The operator-facing name of this mode, for a boot line or a metrics label.
    ///
    /// Deliberately distinct from [`Self::from_env_value`], which accepts several spellings: the
    /// label is the canonical one, so a log line and an env var cannot disagree about which mode
    /// actually ran.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Batched => "chunked",
            Self::TokenLoop => "token-loop",
        }
    }

    /// Chunk size this mode forwards a prompt in, given the configured batched chunk size.
    #[must_use]
    pub const fn chunk_tokens(self, configured: usize) -> usize {
        match self {
            Self::Batched => configured,
            Self::TokenLoop => TOKEN_LOOP_CHUNK_TOKENS,
        }
    }

    /// True when the mode forwards one prompt token per engine call.
    #[must_use]
    pub const fn is_token_loop(self) -> bool {
        matches!(self, Self::TokenLoop)
    }
}

/// What happens to a preempted request when it resumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionPolicy {
    /// Default: reload the retained prefix and re-forward the rest of the history.
    #[default]
    Recompute,
    /// Resume from the reattached history without re-forwarding; the KV bytes are the device's
    /// swap copy, accounted but not moved here.
    Swap,
}

impl PreemptionPolicy {
    /// True when a resumed request returns to decode without a prefill forward.
    #[must_use]
    pub const fn resumes_without_prefill(self) -> bool {
        matches!(self, Self::Swap)
    }
}

/// The batching knobs, with the specification/measured defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchConfig {
    /// Prefill execution policy; [PrefillMode::Batched] by default.
    pub prefill_mode: PrefillMode,
    /// Highest number of concurrently running sequences (max_num_seqs).
    pub max_num_seqs: usize,
    /// Tokens a step may batch (max_num_batched_tokens), across decodes and prefill chunks.
    pub max_num_batched_tokens: usize,
    /// Largest prefill chunk in tokens (one forward per chunk).
    pub prefill_chunk_tokens: usize,
    /// Longest prompt the batcher admits; longer prompts are refused at submit.
    pub max_prompt_tokens: usize,
    /// Paged-KV pool size in 64-token pages.
    pub kv_pages: u32,
    /// SWA tail slots the pool holds; one running sequence owns one.
    pub kv_tails: u32,
    /// What a preempted request does when it resumes.
    pub preemption: PreemptionPolicy,
    /// Serve the decode phase before the prefill phase in a mixed step.
    pub decode_first: bool,
    /// Plan the prefill from the end of the reusable prefix rather than reforwarding it.
    ///
    /// false is the conservative compatibility path: the full prompt is forwarded even when
    /// its complete blocks are cached, which keeps the SWA-tail replay decision unchanged (the
    /// KV layer's PrefixMatchPlan remains the authority on that decision and this batcher does
    /// not model the replay).
    pub prefix_reuse: bool,
    /// Bytes a swapped sequence's KV would occupy per token, for the swap ledger.
    pub swap_bytes_per_token: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            prefill_mode: PrefillMode::Batched,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
            max_prompt_tokens: MAX_CONTEXT_TOKENS,
            kv_pages: DEFAULT_KV_PAGES,
            kv_tails: DEFAULT_KV_TAILS,
            preemption: PreemptionPolicy::Recompute,
            decode_first: true,
            prefix_reuse: true,
            swap_bytes_per_token: GLOBAL_KV_BYTES_PER_TOKEN,
        }
    }
}

impl BatchConfig {
    /// Checks every knob and returns the first one that is out of range.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidConfig] naming the offending knob and its value.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        if self.max_num_seqs == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_num_seqs must be at least 1".to_owned(),
            ));
        }
        if self.max_num_batched_tokens == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_num_batched_tokens must be at least 1".to_owned(),
            ));
        }
        if self.prefill_chunk_tokens == 0 {
            return Err(SchedulerError::InvalidConfig(
                "prefill_chunk_tokens must be at least 1".to_owned(),
            ));
        }
        if self.prefill_chunk_tokens > self.max_num_batched_tokens {
            return Err(SchedulerError::InvalidConfig(format!(
                "prefill_chunk_tokens {} exceeds max_num_batched_tokens {}",
                self.prefill_chunk_tokens, self.max_num_batched_tokens
            )));
        }
        if self.max_prompt_tokens == 0 {
            return Err(SchedulerError::InvalidConfig(
                "max_prompt_tokens must be at least 1".to_owned(),
            ));
        }
        if self.max_prompt_tokens > MAX_CONTEXT_TOKENS {
            return Err(SchedulerError::InvalidConfig(format!(
                "max_prompt_tokens {} exceeds the model limit of {MAX_CONTEXT_TOKENS}",
                self.max_prompt_tokens
            )));
        }
        if self.kv_pages == 0 {
            return Err(SchedulerError::InvalidConfig(
                "kv_pages must be at least 1".to_owned(),
            ));
        }
        if self.kv_tails == 0 {
            return Err(SchedulerError::InvalidConfig(
                "kv_tails must be at least 1".to_owned(),
            ));
        }
        if self.kv_tails < self.max_num_seqs as u32 {
            return Err(SchedulerError::InvalidConfig(format!(
                "kv_tails {} is fewer than max_num_seqs {}; a running sequence needs one tail slot",
                self.kv_tails, self.max_num_seqs
            )));
        }
        if self.swap_bytes_per_token == 0 {
            return Err(SchedulerError::InvalidConfig(
                "swap_bytes_per_token must be at least 1".to_owned(),
            ));
        }
        Ok(())
    }

    /// The chunk size the configured mode forwards at: the batched chunk, or one token.
    #[must_use]
    pub const fn chunk_tokens(&self) -> usize {
        self.prefill_mode.chunk_tokens(self.prefill_chunk_tokens)
    }

    /// Same configuration with a different prefill mode.
    #[must_use]
    pub const fn with_prefill_mode(mut self, mode: PrefillMode) -> Self {
        self.prefill_mode = mode;
        self
    }

    /// Same configuration with a different prefill chunk size.
    #[must_use]
    pub const fn with_prefill_chunk_tokens(mut self, tokens: usize) -> Self {
        self.prefill_chunk_tokens = tokens;
        self
    }
}

/// One forward-bearing (or bookkeeping) operation a step schedules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduledOp {
    /// One forward over a contiguous prompt range; the last chunk of a prompt also produces the
    /// request's first generated token.
    PrefillChunk {
        /// Request the chunk belongs to.
        seq: RequestId,
        /// Token range inside the request's prompt (or consumed history on a recompute).
        range: Range<usize>,
        /// True when this chunk is the last one and the forward produces the next token.
        emits_token: bool,
    },
    /// One decode forward at the next position.
    Decode {
        /// Request the token belongs to.
        seq: RequestId,
        /// Position the token will occupy in the sequence, counted from the prompt.
        position: usize,
    },
    /// A complete cached prefix the request's prefill reuses; no forward.
    PrefixReuse {
        /// Request the prefix belongs to.
        seq: RequestId,
        /// Tokens whose KV is reused.
        tokens: usize,
    },
    /// A victim was preempted and its pages returned.
    Preempted {
        /// Victim request.
        seq: RequestId,
        /// Pages the paged cache returned.
        reclaimed_pages: usize,
        /// Of those, complete blocks whose hash survived for a later prefix hit.
        retained_blocks: usize,
    },
    /// A swapped request resumed; the history was reattached.
    SwappedIn {
        /// Resumed request.
        seq: RequestId,
        /// Tokens reattached.
        restored_tokens: usize,
        /// Of those, tokens restored from surviving complete blocks rather than the swap copy.
        from_prefix_cache: usize,
    },
}

impl ScheduledOp {
    /// The request this op belongs to.
    #[must_use]
    pub const fn seq(&self) -> RequestId {
        match self {
            Self::PrefillChunk { seq, .. }
            | Self::Decode { seq, .. }
            | Self::PrefixReuse { seq, .. }
            | Self::Preempted { seq, .. }
            | Self::SwappedIn { seq, .. } => *seq,
        }
    }

    /// True when the op is one engine call on the reference host path.
    #[must_use]
    pub const fn calls_a_forward(&self) -> bool {
        matches!(self, Self::PrefillChunk { .. } | Self::Decode { .. })
    }

    /// True when the op produces a token the executor must hand back to
    /// [ContinuousBatcher::accept_step].
    #[must_use]
    pub const fn produces_token(&self) -> bool {
        match self {
            Self::PrefillChunk { emits_token, .. } => *emits_token,
            Self::Decode { .. } => true,
            _ => false,
        }
    }
}

/// One scheduling step's decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchStep {
    /// Step number this plan is for.
    pub step: u64,
    /// Operations in execution order.
    pub ops: Vec<ScheduledOp>,
    /// Prompt tokens the step forwards.
    pub prefill_tokens: usize,
    /// Decode tokens the step forwards.
    pub decode_tokens: usize,
    /// Cached tokens the step reuses without forwarding.
    pub prefix_reused_tokens: usize,
    /// Requests admitted at this step.
    pub admitted: Vec<RequestId>,
    /// Requests preempted at this step.
    pub preempted: Vec<RequestId>,
    /// Requests that resumed at this step.
    pub swapped_in: Vec<RequestId>,
    /// The token budget the step planned against.
    pub token_budget: usize,
}

impl BatchStep {
    /// An empty plan for step with a token budget.
    #[must_use]
    pub const fn new(step: u64, token_budget: usize) -> Self {
        Self {
            step,
            ops: Vec::new(),
            prefill_tokens: 0,
            decode_tokens: 0,
            prefix_reused_tokens: 0,
            admitted: Vec::new(),
            preempted: Vec::new(),
            swapped_in: Vec::new(),
            token_budget,
        }
    }

    /// True when the step schedules no work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Engine calls on the reference host path: one per prefill chunk and one per decode.
    #[must_use]
    pub fn host_forwards(&self) -> usize {
        self.ops.iter().filter(|op| op.calls_a_forward()).count()
    }

    /// Device launches a fused step is expected to make: one when the step forwards anything.
    ///
    /// This is a planning shape, not a measurement: whether the device path can fuse a mixed
    /// prefill/decode step into one launch is the WS-E/device work.
    #[must_use]
    pub fn device_forward_launches(&self) -> usize {
        if self.host_forwards() > 0 { 1 } else { 0 }
    }

    /// Every op scheduled for seq, in order.
    #[must_use]
    pub fn ops_for(&self, seq: RequestId) -> Vec<&ScheduledOp> {
        self.ops.iter().filter(|op| op.seq() == seq).collect()
    }
}

/// A standalone token-budget chunk plan for one prompt range.
///
/// The per-request form of the batched prefill policy: contiguous ranges of at most
/// chunk_tokens and, in aggregate, never more than max_num_batched_tokens per step. The batcher
/// allocates the budget across requests; this type is the single-request decomposition the
/// tests and the engine executor use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBudgetPlan {
    ranges: Vec<Range<usize>>,
    chunk_tokens: usize,
    max_num_batched_tokens: usize,
}

impl TokenBudgetPlan {
    /// Plans start..prompt_tokens in token-budget chunks.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidArgument] for an empty range, a zero chunk size, a chunk
    /// above the token budget, or a start beyond the prompt.
    pub fn plan(
        prompt_tokens: usize,
        start: usize,
        chunk_tokens: usize,
        max_num_batched_tokens: usize,
    ) -> Result<Self, SchedulerError> {
        if prompt_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a token-budget plan needs at least one prompt token".to_owned(),
            ));
        }
        if start > prompt_tokens {
            return Err(SchedulerError::InvalidArgument(format!(
                "the plan starts at {start} beyond the {prompt_tokens} token prompt"
            )));
        }
        if chunk_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a token-budget plan needs a positive chunk size".to_owned(),
            ));
        }
        if max_num_batched_tokens == 0 || chunk_tokens > max_num_batched_tokens {
            return Err(SchedulerError::InvalidArgument(format!(
                "a {chunk_tokens} token chunk does not fit a {max_num_batched_tokens} token budget"
            )));
        }
        let mut ranges = Vec::new();
        let mut begin = start;
        while begin < prompt_tokens {
            let end = (begin + chunk_tokens).min(prompt_tokens);
            ranges.push(begin..end);
            begin = end;
        }
        Ok(Self {
            ranges,
            chunk_tokens,
            max_num_batched_tokens,
        })
    }

    /// The token ranges, in order.
    #[must_use]
    pub fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Chunks the plan forwards.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.ranges.len()
    }

    /// Chunk size the plan splits at.
    #[must_use]
    pub const fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    /// Token budget the plan was built against.
    #[must_use]
    pub const fn max_num_batched_tokens(&self) -> usize {
        self.max_num_batched_tokens
    }

    /// Tokens of the last chunk; only the last one may be short.
    #[must_use]
    pub fn last_chunk_tokens(&self) -> usize {
        self.ranges
            .last()
            .map_or(0, |range| range.end - range.start)
    }

    /// Tokens the plan covers.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum()
    }
}

/// A request's phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Queued, not admitted yet.
    Waiting,
    /// Admitted; its prompt is still being forwarded.
    Prefill,
    /// Prompt forwarded; emitting one token per step.
    Decode,
    /// Preempted; waiting to resume.
    Swapped,
    /// Done; its KV is freed.
    Finished,
}

/// A request's observable state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestSnapshot {
    /// Request handle.
    pub id: RequestId,
    /// Current phase.
    pub phase: RequestPhase,
    /// Prompt tokens the request brought.
    pub prompt_tokens: usize,
    /// Tokens consumed so far: prompt plus generated.
    pub consumed_tokens: usize,
    /// Tokens generated so far.
    pub generated_tokens: usize,
    /// The request's completion cap.
    pub max_new_tokens: usize,
    /// How far the prefill has forwarded.
    pub prefill_cursor: usize,
    /// Times the request was preempted.
    pub preemptions: u32,
    /// Complete cached prompt tokens reused at admission.
    pub prefix_hit_tokens: usize,
    /// Bytes a swap of this request would carry while it is swapped; zero otherwise.
    pub swap_bytes: usize,
    /// Step the request arrived at.
    pub arrival_step: u64,
}

/// Cumulative batcher counters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BatchStats {
    /// Steps planned.
    pub steps: u64,
    /// Requests submitted.
    pub submitted: u64,
    /// Requests admitted.
    pub admitted: u64,
    /// Requests finished.
    pub finished: u64,
    /// Preemptions.
    pub preemptions: u64,
    /// Swap-in operations.
    pub swap_ins: u64,
    /// Prompt tokens forwarded by prefill chunks.
    pub prefill_tokens: u64,
    /// Decode tokens forwarded.
    pub decode_tokens: u64,
    /// Prompt tokens reused from the prefix cache instead of forwarded.
    pub prefix_reused_tokens: u64,
    /// Engine calls the reference host path would make: one per forward-bearing op.
    pub host_forwards: u64,
    /// Steps that forwarded anything (the expected device launch count).
    pub device_launches: u64,
    /// Paged-cache block evictions observed.
    pub kv_evictions: u64,
    /// Highest number of simultaneously running sequences.
    pub max_running: usize,
}

/// A batcher snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSnapshot {
    /// Per-request state, in request-id order.
    pub requests: Vec<RequestSnapshot>,
    /// Cumulative counters.
    pub stats: BatchStats,
    /// Paged-cache counters.
    pub kv: PagedStats,
}

/// A TTFT estimate for a prefill, labelled as an estimate.
///
/// The per-token costs are *measured* numbers for the two paths that were measured, but the
/// mapping from them to this request is not: a token-budget chunked forward is not the measured
/// whole-prompt chunk, and an endpoint is not the bench harness. No caller may report the
/// result of this type as a TTFT measurement or as a device result; it exists so a prefix-cache
/// decision can be discussed with explicit provenance instead of an invented percentage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillCostEstimate {
    /// Path the estimate is for.
    pub mode: PrefillMode,
    /// Prompt tokens the request brought.
    pub prompt_tokens: usize,
    /// Prompt tokens the prefill reuses from the prefix cache.
    pub reused_tokens: usize,
    /// Prompt tokens the prefill actually forwards.
    pub forwarded_tokens: usize,
    /// Per-forwarded-token cost the estimate used, in milliseconds.
    pub ms_per_forwarded_token: f64,
}

impl PrefillCostEstimate {
    /// Estimates a prefill's TTFT from a cached-token count.
    ///
    /// [PrefillMode::Batched] uses the measured batch-17 whole-prompt *endpoint* cost (the
    /// conservative of the measured chunk numbers, because the token-budget chunked path is
    /// unmeasured on the endpoint); [PrefillMode::TokenLoop] uses the measured token-loop cost.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidArgument] when reused_tokens exceeds the prompt.
    pub fn estimate(
        mode: PrefillMode,
        prompt_tokens: usize,
        reused_tokens: usize,
    ) -> Result<Self, SchedulerError> {
        if reused_tokens > prompt_tokens {
            return Err(SchedulerError::InvalidArgument(format!(
                "{reused_tokens} reused tokens exceed the {prompt_tokens} token prompt"
            )));
        }
        let ms_per_forwarded_token = match mode {
            PrefillMode::Batched => MEASURED_ENDPOINT_CHUNK_MS_PER_PROMPT_TOKEN,
            PrefillMode::TokenLoop => MEASURED_TOKEN_LOOP_MS_PER_PROMPT_TOKEN,
        };
        Ok(Self {
            mode,
            prompt_tokens,
            reused_tokens,
            forwarded_tokens: prompt_tokens - reused_tokens,
            ms_per_forwarded_token,
        })
    }

    /// The estimated TTFT in milliseconds; an estimate, never a measurement.
    #[must_use]
    pub fn estimated_ttft_ms(&self) -> f64 {
        self.forwarded_tokens as f64 * self.ms_per_forwarded_token
    }

    /// The estimated TTFT reduction from reusing the prefix, in milliseconds.
    #[must_use]
    pub fn estimated_saving_ms(&self) -> f64 {
        self.reused_tokens as f64 * self.ms_per_forwarded_token
    }
}

/// One request's bookkeeping.
#[derive(Debug, Clone)]
struct Request {
    id: RequestId,
    prompt: Vec<u32>,
    output: Vec<u32>,
    max_new_tokens: usize,
    prefill_cursor: usize,
    phase: RequestPhase,
    preemptions: u32,
    prefix_hit_tokens: usize,
    swap_bytes: usize,
    arrival_step: u64,
}

impl Request {
    fn history_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    fn history(&self) -> Vec<u32> {
        let mut history = Vec::with_capacity(self.history_len());
        history.extend_from_slice(&self.prompt);
        history.extend_from_slice(&self.output);
        history
    }

    fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            id: self.id,
            phase: self.phase,
            prompt_tokens: self.prompt.len(),
            consumed_tokens: self.history_len(),
            generated_tokens: self.output.len(),
            max_new_tokens: self.max_new_tokens,
            prefill_cursor: self.prefill_cursor,
            preemptions: self.preemptions,
            prefix_hit_tokens: self.prefix_hit_tokens,
            swap_bytes: self.swap_bytes,
            arrival_step: self.arrival_step,
        }
    }
}

/// Pages a token count occupies in the 64-token paged pool.
#[must_use]
pub const fn pages_for(tokens: usize) -> usize {
    tokens.div_ceil(KV_PAGE_TOKENS)
}

/// The continuous batcher: admission, mixed prefill/decode steps, preemption and prefix reuse.
#[derive(Debug)]
pub struct ContinuousBatcher {
    config: BatchConfig,
    kv: PagedKvCache,
    requests: Vec<Request>,
    waiting: VecDeque<RequestId>,
    running: Vec<RequestId>,
    swapped: VecDeque<RequestId>,
    step: u64,
    next_id: u64,
    decode_cursor: usize,
    prefill_cursor: usize,
    stats: BatchStats,
}

impl ContinuousBatcher {
    /// A batcher over a fresh paged-KV pool.
    ///
    /// # Errors
    /// Returns the configuration errors, or the paged cache's geometry error when the pool is
    /// invalid.
    pub fn new(config: BatchConfig) -> Result<Self, SchedulerError> {
        config.validate()?;
        let geometry = PagedGeometry::new(config.kv_pages, config.kv_tails)?;
        Ok(Self {
            config,
            kv: PagedKvCache::new(geometry),
            requests: Vec::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: VecDeque::new(),
            step: 0,
            next_id: 0,
            decode_cursor: 0,
            prefill_cursor: 0,
            stats: BatchStats::default(),
        })
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// The next step number.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Queues a request for max_new_tokens tokens.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidArgument] for an empty prompt, a prompt longer than
    /// [BatchConfig::max_prompt_tokens], or a zero completion cap.
    pub fn submit(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<RequestId, SchedulerError> {
        if prompt.is_empty() {
            return Err(SchedulerError::InvalidArgument(
                "a batched request needs at least one prompt token".to_owned(),
            ));
        }
        if prompt.len() > self.config.max_prompt_tokens {
            return Err(SchedulerError::InvalidArgument(format!(
                "a {} token prompt exceeds the configured {} token maximum",
                prompt.len(),
                self.config.max_prompt_tokens
            )));
        }
        if max_new_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a batched request needs a positive completion cap".to_owned(),
            ));
        }
        let id = RequestId(self.next_id);
        self.next_id += 1;
        self.requests.push(Request {
            id,
            prompt,
            output: Vec::new(),
            max_new_tokens,
            prefill_cursor: 0,
            phase: RequestPhase::Waiting,
            preemptions: 0,
            prefix_hit_tokens: 0,
            swap_bytes: 0,
            arrival_step: self.step,
        });
        self.waiting.push_back(id);
        self.stats.submitted += 1;
        Ok(id)
    }

    /// One request, if it is known.
    #[must_use]
    pub fn request(&self, id: RequestId) -> Option<RequestSnapshot> {
        self.lookup(id).map(Request::snapshot)
    }

    /// Requests currently running.
    #[must_use]
    pub fn running(&self) -> &[RequestId] {
        &self.running
    }

    /// Requests currently waiting.
    #[must_use]
    pub fn waiting(&self) -> &VecDeque<RequestId> {
        &self.waiting
    }

    /// Requests currently swapped out.
    #[must_use]
    pub fn swapped(&self) -> &VecDeque<RequestId> {
        &self.swapped
    }

    /// Cumulative counters.
    #[must_use]
    pub const fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// A snapshot of every request and counter.
    #[must_use]
    pub fn snapshot(&self) -> BatchSnapshot {
        BatchSnapshot {
            requests: self.requests.iter().map(Request::snapshot).collect(),
            stats: self.stats.clone(),
            kv: self.kv.stats(),
        }
    }

    /// Checks the batcher's bookkeeping and the paged cache's invariants.
    ///
    /// # Errors
    /// Returns [SchedulerError::Invariant] for a request in the wrong list or phase, or the
    /// paged cache's error when its own check fails.
    pub fn check_invariants(&self) -> Result<(), SchedulerError> {
        self.kv.check_invariants()?;
        let mut listed = 0usize;
        for request in &self.requests {
            match request.phase {
                RequestPhase::Waiting => {
                    if !self.waiting.contains(&request.id) {
                        return Err(invariant(format!(
                            "request {} is Waiting but not queued",
                            request.id.0
                        )));
                    }
                    listed += 1;
                }
                RequestPhase::Prefill | RequestPhase::Decode => {
                    if !self.running.contains(&request.id) {
                        return Err(invariant(format!(
                            "request {} is {:?} but not running",
                            request.id.0, request.phase
                        )));
                    }
                    listed += 1;
                }
                RequestPhase::Swapped => {
                    if !self.swapped.contains(&request.id) {
                        return Err(invariant(format!(
                            "request {} is Swapped but not queued",
                            request.id.0
                        )));
                    }
                    listed += 1;
                }
                RequestPhase::Finished => {}
            }
        }
        let expected = self.waiting.len() + self.running.len() + self.swapped.len();
        if listed != expected {
            return Err(invariant(format!(
                "{listed} requests with a live phase, {expected} in the queues"
            )));
        }
        Ok(())
    }

    /// Plans one step.
    ///
    /// # Errors
    /// Returns a [SchedulerError] when the paged cache refuses an operation the plan should have
    /// made room for; that is an internal inconsistency, not a normal refusal.
    pub fn plan_step(&mut self) -> Result<BatchStep, SchedulerError> {
        let mut step = BatchStep::new(self.step, self.config.max_num_batched_tokens);
        self.swap_in_pass(&mut step)?;
        let mut budget = self.config.max_num_batched_tokens;
        if self.config.decode_first {
            self.decode_pass(&mut step, &mut budget)?;
        }
        self.admission_pass(&mut step)?;
        self.prefill_pass(&mut step, &mut budget)?;
        if !self.config.decode_first {
            self.decode_pass(&mut step, &mut budget)?;
        }
        self.step += 1;
        self.stats.steps += 1;
        self.stats.host_forwards += step.host_forwards() as u64;
        self.stats.device_launches += step.device_forward_launches() as u64;
        self.stats.prefill_tokens += step.prefill_tokens as u64;
        self.stats.decode_tokens += step.decode_tokens as u64;
        self.stats.prefix_reused_tokens += step.prefix_reused_tokens as u64;
        self.stats.kv_evictions = self.kv.stats().evictions;
        self.stats.max_running = self.stats.max_running.max(self.running.len());
        Ok(step)
    }

    /// Records the tokens the executor produced for step and frees finished requests.
    ///
    /// produced is one (request, token) per token-producing op of the step, in op order. The
    /// tokens are appended to their requests' history and to the paged cache, so a later
    /// preemption can replay the real sequence.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidArgument] when the produced list does not match the
    /// step's token-producing ops, and a paging error when the cache refuses a token the plan
    /// did not reserve a page for.
    pub fn accept_step(
        &mut self,
        step: &BatchStep,
        produced: &[(RequestId, u32)],
    ) -> Result<Vec<RequestId>, SchedulerError> {
        let producers: Vec<&ScheduledOp> =
            step.ops.iter().filter(|op| op.produces_token()).collect();
        if producers.len() != produced.len() {
            return Err(SchedulerError::InvalidArgument(format!(
                "step {} produced {} tokens for {} token-producing ops",
                step.step,
                produced.len(),
                producers.len()
            )));
        }
        let mut finished = Vec::new();
        for (op, (seq, token)) in producers.iter().zip(produced) {
            if op.seq() != *seq {
                return Err(SchedulerError::InvalidArgument(format!(
                    "step {} produced a token for request {} where request {} was expected",
                    step.step,
                    seq.0,
                    op.seq().0
                )));
            }
            if self.lookup(*seq).is_none() {
                return Err(SchedulerError::InvalidArgument(format!(
                    "request {} is not known",
                    seq.0
                )));
            }
            self.kv.append(seq.0, &[*token])?;
            let request = self.lookup_mut(*seq).expect("checked above");
            request.output.push(*token);
            if matches!(
                op,
                ScheduledOp::PrefillChunk {
                    emits_token: true,
                    ..
                }
            ) && request.phase == RequestPhase::Prefill
            {
                request.phase = RequestPhase::Decode;
            }
            if request.output.len() >= request.max_new_tokens {
                request.phase = RequestPhase::Finished;
                request.swap_bytes = 0;
                self.running.retain(|candidate| *candidate != *seq);
                self.finish(*seq)?;
                finished.push(*seq);
            }
        }
        self.stats.finished += finished.len() as u64;
        self.stats.kv_evictions = self.kv.stats().evictions;
        Ok(finished)
    }

    /// Cancels a live request, freeing its KV like a finished one.
    ///
    /// # Errors
    /// Returns [SchedulerError::InvalidArgument] for an unknown request, or when the paged
    /// cache no longer knows it.
    pub fn cancel(&mut self, id: RequestId) -> Result<(), SchedulerError> {
        let phase = self
            .lookup(id)
            .ok_or_else(|| {
                SchedulerError::InvalidArgument(format!("request {} is not known", id.0))
            })?
            .phase;
        match phase {
            RequestPhase::Finished => {}
            RequestPhase::Waiting => {
                self.waiting.retain(|candidate| *candidate != id);
                self.lookup_mut(id).expect("known").phase = RequestPhase::Finished;
            }
            RequestPhase::Swapped => {
                self.swapped.retain(|candidate| *candidate != id);
                self.lookup_mut(id).expect("known").phase = RequestPhase::Finished;
            }
            RequestPhase::Prefill | RequestPhase::Decode => {
                self.running.retain(|candidate| *candidate != id);
                self.finish(id)?;
                self.lookup_mut(id).expect("known").phase = RequestPhase::Finished;
            }
        }
        Ok(())
    }

    fn lookup(&self, id: RequestId) -> Option<&Request> {
        self.requests.get(usize::try_from(id.0).ok()?)
    }

    fn lookup_mut(&mut self, id: RequestId) -> Option<&mut Request> {
        self.requests.get_mut(usize::try_from(id.0).ok()?)
    }

    fn finish(&mut self, id: RequestId) -> Result<(), SchedulerError> {
        self.kv.free_sequence(id.0)?;
        Ok(())
    }

    /// Reattaches swapped requests, oldest first, while their pages fit.
    fn swap_in_pass(&mut self, step: &mut BatchStep) -> Result<(), SchedulerError> {
        while let Some(&id) = self.swapped.front() {
            let history = self.lookup(id).map(Request::history).unwrap_or_default();
            if history.is_empty() {
                self.swapped.pop_front();
                continue;
            }
            let need = pages_for(history.len());
            if (self.kv.stats().pages_free as usize) < need {
                break;
            }
            let report = self.kv.append(id.0, &history)?;
            let shared = report.shared * KV_PAGE_TOKENS;
            let resumes_without_prefill = self.config.preemption.resumes_without_prefill();
            let request = self.lookup_mut(id).expect("queued request is known");
            request.prefix_hit_tokens += shared;
            request.swap_bytes = 0;
            if resumes_without_prefill {
                request.prefill_cursor = history.len();
                request.phase = RequestPhase::Decode;
            } else {
                request.prefill_cursor = shared.min(history.len().saturating_sub(1));
                request.phase = RequestPhase::Prefill;
            }
            self.running.push(id);
            self.swapped.pop_front();
            step.ops.push(ScheduledOp::SwappedIn {
                seq: id,
                restored_tokens: history.len(),
                from_prefix_cache: shared,
            });
            step.swapped_in.push(id);
            self.stats.swap_ins += 1;
        }
        Ok(())
    }

    /// The decode phase: one token per running decoding sequence, in rotation.
    fn decode_pass(
        &mut self,
        step: &mut BatchStep,
        budget: &mut usize,
    ) -> Result<(), SchedulerError> {
        let count = self.running.len();
        if count == 0 {
            return Ok(());
        }
        let start = self.decode_cursor % count;
        let mut order: Vec<RequestId> = self.running.clone();
        order.rotate_left(start);
        let mut served = false;
        for id in order {
            if *budget == 0 {
                break;
            }
            if !self.running.contains(&id)
                || self.lookup(id).map(|request| request.phase) != Some(RequestPhase::Decode)
            {
                continue;
            }
            let position = self
                .lookup(id)
                .expect("running request is known")
                .history_len();
            if position.is_multiple_of(KV_PAGE_TOKENS) && !self.ensure_free_pages(step, 1)? {
                continue;
            }
            step.ops.push(ScheduledOp::Decode { seq: id, position });
            step.decode_tokens += 1;
            *budget -= 1;
            served = true;
        }
        if served {
            self.decode_cursor = (start + 1) % count;
        }
        Ok(())
    }

    /// Admission: FIFO from the waiting queue while a slot and pages are available.
    fn admission_pass(&mut self, step: &mut BatchStep) -> Result<(), SchedulerError> {
        while self.running.len() < self.config.max_num_seqs {
            let Some(&id) = self.waiting.front() else {
                break;
            };
            let prompt = self
                .lookup(id)
                .expect("queued request is known")
                .prompt
                .clone();
            let need = pages_for(prompt.len());
            if (self.kv.stats().pages_free as usize) < need {
                if !self.preempt_one(step)? {
                    break;
                }
                continue;
            }
            self.waiting.pop_front();
            self.kv.add_sequence(id.0)?;
            let report = self.kv.append(id.0, &prompt)?;
            let hits = report.shared * KV_PAGE_TOKENS;
            let reuse = self.config.prefix_reuse;
            let request = self.lookup_mut(id).expect("queued request is known");
            request.phase = RequestPhase::Prefill;
            request.prefix_hit_tokens = hits;
            request.prefill_cursor = if reuse {
                hits.min(request.prompt.len().saturating_sub(1))
            } else {
                0
            };
            self.running.push(id);
            step.admitted.push(id);
            self.stats.admitted += 1;
            if reuse && hits > 0 {
                step.ops.push(ScheduledOp::PrefixReuse {
                    seq: id,
                    tokens: hits,
                });
                step.prefix_reused_tokens += hits;
            }
        }
        Ok(())
    }

    /// The prefill phase: at most one token-budget chunk per prefilling sequence, in rotation.
    fn prefill_pass(
        &mut self,
        step: &mut BatchStep,
        budget: &mut usize,
    ) -> Result<(), SchedulerError> {
        let count = self.running.len();
        if count == 0 {
            return Ok(());
        }
        let start = self.prefill_cursor % count;
        let chunk_tokens = self.config.chunk_tokens();
        let mut order: Vec<RequestId> = self.running.clone();
        order.rotate_left(start);
        for id in order {
            if *budget == 0 {
                break;
            }
            if !self.running.contains(&id) {
                continue;
            }
            let (target, cursor) = match self.lookup(id) {
                Some(request) if request.phase == RequestPhase::Prefill => {
                    (request.history_len(), request.prefill_cursor)
                }
                _ => continue,
            };
            if cursor >= target {
                continue;
            }
            let take = (target - cursor).min(chunk_tokens).min(*budget);
            if take == 0 {
                continue;
            }
            let range = cursor..cursor + take;
            let emits_token = range.end == target;
            if emits_token
                && target.is_multiple_of(KV_PAGE_TOKENS)
                && !self.ensure_free_pages(step, 1)?
            {
                continue;
            }
            step.ops.push(ScheduledOp::PrefillChunk {
                seq: id,
                range: range.clone(),
                emits_token,
            });
            step.prefill_tokens += take;
            *budget -= take;
            self.lookup_mut(id)
                .expect("running request is known")
                .prefill_cursor = range.end;
        }
        self.prefill_cursor = (start + 1) % count;
        Ok(())
    }

    /// Makes sure needed pages are free, preempting victims when they are not.
    ///
    /// A victim is never a request that already has an operation in this step, so a planned
    /// forward is never invalidated by a later preemption in the same step. Returns whether the
    /// pages are available afterwards.
    fn ensure_free_pages(
        &mut self,
        step: &mut BatchStep,
        needed: usize,
    ) -> Result<bool, SchedulerError> {
        while (self.kv.stats().pages_free as usize) < needed {
            if !self.preempt_one(step)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Preempts one victim, last-admitted first.
    fn preempt_one(&mut self, step: &mut BatchStep) -> Result<bool, SchedulerError> {
        let planned: Vec<RequestId> = step.ops.iter().map(ScheduledOp::seq).collect();
        let victim = self
            .running
            .iter()
            .rev()
            .copied()
            .find(|id| !planned.contains(id) && !step.preempted.contains(id));
        let Some(victim) = victim else {
            return Ok(false);
        };
        let history = self
            .lookup(victim)
            .map(Request::history)
            .unwrap_or_default();
        let report = self.kv.preempt(victim.0)?;
        self.running.retain(|candidate| *candidate != victim);
        let swap_bytes = history.len() * self.config.swap_bytes_per_token + SWA_TAIL_BYTES;
        let request = self.lookup_mut(victim).expect("running request is known");
        request.phase = RequestPhase::Swapped;
        request.preemptions += 1;
        request.swap_bytes = swap_bytes;
        self.swapped.push_back(victim);
        step.ops.push(ScheduledOp::Preempted {
            seq: victim,
            reclaimed_pages: report.freed_pages,
            retained_blocks: report.retained_blocks,
        });
        step.preempted.push(victim);
        self.stats.preemptions += 1;
        Ok(true)
    }
}

fn invariant(detail: String) -> SchedulerError {
    SchedulerError::Invariant(detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_batched_prefill_policy_with_the_spec_budget() {
        let config = BatchConfig::default();
        config.validate().expect("the defaults are valid");
        assert_eq!(config.prefill_mode, PrefillMode::Batched);
        assert!(!config.prefill_mode.is_token_loop());
        assert_eq!(config.chunk_tokens(), 4096);
        assert_eq!(config.max_num_seqs, 16);
        assert_eq!(config.max_num_batched_tokens, 8192);
        assert_eq!(config.max_prompt_tokens, MAX_CONTEXT_TOKENS);
        assert_eq!(config.kv_tails, 16);
        assert!(config.decode_first);
        assert!(config.prefix_reuse);
        assert_eq!(config.preemption, PreemptionPolicy::Recompute);
        assert_eq!(config.swap_bytes_per_token, 890);
        let compat = config.with_prefill_mode(PrefillMode::TokenLoop);
        assert_eq!(compat.chunk_tokens(), TOKEN_LOOP_CHUNK_TOKENS);
        assert_eq!(BatchConfig::default().prefill_mode, PrefillMode::Batched);
    }

    #[test]
    fn the_mode_parser_accepts_the_documented_spellings_and_refuses_others() {
        for value in ["batched", "chunk", "chunked", "BATCHED", " chunk "] {
            assert_eq!(
                PrefillMode::from_env_value(value),
                Some(PrefillMode::Batched),
                "{value:?}"
            );
        }
        for value in ["token-loop", "token_loop", "loop", "TOKEN"] {
            assert_eq!(
                PrefillMode::from_env_value(value),
                Some(PrefillMode::TokenLoop),
                "{value:?}"
            );
        }
        assert_eq!(PrefillMode::from_env_value("decode"), None);
        assert_eq!(PrefillMode::from_env_value(""), None);
    }

    #[test]
    fn validation_names_each_knob_it_rejects() {
        fn max_num_seqs(c: &mut BatchConfig) {
            c.max_num_seqs = 0;
        }
        fn max_tokens(c: &mut BatchConfig) {
            c.max_num_batched_tokens = 0;
        }
        fn chunk_zero(c: &mut BatchConfig) {
            c.prefill_chunk_tokens = 0;
        }
        fn chunk_over_budget(c: &mut BatchConfig) {
            c.prefill_chunk_tokens = 16384;
        }
        fn prompt_zero(c: &mut BatchConfig) {
            c.max_prompt_tokens = 0;
        }
        fn prompt_over_model(c: &mut BatchConfig) {
            c.max_prompt_tokens = MAX_CONTEXT_TOKENS + 1;
        }
        fn pages_zero(c: &mut BatchConfig) {
            c.kv_pages = 0;
        }
        fn tails_zero(c: &mut BatchConfig) {
            c.kv_tails = 0;
        }
        fn tails_under_seqs(c: &mut BatchConfig) {
            c.kv_tails = 2;
        }
        fn swap_zero(c: &mut BatchConfig) {
            c.swap_bytes_per_token = 0;
        }
        type Mutator = fn(&mut BatchConfig);
        let cases: [(&str, Mutator); 10] = [
            ("max_num_seqs", max_num_seqs),
            ("max_num_batched_tokens", max_tokens),
            ("prefill_chunk_tokens", chunk_zero),
            ("exceeds max_num_batched_tokens", chunk_over_budget),
            ("max_prompt_tokens", prompt_zero),
            ("model limit", prompt_over_model),
            ("kv_pages", pages_zero),
            ("kv_tails", tails_zero),
            ("fewer than max_num_seqs", tails_under_seqs),
            ("swap_bytes_per_token", swap_zero),
        ];
        for (needle, mutate) in cases {
            let mut config = BatchConfig::default();
            mutate(&mut config);
            let error = config
                .validate()
                .expect_err("the mutated configuration must be rejected");
            assert!(
                error.to_string().contains(needle),
                "expected {needle:?} in {error}"
            );
        }
    }

    #[test]
    fn the_token_budget_plan_is_contiguous_and_never_exceeds_the_budget() {
        for (prompt, chunk, budget) in [
            (10_000usize, 4096usize, 8192usize),
            (37, 4096, 8192),
            (9, 2, 2),
            (5, 1, 1),
            (1, 4096, 8192),
        ] {
            let plan = TokenBudgetPlan::plan(prompt, 0, chunk, budget).expect("plan");
            assert_eq!(plan.total_tokens(), prompt, "{prompt} tokens are covered");
            assert!(plan.chunks() >= 1);
            let mut next = 0usize;
            for range in plan.ranges() {
                assert_eq!(range.start, next, "chunks are contiguous and ordered");
                assert!(!range.is_empty());
                assert!(range.end - range.start <= chunk);
                assert!(range.end - range.start <= budget);
                next = range.end;
            }
            assert_eq!(next, prompt);
        }
        let plan = TokenBudgetPlan::plan(10_000, 0, 4096, 8192).expect("plan");
        assert_eq!(plan.chunks(), 3, "10000 tokens at 4096 per chunk");
        assert_eq!(plan.last_chunk_tokens(), 10_000 - 2 * 4096);
        let tail = TokenBudgetPlan::plan(10_000, 9_000, 4096, 8192).expect("plan");
        assert_eq!(tail.ranges().len(), 1);
        assert_eq!(tail.ranges()[0], 9_000..10_000);
    }

    #[test]
    fn the_token_budget_plan_refuses_impossible_shapes() {
        assert!(TokenBudgetPlan::plan(0, 0, 4096, 8192).is_err());
        assert!(TokenBudgetPlan::plan(10, 11, 4096, 8192).is_err());
        assert!(TokenBudgetPlan::plan(10, 0, 0, 8192).is_err());
        assert!(TokenBudgetPlan::plan(10, 0, 8192, 4096).is_err());
        assert!(TokenBudgetPlan::plan(10, 0, 4096, 0).is_err());
    }

    #[test]
    fn the_ttft_estimate_is_labelled_and_bounded() {
        let batched = PrefillCostEstimate::estimate(PrefillMode::Batched, 37, 0).expect("estimate");
        assert_eq!(batched.forwarded_tokens, 37);
        assert!((batched.estimated_ttft_ms() - 37.0 * 1179.0).abs() < 1e-9);
        let reused = PrefillCostEstimate::estimate(PrefillMode::Batched, 37, 32).expect("estimate");
        assert_eq!(reused.forwarded_tokens, 5);
        assert!(reused.estimated_ttft_ms() < batched.estimated_ttft_ms());
        assert!((reused.estimated_saving_ms() - 32.0 * 1179.0).abs() < 1e-9);
        let loop_estimate =
            PrefillCostEstimate::estimate(PrefillMode::TokenLoop, 37, 0).expect("estimate");
        assert!((loop_estimate.estimated_ttft_ms() - 37.0 * 945.0).abs() < 1e-9);
        assert!(PrefillCostEstimate::estimate(PrefillMode::Batched, 10, 11).is_err());
    }

    #[test]
    fn a_step_counts_one_forward_per_chunk_and_fuses_decodes() {
        let mut step = BatchStep::new(3, 8192);
        step.ops.push(ScheduledOp::PrefillChunk {
            seq: RequestId(0),
            range: 0..4,
            emits_token: false,
        });
        step.ops.push(ScheduledOp::PrefillChunk {
            seq: RequestId(0),
            range: 4..7,
            emits_token: true,
        });
        step.ops.push(ScheduledOp::Decode {
            seq: RequestId(1),
            position: 10,
        });
        step.ops.push(ScheduledOp::PrefixReuse {
            seq: RequestId(2),
            tokens: 128,
        });
        assert_eq!(step.host_forwards(), 3);
        assert_eq!(step.device_forward_launches(), 1);
        assert_eq!(step.ops_for(RequestId(0)).len(), 2);
        assert!(step.ops[1].produces_token());
        assert!(!step.ops[0].produces_token());
        assert!(step.ops[2].produces_token());
        assert!(!step.ops[3].produces_token());
        let empty = BatchStep::new(0, 1);
        assert!(empty.is_empty());
        assert_eq!(empty.host_forwards(), 0);
        assert_eq!(empty.device_forward_launches(), 0);
    }

    #[test]
    fn pages_are_whole_sixty_four_token_blocks() {
        assert_eq!(pages_for(0), 0);
        assert_eq!(pages_for(1), 1);
        assert_eq!(pages_for(64), 1);
        assert_eq!(pages_for(65), 2);
        assert_eq!(KV_PAGE_TOKENS, 64);
    }
}
