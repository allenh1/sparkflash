//! One-forward-per-chunk batched prefill execution over the reference engine.
//!
//! This is the executor half of the scheduler's batched prefill policy (WS-F). The scheduler
//! plans token-budget chunks; this module runs them as one [InferenceEngine::prefill] call per
//! chunk and counts the forwards, so a test can assert the per-token loop is gone. Nothing here
//! is a device path: the reference engine runs one sequence at a time on f32 and this module
//! does not fuse a batch. What it does fix is the execution contract the device path has to
//! keep: a contiguous prompt range is forwarded exactly once, every prompt token is written to
//! the KV exactly once, and the outputs are bit-identical to the whole-prompt call and to the
//! pre-batch-17 one-token loop.
//!
//! The compatibility path is [BatchedPrefill::token_loop]: one decode-shaped forward per prompt
//! token, the shape the CLI shipped before batch 17 and still ships as its default until the
//! endpoint chunk overhead is understood. It is one call per token by construction, which is
//! exactly what [PrefillRun::forwards] exists to show.
//!
//! Prefix caching meets this module through [SequenceState::fork]: a state that already holds a
//! cached prefix can run only the uncached suffix, and [BatchedPrefill::prefill_ranges] refuses
//! ranges that do not start at the state's consumed position, so a cached prefix can never be
//! silently forwarded twice.

use std::ops::Range;

use crate::forward::{InferenceEngine, SequenceState, StepOutput};
use crate::linalg::EngineError;

/// What one prefill execution did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillRun {
    /// Chunks forwarded: the plan's range count.
    pub chunks: usize,
    /// Engine calls made: one per chunk. On the batched path this never equals the prompt token
    /// count unless the prompt is shorter than two chunks.
    pub forwards: usize,
    /// Prompt tokens written to the KV exactly once, in total.
    pub tokens_written: usize,
}

/// A batched prefill executor over one engine.
pub struct BatchedPrefill<'e> {
    engine: &'e InferenceEngine,
}

impl<'e> BatchedPrefill<'e> {
    /// An executor over engine.
    #[must_use]
    pub fn new(engine: &'e InferenceEngine) -> Self {
        Self { engine }
    }

    /// Forwards prompt\[start..\] in token-budget chunks.
    ///
    /// start is the state's already-consumed position, so the budget applies to the uncached
    /// suffix. chunk_tokens is the largest chunk and max_num_batched_tokens the step budget; a
    /// chunk is bounded by the smaller of the two, and a configured chunk larger than the step
    /// budget is refused rather than silently split into something else.
    ///
    /// # Errors
    /// [EngineError::Shape] when the prompt is empty, there is nothing left to prefill, the
    /// state does not hold a prefix of the prompt, or either size is zero or the chunk exceeds
    /// the budget.
    pub fn prefill_token_budget(
        &self,
        state: &mut SequenceState,
        prompt: &[u32],
        chunk_tokens: usize,
        max_num_batched_tokens: usize,
    ) -> Result<(StepOutput, PrefillRun), EngineError> {
        if chunk_tokens == 0 {
            return Err(shape(
                "batched prefill",
                "the chunk size must be at least one token".to_owned(),
            ));
        }
        if max_num_batched_tokens == 0 || chunk_tokens > max_num_batched_tokens {
            return Err(shape(
                "batched prefill",
                format!(
                    "a {chunk_tokens} token chunk does not fit a {max_num_batched_tokens} token budget"
                ),
            ));
        }
        let ranges = token_budget_ranges(
            state.len(),
            prompt.len(),
            chunk_tokens,
            max_num_batched_tokens,
        )?;
        self.prefill_ranges(state, prompt, &ranges)
    }

    /// Forwards the explicit ranges of prompt on state.
    ///
    /// The ranges must start at state's consumed position, be contiguous and non-empty, and end
    /// at the prompt's end. One [InferenceEngine::prefill] call runs per range.
    ///
    /// # Errors
    /// [EngineError::Shape] for any range that violates those rules or a state that does not
    /// hold a prefix of the prompt.
    pub fn prefill_ranges(
        &self,
        state: &mut SequenceState,
        prompt: &[u32],
        ranges: &[Range<usize>],
    ) -> Result<(StepOutput, PrefillRun), EngineError> {
        if prompt.is_empty() {
            return Err(shape(
                "batched prefill",
                "a prefill needs at least one prompt token".to_owned(),
            ));
        }
        if state.len() > prompt.len() {
            return Err(shape(
                "batched prefill",
                format!(
                    "the sequence already holds {} tokens, more than the {} token prompt",
                    state.len(),
                    prompt.len()
                ),
            ));
        }
        if state.tokens() != &prompt[..state.len()] {
            return Err(shape(
                "batched prefill",
                "the sequence's consumed tokens are not a prefix of the prompt".to_owned(),
            ));
        }
        if ranges.is_empty() {
            return Err(shape(
                "batched prefill",
                "a batched prefill needs at least one chunk".to_owned(),
            ));
        }
        let mut cursor = state.len();
        for range in ranges {
            if range.start != cursor {
                return Err(shape(
                    "batched prefill",
                    format!(
                        "chunk {}..{} does not continue from the consumed position {cursor}",
                        range.start, range.end
                    ),
                ));
            }
            if range.is_empty() {
                return Err(shape(
                    "batched prefill",
                    format!("chunk {}..{} is empty", range.start, range.end),
                ));
            }
            cursor = range.end;
        }
        if cursor != prompt.len() {
            return Err(shape(
                "batched prefill",
                format!(
                    "the chunks end at {cursor} but the prompt ends at {}",
                    prompt.len()
                ),
            ));
        }
        let mut output = None;
        let mut forwards = 0usize;
        let mut tokens_written = 0usize;
        for range in ranges {
            output = Some(self.engine.prefill(state, &prompt[range.clone()])?);
            forwards += 1;
            tokens_written += range.end - range.start;
        }
        let output = output.expect("the ranges are non-empty");
        Ok((
            output,
            PrefillRun {
                chunks: ranges.len(),
                forwards,
                tokens_written,
            },
        ))
    }

    /// The pre-batch-17 compatibility path: one decode-shaped forward per prompt token.
    ///
    /// This is the shape the endpoint shipped (and still ships by default) before the token
    /// budget chunking: the prompt is fed one token at a time, with the growing prefix as the
    /// sequence's consumed tokens. It exists so the old path can be reproduced and compared, and
    /// so the batched path has something to be bit-identical to.
    ///
    /// # Errors
    /// As [Self::prefill_ranges]: the prompt must be non-empty and continue the state.
    pub fn token_loop(
        &self,
        state: &mut SequenceState,
        prompt: &[u32],
    ) -> Result<(StepOutput, PrefillRun), EngineError> {
        let start = state.len();
        if start >= prompt.len() {
            return Err(shape(
                "token-loop prefill",
                format!(
                    "the sequence already holds {start} of the {} prompt tokens",
                    prompt.len()
                ),
            ));
        }
        let ranges: Vec<Range<usize>> = (start..prompt.len())
            .map(|index| index..index + 1)
            .collect();
        self.prefill_ranges(state, prompt, &ranges)
    }
}

/// The contiguous token ranges a batched prefill of start..end uses.
///
/// # Errors
/// [EngineError::Shape] for an empty range, a zero chunk, or a chunk larger than the step
/// budget.
pub fn token_budget_ranges(
    start: usize,
    end: usize,
    chunk_tokens: usize,
    max_num_batched_tokens: usize,
) -> Result<Vec<Range<usize>>, EngineError> {
    if start >= end {
        return Err(shape(
            "batched prefill",
            format!("there is nothing to prefill: the sequence holds {start} of {end} tokens"),
        ));
    }
    if chunk_tokens == 0 {
        return Err(shape(
            "batched prefill",
            "the chunk size must be at least one token".to_owned(),
        ));
    }
    if max_num_batched_tokens == 0 || chunk_tokens > max_num_batched_tokens {
        return Err(shape(
            "batched prefill",
            format!(
                "a {chunk_tokens} token chunk does not fit a {max_num_batched_tokens} token budget"
            ),
        ));
    }
    let mut ranges = Vec::new();
    let mut begin = start;
    while begin < end {
        let next = (begin + chunk_tokens).min(end);
        ranges.push(begin..next);
        begin = next;
    }
    Ok(ranges)
}

fn shape(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}
