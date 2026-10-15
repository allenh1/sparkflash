//! Spec 5.6.2: short-request prefill bypass.
//!
//! A long cold prefill runs in `chunk_tokens` chunks and, between two chunks, the engine runs
//! `prefill_decode_interval` decode steps. A request whose prompt is at most
//! `DSV41_PREFILL_BYPASS = 256` tokens does not queue behind those chunks: it is admitted in
//! the next inter-chunk gap and decoded alongside that gap's decode step, which is what turns
//! the interactive TTFT from the length of the remaining prefilled whole chunks into at most
//! one chunk.
//!
//! This module is the state machine that decides that timeline, in engine steps, plus the
//! cost model that turns the timeline into seconds. The step timeline is the contract; the
//! seconds are derived from it and from two costs, one of which the specification supplies
//! and one of which is an explicit assumption (see `TimelineCost`).

use std::ops::Range;

use crate::RequestId;
use crate::config::{PREFILL_REFERENCE_SECONDS, SchedulerConfig};
use crate::error::SchedulerError;

/// Why a request was treated the way it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionKind {
    /// The prompt fit the bypass threshold, so the request cut in between two chunks.
    Bypassed,
    /// The prompt was too long, so the request waits for the background prefill to finish.
    QueuedBehindPrefill,
}

/// One request's place in the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    /// Request this admission belongs to.
    pub request: RequestId,
    /// Step at which the request arrived.
    pub arrival_step: u64,
    /// Step whose inter-chunk gap carries the request's own prefill.
    pub admitted_step: u64,
    /// Step whose decode produces the request's first token.
    pub first_token_step: u64,
    /// Prefill slots the request's own prompt occupies.
    pub prefill_slots: usize,
    /// Which rule produced this admission.
    pub kind: AdmissionKind,
}

impl Admission {
    /// Steps between arrival and the first token.
    #[must_use]
    pub const fn ttft_steps(&self) -> u64 {
        self.first_token_step - self.arrival_step
    }

    /// True when the request cut in front of the background prefill.
    #[must_use]
    pub const fn bypassed(&self) -> bool {
        matches!(self.kind, AdmissionKind::Bypassed)
    }
}

/// The background cold prefill's step schedule.
///
/// Invariant: step `i * stride()` is prefill chunk `i`, and the `decode_interval` steps after
/// it are that chunk's inter-chunk decode window. Every decision this module makes is a
/// position in that schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillTimeline {
    prompt_tokens: usize,
    chunk_tokens: usize,
    chunks: usize,
    decode_interval: usize,
}

impl PrefillTimeline {
    /// The schedule for a background prefill of `prompt_tokens` tokens.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidConfig` for an invalid configuration and
    /// `SchedulerError::InvalidArgument` for an empty prompt.
    pub fn new(prompt_tokens: usize, config: &SchedulerConfig) -> Result<Self, SchedulerError> {
        config.validate()?;
        if prompt_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "a background prefill needs at least one prompt token".to_owned(),
            ));
        }
        Ok(Self {
            prompt_tokens,
            chunk_tokens: config.chunk_tokens,
            chunks: prompt_tokens.div_ceil(config.chunk_tokens),
            decode_interval: config.prefill_decode_interval,
        })
    }

    /// Prompt tokens the background prefill covers.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Chunk size the prefill runs in.
    #[must_use]
    pub const fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    /// Chunks the prompt is prefilled in.
    #[must_use]
    pub const fn chunks(&self) -> usize {
        self.chunks
    }

    /// Decode steps between two chunks.
    #[must_use]
    pub const fn decode_interval(&self) -> usize {
        self.decode_interval
    }

    /// Steps one (chunk, window) period spans.
    #[must_use]
    pub const fn stride(&self) -> u64 {
        (self.decode_interval + 1) as u64
    }

    /// True when `step` runs a prefill chunk rather than a decode.
    #[must_use]
    pub const fn is_chunk_step(&self, step: u64) -> bool {
        step.is_multiple_of(self.stride())
    }

    /// The index of the chunk that runs at `step`, when `step` is a chunk step.
    #[must_use]
    pub const fn chunk_index(&self, step: u64) -> Option<usize> {
        if !self.is_chunk_step(step) {
            return None;
        }
        let index = (step / self.stride()) as usize;
        if index < self.chunks {
            Some(index)
        } else {
            None
        }
    }

    /// The step prefill chunk `index` runs at.
    ///
    /// # Panics
    /// Panics when `index >= chunks()`; callers iterate over `0..chunks()`.
    #[must_use]
    pub fn chunk_step(&self, index: usize) -> u64 {
        assert!(index < self.chunks, "chunk {index} is outside the prefill");
        index as u64 * self.stride()
    }

    /// The inter-chunk decode window that follows chunk `index`.
    ///
    /// # Panics
    /// Panics when `index >= chunks()`.
    #[must_use]
    pub fn window_after(&self, index: usize) -> Range<u64> {
        let start = self.chunk_step(index) + 1;
        start..start + self.decode_interval as u64
    }

    /// The step the background prefill's last chunk runs at.
    #[must_use]
    pub fn last_chunk_step(&self) -> u64 {
        self.chunk_step(self.chunks - 1)
    }

    /// The step at which a request arriving at `arrival` gets decoded.
    ///
    /// The rule, in order: take the first inter-chunk window that still has a decode step
    /// after the arrival; if the arrival is already inside that window, admit it at the very
    /// next step, otherwise at the window's first step. An arrival after the last window
    /// decodes immediately, because the background prefill is over.
    #[must_use]
    pub fn next_admission_step(&self, arrival: u64) -> u64 {
        for index in 0..self.chunks {
            let window = self.window_after(index);
            let candidate = (arrival + 1).max(window.start);
            if candidate < window.end {
                return candidate;
            }
        }
        arrival + 1
    }

    /// Seconds the schedule spends between two steps, using `cost`.
    #[must_use]
    pub fn seconds_between(&self, from: u64, to: u64, cost: &TimelineCost) -> f64 {
        let mut chunk_steps = 0u64;
        let mut decode_steps = 0u64;
        for step in (from + 1)..=to {
            if self.is_chunk_step(step) {
                chunk_steps += 1;
            } else {
                decode_steps += 1;
            }
        }
        chunk_steps as f64 * cost.chunk_seconds + decode_steps as f64 * cost.decode_step_seconds
    }
}

/// The cost model that turns the step timeline into seconds.
///
/// `chunk_seconds` is derived from the specification's own reference prefill (78 s for a 400K
/// token cold prefill, i.e. 78 s divided by that prompt's chunk count). `decode_step_seconds`
/// is *assumed*: the specification gives 0.55 ms as the per-step saving from removing the
/// draft broadcast, not the step cost itself, so the default lives in
/// `SchedulerConfig::assumed_decode_step_seconds` and every test that reports seconds says
/// which number is assumed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineCost {
    /// Seconds one prefill chunk takes.
    pub chunk_seconds: f64,
    /// Seconds one decode step takes (assumed).
    pub decode_step_seconds: f64,
}

impl TimelineCost {
    /// The cost model implied by the specification's reference prefill.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for a non-positive or non-finite reference.
    pub fn from_spec(
        config: &SchedulerConfig,
        reference_seconds: f64,
        reference_prompt_tokens: usize,
    ) -> Result<Self, SchedulerError> {
        config.validate()?;
        if !reference_seconds.is_finite() || reference_seconds <= 0.0 {
            return Err(SchedulerError::InvalidArgument(
                "the reference prefill time must be positive".to_owned(),
            ));
        }
        if reference_prompt_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the reference prompt must not be empty".to_owned(),
            ));
        }
        let chunks = reference_prompt_tokens.div_ceil(config.chunk_tokens);
        Ok(Self {
            chunk_seconds: reference_seconds / chunks as f64,
            decode_step_seconds: config.assumed_decode_step_seconds,
        })
    }

    /// The specification's reference prefill as a cost model.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the configuration is invalid.
    pub fn reference(config: &SchedulerConfig) -> Result<Self, SchedulerError> {
        Self::from_spec(
            config,
            PREFILL_REFERENCE_SECONDS,
            crate::config::PREFILL_REFERENCE_PROMPT_TOKENS,
        )
    }

    /// Seconds the whole background prefill occupies, windows included.
    ///
    /// This is an upper bound on the specification's 78 s figure: the reference time covers
    /// the chunk compute, and the model adds the inter-chunk decode windows on top.
    #[must_use]
    pub fn background_seconds(&self, timeline: &PrefillTimeline) -> f64 {
        let chunks = timeline.chunks() as f64;
        chunks * self.chunk_seconds
            + chunks * timeline.decode_interval() as f64 * self.decode_step_seconds
    }
}

/// Admits one request against a running background prefill.
///
/// # Errors
/// Returns `SchedulerError::InvalidArgument` for an empty prompt.
pub fn admit(
    request: RequestId,
    prompt_tokens: usize,
    arrival_step: u64,
    timeline: &PrefillTimeline,
    config: &SchedulerConfig,
) -> Result<Admission, SchedulerError> {
    if prompt_tokens == 0 {
        return Err(SchedulerError::InvalidArgument(
            "a request needs at least one prompt token".to_owned(),
        ));
    }
    let prefill_slots = prompt_tokens.div_ceil(config.chunk_tokens);
    if prompt_tokens <= config.bypass_max_tokens {
        let admitted_step = timeline.next_admission_step(arrival_step);
        Ok(Admission {
            request,
            arrival_step,
            admitted_step,
            first_token_step: admitted_step,
            prefill_slots,
            kind: AdmissionKind::Bypassed,
        })
    } else {
        let admitted_step = timeline.last_chunk_step() + timeline.stride() * prefill_slots as u64;
        Ok(Admission {
            request,
            arrival_step,
            admitted_step,
            first_token_step: admitted_step + 1,
            prefill_slots,
            kind: AdmissionKind::QueuedBehindPrefill,
        })
    }
}

/// Seconds between a request's arrival and its first token.
#[must_use]
pub fn ttft_seconds(admission: &Admission, timeline: &PrefillTimeline, cost: &TimelineCost) -> f64 {
    timeline.seconds_between(admission.arrival_step, admission.admitted_step, cost)
        + admission.prefill_slots as f64 * cost.chunk_seconds
        + cost.decode_step_seconds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchedulerConfig;

    fn timeline() -> (SchedulerConfig, PrefillTimeline) {
        let config = SchedulerConfig::default();
        let timeline = PrefillTimeline::new(400_000, &config).expect("timeline");
        (config, timeline)
    }

    /// A request identity for the tests; the scheduler only uses it as a name.
    fn req(id: u64) -> crate::RequestId {
        crate::RequestId(id)
    }

    #[test]
    fn the_timeline_places_chunks_and_windows_from_the_interval() {
        let (config, timeline) = timeline();
        assert_eq!(timeline.chunks(), 400_000usize.div_ceil(4096));
        assert_eq!(timeline.chunks(), 98);
        assert_eq!(timeline.stride(), 1 + config.prefill_decode_interval as u64);
        assert_eq!(timeline.stride(), 3);
        assert_eq!(timeline.chunk_step(0), 0);
        assert_eq!(timeline.chunk_step(1), 3);
        assert_eq!(timeline.window_after(0), 1..3);
        assert_eq!(timeline.window_after(1), 4..6);
        assert_eq!(timeline.last_chunk_step(), 291);
        assert!(timeline.is_chunk_step(291));
        assert!(!timeline.is_chunk_step(292));
        assert_eq!(timeline.chunk_index(291), Some(97));
        assert_eq!(timeline.chunk_index(292), None);
        assert_eq!(timeline.chunk_index(300), None, "past the last chunk");
    }

    #[test]
    fn a_short_request_is_admitted_between_two_chunks() {
        let (config, timeline) = timeline();
        let expectations = [
            (0u64, 1u64),
            (1, 2),
            (2, 4),
            (3, 4),
            // Step 6 is a chunk step, so an arrival at 5 waits for window 2 (steps 7..9).
            (5, 7),
            (6, 7),
            (291, 292),
            (292, 293),
            (294, 295),
            (500, 501),
        ];
        for (arrival, first_token_step) in expectations {
            let admission = admit(req(7), 256, arrival, &timeline, &config).expect("admitted");
            assert!(admission.bypassed(), "a 256-token prompt must bypass");
            assert_eq!(
                admission.first_token_step, first_token_step,
                "arrival {arrival}"
            );
            assert_eq!(admission.prefill_slots, 1);
            assert_eq!(admission.ttft_steps(), first_token_step - arrival);
            // The admission can belong to a later window than the first: an arrival at step
            // 2 can only act at step 3, which is a chunk step, so it waits for window 1.
            let admitted = admission.admitted_step;
            let in_a_window = (0..timeline.chunks())
                .any(|chunk| timeline.window_after(chunk).contains(&admitted));
            assert!(
                in_a_window || admitted > timeline.last_chunk_step(),
                "arrival {arrival} was admitted at step {admitted}, outside every inter-chunk window"
            );
        }
    }

    #[test]
    fn two_short_requests_share_the_inter_chunk_decode_step() {
        let (config, timeline) = timeline();
        let first = admit(req(1), 200, 0, &timeline, &config).expect("admitted");
        let second = admit(req(2), 3, 0, &timeline, &config).expect("admitted");
        assert_eq!(first.admitted_step, second.admitted_step);
        assert_eq!(first.admitted_step, 1);
        let window = timeline.window_after(0);
        assert!(window.contains(&first.admitted_step));
        // Arrivals one step apart land in the same window, so both prefills share the gap.
        let third = admit(req(3), 64, 1, &timeline, &config).expect("admitted");
        assert!(window.contains(&third.admitted_step));
        assert_ne!(third.admitted_step, first.admitted_step);
        assert_eq!(third.admitted_step - first.admitted_step, 1);
    }

    #[test]
    fn the_threshold_is_inclusive_and_one_token_more_waits_for_the_prefill() {
        let (config, timeline) = timeline();
        let at_threshold =
            admit(req(1), config.bypass_max_tokens, 0, &timeline, &config).expect("ok");
        assert!(at_threshold.bypassed());
        assert_eq!(at_threshold.first_token_step, 1);
        let over = admit(req(2), config.bypass_max_tokens + 1, 0, &timeline, &config).expect("ok");
        assert_eq!(over.kind, AdmissionKind::QueuedBehindPrefill);
        assert_eq!(over.prefill_slots, 1);
        assert_eq!(
            over.admitted_step,
            timeline.last_chunk_step() + timeline.stride()
        );
        assert_eq!(over.first_token_step, over.admitted_step + 1);
        assert_eq!(over.admitted_step, 294);
        assert_eq!(over.first_token_step, 295);
        assert!(over.ttft_steps() > at_threshold.ttft_steps());
        let long = admit(req(3), 8192, 0, &timeline, &config).expect("ok");
        assert_eq!(long.prefill_slots, 2);
        assert_eq!(long.admitted_step, 291 + 2 * 3);
        assert!(admit(req(4), 0, 0, &timeline, &config).is_err());
        assert!(PrefillTimeline::new(0, &config).is_err());
    }

    #[test]
    fn the_ttft_the_spec_quotes_brackets_the_timeline() {
        let (config, timeline) = timeline();
        let cost = TimelineCost::reference(&config).expect("cost model");
        assert!((cost.chunk_seconds - 78.0 / 98.0).abs() < 1e-12);
        assert_eq!(cost.decode_step_seconds, config.assumed_decode_step_seconds);
        // Worst-case bypassed arrival: the last decode step of a window waits for the next one.
        let bypassed = admit(req(1), 256, 2, &timeline, &config).expect("ok");
        let bypassed_seconds = ttft_seconds(&bypassed, &timeline, &cost);
        assert!(
            bypassed_seconds < 4.6,
            "the bypass must beat the spec's 4.6 s claim, got {bypassed_seconds}"
        );
        assert!(bypassed_seconds > cost.chunk_seconds);
        let queued = admit(req(2), 4000, 2, &timeline, &config).expect("ok");
        let queued_seconds = ttft_seconds(&queued, &timeline, &cost);
        assert!(
            queued_seconds >= 80.0,
            "a queued request waits out the whole prefill, got {queued_seconds}"
        );
        assert!(queued_seconds / bypassed_seconds > 40.0);
        let background = cost.background_seconds(&timeline);
        assert!(background >= 78.0);
        assert!(
            (background - (78.0 + 98.0 * 2.0 * config.assumed_decode_step_seconds)).abs() < 1e-9
        );
    }

    #[test]
    fn the_cost_model_refuses_a_reference_it_cannot_use() {
        let config = SchedulerConfig::default();
        assert!(TimelineCost::from_spec(&config, 0.0, 400_000).is_err());
        assert!(TimelineCost::from_spec(&config, f64::NAN, 400_000).is_err());
        assert!(TimelineCost::from_spec(&config, 78.0, 0).is_err());
        let cost = TimelineCost::from_spec(&config, 39.0, 200_000).expect("cost model");
        assert!((cost.chunk_seconds - 39.0 / 49.0).abs() < 1e-12);
        assert!(TimelineCost::reference(&config).is_ok());
    }
}
