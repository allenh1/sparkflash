//! Step boundaries, and the D9 trace binding that has to happen at them.
//!
//! docs/DECISIONS.md D9 records the obligation: the trace request id must move onto the
//! engine request and be re-bound at each step boundary inside this crate, where the step is
//! single-threaded, instead of relying on whichever thread the HTTP task happened to be on.
//! `StepBoundaryScheduler::plan_step` is that boundary: it decides the step's phase, advances
//! every live request by one chunk or one token, and re-emits a `TraceBinding` for each of
//! them with the step as its epoch, so no span can be attributed to a stale id.
//!
//! The binding API is `sparkflash_trace::controller`, re-exported from the crate root as
//! `bind_current_request`, `current_request`, `is_request_traced`, `request_filter` and
//! `active_mask`. `plan_step` is where D9 is satisfied: for every live request it binds that
//! request's id, records the SCHED_BATCH span while the binding is in place, and restores the
//! previous binding, so the engine thread is never left dirty and a capture targeted at one
//! request records that request's spans and nobody else's.
//!
//! `EngineRequest::trace_request_id` is the id an admin capture filters on - the HTTP layer's
//! `X-Request-Id` (docs/DECISIONS.md D9) - not a scheduler-private tag.

use crate::RequestId;
use crate::bypass::{AdmissionKind, PrefillTimeline, admit};
use crate::config::SchedulerConfig;
use crate::error::SchedulerError;
use sparkflash_trace::{
    TraceScope, active_mask, bind_current_request, current_request, request_filter,
    spark_trace_span,
};

/// What a live request is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting for its own prefill, which the gap at `EngineRequest::admitted_step` carries.
    /// Its first token lands at `EngineRequest::first_token_step`; the background prefill's
    /// chunk steps are not this request's schedule (review M5 A1).
    Prefilling,
    /// Emitting one token per step.
    Decoding {
        /// Tokens emitted so far.
        tokens: u64,
    },
    /// Done; it is no longer bound.
    Finished,
}

/// One engine request, carrying its own trace id (D9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineRequest {
    /// Engine request id.
    pub id: RequestId,
    /// The trace request id this request's spans are bound to: the id an admin capture filters
    /// on, which is the HTTP layer's `X-Request-Id`.
    pub trace_request_id: String,
    /// Prompt tokens the request brought.
    pub prompt_tokens: usize,
    /// Step the request arrived at.
    pub arrival_step: u64,
    /// Step the request was admitted at.
    pub admitted_step: u64,
    /// Step its first token is decoded at.
    pub first_token_step: u64,
    /// Prefill chunks its prompt needs.
    pub prefill_slots: usize,
    /// Which admission rule admitted it.
    pub kind: AdmissionKind,
    /// Current state.
    pub state: RequestState,
}

impl EngineRequest {
    /// True when the request is finished and must not be bound again.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, RequestState::Finished)
    }

    /// Tokens the request has decoded.
    #[must_use]
    pub const fn decoded_tokens(&self) -> u64 {
        match self.state {
            RequestState::Decoding { tokens } => tokens,
            _ => 0,
        }
    }
}

/// The phase of one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPhase {
    /// A background prefill chunk runs at this step.
    PrefillChunk {
        /// Chunk index inside the background prefill.
        index: usize,
    },
    /// An inter-chunk decode step.
    InterChunkDecode,
    /// The background prefill is over; only the live requests decode.
    Drained,
}

/// The trace binding a step boundary installs for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBinding {
    /// Engine request the binding belongs to.
    pub request: RequestId,
    /// Trace request id to bind on the engine thread.
    pub trace_request_id: String,
    /// Step the binding was made at; a binding with an older epoch is stale.
    pub epoch: u64,
}

/// One step's decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPlan {
    /// Step this plan is for.
    pub step: u64,
    /// What the step runs.
    pub phase: StepPhase,
    /// Requests scheduled in this step, in request-id order.
    pub scheduled: Vec<RequestId>,
    /// Re-bound trace ids, one per scheduled request.
    pub bindings: Vec<TraceBinding>,
}

impl StepPlan {
    /// The binding for `request`, when this step scheduled it.
    #[must_use]
    pub fn binding(&self, request: RequestId) -> Option<TraceBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.request == request)
            .cloned()
    }
}

/// Restores the trace binding a D9 step put on this thread, including on unwind.
struct BindingGuard(Option<String>);

impl Drop for BindingGuard {
    fn drop(&mut self) {
        bind_current_request(self.0.take());
    }
}

/// The step-boundary scheduler over one background cold prefill.
#[derive(Debug, Clone)]
pub struct StepBoundaryScheduler {
    config: SchedulerConfig,
    timeline: PrefillTimeline,
    requests: Vec<EngineRequest>,
    step: u64,
    next_id: u64,
}

impl StepBoundaryScheduler {
    /// A scheduler whose background prefill covers `background_prompt_tokens`.
    ///
    /// # Errors
    /// Returns the configuration or timeline errors for an invalid setup.
    pub fn new(
        config: &SchedulerConfig,
        background_prompt_tokens: usize,
    ) -> Result<Self, SchedulerError> {
        Ok(Self {
            config: config.clone(),
            timeline: PrefillTimeline::new(background_prompt_tokens, config)?,
            requests: Vec::new(),
            step: 0,
            next_id: 0,
        })
    }

    /// The step the next plan is for.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// The background prefill's schedule.
    #[must_use]
    pub const fn timeline(&self) -> &PrefillTimeline {
        &self.timeline
    }

    /// Every request the scheduler knows about.
    #[must_use]
    pub fn requests(&self) -> &[EngineRequest] {
        &self.requests
    }

    /// Looks one request up.
    #[must_use]
    pub fn request(&self, id: RequestId) -> Option<&EngineRequest> {
        self.requests.iter().find(|request| request.id == id)
    }

    /// Admits a request arriving at the current step.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for an empty prompt.
    pub fn submit(
        &mut self,
        trace_request_id: impl Into<String>,
        prompt_tokens: usize,
    ) -> Result<RequestId, SchedulerError> {
        let step = self.step;
        self.submit_at(trace_request_id.into(), prompt_tokens, step)
    }

    /// Admits a request that arrived at `arrival_step`.
    ///
    /// # Errors
    /// Returns `SchedulerError::StepOutOfOrder` when the arrival is in the past, and
    /// `SchedulerError::InvalidArgument` for an empty prompt.
    pub fn submit_at(
        &mut self,
        trace_request_id: impl Into<String>,
        prompt_tokens: usize,
        arrival_step: u64,
    ) -> Result<RequestId, SchedulerError> {
        if arrival_step < self.step {
            return Err(SchedulerError::StepOutOfOrder {
                requested: arrival_step,
                next: self.step,
            });
        }
        let id = RequestId(self.next_id);
        let admission = admit(
            id,
            prompt_tokens,
            arrival_step,
            &self.timeline,
            &self.config,
        )?;
        self.next_id += 1;
        self.requests.push(EngineRequest {
            id,
            trace_request_id: trace_request_id.into(),
            prompt_tokens,
            arrival_step,
            admitted_step: admission.admitted_step,
            first_token_step: admission.first_token_step,
            prefill_slots: admission.prefill_slots,
            kind: admission.kind,
            // first_token_step is the authority. A request that arrives at or after it decodes
            // from the step it arrived at, instead of waiting for a prefill already in its past.
            state: if admission.first_token_step <= arrival_step {
                RequestState::Decoding { tokens: 0 }
            } else {
                RequestState::Prefilling
            },
        });
        Ok(id)
    }

    /// Marks a request finished; it is not bound again.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for an unknown request.
    pub fn finish(&mut self, id: RequestId) -> Result<(), SchedulerError> {
        let request = self
            .requests
            .iter_mut()
            .find(|request| request.id == id)
            .ok_or_else(|| {
                SchedulerError::InvalidArgument(format!("request {} is not known", id.0))
            })?;
        request.state = RequestState::Finished;
        Ok(())
    }

    /// Plans one step and advances the clock.
    ///
    /// The binding pass runs before the advance pass, so a request that finishes *in* this step
    /// is still bound in it and disappears at the next boundary.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the background prefill has no chunks,
    /// which the constructor prevents.
    pub fn plan_step(&mut self) -> Result<StepPlan, SchedulerError> {
        let step = self.step;
        let phase = match self.timeline.chunk_index(step) {
            Some(index) => StepPhase::PrefillChunk { index },
            None if step > self.timeline.last_chunk_step() => StepPhase::Drained,
            None => StepPhase::InterChunkDecode,
        };
        // A request is live from the step it arrived at, not from the step it was admitted at:
        // a queued request is still consuming its own prompt while the background prefill runs,
        // so its trace id has to be bound at those boundaries too.
        let live: Vec<usize> = self
            .requests
            .iter()
            .enumerate()
            .filter(|(_, request)| !request.is_finished() && request.arrival_step <= step)
            .map(|(index, _)| index)
            .collect();
        let bindings: Vec<TraceBinding> = live
            .iter()
            .map(|index| {
                let request = &self.requests[*index];
                TraceBinding {
                    request: request.id,
                    trace_request_id: request.trace_request_id.clone(),
                    epoch: step,
                }
            })
            .collect();
        // D9: the step is single-threaded, so this boundary is the one place the engine
        // thread's request binding can be established without the HTTP task's thread coming
        // into it. Each live request binds its own id, records its span while bound, and then
        // puts the previous binding back, so the engine thread is never left dirty.
        //
        // Cost: one relaxed atomic load per step in the steady state, because the whole loop is
        // skipped while the scope is disarmed. While a capture is armed, a request whose span
        // the installed filter would drop is skipped rather than allocated for, so an armed,
        // filtered capture costs at most one binding per step.
        if active_mask() & TraceScope::SCHED_BATCH != 0 {
            let filter = request_filter();
            for binding in &bindings {
                if filter
                    .as_deref()
                    .is_some_and(|wanted| wanted != binding.trace_request_id)
                {
                    continue;
                }
                let previous = bind_current_request(Some(binding.trace_request_id.clone()));
                // Restored by Drop, so the thread cannot be left dirty even if the check below
                // trips in a debug or test build (review M5 A5).
                let restore = BindingGuard(previous);
                // D9 in-boundary check: the request's own id is the current binding while its
                // span is emitted. debug_assert so a release build pays nothing for it.
                debug_assert_eq!(
                    current_request().as_deref(),
                    Some(binding.trace_request_id.as_str()),
                    "the request's id must be bound for its own span"
                );
                spark_trace_span!(
                    TraceScope::SCHED_BATCH,
                    "sched_step_bind",
                    binding.request.0 as u32
                );
                drop(restore);
            }
        }
        let scheduled: Vec<RequestId> = bindings.iter().map(|binding| binding.request).collect();
        for index in &live {
            let request = &mut self.requests[*index];
            match request.state {
                RequestState::Prefilling => {
                    // This request's own prefill runs in the gap at admitted_step, so it is
                    // decode-ready from that step. The first token lands at first_token_step:
                    // the same step for a bypassed request, whose prefill and decode share the
                    // gap, and the next one for a request queued behind the background prefill.
                    // Deriving the count from the step keeps the realized timeline exactly the
                    // admitted one, whether or not a PrefillChunk phase runs again (review M5 A1).
                    if step >= request.admitted_step {
                        let tokens = if step >= request.first_token_step {
                            step - request.first_token_step + 1
                        } else {
                            0
                        };
                        request.state = RequestState::Decoding { tokens };
                    }
                }
                RequestState::Decoding { tokens } => {
                    request.state = RequestState::Decoding { tokens: tokens + 1 };
                }
                RequestState::Finished => {}
            }
        }
        self.step += 1;
        Ok(StepPlan {
            step,
            phase,
            scheduled,
            bindings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchedulerConfig;

    fn scheduler() -> StepBoundaryScheduler {
        StepBoundaryScheduler::new(&SchedulerConfig::default(), 400_000).expect("scheduler")
    }

    /// D9: the id lives on the request and every boundary re-emits it with the step as epoch.
    #[test]
    fn the_trace_id_travels_on_the_request_and_is_rebound_at_every_boundary() {
        let mut scheduler = scheduler();
        let first = scheduler.submit("req-900", 256).expect("admitted");
        let second = scheduler.submit("req-901", 4000).expect("admitted");
        assert_eq!(
            scheduler.request(first).expect("known").trace_request_id,
            "req-900"
        );
        assert_eq!(
            scheduler.request(second).expect("known").trace_request_id,
            "req-901"
        );
        for step in 0..6u64 {
            let plan = scheduler.plan_step().expect("plan");
            assert_eq!(plan.step, step);
            assert_eq!(plan.bindings.len(), 2, "both requests stay live");
            for binding in &plan.bindings {
                assert_eq!(binding.epoch, step, "a binding is only valid for its step");
            }
            assert_eq!(
                plan.binding(first).expect("bound").trace_request_id,
                "req-900"
            );
            assert_eq!(
                plan.binding(second).expect("bound").trace_request_id,
                "req-901"
            );
            assert_eq!(plan.scheduled, vec![first, second]);
        }
    }

    /// A request that arrives after a boundary is bound at the next one, not retroactively.
    #[test]
    fn a_request_submitted_mid_step_is_bound_at_the_next_boundary() {
        let mut scheduler = scheduler();
        let first = scheduler.submit("req-1", 128).expect("admitted");
        let plan = scheduler.plan_step().expect("plan");
        assert_eq!(plan.scheduled, vec![first]);
        let late = scheduler.submit("req-2", 128).expect("admitted");
        let plan = scheduler.plan_step().expect("plan");
        assert_eq!(plan.bindings.len(), 2);
        assert_eq!(plan.binding(late).expect("bound").epoch, plan.step);
        let err = scheduler
            .submit_at("req-3", 128, 0)
            .expect_err("a past arrival is out of order");
        assert!(matches!(err, SchedulerError::StepOutOfOrder { .. }));
    }

    /// A finished request stops being bound.
    #[test]
    fn a_finished_request_is_not_rebound() {
        let mut scheduler = scheduler();
        let first = scheduler.submit("req-1", 256).expect("admitted");
        let second = scheduler.submit("req-2", 256).expect("admitted");
        scheduler.plan_step().expect("plan");
        scheduler.finish(first).expect("known");
        let plan = scheduler.plan_step().expect("plan");
        assert_eq!(plan.scheduled, vec![second]);
        assert!(plan.binding(first).is_none(), "no stale binding");
        assert!(scheduler.request(first).expect("known").is_finished());
        assert!(scheduler.finish(RequestId(99)).is_err());
    }

    /// The step phases follow the timeline's interval.
    #[test]
    fn chunk_and_decode_steps_follow_the_decode_interval() {
        let mut scheduler = scheduler();
        let phases: Vec<StepPhase> = (0..8)
            .map(|_| scheduler.plan_step().expect("plan").phase)
            .collect();
        assert_eq!(
            phases,
            vec![
                StepPhase::PrefillChunk { index: 0 },
                StepPhase::InterChunkDecode,
                StepPhase::InterChunkDecode,
                StepPhase::PrefillChunk { index: 1 },
                StepPhase::InterChunkDecode,
                StepPhase::InterChunkDecode,
                StepPhase::PrefillChunk { index: 2 },
                StepPhase::InterChunkDecode,
            ]
        );
    }

    /// A short request decodes from its admission step; a long one prefills first.
    #[test]
    fn a_bypassed_request_decodes_before_a_queued_one() {
        let mut scheduler = scheduler();
        let short = scheduler.submit("req-1", 256).expect("admitted");
        let long = scheduler.submit("req-2", 8192).expect("admitted");
        let short_request = scheduler.request(short).expect("known");
        assert_eq!(short_request.kind, AdmissionKind::Bypassed);
        assert_eq!(short_request.first_token_step, 1);
        assert_eq!(scheduler.request(long).expect("known").prefill_slots, 2);
        let plan = scheduler.plan_step().expect("plan");
        assert_eq!(plan.step, 0);
        let plan = scheduler.plan_step().expect("plan");
        assert_eq!(plan.step, 1);
        let short_tokens = scheduler.request(short).expect("known").decoded_tokens();
        assert_eq!(
            short_tokens, 1,
            "the bypassed request decodes exactly at its declared first_token_step"
        );
        let long_request = scheduler.request(long).expect("known");
        assert_eq!(
            long_request.first_token_step, 298,
            "8192 tokens wait for the background prefill plus one stride per own chunk"
        );
        assert_eq!(
            long_request.decoded_tokens(),
            0,
            "the queued request is still prefilling"
        );
    }

    /// Every admission declares the step its first token lands at, and the step machine has to
    /// realize exactly that: for a request queued behind the background prefill, for a bypassed
    /// one, and for one that arrives after the background's last chunk, when no PrefillChunk
    /// phase ever runs again (review M5 A1). 400K tokens at the default 4096 chunk is 98 chunks,
    /// so the last chunk step is 291 and arrivals at 292 and 294 are past the drain.
    #[test]
    fn the_step_machine_realizes_the_admitted_timeline() {
        // The bool is whether the request is queued behind the background prefill, in which case
        // its declared first token is later than its arrival. The last two rows arrive after the
        // step they were declared for, which is the other branch of the initial state: they decode
        // from the step they arrived at rather than before it (review M5 test-quality 1).
        let cases = [
            (256usize, 0u64, true),
            (4000, 0, true),
            (8192, 0, true),
            (256, 292, true),
            (4000, 292, true),
            (256, 294, true),
            (8192, 294, true),
            (4000, 300, false),
            (256, 500, false),
        ];
        for (tokens, arrival, queued) in cases {
            let mut engine = scheduler();
            let id = engine
                .submit_at("req-timeline", tokens, arrival)
                .expect("admitted");
            let request = engine.request(id).expect("known");
            let (admitted, declared) = (request.admitted_step, request.first_token_step);
            if queued {
                assert!(
                    declared > arrival,
                    "{tokens} tokens arriving at {arrival}: declared first token {declared}"
                );
            }
            let expected = declared.max(arrival);
            let mut actual = None;
            for _ in 0..=(expected + 3) {
                let plan = engine.plan_step().expect("plan");
                if actual.is_none() && engine.request(id).expect("known").decoded_tokens() > 0 {
                    actual = Some(plan.step);
                }
            }
            assert_eq!(
                actual,
                Some(expected),
                "{tokens} tokens arriving at {arrival} (queued {queued}): admitted {admitted}, declared first token {declared}"
            );
            assert!(
                actual.expect("a first decode") >= admitted,
                "{tokens} tokens arriving at {arrival} decoded before it was admitted"
            );
        }
    }

    /// Serialises this binary's own tests that drive the process-wide trace state: the mask, the
    /// filter and the thread binding are globals. Only the D9 proof takes it today, and the real
    /// safeguard there is that the proof reads its evidence from its own thread: the mask is
    /// process-wide, so the other tests in this binary keep recording their own step spans on
    /// their own threads while it is armed, and a lock held only by the tests that take it
    /// cannot stop them. The trace crate's own lock is crate-private
    /// (sparkflash-trace/src/lib.rs:163, `fn lock` at 169), so this binary keeps its own, exactly as
    /// sparkflash-trace/tests/request_filter.rs does.
    fn trace_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Puts the process-wide trace state back however the test ends.
    struct TraceState {
        filter: Option<String>,
        binding: Option<String>,
    }

    impl Drop for TraceState {
        fn drop(&mut self) {
            sparkflash_trace::bind_current_request(self.binding.take());
            sparkflash_trace::set_request_filter(self.filter.take());
            sparkflash_trace::disarm();
        }
    }

    /// D9 proved rather than asserted: with a filter installed, the step boundary binds the
    /// matching request, records that request's span, drops the non-matching request's span,
    /// and leaves the engine thread's previous binding in place.
    #[test]
    fn the_boundary_binds_the_filtered_request_and_restores_the_previous_binding() {
        let _serialised = trace_lock();
        sparkflash_trace::disarm();
        let state = TraceState {
            filter: request_filter(),
            binding: bind_current_request(Some("outer".to_owned())),
        };
        // arm() clears any filter, so the filter is installed afterwards
        // (sparkflash-trace/src/controller.rs:125-129).
        assert_ne!(sparkflash_trace::arm(TraceScope::SCHED_BATCH, 8), 0);
        sparkflash_trace::clear_cpu_events();

        let mut engine = scheduler();
        let matched = engine.submit("req-900", 256).expect("admitted");
        let other = engine.submit("req-901", 256).expect("admitted");
        assert_eq!(
            sparkflash_trace::set_request_filter(Some("req-900".to_owned())),
            None
        );

        let plan = engine.plan_step().expect("plan");
        assert_eq!(plan.scheduled, vec![matched, other]);
        assert_eq!(
            plan.binding(matched).expect("bound").trace_request_id,
            "req-900"
        );
        assert_eq!(
            plan.binding(other).expect("bound").trace_request_id,
            "req-901"
        );

        // The engine thread is not left dirty, and with the filter targeting another request an
        // unbound thread records nothing.
        assert_eq!(
            sparkflash_trace::current_request().as_deref(),
            Some("outer")
        );
        assert!(
            !sparkflash_trace::is_request_traced(),
            "outer is not the filtered request"
        );

        // The ring is the proof that the binding itself happened: only the span emitted while
        // "req-900" was bound survives is_request_traced(), and the meta carries the engine
        // request id, so the surviving span belongs to the matching request.
        //
        // The events of this thread only: the mask is process-wide, so the other tests in this
        // binary record their own step spans on their own threads while this test has it armed,
        // and a lock held only by the tests that use it cannot stop them.
        let thread = sparkflash_trace::current_thread_state().thread_id();
        fn recorded(label: &str, thread: u32) -> Vec<u32> {
            let mut metas: Vec<u32> = sparkflash_trace::drain_cpu_events()
                .into_iter()
                .filter(|event| event.label == label && event.thread_id == thread)
                .map(|event| event.meta)
                .collect();
            metas.sort_unstable();
            metas
        }
        assert_eq!(
            recorded("sched_step_bind", thread),
            vec![matched.0 as u32],
            "only the filtered request records"
        );

        // Without a filter every live request records, which is what an unfiltered
        // sparkflash trace --scopes all capture needs.
        sparkflash_trace::clear_cpu_events();
        assert_eq!(
            sparkflash_trace::set_request_filter(None),
            Some("req-900".to_owned())
        );
        engine.plan_step().expect("plan");
        assert_eq!(
            recorded("sched_step_bind", thread),
            vec![matched.0 as u32, other.0 as u32]
        );
        drop(state);
    }
}
