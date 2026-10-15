//! Runtime arming and disarming of trace scopes.
//!
//! Two atomics drive everything:
//!
//! * GLOBAL_TRACE_MASK - the L1-cached bitmask every CPU trace point tests.
//! * TRACE_STEPS_REMAINING - a step countdown; when it reaches zero the mask is cleared
//!   automatically, so a trace burst can never leak into steady-state serving.
//!
//! The same mask word is mirrored into the coherent-UMA GPU ring header, which is what makes
//! it possible to arm in-graph GPU trace points without recapturing a CUDA graph.

use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::event::GpuTraceRingHeader;
use crate::scope::{ScopeParseError, TraceScope, parse_scopes_with_aliases};

/// CPU-side, L1-cached scope bitmask.
pub static GLOBAL_TRACE_MASK: AtomicU64 = AtomicU64::new(TraceScope::NONE);
/// Remaining engine steps for which the mask stays armed.
pub static TRACE_STEPS_REMAINING: AtomicU32 = AtomicU32::new(0);

/// Set while a request-ID filter is installed, so the hot path can skip the lock entirely.
static REQUEST_FILTER_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The request id trace recording is restricted to, when one is installed.
static REQUEST_FILTER: Mutex<Option<String>> = Mutex::new(None);

thread_local! {
    /// Request id bound to this thread by the protocol layer.
    static CURRENT_REQUEST: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Restricts trace recording to one request id and returns the previous filter.
///
/// This is what makes `TraceAdminRequest::request_id` substantive: with a filter installed,
/// only trace points executed by a thread that bound the matching request are recorded, so a
/// 45-step trace of one request is not drowned by concurrent traffic.
pub fn set_request_filter(id: Option<String>) -> Option<String> {
    let mut guard = REQUEST_FILTER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = guard.take();
    REQUEST_FILTER_ACTIVE.store(id.is_some(), Ordering::Release);
    *guard = id;
    previous
}

/// The installed request-ID filter, if any.
#[must_use]
pub fn request_filter() -> Option<String> {
    if !REQUEST_FILTER_ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    REQUEST_FILTER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Binds a request id to the calling thread and returns the previous binding.
pub fn bind_current_request(id: Option<String>) -> Option<String> {
    CURRENT_REQUEST.with(|cell| cell.replace(id))
}

/// The request id bound to the calling thread.
#[must_use]
pub fn current_request() -> Option<String> {
    CURRENT_REQUEST.with(|cell| cell.borrow().clone())
}

/// True when the calling thread is allowed to record right now.
///
/// With no filter installed this is always true and costs one relaxed load.
#[must_use]
pub fn is_request_traced() -> bool {
    if !REQUEST_FILTER_ACTIVE.load(Ordering::Acquire) {
        return true;
    }
    match request_filter() {
        None => true,
        Some(wanted) => current_request().is_some_and(|bound| bound == wanted),
    }
}

/// Returns true when any bit of `scope` is armed and this thread may record.
///
/// One relaxed load when tracing is disabled, which is the steady-state cost.
#[must_use]
#[inline(always)]
pub fn is_scope_enabled(scope: u64) -> bool {
    if scope == 0 {
        return false;
    }
    if GLOBAL_TRACE_MASK.load(Ordering::Relaxed) & scope == 0 {
        return false;
    }
    is_request_traced()
}

/// The currently armed scope mask.
#[must_use]
pub fn active_mask() -> u64 {
    GLOBAL_TRACE_MASK.load(Ordering::Relaxed)
}

/// Steps left before the mask auto-disarms.
#[must_use]
pub fn steps_remaining() -> u32 {
    TRACE_STEPS_REMAINING.load(Ordering::Relaxed)
}

/// True while a trace burst is in flight.
#[must_use]
pub fn is_armed() -> bool {
    active_mask() != TraceScope::NONE
}

/// Arms a scope mask for `steps` engine steps.
///
/// Returns the effective mask. A zero mask or zero step count disarms instead.
pub fn arm(scopes: u64, steps: u32) -> u64 {
    // Every new capture starts unfiltered. A caller that wants single-request targeting installs
    // the filter explicitly AFTER arming (see set_request_filter). If arming did not clear it, a
    // filter left behind by an earlier capture would silently swallow this one - which is exactly
    // what happened to the X-SparkFlash-Trace header path before this line existed.
    set_request_filter(None);
    let effective = scopes & TraceScope::ALL;
    if effective == 0 || steps == 0 {
        disarm();
        return TraceScope::NONE;
    }
    GLOBAL_TRACE_MASK.store(effective, Ordering::Release);
    TRACE_STEPS_REMAINING.store(steps, Ordering::Release);
    effective
}

/// Clears the mask, the countdown and any request-ID filter.
///
/// Dropping the filter here is deliberate: after a disarm there is no capture for a filter to
/// belong to, and a stale one would make the next capture record nothing at all.
pub fn disarm() {
    GLOBAL_TRACE_MASK.store(TraceScope::NONE, Ordering::Release);
    TRACE_STEPS_REMAINING.store(0, Ordering::Release);
    set_request_filter(None);
}

/// Advances the countdown by one engine step, auto-disarming at zero.
///
/// Returns the number of steps left after the tick.
pub fn tick_step() -> u32 {
    let remaining = TRACE_STEPS_REMAINING.load(Ordering::Acquire);
    if remaining == 0 {
        GLOBAL_TRACE_MASK.store(TraceScope::NONE, Ordering::Release);
        // Keep the filter in lockstep with the mask: a filter that outlives its capture is the
        // exact hazard arm() clears it for, and tick_step() is the other way a capture ends.
        set_request_filter(None);
        return 0;
    }
    let next = remaining - 1;
    TRACE_STEPS_REMAINING.store(next, Ordering::Release);
    if next == 0 {
        GLOBAL_TRACE_MASK.store(TraceScope::NONE, Ordering::Release);
        set_request_filter(None);
    }
    next
}

/// Mirrors the CPU mask into a GPU ring header so in-graph trace points see it.
pub fn mirror_mask_to_gpu(header: &GpuTraceRingHeader, mask: u64) {
    header.publish_mask(mask & TraceScope::ALL);
}

/// Arms CPU and GPU trace points together.
pub fn arm_with_gpu(header: &GpuTraceRingHeader, scopes: u64, steps: u32) -> u64 {
    let effective = arm(scopes, steps);
    mirror_mask_to_gpu(header, effective);
    effective
}

/// Disarms CPU and GPU trace points together.
pub fn disarm_with_gpu(header: &GpuTraceRingHeader) {
    disarm();
    mirror_mask_to_gpu(header, TraceScope::NONE);
}

/// Request body of POST /v1/admin/trace.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraceAdminRequest {
    /// Scope names or aliases, e.g. ["engram", "ROCENANTE_RDMA"].
    #[serde(default)]
    pub scopes: Vec<String>,
    /// Alternative to `scopes`: a comma separated list.
    #[serde(default)]
    pub scope_list: Option<String>,
    /// Number of engine steps to trace; zero disarms.
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// Optional request id to restrict tracing to one request.
    #[serde(default)]
    pub request_id: Option<String>,
}

fn default_steps() -> u32 {
    64
}

impl TraceAdminRequest {
    /// Resolves the requested scopes into a mask.
    ///
    /// # Errors
    /// Returns ScopeParseError when a name is unknown.
    pub fn resolve_mask(&self) -> Result<u64, ScopeParseError> {
        if let Some(list) = &self.scope_list {
            return parse_scopes_with_aliases(list);
        }
        if self.scopes.is_empty() {
            return Ok(TraceScope::ALL);
        }
        parse_scopes_with_aliases(&self.scopes.join(","))
    }
}

/// Response body of POST /v1/admin/trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAdminResponse {
    /// Scope mask that is now armed.
    pub active_mask: u64,
    /// Human readable scope names.
    pub active_scopes: Vec<String>,
    /// Steps remaining.
    pub steps_remaining: u32,
    /// Registered CPU trace rings.
    pub cpu_threads: usize,
    /// CPU events currently buffered.
    pub cpu_events_buffered: u64,
    /// CPU events dropped because a ring was full.
    pub cpu_events_dropped: u64,
    /// GPU ring capacity in events.
    pub gpu_ring_capacity: usize,
    /// GPU events written to the ring.
    pub gpu_events_written: u32,
}

impl TraceAdminResponse {
    /// Snapshots the current trace state.
    #[must_use]
    pub fn snapshot(gpu: Option<&GpuTraceRingHeader>) -> Self {
        let mask = active_mask();
        let (threads, buffered, dropped) = crate::event::cpu_ring_stats();
        Self {
            active_mask: mask,
            active_scopes: TraceScope::describe(mask)
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
            steps_remaining: steps_remaining(),
            cpu_threads: threads,
            cpu_events_buffered: buffered,
            cpu_events_dropped: dropped,
            gpu_ring_capacity: gpu.map_or(0, GpuTraceRingHeader::capacity),
            gpu_events_written: gpu.map_or(0, GpuTraceRingHeader::written),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every test that touches the global mask is serialized through the CRATE-WIDE lock, because
    // the mask, the countdown and the request filter are process-wide and the span/CPU-ring tests
    // in lib.rs and event.rs read the same global registry from the same test binary.
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        crate::test_support::lock()
    }

    #[test]
    fn arm_and_disarm_toggle_the_mask() {
        let _g = guard();
        disarm();
        assert!(!is_scope_enabled(TraceScope::MEGA_MHC));
        assert_eq!(
            arm(TraceScope::MEGA_MHC | TraceScope::CSA2_ATTN, 8),
            TraceScope::MEGA_MHC | TraceScope::CSA2_ATTN
        );
        assert!(is_scope_enabled(TraceScope::MEGA_MHC));
        assert!(!is_scope_enabled(TraceScope::MOE_MXFP4));
        assert_eq!(steps_remaining(), 8);
        disarm();
        assert!(!is_armed());
        assert_eq!(active_mask(), TraceScope::NONE);
    }

    #[test]
    fn countdown_auto_disarms() {
        let _g = guard();
        arm(TraceScope::ENGRAM_PREFETCH, 2);
        assert_eq!(tick_step(), 1);
        assert!(is_scope_enabled(TraceScope::ENGRAM_PREFETCH));
        assert_eq!(tick_step(), 0);
        assert!(!is_armed());
        assert_eq!(tick_step(), 0);
    }

    #[test]
    fn arm_with_zero_steps_disarms() {
        let _g = guard();
        arm(TraceScope::ALL, 10);
        assert!(is_armed());
        assert_eq!(arm(TraceScope::ALL, 0), TraceScope::NONE);
        assert!(!is_armed());
    }

    #[test]
    fn mask_mirrors_into_the_gpu_ring_header() {
        let _g = guard();
        let header = GpuTraceRingHeader::new(1024);
        arm_with_gpu(&header, TraceScope::ROCENANTE_RDMA, 4);
        assert_eq!(header.mask(), TraceScope::ROCENANTE_RDMA);
        disarm_with_gpu(&header);
        assert_eq!(header.mask(), TraceScope::NONE);
    }

    #[test]
    fn admin_request_resolves_aliases_and_defaults() {
        let _g = guard();
        let request: TraceAdminRequest =
            serde_json::from_str(r#"{"scopes":["engram","roce"],"steps":45}"#).unwrap();
        assert_eq!(
            request.resolve_mask().unwrap(),
            TraceScope::ENGRAM_PREFETCH
                | TraceScope::ENGRAM_IO_URING
                | TraceScope::ENGRAM_GPU_GATE
                | TraceScope::ROCENANTE_RDMA
        );
        let default_all: TraceAdminRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(default_all.resolve_mask().unwrap(), TraceScope::ALL);
        assert_eq!(default_all.steps, 64);

        arm(request.resolve_mask().unwrap(), request.steps);
        let snapshot = TraceAdminResponse::snapshot(None);
        assert_eq!(snapshot.steps_remaining, 45);
        assert!(
            snapshot
                .active_scopes
                .contains(&"ROCENANTE_RDMA".to_owned())
        );
        disarm();
    }

    #[test]
    fn global_trace_mask_is_zero_cost_when_nothing_is_armed() {
        let _g = guard();
        disarm();
        // A disabled span must not allocate a guard.
        assert!(crate::ScopedCpuSpan::enter(TraceScope::ALL, "noop", 0).is_none());
    }
}
