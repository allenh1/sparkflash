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

//! Zero-overhead scoped trace points for SparkFlash-RS.
//!
//! Three gating levels keep production serving free of tracing cost:
//!
//! 1. Compile time. Without the `trace-points` feature (build the crate with
//!    --no-default-features) spark_trace_span! expands to nothing: no loads, no branches,
//!    no registers.
//! 2. A single L1-cached 64-bit mask (GLOBAL_TRACE_MASK) plus a step countdown
//!    (TRACE_STEPS_REMAINING). When disarmed, a trace point costs one relaxed load and one
//!    predictable branch.
//! 3. Per-scope granularity. Only the subsystems named in the mask record anything.
//!
//! GPU trace points inside already-captured CUDA graphs read the same mask through a
//! coherent-unified-memory header (`GpuTraceRingHeader`), so arming a scope never forces a
//! graph recapture.

pub mod controller;
pub mod event;
pub mod export;
pub mod scope;
pub mod timer;

pub use controller::{
    GLOBAL_TRACE_MASK, TRACE_STEPS_REMAINING, TraceAdminRequest, TraceAdminResponse, active_mask,
    arm, arm_with_gpu, bind_current_request, current_request, disarm, disarm_with_gpu, is_armed,
    is_request_traced, is_scope_enabled, mirror_mask_to_gpu, request_filter, set_request_filter,
    steps_remaining, tick_step,
};
pub use event::{
    CpuTraceEvent, GPU_RING_HEADER_BYTES, GPU_RING_HEADER_USED_BYTES, GPU_TRACE_EVENT_ALIGN,
    GPU_TRACE_EVENT_BYTES, GpuTraceEvent, GpuTraceRing, GpuTraceRingHeader, MAX_RING_CAPACITY,
    ThreadTraceState, clear_cpu_events, cpu_ring_stats, current_thread_state, drain_cpu_events,
};
pub use export::{
    ScopeSummary, TraceExportOptions, summarize, to_chrome_trace_json, write_chrome_trace_gz,
    write_chrome_trace_json,
};
pub use scope::{
    ScopeParseError, TraceScope, parse_scope_alias, parse_scope_token, parse_scopes,
    parse_scopes_with_aliases,
};
pub use timer::{counter_frequency, read_arm_cntvct_ns, read_counter};

use std::sync::Arc;

/// An open CPU span. Created by ScopedCpuSpan::enter and closed on drop.
///
/// The type is intentionally opaque: trace points are always written through
/// spark_trace_span! so that the compile-time switch can erase them.
pub struct ScopedCpuSpan {
    state: Arc<ThreadTraceState>,
    scope: u64,
    label: &'static str,
    meta: u32,
    start_ns: u64,
}

impl ScopedCpuSpan {
    /// Starts a span when any bit of `scope` is armed, otherwise returns None and costs a
    /// single relaxed atomic load.
    ///
    /// The decision is taken here and not revisited at `Drop`: a span already in flight when an
    /// admin installs or changes the request filter is still recorded, even if its request is
    /// filtered out by the time it ends. That is accepted, not overlooked (review M5 A5): a
    /// capture collects the spans that were already running when it was installed - bounded by
    /// the number of threads, not by the capture's duration - and re-reading the filter at
    /// `Drop` would make one span's cost depend on two readings of the same global state.
    #[inline(always)]
    #[must_use]
    pub fn enter(scope: u64, label: &'static str, meta: u32) -> Option<Self> {
        if !is_scope_enabled(scope) {
            return None;
        }
        Some(Self {
            state: current_thread_state(),
            scope,
            label,
            meta,
            start_ns: read_arm_cntvct_ns(),
        })
    }

    /// Scope bit this span is attributed to.
    #[must_use]
    pub fn scope(&self) -> u64 {
        self.scope
    }

    /// Span label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }
}

impl Drop for ScopedCpuSpan {
    #[inline(always)]
    fn drop(&mut self) {
        let end_ns = read_arm_cntvct_ns();
        let thread_id = self.state.thread_id();
        self.state.record(CpuTraceEvent {
            label: self.label,
            scope: self.scope,
            meta: self.meta,
            thread_id,
            start_ns: self.start_ns,
            end_ns,
        });
    }
}

/// Records a CPU span for the duration of the enclosing block.
///
/// With the `trace-points` feature disabled this expands to nothing at all.
///
/// ```
/// use sparkflash_trace::{TraceScope, spark_trace_span};
/// spark_trace_span!(TraceScope::KV_RADIX, "plan_request", 4096u32);
/// ```
#[cfg(feature = "trace-points")]
#[macro_export]
macro_rules! spark_trace_span {
    ($scope:expr, $label:expr, $meta:expr) => {
        let _spark_trace_guard = $crate::ScopedCpuSpan::enter($scope, $label, $meta);
    };
}

/// No-op expansion used when the crate is built without `trace-points`.
#[cfg(not(feature = "trace-points"))]
#[macro_export]
macro_rules! spark_trace_span {
    ($($tt:tt)*) => {};
}

/// Evaluates to true when the scope is armed. Zero cost when the feature is off.
#[cfg(feature = "trace-points")]
#[macro_export]
macro_rules! spark_trace_enabled {
    ($scope:expr) => {
        $crate::is_scope_enabled($scope)
    };
}

/// Always false when tracing is compiled out.
#[cfg(not(feature = "trace-points"))]
#[macro_export]
macro_rules! spark_trace_enabled {
    ($($tt:tt)*) => {
        false
    };
}

/// One lock for every unit test in this crate that touches process-global trace state.
///
/// The mask, the step countdown, the request filter, the thread registry and every per-thread CPU
/// ring are process-wide, and the unit tests for them live in three different modules
/// (controller.rs, lib.rs and event.rs) inside ONE test binary, where tests run on parallel
/// threads. A lock per module therefore serialises nothing: an event.rs ring test registering its
/// thread makes drain_cpu_events() (which drains EVERY registered thread) steal or lose events
/// that lib.rs's span tests are asserting on. That produced a gate that failed 0.11% of full
/// binary runs and 40% of focused runs before this lock was shared.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Serialises the caller against every other test that touches process-global trace state.
    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "trace-points")]
    use crate::controller::{arm, disarm};

    /// Serialises the tests that touch the process-global mask, countdown, filter or CPU rings.
    /// Without the trace-points feature nothing is recorded, so nothing needs serialising.
    #[cfg(feature = "trace-points")]
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        crate::test_support::lock()
    }

    #[cfg(feature = "trace-points")]
    #[test]
    fn spans_are_recorded_only_while_armed() {
        let _g = guard();
        disarm();
        clear_cpu_events();
        {
            spark_trace_span!(TraceScope::MEGA_MHC, "disabled_span", 1u32);
        }
        assert!(drain_cpu_events().is_empty());

        arm(TraceScope::MEGA_MHC, 4);
        {
            spark_trace_span!(TraceScope::MEGA_MHC, "enabled_span", 2u32);
        }
        let events = drain_cpu_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].label, "enabled_span");
        assert_eq!(events[0].meta, 2);
        assert_eq!(events[0].scope, TraceScope::MEGA_MHC);
        disarm();
    }

    #[cfg(feature = "trace-points")]
    #[test]
    fn two_spans_in_one_scope_coexist() {
        let _g = guard();
        arm(TraceScope::ALL, 4);
        clear_cpu_events();
        {
            spark_trace_span!(TraceScope::CSA2_ATTN, "outer_span", 0u32);
            {
                spark_trace_span!(TraceScope::CSA2_ATTN, "inner_span", 0u32);
            }
        }
        let mut labels: Vec<&str> = drain_cpu_events().iter().map(|e| e.label).collect();
        labels.sort_unstable();
        assert_eq!(labels, vec!["inner_span", "outer_span"]);
        disarm();
    }

    #[test]
    fn macro_is_available_when_tracing_is_compiled_out() {
        // This test RECORDS when it runs while another test has the mask armed, and
        // drain_cpu_events() walks every registered thread, so it must hold the same lock as the
        // span tests above even though its own assertions do not look at the ring.
        let _global_state = crate::test_support::lock();
        // Compiles (and does nothing) under --no-default-features.
        spark_trace_span!(TraceScope::ALL, "erased", 0u32);
        assert!(!spark_trace_enabled!(TraceScope::ALL) || is_scope_enabled(TraceScope::ALL));
    }
}
