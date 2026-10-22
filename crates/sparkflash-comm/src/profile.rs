//! Opt-in phase profiler for the host collective path.
//!
//! The profiler is off unless `SPARK_COMM_PROFILE` is set. With it off, an instrumented call
//! pays one relaxed atomic load; a per-peer reader thread pays one `Option<Instant>` check.
//! With it on, every instrumented all-reduce records the time it spent in prepare, in the
//! per-peer sends, in collect (joining the readers, i.e. waiting for peer payloads), and in the
//! local canonical reduction, plus the per-peer wait measured inside the reader threads. One
//! window line is printed to stderr every `SPARK_COMM_PROFILE_EVERY` completed calls (default
//! 40, `0` disables the periodic line). All printed numbers are microseconds per call.
//!
//! The reader-thread wait is the evidence that separates "the transport is slow" from "this rank
//! arrived first and waited for its peers": it starts when the reader thread starts and ends when
//! its frame is complete, so a rank that reaches the call early shows a large wait and the rank
//! that reaches it last shows almost none. Arrival order never changes the canonical reduction;
//! this only attributes the wall time.
//!
//! The profiler never affects the arithmetic. It reads clocks and atomics only.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static ENABLED: OnceLock<bool> = OnceLock::new();
static EVERY: OnceLock<u64> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var_os("SPARK_COMM_PROFILE")
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

fn every() -> u64 {
    *EVERY.get_or_init(|| {
        std::env::var("SPARK_COMM_PROFILE_EVERY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(40)
    })
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static SEND_NS: AtomicU64 = AtomicU64::new(0);
static SEND_MAX_NS: AtomicU64 = AtomicU64::new(0);
static COLLECT_NS: AtomicU64 = AtomicU64::new(0);
static COLLECT_MAX_NS: AtomicU64 = AtomicU64::new(0);
static REDUCE_NS: AtomicU64 = AtomicU64::new(0);
static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static TOTAL_MAX_NS: AtomicU64 = AtomicU64::new(0);
static PEER_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static PEER_WAIT_MAX_NS: AtomicU64 = AtomicU64::new(0);
static PEER_WAITS: AtomicU64 = AtomicU64::new(0);
static SEQUENTIAL_CALLS: AtomicU64 = AtomicU64::new(0);
static THREAD_CALLS: AtomicU64 = AtomicU64::new(0);

/// A timer for one collective call. Every method is a no-op when `SPARK_COMM_PROFILE` is unset.
pub(crate) struct CallTimer {
    inner: Option<TimerInner>,
}

struct TimerInner {
    start: Instant,
    phase: Instant,
    prepare: Duration,
    send: Duration,
    send_max: Duration,
    collect: Duration,
    collect_max: Duration,
    reduce: Duration,
}

impl CallTimer {
    /// Starts a timer, or returns a disabled timer.
    pub(crate) fn start() -> Self {
        if enabled() {
            let now = Instant::now();
            Self {
                inner: Some(TimerInner {
                    start: now,
                    phase: now,
                    prepare: Duration::ZERO,
                    send: Duration::ZERO,
                    send_max: Duration::ZERO,
                    collect: Duration::ZERO,
                    collect_max: Duration::ZERO,
                    reduce: Duration::ZERO,
                }),
            }
        } else {
            Self { inner: None }
        }
    }

    /// Closes the prepare phase (the reader-thread handoff).
    pub(crate) fn mark_prepare(&mut self) {
        if let Some(inner) = &mut self.inner {
            let now = Instant::now();
            inner.prepare += now.duration_since(inner.phase);
            inner.phase = now;
        }
    }

    /// Returns a clock for one send, or `None` when the profiler is disabled.
    pub(crate) fn send_clock(&self) -> Option<Instant> {
        self.inner.as_ref().map(|_| Instant::now())
    }

    /// Records one completed send.
    pub(crate) fn record_send(&mut self, clock: Option<Instant>) {
        if let (Some(inner), Some(clock)) = (&mut self.inner, clock) {
            let elapsed = clock.elapsed();
            inner.send += elapsed;
            if elapsed > inner.send_max {
                inner.send_max = elapsed;
            }
        }
    }

    /// Closes the send phase; the next phase is collect.
    pub(crate) fn mark_send(&mut self) {
        if let Some(inner) = &mut self.inner {
            inner.phase = Instant::now();
        }
    }

    /// Closes the collect phase (the wait for peer payloads).
    pub(crate) fn mark_collect(&mut self) {
        if let Some(inner) = &mut self.inner {
            let now = Instant::now();
            let elapsed = now.duration_since(inner.phase);
            inner.collect += elapsed;
            if elapsed > inner.collect_max {
                inner.collect_max = elapsed;
            }
            inner.phase = now;
        }
    }

    /// Closes the canonical-reduce phase and publishes the call.
    pub(crate) fn mark_reduce(mut self) {
        if let Some(inner) = &mut self.inner {
            let now = Instant::now();
            inner.reduce += now.duration_since(inner.phase);
            let prepare = inner.prepare;
            let send = inner.send;
            let send_max = inner.send_max;
            let collect = inner.collect;
            let collect_max = inner.collect_max;
            let reduce = inner.reduce;
            let total = inner.start.elapsed();
            record(CallRecord {
                prepare,
                send,
                send_max,
                collect,
                collect_max,
                reduce,
                total,
            });
        }
    }
}

struct CallRecord {
    prepare: Duration,
    send: Duration,
    send_max: Duration,
    collect: Duration,
    collect_max: Duration,
    reduce: Duration,
    total: Duration,
}

fn record(record: CallRecord) {
    let nanos = |duration: Duration| duration.as_nanos().min(u128::from(u64::MAX)) as u64;
    PREPARE_NS.fetch_add(nanos(record.prepare), Ordering::Relaxed);
    SEND_NS.fetch_add(nanos(record.send), Ordering::Relaxed);
    SEND_MAX_NS.fetch_max(nanos(record.send_max), Ordering::Relaxed);
    COLLECT_NS.fetch_add(nanos(record.collect), Ordering::Relaxed);
    COLLECT_MAX_NS.fetch_max(nanos(record.collect_max), Ordering::Relaxed);
    REDUCE_NS.fetch_add(nanos(record.reduce), Ordering::Relaxed);
    TOTAL_NS.fetch_add(nanos(record.total), Ordering::Relaxed);
    TOTAL_MAX_NS.fetch_max(nanos(record.total), Ordering::Relaxed);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let every = every();
    if every != 0 && calls.is_multiple_of(every) {
        print_window();
    }
}

fn print_window() {
    let calls = CALLS.swap(0, Ordering::Relaxed);
    if calls == 0 {
        return;
    }
    let per_call_us = |nanos: u64| nanos as f64 / calls as f64 / 1000.0;
    let prepare = per_call_us(PREPARE_NS.swap(0, Ordering::Relaxed));
    let send = per_call_us(SEND_NS.swap(0, Ordering::Relaxed));
    let collect = per_call_us(COLLECT_NS.swap(0, Ordering::Relaxed));
    let reduce = per_call_us(REDUCE_NS.swap(0, Ordering::Relaxed));
    let total = per_call_us(TOTAL_NS.swap(0, Ordering::Relaxed));
    let send_max = SEND_MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    let collect_max = COLLECT_MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    let total_max = TOTAL_MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    let peer_waits = PEER_WAITS.swap(0, Ordering::Relaxed);
    let peer_wait = per_call_us(PEER_WAIT_NS.swap(0, Ordering::Relaxed));
    let peer_wait_max = PEER_WAIT_MAX_NS.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    let sequential_calls = SEQUENTIAL_CALLS.swap(0, Ordering::Relaxed);
    let thread_calls = THREAD_CALLS.swap(0, Ordering::Relaxed);
    eprintln!(
        "comm-profile window_calls={calls} prepare_us={prepare:.1} send_us={send:.1} \
send_max_us={send_max:.1} collect_us={collect:.1} collect_max_us={collect_max:.1} \
reduce_us={reduce:.1} total_us={total:.1} total_max_us={total_max:.1} \
peer_wait_us={peer_wait:.1} peer_wait_max_us={peer_wait_max:.1} peer_waits={peer_waits} \
sequential_calls={sequential_calls} thread_calls={thread_calls}"
    );
}

/// Starts a clock for one peer-reader wait, or `None` when the profiler is disabled.
pub(crate) fn peer_wait_clock() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Records which small-frame read path an exchange selected.
pub(crate) fn record_read_path(sequential: bool) {
    if !enabled() {
        return;
    }
    if sequential {
        SEQUENTIAL_CALLS.fetch_add(1, Ordering::Relaxed);
    } else {
        THREAD_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Records one completed peer-reader wait.
pub(crate) fn record_peer_wait(clock: Option<Instant>) {
    let Some(clock) = clock else {
        return;
    };
    let nanos = clock.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    PEER_WAIT_NS.fetch_add(nanos, Ordering::Relaxed);
    PEER_WAIT_MAX_NS.fetch_max(nanos, Ordering::Relaxed);
    PEER_WAITS.fetch_add(1, Ordering::Relaxed);
}
