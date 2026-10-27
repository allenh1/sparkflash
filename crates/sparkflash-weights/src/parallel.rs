//! The reference pass's data-parallel helpers, and the switch that pins their thread count.
//!
//! The helpers here split *independent output work* across threads: separate output elements,
//! separate output rows, or separate output chunks. No helper reorders a per-output reduction or
//! changes a per-element formula, so a parallel run produces bit-identical values to the serial run
//! by construction. That is the property the sequence fixture relies on: the committed CI fixture
//! is regenerated under both settings and must stay byte-identical, and the golden fixture's values
//! do not depend on the thread count.
//!
//! Two controls exist:
//!
//! * [with_reference_threads] pins the thread count for one call on the current thread (tests use
//!   1 and a small number to prove byte-equality);
//! * the `SPARKFLASH_REFERENCE_THREADS` environment variable sets the process-wide default, and
//!   otherwise the machine's available parallelism is used.
//!
//! Nested calls run serially: a function called from inside a worker is already inside the parallel
//! region and does not spawn a second level.
//!
//! The machine thread count is cached: [std::thread::available_parallelism] is not a cheap fact on
//! this host - it reads the cgroup CPU-quota files every time - and it measured 27.6 us per call,
//! which the helpers pay on every invocation whether or not they end up spawning anything. The
//! count is a property of the process, so it is computed once; the environment override is still
//! read per call, so a run that sets it after startup still sees it.

use std::cell::Cell;

thread_local! {
    static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
    static OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The machine's parallelism, computed once: asking for it costs a cgroup read.
fn cached_parallelism() -> usize {
    static MACHINE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MACHINE.get_or_init(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    })
}

/// The thread count used when nothing pins one.
fn machine_threads() -> usize {
    if let Ok(text) = std::env::var("SPARKFLASH_REFERENCE_THREADS")
        && let Ok(value) = text.trim().parse::<usize>()
        && value > 0
    {
        return value;
    }
    cached_parallelism()
}

/// The thread count the helpers will use on the current thread.
#[must_use]
pub fn reference_threads() -> usize {
    OVERRIDE.with(Cell::get).unwrap_or_else(machine_threads)
}

/// Runs `body` with a fixed reference thread count; 0 is treated as 1 (serial).
///
/// This is the test hook that proves the parallel helpers are value-transparent.
pub fn with_reference_threads<R>(threads: usize, body: impl FnOnce() -> R) -> R {
    let previous = OVERRIDE.with(|cell| cell.replace(Some(threads.max(1))));
    let result = body();
    OVERRIDE.with(|cell| cell.set(previous));
    result
}

fn inside_parallel() -> bool {
    IN_PARALLEL.with(Cell::get)
}

/// Marks the current thread as a worker inside a parallel region.
///
/// A routine that spawns its own workers (the MoE expert pass does, so one thread owns a whole
/// expert's read, decode and arithmetic) calls this first: the generic helpers then stay serial
/// inside the worker instead of oversubscribing the machine.
pub fn enter_worker() {
    IN_PARALLEL.with(|cell| cell.set(true));
}

/// Runs `body(offset, chunk)` on contiguous chunks of `data`, in parallel when it pays.
///
/// The chunk boundaries are always element boundaries; `offset` is the chunk's element index in
/// the whole slice. The closure must only write its own chunk: every chunk is disjoint, which is
/// why the result cannot depend on scheduling.
pub fn par_chunks_mut<T, F>(data: &mut [T], min_chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let workers = reference_threads();
    let len = data.len();
    if workers <= 1 || len <= min_chunk || inside_parallel() {
        body(0, data);
        return;
    }
    let chunk = len.div_ceil(workers).max(min_chunk);
    std::thread::scope(|scope| {
        for (index, part) in data.chunks_mut(chunk).enumerate() {
            let body = &body;
            let offset = index * chunk;
            scope.spawn(move || {
                IN_PARALLEL.with(|cell| cell.set(true));
                body(offset, part);
            });
        }
    });
}

/// Runs `body(first_unit, chunk)` on contiguous whole units of `data`, in parallel when it pays.
///
/// `unit` is the width of one independent unit (a row); `data.len()` must be a whole number of
/// them. Splitting on unit boundaries is what lets a row-wise routine keep its row arithmetic and
/// still run on several threads: the row index a chunk starts at is `first_unit`, and the rows in
/// the chunk are exactly the serial routine's rows, visited in the same order.
pub fn par_aligned_mut<T, F>(data: &mut [T], unit: usize, min_units: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if unit == 0 || !data.len().is_multiple_of(unit) {
        body(0, data);
        return;
    }
    let units = data.len() / unit;
    let workers = reference_threads();
    if workers <= 1 || units <= min_units || inside_parallel() {
        body(0, data);
        return;
    }
    let workers = workers.min(units.div_ceil(min_units.max(1)));
    let chunk_units = units.div_ceil(workers).max(min_units.max(1));
    let chunk = chunk_units * unit;
    std::thread::scope(|scope| {
        for (index, part) in data.chunks_mut(chunk).enumerate() {
            let body = &body;
            let offset = index * chunk_units;
            scope.spawn(move || {
                IN_PARALLEL.with(|cell| cell.set(true));
                body(offset, part);
            });
        }
    });
}
