//! Row prefetch: a 96-deep asynchronous read engine in front of the Engram tables.
//!
//! The consumer walks rows in increasing order (the Engram lookups of one decode step), so
//! the engine issues the reads it is about to need and hands them back on `take`. Two
//! backends implement the same request/ready-map protocol:
//!
//! * the `io-uring` backend (behind the `io-uring` feature) keeps a ring of
//!   `[DEFAULT_QUEUE_DEPTH]` reads in flight, one slot each, on a single worker thread, and
//!   is the backend that actually removes the syscall-per-row cost;
//! * the portable backend runs `pread` on a small thread pool, which is the default and the
//!   fallback whenever the ring cannot be created or fails its start-up probe.
//!
//! Neither backend can make a request fail: a request the engine cannot accept is counted
//! as backpressure and dropped, because the caller can always read the row synchronously
//! through the same `[RowTable]`. `[PrefetchEngine::fallback_reason]` says why the portable
//! backend was chosen.
//!
//! Trace points `ENGRAM_PREFETCH` (one per issued read) and `ENGRAM_IO_URING` (one per
//! ring submission) can be armed through sparkflash-trace.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sparkflash_trace::{TraceScope, spark_trace_span};

use crate::layout::ENGRAM_ROW_BYTES;
use crate::store::RowTable;

/// Ring depth and queue depth: how many reads the engine keeps outstanding.
pub const DEFAULT_QUEUE_DEPTH: usize = 96;

/// Default lookahead, in rows, past the consumer's cursor.
pub const DEFAULT_LEAD_ROWS: u64 = 32;

/// Default cap on the rows one batch of prefetch requests may cover.
pub const DEFAULT_MAX_BATCH_ROWS: usize = 64;

/// Default number of read threads used by the portable backend.
pub const DEFAULT_WORKERS: usize = 8;

/// Rows that may sit completed but unconsumed before the oldest is dropped.
pub const DEFAULT_READY_CAPACITY: usize = 4 * DEFAULT_QUEUE_DEPTH;

/// Which backend is serving the requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefetchBackend {
    /// The `io-uring` ring backend.
    IoUring,
    /// The portable `pread` thread pool.
    ThreadPool,
    /// Requests are served inline on the calling thread; no workers are started.
    Inline,
}

/// Backend preference for `[PrefetchEngine::with_backend]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendRequest {
    /// Use the ring when the feature is compiled in and its probe succeeds.
    #[default]
    Auto,
    /// Never use the ring; always the thread pool.
    ThreadPool,
    /// Require the ring, failing construction when it cannot start.
    IoUring,
    /// Read inline on the calling thread; deterministic, used by unit tests.
    Inline,
}

/// Prefetch geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchConfig {
    /// Outstanding reads the engine will keep in flight.
    pub queue_depth: usize,
    /// Lookahead in rows for `[PrefetchEngine::prefetch_ahead]`.
    pub lead_rows: u64,
    /// Rows one batch request may cover.
    pub max_batch_rows: usize,
    /// Read threads for the portable backend.
    pub workers: usize,
    /// Completed rows retained before the oldest is dropped.
    pub ready_capacity: usize,
    /// Window size used by the ring backend, in bytes; a multiple of 512.
    pub window_bytes: u64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            queue_depth: DEFAULT_QUEUE_DEPTH,
            lead_rows: DEFAULT_LEAD_ROWS,
            max_batch_rows: DEFAULT_MAX_BATCH_ROWS,
            workers: DEFAULT_WORKERS,
            ready_capacity: DEFAULT_READY_CAPACITY,
            window_bytes: crate::store::DEFAULT_WINDOW_BYTES,
        }
    }
}

impl PrefetchConfig {
    /// Validates the geometry.
    ///
    /// # Errors
    /// When the queue depth is zero or absurd, the batch size is zero, or the window is not
    /// a 512-byte multiple.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.queue_depth >= 1 && self.queue_depth <= 4096,
            "queue depth {} is out of 1..=4096",
            self.queue_depth
        );
        anyhow::ensure!(
            self.max_batch_rows >= 1,
            "max batch rows must be at least one"
        );
        anyhow::ensure!(
            self.ready_capacity >= self.queue_depth,
            "ready capacity {} must hold at least one queue depth ({})",
            self.ready_capacity,
            self.queue_depth
        );
        anyhow::ensure!(
            self.window_bytes >= crate::store::MIN_ALIGN_BYTES
                && self
                    .window_bytes
                    .is_multiple_of(crate::store::MIN_ALIGN_BYTES),
            "window size {} must be a multiple of {}",
            self.window_bytes,
            crate::store::MIN_ALIGN_BYTES
        );
        Ok(())
    }

    /// Number of read threads to start: the configured count, or a core-derived default.
    #[must_use]
    pub fn resolved_workers(&self) -> usize {
        if self.workers == 0 {
            std::thread::available_parallelism()
                .map(|cores| cores.get().clamp(1, self.queue_depth))
                .unwrap_or(4)
        } else {
            self.workers.min(self.queue_depth).max(1)
        }
    }
}

/// Prefetch counters.
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Requests accepted by a backend.
    pub issued: AtomicU64,
    /// Requests whose row arrived in the ready map.
    pub completed: AtomicU64,
    /// Completed rows dropped without being consumed.
    pub wasted: AtomicU64,
    /// Requests whose read failed, counted in rows.
    ///
    /// The Inline backend serves a contiguous group of rows with one read call, so a failed
    /// read counts every row in that group. `pool_read_failures` counts the failed reads
    /// themselves, which is what makes that granularity visible instead of inferred.
    pub failed: AtomicU64,
    /// Inline reads that failed, one per `read_rows_into` call.
    pub pool_read_failures: AtomicU64,
    /// Requests refused because the queue was full.
    pub backpressure: AtomicU64,
    /// Requests refused because the backend's receiver was gone. The ring worker drops its
    /// receiver when it retires a lost completion and exits, so this - not backpressure - is
    /// the counter that says the backend died rather than that it was busy.
    pub channel_dead: AtomicU64,
    /// Rows handed back to the consumer by `take`.
    pub consumed: AtomicU64,
    /// Row bytes read by the engine, including the window padding it slices away.
    pub bytes_read: AtomicU64,
    /// Sum of the lead distances sampled at `take`, in decode steps.
    pub lead_steps_total: AtomicU64,
    /// Number of lead distances sampled.
    pub lead_samples: AtomicU64,
    /// Submissions made to the io_uring ring.
    pub ring_submissions: AtomicU64,
    /// Blocking reads made by the portable thread pool.
    pub pool_reads: AtomicU64,
}

/// Plain-value view of the prefetch counters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefetchSnapshot {
    /// Requests accepted by a backend.
    pub issued: u64,
    /// Requests whose row arrived in the ready map.
    pub completed: u64,
    /// Completed rows dropped without being consumed.
    pub wasted: u64,
    /// Requests whose read failed, counted in rows.
    pub failed: u64,
    /// Inline reads that failed. Together with `failed` it shows the group size a failure
    /// cost, because the Inline backend reads a contiguous group in one call.
    pub pool_read_failures: u64,
    /// Requests refused because the queue was full.
    pub backpressure: u64,
    /// Requests refused because the backend's receiver was gone: the backend died rather than
    /// ran out of room.
    pub channel_dead: u64,
    /// Rows handed back by `take`.
    pub consumed: u64,
    /// Row bytes read by the engine.
    pub bytes_read: u64,
    /// Mean lead distance of consumed rows, in decode steps.
    pub avg_lead_steps: f64,
    /// Number of lead samples behind `avg_lead_steps`.
    pub lead_samples: u64,
    /// Configured queue depth.
    pub queue_depth: usize,
    /// Current lookahead in rows.
    pub lead_rows: u64,
    /// Configured batch cap.
    pub max_batch_rows: usize,
    /// Rows completed but not yet consumed.
    pub ready: usize,
    /// Bookkeeping entries behind `ready`: one per ready row. It is the direct check that
    /// consuming rows does not leak queue state.
    pub ready_order: usize,
    /// Requests accepted but not yet completed.
    pub in_flight: usize,
    /// Backend serving the requests.
    pub backend: PrefetchBackend,
    /// Submissions made to the io_uring ring.
    pub ring_submissions: u64,
    /// Blocking reads made by the portable thread pool.
    pub pool_reads: u64,
    /// True when the io_uring worker stopped serving after a lost completion and left its
    /// rows to the synchronous path. Always false on the Inline and ThreadPool backends.
    pub ring_stalled: bool,
}

/// One completed prefetch.
#[derive(Debug, Clone, Copy)]
struct Entry {
    data: [u8; ENGRAM_ROW_BYTES],
    issued_step: u64,
}

/// Completed rows waiting for the consumer.
///
/// `order` maps the arrival sequence of every row in `rows` to its key, so a row that is
/// consumed, or dropped past the capacity, leaves both in one step. The two maps hold
/// exactly the same keys, which is what keeps the bookkeeping bounded: trimming only while
/// the map is over capacity grows by one entry per completed row for the whole life of a
/// decode that keeps up with its prefetches.
struct ReadyMap {
    rows: HashMap<u64, (Entry, u64)>,
    order: BTreeMap<u64, u64>,
    next_sequence: u64,
    capacity: usize,
}

impl ReadyMap {
    fn new(capacity: usize) -> Self {
        Self {
            rows: HashMap::with_capacity(capacity.min(4096)),
            order: BTreeMap::new(),
            next_sequence: 0,
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    /// Bookkeeping entries; always equal to `len`.
    fn order_len(&self) -> usize {
        self.order.len()
    }

    fn contains(&self, key: u64) -> bool {
        self.rows.contains_key(&key)
    }

    /// Stores a completed row, dropping the oldest unconsumed rows past the capacity.
    /// Returns how many rows were dropped.
    fn insert(&mut self, key: u64, entry: Entry) -> u64 {
        if let Some((slot, _)) = self.rows.get_mut(&key) {
            *slot = entry;
            return 0;
        }
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.rows.insert(key, (entry, sequence));
        self.order.insert(sequence, key);
        let mut dropped = 0;
        while self.rows.len() > self.capacity {
            let Some((sequence, oldest)) = self.order.iter().next().map(|(s, k)| (*s, *k)) else {
                break;
            };
            self.order.remove(&sequence);
            if self.rows.remove(&oldest).is_some() {
                dropped += 1;
            }
        }
        dropped
    }

    fn take(&mut self, key: u64) -> Option<Entry> {
        let (entry, sequence) = self.rows.remove(&key)?;
        self.order.remove(&sequence);
        Some(entry)
    }

    /// Drops every completed row, returning how many there were.
    fn clear(&mut self) -> u64 {
        let dropped = self.rows.len() as u64;
        self.rows.clear();
        self.order.clear();
        dropped
    }

    /// True when the two maps hold exactly the same keys.
    #[cfg(test)]
    fn invariants_hold(&self) -> bool {
        self.order.len() == self.rows.len()
            && self.order.values().all(|key| self.rows.contains_key(key))
    }
}

/// Packs `(table, row)` into a ready-map key. Rows wider than 48 bits cannot be keyed and
/// are never prefetched.
fn job_key(table: usize, row: u64) -> Option<u64> {
    if row > 0x0000_FFFF_FFFF_FFFF {
        return None;
    }
    Some(((table as u64) << 48) | row)
}

/// One read request.
#[derive(Debug, Clone, Copy)]
struct Job {
    table: usize,
    row: u64,
    key: u64,
    issued_step: u64,
}

/// How many prefetch failures are remembered so a waiter can be told why its row is
/// missing. Bounded, because a consumer that never asks for a row must not be able to grow
/// the record without limit.
const MAX_RECORDED_FAILURES: usize = 64;

/// State shared between the engine and its backend workers.
struct Shared {
    ready: Mutex<ReadyMap>,
    stats: PrefetchStats,
    step: AtomicU64,
    in_flight: AtomicUsize,
    stop: AtomicBool,
    /// Most recent failed requests, newest last, keyed by ready-map key.
    failures: Mutex<VecDeque<(u64, String)>>,
    /// Set by the io_uring worker when it stops serving after a lost completion.
    ring_stalled: AtomicBool,
    /// Orders acceptance against retirement (review A1, round 4).
    ///
    /// `Shared::accept` holds it across the reservation and the hand-off, and the retire path
    /// holds it from before it raises `ring_stalled` until after it has drained every accepted
    /// request and brought the shared count back to zero. Without it a request could be accepted
    /// into a queue the retirement had already walked: counted in flight forever, with no reason
    /// for its waiter and no worker left to complete it.
    retire_gate: Mutex<()>,
}

impl Shared {
    /// Remembers why a request could not produce a row, so that
    /// `[PrefetchEngine::fill_or_read]` can report the prefetch failure instead of leaving
    /// the caller to guess.
    fn record_failure(&self, key: u64, message: String) {
        let mut failures = self.failures.lock();
        if failures.len() >= MAX_RECORDED_FAILURES {
            failures.pop_front();
        }
        failures.push_back((key, message));
    }

    /// Takes the recorded failure for `key`, if there is one.
    fn take_failure(&self, key: u64) -> Option<String> {
        let mut failures = self.failures.lock();
        let index = failures.iter().position(|(failed, _)| *failed == key)?;
        failures.remove(index).map(|(_, message)| message)
    }

    /// Classifies a send the queue refused. Returns false: the row was not accepted.
    ///
    /// A full queue is backpressure - the row was refused now and the caller may ask again - so
    /// it is counted as such. A disconnected queue is a dead backend: the ring worker breaks out
    /// of its loop when it retires a lost completion, which drops its receiver while the engine
    /// still holds the sender, and a worker that fails to start drops it just the same. That row
    /// is not retried on this backend and the stall flag is raised, which is what sends
    /// `fill_or_read` down the synchronous path; the two are counted separately so a dead ring
    /// can never be read as healthy backpressure (review A3).
    fn note_refused_send(&self, error: &TrySendError<Job>) -> bool {
        match error {
            TrySendError::Full(_) => {
                self.stats.backpressure.fetch_add(1, Ordering::Relaxed);
                false
            }
            TrySendError::Disconnected(_) => self.note_dead_backend(),
        }
    }

    /// Records that this backend will not serve the row at all - the ring stopped after a lost
    /// completion, or the queue's receiver is gone - and returns false: the row was not accepted.
    ///
    /// Both are the same capability claim: this backend is dead for the rest of the engine's
    /// life, so the consumer must read synchronously rather than wait for a worker that will
    /// never run.
    fn note_dead_backend(&self) -> bool {
        self.stats.channel_dead.fetch_add(1, Ordering::Relaxed);
        self.ring_stalled.store(true, Ordering::Relaxed);
        false
    }

    /// Reserves and hands one job to a backend channel, under the retirement gate.
    ///
    /// Returns true when the job was accepted, false when the backend will not take it: the ring
    /// has stopped serving, or the queue refused it. The gate is held across the reservation and
    /// the hand-off, so an accepted request cannot escape into a queue the retirement has already
    /// walked (review A1, round 4). The reservation happens before the hand-off because a worker
    /// can complete the job and decrement the count before this thread's own increment lands,
    /// which would take the counter below the true value (review A7).
    fn accept(&self, tx: &Sender<Job>, job: Job) -> bool {
        let _gate = self.retire_gate.lock();
        if self.ring_stalled.load(Ordering::Relaxed) {
            return self.note_dead_backend();
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        match tx.try_send(job) {
            Ok(()) => true,
            Err(error) => {
                self.in_flight.fetch_sub(1, Ordering::Relaxed);
                self.note_refused_send(&error)
            }
        }
    }
}

/// Handles of the portable backend.
struct PoolHandle {
    tx: Sender<Job>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// Reads rows ahead of the consumer and hands them back on `take`.
pub struct PrefetchEngine {
    source: Arc<dyn RowTable>,
    config: PrefetchConfig,
    lead_rows: AtomicU64,
    shared: Arc<Shared>,
    backend: PrefetchBackend,
    fallback_reason: Option<String>,
    pool: Option<PoolHandle>,
    #[cfg(feature = "io-uring")]
    ring: Option<uring::RingHandle>,
}

impl PrefetchEngine {
    /// Starts an engine on the best available backend.
    ///
    /// # Errors
    /// When the configuration is invalid.
    pub fn new(source: Arc<dyn RowTable>, config: PrefetchConfig) -> anyhow::Result<Self> {
        Self::with_backend(source, config, BackendRequest::Auto)
    }

    /// Starts an engine with an explicit backend preference.
    ///
    /// # Errors
    /// When the configuration is invalid, or the ring was required and could not start.
    pub fn with_backend(
        source: Arc<dyn RowTable>,
        config: PrefetchConfig,
        request: BackendRequest,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let shared = Arc::new(Shared {
            ready: Mutex::new(ReadyMap::new(config.ready_capacity)),
            stats: PrefetchStats::default(),
            step: AtomicU64::new(0),
            in_flight: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            failures: Mutex::new(VecDeque::new()),
            ring_stalled: AtomicBool::new(false),
            retire_gate: Mutex::new(()),
        });
        let mut fallback_reason: Option<String> = None;
        let mut pool = None;
        #[cfg(feature = "io-uring")]
        let mut ring = None;
        let backend = match request {
            BackendRequest::Inline => PrefetchBackend::Inline,
            BackendRequest::ThreadPool => {
                pool = Some(start_pool(&source, &config, &shared)?);
                PrefetchBackend::ThreadPool
            }
            BackendRequest::IoUring => {
                #[cfg(feature = "io-uring")]
                {
                    ring = Some(uring::start(&source, &config, &shared)?);
                    PrefetchBackend::IoUring
                }
                #[cfg(not(feature = "io-uring"))]
                {
                    anyhow::bail!(
                        "the io-uring backend requires building sparkflash-engram with \
                         --features io-uring"
                    );
                }
            }
            BackendRequest::Auto => {
                #[cfg(feature = "io-uring")]
                {
                    match uring::start(&source, &config, &shared) {
                        Ok(handle) => {
                            ring = Some(handle);
                            PrefetchBackend::IoUring
                        }
                        Err(error) => {
                            fallback_reason =
                                Some(format!("io_uring unavailable, using pread pool: {error}"));
                            pool = Some(start_pool(&source, &config, &shared)?);
                            PrefetchBackend::ThreadPool
                        }
                    }
                }
                #[cfg(not(feature = "io-uring"))]
                {
                    // Compiled without the feature: say so, so a caller can tell "not in this
                    // build" from "tried and failed".
                    fallback_reason = Some(
                        "the io-uring feature is not compiled into this build, using the \
                         pread pool"
                            .to_owned(),
                    );
                    pool = Some(start_pool(&source, &config, &shared)?);
                    PrefetchBackend::ThreadPool
                }
            }
        };
        Ok(Self {
            lead_rows: AtomicU64::new(config.lead_rows),
            config,
            source,
            shared,
            backend,
            fallback_reason,
            pool,
            #[cfg(feature = "io-uring")]
            ring,
        })
    }

    /// Backend actually serving this engine.
    #[must_use]
    pub fn backend(&self) -> PrefetchBackend {
        self.backend
    }

    /// Why the portable backend was used instead of the ring, when it was.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// The engine's configuration.
    #[must_use]
    pub fn config(&self) -> &PrefetchConfig {
        &self.config
    }

    /// The row source being prefetched from.
    #[must_use]
    pub fn source(&self) -> &Arc<dyn RowTable> {
        &self.source
    }
}

/// Starts the portable `pread` pool.
fn start_pool(
    source: &Arc<dyn RowTable>,
    config: &PrefetchConfig,
    shared: &Arc<Shared>,
) -> anyhow::Result<PoolHandle> {
    let (tx, rx) = crossbeam_channel::bounded::<Job>(config.queue_depth);
    let workers = config.resolved_workers();
    let mut handles = Vec::with_capacity(workers);
    for index in 0..workers {
        let source = Arc::clone(source);
        let shared = Arc::clone(shared);
        let rx: Receiver<Job> = rx.clone();
        let handle = std::thread::Builder::new()
            .name(format!("engram-prefetch-{index}"))
            .spawn(move || pool_worker(&source, &rx, &shared))
            .map_err(|error| anyhow::anyhow!("spawn prefetch worker {index}: {error}"))?;
        handles.push(handle);
    }
    Ok(PoolHandle {
        tx,
        workers: handles,
    })
}

/// One portable backend worker: blocking `pread` through the row source.
fn pool_worker(source: &Arc<dyn RowTable>, rx: &Receiver<Job>, shared: &Arc<Shared>) {
    while let Ok(job) = rx.recv() {
        if shared.stop.load(Ordering::Relaxed) {
            // Every job still in the queue was accepted before this worker saw the stop flag and
            // no other worker will take it - this one is about to drop the receiver - so all of
            // them are retired here with their own reasons instead of being dropped while still
            // counted in flight. One job was covered before; the queue is as deep as the
            // configuration says, so several can be waiting (review A2, round 4).
            for abandoned in std::iter::once(job).chain(rx.try_iter()) {
                retire_unread(
                    shared,
                    &abandoned,
                    format!(
                        "the prefetch pool stopped before reading row {} of table {}; the row must be read inline",
                        abandoned.row, abandoned.table
                    ),
                );
            }
            break;
        }
        let result = {
            spark_trace_span!(
                TraceScope::ENGRAM_PREFETCH,
                "engram_prefetch_row",
                job.row as u32
            );
            source.read_row(job.table, job.row)
        };
        shared.in_flight.fetch_sub(1, Ordering::Relaxed);
        match result {
            Ok(data) => {
                shared.stats.pool_reads.fetch_add(1, Ordering::Relaxed);
                shared
                    .stats
                    .bytes_read
                    .fetch_add(ENGRAM_ROW_BYTES as u64, Ordering::Relaxed);
                let dropped = shared.ready.lock().insert(
                    job.key,
                    Entry {
                        data,
                        issued_step: job.issued_step,
                    },
                );
                if dropped > 0 {
                    shared.stats.wasted.fetch_add(dropped, Ordering::Relaxed);
                }
                shared.stats.completed.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                shared.stats.failed.fetch_add(1, Ordering::Relaxed);
                shared.record_failure(
                    job.key,
                    format!(
                        "the prefetch pool could not read row {} of table {}: {error}",
                        job.row, job.table
                    ),
                );
            }
        }
    }
}

impl PrefetchEngine {
    /// Current lookahead, in rows.
    #[must_use]
    pub fn lead_rows(&self) -> u64 {
        self.lead_rows.load(Ordering::Relaxed)
    }

    /// Changes the lookahead, in rows. Takes effect on the next `prefetch_ahead`.
    pub fn set_lead_rows(&self, rows: u64) {
        self.lead_rows.store(rows, Ordering::Relaxed);
    }

    /// Advances the decode step that lead distances are measured in.
    pub fn advance_step(&self) {
        self.shared.step.fetch_add(1, Ordering::Relaxed);
    }

    /// Decode step the engine is on.
    #[must_use]
    pub fn step(&self) -> u64 {
        self.shared.step.load(Ordering::Relaxed)
    }

    /// Rows completed but not yet consumed.
    #[must_use]
    pub fn ready_len(&self) -> usize {
        self.shared.ready.lock().len()
    }

    /// Requests accepted but not yet completed.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::Relaxed)
    }

    fn sender(&self) -> Option<Sender<Job>> {
        if let Some(pool) = &self.pool {
            return Some(pool.tx.clone());
        }
        #[cfg(feature = "io-uring")]
        if let Some(ring) = &self.ring {
            return Some(ring.tx.clone());
        }
        None
    }

    /// Requests one row.
    ///
    /// Returns `false` when the row is already ready or the request was refused because the
    /// queue was full; neither is an error, since the caller can always read the row
    /// synchronously.
    pub fn request(&self, table: usize, row: u64) -> bool {
        if row > u64::from(u32::MAX) {
            return false;
        }
        let Some(key) = job_key(table, row) else {
            return false;
        };
        if self.shared.ready.lock().contains(key) {
            return false;
        }
        let job = Job {
            table,
            row,
            key,
            issued_step: self.step(),
        };
        if self.backend == PrefetchBackend::Inline {
            // `issued` counts accepted rows on every backend and is counted before the read is
            // attempted: counting it only when the read succeeded left accepted rows
            // uncounted on a failure (review A5).
            self.shared.stats.issued.fetch_add(1, Ordering::Relaxed);
            self.read_inline(std::slice::from_ref(&job));
            return true;
        }
        match self.sender() {
            Some(tx) => {
                let accepted = self.shared.accept(&tx, job);
                if accepted {
                    self.shared.stats.issued.fetch_add(1, Ordering::Relaxed);
                }
                accepted
            }
            None => {
                self.shared
                    .stats
                    .backpressure
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Requests a batch of rows, split into `max_batch_rows` chunks so one batch never
    /// exceeds the queue depth. Returns how many rows were accepted.
    pub fn request_rows(&self, table: usize, rows: &[u32]) -> usize {
        let chunk_rows = self.config.max_batch_rows.max(1);
        let mut accepted = 0;
        for chunk in rows.chunks(chunk_rows) {
            if self.backend == PrefetchBackend::Inline {
                let jobs: Vec<Job> = chunk
                    .iter()
                    .filter_map(|row| {
                        let row = u64::from(*row);
                        let key = job_key(table, row)?;
                        if self.shared.ready.lock().contains(key) {
                            return None;
                        }
                        Some(Job {
                            table,
                            row,
                            key,
                            issued_step: self.step(),
                        })
                    })
                    .collect();
                if !jobs.is_empty() {
                    accepted += jobs.len();
                    self.shared
                        .stats
                        .issued
                        .fetch_add(jobs.len() as u64, Ordering::Relaxed);
                    self.read_inline(&jobs);
                }
            } else {
                for row in chunk {
                    if self.request(table, u64::from(*row)) {
                        accepted += 1;
                    }
                }
            }
        }
        accepted
    }

    /// Issues requests for the `lead_rows` rows that follow the consumer's cursor.
    ///
    /// `rows` must be sorted ascending, which is how an Engram lookup batch is walked.
    /// Returns how many requests were accepted.
    pub fn prefetch_ahead(&self, table: usize, rows: &[u32], cursor: usize) -> usize {
        let start = cursor.saturating_add(1);
        if start >= rows.len() {
            return 0;
        }
        let lead = usize::try_from(self.lead_rows()).unwrap_or(usize::MAX);
        let end = start.saturating_add(lead).min(rows.len());
        self.request_rows(table, &rows[start..end])
    }

    /// Takes a completed row, recording how many decode steps of lead it arrived with.
    #[must_use]
    pub fn take(&self, table: usize, row: u64) -> Option<[u8; ENGRAM_ROW_BYTES]> {
        let key = job_key(table, row)?;
        let entry = self.shared.ready.lock().take(key)?;
        let lead = self.step().saturating_sub(entry.issued_step);
        self.shared.stats.consumed.fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .lead_steps_total
            .fetch_add(lead, Ordering::Relaxed);
        self.shared
            .stats
            .lead_samples
            .fetch_add(1, Ordering::Relaxed);
        Some(entry.data)
    }

    /// Takes a prefetched row, or reads it synchronously through the row source.
    ///
    /// # Errors
    /// When no prefetch was ready and the synchronous read fails.
    pub fn fill_or_read(&self, table: usize, row: u64) -> anyhow::Result<[u8; ENGRAM_ROW_BYTES]> {
        if let Some(data) = self.take(table, row) {
            return Ok(data);
        }
        // A request that was accepted and then failed must not look like a miss: report what
        // went wrong, and still try the synchronous read so a transient prefetch failure does
        // not become a hard failure.
        let failure = job_key(table, row).and_then(|key| self.shared.take_failure(key));
        match (failure, self.source.read_row(table, row)) {
            (_, Ok(data)) => Ok(data),
            (Some(message), Err(error)) => {
                anyhow::bail!("{message}; the synchronous read also failed: {error}")
            }
            (None, Err(error)) => Err(error),
        }
    }

    /// Drops a completed row without consuming it, counting it as wasted.
    pub fn discard(&self, table: usize, row: u64) -> bool {
        let Some(key) = job_key(table, row) else {
            return false;
        };
        let dropped = self.shared.ready.lock().take(key).is_some();
        if dropped {
            self.shared.stats.wasted.fetch_add(1, Ordering::Relaxed);
        }
        dropped
    }

    /// Drops every completed row, returning how many were thrown away.
    pub fn flush(&self) -> u64 {
        let dropped = self.shared.ready.lock().clear();
        if dropped > 0 {
            self.shared
                .stats
                .wasted
                .fetch_add(dropped, Ordering::Relaxed);
        }
        dropped
    }

    /// Snapshot of the counters and queue state.
    #[must_use]
    pub fn snapshot(&self) -> PrefetchSnapshot {
        let stats = &self.shared.stats;
        let samples = stats.lead_samples.load(Ordering::Relaxed);
        PrefetchSnapshot {
            issued: stats.issued.load(Ordering::Relaxed),
            completed: stats.completed.load(Ordering::Relaxed),
            wasted: stats.wasted.load(Ordering::Relaxed),
            failed: stats.failed.load(Ordering::Relaxed),
            pool_read_failures: stats.pool_read_failures.load(Ordering::Relaxed),
            backpressure: stats.backpressure.load(Ordering::Relaxed),
            channel_dead: stats.channel_dead.load(Ordering::Relaxed),
            consumed: stats.consumed.load(Ordering::Relaxed),
            bytes_read: stats.bytes_read.load(Ordering::Relaxed),
            avg_lead_steps: if samples == 0 {
                0.0
            } else {
                stats.lead_steps_total.load(Ordering::Relaxed) as f64 / samples as f64
            },
            lead_samples: samples,
            queue_depth: self.config.queue_depth,
            lead_rows: self.lead_rows(),
            max_batch_rows: self.config.max_batch_rows,
            ready: self.ready_len(),
            ready_order: self.shared.ready.lock().order_len(),
            in_flight: self.in_flight(),
            backend: self.backend,
            ring_submissions: stats.ring_submissions.load(Ordering::Relaxed),
            pool_reads: stats.pool_reads.load(Ordering::Relaxed),
            ring_stalled: self.shared.ring_stalled.load(Ordering::Relaxed),
        }
    }

    /// Serves jobs on the calling thread, merging the rows of each table into one read.
    ///
    /// `issued` is not counted here: it belongs to the two acceptance points above, so an
    /// accepted row is counted exactly once whatever the read then does.
    ///
    /// A read covers a contiguous group of rows and is all-or-nothing: when
    /// `read_rows_into` fails, none of the group's rows could be read, so `failed` grows by
    /// the whole group while `pool_read_failures` grows by one. The ThreadPool backend reads
    /// a row at a time, so three bad rows fail three rows there and one group here. Each
    /// affected row also gets its reason recorded, so `fill_or_read` reports the failure
    /// instead of leaving the caller with a silent miss.
    fn read_inline(&self, jobs: &[Job]) {
        let mut tables: Vec<usize> = jobs.iter().map(|job| job.table).collect();
        tables.sort_unstable();
        tables.dedup();
        for table in tables {
            let group: Vec<&Job> = jobs.iter().filter(|job| job.table == table).collect();
            let wanted: Vec<u32> = group.iter().map(|job| job.row as u32).collect();
            let mut out = vec![[0u8; ENGRAM_ROW_BYTES]; group.len()];
            match self.source.read_rows_into(table, &wanted, &mut out) {
                Ok(()) => {
                    let rows = group.len() as u64;
                    self.shared.stats.pool_reads.fetch_add(1, Ordering::Relaxed);
                    self.shared
                        .stats
                        .bytes_read
                        .fetch_add(rows * ENGRAM_ROW_BYTES as u64, Ordering::Relaxed);
                    for (job, data) in group.iter().zip(out) {
                        let dropped = self.shared.ready.lock().insert(
                            job.key,
                            Entry {
                                data,
                                issued_step: job.issued_step,
                            },
                        );
                        if dropped > 0 {
                            self.shared
                                .stats
                                .wasted
                                .fetch_add(dropped, Ordering::Relaxed);
                        }
                        self.shared.stats.completed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(error) => {
                    // One read failed and every row it covered is unreadable. Both counters
                    // are bumped so the granularity is visible rather than inferred.
                    self.shared
                        .stats
                        .pool_read_failures
                        .fetch_add(1, Ordering::Relaxed);
                    self.shared
                        .stats
                        .failed
                        .fetch_add(group.len() as u64, Ordering::Relaxed);
                    for job in &group {
                        self.shared.record_failure(
                            job.key,
                            format!(
                                "the inline read of row {} of table {} failed: {error}",
                                job.row, job.table
                            ),
                        );
                    }
                }
            }
        }
    }
}

impl Drop for PrefetchEngine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(pool) = self.pool.take() {
            drop(pool.tx);
            for worker in pool.workers {
                let _ = worker.join();
            }
        }
        #[cfg(feature = "io-uring")]
        if let Some(ring) = self.ring.take() {
            ring.stop();
            drop(ring.tx);
            let _ = ring.worker.join();
        }
    }
}

/// One in-flight ring read.
///
/// Compiled for the ring and for the tests that drive its retirement seam; without the feature
/// nothing outside the tests uses it, and a production build must not carry it (review A1,
/// round 4).
#[cfg(any(feature = "io-uring", test))]
struct InFlight {
    job: Job,
    /// File offset of the read, 512-byte aligned.
    offset: u64,
    /// Length of the read, 512-byte aligned.
    len: u32,
}

/// Moves every live slot out, so retiring a lost completion cannot reuse its buffer.
#[cfg(any(feature = "io-uring", test))]
fn retire_slots(slots: &mut [Option<InFlight>]) -> Vec<InFlight> {
    let mut retired = Vec::new();
    for slot in slots.iter_mut() {
        if let Some(in_flight) = slot.take() {
            retired.push(in_flight);
        }
    }
    retired
}

/// Retires every accepted request of a worker that stopped serving: the one place where
/// acceptance and retirement are ordered.
///
/// Under the gate it raises the stall flag, runs `drain` - which is the caller's own
/// slot/backlog/channel walk, and is a closure so a test can pin this exact point - and brings
/// the shared in-flight count back to zero. Setting the flag before the drain is what makes
/// `Shared::accept` refuse every request that comes afterwards, so no request can be accepted
/// into a queue this retirement has already walked (review A1, round 4).
#[cfg(any(feature = "io-uring", test))]
fn retire_accepted<D>(shared: &Shared, drain: D) -> (Vec<InFlight>, Vec<Job>)
where
    D: FnOnce() -> (Vec<InFlight>, Vec<Job>),
{
    let _gate = shared.retire_gate.lock();
    shared.ring_stalled.store(true, Ordering::Relaxed);
    let (retired, queued) = drain();
    let lost = retired.len() + queued.len();
    // A compare-exchange loop rather than fetch_update: fetch_update is deprecated in favour of
    // try_update on newer stable toolchains, this repository builds on the stable channel, and
    // naming either function ties the code to one release. The loop is what fetch_update does
    // internally, and it compiles on every toolchain the workspace supports.
    let mut observed = shared.in_flight.load(Ordering::Relaxed);
    loop {
        let next = observed.saturating_sub(lost);
        match shared.in_flight.compare_exchange_weak(
            observed,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(current) => observed = current,
        }
    }
    (retired, queued)
}

/// Records a lost completion: one failure per lost request and a reason for each waiter.
/// `retired` are the requests that were in the ring; `queued` are the accepted requests that
/// never reached it, which are just as lost because the worker that would have submitted them is
/// exiting. The stall flag and the shared count belong to [retire_accepted], which owns the
/// ordering against `Shared::accept`.
#[cfg(any(feature = "io-uring", test))]
fn record_lost_completion(
    shared: &Shared,
    retired: &[InFlight],
    queued: &[Job],
    timeout: std::time::Duration,
) -> usize {
    for in_flight in retired {
        shared.stats.failed.fetch_add(1, Ordering::Relaxed);
        shared.record_failure(
            in_flight.job.key,
            format!(
                "the io_uring completion for row {} of table {} never arrived within {timeout:?}; the ring was stopped and the row must be read synchronously",
                in_flight.job.row, in_flight.job.table
            ),
        );
    }
    for job in queued {
        shared.stats.failed.fetch_add(1, Ordering::Relaxed);
        shared.record_failure(
            job.key,
            format!(
                "row {} of table {} was accepted but never submitted when the ring stopped after a lost completion; the row must be read synchronously",
                job.row, job.table
            ),
        );
    }
    retired.len() + queued.len()
}

/// Retires an accepted job this worker will never read, with its own reason, so its waiter is
/// told why instead of waiting for a row nobody will produce (review A2).
fn retire_unread(shared: &Shared, job: &Job, reason: String) {
    shared.in_flight.fetch_sub(1, Ordering::Relaxed);
    shared.stats.failed.fetch_add(1, Ordering::Relaxed);
    shared.record_failure(job.key, reason);
}

/// The optional `io_uring` backend.
#[cfg(feature = "io-uring")]
mod uring {
    use std::collections::VecDeque;
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::time::{Duration, Instant};

    use io_uring::{IoUring, opcode, types};

    use super::{
        AtomicBool, ENGRAM_ROW_BYTES, Entry, InFlight, Job, Ordering, PrefetchConfig, RowTable,
        Sender, Shared, record_lost_completion, retire_accepted, retire_slots,
    };
    use crate::store::AlignedBuffer;
    use sparkflash_trace::{TraceScope, spark_trace_span};

    /// Bytes one ring slot reads for a device with `align`-byte logical blocks: a row starts
    /// at an arbitrary 8-byte offset inside one alignment unit, so two units always cover a
    /// 264-byte row. On a 512-byte device that is the 1024 bytes this slot has always been;
    /// a 4096-byte device needs 8192.
    fn slot_bytes(align: u64) -> usize {
        (2 * align.max(crate::store::MIN_ALIGN_BYTES)) as usize
    }

    /// How long the start-up probe waits for its completion.
    const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

    /// How long `start` waits for the worker to report readiness.
    const START_TIMEOUT: Duration = Duration::from_secs(5);

    /// How long the worker waits for a completion before it concludes the ring lost one.
    ///
    /// `submit_and_wait` with no bound is what let a single lost CQE wedge the worker inside
    /// `io_uring_enter` and `Drop`'s join with it (review item 3). The wait is bounded by this
    /// timeout instead. A timeout with requests outstanding is treated as unrecoverable for
    /// slot reuse, because the kernel may still write the buffer the slot points at: reusing
    /// the slot could publish a torn row, which is exactly what the short-completion guard
    /// exists to prevent. The worker therefore retires every in-flight request as failed,
    /// marks the ring stalled and exits, leaving those rows to the synchronous path. A false
    /// positive on a pathologically slow device costs the ring, not correctness.
    const CQE_TIMEOUT: Duration = Duration::from_millis(250);

    /// What one bounded completion wait saw.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WaitOutcome {
        /// At least one completion is ready to be reaped.
        Satisfied,
        /// The bounded wait expired with requests outstanding, so no completion is coming.
        TimedOut,
        /// The submission itself failed with an errno other than `ETIME`.
        Failed(i32),
    }

    /// Linux `ETIME`: "Timer expired". The kernel reports an expired bounded wait in
    /// `submit_with_args` with this errno, which the standard library maps to
    /// `ErrorKind::Uncategorized` rather than to `ErrorKind::TimedOut`, so it has to be
    /// matched by its raw value.
    const ETIME: i32 = 62;

    /// The engine's handle on the ring worker.
    pub(super) struct RingHandle {
        pub(super) tx: Sender<Job>,
        pub(super) worker: std::thread::JoinHandle<()>,
        stop: std::sync::Arc<AtomicBool>,
    }

    impl RingHandle {
        /// Asks the worker to stop; `Drop` joins it.
        pub(super) fn stop(&self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Opens every table and starts a ring worker, failing when the ring cannot be created
    /// or its probe read does not complete in time.
    pub(super) fn start(
        source: &std::sync::Arc<dyn RowTable>,
        config: &PrefetchConfig,
        shared: &std::sync::Arc<Shared>,
    ) -> anyhow::Result<RingHandle> {
        let depth = config.queue_depth.max(1);
        let (tx, rx) = crossbeam_channel::bounded::<Job>(depth);
        let (ready_tx, ready_rx) = crossbeam_channel::bounded::<anyhow::Result<()>>(1);
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let worker_stop = std::sync::Arc::clone(&stop);
        let source = std::sync::Arc::clone(source);
        let shared = std::sync::Arc::clone(shared);
        let worker = std::thread::Builder::new()
            .name("engram-io-uring".to_string())
            .spawn(
                move || match RingWorker::new(&source, depth, &shared, rx, worker_stop) {
                    Ok(mut worker) => {
                        let _ = ready_tx.send(Ok(()));
                        worker.run();
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                },
            )
            .map_err(|error| anyhow::anyhow!("spawn io_uring worker: {error}"))?;
        match ready_rx.recv_timeout(START_TIMEOUT) {
            Ok(Ok(())) => Ok(RingHandle { tx, worker, stop }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                stop.store(true, Ordering::Relaxed);
                let _ = worker.join();
                anyhow::bail!("the io_uring worker did not become ready within {START_TIMEOUT:?}")
            }
        }
    }

    /// Test-only alignment override, keyed by table path.
    ///
    /// A ring test has to be able to drive a 4096-byte-logical-block device on a filesystem that
    /// reports 512: that combination is the one the tail case needs, and the one a real rank file
    /// lands on. Keyed by path so tests running beside it keep their own filesystem's alignment
    /// (review A3, round 4).
    #[cfg(test)]
    static ALIGN_OVERRIDE: std::sync::Mutex<Vec<(std::path::PathBuf, u64)>> =
        std::sync::Mutex::new(Vec::new());

    /// Forces the alignment this worker uses for one table path; tests only.
    #[cfg(test)]
    pub(super) fn force_alignment(path: &std::path::Path, align: u64) {
        let mut forced = ALIGN_OVERRIDE.lock().expect("the alignment override");
        forced.retain(|(forced, _)| forced != path);
        forced.push((path.to_path_buf(), align));
    }

    /// The `O_DIRECT` alignment of one table: forced in tests, else asked of the filesystem.
    fn alignment_for(path: &std::path::Path) -> u64 {
        #[cfg(test)]
        {
            if let Some((_, align)) = ALIGN_OVERRIDE
                .lock()
                .expect("the alignment override")
                .iter()
                .find(|(forced, _)| forced == path)
            {
                return *align;
            }
        }
        crate::store::device_alignment(path)
    }

    /// The ring, its descriptors and its buffers, all owned by one thread.
    struct RingWorker {
        ring: IoUring,
        files: Vec<File>,
        file_bytes: Vec<u64>,
        buffers: Vec<AlignedBuffer>,
        slots: Vec<Option<InFlight>>,
        pending: VecDeque<Job>,
        rx: crossbeam_channel::Receiver<Job>,
        shared: std::sync::Arc<Shared>,
        stop: std::sync::Arc<AtomicBool>,
        in_flight: usize,
        /// `O_DIRECT` alignment every SQE has to respect: the largest logical block size
        /// among the tables this worker serves.
        align: u64,
        /// Bytes one slot reads, two alignment units.
        slot_bytes: usize,
    }

    impl RingWorker {
        fn new(
            source: &std::sync::Arc<dyn RowTable>,
            depth: usize,
            shared: &std::sync::Arc<Shared>,
            rx: crossbeam_channel::Receiver<Job>,
            stop: std::sync::Arc<AtomicBool>,
        ) -> anyhow::Result<Self> {
            let tables = source.table_count();
            anyhow::ensure!(tables >= 1, "the row source exposes no tables");
            let mut files = Vec::with_capacity(tables);
            let mut file_bytes = Vec::with_capacity(tables);
            let mut align = crate::store::MIN_ALIGN_BYTES;
            for table in 0..tables {
                let spec = source
                    .spec(table)
                    .ok_or_else(|| anyhow::anyhow!("table {table} has no spec"))?;
                let table_align = alignment_for(&spec.path);
                // The ring reads through buffers aligned to the scratch alignment; a device
                // that needs more cannot be served by any SQE, so the engine falls back to
                // the pool (or errors, when the ring was required) instead of issuing reads
                // the kernel will refuse.
                anyhow::ensure!(
                    table_align <= crate::store::MAX_ALIGN_BYTES,
                    "the device behind {} needs a {table_align}-byte O_DIRECT alignment, \
                     more than the {} byte maximum",
                    spec.path.display(),
                    crate::store::MAX_ALIGN_BYTES
                );
                align = align.max(table_align);
                let file = crate::store::open_direct(&spec.path).or_else(|direct| {
                    File::open(&spec.path).map_err(|plain| {
                        anyhow::anyhow!(
                            "open {} for the ring failed (O_DIRECT: {direct}; buffered: {plain})",
                            spec.path.display()
                        )
                    })
                })?;
                let length = file
                    .metadata()
                    .map_err(|error| anyhow::anyhow!("stat {}: {error}", spec.path.display()))?
                    .len();
                let needed = spec.physical_rows * ENGRAM_ROW_BYTES as u64;
                anyhow::ensure!(
                    length >= needed && length.is_multiple_of(crate::store::MIN_ALIGN_BYTES),
                    "{} is {length} bytes, which cannot serve {} rows",
                    spec.path.display(),
                    spec.physical_rows
                );
                files.push(file);
                file_bytes.push(length);
            }
            let slot_bytes = slot_bytes(align);
            let ring = IoUring::new(depth as u32).map_err(|error| {
                anyhow::anyhow!("create an io_uring ring of depth {depth}: {error}")
            })?;
            let mut buffers = Vec::with_capacity(depth);
            for _ in 0..depth {
                buffers.push(AlignedBuffer::new(slot_bytes));
            }
            let mut worker = Self {
                ring,
                files,
                file_bytes,
                buffers,
                slots: (0..depth).map(|_| None).collect(),
                pending: VecDeque::new(),
                rx,
                shared: std::sync::Arc::clone(shared),
                stop,
                in_flight: 0,
                align,
                slot_bytes,
            };
            worker.probe()?;
            Ok(worker)
        }

        /// Reads the first row of the first table through the ring, proving that a
        /// submission actually completes before the engine depends on it.
        fn probe(&mut self) -> anyhow::Result<()> {
            let job = Job {
                table: 0,
                row: 0,
                key: 0,
                issued_step: 0,
            };
            if self.file_bytes[0] < ENGRAM_ROW_BYTES as u64 {
                anyhow::bail!("table 0 holds no rows");
            }
            let entry = self.build_entry(0, &job)?;
            let expected = self.slots[0].as_ref().map_or(0, |in_flight| in_flight.len);
            // SAFETY: slot 0 is free (nothing is in flight) and its buffer stays alive
            // until the completion is reaped.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|error| anyhow::anyhow!("push the ring probe: {error}"))?;
            }
            let _ = self.ring.submit();
            let deadline = Instant::now() + PROBE_TIMEOUT;
            while Instant::now() < deadline {
                let mut completed = false;
                for cqe in self.ring.completion() {
                    // A completion only proves the buffer holds the row when the kernel
                    // reports exactly the length the SQE asked for.
                    if cqe.result() != expected as i32 {
                        anyhow::bail!(
                            "the io_uring probe read of {expected} bytes returned {}",
                            cqe.result()
                        );
                    }
                    completed = true;
                }
                if completed {
                    // Free the probe's slot so it can serve a real request.
                    self.slots[0] = None;
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            anyhow::bail!("the io_uring probe read did not complete within {PROBE_TIMEOUT:?}")
        }

        /// Builds the read SQE for `job` in `slot`.
        fn build_entry(
            &mut self,
            slot: usize,
            job: &Job,
        ) -> anyhow::Result<io_uring::squeue::Entry> {
            let row_bytes = ENGRAM_ROW_BYTES as u64;
            let offset = job.row * row_bytes;
            let align = self.align;
            let start = offset / align * align;
            let end = (offset + row_bytes).div_ceil(align) * align;
            let len = end - start;
            // The window may run past the end of a file whose length is not an alignment
            // multiple - every packed rank file is padded to 512, not to the 4096 a 4Kn device
            // needs - and an aligned O_DIRECT read that reaches past EOF completes short instead
            // of failing. What must exist is the row itself: `reap` publishes a row only when
            // the completion covers its bytes, so a window that overruns the padded end is
            // served rather than refused. Refusing it is what stopped the ring from ever serving
            // the last rows of a rank file (review A3, round 4).
            let row_end = offset + row_bytes;
            anyhow::ensure!(
                len as usize <= self.slot_bytes && row_end <= self.file_bytes[job.table],
                "row {} of table {} ends at {row_end}, past the {} byte file",
                job.row,
                job.table,
                self.file_bytes[job.table]
            );
            let buffer = &mut self.buffers[slot];
            let pointer = buffer.as_mut_slice().as_mut_ptr();
            let fd = types::Fd(self.files[job.table].as_raw_fd());
            self.slots[slot] = Some(InFlight {
                job: *job,
                offset: start,
                len: len as u32,
            });
            Ok(opcode::Read::new(fd, pointer, len as u32)
                .offset(start)
                .build()
                .user_data(slot as u64))
        }

        fn run(&mut self) {
            loop {
                if self.stop.load(Ordering::Relaxed)
                    && self.pending.is_empty()
                    && self.in_flight == 0
                {
                    break;
                }
                self.fill_ring();
                if self.in_flight > 0 {
                    match self.wait_for_completions() {
                        WaitOutcome::Satisfied => {
                            self.reap();
                        }
                        WaitOutcome::TimedOut => {
                            // The bounded wait expired with requests outstanding, so no
                            // completion is coming for them.
                            let retired = self.retire_after_lost_completion();
                            spark_trace_span!(
                                TraceScope::ENGRAM_IO_URING,
                                "engram_io_uring_stalled",
                                retired as u32
                            );
                            break;
                        }
                        WaitOutcome::Failed(errno) => {
                            // A submission that keeps failing is a stall: retiring here is what
                            // keeps the promise the bounded wait makes (review A5, round 4). The
                            // errno is the span's meta; the retired requests are counted as
                            // failures by the retirement itself.
                            let _retired = self.retire_after_lost_completion();
                            spark_trace_span!(
                                TraceScope::ENGRAM_IO_URING,
                                "engram_io_uring_wait_failed",
                                errno as u32
                            );
                            break;
                        }
                    }
                } else {
                    match self.rx.recv_timeout(Duration::from_micros(200)) {
                        Ok(job) => self.pending.push_back(job),
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            }
        }

        /// Moves pending jobs into the ring while a slot and an SQ entry are free.
        fn fill_ring(&mut self) {
            while self.in_flight < self.slots.len() {
                if self.pending.is_empty() {
                    match self.rx.try_recv() {
                        Ok(job) => self.pending.push_back(job),
                        Err(_) => break,
                    }
                }
                let Some(job) = self.pending.pop_front() else {
                    break;
                };
                let Some(slot) = self.slots.iter().position(Option::is_none) else {
                    self.pending.push_front(job);
                    break;
                };
                match self.build_entry(slot, &job) {
                    Ok(entry) => {
                        let submissions = {
                            spark_trace_span!(
                                TraceScope::ENGRAM_IO_URING,
                                "engram_io_uring_submit",
                                slot as u32
                            );
                            // SAFETY: slot is free and the SQE points at its buffer, which
                            // is not moved until the completion is reaped.
                            unsafe { self.ring.submission().push(&entry) }
                        };
                        if let Err(error) = submissions {
                            self.slots[slot] = None;
                            self.shared.in_flight.fetch_sub(1, Ordering::Relaxed);
                            self.shared.stats.failed.fetch_add(1, Ordering::Relaxed);
                            self.shared.record_failure(
                                job.key,
                                format!(
                                    "submitting row {} of table {} to io_uring failed: {error}",
                                    job.row, job.table
                                ),
                            );
                            continue;
                        }
                        self.in_flight += 1;
                        // `issued` belongs to `request`, which counts one accepted request
                        // per row; the ring counts its own submissions.
                        self.shared
                            .stats
                            .ring_submissions
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(error) => {
                        self.shared.in_flight.fetch_sub(1, Ordering::Relaxed);
                        self.shared.stats.failed.fetch_add(1, Ordering::Relaxed);
                        self.shared.record_failure(
                            job.key,
                            format!(
                                "row {} of table {} cannot be read through the ring: {error}",
                                job.row, job.table
                            ),
                        );
                    }
                }
            }
        }

        /// Collects completions and publishes their rows.
        ///
        /// A completion only proves that the slot buffer holds the row when the kernel
        /// reports exactly the length the SQE asked for. Anything else - a short read after
        /// the file changed underneath the ring, or a negative errno - is a failure: the row
        /// is never published from the reused slot buffer, the attempt is counted failed, the
        /// waiter is told why through PrefetchEngine::fill_or_read, and the accepted request
        /// is retired so in_flight returns to zero.
        /// Returns how many completions it collected. Progress is decided by
        /// [`Self::wait_for_completions`], not by this count, which `run` discards
        /// (review A4, round 4).
        fn reap(&mut self) -> usize {
            let mut reaped = 0usize;
            for cqe in self.ring.completion() {
                let slot = cqe.user_data() as usize;
                reaped += 1;
                let result = cqe.result();
                let Some(in_flight) = self.slots.get_mut(slot).and_then(Option::take) else {
                    // A completion for a slot this worker no longer owns (the slot set was
                    // retired under it) must not move either counter: decrementing before the
                    // lookup leaked the shared count (review A7).
                    continue;
                };
                // One accepted request is retired here, whatever the outcome, and both counters
                // move together for it.
                self.in_flight = self.in_flight.saturating_sub(1);
                self.shared.in_flight.fetch_sub(1, Ordering::Relaxed);
                // The guard is the row's own span, not the request length: an aligned window
                // can overrun the padded end of the file, so the kernel may return fewer bytes
                // than were asked for while still filling the buffer with the whole row. A
                // completion that does not cover the row is the failure, because publishing
                // from the reused slot buffer would hand the consumer another row's bytes
                // (review A3, round 4).
                let needed = i32::try_from(
                    in_flight.job.row * ENGRAM_ROW_BYTES as u64 + ENGRAM_ROW_BYTES as u64
                        - in_flight.offset,
                )
                .unwrap_or(i32::MAX);
                if result < needed {
                    self.shared.stats.failed.fetch_add(1, Ordering::Relaxed);
                    self.shared.record_failure(
                        in_flight.job.key,
                        format!(
                            "the io_uring read of row {} of table {} asked for {} bytes at {} and returned {}, which does not cover the {needed} bytes the row needs",
                            in_flight.job.row,
                            in_flight.job.table,
                            in_flight.len,
                            in_flight.offset,
                            result
                        ),
                    );
                    continue;
                }
                let begin =
                    (in_flight.job.row * ENGRAM_ROW_BYTES as u64 - in_flight.offset) as usize;
                let data = &self.buffers[slot].as_slice()[begin..begin + ENGRAM_ROW_BYTES];
                let mut row = [0u8; ENGRAM_ROW_BYTES];
                row.copy_from_slice(data);
                let dropped = self.shared.ready.lock().insert(
                    in_flight.job.key,
                    Entry {
                        data: row,
                        issued_step: in_flight.job.issued_step,
                    },
                );
                if dropped > 0 {
                    self.shared
                        .stats
                        .wasted
                        .fetch_add(dropped, Ordering::Relaxed);
                }
                self.shared.stats.completed.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .bytes_read
                    .fetch_add(u64::from(in_flight.len), Ordering::Relaxed);
            }
            reaped
        }

        /// Waits for at least one completion, bounded by `CQE_TIMEOUT`.
        ///
        /// The wait is satisfied, expired, or failed. A persistent submission error - EINVAL,
        /// EBUSY - used to be reported as a satisfied wait for every errno but `ETIME`, which
        /// left the worker reaping nothing and spinning with requests outstanding: the opposite
        /// of the bounded wait this constant exists to provide, so a failed submission is a
        /// stall too (review A5, round 4).
        fn wait_for_completions(&self) -> WaitOutcome {
            let timeout = types::Timespec::from(CQE_TIMEOUT);
            let args = types::SubmitArgs::new().timespec(&timeout);
            match self.ring.submitter().submit_with_args(1, &args) {
                Ok(_) => WaitOutcome::Satisfied,
                Err(error) => match error.raw_os_error() {
                    Some(ETIME) => WaitOutcome::TimedOut,
                    other => WaitOutcome::Failed(other.unwrap_or_default()),
                },
            }
        }

        /// Retires every accepted request after the ring stopped delivering completions.
        ///
        /// The slot buffers may still be written by the kernel, so the slots are dropped rather
        /// than reused and the worker exits with its ring. Accepted requests live in three
        /// places, not one: the ring's live slots, this worker's local backlog, and the shared
        /// channel. The worker is about to exit, so nothing downstream will ever complete the
        /// last two - and because the channel capacity is a second queue on top of the ring
        /// depth, up to twice the ring depth of accepted requests can exist. Leaving them behind
        /// kept them counted in the shared in-flight count forever and gave their waiters no
        /// reason (review A1). Every one of them is counted failed, gets its own reason, and the
        /// shared count returns to zero. Returns how many were lost.
        fn retire_after_lost_completion(&mut self) -> usize {
            let shared = std::sync::Arc::clone(&self.shared);
            let (retired, queued) = retire_accepted(&shared, || {
                let retired = retire_slots(&mut self.slots);
                let mut queued: Vec<Job> = self.pending.drain(..).collect();
                queued.extend(self.rx.try_iter());
                (retired, queued)
            });
            self.in_flight = 0;
            record_lost_completion(&shared, &retired, &queued, CQE_TIMEOUT)
        }
    }

    #[cfg(test)]
    mod timeout_tests {
        use std::collections::VecDeque;
        use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        use io_uring::{IoUring, types};

        use super::super::{InFlight, record_lost_completion, retire_accepted, retire_slots};
        use super::{CQE_TIMEOUT, ETIME};
        use crate::prefetch::{Job, Mutex, PrefetchStats, ReadyMap, Shared};

        fn shared_with(in_flight: usize) -> Shared {
            Shared {
                ready: Mutex::new(ReadyMap::new(8)),
                stats: PrefetchStats::default(),
                step: AtomicU64::new(0),
                in_flight: AtomicUsize::new(in_flight),
                stop: AtomicBool::new(false),
                failures: Mutex::new(VecDeque::new()),
                ring_stalled: AtomicBool::new(false),
                retire_gate: Mutex::new(()),
            }
        }

        fn in_flight(row: u64, slot: usize) -> InFlight {
            InFlight {
                job: Job {
                    table: 0,
                    row,
                    key: row + 1,
                    issued_step: 0,
                },
                offset: (slot * 512) as u64,
                len: 1024,
            }
        }

        /// The bounded wait returns a timeout instead of blocking. This is the property that
        /// stops a lost completion from wedging the worker inside `io_uring_enter`, and with
        /// it `Drop`'s join.
        #[test]
        fn the_completion_wait_returns_a_timeout() {
            let ring = IoUring::new(8).expect("an io_uring ring");
            let timeout = types::Timespec::from(Duration::from_millis(20));
            let args = types::SubmitArgs::new().timespec(&timeout);
            let started = Instant::now();
            let error = ring
                .submitter()
                .submit_with_args(1, &args)
                .expect_err("nothing was submitted, so nothing can complete");
            let elapsed = started.elapsed();
            assert_eq!(
                error.raw_os_error(),
                Some(ETIME),
                "the bounded wait must report the timeout as ETIME: {error:?}"
            );
            assert!(
                elapsed >= Duration::from_millis(10),
                "the wait returned before its timeout: {elapsed:?}"
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "the wait was not bounded: {elapsed:?}"
            );
        }

        /// A lost completion retires every request that was in the ring, tells each waiter why,
        /// clears the slots and publishes nothing.
        ///
        /// This covers the ring's own slots; it builds them by hand, so it cannot see an accepted
        /// request that was still in the worker's backlog or in the shared queue. Those are
        /// covered by a_stall_retires_the_slots_the_backlog_and_the_queue, which drives the
        /// worker's own retirement path.
        #[test]
        fn a_lost_completion_retires_every_request_in_the_ring() {
            let shared = shared_with(2);
            let mut slots = vec![Some(in_flight(4, 0)), None, Some(in_flight(9, 1))];
            let (retired, queued) =
                retire_accepted(&shared, || (retire_slots(&mut slots), Vec::new()));
            assert_eq!(retired.len(), 2, "a free slot is not a request");
            assert!(slots.iter().all(Option::is_none));
            assert!(queued.is_empty());
            assert_eq!(
                record_lost_completion(&shared, &retired, &queued, CQE_TIMEOUT),
                2
            );
            assert_eq!(shared.stats.failed.load(Ordering::Relaxed), 2);
            assert_eq!(
                shared.in_flight.load(Ordering::Relaxed),
                0,
                "every retired request must leave the shared in-flight count"
            );
            assert!(shared.ring_stalled.load(Ordering::Relaxed));
            assert_eq!(
                shared.ready.lock().len(),
                0,
                "a lost completion must never publish a row"
            );
            let reason = shared.take_failure(5).expect("row 4 waits on key 5");
            assert!(reason.contains("never arrived"), "{reason}");
            assert!(shared.take_failure(10).is_some());
            assert!(
                shared.take_failure(5).is_none(),
                "a failure is reported once"
            );
        }

        /// A stall with a backlog retires the ring's slots, the worker's own backlog and the jobs
        /// still waiting in the channel. Accepted requests live in all three, the worker exits on
        /// a stall, and the channel is a second queue on top of the ring depth, so the last two
        /// would otherwise stay counted in flight with no reason for their waiters - the defect
        /// review A1 found. A counter that cannot move cannot show this, so the assertions are
        /// completed + failed == accepted, a reason for every accepted key, and zero in flight.
        #[test]
        fn a_stall_retires_the_slots_the_backlog_and_the_queue() {
            let depth = 2;
            let (tx, rx) = crossbeam_channel::bounded::<Job>(depth);
            let shared = std::sync::Arc::new(shared_with(0));
            let mut worker = super::RingWorker {
                ring: IoUring::new(depth as u32 + 1).expect("an io_uring ring"),
                files: Vec::new(),
                file_bytes: Vec::new(),
                buffers: Vec::new(),
                // Two accepted requests in the ring, one in the worker's backlog, one queued.
                slots: vec![Some(in_flight(4, 0)), Some(in_flight(9, 1))],
                pending: vec![Job {
                    table: 0,
                    row: 16,
                    key: 17,
                    issued_step: 0,
                }]
                .into(),
                rx,
                shared: std::sync::Arc::clone(&shared),
                stop: std::sync::Arc::new(AtomicBool::new(false)),
                in_flight: depth,
                align: crate::store::MIN_ALIGN_BYTES,
                slot_bytes: 1024,
            };
            tx.try_send(Job {
                table: 0,
                row: 25,
                key: 26,
                issued_step: 0,
            })
            .expect("the queue has room");
            let accepted = 4usize;
            shared.in_flight.store(accepted, Ordering::Relaxed);
            shared
                .stats
                .issued
                .store(accepted as u64, Ordering::Relaxed);

            let lost = worker.retire_after_lost_completion();
            assert_eq!(lost, accepted, "every accepted request is retired");
            assert_eq!(
                shared.stats.completed.load(Ordering::Relaxed)
                    + shared.stats.failed.load(Ordering::Relaxed),
                accepted as u64,
                "completed + failed must account for every accepted request"
            );
            assert_eq!(shared.in_flight.load(Ordering::Relaxed), 0);
            assert!(worker.slots.iter().all(Option::is_none));
            assert!(worker.pending.is_empty());
            assert!(worker.rx.is_empty(), "the queue is drained, not abandoned");
            // Keys 5 and 10 were in the ring, 17 was in the worker's backlog and 26 was still in
            // the channel; each one gets exactly one reason.
            for key in [5u64, 10, 17, 26] {
                assert!(
                    shared.take_failure(key).is_some(),
                    "key {key} has no reason"
                );
            }
            for key in [5u64, 10, 17, 26] {
                assert!(shared.take_failure(key).is_none(), "one reason per request");
            }
        }

        /// The watchdog timeout is a documented, bounded constant rather than an accident of
        /// the syscall.
        #[test]
        fn the_stall_timeout_is_bounded() {
            assert!(CQE_TIMEOUT >= Duration::from_millis(10));
            assert!(CQE_TIMEOUT <= Duration::from_secs(5));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryTable, TableSpec, TailFill};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    fn spec(rows: u64) -> TableSpec {
        TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: PathBuf::from("/tmp/prefetch-never-opened.bin"),
            row_start: 0,
            physical_rows: rows,
            logical_rows: rows,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        }
    }

    /// The shared state a hand-built prefetch test needs. The io_uring tests keep their own copy
    /// of this, because a test of the portable path has to run without the io-uring feature.
    fn shared_for_tests(in_flight: usize) -> Shared {
        Shared {
            ready: Mutex::new(ReadyMap::new(8)),
            stats: PrefetchStats::default(),
            step: AtomicU64::new(0),
            in_flight: AtomicUsize::new(in_flight),
            stop: AtomicBool::new(false),
            failures: Mutex::new(VecDeque::new()),
            ring_stalled: AtomicBool::new(false),
            retire_gate: Mutex::new(()),
        }
    }

    /// A dead queue is not backpressure: it moves its own counter and raises the stall flag, so
    /// the consumer falls back to the synchronous path instead of reading a dead backend as a
    /// healthy, busy one (review A3).
    #[test]
    fn a_dead_queue_is_counted_as_stalled_not_as_busy() {
        let shared = shared_for_tests(0);
        let job = Job {
            table: 0,
            row: 7,
            key: 8,
            issued_step: 0,
        };
        // A real disconnected queue: dropping the receiver is what a stopping worker does.
        let (tx, rx) = crossbeam_channel::bounded::<Job>(1);
        drop(rx);
        let disconnected = tx.try_send(job).expect_err("the receiver is gone");
        assert!(matches!(
            disconnected,
            crossbeam_channel::TrySendError::Disconnected(_)
        ));
        assert!(!shared.note_refused_send(&disconnected));
        assert_eq!(shared.stats.channel_dead.load(Ordering::Relaxed), 1);
        assert_eq!(
            shared.stats.backpressure.load(Ordering::Relaxed),
            0,
            "a dead backend is not backpressure"
        );
        assert!(
            shared.ring_stalled.load(Ordering::Relaxed),
            "a dead backend must send the reader to the synchronous path"
        );
        // A full queue is the other case, and it must not look like a dead backend.
        let (tx, rx) = crossbeam_channel::bounded::<Job>(1);
        tx.try_send(job).expect("the first job fits");
        let full = tx.try_send(job).expect_err("the queue is full");
        assert!(matches!(full, crossbeam_channel::TrySendError::Full(_)));
        assert!(!shared.note_refused_send(&full));
        assert_eq!(shared.stats.backpressure.load(Ordering::Relaxed), 1);
        assert_eq!(shared.stats.channel_dead.load(Ordering::Relaxed), 1);
        drop(rx);
    }

    /// A worker that takes a job after the stop flag is set retires it: the job was accepted
    /// before the worker saw the flag, no other worker will take it, and dropping it would leave
    /// the shared in-flight count permanently high with no reason for its waiter (review A2).
    #[test]
    fn a_stopped_worker_retires_every_job_it_already_took() {
        let source = memory_table(8);
        let config = PrefetchConfig {
            queue_depth: 4,
            workers: 1,
            max_batch_rows: 1,
            ready_capacity: 8,
            ..config()
        };
        let shared = Arc::new(shared_for_tests(0));
        // Set before the worker starts, so the first job it takes is the hand-over race the fix
        // covers rather than a shutdown that happens to be a little later.
        shared.stop.store(true, Ordering::Relaxed);
        let pool = start_pool(&source, &config, &shared).expect("a portable pool");
        let rows = [3u64, 5, 7];
        for row in rows {
            shared.in_flight.fetch_add(1, Ordering::Relaxed);
            shared.stats.issued.fetch_add(1, Ordering::Relaxed);
            pool.tx
                .try_send(Job {
                    table: 0,
                    row,
                    key: row + 1,
                    issued_step: 0,
                })
                .expect("the pool is running");
        }
        for worker in pool.workers {
            worker.join().expect("the worker thread");
        }
        assert_eq!(shared.in_flight.load(Ordering::Relaxed), 0);
        assert_eq!(
            shared.stats.failed.load(Ordering::Relaxed),
            rows.len() as u64
        );
        for row in rows {
            let reason = shared
                .take_failure(row + 1)
                .expect("the waiter is told why");
            assert!(reason.contains("stopped"), "{reason}");
        }
    }

    /// The accept/retire escape (review A1, round 4): a request accepted into a queue the
    /// retirement has already walked is orphaned - counted in flight forever, with no reason for
    /// its waiter and no worker left to complete it. The reviewer reproduced it deterministically
    /// by pinning the worker between the drain and the count adjustment; this runs the engine's
    /// own accept path - Shared::accept, the same call PrefetchEngine::request makes - against a
    /// real channel after a real retirement, and asserts what the escape breaks: every accepted
    /// request has a reason, nothing is left in the queue, and the count returns to zero.
    #[test]
    fn a_request_cannot_escape_into_a_queue_the_retirement_walked() {
        let shared = Arc::new(shared_for_tests(0));
        let (tx, rx) = crossbeam_channel::bounded::<Job>(4);
        // Three accepted requests, one in each place the retirement has to walk: the ring's
        // slots, the worker's backlog, and the shared queue.
        let mut slots = vec![Some(InFlight {
            job: job_for(4),
            offset: 0,
            len: 1024,
        })];
        let mut pending = VecDeque::from([job_for(9)]);
        tx.try_send(job_for(12)).expect("the queue has room");
        shared.in_flight.store(3, Ordering::Relaxed);

        let (retired, queued) = retire_accepted(&shared, || {
            let retired = retire_slots(&mut slots);
            let mut queued: Vec<Job> = pending.drain(..).collect();
            queued.extend(rx.try_iter());
            (retired, queued)
        });
        assert_eq!(
            record_lost_completion(&shared, &retired, &queued, Duration::from_millis(250)),
            3,
            "the ring, the backlog and the queue are all retired"
        );
        assert_eq!(
            retired.len(),
            1,
            "the ring's own request is the retired one"
        );
        assert_eq!(retired[0].job.row, 4);
        assert_eq!(retired[0].offset, 0, "the retired request keeps its window");
        assert_eq!(retired[0].len, 1024);
        assert_eq!(shared.in_flight.load(Ordering::Relaxed), 0);
        assert!(rx.is_empty(), "the retirement drained the queue");

        // The escape: the consumer asks for another row after the worker stopped serving.
        let escaped = shared.accept(&tx, job_for(20));
        assert!(
            !escaped,
            "a request accepted after the retirement walked the queue is orphaned"
        );
        assert_eq!(
            shared.in_flight.load(Ordering::Relaxed),
            0,
            "a refused request must give its reservation back"
        );
        assert!(
            rx.try_recv().is_err(),
            "the refused request must not be left in the queue"
        );
        assert_eq!(
            shared.stats.channel_dead.load(Ordering::Relaxed),
            1,
            "the refusal is a dead backend, not backpressure"
        );
        assert_eq!(shared.stats.backpressure.load(Ordering::Relaxed), 0);
        for key in [5u64, 10, 13] {
            assert!(
                shared.take_failure(key).is_some(),
                "key {key} was accepted and must have a reason"
            );
        }
    }

    /// The gate orders acceptance against retirement (review round 5, item 2). The escape test
    /// above calls `accept` only after the retirement has returned, so it passes whichever order the
    /// two take the gate in; this probe pins the retirement inside its drain with a channel, spawns
    /// the accepts while the gate is provably held, and requires them to wait for it. An accept that
    /// answers around the gate is the ordering that lets a request slip into a queue the retirement
    /// has already walked.
    ///
    /// Nothing sleeps to win the race: the retirement blocks on `go`, so the gate stays held while
    /// the accepts are spawned, and the one bounded wait asserts that they have *not* returned,
    /// which the held gate guarantees.
    #[test]
    fn acceptance_waits_for_the_retirement_gate_and_is_then_refused() {
        let shared = Arc::new(shared_for_tests(0));
        let (tx, rx) = crossbeam_channel::bounded::<Job>(64);
        tx.try_send(job_for(4)).expect("the queue has room");
        shared.in_flight.store(1, Ordering::Relaxed);

        let (in_drain_tx, in_drain_rx) = crossbeam_channel::bounded::<()>(1);
        let (go_tx, go_rx) = crossbeam_channel::bounded::<()>(1);
        let drain_rx = rx.clone();
        let retirement = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                retire_accepted(&shared, || {
                    in_drain_tx.send(()).expect("the test waits for the drain");
                    go_rx.recv().expect("the test releases the drain");
                    let queued: Vec<Job> = drain_rx.try_iter().collect();
                    (Vec::new(), queued)
                })
            })
        };
        in_drain_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the retirement reaches its drain");

        // The witness: one accept spawned while the retirement holds the gate. It must wait.
        let (done_tx, done_rx) = crossbeam_channel::bounded::<bool>(17);
        {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            let done_tx = done_tx.clone();
            std::thread::spawn(move || {
                let accepted = shared.accept(&tx, job_for(20));
                done_tx.send(accepted).expect("the test collects");
            });
        }
        let early = done_rx.recv_timeout(Duration::from_millis(150));
        assert!(
            early.is_err(),
            "an accept returned {early:?} while the retirement held the gate; it must wait for it"
        );

        // The storm: sixteen accepts, none of which may be accepted behind the retirement.
        let mut storm = Vec::new();
        for row in 0..16u64 {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            let done_tx = done_tx.clone();
            storm.push(std::thread::spawn(move || {
                let accepted = shared.accept(&tx, job_for(100 + row));
                done_tx.send(accepted).expect("the test collects");
            }));
        }
        drop(done_tx);
        go_tx.send(()).expect("the drain is waiting");
        let (retired, queued) = retirement.join().expect("the retirement thread");
        let mut accepted = 0;
        for _ in 0..17 {
            if done_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("every accept reports")
            {
                accepted += 1;
            }
        }
        for thread in storm {
            thread.join().expect("a storm thread");
        }

        assert_eq!(accepted, 0, "every accept behind the gate is refused");
        assert_eq!(
            shared.stats.channel_dead.load(Ordering::Relaxed),
            17,
            "the witness and the sixteen storm accepts are all dead-backend refusals"
        );
        assert_eq!(shared.stats.backpressure.load(Ordering::Relaxed), 0);
        assert_eq!(
            shared.in_flight.load(Ordering::Relaxed),
            0,
            "no refused request keeps its reservation"
        );
        assert!(
            rx.try_recv().is_err(),
            "no refused request is left in the queue"
        );
        assert_eq!(
            record_lost_completion(&shared, &retired, &queued, Duration::from_millis(250)),
            1,
            "the retirement drained the one accepted request"
        );
        assert!(
            shared.take_failure(job_for(4).key).is_some(),
            "the drained request has a reason for its waiter"
        );
    }

    /// The ordering the probe above cannot see: an accept whose stall check happens *before* it
    /// takes the gate. This storm is already running while the gate is still free, so an accept that
    /// reads the flag before locking reads it while it is clear, then parks behind the retirement,
    /// and - if it does not check again inside the gate - slips into the queue after the retirement
    /// has walked it. That request is orphaned: counted in flight, unreachable by any worker, and
    /// with no reason for its waiter.
    ///
    /// The assertion is the invariant rather than a count of refusals, because requests accepted
    /// before the retirement are legitimately drained by it. Every round is a fresh race, so the
    /// mutation cannot hide in one lucky interleaving. The fixed 2 ms is the drain being held under
    /// the gate, which gives the attempts that were in flight when the flag went up time to park.
    #[test]
    fn a_stale_stall_check_cannot_slip_past_the_retirement_gate() {
        for round in 0..25 {
            let shared = Arc::new(shared_for_tests(0));
            let (tx, rx) = crossbeam_channel::bounded::<Job>(64);
            let stop = Arc::new(AtomicBool::new(false));
            let started = Arc::new(AtomicUsize::new(0));
            let mut storm = Vec::new();
            for row in 0..16u64 {
                let shared = Arc::clone(&shared);
                let tx = tx.clone();
                let stop = Arc::clone(&stop);
                let started = Arc::clone(&started);
                storm.push(std::thread::spawn(move || {
                    started.fetch_add(1, Ordering::Relaxed);
                    while !stop.load(Ordering::Relaxed) {
                        let _ = shared.accept(&tx, job_for(100 + row));
                    }
                }));
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            while started.load(Ordering::Relaxed) < 16 {
                assert!(Instant::now() < deadline, "the storm threads did not start");
                std::thread::yield_now();
            }

            let (retired, queued) = retire_accepted(&shared, || {
                std::thread::sleep(Duration::from_millis(2));
                let queued: Vec<Job> = rx.try_iter().collect();
                (Vec::new(), queued)
            });
            let _ = record_lost_completion(&shared, &retired, &queued, Duration::from_millis(250));

            stop.store(true, Ordering::Relaxed);
            for thread in storm {
                thread.join().expect("a storm thread");
            }
            let leftover = rx.len();
            let in_flight = shared.in_flight.load(Ordering::Relaxed);
            assert!(
                leftover == 0 && in_flight == 0,
                "round {round}: {leftover} request(s) slipped into a queue the retirement walked, \
                 with in_flight {in_flight}"
            );
        }
    }

    /// One accepted job for row `row`, keyed row + 1.
    fn job_for(row: u64) -> Job {
        Job {
            table: 0,
            row,
            key: row + 1,
            issued_step: 0,
        }
    }

    /// A patterned in-memory table: the first eight bytes of row N are N.
    fn memory_table(rows: u64) -> Arc<dyn RowTable> {
        let mut table = MemoryTable::new().with_layer(spec(rows));
        table.fill_pattern(0).unwrap();
        Arc::new(table)
    }

    fn config() -> PrefetchConfig {
        PrefetchConfig {
            queue_depth: 96,
            lead_rows: 32,
            max_batch_rows: 64,
            workers: 2,
            ready_capacity: 4096,
            window_bytes: 512,
        }
    }

    fn first(data: [u8; ENGRAM_ROW_BYTES]) -> u64 {
        u64::from_le_bytes(data[..8].try_into().unwrap())
    }

    /// A row source whose reads fail whenever a requested row is marked bad, the way the
    /// reviewer injected failures: a merged read fails as a group.
    struct FailingTable {
        inner: Arc<dyn RowTable>,
        bad: Vec<u32>,
    }

    impl RowTable for FailingTable {
        fn table_count(&self) -> usize {
            self.inner.table_count()
        }

        fn spec(&self, table: usize) -> Option<&TableSpec> {
            self.inner.spec(table)
        }

        fn read_row(&self, table: usize, row: u64) -> anyhow::Result<[u8; ENGRAM_ROW_BYTES]> {
            if self.bad.contains(&u32::try_from(row).unwrap_or(u32::MAX)) {
                anyhow::bail!("injected failure for row {row}");
            }
            self.inner.read_row(table, row)
        }

        fn read_rows_into(
            &self,
            table: usize,
            rows: &[u32],
            out: &mut [[u8; ENGRAM_ROW_BYTES]],
        ) -> anyhow::Result<()> {
            if let Some(row) = rows.iter().find(|row| self.bad.contains(row)) {
                anyhow::bail!("injected failure for row {row}");
            }
            self.inner.read_rows_into(table, rows, out)
        }

        fn read_ops(&self) -> u64 {
            self.inner.read_ops()
        }
    }

    fn failing_table(rows: u64, bad: Vec<u32>) -> Arc<dyn RowTable> {
        let mut table = MemoryTable::new().with_layer(spec(rows));
        table.fill_pattern(0).unwrap();
        Arc::new(FailingTable {
            inner: Arc::new(table),
            bad,
        })
    }

    /// Review A5: on the Inline backend an accepted row is counted issued whatever the read
    /// then does, so `issued == accepted` and `completed + failed == accepted` hold, and the
    /// failure granularity is visible instead of implied: one merged read failed, so every
    /// row of that one group is counted failed.
    #[test]
    fn a_mixed_inline_run_counts_every_accepted_row_once() {
        let requested: Vec<u32> = (0..20).collect();

        let healthy = PrefetchEngine::with_backend(
            failing_table(64, Vec::new()),
            config(),
            BackendRequest::Inline,
        )
        .expect("an inline engine");
        let accepted = healthy.request_rows(0, &requested);
        assert_eq!(accepted, requested.len());
        let snapshot = healthy.snapshot();
        assert_eq!(snapshot.backend, PrefetchBackend::Inline);
        assert_eq!(
            snapshot.issued, accepted as u64,
            "issued must equal accepted"
        );
        assert_eq!(
            snapshot.completed + snapshot.failed,
            accepted as u64,
            "every accepted row is either completed or failed"
        );
        assert_eq!(snapshot.completed, accepted as u64);
        assert_eq!(snapshot.failed, 0);
        assert_eq!(snapshot.pool_read_failures, 0);

        let mixed = PrefetchEngine::with_backend(
            failing_table(64, vec![3, 7, 11]),
            config(),
            BackendRequest::Inline,
        )
        .expect("an inline engine");
        let accepted = mixed.request_rows(0, &requested);
        assert_eq!(accepted, requested.len());
        let snapshot = mixed.snapshot();
        assert_eq!(
            snapshot.issued, accepted as u64,
            "a failed read must not uncount the rows it accepted"
        );
        assert_eq!(
            snapshot.completed + snapshot.failed,
            accepted as u64,
            "every accepted row is either completed or failed"
        );
        assert_eq!(
            snapshot.completed, 0,
            "one read covers the group, so a bad row fails all of it"
        );
        assert_eq!(
            snapshot.failed, accepted as u64,
            "the merged read failed as a group"
        );
        assert_eq!(
            snapshot.pool_read_failures, 1,
            "exactly one merged read was attempted"
        );
        assert!(
            mixed.take(0, 3).is_none(),
            "a failed row must not be published"
        );
    }

    /// A row source that sleeps before every read, to exercise backpressure.
    struct SlowTable {
        inner: Arc<dyn RowTable>,
        delay: Duration,
    }

    impl RowTable for SlowTable {
        fn table_count(&self) -> usize {
            self.inner.table_count()
        }

        fn spec(&self, table: usize) -> Option<&TableSpec> {
            self.inner.spec(table)
        }

        fn read_row(&self, table: usize, row: u64) -> anyhow::Result<[u8; ENGRAM_ROW_BYTES]> {
            std::thread::sleep(self.delay);
            self.inner.read_row(table, row)
        }

        fn read_rows_into(
            &self,
            table: usize,
            rows: &[u32],
            out: &mut [[u8; ENGRAM_ROW_BYTES]],
        ) -> anyhow::Result<()> {
            std::thread::sleep(self.delay);
            self.inner.read_rows_into(table, rows, out)
        }

        fn read_ops(&self) -> u64 {
            self.inner.read_ops()
        }
    }

    /// Polls `take` until the row arrives or the deadline passes.
    fn wait_for(
        engine: &PrefetchEngine,
        table: usize,
        row: u64,
        timeout: Duration,
    ) -> Option<[u8; ENGRAM_ROW_BYTES]> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(data) = engine.take(table, row) {
                return Some(data);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn configuration_is_validated() {
        assert!(config().validate().is_ok());
        for bad in [
            PrefetchConfig {
                queue_depth: 0,
                ..config()
            },
            PrefetchConfig {
                queue_depth: 5000,
                ..config()
            },
            PrefetchConfig {
                max_batch_rows: 0,
                ..config()
            },
            PrefetchConfig {
                window_bytes: 100,
                ..config()
            },
            PrefetchConfig {
                ready_capacity: 1,
                ..config()
            },
        ] {
            assert!(bad.validate().is_err(), "accepted a bad configuration");
        }
        assert_eq!(
            PrefetchConfig {
                workers: 4,
                ..config()
            }
            .resolved_workers(),
            4
        );
        let automatic = PrefetchConfig {
            workers: 0,
            ..config()
        };
        assert!(automatic.resolved_workers() >= 1);
        assert!(automatic.resolved_workers() <= automatic.queue_depth);
    }

    #[test]
    fn inline_engine_tracks_lead_depth_and_consumption() {
        let engine =
            PrefetchEngine::with_backend(memory_table(64), config(), BackendRequest::Inline)
                .unwrap();
        assert_eq!(engine.backend(), PrefetchBackend::Inline);
        assert_eq!(engine.request_rows(0, &[0, 1, 2, 3]), 4);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.issued, 4);
        assert_eq!(snapshot.completed, 4);
        assert_eq!(snapshot.ready, 4);
        assert_eq!(snapshot.in_flight, 0);
        assert_eq!(snapshot.pool_reads, 1, "one merged read for the batch");
        // A row that is already ready is not fetched twice.
        assert_eq!(engine.request_rows(0, &[0, 1]), 0);
        assert_eq!(first(engine.take(0, 0).unwrap()), 0);
        engine.advance_step();
        engine.advance_step();
        assert_eq!(first(engine.take(0, 1).unwrap()), 1);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.consumed, 2);
        assert_eq!(snapshot.lead_samples, 2);
        // Row 0 was consumed at step 0 and row 1 at step 2, so the mean lead is 1.
        assert!((snapshot.avg_lead_steps - 1.0).abs() < 1e-9);
        assert!(engine.take(0, 9).is_none());
        // A row that was never prefetched falls back to the source.
        assert_eq!(first(engine.fill_or_read(0, 9).unwrap()), 9);
    }

    #[test]
    fn prefetch_ahead_respects_the_lead_depth() {
        let engine =
            PrefetchEngine::with_backend(memory_table(64), config(), BackendRequest::Inline)
                .unwrap();
        let rows: Vec<u32> = (0..32).collect();
        engine.set_lead_rows(4);
        assert_eq!(engine.lead_rows(), 4);
        assert_eq!(engine.prefetch_ahead(0, &rows, 0), 4);
        assert!(!engine.request(0, 1), "row 1 is already prefetched");
        engine.set_lead_rows(8);
        assert_eq!(engine.prefetch_ahead(0, &rows, 5), 8);
        assert_eq!(engine.ready_len(), 12);
        assert_eq!(engine.prefetch_ahead(0, &rows, rows.len() - 1), 0);
        assert_eq!(engine.prefetch_ahead(0, &rows, rows.len()), 0);
    }

    #[test]
    fn portable_pool_completes_every_request() {
        let engine =
            PrefetchEngine::with_backend(memory_table(64), config(), BackendRequest::ThreadPool)
                .unwrap();
        assert_eq!(engine.backend(), PrefetchBackend::ThreadPool);
        assert!(engine.fallback_reason().is_none());
        let rows: Vec<u32> = (0..32).collect();
        assert_eq!(engine.request_rows(0, &rows), 32);
        for row in 0..32u64 {
            let data = wait_for(&engine, 0, row, Duration::from_secs(10))
                .unwrap_or_else(|| panic!("row {row} never arrived"));
            assert_eq!(first(data), row);
        }
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.issued, 32);
        assert_eq!(snapshot.completed, 32);
        assert_eq!(snapshot.consumed, 32);
        assert_eq!(snapshot.backpressure, 0);
        assert_eq!(snapshot.ready, 0);
        assert_eq!(snapshot.in_flight, 0);
        assert!(snapshot.pool_reads >= 1);
        assert_eq!(snapshot.bytes_read, 32 * ENGRAM_ROW_BYTES as u64);
        assert_eq!(engine.backend(), PrefetchBackend::ThreadPool);
    }

    #[test]
    fn a_full_queue_is_reported_as_backpressure() {
        let slow = Arc::new(SlowTable {
            inner: memory_table(64),
            delay: Duration::from_millis(5),
        });
        let config = PrefetchConfig {
            queue_depth: 2,
            workers: 1,
            max_batch_rows: 1,
            ready_capacity: 64,
            ..config()
        };
        let engine =
            PrefetchEngine::with_backend(slow, config, BackendRequest::ThreadPool).unwrap();
        let rows: Vec<u32> = (0..40).collect();
        let accepted = engine.request_rows(0, &rows);
        let snapshot = engine.snapshot();
        assert!(
            snapshot.backpressure > 0,
            "a two-deep queue served by one slow worker must refuse requests"
        );
        assert!(accepted < rows.len());
        assert_eq!(snapshot.issued, accepted as u64);
        // A refused send is not an accepted one: the reservation is given back, so the count can
        // never run ahead of the requests the engine actually handed over (review A7).
        assert_eq!(
            snapshot.in_flight as u64 + snapshot.completed + snapshot.failed,
            snapshot.issued,
            "every accepted request is in flight, completed or failed"
        );
        assert_eq!(snapshot.queue_depth, 2);
    }

    #[test]
    fn exceeding_the_ready_capacity_drops_the_oldest_row_as_wasted() {
        let config = PrefetchConfig {
            queue_depth: 4,
            ready_capacity: 4,
            max_batch_rows: 1,
            workers: 1,
            lead_rows: 1,
            window_bytes: 512,
        };
        let engine =
            PrefetchEngine::with_backend(memory_table(16), config, BackendRequest::Inline).unwrap();
        assert_eq!(engine.request_rows(0, &[0, 1, 2, 3, 4, 5]), 6);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.ready, 4);
        assert_eq!(
            snapshot.ready_order, 4,
            "one bookkeeping entry per ready row"
        );
        assert_eq!(snapshot.wasted, 2);
        assert!(engine.take(0, 0).is_none(), "row 0 was dropped");
        assert_eq!(first(engine.take(0, 5).unwrap()), 5);
    }

    #[test]
    fn discard_and_flush_count_wasted_rows() {
        let engine =
            PrefetchEngine::with_backend(memory_table(16), config(), BackendRequest::Inline)
                .unwrap();
        assert_eq!(engine.request_rows(0, &[0, 1, 2, 3]), 4);
        assert!(engine.discard(0, 0));
        assert!(!engine.discard(0, 0));
        assert_eq!(engine.flush(), 3);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.wasted, 4);
        assert_eq!(snapshot.ready, 0);
        assert_eq!(snapshot.consumed, 0);
    }

    #[test]
    fn snapshot_serialises_for_the_stats_endpoint() {
        let engine =
            PrefetchEngine::with_backend(memory_table(16), config(), BackendRequest::Inline)
                .unwrap();
        engine.request_rows(0, &[0, 1]);
        let _ = engine.take(0, 0);
        let json = serde_json::to_value(engine.snapshot()).unwrap();
        for key in [
            "issued",
            "completed",
            "wasted",
            "failed",
            "backpressure",
            "consumed",
            "bytes_read",
            "avg_lead_steps",
            "lead_samples",
            "queue_depth",
            "lead_rows",
            "max_batch_rows",
            "ready",
            "in_flight",
            "backend",
            "ring_submissions",
            "pool_reads",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["queue_depth"], 96);
        assert_eq!(json["backend"], "Inline");
    }

    #[test]
    fn dropping_the_engine_stops_its_workers() {
        let source = memory_table(16);
        {
            let engine = PrefetchEngine::with_backend(
                Arc::clone(&source),
                config(),
                BackendRequest::ThreadPool,
            )
            .unwrap();
            engine.request_rows(0, &[0, 1, 2, 3]);
        }
        // The drop above joined every worker; a second engine still works afterwards.
        let engine =
            PrefetchEngine::with_backend(source, config(), BackendRequest::ThreadPool).unwrap();
        assert!(engine.request(0, 0));
        assert!(wait_for(&engine, 0, 0, Duration::from_secs(10)).is_some());
    }

    /// With the `io-uring` feature compiled out, `Auto` must fall back to the portable pool
    /// and say why. Without a reason the caller cannot tell a deliberate build choice from a
    /// device that refused direct I/O.
    #[cfg(not(feature = "io-uring"))]
    #[test]
    fn auto_without_the_ring_feature_explains_the_fallback() {
        let engine = PrefetchEngine::new(memory_table(16), config()).unwrap();
        assert_eq!(engine.backend(), PrefetchBackend::ThreadPool);
        let reason = engine.fallback_reason().unwrap_or_default();
        assert!(
            reason.contains("io-uring"),
            "the portable backend must be explained: {reason:?}"
        );
        assert!(
            engine.fallback_reason().is_some(),
            "a fallback without a reason is indistinguishable from a silent downgrade"
        );
    }

    fn sample_entry(byte: u8) -> Entry {
        Entry {
            data: [byte; ENGRAM_ROW_BYTES],
            issued_step: 0,
        }
    }

    #[test]
    fn consuming_rows_does_not_grow_the_queue_bookkeeping() {
        // A decode loop requests and consumes fresh rows forever. The ready map may hold at
        // most ready_capacity rows and its bookkeeping must stay the same size: an
        // implementation that left consumed keys in the arrival order grows by one entry per
        // row, which is 8 bytes per row and 62 MB over eight million rows.
        let mut map = ReadyMap::new(8);
        for cycle in 0..300_000u64 {
            assert_eq!(map.insert(cycle, sample_entry(cycle as u8)), 0);
            assert!(map.take(cycle).is_some());
            assert!(
                map.order_len() <= 8,
                "cycle {cycle}: bookkeeping grew to {}",
                map.order_len()
            );
            assert_eq!(map.len(), 0);
            assert!(map.invariants_hold());
        }
        assert_eq!(map.order_len(), 0, "a drained map keeps no bookkeeping");

        // Consuming rows out of arrival order is bounded as well.
        for cycle in 1_000_000..1_050_000u64 {
            assert_eq!(map.insert(cycle, sample_entry(1)), 0);
            assert_eq!(map.insert(cycle + 1, sample_entry(2)), 0);
            assert!(map.take(cycle).is_some());
            assert!(map.take(cycle + 1).is_some());
            assert!(
                map.order_len() <= 8,
                "cycle {cycle}: bookkeeping grew to {}",
                map.order_len()
            );
        }
        assert!(map.invariants_hold());
        assert_eq!(map.order_len(), 0);
    }
}

/// Tests for the `io_uring` backend; they exist only when the feature is compiled in.
#[cfg(all(test, feature = "io-uring"))]
mod uring_tests {
    use super::*;
    use crate::packer::align_up;
    use crate::store::{DirectTable, MemoryTable, TableSpec, TailFill};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    /// Writes a dense packed table exactly like the packer does and returns its spec.
    fn packed_file(dir: &std::path::Path, rows: u64) -> TableSpec {
        let path = dir.join("engram-l1-r0of1.bin");
        let mut bytes = vec![0u8; (rows * ENGRAM_ROW_BYTES as u64) as usize];
        for row in 0..rows {
            let start = row as usize * ENGRAM_ROW_BYTES;
            bytes[start..start + 8].copy_from_slice(&row.to_le_bytes());
            for (offset, byte) in bytes[start + 8..start + ENGRAM_ROW_BYTES]
                .iter_mut()
                .enumerate()
            {
                *byte = (row as u8).wrapping_add(offset as u8);
            }
        }
        bytes.resize(align_up(bytes.len() as u64) as usize, 0);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: PathBuf::from(&path),
            row_start: 0,
            physical_rows: rows,
            logical_rows: rows,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        }
    }

    fn ring_config() -> PrefetchConfig {
        PrefetchConfig {
            queue_depth: 8,
            max_batch_rows: 8,
            workers: 1,
            ready_capacity: 64,
            lead_rows: 4,
            window_bytes: 512,
        }
    }

    /// Proves that a file-backed table is served by the ring when the ring can serve it, that
    /// the row comes back byte for byte, and that choosing the pool instead is explained.
    ///
    /// The file is 37 rows: 37 * 264 = 9768 bytes, padded to 512 means 10240 = 2 * 4096 + 2048,
    /// so on a 4096-byte-logical-block device the last six row windows round past the padded end.
    /// That is the shape every real rank file has: layer 0 ends at 2048 mod 4096 and layer 1 at
    /// 3584, so a ring that refused those windows served 31 of 37 rows and fell back to
    /// synchronous reads for the tail. The device is forced to 4096 because a test filesystem
    /// reports 512, and the alignment is what decides whether the tail is servable (review A3,
    /// round 4). A successful open(O_DIRECT) on tmpfs is not evidence of unbuffered I/O, which is
    /// why this asserts which backend served the row, not how the kernel served it.
    #[test]
    fn the_ring_or_a_documented_fallback_serves_rows() {
        let dir = tempfile::tempdir().unwrap();
        let spec = packed_file(dir.path(), 37);
        uring::force_alignment(&spec.path, crate::store::MAX_ALIGN_BYTES);
        let source: Arc<dyn RowTable> = Arc::new(DirectTable::open(vec![spec]).unwrap());
        let engine =
            PrefetchEngine::with_backend(Arc::clone(&source), ring_config(), BackendRequest::Auto)
                .unwrap();
        if engine.backend() == PrefetchBackend::IoUring {
            assert!(engine.fallback_reason().is_none());
        } else {
            assert_eq!(engine.backend(), PrefetchBackend::ThreadPool);
            assert!(
                engine.fallback_reason().is_some(),
                "choosing the portable backend must be explained"
            );
        }
        // The last row sits in the last window of the file, which overruns the padded end on a
        // 4096-byte-logical-block device.
        assert!(engine.request(0, 36));
        let deadline = Instant::now() + Duration::from_secs(10);
        let data = loop {
            if let Some(data) = engine.take(0, 36) {
                break data;
            }
            assert!(Instant::now() < deadline, "row 36 never arrived");
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), 36);
        assert_eq!(data[8], 36u64 as u8);
        if engine.backend() == PrefetchBackend::IoUring {
            assert_eq!(
                engine.snapshot().failed,
                0,
                "no row is refused, including the ones whose windows overrun the padded end"
            );
            assert!(engine.snapshot().ring_submissions >= 1);
        } else {
            assert!(engine.snapshot().pool_reads >= 1);
        }
    }

    /// The tail a 512-padded rank file leaves: the ring serves the final rows whose windows round
    /// past the padded end, every row arrives byte for byte, and nothing is published from bytes
    /// the kernel did not write. Before the fix the ring refused those windows, so a rank file on
    /// the device this project targets had a tail it never served (review A3, round 4).
    #[test]
    fn the_ring_serves_a_tail_window_that_overruns_the_padded_end() {
        let dir = tempfile::tempdir().unwrap();
        let spec = packed_file(dir.path(), 37);
        let padded = std::fs::metadata(&spec.path)
            .expect("the packed file")
            .len();
        assert_eq!(padded % 512, 0, "the packer pads to 512");
        assert_ne!(padded % 4096, 0, "and not to 4096: that is the tail");
        uring::force_alignment(&spec.path, crate::store::MAX_ALIGN_BYTES);
        let source: Arc<dyn RowTable> = Arc::new(DirectTable::open(vec![spec]).unwrap());
        let engine = PrefetchEngine::with_backend(
            Arc::clone(&source),
            wide_ring_config(),
            BackendRequest::IoUring,
        )
        .expect("the ring is required by this test");
        let rows: Vec<u32> = (0..37u32).collect();
        assert_eq!(engine.request_rows(0, &rows), 37);
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let snapshot = engine.snapshot();
            if snapshot.completed + snapshot.failed >= 37 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the ring never finished: {snapshot:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.completed, 37, "every row is served");
        assert_eq!(
            snapshot.failed, 0,
            "a window that overruns the padded end is served, not refused"
        );
        assert_eq!(snapshot.in_flight, 0);
        for row in 0..37u64 {
            let data = engine
                .take(0, row)
                .unwrap_or_else(|| panic!("row {row} should have arrived"));
            assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), row);
            assert_eq!(data[8], row as u8);
        }
    }

    /// A memory-backed source has no file to open, so `Auto` must fall back and say so.
    #[test]
    fn a_memory_source_falls_back_with_a_reason() {
        let spec = TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: PathBuf::from("/tmp/engram-memory-double.bin"),
            row_start: 0,
            physical_rows: 8,
            logical_rows: 8,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        };
        let mut table = MemoryTable::new().with_layer(spec);
        table.fill_pattern(0).unwrap();
        let engine = PrefetchEngine::new(Arc::new(table), ring_config()).unwrap();
        assert_eq!(engine.backend(), PrefetchBackend::ThreadPool);
        let reason = engine.fallback_reason().unwrap_or_default();
        assert!(reason.contains("io_uring"), "unexpected reason: {reason}");
    }

    /// Asking for the ring explicitly must fail loudly instead of falling back silently.
    #[test]
    fn requiring_the_ring_on_a_memory_source_fails() {
        let spec = TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: PathBuf::from("/tmp/engram-memory-double.bin"),
            row_start: 0,
            physical_rows: 8,
            logical_rows: 8,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        };
        let mut table = MemoryTable::new().with_layer(spec);
        table.fill_pattern(0).unwrap();
        let engine =
            PrefetchEngine::with_backend(Arc::new(table), ring_config(), BackendRequest::IoUring);
        assert!(engine.is_err(), "the ring cannot serve a memory source");
    }

    /// A queue wide enough to hold a whole test file at once, so a test can request every row
    /// without depending on how fast the worker drains the channel.
    fn wide_ring_config() -> PrefetchConfig {
        PrefetchConfig {
            queue_depth: 512,
            max_batch_rows: 512,
            workers: 1,
            ready_capacity: 512,
            lead_rows: 8,
            window_bytes: 512,
        }
    }

    /// One owner per counter: every accepted request is counted once by issued, retired once,
    /// and the ready map holds one bookkeeping entry per ready row.
    #[test]
    fn the_ring_counts_each_accepted_request_once() {
        let dir = tempfile::tempdir().unwrap();
        let spec = packed_file(dir.path(), 512);
        let source: Arc<dyn RowTable> = Arc::new(DirectTable::open(vec![spec]).unwrap());
        let engine = PrefetchEngine::with_backend(
            Arc::clone(&source),
            wide_ring_config(),
            BackendRequest::Auto,
        )
        .unwrap();
        if engine.backend() != PrefetchBackend::IoUring {
            eprintln!(
                "skipping: the ring is unavailable here ({})",
                engine.fallback_reason().unwrap_or_default()
            );
            return;
        }
        let rows: Vec<u32> = (0..8).collect();
        let accepted = engine.request_rows(0, &rows) as u64;
        assert_eq!(accepted, 8);
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let snapshot = engine.snapshot();
            if snapshot.completed + snapshot.failed >= accepted {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the ring never finished: {snapshot:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let snapshot = engine.snapshot();
        assert_eq!(
            snapshot.issued, accepted,
            "issued must count one accepted request per row, not one per ring submission"
        );
        assert_eq!(
            snapshot.ring_submissions, accepted,
            "each row is submitted to the ring exactly once"
        );
        assert_eq!(snapshot.completed, accepted);
        assert_eq!(snapshot.failed, 0);
        assert_eq!(snapshot.in_flight, 0, "every accepted request is retired");
        for row in 0..8u64 {
            let data = engine.take(0, row).expect("row");
            assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), row);
        }
        let snapshot = engine.snapshot();
        assert_eq!(
            snapshot.ready_order, snapshot.ready,
            "the queue bookkeeping holds one entry per ready row"
        );
    }

    /// A completion that does not cover the row must never be published: the slot buffer still
    /// holds whatever the previous read put there, so publishing it would hand the consumer
    /// another row's bytes days after the file changed. Truncating the file underneath the ring
    /// is the reviewer reproducer for that. The guard is the row's own span, not the request
    /// length, so a short completion that covers the whole row is served - which is exactly the
    /// tail case above (review A3, round 4).
    #[test]
    fn a_completion_that_does_not_cover_the_row_is_never_published() {
        let dir = tempfile::tempdir().unwrap();
        // 512 * 264 bytes is exactly 33 alignment units, so every window of the intact file
        // is servable and the only reason a completion can be short is the truncation below.
        let spec = packed_file(dir.path(), 512);
        let path = spec.path.clone();
        let source: Arc<dyn RowTable> = Arc::new(DirectTable::open(vec![spec]).unwrap());
        let engine = PrefetchEngine::with_backend(
            Arc::clone(&source),
            wide_ring_config(),
            BackendRequest::Auto,
        )
        .unwrap();
        if engine.backend() != PrefetchBackend::IoUring {
            eprintln!(
                "skipping: the ring is unavailable here ({})",
                engine.fallback_reason().unwrap_or_default()
            );
            return;
        }
        // Shrink the file under the ring. Both the store and the ring have already cached its
        // length, so a window that reaches past byte 4096 now completes short while the slot
        // buffer still holds an earlier row.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(4096).unwrap();
        drop(file);

        let accepted = engine.request_rows(0, &(0..512).collect::<Vec<u32>>()) as u64;
        assert_eq!(accepted, 512, "the wide queue accepts the whole file");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let snapshot = engine.snapshot();
            if snapshot.completed + snapshot.failed >= accepted {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the ring never finished: {snapshot:?}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.completed + snapshot.failed, accepted);
        assert_eq!(snapshot.in_flight, 0);
        // Rows 0..15 occupy the window [0, 4096); row 15 already needs the next unit, so 15
        // rows are readable in a 4096-byte file and the rest cannot be read at all.
        assert_eq!(
            snapshot.ready, 15,
            "only the rows inside the file can arrive"
        );
        assert_eq!(snapshot.completed, 15);
        assert_eq!(snapshot.failed, 497);
        assert_eq!(snapshot.ready_order, snapshot.ready);
        // The rows that were not truncated still arrive byte for byte.
        for row in 0..15u64 {
            let data = engine
                .take(0, row)
                .unwrap_or_else(|| panic!("row {row} should have arrived"));
            assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), row);
        }
        // The rest are absent, never another row bytes.
        for row in 15..512u64 {
            assert!(
                engine.take(0, row).is_none(),
                "row {row} must not be published from a reused slot buffer"
            );
        }
        // A waiter asking for one of them is woken with an error instead of a row.
        let error = engine
            .fill_or_read(0, 511)
            .expect_err("row 511 is past the end of the truncated file");
        assert!(!error.to_string().is_empty(), "{error:#}");
    }
}
