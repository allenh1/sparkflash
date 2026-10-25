//! Speculative decoding with the on-device DSpark drafter (SPARKFLASH_DSPARK=1).
//!
//! The prompt runs as multi-row steps of up to `R_MAX` positions, each seeding the drafter's
//! rings with its rows' taps. Every generation step then drafts k tokens from the anchor (the last
//! emitted token, not yet run through the backbone), verifies [anchor, d1..dk] as one rows step,
//! accepts the drafts while each equals the verify argmax before it, and emits the accepted drafts
//! plus the verify argmax after them. The rows step is bit-identical to the single-row step, so the
//! emitted sequence is the single-row greedy sequence whatever the drafter proposes.
//!
//! A rejected row leaves position-addressed state behind; [`DeviceLayer::rewind_rows`] moves the
//! counters back and the next step overwrites it. The Engram entries of the verify rows are written
//! from the step's own token history before it runs.

use super::batch::{Batch, Emission, Pools};
use super::draft::DraftState;
use super::rows::RowsState;
use super::sample::Sampler;
use super::*;
use crate::device_link::{DeviceFinish, DeviceLink, DeviceOutcome};
use sparkflash_kernels::cuda_driver::HostMapped;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// The transport sequence range the batched embedding gathers use.
const SPEC_EMBED_SEQ: u64 = u64::MAX - 0xE_0000;

/// What the spec loop needs from the forward loop's scope.
pub(super) struct SpecInputs<'s> {
    pub(super) config: &'s sparkflash_engine::config::ModelConfig,
    pub(super) tail: &'s crate::tp4_tail::TailStage,
    pub(super) transport: &'s dyn sparkflash_comm::transport::Transport,
    pub(super) engram: Option<&'s EngramReader>,
    pub(super) staged: &'s [Option<SeamStagedWeights>],
    pub(super) embed: Deviceptr,
    /// The harness's prompt (serving passes each request's own).
    pub(super) prompt: &'s [u32],
    pub(super) rank: usize,
    pub(super) world: usize,
    /// The bulk prefill's rows state (super::bulk), when one is open.
    pub(super) bulk: Option<&'s RowsState>,
}

/// The shortest uncached prompt the bulk prefill takes (SPARKFLASH_PREFILL_BULK_MIN, default 256):
/// shorter ones run the exact rows step, whose output is bit-identical to M = 1.
pub(super) fn bulk_min() -> usize {
    std::env::var("SPARKFLASH_PREFILL_BULK_MIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256)
}

/// Gathers the embedding rows of `tokens` (f32, `[n][dim]`) with one all-gather: each owner rank
/// contributes its rows, zeros elsewhere.
pub(super) fn gather_rows(
    cuda: &Cuda,
    inputs: &SpecInputs<'_>,
    counter: &Cell<u64>,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
    let n = tokens.len();
    let row_bytes = HIDDEN * 2;
    let mut local = vec![0u16; n * HIDDEN];
    let mut bytes = vec![0u8; row_bytes];
    for (i, &token) in tokens.iter().enumerate() {
        let (owner, row) = embedding_location(token);
        if owner == inputs.rank {
            cuda.copy_dtoh(
                &mut bytes,
                &Deviceptr(inputs.embed.0 + (row * row_bytes) as u64),
            )?;
            for (slot, word) in local[i * HIDDEN..(i + 1) * HIDDEN]
                .iter_mut()
                .zip(bytes.as_chunks::<2>().0)
            {
                *slot = u16::from_le_bytes(*word);
            }
        }
    }
    let mut gathered = vec![0u16; inputs.world * n * HIDDEN];
    let call = counter.get();
    counter.set(call.wrapping_add(1));
    inputs
        .transport
        .all_gather_bf16(
            &local,
            &mut gathered,
            SPEC_EMBED_SEQ.wrapping_sub(call % 0x1_0000),
        )
        .map_err(|error| format!("spec embedding all-gather: {error}"))?;
    let mut out = Vec::with_capacity(n * HIDDEN);
    for (i, &token) in tokens.iter().enumerate() {
        let (owner, _) = embedding_location(token);
        let at = (owner * n + i) * HIDDEN;
        out.extend(
            gathered[at..at + HIDDEN]
                .iter()
                .map(|&word| f32::from_bits(u32::from(word) << 16)),
        );
    }
    Ok(out)
}

/// The paged Engram tables, opened for concurrent reads: per Engram slot, the codes and scales
/// files and the byte each tensor starts at.
pub(super) struct EngramReader {
    slots: Vec<(Arc<fs::File>, u64, Arc<fs::File>, u64)>,
    /// Entries already read, by (file, byte): the tables are weights, so an n-gram seen again (or a
    /// redrafted row) costs a copy instead of a device read.
    cache: Mutex<EntryCache>,
    cached: bool,
}

/// Reader threads per gather: the lookups are small random reads, latency-bound one at a time, and
/// more threads than this only contend with the launch thread.
pub(super) const ENGRAM_READ_THREADS: usize = 32;
/// Reader threads for a read that overlaps the draft, which has time to spare and host work of its
/// own.
pub(super) const ENGRAM_AHEAD_THREADS: usize = 8;
/// Reader threads for a bulk prefill chunk's Engram reads (SPARKFLASH_PREFILL_READ_THREADS): a
/// chunk issues 96 small random reads a row, far more than a verify step's queue.
pub(super) fn prefill_read_threads() -> usize {
    std::env::var("SPARKFLASH_PREFILL_READ_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128usize)
        .clamp(1, 256)
}
/// The most entries the reader keeps (about 130 B each).
const ENGRAM_CACHE_ENTRIES: usize = 1 << 20;
/// Submission queue entries of the io_uring reader: a verify step's gather is under 300 reads.
const ENGRAM_URING_ENTRIES: u32 = 512;

impl EngramReader {
    pub(super) fn open(
        config: &sparkflash_engine::config::ModelConfig,
        source: &dyn sparkflash_engine::weights::WeightSource,
    ) -> Result<Self, String> {
        use std::os::fd::AsRawFd as _;
        let region = |name: String| -> Result<(Arc<fs::File>, u64), String> {
            let (file, start, _len) = source
                .file_region(&name)
                .ok_or_else(|| format!("engram reader: {name} is not file-backed"))?
                .map_err(|error| format!("engram reader: {name}: {error}"))?;
            // Readahead only multiplies the bytes a random 256-byte lookup moves.
            // SAFETY: a valid open descriptor; the advice changes no memory.
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
            Ok((file, start))
        };
        let mut slots = Vec::new();
        for layer in &config.engram_layer_ids {
            let (codes, codes_at) = region(format!("layers.{layer}.engram.embed.weight"))?;
            let (scales, scales_at) = region(format!("layers.{layer}.engram.embed.scale"))?;
            slots.push((codes, codes_at, scales, scales_at));
        }
        Ok(Self {
            slots,
            cache: Mutex::new(std::collections::HashMap::new()),
            cached: std::env::var("SPARKFLASH_ENGRAM_CACHE").as_deref() != Ok("0"),
        })
    }
}

/// Keeps the Engram drive out of its idle power states while sequences run. A decode step's reads
/// come in bursts ~45 ms apart, and the Sparks' drives answer a random read ~0.8 ms late after a
/// few ms idle against ~0.3 ms back to back: a 120-read burst every 40 ms lands in 1.8 ms (p50)
/// idle between bursts and in 0.9 ms with this thread reading. While armed it reads one random 4 KB
/// page of the first Engram table's file with O_DIRECT (so the page cache is untouched), back to
/// back; disarmed it waits. SPARKFLASH_ENGRAM_KEEPALIVE=0 starts no thread. Dropping it stops the
/// thread (the scope joins it).
pub(super) struct Keepalive(Arc<KeepState>);

/// The keepalive thread's switch: armed, and the stop flag the disarmed thread waits on.
struct KeepState {
    armed: std::sync::atomic::AtomicBool,
    stop: Mutex<bool>,
    wake: Condvar,
}

impl Keepalive {
    /// Starts the reader on `scope` over `reader`'s first table file; None when disabled or the
    /// file cannot be reopened for direct reads.
    pub(super) fn start<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        reader: &EngramReader,
    ) -> Option<Self> {
        use std::os::fd::AsRawFd as _;
        use std::os::unix::fs::{FileExt as _, OpenOptionsExt as _};
        if std::env::var("SPARKFLASH_ENGRAM_KEEPALIVE").as_deref() == Ok("0") {
            return None;
        }
        let (codes, ..) = reader.slots.first()?;
        let opened = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", codes.as_raw_fd()))
            .and_then(|file| Ok((file.metadata()?.len(), file)));
        let (len, file) = match opened {
            Ok((len, file)) if len >= 8192 => (len, file),
            Ok(_) => return None,
            Err(error) => {
                eprintln!("engram keepalive: {error}; none");
                return None;
            }
        };
        let keep = Arc::new(KeepState {
            armed: std::sync::atomic::AtomicBool::new(false),
            stop: Mutex::new(false),
            wake: Condvar::new(),
        });
        let worker = keep.clone();
        scope.spawn(move || {
            let mut page = vec![0u8; 8192];
            let at = page.as_ptr().align_offset(4096);
            let page = &mut page[at..at + 4096];
            let mut seed = 0x9E37_79B9_7F4A_7C15u64;
            loop {
                if !worker.armed.load(Ordering::Acquire) {
                    let mut stop = worker.stop.lock().unwrap_or_else(|p| p.into_inner());
                    while !*stop && !worker.armed.load(Ordering::Acquire) {
                        stop = worker.wake.wait(stop).unwrap_or_else(|p| p.into_inner());
                    }
                    if *stop {
                        return;
                    }
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if let Err(error) = file.read_exact_at(page, (seed % (len - 4096)) & !4095) {
                    eprintln!("engram keepalive: {error}; stopping");
                    return;
                }
            }
        });
        Some(Self(keep))
    }

    /// Arms (reads on) or disarms the reader.
    pub(super) fn set(&self, on: bool) {
        if self.0.armed.swap(on, Ordering::AcqRel) != on && on {
            let _guard = self.0.stop.lock().unwrap_or_else(|p| p.into_inner());
            self.0.wake.notify_all();
        }
    }
}

impl Drop for Keepalive {
    fn drop(&mut self) {
        self.0.armed.store(false, Ordering::Release);
        *self.0.stop.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.0.wake.notify_all();
    }
}

/// The reads that fill the Engram entries of rows p0..p0+n: each slot's entries are contiguous in
/// the arena, so a slot is one span of the staging buffer (slot s at s * n * stride) and one upload.
pub(super) struct EngramGather<'r> {
    reader: &'r EngramReader,
    n: usize,
    stride: usize,
    /// The staging rows per slot, and the row of it p0 lands in.
    cap: usize,
    row0: usize,
    /// The arena row staging row 0 uploads to: p0 - row0 (by position), or the step row of the
    /// sequence's first row when the arena is indexed by step row (lanes).
    pub(super) at: usize,
    /// Each row's skip flag per slot, `[n][slots]` (the dead-token rule): by position they are set
    /// at plan time, by step row when the gather is applied ([`DeviceLayer::engram_apply`]).
    skips: Vec<bool>,
    /// (slot, file, byte, offset in the staging buffer, length), slot 0's reads first.
    jobs: Vec<ReadJob<'r>>,
    /// Reads per slot.
    counts: Vec<usize>,
    /// Whether the reads go through the reader's entry cache: a bulk prefill chunk's do not (its
    /// thousands of reads would serialise on the cache lock and fill the cache with prompt rows).
    pub(super) cached: bool,
}

/// One Engram read: (slot, file, byte offset, staging offset, length).
type ReadJob<'f> = (usize, &'f fs::File, u64, usize, usize);

/// Entries already read, by (file, byte).
type EntryCache = std::collections::HashMap<(usize, u64), Box<[u8]>>;

/// A gather with reads in flight on the ring, and its outstanding count.
type LiveGather<'r> = (Arc<GatherTask<'r>>, usize);

/// A staging buffer the reader threads fill at disjoint offsets.
pub(super) struct Staging(*mut u8, usize);

impl Staging {
    /// The whole of a pinned buffer.
    pub(super) fn of(pinned: &HostMapped) -> Self {
        Self(pinned.host_ptr().cast(), pinned.len())
    }
}
// SAFETY: every job writes its own range (checked against the length), and the buffer is read
// only after the slot's reads have all landed.
unsafe impl Sync for Staging {}
// SAFETY: as above; the buffer outlives every thread that holds it (scoped threads).
unsafe impl Send for Staging {}

/// One step's read progress: the next job to take and, per slot, the reads still outstanding.
struct ReadProgress {
    next: AtomicUsize,
    remaining: Vec<AtomicUsize>,
    failed: Mutex<Option<String>>,
    landed: Condvar,
}

impl EngramGather<'_> {
    pub(super) fn slots(&self) -> usize {
        self.counts.len()
    }

    /// One gather for several sequences' rows in one step, each part staged at the rows after the
    /// one before (the arena indexed by step row): slot 0's reads still go first. None for no
    /// parts.
    ///
    /// # Errors
    /// Parts that do not tile consecutive staging rows of one layout.
    pub(super) fn merge(parts: Vec<Self>) -> Result<Option<Self>, String> {
        let mut parts = parts.into_iter();
        let Some(mut merged) = parts.next() else {
            return Ok(None);
        };
        for part in parts {
            if merged.at != 0
                || part.at != 0
                || part.row0 != merged.row0 + merged.n
                || part.cap != merged.cap
                || part.stride != merged.stride
                || part.slots() != merged.slots()
                || !std::ptr::eq(part.reader, merged.reader)
            {
                return Err("engram gather: the parts do not tile one staging buffer".to_owned());
            }
            merged.n += part.n;
            merged.skips.extend(part.skips);
            for (count, more) in merged.counts.iter_mut().zip(&part.counts) {
                *count += more;
            }
            merged.jobs.extend(part.jobs);
            merged.cached &= part.cached;
        }
        // Each slot's reads together, slot 0's first (the sort is stable: rows stay in order).
        merged.jobs.sort_by_key(|job| job.0);
        Ok(Some(merged))
    }

    /// Slot `slot`'s staged rows 0..row0 + n: (byte offset in the staging buffer, bytes).
    fn slot_span(&self, slot: usize) -> (usize, usize) {
        (
            slot * self.cap * self.stride,
            (self.row0 + self.n) * self.stride,
        )
    }

    fn progress(&self) -> ReadProgress {
        ReadProgress {
            next: AtomicUsize::new(0),
            remaining: self
                .counts
                .iter()
                .map(|&count| AtomicUsize::new(count))
                .collect(),
            failed: Mutex::new(None),
            landed: Condvar::new(),
        }
    }

    /// Job `i`'s staging bytes, or an error if the job lands outside the buffer.
    ///
    /// # Safety
    /// No other live slice of job `i`'s range may exist: each index is held by one reader at a
    /// time (the pool's `next` counter hands it out once; the ring takes it at submit and again only
    /// after its completion).
    // The &mut comes from the shared buffer by design: the ranges are disjoint per job (# Safety).
    #[allow(clippy::mut_from_ref)]
    unsafe fn job_slice<'b>(&self, i: usize, staging: &'b Staging) -> Result<&'b mut [u8], String> {
        let (_, _, _, dst, len) = self.jobs[i];
        if dst + len > staging.1 {
            return Err("engram reader: a read lands outside the staging buffer".to_owned());
        }
        // SAFETY: inside the buffer (checked) and disjoint from every other job's range.
        Ok(unsafe { std::slice::from_raw_parts_mut(staging.0.add(dst), len) })
    }

    /// Copies job `i`'s entry from the reader's cache into `slice`; false on a miss.
    fn cached_copy(&self, i: usize, slice: &mut [u8]) -> bool {
        let (_, file, at, _, _) = self.jobs[i];
        let key = (std::ptr::from_ref(file) as usize, at);
        (self.reader.cached && self.cached)
            .then(|| self.reader.cache.lock().ok())
            .flatten()
            .and_then(|cache| cache.get(&key).map(|bytes| slice.copy_from_slice(bytes)))
            .is_some()
    }

    /// Keeps job `i`'s freshly read entry for a later lookup of the same table bytes.
    fn remember(&self, i: usize, slice: &[u8]) {
        let (_, file, at, _, _) = self.jobs[i];
        if self.reader.cached
            && self.cached
            && let Ok(mut cache) = self.reader.cache.lock()
            && cache.len() < ENGRAM_CACHE_ENTRIES
        {
            cache.insert((std::ptr::from_ref(file) as usize, at), slice.into());
        }
    }

    /// Takes reads until none is left.
    fn work(&self, progress: &ReadProgress, staging: &Staging) {
        use std::os::unix::fs::FileExt as _;
        loop {
            let i = progress.next.fetch_add(1, Ordering::Relaxed);
            let Some(&(slot, file, at, _, _)) = self.jobs.get(i) else {
                break;
            };
            // SAFETY: `next` gave index i to this thread alone.
            let result = unsafe { self.job_slice(i, staging) }.and_then(|slice| {
                if self.cached_copy(i, slice) {
                    return Ok(());
                }
                file.read_exact_at(slice, at)
                    .map(|()| self.remember(i, slice))
                    .map_err(|error| format!("engram reader: read at {at}: {error}"))
            });
            if let Err(error) = result {
                progress
                    .failed
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .get_or_insert(error);
            }
            if progress.remaining[slot].fetch_sub(1, Ordering::AcqRel) == 1 {
                let _guard = progress.failed.lock().unwrap_or_else(|p| p.into_inner());
                progress.landed.notify_all();
            }
        }
    }

    /// Runs every read on `threads` threads and returns once all have landed.
    fn read_all(&self, staging: &Staging, threads: usize) -> Result<(), String> {
        let progress = self.progress();
        std::thread::scope(|scope| {
            for _ in 0..threads.min(self.jobs.len()) {
                scope.spawn(|| self.work(&progress, staging));
            }
        });
        (0..self.slots()).try_for_each(|slot| Self::wait(&progress, slot))
    }

    /// Blocks until every read of `slot` has landed.
    fn wait(progress: &ReadProgress, slot: usize) -> Result<(), String> {
        let mut failed = progress.failed.lock().unwrap_or_else(|p| p.into_inner());
        while progress.remaining[slot].load(Ordering::Acquire) != 0 {
            failed = progress
                .landed
                .wait(failed)
                .unwrap_or_else(|p| p.into_inner());
        }
        failed.as_ref().map_or(Ok(()), |error| Err(error.clone()))
    }
}

/// One gather handed to the reader pool: its reads, their progress and where they land.
pub(super) struct GatherTask<'r> {
    gather: EngramGather<'r>,
    progress: ReadProgress,
    staging: Staging,
}

/// Reader threads kept for a whole generation, so a step's reads start no threads.
pub(super) struct ReadPool<'r> {
    workers: Vec<std::sync::mpsc::Sender<Arc<GatherTask<'r>>>>,
}

impl<'r> ReadPool<'r> {
    fn start<'scope>(scope: &'scope std::thread::Scope<'scope, '_>, threads: usize) -> Self
    where
        'r: 'scope,
    {
        let workers = (0..threads)
            .map(|_| {
                let (sender, tasks) = std::sync::mpsc::channel::<Arc<GatherTask<'r>>>();
                scope.spawn(move || {
                    for task in tasks {
                        task.gather.work(&task.progress, &task.staging);
                    }
                });
                sender
            })
            .collect();
        Self { workers }
    }

    /// Starts `gather`'s reads into `staging` on up to `threads` of the workers.
    fn submit(
        &self,
        gather: EngramGather<'r>,
        staging: Staging,
        threads: usize,
    ) -> Result<Arc<GatherTask<'r>>, String> {
        if gather.slots() * gather.cap * gather.stride > staging.1 {
            return Err("engram gather: the staging buffer is too small".to_owned());
        }
        let task = Arc::new(GatherTask {
            progress: gather.progress(),
            gather,
            staging,
        });
        for worker in self
            .workers
            .iter()
            .take(threads.min(task.gather.jobs.len()))
        {
            worker
                .send(task.clone())
                .map_err(|_| "engram gather: a reader thread is gone".to_owned())?;
        }
        Ok(task)
    }
}

/// The step's Engram reads on one io_uring, driven from the launch thread: a gather's misses are
/// all in flight at once (independent random 256-byte reads, so the drive's queue depth bounds
/// them, not a thread count), and a wait reaps completions until its slot has landed.
pub(super) struct UringReads<'r> {
    ring: RefCell<io_uring::IoUring>,
    /// The gathers with reads in flight and their outstanding count, by user_data's high half.
    live: RefCell<Vec<Option<LiveGather<'r>>>>,
}

impl<'r> UringReads<'r> {
    fn open() -> Result<Self, String> {
        let ring = io_uring::IoUring::new(ENGRAM_URING_ENTRIES)
            .map_err(|error| format!("engram reader: io_uring: {error}"))?;
        Ok(Self {
            ring: RefCell::new(ring),
            live: RefCell::new(Vec::new()),
        })
    }

    fn submit(
        &self,
        gather: EngramGather<'r>,
        staging: Staging,
    ) -> Result<Arc<GatherTask<'r>>, String> {
        use std::os::fd::AsRawFd as _;
        if gather.slots() * gather.cap * gather.stride > staging.1 {
            return Err("engram gather: the staging buffer is too small".to_owned());
        }
        let task = Arc::new(GatherTask {
            progress: gather.progress(),
            gather,
            staging,
        });
        let mut live = self.live.borrow_mut();
        let t = live.iter().position(Option::is_none).unwrap_or_else(|| {
            live.push(None);
            live.len() - 1
        });
        let mut ring = self.ring.borrow_mut();
        let mut in_flight = 0usize;
        for (i, &(slot, file, at, _, len)) in task.gather.jobs.iter().enumerate() {
            // SAFETY: the task is new, so nothing else holds index i yet.
            let slice = unsafe { task.gather.job_slice(i, &task.staging) }?;
            if task.gather.cached_copy(i, slice) {
                task.progress.remaining[slot].fetch_sub(1, Ordering::Relaxed);
                continue;
            }
            let entry = io_uring::opcode::Read::new(
                io_uring::types::Fd(file.as_raw_fd()),
                slice.as_mut_ptr(),
                len as u32,
            )
            .offset(at)
            .build()
            .user_data(((t as u64) << 32) | i as u64);
            // SAFETY: the buffer is the job's own staging range, which outlives the read (the task
            // stays in `live` until every read of it has completed).
            while unsafe { ring.submission().push(&entry) }.is_err() {
                ring.submit()
                    .map_err(|error| format!("engram reader: submit: {error}"))?;
            }
            in_flight += 1;
        }
        ring.submit()
            .map_err(|error| format!("engram reader: submit: {error}"))?;
        if in_flight > 0 {
            live[t] = Some((task.clone(), in_flight));
        }
        Ok(task)
    }

    /// Blocks for at least one completion and retires every completion queued.
    fn reap(&self) -> Result<(), String> {
        use std::os::unix::fs::FileExt as _;
        let done: Vec<(u64, i32)> = {
            let mut ring = self.ring.borrow_mut();
            ring.submit_and_wait(1)
                .map_err(|error| format!("engram reader: wait: {error}"))?;
            ring.completion()
                .map(|entry| (entry.user_data(), entry.result()))
                .collect()
        };
        let mut live = self.live.borrow_mut();
        for (data, result) in done {
            let (t, i) = ((data >> 32) as usize, (data & 0xffff_ffff) as usize);
            let Some((task, in_flight)) = live.get_mut(t).and_then(Option::as_mut) else {
                continue;
            };
            let gather = &task.gather;
            let (slot, file, at, _, len) = gather.jobs[i];
            // SAFETY: index i's read has completed, and its submit-time slice is gone.
            let outcome = unsafe { gather.job_slice(i, &task.staging) }.and_then(|slice| {
                let got = usize::try_from(result).map_err(|_| {
                    format!(
                        "engram reader: read at {at}: {}",
                        std::io::Error::from_raw_os_error(-result)
                    )
                })?;
                if got < len {
                    // A short read: finish it in place.
                    file.read_exact_at(&mut slice[got..], at + got as u64)
                        .map_err(|error| format!("engram reader: read at {at}: {error}"))?;
                }
                gather.remember(i, slice);
                Ok(())
            });
            if let Err(error) = outcome {
                task.progress
                    .failed
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .get_or_insert(error);
            }
            task.progress.remaining[slot].fetch_sub(1, Ordering::Release);
            *in_flight -= 1;
            if *in_flight == 0 {
                live[t] = None;
            }
        }
        Ok(())
    }

    fn wait(&self, task: &GatherTask<'r>, slot: usize) -> Result<(), String> {
        while task.progress.remaining[slot].load(Ordering::Acquire) != 0 {
            self.reap()?;
        }
        let failed = task
            .progress
            .failed
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        failed.as_ref().map_or(Ok(()), |error| Err(error.clone()))
    }
}

/// Where a generation's Engram reads run: one io_uring (the default) or the reader threads
/// (SPARKFLASH_ENGRAM_URING=0, or a kernel without io_uring).
pub(super) enum Reads<'r> {
    Uring(Box<UringReads<'r>>),
    Pool(ReadPool<'r>),
}

impl<'r> Reads<'r> {
    /// A decode step's reads: one io_uring unless SPARKFLASH_ENGRAM_URING=0 (or the kernel has
    /// none), else reader threads.
    pub(super) fn decode<'scope>(scope: &'scope std::thread::Scope<'scope, '_>) -> Self
    where
        'r: 'scope,
    {
        match (std::env::var("SPARKFLASH_ENGRAM_URING").as_deref() != Ok("0"))
            .then(UringReads::open)
        {
            Some(Ok(uring)) => Self::Uring(Box::new(uring)),
            Some(Err(error)) => {
                eprintln!("spec: {error}; reading on threads");
                Self::Pool(ReadPool::start(scope, ENGRAM_READ_THREADS))
            }
            None => Self::Pool(ReadPool::start(scope, ENGRAM_READ_THREADS)),
        }
    }

    /// A bulk prefill chunk's reads: reader threads (SPARKFLASH_PREFILL_READ_THREADS).
    pub(super) fn prefill<'scope>(scope: &'scope std::thread::Scope<'scope, '_>) -> Self
    where
        'r: 'scope,
    {
        Self::Pool(ReadPool::start(scope, prefill_read_threads()))
    }

    pub(super) fn submit(
        &self,
        gather: EngramGather<'r>,
        staging: Staging,
        threads: usize,
    ) -> Result<Arc<GatherTask<'r>>, String> {
        match self {
            Self::Uring(uring) => uring.submit(gather, staging),
            Self::Pool(pool) => pool.submit(gather, staging, threads),
        }
    }

    /// Blocks until `slot`'s reads of `task` have landed.
    fn wait(&self, task: &GatherTask<'r>, slot: usize) -> Result<(), String> {
        match self {
            Self::Uring(uring) => uring.wait(task, slot),
            Self::Pool(_) => EngramGather::wait(&task.progress, slot),
        }
    }

    pub(super) fn wait_all(&self, task: &GatherTask<'r>) -> Result<(), String> {
        (0..task.gather.slots()).try_for_each(|slot| self.wait(task, slot))
    }
}

impl DeviceLayer<'_> {
    /// Hashes rows p0..p0+n of `history` (the arena build's n-gram hashes), sets their skip flags
    /// and returns the table reads that fill their entries, staged at rows row0.. of `cap` per slot.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn engram_plan<'r>(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        reader: &'r EngramReader,
        history: &[u32],
        p0: usize,
        n: usize,
        row0: usize,
        cap: usize,
    ) -> Result<Option<EngramGather<'r>>, String> {
        let Some(engram) = self.engram.as_mut() else {
            return Ok(None);
        };
        let by_row = engram.by_row;
        if (!by_row && (p0 + n > engram.positions || row0 > p0))
            || p0 + n > history.len()
            || row0 + n > cap
        {
            return Err(format!(
                "engram set: rows {p0}..{} are outside the arena",
                p0 + n
            ));
        }
        let layout = engram.layout;
        let (columns, head_dim, scale_width, stride) = (
            layout.columns(),
            layout.head_dim(),
            layout.scale_width(),
            layout.stride(),
        );
        let align = crate::engram_device_state::ARENA_ALIGN;
        let scales_offset = layout.codes_len().div_ceil(align) * align;
        let slots = engram.layer_ids.len();
        let mut per_slot: Vec<Vec<ReadJob<'_>>> = vec![Vec::new(); slots];
        let mut hashes = Vec::new();
        let mut skips = Vec::with_capacity(n * slots);
        sparkflash_engine::forward::engram::ngram_hashes(
            config,
            &history[..p0 + n],
            p0,
            &mut hashes,
        )
        .map_err(|error| format!("engram set: hashes at {p0}..{}: {error}", p0 + n))?;
        for row in 0..n {
            let position = p0 + row;
            for (slot, slot_jobs) in per_slot.iter_mut().enumerate() {
                let layer = engram.layer_ids[slot];
                let at = (row * slots + slot) * columns;
                let ids = hashes
                    .get(at..at + columns)
                    .ok_or("engram set: short hash row")?;
                let table_rows = config
                    .engram_rows(layer)
                    .ok_or("engram set: no table height")?;
                let skip = ids.contains(&sparkflash_engine::forward::engram::HASH_SENTINEL)
                    || ids.iter().any(|&id| id as usize >= table_rows);
                if !by_row {
                    engram.skip[slot * engram.positions + position] = skip;
                }
                skips.push(skip);
                if skip {
                    continue;
                }
                let (codes, codes_at, scales, scales_at) = &reader.slots[slot];
                let base = (slot * cap + row0 + row) * stride;
                for (column, &id) in ids.iter().enumerate() {
                    let id = u64::from(id);
                    slot_jobs.push((
                        slot,
                        codes,
                        codes_at + id * head_dim as u64,
                        base + column * head_dim,
                        head_dim,
                    ));
                    slot_jobs.push((
                        slot,
                        scales,
                        scales_at + id * scale_width as u64,
                        base + scales_offset + column * scale_width,
                        scale_width,
                    ));
                }
            }
        }
        let counts = per_slot.iter().map(Vec::len).collect();
        let jobs = per_slot.into_iter().flatten().collect();
        Ok(Some(EngramGather {
            reader,
            n,
            stride,
            cap,
            row0,
            at: if by_row { 0 } else { p0 - row0 },
            skips,
            jobs,
            counts,
            cached: true,
        }))
    }

    /// Sets the skip flags of `gather`'s rows at their arena rows, when the arena is indexed by
    /// step row (by position, the plan set them). Before the step's layers are queued: the flags
    /// are host metadata the Engram layers read at launch.
    pub(super) fn engram_apply(&mut self, gather: &EngramGather<'_>) -> Result<(), String> {
        let Some(engram) = self.engram.as_mut().filter(|engram| engram.by_row) else {
            return Ok(());
        };
        let slots = gather.slots();
        let (positions, first) = (engram.positions, gather.at + gather.row0);
        if first + gather.n > positions || gather.skips.len() != gather.n * slots {
            return Err(format!(
                "engram apply: rows {first}..{} are outside the {positions}-row arena",
                first + gather.n
            ));
        }
        for row in 0..gather.n {
            for slot in 0..slots {
                engram.skip[slot * positions + first + row] = gather.skips[row * slots + slot];
            }
        }
        Ok(())
    }

    /// The pinned staging buffer of a gather of up to `rows` rows (every slot), or None without
    /// Engram state.
    pub(super) fn engram_staging(&self, rows: usize) -> Result<Option<HostMapped>, String> {
        let Some(engram) = self.engram.as_ref() else {
            return Ok(None);
        };
        let bytes = engram.layer_ids.len() * rows * engram.layout.stride();
        Ok(Some(self.cuda.alloc_host_mapped(bytes, &mut 0)?))
    }

    /// The arena address of slot `slot`'s entry at `position`.
    fn engram_entry(
        &self,
        slot: usize,
        position: usize,
        stride: usize,
    ) -> Result<Deviceptr, String> {
        let engram = self
            .engram
            .as_ref()
            .ok_or("engram upload: no Engram state")?;
        Ok(Deviceptr(
            engram.arena.0 + ((slot * engram.positions + position) * stride) as u64,
        ))
    }

    /// The Engram entries of rows p0..p0+n, read and uploaded before returning.
    pub(super) fn engram_set_rows(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        reader: &EngramReader,
        history: &[u32],
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let Some(gather) = self.engram_plan(config, reader, history, p0, n, 0, n)? else {
            return Ok(());
        };
        self.engram_apply(&gather)?;
        let mut bytes = vec![0u8; gather.slots() * n * gather.stride];
        gather.read_all(
            &Staging(bytes.as_mut_ptr(), bytes.len()),
            ENGRAM_READ_THREADS,
        )?;
        for slot in 0..gather.slots() {
            let (offset, len) = gather.slot_span(slot);
            let at = self.engram_entry(slot, gather.at, gather.stride)?;
            self.cuda.copy_htod(&at, &bytes[offset..offset + len])?;
        }
        Ok(())
    }

    /// The rows step's layers. With a gather (its reads already running into the pinned `staging`),
    /// each slot's entries, with the rows `ahead` read earlier into the rows before row0, are
    /// queued for upload just before its layer. Returns the ms the launch thread waited for reads.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spec_layers_gathered<'r>(
        &mut self,
        rows: &RowsState,
        staged: &[Option<SeamStagedWeights>],
        engram_layers: &[usize],
        reads: Option<&Reads<'r>>,
        gather: Option<(EngramGather<'r>, Staging)>,
        pinned: Option<&HostMapped>,
        mut ahead: Option<Arc<GatherTask<'r>>>,
        p0: usize,
        n: usize,
    ) -> Result<f64, String> {
        let mut waited = 0.0;
        // The submit issues every miss inline, so it runs under the first layer (the device would
        // otherwise idle through it) unless that layer reads Engram entries itself.
        let submit_at = engram_layers
            .iter()
            .copied()
            .min()
            .map_or(0, |first| first.min(1));
        if let Some((gather, _)) = gather.as_ref() {
            self.engram_apply(gather)?;
        }
        if let Some(early) = ahead.as_deref() {
            self.engram_apply(&early.gather)?;
        }
        let mut pending = gather;
        let mut task = None;
        for (layer, weights) in staged.iter().enumerate() {
            if layer == submit_at
                && let (Some(reads), Some((gather, staging))) = (reads, pending.take())
            {
                task = Some(reads.submit(gather, staging, ENGRAM_READ_THREADS)?);
            }
            if let (Some(reads), Some(task), Some(pinned), Some(slot)) = (
                reads,
                task.as_deref(),
                pinned,
                engram_layers.iter().position(|&target| target == layer),
            ) {
                let started = std::time::Instant::now();
                if let Some(early) = ahead.take() {
                    reads.wait_all(&early)?;
                }
                let gather = &task.gather;
                reads.wait(task, slot)?;
                waited += started.elapsed().as_secs_f64() * 1e3;
                let at = self.engram_entry(slot, gather.at, gather.stride)?;
                let (offset, bytes) = gather.slot_span(slot);
                self.cuda.copy_htod_async(&at, pinned, offset, bytes)?;
            }
            self.set_seam_weights(*weights);
            self.run_layer_rows(rows, layer, p0, n)?;
        }
        self.wait_collectives()?;
        Ok(waited)
    }

    /// A bulk prefill chunk's layers with its Engram reads (`task`, submitted before the chunk
    /// began, staged in `pinned`) in flight: each slot's entries are waited for and queued for upload
    /// just before its layer. `then` runs once every layer is queued and before the collectives are
    /// waited for, so the next chunk's reads go out under this chunk's device time. On return the
    /// device has finished the chunk (the staging is free again). Returns the ms spent waiting for
    /// reads.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_layers_gathered<'r>(
        &mut self,
        rows: &RowsState,
        staged: &[Option<SeamStagedWeights>],
        engram_layers: &[usize],
        reads: &Reads<'r>,
        task: Option<(&GatherTask<'r>, &HostMapped)>,
        p0: usize,
        n: usize,
        then: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<f64, String> {
        let mut waited = 0.0;
        if let Some((task, _)) = task {
            self.engram_apply(&task.gather)?;
        }
        for (layer, weights) in staged.iter().enumerate() {
            if let (Some((task, pinned)), Some(slot)) = (
                task,
                engram_layers.iter().position(|&target| target == layer),
            ) {
                let started = std::time::Instant::now();
                reads.wait(task, slot)?;
                waited += started.elapsed().as_secs_f64() * 1e3;
                let gather = &task.gather;
                let at = self.engram_entry(slot, gather.at, gather.stride)?;
                let (offset, bytes) = gather.slot_span(slot);
                self.cuda.copy_htod_async(&at, pinned, offset, bytes)?;
            }
            self.set_seam_weights(*weights);
            self.run_layer_rows(rows, layer, p0, n)?;
        }
        then(self)?;
        if !self.wait_collectives()? {
            self.cuda.synchronize()?;
        }
        Ok(waited)
    }

    pub(super) fn spec_layers(
        &mut self,
        rows: &RowsState,
        staged: &[Option<SeamStagedWeights>],
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        for (layer, weights) in staged.iter().enumerate() {
            self.set_seam_weights(*weights);
            self.run_layer_rows(rows, layer, p0, n)?;
        }
        self.wait_collectives()?;
        Ok(())
    }
}

/// The modelled cost of a verify of 1..=6 rows on GB10 (ms, the backbone rows step with the
/// head), and of one draft: only the differences matter to the depth choice.
const VERIFY_MS: [f64; 6] = [24.5, 30.2, 36.2, 38.6, 42.6, 48.5];
const DRAFT_MS: f64 = 4.0;
/// The draft-probability bins the acceptance calibration counts in.
const CALIBRATION_EDGES: [f32; 7] = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
/// The pseudo-count of the calibration prior (each bin starts at the draft's own probability).
const CALIBRATION_PRIOR: f64 = 4.0;

/// How many of the k drafts a step verifies (SPARKFLASH_DSPARK_DEPTH=fixed, the default: all k,
/// or adaptive). Adaptive picks the depth d that maximises
///   sum_{j <= d} prod_{i <= j} q_i  -  rate * verify_ms(d + 1),
/// the expected accepted drafts less what the extra rows cost at the run's modelled token rate,
/// where q_i is the acceptance probability of draft i given the ones before it were accepted:
/// the draft's softmax probability, calibrated by the acceptances seen so far in its bin. Every
/// input (the exchanged probabilities, the acceptances, the static cost table) is the same on
/// every rank, so every rank picks the same depth and launches the same collectives. On the chat,
/// code and story prompts (k = 5 adaptive against k = 3 fixed) it moved the rate by -3..+5% per
/// prompt and left the mean unchanged: the draft's probabilities are overconfident on chat and
/// underconfident on code, and 130-token runs are too short to calibrate them.
pub(super) struct DepthPolicy {
    pub(super) adaptive: bool,
    k: usize,
    min_depth: usize,
    accepted: [f64; CALIBRATION_EDGES.len() + 1],
    seen: [f64; CALIBRATION_EDGES.len() + 1],
    tokens: f64,
    ms: f64,
}

impl DepthPolicy {
    pub(super) fn from_env(k: usize) -> Result<Self, String> {
        let adaptive = match std::env::var("SPARKFLASH_DSPARK_DEPTH").as_deref() {
            Ok("adaptive") => true,
            Err(_) | Ok("fixed") => false,
            Ok(other) => {
                return Err(format!(
                    "SPARKFLASH_DSPARK_DEPTH={other}: adaptive or fixed"
                ));
            }
        };
        let min_depth = std::env::var("SPARKFLASH_DSPARK_MIN_DEPTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            .min(k);
        Ok(Self {
            adaptive,
            k,
            min_depth,
            accepted: [0.0; CALIBRATION_EDGES.len() + 1],
            seen: [0.0; CALIBRATION_EDGES.len() + 1],
            tokens: 0.0,
            ms: 0.0,
        })
    }

    fn bin(p: f32) -> usize {
        CALIBRATION_EDGES
            .iter()
            .take_while(|edge| p >= **edge)
            .count()
    }

    pub(super) fn depth(&self, probs: &[f32]) -> usize {
        if !self.adaptive {
            return self.k;
        }
        let rate = if self.ms > 0.0 {
            self.tokens / self.ms
        } else {
            0.06
        };
        let (mut best, mut best_score) = (0usize, -rate * VERIFY_MS[0]);
        let (mut chain, mut gain) = (1.0f64, 0.0f64);
        for d in 1..=self.k.min(VERIFY_MS.len() - 1) {
            let p = f64::from(probs[d - 1]);
            let b = Self::bin(probs[d - 1]);
            chain *=
                (self.accepted[b] + CALIBRATION_PRIOR * p) / (self.seen[b] + CALIBRATION_PRIOR);
            gain += chain;
            let score = gain - rate * VERIFY_MS[d];
            if score > best_score {
                (best, best_score) = (d, score);
            }
        }
        best.max(self.min_depth)
    }

    /// A step verified `depth` drafts and accepted the first `m`: drafts 0..m were accepted and
    /// draft m (when verified) was not.
    pub(super) fn observe(&mut self, probs: &[f32], depth: usize, m: usize) {
        for (i, p) in probs.iter().enumerate().take(depth) {
            let b = Self::bin(*p);
            self.seen[b] += 1.0;
            if i < m {
                self.accepted[b] += 1.0;
            } else {
                break;
            }
        }
        self.tokens += (m + 1) as f64;
        self.ms += DRAFT_MS + VERIFY_MS[depth.min(VERIFY_MS.len() - 1)];
    }
}

/// A generation's per-step callback: the step's tokens and whether the request ends with it
/// anyway; on a step that is not the last, the answer is whether another step runs.
pub(super) type OnStep<'f> = dyn FnMut(&[u32], bool) -> Result<bool, String> + 'f;

/// One generation: the prompt, the budget, the sampler and the ids that end it.
pub(super) struct SpecRequest<'a> {
    pub(super) prompt: &'a [u32],
    /// The most tokens to emit, the prefill's token included.
    pub(super) max_new: usize,
    pub(super) sampler: Sampler,
    /// Emitting one of these ends the generation after it.
    pub(super) stop_ids: &'a [u32],
    /// Cut the emission at exactly `max_new` (serving). The harness lets the last step's accepted
    /// drafts run past it, as it always has, so its token line keeps its length.
    pub(super) exact: bool,
}

/// What one generation emitted and how its steps went.
pub(super) struct SpecOutcome {
    pub(super) emitted: Vec<u32>,
    pub(super) finish: DeviceFinish,
    /// From admission to the first token.
    pub(super) prefill_ms: f64,
    /// The decode steps' wall clock, after the first token.
    pub(super) decode_ms: f64,
    pub(super) steps: usize,
    pub(super) accepted_total: usize,
    pub(super) verified_total: usize,
    pub(super) draft_ms: f64,
    pub(super) verify_ms: f64,
    /// Per-step sums: gather, engram plan, rows + head, read wait (ms).
    pub(super) phase: [f64; 4],
    pub(super) adaptive: bool,
    /// Prompt positions the prefill did not run (a prefix a lane still held).
    pub(super) reused: usize,
}

impl SpecOutcome {
    pub(super) fn device(&self) -> DeviceOutcome {
        DeviceOutcome {
            completion_tokens: self.emitted.len(),
            finish: self.finish,
            prefill_ms: self.prefill_ms,
            decode_ms: self.decode_ms,
            steps: self.steps,
            drafts_verified: self.verified_total,
            drafts_accepted: self.accepted_total,
            cached_tokens: self.reused,
        }
    }
}

/// Cuts one step's `tokens` at the first stop id (kept) and, when `exact`, at what remains of the
/// `max_new` budget after `emitted`; returns the finish the cut implies, if any.
pub(super) fn cut_step(
    tokens: &mut Vec<u32>,
    emitted: usize,
    max_new: usize,
    stop_ids: &[u32],
    exact: bool,
) -> Option<DeviceFinish> {
    let stop = tokens.iter().position(|token| stop_ids.contains(token));
    if let Some(at) = stop {
        tokens.truncate(at + 1);
    }
    if exact {
        let budget = max_new.saturating_sub(emitted);
        if tokens.len() > budget {
            tokens.truncate(budget);
            return Some(DeviceFinish::Length);
        }
    }
    if stop.is_some() {
        return Some(DeviceFinish::Stop);
    }
    (emitted + tokens.len() >= max_new).then_some(DeviceFinish::Length)
}

/// Runs `request` alone on the scheduler (super::batch): the prompt as rows steps of up to
/// `R_MAX` (or bulk chunks), then speculative generation. After every step `on_step(tokens,
/// last)` gets the tokens the step emitted; on a step that is not the last its answer decides, on
/// every rank alike, whether another step runs.
pub(super) fn generate(
    device: &mut DeviceLayer<'_>,
    rows: &RowsState,
    draft: &DraftState,
    inputs: &SpecInputs<'_>,
    k: usize,
    request: &SpecRequest<'_>,
    on_step: &mut OnStep<'_>,
) -> Result<SpecOutcome, String> {
    std::thread::scope(|scope| {
        let pools = Pools::start(scope, inputs);
        let mut batch = Batch::new(device, rows, draft, inputs, &pools, k)?;
        batch.run_one(request, on_step)
    })
}

/// The harness generation: the prompt, then greedy speculative generation of up to `max_new`
/// tokens; rank 0 prints the emitted ids and the rate.
pub(super) fn run_spec(
    device: &mut DeviceLayer<'_>,
    rows: &RowsState,
    draft: &DraftState,
    inputs: &SpecInputs<'_>,
    k: usize,
    max_new: usize,
) -> Result<(), String> {
    let request = SpecRequest {
        prompt: inputs.prompt,
        max_new,
        sampler: sampler_from_env()?,
        stop_ids: &[],
        exact: false,
    };
    let outcome = generate(device, rows, draft, inputs, k, &request, &mut |_, _| {
        Ok(true)
    })?;
    if inputs.rank != 0 {
        return Ok(());
    }
    let SpecOutcome {
        emitted,
        steps,
        accepted_total,
        verified_total,
        draft_ms,
        verify_ms,
        phase,
        ..
    } = &outcome;
    let (steps, wall) = (*steps, outcome.decode_ms / 1e3);
    println!(
        "spec: prefill {} positions in {:.1} ms, first token {}",
        inputs.prompt.len(),
        outcome.prefill_ms,
        emitted[0]
    );
    let generated = emitted.len() - 1;
    println!("spec tokens {emitted:?}");
    println!(
        "spec: k={k} steps {steps} generated {generated} in {:.1} ms = {:.2} tok/s, \
         accept length {:.3}, draft {:.2} ms/step, verify {:.2} ms/step, verify rows {:.2}, depth {}",
        wall * 1e3,
        generated as f64 / wall,
        (accepted_total + steps) as f64 / steps.max(1) as f64,
        draft_ms / steps.max(1) as f64,
        verify_ms / steps.max(1) as f64,
        (verified_total + steps) as f64 / steps.max(1) as f64,
        if outcome.adaptive {
            "adaptive"
        } else {
            "fixed"
        },
    );
    let per = |ms: f64| ms / steps.max(1) as f64;
    println!(
        "spec: per step gather {:.2} ms, engram {:.2} ms, rows+head {:.2} ms (read wait {:.2}), rest {:.2} ms",
        per(phase[0]),
        per(phase[1]),
        per(phase[2]),
        per(phase[3]),
        per(verify_ms - phase[..3].iter().sum::<f64>()),
    );
    Ok(())
}

/// The harness's sampler: greedy unless SPARKFLASH_DSPARK_TEMPERATURE (with _TOP_P, _TOP_K and
/// _SEED) asks for sampling, so the sampled path can be checked against itself across k.
fn sampler_from_env() -> Result<Sampler, String> {
    fn var<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
        match std::env::var(name) {
            Ok(value) => value
                .parse()
                .map_err(|_| format!("{name}={value}: not a number")),
            Err(_) => Ok(default),
        }
    }
    Ok(Sampler {
        temperature: var("SPARKFLASH_DSPARK_TEMPERATURE", 0.0)?,
        top_p: var("SPARKFLASH_DSPARK_TOP_P", 1.0)?,
        top_k: var("SPARKFLASH_DSPARK_TOP_K", 0)?,
        seed: var("SPARKFLASH_DSPARK_SEED", 0)?,
    })
}

/// Serves requests through `link` until it ends, several at once (super::batch). Rank 0's link
/// broadcasts each tick's admissions and cancellations and every rank runs the same scheduler tick
/// on them, so every rank launches the same collectives. A request resumes from the longest prefix
/// of its tokens a free lane still holds (an agent's turns resend the conversation so far) and
/// prefills only the rest; SPARKFLASH_PREFIX_CACHE=0 prefills every request whole.
pub(super) fn serve(
    device: &mut DeviceLayer<'_>,
    rows: &RowsState,
    draft: &DraftState,
    inputs: &SpecInputs<'_>,
    k: usize,
    link: &mut dyn DeviceLink,
) -> Result<(), String> {
    std::thread::scope(|scope| {
        let pools = Pools::start(scope, inputs);
        let mut batch = Batch::new(device, rows, draft, inputs, &pools, k)?;
        // Before rank 0 binds HTTP, so the first client does not pay the lazy module loads and
        // first-use collective setup (a warm 2K prompt's prefill is ~0.3 s faster than a cold
        // one's). SPARKFLASH_WARMUP=0 skips it.
        if std::env::var("SPARKFLASH_WARMUP").as_deref() != Ok("0") {
            let started = std::time::Instant::now();
            batch.warm_up()?;
            if inputs.rank == 0 {
                println!(
                    "sparkflash serve: warm-up {:.0} ms",
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        if inputs.rank == 0 {
            println!("sparkflash serve: {}", batch.describe());
        }
        while let Some(tick) = link.tick(batch.idle(), batch.room())? {
            for id in tick.cancel {
                if let Some(outcome) = batch.cancel(id)? {
                    link.finish(id, Ok(&outcome.device()))?;
                }
            }
            for (id, job) in &tick.admit {
                // The same decision on every rank, before any collective: the group carries on.
                if let Err(error) = batch.admit(*id, job) {
                    link.finish(*id, Err(&error))?;
                }
            }
            let emissions = match batch.tick() {
                Ok(emissions) => emissions,
                Err(error) => {
                    // A failure inside a step may have left the ranks at different collectives.
                    for id in batch.ids() {
                        let _ = link.finish(id, Err(&error));
                    }
                    return Err(error);
                }
            };
            for Emission { seq, tokens, done } in emissions {
                if !tokens.is_empty() {
                    link.emit(seq, &tokens, done.is_some());
                }
                match done {
                    Some(Ok(outcome)) => link.finish(seq, Ok(&outcome.device()))?,
                    Some(Err(error)) => link.finish(seq, Err(&error))?,
                    None => {}
                }
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_is_cut_at_its_stop_id_and_its_budget() {
        let mut step = vec![5, 6, 7, 8];
        assert_eq!(
            cut_step(&mut step, 0, 10, &[7], true),
            Some(DeviceFinish::Stop)
        );
        assert_eq!(step, [5, 6, 7]);
        let mut step = vec![5, 6, 7, 8];
        assert_eq!(
            cut_step(&mut step, 8, 10, &[], true),
            Some(DeviceFinish::Length)
        );
        assert_eq!(step, [5, 6]);
        // The harness keeps the last step's accepted drafts past the budget.
        let mut step = vec![5, 6, 7, 8];
        assert_eq!(
            cut_step(&mut step, 8, 10, &[], false),
            Some(DeviceFinish::Length)
        );
        assert_eq!(step.len(), 4);
        let mut step = vec![5, 6];
        assert_eq!(cut_step(&mut step, 0, 10, &[9], true), None);
    }

    #[test]
    fn merged_gathers_keep_each_slot_together_in_row_order() {
        let reader = EngramReader {
            slots: Vec::new(),
            cache: Mutex::new(std::collections::HashMap::new()),
            cached: true,
        };
        let file = fs::File::open("/dev/null").expect("/dev/null");
        let part = |row0: usize, at: u64| EngramGather {
            reader: &reader,
            n: 1,
            stride: 16,
            cap: 8,
            row0,
            at: 0,
            skips: vec![false, row0 == 1],
            jobs: vec![
                (0, &file, at, row0 * 16, 4),
                (1, &file, at + 1, 128 + row0 * 16, 4),
            ],
            counts: vec![1, 1],
            cached: true,
        };
        let merged = EngramGather::merge(vec![part(0, 100), part(1, 200), part(2, 300)])
            .expect("tiles")
            .expect("parts");
        assert_eq!(
            (merged.n, merged.row0, merged.counts.clone()),
            (3, 0, vec![3, 3])
        );
        assert_eq!(merged.skips, [false, false, false, true, false, false]);
        let order: Vec<(usize, u64)> = merged.jobs.iter().map(|job| (job.0, job.2)).collect();
        assert_eq!(
            order,
            [(0, 100), (0, 200), (0, 300), (1, 101), (1, 201), (1, 301)]
        );
        assert!(EngramGather::merge(vec![part(0, 100), part(2, 300)]).is_err());
        assert!(EngramGather::merge(Vec::new()).expect("empty").is_none());
    }
}
