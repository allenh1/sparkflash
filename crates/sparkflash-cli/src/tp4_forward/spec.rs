//! Speculative decoding with the on-device DSpark drafter (SPARKFLASH_DSPARK=1).
//!
//! The prompt runs as multi-row steps of up to [`R_MAX`] positions, each seeding the drafter's
//! rings with its rows' taps. Every generation step then drafts k tokens from the anchor (the last
//! emitted token, not yet run through the backbone), verifies [anchor, d1..dk] as one rows step,
//! accepts the drafts while each equals the verify argmax before it, and emits the accepted drafts
//! plus the verify argmax after them. The rows step is bit-identical to the single-row step, so the
//! emitted sequence is the single-row greedy sequence whatever the drafter proposes.
//!
//! A rejected row leaves position-addressed state behind; [`DeviceLayer::rewind_rows`] moves the
//! counters back and the next step overwrites it. The Engram entries of the verify rows are written
//! from the step's own token history before it runs.

use super::draft::{AnchorRow, BLOCK, DraftState};
use super::rows::{R_MAX, RowsState};
use super::*;
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
    pub(super) prompt: &'s [u32],
    pub(super) prompt_rows: &'s [f32],
    pub(super) rank: usize,
    pub(super) world: usize,
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
const ENGRAM_READ_THREADS: usize = 32;
/// Reader threads for a read that overlaps the draft, which has time to spare and host work of its
/// own.
const ENGRAM_AHEAD_THREADS: usize = 8;
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

/// The reads that fill the Engram entries of rows p0..p0+n: each slot's entries are contiguous in
/// the arena, so a slot is one span of the staging buffer (slot s at s * n * stride) and one upload.
pub(super) struct EngramGather<'r> {
    reader: &'r EngramReader,
    p0: usize,
    n: usize,
    stride: usize,
    /// The staging rows per slot, and the row of it p0 lands in.
    cap: usize,
    row0: usize,
    /// (slot, file, byte, offset in the staging buffer, length), slot 0's reads first.
    jobs: Vec<ReadJob<'r>>,
    /// Reads per slot.
    counts: Vec<usize>,
}

/// One Engram read: (slot, file, byte offset, staging offset, length).
type ReadJob<'f> = (usize, &'f fs::File, u64, usize, usize);

/// Entries already read, by (file, byte).
type EntryCache = std::collections::HashMap<(usize, u64), Box<[u8]>>;

/// A gather with reads in flight on the ring, and its outstanding count.
type LiveGather<'r> = (Arc<GatherTask<'r>>, usize);

/// A staging buffer the reader threads fill at disjoint offsets.
struct Staging(*mut u8, usize);
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
        self.reader
            .cached
            .then(|| self.reader.cache.lock().ok())
            .flatten()
            .and_then(|cache| cache.get(&key).map(|bytes| slice.copy_from_slice(bytes)))
            .is_some()
    }

    /// Keeps job `i`'s freshly read entry for a later lookup of the same table bytes.
    fn remember(&self, i: usize, slice: &[u8]) {
        let (_, file, at, _, _) = self.jobs[i];
        if self.reader.cached
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
struct GatherTask<'r> {
    gather: EngramGather<'r>,
    progress: ReadProgress,
    staging: Staging,
}

/// Reader threads kept for a whole generation, so a step's reads start no threads.
struct ReadPool<'r> {
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
struct UringReads<'r> {
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
enum Reads<'r> {
    Uring(Box<UringReads<'r>>),
    Pool(ReadPool<'r>),
}

impl<'r> Reads<'r> {
    fn submit(
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

    fn wait_all(&self, task: &GatherTask<'r>) -> Result<(), String> {
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
        if p0 + n > engram.positions || p0 + n > history.len() || row0 + n > cap || row0 > p0 {
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
                engram.skip[slot * engram.positions + position] = skip;
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
            p0,
            n,
            stride,
            cap,
            row0,
            jobs,
            counts,
        }))
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
        let mut bytes = vec![0u8; gather.slots() * n * gather.stride];
        gather.read_all(
            &Staging(bytes.as_mut_ptr(), bytes.len()),
            ENGRAM_READ_THREADS,
        )?;
        for slot in 0..gather.slots() {
            let (offset, len) = gather.slot_span(slot);
            let at = self.engram_entry(slot, p0, gather.stride)?;
            self.cuda.copy_htod(&at, &bytes[offset..offset + len])?;
        }
        Ok(())
    }

    /// The rows step's layers. With a gather (its reads already running into the pinned `staging`),
    /// each slot's entries, with the rows `ahead` read earlier into the rows before row0, are
    /// queued for upload just before its layer. Returns the ms the launch thread waited for reads.
    #[allow(clippy::too_many_arguments)]
    fn spec_layers_gathered<'r>(
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
                let at = self.engram_entry(slot, gather.p0 - gather.row0, gather.stride)?;
                let (offset, bytes) = gather.slot_span(slot);
                self.cuda.copy_htod_async(&at, pinned, offset, bytes)?;
            }
            self.set_seam_weights(*weights);
            self.run_layer_rows(rows, layer, p0, n)?;
        }
        self.wait_collectives()?;
        Ok(waited)
    }

    fn spec_layers(
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
struct DepthPolicy {
    adaptive: bool,
    k: usize,
    min_depth: usize,
    accepted: [f64; CALIBRATION_EDGES.len() + 1],
    seen: [f64; CALIBRATION_EDGES.len() + 1],
    tokens: f64,
    ms: f64,
}

impl DepthPolicy {
    fn from_env(k: usize) -> Result<Self, String> {
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

    fn depth(&self, probs: &[f32]) -> usize {
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
    fn observe(&mut self, probs: &[f32], depth: usize, m: usize) {
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

/// Runs the prompt and then speculative generation of up to `max_new` tokens; rank 0 prints the
/// emitted ids and the rate.
pub(super) fn run_spec(
    device: &mut DeviceLayer<'_>,
    rows: &RowsState,
    draft: &DraftState,
    inputs: &SpecInputs<'_>,
    k: usize,
    max_new: usize,
) -> Result<(), String> {
    let cuda = device.cuda;
    let dim = HIDDEN;
    let prompt_len = inputs.prompt.len();
    let mut policy = DepthPolicy::from_env(k)?;
    let counter = Cell::new(0u64);
    let print = inputs.rank == 0;
    let prefill_started = std::time::Instant::now();
    let mut p = 0usize;
    let mut first = 0u32;
    while p < prompt_len {
        let n = R_MAX.min(prompt_len - p);
        device.begin_rows(rows, p, &inputs.prompt_rows[p * dim..(p + n) * dim])?;
        device.spec_layers(rows, inputs.staged, p, n)?;
        device.draft_seed(rows, draft, p, n)?;
        if p + n == prompt_len {
            let picks = device.head_rows(rows, inputs.tail, inputs.transport, n)?;
            first = picks[n - 1].0;
        }
        p += n;
    }
    cuda.synchronize()?;
    if print {
        println!(
            "spec: prefill {prompt_len} positions in {:.1} ms, first token {first}",
            prefill_started.elapsed().as_secs_f64() * 1e3
        );
    }

    let mut tokens = inputs.prompt.to_vec();
    tokens.push(first);
    let mut emitted = vec![first];
    let mut pos = prompt_len;
    let mut anchor = first;
    // With the exchange, the rows' embeddings are gathered on the device from the ids already there.
    let exchange = device.device_exchange();
    let gather_host = |tokens: &[u32]| match exchange {
        Some(_) => Ok(Vec::new()),
        None => gather_rows(cuda, inputs, &counter, tokens),
    };
    let mut anchor_row = gather_host(&[anchor])?;
    let limit = device.max_positions;
    let mut steps = 0usize;
    let mut accepted_total = 0usize;
    let mut verified_total = 0usize;
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;
    let mut phase = [0.0f64; 4];
    let staging = match inputs.engram {
        Some(_) => device.engram_staging(BLOCK + 1)?,
        None => None,
    };
    let started = std::time::Instant::now();
    std::thread::scope(|scope| -> Result<(), String> {
        // (reader, pinned staging) and the reader threads.
        let engram = inputs.engram.zip(staging.as_ref());
        let pool = engram.map(|_| {
            match (std::env::var("SPARKFLASH_ENGRAM_URING").as_deref() != Ok("0"))
                .then(UringReads::open)
            {
                Some(Ok(uring)) => Reads::Uring(Box::new(uring)),
                Some(Err(error)) => {
                    eprintln!("spec: {error}; reading on threads");
                    Reads::Pool(ReadPool::start(scope, ENGRAM_READ_THREADS))
                }
                None => Reads::Pool(ReadPool::start(scope, ENGRAM_READ_THREADS)),
            }
        });
        let pinned = || {
            staging
                .as_ref()
                .map(|pinned| Staging(pinned.host_ptr().cast(), pinned.len()))
        };
        // The anchor row's Engram reads, run while its block drafts.
        let early = std::env::var("SPARKFLASH_ENGRAM_AHEAD").as_deref() != Ok("0");
        let read_ahead = |device: &mut DeviceLayer<'_>, tokens: &[u32], pos: usize| {
            let (Some((reader, _)), Some(pool), Some(staging), true) =
                (engram, pool.as_ref(), pinned(), early)
            else {
                return Ok(None);
            };
            let Some(gather) =
                device.engram_plan(inputs.config, reader, tokens, pos, 1, 0, BLOCK + 1)?
            else {
                return Ok(None);
            };
            pool.submit(gather, staging, ENGRAM_AHEAD_THREADS).map(Some)
        };
        let mut ahead = read_ahead(device, &tokens, pos)?;
        while emitted.len() < max_new && pos + k + BLOCK + 1 < limit {
            let t0 = std::time::Instant::now();
            let (drafted, probs) = device.draft_block(
                rows,
                draft,
                inputs.tail,
                inputs.transport,
                pos,
                anchor,
                match exchange {
                    Some(exchange) => AnchorRow::Device {
                        embed: inputs.embed.0,
                        exchange,
                    },
                    None => AnchorRow::Host(&anchor_row),
                },
                k,
            )?;
            let t1 = std::time::Instant::now();
            let depth = policy.depth(&probs);
            let drafts = &drafted[..depth];
            verified_total += depth;
            let n = depth + 1;
            tokens.truncate(pos + 1);
            tokens.extend_from_slice(drafts);
            let gather = match engram {
                Some((reader, _)) if early => device.engram_plan(
                    inputs.config,
                    reader,
                    &tokens,
                    pos + 1,
                    depth,
                    1,
                    BLOCK + 1,
                )?,
                Some((reader, _)) => {
                    device.engram_plan(inputs.config, reader, &tokens, pos, n, 0, BLOCK + 1)?
                }
                None => None,
            };
            let te = std::time::Instant::now();
            match exchange {
                Some(exchange) => {
                    device.begin_rows_device(
                        rows,
                        pos,
                        inputs.embed.0,
                        draft.ids(),
                        n,
                        exchange,
                    )?;
                }
                None => {
                    let mut hidden = anchor_row.clone();
                    hidden.extend(gather_rows(cuda, inputs, &counter, drafts)?);
                    device.begin_rows(rows, pos, &hidden)?;
                }
            }
            let tl = std::time::Instant::now();
            phase[3] += device.spec_layers_gathered(
                rows,
                inputs.staged,
                &inputs.config.engram_layer_ids,
                pool.as_ref(),
                gather.zip(pinned()),
                staging.as_ref(),
                ahead.take(),
                pos,
                n,
            )?;
            let picks = device.head_rows(rows, inputs.tail, inputs.transport, n)?;
            let th = std::time::Instant::now();
            phase[1] += (te - t1).as_secs_f64() * 1e3;
            phase[0] += (tl - te).as_secs_f64() * 1e3;
            phase[2] += (th - tl).as_secs_f64() * 1e3;
            let mut m = 0usize;
            while m < depth && drafts[m] == picks[m].0 {
                m += 1;
            }
            let bonus = picks[m].0;
            tokens.truncate(pos + 1 + m);
            tokens.push(bonus);
            policy.observe(&probs, depth, m);
            emitted.extend_from_slice(&drafts[..m]);
            emitted.push(bonus);
            device.rewind_rows(pos + m + 1);
            device.draft_seed(rows, draft, pos, m + 1)?;
            pos += m + 1;
            anchor = bonus;
            anchor_row = gather_host(&[anchor])?;
            ahead = read_ahead(device, &tokens, pos)?;
            let t2 = std::time::Instant::now();
            draft_ms += (t1 - t0).as_secs_f64() * 1e3;
            verify_ms += (t2 - t1).as_secs_f64() * 1e3;
            steps += 1;
            accepted_total += m;
            if print && std::env::var("SPARKFLASH_SPEC_TRACE").as_deref() == Ok("1") {
                println!(
                    "spec step {steps}: pos {pos} drafts {drafted:?} probs {probs:.3?} depth {depth} verify {:?} accepted {m}",
                    picks.iter().map(|pick| pick.0).collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    })?;
    cuda.synchronize()?;
    let wall = started.elapsed().as_secs_f64();
    if let Some(staging) = staging {
        cuda.free_host_mapped(&staging);
    }
    if print {
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
            if policy.adaptive { "adaptive" } else { "fixed" },
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
    }
    Ok(())
}
