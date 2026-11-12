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

use super::batch::{Batch, ENCODER_LAYERS, Emission, Pools};
use super::draft::DraftState;
use super::rows::{R_MAX, RowsState};
use super::sample::Sampler;
use super::*;
use crate::device_link::{DeviceFinish, DeviceLink, DeviceOutcome};
use sparkflash_engram::store::AlignedBuffer;
use sparkflash_kernels::cuda_driver::HostMapped;
use sparkflash_model::speculation::{DepthPolicy, StepCost};
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
    /// More bulk states of the same capacity, for chunks run together (prefill_layers_paired).
    pub(super) bulk_more: &'s [RowsState],
}

/// The shortest uncached prompt the bulk prefill takes (SPARKFLASH_PREFILL_BULK_MIN, default
/// 96). The exact rows step costs about 3.6 ms a row (a synchronous Engram read per 64-row
/// step) while a bulk chunk costs about 0.3 s almost flat (it reads every expert's weights
/// once), so the two cross near 85 rows; below that the exact step is both faster and
/// bit-identical to M = 1.
pub(super) fn bulk_min() -> usize {
    sparkflash_model::active_profile()
        .prefill
        .bulk_min
        .unwrap_or(96)
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
///
/// The files are opened again with `O_DIRECT` unless SPARKFLASH_ENGRAM_DIRECT=off: the tables are
/// ~200 GB read at random (a 256-byte code and an 8-byte scale per lookup), and through the page
/// cache every miss adds a 4 KB page. On a node whose GPU holds most of the same memory, that churn
/// kept the kernel reclaiming while it decoded: a minute of single-stream decode scanned ~1M pages
/// a node, with hundreds of direct-reclaim stalls and some swapping. Read directly, the same minute
/// reclaims nothing, and a lookup moves only the aligned window that covers it.
pub(super) struct EngramReader {
    slots: Vec<(Arc<fs::File>, u64, Arc<fs::File>, u64)>,
    /// The `O_DIRECT` alignment the files are read at, or None when they are read through the page
    /// cache.
    direct: Option<u64>,
    /// The tokenizer's compressed token map: the n-grams are hashed over compressed ids
    /// (engram.py's token map), never over raw ones.
    compressor: sparkflash_engram::compress::TokenizerCompressor,
    /// Entries already read, by (file, byte): the tables are weights, so an n-gram seen again (or a
    /// redrafted row) costs a copy instead of a device read.
    cache: Mutex<EntryCache>,
    cached: bool,
    /// Whether a bulk prefill chunk's reads are split over the ranks by column
    /// ([`EngramShard`]; SPARKFLASH_ENGRAM_SHARD=off reads every column on every rank).
    pub(super) shard: bool,
    /// Whether a gather reads each distinct table entry once and copies it to the rows that look
    /// it up again (SPARKFLASH_ENGRAM_DEDUPE=0 reads every row's entries).
    dedupe: bool,
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
    sparkflash_model::active_profile()
        .prefill
        .read_threads
        .unwrap_or(256usize)
        .clamp(1, 512)
}
/// Reads a pool thread takes from one gather before it looks for an earlier slot's elsewhere.
const READS_PER_TURN: usize = 16;
/// The most entries the reader keeps (about 130 B each).
const ENGRAM_CACHE_ENTRIES: usize = 1 << 20;
/// Submission queue entries of the io_uring reader: a verify step's gather is under 300 reads.
const ENGRAM_URING_ENTRIES: u32 = 512;

impl EngramReader {
    pub(super) fn open(
        config: &sparkflash_engine::config::ModelConfig,
        source: &dyn sparkflash_engine::weights::WeightSource,
        tokenizer: &Path,
    ) -> Result<Self, String> {
        use std::os::fd::AsRawFd as _;
        // The hash constants (primes, offsets, multipliers) are the released model's two layers.
        if config.engram_layer_ids != sparkflash_engram::ENGRAM_LAYERS
            || sparkflash_engine::forward::engram::hash_columns(config)
                != sparkflash_engram::hash::HASH_COLUMNS
        {
            return Err(format!(
                "engram reader: layers {:?} with {} hash columns are not the released Engram layout",
                config.engram_layer_ids,
                sparkflash_engine::forward::engram::hash_columns(config)
            ));
        }
        let compressor =
            sparkflash_engram::compress::TokenizerCompressor::from_tokenizer_json(tokenizer)
                .map_err(|error| format!("engram reader: {}: {error}", tokenizer.display()))?;
        let direct = sparkflash_model::active_profile()
            .diagnostics
            .engram_direct
            .unwrap_or(true);
        let shard = sparkflash_model::active_profile()
            .diagnostics
            .engram_shard
            .as_deref()
            .is_none_or(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false" | "no"
                )
            });
        let dedupe = sparkflash_model::active_profile()
            .diagnostics
            .engram_dedupe
            .as_deref()
            .is_none_or(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false" | "no"
                )
            });
        let mut align = sparkflash_engram::store::MIN_ALIGN_BYTES;
        let mut region = |name: String| -> Result<(Arc<fs::File>, u64), String> {
            let (file, start, _len) = source
                .file_region(&name)
                .ok_or_else(|| format!("engram reader: {name} is not file-backed"))?
                .map_err(|error| format!("engram reader: {name}: {error}"))?;
            if direct {
                let (file, file_align) = reopen_direct(&file).map_err(|error| {
                    format!(
                        "engram reader: {name}: {error} (SPARKFLASH_ENGRAM_DIRECT=off reads the \
                         tables through the page cache)"
                    )
                })?;
                align = align.max(file_align);
                return Ok((Arc::new(file), start));
            }
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
            direct: direct.then_some(align),
            compressor,
            cache: Mutex::new(std::collections::HashMap::new()),
            cached: sparkflash_model::active_profile()
                .diagnostics
                .engram_cache
                .unwrap_or(true),
            shard,
            dedupe,
        })
    }

    /// Where the bytes `[at, at + len)` are read from: (first byte, bytes before them, bytes read).
    /// Through the page cache that is the bytes themselves; direct, the aligned windows that cover
    /// them.
    fn window(&self, at: u64, len: usize) -> (u64, usize, usize) {
        match self.direct {
            None => (at, 0, len),
            Some(align) => {
                let (start, end) = sparkflash_engram::store::window_span(at, len as u64, align);
                (start, (at - start) as usize, (end - start) as usize)
            }
        }
    }
}

/// `file` opened again with `O_DIRECT` through its /proc/self/fd link, once one aligned read has
/// succeeded on it ([`sparkflash_engram::store::open_direct`]), and the alignment its device needs.
fn reopen_direct(file: &fs::File) -> Result<(fs::File, u64), String> {
    use std::os::fd::AsRawFd as _;
    let link = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    let direct =
        sparkflash_engram::store::open_direct(&link).map_err(|error| format!("{error:#}"))?;
    Ok((direct, sparkflash_engram::store::device_alignment(&link)))
}

/// Reads `window` (bytes of `file` from `start`) until its first `need` bytes have landed, `got`
/// of them already: a short read is finished in place, and a direct window may run past the end of
/// the file, where its read stops short.
fn finish_window(
    file: &fs::File,
    window: &mut [u8],
    start: u64,
    mut got: usize,
    need: usize,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt as _;
    while got < need {
        match file.read_at(&mut window[got..], start + got as u64) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => got += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
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
        use std::os::unix::fs::FileExt as _;
        if !sparkflash_model::active_profile()
            .diagnostics
            .engram_keepalive
            .unwrap_or(true)
        {
            return None;
        }
        let (codes, ..) = reader.slots.first()?;
        let opened = reopen_direct(codes).and_then(|(file, _)| {
            let len = file.metadata().map_err(|error| error.to_string())?.len();
            Ok((len, file))
        });
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

/// One rank's share of a sharded gather: the ranks read disjoint column ranges of every row (rank
/// r columns r * columns.., a quarter of the reads each at four ranks), staged compact, and the
/// exchange gives every rank every share ([`DeviceLayer::engram_upload`]). The arena rows end up
/// with the bytes an unsharded gather uploads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EngramShard {
    world: usize,
    rank: usize,
    /// Columns a rank reads.
    columns: usize,
    /// Bytes a staged row: the share's codes, then its scales.
    pitch: usize,
    /// The share's code bytes (where its scales start in a staged row).
    codes: usize,
    /// Where an arena row's scales start.
    scales_offset: usize,
}

impl EngramShard {
    /// Rank `rank`'s share of `layout`'s columns over `world` ranks.
    ///
    /// # Errors
    /// Columns that do not split evenly, or shares the permute cannot move 16 bytes at a time.
    fn of(
        layout: &crate::engram_device_state::EngramArenaLayout,
        world: usize,
        rank: usize,
    ) -> Result<Self, String> {
        let align = crate::engram_device_state::ARENA_ALIGN;
        let scales_offset = layout.codes_len().div_ceil(align) * align;
        let columns = layout.columns() / world.max(1);
        let (codes, scales) = (columns * layout.head_dim(), columns * layout.scale_width());
        if world == 0
            || rank >= world
            || columns * world != layout.columns()
            || ![codes, scales, layout.stride(), scales_offset]
                .iter()
                .all(|bytes| bytes.is_multiple_of(16))
        {
            return Err(format!(
                "engram shard: {} columns of {} + {} bytes do not split into 16-byte shares over \
                 {world} ranks (SPARKFLASH_ENGRAM_SHARD=off reads every column on every rank)",
                layout.columns(),
                layout.head_dim(),
                layout.scale_width()
            ));
        }
        Ok(Self {
            world,
            rank,
            columns,
            pitch: codes + scales,
            codes,
            scales_offset,
        })
    }

    /// The permutes of the exchanged shares into the arena rows: (bytes into a share, bytes into
    /// an arena row, length) for the codes, then the scales; rank r's land at r * length past it.
    fn moves(&self) -> [(usize, usize, usize); 2] {
        [
            (0, 0, self.codes),
            (self.codes, self.scales_offset, self.pitch - self.codes),
        ]
    }
}

/// How a gather stages a row's entries: (first column read, columns read, bytes a staged row,
/// where its scales start), every column in the arena row's own layout without a shard.
fn staged_row(
    layout: &crate::engram_device_state::EngramArenaLayout,
    shard: Option<EngramShard>,
) -> (usize, usize, usize, usize) {
    match shard {
        Some(shard) => (
            shard.rank * shard.columns,
            shard.columns,
            shard.pitch,
            shard.codes,
        ),
        None => {
            let align = crate::engram_device_state::ARENA_ALIGN;
            (
                0,
                layout.columns(),
                layout.stride(),
                layout.codes_len().div_ceil(align) * align,
            )
        }
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
    /// This rank's share when the ranks split the columns (staged `shard.pitch` bytes a row
    /// instead of `stride`).
    shard: Option<EngramShard>,
    /// The entries a gather did not read a second time: (slot, the staging offset they belong at,
    /// the offset of the read that lands the same table bytes, bytes). Copied in after their
    /// slot's reads have landed ([`Self::settle`]).
    dups: Vec<(usize, usize, usize, usize)>,
}

/// One Engram read: (slot, file, byte offset, staging offset, length).
type ReadJob<'f> = (usize, &'f fs::File, u64, usize, usize);

/// Drops every read whose table bytes an earlier read of the same slot already lands, recording
/// it in `dups` as (slot, where the duplicate belongs, the offset of the read that lands it,
/// bytes). The gather copies the entry across once the slot's reads have landed
/// ([`EngramGather::settle`]).
///
/// The key is the slot and the table bytes a read lands, never the row: an n-gram hashed at the
/// same bucket by two rows (repeated text, a redraft, or two columns of one row) is one entry.
fn dedupe_reads<'f>(
    per_slot: &mut [Vec<ReadJob<'f>>],
    dups: &mut Vec<(usize, usize, usize, usize)>,
) {
    let mut seen = std::collections::HashMap::new();
    for slot_jobs in per_slot {
        let mut kept = Vec::with_capacity(slot_jobs.len());
        for job in slot_jobs.drain(..) {
            let key = (job.0, std::ptr::from_ref(job.1) as usize, job.2, job.4);
            match seen.entry(key) {
                std::collections::hash_map::Entry::Occupied(read) => {
                    dups.push((job.0, job.3, *read.get(), job.4));
                }
                std::collections::hash_map::Entry::Vacant(first) => {
                    first.insert(job.3);
                    kept.push(job);
                }
            }
        }
        *slot_jobs = kept;
    }
}

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
                || part.shard != merged.shard
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
            merged.dups.extend(part.dups);
            merged.cached &= part.cached;
        }
        // Each slot's reads together, slot 0's first (the sort is stable: rows stay in order).
        merged.jobs.sort_by_key(|job| job.0);
        Ok(Some(merged))
    }

    /// Bytes a staged row.
    fn pitch(&self) -> usize {
        self.shard.map_or(self.stride, |shard| shard.pitch)
    }

    /// Slot `slot`'s staged rows 0..row0 + n: (byte offset in the staging buffer, bytes).
    fn slot_span(&self, slot: usize) -> (usize, usize) {
        (
            slot * self.cap * self.pitch(),
            (self.row0 + self.n) * self.pitch(),
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

    /// Copies the entries this gather read once into the rows that look the same table bytes up
    /// again, once `slot`'s reads have landed. Idempotent: the source is a read of this slot, so
    /// a second call copies the same bytes over the same bytes.
    fn settle(&self, slot: usize, staging: &Staging) -> Result<(), String> {
        for &(_, dst, src, len) in self.dups.iter().filter(|dup| dup.0 == slot) {
            if src + len > staging.1 || dst + len > staging.1 {
                return Err(
                    "engram reader: a copied read lands outside the staging buffer".to_owned(),
                );
            }
            // SAFETY: both ranges are inside the buffer (checked) and inside one slot's staging
            // rows, whose reads have all landed; the source of a copy is a job that survived the
            // dedupe, so no other write touches it.
            unsafe {
                std::ptr::copy_nonoverlapping(staging.0.add(src), staging.0.add(dst), len);
            }
        }
        Ok(())
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

    /// Reads job `i`'s entry into `slice`: straight in through the page cache, else through
    /// `bounce`, grown to the job's aligned window.
    fn read_job(
        &self,
        i: usize,
        slice: &mut [u8],
        bounce: &mut AlignedBuffer,
    ) -> std::io::Result<()> {
        let (_, file, at, _, len) = self.jobs[i];
        if self.reader.direct.is_none() {
            return finish_window(file, slice, at, 0, len);
        }
        let (start, skip, bytes) = self.reader.window(at, len);
        if bounce.as_slice().len() < bytes {
            *bounce = AlignedBuffer::new(bytes);
        }
        let window = &mut bounce.as_mut_slice()[..bytes];
        finish_window(file, window, start, 0, skip + len)?;
        slice.copy_from_slice(&window[skip..skip + len]);
        Ok(())
    }

    /// Takes reads until none is left.
    fn work(&self, progress: &ReadProgress, staging: &Staging) {
        let mut bounce = AlignedBuffer::new(0);
        while self.take(progress, staging, &mut bounce) {}
    }

    /// The slot of the next read not yet taken, if any is left.
    fn next_slot(&self, progress: &ReadProgress) -> Option<usize> {
        self.jobs
            .get(progress.next.load(Ordering::Relaxed))
            .map(|job| job.0)
    }

    /// Takes the next read and lands it; false when none was left.
    fn take(&self, progress: &ReadProgress, staging: &Staging, bounce: &mut AlignedBuffer) -> bool {
        let i = progress.next.fetch_add(1, Ordering::Relaxed);
        let Some(&(slot, _, at, _, _)) = self.jobs.get(i) else {
            return false;
        };
        // SAFETY: `next` gave index i to this thread alone.
        let result = unsafe { self.job_slice(i, staging) }.and_then(|slice| {
            if self.cached_copy(i, slice) {
                return Ok(());
            }
            self.read_job(i, slice, bounce)
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
        true
    }

    /// Runs every read on `threads` threads and returns once all have landed.
    fn read_all(&self, staging: &Staging, threads: usize) -> Result<(), String> {
        let progress = self.progress();
        std::thread::scope(|scope| {
            for _ in 0..threads.min(self.jobs.len()) {
                scope.spawn(|| self.work(&progress, staging));
            }
        });
        for slot in 0..self.slots() {
            Self::wait(&progress, slot)?;
            self.settle(slot, staging)?;
        }
        Ok(())
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
    /// The ring's direct reads' windows (the pool's threads read through their own).
    windows: Option<Windows>,
}

/// The aligned windows a gather's direct reads land in on the ring, one per job in one arena, each
/// copied to the staging buffer as its read completes.
struct Windows {
    /// Per job: (first byte, bytes before the entry, window bytes, offset in the arena).
    spans: Vec<(u64, usize, usize, usize)>,
    base: Staging,
    /// Owns the memory `base` points into (a Vec's heap block does not move with it).
    _arena: AlignedBuffer,
}

impl Windows {
    fn of(gather: &EngramGather<'_>) -> Self {
        let mut bytes = 0;
        let spans = gather
            .jobs
            .iter()
            .map(|&(_, _, at, _, len)| {
                let (start, skip, window) = gather.reader.window(at, len);
                bytes += window;
                (start, skip, window, bytes - window)
            })
            .collect();
        let mut arena = AlignedBuffer::new(bytes);
        let base = Staging(arena.as_mut_slice().as_mut_ptr(), bytes);
        Self {
            spans,
            base,
            _arena: arena,
        }
    }

    /// Job `i`'s window.
    ///
    /// # Safety
    /// As [`EngramGather::job_slice`]: each index is held by one reader at a time.
    #[allow(clippy::mut_from_ref)]
    unsafe fn window(&self, i: usize) -> &mut [u8] {
        let (_, _, bytes, offset) = self.spans[i];
        // SAFETY: inside the arena (sized from these spans), and disjoint from every other job's.
        unsafe { std::slice::from_raw_parts_mut(self.base.0.add(offset), bytes) }
    }
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
                    let mut bounce = AlignedBuffer::new(0);
                    let mut held: Vec<Arc<GatherTask<'r>>> = Vec::new();
                    loop {
                        held.extend(tasks.try_iter());
                        held.retain(|task| task.gather.next_slot(&task.progress).is_some());
                        // The earliest slot's reads first, the oldest task's of those: a tick's
                        // chunks then all reach their first Engram layer before any read for the
                        // next one lands. A few reads at a time, so a later task's first slot goes
                        // ahead of the rest of this one.
                        let Some(task) = held
                            .iter()
                            .min_by_key(|task| task.gather.next_slot(&task.progress))
                        else {
                            match tasks.recv() {
                                Ok(task) => held.push(task),
                                Err(_) => break,
                            }
                            continue;
                        };
                        for _ in 0..READS_PER_TURN {
                            if !task.gather.take(&task.progress, &task.staging, &mut bounce) {
                                break;
                            }
                        }
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
            windows: None,
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
            windows: gather.reader.direct.map(|_| Windows::of(&gather)),
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
        for (i, &(slot, file, at, _, _)) in task.gather.jobs.iter().enumerate() {
            // SAFETY: the task is new, so nothing else holds index i yet.
            let slice = unsafe { task.gather.job_slice(i, &task.staging) }?;
            if task.gather.cached_copy(i, slice) {
                task.progress.remaining[slot].fetch_sub(1, Ordering::Relaxed);
                continue;
            }
            let (target, start) = match &task.windows {
                // SAFETY: as the staging slice: index i is this submit's alone.
                Some(windows) => (unsafe { windows.window(i) }, windows.spans[i].0),
                None => (slice, at),
            };
            let entry = io_uring::opcode::Read::new(
                io_uring::types::Fd(file.as_raw_fd()),
                target.as_mut_ptr(),
                target.len() as u32,
            )
            .offset(start)
            .build()
            .user_data(((t as u64) << 32) | i as u64);
            // SAFETY: the buffer is the job's own staging range or window, which outlives the read
            // (the task stays in `live` until every read of it has completed).
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
                let read = |window: &mut [u8], start, need| {
                    finish_window(file, window, start, got, need)
                        .map_err(|error| format!("engram reader: read at {at}: {error}"))
                };
                match &task.windows {
                    Some(windows) => {
                        let (start, skip, ..) = windows.spans[i];
                        // SAFETY: index i's read has completed, and its submit-time window is gone.
                        let window = unsafe { windows.window(i) };
                        read(window, start, skip + len)?;
                        slice.copy_from_slice(&window[skip..skip + len]);
                    }
                    None => read(slice, at, len)?,
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
        match (sparkflash_model::active_profile()
            .diagnostics
            .engram_uring
            .unwrap_or(true))
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
            Self::Uring(uring) => uring.wait(task, slot)?,
            Self::Pool(_) => EngramGather::wait(&task.progress, slot)?,
        }
        task.gather.settle(slot, &task.staging)
    }

    pub(super) fn wait_all(&self, task: &GatherTask<'r>) -> Result<(), String> {
        (0..task.gather.slots()).try_for_each(|slot| self.wait(task, slot))
    }
}

impl DeviceLayer<'_> {
    /// Hashes rows p0..p0+n of `history` through the compressed token map, sets their skip flags
    /// and returns the table reads that fill their entries, staged at rows row0.. of `cap` per slot.
    /// With `shard` (world, rank), only this rank's share of the columns ([`EngramShard`]).
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
        shard: Option<(usize, usize)>,
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
        let shard = shard
            .map(|(world, rank)| EngramShard::of(&layout, world, rank))
            .transpose()?;
        let (first, share, pitch, scales_offset) = staged_row(&layout, shard);
        let slots = engram.layer_ids.len();
        let mut per_slot: Vec<Vec<ReadJob<'_>>> = vec![Vec::new(); slots];
        let mut hashes = Vec::new();
        let mut skips = Vec::with_capacity(n * slots);
        sparkflash_engram::hash::hash_history(
            &reader.compressor,
            &history[..p0 + n],
            p0,
            sparkflash_engine::forward::engram::HASH_SENTINEL,
            &mut hashes,
        );
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
                let base = (slot * cap + row0 + row) * pitch;
                for (column, &id) in ids[first..first + share].iter().enumerate() {
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
        // The same n-gram looked up by two rows (repeated text, a redraft, or two columns that
        // hash to one bucket) is one table entry: read it once, copy it to every row that wants
        // it. A verify step over eight lanes repeats a good share of its 2/3/4-grams.
        let mut dups = Vec::new();
        if reader.dedupe {
            dedupe_reads(&mut per_slot, &mut dups);
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
            shard,
            dups,
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

    /// Queues slot `slot`'s staged entries of `gather` (landed in `pinned`) for its arena rows on
    /// the run's stream: one copy, or for a sharded gather a copy of this rank's share into its
    /// block of the rows state's share buffer, the exchange of every rank's (the ring AllGather
    /// past R_MAX rows, as the embedding gather), then the shares' codes and scales permuted into
    /// the rows.
    fn engram_upload(
        &self,
        rows: &RowsState,
        gather: &EngramGather<'_>,
        slot: usize,
        pinned: &HostMapped,
    ) -> Result<(), String> {
        let at = self.engram_entry(slot, gather.at, gather.stride)?;
        let (offset, bytes) = gather.slot_span(slot);
        let Some(shard) = gather.shard else {
            return self.cuda.copy_htod_async(&at, pinned, offset, bytes);
        };
        let staged = gather.row0 + gather.n;
        let shares = rows
            .bulk
            .as_ref()
            .and_then(|bulk| bulk.engram_shares)
            .filter(|&(_, len)| len >= shard.world * bytes)
            .ok_or_else(|| format!("engram upload: no share buffer for {staged} sharded rows"))?
            .0;
        self.cuda.copy_htod_async(
            &Deviceptr(shares + (shard.rank * bytes) as u64),
            pinned,
            offset,
            bytes,
        )?;
        if staged > R_MAX {
            self.gather_slots(shares, bytes / 2, shard.rank, "engram shares")?;
        } else {
            self.exchange_slots(shares, bytes / 2, shard.rank, "engram shares")?;
        }
        let pitches = [(shard.pitch / 16) as u64, (gather.stride / 16) as u64];
        for (from, to, len) in shard.moves() {
            let vecs = (len / 16) as u32;
            self.cuda.launch(
                rows.kernels.rank_permute,
                (
                    (shard.world as u32 * staged as u32 * vecs).div_ceil(256),
                    1,
                    1,
                ),
                (256, 1, 1),
                &[
                    shares + from as u64,
                    at.0 + to as u64,
                    staged as u64,
                    u64::from(vecs),
                    shard.world as u64,
                    pitches[0],
                    pitches[1],
                ],
            )?;
        }
        Ok(())
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
        let Some(gather) = self.engram_plan(config, reader, history, p0, n, 0, n, None)? else {
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
        // One pass over the weights, one ledger step: this is the only place a decode pass's row
        // count is known, and it is the denominator the ledger's per-kernel times divide by.
        sparkflash_kernels::cuda_driver::stage_timing_forward("decode", n);
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
                reads.wait(task, slot)?;
                waited += started.elapsed().as_secs_f64() * 1e3;
                self.engram_upload(rows, &task.gather, slot, pinned)?;
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
        sparkflash_kernels::cuda_driver::stage_timing_forward("prefill", n);
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
                self.engram_upload(rows, &task.gather, slot, pinned)?;
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

    /// Bulk chunks of a tick (consecutive ones of one sequence, or several sequences' own),
    /// their layers interleaved so that each chunk's collectives run on the collective stream under
    /// the other chunks' launches: per layer, the chunks' attention halves in order, then their FFN
    /// halves in the same order. Each chunk's launches are its single-chunk launches in their
    /// order, and a chunk's attention at a layer is queued after the chunk before it has written
    /// that layer's caches, so the results are the chunks' run one after the other. Each chunk's
    /// Engram reads (`task`, staged in its pinned buffer) are waited for and uploaded just before
    /// its slot's layer (the chunks share the arena rows: the run's stream orders each upload after
    /// the previous chunk's reads of them). The first `encoder_only` chunks stop after the
    /// encoder layers ([`ENCODER_LAYERS`]). Each chunk's launches run on its lane
    /// ([`DeviceLayer::enter_lane`]; [`super::lanes::NO_LANE`] keeps the selected one). `then`
    /// runs once every layer is queued. Returns the ms spent waiting for reads.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(super) fn prefill_layers_paired<'r>(
        &mut self,
        chunks: &[(
            &RowsState,
            usize,
            usize,
            usize,
            Option<(&GatherTask<'r>, &HostMapped)>,
        )],
        staged: &[Option<SeamStagedWeights>],
        encoder_only: usize,
        engram_layers: &[usize],
        reads: Option<&Reads<'r>>,
        then: impl FnOnce(&mut Self) -> Result<(), String>,
    ) -> Result<f64, String> {
        if !self.bulk_comm_open() || chunks.is_empty() {
            return Err("paired prefill: no collective stream or no chunk".to_owned());
        }
        // The chunks' layers interleave, so the pass is one ledger step carrying all of their rows.
        sparkflash_kernels::cuda_driver::stage_timing_forward(
            "prefill",
            chunks.iter().map(|&(_, _, _, n, _)| n).sum(),
        );
        if encoder_only > chunks.len() || staged.len() < ENCODER_LAYERS {
            return Err(format!(
                "paired prefill: {encoder_only} encoder-only chunks of {} over {} layers",
                chunks.len(),
                staged.len()
            ));
        }
        for &(rows, _, _, n, _) in chunks {
            if rows.bulk.is_none() || n <= R_MAX {
                return Err(format!("paired prefill: {n} rows are not a bulk chunk"));
            }
            rows.overlap.set(true);
        }
        // Split seams ([Self::enable_prefill_sp]) add an AllGather after each seam: the chunks'
        // seams run first, then their halves past them, so each gather runs under the later
        // chunks' seams (and the earlier chunks' rests).
        let split = chunks
            .iter()
            .any(|&(rows, _, _, n, _)| self.sp_rows(rows, n).is_some());
        // Split, the chunks' routed experts run as one pass over the expert weights when their
        // states share the routed rows ([Self::rows_moe_merged]; SPARKFLASH_PREFILL_MOE_MERGE=0:
        // a pass a chunk). Each row's values are its chunk's own pass's.
        let merged: Vec<(&RowsState, usize)> =
            chunks.iter().map(|&(rows, _, _, n, _)| (rows, n)).collect();
        let merge = split
            && chunks.len() > 1
            && sparkflash_model::active_profile()
                .prefill
                .moe_merge
                .unwrap_or(true)
            && chunks.iter().all(|&(rows, ..)| rows.moe_way.is_some());
        let result: Result<f64, String> = (|| {
            let mut waited = 0.0;
            for (layer, weights) in staged.iter().enumerate() {
                // The decoder layers run the chunks after the encoder-only ones (a suffix, so the
                // merged pass's rows are the suffix of its list too).
                let chunks = if layer < ENCODER_LAYERS {
                    chunks
                } else {
                    &chunks[encoder_only..]
                };
                if chunks.is_empty() {
                    break;
                }
                let merged = &merged[merged.len() - chunks.len()..];
                let merge = merge && chunks.len() > 1;
                self.set_seam_weights(*weights);
                let slot = engram_layers.iter().position(|&target| target == layer);
                for &(rows, lane, p0, n, task) in chunks {
                    self.enter_lane(lane)?;
                    if let (Some(reads), Some((task, pinned)), Some(slot)) = (reads, task, slot) {
                        let started = std::time::Instant::now();
                        reads.wait(task, slot)?;
                        waited += started.elapsed().as_secs_f64() * 1e3;
                        self.engram_apply(&task.gather)?;
                        self.engram_upload(rows, &task.gather, slot, pinned)?;
                    }
                    if split {
                        self.run_layer_attn_seam(rows, layer, p0, n)?;
                    } else {
                        self.run_layer_attn(rows, layer, p0, n)?;
                    }
                }
                if split {
                    for &(rows, lane, p0, n, _) in chunks {
                        self.enter_lane(lane)?;
                        self.run_layer_attn_rest(rows, layer, p0, n)?;
                    }
                    for &(rows, lane, _, n, _) in chunks {
                        self.enter_lane(lane)?;
                        self.run_layer_ffn_seam(rows, layer, n)?;
                    }
                }
                if merge {
                    for &(rows, lane, _, n, _) in chunks {
                        self.enter_lane(lane)?;
                        self.ffn_rest_inputs(rows, layer, n)?;
                    }
                    self.rows_moe_merged(merged, layer)?;
                    // A chunk's reduce goes out behind its own combine, under the next chunk's.
                    for &(rows, lane, _, n, _) in chunks {
                        self.enter_lane(lane)?;
                        self.rows_moe_merged_combine(rows, n)?;
                        self.ffn_rest_tail(rows, layer, n)?;
                    }
                    continue;
                }
                for &(rows, lane, p0, n, _) in chunks {
                    self.enter_lane(lane)?;
                    if split {
                        self.run_layer_ffn_rest(rows, layer, p0, n)?;
                    } else {
                        self.run_layer_ffn(rows, layer, p0, n)?;
                    }
                }
            }
            for &(rows, _, _, n, _) in chunks {
                self.rows_ar_wait(rows, n)?;
            }
            Ok(waited)
        })();
        for &(rows, ..) in chunks {
            rows.end_overlap(result.is_err());
        }
        let waited = result?;
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
        sparkflash_kernels::cuda_driver::stage_timing_forward("prefill", n);
        for (layer, weights) in staged.iter().enumerate() {
            self.set_seam_weights(*weights);
            self.run_layer_rows(rows, layer, p0, n)?;
        }
        self.wait_collectives()?;
        Ok(())
    }
}

/// The modelled wall of a c=1 step on GB10 that verifies `depth` drafts (ms): the draft, the
/// backbone rows step and the head, measured at 27 ms plus 4 ms a verified row (each row reads its
/// own six experts a layer). Only the differences matter to the depth choice; the first step
/// assumes 60 tok/s.
pub(super) const STEP_COST: StepCost = StepCost {
    step_ms: 27.0,
    row_ms: 4.0,
    prior_tokens_per_ms: 0.06,
};

/// The rows the shared step's cost model bends at: a step of up to this many rows is the
/// narrow body, past it the wide one (the bucket split the model is fit over).
pub(super) const BAND_ROWS: usize = 16;

/// What a shared step costs for each sequence past the first, besides its anchor row (ms): its
/// draft block, its segment's seam and select, its head row. This is the term the step model
/// carried before it was measured; a four-number profile value keeps it, so a value that does not
/// name it cannot move the per-sequence part of a step.
pub(super) const SEQ_MS: f64 = 2.3;

/// A shared step's cost model: a0 + a1 * rows up to [BAND_ROWS] rows, b0 + b1 * rows above,
/// and seq_ms for each sequence past the first (ms). The trace's per-tick decode line (rows and
/// milliseconds) is this model's target: one shared step, in rows and in sequences.
pub(super) type StepModel = (f64, f64, f64, f64, f64);

/// The modelled wall of a shared step over the given number of verified rows (ms).
#[must_use]
pub(super) fn batch_step_ms(model: StepModel, rows: usize) -> f64 {
    let (a0, a1, b0, b1, _) = model;
    if rows <= BAND_ROWS {
        a0 + a1 * rows as f64
    } else {
        b0 + b1 * rows as f64
    }
}

/// [batch_step_cost] over one profile value: a0,a1,b0,b1[,seq_ms], or the flat band that
/// reproduces [STEP_COST] over the step's actual rows when the value is absent. (A malformed
/// value never reaches here in a serving process: `ModelProfile::validate` refuses it at boot;
/// the fallback below is the function's own contract for an unvalidated caller.)
///
/// The flat model's row count is the step's rows (`seqs + drafts`, the count the trace and the
/// fit use), while [STEP_COST] prices a c=1 step by its *drafts*: 27 ms covers the anchor and
/// 4 ms each verified row. The same curve over the step's rows is therefore `(27 - 4) + 4 * rows`,
/// and with it an unset knob is exactly the pre-bend model the share ran - 27 + 4 x drafts plus
/// [SEQ_MS] a sequence past the first - so the switch cannot move a step it was not asked to move.
#[must_use]
pub(super) fn batch_step_cost_from(value: Option<&str>) -> StepModel {
    let flat = (
        STEP_COST.step_ms - STEP_COST.row_ms,
        STEP_COST.row_ms,
        STEP_COST.step_ms - STEP_COST.row_ms,
        STEP_COST.row_ms,
        SEQ_MS,
    );
    let parsed = value.and_then(|text| {
        let parts = text
            .split(',')
            .map(|part| part.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let sane = parts.iter().all(|ms| ms.is_finite() && *ms >= 0.0);
        (sane && (parts.len() == 4 || parts.len() == 5)).then(|| {
            (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts.get(4).copied().unwrap_or(SEQ_MS),
            )
        })
    });
    parsed.unwrap_or(flat)
}

/// The shared step's cost model the profile names ([batch_step_cost_from]).
#[must_use]
pub(super) fn batch_step_cost() -> StepModel {
    batch_step_cost_from(
        sparkflash_model::active_profile()
            .speculation
            .batch_spec_cost
            .as_deref(),
    )
}

/// The depth policy's per-row cost, from `SPARKFLASH_SPEC_ROW_MS` when it names a non-negative
/// number, else [STEP_COST]'s.
///
/// This exists for one A/B (issue #20 WS4): a c1 step of six rows took 38 ms and carries about
/// eleven experts, so with 22.8 ms fixed, 0.97 ms an expert and 0.71 ms a row of seam, head and
/// collectives the marginal row is **0.75 ms** - not the 4 ms `STEP_COST` charges, which was
/// justified as "each row reads its own six experts a layer" and is five times the measured cost.
/// Only the depth policy takes it, not `step_ms`, so the multi-sequence share's term ([SEQ_MS]) is
/// untouched and the A/B moves one decision.
///
/// It stays an override rather than the default until that A/B has run at both shapes: the cheaper
/// row makes the policy verify deeper, which is only right if the extra rows really do cost what
/// this says.
#[must_use]
pub(super) fn row_cost_ms_from(value: Option<&str>) -> Option<f64> {
    value
        .and_then(|text| text.trim().parse::<f64>().ok())
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
}

/// [row_cost_ms_from] over the typed profile (`speculation.row_ms`,
/// `SPARKFLASH_SPEC_ROW_MS`).
#[must_use]
fn row_cost_ms() -> Option<f64> {
    row_cost_ms_from(
        sparkflash_model::active_profile()
            .speculation
            .row_ms
            .as_deref(),
    )
}

/// The cost model the depth policy is built on.
#[must_use]
pub(super) fn depth_cost() -> StepCost {
    match row_cost_ms() {
        Some(row_ms) => StepCost {
            row_ms,
            ..STEP_COST
        },
        None => STEP_COST,
    }
}

/// How many of the k drafts a step verifies: the shared [DepthPolicy] over [STEP_COST], from the
/// profile's `speculation.depth_policy` (adaptive, the default, or fixed) and `min_depth`.
/// Every input (the exchanged probabilities, the acceptances, the static cost model) is the same
/// on every rank, so every rank picks the same depth and launches the same collectives. At c=1 on
/// GB10 (k = 5) it took prose at 2K context from 52.0 / 56.8 to 59.9 / 65.3 tok/s and the
/// explanation prompts from 57.1 / 67.4 to 64.8 / 73.0, and left code at 107-112 (its drafts are
/// accepted to the full depth); pinned to depth 5 (`speculation.min_depth = 5`) it ran as fixed,
/// so the host read of the probabilities costs nothing.
pub(super) fn depth_policy(k: usize) -> DepthPolicy {
    DepthPolicy::from_profile(k, depth_cost())
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
        let mut batch = Batch::new(device, rows, draft, inputs, &pools, k)
            .inspect_err(|error| link.fail(error))?;
        let served = serve_ticks(&mut batch, inputs, link);
        // The link hears of a failure before the scheduler and its read pools are torn down.
        if let Err(error) = &served {
            link.fail(error);
        }
        served
    })
}

/// [`serve`]'s warm-up and tick loop on its scheduler.
fn serve_ticks(
    batch: &mut Batch<'_, '_, '_>,
    inputs: &SpecInputs<'_>,
    link: &mut dyn DeviceLink,
) -> Result<(), String> {
    // Before rank 0 binds HTTP, so the first client does not pay the lazy module loads and
    // first-use collective setup (a warm 2K prompt's prefill is ~0.3 s faster than a cold
    // one's). SPARKFLASH_WARMUP=0 skips it.
    if sparkflash_model::active_profile()
        .prefill
        .warmup
        .unwrap_or(true)
    {
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
    // SPARKFLASH_TICK_AHEAD=0: rank 0 decides each tick when the last one ends, and the
    // workers wait for its frame at every step.
    let send_ahead = sparkflash_model::active_profile()
        .prefill
        .tick_ahead
        .unwrap_or(true);
    while let Some(tick) = link.tick(batch.idle(), batch.room(), batch.hold_prefill())? {
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
        let mut ahead = |room, hold| {
            if send_ahead {
                link.ahead(room, hold)
            } else {
                Ok(())
            }
        };
        let emissions = match batch.tick(tick.hold_prefill, &mut ahead) {
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

    /// A reader over no tables, reading direct at `direct`'s alignment (None: through the cache).
    fn reader(direct: Option<u64>) -> EngramReader {
        EngramReader {
            slots: Vec::new(),
            direct,
            compressor: sparkflash_engram::compress::TokenizerCompressor::from_lookup(
                vec![0; 3],
                sparkflash_engram::layout::COMPRESSED_VOCAB_SIZE,
            )
            .expect("lookup"),
            cache: Mutex::new(std::collections::HashMap::new()),
            cached: true,
            shard: true,
            dedupe: true,
        }
    }

    #[test]
    fn direct_windows_land_the_same_bytes_as_reads_through_the_cache() {
        use std::io::Write as _;
        // 5000 bytes, not a whole number of 512-byte blocks, so the last window runs past the end.
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i * 7 + i / 251) as u8).collect();
        let mut file = tempfile::tempfile().expect("a temporary file");
        file.write_all(&bytes).expect("write");
        // A row's two reads at the start, straddling a block, inside one, and at the end.
        let reads = [
            (0u64, 256usize),
            (300, 256),
            (510, 8),
            (4096, 8),
            (4990, 10),
            (4744, 256),
        ];
        let total: usize = reads.iter().map(|&(_, len)| len).sum();
        let want: Vec<u8> = reads
            .iter()
            .flat_map(|&(at, len)| bytes[at as usize..at as usize + len].to_vec())
            .collect();
        for direct in [None, Some(512), Some(4096)] {
            let reader = reader(direct);
            let gather = || {
                let mut dst = 0;
                let jobs = reads
                    .iter()
                    .map(|&(at, len)| {
                        dst += len;
                        (0, &file, at, dst - len, len)
                    })
                    .collect();
                EngramGather {
                    reader: &reader,
                    n: 1,
                    stride: total,
                    cap: 1,
                    row0: 0,
                    at: 0,
                    skips: vec![false],
                    jobs,
                    counts: vec![reads.len()],
                    cached: false,
                    shard: None,
                    dups: Vec::new(),
                }
            };
            let mut got = vec![0u8; total];
            gather()
                .read_all(&Staging(got.as_mut_ptr(), total), 2)
                .expect("the pool's reads");
            assert_eq!(got, want, "pool, direct {direct:?}");
            match UringReads::open() {
                Ok(ring) => {
                    let mut got = vec![0u8; total];
                    let task = ring
                        .submit(gather(), Staging(got.as_mut_ptr(), total))
                        .expect("submit");
                    ring.wait(&task, 0).expect("the ring's reads");
                    drop(task);
                    assert_eq!(got, want, "ring, direct {direct:?}");
                }
                // The CI runner's seccomp profile forbids io_uring_setup (engram's ring_or_skip).
                Err(error)
                    if !sparkflash_model::active_profile()
                        .diagnostics
                        .require_io_uring
                        .unwrap_or(false) =>
                {
                    eprintln!("skipped the ring's reads ({error}): no io_uring here");
                }
                Err(error) => panic!("SPARKFLASH_REQUIRE_IO_URING is set: {error}"),
            }
        }
        // A direct window is its aligned span; through the cache, the bytes themselves.
        assert_eq!(reader(Some(512)).window(300, 256), (0, 300, 1024));
        assert_eq!(reader(Some(512)).window(4990, 10), (4608, 382, 512));
        assert_eq!(reader(None).window(300, 256), (300, 0, 256));
        // A short read is finished from where it stopped.
        let mut window = vec![0u8; 512];
        window[..100].copy_from_slice(&bytes[4608..4708]);
        finish_window(&file, &mut window, 4608, 100, 392).expect("the rest of the window");
        assert_eq!(window[..392], bytes[4608..]);
        assert!(finish_window(&file, &mut window, 4608, 0, 393).is_err());
    }

    #[test]
    fn pooled_engram_reads_land_every_task_s_bytes() {
        use std::io::Write as _;
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 13 + i / 97) as u8).collect();
        let mut file = tempfile::tempfile().expect("a temporary file");
        file.write_all(&bytes).expect("write");
        // (slot, byte, length): slot 0 repeats one entry, slot 1 reads one entry three times.
        let entries = [
            (0, 300u64, 16usize),
            (0, 0, 16),
            (0, 300, 16),
            (1, 1000, 8),
            (1, 1000, 8),
            (1, 1000, 8),
        ];
        let total: usize = entries.iter().map(|entry| entry.2).sum();
        let want: Vec<u8> = entries
            .iter()
            .flat_map(|&(_, at, len)| bytes[at as usize..at as usize + len].to_vec())
            .collect();
        let reader = reader(Some(512));
        let gather = || {
            let mut dst = 0;
            let jobs = entries
                .iter()
                .map(|&(slot, at, len)| {
                    dst += len;
                    (slot, &file, at, dst - len, len)
                })
                .collect();
            EngramGather {
                reader: &reader,
                n: 1,
                stride: total,
                cap: 1,
                row0: 0,
                at: 0,
                skips: vec![false, false],
                jobs,
                counts: vec![3, 3],
                cached: false,
                shard: None,
                dups: Vec::new(),
            }
        };
        // Two tasks on one pool, whose workers take the earliest slot of either; the later one
        // waited for first.
        let (mut first, mut second) = (vec![0u8; 2 * total], vec![0u8; 2 * total]);
        std::thread::scope(|scope| {
            let reads = Reads::Pool(ReadPool::start(scope, 2));
            let one = reads
                .submit(gather(), Staging(first.as_mut_ptr(), first.len()), 2)
                .expect("submit");
            let two = reads
                .submit(gather(), Staging(second.as_mut_ptr(), second.len()), 2)
                .expect("submit");
            reads.wait_all(&two).expect("the second task's reads");
            reads.wait_all(&one).expect("the first task's reads");
        });
        assert_eq!(first[..total], want, "the first task");
        assert_eq!(second[..total], want, "the second task");
    }

    #[test]
    fn a_repeated_entry_is_read_once_and_copied_to_every_row_that_wants_it() {
        use std::io::Write as _;
        let mut file = tempfile::tempfile().expect("a temporary file");
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 13 + 5) as u8).collect();
        file.write_all(&bytes).expect("write");
        // Each slot reads one entry into two rows: the second row is a copy, not a read.
        let mut per_slot = vec![
            vec![
                (0usize, &file, 0u64, 0usize, 256usize),
                (0, &file, 0, 1024, 256),
            ],
            vec![(1, &file, 1024, 2048, 256), (1, &file, 1024, 3072, 256)],
        ];
        let mut dups = Vec::new();
        dedupe_reads(&mut per_slot, &mut dups);
        assert_eq!(
            per_slot.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1],
            "one read a slot"
        );
        assert_eq!(
            dups,
            vec![(0, 1024, 0, 256), (1, 3072, 2048, 256)],
            "the second row copies the first"
        );
        let reader = reader(None);
        let (stride, cap, slots) = (1024, 2, 2);
        let gather = EngramGather {
            reader: &reader,
            n: 2,
            stride,
            cap,
            row0: 0,
            at: 0,
            skips: vec![false, false],
            jobs: per_slot.into_iter().flatten().collect(),
            counts: vec![1, 1],
            cached: true,
            shard: None,
            dups,
        };
        let mut staged = vec![0u8; slots * cap * stride];
        gather
            .read_all(&Staging(staged.as_mut_ptr(), staged.len()), 2)
            .expect("the reads and their copies");
        for slot in 0..slots {
            let entry = slot * cap * stride;
            for row in 0..2 {
                let at = entry + row * stride;
                assert_eq!(
                    staged[at..at + 256],
                    bytes[slot * 1024..slot * 1024 + 256],
                    "slot {slot} row {row}"
                );
            }
        }
    }

    #[test]
    fn sharded_rows_permute_into_the_unsharded_rows() {
        let layout =
            crate::engram_device_state::EngramArenaLayout::new(24, 256).expect("the layout");
        let (head, scale) = (layout.head_dim(), layout.scale_width());
        let rows = 3;
        // Row y's column j: its code bytes and its scale bytes.
        let code = |y: usize, j: usize, k: usize| (y * 31 + j * 7 + k) as u8;
        let scale_byte = |y: usize, j: usize, k: usize| (200 + y * 3 + j * 5 + k) as u8;
        // Stages every row as `shard` does, at row y of [rows][pitch].
        let stage = |shard: Option<EngramShard>| {
            let (first, count, pitch, scales_at) = staged_row(&layout, shard);
            let mut staged = vec![0u8; rows * pitch];
            for y in 0..rows {
                for c in 0..count {
                    let (row, j) = (y * pitch, first + c);
                    for k in 0..head {
                        staged[row + c * head + k] = code(y, j, k);
                    }
                    for k in 0..scale {
                        staged[row + scales_at + c * scale + k] = scale_byte(y, j, k);
                    }
                }
            }
            staged
        };
        let want = stage(None);
        let world = 4;
        let shards: Vec<EngramShard> = (0..world)
            .map(|rank| EngramShard::of(&layout, world, rank).expect("a share"))
            .collect();
        // The exchanged shares, [world][rows][pitch], permuted as the device does.
        let shares: Vec<u8> = shards
            .iter()
            .flat_map(|&shard| stage(Some(shard)))
            .collect();
        let (pitch, stride) = (shards[0].pitch, layout.stride());
        let mut got = vec![0u8; rows * stride];
        for (from, to, len) in shards[0].moves() {
            assert_eq!((from % 16, to % 16, len % 16), (0, 0, 0));
            for rank in 0..world {
                for y in 0..rows {
                    let src = (rank * rows + y) * pitch + from;
                    let dst = y * stride + to + rank * len;
                    got[dst..dst + len].copy_from_slice(&shares[src..src + len]);
                }
            }
        }
        // Every byte an unsharded gather stages (the alignment padding is never read).
        let (_, _, _, scales_at) = staged_row(&layout, None);
        for y in 0..rows {
            let row = y * stride;
            let codes = row..row + layout.codes_len();
            let scales = row + scales_at..row + scales_at + layout.scales_len();
            assert_eq!(got[codes.clone()], want[codes], "row {y}'s codes");
            assert_eq!(got[scales.clone()], want[scales], "row {y}'s scales");
        }
        assert!(EngramShard::of(&layout, 5, 0).is_err());
        assert!(EngramShard::of(&layout, 4, 4).is_err());
        assert!(EngramShard::of(&layout, 0, 0).is_err());
    }

    #[test]
    fn merged_gathers_keep_each_slot_together_in_row_order() {
        let reader = reader(None);
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
            shard: None,
            dups: Vec::new(),
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
