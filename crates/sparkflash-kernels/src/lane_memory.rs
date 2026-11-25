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

//! Sequence state that physical memory backs as it grows: the lanes several served sequences
//! keep side by side.
//!
//! A lane is one sequence's position-addressed state laid out in a span of reserved virtual
//! memory ([LaneShape]: every part's offset, length and bytes a row). Physical memory backs a span
//! lazily, in granularity chunks (2 MiB on GB10), over the prefix of each part a sequence of that
//! many positions touches ([LaneSpan::ensure]); the chunks come from, and go back to, one
//! [ChunkPool] whose budget is a fixed number of chunks. The budget is a number, not the device's
//! free memory, so every rank of a group that makes the same requests makes the same refusals.
//!
//! A part that is a window ring (read only over its last window of rows) may give its old rows
//! back as the sequence grows ([LaneSpan::slide]).
//!
//! Every chunk is zeroed when it is mapped, so a lane's memory reads as a fresh allocation's did.

use crate::cuda_driver::{Cuda, Deviceptr};

/// The device operations a span needs: [Cuda]'s virtual memory calls, and a stand-in in tests.
pub trait ChunkDevice {
    /// Creates `bytes` of physical memory, mapped nowhere.
    ///
    /// # Errors
    /// The driver's failure (out of memory, say).
    fn create(&self, bytes: usize) -> Result<u64, String>;
    /// Releases a handle [Self::create] made.
    fn release(&self, handle: u64);
    /// Maps `handle` at `at`.
    ///
    /// # Errors
    /// The driver's failure.
    fn map(&self, at: u64, bytes: usize, handle: u64) -> Result<(), String>;
    /// Unmaps `bytes` at `at`. Nothing queued may still read it.
    ///
    /// # Errors
    /// The driver's failure.
    fn unmap(&self, at: u64, bytes: usize) -> Result<(), String>;
    /// Queues zeroing `bytes` at `at`.
    ///
    /// # Errors
    /// The driver's failure.
    fn zero(&self, at: u64, bytes: usize) -> Result<(), String>;
    /// Queues a copy of `bytes` from `src` to `dst`.
    ///
    /// # Errors
    /// The driver's failure.
    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), String>;
}

impl ChunkDevice for Cuda {
    fn create(&self, bytes: usize) -> Result<u64, String> {
        self.vmm_create(bytes)
    }

    fn release(&self, handle: u64) {
        self.vmm_release(handle);
    }

    fn map(&self, at: u64, bytes: usize, handle: u64) -> Result<(), String> {
        self.vmm_map(Deviceptr(at), bytes, handle)
    }

    fn unmap(&self, at: u64, bytes: usize) -> Result<(), String> {
        self.vmm_unmap(Deviceptr(at), bytes)
    }

    fn zero(&self, at: u64, bytes: usize) -> Result<(), String> {
        self.memset_d8_async(Deviceptr(at), 0, bytes)
    }

    fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
        self.copy_dtod_async(&Deviceptr(dst), &Deviceptr(src), bytes)
    }
}

/// One part of a lane's layout: `len` bytes at `offset`, of which a sequence of `rows` positions
/// touches the first `max(rows, min_rows) * row_bytes` (all of it when `row_bytes` is 0). A window
/// ring `slides`: below the lane's floor its rows are not read again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Where the part starts, from the span's base.
    pub offset: usize,
    /// The part's bytes.
    pub len: usize,
    /// The bytes one position adds, or 0 for a part a sequence touches whole.
    pub row_bytes: usize,
    /// The rows a sequence touches however short it is.
    pub min_rows: usize,
    /// Whether the part is a window ring that gives its old rows back.
    pub slides: bool,
    /// Whether the part is a recurrent state: the state after the sequence's last position, not
    /// a row a position, so a prefix of it means nothing and [LaneSpan::copy_prefix] leaves it to
    /// the arm (which copies a state saved at the prefix's end, if it holds one).
    pub state: bool,
}

impl Region {
    /// The bytes from `offset` a sequence of `rows` positions touches.
    #[must_use]
    pub fn touched(&self, rows: usize) -> usize {
        if self.row_bytes == 0 {
            self.len
        } else {
            self.len.min(rows.max(self.min_rows) * self.row_bytes)
        }
    }

    /// The bytes `[from, to)` from `offset` a sequence of `rows` positions still reads when its
    /// window rings start at row `floor`.
    #[must_use]
    pub fn live(&self, rows: usize, floor: usize) -> (usize, usize) {
        let to = self.touched(rows);
        let from = if self.slides {
            (floor * self.row_bytes).min(to)
        } else {
            0
        };
        (from, to)
    }
}

/// A lane layout as it is built: the next free offset and the regions so far.
#[derive(Debug, Default)]
pub struct Layout {
    cursor: usize,
    regions: Vec<Region>,
}

impl Layout {
    /// The offset of `len` fresh bytes (256-byte aligned, the slab's alignment).
    pub fn reserve(&mut self, len: usize) -> usize {
        let offset = round_up(self.cursor, 256);
        self.cursor = offset + len;
        offset
    }

    /// A region over `len` bytes at `offset`.
    pub fn region(&mut self, offset: usize, len: usize, row_bytes: usize, min_rows: usize) {
        self.regions.push(Region {
            offset,
            len,
            row_bytes,
            min_rows,
            slides: false,
            state: false,
        });
    }

    /// A window ring's region: it slides.
    pub fn window(&mut self, offset: usize, len: usize, row_bytes: usize) {
        self.region(offset, len, row_bytes, 0);
        if let Some(region) = self.regions.last_mut() {
            region.slides = true;
        }
    }

    /// [Self::reserve] and one recurrent state's region over it, `len` bytes a sequence touches
    /// whole ([Region::state]).
    pub fn state(&mut self, len: usize) -> usize {
        let offset = self.part(len, 0, 0);
        if let Some(region) = self.regions.last_mut() {
            region.state = true;
        }
        offset
    }

    /// [Self::reserve] and one region over it.
    pub fn part(&mut self, len: usize, row_bytes: usize, min_rows: usize) -> usize {
        let offset = self.reserve(len);
        self.region(offset, len, row_bytes, min_rows);
        offset
    }

    /// The finished shape: a span as long as the layout, in whole chunks of `granularity`, backed
    /// in steps of `quantum` rows.
    #[must_use]
    pub fn finish(self, granularity: usize, quantum: usize) -> LaneShape {
        let granularity = granularity.max(1);
        LaneShape {
            stride: round_up(self.cursor.max(1), granularity),
            granularity,
            quantum: quantum.max(1),
            regions: self.regions,
        }
    }
}

/// Every lane's layout: its parts, its span's length and the chunk size.
#[derive(Clone, Debug)]
pub struct LaneShape {
    /// A span's bytes, a whole number of chunks.
    pub stride: usize,
    /// The chunk size.
    pub granularity: usize,
    /// [LaneSpan::ensure] backs rows in steps of this many, so a decode step rarely maps anything.
    pub quantum: usize,
    /// The parts.
    pub regions: Vec<Region>,
}

impl LaneShape {
    /// A span's chunks.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.stride / self.granularity
    }

    /// `rows` rounded up to the backing step.
    #[must_use]
    pub fn quantized(&self, rows: usize) -> usize {
        rows.div_ceil(self.quantum) * self.quantum
    }

    /// The chunk indices `[first, end)` holding `region`'s live bytes at `rows` rows and window
    /// floor `floor`.
    fn chunk_span(&self, region: &Region, rows: usize, floor: usize) -> (usize, usize) {
        let (from, to) = region.live(rows, floor);
        if from == to {
            return (0, 0);
        }
        (
            (region.offset + from) / self.granularity,
            (region.offset + to).div_ceil(self.granularity),
        )
    }

    /// Which chunks hold bytes a sequence of `rows` rows reads with its window rings from row
    /// `floor`.
    #[must_use]
    pub fn wanted(&self, rows: usize, floor: usize) -> Vec<bool> {
        let mut wanted = vec![false; self.chunks()];
        for region in &self.regions {
            let (first, end) = self.chunk_span(region, rows, floor);
            wanted[first..end].fill(true);
        }
        wanted
    }

    /// The chunks one sequence of `rows` positions (rounded to the step) holds with its window
    /// rings from row `floor`.
    #[must_use]
    pub fn chunks_for(&self, rows: usize, floor: usize) -> usize {
        self.wanted(self.quantized(rows), floor)
            .iter()
            .filter(|&&want| want)
            .count()
    }
}

/// The physical chunks every lane of a pool may hold together.
#[derive(Debug)]
pub struct ChunkPool {
    granularity: usize,
    /// Chunks created and unmapped, ready for the next lane that grows.
    free: Vec<u64>,
    created: usize,
    budget: usize,
}

impl ChunkPool {
    /// A pool of at most `budget_bytes` (rounded down to whole chunks of `granularity`).
    #[must_use]
    pub fn new(granularity: usize, budget_bytes: usize) -> Self {
        let granularity = granularity.max(1);
        Self {
            granularity,
            free: Vec::new(),
            created: 0,
            budget: budget_bytes / granularity,
        }
    }

    /// The chunk size.
    #[must_use]
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// The chunks the pool may hold.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Chunks not yet promised to any lane: the free ones and those the budget can still create.
    #[must_use]
    pub fn spare(&self) -> usize {
        self.free.len() + self.budget.saturating_sub(self.created)
    }

    /// The chunks held by lanes now.
    #[must_use]
    pub fn mapped(&self) -> usize {
        self.created - self.free.len()
    }

    fn take(&mut self, device: &impl ChunkDevice) -> Result<u64, String> {
        if let Some(handle) = self.free.pop() {
            return Ok(handle);
        }
        let handle = device.create(self.granularity)?;
        self.created += 1;
        Ok(handle)
    }

    fn give(&mut self, handle: u64) {
        self.free.push(handle);
    }

    /// Releases every free chunk; lanes still holding chunks keep them.
    pub fn drain(&mut self, device: &impl ChunkDevice) {
        self.created -= self.free.len();
        for handle in self.free.drain(..) {
            device.release(handle);
        }
    }
}

/// One lane's span: where it starts and the chunk behind each chunk of it.
#[derive(Debug)]
pub struct LaneSpan {
    base: u64,
    /// The physical chunk behind each chunk of the span (0: none).
    chunks: Vec<u64>,
    /// The rows [Self::ensure] has backed.
    backed: usize,
    /// The first row its window rings still back ([Self::slide]).
    floor: usize,
}

impl LaneSpan {
    /// A span of `shape` at `base`, a reserved range of `shape.stride` bytes, backing nothing.
    #[must_use]
    pub fn new(base: u64, shape: &LaneShape) -> Self {
        Self {
            base,
            chunks: vec![0; shape.chunks()],
            backed: 0,
            floor: 0,
        }
    }

    /// Where the span starts.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// The rows backed.
    #[must_use]
    pub fn backed(&self) -> usize {
        self.backed
    }

    /// The first row the window rings still back.
    #[must_use]
    pub fn floor(&self) -> usize {
        self.floor
    }

    /// The chunks mapped.
    #[must_use]
    pub fn mapped(&self) -> usize {
        self.chunks.iter().filter(|&&handle| handle != 0).count()
    }

    fn at(&self, shape: &LaneShape, chunk: usize) -> u64 {
        self.base + (chunk * shape.granularity) as u64
    }

    /// The chunks this span would have to map to back `rows` rows.
    #[must_use]
    pub fn chunks_needed(&self, shape: &LaneShape, rows: usize) -> usize {
        shape
            .wanted(shape.quantized(rows), self.floor)
            .iter()
            .zip(&self.chunks)
            .filter(|&(&want, &mapped)| want && mapped == 0)
            .count()
    }

    /// Whether `pool` can back `rows` rows here now.
    #[must_use]
    pub fn fits(&self, shape: &LaneShape, pool: &ChunkPool, rows: usize) -> bool {
        rows <= self.backed || self.chunks_needed(shape, rows) <= pool.spare()
    }

    /// Backs the span for sequences of up to `rows` positions. Returns false, mapping nothing
    /// more, when the pool has too few chunks left (the caller frees a lane and retries).
    ///
    /// # Errors
    /// A driver failure creating, mapping or zeroing a chunk.
    pub fn ensure(
        &mut self,
        device: &impl ChunkDevice,
        shape: &LaneShape,
        pool: &mut ChunkPool,
        rows: usize,
    ) -> Result<bool, String> {
        if rows <= self.backed {
            return Ok(true);
        }
        let rows = shape.quantized(rows);
        if self.chunks_needed(shape, rows) > pool.spare() {
            return Ok(false);
        }
        let wanted = shape.wanted(rows, self.floor);
        for (chunk, want) in wanted.into_iter().enumerate() {
            if !want || self.chunks[chunk] != 0 {
                continue;
            }
            let handle = pool.take(device)?;
            let at = self.at(shape, chunk);
            if let Err(error) = device.map(at, shape.granularity, handle) {
                pool.give(handle);
                return Err(format!("lane memory: chunk {chunk}: {error}"));
            }
            self.chunks[chunk] = handle;
            // A chunk may come from another lane (or another process's freed memory): every part
            // starts zeroed, as a fresh allocation did.
            device.zero(at, shape.granularity)?;
        }
        self.backed = rows;
        Ok(true)
    }

    /// The chunks this span would give back if its window rings started at row `floor`: those it
    /// maps that no part reads from there on.
    #[must_use]
    pub fn slid_chunks(&self, shape: &LaneShape, floor: usize) -> Vec<usize> {
        if floor <= self.floor {
            return Vec::new();
        }
        let wanted = shape.wanted(self.backed, floor);
        (0..self.chunks.len())
            .filter(|&chunk| self.chunks[chunk] != 0 && !wanted[chunk])
            .collect()
    }

    /// Moves the window floor to row `floor`, returning [Self::slid_chunks] to the pool. Nothing
    /// queued may still read rows below `floor`.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub fn slide(
        &mut self,
        device: &impl ChunkDevice,
        shape: &LaneShape,
        pool: &mut ChunkPool,
        floor: usize,
    ) -> Result<usize, String> {
        let slid = self.slid_chunks(shape, floor);
        for &chunk in &slid {
            let handle = std::mem::take(&mut self.chunks[chunk]);
            device.unmap(self.at(shape, chunk), shape.granularity)?;
            pool.give(handle);
        }
        self.floor = self.floor.max(floor);
        Ok(slid.len())
    }

    /// Copies every part `src` holds for positions `..rows` into this span (the window rings from
    /// row `floor`, at or above both spans' floors; no recurrent state, [Region::state]), backing
    /// it that far first. Returns false, copying nothing, when the pool cannot back it.
    ///
    /// # Errors
    /// A driver failure backing this span or queueing a copy.
    pub fn copy_prefix(
        &mut self,
        device: &impl ChunkDevice,
        shape: &LaneShape,
        pool: &mut ChunkPool,
        src: &Self,
        rows: usize,
        floor: usize,
    ) -> Result<bool, String> {
        if !self.ensure(device, shape, pool, rows)? {
            return Ok(false);
        }
        for region in shape.regions.iter().filter(|region| !region.state) {
            let (from, to) = region.live(rows, floor);
            if from < to {
                let at = (region.offset + from) as u64;
                device.copy(self.base + at, src.base + at, to - from)?;
            }
        }
        Ok(true)
    }

    /// Zeroes every backed byte: the span reads as fresh again and keeps its chunks.
    ///
    /// # Errors
    /// A driver failure queueing a memset.
    pub fn clear(&self, device: &impl ChunkDevice, shape: &LaneShape) -> Result<(), String> {
        let mut chunk = 0;
        while chunk < self.chunks.len() {
            if self.chunks[chunk] == 0 {
                chunk += 1;
                continue;
            }
            let first = chunk;
            while chunk < self.chunks.len() && self.chunks[chunk] != 0 {
                chunk += 1;
            }
            device.zero(self.at(shape, first), (chunk - first) * shape.granularity)?;
        }
        Ok(())
    }

    /// Returns every chunk to the pool. Nothing queued may still read the span.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub fn release(
        &mut self,
        device: &impl ChunkDevice,
        shape: &LaneShape,
        pool: &mut ChunkPool,
    ) -> Result<(), String> {
        for chunk in 0..self.chunks.len() {
            let handle = std::mem::take(&mut self.chunks[chunk]);
            if handle == 0 {
                continue;
            }
            device.unmap(self.at(shape, chunk), shape.granularity)?;
            pool.give(handle);
        }
        self.backed = 0;
        self.floor = 0;
        Ok(())
    }
}

/// The lanes a default [LanePool::budget] can always start together: the most any arm steps
/// (Dory's recurrence group drafts up to 16).
pub const FLOOR_LANES: usize = 16;

/// One rank's lanes: the [LaneShape] every lane of the rank shares and the [ChunkPool] their
/// chunks come from, which the rank's sequences share under a lock. An arm lays its sequence
/// state out once ([Layout]), each sequence reserves one span of [LaneShape::stride] bytes and
/// keeps a [LaneSpan] over it, and this backs, refuses and releases the spans' rows.
#[derive(Debug)]
pub struct LanePool {
    shape: LaneShape,
    pool: std::sync::Mutex<ChunkPool>,
}

impl LanePool {
    /// A pool of `budget_bytes` ([Self::budget]) behind lanes of `shape`.
    #[must_use]
    pub fn new(shape: LaneShape, budget_bytes: usize) -> Self {
        let pool = ChunkPool::new(shape.granularity, budget_bytes);
        Self {
            shape,
            pool: std::sync::Mutex::new(pool),
        }
    }

    /// The bytes a pool of lanes of `shape` holds: `configured_mb` MiB when the profile sets it
    /// (`runtime.lane_memory_mb`, SPARKFLASH_LANE_MEMORY_MB), else what one sequence of `rows`
    /// positions holds, so several sequences share what one whole context would. Either way it
    /// is a number, the same on every rank of a group, so every rank refuses the same growth.
    ///
    /// The default is never less than [FLOOR_LANES] lanes' first backing step. Every part of a
    /// lane takes whole chunks, so at a short context one sequence's worth may be a single chunk
    /// and a second lane could not start at all. The budget only caps what the lanes map, and a
    /// sequence never maps more than its own span, so the floor costs nothing until several
    /// lanes grow at once.
    #[must_use]
    pub fn budget(shape: &LaneShape, rows: usize, configured_mb: Option<usize>) -> usize {
        configured_mb.map_or_else(
            || {
                let whole = shape.chunks_for(rows, 0);
                let starts = FLOOR_LANES * shape.chunks_for(1, 0);
                whole.max(starts) * shape.granularity
            },
            |mb| mb << 20,
        )
    }

    /// The lanes' layout.
    #[must_use]
    pub fn shape(&self) -> &LaneShape {
        &self.shape
    }

    /// The pool, locked.
    pub fn pool(&self) -> std::sync::MutexGuard<'_, ChunkPool> {
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Backs `span` for a sequence of `rows` positions.
    ///
    /// # Errors
    /// [Self::short]'s refusal when the pool has too few chunks left, or a driver failure.
    pub fn grow(
        &self,
        device: &impl ChunkDevice,
        span: &mut LaneSpan,
        rows: usize,
    ) -> Result<(), String> {
        let mut pool = self.pool();
        if span.ensure(device, &self.shape, &mut pool, rows)? {
            return Ok(());
        }
        Err(self.short(span, &pool, rows))
    }

    /// Whether the pool can back `rows` positions in `span` now, counting the chunks a reset
    /// gives back first when `reset`.
    #[must_use]
    pub fn room(&self, span: &LaneSpan, rows: usize, reset: bool) -> bool {
        let pool = self.pool();
        if reset {
            self.shape.chunks_for(rows, 0) <= pool.spare() + span.mapped()
        } else {
            span.fits(&self.shape, &pool, rows)
        }
    }

    /// The refusal of a span that needs more chunks for `rows` positions than `pool` has spare.
    #[must_use]
    pub fn short(&self, span: &LaneSpan, pool: &ChunkPool, rows: usize) -> String {
        let chunk = self.shape.granularity;
        format!(
            "lane memory: {rows} positions need {} MiB more and the sequences' pool has {} MiB of \
             {} MiB spare (runtime.lane_memory_mb / SPARKFLASH_LANE_MEMORY_MB sets it)",
            (span.chunks_needed(&self.shape, rows) * chunk) >> 20,
            (pool.spare() * chunk) >> 20,
            (pool.budget() * chunk) >> 20
        )
    }

    /// Returns every chunk `span` holds to the pool. Nothing queued may still read the span.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub fn release(&self, device: &impl ChunkDevice, span: &mut LaneSpan) -> Result<(), String> {
        if span.mapped() == 0 {
            return Ok(());
        }
        let mut pool = self.pool();
        span.release(device, &self.shape, &mut pool)
    }

    /// One line for the boot log: a lane's span, the pool and the chunk size.
    #[must_use]
    pub fn describe(&self) -> String {
        let chunk = self.shape.granularity;
        format!(
            "{} MiB a lane at most, a {} MiB pool, {} KiB chunks",
            self.shape.stride >> 20,
            (self.pool().budget() * chunk) >> 20,
            chunk >> 10
        )
    }
}

/// The rows a sequence's lane backs at a time ([LaneShape::quantum]), so a decode step rarely maps
/// anything.
pub const LANE_QUANTUM: usize = 2048;

/// Why a lane could not be backed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackError {
    /// The pool has too few chunks left ([LanePool::short]), or the driver would not map one:
    /// the request's refusal.
    Refused(String),
    /// A device failure around the backing (making the context current, waiting for the zeroes).
    Device(String),
}

impl std::fmt::Display for BackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(error) | Self::Device(error) => f.write_str(error),
        }
    }
}

/// Every sequence of one kind on a device: the [LanePool] behind their lanes, the arm's parts
/// (where each cache lies in a span, as offsets from its base) and the rows a sequence backs past
/// its last position.
///
/// This is what an arm's sequences that keep position-addressed caches need of lane memory. The
/// arm lays its caches out once ([SequenceLanes::plan]), each sequence reserves a span
/// ([SequenceLane::reserve]) and points its caches at `base + offset`, and before a forward
/// writes rows the sequence backs them ([SequenceLane::back]). The chunks come from one pool the
/// sequences share, so several hold together what one whole context would: the budget is one
/// full sequence's worth unless `runtime.lane_memory_mb` (SPARKFLASH_LANE_MEMORY_MB) says
/// otherwise. It is a number, the same on every rank, so every rank of a group refuses the same
/// growth, and the serving engine asks [SequenceLane::room] before it hands a slot a job.
#[derive(Debug)]
pub struct SequenceLanes<P> {
    pool: LanePool,
    capacity: usize,
    read_past: usize,
    parts: P,
}

impl<P> SequenceLanes<P> {
    /// Lanes for sequences of `capacity` positions: `parts` lays every cache out over
    /// `capacity` rows in the [Layout] it is handed and returns where each one lies, and a
    /// sequence backs `read_past` rows past its last position too (the rows a kernel's tile
    /// reads past it, or a drafter's drafts write). The pool holds `configured_mb` MiB when that
    /// is set, else one whole sequence's worth ([LanePool::budget]).
    ///
    /// # Errors
    /// A device without virtual memory management.
    pub fn plan(
        cuda: &Cuda,
        capacity: usize,
        read_past: usize,
        configured_mb: Option<usize>,
        parts: impl FnOnce(&mut Layout) -> P,
    ) -> Result<Self, String> {
        let granularity = cuda.vmm_granularity().ok_or_else(|| {
            "the device has no virtual memory management for the sequences' caches".to_owned()
        })?;
        let mut layout = Layout::default();
        let parts = parts(&mut layout);
        let shape = layout.finish(granularity, LANE_QUANTUM);
        let budget = LanePool::budget(&shape, capacity + read_past, configured_mb);
        Ok(Self {
            pool: LanePool::new(shape, budget),
            capacity,
            read_past,
            parts,
        })
    }

    /// Where each cache lies in a lane.
    #[must_use]
    pub fn parts(&self) -> &P {
        &self.parts
    }

    /// The pool behind the lanes.
    #[must_use]
    pub fn pool(&self) -> &LanePool {
        &self.pool
    }

    /// The positions a sequence holds at most.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The rows a lane backs for a sequence of `rows` positions.
    fn rows(&self, rows: usize) -> usize {
        rows.min(self.capacity) + self.read_past
    }

    /// [LanePool::describe], for the boot log.
    #[must_use]
    pub fn describe(&self) -> String {
        self.pool.describe()
    }

    /// Gives the pool's chunks back to the device. Every lane has released its own by now: each
    /// holds these lanes.
    pub fn drain(&self, cuda: &Cuda) {
        let _ = cuda.make_current();
        self.pool.pool().drain(cuda);
    }
}

/// One sequence's span of its [SequenceLanes].
///
/// The arm frees it ([SequenceLane::free]) when the sequence goes, on the device that reserved
/// it; a span dropped without that keeps its chunks out of the pool for good.
#[derive(Debug)]
pub struct SequenceLane<P> {
    lanes: std::sync::Arc<SequenceLanes<P>>,
    span: LaneSpan,
    freed: bool,
}

impl<P> SequenceLane<P> {
    /// Reserves a span for one sequence of `lanes`; nothing backs it yet.
    ///
    /// # Errors
    /// The driver's refusal of the reservation.
    pub fn reserve(cuda: &Cuda, lanes: &std::sync::Arc<SequenceLanes<P>>) -> Result<Self, String> {
        let shape = lanes.pool.shape();
        let base = cuda.vmm_reserve(shape.stride)?;
        Ok(Self {
            span: LaneSpan::new(base.0, shape),
            lanes: std::sync::Arc::clone(lanes),
            freed: false,
        })
    }

    /// The span's first byte: a cache at offset `o` of the parts lies at `base().offset(o)`.
    #[must_use]
    pub fn base(&self) -> Deviceptr {
        Deviceptr(self.span.base())
    }

    /// The lanes this span is one of.
    #[must_use]
    pub fn lanes(&self) -> &std::sync::Arc<SequenceLanes<P>> {
        &self.lanes
    }

    /// The span.
    #[must_use]
    pub fn span(&self) -> &LaneSpan {
        &self.span
    }

    /// Backs the lane for a sequence of `rows` positions and the rows past them the kernels
    /// read or write. The new chunks are zeroed on the default stream, so this waits for the
    /// device before it returns when it mapped any: a run stream does not wait for the default
    /// one.
    ///
    /// # Errors
    /// [BackError::Refused] when the pool has too few chunks left, [BackError::Device] for a
    /// device failure.
    pub fn back(&mut self, cuda: &Cuda, rows: usize) -> Result<(), BackError> {
        let rows = self.lanes.rows(rows);
        if rows <= self.span.backed() {
            return Ok(());
        }
        // Zeroing the new chunks needs the context on the calling thread, which an engine
        // thread may not have made current before a prompt.
        cuda.make_current().map_err(BackError::Device)?;
        self.lanes
            .pool
            .grow(cuda, &mut self.span, rows)
            .map_err(BackError::Refused)?;
        cuda.synchronize().map_err(BackError::Device)
    }

    /// Whether the pool can back this lane for `rows` positions now, counting the chunks a
    /// reset gives back first when `reset`: the answer [SequenceLane::back] would give.
    #[must_use]
    pub fn room(&self, rows: usize, reset: bool) -> bool {
        self.lanes
            .pool
            .room(&self.span, self.lanes.rows(rows), reset)
    }

    /// Gives every chunk back to the pool, once the device is done with them; the rows come
    /// back zeroed as the sequence grows again.
    ///
    /// # Errors
    /// A driver failure.
    pub fn release(&mut self, cuda: &Cuda) -> Result<(), String> {
        if self.span.mapped() == 0 {
            return Ok(());
        }
        cuda.synchronize()?;
        self.lanes.pool.release(cuda, &mut self.span)
    }

    /// Gives the lane back: its chunks to the pool and its span to the device. A second call
    /// does nothing.
    pub fn free(&mut self, cuda: &Cuda) {
        if self.freed {
            return;
        }
        self.freed = true;
        // A sequence can go from a thread that has not touched the device yet.
        let _ = cuda.make_current();
        let _ = self.release(cuda);
        cuda.vmm_free(self.base(), self.lanes.pool.shape().stride);
        // The last lane of lanes whose arm has gone ([LanePlans::drain] drained the rest)
        // gives the chunks back to the device.
        if std::sync::Arc::strong_count(&self.lanes) == 1 {
            self.lanes.drain(cuda);
        }
    }
}

/// An arm's [SequenceLanes], one for each kind of sequence (`K`: a capacity, whether it drafts),
/// planned on first use.
#[derive(Debug)]
pub struct LanePlans<K, P> {
    planned: std::sync::Mutex<Vec<(K, std::sync::Arc<SequenceLanes<P>>)>>,
}

impl<K, P> Default for LanePlans<K, P> {
    fn default() -> Self {
        Self {
            planned: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl<K: PartialEq, P> LanePlans<K, P> {
    /// The lanes of sequences of `key`, planned by `plan` the first time; `planned` is told of
    /// a new plan (for the boot log).
    ///
    /// # Errors
    /// `plan`'s.
    pub fn get<E>(
        &self,
        key: K,
        plan: impl FnOnce() -> Result<SequenceLanes<P>, E>,
        planned: impl FnOnce(&SequenceLanes<P>),
    ) -> Result<std::sync::Arc<SequenceLanes<P>>, E> {
        let mut plans = self
            .planned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, lanes)) = plans.iter().find(|(planned, _)| *planned == key) {
            return Ok(std::sync::Arc::clone(lanes));
        }
        let lanes = std::sync::Arc::new(plan()?);
        planned(&lanes);
        plans.push((key, std::sync::Arc::clone(&lanes)));
        Ok(lanes)
    }

    /// Gives every plan's spare chunks back to the device, as the arm goes. A sequence that
    /// outlives the arm gives its own back when it is freed.
    pub fn drain(&self, cuda: &Cuda) {
        let plans = self
            .planned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, lanes) in plans.iter() {
            lanes.drain(cuda);
        }
    }
}

/// Rounds `value` up to a multiple of `to`.
#[must_use]
pub fn round_up(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    /// A device that records what is mapped where and zeroes nothing it need not.
    #[derive(Default)]
    struct Fake {
        next: RefCell<u64>,
        mapped: RefCell<BTreeMap<u64, u64>>,
        zeroed: RefCell<Vec<(u64, usize)>>,
        copies: RefCell<Vec<(u64, u64, usize)>>,
    }

    impl ChunkDevice for Fake {
        fn create(&self, _bytes: usize) -> Result<u64, String> {
            let mut next = self.next.borrow_mut();
            *next += 1;
            Ok(*next)
        }

        fn release(&self, _handle: u64) {}

        fn map(&self, at: u64, _bytes: usize, handle: u64) -> Result<(), String> {
            let old = self.mapped.borrow_mut().insert(at, handle);
            assert!(old.is_none(), "chunk at {at} mapped twice");
            Ok(())
        }

        fn unmap(&self, at: u64, _bytes: usize) -> Result<(), String> {
            self.mapped
                .borrow_mut()
                .remove(&at)
                .map(|_| ())
                .ok_or_else(|| format!("nothing mapped at {at}"))
        }

        fn zero(&self, at: u64, bytes: usize) -> Result<(), String> {
            self.zeroed.borrow_mut().push((at, bytes));
            Ok(())
        }

        fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), String> {
            self.copies.borrow_mut().push((dst, src, bytes));
            Ok(())
        }
    }

    /// Two parts: a 64-byte-a-row store of 4096 rows and a 16-byte-a-row ring of 4096 rows, in
    /// 1 KiB chunks, backed 16 rows at a time.
    fn shape() -> LaneShape {
        let mut layout = Layout::default();
        layout.part(4096 * 64, 64, 0);
        let ring = layout.reserve(4096 * 16);
        layout.window(ring, 4096 * 16, 16);
        layout.finish(1024, 16)
    }

    #[test]
    fn a_span_backs_only_the_rows_a_sequence_touches_and_zeroes_them() {
        let shape = shape();
        let device = Fake::default();
        let mut pool = ChunkPool::new(1024, 1 << 20);
        let mut span = LaneSpan::new(0, &shape);
        assert!(span.ensure(&device, &shape, &mut pool, 20).unwrap());
        // 32 rows (20 rounded to the step): 2 KiB of store, 512 bytes of ring.
        assert_eq!(span.backed(), 32);
        assert_eq!(span.mapped(), 3);
        assert_eq!(device.zeroed.borrow().len(), 3);
        // A step inside the backed rows maps nothing.
        assert!(span.ensure(&device, &shape, &mut pool, 30).unwrap());
        assert_eq!(span.mapped(), 3);
    }

    #[test]
    fn a_pool_refuses_growth_past_its_budget_and_takes_back_a_released_lane() {
        let shape = shape();
        let device = Fake::default();
        let mut pool = ChunkPool::new(1024, 4 * 1024);
        let mut first = LaneSpan::new(0, &shape);
        let mut second = LaneSpan::new(shape.stride as u64, &shape);
        assert!(first.ensure(&device, &shape, &mut pool, 32).unwrap());
        assert_eq!(pool.spare(), 1);
        assert!(!second.fits(&shape, &pool, 32));
        assert!(!second.ensure(&device, &shape, &mut pool, 32).unwrap());
        assert_eq!(second.mapped(), 0, "a refused growth maps nothing");
        first.release(&device, &shape, &mut pool).unwrap();
        assert_eq!(pool.spare(), 4);
        assert!(second.ensure(&device, &shape, &mut pool, 32).unwrap());
        assert_eq!(pool.mapped(), 3);
    }

    #[test]
    fn a_lane_pool_refuses_alike_and_counts_a_reset_as_room() {
        let shape = shape();
        // A whole 4096-row sequence's worth is 320 chunks (256 of store, 64 of ring); at 32
        // rows it would be three, and the floor lets FLOOR_LANES lanes take their first 16 rows
        // (two chunks each).
        assert_eq!(LanePool::budget(&shape, 4096, None), 320 * 1024);
        assert_eq!(LanePool::budget(&shape, 32, None), FLOOR_LANES * 2 * 1024);
        assert_eq!(LanePool::budget(&shape, 32, Some(2)), 2 << 20);
        // A pool of one 32-row sequence's three chunks.
        let lanes = LanePool::new(shape.clone(), 3 * 1024);
        let device = Fake::default();
        let mut first = LaneSpan::new(0, &shape);
        let mut second = LaneSpan::new(shape.stride as u64, &shape);
        lanes.grow(&device, &mut first, 32).unwrap();
        let refusal = lanes.grow(&device, &mut second, 32).unwrap_err();
        assert!(refusal.contains("SPARKFLASH_LANE_MEMORY_MB"), "{refusal}");
        assert_eq!(second.mapped(), 0);
        assert!(!lanes.room(&second, 32, false));
        // A reset of the first gives its chunks back before it grows again.
        assert!(lanes.room(&first, 32, true));
        lanes.release(&device, &mut first).unwrap();
        assert_eq!(first.mapped(), 0);
        lanes.grow(&device, &mut second, 32).unwrap();
        assert!(lanes.describe().contains("KiB chunks"));
    }

    #[test]
    fn a_ring_slides_its_old_rows_back_and_a_copy_reads_both_floors() {
        let shape = shape();
        let device = Fake::default();
        let mut pool = ChunkPool::new(1024, 1 << 20);
        let mut span = LaneSpan::new(0, &shape);
        assert!(span.ensure(&device, &shape, &mut pool, 256).unwrap());
        let before = span.mapped();
        // The ring's first 128 rows are two 1 KiB chunks.
        assert_eq!(span.slide(&device, &shape, &mut pool, 128).unwrap(), 2);
        assert_eq!(span.mapped(), before - 2);
        assert_eq!(span.floor(), 128);
        let mut copy = LaneSpan::new(shape.stride as u64, &shape);
        assert!(
            copy.copy_prefix(&device, &shape, &mut pool, &span, 200, 128)
                .unwrap()
        );
        let copies = device.copies.borrow();
        assert_eq!(copies.len(), 2);
        assert_eq!(copies[0].2, 200 * 64);
        assert_eq!(copies[1].2, (200 - 128) * 16);
    }

    #[test]
    fn a_prefix_copy_leaves_a_recurrent_state_alone() {
        let mut layout = Layout::default();
        layout.part(4096 * 64, 64, 0);
        let state = layout.state(512);
        let shape = layout.finish(1024, 16);
        assert!(shape.regions[1].state && shape.regions[1].row_bytes == 0);
        assert_eq!(state, 4096 * 64);
        let device = Fake::default();
        let mut pool = ChunkPool::new(1024, 1 << 20);
        let mut span = LaneSpan::new(0, &shape);
        assert!(span.ensure(&device, &shape, &mut pool, 16).unwrap());
        let mut copy = LaneSpan::new(shape.stride as u64, &shape);
        assert!(
            copy.copy_prefix(&device, &shape, &mut pool, &span, 16, 0)
                .unwrap()
        );
        // The state is backed whole, and only the positional part's rows are copied.
        assert_eq!(copy.mapped(), span.mapped());
        let copies = device.copies.borrow();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].2, 16 * 64);
    }

    #[test]
    fn clearing_zeroes_each_run_of_mapped_chunks_once() {
        let shape = shape();
        let device = Fake::default();
        let mut pool = ChunkPool::new(1024, 1 << 20);
        let mut span = LaneSpan::new(0, &shape);
        assert!(span.ensure(&device, &shape, &mut pool, 64).unwrap());
        device.zeroed.borrow_mut().clear();
        span.clear(&device, &shape).unwrap();
        // The store's four chunks, then the ring's one.
        assert_eq!(*device.zeroed.borrow(), vec![(0, 4096), (256 * 1024, 1024)]);
    }
}
