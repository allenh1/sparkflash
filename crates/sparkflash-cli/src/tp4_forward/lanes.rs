//! Per-sequence device state for serving several sequences at once: "lanes".
//!
//! A lane is one sequence's position-addressed history: every layer's window ring, the kv-source
//! owners' compressor rings, compressed stores and index keys, and the drafter stages' rings. The
//! lanes are `count` strides of one layout in one virtual reservation, each part laid out exactly
//! as the single sequence's allocation was, so a kernel addresses a lane's state as it addressed
//! the one sequence's: only the base pointers change ([`DeviceLayer::select_lane`]).
//!
//! Physical memory backs a lane lazily, in granularity chunks (2 MiB on GB10), over the prefix of
//! each part a sequence of that many positions touches ([`Lanes::ensure`]); a lane given up returns
//! its chunks to a pool ([`Lanes::release`]). Every stage reads a part only below the rows its
//! position reached (the window ring at or below the position, the store and index keys below the
//! compressed count, the drafter ring below its block), so the unbacked tail is never read.
//!
//! The window rings (every layer's and the drafter stages') are position-addressed but read only
//! the last window of rows, so a long sequence gives their old rows back as it grows
//! ([`Lanes::slide`]): a lane keeps [`SLIDE_KEEP`] rows behind its position, and a 1M-token
//! sequence holds its compressed stores and index keys plus a few chunks of each window ring.

use super::*;

/// The physical memory every lane's chunks may hold together (SPARKFLASH_LANE_MEMORY_MB overrides
/// it). A fixed number, not the device's free memory: every rank runs the same scheduler, so every
/// rank must refuse the same growth. A 250K-token sequence holds ~1.9 GB, so this keeps eight
/// agents' conversations (and the free lanes' cached prefixes) resident, and a serving rank still
/// has ~10 GiB available.
const DEFAULT_BUDGET_BYTES: usize = 24 << 30;
/// [`Lanes::ensure`] backs rows in steps of this many, so a decode step rarely maps anything.
const ROW_QUANTUM: usize = 2048;
/// The window ring rows a lane keeps behind its position: the window (128 rows) and the verify
/// rewinds read far less, and a request resuming the lane's prefix may start this far back.
pub(super) const SLIDE_KEEP: usize = 16384;
/// The rows a lane's floor advances by at a time, so a slide (which waits for the device) is rare.
pub(super) const SLIDE_STEP: usize = 4096;

/// The window floor of a lane at position `pos`: [`SLIDE_KEEP`] rows behind it, in [`SLIDE_STEP`]
/// steps.
pub(super) fn slide_floor(pos: usize) -> usize {
    pos.saturating_sub(SLIDE_KEEP) / SLIDE_STEP * SLIDE_STEP
}

/// One part of the lane layout: `len` bytes at `offset`, of which a sequence of `rows` positions
/// touches the first `max(rows, min_rows) * row_bytes` (all of it when `row_bytes` is 0). A window
/// ring `slides`: below the lane's floor its rows are not read again.
#[derive(Clone, Copy, Debug)]
struct Region {
    offset: usize,
    len: usize,
    row_bytes: usize,
    min_rows: usize,
    slides: bool,
}

impl Region {
    /// The bytes from `offset` a sequence of `rows` positions touches.
    fn touched(&self, rows: usize) -> usize {
        if self.row_bytes == 0 {
            self.len
        } else {
            self.len.min(rows.max(self.min_rows) * self.row_bytes)
        }
    }

    /// The bytes `[from, to)` from `offset` a sequence of `rows` positions still reads when its
    /// window rings start at row `floor`.
    fn live(&self, rows: usize, floor: usize) -> (usize, usize) {
        let to = self.touched(rows);
        let from = if self.slides {
            (floor * self.row_bytes).min(to)
        } else {
            0
        };
        (from, to)
    }
}

/// Where each part of a lane starts, relative to the lane's base.
#[derive(Clone, Copy, Debug)]
struct Offsets {
    kv_ring: usize,
    wkv: usize,
    wgate: usize,
    store_codes: usize,
    store_scales: usize,
    index_keys: usize,
    /// Each drafter stage's (codes, scales) ring.
    draft: [(usize, usize); DRAFT_STAGES],
}

/// The drafter's stages (draft.rs `STAGES`).
pub(super) const DRAFT_STAGES: usize = 3;

/// One lane's device pointers, as the DeviceLayer fields hold them.
#[derive(Clone, Copy, Debug)]
pub(super) struct LanePointers {
    pub(super) kv_ring: u64,
    pub(super) wkv: u64,
    pub(super) wgate: u64,
    pub(super) store_codes: u64,
    pub(super) store_scales: u64,
    pub(super) index_keys: u64,
    pub(super) draft: [(u64, u64); DRAFT_STAGES],
}

/// The host counters of a lane's device state, swapped in and out with its pointers.
#[derive(Clone, Debug, Default)]
pub(super) struct LaneHost {
    pub(super) positions_begun: usize,
    pub(super) appended_rows: Vec<usize>,
}

/// One sequence's rows in a rows step that serves several: step rows `row0..row0 + n` are
/// positions `p0..p0 + n` of lane `lane`. A step's segments are in row order and tile its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Seg {
    pub(super) lane: usize,
    pub(super) p0: usize,
    pub(super) n: usize,
    pub(super) row0: usize,
}

/// The lane of the implicit segment of a single-sequence step: no lane is selected for it.
pub(super) const NO_LANE: usize = usize::MAX;

/// The geometry [`DeviceLayer::attach_lanes`] lays a lane out from.
pub(super) struct LaneGeometry {
    pub(super) count: usize,
    /// The drafter stage rings' rows (DraftState's `ring_rows`), or 0 without a drafter.
    pub(super) draft_ring_rows: usize,
    /// The most rows one step uploads Engram entries for: the Engram arena is indexed by step row
    /// from here on, with this many rows per slot.
    pub(super) engram_rows: usize,
}

/// Every lane's state: the reservation, the layout and the chunk pool.
pub(super) struct Lanes<'a> {
    cuda: &'a Cuda,
    base: u64,
    stride: usize,
    granularity: usize,
    regions: Vec<Region>,
    offsets: Offsets,
    /// Per lane, the physical chunk behind each chunk of its stride (0: none).
    chunks: Vec<Vec<u64>>,
    /// Per lane, the rows [Self::ensure] has backed.
    backed: Vec<usize>,
    /// Per lane, the first row its window rings still back ([Self::slide]).
    floor: Vec<usize>,
    /// Chunks created and unmapped, ready for the next lane that grows.
    free: Vec<u64>,
    created: usize,
    budget: usize,
    pub(super) host: Vec<LaneHost>,
    /// The lane whose pointers and counters the DeviceLayer holds.
    pub(super) selected: usize,
}

impl Lanes<'_> {
    pub(super) fn count(&self) -> usize {
        self.chunks.len()
    }

    /// Lane `lane`'s pointers.
    pub(super) fn pointers(&self, lane: usize) -> LanePointers {
        let base = self.base + (lane * self.stride) as u64;
        let at = |offset: usize| base + offset as u64;
        let o = self.offsets;
        LanePointers {
            kv_ring: at(o.kv_ring),
            wkv: at(o.wkv),
            wgate: at(o.wgate),
            store_codes: at(o.store_codes),
            store_scales: at(o.store_scales),
            index_keys: at(o.index_keys),
            draft: o.draft.map(|(codes, scales)| (at(codes), at(scales))),
        }
    }

    /// The rows lane `lane` has backed.
    pub(super) fn backed(&self, lane: usize) -> usize {
        self.backed.get(lane).copied().unwrap_or(0)
    }

    /// The first row lane `lane`'s window rings still back.
    pub(super) fn floor(&self, lane: usize) -> usize {
        self.floor.get(lane).copied().unwrap_or(0)
    }

    /// Chunks not yet promised to any lane: the pool's free ones and those the budget can still
    /// create.
    pub(super) fn spare_chunks(&self) -> usize {
        self.free.len() + self.budget.saturating_sub(self.created)
    }

    /// The chunks lane `lane` would have to map to back `rows` rows.
    pub(super) fn chunks_needed(&self, lane: usize, rows: usize) -> usize {
        let rows = rows.div_ceil(ROW_QUANTUM) * ROW_QUANTUM;
        self.wanted(lane, rows, self.floor[lane])
            .iter()
            .zip(&self.chunks[lane])
            .filter(|&(&want, &mapped)| want && mapped == 0)
            .count()
    }

    /// Whether one sequence of `rows` positions fits the budget with every other lane empty: its
    /// stores for every row, and its window rings from the floor of row `rows - ahead` (a tick
    /// backs up to `ahead` rows before its lane's floor slides).
    pub(super) fn fits(&self, rows: usize, ahead: usize) -> bool {
        let rows = rows.div_ceil(ROW_QUANTUM) * ROW_QUANTUM;
        let floor = slide_floor(rows.saturating_sub(ahead));
        let wanted = self.wanted(0, rows, floor);
        wanted.iter().filter(|&&want| want).count() <= self.budget
    }

    /// Which of lane `lane`'s chunks hold bytes a sequence of `rows` rows reads with its window
    /// rings from row `floor`.
    fn wanted(&self, lane: usize, rows: usize, floor: usize) -> Vec<bool> {
        let mut wanted = vec![false; self.chunks[lane].len()];
        for region in &self.regions {
            let (first, end) = self.chunk_span(region, rows, floor);
            wanted[first..end].fill(true);
        }
        wanted
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

    /// Backs lane `lane` for sequences of up to `rows` positions. Returns false, mapping nothing
    /// more, when the budget has too few chunks left (the caller frees a lane and retries).
    ///
    /// # Errors
    /// A driver failure creating, mapping or zeroing a chunk.
    pub(super) fn ensure(&mut self, lane: usize, rows: usize) -> Result<bool, String> {
        if lane >= self.count() {
            return Err(format!("lanes: lane {lane} of {}", self.count()));
        }
        if rows <= self.backed[lane] {
            return Ok(true);
        }
        let rows = rows.div_ceil(ROW_QUANTUM) * ROW_QUANTUM;
        if self.chunks_needed(lane, rows) > self.spare_chunks() {
            return Ok(false);
        }
        let lane_base = self.base + (lane * self.stride) as u64;
        let floor = self.floor[lane];
        for index in 0..self.regions.len() {
            let (first, end) = self.chunk_span(&self.regions[index], rows, floor);
            for chunk in first..end {
                if self.chunks[lane][chunk] != 0 {
                    continue;
                }
                let handle = match self.free.pop() {
                    Some(handle) => handle,
                    None => {
                        let handle = self.cuda.vmm_create(self.granularity)?;
                        self.created += 1;
                        handle
                    }
                };
                let at = Deviceptr(lane_base + (chunk * self.granularity) as u64);
                if let Err(error) = self.cuda.vmm_map(at, self.granularity, handle) {
                    self.free.push(handle);
                    return Err(format!("lanes: lane {lane} chunk {chunk}: {error}"));
                }
                self.chunks[lane][chunk] = handle;
                // A chunk may come from another lane (or another process's freed memory): every
                // part starts zeroed, as the single sequence's allocations did.
                self.cuda.memset_d8_async(at, 0, self.granularity)?;
            }
        }
        self.backed[lane] = rows;
        Ok(true)
    }

    /// The chunks lane `lane` would give back if its window rings started at row `floor`: those
    /// it maps that no part reads from there on.
    pub(super) fn slid_chunks(&self, lane: usize, floor: usize) -> Vec<usize> {
        if floor <= self.floor[lane] {
            return Vec::new();
        }
        let wanted = self.wanted(lane, self.backed[lane], floor);
        (0..self.chunks[lane].len())
            .filter(|&chunk| self.chunks[lane][chunk] != 0 && !wanted[chunk])
            .collect()
    }

    /// Moves lane `lane`'s window floor to row `floor`, returning [Self::slid_chunks] to the
    /// pool. Nothing queued may still read rows below `floor`.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub(super) fn slide(&mut self, lane: usize, floor: usize) -> Result<usize, String> {
        let slid = self.slid_chunks(lane, floor);
        let lane_base = self.base + (lane * self.stride) as u64;
        for &chunk in &slid {
            let handle = std::mem::take(&mut self.chunks[lane][chunk]);
            self.cuda.vmm_unmap(
                Deviceptr(lane_base + (chunk * self.granularity) as u64),
                self.granularity,
            )?;
            self.free.push(handle);
        }
        self.floor[lane] = self.floor[lane].max(floor);
        Ok(slid.len())
    }

    /// Copies every part lane `src` holds for positions `..rows` into lane `dst` (the window rings
    /// from row `floor`, at or above both lanes' floors), backing `dst` that far first. Returns
    /// false, copying nothing, when the budget cannot back it.
    ///
    /// # Errors
    /// A driver failure backing `dst` or queueing a copy.
    fn copy_prefix(
        &mut self,
        dst: usize,
        src: usize,
        rows: usize,
        floor: usize,
    ) -> Result<bool, String> {
        if !self.ensure(dst, rows)? {
            return Ok(false);
        }
        let dst_base = self.base + (dst * self.stride) as u64;
        let src_base = self.base + (src * self.stride) as u64;
        for region in &self.regions {
            let (from, to) = region.live(rows, floor);
            if from < to {
                let at = (region.offset + from) as u64;
                self.cuda.copy_dtod_async(
                    &Deviceptr(dst_base + at),
                    &Deviceptr(src_base + at),
                    to - from,
                )?;
            }
        }
        Ok(true)
    }

    /// Returns every chunk of lane `lane` to the pool. The device must be idle: nothing queued may
    /// still read the lane.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub(super) fn release(&mut self, lane: usize) -> Result<(), String> {
        let lane_base = self.base + (lane * self.stride) as u64;
        for chunk in 0..self.chunks[lane].len() {
            let handle = std::mem::take(&mut self.chunks[lane][chunk]);
            if handle == 0 {
                continue;
            }
            self.cuda.vmm_unmap(
                Deviceptr(lane_base + (chunk * self.granularity) as u64),
                self.granularity,
            )?;
            self.free.push(handle);
        }
        self.backed[lane] = 0;
        self.floor[lane] = 0;
        if let Some(host) = self.host.get_mut(lane) {
            host.positions_begun = 0;
            host.appended_rows.iter_mut().for_each(|count| *count = 0);
        }
        Ok(())
    }

    /// The physical bytes the lanes hold and may hold.
    pub(super) fn memory(&self) -> (usize, usize) {
        let mapped: usize = self
            .chunks
            .iter()
            .map(|lane| lane.iter().filter(|&&handle| handle != 0).count())
            .sum();
        (mapped * self.granularity, self.budget * self.granularity)
    }
}

impl Drop for Lanes<'_> {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
        for lane in 0..self.chunks.len() {
            let _ = self.release(lane);
        }
        for handle in self.free.drain(..) {
            self.cuda.vmm_release(handle);
        }
        self.cuda
            .vmm_free(Deviceptr(self.base), self.stride * self.chunks.len());
    }
}

/// The lane layout as it is built: the next free offset and the regions so far.
#[derive(Default)]
struct Layout {
    cursor: usize,
    regions: Vec<Region>,
}

impl Layout {
    /// The offset of `len` fresh bytes (256-byte aligned, the slab's alignment).
    fn reserve(&mut self, len: usize) -> usize {
        let offset = round_up(self.cursor, 256);
        self.cursor = offset + len;
        offset
    }

    fn region(&mut self, offset: usize, len: usize, row_bytes: usize, min_rows: usize) {
        self.regions.push(Region {
            offset,
            len,
            row_bytes,
            min_rows,
            slides: false,
        });
    }

    /// A window ring's region: it slides.
    fn window(&mut self, offset: usize, len: usize, row_bytes: usize) {
        self.region(offset, len, row_bytes, 0);
        if let Some(region) = self.regions.last_mut() {
            region.slides = true;
        }
    }

    /// [Self::reserve] and one region over it.
    fn part(&mut self, len: usize, row_bytes: usize, min_rows: usize) -> usize {
        let offset = self.reserve(len);
        self.region(offset, len, row_bytes, min_rows);
        offset
    }
}

/// Rounds `value` up to a multiple of `to`.
fn round_up(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

impl<'a> DeviceLayer<'a> {
    /// Moves the run's sequence state into `geometry.count` lanes: frees the single sequence's
    /// window ring, its bf16 cache (the rows path never reads it; the M = 1 path refuses from here
    /// on), the compressor rings, stores and index keys, and lays every lane out in one
    /// reservation. Lane 0 is selected and backs nothing yet.
    ///
    /// # Errors
    /// No seam ring, no virtual memory management, or a driver failure.
    pub(super) fn attach_lanes(&mut self, geometry: &LaneGeometry) -> Result<(), String> {
        use sparkflash_kernels::compressed_cache_ref::{CODE_BYTES, SCALE_BYTES};
        use sparkflash_kernels::kv_path_ref::{WINDOW_CODE_BYTES, WINDOW_SCALE_BYTES};
        if self.lanes.is_some() {
            return Err("lanes: already attached".to_owned());
        }
        let count = geometry.count;
        if count == 0 {
            return Err("lanes: zero lanes".to_owned());
        }
        let Some(ring) = self.seam_ring else {
            return Err("lanes: the run has no compressor ring".to_owned());
        };
        let cuda = self.cuda;
        let granularity = cuda
            .vmm_granularity()
            .ok_or("lanes: the device has no virtual memory management")?;
        let mut layout = Layout::default();
        let layers = self.appended_rows.len();
        let window = self.kv_ring;
        // Every layer's ring at the single allocation's offsets: its codes, then its scales.
        let kv_ring = layout.reserve(window.total_bytes(layers)?);
        for layer in 0..layers {
            let at = kv_ring + window.layer_offset(layer);
            layout.window(at, window.codes_bytes(), WINDOW_CODE_BYTES);
            layout.window(
                at + window.scales_offset(0)?,
                window.scales_bytes(),
                WINDOW_SCALE_BYTES,
            );
        }
        let owners = self.kv_source_layers.len().max(1);
        let ring_span = owners * ring.rows * ring.head_dim * 4;
        let wkv = layout.part(ring_span, 0, 0);
        let wgate = layout.part(ring_span, 0, 0);
        let (code_bytes, scale_bytes) = seam_store_owner_bytes(self.max_positions)?;
        let min_rows = sparkflash_kernels::attention_operands::EXPANDED_ROWS;
        // The owners of a store are contiguous (owner o at o * the owner's bytes).
        let mut owner_parts = |bytes: usize, row_bytes: usize, min_rows: usize| {
            let first = layout.reserve(owners * bytes);
            for owner in 0..owners {
                layout.region(first + owner * bytes, bytes, row_bytes, min_rows);
            }
            first
        };
        let store_codes = owner_parts(code_bytes, CODE_BYTES, min_rows);
        let store_scales = owner_parts(scale_bytes, SCALE_BYTES, min_rows);
        let key_bytes = self.max_positions * self.index_head_dim * 4;
        let index_keys = owner_parts(key_bytes.max(4), self.index_head_dim * 4, 0);
        let mut draft = [(0usize, 0usize); DRAFT_STAGES];
        let draft_rows = geometry.draft_ring_rows.max(1);
        let latent = self.head_width as usize;
        for stage in &mut draft {
            stage.0 = layout.reserve(draft_rows * latent);
            layout.window(stage.0, draft_rows * latent, latent);
            stage.1 = layout.reserve(draft_rows * latent / 32);
            layout.window(stage.1, draft_rows * latent / 32, latent / 32);
        }
        let Layout { cursor, regions } = layout;
        let stride = round_up(cursor, granularity);
        let offsets = Offsets {
            kv_ring,
            wkv,
            wgate,
            store_codes,
            store_scales,
            index_keys,
            draft,
        };

        // The single sequence's state goes before the lanes take memory.
        cuda.synchronize()?;
        let mut freed = vec![
            Deviceptr(self.kv_cache.device),
            Deviceptr(self.kv_ring_base),
            ring.wkv,
            ring.wgate,
            ring.store_codes,
            ring.store_scales,
            ring.index_keys,
        ];
        freed.retain(|ptr| ptr.0 != 0);
        cuda.free_buffers(&freed.iter().collect::<Vec<_>>());
        self.kv_cache_off = true;
        self.kv_ring_base = 0;
        if let Some(engram) = self.engram.as_mut() {
            let rows = geometry.engram_rows.max(1);
            let slots = engram.layer_ids.len();
            let bytes = slots * rows * engram.layout.stride();
            cuda.free_buffers(&[&engram.arena]);
            engram.arena = cuda.alloc(bytes.max(4), &mut 0)?;
            cuda.memset_d8_async(engram.arena, 0, bytes.max(4))?;
            engram.arena_bytes = bytes;
            engram.positions = rows;
            engram.skip = vec![true; slots * rows];
            engram.by_row = true;
        }
        let base = cuda.vmm_reserve(stride * count)?.0;
        let budget_bytes = match std::env::var("SPARKFLASH_LANE_MEMORY_MB") {
            Ok(text) => {
                text.parse::<usize>()
                    .map_err(|_| format!("SPARKFLASH_LANE_MEMORY_MB={text} is not a number"))?
                    << 20
            }
            Err(_) => DEFAULT_BUDGET_BYTES,
        };
        let lanes = Lanes {
            cuda,
            base,
            stride,
            granularity,
            regions,
            offsets,
            chunks: vec![vec![0; stride / granularity]; count],
            backed: vec![0; count],
            floor: vec![0; count],
            free: Vec::new(),
            created: 0,
            budget: budget_bytes / granularity,
            host: vec![
                LaneHost {
                    positions_begun: 0,
                    appended_rows: vec![0; layers],
                };
                count
            ],
            selected: usize::MAX,
        };
        eprintln!(
            "lanes: {count} lanes of {} MiB virtual each, {} MiB physical budget, {} KiB chunks",
            stride >> 20,
            budget_bytes >> 20,
            granularity >> 10
        );
        self.lanes = Some(Box::new(lanes));
        self.positions_begun = 0;
        self.appended_rows.iter_mut().for_each(|count| *count = 0);
        self.select_lane(0)
    }

    /// Points the run's sequence state at lane `lane`: its window ring, compressor rings, stores,
    /// index keys and drafter rings, and swaps its position counters in.
    ///
    /// # Errors
    /// No lanes, or a lane outside them.
    pub(super) fn select_lane(&mut self, lane: usize) -> Result<(), String> {
        let lanes = self.lanes.as_mut().ok_or("lanes: none attached")?;
        if lane >= lanes.count() {
            return Err(format!("lanes: lane {lane} of {}", lanes.count()));
        }
        if lanes.selected == lane {
            return Ok(());
        }
        if let Some(host) = lanes.host.get_mut(lanes.selected) {
            host.positions_begun = self.positions_begun;
            std::mem::swap(&mut host.appended_rows, &mut self.appended_rows);
        }
        let host = &mut lanes.host[lane];
        self.positions_begun = host.positions_begun;
        std::mem::swap(&mut host.appended_rows, &mut self.appended_rows);
        lanes.selected = lane;
        let pointers = lanes.pointers(lane);
        self.kv_ring_base = pointers.kv_ring;
        if let Some(ring) = self.seam_ring.as_mut() {
            ring.wkv = Deviceptr(pointers.wkv);
            ring.wgate = Deviceptr(pointers.wgate);
            ring.store_codes = Deviceptr(pointers.store_codes);
            ring.store_scales = Deviceptr(pointers.store_scales);
            ring.index_keys = Deviceptr(pointers.index_keys);
        }
        self.draft_rings = Some(pointers.draft);
        self.attn_window = AttentionWindow::default();
        Ok(())
    }

    /// The step's segments: the ones [Self::set_segments] installed, or the single sequence's
    /// rows p0..p0 + n on the selected state.
    pub(super) fn step_segs(&self, p0: usize, n: usize) -> Vec<Seg> {
        if self.segs.is_empty() {
            vec![Seg {
                lane: NO_LANE,
                p0,
                n,
                row0: 0,
            }]
        } else {
            self.segs.clone()
        }
    }

    /// Refuses a step while the planned lanes are not attached: the single sequence's own state
    /// is then a token allocation that the stages' `max_positions` strides overrun.
    pub(super) fn check_state(&self) -> Result<(), String> {
        if self.lanes.is_none() && self.sequence_rows < self.max_positions {
            return Err(format!(
                "layer device: the single sequence's state holds {} rows of the {}-position run, \
                 for lanes to replace before the first step",
                self.sequence_rows, self.max_positions
            ));
        }
        Ok(())
    }

    /// Selects `seg`'s lane (a single-sequence step's implicit segment selects nothing).
    pub(super) fn enter_seg(&mut self, seg: &Seg) -> Result<(), String> {
        if seg.lane == NO_LANE {
            Ok(())
        } else {
            self.select_lane(seg.lane)
        }
    }

    /// Installs the segments the next rows step runs (empty: the single sequence again). Checks
    /// that they tile rows 0.. in order and that each lane appears once.
    pub(super) fn set_segments(&mut self, segs: &[Seg]) -> Result<(), String> {
        let mut next = 0;
        for (index, seg) in segs.iter().enumerate() {
            if seg.row0 != next || seg.n == 0 || seg.lane == NO_LANE {
                return Err(format!(
                    "rows: segment {index} {seg:?} does not continue at row {next}"
                ));
            }
            if segs[..index].iter().any(|other| other.lane == seg.lane) {
                return Err(format!("rows: lane {} has two segments", seg.lane));
            }
            next += seg.n;
        }
        self.segs.clear();
        self.segs.extend_from_slice(segs);
        Ok(())
    }

    /// Clears the step's segments: the next steps are the selected lane's alone.
    pub(super) fn clear_segments(&mut self) {
        self.segs.clear();
    }

    /// [DeviceLayer::begin_rows_device] for one step over several lanes' rows: `segs` tile the
    /// step's rows (the token ids at `ids`, in step order), each segment its lane's next positions.
    /// The segments stay installed for the step's layers, head and drafter seed.
    ///
    /// # Errors
    /// Segments that do not tile 1..=R_MAX rows, or a segment that does not continue its lane.
    pub(super) fn begin_rows_segments(
        &mut self,
        rows: &rows::RowsState,
        segs: &[Seg],
        embed: u64,
        ids: u64,
        exchange: (usize, usize),
    ) -> Result<usize, String> {
        self.set_segments(segs)?;
        let n: usize = segs.iter().map(|seg| seg.n).sum();
        if n > rows::R_MAX {
            self.segs.clear();
            return Err(format!(
                "rows: {n} rows of several lanes are more than {}",
                rows::R_MAX
            ));
        }
        for seg in segs {
            self.enter_seg(seg)?;
            self.check_begin(seg.p0, seg.n, rows::R_MAX)?;
        }
        self.upload_row_table(rows, segs)?;
        self.embed_rows_on_device(rows, embed, ids, n, n, 0, exchange, false)?;
        for seg in segs {
            self.enter_seg(seg)?;
            self.positions_begun = seg.p0 + seg.n;
        }
        Ok(n)
    }

    /// Gives up lane `lane`'s memory (its state is gone: the lane starts again at position 0).
    ///
    /// # Errors
    /// No lanes, or a driver failure.
    pub(super) fn release_lane(&mut self, lane: usize) -> Result<(), String> {
        self.cuda.synchronize()?;
        let lanes = self.lanes.as_mut().ok_or("lanes: none attached")?;
        lanes.release(lane)?;
        if lanes.selected == lane {
            self.positions_begun = 0;
            self.appended_rows.iter_mut().for_each(|count| *count = 0);
        }
        Ok(())
    }

    /// Backs lane `lane` for `rows` positions; false when the budget cannot.
    ///
    /// # Errors
    /// No lanes, or a driver failure.
    pub(super) fn ensure_lane(&mut self, lane: usize, rows: usize) -> Result<bool, String> {
        self.lanes
            .as_mut()
            .ok_or("lanes: none attached")?
            .ensure(lane, rows)
    }

    /// Whether one sequence of `rows` positions can be backed at all ([`Lanes::fits`]); true
    /// without lanes.
    pub(super) fn lane_fits(&self, rows: usize, ahead: usize) -> bool {
        self.lanes
            .as_ref()
            .is_none_or(|lanes| lanes.fits(rows, ahead))
    }

    /// The lanes attached, or 0.
    pub(super) fn lane_count(&self) -> usize {
        self.lanes.as_ref().map_or(0, |lanes| lanes.count())
    }

    /// The physical bytes the lanes hold and may hold.
    pub(super) fn lane_memory(&self) -> (usize, usize) {
        self.lanes.as_ref().map_or((0, 0), |lanes| lanes.memory())
    }

    /// The rows lane `lane` has backed (0 without lanes).
    pub(super) fn lane_backed(&self, lane: usize) -> usize {
        self.lanes.as_ref().map_or(0, |lanes| lanes.backed(lane))
    }

    /// The first row lane `lane`'s window rings still back (0 without lanes).
    pub(super) fn lane_floor(&self, lane: usize) -> usize {
        self.lanes.as_ref().map_or(0, |lanes| lanes.floor(lane))
    }

    /// Gives back lane `lane`'s window ring rows below `floor` (waiting for the device first when
    /// that unmaps anything). Returns the chunks given back.
    ///
    /// # Errors
    /// No lanes, or a driver failure.
    pub(super) fn slide_lane(&mut self, lane: usize, floor: usize) -> Result<usize, String> {
        let lanes = self.lanes.as_ref().ok_or("lanes: none attached")?;
        if lanes.slid_chunks(lane, floor).is_empty() {
            return Ok(0);
        }
        self.cuda.synchronize()?;
        self.lanes
            .as_mut()
            .ok_or("lanes: none attached")?
            .slide(lane, floor)
    }

    /// Gives lane `dst` lane `src`'s state of positions `..rows`, a resume boundary `src` has run
    /// past: its window rings start at the higher of the two floors, and its position counters
    /// are `src`'s cut at `rows`, so it resumes at `rows` as if it had run them. False, copying
    /// nothing, when the budget cannot back `dst` that far.
    ///
    /// # Errors
    /// No lanes, a lane outside them, or a driver failure.
    pub(super) fn copy_lane_prefix(
        &mut self,
        dst: usize,
        src: usize,
        rows: usize,
    ) -> Result<bool, String> {
        let lanes = self.lanes.as_ref().ok_or("lanes: none attached")?;
        if dst == src || dst.max(src) >= lanes.count() {
            return Err(format!(
                "lanes: lane {src} to lane {dst} of {}",
                lanes.count()
            ));
        }
        let floor = lanes.floor(dst).max(lanes.floor(src));
        self.slide_lane(dst, floor)?;
        let lanes = self.lanes.as_mut().ok_or("lanes: none attached")?;
        if !lanes.copy_prefix(dst, src, rows, floor)? {
            return Ok(false);
        }
        let counts: Vec<usize> = if lanes.selected == src {
            &self.appended_rows
        } else {
            &lanes.host[src].appended_rows
        }
        .iter()
        .map(|&count| count.min(rows))
        .collect();
        if lanes.selected == dst {
            self.positions_begun = rows;
            self.appended_rows = counts;
            self.attn_window = AttentionWindow::default();
        } else {
            let host = &mut lanes.host[dst];
            host.positions_begun = rows;
            host.appended_rows = counts;
        }
        Ok(true)
    }
}
