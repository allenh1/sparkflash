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
use sparkflash_kernels::checkpoint::Checkpoint;
use sparkflash_kernels::lane_memory::{ChunkPool, LaneShape, LaneSpan, Layout};

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

/// Where each part of a lane starts, relative to the lane's base.
#[derive(Clone, Debug)]
struct Offsets {
    kv_ring: usize,
    wkv: usize,
    wgate: usize,
    store_codes: usize,
    store_scales: usize,
    index_keys: usize,
    /// Each drafter stage's (codes, scales) ring.
    draft: Vec<(usize, usize)>,
    /// Per layer, a grouped-query block's BF16 (keys, values) cache, position-addressed; `None`
    /// at every other block.
    grouped: Vec<Option<(usize, usize)>>,
    /// Per layer, a Gated DeltaNet block's (conv history, state), each touched whole; `None` at
    /// every other block.
    recurrent: Vec<Option<(usize, usize)>>,
}

/// One lane's device pointers, as the DeviceLayer fields hold them.
#[derive(Clone, Debug)]
pub(super) struct LanePointers {
    pub(super) kv_ring: u64,
    pub(super) wkv: u64,
    pub(super) wgate: u64,
    pub(super) store_codes: u64,
    pub(super) store_scales: u64,
    pub(super) index_keys: u64,
    pub(super) draft: Vec<(u64, u64)>,
    pub(super) grouped: Vec<Option<(u64, u64)>>,
    pub(super) recurrent: Vec<Option<(u64, u64)>>,
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

/// Every lane's state: the reservation, the layout and the chunk pool ([sparkflash_kernels::
/// lane_memory] holds the backing every lane arm shares).
pub(super) struct Lanes<'a> {
    cuda: &'a Cuda,
    base: u64,
    shape: LaneShape,
    offsets: Offsets,
    /// Per lane, its span of the reservation and the chunks behind it.
    spans: Vec<LaneSpan>,
    pool: ChunkPool,
    pub(super) host: Vec<LaneHost>,
    /// The lane whose pointers and counters the DeviceLayer holds.
    pub(super) selected: usize,
    /// Per lane, its recurrent regions' rewind point in the lane's checkpoint arena; empty when
    /// the stack has no recurrent block.
    checkpoints: Vec<Checkpoint>,
    /// The recurrent regions' bytes, in the checkpoint's order.
    recurrent_sizes: Vec<usize>,
}

impl Lanes<'_> {
    pub(super) fn count(&self) -> usize {
        self.spans.len()
    }

    /// Lane `lane`'s pointers.
    pub(super) fn pointers(&self, lane: usize) -> LanePointers {
        let base = self.base + (lane * self.shape.stride) as u64;
        let at = |offset: usize| base + offset as u64;
        let o = &self.offsets;
        LanePointers {
            kv_ring: at(o.kv_ring),
            wkv: at(o.wkv),
            wgate: at(o.wgate),
            store_codes: at(o.store_codes),
            store_scales: at(o.store_scales),
            index_keys: at(o.index_keys),
            draft: o
                .draft
                .iter()
                .map(|&(codes, scales)| (at(codes), at(scales)))
                .collect(),
            grouped: o
                .grouped
                .iter()
                .map(|part| part.map(|(keys, values)| (at(keys), at(values))))
                .collect(),
            recurrent: o
                .recurrent
                .iter()
                .map(|part| part.map(|(history, state)| (at(history), at(state))))
                .collect(),
        }
    }

    /// Lane `lane`'s recurrent regions in the checkpoint's order: each recurrent block's conv
    /// history and state, in block order ([super::rows::recurrent_sizes]).
    fn recurrent_live(&self, lane: usize) -> Vec<Deviceptr> {
        self.pointers(lane)
            .recurrent
            .into_iter()
            .flatten()
            .flat_map(|(history, state)| [Deviceptr(history), Deviceptr(state)])
            .collect()
    }

    /// Lane `lane`'s checkpoint, when the stack has recurrent blocks.
    fn checkpoint(&self, lane: usize) -> Option<&Checkpoint> {
        self.checkpoints.get(lane)
    }

    /// The rows lane `lane` has backed.
    pub(super) fn backed(&self, lane: usize) -> usize {
        self.spans.get(lane).map_or(0, LaneSpan::backed)
    }

    /// The first row lane `lane`'s window rings still back.
    pub(super) fn floor(&self, lane: usize) -> usize {
        self.spans.get(lane).map_or(0, LaneSpan::floor)
    }

    /// Whether one sequence of `rows` positions fits the budget with every other lane empty: its
    /// stores for every row, and its window rings from the floor of row `rows - ahead` (a tick
    /// backs up to `ahead` rows before its lane's floor slides).
    pub(super) fn fits(&self, rows: usize, ahead: usize) -> bool {
        let floor = slide_floor(self.shape.quantized(rows).saturating_sub(ahead));
        self.shape.chunks_for(rows, floor) <= self.pool.budget()
    }

    /// Backs lane `lane` for sequences of up to `rows` positions. Returns false, mapping nothing
    /// more, when the budget has too few chunks left (the caller frees a lane and retries).
    ///
    /// # Errors
    /// A driver failure creating, mapping or zeroing a chunk.
    pub(super) fn ensure(&mut self, lane: usize, rows: usize) -> Result<bool, String> {
        let count = self.count();
        let span = self
            .spans
            .get_mut(lane)
            .ok_or_else(|| format!("lanes: lane {lane} of {count}"))?;
        span.ensure(self.cuda, &self.shape, &mut self.pool, rows)
            .map_err(|error| format!("lanes: lane {lane}: {error}"))
    }

    /// The chunks lane `lane` would give back if its window rings started at row `floor`: those
    /// it maps that no part reads from there on.
    pub(super) fn slid_chunks(&self, lane: usize, floor: usize) -> Vec<usize> {
        self.spans[lane].slid_chunks(&self.shape, floor)
    }

    /// Moves lane `lane`'s window floor to row `floor`, returning [Self::slid_chunks] to the
    /// pool. Nothing queued may still read rows below `floor`.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub(super) fn slide(&mut self, lane: usize, floor: usize) -> Result<usize, String> {
        self.spans[lane].slide(self.cuda, &self.shape, &mut self.pool, floor)
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
        let (dst_span, src_span) = if dst < src {
            let (low, high) = self.spans.split_at_mut(src);
            (&mut low[dst], &high[0])
        } else {
            let (low, high) = self.spans.split_at_mut(dst);
            (&mut high[0], &low[src])
        };
        dst_span.copy_prefix(
            self.cuda,
            &self.shape,
            &mut self.pool,
            src_span,
            rows,
            floor,
        )
    }

    /// Returns every chunk of lane `lane` to the pool. The device must be idle: nothing queued may
    /// still read the lane.
    ///
    /// # Errors
    /// A driver failure unmapping a chunk.
    pub(super) fn release(&mut self, lane: usize) -> Result<(), String> {
        self.spans[lane].release(self.cuda, &self.shape, &mut self.pool)?;
        if let Some(checkpoint) = self.checkpoints.get_mut(lane) {
            checkpoint.forget();
        }
        if let Some(host) = self.host.get_mut(lane) {
            host.positions_begun = 0;
            host.appended_rows.iter_mut().for_each(|count| *count = 0);
        }
        Ok(())
    }

    /// The physical bytes the lanes hold and may hold.
    pub(super) fn memory(&self) -> (usize, usize) {
        let mapped: usize = self.spans.iter().map(LaneSpan::mapped).sum();
        let granularity = self.shape.granularity;
        (mapped * granularity, self.pool.budget() * granularity)
    }
}

impl Drop for Lanes<'_> {
    fn drop(&mut self) {
        let _ = self.cuda.synchronize();
        for lane in 0..self.spans.len() {
            let _ = self.release(lane);
        }
        self.pool.drain(self.cuda);
        self.cuda
            .vmm_free(Deviceptr(self.base), self.shape.stride * self.spans.len());
    }
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
        let mut draft = vec![(0usize, 0usize); crate::plan::n_mtp_layers()];
        let draft_rows = geometry.draft_ring_rows.max(1);
        let latent = self.head_width as usize;
        for stage in &mut draft {
            stage.0 = layout.reserve(draft_rows * latent);
            layout.window(stage.0, draft_rows * latent, latent);
            stage.1 = layout.reserve(draft_rows * latent / 32);
            layout.window(stage.1, draft_rows * latent / 32, latent / 32);
        }
        // A grouped-query block's BF16 keys and values, one row of the rank's key/value heads a
        // position, backed as far as the lane's positions reach.
        let mut grouped = vec![None; layers];
        if let Some(shape) = self.grouped_shape()? {
            let row_bytes = shape.kv_width() * 2;
            for &layer in &crate::plan::description().values().gqa_layers {
                let slot = grouped.get_mut(layer).ok_or_else(|| {
                    format!("lanes: grouped-query block {layer} of {layers} layers")
                })?;
                let bytes = self.max_positions * row_bytes;
                *slot = Some((
                    layout.part(bytes, row_bytes, 0),
                    layout.part(bytes, row_bytes, 0),
                ));
            }
        }
        // A Gated DeltaNet block's conv history and state, each a sequence's whole, then one
        // checkpoint arena for them all: nothing at all for a stack with no recurrent block.
        let mut recurrent = vec![None; layers];
        let mut checkpoint = None;
        let mut recurrent_sizes = Vec::new();
        if let Some(geometry) = super::rows::recurrent_geometry()? {
            recurrent_sizes = super::rows::recurrent_sizes(&geometry);
            for (block, &layer) in crate::plan::description()
                .values()
                .gdn_layers
                .iter()
                .enumerate()
            {
                let slot = recurrent.get_mut(layer).ok_or_else(|| {
                    format!("lanes: Gated DeltaNet block {layer} of {layers} layers")
                })?;
                *slot = Some((
                    layout.state(recurrent_sizes[2 * block]),
                    layout.state(recurrent_sizes[2 * block + 1]),
                ));
            }
            checkpoint = Some(layout.state(Checkpoint::arena_bytes(&recurrent_sizes)));
        }
        let shape = layout.finish(granularity, ROW_QUANTUM);
        let stride = shape.stride;
        let offsets = Offsets {
            kv_ring,
            wkv,
            wgate,
            store_codes,
            store_scales,
            index_keys,
            draft,
            grouped,
            recurrent,
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
        let budget_bytes = sparkflash_model::active_profile()
            .runtime
            .lane_memory_mb
            .map_or(DEFAULT_BUDGET_BYTES, |mb| mb << 20);
        let lanes = Lanes {
            cuda,
            base,
            spans: (0..count)
                .map(|lane| LaneSpan::new(base + (lane * stride) as u64, &shape))
                .collect(),
            shape,
            offsets,
            pool: ChunkPool::new(granularity, budget_bytes),
            host: vec![
                LaneHost {
                    positions_begun: 0,
                    appended_rows: vec![0; layers],
                };
                count
            ],
            selected: usize::MAX,
            checkpoints: checkpoint.map_or_else(Vec::new, |offset| {
                (0..count)
                    .map(|lane| {
                        Checkpoint::new(
                            Deviceptr(base + (lane * stride + offset) as u64),
                            recurrent_sizes.clone(),
                        )
                    })
                    .collect()
            }),
            recurrent_sizes,
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

    /// The segments the step installed (empty for a single-lane step).
    pub(super) fn segments(&self) -> &[Seg] {
        &self.segs
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
        self.enter_lane(seg.lane)
    }

    /// Selects lane `lane`, or nothing for [`NO_LANE`] (the state already selected).
    pub(super) fn enter_lane(&mut self, lane: usize) -> Result<(), String> {
        if lane == NO_LANE {
            Ok(())
        } else {
            self.select_lane(lane)
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

    /// One lane's bytes over its positions, rounded up (0 without lanes).
    pub(super) fn lane_row_bytes(&self) -> usize {
        self.lanes.as_ref().map_or(0, |lanes| {
            lanes.shape.stride.div_ceil(self.max_positions.max(1))
        })
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
    /// Whether the lanes hold recurrent state: the stack has a Gated DeltaNet block.
    pub(super) fn has_recurrent(&self) -> bool {
        self.lanes
            .as_ref()
            .is_some_and(|lanes| !lanes.checkpoints.is_empty())
    }

    /// The bytes one lane's live recurrent state holds (0 for a stack with no recurrent block).
    pub(super) fn recurrent_bytes(&self) -> usize {
        self.lanes
            .as_ref()
            .map_or(0, |lanes| lanes.recurrent_sizes.iter().sum())
    }

    /// The position lane `lane`'s recurrent checkpoint holds, when it holds one.
    pub(super) fn recurrent_point(&self, lane: usize) -> Option<usize> {
        self.lanes
            .as_ref()
            .and_then(|lanes| lanes.checkpoint(lane))
            .and_then(Checkpoint::point)
    }

    /// Puts lane `lane`'s recurrent state at `start`, the position its rows resume from
    /// ([sparkflash_runtime::tick::recurrent_start]): zero at 0 (the checkpoint forgets its
    /// point, a prompt from the start saves its own), else the checkpoint's copy, which must hold
    /// `start`. Nothing for a stack with no recurrent block. The lane must be backed.
    ///
    /// # Errors
    /// A start that is neither 0 nor the checkpoint's point, or a driver failure.
    pub(super) fn recurrent_resume(&mut self, lane: usize, start: usize) -> Result<(), String> {
        let cuda = self.cuda;
        let Some(lanes) = self
            .lanes
            .as_mut()
            .filter(|lanes| !lanes.checkpoints.is_empty())
        else {
            return Ok(());
        };
        let live = lanes.recurrent_live(lane);
        let checkpoint = lanes
            .checkpoints
            .get_mut(lane)
            .ok_or_else(|| format!("lanes: no checkpoint for lane {lane}"))?;
        if start == 0 {
            checkpoint.forget();
            for (region, &bytes) in live.iter().zip(&lanes.recurrent_sizes) {
                cuda.memset_d8_async(*region, 0, bytes)?;
            }
            return Ok(());
        }
        checkpoint
            .restore(cuda, None, &live, start)
            .map_err(|error| format!("lanes: lane {lane} resumes at {start}: {error}"))
    }

    /// Saves lane `lane`'s recurrent state as the state after `point` positions: the prefill's
    /// save at its resume boundary below the prompt's end. Nothing for a stack with no recurrent
    /// block.
    ///
    /// # Errors
    /// A driver failure.
    pub(super) fn recurrent_save(&mut self, lane: usize, point: usize) -> Result<(), String> {
        let cuda = self.cuda;
        let Some(lanes) = self
            .lanes
            .as_mut()
            .filter(|lanes| !lanes.checkpoints.is_empty())
        else {
            return Ok(());
        };
        let live = lanes.recurrent_live(lane);
        lanes
            .checkpoints
            .get_mut(lane)
            .ok_or_else(|| format!("lanes: no checkpoint for lane {lane}"))?
            .save(cuda, None, &live, point)
            .map_err(|error| format!("lanes: lane {lane} saves at {point}: {error}"))
    }

    /// Gives lane `dst` lane `src`'s recurrent state at `point`, the point `src`'s checkpoint
    /// holds and the adopted prefix's length: `dst`'s live regions and its checkpoint, so it
    /// resumes there and may resume there again. Nothing for a stack with no recurrent block.
    ///
    /// # Errors
    /// A point `src` does not hold, or a driver failure.
    pub(super) fn recurrent_adopt(
        &mut self,
        dst: usize,
        src: usize,
        point: usize,
    ) -> Result<(), String> {
        let cuda = self.cuda;
        let Some(lanes) = self
            .lanes
            .as_mut()
            .filter(|lanes| !lanes.checkpoints.is_empty())
        else {
            return Ok(());
        };
        let live = lanes.recurrent_live(dst);
        lanes
            .checkpoint(src)
            .ok_or_else(|| format!("lanes: no checkpoint for lane {src}"))?
            .restore(cuda, None, &live, point)
            .map_err(|error| format!("lanes: lane {dst} adopts lane {src} at {point}: {error}"))?;
        self.recurrent_save(dst, point)
    }

    /// Commits pass row `row`'s recurrent snapshots into lane `lane`'s live regions: the state
    /// after the last row a verify kept. Nothing for a stack with no recurrent block.
    ///
    /// # Errors
    /// A rows state without a snapshot slab, or a driver failure.
    pub(super) fn commit_recurrent(
        &self,
        rows: &super::rows::RowsState,
        lane: usize,
        row: usize,
    ) -> Result<(), String> {
        let Some(lanes) = self
            .lanes
            .as_ref()
            .filter(|lanes| !lanes.checkpoints.is_empty())
        else {
            return Ok(());
        };
        rows.recurrent
            .as_ref()
            .and_then(|recurrent| recurrent.snapshots.as_ref())
            .ok_or("lanes: a verify kept fewer rows than it ran over a rows state with no snapshot slab")?
            .commit(self.cuda, None, row, &lanes.recurrent_live(lane))
            .map_err(|error| format!("lanes: lane {lane} commits row {row}: {error}"))
    }
}
