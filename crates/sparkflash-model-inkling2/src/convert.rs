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

//! The Inkling-2-Small rank-file converter: one plan-order file per TP rank.
//!
//! The arm already knows exactly which bytes of which released tensor every rank holds, because
//! [InklingTensorPlan] is the same plan the host reference and the device arm read. So the
//! converter is a walk over that plan through the shared rank-file container
//! ([sparkflash_weights::rank_file]) rather than a second transcription of the checkpoint, and it
//! reuses [sparkflash_weights::rank_file::SlotPlan], [SlotFileWriter] and [SlotCursor] exactly as
//! [sparkflash_ops::slot_pack::SlotPack] does for the plan-order arms.
//!
//! Two things are this arm's own. A split can be on an inner axis (the output projections shard
//! their contraction axis), so one rank's payload is a strided run of rows rather than one
//! contiguous byte range: a [RankSlot] carries the source ranges that make up its payload in
//! payload order, not just one offset. And a tensor larger than [SLOT_CHUNK_BYTES] is written as
//! several chunks of the same tensor, so a load streams it rather than materialising it.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sparkflash_model::TpPlan;
use sparkflash_weights::rank_file::{
    SlotCursor, SlotFileWriter, SlotIndex, SlotPlan, SlotRecords, WrittenRank, fnv1a,
};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{TensorSpec, WeightError, WeightSource};

use crate::config::Inkling2Config;
use crate::tensor::InklingTensorPlan;

/// The largest payload one slot holds. A released expert tensor is a few gigabytes; chunking it
/// keeps every read under a weight source's single-read limit and lets a load stream the tensor.
pub const SLOT_CHUNK_BYTES: usize = 512 << 20;

/// What an Inkling rank file records beside its slots. The plan names every weight, so there is
/// nothing else to carry; the empty record is still written so the container's model name is
/// checked on a load.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InklingRecords {}

impl SlotRecords for InklingRecords {
    const MODEL: &'static str = "inkling2";
}

/// One slot of one rank: the released tensor it streams and the source ranges its payload
/// concatenates, in payload order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankSlot {
    /// The slot key: the released name, with a chunk suffix when the tensor spans several slots.
    pub key: String,
    /// The released tensor the slot belongs to.
    pub name: String,
    /// The checkpoint byte ranges the payload holds, in payload order.
    pub ranges: Vec<(u64, usize)>,
    /// The payload length in bytes.
    pub bytes: usize,
}

impl RankSlot {
    /// Whether this slot is a chunk of a larger tensor.
    #[must_use]
    pub fn is_chunk(&self) -> bool {
        self.key != self.name
    }

    /// The payload the slot holds, read through the source's range path when it has one.
    ///
    /// # Errors
    /// A source read error, or a range outside the tensor.
    pub fn read(&self, source: &dyn WeightSource) -> Result<Vec<u8>, String> {
        let mut payload = Vec::with_capacity(self.bytes);
        for (offset, len) in &self.ranges {
            payload.extend_from_slice(&read_range(source, &self.name, *offset, *len)?);
        }
        debug_assert_eq!(payload.len(), self.bytes);
        Ok(payload)
    }
}

/// One rank's payload byte layout of one released tensor: the ranges, in payload order, and the
/// payload length.
fn rank_ranges(
    config: &Inkling2Config,
    layout: TpLayout,
    plan: &InklingTensorPlan,
    index: usize,
) -> Result<(Vec<(u64, usize)>, usize), String> {
    let _ = config;
    let full = &plan.plan().specs()[index];
    let held = plan.rank_spec_at(index);
    let bytes = usize::try_from(held.storage_bytes())
        .map_err(|_| format!("{}: the rank payload does not fit this machine", full.name))?;
    let slices = plan.rank_slices_at(index);
    if slices.is_empty() {
        return Ok((vec![(0, bytes)], bytes));
    }
    if full.shape.is_empty() {
        return Err(format!("{}: a scalar cannot be sharded", full.name));
    }
    // Every axis the plan slices, as the run this rank holds on it.
    let mut bounds: Vec<(usize, usize)> = full.shape.iter().map(|width| (0, *width)).collect();
    let mut slices = slices.to_vec();
    slices.retain(|slice| {
        // A split that hands this rank the whole axis is not a shard, and treating it as one
        // would cut a row into one-byte runs for nothing.
        !(slice.axis < full.shape.len()
            && slice.split.offset == 0
            && slice.split.per_rank == full.shape[slice.axis])
    });
    if slices.is_empty() {
        return Ok((vec![(0, bytes)], bytes));
    }
    for slice in &slices {
        let axis = slice.axis;
        if axis >= full.shape.len() {
            return Err(format!(
                "{}: the split names axis {axis} of a {} axis tensor",
                full.name,
                full.shape.len()
            ));
        }
        bounds[axis] = (slice.split.offset, slice.split.per_rank);
    }
    let last = slices.iter().map(|slice| slice.axis).max().unwrap_or(0);
    let inner: usize = full.shape[last + 1..].iter().product();
    let run_elements = bounds[last].1 * inner;
    let run_bytes = full.dtype.storage_bytes(run_elements);
    if run_bytes != bounds[last].1 * full.dtype.storage_bytes(inner) {
        return Err(format!(
            "{}: a {run_elements} element run is not byte aligned",
            full.name
        ));
    }
    // Prefixes over every axis before the last sliced one; the run itself starts at the last
    // sliced axis and covers its whole share plus the untouched inner axes, so it is contiguous.
    let radices: Vec<usize> = (0..last).map(|axis| bounds[axis].1).collect();
    let total: usize = radices.iter().product();
    let mut ranges = Vec::with_capacity(total.max(1));
    let mut counter = vec![0usize; last];
    for _ in 0..total.max(1) {
        let mut element = 0usize;
        for axis in 0..=last {
            let index = if axis == last {
                bounds[last].0
            } else {
                bounds[axis].0 + counter[axis]
            };
            element = element * full.shape[axis] + index;
        }
        // `element` indexes the run's first corner; the inner axes after the last sliced one
        // are all held, so the byte offset scales by their product.
        ranges.push((full.dtype.storage_bytes(element * inner) as u64, run_bytes));
        for axis in (0..last).rev() {
            counter[axis] += 1;
            if counter[axis] < radices[axis] {
                break;
            }
            counter[axis] = 0;
        }
    }
    // Coalesce byte-adjacent runs so a row-wise split of a wide matrix is one range, not
    // thousands of neighbouring ones.
    let mut merged: Vec<(u64, usize)> = Vec::with_capacity(ranges.len());
    for (offset, len) in ranges {
        match merged.last_mut() {
            Some((last_offset, last_len)) if *last_offset + *last_len as u64 == offset => {
                *last_len += len;
            }
            _ => merged.push((offset, len)),
        }
    }
    let ranges = merged;
    let sum: usize = ranges.iter().map(|(_, len)| *len).sum();
    if sum != bytes {
        return Err(format!(
            "{}: the rank ranges hold {sum} bytes and the plan says {bytes}",
            full.name
        ));
    }
    let _ = (layout, held);
    Ok((ranges, bytes))
}

/// Every slot rank `layout.rank` holds, in plan order.
///
/// # Errors
/// A world the model does not divide, an unaligned split, or a plan whose rank payload disagrees
/// with the split the plan records.
pub fn rank_slots(config: &Inkling2Config, layout: TpLayout) -> Result<Vec<RankSlot>, String> {
    let tp = TpPlan::new(layout.world_size, layout.rank)
        .map_err(|error| format!("TP{}/rank{}: {error}", layout.world_size, layout.rank))?;
    let plan = InklingTensorPlan::served(config, tp)?;
    let mut slots = Vec::new();
    for index in 0..plan.plan().len() {
        let name = plan.plan().specs()[index].name.clone();
        let (ranges, bytes) = rank_ranges(config, layout, &plan, index)?;
        if bytes == 0 {
            continue;
        }
        let chunks = bytes.div_ceil(SLOT_CHUNK_BYTES).max(1);
        // Split the range list into chunk-sized payloads with a cursor, so a tensor of many
        // ranges is one pass rather than one shift per range.
        let mut pending: std::collections::VecDeque<(u64, usize)> = ranges.into();
        let mut chunk = 0usize;
        let mut taken = 0usize;
        while taken < bytes {
            let want = SLOT_CHUNK_BYTES.min(bytes - taken);
            let mut pieces = Vec::new();
            let mut room = want;
            while room > 0 {
                let Some((offset, len)) = pending.pop_front() else {
                    return Err(format!("{name}: the ranges end after {taken} bytes"));
                };
                let take = len.min(room);
                pieces.push((offset, take));
                if len > take {
                    pending.push_front((offset + take as u64, len - take));
                }
                room -= take;
            }
            let key = if chunks == 1 {
                name.clone()
            } else {
                format!("{name}#{chunk}")
            };
            slots.push(RankSlot {
                key,
                name: name.clone(),
                ranges: pieces,
                bytes: want,
            });
            taken += want;
            chunk += 1;
        }
    }
    Ok(slots)
}

/// One byte range of one released tensor, through the source's range path when it has one.
fn read_range(
    source: &dyn WeightSource,
    name: &str,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, String> {
    match source.read_ranges(name, &[(offset, len)]) {
        Some(Ok(bytes)) => Ok(bytes),
        Some(Err(error)) => Err(format!("{name}: {error}")),
        None => {
            let all = source
                .read(name)
                .map_err(|error| format!("{name}: {error}"))?;
            let start = usize::try_from(offset)
                .map_err(|_| format!("{name}: offset {offset} does not fit this machine"))?;
            all.get(start..start + len)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| {
                    format!(
                        "{name}: range {offset}..{} is outside the tensor",
                        offset as usize + len
                    )
                })
        }
    }
}

/// Writes rank `layout.rank`'s converted file into `output_dir`.
///
/// # Errors
/// What [rank_slots] refuses, a source read error, or a file that cannot be created or written.
pub fn convert_rank(
    source: &dyn WeightSource,
    config: &Inkling2Config,
    layout: TpLayout,
    output_dir: &Path,
    label: &str,
) -> Result<WrittenRank, String> {
    let slots = rank_slots(config, layout)?;
    let mut placing = SlotPlan::default();
    for slot in &slots {
        placing.place(&slot.key, slot.bytes);
    }
    let index = placing.into_index(layout, label, InklingRecords {});
    let mut writer = SlotFileWriter::create(output_dir, index)?;
    for slot in &slots {
        let payload = slot.read(source)?;
        writer.put(&slot.key, slot.bytes, &payload)?;
    }
    writer.finish()
}

/// Streams every slot of a converted rank file to `visit`, in plan order, after checking that the
/// file is this rank's and holds exactly this plan.
///
/// This is the loader the device arm will call with an upload closure; [verify_rank] is the same
/// walk with a comparing closure, and the proof that the file and the checkpoint agree.
///
/// # Errors
/// A file that is another model's or rank's, a slot that is not where the plan puts it, or a
/// visitor that fails.
pub fn visit_rank<F>(
    path: &Path,
    config: &Inkling2Config,
    layout: TpLayout,
    mut visit: F,
) -> Result<SlotIndex<InklingRecords>, String>
where
    F: FnMut(&RankSlot, &[u8]) -> Result<(), String>,
{
    let (preamble, index) = SlotIndex::<InklingRecords>::read(path, layout)?;
    let slots = rank_slots(config, layout)?;
    let mut cursor = SlotCursor::new(&index.slots);
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut payload = Vec::new();
    for (position, slot) in slots.iter().enumerate() {
        let at = cursor.take(&slot.key, slot.bytes)?;
        payload.clear();
        payload.resize(slot.bytes, 0);
        file.seek(SeekFrom::Start(preamble.data_offset + at as u64))
            .map_err(|error| format!("{}: seek to {at}: {error}", path.display()))?;
        file.read_exact(&mut payload)
            .map_err(|error| format!("{}: read slot {}: {error}", path.display(), slot.key))?;
        let digest = fnv1a(&payload);
        if digest != index.slots[position].digest {
            return Err(format!(
                "{}: slot {} digest mismatch (measured {digest:#018x}, header {:#018x})",
                path.display(),
                slot.key,
                index.slots[position].digest
            ));
        }
        visit(slot, &payload)?;
    }
    cursor.finish()?;
    Ok(index)
}

/// What [verify_rank] measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankVerification {
    /// Slots the file holds.
    pub slots: usize,
    /// Payload bytes checked, file against checkpoint.
    pub bytes: u64,
}

/// Checks a converted rank file against the checkpoint it was written from, slot by slot.
///
/// # Errors
/// What [visit_rank] refuses, a checkpoint read error, or any slot whose bytes differ from the
/// checkpoint range the plan names.
pub fn verify_rank(
    path: &Path,
    source: &dyn WeightSource,
    config: &Inkling2Config,
    layout: TpLayout,
) -> Result<RankVerification, String> {
    let mut bytes = 0u64;
    let index = visit_rank(path, config, layout, |slot, payload| {
        let expected = slot.read(source)?;
        if expected != payload {
            return Err(format!(
                "{}: slot {} does not match the checkpoint",
                path.display(),
                slot.key
            ));
        }
        bytes += slot.bytes as u64;
        Ok(())
    })?;
    Ok(RankVerification {
        slots: index.slots.len(),
        bytes,
    })
}

/// A [WeightSource] over one converted rank file.
///
/// The plan decides what the file holds, so this opens with the same [rank_slots] walk the
/// converter wrote and presents each planned tensor under its released name with the rank-local
/// shape. A rank file's payload is contiguous and in plan order, so a read is one `pread` per
/// slot it touches rather than one per source range: the loader is a walk, not a re-tiling.
pub struct RankFileSource {
    path: PathBuf,
    file: std::sync::Mutex<File>,
    data_offset: u64,
    index: SlotIndex<InklingRecords>,
    specs: std::collections::BTreeMap<String, TensorSpec>,
    slots: Vec<RankSlot>,
}

impl std::fmt::Debug for RankFileSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RankFileSource")
            .field("path", &self.path)
            .field("slots", &self.slots.len())
            .field("tensors", &self.specs.len())
            .finish()
    }
}

impl RankFileSource {
    /// Opens one converted rank file and checks its slots against the plan it was written from.
    ///
    /// # Errors
    /// A file that is another model's or rank's, a slot that is not where the plan puts it, or a
    /// file that cannot be opened.
    pub fn open(path: &Path, config: &Inkling2Config, layout: TpLayout) -> Result<Self, String> {
        let (preamble, index) = SlotIndex::<InklingRecords>::read(path, layout)?;
        let slots = rank_slots(config, layout)?;
        let mut cursor = SlotCursor::new(&index.slots);
        for slot in &slots {
            cursor.take(&slot.key, slot.bytes)?;
        }
        cursor.finish()?;
        let tp = TpPlan::new(layout.world_size, layout.rank)
            .map_err(|error| format!("TP{}/rank{}: {error}", layout.world_size, layout.rank))?;
        let plan = InklingTensorPlan::served(config, tp)?;
        let specs = (0..plan.plan().len())
            .map(|index| plan.rank_spec_at(index).clone())
            .map(|spec| (spec.name.clone(), spec))
            .collect();
        let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            file: std::sync::Mutex::new(file),
            data_offset: preamble.data_offset,
            index,
            specs,
            slots,
        })
    }

    /// The slots the file holds, in plan order.
    #[must_use]
    pub fn slots(&self) -> &[RankSlot] {
        &self.slots
    }

    /// The converted payload bytes of one tensor, or `None` when the rank does not hold it.
    #[must_use]
    pub fn tensor_bytes(&self, name: &str) -> Option<u64> {
        self.specs.get(name).map(|spec| spec.storage_bytes() as u64)
    }

    /// Reads `len` payload bytes at `offset` of one tensor's rank-local bytes.
    fn read_tensor(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>, WeightError> {
        if !self.specs.contains_key(name) {
            return Err(WeightError::Missing {
                name: name.to_string(),
            });
        }
        let mut out = Vec::with_capacity(len);
        let mut tensor_start = 0u64;
        let mut remaining = len as u64;
        let mut wanted = offset;
        for (position, slot) in self.slots.iter().enumerate() {
            if slot.name != name {
                continue;
            }
            let slot_start = tensor_start;
            tensor_start += slot.bytes as u64;
            if remaining == 0 {
                break;
            }
            let slot_end = slot_start + slot.bytes as u64;
            if wanted >= slot_end {
                continue;
            }
            let local = wanted.max(slot_start) - slot_start;
            let take = (slot_end - (slot_start + local)).min(remaining);
            let header = &self.index.slots[position];
            let at = self.data_offset + header.offset as u64 + local;
            let mut buffer = vec![0u8; take as usize];
            let mut file = self.file.lock().map_err(|_| WeightError::Io {
                name: name.to_string(),
                detail: "the rank file lock is poisoned".to_owned(),
            })?;
            file.seek(SeekFrom::Start(at))
                .and_then(|_| file.read_exact(&mut buffer))
                .map_err(|error| WeightError::Io {
                    name: name.to_string(),
                    detail: error.to_string(),
                })?;
            out.extend_from_slice(&buffer);
            wanted += take;
            remaining -= take;
        }
        if remaining != 0 {
            return Err(WeightError::Malformed {
                name: name.to_string(),
                detail: format!(
                    "{len} bytes at {offset} run past the {} the rank file holds",
                    self.tensor_bytes(name).unwrap_or(0)
                ),
            });
        }
        Ok(out)
    }
}

impl WeightSource for RankFileSource {
    fn directory(&self) -> Option<&Path> {
        self.path.parent()
    }

    fn names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.specs
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_string(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let bytes = self
            .tensor_bytes(name)
            .ok_or_else(|| WeightError::Missing {
                name: name.to_string(),
            })?;
        let bytes = usize::try_from(bytes).map_err(|_| WeightError::TooLarge {
            name: name.to_string(),
            bytes: bytes as u128,
            limit: usize::MAX as u128,
        })?;
        self.read_tensor(name, 0, bytes)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        Some((|| {
            let mut out = Vec::new();
            for (offset, len) in ranges {
                out.extend_from_slice(&self.read_tensor(name, *offset, *len)?);
            }
            Ok(out)
        })())
    }
}

#[cfg(test)]
mod tests {
    use super::rank_slots;
    use crate::config::Inkling2Config;
    use crate::tensor::InklingTensorPlan;
    use sparkflash_model::TpPlan;
    use sparkflash_weights::tp::TpLayout;

    fn released() -> Inkling2Config {
        Inkling2Config::from_json(
            include_str!("../config/inkling2_small.hf.json"),
            Some(include_str!("../config/inkling2_small.hf_quant.json")),
        )
        .expect("the released config parses")
    }

    /// The four rank files must tile the checkpoint: a sharded tensor's ranks cover it once,
    /// exactly, and a replicated tensor is the same bytes on every rank. Nothing may fall between
    /// ranks and nothing may be counted twice, which is what makes the per-node load a partition.
    #[test]
    fn the_four_ranks_tile_the_checkpoint() {
        use std::collections::BTreeMap;

        let config = released();
        let served = InklingTensorPlan::served(&config, TpPlan::new(1, 0).expect("TP1"))
            .expect("the served plan");
        // A tensor's largest rank slot, and each rank's (offset, length) ranges of it.
        type Coverage = (usize, Vec<Vec<(u64, usize)>>);
        let mut coverage: BTreeMap<String, Coverage> = BTreeMap::new();
        for rank in 0..4 {
            let layout = TpLayout {
                rank,
                world_size: 4,
            };
            let slots = rank_slots(&config, layout).expect("rank slots");
            let plan = InklingTensorPlan::served(&config, TpPlan::new(4, rank).expect("TP4"))
                .expect("the rank plan");
            let bytes: u64 = slots.iter().map(|slot| slot.bytes as u64).sum();
            assert_eq!(
                bytes,
                u64::try_from(plan.rank_storage_bytes()).expect("the rank fits u64"),
                "rank {rank} holds what its plan says"
            );
            // A rank has to fit the box: 113.7 GiB after the memory floor.
            assert!(
                bytes <= 113_700_000_000,
                "rank {rank} holds {bytes} bytes, past the 113.7 GB budget"
            );
            for slot in &slots {
                let entry = coverage
                    .entry(slot.name.clone())
                    .or_insert_with(|| (0, vec![Vec::new(); 4]));
                entry.0 = slot.bytes.max(entry.0) + entry.0 - entry.0;
                entry.1[rank].push(slot.ranges[0]);
                for range in slot.ranges.iter().skip(1) {
                    entry.1[rank].push(*range);
                }
            }
        }
        for (name, (_, per_rank)) in &coverage {
            let spec = served
                .plan()
                .spec(&served.plan().specs()[0].name)
                .map(|_| ());
            let _ = spec;
            let full = served
                .plan()
                .specs()
                .iter()
                .find(|spec| &spec.name == name)
                .map(|spec| u64::try_from(spec.storage_bytes()).expect("fits"))
                .expect("the covered tensor is planned");
            let replicated = per_rank.iter().all(|rank| rank == &per_rank[0]);
            if replicated {
                assert_eq!(
                    per_rank[0].iter().map(|(_, len)| *len as u64).sum::<u64>(),
                    full,
                    "{name}: a replicated tensor must be the whole tensor"
                );
                continue;
            }
            let mut runs: Vec<(u64, usize)> = per_rank.iter().flatten().copied().collect();
            runs.sort_unstable();
            let mut end = 0u64;
            for (offset, len) in runs {
                assert_eq!(offset, end, "{name}: the ranks leave a gap at {end}");
                end += len as u64;
            }
            assert_eq!(end, full, "{name}: the ranks do not cover the whole tensor");
        }
        assert!(
            coverage.contains_key("model.llm.layers.0.mlp.experts.w13_weight"),
            "the expert tensor is covered"
        );
    }

    /// Every planned tensor reaches the rank file, and a slot's ranges add up to its payload.
    #[test]
    fn every_planned_tensor_reaches_the_rank_file() {
        let config = released();
        let layout = TpLayout {
            rank: 1,
            world_size: 4,
        };
        let slots = rank_slots(&config, layout).expect("rank slots");
        let names: std::collections::BTreeSet<&str> =
            slots.iter().map(|slot| slot.name.as_str()).collect();
        for slot in &slots {
            let sum: usize = slot.ranges.iter().map(|(_, len)| *len).sum();
            assert_eq!(
                sum, slot.bytes,
                "{}: the ranges and payload disagree",
                slot.key
            );
        }
        assert!(names.contains("model.llm.embed.weight"));
        assert!(names.contains("model.llm.layers.0.mlp.experts.w13_weight"));
        assert!(names.contains("model.llm.layers.0.attn.wo_ud.weight"));
    }

    /// The output projections shard their contraction axis, so their payload is one row range per
    /// row of the matrix and a rank holds every row's own slice.
    #[test]
    fn a_contraction_axis_shard_is_one_range_per_row() {
        let config = released();
        let layout = TpLayout {
            rank: 2,
            world_size: 4,
        };
        let slots = rank_slots(&config, layout).expect("rank slots");
        let projection = slots
            .iter()
            .find(|slot| slot.name == "model.llm.layers.0.attn.wo_ud.weight")
            .expect("the output projection is planned");
        assert!(
            projection.ranges.len() > 1,
            "an inner-axis split is more than one range"
        );
        // [hidden, heads * head_dim] sharded on the contraction axis: every one of the hidden
        // rows contributes its own run of head_dim * heads / world columns.
        assert_eq!(projection.ranges.len(), config.hidden_size);
        // A range is in bytes, and a row's run is a quarter of the 6,144-wide contraction at two
        // bytes a value.
        let per_row = config.num_attention_heads * config.head_dim / 4 * 2;
        assert!(
            projection.ranges.iter().all(|(_, len)| *len == per_row),
            "every row's run is one rank's width"
        );
        assert_eq!(projection.bytes, config.hidden_size * per_row);
    }
}
