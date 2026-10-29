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

//! Pre-converted per-rank checkpoint format (`sparkflash convert`) and fast parallel direct-I/O
//! arena loader.
//!
//! **Attribution:** Pre-converting per-rank weight layouts and loading them via parallel
//! `O_DIRECT` reads for fast (~20-25 s) cold boot on DGX Spark draws on prior art in
//! [`b12x`](https://github.com/local-inference-lab/b12x) (`b12x/loader/_direct.c`) and
//! [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0; Luke Alonso
//! `@lukealonso`, `@original-el8`, and contributors). See the root `NOTICE` file.
//!
//! Instead of seeking and reading one tensor at a time on one thread, allocating a fresh
//! `Vec<u8>` per tensor, and re-tiling B12x weights and re-digesting on the CPU on every boot,
//! `sparkflash convert` writes one contiguous file per rank (`rank-{rank}-of-{world}.bin`) already
//! in the exact device arena layout:
//!
//! - Dense FP8 weights are already B12x-tiled (`TensorLayout::B12x`).
//! - Routed MoE down-projection (`w2`) weights and scales are already padded to the 640-wide K
//!   layout (`320` packed FP4 bytes and `20` E8M0 scale bytes per output row,
//!   `TensorLayout::MoePadded640`), eliminating the "+25.2 GiB/rank boot repack" noted in
//!   `resident_weights.rs`.
//! - Scales are stored immediately next to their weights (`foo.weight` followed by `foo.scale`).
//! - A small index (`name -> offset, length, layout, digest`) and `header_hash` + `file_digest`
//!   sit in a 4096-byte-aligned header block at the start of the file.
//!
//! At boot, loading checks the file size plus `header_hash` (a few kilobytes), allocates one
//! contiguous unified/host-mapped slab (or `mmap`s and registers the file with
//! `cuMemHostRegister`), and reads the payload straight into the arena using parallel `O_DIRECT`
//! `pread` in 8-16 MB chunks with 4-8 reads in flight. Full payload re-digesting is an opt-in
//! `--verify-weights` flag.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sparkflash_engine::safetensors::{CheckpointIndex, SafeTensorsSource};
use sparkflash_kernels::cuda_driver::DirectReadReport;
use sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes;

use crate::arena::{
    AllocationMode, ArenaAllocator, ArenaEvidence, ArenaSlice, DIRECT_CHUNK_BYTES,
    DIRECT_CONCURRENCY, FNV_OFFSET, FNV_PRIME, GroupArena, ResidentArena, TensorLayout, align_up,
    b12x_layout, elapsed_ms, fnv1a, moe_padded_640_layout, order_scales_next_to_weights,
    pad_moe_rows_640, tile_b12x,
};
use crate::resident::{HeaderEntry, Placement, ResidentGroup, ResidentPlan};

/// Magic bytes at offset 0 of every converted per-rank file.
pub const CONVERTED_MAGIC: &[u8; 8] = b"SPKFRANK";
/// Format schema version.
pub const CONVERTED_VERSION: u32 = 1;
/// Fixed binary preamble length in bytes before the JSON index.
pub const PREAMBLE_BYTES: usize = 64;
/// Alignment of the data section start and padded length, matching 4 KiB page / `O_DIRECT` blocks.
pub const DATA_ALIGN_BYTES: usize = 4096;
/// Manifest filename written in the converted output directory.
pub const CONVERTED_MANIFEST_NAME: &str = "sparkflash-converted-manifest.json";

/// Canonical filename for one rank's converted file.
#[must_use]
pub fn rank_file_name(rank: usize, world: usize) -> String {
    format!("rank-{rank}-of-{world}.bin")
}

/// Returns the path to a rank's converted file inside `dir` if it exists on disk.
#[must_use]
pub fn find_converted_rank_file(dir: &Path, rank: usize, world: usize) -> Option<PathBuf> {
    let candidate = dir.join(rank_file_name(rank, world));
    candidate.is_file().then_some(candidate)
}

/// Metadata for one tensor stored inside a converted rank file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedTensorEntry {
    /// Checkpoint tensor name.
    pub name: String,
    /// Resident group.
    pub group: ResidentGroup,
    /// Byte offset relative to `data_offset` (and therefore inside the single arena slab).
    pub slab_offset: usize,
    /// Byte offset inside the group's sub-slab (`ArenaSlice::offset`).
    pub group_offset: usize,
    /// Payload length in bytes (tiled or 640-padded when applicable).
    pub bytes: usize,
    /// Unpadded rank shard bytes from the checkpoint plan.
    pub stored_bytes: u64,
    /// Device tensor layout (`Stored`, `B12x`, or `MoePadded640`).
    pub layout: TensorLayout,
    /// Safetensors dtype word.
    pub dtype: String,
    /// Full checkpoint stored shape.
    pub stored_shape: Vec<usize>,
    /// Per-rank shard shape (with `[rows, 320]` / `[rows, 20]` for `MoePadded640`).
    pub shard_shape: Vec<usize>,
    /// FNV-1a 64 digest of the stored payload bytes, computed at conversion time.
    pub digest: u64,
}

/// Metadata for one `ResidentGroup` sub-slab inside a converted rank file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedGroupEntry {
    /// Design group.
    pub group: ResidentGroup,
    /// Starting byte offset relative to `data_offset`.
    pub slab_offset: usize,
    /// Group sub-slab size in bytes.
    pub bytes: usize,
    /// Number of tensors in this group.
    pub tensors: usize,
    /// Number of B12x-tiled tensors in this group.
    pub tiled: usize,
    /// FNV-1a 64 digest of the group's tensor digests in layout order.
    pub digest: u64,
}

/// JSON index stored in the header region (`64..64 + index_len`) of a converted rank file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedRankHeader {
    /// Format version (`CONVERTED_VERSION`).
    pub version: u32,
    /// TP rank (`0..world`).
    pub rank: usize,
    /// TP world size.
    pub world: usize,
    /// `ResidentPlan::plan_hash()` of the source plan.
    pub plan_hash: u64,
    /// Total unpadded shard bytes across all resident tensors.
    pub stored_bytes: u64,
    /// Total converted payload bytes (including B12x and 640-wide MoE padding).
    pub payload_bytes: u64,
    /// Total 256-byte-aligned arena bytes (`data_bytes`).
    pub allocated_bytes: u64,
    /// Total 4096-byte-aligned payload bytes on disk (`padded_data_bytes`).
    pub padded_data_bytes: u64,
    /// Byte offset in the file where the contiguous arena data begins (multiple of 4096).
    pub data_offset: u64,
    /// Count of B12x-tiled tensors.
    pub tiled_tensors: usize,
    /// Count of 640-wide padded MoE `w2` tensors (`w2.weight` + `w2.scale`).
    pub padded_moe_tensors: usize,
    /// FNV-1a 64 digest of all tensor digests in slab order.
    pub file_digest: u64,
    /// Group sub-slabs in layout order.
    pub groups: Vec<ConvertedGroupEntry>,
    /// All resident tensors in slab order (with scales immediately adjacent to their weights).
    pub tensors: Vec<ConvertedTensorEntry>,
}

/// Decoded 64-byte binary preamble at offset 0 of a converted rank file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertedPreamble {
    /// Schema version.
    pub version: u32,
    /// Rank id.
    pub rank: usize,
    /// World size.
    pub world: usize,
    /// Byte length of the serialized JSON header starting at byte 64.
    pub index_len: u64,
    /// 4096-aligned byte offset where the arena data starts.
    pub data_offset: u64,
    /// Logical arena size (`allocated_bytes`).
    pub data_bytes: u64,
    /// 4096-aligned arena data size on disk.
    pub padded_data_bytes: u64,
    /// FNV-1a 64 hash of the JSON header bytes `64..64 + index_len`.
    pub header_hash: u64,
    /// FNV-1a 64 digest of the arena payload.
    pub file_digest: u64,
}

impl ConvertedPreamble {
    fn encode(&self) -> [u8; PREAMBLE_BYTES] {
        let mut out = [0u8; PREAMBLE_BYTES];
        out[0..8].copy_from_slice(CONVERTED_MAGIC);
        out[8..12].copy_from_slice(&self.version.to_le_bytes());
        out[12..16].copy_from_slice(&(self.rank as u32).to_le_bytes());
        out[16..20].copy_from_slice(&(self.world as u32).to_le_bytes());
        out[20..24].copy_from_slice(&0u32.to_le_bytes());
        out[24..32].copy_from_slice(&self.index_len.to_le_bytes());
        out[32..40].copy_from_slice(&self.data_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.data_bytes.to_le_bytes());
        out[48..56].copy_from_slice(&self.padded_data_bytes.to_le_bytes());
        out[56..64].copy_from_slice(&self.header_hash.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8; PREAMBLE_BYTES], path: &Path) -> Result<Self, String> {
        if &bytes[0..8] != CONVERTED_MAGIC {
            return Err(format!(
                "{}: invalid magic {:?}, expected {:?}",
                path.display(),
                &bytes[0..8],
                CONVERTED_MAGIC
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != CONVERTED_VERSION {
            return Err(format!(
                "{}: unsupported converted version {version}, expected {CONVERTED_VERSION}",
                path.display()
            ));
        }
        let rank = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let world = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let index_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let data_bytes = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        let padded_data_bytes = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
        let header_hash = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
        Ok(Self {
            version,
            rank,
            world,
            index_len,
            data_offset,
            data_bytes,
            padded_data_bytes,
            header_hash,
            file_digest: 0,
        })
    }
}

/// Summary of one converted rank file in `sparkflash-converted-manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedRankSummary {
    /// Rank index (`0..world`).
    pub rank: usize,
    /// World size.
    pub world: usize,
    /// Relative filename (`rank-{rank}-of-{world}.bin`).
    pub file_name: String,
    /// Total file size in bytes (`data_offset + padded_data_bytes`).
    pub file_bytes: u64,
    /// Byte offset where the 4096-aligned arena data starts.
    pub data_offset: u64,
    /// Logical arena bytes (`allocated_bytes`).
    pub allocated_bytes: u64,
    /// Payload bytes before 256-byte inter-tensor alignment padding.
    pub payload_bytes: u64,
    /// Unpadded checkpoint shard bytes.
    pub stored_bytes: u64,
    /// Resident tensor count.
    pub tensors: usize,
    /// B12x-tiled tensor count.
    pub tiled_tensors: usize,
    /// 640-wide padded MoE `w2` tensor count.
    pub padded_moe_tensors: usize,
    /// FNV-1a 64 hash of the serialized index header.
    pub header_hash: String,
    /// FNV-1a 64 digest of the converted file payload.
    pub file_digest: String,
    /// Conversion wall-clock time in milliseconds.
    pub convert_ms: u64,
}

/// Top-level manifest written to `sparkflash-converted-manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedManifest {
    /// Format identifier (`"sparkflash-converted-checkpoint"`).
    pub format: String,
    /// Format version (`1`).
    pub version: u32,
    /// TP world size.
    pub world_size: usize,
    /// Whether routed MoE `w2` experts were padded to 640-wide K.
    pub pad_moe_640: bool,
    /// Converted rank summaries.
    pub ranks: Vec<ConvertedRankSummary>,
}

/// Options for `convert_checkpoint` / `convert_rank`.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Path to the source safetensors checkpoint directory.
    pub checkpoint_dir: PathBuf,
    /// Directory where `rank-{rank}-of-{world}.bin` and the manifest are written.
    pub output_dir: PathBuf,
    /// TP world size (default 4).
    pub world_size: usize,
    /// Specific ranks to convert (empty means all `0..world_size`).
    pub ranks: Vec<usize>,
    /// Pad routed MoE `w2.weight` and `w2.scale` rows from 576-wide (`288`/`18` bytes) to
    /// 640-wide (`320`/`20` bytes) during conversion (default `true`).
    pub pad_moe_640: bool,
    /// Verify the written file digest after writing (default `true`).
    pub verify: bool,
}

impl ConvertOptions {
    /// Default options for converting `checkpoint_dir` into `output_dir` for TP=4.
    #[must_use]
    pub fn new(checkpoint_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            output_dir: output_dir.into(),
            world_size: 4,
            ranks: Vec::new(),
            pad_moe_640: true,
            verify: true,
        }
    }

    /// Returns the sorted list of ranks to convert.
    pub fn selected_ranks(&self) -> Result<Vec<usize>, String> {
        if self.world_size == 0 {
            return Err("world_size must be at least 1".to_string());
        }
        let mut ranks = if self.ranks.is_empty() {
            (0..self.world_size).collect()
        } else {
            self.ranks.clone()
        };
        ranks.sort_unstable();
        ranks.dedup();
        for &rank in &ranks {
            if rank >= self.world_size {
                return Err(format!(
                    "rank {rank} is outside world_size {}",
                    self.world_size
                ));
            }
        }
        Ok(ranks)
    }
}

/// Options controlling how a converted rank file is loaded at boot.
#[derive(Debug, Clone, Copy)]
pub struct ConvertedLoadOptions {
    /// Allocation mode (`UnifiedSingle`, `HostMappedSingle`, `MmapRegistered`, or `Device`).
    pub mode: AllocationMode,
    /// Chunk size in bytes for parallel `O_DIRECT` / `pread` (default 16 MiB).
    pub chunk_bytes: usize,
    /// Number of concurrent `pread` worker threads in flight (default 8).
    pub concurrency: usize,
    /// Whether to open the converted file with `O_DIRECT` (default `true`).
    pub direct_io: bool,
    /// Whether to re-compute and verify every tensor's FNV-1a digest at boot (`--verify-weights`,
    /// default `false`; boot always checks file size + `header_hash`).
    pub verify_weights: bool,
}

impl Default for ConvertedLoadOptions {
    fn default() -> Self {
        Self {
            mode: AllocationMode::UnifiedSingle,
            chunk_bytes: DIRECT_CHUNK_BYTES,
            concurrency: DIRECT_CONCURRENCY,
            direct_io: true,
            verify_weights: false,
        }
    }
}

/// Evidence returned alongside the `ResidentArena` when loaded from a converted rank file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedLoadReport {
    /// Path of the converted file loaded.
    pub path: PathBuf,
    /// Parsed header from the converted file.
    pub header: ConvertedRankHeader,
    /// Header hash verified at boot.
    pub header_hash: u64,
    /// Direct I/O read report (None when loaded via `MmapRegistered`).
    pub read_report: Option<DirectReadReport>,
    /// Whether full payload digest verification (`--verify-weights`) was performed.
    pub weights_verified: bool,
}

fn align_up_page(bytes: usize) -> usize {
    match bytes % DATA_ALIGN_BYTES {
        0 => bytes,
        rem => bytes + (DATA_ALIGN_BYTES - rem),
    }
}

/// Loads all safetensors headers once from `checkpoint_dir` and returns `(SafeTensorsSource, Vec<HeaderEntry>)`.
pub fn load_checkpoint_headers_once(
    checkpoint_dir: &Path,
) -> Result<(SafeTensorsSource, Vec<HeaderEntry>), String> {
    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    let index = CheckpointIndex::load(&index_path)
        .map_err(|error| format!("read {}: {error}", index_path.display()))?;
    let source = SafeTensorsSource::open(checkpoint_dir, &index_path)
        .map_err(|error| format!("open checkpoint {}: {error}", checkpoint_dir.display()))?;
    source
        .preload_headers()
        .map_err(|error| format!("preload headers in {}: {error}", checkpoint_dir.display()))?;
    let mut entries = Vec::with_capacity(index.weight_map.len());
    for name in index.names() {
        let header = source
            .header(name)
            .map_err(|error| format!("{name}: {error}"))?;
        let tensor = header
            .tensor(name)
            .map_err(|error| format!("{name}: {error}"))?;
        entries.push(HeaderEntry::new(
            name.to_string(),
            tensor.dtype.clone(),
            tensor.shape.clone(),
        ));
    }
    Ok((source, entries))
}

/// Converts every selected rank in `options` from safetensors into per-rank contiguous files and
/// writes `sparkflash-converted-manifest.json`.
pub fn convert_checkpoint(options: &ConvertOptions) -> Result<ConvertedManifest, String> {
    let ranks = options.selected_ranks()?;
    fs::create_dir_all(&options.output_dir)
        .map_err(|error| format!("create {}: {error}", options.output_dir.display()))?;
    let (source, entries) = load_checkpoint_headers_once(&options.checkpoint_dir)?;
    let mut summaries = Vec::with_capacity(ranks.len());
    for rank in ranks {
        let summary = convert_rank_with_source(
            &source,
            entries.clone(),
            rank,
            options.world_size,
            &options.output_dir,
            options.pad_moe_640,
            options.verify,
        )?;
        summaries.push(summary);
    }
    let manifest = ConvertedManifest {
        format: "sparkflash-converted-checkpoint".to_string(),
        version: CONVERTED_VERSION,
        world_size: options.world_size,
        pad_moe_640: options.pad_moe_640,
        ranks: summaries,
    };
    let manifest_path = options.output_dir.join(CONVERTED_MANIFEST_NAME);
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialise manifest: {error}"))?;
    fs::write(&manifest_path, manifest_bytes)
        .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
    Ok(manifest)
}

/// Converts a single rank into `output_dir/rank-{rank}-of-{world}.bin`.
#[allow(clippy::too_many_lines)]
pub fn convert_rank_with_source(
    source: &SafeTensorsSource,
    entries: Vec<HeaderEntry>,
    rank: usize,
    world: usize,
    output_dir: &Path,
    pad_moe_640: bool,
    verify: bool,
) -> Result<ConvertedRankSummary, String> {
    let started = Instant::now();
    let plan = ResidentPlan::plan(rank, world, entries)
        .map_err(|error| format!("plan rank {rank}/{world}: {error}"))?;
    let shard_plan = plan
        .shard_plan()
        .map_err(|error| format!("shard plan rank {rank}/{world}: {error}"))?;

    struct PlannedSlot {
        planned_index: usize,
        name: String,
        group: ResidentGroup,
        stored_bytes: u64,
        layout: TensorLayout,
        b12x: Option<(usize, usize)>,
        moe_pad: Option<(usize, usize, usize)>,
        payload_bytes: usize,
        dtype: String,
        stored_shape: Vec<usize>,
        shard_shape: Vec<usize>,
    }

    let planned_shards = plan.shards();
    let mut raw_slots: Vec<PlannedSlot> = Vec::new();
    for (index, planned) in planned_shards.iter().enumerate() {
        if planned.placement != Placement::Resident {
            continue;
        }
        let b12x = b12x_layout(planned);
        let moe_pad = if pad_moe_640 {
            moe_padded_640_layout(planned)
        } else {
            None
        };
        let (layout, payload_bytes, shard_shape) = if let Some((cols, inner)) = b12x {
            (
                TensorLayout::B12x,
                b12x_buffer_bytes(cols, inner),
                planned.shard_shape.clone(),
            )
        } else if let Some((rows, _unpadded, padded_row)) = moe_pad {
            (
                TensorLayout::MoePadded640,
                rows * padded_row,
                vec![rows, padded_row],
            )
        } else {
            let len = usize::try_from(planned.shard_bytes)
                .map_err(|_| format!("{}: shard bytes do not fit usize", planned.name))?;
            (TensorLayout::Stored, len, planned.shard_shape.clone())
        };
        raw_slots.push(PlannedSlot {
            planned_index: index,
            name: planned.name.clone(),
            group: planned.group,
            stored_bytes: planned.shard_bytes,
            layout,
            b12x,
            moe_pad,
            payload_bytes,
            dtype: planned.dtype.clone(),
            stored_shape: planned.stored_shape.clone(),
            shard_shape,
        });
    }

    let mut by_group: BTreeMap<ResidentGroup, Vec<usize>> = BTreeMap::new();
    for (idx, slot) in raw_slots.iter().enumerate() {
        by_group.entry(slot.group).or_default().push(idx);
    }
    // Order scales immediately next to their weights inside every group.
    for indices in by_group.values_mut() {
        let local_order =
            order_scales_next_to_weights(indices.len(), |i| raw_slots[indices[i]].name.as_str());
        let reordered: Vec<usize> = local_order.into_iter().map(|i| indices[i]).collect();
        *indices = reordered;
    }

    // Pass 1: compute exact offsets and dummy index structure so we know the exact header size and
    // 4096-byte aligned `data_offset` before streaming the payload bytes.
    let mut group_entries: Vec<ConvertedGroupEntry> = Vec::with_capacity(by_group.len());
    let mut tensor_entries: Vec<ConvertedTensorEntry> = Vec::with_capacity(raw_slots.len());
    let mut slab_cursor = 0usize;
    let mut total_stored_bytes = 0u64;
    let mut total_payload_bytes = 0u64;
    let mut tiled_tensors = 0usize;
    let mut padded_moe_tensors = 0usize;

    for (group, indices) in &by_group {
        let group_slab_offset = slab_cursor;
        let mut group_offset = 0usize;
        let mut group_tiled = 0usize;
        for &slot_idx in indices {
            let slot = &raw_slots[slot_idx];
            if slot.layout == TensorLayout::B12x {
                group_tiled += 1;
                tiled_tensors += 1;
            } else if slot.layout == TensorLayout::MoePadded640 {
                padded_moe_tensors += 1;
            }
            total_stored_bytes = total_stored_bytes.saturating_add(slot.stored_bytes);
            total_payload_bytes = total_payload_bytes.saturating_add(slot.payload_bytes as u64);
            tensor_entries.push(ConvertedTensorEntry {
                name: slot.name.clone(),
                group: *group,
                slab_offset: group_slab_offset + group_offset,
                group_offset,
                bytes: slot.payload_bytes,
                stored_bytes: slot.stored_bytes,
                layout: slot.layout,
                dtype: slot.dtype.clone(),
                stored_shape: slot.stored_shape.clone(),
                shard_shape: slot.shard_shape.clone(),
                digest: u64::MAX,
            });
            group_offset = align_up(group_offset + slot.payload_bytes);
        }
        group_entries.push(ConvertedGroupEntry {
            group: *group,
            slab_offset: group_slab_offset,
            bytes: group_offset,
            tensors: indices.len(),
            tiled: group_tiled,
            digest: u64::MAX,
        });
        slab_cursor = align_up(group_slab_offset + group_offset);
    }

    let allocated_bytes = slab_cursor as u64;
    let padded_data_bytes = align_up_page(slab_cursor) as u64;
    let placeholder_header = ConvertedRankHeader {
        version: CONVERTED_VERSION,
        rank,
        world,
        plan_hash: plan.plan_hash(),
        stored_bytes: total_stored_bytes,
        payload_bytes: total_payload_bytes,
        allocated_bytes,
        padded_data_bytes,
        data_offset: DATA_ALIGN_BYTES as u64,
        tiled_tensors,
        padded_moe_tensors,
        file_digest: u64::MAX,
        groups: group_entries.clone(),
        tensors: tensor_entries.clone(),
    };
    let placeholder_json = serde_json::to_vec(&placeholder_header)
        .map_err(|error| format!("serialise header placeholder: {error}"))?;
    // Reserve generous slack (one extra 4096 page) so final JSON with real digests always fits
    // inside `data_offset`.
    let data_offset = align_up_page(PREAMBLE_BYTES + placeholder_json.len() + 1024) as u64;

    fs::create_dir_all(output_dir)
        .map_err(|error| format!("create {}: {error}", output_dir.display()))?;
    let file_name = rank_file_name(rank, world);
    let out_path = output_dir.join(&file_name);
    let file = File::create(&out_path)
        .map_err(|error| format!("create {}: {error}", out_path.display()))?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    // Seek to `data_offset` and stream each tensor's converted payload sequentially.
    writer
        .seek(SeekFrom::Start(data_offset))
        .map_err(|error| format!("seek {}: {error}", out_path.display()))?;

    let max_stored = raw_slots
        .iter()
        .filter_map(|slot| usize::try_from(slot.stored_bytes).ok())
        .max()
        .unwrap_or(0);
    let mut scratch = vec![0u8; max_stored];
    let zeros = [0u8; DATA_ALIGN_BYTES];

    let mut file_digest = FNV_OFFSET;
    let mut tensor_cursor = 0usize;
    let mut written_in_data = 0usize;

    for (group_idx, (_group, indices)) in by_group.iter().enumerate() {
        let mut group_digest = FNV_OFFSET;
        for &slot_idx in indices {
            let slot = &raw_slots[slot_idx];
            let entry = &mut tensor_entries[tensor_cursor];
            tensor_cursor += 1;

            if written_in_data < entry.slab_offset {
                let pad = entry.slab_offset - written_in_data;
                write_zeros(&mut writer, &zeros, pad, &out_path)?;
                written_in_data += pad;
            }
            let _ = &planned_shards[slot.planned_index];
            let stored_len = usize::try_from(slot.stored_bytes)
                .map_err(|_| format!("{}: stored bytes do not fit usize", slot.name))?;
            shard_plan
                .read_into(source, &slot.name, &mut scratch[..stored_len])
                .map_err(|error| format!("convert read {}: {error}", slot.name))?;
            let raw = &scratch[..stored_len];
            let transformed: Option<Vec<u8>> = if let Some((cols, inner)) = slot.b12x {
                Some(
                    tile_b12x(raw, cols, inner)
                        .map_err(|error| format!("convert tile {}: {error}", slot.name))?,
                )
            } else if let Some((rows, unpadded_row, padded_row)) = slot.moe_pad {
                Some(
                    pad_moe_rows_640(raw, rows, unpadded_row, padded_row)
                        .map_err(|error| format!("convert pad {}: {error}", slot.name))?,
                )
            } else {
                None
            };
            let payload: &[u8] = transformed.as_deref().unwrap_or(raw);
            if payload.len() != entry.bytes {
                return Err(format!(
                    "{}: converted payload is {} bytes, expected {}",
                    slot.name,
                    payload.len(),
                    entry.bytes
                ));
            }
            writer
                .write_all(payload)
                .map_err(|error| format!("write {}: {error}", out_path.display()))?;
            written_in_data += payload.len();

            let slice_digest = fnv1a(payload);
            entry.digest = slice_digest;
            group_digest ^= slice_digest;
            group_digest = group_digest.wrapping_mul(FNV_PRIME);
            file_digest ^= slice_digest;
            file_digest = file_digest.wrapping_mul(FNV_PRIME);
        }
        group_entries[group_idx].digest = group_digest;
    }

    let target_data_len = padded_data_bytes as usize;
    if written_in_data < target_data_len {
        let pad = target_data_len - written_in_data;
        write_zeros(&mut writer, &zeros, pad, &out_path)?;
    }

    let final_header = ConvertedRankHeader {
        version: CONVERTED_VERSION,
        rank,
        world,
        plan_hash: plan.plan_hash(),
        stored_bytes: total_stored_bytes,
        payload_bytes: total_payload_bytes,
        allocated_bytes,
        padded_data_bytes,
        data_offset,
        tiled_tensors,
        padded_moe_tensors,
        file_digest,
        groups: group_entries,
        tensors: tensor_entries,
    };
    let index_bytes = serde_json::to_vec(&final_header)
        .map_err(|error| format!("serialise converted header: {error}"))?;
    if PREAMBLE_BYTES + index_bytes.len() > data_offset as usize {
        return Err(format!(
            "{}: header grew to {} bytes, exceeding reserved data_offset {data_offset}",
            out_path.display(),
            PREAMBLE_BYTES + index_bytes.len()
        ));
    }
    let header_hash = fnv1a(&index_bytes);
    let preamble = ConvertedPreamble {
        version: CONVERTED_VERSION,
        rank,
        world,
        index_len: index_bytes.len() as u64,
        data_offset,
        data_bytes: allocated_bytes,
        padded_data_bytes,
        header_hash,
        file_digest,
    };

    writer
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("seek start {}: {error}", out_path.display()))?;
    writer
        .write_all(&preamble.encode())
        .map_err(|error| format!("write preamble {}: {error}", out_path.display()))?;
    writer
        .write_all(&index_bytes)
        .map_err(|error| format!("write header {}: {error}", out_path.display()))?;
    let header_pad = (data_offset as usize) - (PREAMBLE_BYTES + index_bytes.len());
    write_zeros(&mut writer, &zeros, header_pad, &out_path)?;
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", out_path.display()))?;
    drop(writer);

    if verify {
        verify_converted_file(&out_path)?;
    }

    let convert_ms = elapsed_ms(started);
    Ok(ConvertedRankSummary {
        rank,
        world,
        file_name,
        file_bytes: data_offset + padded_data_bytes,
        data_offset,
        allocated_bytes,
        payload_bytes: total_payload_bytes,
        stored_bytes: total_stored_bytes,
        tensors: final_header.tensors.len(),
        tiled_tensors,
        padded_moe_tensors,
        header_hash: format!("{header_hash:#018x}"),
        file_digest: format!("{file_digest:#018x}"),
        convert_ms,
    })
}

fn write_zeros(
    writer: &mut BufWriter<File>,
    zeros: &[u8; DATA_ALIGN_BYTES],
    mut remaining: usize,
    path: &Path,
) -> Result<(), String> {
    while remaining > 0 {
        let step = remaining.min(zeros.len());
        writer
            .write_all(&zeros[..step])
            .map_err(|error| format!("pad {}: {error}", path.display()))?;
        remaining -= step;
    }
    Ok(())
}

/// Performs the cheap boot check on a converted rank file: verifies the file size against
/// `data_offset + padded_data_bytes` and verifies `fnv1a(index_bytes) == header_hash`.
///
/// Touching no weight bytes, this finishes in under a millisecond.
pub fn inspect_converted_rank(
    path: &Path,
) -> Result<(ConvertedPreamble, ConvertedRankHeader), String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();
    if file_len < PREAMBLE_BYTES as u64 {
        return Err(format!(
            "{}: file is {} bytes, shorter than the {PREAMBLE_BYTES}-byte preamble",
            path.display(),
            file_len
        ));
    }
    let mut preamble_buf = [0u8; PREAMBLE_BYTES];
    file.read_exact(&mut preamble_buf)
        .map_err(|error| format!("read preamble {}: {error}", path.display()))?;
    let mut preamble = ConvertedPreamble::decode(&preamble_buf, path)?;
    let expected_file_len = preamble
        .data_offset
        .checked_add(preamble.padded_data_bytes)
        .ok_or_else(|| format!("{}: file size arithmetic overflow", path.display()))?;
    if file_len != expected_file_len {
        return Err(format!(
            "{}: file size is {file_len} bytes, but header declares {expected_file_len} ({} header + {} data)",
            path.display(),
            preamble.data_offset,
            preamble.padded_data_bytes
        ));
    }
    if !(preamble.data_offset as usize).is_multiple_of(DATA_ALIGN_BYTES)
        || !(preamble.padded_data_bytes as usize).is_multiple_of(DATA_ALIGN_BYTES)
    {
        return Err(format!(
            "{}: data_offset ({}) or padded_data_bytes ({}) is not {DATA_ALIGN_BYTES}-byte aligned",
            path.display(),
            preamble.data_offset,
            preamble.padded_data_bytes
        ));
    }
    let index_end = (PREAMBLE_BYTES as u64)
        .checked_add(preamble.index_len)
        .ok_or_else(|| format!("{}: index_len overflow", path.display()))?;
    if index_end > preamble.data_offset {
        return Err(format!(
            "{}: index end {index_end} exceeds data_offset {}",
            path.display(),
            preamble.data_offset
        ));
    }
    let index_len = usize::try_from(preamble.index_len)
        .map_err(|_| format!("{}: index_len does not fit usize", path.display()))?;
    let mut index_bytes = vec![0u8; index_len];
    file.read_exact(&mut index_bytes)
        .map_err(|error| format!("read index {}: {error}", path.display()))?;
    let measured_header_hash = fnv1a(&index_bytes);
    if measured_header_hash != preamble.header_hash {
        return Err(format!(
            "{}: header hash mismatch (measured {measured_header_hash:#018x}, expected {:#018x})",
            path.display(),
            preamble.header_hash
        ));
    }
    let header: ConvertedRankHeader = serde_json::from_slice(&index_bytes)
        .map_err(|error| format!("parse index in {}: {error}", path.display()))?;
    preamble.file_digest = header.file_digest;
    Ok((preamble, header))
}

/// Re-reads and verifies every tensor's FNV-1a digest and the overall `file_digest` of a
/// converted rank file on disk. Called at conversion time and when `--verify-weights` is passed.
pub fn verify_converted_file(path: &Path) -> Result<ConvertedRankHeader, String> {
    let (preamble, header) = inspect_converted_rank(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let max_tensor = header.tensors.iter().map(|t| t.bytes).max().unwrap_or(0);
    let mut buf = vec![0u8; max_tensor];
    let mut rolling_digest = FNV_OFFSET;
    for entry in &header.tensors {
        let at = preamble.data_offset + entry.slab_offset as u64;
        file.read_exact_at(&mut buf[..entry.bytes], at)
            .map_err(|error| {
                format!(
                    "verify read {} ({}) at {at}: {error}",
                    entry.name,
                    path.display()
                )
            })?;
        let digest = fnv1a(&buf[..entry.bytes]);
        if digest != entry.digest {
            return Err(format!(
                "{}: tensor {} digest mismatch (measured {digest:#018x}, header {:#018x})",
                path.display(),
                entry.name,
                entry.digest
            ));
        }
        rolling_digest ^= digest;
        rolling_digest = rolling_digest.wrapping_mul(FNV_PRIME);
    }
    if rolling_digest != header.file_digest {
        return Err(format!(
            "{}: file digest mismatch (measured {rolling_digest:#018x}, header {:#018x})",
            path.display(),
            header.file_digest
        ));
    }
    Ok(header)
}

/// Loads a converted per-rank file into a `ResidentArena`.
///
/// Boot performs the cheap size + `header_hash` check first, makes one contiguous allocation (or
/// `mmap`s and registers the file with `cuMemHostRegister` when `options.mode ==
/// AllocationMode::MmapRegistered`), and reads the payload in parallel 8-16 MB `O_DIRECT` chunks
/// with no CPU re-tiling and no CPU digesting. When `options.verify_weights` is true, it also
/// verifies every tensor digest and `file_digest`.
pub fn load_converted_rank_arena(
    allocator: &mut dyn ArenaAllocator,
    path: &Path,
    expected_rank: usize,
    expected_world: usize,
    options: ConvertedLoadOptions,
) -> Result<(ResidentArena, ConvertedLoadReport), String> {
    let (preamble, header) = inspect_converted_rank(path)?;
    if header.rank != expected_rank || header.world != expected_world {
        return Err(format!(
            "{}: converted file is rank {}/{}, but caller requested rank {expected_rank}/{expected_world}",
            path.display(),
            header.rank,
            header.world
        ));
    }
    let padded_len = usize::try_from(header.padded_data_bytes)
        .map_err(|_| format!("{}: padded_data_bytes does not fit usize", path.display()))?;

    let allocate_started = Instant::now();
    let (base, allocate_ms, upload_ms, read_report) =
        if options.mode == AllocationMode::MmapRegistered {
            let base = allocator.mmap_from_file(path, preamble.data_offset, padded_len)?;
            let alloc_ms = elapsed_ms(allocate_started);
            (base, alloc_ms, 0, None)
        } else {
            let base = allocator.allocate(padded_len).map_err(|error| {
                format!("arena converted single slab ({padded_len} bytes): {error}")
            })?;
            let alloc_ms = elapsed_ms(allocate_started);
            let upload_started = Instant::now();
            let report = allocator.upload_from_file_parallel(
                &base,
                path,
                preamble.data_offset,
                padded_len,
                options.chunk_bytes,
                options.concurrency,
                options.direct_io,
            )?;
            let up_ms = elapsed_ms(upload_started);
            (base, alloc_ms, up_ms, Some(report))
        };

    if options.verify_weights {
        verify_converted_file(path)?;
    }

    let mut slices = BTreeMap::new();
    for entry in &header.tensors {
        slices.insert(
            entry.name.clone(),
            ArenaSlice {
                name: entry.name.clone(),
                group: entry.group,
                ptr: base.offset(entry.slab_offset),
                offset: entry.group_offset,
                bytes: entry.bytes,
                layout: entry.layout,
                digest: entry.digest,
            },
        );
    }
    let groups: Vec<GroupArena> = header
        .groups
        .iter()
        .map(|g| GroupArena {
            group: g.group,
            bytes: g.bytes,
            tensors: g.tensors,
            tiled: g.tiled,
            digest: g.digest,
        })
        .collect();
    let evidence = ArenaEvidence {
        mode: options.mode,
        tensors: header.tensors.len(),
        tiled_tensors: header.tiled_tensors,
        stored_bytes: header.stored_bytes,
        bytes: header.payload_bytes,
        allocated_bytes: header.allocated_bytes,
        allocate_ms,
        upload_ms,
        digest: header.file_digest,
    };
    let arena =
        ResidentArena::from_converted_parts(slices, groups, header.allocated_bytes, evidence);
    let report = ConvertedLoadReport {
        path: path.to_path_buf(),
        header,
        header_hash: preamble.header_hash,
        read_report,
        weights_verified: options.verify_weights,
    };
    Ok((arena, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_kernels::cuda_driver::Deviceptr;

    #[derive(Default)]
    struct TestAllocator {
        next: u64,
        uploaded: Vec<(u64, Vec<u8>)>,
    }

    impl TestAllocator {
        fn new() -> Self {
            Self {
                next: 0x10_0000,
                uploaded: Vec::new(),
            }
        }
    }

    impl ArenaAllocator for TestAllocator {
        fn allocate(&mut self, bytes: usize) -> Result<Deviceptr, String> {
            let ptr = Deviceptr(self.next);
            self.next += bytes as u64;
            Ok(ptr)
        }

        fn upload(&mut self, destination: &Deviceptr, bytes: &[u8]) -> Result<(), String> {
            self.uploaded.push((destination.0, bytes.to_vec()));
            Ok(())
        }
    }

    fn write_synthetic_checkpoint(dir: &Path) {
        // Build a synthetic safetensors shard with:
        // - B12x attention projection (`layers.0.attn.wq_a.weight` [32, 32] F8_E4M3 + `.scale` [1, 1] F8_E8M0)
        // - Routed MoE down projection (`layers.0.ffn.experts.0.w2.weight` [4, 1152] I8 -> [4, 288] per rank -> padded to [4, 320])
        //   and `.w2.scale` [4, 72] F8_E8M0 -> [4, 18] per rank -> padded to [4, 20]
        // - Tail norm (`norm.weight` [32] BF16)
        let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
            (
                "layers.0.attn.wq_a.scale",
                "F8_E8M0",
                vec![1, 1],
                vec![0x7f],
            ),
            (
                "layers.0.attn.wq_a.weight",
                "F8_E4M3",
                vec![32, 32],
                (0..1024usize).map(|i| (i % 251) as u8).collect(),
            ),
            (
                "layers.0.ffn.experts.0.w2.scale",
                "F8_E8M0",
                vec![4, 72],
                (0..4 * 72).map(|i| ((i % 120) + 1) as u8).collect(),
            ),
            (
                "layers.0.ffn.experts.0.w2.weight",
                "I8",
                vec![4, 1152],
                (0..4 * 1152).map(|i| (i % 253) as u8).collect(),
            ),
            ("norm.weight", "BF16", vec![32], vec![1u8; 64]),
        ];

        let shard_name = "model-00001-of-00001.safetensors";
        let mut header = String::from("{");
        let mut offset = 0u64;
        for (idx, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if idx > 0 {
                header.push(',');
            }
            let begin = offset;
            offset += bytes.len() as u64;
            let shape_text = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\"data_offsets\":[{begin},{offset}]}}"
            ));
        }
        header.push('}');
        let mut header_bytes = header.into_bytes();
        while (8 + header_bytes.len()) % 8 != 0 {
            header_bytes.push(b' ');
        }
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        for (_, _, _, bytes) in &tensors {
            file_bytes.extend_from_slice(bytes);
        }
        fs::write(dir.join(shard_name), file_bytes).expect("write shard");

        let mut weight_map = String::new();
        for (idx, (name, _, _, _)) in tensors.iter().enumerate() {
            if idx > 0 {
                weight_map.push(',');
            }
            weight_map.push_str(&format!("\"{name}\":\"{shard_name}\""));
        }
        let index_text = format!("{{\"metadata\":{{}},\"weight_map\":{{{weight_map}}}}}");
        fs::write(dir.join("model.safetensors.index.json"), index_text).expect("write index");
    }

    #[test]
    fn convert_writes_b12x_and_640_padded_moe_with_scales_adjacent_and_loads_via_parallel_pread() {
        let ckpt_dir = tempfile::tempdir().expect("ckpt_dir");
        let out_dir = tempfile::tempdir().expect("out_dir");
        write_synthetic_checkpoint(ckpt_dir.path());

        let mut options = ConvertOptions::new(ckpt_dir.path(), out_dir.path());
        options.ranks = vec![0, 1];
        let manifest = convert_checkpoint(&options).expect("convert_checkpoint");
        assert_eq!(manifest.ranks.len(), 2);
        assert_eq!(manifest.ranks[0].tiled_tensors, 1);
        assert_eq!(manifest.ranks[0].padded_moe_tensors, 2);

        let rank0_path = out_dir.path().join("rank-0-of-4.bin");
        let (_preamble, header) = inspect_converted_rank(&rank0_path).expect("inspect rank 0");

        // Verify scales are stored immediately after their weights inside each group.
        let attn_names: Vec<&str> = header
            .tensors
            .iter()
            .filter(|t| t.group == ResidentGroup::AttentionReplicated)
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(
            attn_names,
            vec!["layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"],
            "scale must immediately follow its weight"
        );

        let moe_entries: Vec<&ConvertedTensorEntry> = header
            .tensors
            .iter()
            .filter(|t| t.group == ResidentGroup::ExpertsRow)
            .collect();
        assert_eq!(moe_entries.len(), 2);
        assert_eq!(moe_entries[0].name, "layers.0.ffn.experts.0.w2.weight");
        assert_eq!(moe_entries[0].layout, TensorLayout::MoePadded640);
        assert_eq!(moe_entries[0].shard_shape, vec![4, 320]);
        assert_eq!(moe_entries[0].bytes, 4 * 320);
        assert_eq!(moe_entries[1].name, "layers.0.ffn.experts.0.w2.scale");
        assert_eq!(moe_entries[1].layout, TensorLayout::MoePadded640);
        assert_eq!(moe_entries[1].shard_shape, vec![4, 20]);
        assert_eq!(moe_entries[1].bytes, 4 * 20);

        // Load into ResidentArena via parallel direct I/O.
        let mut allocator = TestAllocator::new();
        let (arena, report) = load_converted_rank_arena(
            &mut allocator,
            &rank0_path,
            0,
            4,
            ConvertedLoadOptions {
                chunk_bytes: 4096,
                concurrency: 4,
                verify_weights: true,
                ..ConvertedLoadOptions::default()
            },
        )
        .expect("load converted arena");
        assert!(report.weights_verified);
        assert_eq!(arena.len(), 5);
        assert_eq!(
            arena
                .slice("layers.0.ffn.experts.0.w2.weight")
                .expect("w2 weight")
                .bytes,
            4 * 320
        );
        assert_eq!(
            arena
                .slice("layers.0.ffn.experts.0.w2.scale")
                .expect("w2 scale")
                .bytes,
            4 * 20
        );
    }
}
