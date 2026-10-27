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

//! safetensors -> per-rank NVMe Engram table packer.
//!
//! The checkpoint stores each Engram table as two *separate* tensors per layer
//! (`layers.{1,14}.engram.embed.weight`, F8_E4M3 `[rows, 256]`, and
//! `layers.{1,14}.engram.embed.scale`, F8_E8M0 `[rows, 8]`). Both live in a single
//! safetensors shard (47 for layer 1, 48 for layer 14). The runtime wants one dense
//! 264-byte row per lookup (`docs/DECISIONS.md` D4): 256 fp8 values followed by the 8
//! ue8m0 block scales, *not* the specification's 512-byte padded stride.
//!
//! This module turns those two tensors into four per-rank files per layer
//! (`engram-l{1,14}-r{0..3}of4.bin`) by copying rows, never by materialising a shard:
//!
//! * the shard is `mmap`ed read-only and only the rank's contiguous row range is copied
//!   out, so RSS stays at one chunk buffer regardless of the 101 GB shard size;
//! * rows are interleaved into a reusable chunk buffer and streamed to disk, so both
//!   layers' output is never resident at the same time;
//! * the rank split is the reference's `div_ceil` row split (`convert.py:135-141`), and
//!   the last rank's tail rows are written with exactly the reference's padding: zero
//!   weights and a ue8m0 scale of 1.0 (`0x7F`);
//! * every output file is padded to a 512-byte multiple so an O_DIRECT reader can issue
//!   aligned windows (see `crate::store`); the logical row count is recorded in the
//!   manifest and the sub-512-byte tail is never addressed.
//!
//! `plan` needs nothing but the safetensors index JSON and the compiled-in geometry, so
//! `--dry-run` is exact and testable with no weights present. `execute` is the only
//! entry point that opens a shard.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Advice, Mmap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::layout::{
    ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES, SCALE_BLOCK,
    num_embeddings, rows_per_rank,
};

/// Minimum alignment of an output file's length, so an O_DIRECT reader can address any
/// window it needs without reading past the end of the file.
pub const FILE_ALIGN_BYTES: u64 = 512;

/// Default rank count, matching the 4x DGX Spark deployment.
pub const DEFAULT_WORLD_SIZE: usize = 4;

/// Default streamed chunk size in rows (16,384 rows = 4 MiB of output per chunk).
pub const DEFAULT_CHUNK_ROWS: u64 = 16_384;

/// Default free-memory floor. The packer only needs a chunk buffer plus kernel page
/// cache for the mapping it touches, but a 101 GB sequential read will otherwise evict
/// everything else on the box, so refuse to run when the machine is already under
/// memory pressure.
pub const DEFAULT_MIN_FREE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// safetensors dtype of the Engram weight tensor.
pub const WEIGHT_DTYPE: &str = "F8_E4M3";

/// safetensors dtype of the Engram scale tensor.
pub const SCALE_DTYPE: &str = "F8_E8M0";

/// Name of the manifest written next to the tables.
pub const MANIFEST_NAME: &str = "manifest.json";

/// Manifest `format` discriminator.
pub const FORMAT_NAME: &str = "sparkflash-engram-pack";

/// Manifest schema version.
pub const FORMAT_VERSION: u32 = 1;

/// ue8m0 encoding of 1.0 (exponent 127), the value `convert.py` pads the tail scales
/// with. `0x00` would be 2^-127, not zero, which is why the tail is not zero-filled.
pub const PAD_SCALE_BYTE: u8 = 0x7F;

/// Maximum accepted safetensors header length, as a sanity bound before allocating.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Upper bound on `--world-size`, matching the largest plausible deployment.
const MAX_WORLD_SIZE: usize = 64;

// ---------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------

/// Everything one `pack-engram` run needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackOptions {
    /// Path to `model.safetensors.index.json`.
    pub index_path: PathBuf,
    /// Directory holding the safetensors shards; defaults to the index's parent.
    pub checkpoint_dir: Option<PathBuf>,
    /// Directory the rank files and `manifest.json` are written to.
    pub output_dir: PathBuf,
    /// Backbone layer ids to pack (must be `[ENGRAM_LAYERS]` members).
    pub layers: Vec<usize>,
    /// Tensor-parallel world size.
    pub world_size: usize,
    /// Ranks to write; empty means every rank in `0..world_size`.
    pub ranks: Vec<usize>,
    /// Re-read every written file and compare sha256 and sampled rows.
    pub verify: bool,
    /// Print the plan and perform no weight I/O.
    pub dry_run: bool,
    /// `tokenizer.json` the packed tables were built against.
    pub tokenizer_path: Option<PathBuf>,
    /// Free-memory floor in bytes.
    pub min_free_bytes: u64,
    /// Rows per streamed chunk.
    pub chunk_rows: u64,
    /// `sync_all` each output file before moving on.
    pub fsync: bool,
    /// Overwrite existing output files.
    pub force: bool,
}

impl PackOptions {
    /// Options writing into `output_dir` from the index at `index_path`, every other
    /// field at its documented default.
    #[must_use]
    pub fn new(index_path: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            index_path: index_path.into(),
            checkpoint_dir: None,
            output_dir: output_dir.into(),
            layers: ENGRAM_LAYERS.to_vec(),
            world_size: DEFAULT_WORLD_SIZE,
            ranks: Vec::new(),
            verify: false,
            dry_run: false,
            tokenizer_path: None,
            min_free_bytes: DEFAULT_MIN_FREE_BYTES,
            chunk_rows: DEFAULT_CHUNK_ROWS,
            fsync: true,
            force: false,
        }
    }

    /// Ranks this run writes, in ascending order.
    ///
    /// # Errors
    /// When a requested rank is outside `0..world_size` or `world_size` is out of range.
    pub fn selected_ranks(&self) -> Result<Vec<usize>> {
        ensure!(
            self.world_size > 0 && self.world_size <= MAX_WORLD_SIZE,
            "world size {} is outside 1..={MAX_WORLD_SIZE}",
            self.world_size
        );
        ensure!(self.chunk_rows > 0, "chunk_rows must be non-zero");
        ensure!(!self.layers.is_empty(), "at least one layer is required");
        let mut ranks: Vec<usize> = if self.ranks.is_empty() {
            (0..self.world_size).collect()
        } else {
            self.ranks.clone()
        };
        ranks.sort_unstable();
        ranks.dedup();
        for &rank in &ranks {
            ensure!(
                rank < self.world_size,
                "rank {rank} is outside world size {}",
                self.world_size
            );
        }
        Ok(ranks)
    }

    /// Directory holding the safetensors shards.
    ///
    /// # Errors
    /// When the checkpoint directory cannot be derived from the index path.
    pub fn resolved_checkpoint_dir(&self) -> Result<PathBuf> {
        if let Some(dir) = &self.checkpoint_dir {
            return Ok(dir.clone());
        }
        self.index_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot derive the checkpoint directory from index path {}; pass --checkpoint-dir",
                    self.index_path.display()
                )
            })
    }
}

// ---------------------------------------------------------------------------------
// safetensors index
// ---------------------------------------------------------------------------------

/// The subset of `model.safetensors.index.json` the packer reads.
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsIndex {
    /// tensor name -> shard file name.
    pub weight_map: BTreeMap<String, String>,
    /// Optional HF metadata block (`total_size`); reported but otherwise ignored.
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl SafetensorsIndex {
    /// Parses an index JSON file.
    ///
    /// # Errors
    /// When the file cannot be read or is not a safetensors index.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("open safetensors index {}", path.display()))?;
        serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("parse safetensors index {}", path.display()))
    }

    /// Shard file that holds `tensor`.
    ///
    /// # Errors
    /// When the tensor is absent from the index.
    pub fn shard_of(&self, tensor: &str) -> Result<&str> {
        self.weight_map
            .get(tensor)
            .map(String::as_str)
            .ok_or_else(|| {
                let mut known: Vec<&str> = self
                    .weight_map
                    .keys()
                    .filter(|key| key.contains("engram"))
                    .map(String::as_str)
                    .collect();
                known.sort_unstable();
                known.truncate(32);
                anyhow::anyhow!(
                    "tensor {tensor} is not present in the safetensors index (engram tensors found: {})",
                    if known.is_empty() {
                        "none".to_owned()
                    } else {
                        known.join(", ")
                    }
                )
            })
    }
}

/// One tensor as described by a safetensors file header.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TensorInfo {
    /// safetensors dtype string, e.g. `F8_E4M3`.
    pub dtype: String,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// `[start, end)` offsets relative to the end of the file header.
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    /// Number of bytes the tensor occupies in the shard.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }
}

/// Parsed safetensors header: per-tensor descriptors plus where the data section starts.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Tensor descriptors, keyed by full tensor name.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// The shard's `__metadata__` object, kept verbatim.
    ///
    /// safetensors reserves exactly one non-tensor entry, `__metadata__`, whose value is an
    /// object of arbitrary JSON (Hugging Face writes `{"format": "pt"}`). It is kept rather
    /// than dropped so a caller can see what the shard said about itself.
    pub metadata: Option<serde_json::Value>,
    /// Absolute file offset of the first data byte (8 + header length).
    pub data_start: u64,
    /// Length of the shard file itself.
    ///
    /// A header is just a promise: it can describe more data than the shard holds, for
    /// example when a download was cut short. Every tensor offset is checked against this
    /// before any byte of the file is sliced.
    pub file_len: u64,
}

/// The one non-tensor key safetensors reserves: an object of arbitrary JSON.
const METADATA_KEY: &str = "__metadata__";

/// Splits a shard header into its tensor descriptors and its metadata object.
///
/// A safetensors header is a JSON object whose entries are tensor descriptors, except for the
/// optional `__metadata__` entry. That entry is not a tensor: the container specification
/// lets it hold any JSON at all - strings, numbers, booleans, arrays, nested objects - and
/// Hugging Face writes `{"format": "pt"}` there. Deserialising the whole object straight into
/// a tensor map therefore rejects a perfectly standard shard with "missing field dtype", so
/// every entry is read as JSON first and only the non-metadata entries are required to be
/// tensor descriptors. An entry that is neither `__metadata__` nor a tensor is still an
/// error, and the error names the offending key.
fn parse_header(
    header: &[u8],
) -> anyhow::Result<(BTreeMap<String, TensorInfo>, Option<serde_json::Value>)> {
    let entries: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(header).context("the safetensors header is not a JSON object")?;
    let mut metadata = None;
    let mut tensors = BTreeMap::new();
    for (name, value) in entries {
        if name == METADATA_KEY {
            metadata = Some(value);
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value)
            .with_context(|| format!("{name} is not a valid safetensors tensor descriptor"))?;
        tensors.insert(name, info);
    }
    Ok((tensors, metadata))
}

impl SafetensorsHeader {
    /// Reads and parses only the JSON header of a safetensors file. The header of a
    /// 101 GB shard is a few kilobytes, so this never touches the tensor data.
    ///
    /// # Errors
    /// When the file is unreadable, truncated, or not a safetensors container.
    pub fn read(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("open safetensors shard {}", path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("stat safetensors shard {}", path.display()))?
            .len();
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .with_context(|| format!("read safetensors header length of {}", path.display()))?;
        let header_len = u64::from_le_bytes(len_bytes);
        ensure!(
            header_len > 2 && header_len <= MAX_HEADER_BYTES,
            "{} is not a safetensors file (header length {header_len})",
            path.display()
        );
        ensure!(
            8 + header_len <= file_len,
            "{} is truncated: its header claims {header_len} bytes but the shard is only \
             {file_len} bytes",
            path.display()
        );
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)
            .with_context(|| format!("read safetensors header of {}", path.display()))?;
        // The chain is flattened into this message rather than attached as context: the key that
        // failed is named by parse_header, this is the message a caller sees with {}, and an
        // outer context would hide it until someone printed the alternate form (review A6).
        let (tensors, metadata) = parse_header(&header).map_err(|error| {
            anyhow::anyhow!("parse safetensors header of {}: {error:#}", path.display())
        })?;
        Ok(Self {
            tensors,
            metadata,
            data_start: 8 + header_len,
            file_len,
        })
    }

    /// Descriptor of `name`.
    ///
    /// # Errors
    /// When the tensor is absent from the header.
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name} is missing from the shard header"))
    }
}

// ---------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------

/// A tensor as planned from the index alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanTensor {
    /// Full tensor name.
    pub name: String,
    /// Expected safetensors dtype.
    pub dtype: String,
    /// Expected shape.
    pub shape: [u64; 2],
}

impl PlanTensor {
    /// Number of bytes this tensor occupies in the source shard.
    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.shape[0].saturating_mul(self.shape[1])
    }
}

/// One output rank file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardPlan {
    /// Rank index.
    pub rank: usize,
    /// File name, `engram-l{layer}-r{rank}of{world}.bin`.
    pub file: String,
    /// First global table row in this file.
    pub row_start: u64,
    /// One past the last global table row in this file.
    pub row_end: u64,
    /// Rows stored in the file (including the padded tail).
    pub rows: u64,
    /// Rows that exist in the source table.
    pub source_rows: u64,
    /// Tail rows that do not exist in the source and are written as padding.
    pub padded_rows: u64,
    /// Logical row bytes before file alignment (`rows * 264`).
    pub data_bytes: u64,
    /// File length, `data_bytes` rounded up to `[FILE_ALIGN_BYTES]`.
    pub file_bytes: u64,
    /// Zero bytes appended for alignment.
    pub padding_bytes: u64,
}

/// One layer's plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlan {
    /// Backbone layer id (1 or 14).
    pub layer: usize,
    /// Index into `[ENGRAM_LAYERS]`.
    pub layer_index: usize,
    /// safetensors shard holding both tensors.
    pub source_shard: String,
    /// Weight tensor plan.
    pub weight: PlanTensor,
    /// Scale tensor plan.
    pub scale: PlanTensor,
    /// Physical row count from the checkpoint config.
    pub table_rows: u64,
    /// Dense row stride in the packed file.
    pub row_bytes: u64,
    /// Output files, one per selected rank.
    pub shards: Vec<ShardPlan>,
}

impl LayerPlan {
    /// Sum of the selected shards' logical data bytes.
    #[must_use]
    pub fn data_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.data_bytes).sum()
    }

    /// Sum of the selected shards' file bytes.
    #[must_use]
    pub fn file_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.file_bytes).sum()
    }
}

/// The complete, weight-free packing plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackPlan {
    /// Index file the plan was derived from.
    pub index_path: PathBuf,
    /// sha256 of the index file, when it was hashed.
    pub index_sha256: Option<String>,
    /// Checkpoint directory holding the shards.
    pub checkpoint_dir: PathBuf,
    /// Output directory.
    pub output_dir: PathBuf,
    /// World size.
    pub world_size: usize,
    /// Ranks that will be written.
    pub ranks: Vec<usize>,
    /// Per-layer plans, in the order given by `--layers`.
    pub layers: Vec<LayerPlan>,
    /// Total physical rows across the planned layers.
    pub total_rows: u64,
    /// Total logical data bytes across the planned files.
    pub total_data_bytes: u64,
    /// Total file bytes (including alignment padding) across the planned files.
    pub total_file_bytes: u64,
    /// `tokenizer.json` path the run is pinned to.
    pub tokenizer_path: Option<PathBuf>,
    /// sha256 of `tokenizer.json`, when it was hashed.
    pub tokenizer_sha256: Option<String>,
}

/// Rounds `value` up to the next multiple of `[FILE_ALIGN_BYTES]`.
#[must_use]
pub fn align_up(value: u64) -> u64 {
    value.div_ceil(FILE_ALIGN_BYTES) * FILE_ALIGN_BYTES
}

/// Returns the padded tail row exactly as `convert.py` writes it: zero weights and a
/// ue8m0 exponent of 127 (1.0) in every block scale.
#[must_use]
pub fn pad_row() -> [u8; ENGRAM_ROW_BYTES] {
    let mut row = [0u8; ENGRAM_ROW_BYTES];
    row[ENGRAM_HEAD_DIM..].fill(PAD_SCALE_BYTE);
    row
}

/// Builds the packing plan from the safetensors index and the compiled-in geometry.
///
/// This performs no weight I/O: the only files it opens are the index JSON and
/// `tokenizer.json`, and only to hash them.
///
/// # Errors
/// When the index is unreadable, an expected tensor is missing, both tensors are not in
/// the same shard, or an option is out of range.
pub fn plan(options: &PackOptions) -> Result<PackPlan> {
    let ranks = options.selected_ranks()?;
    let checkpoint_dir = options.resolved_checkpoint_dir()?;
    let index = SafetensorsIndex::load(&options.index_path)?;
    let index_sha256 = sha256_file(&options.index_path).ok();

    let tokenizer_path = options
        .tokenizer_path
        .clone()
        .or_else(|| Some(checkpoint_dir.join("tokenizer.json")));
    let tokenizer_sha256 = tokenizer_path
        .as_deref()
        .and_then(|path| sha256_file(path).ok());

    let mut layers = Vec::with_capacity(options.layers.len());
    for &layer in &options.layers {
        let layer_index = ENGRAM_LAYERS
            .iter()
            .position(|candidate| *candidate == layer)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {layer} has no Engram table; known Engram layers are {}",
                    ENGRAM_LAYERS
                        .iter()
                        .map(|known| known.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        let weight_name = format!("layers.{layer}.engram.embed.weight");
        let scale_name = format!("layers.{layer}.engram.embed.scale");
        let weight_shard = index.shard_of(&weight_name)?;
        let scale_shard = index.shard_of(&scale_name)?;
        ensure!(
            weight_shard == scale_shard,
            "layer {layer}: {weight_name} lives in {weight_shard} but {scale_name} lives in \
             {scale_shard}; the packer needs both in one shard"
        );
        let table_rows = num_embeddings(layer_index);
        let shard_rows = rows_per_rank(layer_index, options.world_size);

        let mut shards = Vec::with_capacity(ranks.len());
        for &rank in &ranks {
            let row_start = rank as u64 * shard_rows;
            let row_end = row_start + shard_rows;
            let source_rows = if row_start >= table_rows {
                0
            } else {
                shard_rows.min(table_rows - row_start)
            };
            let data_bytes = shard_rows * ENGRAM_ROW_BYTES as u64;
            let file_bytes = align_up(data_bytes);
            shards.push(ShardPlan {
                rank,
                file: shard_file_name(layer, rank, options.world_size),
                row_start,
                row_end,
                rows: shard_rows,
                source_rows,
                padded_rows: shard_rows - source_rows,
                data_bytes,
                file_bytes,
                padding_bytes: file_bytes - data_bytes,
            });
        }

        layers.push(LayerPlan {
            layer,
            layer_index,
            source_shard: weight_shard.to_owned(),
            weight: PlanTensor {
                name: weight_name,
                dtype: WEIGHT_DTYPE.to_owned(),
                shape: [table_rows, ENGRAM_HEAD_DIM as u64],
            },
            scale: PlanTensor {
                name: scale_name,
                dtype: SCALE_DTYPE.to_owned(),
                shape: [table_rows, ENGRAM_SCALE_BYTES as u64],
            },
            table_rows,
            row_bytes: ENGRAM_ROW_BYTES as u64,
            shards,
        });
    }

    let total_rows = layers.iter().map(|layer| layer.table_rows).sum();
    let total_data_bytes = layers.iter().map(LayerPlan::data_bytes).sum();
    let total_file_bytes = layers.iter().map(LayerPlan::file_bytes).sum();
    Ok(PackPlan {
        index_path: options.index_path.clone(),
        index_sha256,
        checkpoint_dir,
        output_dir: options.output_dir.clone(),
        world_size: options.world_size,
        ranks,
        layers,
        total_rows,
        total_data_bytes,
        total_file_bytes,
        tokenizer_path,
        tokenizer_sha256,
    })
}

/// File name of one layer/rank table.
#[must_use]
pub fn shard_file_name(layer: usize, rank: usize, world_size: usize) -> String {
    format!("engram-l{layer}-r{rank}of{world_size}.bin")
}

/// Human-readable rendering of a plan, used by `--dry-run`.
#[must_use]
pub fn format_plan(plan: &PackPlan, options: &PackOptions) -> String {
    let mut out = String::new();
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let _ = writeln!(out, "sparkflash pack-engram -- DRY RUN (no weight I/O)");
    let _ = writeln!(
        out,
        "  index        : {}  sha256 {}",
        plan.index_path.display(),
        plan.index_sha256.as_deref().unwrap_or("<not hashed>")
    );
    let _ = writeln!(out, "  checkpoint   : {}", plan.checkpoint_dir.display());
    let _ = writeln!(
        out,
        "  tokenizer    : {}  sha256 {}",
        plan.tokenizer_path
            .as_deref()
            .map_or_else(|| "<none>".to_owned(), |path| path.display().to_string()),
        plan.tokenizer_sha256.as_deref().unwrap_or("<not hashed>")
    );
    let _ = writeln!(out, "  output dir   : {}", plan.output_dir.display());
    let _ = writeln!(
        out,
        "  world size   : {}  ranks: {}",
        plan.world_size,
        plan.ranks
            .iter()
            .map(|rank| rank.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let _ = writeln!(
        out,
        "  row layout   : dense {} B = {} F8_E4M3 + {} F8_E8M0 ({} values per scale); \
         file padding {FILE_ALIGN_BYTES} B",
        ENGRAM_ROW_BYTES, ENGRAM_HEAD_DIM, ENGRAM_SCALE_BYTES, SCALE_BLOCK
    );
    for layer in &plan.layers {
        let _ = writeln!(
            out,
            "  layer {} (engram layer index {}, {} table rows)",
            layer.layer, layer.layer_index, layer.table_rows
        );
        let _ = writeln!(out, "    source shard : {}", layer.source_shard);
        let _ = writeln!(
            out,
            "    weight       : {}  dtype {}  shape [{}, {}]  {} bytes",
            layer.weight.name,
            layer.weight.dtype,
            layer.weight.shape[0],
            layer.weight.shape[1],
            layer.weight.source_bytes()
        );
        let _ = writeln!(
            out,
            "    scale        : {}  dtype {}  shape [{}, {}]  {} bytes",
            layer.scale.name,
            layer.scale.dtype,
            layer.scale.shape[0],
            layer.scale.shape[1],
            layer.scale.source_bytes()
        );
        let _ = writeln!(
            out,
            "    {:>4}  {:<24} {:>12}  {:<28} {:>12} {:>16}",
            "rank", "file", "rows", "row range", "padded", "file bytes"
        );
        for shard in &layer.shards {
            let _ = writeln!(
                out,
                "    {:>4}  {:<24} {:>12}  {:<28} {:>12} {:>16}",
                shard.rank,
                shard.file,
                shard.rows,
                format!("{}..{}", shard.row_start, shard.row_end),
                shard.padded_rows,
                shard.file_bytes
            );
        }
        let _ = writeln!(
            out,
            "    layer total  : {} files, {} data bytes, {} file bytes ({:.3} GiB)",
            layer.shards.len(),
            layer.data_bytes(),
            layer.file_bytes(),
            gib(layer.file_bytes())
        );
    }
    let _ = writeln!(
        out,
        "  totals       : {} table rows, {} files, {} data bytes, {} file bytes ({:.3} GiB)",
        plan.total_rows,
        plan.layers
            .iter()
            .map(|layer| layer.shards.len())
            .sum::<usize>(),
        plan.total_data_bytes,
        plan.total_file_bytes,
        gib(plan.total_file_bytes)
    );
    let _ = writeln!(
        out,
        "  manifest     : {}",
        plan.output_dir.join(MANIFEST_NAME).display()
    );
    let _ = writeln!(
        out,
        "  memory floor : {} bytes ({:.2} GiB), chunk {} rows ({} bytes)",
        options.min_free_bytes,
        gib(options.min_free_bytes),
        options.chunk_rows,
        options.chunk_rows * ENGRAM_ROW_BYTES as u64
    );
    match mem_available_bytes() {
        Ok(available) => {
            let verdict = if available >= options.min_free_bytes {
                "ok"
            } else {
                "BELOW FLOOR -- a real run would refuse"
            };
            let _ = writeln!(
                out,
                "  MemAvailable : {} bytes ({:.2} GiB) -- {verdict}",
                available,
                gib(available)
            );
        }
        Err(error) => {
            let _ = writeln!(out, "  MemAvailable : unavailable ({error:#})");
        }
    }
    out
}

// ---------------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------------

/// Manifest written next to the packed tables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackManifest {
    /// Format discriminator.
    pub format: String,
    /// Schema version.
    pub format_version: u32,
    /// Producing command.
    pub generated_by: String,
    /// Dense-row layout description.
    pub layout: LayoutDescription,
    /// World size.
    pub world_size: usize,
    /// Backbone layers packed.
    pub layers: Vec<usize>,
    /// Index the tables were packed from.
    pub index: String,
    /// sha256 of the index file.
    pub index_sha256: Option<String>,
    /// `tokenizer.json` the tables were built against.
    pub tokenizer: Option<String>,
    /// sha256 of that `tokenizer.json`.
    pub tokenizer_sha256: Option<String>,
    /// Total physical rows across layers.
    pub total_rows: u64,
    /// Total logical data bytes.
    pub total_data_bytes: u64,
    /// Total file bytes including alignment padding.
    pub total_file_bytes: u64,
    /// Per-file entries.
    pub shards: Vec<ManifestShard>,
    /// Per-layer source tensor descriptors.
    pub sources: Vec<ManifestSource>,
}

/// Dense-row layout description inside the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutDescription {
    /// Bytes per dense row.
    pub row_bytes: u64,
    /// Leading weight bytes per row.
    pub weight_bytes: u64,
    /// Trailing scale bytes per row.
    pub scale_bytes: u64,
    /// Values per ue8m0 block scale.
    pub scale_block: u64,
    /// Dtype of the weight bytes.
    pub dtype_weight: String,
    /// Dtype of the scale bytes.
    pub dtype_scale: String,
    /// File-length alignment.
    pub file_alignment: u64,
    /// The padded-stride layout is explicitly not used.
    pub dense: bool,
}

/// One file's manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestShard {
    /// File name.
    pub file: String,
    /// Backbone layer id.
    pub layer: usize,
    /// Rank.
    pub rank: usize,
    /// World size.
    pub world_size: usize,
    /// First global row.
    pub row_start: u64,
    /// One past the last global row.
    pub row_end: u64,
    /// Rows in the file including padding.
    pub rows: u64,
    /// Tail rows that do not exist in the source.
    pub padded_rows: u64,
    /// Row stride in bytes.
    pub row_stride: u64,
    /// Dtype of the row's weight bytes.
    pub dtype: String,
    /// Logical data bytes.
    pub data_bytes: u64,
    /// Actual file length.
    pub file_bytes: u64,
    /// Bytes appended for alignment.
    pub padding_bytes: u64,
    /// sha256 of the complete file.
    pub sha256: String,
    /// Source shard this file was cut from.
    pub source_shard: String,
}

/// One layer's manifest source entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSource {
    /// Backbone layer id.
    pub layer: usize,
    /// Index into `[ENGRAM_LAYERS]`.
    pub layer_index: usize,
    /// safetensors shard holding both tensors.
    pub shard: String,
    /// Source weight tensor.
    pub weight: PlanTensor,
    /// Source scale tensor.
    pub scale: PlanTensor,
    /// Absolute data offset of the weight tensor inside the shard.
    pub weight_data_offset: u64,
    /// Absolute data offset of the scale tensor inside the shard.
    pub scale_data_offset: u64,
    /// Output files written for this layer.
    pub files: Vec<String>,
}

/// Result of one `pack-engram` invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackReport {
    /// True when nothing was written.
    pub dry_run: bool,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Manifest path (written during a real run).
    pub manifest_path: PathBuf,
    /// Files written, in order.
    pub files_written: Vec<String>,
    /// Total bytes written including alignment padding.
    pub bytes_written: u64,
    /// Files re-read by `--verify`.
    pub verified_files: usize,
    /// Sampled rows byte-compared by `--verify`.
    pub verified_sample_rows: u64,
}

/// Plans, and for a dry run prints the plan and touches no shard.
///
/// # Errors
/// Propagates `[plan]` and `[execute]` errors.
pub fn run(options: &PackOptions) -> Result<PackReport> {
    let plan = plan(options)?;
    if options.dry_run {
        print!("{}", format_plan(&plan, options));
        return Ok(PackReport {
            dry_run: true,
            output_dir: plan.output_dir.clone(),
            manifest_path: plan.output_dir.join(MANIFEST_NAME),
            files_written: Vec::new(),
            bytes_written: 0,
            verified_files: 0,
            verified_sample_rows: 0,
        });
    }
    execute(&plan, options)
}

/// Executes a plan: opens the shards, streams the rank files and writes the manifest.
///
/// The shard is `mmap`ed and only the rank's contiguous row range is copied, so no
/// whole-shard buffer is ever allocated. `MADV_SEQUENTIAL` is set on the mapping and
/// `MADV_DONTNEED` is issued after each output file, which bounds page-cache growth on a
/// machine that has to keep serving other work.
///
/// # Errors
/// When free memory is below the configured floor, an output file already exists and
/// `--force` was not given, the shard header disagrees with the geometry, or any I/O
/// fails.
pub fn execute(plan: &PackPlan, options: &PackOptions) -> Result<PackReport> {
    // The plan is validated where it is built, but the options are not: this entry point is
    // public, and a zero chunk size used to be accepted here and looped forever.
    ensure!(options.chunk_rows > 0, "chunk_rows must be non-zero");
    let available = mem_available_bytes()?;
    check_free_memory(options.min_free_bytes, available)?;
    std::fs::create_dir_all(&plan.output_dir)
        .with_context(|| format!("create output directory {}", plan.output_dir.display()))?;

    let mut manifest_shards = Vec::new();
    let mut manifest_sources = Vec::new();
    let mut files_written = Vec::new();
    let mut bytes_written = 0u64;

    for layer in &plan.layers {
        let shard_path = plan.checkpoint_dir.join(&layer.source_shard);
        let header = SafetensorsHeader::read(&shard_path)?;
        let weight_info = validate_tensor(&header, &layer.weight, &shard_path)?;
        let scale_info = validate_tensor(&header, &layer.scale, &shard_path)?;
        let file = File::open(&shard_path)
            .with_context(|| format!("open source shard {}", shard_path.display()))?;
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap source shard {}", shard_path.display()))?;
        map.advise(Advice::Sequential)
            .with_context(|| format!("advise Sequential on {}", shard_path.display()))?;
        let source = SourceLayer {
            map,
            weight_offset: header.data_start + weight_info.data_offsets[0],
            scale_offset: header.data_start + scale_info.data_offsets[0],
        };

        let mut layer_files = Vec::new();
        for shard in &layer.shards {
            let path = plan.output_dir.join(&shard.file);
            if path.exists() && !options.force {
                bail!(
                    "{} already exists; pass --force to overwrite it",
                    path.display()
                );
            }
            let sha256 = write_shard(&source, shard, &path, options)?;
            let file_bytes = std::fs::metadata(&path)
                .with_context(|| format!("stat {}", path.display()))?
                .len();
            ensure!(
                file_bytes == shard.file_bytes,
                "{} is {file_bytes} bytes, expected {}",
                path.display(),
                shard.file_bytes
            );
            files_written.push(shard.file.clone());
            layer_files.push(shard.file.clone());
            bytes_written += file_bytes;
            manifest_shards.push(ManifestShard {
                file: shard.file.clone(),
                layer: layer.layer,
                rank: shard.rank,
                world_size: plan.world_size,
                row_start: shard.row_start,
                row_end: shard.row_end,
                rows: shard.rows,
                padded_rows: shard.padded_rows,
                row_stride: ENGRAM_ROW_BYTES as u64,
                dtype: WEIGHT_DTYPE.to_owned(),
                data_bytes: shard.data_bytes,
                file_bytes: shard.file_bytes,
                padding_bytes: shard.padding_bytes,
                sha256,
                source_shard: layer.source_shard.clone(),
            });
        }

        // Release the shard's page cache; every rank reads a disjoint range, so nothing
        // this run still needs lives in the mapping.
        release_pages(&source.map);

        manifest_sources.push(ManifestSource {
            layer: layer.layer,
            layer_index: layer.layer_index,
            shard: layer.source_shard.clone(),
            weight: layer.weight.clone(),
            scale: layer.scale.clone(),
            weight_data_offset: source.weight_offset,
            scale_data_offset: source.scale_offset,
            files: layer_files,
        });
    }

    let manifest = PackManifest {
        format: FORMAT_NAME.to_owned(),
        format_version: FORMAT_VERSION,
        generated_by: "sparkflash pack-engram".to_owned(),
        layout: LayoutDescription {
            row_bytes: ENGRAM_ROW_BYTES as u64,
            weight_bytes: ENGRAM_HEAD_DIM as u64,
            scale_bytes: ENGRAM_SCALE_BYTES as u64,
            scale_block: SCALE_BLOCK as u64,
            dtype_weight: WEIGHT_DTYPE.to_owned(),
            dtype_scale: SCALE_DTYPE.to_owned(),
            file_alignment: FILE_ALIGN_BYTES,
            dense: true,
        },
        world_size: plan.world_size,
        layers: plan.layers.iter().map(|layer| layer.layer).collect(),
        index: plan.index_path.display().to_string(),
        index_sha256: plan.index_sha256.clone(),
        tokenizer: plan
            .tokenizer_path
            .as_deref()
            .map(|path| path.display().to_string()),
        tokenizer_sha256: plan.tokenizer_sha256.clone(),
        total_rows: plan.total_rows,
        total_data_bytes: plan.total_data_bytes,
        total_file_bytes: plan.total_file_bytes,
        shards: manifest_shards,
        sources: manifest_sources,
    };
    let manifest_path = plan.output_dir.join(MANIFEST_NAME);
    write_manifest(&manifest, &manifest_path)?;

    let mut verified_files = 0usize;
    let mut verified_sample_rows = 0u64;
    if options.verify {
        let (files, rows) = verify(plan, &manifest)?;
        verified_files = files;
        verified_sample_rows = rows;
    }

    Ok(PackReport {
        dry_run: false,
        output_dir: plan.output_dir.clone(),
        manifest_path,
        files_written,
        bytes_written,
        verified_files,
        verified_sample_rows,
    })
}

/// Source shard view: one mapping plus both tensors' absolute offsets.
struct SourceLayer {
    map: Mmap,
    weight_offset: u64,
    scale_offset: u64,
}

impl SourceLayer {
    fn weight_row(&self, row: u64) -> &[u8] {
        let start = self.weight_offset as usize + row as usize * ENGRAM_HEAD_DIM;
        &self.map[start..start + ENGRAM_HEAD_DIM]
    }

    fn scale_row(&self, row: u64) -> &[u8] {
        let start = self.scale_offset as usize + row as usize * ENGRAM_SCALE_BYTES;
        &self.map[start..start + ENGRAM_SCALE_BYTES]
    }
}

/// Drops a mapping's resident pages with `MADV_DONTNEED`. Best effort by design: the
/// only cost of a failure is that the kernel keeps the pages cached, and the caller drops
/// the mapping immediately afterwards anyway.
fn release_pages(map: &Mmap) {
    if map.is_empty() {
        return;
    }
    // SAFETY: the pointer and length come from the live mapping and MADV_DONTNEED only
    // releases resident pages of an address range this process owns.
    unsafe {
        libc::madvise(
            map.as_ptr().cast_mut().cast(),
            map.len(),
            libc::MADV_DONTNEED,
        );
    }
}

fn validate_tensor(
    header: &SafetensorsHeader,
    plan: &PlanTensor,
    shard_path: &Path,
) -> Result<TensorInfo> {
    let info = header.tensor(&plan.name)?;
    ensure!(
        info.dtype == plan.dtype,
        "{}: {} has dtype {}, expected {}",
        shard_path.display(),
        plan.name,
        info.dtype,
        plan.dtype
    );
    ensure!(
        info.shape.len() == 2 && info.shape[0] == plan.shape[0] && info.shape[1] == plan.shape[1],
        "{}: {} has shape {:?}, expected {:?}",
        shard_path.display(),
        plan.name,
        info.shape,
        plan.shape
    );
    ensure!(
        info.data_offsets[0] <= info.data_offsets[1],
        "{}: {} has inverted data offsets {:?}",
        shard_path.display(),
        plan.name,
        info.data_offsets
    );
    // A header can describe more data than the shard holds; slicing the mapping at such an
    // offset panics, so refuse it with an error instead.
    let end = header
        .data_start
        .checked_add(info.data_offsets[1])
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{}: {} ends past the end of the address space",
                shard_path.display(),
                plan.name
            )
        })?;
    ensure!(
        end <= header.file_len,
        "{}: {} ends at byte {end} but the shard is only {} bytes (truncated shard)",
        shard_path.display(),
        plan.name,
        header.file_len
    );
    ensure!(
        info.byte_len() == plan.source_bytes(),
        "{}: {} occupies {} bytes, expected {}",
        shard_path.display(),
        plan.name,
        info.byte_len(),
        plan.source_bytes()
    );
    Ok(info.clone())
}

/// Streams one rank file: interleaves weight and scale rows into a reusable chunk
/// buffer, hashes, fsyncs and aligns the length. Returns the file's sha256.
fn write_shard(
    source: &SourceLayer,
    shard: &ShardPlan,
    path: &Path,
    options: &PackOptions,
) -> Result<String> {
    let row_bytes = ENGRAM_ROW_BYTES as u64;
    ensure!(options.chunk_rows > 0, "chunk_rows must be non-zero");
    let chunk_rows = options.chunk_rows.min(shard.rows.max(1));
    let mut buffer = vec![0u8; (chunk_rows * row_bytes) as usize];
    let pad = pad_row();
    let mut hasher = Sha256::new();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

    let mut row = 0u64;
    while row < shard.rows {
        let count = (shard.rows - row).min(chunk_rows);
        let real = if row >= shard.source_rows {
            0
        } else {
            (shard.source_rows - row).min(count)
        };
        let chunk = &mut buffer[..(count * row_bytes) as usize];
        for index in 0..real {
            let source_row = shard.row_start + row + index;
            let slot = &mut chunk[(index * row_bytes) as usize..((index + 1) * row_bytes) as usize];
            slot[..ENGRAM_HEAD_DIM].copy_from_slice(source.weight_row(source_row));
            slot[ENGRAM_HEAD_DIM..].copy_from_slice(source.scale_row(source_row));
        }
        for index in real..count {
            let slot = &mut chunk[(index * row_bytes) as usize..((index + 1) * row_bytes) as usize];
            slot.copy_from_slice(&pad);
        }
        writer
            .write_all(chunk)
            .with_context(|| format!("write {}", path.display()))?;
        hasher.update(chunk);
        row += count;
    }

    if shard.padding_bytes > 0 {
        let zeros = [0u8; FILE_ALIGN_BYTES as usize];
        let tail = &zeros[..shard.padding_bytes as usize];
        writer
            .write_all(tail)
            .with_context(|| format!("write alignment padding to {}", path.display()))?;
        hasher.update(tail);
    }
    let file = writer
        .into_inner()
        .map_err(|error| error.into_error())
        .with_context(|| format!("flush {}", path.display()))?;
    if options.fsync {
        file.sync_all()
            .with_context(|| format!("fsync {}", path.display()))?;
    }
    Ok(to_hex(&hasher.finalize()))
}

fn write_manifest(manifest: &PackManifest, path: &Path) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, manifest)
        .with_context(|| format!("serialize manifest {}", path.display()))?;
    writer
        .write_all(b"\n")
        .with_context(|| format!("write manifest {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flush manifest {}", path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|error| error.into_error())
        .with_context(|| format!("flush manifest {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("fsync manifest {}", path.display()))?;
    Ok(())
}

/// Re-reads every file named in the manifest: streams it through sha256 and compares a
/// sample of rows against the source shard. Returns `(files, sampled rows)`.
///
/// # Errors
/// When a file's length or sha256 disagrees with the manifest, or a sampled row does not
/// match the source.
pub fn verify(plan: &PackPlan, manifest: &PackManifest) -> Result<(usize, u64)> {
    let mut sampled_total = 0u64;
    for entry in &manifest.shards {
        let layer_plan = plan
            .layers
            .iter()
            .find(|layer| layer.layer == entry.layer)
            .ok_or_else(|| {
                anyhow::anyhow!("manifest names layer {} but the plan does not", entry.layer)
            })?;
        let source_path = plan.checkpoint_dir.join(&entry.source_shard);
        let header = SafetensorsHeader::read(&source_path)?;
        let weight_info = validate_tensor(&header, &layer_plan.weight, &source_path)?;
        let scale_info = validate_tensor(&header, &layer_plan.scale, &source_path)?;
        let file = File::open(&source_path)
            .with_context(|| format!("open source shard {}", source_path.display()))?;
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap source shard {}", source_path.display()))?;
        let source = SourceLayer {
            map,
            weight_offset: header.data_start + weight_info.data_offsets[0],
            scale_offset: header.data_start + scale_info.data_offsets[0],
        };

        let path = plan.output_dir.join(&entry.file);
        let actual_bytes = std::fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        ensure!(
            actual_bytes == entry.file_bytes,
            "{} is {actual_bytes} bytes, manifest says {}",
            path.display(),
            entry.file_bytes
        );
        let actual_sha = sha256_file(&path)?;
        ensure!(
            actual_sha == entry.sha256,
            "{} sha256 {actual_sha} does not match manifest {}",
            path.display(),
            entry.sha256
        );

        let table = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let real_rows = entry.rows - entry.padded_rows;
        let mut sampled = 0u64;
        for row in sample_rows(entry.rows) {
            let mut got = [0u8; ENGRAM_ROW_BYTES];
            table
                .read_exact_at(&mut got, row * ENGRAM_ROW_BYTES as u64)
                .with_context(|| format!("read row {row} of {}", path.display()))?;
            let want = if row < real_rows {
                let source_row = entry.row_start + row;
                let mut expected = [0u8; ENGRAM_ROW_BYTES];
                expected[..ENGRAM_HEAD_DIM].copy_from_slice(source.weight_row(source_row));
                expected[ENGRAM_HEAD_DIM..].copy_from_slice(source.scale_row(source_row));
                expected
            } else {
                pad_row()
            };
            ensure!(
                got == want,
                "{}: row {row} does not match the source shard",
                path.display()
            );
            sampled += 1;
        }
        release_pages(&source.map);
        sampled_total += sampled;
    }
    Ok((manifest.shards.len(), sampled_total))
}

/// Evenly spaced sample rows plus both ends.
#[must_use]
pub fn sample_rows(rows: u64) -> Vec<u64> {
    if rows == 0 {
        return Vec::new();
    }
    const SAMPLES: u64 = 64;
    let mut out = Vec::with_capacity(SAMPLES as usize + 2);
    out.push(0);
    for index in 0..SAMPLES {
        out.push(index * (rows - 1) / (SAMPLES - 1));
    }
    out.push(rows - 1);
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------------
// Memory / hashing helpers
// ---------------------------------------------------------------------------------

/// Free-memory floor check, split out from `/proc` so it is unit testable.
///
/// # Errors
/// When `available` is below `required`, with both figures in the message.
pub fn check_free_memory(required: u64, available: u64) -> Result<()> {
    if available < required {
        bail!(
            "refusing to pack: {} bytes ({:.2} GiB) of MemAvailable is below the {} bytes \
             ({:.2} GiB) floor; free memory or lower --min-free-gib",
            available,
            available as f64 / (1024.0 * 1024.0 * 1024.0),
            required,
            required as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    Ok(())
}

/// Physical memory the kernel reports as available, read from `/proc/meminfo`.
///
/// # Errors
/// When `/proc/meminfo` is unreadable and the `sysconf` fallback also fails.
pub fn mem_available_bytes() -> Result<u64> {
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kib = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("malformed MemAvailable line"))?
                    .parse::<u64>()
                    .context("parse MemAvailable")?;
                return Ok(kib * 1024);
            }
        }
    }
    // Fallback for a kernel without /proc.
    let pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(
        pages > 0 && page_size > 0,
        "cannot determine available memory (sysconf returned {pages} pages of {page_size} bytes)"
    );
    Ok(pages as u64 * page_size as u64)
}

/// Lower-case hex rendering.
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Streaming sha256 of a file, returned as lower-case hex.
///
/// # Errors
/// When the file cannot be opened or read.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(to_hex(&hasher.finalize()))
}

// ---------------------------------------------------------------------------------
// CLI surface (shared by the pack-engram bin and sparkflash pack-engram)
// ---------------------------------------------------------------------------------

/// Command line arguments for `pack-engram`.
#[derive(Debug, Clone, clap::Args)]
#[command(
    name = "pack-engram",
    about = "Extract the two Engram tables from the safetensors shards into dense per-rank files"
)]
pub struct CliArgs {
    /// Path to model.safetensors.index.json.
    #[arg(long, env = "SPARK_ENGRAM_INDEX", value_name = "PATH")]
    pub index: Option<PathBuf>,
    /// Directory holding the safetensors shards (default: the index's parent).
    #[arg(long, value_name = "DIR")]
    pub checkpoint_dir: Option<PathBuf>,
    /// Directory to write the rank files and manifest.json into.
    #[arg(long, value_name = "DIR", default_value = "/var/lib/sparkflash/engram")]
    pub output_dir: PathBuf,
    /// Comma-separated backbone layer ids.
    #[arg(long, value_delimiter = ',', default_value = "1,14")]
    pub layers: Vec<usize>,
    /// Tensor-parallel world size.
    #[arg(long, default_value_t = DEFAULT_WORLD_SIZE)]
    pub world_size: usize,
    /// Comma-separated ranks to write (default: every rank).
    #[arg(long, value_delimiter = ',')]
    pub rank: Vec<usize>,
    /// Plan only: read the index, open no shard, print the plan.
    #[arg(long)]
    pub dry_run: bool,
    /// Re-read every written file and compare sha256 plus sampled rows.
    #[arg(long)]
    pub verify: bool,
    /// tokenizer.json the tables are built against (default: beside the index).
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,
    /// Free-memory floor; refuse to run below it.
    #[arg(long, default_value_t = DEFAULT_MIN_FREE_BYTES / (1024 * 1024 * 1024))]
    pub min_free_gib: u64,
    /// Rows per streamed chunk.
    #[arg(long, default_value_t = DEFAULT_CHUNK_ROWS)]
    pub chunk_rows: u64,
    /// Overwrite existing output files.
    #[arg(long)]
    pub force: bool,
    /// Do not fsync each output file before moving on.
    #[arg(long)]
    pub no_fsync: bool,
}

impl CliArgs {
    /// Resolves the CLI surface into library options.
    ///
    /// # Errors
    /// When no index path can be determined, or `--min-free-gib` overflows.
    pub fn into_options(self) -> Result<PackOptions> {
        let checkpoint_dir = self.checkpoint_dir.clone();
        let index_path = match self.index {
            Some(path) => path,
            None => {
                let dir = checkpoint_dir.clone().ok_or_else(|| {
                    anyhow::anyhow!(
                        "no safetensors index given: pass --index PATH (or set SPARK_ENGRAM_INDEX)"
                    )
                })?;
                dir.join("model.safetensors.index.json")
            }
        };
        let min_free_bytes = self
            .min_free_gib
            .checked_mul(1024 * 1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("--min-free-gib {} overflows", self.min_free_gib))?;
        Ok(PackOptions {
            index_path,
            checkpoint_dir,
            output_dir: self.output_dir,
            layers: self.layers,
            world_size: self.world_size,
            ranks: self.rank,
            verify: self.verify,
            dry_run: self.dry_run,
            tokenizer_path: self.tokenizer,
            min_free_bytes,
            chunk_rows: self.chunk_rows,
            fsync: !self.no_fsync,
            force: self.force,
        })
    }
}

/// CLI entry point shared by the bin and the `sparkflash` subcommand.
///
/// # Errors
/// Propagates planning and I/O failures.
pub fn run_cli(args: CliArgs) -> Result<PackReport> {
    let options = args.into_options()?;
    run(&options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rounds_up_to_the_documented_stride() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 512);
        assert_eq!(align_up(512), 512);
        assert_eq!(align_up(513), 1024);
        // 96,001,542 rows * 264 bytes = 25,344,407,088 bytes.
        assert_eq!(align_up(25_344_407_088), 25_344_407_552);
    }

    #[test]
    fn pad_row_has_zero_weights_and_unit_scales() {
        let row = pad_row();
        assert_eq!(row.len(), 264);
        assert!(row[..256].iter().all(|byte| *byte == 0));
        assert!(row[256..].iter().all(|byte| *byte == PAD_SCALE_BYTE));
    }

    #[test]
    fn memory_floor_refuses_below_and_allows_above() {
        assert!(check_free_memory(100, 100).is_ok());
        assert!(check_free_memory(100, 101).is_ok());
        let error = check_free_memory(100, 99).unwrap_err().to_string();
        assert!(error.contains("refusing to pack"), "{error}");
        assert!(error.contains("GiB"), "{error}");
        assert!(mem_available_bytes().unwrap() > 0);
    }

    #[test]
    fn hex_encoding_is_lower_case_and_padded() {
        assert_eq!(to_hex(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
        assert_eq!(to_hex(&[]), "");
    }

    #[test]
    fn sample_rows_cover_both_ends() {
        assert!(sample_rows(0).is_empty());
        assert_eq!(sample_rows(1), vec![0]);
        let rows = sample_rows(1000);
        assert_eq!(rows.first(), Some(&0));
        assert_eq!(rows.last(), Some(&999));
        assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(rows.len() <= 66);
    }

    #[test]
    fn shard_file_name_matches_the_spec() {
        assert_eq!(shard_file_name(1, 0, 4), "engram-l1-r0of4.bin");
        assert_eq!(shard_file_name(14, 3, 4), "engram-l14-r3of4.bin");
    }

    #[test]
    fn options_reject_impossible_ranks() {
        let mut options = PackOptions::new("/tmp/index.json", "/tmp/out");
        options.ranks = vec![4];
        assert!(options.selected_ranks().is_err());
        options.ranks = vec![2, 1, 2];
        assert_eq!(options.selected_ranks().unwrap(), vec![1, 2]);
        options.world_size = 0;
        options.ranks.clear();
        assert!(options.selected_ranks().is_err());
    }

    /// Writes a safetensors shard whose header describes `offsets` of tensor data while the
    /// file only holds `present` data bytes after it.
    fn synthetic_shard(
        path: &Path,
        name: &str,
        shape: [u64; 2],
        offsets: [u64; 2],
        present: usize,
    ) {
        let header = serde_json::json!({
            name: {
                "dtype": "F8_E4M3",
                "shape": shape,
                "data_offsets": offsets,
            }
        });
        let mut json = serde_json::to_vec(&header).expect("serialise the header");
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        bytes.extend(std::iter::repeat_n(0u8, present));
        std::fs::write(path, bytes).expect("write the synthetic shard");
    }

    fn weight_plan() -> PlanTensor {
        PlanTensor {
            name: "layers.1.engram.embed.weight".to_owned(),
            dtype: "F8_E4M3".to_owned(),
            shape: [3, 8],
        }
    }

    /// A header may promise more data than the shard holds - a cut-short download looks
    /// exactly like this - and slicing the mapping at such an offset panics.
    #[test]
    fn a_truncated_shard_is_rejected_not_sliced() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("model-00047-of-00050.safetensors");
        synthetic_shard(&path, "layers.1.engram.embed.weight", [3, 8], [0, 24], 8);
        let header = SafetensorsHeader::read(&path).expect("the header itself is complete");
        assert_eq!(header.file_len, header.data_start + 8);
        let error = validate_tensor(&header, &weight_plan(), &path)
            .expect_err("the tensor does not fit the shard");
        let message = error.to_string();
        assert!(message.contains("truncated shard"), "{message}");
        assert!(message.contains("only"), "{message}");
    }

    #[test]
    fn inverted_data_offsets_are_rejected() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("model-00047-of-00050.safetensors");
        synthetic_shard(&path, "layers.1.engram.embed.weight", [3, 8], [24, 8], 32);
        let header = SafetensorsHeader::read(&path).expect("the header itself is complete");
        let error =
            validate_tensor(&header, &weight_plan(), &path).expect_err("the offsets are inverted");
        let message = error.to_string();
        assert!(message.contains("inverted data offsets"), "{message}");
    }

    /// A header that runs past the end of the file is refused while it is being parsed.
    #[test]
    fn a_truncated_header_is_rejected() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("model-00047-of-00050.safetensors");
        let mut bytes = 4096u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, &bytes).expect("write the truncated shard");
        let error = SafetensorsHeader::read(&path).expect_err("the header is missing");
        assert!(error.to_string().contains("truncated"), "{error}");
    }

    /// Writes a shard whose header is exactly `header`, followed by `present` data bytes.
    fn write_header_only_shard(path: &Path, header: &serde_json::Value, present: usize) {
        let mut json = serde_json::to_vec(header).expect("serialise the header");
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        bytes.extend(std::iter::repeat_n(0u8, present));
        std::fs::write(path, bytes).expect("write the shard");
    }

    /// A standard Hugging Face shard carries `__metadata__`: an object whose fields are not
    /// tensors and may be any JSON at all. It used to be deserialised as a tensor, and the
    /// whole read failed with "missing field dtype".
    #[test]
    fn a_metadata_object_is_skipped_whatever_json_it_holds() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let payloads = [
            serde_json::json!({}),
            serde_json::json!({"format": "pt"}),
            serde_json::json!({
                "format": "pt",
                "count": 7,
                "ratio": 0.5,
                "ok": true,
                "nothing": null,
                "tags": ["a", "b"],
                "nested": {"a": [1, 2, {"b": "c"}]},
            }),
        ];
        for (index, metadata) in payloads.iter().enumerate() {
            let path = dir.path().join(format!("model-{index}.safetensors"));
            write_header_only_shard(
                &path,
                &serde_json::json!({
                    METADATA_KEY: metadata,
                    "layers.1.engram.embed.weight": {
                        "dtype": WEIGHT_DTYPE,
                        "shape": [2u64, ENGRAM_HEAD_DIM as u64],
                        "data_offsets": [0u64, 2 * ENGRAM_HEAD_DIM as u64],
                    },
                }),
                1,
            );
            let header = SafetensorsHeader::read(&path).expect("a standard header parses");
            assert_eq!(header.tensors.len(), 1, "metadata must not become a tensor");
            assert!(!header.tensors.contains_key(METADATA_KEY));
            assert_eq!(header.metadata.as_ref(), Some(metadata));
        }
    }

    /// Only `__metadata__` may be a non-tensor; every other key has to be a tensor descriptor,
    /// and the error says which key was wrong.
    #[test]
    fn a_non_metadata_entry_that_is_not_a_tensor_is_rejected_by_name() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let path = dir.path().join("model-bad.safetensors");
        write_header_only_shard(
            &path,
            &serde_json::json!({
                METADATA_KEY: {"format": "pt"},
                "layers.1.engram.embed.weight": "pt",
            }),
            8,
        );
        let error = SafetensorsHeader::read(&path).expect_err("a string is not a tensor");
        // The top-level Display has to name the key: a bare "parse safetensors header of <path>"
        // makes the operator go hunting through the file, and the alternate form is not what a
        // caller prints by accident (review A6).
        let plain = format!("{error}");
        assert!(
            plain.contains("layers.1.engram.embed.weight"),
            "the top-level message must name the offending key: {plain}"
        );
        assert!(
            plain.contains("parse safetensors header of"),
            "the file being parsed must still be named: {plain}"
        );
        let chain = format!("{error:#}");
        assert!(
            chain.contains("layers.1.engram.embed.weight"),
            "the error must name the offending key: {chain}"
        );
    }

    /// `execute` is public, so it cannot rely on the caller having validated the options: a
    /// zero chunk size used to be accepted there and wrapped the chunk loop forever.
    #[test]
    fn a_zero_chunk_size_is_refused_before_any_work() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let mut options = PackOptions::new(dir.path().join("index.json"), dir.path());
        options.chunk_rows = 0;
        let error = options
            .selected_ranks()
            .expect_err("zero chunk rows is invalid");
        assert!(error.to_string().contains("chunk_rows"), "{error}");

        // An empty plan would otherwise succeed, so the only possible error here is the
        // option check itself.
        let plan = PackPlan {
            index_path: dir.path().join("index.json"),
            index_sha256: None,
            checkpoint_dir: dir.path().to_path_buf(),
            output_dir: dir.path().join("out"),
            world_size: 1,
            ranks: vec![0],
            layers: Vec::new(),
            total_rows: 0,
            total_data_bytes: 0,
            total_file_bytes: 0,
            tokenizer_path: None,
            tokenizer_sha256: None,
        };
        let error = execute(&plan, &options).expect_err("a zero chunk size is refused");
        assert!(
            error.to_string().contains("chunk_rows"),
            "execute must validate the options before it touches the checkpoint: {error}"
        );
        assert!(!plan.output_dir.join("engram-l1-r0of1.bin").exists());
    }

    /// Writes a safetensors shard holding the weight and scale tensors of `rows` rows.
    fn write_layer_shard(path: &Path, rows: u64) {
        write_layer_shard_with_metadata(path, rows, None);
    }

    /// The same shard, optionally carrying a `__metadata__` object - which is what a real
    /// Hugging Face shard has and what the reader used to reject.
    fn write_layer_shard_with_metadata(
        path: &Path,
        rows: u64,
        metadata: Option<&serde_json::Value>,
    ) {
        let weight_bytes = rows * ENGRAM_HEAD_DIM as u64;
        let scale_bytes = rows * ENGRAM_SCALE_BYTES as u64;
        let mut header = serde_json::json!({
            "layers.1.engram.embed.weight": {
                "dtype": WEIGHT_DTYPE,
                "shape": [rows, ENGRAM_HEAD_DIM as u64],
                "data_offsets": [0u64, weight_bytes],
            },
            "layers.1.engram.embed.scale": {
                "dtype": SCALE_DTYPE,
                "shape": [rows, ENGRAM_SCALE_BYTES as u64],
                "data_offsets": [weight_bytes, weight_bytes + scale_bytes],
            },
        });
        if let Some(metadata) = metadata {
            header[METADATA_KEY] = metadata.clone();
        }
        let mut json = serde_json::to_vec(&header).expect("serialise the header");
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        for index in 0..(weight_bytes + scale_bytes) as usize {
            bytes.push((index % 251) as u8);
        }
        std::fs::write(path, bytes).expect("write the layer shard");
    }

    /// The reviewer's regression was in the reader the packer itself uses, so the regression
    /// test has to run the whole `execute` path over a shard whose header carries metadata.
    #[test]
    fn a_shard_with_metadata_packs_end_to_end() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let shard = "model-00047-of-00050.safetensors";
        let metadata = serde_json::json!({
            "format": "pt",
            "count": 3,
            "ok": true,
            "nested": {"a": [1, 2]},
        });
        write_layer_shard_with_metadata(&dir.path().join(shard), 4, Some(&metadata));
        let plan = small_plan(dir.path(), shard, 4);
        let options = PackOptions::new(dir.path().join("index.json"), dir.path().join("out"));
        execute(&plan, &options).expect("a standard header with metadata must pack");
        let written =
            std::fs::read(plan.output_dir.join("engram-l1-r0of1.bin")).expect("rank file");
        assert_eq!(written.len() as u64, plan.layers[0].shards[0].file_bytes);
        let source_path = dir.path().join(shard);
        let header = SafetensorsHeader::read(&source_path).expect("header");
        assert_eq!(header.metadata.as_ref(), Some(&metadata));
        let source = std::fs::read(&source_path).expect("source shard");
        let data_start = header.data_start as usize;
        // Row 0 is the interleave of the weight tensor's row 0 and the scale tensor's row 0.
        assert_eq!(
            &written[..ENGRAM_HEAD_DIM],
            &source[data_start..data_start + ENGRAM_HEAD_DIM]
        );
        let scale_start = data_start + 4 * ENGRAM_HEAD_DIM;
        assert_eq!(
            &written[ENGRAM_HEAD_DIM..ENGRAM_ROW_BYTES],
            &source[scale_start..scale_start + ENGRAM_SCALE_BYTES]
        );
        // Everything past the fourth row is the zero padding that aligns the file.
        assert!(
            written[4 * ENGRAM_ROW_BYTES..]
                .iter()
                .all(|byte| *byte == 0),
            "the file padding must be zero"
        );
    }

    /// A one-layer plan over `write_layer_shard`, so the whole `execute` path can run on a
    /// few kilobytes instead of the 101 GB checkpoint it is normally pointed at.
    fn small_plan(dir: &Path, shard_name: &str, rows: u64) -> PackPlan {
        let data_bytes = rows * ENGRAM_ROW_BYTES as u64;
        let file_bytes = align_up(data_bytes);
        PackPlan {
            index_path: dir.join("index.json"),
            index_sha256: None,
            checkpoint_dir: dir.to_path_buf(),
            output_dir: dir.join("out"),
            world_size: 1,
            ranks: vec![0],
            layers: vec![LayerPlan {
                layer: 1,
                layer_index: 0,
                source_shard: shard_name.to_owned(),
                weight: PlanTensor {
                    name: "layers.1.engram.embed.weight".to_owned(),
                    dtype: WEIGHT_DTYPE.to_owned(),
                    shape: [rows, ENGRAM_HEAD_DIM as u64],
                },
                scale: PlanTensor {
                    name: "layers.1.engram.embed.scale".to_owned(),
                    dtype: SCALE_DTYPE.to_owned(),
                    shape: [rows, ENGRAM_SCALE_BYTES as u64],
                },
                table_rows: rows,
                row_bytes: ENGRAM_ROW_BYTES as u64,
                shards: vec![ShardPlan {
                    rank: 0,
                    file: "engram-l1-r0of1.bin".to_owned(),
                    row_start: 0,
                    row_end: rows,
                    rows,
                    source_rows: rows,
                    padded_rows: 0,
                    data_bytes,
                    file_bytes,
                    padding_bytes: file_bytes - data_bytes,
                }],
            }],
            total_rows: rows,
            total_data_bytes: data_bytes,
            total_file_bytes: file_bytes,
            tokenizer_path: None,
            tokenizer_sha256: None,
        }
    }

    /// The reviewer reproducer: a shard whose header describes the full tensor while the file
    /// stops early. Slicing the mapping used to panic; now `execute` refuses it.
    #[test]
    fn a_truncated_shard_fails_execute_instead_of_panicking() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let rows = 40u64;
        let shard_name = "model-00047-of-00048.safetensors";
        let shard = dir.path().join(shard_name);
        write_layer_shard(&shard, rows);
        let file_len = std::fs::metadata(&shard).expect("stat the shard").len();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .expect("open the shard");
        file.set_len(file_len - 1_000).expect("truncate the shard");
        drop(file);

        let plan = small_plan(dir.path(), shard_name, rows);
        let options = PackOptions::new(plan.index_path.clone(), plan.output_dir.clone());
        let error = execute(&plan, &options).expect_err("the shard is truncated");
        assert!(error.to_string().contains("truncated shard"), "{error}");
        assert!(
            !plan.output_dir.join("engram-l1-r0of1.bin").exists(),
            "no rank file may be written from a truncated shard"
        );
    }
}

/// End-to-end check that what the packer writes is exactly what the store reads back.
///
/// The writer is private, so this test lives beside it; everything it needs is public, so it
/// exercises the same `write_shard` the binary calls and the same `DirectTable` the server
/// calls, over a real file in a temporary directory.
#[cfg(test)]
mod round_trip {
    use super::*;
    use crate::store::{DirectTable, RowTable, TableSpec, TailFill};
    use std::fs::File;

    /// Weight bytes of a synthetic source row.
    fn weight(row: u64) -> Vec<u8> {
        (0..ENGRAM_HEAD_DIM)
            .map(|index| (row as u8).wrapping_add(index as u8))
            .collect()
    }

    /// Scale bytes of a synthetic source row.
    fn scale(row: u64) -> Vec<u8> {
        (0..ENGRAM_SCALE_BYTES)
            .map(|index| 0x7F_u8.wrapping_add(row as u8).wrapping_add(index as u8))
            .collect()
    }

    #[test]
    fn a_written_shard_reads_back_through_the_store() {
        let dir = tempfile::tempdir().expect("temporary directory");
        // 4,099 rows with two padded tail rows: more than one chunk, and both ends of the
        // chunk loop are exercised.
        let rows = 4_099u64;
        let source_rows = 4_097u64;
        let mut payload = vec![0u8; (rows * ENGRAM_ROW_BYTES as u64) as usize];
        for row in 0..source_rows {
            let start = (row * ENGRAM_HEAD_DIM as u64) as usize;
            payload[start..start + ENGRAM_HEAD_DIM].copy_from_slice(&weight(row));
        }
        let scale_base = (source_rows * ENGRAM_HEAD_DIM as u64) as usize;
        for row in 0..source_rows {
            let start = scale_base + (row * ENGRAM_SCALE_BYTES as u64) as usize;
            payload[start..start + ENGRAM_SCALE_BYTES].copy_from_slice(&scale(row));
        }
        let source_path = dir.path().join("synthetic-source.bin");
        std::fs::write(&source_path, &payload).expect("write the synthetic source");
        let source_file = File::open(&source_path).expect("open the synthetic source");
        // SAFETY: the file is not modified while it is mapped.
        let map = unsafe { Mmap::map(&source_file) }.expect("map the synthetic source");
        let source = SourceLayer {
            map,
            weight_offset: 0,
            scale_offset: scale_base as u64,
        };

        let data_bytes = rows * ENGRAM_ROW_BYTES as u64;
        let file_bytes = align_up(data_bytes);
        let shard = ShardPlan {
            rank: 0,
            file: "engram-l1-r0of1.bin".to_owned(),
            row_start: 0,
            row_end: rows,
            rows,
            source_rows,
            padded_rows: rows - source_rows,
            data_bytes,
            file_bytes,
            padding_bytes: file_bytes - data_bytes,
        };
        let mut options = PackOptions::new(dir.path().join("index.json"), dir.path());
        options.chunk_rows = 1_000;
        let out = dir.path().join(&shard.file);
        let sha = write_shard(&source, &shard, &out, &options).expect("write the shard");
        assert_eq!(sha, sha256_file(&out).expect("hash the shard"));
        assert_eq!(
            std::fs::metadata(&out).expect("stat").len(),
            file_bytes,
            "the writer must pad to the declared file length"
        );

        // The reader must return exactly the interleaved source rows, and the tail rows the
        // reference padding, not whatever happens to be in the file.
        let table = DirectTable::open(vec![TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: out.clone(),
            row_start: 0,
            physical_rows: rows,
            logical_rows: rows,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        }])
        .expect("open the packed shard");
        for row in [0u64, 1, 999, 1_000, 4_096] {
            let got = table.read_row(0, row).expect("read a real row");
            assert_eq!(
                &got[..ENGRAM_HEAD_DIM],
                &weight(row)[..],
                "weights of row {row}"
            );
            assert_eq!(
                &got[ENGRAM_HEAD_DIM..],
                &scale(row)[..],
                "scales of row {row}"
            );
        }
        let padding = pad_row();
        for row in source_rows..rows {
            let got = table.read_row(0, row).expect("read a padded row");
            assert_eq!(got, padding, "row {row} is padding");
        }

        // The manifest's sha256 and geometry describe the file on disk.
        let manifest = serde_json::json!({
            "format": "sparkflash-engram",
            "file_bytes": file_bytes,
            "sha256": sha,
            "rows": shard.rows,
            "source_rows": shard.source_rows,
        });
        assert_eq!(manifest["file_bytes"], file_bytes);
        assert_eq!(manifest["sha256"], sha);
        assert_eq!(manifest["rows"], rows);
        assert_eq!(manifest["source_rows"], source_rows);
    }

    /// The golden fixture is generated by an independent implementation: small python, see
    /// `tests/fixtures/store/golden_interleave.py`. The writer must reproduce those bytes
    /// exactly, including the reference padding of the short rank tail and the 512-byte file
    /// alignment, so the committed fixture pins the layout rather than the writer pinning
    /// itself.
    #[test]
    fn the_writer_reproduces_the_independently_generated_shard() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/store");
        let text = std::fs::read_to_string(format!("{fixture}/golden_expect.json"))
            .expect("read golden_expect.json");
        let expect: serde_json::Value =
            serde_json::from_str(&text).expect("parse golden_expect.json");
        let rows = expect["rows"].as_u64().expect("rows");
        let source_rows = expect["source_rows"].as_u64().expect("source_rows");
        let file_bytes = expect["file_bytes"].as_u64().expect("file_bytes");

        let source_path = PathBuf::from(format!("{fixture}/golden_source.bin"));
        let source_file = File::open(&source_path).expect("open the golden source");
        // SAFETY: the fixture is not modified while it is mapped.
        let map = unsafe { Mmap::map(&source_file) }.expect("map the golden source");
        let source = SourceLayer {
            map,
            weight_offset: 0,
            scale_offset: source_rows * ENGRAM_HEAD_DIM as u64,
        };

        let data_bytes = rows * ENGRAM_ROW_BYTES as u64;
        assert_eq!(align_up(data_bytes), file_bytes);
        let shard = ShardPlan {
            rank: 0,
            file: "engram-l1-r0of1.bin".to_owned(),
            row_start: 0,
            row_end: rows,
            rows,
            source_rows,
            padded_rows: rows - source_rows,
            data_bytes,
            file_bytes,
            padding_bytes: file_bytes - data_bytes,
        };
        let options = PackOptions::new(dir.path().join("index.json"), dir.path());
        let out = dir.path().join(&shard.file);
        let sha = write_shard(&source, &shard, &out, &options).expect("write the shard");

        let golden =
            std::fs::read(format!("{fixture}/golden_l1_r0of1.bin")).expect("read the golden shard");
        let written = std::fs::read(&out).expect("read the written shard");
        assert_eq!(
            written, golden,
            "the writer must reproduce the independent golden byte for byte"
        );
        assert_eq!(sha, expect["file_sha256"].as_str().expect("file_sha256"));
        assert_eq!(
            sha256_file(&source_path).expect("hash the golden source"),
            expect["source_sha256"].as_str().expect("source_sha256")
        );
    }
}
