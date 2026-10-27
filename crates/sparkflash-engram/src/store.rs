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

//! On-disk Engram table reader: 512-byte-aligned windows sliced to dense 264-byte rows.
//!
//! `crate::packer` writes rows densely at 264 bytes and pads each file to a 512-byte
//! multiple. A row therefore starts at an arbitrary 8-byte offset inside a 512-byte
//! sector, and 264 is not a divisor of 512, so a row *straddles* two windows roughly one
//! time in two (`264 * 2 > 512`). Every read here is expressed as a whole number of
//! 512-byte-aligned windows and sliced afterwards, which is what makes `O_DIRECT` legal:
//! the kernel wants the file offset, the length *and* the buffer address to be block
//! aligned, and an unaligned request is rejected with `EINVAL` rather than being fixed up.
//!
//! Three readers sit behind the same trait:
//!
//! * `[DirectTable]` opens each rank file with `O_DIRECT` and issues aligned windowed
//!   reads through `pread`. The alignment it requires is derived per file from the
//!   logical block size of the device behind it (`[device_alignment]`) and every handle
//!   is probed with one real aligned read, because `open(O_DIRECT)` succeeding proves
//!   nothing on its own (tmpfs accepts the flag and ignores it). If `O_DIRECT` is
//!   unsupported (tmpfs, some overlay filesystems) the reader falls back to an `mmap`
//!   plus `MADV_RANDOM` and records why; if the kernel refuses a direct read later it
//!   latches the plain-handle path instead of pretending to be direct. Opening only fails
//!   because of `O_DIRECT` when the caller asked for it with `[BackendPreference::Require]`.
//! * `[MemoryTable]` is the test double: no files, no NVMe, deterministic.
//! * any caller can implement `[RowTable]`, which is what the cache and prefetch tests do.
//!
//! Rows past the logical end of a table (the padded tail the packer writes for the last
//! rank) are synthesised from `[TailFill]` and never read from disk, so a short file
//! cannot produce a read past EOF.

use std::fs::File;
use std::os::unix::fs::{FileExt, FileTypeExt, MetadataExt};
use std::os::unix::io::FromRawFd;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Advice, Mmap};

use crate::layout::{ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_ROW_BYTES};
use crate::packer::{PAD_SCALE_BYTE, align_up};

/// Minimum alignment of an `O_DIRECT` request (offset, length and buffer address).
///
/// This is the floor, not the whole story: a device with a larger logical block size
/// (4Kn drives, dm-crypt volumes, some NVMe namespaces) rejects any request that is not a
/// multiple of *its* logical block size with `EINVAL`, so the reader also derives an
/// alignment per file with `[device_alignment]` and uses `[effective_window]` to raise a
/// window to it. A 512-byte device keeps both 512-byte alignments and 512-byte windows.
pub const MIN_ALIGN_BYTES: u64 = 512;

/// Alignment assumed when neither the block device nor the filesystem reports one.
pub const DEFAULT_ALIGN_BYTES: u64 = MIN_ALIGN_BYTES;

/// Largest alignment this reader can serve with `O_DIRECT`; the scratch buffers are
/// aligned to this much, so a device with a bigger logical block size is served through
/// the mapped backend instead of failing every read.
pub const MAX_ALIGN_BYTES: u64 = 4096;

/// Default window size. 4 KiB is the logical block size of the target PCIe Gen5 NVMe
/// devices and keeps ~15 rows in one request while still being a multiple of
/// `[MIN_ALIGN_BYTES]`; tests use 512 to force the straddle path on every row.
pub const DEFAULT_WINDOW_BYTES: u64 = 4096;

/// Default gap below which two requested rows are merged into one read.
pub const DEFAULT_MERGE_GAP_BYTES: u64 = DEFAULT_WINDOW_BYTES;

/// Default cap on one batched read, so a fully dense batch of millions of rows is split
/// into many bounded requests instead of one enormous scratch buffer.
pub const DEFAULT_MAX_IO_BYTES: u64 = 1024 * 1024;

/// Buffer alignment used for `O_DIRECT` scratch space; a multiple of every logical block
/// size this reader supports.
const SCRATCH_ALIGN: usize = MAX_ALIGN_BYTES as usize;

/// How a row past the logical end of a table is served.
///
/// The packer materialises the same bytes on disk, so both paths agree; the variant
/// exists because a caller may also point the reader at a file that contains only the
/// real rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailFill {
    /// Exactly what `convert.py` and `[crate::packer]` write: zero weights and a ue8m0
    /// exponent of 127 (1.0) in all eight block scales.
    #[default]
    Reference,
    /// All-zero bytes.
    Zeros,
}

impl TailFill {
    /// The row this fill produces.
    #[must_use]
    pub fn row(&self) -> [u8; ENGRAM_ROW_BYTES] {
        match self {
            Self::Reference => {
                let mut row = [0u8; ENGRAM_ROW_BYTES];
                row[ENGRAM_HEAD_DIM..].fill(PAD_SCALE_BYTE);
                row
            }
            Self::Zeros => [0u8; ENGRAM_ROW_BYTES],
        }
    }
}

/// Read path a table is actually being served on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadBackend {
    /// Opened with `O_DIRECT` and serving every request as an aligned windowed `pread`.
    Direct,
    /// `O_DIRECT` was unsupported at open time: the file is mapped with `MADV_RANDOM`.
    Mapped,
    /// The kernel refused an `O_DIRECT` request at read time (typically `EINVAL` on a
    /// device whose logical block size exceeds the request alignment); reads use the plain
    /// file handle from then on, which is slower but always correct.
    Buffered,
}

/// Which backend `[DirectTable::open]` should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Try `O_DIRECT`, fall back to the mapping, never fail because of alignment.
    #[default]
    Auto,
    /// Require `O_DIRECT`; a filesystem that rejects it is an error (used by tests that
    /// want to prove the direct path really engaged).
    Require,
    /// Skip `O_DIRECT` entirely.
    Mapped,
}

/// One rank file of one Engram layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSpec {
    /// Backbone layer id (1 or 14).
    pub layer_id: usize,
    /// Index into `[ENGRAM_LAYERS]`.
    pub layer_index: usize,
    /// Rank that owns this file.
    pub rank: usize,
    /// World size the file was packed for.
    pub world_size: usize,
    /// Path of the rank file.
    pub path: PathBuf,
    /// First global table row stored in this file.
    pub row_start: u64,
    /// Rows physically stored in the file, including the padded tail.
    pub physical_rows: u64,
    /// Rows the global table actually has (the last rank's tail exceeds this).
    pub logical_rows: u64,
    /// Aligned window size in bytes; a multiple of `[MIN_ALIGN_BYTES]`.
    pub window_bytes: u64,
    /// How the padded tail is served.
    pub tail_fill: TailFill,
}

impl TableSpec {
    /// Spec for one rank of one layer, deriving every row count from `[crate::layout]`.
    ///
    /// # Errors
    /// When the layer is not an Engram layer or the rank is outside the world.
    pub fn new(
        layer_id: usize,
        rank: usize,
        world_size: usize,
        path: impl Into<PathBuf>,
    ) -> Result<Self> {
        let layer_index = ENGRAM_LAYERS
            .iter()
            .position(|candidate| *candidate == layer_id)
            .ok_or_else(|| anyhow::anyhow!("layer {layer_id} has no Engram table"))?;
        ensure!(
            rank < world_size,
            "rank {rank} is outside world size {world_size}"
        );
        let logical_rows = crate::layout::num_embeddings(layer_index);
        Ok(Self {
            layer_id,
            layer_index,
            rank,
            world_size,
            path: path.into(),
            row_start: rank as u64 * crate::layout::rows_per_rank(layer_index, world_size),
            physical_rows: crate::layout::rows_per_rank(layer_index, world_size),
            logical_rows,
            window_bytes: DEFAULT_WINDOW_BYTES,
            tail_fill: TailFill::default(),
        })
    }

    /// Overrides the window size.
    ///
    /// # Errors
    /// When `window_bytes` is not a non-zero multiple of `[MIN_ALIGN_BYTES]`.
    pub fn with_window_bytes(mut self, window_bytes: u64) -> Result<Self> {
        ensure!(
            window_bytes >= MIN_ALIGN_BYTES && window_bytes.is_multiple_of(MIN_ALIGN_BYTES),
            "window size {window_bytes} must be a non-zero multiple of {MIN_ALIGN_BYTES}"
        );
        self.window_bytes = window_bytes;
        Ok(self)
    }

    /// Overrides the padded-tail fill.
    #[must_use]
    pub fn with_tail_fill(mut self, tail_fill: TailFill) -> Self {
        self.tail_fill = tail_fill;
        self
    }

    /// Rows physically stored in the file.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.physical_rows
    }

    /// Lowest global row of this file.
    #[must_use]
    pub fn global_start(&self) -> u64 {
        self.row_start
    }

    /// Global row of a local row index.
    #[must_use]
    pub fn global_row(&self, local: u64) -> u64 {
        self.row_start + local
    }

    /// Local row index of a global row, when this rank owns it.
    #[must_use]
    pub fn local_row(&self, global: u64) -> Option<u64> {
        let local = global.checked_sub(self.row_start)?;
        (local < self.physical_rows).then_some(local)
    }

    /// True when the local row has no source row behind it (the padded tail).
    #[must_use]
    pub fn is_tail(&self, local: u64) -> bool {
        self.global_row(local) >= self.logical_rows
    }

    /// Rows of this file that exist in the source table.
    #[must_use]
    pub fn source_rows(&self) -> u64 {
        self.logical_rows
            .saturating_sub(self.row_start)
            .min(self.physical_rows)
    }

    /// Length the packed file must have.
    #[must_use]
    pub fn file_bytes(&self) -> u64 {
        align_up(self.physical_rows * ENGRAM_ROW_BYTES as u64)
    }
}

/// Byte span of the aligned windows that cover `[offset, offset + len)`.
///
/// The returned `(start, end)` are both multiples of `window`, and `end` is exclusive.
#[must_use]
pub fn window_span(offset: u64, len: u64, window: u64) -> (u64, u64) {
    debug_assert!(window > 0);
    let start = offset / window * window;
    let end = (offset + len).div_ceil(window) * window;
    (start, end)
}

/// Random access to packed Engram rows, addressed by *local* row index inside a rank file.
///
/// `layer_id`/`local_row` translation lives on `[TableSpec]`; use `[RowTable::read_global]`
/// when the caller only has a global hash id.
pub trait RowTable: Send + Sync {
    /// Number of rank files (tables) this reader exposes.
    fn table_count(&self) -> usize;

    /// Metadata of a table.
    fn spec(&self, table: usize) -> Option<&TableSpec>;

    /// Reads one row by local index.
    ///
    /// # Errors
    /// When `table` is out of range or `row` is past the end of that file.
    fn read_row(&self, table: usize, row: u64) -> Result<[u8; ENGRAM_ROW_BYTES]>;

    /// Reads many rows, merging nearby offsets into as few system calls as possible.
    ///
    /// # Errors
    /// When any requested row is out of range or an I/O fails.
    fn read_rows(&self, table: usize, rows: &[u32]) -> Result<Vec<[u8; ENGRAM_ROW_BYTES]>> {
        let mut out = vec![[0u8; ENGRAM_ROW_BYTES]; rows.len()];
        self.read_rows_into(table, rows, &mut out)?;
        Ok(out)
    }

    /// Batched read into a caller-owned buffer.
    ///
    /// # Errors
    /// When `out` and `rows` have different lengths, a row is out of range, or an I/O
    /// fails.
    fn read_rows_into(
        &self,
        table: usize,
        rows: &[u32],
        out: &mut [[u8; ENGRAM_ROW_BYTES]],
    ) -> Result<()> {
        ensure!(
            rows.len() == out.len(),
            "{} rows requested but {} output slots given",
            rows.len(),
            out.len()
        );
        for (slot, row) in out.iter_mut().zip(rows) {
            *slot = self.read_row(table, *row as u64)?;
        }
        Ok(())
    }

    /// Underlying read operations issued so far. Used by tests to prove batching really
    /// merged requests; implementations that do no I/O report zero.
    fn read_ops(&self) -> u64 {
        0
    }

    /// Reads a row by *global* table row, returning `None` when the rank files behind this
    /// reader do not own it.
    ///
    /// # Errors
    /// When the owning table exists but the read fails, or two tables claim the same
    /// global row.
    fn read_global(
        &self,
        layer_id: usize,
        global_row: u64,
    ) -> Result<Option<[u8; ENGRAM_ROW_BYTES]>> {
        let mut found = None;
        for table in 0..self.table_count() {
            let Some(spec) = self.spec(table) else {
                continue;
            };
            if spec.layer_id != layer_id {
                continue;
            }
            if let Some(local) = spec.local_row(global_row) {
                ensure!(
                    found.is_none(),
                    "global row {global_row} of layer {layer_id} is owned by more than one table"
                );
                found = Some(self.read_row(table, local)?);
            }
        }
        Ok(found)
    }
}

/// In-memory `[RowTable]` for tests and for validating a plan with no files at all.
///
/// Tail rows resolve exactly like the on-disk reader's: through `[TableSpec::tail_fill]`,
/// never through the stored rows.
#[derive(Debug, Default)]
pub struct MemoryTable {
    layers: Vec<MemoryLayer>,
    reads: std::sync::atomic::AtomicU64,
}

/// One in-memory table.
#[derive(Debug, Clone)]
pub struct MemoryLayer {
    /// Metadata, including the row range and tail fill.
    pub spec: TableSpec,
    /// Stored rows; a tail row's contents are ignored in favour of the tail fill.
    pub data: Vec<[u8; ENGRAM_ROW_BYTES]>,
}

impl MemoryTable {
    /// An empty reader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a table whose rows are all zero, sized from the spec.
    #[must_use]
    pub fn with_layer(mut self, spec: TableSpec) -> Self {
        let rows = spec.physical_rows as usize;
        self.layers.push(MemoryLayer {
            spec,
            data: vec![[0u8; ENGRAM_ROW_BYTES]; rows],
        });
        self
    }

    /// Appends a table with explicit row contents.
    ///
    /// # Errors
    /// When `data` does not have exactly `spec.physical_rows` entries.
    pub fn with_layer_data(
        mut self,
        spec: TableSpec,
        data: Vec<[u8; ENGRAM_ROW_BYTES]>,
    ) -> Result<Self> {
        ensure!(
            data.len() as u64 == spec.physical_rows,
            "{} rows of data for a table with {} rows",
            data.len(),
            spec.physical_rows
        );
        self.layers.push(MemoryLayer { spec, data });
        Ok(self)
    }

    /// Overwrites one non-tail row.
    ///
    /// # Errors
    /// When the table or row index is out of range.
    pub fn set_row(&mut self, table: usize, row: u64, data: &[u8; ENGRAM_ROW_BYTES]) -> Result<()> {
        let layer = self
            .layers
            .get_mut(table)
            .ok_or_else(|| anyhow::anyhow!("table {table} is out of range"))?;
        let slot = layer
            .data
            .get_mut(row as usize)
            .ok_or_else(|| anyhow::anyhow!("row {row} is out of range for table {table}"))?;
        *slot = *data;
        Ok(())
    }

    /// Writes a deterministic, self-describing pattern into every non-tail row: the
    /// little-endian row index in the first eight weight bytes and a per-byte mix over the
    /// rest, so a mis-sliced read cannot coincidentally match.
    pub fn fill_pattern(&mut self, table: usize) -> Result<()> {
        let layer = self
            .layers
            .get_mut(table)
            .ok_or_else(|| anyhow::anyhow!("table {table} is out of range"))?;
        let layer_id = layer.spec.layer_id as u64;
        let base = layer.spec.row_start;
        for (index, row) in layer.data.iter_mut().enumerate() {
            let global = base + index as u64;
            row[..8].copy_from_slice(&global.to_le_bytes());
            for (offset, byte) in row[8..].iter_mut().enumerate() {
                *byte = (global as u8)
                    .wrapping_mul(31)
                    .wrapping_add((offset as u8).wrapping_mul(7))
                    .wrapping_add(layer_id as u8);
            }
        }
        Ok(())
    }

    /// Number of rows stored in a table.
    #[must_use]
    pub fn stored_rows(&self, table: usize) -> usize {
        self.layers.get(table).map_or(0, |layer| layer.data.len())
    }
}

impl RowTable for MemoryTable {
    fn table_count(&self) -> usize {
        self.layers.len()
    }

    fn spec(&self, table: usize) -> Option<&TableSpec> {
        self.layers.get(table).map(|layer| &layer.spec)
    }

    fn read_row(&self, table: usize, row: u64) -> Result<[u8; ENGRAM_ROW_BYTES]> {
        let layer = self
            .layers
            .get(table)
            .ok_or_else(|| anyhow::anyhow!("table {table} is out of range"))?;
        ensure!(
            row < layer.spec.physical_rows,
            "row {row} is past the {} rows of table {table}",
            layer.spec.physical_rows
        );
        self.reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if layer.spec.is_tail(row) {
            return Ok(layer.spec.tail_fill.row());
        }
        Ok(layer.data[row as usize])
    }

    fn read_rows_into(
        &self,
        table: usize,
        rows: &[u32],
        out: &mut [[u8; ENGRAM_ROW_BYTES]],
    ) -> Result<()> {
        ensure!(
            rows.len() == out.len(),
            "{} rows requested but {} output slots given",
            rows.len(),
            out.len()
        );
        let layer = self
            .layers
            .get(table)
            .ok_or_else(|| anyhow::anyhow!("table {table} is out of range"))?;
        for (slot, row) in out.iter_mut().zip(rows) {
            let row = *row as u64;
            ensure!(
                row < layer.spec.physical_rows,
                "row {row} is past the {} rows of table {table}",
                layer.spec.physical_rows
            );
            *slot = if layer.spec.is_tail(row) {
                layer.spec.tail_fill.row()
            } else {
                layer.data[row as usize]
            };
        }
        self.reads
            .fetch_add(rows.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn read_ops(&self) -> u64 {
        self.reads.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A heap buffer whose first byte is 4096-byte aligned, as `O_DIRECT` requires.
pub(crate) struct AlignedBuffer {
    storage: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBuffer {
    pub(crate) fn new(len: usize) -> Self {
        let storage = vec![0u8; len + SCRATCH_ALIGN];
        let base = storage.as_ptr() as usize;
        let offset = (SCRATCH_ALIGN - base % SCRATCH_ALIGN) % SCRATCH_ALIGN;
        Self {
            storage,
            offset,
            len,
        }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        let (offset, len) = (self.offset, self.len);
        &mut self.storage[offset..offset + len]
    }
}

/// Alignment an `O_DIRECT` request needs, given the block sizes the kernel reports.
///
/// `logical` is the logical block size of the block device behind the file when it could
/// be determined (sysfs `queue/logical_block_size`, or `BLKSSZGET` when the path is itself
/// a block device). `preferred` is `statx`'s `stx_blksize`: on ext4 and tmpfs that is
/// the same 4096 the device uses, but some filesystems report a *preferred I/O size*
/// instead (ZFS reports 128 KiB), so it is only a clamped fallback.
///
/// The result is at least `[MIN_ALIGN_BYTES]` and rounded up to a multiple of it, because
/// the packed files are 512-byte aligned and every logical block size a device reports is a
/// power-of-two multiple of 512.
#[must_use]
pub fn alignment_from_block_sizes(logical: Option<u64>, preferred: Option<u64>) -> u64 {
    let chosen = match logical {
        Some(value) if value >= MIN_ALIGN_BYTES => value,
        _ => preferred
            .unwrap_or(DEFAULT_ALIGN_BYTES)
            .clamp(MIN_ALIGN_BYTES, MAX_ALIGN_BYTES),
    };
    align_up(chosen).max(MIN_ALIGN_BYTES)
}

/// `O_DIRECT` alignment required for `path`.
///
/// Resolution order: the logical block size of the device backing the file, then
/// `BLKSSZGET` when `path` is a block device, then `statx`'s preferred I/O size clamped to
/// `[MIN_ALIGN_BYTES]`..=`[MAX_ALIGN_BYTES]`, then `[DEFAULT_ALIGN_BYTES]`.
#[must_use]
pub fn device_alignment(path: &Path) -> u64 {
    alignment_from_block_sizes(logical_block_size(path), preferred_io_size(path))
}

/// Window a table's reads actually use.
///
/// A window has to be a legal `O_DIRECT` length, so it is raised to the device alignment
/// when the spec asks for less; a spec that asks for more keeps its own window. On a
/// 512-byte device a 512-byte window stays 512 bytes, and on a 4096-byte device it becomes
/// 4096.
#[must_use]
pub fn effective_window(window_bytes: u64, align_bytes: u64) -> u64 {
    let align = align_bytes.max(MIN_ALIGN_BYTES);
    window_bytes.max(align).div_ceil(align) * align
}

/// Logical block size of the device behind `path`, when it can be determined.
fn logical_block_size(path: &Path) -> Option<u64> {
    let metadata = std::fs::metadata(path).ok()?;
    if let Some(size) = sysfs_logical_block_size(metadata.dev()) {
        return Some(size);
    }
    // A path that *is* a block device is not reachable through /sys/dev/block by st_dev on
    // every kernel, so ask the device itself.
    if metadata.file_type().is_block_device() {
        return blksszget(path);
    }
    None
}

/// `/sys/dev/block/<major>:<minor>/queue/logical_block_size` for a device number.
fn sysfs_logical_block_size(device: u64) -> Option<u64> {
    let (major, minor) = (libc::major(device), libc::minor(device));
    let path = PathBuf::from(format!(
        "/sys/dev/block/{major}:{minor}/queue/logical_block_size"
    ));
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|size| *size >= MIN_ALIGN_BYTES)
}

/// `BLKSSZGET` on a block-device path: the same number `blockdev --getss` prints.
fn blksszget(path: &Path) -> Option<u64> {
    use std::os::unix::io::AsRawFd;
    let file = File::open(path).ok()?;
    let mut size: libc::c_int = 0;
    // SAFETY: BLKSSZGET writes one c_int through the pointer for a block-device descriptor.
    let result = unsafe { libc::ioctl(file.as_raw_fd(), libc::BLKSSZGET, &mut size) };
    (result == 0 && size >= MIN_ALIGN_BYTES as libc::c_int).then_some(size as u64)
}

/// `statx`'s `stx_blksize` (Rust exposes it as `MetadataExt::blksize`).
fn preferred_io_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .map(|metadata| metadata.blksize())
}

/// True for the errors a kernel returns when it cannot satisfy an `O_DIRECT` request: a
/// request that is not a multiple of the device's logical block size, or a filesystem that
/// does not support direct I/O at all. (`ENOTSUP` and `EOPNOTSUPP` are the same value on
/// Linux.)
fn is_alignment_error(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL) | Some(libc::ENOTSUP)
    )
}

/// One opened rank file plus the handles the chosen backend needs.
#[derive(Debug)]
struct OpenTable {
    spec: TableSpec,
    /// `O_DIRECT` handle; `None` for the mapped backend.
    direct: Option<File>,
    /// Plain handle: always available, and the one used for any request the direct path
    /// cannot serve (the EOF-clamped tail, and everything after a refused direct read).
    buffered: File,
    /// Mapping used by the mapped backend.
    map: Option<Mmap>,
    /// Actual file length.
    file_bytes: u64,
    /// `O_DIRECT` alignment this file's device requires; at least `[MIN_ALIGN_BYTES]`.
    align: u64,
    /// Window used to express reads: the spec's window raised to `align`.
    read_window: u64,
    /// Set when the kernel refused a direct read at run time; reads then go through
    /// `buffered` and `[OpenTable::backend]` reports `[ReadBackend::Buffered]`.
    direct_failed: std::sync::atomic::AtomicBool,
}

impl OpenTable {
    /// Backend this table is actually being read through.
    fn backend(&self) -> ReadBackend {
        match &self.direct {
            Some(_)
                if !self
                    .direct_failed
                    .load(std::sync::atomic::Ordering::Relaxed) =>
            {
                ReadBackend::Direct
            }
            Some(_) => ReadBackend::Buffered,
            None if self.map.is_some() => ReadBackend::Mapped,
            None => ReadBackend::Buffered,
        }
    }

    /// True while this table's reads go through `O_DIRECT`.
    fn uses_o_direct(&self) -> bool {
        self.backend() == ReadBackend::Direct
    }

    /// Window this table's reads are expressed in.
    fn read_window(&self) -> u64 {
        self.read_window
    }
}

/// `O_DIRECT`-capable reader over packed rank files.
///
/// Every request is expressed as one or more aligned windows; the reader asserts that the
/// offset, the length and the scratch address are all multiples of `[MIN_ALIGN_BYTES]`
/// before it issues an `O_DIRECT` `pread`, so an unaligned request cannot reach the
/// kernel.
#[derive(Debug)]
pub struct DirectTable {
    tables: Vec<OpenTable>,
    /// First reason a table could not be read with `O_DIRECT`, recorded at open time or
    /// when the kernel refused a read later.
    fallback_reason: parking_lot::Mutex<Option<String>>,
    read_ops: std::sync::atomic::AtomicU64,
    /// Requested rows closer than this are merged into one read.
    merge_gap_bytes: u64,
    /// Upper bound on one merged read, before window rounding.
    max_io_bytes: u64,
}

impl DirectTable {
    /// Opens every spec, preferring `O_DIRECT` and falling back to `mmap`.
    ///
    /// # Errors
    /// When a file cannot be opened, is shorter than its spec requires, or is not
    /// 512-byte aligned in length, or when `O_DIRECT` was required and refused.
    pub fn open(specs: Vec<TableSpec>) -> Result<Self> {
        Self::open_with(specs, BackendPreference::Auto, &mut |_| {})
    }

    /// As `[DirectTable::open]`, with an explicit backend preference and a sink for the
    /// fallback warning.
    ///
    /// The `O_DIRECT` alignment is derived per file from the logical block size of the
    /// device behind it (see `[device_alignment]`), and each file is probed with one real
    /// aligned read before its direct handle is accepted.
    ///
    /// # Errors
    /// As `[DirectTable::open]`; `BackendPreference::Require` turns a filesystem that
    /// rejects `O_DIRECT` into an error.
    pub fn open_with(
        specs: Vec<TableSpec>,
        preference: BackendPreference,
        log: &mut dyn FnMut(&str),
    ) -> Result<Self> {
        Self::open_impl(specs, preference, None, log)
    }

    /// As `[DirectTable::open_with]`, with the `O_DIRECT` alignment forced instead of
    /// detected.
    ///
    /// This is for tests and for operators who know the logical block size of the device
    /// behind every file; `[device_alignment]` decides otherwise. Forcing 512 on a device
    /// that needs 4096 exercises the honest fallback: the probe fails with `EINVAL` and the
    /// table is served through the mapping instead of claiming `[ReadBackend::Direct]`.
    ///
    /// # Errors
    /// When `align_bytes` is not a multiple of `[MIN_ALIGN_BYTES]` or exceeds
    /// `[MAX_ALIGN_BYTES]`, plus everything `[DirectTable::open_with]` reports.
    pub fn open_with_alignment(
        specs: Vec<TableSpec>,
        preference: BackendPreference,
        align_bytes: u64,
        log: &mut dyn FnMut(&str),
    ) -> Result<Self> {
        ensure!(
            align_bytes >= MIN_ALIGN_BYTES
                && align_bytes.is_multiple_of(MIN_ALIGN_BYTES)
                && align_bytes <= MAX_ALIGN_BYTES,
            "alignment {align_bytes} must be a multiple of {MIN_ALIGN_BYTES} and at most \
             {MAX_ALIGN_BYTES}"
        );
        Self::open_impl(specs, preference, Some(align_bytes), log)
    }

    fn open_impl(
        specs: Vec<TableSpec>,
        preference: BackendPreference,
        align_override: Option<u64>,
        log: &mut dyn FnMut(&str),
    ) -> Result<Self> {
        ensure!(!specs.is_empty(), "at least one table spec is required");
        let mut tables = Vec::with_capacity(specs.len());
        let mut fallback_reason: Option<String> = None;
        for spec in specs {
            let buffered = File::open(&spec.path)
                .with_context(|| format!("open rank file {}", spec.path.display()))?;
            let file_bytes = buffered
                .metadata()
                .with_context(|| format!("stat {}", spec.path.display()))?
                .len();
            let needed = spec.physical_rows * ENGRAM_ROW_BYTES as u64;
            ensure!(
                file_bytes >= needed,
                "{} is {file_bytes} bytes but its spec needs {needed} bytes for {} rows; \
                 re-run the packer",
                spec.path.display(),
                spec.physical_rows
            );
            ensure!(
                file_bytes.is_multiple_of(MIN_ALIGN_BYTES),
                "{} length {file_bytes} is not a multiple of {MIN_ALIGN_BYTES}; re-run the packer",
                spec.path.display()
            );
            let align = align_override.unwrap_or_else(|| device_alignment(&spec.path));
            let read_window = effective_window(spec.window_bytes, align);
            let unusable = (align > MAX_ALIGN_BYTES).then(|| {
                format!(
                    "the device behind {} needs a {align}-byte O_DIRECT alignment, more than \
                     the {MAX_ALIGN_BYTES}-byte scratch alignment",
                    spec.path.display()
                )
            });
            let (direct, map) = match preference {
                BackendPreference::Mapped => (None, Some(map_file(&buffered, &spec.path)?)),
                BackendPreference::Require => (
                    Some(open_direct_aligned(&spec.path, align, file_bytes)?),
                    None,
                ),
                BackendPreference::Auto => match unusable {
                    Some(reason) => {
                        log(&reason);
                        if fallback_reason.is_none() {
                            fallback_reason = Some(reason);
                        }
                        (None, Some(map_file(&buffered, &spec.path)?))
                    }
                    None => match open_direct_aligned(&spec.path, align, file_bytes) {
                        Ok(file) => (Some(file), None),
                        Err(error) => {
                            let reason = format!(
                                "O_DIRECT unavailable for {} ({error}); falling back to \
                                 mmap + MADV_RANDOM",
                                spec.path.display()
                            );
                            log(&reason);
                            if fallback_reason.is_none() {
                                fallback_reason = Some(reason);
                            }
                            (None, Some(map_file(&buffered, &spec.path)?))
                        }
                    },
                },
            };
            tables.push(OpenTable {
                spec,
                direct,
                buffered,
                map,
                file_bytes,
                align,
                read_window,
                direct_failed: std::sync::atomic::AtomicBool::new(false),
            });
        }
        Ok(Self {
            tables,
            fallback_reason: parking_lot::Mutex::new(fallback_reason),
            read_ops: std::sync::atomic::AtomicU64::new(0),
            merge_gap_bytes: DEFAULT_MERGE_GAP_BYTES,
            max_io_bytes: DEFAULT_MAX_IO_BYTES,
        })
    }
}

impl DirectTable {
    /// Overrides the merge gap: requested rows whose byte offsets are within this many
    /// bytes of each other are fetched by one read.
    #[must_use]
    pub fn with_merge_gap_bytes(mut self, merge_gap_bytes: u64) -> Self {
        self.merge_gap_bytes = merge_gap_bytes.max(MIN_ALIGN_BYTES);
        self
    }

    /// Overrides the cap on one merged read, before window rounding.
    #[must_use]
    pub fn with_max_io_bytes(mut self, max_io_bytes: u64) -> Self {
        self.max_io_bytes = max_io_bytes.max(MIN_ALIGN_BYTES);
        self
    }

    /// Number of tables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// True when there are no tables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Backend a table is being served on right now.
    ///
    /// A table that latched the buffered path after a refused direct read reports
    /// `[ReadBackend::Buffered]`, so this never claims a backend that is not reading.
    #[must_use]
    pub fn backend_of(&self, table: usize) -> Option<ReadBackend> {
        self.tables.get(table).map(OpenTable::backend)
    }

    /// True when every table is on the `O_DIRECT` path.
    #[must_use]
    pub fn uses_o_direct(&self) -> bool {
        !self.tables.is_empty() && self.tables.iter().all(OpenTable::uses_o_direct)
    }

    /// `O_DIRECT` alignment in force for a table.
    #[must_use]
    pub fn alignment_of(&self, table: usize) -> Option<u64> {
        self.tables.get(table).map(|entry| entry.align)
    }

    /// Window a table's reads use; at least the spec's window and always a multiple of the
    /// table's alignment.
    #[must_use]
    pub fn window_of(&self, table: usize) -> Option<u64> {
        self.tables.get(table).map(OpenTable::read_window)
    }

    /// Why a non-`O_DIRECT` path was taken, when one was. The reason is recorded both for a
    /// fallback at open time and for a direct read the kernel refused later.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<String> {
        self.fallback_reason.lock().clone()
    }

    /// Records the first reason a table could not be read with `O_DIRECT`.
    fn note_fallback(&self, reason: String) {
        let mut slot = self.fallback_reason.lock();
        if slot.is_none() {
            *slot = Some(reason);
        }
    }

    /// Actual length of a table's file.
    #[must_use]
    pub fn file_bytes(&self, table: usize) -> Option<u64> {
        self.tables.get(table).map(|entry| entry.file_bytes)
    }

    fn entry(&self, table: usize) -> Result<&OpenTable> {
        self.tables
            .get(table)
            .ok_or_else(|| anyhow::anyhow!("table {table} is out of range"))
    }

    /// One read of `len` bytes at `offset` into `buf`.
    ///
    /// The request is issued with `O_DIRECT` only when the offset, the length and the buffer
    /// address are all multiples of the table's alignment. Everything else — in particular
    /// the final clamp that is shorter than one alignment unit, and every read after a
    /// refusal — goes through the mapping or the plain handle, which is slower and always
    /// correct.
    fn read_at(&self, table: usize, offset: u64, len: u64, buf: &mut [u8]) -> Result<()> {
        let entry = self.entry(table)?;
        ensure!(
            buf.len() as u64 == len,
            "scratch buffer is {} bytes but the request is {len}",
            buf.len()
        );
        ensure!(
            offset + len <= entry.file_bytes,
            "refusing to read {len} bytes at {offset} of {} which is {} bytes",
            entry.spec.path.display(),
            entry.file_bytes
        );
        let align = entry.align;
        let direct = entry
            .direct
            .as_ref()
            .filter(|_| {
                !entry
                    .direct_failed
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .filter(|_| offset.is_multiple_of(align) && len.is_multiple_of(align))
            .filter(|_| (buf.as_ptr() as usize).is_multiple_of(align as usize));
        match direct {
            Some(file) => match file.read_exact_at(buf, offset) {
                Ok(()) => {
                    self.read_ops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(error) if is_alignment_error(&error) => {
                    // The kernel will not serve this file unbuffered (a 4 KiB-logical-block
                    // device asked for a 512-byte read, or the filesystem dropped support):
                    // remember it, stop claiming `ReadBackend::Direct`, and read this request
                    // through the plain handle.
                    entry
                        .direct_failed
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    self.note_fallback(format!(
                        "the kernel refused an O_DIRECT read of {len} bytes at {offset} on {} \
                         ({error}); reads now use the buffered handle",
                        entry.spec.path.display()
                    ));
                    self.read_plain(table, offset, len, buf)?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "O_DIRECT read of {len} bytes at {offset} from {}",
                            entry.spec.path.display()
                        )
                    });
                }
            },
            None => self.read_plain(table, offset, len, buf)?,
        }
        Ok(())
    }

    /// Reads through the mapping when there is one, and through the plain handle otherwise.
    /// No alignment or length rule applies to this path, and it counts as one read.
    fn read_plain(&self, table: usize, offset: u64, len: u64, buf: &mut [u8]) -> Result<()> {
        let entry = self.entry(table)?;
        match &entry.map {
            Some(map) => {
                let (start, end) = (offset as usize, (offset + len) as usize);
                ensure!(
                    end <= map.len(),
                    "mapping of {} is {} bytes but the request ends at {end}",
                    entry.spec.path.display(),
                    map.len()
                );
                buf.copy_from_slice(&map[start..end]);
            }
            None => entry.buffered.read_exact_at(buf, offset).with_context(|| {
                format!(
                    "buffered read of {len} bytes at {offset} from {}",
                    entry.spec.path.display()
                )
            })?,
        }
        self.read_ops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Reads the window span that covers `[offset, offset + len)`, clamping the request at
    /// the end of the file and zero-filling the rest of `buf`.
    ///
    /// The clamp is deliberately not rounded: `[DirectTable::read_at]` only takes the direct
    /// path for a request it can serve legally, so a final span that is shorter than one
    /// alignment unit is read through the mapping or the plain handle instead of failing or
    /// reading past the end of the file.
    fn read_windows(
        &self,
        table: usize,
        offset: u64,
        len: u64,
        buf: &mut AlignedBuffer,
    ) -> Result<()> {
        let entry = self.entry(table)?;
        let window = entry.read_window();
        ensure!(
            offset.is_multiple_of(window) && (offset + len).is_multiple_of(window),
            "window span {offset}..{} is not aligned to {window}",
            offset + len
        );
        let available = entry.file_bytes.saturating_sub(offset).min(len);
        if available > 0 {
            // A direct table reads the `O_DIRECT`-aligned prefix of the span directly and
            // sends only the short tail (the clamp at the end of the file) through the
            // mapping or the plain handle, so the final unaligned window does not drag the
            // whole span off the fast path. A table that is not on the direct path reads the
            // span in one piece.
            let direct_len = if entry.uses_o_direct() {
                available - available % entry.align
            } else {
                0
            };
            if direct_len > 0 {
                self.read_at(
                    table,
                    offset,
                    direct_len,
                    &mut buf.as_mut_slice()[..direct_len as usize],
                )?;
            }
            if direct_len < available {
                let tail = &mut buf.as_mut_slice()[direct_len as usize..available as usize];
                self.read_plain(table, offset + direct_len, available - direct_len, tail)?;
            }
        }
        if available < len {
            buf.as_mut_slice()[available as usize..].fill(0);
        }
        Ok(())
    }
}

/// Opens `path` with `O_DIRECT` at the alignment its device requires.
///
/// A successful `open` proves nothing about direct I/O on its own: tmpfs accepts the flag
/// and ignores it, and a device with a larger logical block size accepts the open and then
/// rejects every misaligned request with `EINVAL`. This therefore only returns the handle
/// after one real aligned read at offset 0 has succeeded.
#[cfg(feature = "io-uring")]
pub(crate) fn open_direct(path: &Path) -> Result<File> {
    let file_bytes = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    open_direct_aligned(path, device_alignment(path), file_bytes)
}

/// As `[open_direct]`, with a known alignment and file length.
fn open_direct_aligned(path: &Path, align: u64, file_bytes: u64) -> Result<File> {
    use std::os::unix::ffi::OsStrExt;
    ensure!(
        (MIN_ALIGN_BYTES..=MAX_ALIGN_BYTES).contains(&align),
        "{} needs a {align}-byte O_DIRECT alignment; this reader serves \
         {MIN_ALIGN_BYTES}..={MAX_ALIGN_BYTES}",
        path.display()
    );
    let probe_len = align.min(file_bytes - file_bytes % align);
    ensure!(
        probe_len >= align,
        "{} is {file_bytes} bytes, shorter than one {align}-byte direct-I/O unit",
        path.display()
    );
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("path {} contains a NUL byte", path.display()))?;
    // SAFETY: c_path is a valid NUL-terminated string that outlives the call.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        bail!("open O_DIRECT {}: {error}", path.display());
    }
    // SAFETY: fd is a fresh descriptor owned by this process, handed to File exactly once.
    let file = unsafe { File::from_raw_fd(fd) };
    let mut probe = AlignedBuffer::new(probe_len as usize);
    file.read_exact_at(probe.as_mut_slice(), 0)
        .map_err(|error| {
            anyhow::anyhow!(
                "O_DIRECT probe read of {probe_len} bytes at 0 on {} failed ({error})",
                path.display()
            )
        })?;
    Ok(file)
}

/// Maps a rank file read-only and advises random access.
fn map_file(file: &File, path: &Path) -> Result<Mmap> {
    // SAFETY: the mapping is read-only, and it keeps its own reference to the file, so it
    // stays valid after the descriptor is dropped.
    let map =
        unsafe { Mmap::map(file) }.with_context(|| format!("mmap rank file {}", path.display()))?;
    map.advise(Advice::Random)
        .with_context(|| format!("advise MADV_RANDOM on {}", path.display()))?;
    Ok(map)
}

impl DirectTable {
    /// Reads one row into a caller buffer, synthesising tail rows without any I/O.
    fn read_row_into(
        &self,
        table: usize,
        row: u64,
        out: &mut [u8; ENGRAM_ROW_BYTES],
    ) -> Result<()> {
        let entry = self.entry(table)?;
        ensure!(
            row < entry.spec.physical_rows,
            "row {row} is past the {} rows of table {table}",
            entry.spec.physical_rows
        );
        if entry.spec.is_tail(row) {
            *out = entry.spec.tail_fill.row();
            return Ok(());
        }
        let window = entry.read_window();
        let offset = row * ENGRAM_ROW_BYTES as u64;
        let (start, end) = window_span(offset, ENGRAM_ROW_BYTES as u64, window);
        let mut scratch = AlignedBuffer::new((end - start) as usize);
        self.read_windows(table, start, end - start, &mut scratch)?;
        let begin = (offset - start) as usize;
        out.copy_from_slice(&scratch.as_slice()[begin..begin + ENGRAM_ROW_BYTES]);
        Ok(())
    }

    /// Batched read: groups rows whose byte offsets are close into runs and issues exactly
    /// one aligned read per run.
    fn read_rows_batched(
        &self,
        table: usize,
        rows: &[u32],
        out: &mut [[u8; ENGRAM_ROW_BYTES]],
    ) -> Result<()> {
        ensure!(
            rows.len() == out.len(),
            "{} rows requested but {} output slots given",
            rows.len(),
            out.len()
        );
        let entry = self.entry(table)?;
        let row_bytes = ENGRAM_ROW_BYTES as u64;
        let window = entry.read_window();

        let mut order: Vec<usize> = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let row = *row as u64;
            ensure!(
                row < entry.spec.physical_rows,
                "row {row} is past the {} rows of table {table}",
                entry.spec.physical_rows
            );
            if entry.spec.is_tail(row) {
                out[index] = entry.spec.tail_fill.row();
            } else {
                order.push(index);
            }
        }
        order.sort_unstable_by_key(|index| rows[*index]);

        let mut cursor = 0;
        while cursor < order.len() {
            let first = rows[order[cursor]] as u64;
            let mut last = first;
            let mut end = cursor + 1;
            while end < order.len() {
                let next = rows[order[end]] as u64;
                let span = (next + 1) * row_bytes - first * row_bytes;
                let gap = next.saturating_sub(last + 1).saturating_mul(row_bytes);
                if span > self.max_io_bytes || gap > self.merge_gap_bytes {
                    break;
                }
                last = next;
                end += 1;
            }
            let read_start = first * row_bytes / window * window;
            let read_end = ((last + 1) * row_bytes).div_ceil(window) * window;
            let mut scratch = AlignedBuffer::new((read_end - read_start) as usize);
            self.read_windows(table, read_start, read_end - read_start, &mut scratch)?;
            let base = scratch.as_slice();
            for &slot in &order[cursor..end] {
                let offset = rows[slot] as u64 * row_bytes;
                let begin = (offset - read_start) as usize;
                out[slot].copy_from_slice(&base[begin..begin + ENGRAM_ROW_BYTES]);
            }
            cursor = end;
        }
        Ok(())
    }
}

impl RowTable for DirectTable {
    fn table_count(&self) -> usize {
        self.tables.len()
    }

    fn spec(&self, table: usize) -> Option<&TableSpec> {
        self.tables.get(table).map(|entry| &entry.spec)
    }

    fn read_row(&self, table: usize, row: u64) -> Result<[u8; ENGRAM_ROW_BYTES]> {
        let mut out = [0u8; ENGRAM_ROW_BYTES];
        self.read_row_into(table, row, &mut out)?;
        Ok(out)
    }

    fn read_rows_into(
        &self,
        table: usize,
        rows: &[u32],
        out: &mut [[u8; ENGRAM_ROW_BYTES]],
    ) -> Result<()> {
        self.read_rows_batched(table, rows, out)
    }

    fn read_ops(&self) -> u64 {
        self.read_ops.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Builds the table specs for one rank from a manifest produced by `[crate::packer]`.
///
/// The manifest is treated as the source of truth for the row ranges and cross-checked
/// against the compiled-in geometry, so a stale manifest is rejected instead of silently
/// serving wrong rows.
///
/// # Errors
/// When the manifest targets a different world size, is missing a layer/rank, or disagrees
/// with the compiled-in row layout.
pub fn specs_from_manifest(
    manifest: &crate::packer::PackManifest,
    output_dir: &Path,
    rank: usize,
    world_size: usize,
) -> Result<Vec<TableSpec>> {
    ensure!(
        manifest.world_size == world_size,
        "manifest was packed for world size {} but {world_size} was requested",
        manifest.world_size
    );
    ensure!(
        manifest.layout.row_bytes == ENGRAM_ROW_BYTES as u64,
        "manifest row stride {} is not the dense {ENGRAM_ROW_BYTES}-byte layout",
        manifest.layout.row_bytes
    );
    let mut specs = Vec::with_capacity(manifest.sources.len());
    for source in &manifest.sources {
        let entry = manifest
            .shards
            .iter()
            .find(|shard| shard.layer == source.layer && shard.rank == rank)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "manifest has no shard for layer {} rank {rank}",
                    source.layer
                )
            })?;
        let mut spec =
            TableSpec::new(source.layer, rank, world_size, output_dir.join(&entry.file))?;
        ensure!(
            spec.logical_rows == source.weight.shape[0],
            "layer {}: manifest tensor has {} rows but the compiled layout says {}",
            source.layer,
            source.weight.shape[0],
            spec.logical_rows
        );
        ensure!(
            spec.physical_rows == entry.rows && spec.row_start == entry.row_start,
            "layer {} rank {rank}: manifest row range {}..{} disagrees with the compiled layout",
            source.layer,
            entry.row_start,
            entry.row_end
        );
        ensure!(
            entry.row_stride == ENGRAM_ROW_BYTES as u64,
            "layer {} rank {rank}: manifest row stride {} is not {ENGRAM_ROW_BYTES}",
            source.layer,
            entry.row_stride
        );
        spec = spec.with_window_bytes(DEFAULT_WINDOW_BYTES)?;
        specs.push(spec);
    }
    specs.sort_by_key(|spec| spec.layer_index);
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_spec(
        layer_id: usize,
        layer_index: usize,
        row_start: u64,
        physical_rows: u64,
        logical_rows: u64,
    ) -> TableSpec {
        TableSpec {
            layer_id,
            layer_index,
            rank: 0,
            world_size: 1,
            path: PathBuf::from("/tmp/does-not-need-to-exist.bin"),
            row_start,
            physical_rows,
            logical_rows,
            window_bytes: MIN_ALIGN_BYTES,
            tail_fill: TailFill::Reference,
        }
    }

    #[test]
    fn window_span_covers_a_straddling_row() {
        // Row 0 sits in the first window.
        assert_eq!(window_span(0, 264, 512), (0, 512));
        // Row 1 starts at 264 and ends at 528: it straddles windows 0 and 1.
        assert_eq!(window_span(264, 264, 512), (0, 1024));
        // Row 2 fits exactly inside the second window.
        assert_eq!(window_span(528, 264, 512), (512, 1024));
        // Row 3 straddles windows 1 and 2.
        assert_eq!(window_span(792, 264, 512), (512, 1536));
        // A 4 KiB window straddles far less often but still must be handled.
        assert_eq!(window_span(3800, 264, 4096), (0, 4096));
        assert_eq!(window_span(3840, 264, 4096), (0, 8192));
    }

    #[test]
    fn window_size_must_be_an_aligned_multiple() {
        let spec = small_spec(1, 0, 0, 16, 16);
        assert!(spec.clone().with_window_bytes(512).is_ok());
        assert!(spec.clone().with_window_bytes(4096).is_ok());
        assert!(spec.clone().with_window_bytes(1024).is_ok());
        assert!(spec.with_window_bytes(100).is_err());
        assert!(small_spec(1, 0, 0, 16, 16).with_window_bytes(0).is_err());
    }

    #[test]
    fn tail_fill_matches_the_reference_padding() {
        assert_eq!(TailFill::default(), TailFill::Reference);
        let reference = TailFill::Reference.row();
        assert!(reference[..ENGRAM_HEAD_DIM].iter().all(|byte| *byte == 0));
        assert!(
            reference[ENGRAM_HEAD_DIM..]
                .iter()
                .all(|byte| *byte == PAD_SCALE_BYTE)
        );
        assert!(TailFill::Zeros.row().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn table_spec_derives_the_reference_row_split() {
        // Layer 14 is the layer whose split does not divide evenly: 384,016,682 rows over
        // 4 ranks is 96,004,171 rows per rank, so the last rank pads two rows.
        let spec = TableSpec::new(14, 3, 4, "/tmp/l14r3.bin").unwrap();
        assert_eq!(spec.layer_index, 1);
        assert_eq!(spec.logical_rows, 384_016_682);
        assert_eq!(spec.physical_rows, 96_004_171);
        assert_eq!(spec.row_start, 288_012_513);
        assert_eq!(spec.source_rows(), 96_004_169);
        assert!(!spec.is_tail(96_004_168));
        assert!(spec.is_tail(96_004_169));
        assert!(spec.is_tail(96_004_170));
        assert_eq!(spec.local_row(288_012_513), Some(0));
        assert_eq!(spec.local_row(288_012_512), None);
        assert_eq!(spec.global_row(96_004_170), 384_016_683);
        assert_eq!(spec.file_bytes(), 25_345_101_312);
        assert_eq!(spec.file_bytes() % MIN_ALIGN_BYTES, 0);

        // Layer 1 divides exactly.
        let even = TableSpec::new(1, 3, 4, "/tmp/l1r3.bin").unwrap();
        assert_eq!(even.logical_rows, 384_006_168);
        assert_eq!(even.physical_rows, 96_001_542);
        assert_eq!(even.source_rows(), 96_001_542);
        assert!(!even.is_tail(96_001_541));
        assert_eq!(even.file_bytes(), 25_344_407_552);
    }

    #[test]
    fn memory_table_serves_patterns_and_tail_rows() {
        let spec = small_spec(1, 0, 100, 8, 106);
        let mut table = MemoryTable::new().with_layer(spec);
        table.fill_pattern(0).unwrap();
        let row = table.read_row(0, 3).unwrap();
        assert_eq!(u64::from_le_bytes(row[..8].try_into().unwrap()), 103);
        // Rows 6 and 7 are past the logical end and come from the tail fill.
        let tail = table.read_row(0, 6).unwrap();
        assert_eq!(tail, TailFill::Reference.row());
        assert!(table.read_row(0, 8).is_err());
        assert_eq!(table.read_ops(), 2);
        assert_eq!(table.stored_rows(0), 8);
    }

    #[test]
    fn read_global_finds_the_owning_table() {
        let mut table = MemoryTable::new()
            .with_layer(small_spec(1, 0, 0, 4, 8))
            .with_layer(small_spec(14, 1, 4, 4, 8));
        table.fill_pattern(0).unwrap();
        table.fill_pattern(1).unwrap();
        // The pattern's first eight bytes are the global row index.
        let first =
            |bytes: [u8; ENGRAM_ROW_BYTES]| u64::from_le_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(first(table.read_global(1, 3).unwrap().unwrap()), 3);
        // Global row 2 of layer 14 belongs to no table this reader holds.
        assert!(table.read_global(14, 2).unwrap().is_none());
        // Table 1 starts at global row 4, so global row 5 is its local row 1.
        assert_eq!(first(table.read_global(14, 5).unwrap().unwrap()), 5);
        assert!(table.read_global(7, 0).unwrap().is_none());
    }

    #[test]
    fn alignment_derivation_covers_both_device_shapes() {
        // 512-byte logical blocks (the deployment target) keep a 512-byte alignment, so a
        // 512-byte window stays legal and cheap.
        assert_eq!(alignment_from_block_sizes(Some(512), Some(4096)), 512);
        // 4096-byte logical blocks (4Kn drives, dm-crypt volumes such as this box's
        // /var/tmp) require 4096: a 512-byte request there is refused with EINVAL.
        assert_eq!(alignment_from_block_sizes(Some(4096), Some(4096)), 4096);
        assert_eq!(alignment_from_block_sizes(Some(2048), None), 2048);
        // A device that reports more than the scratch alignment is passed through, so the
        // caller can fall back to the mapping instead of issuing reads that cannot work.
        assert_eq!(alignment_from_block_sizes(Some(8192), None), 8192);
        // Filesystems that report a preferred I/O size rather than a logical block size
        // (ZFS reports 128 KiB here) are clamped to what a scratch buffer can serve.
        assert_eq!(
            alignment_from_block_sizes(None, Some(131_072)),
            MAX_ALIGN_BYTES
        );
        assert_eq!(alignment_from_block_sizes(None, Some(512)), 512);
        assert_eq!(alignment_from_block_sizes(None, None), MIN_ALIGN_BYTES);
        assert_eq!(alignment_from_block_sizes(Some(0), None), MIN_ALIGN_BYTES);
        // Odd inputs round up to the 512-byte unit the format uses.
        assert_eq!(alignment_from_block_sizes(Some(600), None), 1024);

        // A window is never lowered, and is always a multiple of the alignment.
        assert_eq!(effective_window(512, 512), 512);
        assert_eq!(effective_window(512, 4096), 4096);
        assert_eq!(effective_window(4096, 512), 4096);
        assert_eq!(effective_window(1536, 1024), 2048);
        assert_eq!(effective_window(DEFAULT_WINDOW_BYTES, 512), 4096);
        assert_eq!(effective_window(DEFAULT_WINDOW_BYTES, 4096), 4096);

        // A real path resolves to something this reader can serve.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alignment.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let detected = device_alignment(&path);
        assert!(
            (MIN_ALIGN_BYTES..=MAX_ALIGN_BYTES).contains(&detected)
                && detected.is_multiple_of(MIN_ALIGN_BYTES),
            "detected {detected}"
        );
    }

    #[test]
    fn only_alignment_errors_latch_the_buffered_path() {
        use std::io::{Error, ErrorKind};
        // EINVAL/ENOTSUP are what a 4 KiB-logical-block device and a filesystem without
        // direct I/O return; those mean "use the buffered path from now on".
        assert!(is_alignment_error(&Error::from_raw_os_error(libc::EINVAL)));
        assert!(is_alignment_error(&Error::from_raw_os_error(libc::ENOTSUP)));
        // A real I/O error must surface instead of being papered over.
        assert!(!is_alignment_error(&Error::from_raw_os_error(libc::EIO)));
        assert!(!is_alignment_error(&Error::new(
            ErrorKind::UnexpectedEof,
            "short read"
        )));
    }

    #[test]
    fn an_impossible_alignment_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let spec = TableSpec {
            path: path.clone(),
            ..small_spec(1, 0, 0, 8, 8)
        };
        let open = |align: u64| {
            DirectTable::open_with_alignment(
                vec![spec.clone()],
                BackendPreference::Mapped,
                align,
                &mut |_| {},
            )
        };
        assert!(open(512).is_ok());
        assert!(open(4096).is_ok());
        assert!(open(256).is_err(), "below the 512-byte floor");
        assert!(open(1536).is_ok(), "1536 is a legal multiple of 512");
        assert!(open(8192).is_err(), "above the scratch alignment");
        assert!(open(0).is_err());
    }
}
