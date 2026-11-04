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

//! The per-rank converted container `sparkflash convert` writes: one file a tensor-parallel rank,
//! `rank-{rank}-of-{world}.bin`, holding that rank's slices already in the order its device
//! memory takes them, so a boot reads the payload in large parallel direct-I/O chunks instead of
//! seeking, slicing and repacking tensor by tensor.
//!
//! **Attribution:** pre-converting per-rank layouts and loading them with parallel `O_DIRECT`
//! reads draws on prior art in [`b12x`](https://github.com/local-inference-lab/b12x)
//! (`b12x/loader/_direct.c`) and [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference)
//! (Apache-2.0; Luke Alonso `@lukealonso`, `@original-el8`, and contributors). See the root
//! `NOTICE` file.
//!
//! A file is a 64-byte [ConvertedPreamble], a JSON index from byte 64, zeros up to `data_offset`
//! (a multiple of [DATA_ALIGN_BYTES]), and the data, zero-padded to whole pages. The preamble
//! carries the index's FNV-1a 64 hash, so a boot checks the file's size and its index in a few
//! kilobytes without touching a weight; each payload's own digest, and the digest of those digests
//! in file order, are there for a full verify.
//!
//! The index is the model pack's: the V4.1 arm records resident groups and tiled layouts. The
//! Qwen and GLM arms write in walk order and share one index, a [SlotIndex]: every upload of the
//! rank's weight walk is a [Slot], planned by a first pass ([SlotPlan]), streamed by a second
//! ([SlotFileWriter]) and found again by the load's own walk ([SlotCursor]). This module owns what
//! every rank file shares - the preamble, the index hash, the payload digests, the page alignment,
//! the streaming writer and the payload verify - and the slot index the walk-ordered packs share.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::tp::TpLayout;
use crate::weights::check_length;

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

/// The FNV-1a 64 offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 over a byte slice.
#[must_use]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// `digest` folded into a running digest of digests, the way a file's and a group's are built.
#[must_use]
pub fn fold_digest(running: u64, digest: u64) -> u64 {
    (running ^ digest).wrapping_mul(FNV_PRIME)
}

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

/// `bytes` rounded up to whole [DATA_ALIGN_BYTES] pages.
#[must_use]
pub fn align_up_page(bytes: usize) -> usize {
    bytes.next_multiple_of(DATA_ALIGN_BYTES)
}

/// The `data_offset` for an index that serialises to `index_len` bytes with placeholder digests.
///
/// A kilobyte of slack past the preamble and index is reserved before rounding to a page, so the
/// final index, with real digests in place of the placeholders, always fits ahead of the data.
#[must_use]
pub fn data_offset_for(index_len: usize) -> u64 {
    align_up_page(PREAMBLE_BYTES + index_len + 1024) as u64
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
    /// Byte offset in the file where the data begins (multiple of 4096).
    pub data_offset: u64,
    /// Logical data bytes (`allocated_bytes` for the V4.1 arena).
    pub data_bytes: u64,
    /// Data bytes padded to whole 4096-byte pages, as stored on disk.
    pub padded_data_bytes: u64,
    /// FNV-1a 64 hash of the JSON index bytes.
    pub header_hash: u64,
    /// The index's digest of payload digests (not encoded in the preamble; a reader fills it from
    /// its index).
    pub file_digest: u64,
}

impl ConvertedPreamble {
    /// The 64 bytes at the start of the file.
    #[must_use]
    pub fn encode(&self) -> [u8; PREAMBLE_BYTES] {
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

    /// Decodes a preamble, refusing another magic or version. `file_digest` is left zero.
    ///
    /// # Errors
    /// When the magic or the version is not this format's.
    pub fn decode(bytes: &[u8; PREAMBLE_BYTES], path: &Path) -> Result<Self, String> {
        if &bytes[0..8] != CONVERTED_MAGIC {
            return Err(format!(
                "{}: invalid magic {:?}, expected {:?}",
                path.display(),
                &bytes[0..8],
                CONVERTED_MAGIC
            ));
        }
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let long = |at: usize| {
            let mut eight = [0u8; 8];
            eight.copy_from_slice(&bytes[at..at + 8]);
            u64::from_le_bytes(eight)
        };
        let version = word(8);
        if version != CONVERTED_VERSION {
            return Err(format!(
                "{}: unsupported converted version {version}, expected {CONVERTED_VERSION}",
                path.display()
            ));
        }
        Ok(Self {
            version,
            rank: word(12) as usize,
            world: word(16) as usize,
            index_len: long(24),
            data_offset: long(32),
            data_bytes: long(40),
            padded_data_bytes: long(48),
            header_hash: long(56),
            file_digest: 0,
        })
    }
}

/// Streams one rank file: the payloads first, from `data_offset`, then the preamble and the index
/// ahead of them once every digest is known.
pub struct RankFileWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    data_offset: u64,
    written: usize,
    file_digest: u64,
}

impl RankFileWriter {
    /// Creates `path`, leaving `data_offset` bytes ([data_offset_for]) for the preamble and index.
    ///
    /// # Errors
    /// When the file cannot be created or the seek fails.
    pub fn create(path: &Path, data_offset: u64) -> Result<Self, String> {
        let file =
            File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);
        writer
            .seek(SeekFrom::Start(data_offset))
            .map_err(|error| format!("seek {}: {error}", path.display()))?;
        Ok(Self {
            writer,
            path: path.to_path_buf(),
            data_offset,
            written: 0,
            file_digest: FNV_OFFSET,
        })
    }

    /// Writes `payload` at data offset `at`, zero-filling from the end of the previous payload,
    /// and folds its digest, which it returns, into the file's.
    ///
    /// # Errors
    /// When `at` is behind what is already written, or the write fails.
    pub fn put(&mut self, at: usize, payload: &[u8]) -> Result<u64, String> {
        if at < self.written {
            return Err(format!(
                "{}: a payload at data offset {at} is behind the {} bytes already written",
                self.path.display(),
                self.written
            ));
        }
        self.zeros(at - self.written)?;
        self.writer
            .write_all(payload)
            .map_err(|error| format!("write {}: {error}", self.path.display()))?;
        self.written = at + payload.len();
        let digest = fnv1a(payload);
        self.file_digest = fold_digest(self.file_digest, digest);
        Ok(digest)
    }

    /// The digest of every payload's digest so far, in the order they were put.
    #[must_use]
    pub fn file_digest(&self) -> u64 {
        self.file_digest
    }

    fn zeros(&mut self, mut remaining: usize) -> Result<(), String> {
        let zeros = [0u8; DATA_ALIGN_BYTES];
        while remaining > 0 {
            let step = remaining.min(zeros.len());
            self.writer
                .write_all(&zeros[..step])
                .map_err(|error| format!("pad {}: {error}", self.path.display()))?;
            remaining -= step;
        }
        Ok(())
    }

    /// Pads the data to `padded_data_bytes`, writes the preamble and `index` ahead of it, and
    /// flushes, returning the index's hash.
    ///
    /// # Errors
    /// When the data overran `padded_data_bytes`, the index outgrew `data_offset`, or a write fails.
    pub fn finish(
        mut self,
        rank: usize,
        world: usize,
        index: &[u8],
        data_bytes: u64,
        padded_data_bytes: u64,
    ) -> Result<u64, String> {
        let padded = usize::try_from(padded_data_bytes)
            .map_err(|_| format!("{}: padded data does not fit usize", self.path.display()))?;
        if self.written > padded {
            return Err(format!(
                "{}: {} data bytes written past the {padded} padded",
                self.path.display(),
                self.written
            ));
        }
        self.zeros(padded - self.written)?;
        let data_offset = self.data_offset;
        if PREAMBLE_BYTES + index.len() > data_offset as usize {
            return Err(format!(
                "{}: header grew to {} bytes, exceeding reserved data_offset {data_offset}",
                self.path.display(),
                PREAMBLE_BYTES + index.len()
            ));
        }
        let header_hash = fnv1a(index);
        let preamble = ConvertedPreamble {
            version: CONVERTED_VERSION,
            rank,
            world,
            index_len: index.len() as u64,
            data_offset,
            data_bytes,
            padded_data_bytes,
            header_hash,
            file_digest: self.file_digest,
        };
        self.writer
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("seek start {}: {error}", self.path.display()))?;
        self.writer
            .write_all(&preamble.encode())
            .map_err(|error| format!("write preamble {}: {error}", self.path.display()))?;
        self.writer
            .write_all(index)
            .map_err(|error| format!("write header {}: {error}", self.path.display()))?;
        self.zeros(data_offset as usize - (PREAMBLE_BYTES + index.len()))?;
        self.writer
            .flush()
            .map_err(|error| format!("flush {}: {error}", self.path.display()))?;
        Ok(header_hash)
    }
}

/// Reads a rank file's preamble and index, checking the file's size against the preamble, the
/// page alignment, and the index's hash. Touching no payload, this finishes in under a
/// millisecond.
///
/// # Errors
/// When the file is short, misaligned, another format or version, or its index does not hash to
/// the preamble's or does not parse as `H`.
pub fn read_index<H: DeserializeOwned>(path: &Path) -> Result<(ConvertedPreamble, H), String> {
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
    let preamble = ConvertedPreamble::decode(&preamble_buf, path)?;
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
    let header: H = serde_json::from_slice(&index_bytes)
        .map_err(|error| format!("parse index in {}: {error}", path.display()))?;
    Ok((preamble, header))
}

/// One payload as an index records it, for [verify_payloads].
#[derive(Debug, Clone, Copy)]
pub struct PayloadDigest<'a> {
    /// What the index calls it.
    pub name: &'a str,
    /// Its byte offset from `data_offset`.
    pub offset: usize,
    /// Its length in bytes.
    pub bytes: usize,
    /// Its FNV-1a 64 digest.
    pub digest: u64,
}

/// Re-reads every payload of a rank file and checks each digest, and the digest of digests in
/// the order given against `file_digest`.
///
/// # Errors
/// When a read fails or any digest differs.
pub fn verify_payloads<'a>(
    path: &Path,
    data_offset: u64,
    payloads: impl IntoIterator<Item = PayloadDigest<'a>>,
    file_digest: u64,
) -> Result<(), String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut buf = Vec::new();
    let mut rolling_digest = FNV_OFFSET;
    for entry in payloads {
        if buf.len() < entry.bytes {
            buf.resize(entry.bytes, 0);
        }
        let at = data_offset + entry.offset as u64;
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
        rolling_digest = fold_digest(rolling_digest, digest);
    }
    if rolling_digest != file_digest {
        return Err(format!(
            "{}: file digest mismatch (measured {rolling_digest:#018x}, header {file_digest:#018x})",
            path.display()
        ));
    }
    Ok(())
}

/// Where each [Slot] starts: the alignment of a device allocation, which the kernels' vector
/// loads assume of every weight.
pub const SLOT_ALIGN_BYTES: usize = 256;

/// One upload of a rank's weight walk, as a [SlotIndex] records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    /// What the walk uploads it as: the checkpoint names it is built from (joined by `+`, a scale
    /// grid's ending `_scale_inv`), or a derived table's name.
    pub key: String,
    /// Its byte offset from the file's `data_offset`, a multiple of [SLOT_ALIGN_BYTES].
    pub offset: usize,
    /// Its length in bytes.
    pub bytes: usize,
    /// The FNV-1a 64 digest of its bytes.
    pub digest: u64,
}

/// What a walk-ordered pack records in its [SlotIndex] beside the slots, and the model name its
/// files carry.
pub trait SlotRecords: Serialize + DeserializeOwned {
    /// What the pack's rank files name as their model.
    const MODEL: &'static str;
}

/// The index of a rank file written in walk order: every upload of the rank's weight walk as a
/// [Slot], in the order the walk makes them, so a load that runs the same walk finds each weight
/// at the next slot. `X` is what the pack records beside the slots; its fields sit in the JSON
/// object between `source` and `slots`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotIndex<X> {
    /// [SlotRecords::MODEL].
    pub model: String,
    /// The rank, `0..world`.
    pub rank: usize,
    /// The tensor-parallel world the walk was sharded for.
    pub world: usize,
    /// What the converter read, as it was given.
    pub source: String,
    /// The pack's own records.
    #[serde(flatten)]
    pub extra: X,
    /// Every upload, in walk order.
    pub slots: Vec<Slot>,
    /// The end of the last slot.
    pub data_bytes: u64,
    /// The digest of the slots' digests, in order.
    pub file_digest: u64,
}

impl<X: SlotRecords> SlotIndex<X> {
    /// Reads a rank file's preamble and index ([read_index]), once it is a [SlotRecords::MODEL]
    /// file of rank `layout.rank` in a world of `layout.world_size` whose slots lie where
    /// [SlotPlan] places them. Touching no weight, this is the boot check.
    ///
    /// # Errors
    /// What [read_index] refuses, another model's or rank's file, and a slot [SlotPlan] would not
    /// have placed there or data that does not end with the last slot.
    pub fn read(path: &Path, layout: TpLayout) -> Result<(ConvertedPreamble, Self), String> {
        let (mut preamble, index): (_, Self) = read_index(path)?;
        let shown = path.display();
        if index.model != X::MODEL {
            return Err(format!(
                "{shown} is a {} rank file, not {}",
                index.model,
                X::MODEL
            ));
        }
        for (what, by_preamble, by_index, wanted) in [
            ("rank", preamble.rank, index.rank, layout.rank),
            ("world", preamble.world, index.world, layout.world_size),
        ] {
            if by_preamble != wanted || by_index != wanted {
                return Err(format!(
                    "{shown} is {what} {by_preamble} by its preamble and {by_index} by its index, \
                     where this load is {what} {wanted}"
                ));
            }
        }
        if preamble.data_bytes != index.data_bytes || index.data_bytes > preamble.padded_data_bytes
        {
            return Err(format!(
                "{shown} holds {} data bytes by its preamble, {} by its index, in {} padded",
                preamble.data_bytes, index.data_bytes, preamble.padded_data_bytes
            ));
        }
        // Each slot sits where SlotPlan puts it: in walk order from 0, at the aligned end of the
        // one before. No two walk entries can then share bytes, which their digests alone would
        // not catch.
        let mut end = 0usize;
        for slot in &index.slots {
            let placed = end.next_multiple_of(SLOT_ALIGN_BYTES);
            let next = slot.offset.checked_add(slot.bytes);
            let Some(next) =
                next.filter(|next| slot.offset == placed && *next as u64 <= index.data_bytes)
            else {
                return Err(format!(
                    "{shown}: slot {} at {} ({} bytes) is not where the walk places it ({placed}) \
                     inside the {} data bytes",
                    slot.key, slot.offset, slot.bytes, index.data_bytes
                ));
            };
            end = next;
        }
        if end as u64 != index.data_bytes {
            return Err(format!(
                "{shown}: the slots end at byte {end}, but the index holds {} data bytes",
                index.data_bytes
            ));
        }
        preamble.file_digest = index.file_digest;
        Ok((preamble, index))
    }

    /// Re-reads every slot of a rank file and checks its digest, and the digest of the digests.
    ///
    /// # Errors
    /// What [Self::read] refuses, and a slot or file digest that differs.
    pub fn verify(path: &Path, layout: TpLayout) -> Result<Self, String> {
        let (preamble, index) = Self::read(path, layout)?;
        verify_payloads(
            path,
            preamble.data_offset,
            index.slots.iter().map(|slot| PayloadDigest {
                name: &slot.key,
                offset: slot.offset,
                bytes: slot.bytes,
                digest: slot.digest,
            }),
            index.file_digest,
        )?;
        Ok(index)
    }
}

/// Slot `next` of `slots`, once it is `key` at `len` bytes, the upload the walk makes next.
fn expect_slot<'s>(
    slots: &'s [Slot],
    next: usize,
    key: &str,
    len: usize,
) -> Result<&'s Slot, String> {
    let Some(slot) = slots.get(next) else {
        return Err(format!(
            "the rank file ends before {key}, which this configuration loads"
        ));
    };
    if slot.key != key || slot.bytes != len {
        return Err(format!(
            "the rank file's slot {next} is {} ({} bytes), where this configuration loads {key} \
             ({len} bytes)",
            slot.key, slot.bytes
        ));
    }
    Ok(slot)
}

/// A rank's first pass over its walk: each upload placed at the next aligned offset after the
/// last, reading no weight, so the index's size, and with it where the data starts, is known
/// before the second pass ([SlotFileWriter]) streams the bytes.
#[derive(Debug, Default)]
pub struct SlotPlan {
    slots: Vec<Slot>,
    end: usize,
}

impl SlotPlan {
    /// Places the walk's next upload, `key` at `len` bytes.
    pub fn place(&mut self, key: &str, len: usize) {
        let offset = self.end.next_multiple_of(SLOT_ALIGN_BYTES);
        self.slots.push(Slot {
            key: key.to_owned(),
            offset,
            bytes: len,
            // The widest digest, so the placeholder index is never shorter than the final one.
            digest: u64::MAX,
        });
        self.end = offset + len;
    }

    /// The slots placed so far, in walk order.
    #[must_use]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// The first pass's index of rank `layout`: these slots and the pack's `records`, with
    /// `source` as what was read and placeholder digests as wide as any real one.
    #[must_use]
    pub fn into_index<X: SlotRecords>(
        self,
        layout: TpLayout,
        source: &str,
        records: X,
    ) -> SlotIndex<X> {
        SlotIndex {
            model: X::MODEL.to_owned(),
            rank: layout.rank,
            world: layout.world_size,
            source: source.to_owned(),
            extra: records,
            slots: self.slots,
            data_bytes: self.end as u64,
            file_digest: u64::MAX,
        }
    }
}

/// One rank file a [SlotFileWriter] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenRank {
    /// The file written.
    pub path: PathBuf,
    /// Its size: `data_offset` and the page-padded data.
    pub file_bytes: u64,
    /// Where the data starts.
    pub data_offset: u64,
    /// The end of the last slot.
    pub data_bytes: u64,
    /// The uploads it holds.
    pub slots: usize,
    /// The FNV-1a 64 hash of the index.
    pub header_hash: u64,
    /// The digest of the slots' digests.
    pub file_digest: u64,
}

/// A rank's second pass over its walk: each upload's bytes streamed to the slot the first pass
/// placed it at, then the index, with every digest, and the preamble ahead of them.
pub struct SlotFileWriter<X> {
    path: PathBuf,
    data_offset: u64,
    writer: RankFileWriter,
    index: SlotIndex<X>,
    next: usize,
}

impl<X: SlotRecords> SlotFileWriter<X> {
    /// Creates rank `index.rank`'s file in `output_dir`, as [rank_file_name] names it, with its
    /// data where the first pass's `index` ([SlotPlan::into_index]) puts it.
    ///
    /// # Errors
    /// When the index does not serialise, or the directory or file cannot be created.
    pub fn create(output_dir: &Path, index: SlotIndex<X>) -> Result<Self, String> {
        let placeholder =
            serde_json::to_vec(&index).map_err(|error| format!("serialise the index: {error}"))?;
        let data_offset = data_offset_for(placeholder.len());
        fs::create_dir_all(output_dir)
            .map_err(|error| format!("create {}: {error}", output_dir.display()))?;
        let path = output_dir.join(rank_file_name(index.rank, index.world));
        let writer = RankFileWriter::create(&path, data_offset)?;
        Ok(Self {
            path,
            data_offset,
            writer,
            index,
            next: 0,
        })
    }

    /// Streams the walk's next upload, `key` at `len` bytes, to its slot.
    ///
    /// # Errors
    /// When `bytes` is not `len` long, the next slot is another upload, or the write fails.
    pub fn put(&mut self, key: &str, len: usize, bytes: &[u8]) -> Result<(), String> {
        check_length(key, len, bytes)?;
        let at = expect_slot(&self.index.slots, self.next, key, len)?.offset;
        self.index.slots[self.next].digest = self.writer.put(at, bytes)?;
        self.next += 1;
        Ok(())
    }

    /// Writes the index and the preamble once the walk has put every slot, and returns the file.
    ///
    /// # Errors
    /// When the second walk placed another number of uploads than the first, or a write fails.
    pub fn finish(self) -> Result<WrittenRank, String> {
        let Self {
            path,
            data_offset,
            writer,
            mut index,
            next,
        } = self;
        if next != index.slots.len() {
            return Err(format!(
                "the second walk placed {next} weights where the first placed {}",
                index.slots.len()
            ));
        }
        index.file_digest = writer.file_digest();
        let bytes =
            serde_json::to_vec(&index).map_err(|error| format!("serialise the index: {error}"))?;
        let data_bytes = usize::try_from(index.data_bytes)
            .map_err(|_| format!("{}: the data does not fit usize", path.display()))?;
        let padded_data_bytes = align_up_page(data_bytes) as u64;
        let header_hash = writer.finish(
            index.rank,
            index.world,
            &bytes,
            index.data_bytes,
            padded_data_bytes,
        )?;
        Ok(WrittenRank {
            path,
            file_bytes: data_offset + padded_data_bytes,
            data_offset,
            data_bytes: index.data_bytes,
            slots: index.slots.len(),
            header_hash,
            file_digest: index.file_digest,
        })
    }
}

/// A load's pass over a rank file's slots: each upload of its walk must be the next slot, the
/// same weight at the same length, and the walk must take every slot.
#[derive(Debug)]
pub struct SlotCursor<'a> {
    slots: &'a [Slot],
    next: usize,
}

impl<'a> SlotCursor<'a> {
    /// A pass from the first of `slots`.
    #[must_use]
    pub fn new(slots: &'a [Slot]) -> Self {
        Self { slots, next: 0 }
    }

    /// The data offset of the walk's next upload, once the next slot is `key` at `len` bytes.
    ///
    /// # Errors
    /// When the file ends first, or its next slot is another upload or length.
    pub fn take(&mut self, key: &str, len: usize) -> Result<usize, String> {
        let at = expect_slot(self.slots, self.next, key, len)?.offset;
        self.next += 1;
        Ok(at)
    }

    /// Once the walk has taken every slot.
    ///
    /// # Errors
    /// When the file holds slots the walk did not take.
    pub fn finish(&self) -> Result<(), String> {
        if self.next == self.slots.len() {
            Ok(())
        } else {
            Err(format!(
                "the rank file holds {} weights, and this configuration loads {}",
                self.slots.len(),
                self.next
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Index {
        payloads: Vec<(String, usize, usize, u64)>,
        file_digest: u64,
    }

    fn scratch(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-rank-file-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory.join(rank_file_name(1, 2))
    }

    /// Two payloads, the second past a gap, round-trip through the writer, the index read and the
    /// verify; a flipped payload byte and a flipped index byte are each refused.
    #[test]
    fn a_rank_file_round_trips_and_refuses_a_flipped_byte() {
        let path = scratch("round-trip");
        let payloads: [(&str, usize, Vec<u8>); 2] =
            [("a", 0, vec![1, 2, 3]), ("b", 256, vec![9; 5000])];
        let placeholder = Index {
            payloads: payloads
                .iter()
                .map(|(name, at, bytes)| ((*name).to_owned(), *at, bytes.len(), u64::MAX))
                .collect(),
            file_digest: u64::MAX,
        };
        let data_offset = data_offset_for(serde_json::to_vec(&placeholder).unwrap().len());
        let mut writer = RankFileWriter::create(&path, data_offset).unwrap();
        let mut index = placeholder;
        for (slot, (_, at, bytes)) in index.payloads.iter_mut().zip(&payloads) {
            slot.3 = writer.put(*at, bytes).unwrap();
        }
        index.file_digest = writer.file_digest();
        let data = 256 + 5000;
        let index_bytes = serde_json::to_vec(&index).unwrap();
        writer
            .finish(1, 2, &index_bytes, data as u64, align_up_page(data) as u64)
            .unwrap();

        let (preamble, read): (_, Index) = read_index(&path).unwrap();
        assert_eq!(read, index);
        assert_eq!((preamble.rank, preamble.world), (1, 2));
        assert_eq!(preamble.data_offset, data_offset);
        let verify = |path: &Path| {
            verify_payloads(
                path,
                data_offset,
                read.payloads
                    .iter()
                    .map(|(name, offset, bytes, digest)| PayloadDigest {
                        name,
                        offset: *offset,
                        bytes: *bytes,
                        digest: *digest,
                    }),
                read.file_digest,
            )
        };
        verify(&path).unwrap();

        let mut flipped = std::fs::read(&path).unwrap();
        flipped[data_offset as usize + 300] ^= 1;
        std::fs::write(&path, &flipped).unwrap();
        assert!(
            verify(&path)
                .unwrap_err()
                .contains("tensor b digest mismatch")
        );
        flipped[data_offset as usize + 300] ^= 1;
        flipped[PREAMBLE_BYTES + 3] ^= 1;
        std::fs::write(&path, &flipped).unwrap();
        assert!(
            read_index::<Index>(&path)
                .unwrap_err()
                .contains("header hash mismatch")
        );
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Records {
        scales: Vec<u32>,
    }

    impl SlotRecords for Records {
        const MODEL: &'static str = "fixture";
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Other {}

    impl SlotRecords for Other {
        const MODEL: &'static str = "other";
    }

    /// A three-upload walk planned, written and taken again: the slots sit on 256-byte offsets,
    /// the pack's records sit between the source and the slots, a write or load that walks
    /// another way is refused, and a file of another rank or model is refused.
    #[test]
    fn a_slot_file_holds_its_walk_and_a_load_takes_it_again() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let layout = TpLayout {
            rank: 1,
            world_size: 2,
        };
        let walk: [(&str, Vec<u8>); 3] =
            [("a", vec![1, 2, 3]), ("b", vec![7; 300]), ("c", vec![9; 5])];
        let mut plan = SlotPlan::default();
        for (key, bytes) in &walk {
            plan.place(key, bytes.len());
        }
        let offsets: Vec<usize> = plan.slots().iter().map(|slot| slot.offset).collect();
        assert_eq!(offsets, [0, 256, 768]);
        let index = plan.into_index(layout, "fixture", Records { scales: vec![5] });
        let mut writer = SlotFileWriter::create(directory.path(), index).expect("the file opens");
        let error = writer.put("b", 300, &walk[1].1).unwrap_err();
        assert!(error.contains("slot 0 is a (3 bytes)"), "{error}");
        let error = writer.put("a", 3, &[1, 2]).unwrap_err();
        assert!(error.contains("a came to 2 bytes"), "{error}");
        for (key, bytes) in &walk {
            writer
                .put(key, bytes.len(), bytes)
                .expect("the walk writes");
        }
        let written = writer.finish().expect("the file finishes");
        assert_eq!(written.path, directory.path().join(rank_file_name(1, 2)));
        assert_eq!(
            std::fs::metadata(&written.path).unwrap().len(),
            written.file_bytes
        );
        assert_eq!((written.slots, written.data_bytes), (3, 773));

        let bytes = std::fs::read(&written.path).unwrap();
        let (preamble, _): (_, serde_json::Value) = read_index(&written.path).unwrap();
        let json = std::str::from_utf8(
            &bytes[PREAMBLE_BYTES..PREAMBLE_BYTES + preamble.index_len as usize],
        )
        .unwrap();
        let at = |field: &str| json.find(&format!("\"{field}\":")).expect(field);
        assert!(
            at("source") < at("scales") && at("scales") < at("slots"),
            "{json}"
        );

        let index = SlotIndex::<Records>::verify(&written.path, layout).expect("it verifies");
        assert_eq!(index.extra.scales, [5]);
        assert_eq!(index.file_digest, written.file_digest);
        let at = written.data_offset as usize + index.slots[1].offset;
        assert_eq!(bytes[at..at + 300], walk[1].1);
        let mut cursor = SlotCursor::new(&index.slots);
        assert_eq!(cursor.take("a", 3).unwrap(), 0);
        let error = cursor.finish().unwrap_err();
        assert!(
            error.contains("holds 3 weights, and this configuration loads 1"),
            "{error}"
        );
        let error = cursor.take("c", 5).unwrap_err();
        assert!(error.contains("slot 1 is b"), "{error}");
        assert_eq!(cursor.take("b", 300).unwrap(), 256);
        assert_eq!(cursor.take("c", 5).unwrap(), 768);
        cursor.finish().expect("the walk took every slot");
        let error = cursor.take("d", 1).unwrap_err();
        assert!(error.contains("ends before d"), "{error}");

        let other_rank = TpLayout {
            rank: 0,
            world_size: 2,
        };
        let error = SlotIndex::<Records>::read(&written.path, other_rank).unwrap_err();
        assert!(error.contains("rank 1 by its preamble"), "{error}");
        let error = SlotIndex::<Other>::read(&written.path, layout).unwrap_err();
        assert!(error.contains("a fixture rank file, not other"), "{error}");
    }

    /// An index off [SlotPlan]'s placement is refused though its header hash is sound: two walk
    /// entries on the same bytes each verify against their digest, so only the placement check
    /// keeps a load from reading one weight's bytes as another's.
    #[test]
    fn an_index_off_the_plan_is_refused() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let layout = TpLayout {
            rank: 1,
            world_size: 2,
        };
        let read = |name: &str, edit: fn(&mut SlotIndex<Records>)| {
            let mut plan = SlotPlan::default();
            plan.place("a", 3);
            plan.place("b", 3);
            let mut index = plan.into_index(layout, "fixture", Records { scales: vec![] });
            edit(&mut index);
            let path = directory.path().join(name);
            let index_bytes = serde_json::to_vec(&index).unwrap();
            let mut writer =
                RankFileWriter::create(&path, data_offset_for(index_bytes.len())).unwrap();
            for slot in &index.slots {
                if writer.put(slot.offset, &[1, 2, 3]).is_err() {
                    break;
                }
            }
            let padded = align_up_page(index.data_bytes as usize) as u64;
            writer
                .finish(1, 2, &index_bytes, index.data_bytes, padded)
                .unwrap();
            SlotIndex::<Records>::read(&path, layout).map(|_| ())
        };
        read("planned", |_| {}).expect("the plan's own placement reads");
        let error = read("aliased", |index| index.slots[1].offset = 0).unwrap_err();
        assert!(
            error.contains("slot b at 0 (3 bytes) is not where the walk places it (256)"),
            "{error}"
        );
        let error = read("unordered", |index| index.slots.swap(0, 1)).unwrap_err();
        assert!(error.contains("slot b at 256"), "{error}");
        let error = read("shifted", |index| index.slots[1].offset = 512).unwrap_err();
        assert!(error.contains("places it (256)"), "{error}");
        let error = read("trailing", |index| index.data_bytes += 256).unwrap_err();
        assert!(
            error.contains("the slots end at byte 259, but the index holds 515 data bytes"),
            "{error}"
        );
    }
}
