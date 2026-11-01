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

//! Per-rank converted weights: one file a rank, the bytes [DeviceRank::load] uploads, in its
//! order, so a rank boots from one parallel direct-I/O read with no checkpoint on the node.
//!
//! **Attribution:** pre-converting per-rank layouts and loading them with parallel `O_DIRECT`
//! reads draws on prior art in [`b12x`](https://github.com/local-inference-lab/b12x)
//! (`b12x/loader/_direct.c`) and [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference)
//! (Apache-2.0; Luke Alonso `@lukealonso`, `@original-el8`, and contributors). See the root
//! `NOTICE` file.
//!
//! The file is the container the V4.1 arm writes ([sparkflash_weights::rank_file]) with this
//! pack's index: every upload of the rank's walk as a slot (its key, offset, length and digest),
//! and the spec of every checkpoint tensor the walk reads, so [DeviceRank::load_converted] runs
//! the same walk over the specs and finds each weight in the slab the file was read into.
//! [convert_rank] walks the checkpoint twice on the host: once for the slots' offsets, which fix
//! the index's size and so where the data starts, and once to stream each weight's bytes to its
//! slot. Nothing is converted on the way: a slot holds what [DeviceRank::load] would upload.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::BackendError;
use sparkflash_weights::rank_file::{
    ConvertedPreamble, PayloadDigest, RankFileWriter, align_up_page, data_offset_for,
    rank_file_name, verify_payloads,
};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

use crate::config::Glm5NextConfig;
use crate::device::{DeviceRank, Sink, Uploader, check_length, refusal};

/// What a GLM rank file's index names as its model.
pub const CONVERTED_MODEL: &str = "glm5-next";

/// Where each slot starts: the alignment of a device allocation, which the kernels' vector loads
/// assume of every weight.
const SLOT_ALIGN_BYTES: usize = 256;

/// One upload of the rank's walk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedSlot {
    /// The checkpoint names it is built from, joined by `+`; a scale grid's ends `_scale_inv`.
    pub key: String,
    /// Its byte offset from the file's `data_offset`, a multiple of 256.
    pub offset: usize,
    /// Its length in bytes.
    pub bytes: usize,
    /// The FNV-1a 64 digest of its bytes.
    pub digest: u64,
}

/// A checkpoint tensor the walk reads, as the checkpoint describes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedSpec {
    /// The name exactly as the checkpoint spells it.
    pub name: String,
    /// Its format's name ([DType::name]).
    pub dtype: String,
    /// Its whole shape, before the rank's split.
    pub shape: Vec<usize>,
}

/// The JSON index of a GLM rank file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvertedIndex {
    /// [CONVERTED_MODEL].
    pub model: String,
    /// The rank, `0..world`.
    pub rank: usize,
    /// The tensor-parallel world.
    pub world: usize,
    /// What the converter read, as it was given.
    pub source: String,
    /// The spec of every checkpoint tensor the walk reads, in the order it reads them.
    pub specs: Vec<ConvertedSpec>,
    /// Every upload, in walk order.
    pub slots: Vec<ConvertedSlot>,
    /// The end of the last slot.
    pub data_bytes: u64,
    /// The digest of the slots' digests, in order.
    pub file_digest: u64,
}

/// The specs a rank file records, as a source the walk reads shapes and formats from. It holds no
/// bytes: a read is refused.
pub struct ConvertedSpecs {
    names: Vec<String>,
    specs: HashMap<String, TensorSpec>,
}

impl ConvertedSpecs {
    /// The specs of `index`.
    ///
    /// # Errors
    /// When a spec names a format this tree does not know, or one name is recorded twice.
    pub fn new(index: &ConvertedIndex) -> Result<Self, String> {
        let mut specs = HashMap::with_capacity(index.specs.len());
        for spec in &index.specs {
            let dtype = DType::from_name(&spec.dtype).ok_or_else(|| {
                format!(
                    "{} is recorded as {}, no format here",
                    spec.name, spec.dtype
                )
            })?;
            let spec = TensorSpec {
                name: spec.name.clone(),
                dtype,
                shape: spec.shape.clone(),
            };
            if let Some(first) = specs.insert(spec.name.clone(), spec) {
                return Err(format!("{} is recorded twice", first.name));
            }
        }
        Ok(Self {
            names: index.specs.iter().map(|spec| spec.name.clone()).collect(),
            specs,
        })
    }
}

impl WeightSource for ConvertedSpecs {
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.specs
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        Err(WeightError::Refused {
            name: name.to_owned(),
            detail: "a rank file's specs hold no bytes; its slots are read with the file".into(),
        })
    }
}

/// Slot `next` of `slots`, once it is `key` at `len` bytes, the upload the walk makes next.
pub(crate) fn expect_slot<'s>(
    slots: &'s [ConvertedSlot],
    next: usize,
    key: &str,
    len: usize,
) -> Result<&'s ConvertedSlot, BackendError> {
    let Some(slot) = slots.get(next) else {
        return Err(refusal(format!(
            "the rank file ends before {key}, which this configuration loads"
        )));
    };
    if slot.key != key || slot.bytes != len {
        return Err(refusal(format!(
            "the rank file's slot {next} is {} ({} bytes), where this configuration loads {key} \
             ({len} bytes)",
            slot.key, slot.bytes
        )));
    }
    Ok(slot)
}

/// The first pass: each upload's slot, at the next aligned offset, and each tensor's spec.
#[derive(Default)]
struct Planned {
    specs: Vec<ConvertedSpec>,
    slots: Vec<ConvertedSlot>,
    end: usize,
}

impl Sink for Planned {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, BackendError> {
        let offset = self.end.next_multiple_of(SLOT_ALIGN_BYTES);
        self.slots.push(ConvertedSlot {
            key: key.to_owned(),
            offset,
            bytes: len,
            // The widest digest, so the placeholder index is never shorter than the final one.
            digest: u64::MAX,
        });
        self.end = offset + len;
        Ok(Deviceptr(0))
    }

    fn derived(&mut self, _: &[u8]) -> Result<Deviceptr, BackendError> {
        Ok(Deviceptr(0))
    }

    fn spec(&mut self, spec: &TensorSpec) {
        self.specs.push(ConvertedSpec {
            name: spec.name.clone(),
            dtype: spec.dtype.name().to_owned(),
            shape: spec.shape.clone(),
        });
    }
}

/// The second pass: each upload's bytes to the slot the first pass gave it.
struct Writing<'a> {
    writer: RankFileWriter,
    slots: &'a mut [ConvertedSlot],
    next: usize,
}

impl Sink for Writing<'_> {
    fn wants_bytes(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, BackendError> {
        check_length(key, len, &bytes)?;
        let at = expect_slot(self.slots, self.next, key, len)?.offset;
        self.slots[self.next].digest = self.writer.put(at, &bytes).map_err(refusal)?;
        self.next += 1;
        Ok(Deviceptr(0))
    }

    fn derived(&mut self, _: &[u8]) -> Result<Deviceptr, BackendError> {
        Ok(Deviceptr(0))
    }
}

/// One converted rank file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedRank {
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

/// Writes rank `layout.rank`'s file into `output_dir` (as [rank_file_name] names it) from the
/// checkpoint `source`, recording `label` as what was read. It runs on the host; no device is
/// touched.
///
/// # Errors
/// What [DeviceRank::load] refuses of the configuration or the checkpoint, and a file that cannot
/// be written.
pub fn convert_rank<S: WeightSource + ?Sized>(
    source: &S,
    config: &Glm5NextConfig,
    layout: TpLayout,
    output_dir: &Path,
    label: &str,
) -> Result<ConvertedRank, BackendError> {
    DeviceRank::check_rank(config, layout)?;
    let mut planned = Planned::default();
    Uploader::new(source, config, layout, &mut planned).place()?;
    let Planned { specs, slots, end } = planned;
    let mut index = ConvertedIndex {
        model: CONVERTED_MODEL.to_owned(),
        rank: layout.rank,
        world: layout.world_size,
        source: label.to_owned(),
        specs,
        slots,
        data_bytes: end as u64,
        file_digest: u64::MAX,
    };
    let placeholder = serde_json::to_vec(&index).map_err(refusal_of("serialise the index"))?;
    let data_offset = data_offset_for(placeholder.len());
    let padded_data_bytes = align_up_page(end) as u64;

    fs::create_dir_all(output_dir)
        .map_err(|error| refusal(format!("create {}: {error}", output_dir.display())))?;
    let path = output_dir.join(rank_file_name(layout.rank, layout.world_size));
    let mut writing = Writing {
        writer: RankFileWriter::create(&path, data_offset).map_err(refusal)?,
        slots: &mut index.slots,
        next: 0,
    };
    Uploader::new(source, config, layout, &mut writing).place()?;
    let Writing { writer, next, .. } = writing;
    if next != index.slots.len() {
        return Err(refusal(format!(
            "the second walk placed {next} weights where the first placed {}",
            index.slots.len()
        )));
    }
    index.file_digest = writer.file_digest();
    let bytes = serde_json::to_vec(&index).map_err(refusal_of("serialise the index"))?;
    let header_hash = writer
        .finish(
            layout.rank,
            layout.world_size,
            &bytes,
            index.data_bytes,
            padded_data_bytes,
        )
        .map_err(refusal)?;
    Ok(ConvertedRank {
        path,
        file_bytes: data_offset + padded_data_bytes,
        data_offset,
        data_bytes: index.data_bytes,
        slots: index.slots.len(),
        header_hash,
        file_digest: index.file_digest,
    })
}

fn refusal_of(what: &'static str) -> impl Fn(serde_json::Error) -> BackendError {
    move |error| refusal(format!("{what}: {error}"))
}

/// Reads a rank file's preamble and index ([sparkflash_weights::rank_file::read_index]), once it
/// is a GLM file of rank `layout.rank` in a world of `layout.world_size` whose slots lie in its
/// data. Touching no weight, this is the boot check.
///
/// # Errors
/// What the container's check refuses, another model's or rank's file, and a slot past the data.
pub fn read_index(
    path: &Path,
    layout: TpLayout,
) -> Result<(ConvertedPreamble, ConvertedIndex), String> {
    let (mut preamble, index): (_, ConvertedIndex) =
        sparkflash_weights::rank_file::read_index(path)?;
    let shown = path.display();
    if index.model != CONVERTED_MODEL {
        return Err(format!(
            "{shown} is a {} rank file, not {CONVERTED_MODEL}",
            index.model
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
    if preamble.data_bytes != index.data_bytes || index.data_bytes > preamble.padded_data_bytes {
        return Err(format!(
            "{shown} holds {} data bytes by its preamble, {} by its index, in {} padded",
            preamble.data_bytes, index.data_bytes, preamble.padded_data_bytes
        ));
    }
    for slot in &index.slots {
        let end = slot.offset.checked_add(slot.bytes).map(|end| end as u64);
        if !slot.offset.is_multiple_of(SLOT_ALIGN_BYTES)
            || end.is_none_or(|end| end > index.data_bytes)
        {
            return Err(format!(
                "{shown}: slot {} at {} ({} bytes) is unaligned or past the {} data bytes",
                slot.key, slot.offset, slot.bytes, index.data_bytes
            ));
        }
    }
    preamble.file_digest = index.file_digest;
    Ok((preamble, index))
}

/// Re-reads every slot of a rank file and checks its digest, and the digest of the digests.
///
/// # Errors
/// What [read_index] refuses, and a slot or file digest that differs.
pub fn verify_converted(path: &Path, layout: TpLayout) -> Result<ConvertedIndex, String> {
    let (preamble, index) = read_index(path, layout)?;
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

#[cfg(test)]
mod tests {
    use std::os::unix::fs::FileExt;

    use sparkflash_weights::rank_file::PREAMBLE_BYTES;
    use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};

    use super::*;
    use crate::checkpoint::write_synthetic_checkpoint;
    use crate::tensor::LM_PREFIX;

    fn layout(rank: usize, world_size: usize) -> TpLayout {
        TpLayout { rank, world_size }
    }

    fn flip(path: &Path, at: u64) {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut byte = [0u8];
        file.read_exact_at(&mut byte, at).unwrap();
        file.write_all_at(&[byte[0] ^ 1], at).unwrap();
    }

    /// Both ranks of a two-rank world written, checked and replayed: the files are the same bytes
    /// every time, a replicated weight's slot holds the checkpoint's bytes, the walk over the
    /// recorded specs gives the recorded slots, and a file of another rank or world, a flipped
    /// payload byte or a flipped index byte is refused.
    #[test]
    fn a_rank_file_holds_the_walk_and_refuses_what_is_not_it() {
        let config = Glm5NextConfig::small();
        let checkpoint = tempfile::tempdir().expect("a temporary directory");
        write_synthetic_checkpoint(checkpoint.path(), &config, 7, 2).expect("the fixture writes");
        let source =
            SafeTensorsSource::open(checkpoint.path(), &checkpoint.path().join(INDEX_NAME))
                .expect("the fixture opens");
        let (first, second) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let norm = format!("{LM_PREFIX}.norm.weight");
        for rank in 0..2 {
            let converted =
                convert_rank(&source, &config, layout(rank, 2), first.path(), "fixture")
                    .expect("the rank converts");
            let again = convert_rank(&source, &config, layout(rank, 2), second.path(), "fixture")
                .expect("the rank converts again");
            let bytes = fs::read(&converted.path).unwrap();
            assert_eq!(
                bytes,
                fs::read(&again.path).unwrap(),
                "rank {rank} is deterministic"
            );
            assert_eq!(bytes.len() as u64, converted.file_bytes);

            let index = verify_converted(&converted.path, layout(rank, 2)).expect("it verifies");
            assert_eq!(index.slots.len(), converted.slots);
            assert_eq!(index.file_digest, converted.file_digest);
            let slot = index
                .slots
                .iter()
                .find(|slot| slot.key == norm)
                .expect("the norm");
            let at = converted.data_offset as usize + slot.offset;
            assert_eq!(bytes[at..at + slot.bytes], source.read(&norm).unwrap());

            let specs = ConvertedSpecs::new(&index).expect("the specs parse");
            assert!(specs.read(&norm).is_err(), "the specs hold no bytes");
            let mut replay = Planned::default();
            Uploader::new(&specs, &config, layout(rank, 2), &mut replay)
                .place()
                .expect("the walk runs over the recorded specs");
            assert_eq!(replay.specs, index.specs);
            let placed = |slots: &[ConvertedSlot]| -> Vec<(String, usize, usize)> {
                slots
                    .iter()
                    .map(|slot| (slot.key.clone(), slot.offset, slot.bytes))
                    .collect()
            };
            assert_eq!(placed(&replay.slots), placed(&index.slots));
        }

        let zero = first.path().join(rank_file_name(0, 2));
        let error = read_index(&zero, layout(1, 2)).unwrap_err();
        assert!(error.contains("rank 1"), "{error}");
        let error = read_index(&zero, layout(0, 4)).unwrap_err();
        assert!(error.contains("world 4"), "{error}");
        let error = convert_rank(&source, &config, layout(0, 3), first.path(), "fixture")
            .expect_err("three ranks do not split the fixture");
        assert!(error.to_string().contains("split"), "{error}");

        let (preamble, index) = read_index(&zero, layout(0, 2)).unwrap();
        flip(&zero, preamble.data_offset + index.slots[3].offset as u64);
        read_index(&zero, layout(0, 2)).expect("the boot check reads no weight");
        let error = verify_converted(&zero, layout(0, 2)).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");
        // A byte of the index, past the preamble.
        flip(&zero, PREAMBLE_BYTES as u64 + 8);
        let error = read_index(&zero, layout(0, 2)).unwrap_err();
        assert!(error.contains("header hash"), "{error}");
    }
}
