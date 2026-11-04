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
//! The file is the walk-ordered rank file the Qwen arm also writes
//! ([sparkflash_weights::rank_file::SlotIndex]): every upload of the rank's walk is a slot (its
//! key, offset, length and digest), and this pack records the spec of every checkpoint tensor the
//! walk reads ([WalkSpecs]), so [DeviceRank::load_converted] runs the same walk over the specs and
//! finds each weight in the slab the file was read into.
//! [convert_rank] walks the checkpoint twice on the host: once for the slots' offsets, which fix
//! the index's size and so where the data starts, and once to stream each weight's bytes to its
//! slot. Nothing is converted on the way: a slot holds what [DeviceRank::load] would upload.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::BackendError;
use sparkflash_weights::rank_file::{
    SlotFileWriter, SlotIndex, SlotPlan, SlotRecords, WrittenRank,
};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

use crate::config::Glm5NextConfig;
use crate::device::{DeviceRank, Sink, Uploader, refusal};

/// What a GLM rank file's index names as its model.
pub const CONVERTED_MODEL: &str = "glm5-next";

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

/// What a GLM rank file records beside its slots.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalkSpecs {
    /// The spec of every checkpoint tensor the walk reads, in the order it reads them.
    pub specs: Vec<ConvertedSpec>,
}

impl SlotRecords for WalkSpecs {
    const MODEL: &'static str = CONVERTED_MODEL;
}

/// The JSON index of a GLM rank file.
pub type ConvertedIndex = SlotIndex<WalkSpecs>;

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
        let mut specs = HashMap::with_capacity(index.extra.specs.len());
        for spec in &index.extra.specs {
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
            names: index
                .extra
                .specs
                .iter()
                .map(|spec| spec.name.clone())
                .collect(),
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

/// The first pass: each upload's slot, at the next aligned offset, and each tensor's spec.
#[derive(Default)]
struct Planned {
    plan: SlotPlan,
    specs: Vec<ConvertedSpec>,
}

impl Sink for Planned {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, BackendError> {
        self.plan.place(key, len);
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
struct Writing(SlotFileWriter<WalkSpecs>);

impl Sink for Writing {
    fn wants_bytes(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, BackendError> {
        self.0.put(key, len, &bytes).map_err(refusal)?;
        Ok(Deviceptr(0))
    }

    fn derived(&mut self, _: &[u8]) -> Result<Deviceptr, BackendError> {
        Ok(Deviceptr(0))
    }
}

/// Writes rank `layout.rank`'s file into `output_dir` (as
/// [sparkflash_weights::rank_file::rank_file_name] names it) from the checkpoint `source`,
/// recording `label` as what was read. It runs on the host; no device is touched.
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
) -> Result<WrittenRank, BackendError> {
    DeviceRank::check_rank(config, layout)?;
    let mut planned = Planned::default();
    Uploader::new(source, config, layout, &mut planned).place()?;
    let Planned { plan, specs } = planned;
    let index = plan.into_index(layout, label, WalkSpecs { specs });
    let mut writing = Writing(SlotFileWriter::create(output_dir, index).map_err(refusal)?);
    Uploader::new(source, config, layout, &mut writing).place()?;
    writing.0.finish().map_err(refusal)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::FileExt;

    use sparkflash_weights::rank_file::{PREAMBLE_BYTES, Slot, rank_file_name};
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

            let index =
                ConvertedIndex::verify(&converted.path, layout(rank, 2)).expect("it verifies");
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
            assert_eq!(replay.specs, index.extra.specs);
            let placed = |slots: &[Slot]| -> Vec<(String, usize, usize)> {
                slots
                    .iter()
                    .map(|slot| (slot.key.clone(), slot.offset, slot.bytes))
                    .collect()
            };
            assert_eq!(placed(replay.plan.slots()), placed(&index.slots));
        }

        let zero = first.path().join(rank_file_name(0, 2));
        let error = ConvertedIndex::read(&zero, layout(1, 2)).unwrap_err();
        assert!(error.contains("rank 1"), "{error}");
        let error = ConvertedIndex::read(&zero, layout(0, 4)).unwrap_err();
        assert!(error.contains("world 4"), "{error}");
        let error = convert_rank(&source, &config, layout(0, 3), first.path(), "fixture")
            .expect_err("three ranks do not split the fixture");
        assert!(error.to_string().contains("split"), "{error}");

        let (preamble, index) = ConvertedIndex::read(&zero, layout(0, 2)).unwrap();
        flip(&zero, preamble.data_offset + index.slots[3].offset as u64);
        ConvertedIndex::read(&zero, layout(0, 2)).expect("the boot check reads no weight");
        let error = ConvertedIndex::verify(&zero, layout(0, 2)).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");
        // A byte of the index, past the preamble.
        flip(&zero, PREAMBLE_BYTES as u64 + 8);
        let error = ConvertedIndex::read(&zero, layout(0, 2)).unwrap_err();
        assert!(error.contains("header hash"), "{error}");
    }
}
