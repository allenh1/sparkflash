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

//! Per-rank converted weights: one file a rank, the bytes that rank's weight walk uploads, in its
//! order, so a rank boots from one parallel direct-I/O read instead of 77.8 GiB of per-tensor
//! checkpoint reads and allocations.
//!
//! **Attribution:** pre-converting per-rank layouts and loading them with parallel `O_DIRECT`
//! reads draws on prior art in [`b12x`](https://github.com/local-inference-lab/b12x)
//! (`b12x/loader/_direct.c`) and [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference)
//! (Apache-2.0; Luke Alonso `@lukealonso`, `@original-el8`, and contributors). See the root
//! `NOTICE` file.
//!
//! The file is the walk-ordered rank file the GLM arm also writes
//! ([sparkflash_weights::rank_file::SlotIndex]): every upload of the rank's walk is a slot (its
//! key, offset, length and digest), and this pack records the two per-tensor NVFP4 scales of every
//! NVFP4 matrix in walk order ([Nvfp4Scalars]). [convert_rank] walks the checkpoint
//! twice on the host: once for the slots' offsets and the scales, and once to stream each weight's
//! bytes to its slot. [load_converted] reads the file into one device allocation and runs the same
//! walk over a [Slab], which returns each weight's address without reading the checkpoint.
//!
//! The walk is the tensor-parallel one ([crate::device::build_shard]), so a file holds its rank's
//! share of every sharded tensor - the attention and Gated DeltaNet heads, the experts' padded
//! width and the shared expert - exactly as
//! [crate::device::Qwen4ExpDeviceBackend::open_rank] uploads it. A TP=1 file
//! ([sparkflash_weights::rank_file::rank_file_name] of rank 0 in a world of 1) is the whole
//! model.
//!
//! Qwen's PLE n-gram table is host-paged, so the checkpoint directory (and its 51.2 GiB of PLE
//! shards) still has to be present: the converted file replaces the resident weights' upload, not
//! the pager's source.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sparkflash_kernels::cuda_driver::{
    Cuda, DEFAULT_DIRECT_CHUNK_BYTES, DEFAULT_DIRECT_CONCURRENCY, Deviceptr,
};
use sparkflash_weights::rank_file::{
    SlotCursor, SlotFileWriter, SlotIndex, SlotPlan, SlotRecords, WrittenRank,
};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{WeightSource, check_length};

use crate::config::Qwen4ExpConfig;
use crate::device::tp::{Shard, rank_config};
use crate::device::{GpuWeights, Sink, Uploader, build_shard};

/// What a Qwen4-Exp rank file's index names as its model.
pub const CONVERTED_MODEL: &str = "qwen4-exp";

/// What a Qwen4-Exp rank file records beside its slots.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nvfp4Scalars {
    /// The two per-tensor NVFP4 scales of every NVFP4 matrix, in walk order.
    pub nvfp4_scalars: Vec<[u32; 2]>,
}

impl SlotRecords for Nvfp4Scalars {
    const MODEL: &'static str = CONVERTED_MODEL;
}

/// The JSON index of a Qwen4-Exp rank file.
pub type ConvertedIndex = SlotIndex<Nvfp4Scalars>;

/// The first pass: each upload's slot, at the next aligned offset, and each NVFP4 scale pair.
#[derive(Default)]
struct Plan {
    plan: SlotPlan,
    scalars: Vec<[u32; 2]>,
}

impl Sink for Plan {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn wants_scalars(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, String> {
        self.plan.place(key, len);
        Ok(Deviceptr(0))
    }

    fn derived(&mut self, key: &str, len: usize, _: &[u8]) -> Result<Deviceptr, String> {
        self.plan.place(key, len);
        Ok(Deviceptr(0))
    }

    fn nvfp4_scalars(&mut self, scale_2: f32, input_scale: f32) -> Result<(f32, f32), String> {
        self.scalars
            .push([scale_2.to_bits(), input_scale.to_bits()]);
        Ok((scale_2, input_scale))
    }
}

/// The second pass: each upload's bytes to the slot the first pass gave it.
struct Writing(SlotFileWriter<Nvfp4Scalars>);

impl Sink for Writing {
    fn wants_bytes(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, String> {
        self.0.put(key, len, &bytes)?;
        Ok(Deviceptr(0))
    }

    fn derived(&mut self, key: &str, len: usize, _: &[u8]) -> Result<Deviceptr, String> {
        // The table holds device addresses, which change with every allocation, so the file
        // stores a deterministic placeholder and the loader writes the rebuilt table instead.
        self.0.put(key, len, &vec![0; len])?;
        Ok(Deviceptr(0))
    }

    fn nvfp4_scalars(&mut self, scale_2: f32, input_scale: f32) -> Result<(f32, f32), String> {
        Ok((scale_2, input_scale))
    }
}

/// The slab a rank file was read into: each upload is the next slot, which must be the same
/// weight at the same length, and the NVFP4 scales come from the index.
pub(crate) struct Slab<'a> {
    base: Deviceptr,
    slots: SlotCursor<'a>,
    scalars: &'a [[u32; 2]],
    next_scalar: usize,
    cuda: &'a Cuda,
}

impl<'a> Slab<'a> {
    fn new(base: Deviceptr, index: &'a ConvertedIndex, cuda: &'a Cuda) -> Self {
        Self {
            base,
            slots: SlotCursor::new(&index.slots),
            scalars: &index.extra.nvfp4_scalars,
            next_scalar: 0,
            cuda,
        }
    }
}

impl Sink for Slab<'_> {
    fn wants_bytes(&self) -> bool {
        false
    }

    fn wants_scalars(&self) -> bool {
        false
    }

    fn weight(&mut self, key: &str, len: usize, _: Vec<u8>) -> Result<Deviceptr, String> {
        Ok(self.base.offset(self.slots.take(key, len)?))
    }

    fn derived(&mut self, key: &str, len: usize, bytes: &[u8]) -> Result<Deviceptr, String> {
        check_length(key, len, bytes)?;
        let pointer = self.base.offset(self.slots.take(key, len)?);
        self.cuda.copy_htod(&pointer, bytes)?;
        Ok(pointer)
    }

    fn nvfp4_scalars(&mut self, _: f32, _: f32) -> Result<(f32, f32), String> {
        let pair = self.scalars.get(self.next_scalar).ok_or_else(|| {
            "the rank file holds fewer NVFP4 scales than this configuration loads".to_owned()
        })?;
        self.next_scalar += 1;
        Ok((f32::from_bits(pair[0]), f32::from_bits(pair[1])))
    }
}

/// Writes rank `layout.rank`'s file into output_dir (as
/// [sparkflash_weights::rank_file::rank_file_name] names it) from the checkpoint source, recording
/// label as what was read. It runs on the host; no device is touched.
///
/// # Errors
/// A world the model's heads or widths do not split over ([rank_config]), then everything the
/// weight walk refuses of the configuration or the checkpoint, and a file that cannot be written.
pub fn convert_rank(
    source: &dyn WeightSource,
    config: &Qwen4ExpConfig,
    layout: TpLayout,
    directory: &Path,
    output_dir: &Path,
    label: &str,
) -> Result<WrittenRank, String> {
    let shard = Shard::of(layout);
    rank_config(config, shard)?;
    let mut plan = Plan::default();
    let _ = build_shard(
        &Uploader::planning(&mut plan),
        directory,
        source,
        config,
        shard,
    )?;
    let Plan { plan, scalars } = plan;
    let index = plan.into_index(
        layout,
        label,
        Nvfp4Scalars {
            nvfp4_scalars: scalars,
        },
    );
    let mut writing = Writing(SlotFileWriter::create(output_dir, index)?);
    let _ = build_shard(
        &Uploader::planning(&mut writing),
        directory,
        source,
        config,
        shard,
    )?;
    writing.0.finish()
}

/// Reads rank `layout.rank`'s file into one device allocation and returns the weights the
/// rank's walk places in it.
pub(crate) fn load_converted(
    cuda: &Cuda,
    config: &Qwen4ExpConfig,
    directory: &Path,
    path: &Path,
    source: &dyn WeightSource,
    layout: TpLayout,
) -> Result<GpuWeights, String> {
    let shard = Shard::of(layout);
    let (preamble, index) = ConvertedIndex::read(path, layout)?;
    let data = usize::try_from(preamble.padded_data_bytes)
        .map_err(|_| "the rank file's data does not fit this machine's range".to_owned())?;
    let mut total = 0usize;
    let base = cuda.alloc(data.max(4), &mut total)?;
    cuda.upload_file_parallel(
        &base,
        path,
        preamble.data_offset,
        data,
        DEFAULT_DIRECT_CHUNK_BYTES,
        DEFAULT_DIRECT_CONCURRENCY,
        true,
    )?;
    let mut sink = Slab::new(base, &index, cuda);
    let weights = build_shard(
        &Uploader::planning(&mut sink),
        directory,
        source,
        config,
        shard,
    )?;
    sink.slots.finish()?;
    if sink.next_scalar != index.extra.nvfp4_scalars.len() {
        return Err(format!(
            "the rank file holds {} NVFP4 scales, and this configuration loads {}",
            index.extra.nvfp4_scalars.len(),
            sink.next_scalar
        ));
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::FileExt;

    use std::fs;

    use sparkflash_weights::rank_file::{
        PREAMBLE_BYTES, Slot, find_converted_rank_file, rank_file_name,
    };
    use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};

    use super::*;
    use crate::checkpoint::write_synthetic_checkpoint;

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

    /// The file a solo launch reads.
    const TP1: TpLayout = TpLayout {
        rank: 0,
        world_size: 1,
    };

    fn placed(slots: &[Slot]) -> Vec<(String, usize, usize)> {
        slots
            .iter()
            .map(|slot| (slot.key.clone(), slot.offset, slot.bytes))
            .collect()
    }

    #[test]
    fn a_rank_file_holds_the_walk_and_refuses_what_is_not_it() {
        let config = Qwen4ExpConfig::tiny();
        let checkpoint = tempfile::tempdir().expect("a temporary directory");
        write_synthetic_checkpoint(checkpoint.path(), &config, 7).expect("the fixture writes");
        let source =
            SafeTensorsSource::open(checkpoint.path(), &checkpoint.path().join(INDEX_NAME))
                .expect("the fixture opens");
        let (first, second) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let converted = convert_rank(
            &source,
            &config,
            TP1,
            checkpoint.path(),
            first.path(),
            "fixture",
        )
        .expect("the rank converts");
        let again = convert_rank(
            &source,
            &config,
            TP1,
            checkpoint.path(),
            second.path(),
            "fixture",
        )
        .expect("the rank converts again");
        let bytes = fs::read(&converted.path).unwrap();
        assert_eq!(
            bytes,
            fs::read(&again.path).unwrap(),
            "the rank file is deterministic"
        );
        assert_eq!(bytes.len() as u64, converted.file_bytes);

        let index = ConvertedIndex::verify(&converted.path, TP1).expect("it verifies");
        assert_eq!(index.slots.len(), converted.slots);
        assert_eq!(index.file_digest, converted.file_digest);

        let spec = source
            .names()
            .into_iter()
            .find(|name| name.ends_with(".embed_tokens.weight"))
            .or_else(|| {
                source
                    .names()
                    .into_iter()
                    .find(|name| name == "lm_head.weight")
            })
            .expect("the fixture has an embedding or a head");
        let slot = index
            .slots
            .iter()
            .find(|slot| slot.key == spec)
            .expect("the walk uploads it");
        let at = converted.data_offset as usize + slot.offset;
        assert_eq!(bytes[at..at + slot.bytes], source.read(&spec).unwrap());

        let mut replay = Plan::default();
        let _ = build_shard(
            &Uploader::planning(&mut replay),
            checkpoint.path(),
            &source,
            &config,
            Shard::WHOLE,
        )
        .expect("the walk replays");
        assert_eq!(placed(replay.plan.slots()), placed(&index.slots));
        assert_eq!(replay.scalars, index.extra.nvfp4_scalars);

        let error =
            ConvertedIndex::verify(&checkpoint.path().join(rank_file_name(0, 1)), TP1).unwrap_err();
        assert!(
            error.contains("No such file") || error.contains("not found"),
            "{error}"
        );

        let (preamble, index) = ConvertedIndex::read(&converted.path, TP1).unwrap();
        flip(
            &converted.path,
            preamble.data_offset + index.slots[2].offset as u64,
        );
        ConvertedIndex::read(&converted.path, TP1).expect("the boot check reads no weight");
        let error = ConvertedIndex::verify(&converted.path, TP1).unwrap_err();
        assert!(error.contains("digest mismatch"), "{error}");
        flip(&converted.path, PREAMBLE_BYTES as u64 + 8);
        let error = ConvertedIndex::read(&converted.path, TP1).unwrap_err();
        assert!(error.contains("header hash"), "{error}");
    }

    /// A configuration every world 1, 2 and 4 divides the way [rank_config] requires.
    fn tensor_parallel_config() -> Qwen4ExpConfig {
        Qwen4ExpConfig {
            num_attention_heads: 4,
            num_key_value_heads: 4,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            // Two whole 64-row expert tiles at TP=4, one at TP=2.
            moe_intermediate_size: 128,
            ..Qwen4ExpConfig::tiny()
        }
    }

    #[test]
    fn every_tensor_configuration_writes_the_file_its_rank_uploads() {
        let config = tensor_parallel_config();
        let checkpoint = tempfile::tempdir().expect("a temporary directory");
        write_synthetic_checkpoint(checkpoint.path(), &config, 7).expect("the fixture writes");
        let source =
            SafeTensorsSource::open(checkpoint.path(), &checkpoint.path().join(INDEX_NAME))
                .expect("the fixture opens");
        let expert = "model.language_model.layers.0.mlp.experts.0.gate_proj".to_owned();
        for (rank, world_size) in [(0, 1), (0, 2), (1, 2), (0, 4), (3, 4)] {
            let layout = TpLayout { rank, world_size };
            let out = tempfile::tempdir().expect("an output directory");
            let converted = convert_rank(
                &source,
                &config,
                layout,
                checkpoint.path(),
                out.path(),
                "fixture",
            )
            .expect("the rank converts");
            assert_eq!(
                converted.path.file_name().and_then(std::ffi::OsStr::to_str),
                Some(rank_file_name(rank, world_size).as_str())
            );
            let index = ConvertedIndex::verify(&converted.path, layout).expect("it verifies");
            assert_eq!((index.rank, index.world), (rank, world_size));
            assert_eq!(
                find_converted_rank_file(out.path(), rank, world_size).as_deref(),
                Some(converted.path.as_path())
            );

            // The same walk, as the rank's load runs it, places every slot the file holds.
            let shard = Shard::of(layout);
            let mut replay = Plan::default();
            let _ = build_shard(
                &Uploader::planning(&mut replay),
                checkpoint.path(),
                &source,
                &config,
                shard,
            )
            .expect("the walk replays");
            assert_eq!(placed(replay.plan.slots()), placed(&index.slots));
            assert_eq!(replay.scalars, index.extra.nvfp4_scalars);

            // An expert's file slot is its share padded to whole MMA tiles, not the whole matrix.
            let gate = index
                .slots
                .iter()
                .find(|slot| slot.key == expert.clone() + ".weight")
                .expect("the walk uploads the expert's gate");
            let width = shard.expert_width(config.moe_intermediate_size);
            assert_eq!(gate.bytes, width * config.hidden_size / 2);

            // The file says which rank it is, and a load that is not that rank refuses it.
            let other = TpLayout {
                rank: (rank + 1) % world_size,
                world_size,
            };
            if other != layout {
                let error = ConvertedIndex::read(&converted.path, other)
                    .expect_err("another rank's file is not this rank's");
                assert!(error.contains("rank") || error.contains("world"), "{error}");
            }
        }
    }

    #[test]
    fn a_world_the_model_does_not_split_over_is_refused() {
        let config = tensor_parallel_config();
        let checkpoint = tempfile::tempdir().expect("a temporary directory");
        write_synthetic_checkpoint(checkpoint.path(), &config, 7).expect("the fixture writes");
        let source =
            SafeTensorsSource::open(checkpoint.path(), &checkpoint.path().join(INDEX_NAME))
                .expect("the fixture opens");
        let out = tempfile::tempdir().expect("an output directory");
        let error = convert_rank(
            &source,
            &config,
            TpLayout {
                rank: 0,
                world_size: 3,
            },
            checkpoint.path(),
            out.path(),
            "fixture",
        )
        .expect_err("four heads do not split over three ranks");
        assert!(error.contains("do not split"), "{error}");
        let error = convert_rank(
            &source,
            &config,
            TpLayout {
                rank: 2,
                world_size: 2,
            },
            checkpoint.path(),
            out.path(),
            "fixture",
        )
        .expect_err("rank 2 is not a rank of 2");
        assert!(error.contains("not a rank"), "{error}");
    }
}
