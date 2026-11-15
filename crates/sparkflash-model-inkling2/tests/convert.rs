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

//! Converted-weight loading: write a tiny rank file, read it back through [RankFileSource], and
//! serve the same logits from it as from the checkpoint the file was written from.
//!
//! The fixture is generated from the plan, so it holds every tensor the plan names with the
//! right dtype and shape. That makes the test a check on the loader's bookkeeping rather than on
//! a hand-written list of tensor names.

use std::collections::BTreeMap;
use std::sync::Arc;

use sparkflash_model::ModelDescriptor;
use sparkflash_model::TpPlan;
use sparkflash_model_inkling2::Inkling2HostBackend;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::convert::{RankFileSource, convert_rank, verify_rank};
use sparkflash_model_inkling2::forward::Inkling2Weights;
use sparkflash_model_inkling2::tensor::InklingTensorPlan;
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{TensorSpec, WeightError, WeightSource};

/// A source whose every tensor is the plan's shape filled with a name-and-index pattern.
struct Filled {
    specs: BTreeMap<String, TensorSpec>,
    data: BTreeMap<String, Vec<u8>>,
}

impl Filled {
    fn new(config: &Inkling2Config) -> Self {
        let plan = InklingTensorPlan::served(config, TpPlan::new(1, 0).expect("TP1"))
            .expect("the tiny plan");
        let mut specs = BTreeMap::new();
        let mut data = BTreeMap::new();
        for spec in plan.plan().specs() {
            let bytes = usize::try_from(spec.storage_bytes()).expect("the tiny tensor fits");
            let mut buffer = vec![0u8; bytes];
            let seed = spec.name.bytes().fold(1u64, |state, byte| {
                state.wrapping_mul(31).wrapping_add(u64::from(byte))
            });
            for (index, slot) in buffer.iter_mut().enumerate() {
                *slot = (seed.wrapping_add(index as u64 * 7) % 251) as u8;
            }
            specs.insert(spec.name.clone(), spec.clone());
            data.insert(spec.name.clone(), buffer);
        }
        Self { specs, data }
    }
}

impl WeightSource for Filled {
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
        self.data
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_string(),
            })
    }
}

fn layout() -> TpLayout {
    TpLayout {
        rank: 0,
        world_size: 1,
    }
}

/// A converted rank file read through [RankFileSource] must be byte-identical to the checkpoint
/// it was written from, for every tensor the plan names.
#[test]
fn a_rank_file_round_trips_every_planned_tensor() {
    let config = Inkling2Config::tiny();
    let source = Filled::new(&config);
    let directory = tempfile::tempdir().expect("a temporary directory");
    let written = convert_rank(&source, &config, layout(), directory.path(), "tiny")
        .expect("the rank file is written");
    assert!(written.slots > 0);

    let loaded = RankFileSource::open(&written.path, &config, layout()).expect("the file opens");
    for name in source.names() {
        let expected = source.read(&name).expect("the fixture holds it");
        let actual = loaded.read(&name).expect("the rank file holds it");
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name}: the rank file holds another length"
        );
        assert_eq!(actual, expected, "{name}: the rank file holds other bytes");
        assert_eq!(
            loaded.spec(&name).expect("the rank spec").shape,
            source.spec(&name).expect("the checkpoint spec").shape,
            "{name}: the rank spec is another shape"
        );
    }
    assert!(loaded.names().len() == source.names().len());

    // A paged read of the middle of a tensor is the same bytes the whole read has there.
    let name = "model.llm.embed.weight";
    let whole = source.read(name).expect("the fixture holds it");
    let middle = loaded
        .read_ranges(name, &[(4, 16), (32, 8)])
        .expect("the rank file supports ranges")
        .expect("the ranges read");
    assert_eq!(&middle[..16], &whole[4..20]);
    assert_eq!(&middle[16..], &whole[32..40]);
}

/// The device arm reads the data section whole and points each tensor at
/// [RankFileSource::tensor_span]: that span of the file must be the bytes a read of the tensor
/// returns, for every tensor the plan names.
#[test]
fn every_tensor_is_its_span_of_the_data_section() {
    use std::os::unix::fs::FileExt;

    let config = Inkling2Config::tiny();
    let source = Filled::new(&config);
    let directory = tempfile::tempdir().expect("a temporary directory");
    let written = convert_rank(&source, &config, layout(), directory.path(), "tiny")
        .expect("the rank file is written");
    let loaded = RankFileSource::open(&written.path, &config, layout()).expect("the file opens");
    let (path, data_offset, data_bytes) = loaded.data_section();
    assert_eq!(data_offset % 4096, 0, "a direct read starts on a page");
    assert_eq!(data_bytes % 4096, 0, "a direct read is whole pages");
    let file = std::fs::File::open(path).expect("the rank file opens");
    let mut section = vec![0u8; data_bytes];
    file.read_exact_at(&mut section, data_offset)
        .expect("the data section reads");
    for name in source.names() {
        let (start, bytes) = loaded.tensor_span(&name).expect("one span");
        assert_eq!(
            &section[start..start + bytes],
            loaded
                .read(&name)
                .expect("the rank file holds it")
                .as_slice(),
            "{name}: the span holds other bytes"
        );
    }
    assert!(loaded.tensor_span("model.llm.no_such.weight").is_err());
}

/// [verify_rank] compares the file against the checkpoint, not only against its own header.
#[test]
fn a_rank_file_verifies_against_the_checkpoint() {
    let config = Inkling2Config::tiny();
    let source = Filled::new(&config);
    let directory = tempfile::tempdir().expect("a temporary directory");
    let written = convert_rank(&source, &config, layout(), directory.path(), "tiny")
        .expect("the rank file is written");
    let checked =
        verify_rank(&written.path, &source, &config, layout()).expect("the file verifies");
    assert_eq!(checked.slots, written.slots);
    // The payload is every tensor's bytes; the file is those bytes plus the 256-byte alignment
    // between slots, so it must be the smaller of the two and never smaller than the plan.
    let plan =
        InklingTensorPlan::served(&config, TpPlan::new(1, 0).expect("TP1")).expect("the tiny plan");
    assert_eq!(checked.bytes, plan.rank_storage_bytes() as u64);
    assert!(written.data_bytes >= checked.bytes);
}

/// The first token served from converted weights is the first token served from the checkpoint:
/// the loader changes where the bytes come from, not what the forward computes.
#[test]
fn the_host_serves_the_same_logits_from_converted_weights() {
    let config = Inkling2Config::tiny();
    let source = Filled::new(&config);
    let directory = tempfile::tempdir().expect("a temporary directory");
    let written = convert_rank(&source, &config, layout(), directory.path(), "tiny")
        .expect("the rank file is written");
    let loaded = RankFileSource::open(&written.path, &config, layout()).expect("the file opens");

    let descriptor: ModelDescriptor = sparkflash_model::ModelDescriptor::inkling2_small();
    let from_checkpoint =
        Inkling2Weights::load(config.clone(), Arc::new(source)).expect("the checkpoint loads");
    let from_file = Inkling2Weights::load(config.clone(), Arc::new(loaded))
        .expect("the converted weights load");
    let mut left = sparkflash_model_inkling2::Inkling2State::new(&config);
    let mut right = sparkflash_model_inkling2::Inkling2State::new(&config);
    let first = from_checkpoint
        .forward_token(3, &mut left)
        .expect("the checkpoint forward");
    let second = from_file
        .forward_token(3, &mut right)
        .expect("the converted forward");
    assert_eq!(first, second, "the two sources must agree bit for bit");
    let _ = descriptor;
    let _ = Inkling2HostBackend::open;
}
