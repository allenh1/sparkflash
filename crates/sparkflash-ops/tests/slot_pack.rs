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

//! The shared rank-file converter over the architecture plan.

use serde::{Deserialize, Serialize};
use sparkflash_model::TpPlan;
use sparkflash_model::architecture::Architecture;
use sparkflash_ops::slot_pack::SlotPack;
use sparkflash_ops::tensor_plan::ArchitectureTensorPlan;
use sparkflash_weights::rank_file::{SlotIndex, SlotRecords};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{TensorSpec, WeightError, WeightSource};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Extra {
    label: u32,
}

impl SlotRecords for Extra {
    const MODEL: &'static str = "slot-pack-test";
}

struct Bytes(Vec<TensorSpec>);

impl WeightSource for Bytes {
    fn names(&self) -> Vec<String> {
        self.0.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0
            .iter()
            .find(|spec| spec.name == name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.spec(name)?;
        let length = spec.storage_bytes() as usize;
        Ok((0..length).map(|index| (index % 251) as u8).collect())
    }
}

fn architecture() -> Architecture {
    Architecture::from_json(
        r#"{"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"vocab_size":1024,"num_hidden_layers":1,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"layer_types":["linear_attention"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"hc_count":4,"hc_lowrank":320,"indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}"#,
    )
    .expect("the fixture parses")
}

#[test]
fn a_slot_pack_round_trips_every_planned_tensor() {
    let tp = TpPlan::new(1, 0).expect("one rank");
    let plan =
        ArchitectureTensorPlan::served_and_mtp(&architecture(), tp).expect("the plan builds");
    let source = Bytes(plan.specs().to_vec());
    let pack = SlotPack::new(&plan, Extra { label: 7 }).expect("a TP1 plan packs");
    assert_eq!(pack.slots().len(), plan.len());
    assert_eq!(
        pack.slots()[0].key,
        "model.language_model.embed_tokens.weight"
    );

    let out = tempfile::tempdir().expect("an output directory");
    let layout = TpLayout {
        rank: 0,
        world_size: 1,
    };
    let written = pack
        .write(&source, out.path(), layout, "fixture")
        .expect("the pack writes");
    let index = SlotIndex::<Extra>::verify(&written.path, layout).expect("it verifies");
    assert_eq!(index.extra, Extra { label: 7 });
    assert_eq!(index.slots.len(), pack.slots().len());
    for (slot, planned) in index.slots.iter().zip(pack.slots()) {
        assert_eq!(slot.key, planned.key);
        assert_eq!(slot.bytes, planned.bytes());
    }
    let (preamble, read_back) =
        SlotPack::<Extra>::read(&written.path, layout).expect("the index reads back");
    assert_eq!(read_back.extra, Extra { label: 7 });
    let data = std::fs::read(&written.path).expect("the file reads");
    let first = &read_back.slots[0];
    let at = preamble.data_offset as usize + first.offset;
    assert_eq!(
        &data[at..at + first.bytes],
        source
            .read(&first.key)
            .expect("the source reads")
            .as_slice()
    );
}

#[test]
fn a_slot_pack_refuses_a_world_the_arm_walk_owns() {
    let tp = TpPlan::new(2, 0).expect("rank 0 of two");
    let plan = ArchitectureTensorPlan::served(&architecture(), tp).expect("the TP2 plan builds");
    let error =
        SlotPack::new(&plan, Extra { label: 7 }).expect_err("the plan-order converter is TP1 only");
    assert!(
        error.contains("TP1 plan-order") && error.contains("TP2/rank0"),
        "{error}"
    );
}
