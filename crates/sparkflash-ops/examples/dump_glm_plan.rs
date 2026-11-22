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

//! Dump the checkpoint plan of the carried GLM config at TP 1, 2 and 4.
//!
//! This is the reference the layout equality test freezes before the family planner is deleted.

use sparkflash_model::{Architecture, TpPlan};
use sparkflash_ops::ArchitectureTensorPlan;

fn main() {
    let path = std::env::args().nth(1).expect("a config.json path");
    let text = std::fs::read_to_string(&path).expect("the config");
    let architecture = Architecture::from_json(&text).expect("a parsable config");
    let mut out = serde_json::Map::new();
    for world in [1usize, 2, 4] {
        let tp = TpPlan::for_architecture(&architecture, world, 0).expect("a TP plan");
        let plan = ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the checkpoint plan");
        let mut tensors = Vec::new();
        for index in 0..plan.len() {
            let spec = plan.spec_at(index);
            let rank = plan.rank_spec_at(index);
            tensors.push(serde_json::json!({
                "name": spec.name,
                "dtype": format!("{:?}", spec.dtype),
                "shape": spec.shape,
                "rank_shape": rank.shape,
                "role": format!("{:?}", plan.plan().role(index)),
                "run": plan.rank_run_at(index).map(|run| serde_json::json!({
                    "offset": run.offset,
                    "count": run.count,
                    "total": run.total,
                })),
            }));
        }
        out.insert(format!("tp{world}"), serde_json::Value::Array(tensors));
    }
    println!("{}", serde_json::to_string_pretty(&out).expect("JSON"));
}
