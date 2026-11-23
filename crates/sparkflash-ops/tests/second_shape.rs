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

//! The second shape plans at four ranks.
//!
//! The fixture is a DeepSeek-shaped configuration whose width, heads, experts, top-k, layer
//! count, compression ratios, Engram layers and drafter taps all differ from the released model.
//! Every part the released plan resolves must resolve here too: a shape left in code shows up as
//! a part instance with no kernel, not as a device failure.

use sparkflash_model::Target;
use sparkflash_ops::EnginePlan;

const CONFIG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sparkflash-deepseek-forward/tests/fixtures/second-shape/config.json"
);

#[test]
fn the_second_shape_plans_at_four_ranks() {
    for rank in 0..4 {
        let plan = EnginePlan::from_path(CONFIG.as_ref(), Target::Sm121, 4, rank)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        assert!(!plan.tensors().is_empty(), "rank {rank} plans no tensors");
        assert!(
            !plan.symbols().is_empty(),
            "rank {rank} resolves no catalogue symbols"
        );
    }
}
