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

//! The bridge from Dory's typed configuration to the shared architecture.

use sparkflash_model::{Architecture, ConfigError};

const CARRIED_CONFIG: &str = include_str!("../config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json");

/// Parse a released Dory config.json into the shared architecture.
///
/// # Errors
/// A malformed document, or one the shared parser cannot describe.
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    Architecture::from_json(text)
}

/// The shared architecture of the released checkpoint carried in this crate.
///
/// # Errors
/// As [architecture_from_json].
pub fn architecture_from_carried() -> Result<Architecture, ConfigError> {
    architecture_from_json(CARRIED_CONFIG)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::architecture::AttentionKind;

    #[test]
    fn the_shared_walk_reads_36_collapsed_blocks_and_the_loop() {
        let architecture = architecture_from_carried().expect("the released document parses");
        assert_eq!(architecture.family, "dory");
        assert_eq!(architecture.layers.len(), 36);
        let looped = architecture.looped.as_ref().expect("the loop metadata");
        assert_eq!(looped.input_sublayers, 16);
        assert_eq!(looped.recurrent_sublayers, 40);
        assert_eq!(looped.output_sublayers, 16);
        assert_eq!(looped.loops, 2);
        assert!(looped.per_loop_kv);
        assert!(matches!(
            architecture.layers[0].attention.kind,
            AttentionKind::Gqa {
                output_gate: true,
                ..
            }
        ));
        assert_eq!(architecture.layers[0].attention.window, Some(1024));
        assert_eq!(architecture.layers[3].attention.window, None);
        assert_eq!(architecture.layers[3].attention.rope.theta, 10_000_000.0);
    }

    #[test]
    fn the_kv_plan_counts_the_replayed_recurrent_slice() {
        let architecture = architecture_from_carried().expect("the released document parses");
        // BF16 K/V: 36 attention layers at 2 * 4 heads * 256 values * 2 bytes, plus the
        // 20 recurrent layers once more for the second loop.
        let plan = architecture.kv_plan(1, false).expect("the plan");
        assert_eq!(plan.per_token(), (36 + 20) * 2 * 4 * 256 * 2);
        assert_eq!(plan.fixed(), 0);
    }
}
