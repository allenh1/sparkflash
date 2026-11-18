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

//! The bridge from the SenseNova typed configuration to the shared architecture.

use sparkflash_model::{Architecture, ConfigError};

const CARRIED_CONFIG: &str = include_str!("../config/sensenova_loop_9b.hf.json");

/// Parse a released config.json into the shared architecture.
///
/// # Errors
/// A malformed document or one the shared parser cannot describe.
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
    use sparkflash_model::architecture::{AttentionKind, Residual};

    #[test]
    fn the_shared_walk_reads_the_loop_and_scopes_mhc() {
        let architecture = architecture_from_carried().expect("the carried document parses");
        assert_eq!(architecture.family, "sensenova_loop");
        assert_eq!(architecture.layers.len(), 32);
        let looped = architecture.looped.as_ref().expect("the loop metadata");
        assert_eq!(looped.input_sublayers, 4);
        assert_eq!(looped.recurrent_sublayers, 24);
        assert_eq!(looped.output_sublayers, 4);
        assert_eq!(looped.loops, 2);
        assert_eq!(looped.units_per_layer, 1);
        assert!(looped.per_loop_kv);
        assert!(matches!(
            architecture.layers[0].attention.kind,
            AttentionKind::Gdn(_)
        ));
        assert!(matches!(
            architecture.layers[3].attention.kind,
            AttentionKind::Gqa { .. }
        ));
        assert!(matches!(architecture.layers[0].residual, Residual::Plain));
        assert!(matches!(
            architecture.layers[4].residual,
            Residual::Hyper(_)
        ));
        assert!(matches!(
            architecture.layers[27].residual,
            Residual::Hyper(_)
        ));
        assert!(matches!(architecture.layers[28].residual, Residual::Plain));
    }

    #[test]
    fn the_shared_plan_plans_the_released_checkpoint() {
        let architecture = architecture_from_carried().expect("the carried document parses");
        let plan = sparkflash_ops::engine_plan::EnginePlan::for_architecture(
            architecture,
            sparkflash_model::Target::Sm121,
            1,
            0,
        )
        .expect("the released plan plans");
        assert_eq!(plan.family(), "sensenova_loop");
        assert!(plan.boot_summary().contains("sensenova"));
    }

    #[test]
    fn the_kv_plan_replays_the_loop_caches_and_recurrent_state() {
        let architecture = architecture_from_carried().expect("the carried document parses");
        let plan = architecture.kv_plan(1, false).expect("the plan");
        // 14 full-attention visits x 2 * 4 heads * 256 * 2 bytes.
        assert_eq!(plan.per_token(), 14 * 2 * 4 * 256 * 2);
        // The convolution history and the f32 delta-rule state of 42 GDN visits.
        assert!(plan.fixed() > 84 * 1024 * 1024);
    }
}
