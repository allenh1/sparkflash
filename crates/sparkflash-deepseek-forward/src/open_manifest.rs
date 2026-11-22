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

//! The DeepSeek open manifest's model and tensor records.
//!
//! [`sparkflash_kernels::open_manifest`] owns the file and the kernel records, because the driver
//! is the only place that sees every cubin and symbol. This module adds the two things only the
//! forward knows: the model values it reads and the tensors this rank binds.
//!
//! The baseline records every field of the configuration the forward currently opens from, not
//! only the 39 fields it reads today. That is deliberate for the first baseline: a field the
//! forward starts reading later must already be in the manifest, or the first comparison after the
//! change would pass while the value's source moved. Milestone 2 B removes fields the plan does not
//! carry; the rule test's pinned `ModelConfig` counts are what make that move visible.

use std::collections::BTreeSet;

use serde_json::json;
use sparkflash_device::resident::ResidentPlan;
use sparkflash_engine::config::ModelConfig;

/// The schema every line this module writes carries.
pub const SCHEMA: &str = "sparkflash-open-manifest/1";

/// Records every field of the configuration the forward opens from, when the manifest is enabled.
pub fn record_model_config(config: &ModelConfig, rank: usize, world: usize) {
    if !sparkflash_kernels::open_manifest::enabled() {
        return;
    }
    let Ok(value) = serde_json::to_value(config) else {
        return;
    };
    let Some(fields) = value.as_object() else {
        return;
    };
    for (name, value) in fields {
        sparkflash_kernels::open_manifest::record(&json!({
            "schema": SCHEMA,
            "kind": "model",
            "name": name,
            "value": value,
            "rank": rank,
            "world": world,
        }));
    }
}

/// Records every plan shard this rank bound: stored shape, stored format, rank slice and bytes.
pub fn record_tensors(plan: &ResidentPlan, bound: &BTreeSet<String>, rank: usize, world: usize) {
    if !sparkflash_kernels::open_manifest::enabled() {
        eprintln!("open-manifest: record_tensors rank={rank} skipped (disabled)");
        return;
    }
    let shard_count = plan.shards().len();
    let mut matched = 0usize;
    for shard in plan.shards() {
        if !bound.contains(&shard.name) {
            continue;
        }
        matched += 1;
        sparkflash_kernels::open_manifest::record(&json!({
            "schema": SCHEMA,
            "kind": "tensor",
            "name": shard.name,
            "shape": shard.stored_shape,
            "format": shard.dtype,
            "rank": rank,
            "world": world,
            "rank_slice": shard.shard_shape,
            "shard_bytes": shard.shard_bytes,
            "group": shard.group.as_str(),
            "placement": shard.placement.as_str(),
        }));
    }
    eprintln!(
        "open-manifest: record_tensors rank={rank} shards={shard_count} bound={} matched={matched}",
        bound.len()
    );
}
