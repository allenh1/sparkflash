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

//! `kernels/instances.toml` is the build's one list of kernel instances.
//!
//! The build script validates the list while it runs; this test holds it against the other three
//! descriptions a kernel appears in: the source under kernels/, the `PROBED_KERNELS` table in
//! build.rs and the compiled baseline. It needs no nvcc, so a CPU host checks the wiring that the
//! CUDA host's byte comparison assumes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use sparkflash_kernels::baseline::pins;

/// One [instance] table, as this test needs it.
struct Instance {
    source: String,
    entry: String,
    cubin: String,
    family: String,
    catalogue: Option<String>,
    shapes: BTreeMap<String, u64>,
}

fn instances() -> Vec<Instance> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = crate_dir.join("../../kernels/instances.toml");
    let text = std::fs::read_to_string(&path).expect("kernels/instances.toml is readable");
    let document: toml::Value = toml::from_str(&text).expect("kernels/instances.toml is TOML");
    assert_eq!(
        document.get("version").and_then(toml::Value::as_integer),
        Some(1),
        "the instance list carries version = 1"
    );
    document
        .get("instance")
        .and_then(toml::Value::as_array)
        .expect("[[instance]] tables")
        .iter()
        .map(|row| {
            let table = row.as_table().expect("an instance is a table");
            let string = |key: &str| {
                table
                    .get(key)
                    .and_then(toml::Value::as_str)
                    .unwrap_or_else(|| panic!("an instance has no string {key}"))
                    .to_owned()
            };
            let mut shapes = BTreeMap::new();
            for key in [
                "hidden",
                "experts",
                "top_k",
                "intermediate",
                "padded",
                "vocab_per_rank",
                "ranks",
            ] {
                if let Some(value) = table.get(key).and_then(toml::Value::as_integer) {
                    assert!(value > 0, "{key} is positive");
                    shapes.insert(key.to_owned(), value as u64);
                }
            }
            Instance {
                source: string("source"),
                entry: string("entry"),
                cubin: table
                    .get("cubin")
                    .and_then(toml::Value::as_str)
                    .map_or_else(
                        || {
                            string("source")
                                .trim_start_matches("kernels/")
                                .trim_end_matches(".cu")
                                .to_owned()
                        },
                        str::to_owned,
                    ),
                family: string("family"),
                catalogue: table
                    .get("catalogue")
                    .and_then(toml::Value::as_str)
                    .map(str::to_owned),
                shapes,
            }
        })
        .collect()
}

#[test]
fn every_instance_names_a_source_and_an_entry_the_source_carries() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kernels = crate_dir.join("../../kernels");
    let mut entries = BTreeSet::new();
    for instance in instances() {
        assert!(
            entries.insert(instance.entry.clone()),
            "{} is listed twice",
            instance.entry
        );
        let path = kernels.join(&instance.source);
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        assert!(
            body.contains(&instance.entry),
            "{} does not appear in {}",
            instance.entry,
            path.display()
        );
        assert!(
            ["router", "routed", "head"].contains(&instance.family.as_str()),
            "{} names family {:?}",
            instance.entry,
            instance.family
        );
        for key in match instance.family.as_str() {
            "router" => &["hidden", "experts", "top_k"][..],
            "routed" => &["hidden", "intermediate", "padded", "experts", "top_k"][..],
            "head" => &["hidden", "vocab_per_rank", "ranks"][..],
            _ => unreachable!("the family was checked above"),
        } {
            assert!(
                instance.shapes.contains_key(*key),
                "{} ({}) has no {key}",
                instance.entry,
                instance.family
            );
        }
    }
}

#[test]
fn released_instances_are_the_probed_and_pinned_ones_and_the_proof_instance_is_neither() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let build = std::fs::read_to_string(crate_dir.join("build.rs")).expect("build.rs is readable");
    let baseline: BTreeSet<String> = pins()
        .expect("the baseline parses")
        .into_iter()
        .map(|pin| pin.name)
        .collect();

    for instance in instances() {
        let source_name = instance.source.trim_start_matches("kernels/");
        assert!(
            build.contains(&format!("\"{source_name}\"")),
            "{} is not in build.rs PROBED_KERNELS",
            instance.source
        );
        // A released instance's cubin is its source stem; the proof instance deliberately is not.
        let released = instance.cubin == source_name.trim_end_matches(".cu");
        assert_eq!(
            baseline.contains(&instance.cubin),
            released,
            "{}: baseline membership must match whether the cubin is the released stem",
            instance.cubin
        );
    }
}

#[test]
fn the_router_family_has_its_released_shapes_and_a_second_shape() {
    let rows = instances();
    let routers: Vec<&Instance> = rows.iter().filter(|row| row.family == "router").collect();
    let released = routers
        .iter()
        .find(|row| row.cubin == "opt_router_sm121")
        .expect("the released router");
    assert_eq!(released.shapes["hidden"], 5120);
    assert_eq!(released.shapes["experts"], 384);
    assert_eq!(released.shapes["top_k"], 6);

    let second = routers
        .iter()
        .find(|row| row.entry == "opt_router_4096e256k8_sm121")
        .expect("the second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_eq!(second.shapes["experts"], 256);
    assert_eq!(second.shapes["top_k"], 8);
    assert_ne!(second.cubin, released.cubin, "the proof has its own cubin");

    // The DSpark 128/3 router is a second released shape in the released cubin; both are built
    // from the same body, and the released symbol names are unchanged.
    assert!(routers.iter().any(|row| {
        row.cubin == "opt_router_sm121"
            && row.shapes["experts"] == 128
            && row.shapes["top_k"] == 3
            && row.entry == "opt_router_e128k3_sm121"
    }));

    let keys: BTreeSet<&str> = rows
        .iter()
        .filter_map(|row| row.catalogue.as_deref())
        .collect();
    for key in ["deepseek-router", "deepseek-moe", "deepseek-mtp"] {
        assert!(
            keys.contains(key),
            "the catalogue key {key} has no instance"
        );
    }
}

#[test]
fn the_head_family_has_its_released_shape_and_a_second_shape() {
    let rows = instances();
    let heads: Vec<&Instance> = rows.iter().filter(|row| row.family == "head").collect();

    // Both released entries of each source carry the released one-rank shape and share the
    // source's cubin, so the released symbol sets are whole.
    for entry in [
        "cert_lm_head_sm121",
        "cert_lm_head_batched_sm121",
        "lm_head_reduce_sm121",
        "lm_head_reduce_batched_sm121",
    ] {
        let row = heads
            .iter()
            .find(|row| row.entry == entry)
            .unwrap_or_else(|| panic!("{entry} has no head instance"));
        assert_eq!(row.shapes["hidden"], 5120, "{entry}");
        assert_eq!(row.shapes["vocab_per_rank"], 32320, "{entry}");
        assert_eq!(row.shapes["ranks"], 4, "{entry}");
        assert!(row.catalogue.as_deref() == Some("deepseek-head"), "{entry}");
    }
    for source in ["cert_lm_head_sm121.cu", "lm_head_reduce_sm121.cu"] {
        let cubins: BTreeSet<&str> = heads
            .iter()
            .filter(|row| row.source.ends_with(source) && row.catalogue.is_some())
            .map(|row| row.cubin.as_str())
            .collect();
        assert_eq!(cubins.len(), 1, "{source} entries share one released cubin");
    }

    let second = heads
        .iter()
        .find(|row| row.entry == "cert_lm_head_4096v16384_sm121")
        .expect("the head second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_eq!(second.shapes["vocab_per_rank"], 16384);
    assert_eq!(second.shapes["ranks"], 4);
    assert_eq!(second.cubin, "cert_lm_head_4096v16384_sm121");
    assert!(
        second.catalogue.is_none(),
        "the proof shape is not a released catalogue claim"
    );
    assert!(
        !heads
            .iter()
            .any(|row| row.entry == second.entry && row.cubin == "cert_lm_head_sm121"),
        "the proof has its own cubin"
    );
}
