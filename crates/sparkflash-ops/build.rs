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

//! Generates the catalogue's shape rules from `kernels/instances.toml`.
//!
//! A kernel that takes its model shapes as instance arguments is only built for the shapes the
//! instance list names. The catalogue must claim exactly those shapes: a hand-written
//! `hidden_equals(5120)` can claim a width that was never compiled, or hide one that was. Every
//! instance carrying a `catalogue = "<key>"` row contributes its shapes to that key's rule, and
//! the values become equalities or the bounds the source states. The result is emitted as
//! [ShapeConstraint] constants that `src/parts.rs` uses directly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

/// The instance list, relative to this crate.
const INSTANCES: &str = "../../kernels/instances.toml";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={INSTANCES}");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR is set for a build script");
    let generated = generate();
    std::fs::write(Path::new(&out).join("kernel_instances.rs"), generated)
        .expect("write the generated catalogue shapes");
}

fn generate() -> String {
    let text = std::fs::read_to_string(INSTANCES)
        .unwrap_or_else(|error| panic!("sparkflash-ops: cannot read {INSTANCES}: {error}"));
    let document: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("sparkflash-ops: {INSTANCES} is not TOML: {error}"));
    assert_eq!(
        document.get("version").and_then(toml::Value::as_integer),
        Some(1),
        "sparkflash-ops: {INSTANCES} must carry version = 1"
    );
    let rows = document
        .get("instance")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("sparkflash-ops: {INSTANCES} has no [[instance]] tables"));

    // The router/routed convention: each key's exact compiled (hidden, experts, top_k) set.
    let mut shapes: BTreeMap<String, BTreeSet<(u64, u64, u64)>> = BTreeMap::new();
    // The dimensions an instance names, per key, and the values the build produced.
    let mut keys: BTreeSet<String> = BTreeSet::new();
    let mut hidden: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut intermediate: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut experts: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut top_k: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut streams: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut claims_intermediate: BTreeMap<String, bool> = BTreeMap::new();
    let mut claims_hidden: BTreeMap<String, bool> = BTreeMap::new();
    let mut dims: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut max_top_k: BTreeMap<String, u64> = BTreeMap::new();
    let mut intermediate_multiple: BTreeMap<String, u64> = BTreeMap::new();
    for row in rows {
        let table = row
            .as_table()
            .unwrap_or_else(|| panic!("sparkflash-ops: an instance is not a table"));
        let positive = |key: &str| -> Option<u64> {
            table
                .get(key)
                .and_then(toml::Value::as_integer)
                .filter(|value| *value > 0)
                .map(|value| value as u64)
        };
        let Some(key) = table.get("catalogue").and_then(toml::Value::as_str) else {
            continue;
        };
        let key = key.to_owned();
        keys.insert(key.clone());

        let width = positive("hidden")
            .unwrap_or_else(|| panic!("sparkflash-ops: {key} has no positive integer hidden"));
        hidden.entry(key.clone()).or_default().insert(width);

        // The drafter's key shares its routed triple with the routed-expert instances; the
        // drafter rows themselves contribute only the hidden width they compile against.
        if let Some(multiple) = table.get("catalogue_dims").and_then(toml::Value::as_array) {
            let names: BTreeSet<String> = multiple
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .unwrap_or_else(|| {
                            panic!("sparkflash-ops: {key} has a non-string catalogue dimension")
                        })
                        .to_owned()
                })
                .collect();
            assert!(
                names.iter().all(|name| {
                    matches!(
                        name.as_str(),
                        "hidden" | "intermediate" | "experts" | "top_k"
                    )
                }) && !names.is_empty(),
                "sparkflash-ops: {key} names catalogue dimensions outside hidden/intermediate/experts/top_k"
            );
            let slot = dims.entry(key.clone()).or_default();
            assert!(
                slot.is_empty() || *slot == names,
                "sparkflash-ops: every {key} instance must name the same catalogue dimensions"
            );
            *slot = names;
        }

        let claims = table
            .get("catalogue_intermediate")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        *claims_intermediate.entry(key.clone()).or_insert(true) &= claims;
        // A runtime-width body (the quantisers, which take cols as a launch word) has no compiled
        // hidden shape to claim; its instances say so and the generated rule stays open.
        let claims_hidden_flag = table
            .get("catalogue_hidden")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        *claims_hidden.entry(key.clone()).or_insert(true) &= claims_hidden_flag;

        if let Some(width) = table.get("intermediate").and_then(toml::Value::as_integer) {
            assert!(
                width > 0,
                "sparkflash-ops: {key} names a zero intermediate width"
            );
            intermediate
                .entry(key.clone())
                .or_default()
                .insert(width as u64);
        }
        if let Some(count) = positive("hc") {
            streams.entry(key.clone()).or_default().insert(count);
        }
        if let Some(value) = positive("experts") {
            experts.entry(key.clone()).or_default().insert(value);
        }
        if let Some(value) = positive("top_k") {
            top_k.entry(key.clone()).or_default().insert(value);
        }
        // The router/routed convention inserts the triple; a key with catalogue_dims states its
        // own dimensions, and the drafter has no triple of its own.
        let family = table
            .get("family")
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        if !dims.contains_key(&key)
            && family != "drafter"
            && family != "seams"
            && let (Some(e), Some(k)) = (positive("experts"), positive("top_k"))
        {
            shapes.entry(key.clone()).or_default().insert((width, e, k));
        }
        if let Some(multiple) = table.get("intermediate_multiple") {
            let multiple = multiple
                .as_integer()
                .filter(|value| *value > 0)
                .unwrap_or_else(|| {
                    panic!("sparkflash-ops: {key} names a non-positive intermediate_multiple")
                }) as u64;
            let slot = intermediate_multiple.entry(key.clone()).or_insert(multiple);
            assert_eq!(
                *slot, multiple,
                "sparkflash-ops: every {key} instance must name the same intermediate_multiple"
            );
        }
        if let Some(maximum) = table.get("max_top_k") {
            let maximum = maximum
                .as_integer()
                .filter(|value| *value > 0)
                .unwrap_or_else(|| panic!("sparkflash-ops: {key} names a non-positive max_top_k"))
                as u64;
            let slot = max_top_k.entry(key.clone()).or_insert(maximum);
            assert_eq!(
                *slot, maximum,
                "sparkflash-ops: every {key} instance must name the same max_top_k"
            );
        }
    }
    assert!(
        !keys.is_empty(),
        "sparkflash-ops: no instance in {INSTANCES} names a catalogue key"
    );

    let mut generated = String::from(
        "// Generated by sparkflash-ops/build.rs from kernels/instances.toml. Do not edit.\n\
         // Each constant is the exact shape set the build produced for that catalogue key.\n",
    );
    for key in &keys {
        let name = format!("CATALOGUE_{}", key.to_ascii_uppercase().replace('-', "_"));
        let unique = |values: &BTreeMap<String, BTreeSet<u64>>| -> Option<u64> {
            let set = values.get(key)?;
            (set.len() == 1).then(|| *set.iter().next().expect("one value"))
        };
        let mut parts: Vec<String> = Vec::new();
        if let Some(names) = dims.get(key) {
            // The family names exactly the dimensions its sources fix, and the values the build
            // produced. A dimension the body takes at run time is named only with its bound.
            if names.contains("hidden")
                && let Some(width) = unique(&hidden)
            {
                parts.push(format!(".hidden_equals({width})"));
            }
            if let Some(multiple) = intermediate_multiple.get(key) {
                parts.push(format!(".intermediate_multiple({multiple})"));
            } else if names.contains("intermediate")
                && claims_intermediate.get(key).copied().unwrap_or(false)
                && let Some(width) = unique(&intermediate)
            {
                parts.push(format!(".intermediate_equals({width})"));
            }
            if names.contains("experts")
                && let Some(count) = unique(&experts)
            {
                parts.push(format!(".experts_equals({count})"));
            }
            if names.contains("top_k")
                && let Some(count) = unique(&top_k)
            {
                parts.push(format!(".top_k_equals({count})"));
            }
            if let Some(maximum) = max_top_k.get(key) {
                parts.push(format!(".max_top_k({maximum})"));
            }
        } else {
            if claims_hidden.get(key).copied().unwrap_or(true)
                && let Some(width) = unique(&hidden)
            {
                parts.push(format!(".hidden_equals({width})"));
            }
            if let Some(multiple) = intermediate_multiple.get(key) {
                parts.push(format!(".intermediate_multiple({multiple})"));
            } else if claims_intermediate.get(key).copied().unwrap_or(false)
                && let Some(width) = unique(&intermediate)
            {
                parts.push(format!(".intermediate_equals({width})"));
            }
            if let Some(count) = unique(&streams) {
                parts.push(format!(".streams_between({count}, {count})"));
            }
            if let Some(triples) = shapes.get(key)
                && !triples.is_empty()
            {
                let list = triples
                    .iter()
                    .map(|(h, e, k)| format!("({h}usize, {e}usize, {k}usize)"))
                    .collect::<Vec<_>>()
                    .join(", ");
                parts.push(format!(".routed_shapes(&[{list}])"));
            }
        }
        let expression = if parts.is_empty() {
            "ShapeConstraint::ANY".to_owned()
        } else {
            format!("ShapeConstraint::ANY\n    {}", parts.join("\n    "))
        };
        let _ = writeln!(
            generated,
            "#[allow(dead_code)] // a generated rule exists for every catalogue key, used or not"
        );
        let _ = writeln!(
            generated,
            "pub(crate) const {name}: ShapeConstraint = {expression};"
        );
    }
    generated
}
