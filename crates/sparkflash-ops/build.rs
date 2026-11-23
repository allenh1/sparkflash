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
//! instance carrying a `catalogue = "<key>"` row contributes its `(hidden, experts, top_k)` to
//! that key's rule, and the common `hidden`/`intermediate` values become equalities. The result
//! is emitted as [ShapeConstraint] constants that `src/parts.rs` uses directly.

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

    let mut keys: BTreeMap<String, BTreeSet<(u64, u64, u64)>> = BTreeMap::new();
    let mut hidden: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut intermediate: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut experts: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut top_k: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut claims_intermediate: BTreeMap<String, bool> = BTreeMap::new();
    // The rows family names exactly which dimensions a catalogue rule constrains. A key without
    // this field keeps the router/routed rule (every compiled (hidden, experts, top-k) shape).
    let mut dims: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut max_top_k: BTreeMap<String, u64> = BTreeMap::new();
    let mut intermediate_multiple: BTreeMap<String, u64> = BTreeMap::new();
    for row in rows {
        let table = row
            .as_table()
            .unwrap_or_else(|| panic!("sparkflash-ops: an instance is not a table"));
        let get = |key: &str| -> u64 {
            table
                .get(key)
                .and_then(toml::Value::as_integer)
                .filter(|value| *value > 0)
                .unwrap_or_else(|| {
                    panic!("sparkflash-ops: an instance has no positive integer {key}")
                }) as u64
        };
        let Some(key) = table.get("catalogue").and_then(toml::Value::as_str) else {
            continue;
        };
        let shape = (get("hidden"), get("experts"), get("top_k"));
        keys.entry(key.to_owned()).or_default().insert(shape);
        hidden.entry(key.to_owned()).or_default().insert(shape.0);
        experts.entry(key.to_owned()).or_default().insert(shape.1);
        top_k.entry(key.to_owned()).or_default().insert(shape.2);
        if let Some(list) = table.get("catalogue_dims").and_then(toml::Value::as_array) {
            let names: BTreeSet<String> = list
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
            let slot = dims.entry(key.to_owned()).or_default();
            assert!(
                slot.is_empty() || *slot == names,
                "sparkflash-ops: every {key} instance must name the same catalogue dimensions"
            );
            *slot = names;
        }
        if let Some(multiple) = table.get("intermediate_multiple") {
            let multiple = multiple
                .as_integer()
                .filter(|value| *value > 0)
                .unwrap_or_else(|| {
                    panic!("sparkflash-ops: {key} names a non-positive intermediate_multiple")
                }) as u64;
            let slot = intermediate_multiple
                .entry(key.to_owned())
                .or_insert(multiple);
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
            let slot = max_top_k.entry(key.to_owned()).or_insert(maximum);
            assert_eq!(
                *slot, maximum,
                "sparkflash-ops: every {key} instance must name the same max_top_k"
            );
        }
        let claims = table
            .get("catalogue_intermediate")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        *claims_intermediate.entry(key.to_owned()).or_insert(true) &= claims;
        if let Some(width) = table.get("intermediate").and_then(toml::Value::as_integer) {
            assert!(
                width > 0,
                "sparkflash-ops: {key} names a zero intermediate width"
            );
            intermediate
                .entry(key.to_owned())
                .or_default()
                .insert(width as u64);
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
    for (key, shapes) in &keys {
        let name = format!("CATALOGUE_{}", key.to_ascii_uppercase().replace('-', "_"));
        let mut shape_list = String::new();
        for (index, (h, e, k)) in shapes.iter().enumerate() {
            if index > 0 {
                shape_list.push_str(", ");
            }
            let _ = write!(shape_list, "({h}usize, {e}usize, {k}usize)");
        }
        let _ = writeln!(
            generated,
            "#[allow(dead_code)] // a generated rule exists for every catalogue key, used or not"
        );
        let _ = writeln!(
            generated,
            "pub(crate) const {name}: ShapeConstraint = ShapeConstraint::ANY"
        );
        if let Some(names) = dims.get(key) {
            // The rows family: claim exactly the dimensions the source fixes and the values the
            // build produced. A dimension the source takes at run time is not named here unless the
            // instance list gives it a bound (max_top_k), which is what the source states.
            let one = |values: &BTreeMap<String, BTreeSet<u64>>, name: &str| -> Option<u64> {
                let set = values.get(key)?;
                (names.contains(name) && set.len() == 1)
                    .then(|| *set.iter().next().expect("one value"))
            };
            if let Some(value) = one(&hidden, "hidden") {
                let _ = writeln!(generated, "    .hidden_equals({value})");
            }
            if let Some(multiple) = intermediate_multiple.get(key) {
                let _ = writeln!(generated, "    .intermediate_multiple({multiple})");
            } else if let Some(value) = one(&intermediate, "intermediate")
                && claims_intermediate.get(key).copied().unwrap_or(false)
            {
                let _ = writeln!(generated, "    .intermediate_equals({value})");
            }
            if let Some(value) = one(&experts, "experts") {
                let _ = writeln!(generated, "    .experts_equals({value})");
            }
            if let Some(value) = one(&top_k, "top_k") {
                let _ = writeln!(generated, "    .top_k_equals({value})");
            }
            if let Some(value) = max_top_k.get(key) {
                let _ = writeln!(generated, "    .max_top_k({value})");
            }
            let _ = writeln!(generated, ";");
        } else {
            if let Some(widths) = hidden.get(key)
                && widths.len() == 1
            {
                let _ = writeln!(
                    generated,
                    "    .hidden_equals({})",
                    widths.iter().next().expect("one width")
                );
            }
            if let Some(widths) = intermediate.get(key)
                && widths.len() == 1
                && claims_intermediate.get(key).copied().unwrap_or(false)
            {
                let _ = writeln!(
                    generated,
                    "    .intermediate_equals({})",
                    widths.iter().next().expect("one width")
                );
            }
            let _ = writeln!(generated, "    .routed_shapes(&[{shape_list}]);");
        }
    }
    generated
}
