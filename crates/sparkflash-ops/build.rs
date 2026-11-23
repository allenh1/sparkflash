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
    let mut claims_intermediate: BTreeMap<String, bool> = BTreeMap::new();
    // The attention/selection vocabulary: the shape fields those parts expose, one set per key.
    let mut families: BTreeMap<String, String> = BTreeMap::new();
    let mut head_dim: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut index_topk: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut index_heads: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut positions: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
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
        let family = table
            .get("family")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("sparkflash-ops: instance {key} has no family"));
        if let Some(known) = families.insert(key.to_owned(), family.to_owned()) {
            assert_eq!(
                known, family,
                "sparkflash-ops: catalogue key {key} is used by two shape vocabularies"
            );
        }
        if family == "router" || family == "routed" {
            let shape = (get("hidden"), get("experts"), get("top_k"));
            keys.entry(key.to_owned()).or_default().insert(shape);
            hidden.entry(key.to_owned()).or_default().insert(shape.0);
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
            continue;
        }
        // MLA and selection instances: only the fields a ShapeConstraint can name. `head_dim` is
        // the MLA head width; the indexer's own head width is `index_head_dim`.
        keys.entry(key.to_owned()).or_default();
        hidden
            .entry(key.to_owned())
            .or_default()
            .insert(get("hidden"));
        let width_key = if family == "select" {
            "index_head_dim"
        } else {
            "head_dim"
        };
        head_dim
            .entry(key.to_owned())
            .or_default()
            .insert(get(width_key));
        if family == "select" {
            index_topk
                .entry(key.to_owned())
                .or_default()
                .insert(get("index_topk"));
            index_heads
                .entry(key.to_owned())
                .or_default()
                .insert(get("index_heads"));
        }
        positions
            .entry(key.to_owned())
            .or_default()
            .insert(get("positions"));
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
        let family = families.get(key).map(String::as_str).unwrap_or("router");
        let routed = family == "router" || family == "routed";
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
        if routed {
            let _ = writeln!(generated, "    .routed_shapes(&[{shape_list}]);");
            continue;
        }
        // An MLA or selection rule names the exact compiled head width and, for the indexer, the
        // exact candidate count and head ceiling the bodies are built for. A second shape moves
        // those numbers only by adding an instance.
        if let Some(widths) = head_dim.get(key)
            && widths.len() == 1
        {
            let _ = writeln!(
                generated,
                "    .head_dim_equals({})",
                widths.iter().next().expect("one width")
            );
        }
        if let Some(values) = index_topk.get(key)
            && values.len() == 1
        {
            let _ = writeln!(
                generated,
                "    .top_k_equals({})",
                values.iter().next().expect("one value")
            );
        }
        if let Some(values) = index_heads.get(key)
            && values.len() == 1
        {
            let _ = writeln!(
                generated,
                "    .max_heads({})",
                values.iter().next().expect("one value")
            );
        }
        if let Some(values) = positions.get(key)
            && values.len() == 1
        {
            let _ = writeln!(
                generated,
                "    .positions(0, {})",
                values.iter().next().expect("one value")
            );
        }
        generated.push_str(";\n");
    }
    generated
}
