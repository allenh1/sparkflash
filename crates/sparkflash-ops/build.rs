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
    let mut streams: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut claims_intermediate: BTreeMap<String, bool> = BTreeMap::new();
    let mut claims_hidden: BTreeMap<String, bool> = BTreeMap::new();
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
        let hidden_width = positive("hidden")
            .unwrap_or_else(|| panic!("sparkflash-ops: {key} has no positive integer hidden"));
        hidden
            .entry(key.to_owned())
            .or_default()
            .insert(hidden_width);
        keys.entry(key.to_owned()).or_default();
        // The router and routed-expert keys are the only shapes with an expert count; the seam
        // keys carry a hidden width (and, for the hyper-connection body, a stream count) and must
        // not claim a routed shape that was never built.
        if let (Some(experts), Some(top_k)) = (positive("experts"), positive("top_k")) {
            keys.entry(key.to_owned())
                .or_default()
                .insert((hidden_width, experts, top_k));
        }
        if let Some(count) = positive("hc") {
            streams.entry(key.to_owned()).or_default().insert(count);
        }
        let claims = table
            .get("catalogue_intermediate")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        *claims_intermediate.entry(key.to_owned()).or_insert(true) &= claims;
        // A runtime-width body (the quantisers, which take cols as a launch word) has no compiled
        // hidden shape to claim; its instances say so and the generated rule stays open.
        let claims_hidden_flag = table
            .get("catalogue_hidden")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        *claims_hidden
            .entry(key.to_owned())
            .or_insert(true) &= claims_hidden_flag;
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
        let mut method = String::new();
        if let Some(widths) = hidden.get(key)
            && widths.len() == 1
            && claims_hidden.get(key).copied().unwrap_or(true)
        {
            let _ = write!(
                method,
                "    .hidden_equals({})\n",
                widths.iter().next().expect("one width")
            );
        }
        if let Some(widths) = intermediate.get(key)
            && widths.len() == 1
            && claims_intermediate.get(key).copied().unwrap_or(false)
        {
            let _ = write!(
                method,
                "    .intermediate_equals({})\n",
                widths.iter().next().expect("one width")
            );
        }
        if let Some(counts) = streams.get(key)
            && counts.len() == 1
        {
            let count = counts.iter().next().expect("one count");
            let _ = write!(method, "    .streams_between({count}, {count})\n");
        }
        if !shapes.is_empty() {
            let mut shape_list = String::new();
            for (index, (h, e, k)) in shapes.iter().enumerate() {
                if index > 0 {
                    shape_list.push_str(", ");
                }
                let _ = write!(shape_list, "({h}usize, {e}usize, {k}usize)");
            }
            let _ = write!(method, "    .routed_shapes(&[{shape_list}])\n");
        }
        let _ = writeln!(
            generated,
            "#[allow(dead_code)] // a generated rule exists for every catalogue key, used or not"
        );
        let _ = writeln!(
            generated,
            "pub(crate) const {name}: ShapeConstraint = ShapeConstraint::ANY"
        );
        let _ = writeln!(generated, "{};", method.trim_end());
    }
    generated
}
