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

/// The shape values one vision catalogue key draws from its instances.
///
/// A set holds every built value; the generator emits an equality for a singleton, a list for the
/// patch sides (the tower compiles one body per side), and refuses an ambiguous equality rather
/// than quietly widening the rule.
#[derive(Default)]
struct VisionKey {
    hidden: BTreeSet<u64>,
    heads: BTreeSet<u64>,
    patch: BTreeSet<u64>,
    intermediate: BTreeSet<u64>,
    out_hidden: BTreeSet<u64>,
}

/// The shape fields each vision catalogue rule claims.
///
/// The values all come from kernels/instances.toml; this table says which of them the part's
/// compiled bodies actually read, so a rule never claims an unrelated tower dimension. The
/// non-shape part facts (norm kind, epsilon, activation, position, downsample) stay beside the
/// rule in src/parts.rs.
const VISION_FIELDS: &[(&str, &[&str])] = &[
    ("deepseek-vision-patch-embed", &["hidden", "patch"]),
    ("deepseek-vision-position", &["hidden", "heads"]),
    ("deepseek-vision-norm", &["hidden"]),
    ("deepseek-vision-attention", &["hidden", "heads"]),
    ("deepseek-vision-mlp", &["hidden", "intermediate"]),
    ("deepseek-vision-downsample", &["hidden"]),
    ("deepseek-vision-merger", &["hidden", "out_hidden"]),
];

/// The one value a vision field has, or a panic naming the ambiguous key.
fn one(values: &BTreeSet<u64>, key: &str, field: &str) -> u64 {
    assert!(
        values.len() == 1,
        "sparkflash-ops: {key} names {} {field} values {values:?}; an equality needs one",
        values.len()
    );
    *values.iter().next().expect("one value")
}

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
    let mut vision: BTreeMap<String, VisionKey> = BTreeMap::new();
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
        let family = table
            .get("family")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("sparkflash-ops: an instance has no string family"));
        let Some(key) = table.get("catalogue").and_then(toml::Value::as_str) else {
            continue;
        };
        if family == "vision" {
            let row = vision.entry(key.to_owned()).or_default();
            row.hidden.insert(get("hidden"));
            row.heads.insert(get("heads"));
            row.patch.insert(get("patch"));
            row.intermediate.insert(get("intermediate"));
            row.out_hidden.insert(get("out_hidden"));
            // The required keys the vision catalogue rule does not claim are still validated.
            let _ = (get("experts"), get("top_k"));
            continue;
        }
        assert!(
            family == "router" || family == "routed",
            "sparkflash-ops: {key} names family {family:?}, not router, routed or vision"
        );
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
    for (key, shapes) in &vision {
        let fields = VISION_FIELDS
            .iter()
            .find_map(|(name, fields)| (*name == key.as_str()).then_some(*fields))
            .unwrap_or_else(|| {
                panic!(
                    "sparkflash-ops: vision catalogue key {key:?} has no field list in VISION_FIELDS"
                )
            });
        let name = format!("CATALOGUE_{}", key.to_ascii_uppercase().replace('-', "_"));
        let _ = writeln!(
            generated,
            "#[allow(dead_code)] // a generated rule exists for every catalogue key, used or not"
        );
        let _ = writeln!(
            generated,
            "pub(crate) const {name}: ShapeConstraint = ShapeConstraint::ANY"
        );
        if fields.contains(&"hidden") {
            let _ = writeln!(
                generated,
                "    .hidden_equals({})",
                one(&shapes.hidden, key, "hidden")
            );
        }
        if fields.contains(&"patch") {
            assert!(
                !shapes.patch.is_empty(),
                "sparkflash-ops: {key} names no patch side"
            );
            let sides: Vec<String> = shapes
                .patch
                .iter()
                .map(|side| format!("{side}usize"))
                .collect();
            let _ = writeln!(generated, "    .patch_sizes(&[{}])", sides.join(", "));
        }
        if fields.contains(&"heads") {
            let heads = one(&shapes.heads, key, "heads");
            let hidden = one(&shapes.hidden, key, "hidden");
            assert!(
                hidden % heads == 0,
                "sparkflash-ops: {key} has a {hidden}-wide tower over {heads} heads"
            );
            let _ = writeln!(generated, "    .heads_multiple({heads})");
            let _ = writeln!(generated, "    .max_heads({heads})");
            let _ = writeln!(generated, "    .head_dim_equals({})", hidden / heads);
        }
        if fields.contains(&"intermediate") {
            let _ = writeln!(
                generated,
                "    .intermediate_equals({})",
                one(&shapes.intermediate, key, "intermediate")
            );
        }
        if fields.contains(&"out_hidden") {
            let _ = writeln!(
                generated,
                "    .out_hidden_equals({})",
                one(&shapes.out_hidden, key, "out_hidden")
            );
        }
        let _ = writeln!(generated, ";");
    }
    generated
}
