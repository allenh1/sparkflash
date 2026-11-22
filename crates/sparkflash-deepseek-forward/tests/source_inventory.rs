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

//! The inventory gate for milestone 2 A.
//!
//! The counts below are the starting tip's. Each is a ceiling: a change may reduce it, never
//! raise it. A new file with a count above zero is a failure until its budget is stated here.
//! The four-rank sites are a separate table because milestone 3 owns draining them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Counts {
    /// `ModelConfig` mentions (the type, its constructors and its values).
    model_config: usize,
    /// Field reads through a variable named `config`.
    config_reads: usize,
    /// Uses of the forward's named model constants.
    model_constants: usize,
    /// Literal released shapes (5120, 4096, 4608, 384, ...).
    shape_literals: usize,
    /// String literals that spell a checkpoint tensor name.
    tensor_names: usize,
    /// String literals that spell a kernel symbol.
    kernel_symbols: usize,
}

/// The starting tip's per-file ceilings. Only these may fall.
const BUDGET: &[(&str, Counts)] = &[
    (
        "src/attention_device.rs",
        Counts {
            model_config: 11,
            config_reads: 40,
            model_constants: 0,
            shape_literals: 1,
            tensor_names: 10,
            kernel_symbols: 5,
        },
    ),
    (
        "src/device_link.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 1,
            shape_literals: 0,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/engram_device_state.rs",
        Counts {
            model_config: 2,
            config_reads: 6,
            model_constants: 0,
            shape_literals: 0,
            tensor_names: 11,
            kernel_symbols: 0,
        },
    ),
    (
        "src/layer_attention_stages.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 4,
            tensor_names: 0,
            kernel_symbols: 2,
        },
    ),
    (
        "src/layer_device.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 1,
            shape_literals: 16,
            tensor_names: 28,
            kernel_symbols: 21,
        },
    ),
    (
        "src/moe_device.rs",
        Counts {
            model_config: 6,
            config_reads: 16,
            model_constants: 0,
            shape_literals: 3,
            tensor_names: 23,
            kernel_symbols: 9,
        },
    ),
    (
        "src/open_manifest.rs",
        Counts {
            model_config: 2,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 0,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/rdma_exchange.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 1,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/resident_weights.rs",
        Counts {
            model_config: 23,
            config_reads: 68,
            model_constants: 0,
            shape_literals: 5,
            tensor_names: 106,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_combine.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 5,
            tensor_names: 1,
            kernel_symbols: 3,
        },
    ),
    (
        "src/tp4_forward.rs",
        Counts {
            model_config: 46,
            config_reads: 217,
            model_constants: 170,
            shape_literals: 61,
            tensor_names: 191,
            kernel_symbols: 67,
        },
    ),
    (
        "src/tp4_forward/batch.rs",
        Counts {
            model_config: 0,
            config_reads: 6,
            model_constants: 0,
            shape_literals: 25,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_forward/bulk.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 12,
            tensor_names: 0,
            kernel_symbols: 10,
        },
    ),
    (
        "src/tp4_forward/draft.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 4,
            shape_literals: 2,
            tensor_names: 13,
            kernel_symbols: 13,
        },
    ),
    (
        "src/tp4_forward/lanes.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 4,
            shape_literals: 1,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_forward/rows.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 9,
            shape_literals: 4,
            tensor_names: 2,
            kernel_symbols: 6,
        },
    ),
    (
        "src/tp4_forward/spec.rs",
        Counts {
            model_config: 4,
            config_reads: 3,
            model_constants: 8,
            shape_literals: 10,
            tensor_names: 2,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_operands.rs",
        Counts {
            model_config: 2,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 8,
            tensor_names: 15,
            kernel_symbols: 2,
        },
    ),
    (
        "src/tp4_prefill.rs",
        Counts {
            model_config: 4,
            config_reads: 6,
            model_constants: 44,
            shape_literals: 0,
            tensor_names: 1,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_reduce.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 4,
            tensor_names: 0,
            kernel_symbols: 0,
        },
    ),
    (
        "src/tp4_resident.rs",
        Counts {
            model_config: 5,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 4,
            tensor_names: 47,
            kernel_symbols: 1,
        },
    ),
    (
        "src/tp4_smoke.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 0,
            shape_literals: 4,
            tensor_names: 14,
            kernel_symbols: 1,
        },
    ),
    (
        "src/tp4_tail.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 70,
            shape_literals: 2,
            tensor_names: 0,
            kernel_symbols: 4,
        },
    ),
    (
        "src/tp4_token.rs",
        Counts {
            model_config: 0,
            config_reads: 0,
            model_constants: 1,
            shape_literals: 0,
            tensor_names: 0,
            kernel_symbols: 1,
        },
    ),
];

/// The four-rank sites, on their own list so milestone 3 can drain them without touching A's
/// model-value and tensor-name budgets.
const FOUR_RANK_BUDGET: &[(&str, usize)] = &[
    ("src/tp4_forward.rs", 1),
    ("src/tp4_operands.rs", 1),
    ("src/tp4_token.rs", 1),
];

const MODEL_CONSTANTS: &[&str] = &[
    "HIDDEN",
    "VOCAB_PER_RANK",
    "HC_MULT",
    "MLA_LATENT",
    "DEFAULT_COMBINE_ROWS",
    "PREDICTOR_ROWS",
    "WINDOW_QUANT_BLOCK",
    "WINDOW_CODE_BYTES",
    "WINDOW_SCALE_BYTES",
    "SEAM_RING_ROWS",
    "OPT_SEAMS_SCRATCH",
];

const SHAPE_LITERALS: &[&str] = &["5120", "32320", "4096", "2560", "4608", "576", "640", "384"];

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn rust_sources(directory: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory)
        .expect("the forward crate has a source directory")
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Splits source into code with comments and string bodies blanked, plus the string bodies.
fn code_and_strings(source: &str) -> (String, Vec<String>) {
    let bytes = source.as_bytes();
    let mut code = String::with_capacity(source.len());
    let mut strings = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if source[index..].starts_with("//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if source[index..].starts_with("/*") {
            let mut depth = 1usize;
            index += 2;
            while index < bytes.len() && depth > 0 {
                if source[index..].starts_with("/*") {
                    depth += 1;
                    index += 2;
                } else if source[index..].starts_with("*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    if bytes[index] == b'\n' {
                        code.push('\n');
                    }
                    index += 1;
                }
            }
            continue;
        }
        if let Some((start, hashes)) = raw_string_start(source, index) {
            let content_start = start;
            let closing = format!("\"{}", "#".repeat(hashes));
            let end = if let Some(found) = source[content_start..].find(&closing) {
                strings.push(source[content_start..content_start + found].to_string());
                content_start + found + closing.len()
            } else {
                strings.push(source[content_start..].to_string());
                bytes.len()
            };
            code.push(' ');
            index = end;
            continue;
        }
        if bytes[index] == b'"' {
            let mut end = index + 1;
            let mut text = String::new();
            while end < bytes.len() {
                match bytes[end] {
                    b'\\' => {
                        if end + 1 < bytes.len() {
                            text.push(bytes[end + 1] as char);
                        }
                        end += 2;
                    }
                    b'"' => {
                        end += 1;
                        break;
                    }
                    byte => {
                        text.push(byte as char);
                        end += 1;
                    }
                }
            }
            strings.push(text);
            code.push(' ');
            index = end;
            continue;
        }
        if bytes[index] == b'\'' {
            // A character literal ends before the next quote; a lifetime has no closing quote.
            let mut end = index + 1;
            let mut closed = false;
            while end < bytes.len() && end <= index + 6 {
                if bytes[end] == b'\\' {
                    end += 2;
                } else if bytes[end] == b'\'' {
                    end += 1;
                    closed = true;
                    break;
                } else {
                    end += 1;
                }
            }
            if closed {
                code.push(' ');
                index = end;
                continue;
            }
        }
        let ch = source[index..]
            .chars()
            .next()
            .expect("an index in the source");
        code.push(ch);
        index += ch.len_utf8();
    }
    (code, strings)
}

/// If a raw string starts at `index`, returns (body start, hash count).
fn raw_string_start(source: &str, index: usize) -> Option<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut at = index;
    if bytes.get(at) == Some(&b'b') {
        at += 1;
    }
    if bytes.get(at) != Some(&b'r') {
        return None;
    }
    at += 1;
    let mut hashes = 0;
    while bytes.get(at) == Some(&b'#') {
        hashes += 1;
        at += 1;
    }
    (bytes.get(at) == Some(&b'"')).then_some((at + 1, hashes))
}

fn count_file(source: &str) -> (Counts, usize) {
    let (code, strings) = code_and_strings(source);
    let mut counts = Counts {
        model_config: code.matches("ModelConfig").count(),
        config_reads: code.matches("config.").count(),
        ..Counts::default()
    };
    for token in code.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_')) {
        if MODEL_CONSTANTS.contains(&token) {
            counts.model_constants += 1;
        }
        let literal = token.replace('_', "");
        if SHAPE_LITERALS.contains(&literal.as_str()) {
            counts.shape_literals += 1;
        }
    }
    for text in &strings {
        if text.contains("layers.")
            || text.contains("mtp.")
            || text.contains("engram.")
            || text.contains(".weight")
            || text.contains(".scale")
        {
            counts.tensor_names += 1;
        }
        let identifier = !text.is_empty()
            && text
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_');
        if identifier
            && (text.contains("_sm121")
                || text.contains("_bf16")
                || text.contains("_f32")
                || text.contains("_fp8")
                || text.starts_with("b12x_"))
        {
            counts.kernel_symbols += 1;
        }
    }
    let compact: String = code.chars().filter(|ch| !ch.is_whitespace()).collect();
    let four_rank = [
        "world==4",
        "world!=4",
        "world<4",
        "world>4",
        "world_size==4",
        "world_size!=4",
    ]
    .iter()
    .map(|pattern| compact.matches(pattern).count())
    .sum();
    (counts, four_rank)
}

fn inventory() -> (BTreeMap<String, Counts>, BTreeMap<String, usize>) {
    let root = crate_root();
    let mut sources = Vec::new();
    rust_sources(&root.join("src"), &mut sources);
    sources.sort();
    let mut counts = BTreeMap::new();
    let mut four_rank = BTreeMap::new();
    for path in sources {
        let source = std::fs::read_to_string(&path).expect("a Rust source");
        let (file_counts, file_four_rank) = count_file(&source);
        let name = path
            .strip_prefix(&root)
            .expect("a source below the crate root")
            .display()
            .to_string();
        counts.insert(name.clone(), file_counts);
        four_rank.insert(name, file_four_rank);
    }
    (counts, four_rank)
}

#[test]
fn every_inventory_count_only_falls() {
    let (actual, actual_four_rank) = inventory();
    let budget: BTreeMap<&str, Counts> = BUDGET.iter().copied().collect();
    let four_budget: BTreeMap<&str, usize> = FOUR_RANK_BUDGET.iter().copied().collect();
    let mut failures = Vec::new();
    for (file, counts) in &actual {
        let ceiling = budget.get(file.as_str()).copied().unwrap_or_default();
        for (name, value, limit) in [
            ("model_config", counts.model_config, ceiling.model_config),
            ("config_reads", counts.config_reads, ceiling.config_reads),
            (
                "model_constants",
                counts.model_constants,
                ceiling.model_constants,
            ),
            (
                "shape_literals",
                counts.shape_literals,
                ceiling.shape_literals,
            ),
            ("tensor_names", counts.tensor_names, ceiling.tensor_names),
            (
                "kernel_symbols",
                counts.kernel_symbols,
                ceiling.kernel_symbols,
            ),
        ] {
            if value > limit {
                failures.push(format!("{file}: {name} {value} > pinned {limit}"));
            }
        }
        let four_limit = four_budget.get(file.as_str()).copied().unwrap_or(0);
        let four_value = actual_four_rank.get(file).copied().unwrap_or(0);
        if four_value > four_limit {
            failures.push(format!(
                "{file}: four_rank {four_value} > pinned {four_limit}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the forward's source inventory grew; a count may only fall:\n{}\n\nactual: {actual:#?}\nfour-rank: {actual_four_rank:#?}",
        failures.join("\n")
    );
}
