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

//! `sparkflash onboard <hf-id>`: what a checkpoint would need before any code is written.
//!
//! The preview reads the checkpoint's own `config.json` (the operator's `--config`, the local
//! Hugging Face cache, and only then the Hub) through [Architecture], then reads the registrations
//! this tree already has: `ARMS`, `MODELS`, `CATALOGUE`, `WAIVERS` and `DEFAULTS`. Nothing is
//! copied into a second table here.
//!
//! The output answers four questions in order:
//!
//! * which `CATALOGUE` parts the architecture already names, and which are absent;
//! * which named parts have no tensor-core body in a pass and no phase-3a `WAIVERS` line, so a
//!   `PartExec` (a body of their own) has to be written;
//! * the stored format of each weight class;
//! * the fit at TP 1, 2 and 4 and the stored-bytes-per-token floor from `sparkflash-ops`.
//!
//! It opens no device and needs no checkpoint: the configuration is enough.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use sparkflash_model::{
    ARMS, Architecture, DEFAULTS, MEMORY_FLOOR_BYTES, NODE_MEMORY_BUDGET_BYTES, Part, Target,
    canonical_model_type, find_model_by_type, gib,
};
use sparkflash_ops::parts::{CATALOGUE, WAIVERS, Waiver, carried};
use sparkflash_ops::plan::{ComputeClass, KernelPhase, Precision};
use sparkflash_ops::{EnginePlan, bytes_per_token_floor, stored_bytes_floor};

/// The device target the catalogue is read at. The phase-8 preview is a build-time question, so it
/// asks the target this tree compiles kernels for.
const TARGET: Target = Target::Sm121;

/// The tensor-parallel worlds the preview judges.
const WORLDS: [usize; 3] = [1, 2, 4];

/// The row count the catalogue uses for a prompt/prefill probe.
const PROMPT_ROWS: usize = sparkflash_ops::projection::QWEN_GEMM_MIN_BATCH;

/// The row count the catalogue uses for a token/decode probe.
const TOKEN_ROWS: usize = 1;

/// A checkpoint document and where it came from.
struct Input {
    /// The model id the operator named, when one was named.
    id: Option<String>,
    /// The local source for the operator.
    source: String,
    /// The parsed architecture.
    architecture: Architecture,
    /// Declarations [Architecture] reads through but does not model.
    unmodelled: Vec<String>,
}

/// Parse the arguments after `onboard`.
///
/// `sparkflash onboard ORG/REPO [--config FILE]`; with `--config` the id is optional and names
/// only the output's title.
pub fn run(args: impl Iterator<Item = String>) -> ExitCode {
    const USAGE: &str = "usage: sparkflash onboard <hf-id> [--config FILE]";
    let mut id: Option<String> = None;
    let mut config: Option<String> = None;
    let mut args = args.peekable();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => match args.next() {
                Some(path) => config = Some(path),
                None => {
                    eprintln!("sparkflash onboard: --config needs a path; {USAGE}");
                    return ExitCode::FAILURE;
                }
            },
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("sparkflash onboard: unknown argument {other:?}; {USAGE}");
                return ExitCode::FAILURE;
            }
            other => {
                if id.is_some() {
                    eprintln!("sparkflash onboard: one model id, not two; {USAGE}");
                    return ExitCode::FAILURE;
                }
                id = Some(other.to_owned());
            }
        }
    }
    if id.is_none() && config.is_none() {
        eprintln!("sparkflash onboard: {USAGE}");
        return ExitCode::FAILURE;
    }
    let input = match load(id.as_deref(), config.as_deref().map(Path::new)) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("sparkflash onboard: {error}");
            return ExitCode::FAILURE;
        }
    };
    print!("{}", render(&input));
    ExitCode::SUCCESS
}

/// Load a document from `--config`, the hub cache, or the Hub, in that order.
fn load(id: Option<&str>, config: Option<&Path>) -> Result<Input, String> {
    if let Some(path) = config {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        return from_text(id, &text, format!("--config {}", path.display()));
    }
    let Some(id) = id else {
        return Err("no model id and no --config".to_owned());
    };
    if !sparkflash_weights::hub::is_model_id(id) {
        return Err(format!("{id:?} is not a Hugging Face model id"));
    }
    if let Some(directory) = sparkflash_weights::hub::cached_model_dir(id) {
        let path = directory.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let architecture = Architecture::from_checkpoint_dir(&directory)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let unmodelled = unmodelled(&text);
        return Ok(Input {
            id: Some(id.to_owned()),
            source: format!("hub cache {}", directory.display()),
            architecture,
            unmodelled,
        });
    }
    let text = fetch(id)?;
    from_text(Some(id), &text, format!("the Hub ({id})"))
}

/// Parse a document that was read locally or fetched.
fn from_text(id: Option<&str>, text: &str, source: String) -> Result<Input, String> {
    let architecture =
        Architecture::from_json(text).map_err(|error| format!("{source}: {error}"))?;
    Ok(Input {
        id: id.map(str::to_owned),
        source,
        unmodelled: unmodelled(text),
        architecture,
    })
}

/// The declarations [Architecture] deliberately leaves unmodelled.
fn unmodelled(text: &str) -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(text)
        .map(|root| Architecture::unmodelled_features(&root))
        .unwrap_or_default()
}

/// Fetch a released `config.json` from the Hub.
fn fetch(id: &str) -> Result<String, String> {
    let url = format!("https://huggingface.co/{id}/resolve/main/config.json");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();
    let mut request = agent
        .get(&url)
        .set("Accept", "application/json")
        .set("User-Agent", "sparkflash");
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(name)
            && !token.is_empty()
        {
            request = request.set("Authorization", &format!("Bearer {token}"));
            break;
        }
    }
    match request.call() {
        Ok(response) => response
            .into_string()
            .map_err(|error| format!("reading {url}: {error}")),
        Err(ureq::Error::Status(status, _)) => Err(format!("{url} returned HTTP {status}")),
        Err(ureq::Error::Transport(error)) => Err(format!("{url}: {error}")),
    }
}

/// Render the preview for one parsed checkpoint.
fn render(input: &Input) -> String {
    let architecture = &input.architecture;
    let mut out = String::new();
    let title = input.id.as_deref().unwrap_or(architecture.family.as_str());
    let _ = writeln!(out, "sparkflash onboard {title}");
    let _ = writeln!(out, "source: {}", input.source);
    let _ = writeln!(
        out,
        "architecture: {} ({}) - {} layers, hidden {}, vocab {}, context {}",
        architecture.family,
        if architecture.architectures.is_empty() {
            "no architectures list".to_owned()
        } else {
            architecture.architectures.join(", ")
        },
        architecture.layers.len(),
        architecture.hidden_size,
        architecture.vocab_size,
        architecture.max_position_embeddings
    );
    if !input.unmodelled.is_empty() {
        let _ = writeln!(out, "unmodelled: {}", input.unmodelled.join("; "));
    }

    let canonical = canonical_model_type(&architecture.family);
    let arm = ARMS.iter().find(|arm| arm.families().contains(&canonical));
    let model = find_model_by_type(&architecture.family);
    match arm {
        Some(arm) => {
            let conversion = arm.conversion();
            let _ = writeln!(
                out,
                "registered arm: {} (model {}, weights {}, context {}, tokenizer {}, template {}, \
                 conversion TP{} {}, ledger {})",
                arm.id(),
                arm.model_id(),
                if arm.weights().is_empty() {
                    "the checkpoint".to_owned()
                } else {
                    arm.weights().join(", ")
                },
                arm.default_context(),
                arm.tokenizer().tokenizer_id,
                arm.tokenizer().chat_template,
                conversion.world_size,
                if conversion.released {
                    "released"
                } else {
                    "dynamic"
                },
                arm.ledger().arm
            );
        }
        None => {
            let _ = writeln!(
                out,
                "registered arm: none (the catalogue can preview it; a new ARMS entry is part of \
                 what you supply)"
            );
        }
    }
    if let Some(model) = model {
        let _ = writeln!(
            out,
            "registered model: {} ({}, context {}, weights {})",
            model.id, model.label, model.context_tokens, model.weights_id
        );
    } else {
        let _ = writeln!(
            out,
            "registered model: none (MODELS has no descriptor for this family)"
        );
    }
    let defaults = DEFAULTS
        .iter()
        .map(|default| format!("{}={}", default.key, default.value))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "serving defaults (DEFAULTS): {defaults}");

    render_formats(&mut out, architecture, input, arm.is_some());
    render_parts(&mut out, architecture);
    render_fit(&mut out, architecture, arm.map(|arm| arm.default_context()));
    out
}

/// The stored formats and the per-part arithmetic precision.
fn render_formats(out: &mut String, architecture: &Architecture, input: &Input, registered: bool) {
    let quantization = &architecture.quantization;
    let _ = writeln!(
        out,
        "stored formats: dense {}, experts {}, MTP {} (method {}, block {}, scale {}, activation {})",
        quantization.dense_format.name(),
        quantization.expert_format.name(),
        quantization.mtp_expert_format.map_or_else(
            || "same as experts".to_owned(),
            |format| format.name().to_owned()
        ),
        quantization.method.as_deref().unwrap_or("none"),
        quantization
            .block
            .map_or_else(|| "none".to_owned(), |block| block.to_string()),
        quantization.scale.as_deref().unwrap_or("none"),
        quantization.activation.as_deref().unwrap_or("none")
    );
    let _ = writeln!(
        out,
        "checkpoint: id {}, register {}",
        input.id.as_deref().unwrap_or("not named"),
        if registered {
            "already in ARMS"
        } else {
            "new to ARMS"
        }
    );
    let plan = EnginePlan::for_architecture(architecture.clone(), TARGET, 1, 0);
    match plan {
        Ok(plan) => {
            let mut by_part: BTreeMap<Part, BTreeSet<&'static str>> = BTreeMap::new();
            for requirement in plan.requirements() {
                by_part
                    .entry(requirement.part)
                    .or_default()
                    .insert(requirement.precision.name());
            }
            let _ = writeln!(out, "per-part arithmetic formats (TP1):");
            for (part, precisions) in by_part {
                let _ = writeln!(
                    out,
                    "  {:<28} {}",
                    part.name(),
                    precisions.into_iter().collect::<Vec<_>>().join(", ")
                );
            }
        }
        Err(error) => {
            let policy = sparkflash_ops::PrecisionPolicy::from_architecture(architecture);
            let _ = writeln!(
                out,
                "per-part arithmetic formats: no TP1 plan ({error}); policy dense {}, experts {}, \
                 attention {}, MTP {}",
                policy.dense.name(),
                policy.experts.name(),
                policy.attention.name(),
                policy.mtp.name()
            );
        }
    }
}

/// Which named parts the catalogue carries, which it lacks, and which need a tensor-core body.
fn render_parts(out: &mut String, architecture: &Architecture) {
    let available = carried(TARGET);
    let needed = architecture.parts();
    let missing = architecture.missing(&available);
    let _ = writeln!(out, "parts CATALOGUE has ({}):", TARGET);
    let present: Vec<&str> = needed
        .iter()
        .filter(|part| available.contains(*part))
        .map(Part::name)
        .collect();
    let _ = writeln!(
        out,
        "  {}",
        if present.is_empty() {
            "none".to_owned()
        } else {
            present.join(", ")
        }
    );
    let _ = writeln!(out, "parts CATALOGUE lacks ({TARGET}):");
    let _ = writeln!(
        out,
        "  {}",
        if missing.is_empty() {
            "none".to_owned()
        } else {
            missing
                .iter()
                .map(|part| part.name())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    let mut per_part: BTreeMap<Part, Vec<Precision>> = BTreeMap::new();
    match EnginePlan::for_architecture(architecture.clone(), TARGET, 1, 0) {
        Ok(plan) => {
            for requirement in plan.requirements() {
                let precisions = per_part.entry(requirement.part).or_default();
                if !precisions.contains(&requirement.precision) {
                    precisions.push(requirement.precision);
                }
            }
        }
        Err(_) => {
            let policy = sparkflash_ops::PrecisionPolicy::from_architecture(architecture);
            let all = [policy.dense, policy.experts, policy.attention, policy.mtp];
            for part in needed.iter().filter(|part| expects_tensor_core(*part)) {
                per_part.entry(part).or_default().extend(all);
            }
        }
    }

    let mut needing = BTreeSet::new();
    let mut waived = BTreeSet::new();
    for part in needed.iter().filter(|part| expects_tensor_core(*part)) {
        let Some(precisions) = per_part.get(&part) else {
            needing.insert(format!("{}: no TP1 requirement", part.name()));
            continue;
        };
        for precision in precisions {
            for (phase, rows) in [
                (KernelPhase::Prefill, PROMPT_ROWS),
                (KernelPhase::Decode, TOKEN_ROWS),
            ] {
                if has_tensor_core_body(part, *precision, phase, rows) {
                    continue;
                }
                match waiver(part, *precision, phase, rows) {
                    Some(waiver) => {
                        waived.insert(format!(
                            "{} {} {}: {}",
                            part.name(),
                            precision.name(),
                            phase.name(),
                            waiver.measurement
                        ));
                    }
                    None => {
                        needing.insert(format!(
                            "{} {} {} ({} rows): no tensor-core body, no WAIVERS line",
                            part.name(),
                            precision.name(),
                            phase.name(),
                            rows
                        ));
                    }
                }
            }
        }
    }
    let _ = writeln!(out, "parts with tensor-core gaps:");
    if waived.is_empty() {
        let _ = writeln!(out, "  none");
    } else {
        for line in &waived {
            let _ = writeln!(out, "  waived {line}");
        }
    }
    let _ = writeln!(out, "parts needing a PartExec:");
    if needing.is_empty() {
        let _ = writeln!(out, "  none");
    } else {
        for line in &needing {
            let _ = writeln!(out, "  {line}");
        }
    }
}

/// Whether the catalogue exports a tensor-core body for a part, precision, pass and row count.
fn has_tensor_core_body(part: Part, precision: Precision, phase: KernelPhase, rows: usize) -> bool {
    CATALOGUE
        .iter()
        .filter(|entry| entry.part == part && entry.target == TARGET)
        .flat_map(|entry| entry.symbols)
        .any(|symbol| {
            symbol.class == ComputeClass::TensorCore && symbol.serves(precision, phase, rows)
        })
}

/// The waiver that covers a gap, if one does.
fn waiver(
    part: Part,
    precision: Precision,
    phase: KernelPhase,
    rows: usize,
) -> Option<&'static Waiver> {
    WAIVERS.iter().find(|waiver| {
        waiver.part == Some(part)
            && waiver.precision.is_none_or(|waived| waived == precision)
            && waiver.phase.serves(phase)
            && waiver.rows.contains(rows)
    })
}

/// Whether a part's arithmetic is a matrix product, and so expects a tensor-core body.
fn expects_tensor_core(part: Part) -> bool {
    matches!(
        part,
        Part::DenseGemm | Part::DenseMlp | Part::Moe | Part::GroupedMoeGemm | Part::LmHead
    )
}

/// The fit at each world and the stored-bytes-per-token floor.
fn render_fit(out: &mut String, architecture: &Architecture, arm_context: Option<usize>) {
    let declared = architecture.max_position_embeddings;
    let requested = arm_context
        .filter(|context| *context > 0)
        .unwrap_or(declared);
    let context = match (requested, declared) {
        (requested, 0) => requested,
        (requested, declared) => requested.min(declared),
    };
    let budget = NODE_MEMORY_BUDGET_BYTES.saturating_sub(MEMORY_FLOOR_BYTES);
    let _ = writeln!(
        out,
        "fit at {context} tokens (arm/default context, capped by the checkpoint), one DGX Spark \
         node, {} budget:",
        gib(budget)
    );
    for world in WORLDS {
        match EnginePlan::for_architecture(architecture.clone(), TARGET, world, 0) {
            Ok(plan) => {
                let floor = match stored_bytes_floor(&plan) {
                    Ok(floor) => floor,
                    Err(error) => {
                        let _ = writeln!(out, "  TP{world}: no stored-bytes floor ({error})");
                        continue;
                    }
                };
                let bytes_per_token = match bytes_per_token_floor(&plan, floor) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        let _ = writeln!(out, "  TP{world}: no bytes-per-token floor ({error})");
                        continue;
                    }
                };
                let kv = match architecture.kv_plan(world, false) {
                    Ok(kv) => kv,
                    Err(error) => {
                        let _ = writeln!(
                            out,
                            "  TP{world}: weights {}, but no KV plan ({error}); bytes/token floor \
                             {}",
                            gib_u128(floor),
                            bytes_per_token.floor
                        );
                        continue;
                    }
                };
                let cache = u128::from(kv.sequence_bytes(context));
                let total = floor.saturating_add(cache);
                let _ = writeln!(
                    out,
                    "  TP{world}: weights {}, KV {} at {context} tokens, total {} of {} - {}; \
                     bytes/token floor {} (stored_bytes_floor; bytes_per_token_floor within floor: {})",
                    gib_u128(floor),
                    gib_u128(cache),
                    gib_u128(total),
                    gib(budget),
                    if total <= u128::from(budget) {
                        "fits"
                    } else {
                        "does not fit"
                    },
                    bytes_per_token.floor,
                    bytes_per_token.within_floor()
                );
            }
            Err(error) => {
                let _ = writeln!(out, "  TP{world}: no execution plan ({error})");
            }
        }
    }
}

/// A u128 byte count as operator-facing gibibytes.
fn gib_u128(bytes: u128) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u128 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed Dory document, so the test opens no network and no checkpoint.
    const DORY: &str = include_str!(
        "../../sparkflash-model-dory/config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json"
    );

    fn preview() -> String {
        let input = from_text(
            Some("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"),
            DORY,
            "test fixture".to_owned(),
        )
        .expect("the released Dory document parses");
        render(&input)
    }

    #[test]
    fn the_preview_names_the_parts_the_catalogue_has_and_lacks() {
        let text = preview();
        assert!(text.contains("parts CATALOGUE has (sm121):"));
        assert!(text.contains("parts CATALOGUE lacks (sm121):"));
        assert!(text.contains("embedding"), "no embedding in:\n{text}");
        assert!(
            text.contains("gqa_attention"),
            "no gqa_attention in:\n{text}"
        );
        assert!(text.contains("dense_mlp"), "no dense_mlp in:\n{text}");
        assert!(text.contains("lm_head"), "no lm_head in:\n{text}");
        assert!(text.contains("parts needing a PartExec:"));
        assert!(text.contains("parts with tensor-core gaps:"));
    }

    #[test]
    fn the_preview_names_every_missing_part() {
        let input = from_text(None, DORY, "test fixture".to_owned())
            .expect("the released Dory document parses");
        let text = render(&input);
        let missing = input.architecture.missing(&carried(TARGET));
        if missing.is_empty() {
            assert!(
                text.contains("parts CATALOGUE lacks (sm121):\n  none"),
                "empty missing list is not printed as none:\n{text}"
            );
        } else {
            for part in missing {
                assert!(
                    text.contains(part.name()),
                    "missing part {} is not named:\n{text}",
                    part.name()
                );
            }
        }
    }

    #[test]
    fn the_preview_names_a_stored_format_and_a_bytes_per_token_floor() {
        let text = preview();
        assert!(text.contains("stored formats: dense BF16"));
        assert!(text.contains("bytes/token floor"), "no floor in:\n{text}");
        let floor_line = text
            .lines()
            .find(|line| line.contains("TP1:") && line.contains("bytes/token floor"))
            .expect("TP1 prints a bytes-per-token floor");
        assert!(
            floor_line
                .split("bytes/token floor ")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .is_some_and(|number| number.parse::<u128>().is_ok()),
            "the TP1 floor is not a number: {floor_line}"
        );
        assert!(text.contains("fit at 262144 tokens"));
        assert!(text.contains("TP2:"));
        assert!(text.contains("TP4:"));
    }
}
