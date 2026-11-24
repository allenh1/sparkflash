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

//! The device-open gate: the second shape's plan answers for every part the device path opens.
//!
//! The five windows before this gate each found one defect the host could have found: a released
//! literal standing where the plan's answer belongs, in the head's block maxima, the wo_a operand,
//! the shared expert's format, the routed device geometry or the DSpark depth. Each had a
//! hand-written regression test; none of them walked the open. This gate walks the parts in the
//! order the device opens them, at worlds 1, 2 and 4, and refuses by name when a plan fact is
//! missing or a body built for the released shape answers instead of the plan.
//!
//! The gate is plan-driven: it enumerates the plan's layers, parts, stored formats, kernel
//! symbols, draft stages and the families whose instance rows the build generates (routed
//! experts, head screen, NVFP4 prefill). It does not list the five defects; it asks every part the
//! question those defects answered wrongly.

use std::path::{Path, PathBuf};

use sparkflash_deepseek_forward::{plan, tp4_resident, tp4_tail};
use sparkflash_device::resident::{HeaderEntry, OperandLayout, ResidentPlan};
use sparkflash_kernels::moe_routed_device::MoeRoutedGeometry;
use sparkflash_kernels::{baseline, head_screen_ref, lm_head_ref};
use sparkflash_model::{Part, Target};
use sparkflash_model_deepseek_v41::config::ModelConfig;
use sparkflash_model_deepseek_v41::tensor::TensorPlan;
use sparkflash_ops::engine_plan::EnginePlan;

/// The worlds the device arm is measured at.
const WORLDS: [usize; 3] = [1, 2, 4];

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/second-shape")
}

/// The harness that runs this shape, read so the gate can check the two settings a shape whose
/// bodies are built for another geometry must state.
const HARNESS: &str = include_str!("../../../scripts/second-shape.py");

fn harness_states(name: &str, value: &str) -> bool {
    HARNESS
        .lines()
        .any(|line| line.contains(name) && line.contains(value))
}

/// The refusals one world collects, so one run reports every part that fails.
#[derive(Default)]
struct Refusals {
    refused: Vec<String>,
}

impl Refusals {
    /// A named refusal; the detail is built only when the check fails.
    fn require(&mut self, case: &str, holds: bool, detail: impl FnOnce() -> String) {
        if !holds {
            self.refused.push(format!("{case}: {}", detail()));
        }
    }

    /// A result-shaped check; the error becomes the refusal's detail.
    fn report<T>(&mut self, case: &str, result: Result<T, String>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                self.refused.push(format!("{case}: {error}"));
                None
            }
        }
    }
}

#[test]
fn the_second_shape_opens_from_its_own_plan_at_one_two_and_four_ranks() {
    let dir = fixture();
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the fixture config");
    let config = ModelConfig::from_json(&text).expect("the fixture parses");
    let written = TensorPlan::language_model(&config);
    let mut refused: Vec<String> = Vec::new();
    for world in WORLDS {
        let plan = EnginePlan::from_path(&dir.join("config.json"), Target::Sm121, world, 0)
            .unwrap_or_else(|error| panic!("world {world}: the plan: {error}"));
        plan::install(plan.clone()).unwrap_or_else(|error| panic!("world {world}: {error}"));
        let mut gate = Refusals::default();
        check_head(&mut gate);
        check_streams(&mut gate, world);
        check_shape_builds(&mut gate, world);
        check_parts(&mut gate, &plan, world);
        check_routed_dedup(&mut gate, &plan, world);
        check_drafter_block(&mut gate);
        check_wo_a(&mut gate, &dir, world, &written, &plan);
        check_draft(&mut gate, world);
        check_prefill(&mut gate, world);
        refused.extend(
            gate.refused
                .into_iter()
                .map(|line| format!("world {world}: {line}")),
        );
    }
    assert!(
        refused.is_empty(),
        "the device-open gate refused:\n  {}",
        refused.join("\n  ")
    );
}

/// The row kernels are built for one stream count; the plan must state it, or the embedding
/// gather writes rows of a width no kernel reads.
fn check_streams(gate: &mut Refusals, world: usize) {
    let stated = plan::hc_mult();
    let built = sparkflash_deepseek_forward::tp4_forward::rows_streams();
    gate.require("embedding gather streams", stated == built, || {
        format!(
            "world {world}: the plan states {stated} hidden streams; the rows kernels are \
                 built for {built}"
        )
    });
}

/// Every second-shape build states the plan's width and stream count. The stager loads one of
/// these for every released cubin name it covers (`scripts/instances.py --world-builds`), so a row
/// copied from another shape compiles a body whose strides are that shape's. The vision tower's
/// `hidden` is its own width; its output width is the model's.
fn check_shape_builds(gate: &mut Refusals, world: usize) {
    let hidden = plan::hidden().to_string();
    let streams = plan::hc_mult().to_string();
    for row in instance_rows() {
        let Some(cubin) = row.get("cubin") else {
            continue;
        };
        if !cubin.contains("_s2") {
            continue;
        }
        let width_key = if row.get("family").map(String::as_str) == Some("vision") {
            "out_hidden"
        } else {
            "hidden"
        };
        for (key, want) in [(width_key, &hidden), ("hc", &streams)] {
            if let Some(stated) = row.get(key) {
                gate.require("shape build geometry", stated == want, || {
                    format!(
                        "world {world}: {cubin} {} states {key} = {stated}; the plan states {want}",
                        row.get("entry").map_or("?", String::as_str)
                    )
                });
            }
        }
    }
}

/// The certified head's buffer and the screen's block count, at the plan's width.
/// The drafter's dedup down launch states half its rank's intermediate, never a literal.
fn check_drafter_block(gate: &mut Refusals) {
    gate.require(
        "drafter routed block",
        DRAFT.contains("local_inter() / 2") && DRAFT.contains("(down_threads, 1, 1)"),
        || {
            "the drafter's dedup down launch does not derive its block from the plan's per-rank \
             intermediate"
                .to_owned()
        },
    );
}

/// The certified head's buffer and the screen's block count, at the plan's width.
fn check_head(gate: &mut Refusals) {
    let hidden = plan::hidden();
    let blocks = hidden / lm_head_ref::SCREEN_TILE;
    let want_bytes = plan::vocab_per_rank() * blocks * 2;
    gate.require(
        "head block maxima",
        tp4_tail::w_block_max_bytes() == want_bytes,
        || {
            format!(
                "the tail allocates {} bytes for the block maxima; the plan's {}-wide rows \
                 need {} ({} rows x {} blocks x 2)",
                tp4_tail::w_block_max_bytes(),
                hidden,
                want_bytes,
                plan::vocab_per_rank(),
                blocks
            )
        },
    );
    gate.require(
        "head block maxima",
        lm_head_ref::blocks_per_row(hidden) == blocks,
        || {
            format!(
                "the derivation states {} blocks a row at width {hidden}; the plan's width is \
                 {blocks} blocks",
                lm_head_ref::blocks_per_row(hidden)
            )
        },
    );
    let rows = 2;
    let shard = vec![0u8; rows * hidden * 2];
    gate.require(
        "head block maxima",
        lm_head_ref::block_maxima_bf16(&shard, hidden).len() == rows * blocks * 2,
        || {
            format!(
                "the derivation over {rows} plan-wide rows wrote {} bytes; the tail's layout is \
                 {}",
                lm_head_ref::block_maxima_bf16(&shard, hidden).len(),
                rows * blocks * 2
            )
        },
    );
    // The screen is one certified body at 5,120 values a row. A shape that differs must serve the
    // full head; the harness that runs this shape states that. A gate that ignored the width would
    // let the released block count screen a 4,096-wide row.
    let certified = head_screen_ref::K_BLOCKS == hidden / 32;
    gate.require(
        "head screen width",
        certified || harness_states("SPARKFLASH_HEAD_SCREEN", "0"),
        || {
            format!(
                "the screen is built for {} blocks of 32 and the plan's width needs {}; the arm \
                 must run with SPARKFLASH_HEAD_SCREEN=0 and does not state it",
                head_screen_ref::K_BLOCKS,
                hidden / 32
            )
        },
    );
}

/// The drafter's own routed FFN launches one of the same instances, so its block is the plan's
/// too: the source is read here because the drafter is behind the cuda feature and the gate is not.
const DRAFT: &str = include_str!("../src/tp4_forward/draft.rs");

/// The instance catalogue the build compiles from, read so the gate can compare what one cubin was
/// compiled for with what the plan states. A body's launch bounds and its operand strides come
/// from these values, so a row built for another shape answers a launch the driver refuses.
const INSTANCES: &str = include_str!("../../../kernels/instances.toml");

/// One `[[instance]]` row's key/value pairs.
fn instance_rows() -> Vec<std::collections::BTreeMap<String, String>> {
    let mut rows = Vec::new();
    let mut row: Option<std::collections::BTreeMap<String, String>> = None;
    for line in INSTANCES.lines() {
        let line = line.trim();
        if line == "[[instance]]" {
            rows.extend(row.take());
            row = Some(std::collections::BTreeMap::new());
            continue;
        }
        let Some(current) = row.as_mut() else {
            continue;
        };
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        current.insert(key.trim().to_owned(), value.to_owned());
    }
    rows.extend(row);
    rows
}

/// The routed dedup down body's block is the plan's, not the released 288: the instance compiles
/// its launch bounds from one rank's intermediate, so the cubin the plan selects must be one built
/// for that intermediate, and the block must be half of it.
fn check_routed_dedup(gate: &mut Refusals, plan: &EnginePlan, world: usize) {
    let rows = instance_rows();
    for layer in plan.layers() {
        let Some(requirement) = layer
            .requirements
            .iter()
            .find(|requirement| requirement.part == Part::Moe)
        else {
            continue;
        };
        if requirement.geometry.experts == 0 {
            continue;
        }
        // The plan names an image; the staged cubin for this world is that image's build, and the
        // catalogue states which one from the world's suffix.
        if !plan
            .symbols()
            .iter()
            .any(|entry| entry.part == Part::Moe && !entry.symbols.is_empty())
        {
            gate.require("routed dedup block", false, || {
                format!("layer {}: the plan names no MoE image", layer.index)
            });
            continue;
        }
        let geometry = MoeRoutedGeometry::new(
            requirement.geometry.hidden,
            plan::moe_inter_dim(),
            world,
            requirement.geometry.experts,
            requirement.geometry.top_k,
        );
        let geometry = match gate.report("routed dedup block", geometry) {
            Some(geometry) => geometry,
            None => continue,
        };
        let local = geometry.local_inter();
        let block = geometry.dedup_down_threads();
        let suffix = match world {
            1 => "_s2w1",
            2 => "_s2w2",
            4 => "_s2",
            _ => "",
        };
        let built = rows.iter().find(|row| {
            row.get("entry").map(String::as_str) == Some("opt_moe_down_dedup_sm121")
                && row
                    .get("cubin")
                    .is_some_and(|cubin| cubin.ends_with(suffix))
                && !suffix.is_empty()
        });
        gate.require("routed dedup block", built.is_some(), || {
            format!(
                "layer {}: kernels/instances.toml has no opt_moe_down_dedup_sm121 row whose cubin \
                 ends {suffix} for world {world}",
                layer.index
            )
        });
        let Some(built) = built else { continue };
        let compiled = built
            .get("intermediate")
            .and_then(|value| value.parse::<usize>().ok());
        gate.require("routed dedup block", compiled == Some(local), || {
            format!(
                "layer {}: the plan states {local} intermediate columns a rank and the world's \
                 dedup down cubin was compiled for {compiled:?}; the launch would state a block \
                 its launch bounds refuse",
                layer.index
            )
        });
        gate.require("routed dedup block", block * 2 == local, || {
            format!(
                "layer {}: the dedup down block is {block}, not half the {local} intermediate \
                 columns a rank holds",
                layer.index
            )
        });
    }
}

/// Every part the plan executes: its stated format, its selected symbols, the shared expert's own
/// format and, for the routed experts, the geometry an instance was built for.
fn check_parts(gate: &mut Refusals, plan: &EnginePlan, world: usize) {
    for layer in plan.layers() {
        for requirement in &layer.requirements {
            let part = requirement.part;
            let stored = plan.stored_format(part, Some(layer.index));
            gate.require(
                "part stored format",
                stored == Some(requirement.format),
                || {
                    format!(
                        "layer {} {part:?}: the plan states {stored:?} and the part's geometry \
                         states {:?}",
                        layer.index, requirement.format
                    )
                },
            );
            // The plan groups the catalogue by part, not by layer: one part kind selects one
            // symbol set at this target, and a part with no symbols is a part the open cannot
            // launch.
            let symbols = plan.symbols().iter().find(|entry| entry.part == part);
            gate.require(
                "part selected kernel",
                symbols.is_some_and(|entry| !entry.symbols.is_empty() && !entry.image.is_empty()),
                || {
                    format!(
                        "layer {} {part:?}: the plan names no kernel symbols for the part",
                        layer.index
                    )
                },
            );
            if part == Part::Moe && requirement.geometry.shared_experts > 0 {
                let stated = plan.shared_expert_format(layer.index);
                gate.require("shared expert format", stated.is_some(), || {
                    format!(
                        "layer {}: the plan states no format for its {} shared experts",
                        layer.index, requirement.geometry.shared_experts
                    )
                });
                let forward = plan::shared_expert_format(layer.index);
                if let Some(format) = gate.report("shared expert format", forward) {
                    gate.require(
                        "shared expert format",
                        stated == Some(format) && format.block() == 32,
                        || {
                            format!(
                                "layer {}: the forward states {format:?} and the plan {stated:?}",
                                layer.index
                            )
                        },
                    );
                }
            }
            if part == Part::Moe && requirement.geometry.experts > 0 {
                let refused = MoeRoutedGeometry::new(
                    requirement.geometry.hidden,
                    plan::moe_inter_dim(),
                    world,
                    requirement.geometry.experts,
                    requirement.geometry.top_k,
                );
                if let Some(geometry) = gate.report("routed device geometry", refused) {
                    gate.require(
                        "routed device geometry",
                        (
                            geometry.dim,
                            geometry.inter,
                            geometry.world,
                            geometry.experts,
                            geometry.top_k,
                        ) == (
                            requirement.geometry.hidden,
                            plan::moe_inter_dim(),
                            world,
                            requirement.geometry.experts,
                            requirement.geometry.top_k,
                        ),
                        || {
                            format!(
                                "layer {}: the geometry the plan states is not the one the body \
                                 was built for: {geometry:?}",
                                layer.index
                            )
                        },
                    );
                }
            }
        }
    }
}

/// The attention output projection is one rank's rows at every world, for both the stored and the
/// B12x operand layout.
fn check_wo_a(
    gate: &mut Refusals,
    dir: &Path,
    world: usize,
    written: &TensorPlan,
    plan: &EnginePlan,
) {
    let entries = written
        .specs()
        .iter()
        .filter(|spec| spec.name.ends_with("attn.wo_a.weight"))
        .map(|spec| {
            HeaderEntry::new(spec.name.clone(), spec.dtype.name(), spec.shape.clone())
                .with_format(tp4_resident::tensor_format(plan, spec.dtype.name()))
        });
    let resident = match ResidentPlan::plan(0, world, entries) {
        Ok(resident) => resident,
        Err(error) => {
            gate.require("wo_a per-rank stored bytes", false, || {
                format!("{dir:?}: the resident plan: {error}")
            });
            return;
        }
    };
    let mut seen = 0usize;
    for shard in resident.shards() {
        if !shard.name.ends_with("attn.wo_a.weight") {
            continue;
        }
        seen += 1;
        let whole_rows = shard.stored_shape.first().copied().unwrap_or(0);
        let inner = shard.stored_shape.get(1).copied().unwrap_or(0);
        gate.require(
            "wo_a per-rank stored bytes",
            whole_rows.is_multiple_of(world)
                && shard.shard_shape == vec![whole_rows / world, inner]
                && shard.shard_bytes * world as u64 == shard.stored_bytes,
            || {
                format!(
                    "{}: stored {:?} ({} bytes), this rank {:?} ({} bytes) at world {world}",
                    shard.name,
                    shard.stored_shape,
                    shard.stored_bytes,
                    shard.shard_shape,
                    shard.shard_bytes
                )
            },
        );
        match shard.operand_layout() {
            OperandLayout::B12x {
                cols,
                inner: got_inner,
                weight_bytes,
                block,
                ..
            } => {
                gate.require(
                    "wo_a per-rank stored bytes",
                    (cols, got_inner, block) == (whole_rows / world, inner, 32)
                        && weight_bytes == shard.shard_bytes,
                    || {
                        format!(
                            "{}: the B12x operand is {cols} x {got_inner} with {weight_bytes} \
                             bytes (block {block}); the rank's slice is {} x {inner} with {} bytes",
                            shard.name,
                            whole_rows / world,
                            shard.shard_bytes
                        )
                    },
                );
            }
            other => gate.require("wo_a per-rank stored bytes", false, || {
                format!(
                    "{}: the plan states {other:?}; the body reads a B12x tile of this rank's rows",
                    shard.name
                )
            }),
        }
    }
    gate.require("wo_a per-rank stored bytes", seen > 0, || {
        "the resident plan holds no attention output projection".to_owned()
    });
}

/// The DSpark depth and every MTP stage's tensors: the stage the draft block reads is the plan's.
fn check_draft(gate: &mut Refusals, world: usize) {
    let block = plan::dspark_block_size();
    let stages = plan::n_mtp_layers();
    gate.require("DSpark depth", block > 0 && stages > 0, || {
        format!("the plan states a {block}-token block over {stages} stages")
    });
    let depth = gate.report("DSpark depth", plan::draft_depth(None));
    gate.require("DSpark depth", depth == Some(block), || {
        format!("an unstated depth resolves to {depth:?}; the plan's draft block is {block}",)
    });
    gate.require(
        "DSpark depth",
        plan::draft_depth(Some(block)) == Ok(block) && plan::draft_depth(Some(block + 1)).is_err(),
        || {
            format!(
                "the depth authority accepts {block} and refuses {}",
                block + 1
            )
        },
    );
    let last = gate.report("drafter stage claims", plan::mtp_last_stage());
    gate.require(
        "drafter stage claims",
        last == stages.checked_sub(1),
        || {
            format!(
                "the last-markov stage is {last:?}; the plan states {stages} stages at world {world}"
            )
        },
    );
    let Some(last) = last else {
        return;
    };
    let experts = plan::dspark_n_routed_experts();
    for stage in 0..stages {
        for tail in [
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "ffn.gate.weight",
            "ffn.gate.bias",
        ] {
            let _ = gate.report("drafter stage claims", plan::mtp_claim_result(stage, tail));
        }
        for prefix in [
            "attn.wq_a",
            "attn.wkv",
            "attn.wq_b",
            "attn.wo_a",
            "attn.wo_b",
            "ffn.shared_experts.w1",
            "ffn.shared_experts.w3",
            "ffn.shared_experts.w2",
        ] {
            for suffix in ["weight", "scale"] {
                let tail = format!("{prefix}.{suffix}");
                let _ = gate.report("drafter stage claims", plan::mtp_claim_result(stage, &tail));
            }
        }
        for expert in 0..experts {
            for leaf in ["w1", "w3", "w2"] {
                for suffix in ["weight", "scale"] {
                    let tail = format!("ffn.experts.{expert}.{leaf}.{suffix}");
                    let _ =
                        gate.report("drafter stage claims", plan::mtp_claim_result(stage, &tail));
                }
            }
        }
    }
    for tail in [
        "norm.weight",
        "markov_head.head.weight",
        "markov_head.embed.weight",
        "confidence_head.proj.weight",
    ] {
        let _ = gate.report("drafter stage claims", plan::mtp_claim_result(last, tail));
    }
    for tail in ["main_proj.weight", "main_proj.scale", "main_norm.weight"] {
        let _ = gate.report("drafter stage claims", plan::mtp_claim_result(0, tail));
    }
    let beyond = plan::mtp_claim_result(stages, "markov_head.head.weight");
    gate.require("drafter stage claims", beyond.is_err(), || {
        format!(
            "stage {stages} answers for a markov head: {:?}",
            beyond.ok()
        )
    });
}

/// The NVFP4 prefill cubins: the shape's width has a candidate, and the pin does not claim it.
fn check_prefill(gate: &mut Refusals, world: usize) {
    let local = plan::moe_inter_dim() / world;
    let candidates = [
        (512usize, "nvfp4_prefill_sm121_i512"),
        (256, "nvfp4_prefill_sm121_i256"),
        (128, "nvfp4_prefill_sm121_i128"),
    ];
    let candidate = candidates
        .iter()
        .find(|(width, _)| *width == local)
        .map(|&(_, name)| name);
    gate.require("NVFP4 prefill candidate", candidate.is_some(), || {
        format!(
            "the expert width is {local} rows a rank at world {world}; the prefill cubins are \
             built for {}",
            candidates
                .iter()
                .map(|(width, _)| *width)
                .map(|width| width.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    });
    let Some(name) = candidate else {
        return;
    };
    match baseline::pin(name) {
        Ok(None) => {}
        Ok(Some(pin)) => gate.require("NVFP4 prefill candidate", false, || {
            format!(
                "{name} is pinned as {}; a second shape's cubin is built from the instance list \
                 and is not in the released manifest",
                pin.name
            )
        }),
        Err(error) => gate.require("NVFP4 prefill candidate", false, || {
            format!("{name}: the baseline: {error}")
        }),
    }
    gate.require(
        "NVFP4 prefill candidate",
        harness_states("SPARKFLASH_UNPINNED_SECOND_SHAPE", "1"),
        || {
            "the harness loads an unpinned second-shape cubin without stating \
             SPARKFLASH_UNPINNED_SECOND_SHAPE=1"
                .to_owned()
        },
    );
}
