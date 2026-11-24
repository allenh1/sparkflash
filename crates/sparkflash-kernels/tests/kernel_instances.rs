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
                "hc",
                "taps",
                "latent",
                "window",
                "markov_rank",
                "vocab_per_rank",
                "ranks",
                "heads",
                "patch",
                "out_hidden",
                "head_dim",
                "kv_lora",
                "window",
                "index_heads",
                "index_head_dim",
                "index_topk",
                "hsi_blocks",
                "hsi_block_tokens",
                "ring_capacity",
                "positions",
                "ratio",
                "heads",
                "o_groups",
                "o_lora_rank",
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
            entries.insert((instance.cubin.clone(), instance.entry.clone())),
            "{} is listed twice in {}",
            instance.entry,
            instance.cubin
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
            [
                "router", "routed", "seams", "drafter", "rows", "head", "vision", "mla", "select",
                "csa2", "woa",
            ]
            .contains(&instance.family.as_str()),
            "{} names family {:?}",
            instance.entry,
            instance.family
        );
        let keys: &[&str] = match instance.family.as_str() {
            "router" => &["hidden", "experts", "top_k"],
            "routed" => &["hidden", "intermediate", "padded", "experts", "top_k"],
            "seams" => &["hidden"],
            "drafter" => &["hidden", "hc", "taps", "latent", "window", "markov_rank"],
            "rows" => &["hidden", "intermediate", "experts", "top_k"],
            "head" => &["hidden", "vocab_per_rank", "ranks"],
            "vision" => &[
                "hidden",
                "heads",
                "patch",
                "intermediate",
                "out_hidden",
                "experts",
                "top_k",
            ],
            "mla" => &[
                "hidden",
                "heads",
                "head_dim",
                "kv_lora",
                "window",
                "index_heads",
                "index_head_dim",
                "index_topk",
                "hsi_blocks",
                "hsi_block_tokens",
                "ring_capacity",
                "positions",
            ],
            "select" => &[
                "hidden",
                "index_heads",
                "index_head_dim",
                "index_topk",
                "positions",
            ],
            "csa2" => &["hidden", "head_dim", "ratio", "positions"],
            "woa" => &["heads", "head_dim", "o_groups", "o_lora_rank"],
            other => panic!("{} names family {other}", instance.entry),
        };
        for key in keys {
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
fn the_seams_family_has_its_released_shapes_and_a_second_shape() {
    let rows = instances();
    let seams: Vec<&Instance> = rows
        .iter()
        .filter(|row| row.family == "seams" && !row.cubin.contains("_s2"))
        .collect();
    assert!(
        rows.iter()
            .any(|row| row.family == "seams" && row.cubin.contains("_s2")),
        "the seams family has a second-shape build"
    );
    assert!(!seams.is_empty(), "the seams family has instances");

    // Every released seam entry records the model hidden width, and the compiled mHC bodies also
    // name the residual-stream count the source needs.
    for row in &seams {
        if row.entry == "opt_mhc_seam_4096h2_sm121" {
            continue;
        }
        assert_eq!(row.shapes["hidden"], 5120, "{}", row.entry);
    }
    for entry in [
        "opt_mhc_seam",
        "opt_mhc_post",
        "mhc_entry_bf16",
        "mhc_ffn_seams",
        "mhc_attn_residual",
        "mhc_ffn_epilogue",
    ] {
        let row = seams
            .iter()
            .find(|row| row.entry == entry)
            .unwrap_or_else(|| panic!("{entry} is not in the seams family"));
        assert_eq!(row.shapes.get("hc").copied(), Some(4), "{entry}");
    }
    // The runtime-width entries record only the hidden width; they must not claim a stream count
    // they do not compile against.
    for entry in [
        "rope_rotate_bf16",
        "kv_norm_rope_bf16",
        "window_quant_fp8_bf16",
        "kv_append_bf16",
        "rope_rotate_rows_bf16",
        "w8a8_prologue_e4m3",
        "w8a8_prologue_bf16_e4m3",
        "shared_swiglu_f32",
        "rmsnorm_sm121",
    ] {
        let row = seams
            .iter()
            .find(|row| row.entry == entry)
            .unwrap_or_else(|| panic!("{entry} is not in the seams family"));
        assert!(!row.shapes.contains_key("hc"), "{entry} claims hc");
    }

    let second = seams
        .iter()
        .find(|row| row.entry == "opt_mhc_seam_4096h2_sm121")
        .expect("the seams second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_eq!(second.shapes["hc"], 2);
    assert_ne!(second.cubin, "opt_seams_sm121");
    let released = seams
        .iter()
        .find(|row| row.cubin == "opt_seams_sm121")
        .expect("the released seams cubin");
    assert_ne!(second.cubin, released.cubin, "the proof has its own cubin");

    let keys: BTreeSet<&str> = seams
        .iter()
        .filter_map(|row| row.catalogue.as_deref())
        .collect();
    for key in [
        "deepseek-hyperconnection",
        "deepseek-rope",
        "deepseek-quantize",
        "deepseek-rows-norm",
    ] {
        assert!(
            keys.contains(key),
            "the catalogue key {key} has no seam instance"
        );
    }
}

#[test]
fn the_drafter_family_has_its_released_shape_and_a_second_shape() {
    let rows = instances();
    let drafter: Vec<&Instance> = rows.iter().filter(|row| row.family == "drafter").collect();
    let released: Vec<&Instance> = drafter
        .iter()
        .copied()
        .filter(|row| row.cubin == "opt_draft_sm121")
        .collect();
    assert_eq!(
        released.len(),
        15,
        "the released drafter cubin carries fifteen entries"
    );
    for row in &released {
        assert_eq!(row.shapes["hidden"], 5120);
        assert_eq!(row.shapes["hc"], 4);
        assert_eq!(row.shapes["taps"], 3);
        assert_eq!(row.shapes["latent"], 512);
        assert_eq!(row.shapes["window"], 128);
        assert_eq!(row.shapes["markov_rank"], 256);
        assert_eq!(row.catalogue.as_deref(), Some("deepseek-mtp"));
    }

    let second = drafter
        .iter()
        .find(|row| row.entry == "opt_draft_taps_4096h2c2t_sm121")
        .expect("the second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_eq!(second.shapes["hc"], 2);
    assert_eq!(second.shapes["taps"], 2);
    assert_ne!(
        second.cubin, "opt_draft_sm121",
        "the proof has its own cubin"
    );
    assert!(
        second.catalogue.is_none(),
        "the proof is not a catalogue shape"
    );
}

#[test]
fn the_rows_family_has_its_released_shapes_and_a_second_shape() {
    let rows = instances();
    let family: Vec<&Instance> = rows.iter().filter(|row| row.family == "rows").collect();
    assert!(!family.is_empty(), "the rows family has instances");
    let released = family
        .iter()
        .find(|row| row.entry == "opt_rows_lm_head")
        .expect("the released rows LM head");
    assert_eq!(released.shapes["hidden"], 5120);
    assert_eq!(released.cubin, "opt_rows_sm121");
    assert_eq!(released.shapes["intermediate"], 576);
    assert_eq!(released.shapes["experts"], 384);
    assert_eq!(released.shapes["top_k"], 6);

    let second = family
        .iter()
        .find(|row| row.entry == "b12x_hostw_gemv_m1_bf16w_4096_sm121")
        .expect("the rows family's second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_ne!(second.cubin, "b12x_hostw_gemv_m1_sm121");
    assert!(
        second.catalogue.is_none(),
        "the proof claims no catalogue shape"
    );

    for key in [
        "deepseek-rows-norm",
        "deepseek-rows-mlp",
        "deepseek-rows-gemm",
        "deepseek-rows-activation",
        "deepseek-rows-combine",
    ] {
        assert!(
            family
                .iter()
                .any(|row| row.catalogue.as_deref() == Some(key)),
            "the catalogue key {key} has no rows instance"
        );
    }
}

#[test]
fn the_csa2_family_has_its_released_shapes_and_a_second_shape() {
    let rows = instances();
    let family: Vec<&Instance> = rows.iter().filter(|row| row.family == "csa2").collect();
    assert!(!family.is_empty(), "the csa2 family has instances");

    let released = family
        .iter()
        .find(|row| row.entry == "compressed_append_sm121")
        .expect("the released compression append");
    assert_eq!(released.shapes["hidden"], 5120);
    assert_eq!(released.shapes["head_dim"], 512);
    assert_eq!(released.shapes["ratio"], 1);
    assert_eq!(released.shapes["positions"], 1 << 21);
    assert_eq!(released.cubin, "compressed_append_sm121");
    assert_eq!(released.catalogue.as_deref(), Some("deepseek-csa2"));

    for entry in [
        "compressed_selected_fill_sm121",
        "pool_group_f32",
        "opt_rows_group_append",
    ] {
        let row = family
            .iter()
            .find(|row| row.entry == entry)
            .unwrap_or_else(|| panic!("{entry} has no csa2 instance"));
        assert_eq!(row.catalogue.as_deref(), Some("deepseek-csa2"), "{entry}");
        assert_eq!(row.shapes["head_dim"], 512, "{entry}");
    }

    let second = family
        .iter()
        .find(|row| row.entry == "compressed_append_256_sm121")
        .expect("the csa2 second-shape proof instance");
    assert_eq!(second.shapes["hidden"], 4096);
    assert_eq!(second.shapes["head_dim"], 256);
    assert_eq!(second.shapes["ratio"], 2);
    assert_eq!(second.shapes["positions"], 1 << 20);
    assert_eq!(second.cubin, "compressed_append_256_sm121");
    assert_ne!(second.cubin, released.cubin, "the proof has its own cubin");
    assert!(
        second.catalogue.is_none(),
        "the proof shape is not a released catalogue claim"
    );
    let baseline = pins().expect("the baseline parses");
    assert!(
        !baseline.iter().any(|pin| pin.name == second.cubin),
        "the proof cubin is not in the baseline"
    );
    assert!(
        baseline.iter().any(|pin| pin.name == released.cubin),
        "the released cubin keeps its baseline pin"
    );
}

#[test]
fn the_head_family_has_its_released_shape_and_a_second_shape() {
    let rows = instances();
    let heads: Vec<&Instance> = rows
        .iter()
        .filter(|row| row.family == "head" && !row.cubin.contains("_s2"))
        .collect();

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

#[test]
fn the_vision_family_has_its_released_shape_and_a_second_shape() {
    let rows = instances();
    let vision: Vec<&Instance> = rows.iter().filter(|row| row.family == "vision").collect();
    let released: Vec<&&Instance> = vision
        .iter()
        .filter(|row| row.cubin == "opt_vision_sm121")
        .collect();
    assert_eq!(
        released.len(),
        10,
        "the released vision image's ten entries"
    );
    assert_eq!(released[0].shapes["hidden"], 1024);
    assert_eq!(released[0].shapes["heads"], 16);
    assert_eq!(released[0].shapes["intermediate"], 2816);
    assert_eq!(released[0].shapes["out_hidden"], 5120);
    assert_eq!(released[0].shapes["experts"], 384);
    assert_eq!(released[0].shapes["top_k"], 6);
    let sides: BTreeSet<u64> = released.iter().map(|row| row.shapes["patch"]).collect();
    assert_eq!(sides, BTreeSet::from([14, 16]), "both patchify bodies");

    let second = vision
        .iter()
        .find(|row| row.entry == "opt_vision_span_rows_4096_sm121")
        .expect("the second-shape proof instance");
    assert_eq!(second.shapes["out_hidden"], 4096);
    assert_eq!(second.shapes["hidden"], 1024);
    assert_ne!(
        second.cubin, "opt_vision_sm121",
        "the proof has its own cubin"
    );

    let keys: BTreeSet<&str> = vision
        .iter()
        .filter_map(|row| row.catalogue.as_deref())
        .collect();
    for key in [
        "deepseek-vision-patch-embed",
        "deepseek-vision-position",
        "deepseek-vision-norm",
        "deepseek-vision-attention",
        "deepseek-vision-mlp",
        "deepseek-vision-downsample",
        "deepseek-vision-merger",
    ] {
        assert!(
            keys.contains(key),
            "the catalogue key {key} has no vision instance"
        );
    }
}

#[test]
fn the_attention_and_selection_families_have_their_released_shapes_and_a_second_shape() {
    let rows = instances();
    let mla: Vec<&Instance> = rows.iter().filter(|row| row.family == "mla").collect();
    let released = mla
        .iter()
        .find(|row| row.cubin == "flash_mla_csa2_sm121")
        .expect("the released base MLA entry");
    assert_eq!(released.shapes["kv_lora"], 512);
    assert_eq!(released.shapes["heads"], 64);
    assert_eq!(released.shapes["head_dim"], 512);
    assert_eq!(released.shapes["window"], 128);
    assert_eq!(released.shapes["index_heads"], 32);
    assert_eq!(released.shapes["index_head_dim"], 128);
    assert_eq!(released.shapes["index_topk"], 512);
    assert_eq!(released.shapes["hsi_blocks"], 2048);
    assert_eq!(released.shapes["hsi_block_tokens"], 8);
    assert_eq!(released.shapes["ring_capacity"], 134);

    // The proof shape is the ring entry compiled for 32 query heads, a 256-id candidate list and
    // a 64-token window. Its cubin is not a released stem, so the baseline must not name it.
    let second = mla
        .iter()
        .find(|row| row.entry == "flash_mla_csa2_ring_32h256k_sm121")
        .expect("the second-shape proof instance");
    assert_eq!(second.shapes["heads"], 32);
    assert_eq!(second.shapes["index_heads"], 16);
    assert_eq!(second.shapes["index_topk"], 256);
    assert_eq!(second.shapes["window"], 64);
    assert_eq!(second.shapes["hsi_blocks"], 1024);
    assert_ne!(second.cubin, released.cubin, "the proof has its own cubin");

    let select: Vec<&Instance> = rows.iter().filter(|row| row.family == "select").collect();
    let indexer = select
        .iter()
        .find(|row| row.cubin == "opt_select_sm121")
        .expect("the released opt_select entries");
    assert_eq!(indexer.shapes["index_heads"], 32);
    assert_eq!(indexer.shapes["index_head_dim"], 128);
    assert_eq!(indexer.shapes["index_topk"], 512);
    assert_eq!(indexer.shapes["positions"], 1 << 21);

    let keys: BTreeSet<&str> = rows
        .iter()
        .filter_map(|row| row.catalogue.as_deref())
        .collect();
    for key in ["deepseek-mla", "deepseek-indexer"] {
        assert!(
            keys.contains(key),
            "the catalogue key {key} has no instance"
        );
    }
}

#[test]
fn the_woa_family_and_the_b12x_routed_sources_have_released_and_second_shapes() {
    let rows = instances();
    let baseline = pins().expect("the baseline parses");

    // The grouped attention output projection is the new family: five entries share one released
    // cubin and one geometry, and a second shape compiles the same five entries from its own
    // cubin. The reduction width is heads / o_groups * head_dim in both.
    let woa: Vec<&Instance> = rows.iter().filter(|row| row.family == "woa").collect();
    let released: Vec<&&Instance> = woa
        .iter()
        .filter(|row| row.cubin == "b12x_mxfp8_wo_a_sm121")
        .collect();
    assert_eq!(released.len(), 5, "the released wo_a cubin's five entries");
    for row in &released {
        assert_eq!(row.shapes["heads"], 64, "{}", row.entry);
        assert_eq!(row.shapes["head_dim"], 512, "{}", row.entry);
        assert_eq!(row.shapes["o_groups"], 8, "{}", row.entry);
        assert_eq!(row.shapes["o_lora_rank"], 1024, "{}", row.entry);
        assert!(row.catalogue.is_none(), "{} is loaded by symbol", row.entry);
    }
    let second: Vec<&&Instance> = woa
        .iter()
        .filter(|row| row.cubin == "b12x_mxfp8_wo_a_sm121_s2")
        .collect();
    assert_eq!(
        second.len(),
        5,
        "the second-shape wo_a cubin's five entries"
    );
    for row in &second {
        assert_eq!(row.shapes["heads"], 32, "{}", row.entry);
        assert_eq!(row.shapes["head_dim"], 512, "{}", row.entry);
        assert_eq!(row.shapes["o_groups"], 4, "{}", row.entry);
        assert_eq!(row.shapes["o_lora_rank"], 256, "{}", row.entry);
        assert!(
            !baseline.iter().any(|pin| pin.name == row.cubin),
            "{} is a proof cubin",
            row.cubin
        );
    }

    // The two MXFP4 routed-expert sources take the rank's real width, the padded activation row,
    // the expert count and the selected slots from the instance.
    for source in ["b12x_mxfp4_moe_sm121.cu", "b12x_mxfp4_moe_stored_sm121.cu"] {
        let row = rows
            .iter()
            .find(|row| row.source.ends_with(source) && !row.cubin.ends_with("_s2"))
            .unwrap_or_else(|| panic!("{source} has no released instance"));
        assert_eq!(row.family, "routed", "{source}");
        assert_eq!(row.shapes["hidden"], 5120, "{source}");
        assert_eq!(row.shapes["intermediate"], 576, "{source}");
        assert_eq!(row.shapes["padded"], 640, "{source}");
        assert_eq!(row.shapes["experts"], 384, "{source}");
        assert_eq!(row.shapes["top_k"], 6, "{source}");
        assert_eq!(row.catalogue.as_deref(), Some("deepseek-moe"), "{source}");
        assert!(
            baseline.iter().any(|pin| pin.name == row.cubin),
            "{source} keeps its released pin"
        );

        let proof = rows
            .iter()
            .find(|row| row.source.ends_with(source) && row.cubin.ends_with("_s2"))
            .unwrap_or_else(|| panic!("{source} has no second-shape instance"));
        assert_eq!(proof.shapes["hidden"], 4096, "{source}");
        assert_eq!(proof.shapes["intermediate"], 128, "{source}");
        assert_eq!(proof.shapes["padded"], 128, "{source}");
        assert_eq!(proof.shapes["experts"], 64, "{source}");
        assert_eq!(proof.shapes["top_k"], 4, "{source}");
        assert!(
            !baseline.iter().any(|pin| pin.name == proof.cubin),
            "{source}'s proof cubin is not in the baseline"
        );
    }
}

#[test]
fn the_routed_device_pair_is_built_for_the_released_shape_and_every_second_shape_world() {
    let rows = instances();
    let pair: Vec<&Instance> = rows
        .iter()
        .filter(|row| row.source.ends_with("moe_routed_device_sm121.cu"))
        .collect();
    // One compile group holds the stage and the combine; every geometry must state both entries,
    // because the forward resolves them from one cubin and the stage/combine kernels read the same
    // hidden width and top-k.
    for (cubin, world, hidden, intermediate, padded, experts, top_k) in [
        (
            "moe_routed_device_sm121",
            4u64,
            5120u64,
            576u64,
            640u64,
            384u64,
            6u64,
        ),
        ("moe_routed_device_sm121_s2", 4, 4096, 128, 128, 64, 4),
        ("moe_routed_device_sm121_s2w1", 1, 4096, 512, 512, 64, 4),
        ("moe_routed_device_sm121_s2w2", 2, 4096, 256, 256, 64, 4),
    ] {
        let entries: Vec<&&Instance> = pair.iter().filter(|row| row.cubin == cubin).collect();
        assert_eq!(
            entries.len(),
            2,
            "{cubin} holds the stage and combine entries"
        );
        for row in entries {
            assert_eq!(row.family, "routed", "{cubin}");
            assert_eq!(row.shapes["hidden"], hidden, "{cubin}");
            assert_eq!(
                row.shapes["intermediate"], intermediate,
                "{cubin}: the row states the per-rank width, so a world is a build"
            );
            assert_eq!(row.shapes["padded"], padded, "{cubin}");
            assert_eq!(row.shapes["experts"], experts, "{cubin}");
            assert_eq!(row.shapes["top_k"], top_k, "{cubin}");
            assert!(
                row.catalogue.is_none(),
                "{cubin} is loaded by symbol, not decoded by a catalogue rule"
            );
        }
        // The global intermediate the geometry carries is the per-rank width times the rank count
        // its cubin suffix states.
        let global = if hidden == 5120 { 2304 } else { 512 };
        assert_eq!(intermediate * world, global, "{cubin}");
    }
}

#[test]
fn the_second_shape_worlds_have_their_router_and_mxfp4_bodies() {
    let rows = instances();
    for world in [1usize, 2] {
        let suffix = format!("_s2w{world}");
        for source in [
            "moe_router_sm121.cu",
            "opt_router_sm121.cu",
            "b12x_mxfp4_moe_sm121.cu",
            "b12x_mxfp4_moe_stored_sm121.cu",
        ] {
            assert!(
                rows.iter()
                    .any(|row| row.source.ends_with(source) && row.cubin.ends_with(&suffix)),
                "{source} has no {world}-rank second-shape build, so staging would keep the released body"
            );
        }
    }
}

#[test]
fn the_cubin_plan_resolves_the_routed_device_at_4096_and_512() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = crate_dir.join("../../scripts/instances.py");
    for (cubin, intermediate) in [
        ("moe_routed_device_sm121_s2", 128),
        ("moe_routed_device_sm121_s2w1", 512),
        ("moe_routed_device_sm121_s2w2", 256),
    ] {
        let output = std::process::Command::new("python3")
            .arg(&script)
            .args(["--defines-for-cubin", cubin])
            .output()
            .expect("python3 runs scripts/instances.py");
        assert!(
            output.status.success(),
            "scripts/instances.py --defines-for-cubin {cubin}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let defines = String::from_utf8(output.stdout).expect("the plan is UTF-8");
        for needle in [
            "-DSPARK_INSTANCE_MOE_EXPERT_STAGE_SM121_HIDDEN=4096u",
            "-DSPARK_INSTANCE_MOE_EXPERT_STAGE_SM121_EXPERTS=64u",
            "-DSPARK_INSTANCE_MOE_EXPERT_STAGE_SM121_TOP_K=4u",
            "-DSPARK_INSTANCE_MOE_ROUTED_COMBINE_SM121_HIDDEN=4096u",
            "-DSPARK_INSTANCE_MOE_ROUTED_COMBINE_SM121_TOP_K=4u",
        ] {
            assert!(
                defines.contains(needle),
                "{cubin} is missing {needle}: {defines}"
            );
        }
        let per_rank =
            format!("-DSPARK_INSTANCE_MOE_EXPERT_STAGE_SM121_INTERMEDIATE={intermediate}u");
        assert!(
            defines.contains(&per_rank),
            "{cubin} is missing {per_rank}: {defines}"
        );
    }
}
