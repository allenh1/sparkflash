//! The tensor plan, the shipped configuration, and the source that replaces the checkpoint.
//!
//! Three of these tests read the released checkpoint's *index* - a 7.4 MB JSON file that lists names,
//! not weights - when it is present on the host, and print a skip line when it is not. The index is
//! the authoritative statement of what the checkpoint contains, so the plan is checked against it
//! rather than against itself.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::{DType, SCALE_BLOCK, dense_scale_shape, expert_scale_shape};
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource, check_source_against_plan};

/// Where the released checkpoint's index lives on this host, if anywhere.
fn released_index() -> Option<PathBuf> {
    let candidates = ["/home/user/dsv41-research/model.safetensors.index.json"];
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.is_file())
}

/// The names in the released index, or `None` with a printed skip when it is absent.
fn index_names() -> Option<BTreeSet<String>> {
    let path = match released_index() {
        Some(path) => path,
        None => {
            println!("SKIP: the released checkpoint index is not on this host");
            return None;
        }
    };
    let text = std::fs::read_to_string(&path).expect("the index must be readable");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("the index must be JSON");
    let map = parsed["weight_map"]
        .as_object()
        .expect("the index must carry a weight_map");
    println!("read {} names from {}", map.len(), path.display());
    Some(map.keys().cloned().collect())
}

#[test]
fn the_shipped_configuration_parses_and_validates() {
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    assert_eq!(config.dim, 5120);
    assert_eq!(config.n_layers, 40);
    assert_eq!(config.n_mtp_layers, 3);
    assert_eq!(config.vocab_size, 129_280);
    assert_eq!(config.n_heads, 64);
    assert_eq!(config.head_dim, 512);
    assert_eq!(config.rope_head_dim, 64);
    assert_eq!(config.n_routed_experts, 384);
    assert_eq!(config.n_activated_experts, 6);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.q_lora_rank, 1280);
    assert_eq!(config.o_groups, 8);
    assert_eq!(config.o_lora_rank, 1024);
    assert_eq!(config.hc_mult, 4);
    assert_eq!(config.hc_sinkhorn_iters, 20);
    assert_eq!(config.hc_mix_count(), 24);
    assert_eq!(config.hc_width(), 20_480);
    assert_eq!(config.engram_layer_ids, vec![1, 14]);
    assert_eq!(config.engram_num_embeddings, vec![384_006_168, 384_016_682]);
    assert_eq!(config.engram_n_heads, 8);
    assert_eq!(config.engram_head_dim, 256);
    assert_eq!(config.compress_ratios.len(), 43);
    assert_eq!(config.gate_temp, 1.0);
    assert!(config.norm_topk_prob);
}

#[test]
fn the_backbone_and_the_mtp_stages_route_differently() {
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    assert_eq!(config.moe_config(0), (384, 6));
    assert_eq!(config.moe_config(39), (384, 6));
    assert_eq!(config.moe_config(40), (128, 3));
    assert_eq!(config.moe_config(42), (128, 3));
    assert!(!config.is_mtp(39));
    assert!(config.is_mtp(40));
    assert_eq!(config.engram_rows(1), Some(384_006_168));
    assert_eq!(config.engram_rows(14), Some(384_016_682));
    assert_eq!(config.engram_rows(0), None);
    assert_eq!(config.compress_ratio(20), Some(1));
    assert_eq!(config.compress_ratio(42), Some(0));
}

#[test]
fn a_configuration_that_contradicts_itself_is_refused() {
    let mut config = ModelConfig::tiny();
    config.compress_ratios.push(1);
    let error = config
        .validate()
        .expect_err("a stray ratio must be refused");
    assert!(format!("{error}").contains("compress_ratios"), "{error}");

    let mut config = ModelConfig::tiny();
    config.n_shared_experts = 2;
    let error = config
        .validate()
        .expect_err("two shared experts must be refused");
    assert!(format!("{error}").contains("n_shared_experts"), "{error}");

    let mut config = ModelConfig::tiny();
    config.engram_num_embeddings.clear();
    let error = config
        .validate()
        .expect_err("an Engram layer with no row count must be refused");
    assert!(
        format!("{error}").contains("engram_num_embeddings"),
        "{error}"
    );

    let mut config = ModelConfig::tiny();
    config.kv_source_layers = vec![99];
    let error = config
        .validate()
        .expect_err("a KV source outside the backbone must be refused");
    assert!(format!("{error}").contains("kv_source_layers"), "{error}");

    // An MTP target has to NAME a backbone layer. The tiny configuration used to target layer 2 of a
    // 2-layer backbone, which is one past the end, and nothing noticed because the plan only ever uses
    // the number of targets. The released model targets 37, 38 and 39 of 40 layers.
    let mut config = ModelConfig::tiny();
    config.dspark_target_layer_ids = vec![2];
    let error = config
        .validate()
        .expect_err("a target layer past the end of the backbone must be refused");
    assert!(
        format!("{error}").contains("dspark_target_layer_ids"),
        "{error}"
    );

    // The tiny configuration is now a structurally valid model, which is what makes it usable as a
    // stand-in: every identifier in it names something that exists.
    assert_eq!(ModelConfig::tiny().dspark_target_layer_ids, vec![1]);
    ModelConfig::tiny()
        .validate()
        .expect("the tiny configuration must be a valid model");
}

#[test]
fn the_plan_reproduces_the_released_counts() {
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    let layers = plan.under("layers.").count();
    let mtp = plan.under("mtp.").count();
    assert_eq!(
        layers, 93_415,
        "the index holds 93,415 tensors under layers.*"
    );
    assert_eq!(mtp, 2_401, "the index holds 2,401 tensors under mtp.*");
    assert_eq!(plan.len(), 95_819, "the language-model namespaces");
    assert_eq!(
        plan.len() - layers - mtp,
        3,
        "embed, head and the final norm"
    );
    assert_eq!(plan.under("layers.0.").count(), 2_334);
    assert_eq!(plan.under("mtp.0.").count(), 801);
    println!(
        "plan bytes: {} over {} tensors",
        plan.total_bytes(),
        plan.len()
    );
}

/// The bytes the image path occupies, derived from the vision and aligner constructors: 32 blocks of
/// (norm1 1024, norm2 1024, wqkv.weight 3072x1024, wqkv.bias 3072, wo.weight 1024x1024, wo.bias 1024,
/// mlp.w1 5632x1024 fused gate+up, mlp.w2 1024x2816) at two bytes each, plus patch_embed (1024x588 and
/// 1024), vision.norm (1024), the aligner (5120x9216, 5120x5120 and two 5120 biases) and the three
/// 5120-wide image delimiters.
const IMAGE_PATH_BYTES: u128 = 970_536_960;

/// The released index metadata total: the exact sum of stored tensor bytes.
const RELEASED_TOTAL_BYTES: u128 = 510_286_023_000;

#[test]
fn the_plan_bytes_are_exact_not_approximately_right() {
    // This is the strongest check in the crate and it is exact by construction: a safetensors index
    // `metadata.total_size` is the sum of stored tensor bytes, so the language-model plan plus the
    // image path the plan deliberately omits must equal it to the byte. A tolerance would have hidden
    // the one dtype this plan had wrong (the Engram gate weights, 163,840 bytes of it), which is
    // exactly what a 0.5% band did before this test replaced it.
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    let bytes = plan.total_bytes();
    assert_eq!(
        bytes, 509_315_486_040,
        "the language-model bytes are pinned; a dtype or shape change moves this number"
    );
    assert_eq!(
        bytes + IMAGE_PATH_BYTES,
        RELEASED_TOTAL_BYTES,
        "the plan plus the image path must equal the released total_size exactly"
    );
}

#[test]
fn the_released_shapes_and_dtypes_are_pinned() {
    // A transposed wq_b, wkv, wo_a, wo_b or gate is name-identical and, for the FP8 pairs, byte-count
    // invariant: every other test in this file passes with those shapes transposed. So the shapes and
    // the dtypes are pinned here, tensor by tensor, against values derived from the module
    // constructors in inference/model.py.
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    let pinned: [(&str, DType, &[usize]); 30] = [
        ("embed.weight", DType::BF16, &[129_280, 5120]),
        ("head.weight", DType::BF16, &[129_280, 5120]),
        ("norm.weight", DType::BF16, &[5120]),
        ("layers.0.attn.wq_a.weight", DType::F8E4M3, &[1280, 5120]),
        ("layers.0.attn.wq_b.weight", DType::F8E4M3, &[32_768, 1280]),
        ("layers.0.attn.wkv.weight", DType::F8E4M3, &[512, 5120]),
        ("layers.0.attn.wo_a.weight", DType::F8E4M3, &[8192, 4096]),
        ("layers.0.attn.wo_b.weight", DType::F8E4M3, &[5120, 8192]),
        ("layers.0.attn.attn_sink", DType::F32, &[64]),
        ("layers.0.attn.q_norm.weight", DType::BF16, &[1280]),
        ("layers.0.attn.kv_norm.weight", DType::BF16, &[512]),
        ("layers.0.attn_norm.weight", DType::BF16, &[5120]),
        ("layers.0.hc_attn_fn", DType::F32, &[24, 20_480]),
        ("layers.0.hc_attn_base", DType::F32, &[24]),
        ("layers.0.hc_attn_scale", DType::F32, &[3]),
        ("layers.0.ffn.gate.weight", DType::BF16, &[384, 5120]),
        ("layers.0.ffn.gate.bias", DType::F32, &[384]),
        (
            "layers.0.ffn.experts.0.w1.weight",
            DType::F4E2M1,
            &[2304, 5120],
        ),
        (
            "layers.0.ffn.experts.0.w2.weight",
            DType::F4E2M1,
            &[5120, 2304],
        ),
        (
            "layers.0.ffn.shared_experts.w1.weight",
            DType::F8E4M3,
            &[2304, 5120],
        ),
        (
            "layers.2.attn.compressor.wkv.weight",
            DType::BF16,
            &[512, 5120],
        ),
        ("layers.2.attn.compressor.norm.weight", DType::BF16, &[512]),
        (
            "layers.2.attn.compressor.wgate.weight",
            DType::BF16,
            &[512, 5120],
        ),
        (
            "layers.2.attn.indexer.wq_b.weight",
            DType::F8E4M3,
            &[4096, 1280],
        ),
        (
            "layers.2.attn.indexer.weights_proj.weight",
            DType::BF16,
            &[32, 5120],
        ),
        ("layers.2.attn.indexer.wk.weight", DType::BF16, &[128, 512]),
        ("layers.1.engram.wkv.weight", DType::F8E4M3, &[25_600, 6144]),
        ("layers.1.engram.q_weight", DType::BF16, &[4, 5120]),
        ("mtp.0.main_proj.weight", DType::F8E4M3, &[5120, 15_360]),
        (
            "mtp.2.markov_head.embed.weight",
            DType::BF16,
            &[129_280, 256],
        ),
    ];
    for (name, dtype, shape) in pinned {
        let spec = plan
            .get(name)
            .unwrap_or_else(|| panic!("{name} is missing"));
        assert_eq!(spec.dtype, dtype, "{name} dtype");
        assert_eq!(spec.shape, shape.to_vec(), "{name} shape");
    }
    // The two Engram gate weights are the pair the byte identity corrected, so they are asserted
    // again where a reader looking for that story will find them.
    for layer in &config.engram_layer_ids {
        for gate in ["q_weight", "k_weight"] {
            let name = format!("layers.{layer}.engram.{gate}");
            assert_eq!(
                plan.get(&name).expect("planned").dtype,
                DType::BF16,
                "{name}"
            );
        }
    }
}

#[test]
fn the_tiny_plan_derives_its_widths_from_its_configuration() {
    // The released configuration has max_ngram_size 4 and three MTP target layers, so a hardcoded 3
    // in either place is invisible against it. The tiny configuration has 2 and 1, which is what makes
    // these two derivations testable at all.
    let config = ModelConfig::tiny();
    assert_eq!(config.engram_max_ngram_size, 2);
    assert_eq!(config.dspark_target_layer_ids.len(), 1);
    let orders = config.engram_max_ngram_size - 1;
    assert_eq!(
        orders, 1,
        "the tiny configuration has one n-gram order above the unigram"
    );
    let fetched = orders * config.engram_n_heads * config.engram_head_dim;
    assert_eq!(
        fetched, 16,
        "so the fetched width is 1 * 2 * 8, not the released 6144"
    );
    let plan = TensorPlan::language_model(&config);
    let wkv = plan.get("layers.1.engram.wkv.weight").expect("planned");
    assert_eq!(
        wkv.shape,
        vec![config.dim * (config.hc_mult + 1), fetched],
        "(max_ngram_size - 1) * n_heads * head_dim, not a hardcoded 3 * n_heads * head_dim"
    );
    let targets = config.dspark_target_layer_ids.len();
    assert_eq!(
        targets, 1,
        "the tiny configuration has one MTP target layer"
    );
    let main_proj = plan.get("mtp.0.main_proj.weight").expect("planned");
    assert_eq!(
        main_proj.shape,
        vec![config.dim, targets * config.dim],
        "len(dspark_target_layer_ids) * dim, not a hardcoded 3 * dim"
    );
}

#[test]
fn the_names_the_plan_leaves_out_are_the_vision_tower() {
    // The plan covers the language model. What it deliberately does not cover must be exactly the
    // image path, so a language-model tensor can never be dropped by accident and hide inside a
    // smaller count than the index.
    let Some(names) = index_names() else {
        return;
    };
    let excluded: Vec<&String> = names
        .iter()
        .filter(|name| {
            !(name.starts_with("layers.")
                || name.starts_with("mtp.")
                || matches!(
                    name.as_str(),
                    "embed.weight" | "head.weight" | "norm.weight"
                ))
        })
        .collect();
    assert_eq!(excluded.len(), 266);
    assert_eq!(
        excluded.iter().filter(|n| n.starts_with("vision.")).count(),
        259,
        "the vision tower"
    );
    assert_eq!(
        excluded
            .iter()
            .filter(|n| n.starts_with("aligner."))
            .count(),
        4,
        "the aligner"
    );
    for delimiter in ["image_start", "image_end", "image_newline"] {
        assert!(
            excluded.iter().any(|n| n.as_str() == delimiter),
            "{delimiter} is missing from the index"
        );
    }
}

#[test]
fn the_plan_names_are_the_released_index_names() {
    let Some(names) = index_names() else {
        return;
    };
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    let planned: BTreeSet<String> = plan.names().map(str::to_string).collect();
    let from_index: BTreeSet<String> = names
        .iter()
        .filter(|name| {
            name.starts_with("layers.")
                || name.starts_with("mtp.")
                || name.as_str() == "embed.weight"
                || name.as_str() == "head.weight"
                || name.as_str() == "norm.weight"
        })
        .cloned()
        .collect();
    let missing: Vec<&String> = from_index.difference(&planned).take(10).collect();
    let extra: Vec<&String> = planned.difference(&from_index).take(10).collect();
    assert!(
        from_index == planned,
        "the plan and the released index disagree: {} in the index but not planned (e.g. {missing:?}), \
         {} planned but not in the index (e.g. {extra:?})",
        from_index.difference(&planned).count(),
        planned.difference(&from_index).count()
    );
}

#[test]
fn the_two_scale_grids_follow_their_rules() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    // A dense weight uses the square block grid.
    let dense = plan
        .get("layers.0.attn.wq_a.scale")
        .expect("the scale exists");
    assert_eq!(dense.dtype, DType::F8E8M0);
    assert_eq!(
        dense.shape,
        vec![1, 2],
        "ceil(32/32) x ceil(64/32) over the 32x64 weight"
    );
    assert_eq!(dense_scale_shape(config.q_lora_rank, config.dim), [1, 2]);
    // An expert weight uses the row grid: one scale per row per 32 along K.
    let expert = plan
        .get("layers.0.ffn.experts.0.w1.scale")
        .expect("the expert scale exists");
    assert_eq!(expert.shape, vec![32, 2], "[out, ceil(in/32)]");
    assert_eq!(
        expert_scale_shape(config.moe_inter_dim, config.dim),
        [32, 2]
    );
    // The shared expert is a dense Linear, not an FP4 expert.
    let shared = plan
        .get("layers.0.ffn.shared_experts.w1.weight")
        .expect("the shared expert exists");
    assert_eq!(shared.dtype, DType::F8E4M3);
    let routed = plan
        .get("layers.0.ffn.experts.3.w2.weight")
        .expect("a routed expert exists");
    assert_eq!(routed.dtype, DType::F4E2M1);
    assert_eq!(routed.shape, vec![64, 32]);
    assert_eq!(routed.elements(), 2048);
    assert_eq!(
        routed.storage_bytes(),
        1024,
        "FP4 packs two values per byte"
    );
}

#[test]
fn the_tiny_plan_holds_every_family_the_engine_must_handle() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    for name in [
        "embed.weight",
        "norm.weight",
        "head.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.attn_sink",
        "layers.0.attn.q_norm.weight",
        "layers.0.attn.kv_norm.weight",
        "layers.0.hc_attn_fn",
        "layers.0.hc_ffn_scale",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.gate.bias",
        "layers.0.ffn.experts.0.w3.scale",
        "layers.0.ffn.shared_experts.w2.weight",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.compressor.wgate.weight",
        "layers.0.attn.indexer.wq_b.weight",
        "layers.0.attn.indexer.wk.weight",
        "layers.0.attn.indexer.k_norm.weight",
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.1.engram.wkv.weight",
        "layers.1.engram.q_weight",
        "layers.1.engram.k_weight",
        "mtp.0.attn.wq_a.weight",
        "mtp.0.hc_attn_fn",
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.0.main_norm.weight",
        "mtp.0.norm.weight",
        "mtp.0.markov_head.embed.weight",
        "mtp.0.markov_head.head.weight",
        "mtp.0.confidence_head.proj.weight",
        "mtp.0.ffn.experts.1.w1.weight",
    ] {
        assert!(plan.contains(name), "{name} is missing from the plan");
    }
    // Layer 1 of the tiny configuration is not a KV source, so it has no compressor and its indexer
    // has no key projection; layer 0 is a KV source and has both.
    assert!(!plan.contains("layers.1.attn.compressor.wkv.weight"));
    assert!(!plan.contains("layers.1.attn.indexer.wk.weight"));
    assert!(plan.contains("layers.1.attn.indexer.wq_b.weight"));
    // The tiny configuration has a single MTP stage, so it carries both the first-stage extras and
    // the last-stage ones.
    assert!(plan.contains("mtp.0.main_proj.weight"));
    assert!(plan.contains("mtp.0.markov_head.embed.weight"));
    assert!(plan.contains("mtp.0.confidence_head.proj.weight"));
    // The image-span routing bias exists only where a vision tower does (model.py:807), so the tiny
    // configuration - which has no vision - must NOT plan it, while the released one must. Before this
    // was conditional the plan demanded a tensor the tiny checkpoint could never contain, and nothing
    // exercised the difference because both configurations planned it.
    assert!(!config.vision_enabled());
    assert!(
        !plan.contains("layers.0.ffn.gate.bias_vl"),
        "no vision tower, no image bias"
    );
    let released = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    assert!(released.vision_enabled());
    assert!(
        TensorPlan::language_model(&released).contains("layers.0.ffn.gate.bias_vl"),
        "the released configuration has a vision tower and therefore the bias"
    );
}

#[test]
fn no_name_is_planned_twice() {
    let plan = TensorPlan::language_model(&ModelConfig::v41_flash().expect("the config parses"));
    let unique: BTreeSet<&str> = plan.names().collect();
    assert_eq!(unique.len(), plan.len(), "a name was planned twice");
}

#[test]
fn the_synthetic_source_is_a_pure_function_of_the_seed_and_the_name() {
    let plan = TensorPlan::language_model(&ModelConfig::tiny());
    let first = SyntheticWeights::from_plan(&plan, 7);
    let second = SyntheticWeights::from_plan(&plan, 7);
    let other = SyntheticWeights::from_plan(&plan, 8);
    let name = "layers.0.hc_attn_base";
    let a = first.read(name).expect("the tensor exists");
    let b = second.read(name).expect("the tensor exists");
    let c = other.read(name).expect("the tensor exists");
    assert_eq!(a, b, "the same seed and name must give the same bytes");
    assert_ne!(a, c, "a different seed must give different bytes");
    assert_eq!(
        a.len(),
        plan.get("layers.0.hc_attn_base")
            .expect("planned")
            .storage_bytes() as usize,
        "the bytes match the planned f32 coefficient count"
    );
    assert_eq!(
        first.read("layers.0.hc_attn_base").expect("exists").len(),
        8 * 4
    );
    // Different names under the same seed differ, so the name is part of the stream.
    let d = first
        .read("layers.0.hc_ffn_base")
        .expect("the tensor exists");
    assert_ne!(a, d);
}

#[test]
fn every_tensor_in_the_tiny_plan_reads_and_conforms() {
    // NOTE: this source CLONES the plan (synthetic.rs builds its specs from plan.specs()), so the
    // conformance check below cannot fail here and is not evidence of anything. What this test really
    // proves is that every planned tensor materialises at exactly its planned byte length, and that
    // the tiny model is small. The check being able to FIRE is tested separately, by
    // `the_conformance_check_rejects_a_source_that_deviates`, against a source that deviates on
    // purpose.
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let source = SyntheticWeights::from_plan(&plan, 1);
    let report = check_source_against_plan(&source, &plan);
    assert!(report.is_conformant(), "{:?}", report.mismatches);
    assert_eq!(report.matched, plan.len());
    for spec in plan.specs() {
        let bytes = source
            .read(&spec.name)
            .unwrap_or_else(|e| panic!("{} must read: {e}", spec.name));
        assert_eq!(bytes.len() as u128, spec.storage_bytes(), "{}", spec.name);
    }
    // The whole tiny model fits in well under a megabyte, which is the point of it.
    assert!(
        source.total_bytes() < 1024 * 1024,
        "{}",
        source.total_bytes()
    );
}

#[test]
fn the_synthetic_source_refuses_a_tensor_over_its_limit() {
    let plan = TensorPlan::language_model(&ModelConfig::v41_flash().expect("the config parses"));
    let source = SyntheticWeights::from_plan(&plan, 1).with_limit(1024);
    let error = source
        .read("embed.weight")
        .expect_err("a 1.3 GB embedding must not be materialised under a 1 KiB limit");
    match error {
        WeightError::TooLarge { bytes, limit, .. } => {
            assert_eq!(limit, 1024);
            assert_eq!(
                bytes,
                129_280 * 5120 * 2,
                "BF16 embedding: two bytes per element"
            );
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
    // A tensor under the limit still reads (attn_sink is one f32 per head).
    assert!(source.read("layers.0.attn.attn_sink").is_ok());
    assert_eq!(source.limit(), 1024);
    assert_eq!(source.seed(), 1);
}

#[test]
fn an_unknown_name_is_missing_rather_than_empty() {
    let plan = TensorPlan::language_model(&ModelConfig::tiny());
    let source = SyntheticWeights::from_plan(&plan, 1);
    let error = source
        .read("layers.0.attn.not_a_tensor")
        .expect_err("an unknown name must be missing");
    assert!(matches!(error, WeightError::Missing { .. }));
    assert!(!source.contains("layers.0.attn.not_a_tensor"));
    assert_eq!(source.names().len(), plan.len());
    assert_eq!(
        source.spec("norm.weight").expect("norm exists").shape,
        vec![64]
    );
}

#[test]
fn the_packed_formats_round_up() {
    assert_eq!(DType::F4E2M1.storage_bytes(1), 1);
    assert_eq!(DType::F4E2M1.storage_bytes(2), 1);
    assert_eq!(DType::F4E2M1.storage_bytes(3), 2);
    assert_eq!(DType::F4E2M1.storage_bytes(2304 * 5120), 2304 * 5120 / 2);
    assert_eq!(DType::F8E4M3.storage_bytes(7), 7);
    assert_eq!(DType::BF16.storage_bytes(4), 8);
    assert_eq!(DType::F32.storage_bytes(4), 16);
    assert_eq!(DType::F8E8M0.elements_per_byte(), 1);
    assert!(DType::F4E2M1.is_packed());
    assert!(DType::F8E8M0.is_scale());
    assert_eq!(SCALE_BLOCK, 32);
    assert_eq!(DType::F8E4M3.name(), "F8_E4M3");
}

/// A source that hands back whatever it is told to, so the conformance check can be shown firing.
#[derive(Debug)]
struct DeviatingSource {
    inner: SyntheticWeights,
    override_shape: BTreeMap<String, Vec<usize>>,
    override_dtype: BTreeMap<String, DType>,
    hidden: BTreeSet<String>,
}

impl WeightSource for DeviatingSource {
    fn names(&self) -> Vec<String> {
        self.inner
            .names()
            .into_iter()
            .filter(|name| !self.hidden.contains(name))
            .collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        let mut spec = self.inner.spec(name)?;
        if self.hidden.contains(name) {
            return Err(WeightError::Missing {
                name: name.to_string(),
            });
        }
        if let Some(shape) = self.override_shape.get(name) {
            spec.shape = shape.clone();
        }
        if let Some(dtype) = self.override_dtype.get(name) {
            spec.dtype = *dtype;
        }
        Ok(spec)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.inner.read(name)
    }
}

#[test]
fn the_conformance_check_rejects_a_source_that_deviates() {
    // Without this, the check is only ever run against a source built from the plan itself, where it
    // cannot disagree. Three ways to deviate, one per branch of the check, and each message must name
    // the tensor so a failure is actionable.
    let plan = TensorPlan::language_model(&ModelConfig::tiny());
    let mut override_shape = BTreeMap::new();
    override_shape.insert("layers.0.attn.wq_b.weight".to_string(), vec![1280, 32]);
    let mut override_dtype = BTreeMap::new();
    override_dtype.insert("layers.0.hc_attn_fn".to_string(), DType::BF16);
    let mut hidden = BTreeSet::new();
    hidden.insert("norm.weight".to_string());
    let source = DeviatingSource {
        inner: SyntheticWeights::from_plan(&plan, 1),
        override_shape,
        override_dtype,
        hidden,
    };
    let report = check_source_against_plan(&source, &plan);
    assert!(!report.is_conformant(), "the deviation must be caught");
    assert_eq!(report.mismatches.len(), 3, "{:?}", report.mismatches);
    assert_eq!(report.matched, plan.len() - 3);
    assert!(
        report
            .mismatches
            .iter()
            .any(|m| m.contains("layers.0.attn.wq_b.weight") && m.contains("shape")),
        "{:?}",
        report.mismatches
    );
    assert!(
        report
            .mismatches
            .iter()
            .any(|m| m.contains("layers.0.hc_attn_fn") && m.contains("dtype")),
        "{:?}",
        report.mismatches
    );
    assert!(
        report
            .mismatches
            .iter()
            .any(|m| m.contains("norm.weight") && m.contains("no tensor named")),
        "{:?}",
        report.mismatches
    );
    // A deviation the plan does not mention is not the check's business: it compares plan against
    // source, so an extra tensor in the source is not a mismatch here (the loader reports that).
    assert!(source.names().len() < plan.len());
}

#[test]
fn the_plan_matches_the_committed_index_digest() {
    // The released index is 7.4 MB, is not in this repository and is not on a CI runner, so the two
    // index-backed tests above skip there and the central name claim would go unchecked. This test
    // needs no index: it checks the DERIVED plan against a digest committed under tests/fixtures/,
    // which scripts/regen-tensor-fixture.py produces from the released index (the script records that
    // index file sha256 in the fixture). A drift in any name changes the digest and fails here.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v41_flash_tensor_digest.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be committed: {e}", path.display()));
    let mut fields = BTreeMap::new();
    for line in text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let (key, value) = line
            .split_once(' ')
            .unwrap_or_else(|| panic!("malformed fixture line {line:?}"));
        fields.insert(key.to_string(), value.to_string());
    }
    let number = |key: &str| -> u128 {
        fields
            .get(key)
            .unwrap_or_else(|| panic!("{key} is missing from the fixture"))
            .parse()
            .unwrap_or_else(|e| panic!("{key} is not a number: {e}"))
    };
    let field = |key: &str| -> &str {
        fields
            .get(key)
            .unwrap_or_else(|| panic!("{key} is missing from the fixture"))
    };

    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    assert_eq!(
        plan.len() as u128,
        number("language_model"),
        "the plan must hold every language-model tensor the index has"
    );
    let count = |prefix: &str| plan.names().filter(|name| name.starts_with(prefix)).count() as u128;
    assert_eq!(count("layers."), number("layers"));
    assert_eq!(count("mtp."), number("mtp"));
    assert_eq!(
        plan.len() as u128 - count("layers.") - count("mtp."),
        number("globals")
    );

    let mut names: Vec<&str> = plan.names().collect();
    names.sort_unstable();
    assert_eq!(
        format!(
            "{:016x}",
            sparkflash_engine::synthetic::fnv1a(names.join("\n").as_bytes())
        ),
        field("plan_names_fnv1a64"),
        "the derived plan names differ from the committed index digest"
    );

    // The decomposition the digest records, and the byte identity, both have to hold together:
    // the language model plus the image path is the released total, exactly.
    assert_eq!(
        number("language_model") + number("excluded"),
        number("total_tensors")
    );
    assert_eq!(
        number("vision") + number("aligner") + number("image_delimiters"),
        number("excluded")
    );
    assert_eq!(
        plan.total_bytes() + IMAGE_PATH_BYTES,
        number("metadata_total_size"),
        "the plan plus the image path must equal the index metadata total_size exactly"
    );
    // A digest is only as good as the file it came from, so the provenance line is asserted too.
    assert_eq!(field("index_sha256").len(), 64, "the source index sha256");
}

#[test]
fn the_committed_digest_came_from_this_index() {
    // The digest fixture is the only artefact in this repository tied to the released checkpoint, and
    // `the_plan_matches_the_committed_index_digest` trusts it. This test closes that loop where the
    // index exists: it regenerates the fixture from the index by the generator's own rule and compares
    // it line by line, so a fixture that no longer describes the checkpoint fails instead of quietly
    // certifying the plan against a stale digest. Every field is recomputed, the namespace counts
    // included - checking the digest alone would let a count move from one namespace to another and
    // still pass. It skips, loudly, on a machine without the index; the digest test above is what
    // covers that case.
    let index_path = "/home/user/dsv41-research/model.safetensors.index.json";
    if !std::path::Path::new(index_path).is_file() {
        println!("SKIP: the released checkpoint index is not on this host");
        return;
    }

    use sha2::Digest as _;
    let raw = std::fs::read(index_path).expect("the index must read");
    let parsed: serde_json::Value = serde_json::from_slice(&raw).expect("the index must be JSON");
    let weight_map = parsed["weight_map"]
        .as_object()
        .expect("weight_map is an object");
    let mut names: Vec<&str> = weight_map.keys().map(String::as_str).collect();
    names.sort_unstable();

    // The generator's rule, verbatim: five prefixes, and whatever is left is a global tensor.
    let count = |prefix: &str| names.iter().filter(|name| name.starts_with(prefix)).count();
    let vision = count("vision.");
    let aligner = count("aligner.");
    let image = count("image_");
    let layers = count("layers.");
    let mtp = count("mtp.");
    let globals = names.len() - (vision + aligner + image + layers + mtp);
    let excluded = vision + aligner + image;
    let language_model: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| {
            !name.starts_with("vision.")
                && !name.starts_with("aligner.")
                && !name.starts_with("image_")
        })
        .collect();
    assert_eq!(
        language_model.len(),
        layers + mtp + globals,
        "every index name must be classified exactly once"
    );

    let regenerated = [
        "# Generated by scripts/regen-tensor-fixture.py - do not edit by hand.".to_string(),
        format!("# source {index_path}"),
        format!("source_bytes {}", raw.len()),
        format!("index_sha256 {:x}", sha2::Sha256::digest(&raw)),
        format!("total_tensors {}", names.len()),
        format!("language_model {}", language_model.len()),
        format!("layers {layers}"),
        format!("mtp {mtp}"),
        format!("globals {globals}"),
        format!("excluded {excluded}"),
        format!("vision {vision}"),
        format!("aligner {aligner}"),
        format!("image_delimiters {image}"),
        format!(
            "metadata_total_size {}",
            parsed["metadata"]["total_size"].as_u64().expect("a number")
        ),
        format!(
            "plan_names_fnv1a64 {:016x}",
            sparkflash_engine::synthetic::fnv1a(language_model.join("\n").as_bytes())
        ),
    ]
    .join("\n")
        + "\n";

    let fixture = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v41_flash_tensor_digest.txt"),
    )
    .expect("the fixture must be committed");
    assert_eq!(
        regenerated.lines().count(),
        fixture.lines().count(),
        "the fixture has a different number of lines than the index produces"
    );
    for (number, (mine, committed)) in regenerated.lines().zip(fixture.lines()).enumerate() {
        assert_eq!(
            mine,
            committed,
            "line {} of the committed fixture is not what the index produces",
            number + 1
        );
    }
}

#[test]
fn a_zero_vision_layer_count_disables_the_vision_path() {
    // model.py:116-117 documents `vision_n_layers == 0` as disabling the vision path, and model.py:138
    // computes `vision_enabled` as a positive count. `Some(0)` is therefore the reference's own disabled
    // sentinel rather than an enabled tower with no layers, and reading it as enabled would plan 43
    // `ffn.gate.bias_vl` tensors that a checkpoint built that way cannot contain. An absent field and a
    // zero must mean the same thing here.
    let count_bias_vl = |plan: &sparkflash_engine::tensor::TensorPlan| {
        plan.names()
            .filter(|name| name.ends_with(".ffn.gate.bias_vl"))
            .count()
    };

    let enabled = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    assert!(enabled.vision_enabled());
    let enabled_plan = TensorPlan::language_model(&enabled);
    assert_eq!(
        count_bias_vl(&enabled_plan),
        enabled.n_layers + enabled.n_mtp_layers,
        "one image-span bias per backbone block and per MTP stage"
    );

    let mut disabled = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    disabled.vision_n_layers = Some(0);
    assert!(
        !disabled.vision_enabled(),
        "a zero layer count disables the vision path, exactly as an absent field does"
    );
    let disabled_plan = TensorPlan::language_model(&disabled);
    assert_eq!(count_bias_vl(&disabled_plan), 0, "no vision, no image bias");
    assert_eq!(
        disabled_plan.len() + 43,
        enabled_plan.len(),
        "the image bias is the only vision-conditioned tensor, so exactly 43 go"
    );
}
