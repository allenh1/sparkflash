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

//! Architecture-driven tensor plans and the shared parts engine.

use sparkflash_model::architecture::{
    Activation, Architecture, Attention, AttentionKind, DenseMlp, Ffn, HyperConnection, Kda, Layer,
    Moe, Quantization, Residual, Rope, ScaleDType, Score, WeightFormat,
};
use sparkflash_model::{Target, TpPlan};
use sparkflash_ops::PartsDeviceRank;
use sparkflash_ops::engine_plan::{EnginePlan, PrecisionPolicy};
use sparkflash_ops::launchers::LayerLauncher;
use sparkflash_ops::resolver::PlanContext;
use sparkflash_ops::tensor_plan::{ArchitectureTensorPlan, Role};
use sparkflash_weights::{DType, TensorSpec, WeightError, WeightSource};

/// A spec-only source: the plan's names, with zero bytes of the right length.
struct Specs(Vec<TensorSpec>);

impl WeightSource for Specs {
    fn names(&self) -> Vec<String> {
        self.0.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.0
            .iter()
            .find(|spec| spec.name == name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.spec(name)?;
        Ok(vec![0; spec.storage_bytes() as usize])
    }
}

fn rope() -> Rope {
    Rope {
        theta: 10_000.0,
        rotated_dims: 0,
        scaling: None,
        mrope_section: None,
    }
}

fn hyper() -> Residual {
    Residual::Hyper(HyperConnection {
        streams: 4,
        lowrank: None,
        manifold: true,
        sinkhorn_iters: 20,
        epsilon: 1e-6,
    })
}

fn moe() -> Ffn {
    Ffn::Moe(Moe {
        routed_experts: 288,
        active_experts: 8,
        intermediate: 2048,
        shared_experts: 1,
        shared_intermediate: 2048,
        shared_gated: false,
        score: Score::Sigmoid,
        routed_scaling: 2.5,
        norm_topk: true,
        selection_bias: true,
        clamp: Some(10.0),
        activation: Activation::Silu,
    })
}

fn layer(index: usize, kind: AttentionKind, ffn: Ffn) -> Layer {
    Layer {
        index,
        attention: Attention {
            kind,
            heads: 64,
            head_dim: 128,
            window: None,
            rope: rope(),
            compression: None,
            indexer: None,
        },
        ffn,
        residual: hyper(),
        sides: Vec::new(),
    }
}

/// The synthetic hybrid fixture: KDA + GQA + DenseMlp + MoE in one architecture.
fn hybrid() -> Architecture {
    let kda = AttentionKind::Kda(Kda {
        conv_kernel: 4,
        heads: 128,
        head_dim: 128,
        lower_bound: -5.0,
    });
    let gqa = AttentionKind::Gqa {
        kv_heads: 8,
        output_gate: true,
    };
    Architecture {
        family: "glm5_next".to_owned(),
        architectures: vec!["SyntheticHybrid".to_owned()],
        hidden_size: 4096,
        vocab_size: 4096,
        max_position_embeddings: 4096,
        norm_epsilon: 1e-5,
        embedding_tied_to_head: false,
        attention_bias: false,
        layers: vec![
            layer(
                0,
                kda.clone(),
                Ffn::Dense(DenseMlp {
                    intermediate: 12_288,
                    activation: Activation::Silu,
                    clamp: Some(10.0),
                }),
            ),
            layer(
                1,
                gqa.clone(),
                Ffn::Dense(DenseMlp {
                    intermediate: 12_288,
                    activation: Activation::Silu,
                    clamp: Some(10.0),
                }),
            ),
            layer(2, gqa.clone(), moe()),
            layer(3, gqa, moe()),
        ],
        quantization: Quantization {
            dense_format: WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: ScaleDType::F32,
                dynamic_activations: true,
            },
            expert_format: WeightFormat::Fp8Tile {
                block_m: 128,
                block_n: 128,
                scale_dtype: ScaleDType::F32,
                dynamic_activations: true,
            },
            ..Quantization::default()
        },
        vision: None,
        mtp: None,
        tokens: sparkflash_model::architecture::Tokens::default(),
    }
}

fn policy() -> PrecisionPolicy {
    use sparkflash_ops::plan::Precision;
    PrecisionPolicy {
        dense: Precision::Fp8E4M3,
        experts: Precision::Fp8E4M3,
        mtp: Precision::Fp8E4M3,
        attention: Precision::Bf16,
    }
}

#[test]
fn a_hybrid_architecture_plans_every_part_it_names() {
    let architecture = hybrid();
    let tp = TpPlan::new(1, 0).expect("one rank");
    let tensors =
        ArchitectureTensorPlan::served_and_mtp(&architecture, tp).expect("the hybrid plan builds");
    assert!(tensors.len() > 1000, "{} tensors", tensors.len());
    assert!(tensors.role_bytes(Role::Attention) > 0);
    assert!(tensors.role_bytes(Role::DenseMlp) > 0);
    assert!(tensors.role_bytes(Role::Experts) > 0);
    let plan = EnginePlan::from_architecture(architecture, Target::Sm121, tp, policy())
        .expect("the execution plan builds");
    let source = Specs(tensors.plan().specs().to_vec());
    let rank = PartsDeviceRank::open(&plan, &source, PlanContext::SM121_CATALOGUE)
        .expect("every named part resolves");
    assert_eq!(rank.layers().len(), 4);
    assert!(!rank.kernels().is_empty());
    rank.boot_check().expect("the positive fixture boots");
    let scan = rank
        .layers()
        .iter()
        .flat_map(|layer| &layer.launchers)
        .find_map(|launcher| match launcher {
            LayerLauncher::LinearScan(scan) => Some(scan),
            _ => None,
        })
        .expect("the KDA layer has a scan launcher");
    assert_eq!((scan.key_heads, scan.heads), (128, 128));
    let mut scans = 0;
    let mut attentions = 0;
    let mut dense = 0;
    let mut moes = 0;
    for layer in rank.layers() {
        for launcher in &layer.launchers {
            match launcher {
                LayerLauncher::LinearScan(_) => scans += 1,
                LayerLauncher::Attention(_) => attentions += 1,
                LayerLauncher::Moe(_) => moes += 1,
                LayerLauncher::Other(kernel) => {
                    let _ = kernel;
                    dense += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(scans, 1, "one KDA layer");
    assert_eq!(attentions, 3, "three GQA layers");
    assert_eq!(moes, 2, "two MoE layers");
    assert!(dense > 0);
    assert!(!rank.boot_summary().is_empty());
}

#[test]
fn an_unresolvable_served_part_fails_the_boot_gate_in_both_passes() {
    // KDA's catalogue claims only the 128/128 head-width template; a 64-wide KDA layer is
    // planned but has no body. Opening the rank must refuse, naming both passes, instead of
    // returning Ok with the part merely collected.
    let mut architecture = hybrid();
    let AttentionKind::Kda(kda) = &mut architecture.layers[0].attention.kind else {
        panic!("the fixture's first layer is KDA");
    };
    kda.head_dim = 64;
    let tp = TpPlan::new(1, 0).expect("one rank");
    let tensors =
        ArchitectureTensorPlan::served_and_mtp(&architecture, tp).expect("the hybrid plan builds");
    let plan = EnginePlan::from_architecture(architecture, Target::Sm121, tp, policy())
        .expect("the execution plan builds");
    let source = Specs(tensors.plan().specs().to_vec());
    let error = PartsDeviceRank::open(&plan, &source, PlanContext::SM121_CATALOGUE)
        .expect_err("a 64/128 KDA has no catalogue body");
    let text = error.to_string();
    assert!(text.contains("kda_scan"), "{text}");
    assert!(
        text.contains("prefill") && text.contains("decode"),
        "{text}"
    );
    assert!(
        text.contains("2 part instance"),
        "both passes are aggregated: {text}"
    );
}

fn released_family() -> Architecture {
    Architecture::from_json(
        r#"{"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"vocab_size":248320,"num_hidden_layers":1,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"layer_types":["linear_attention"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"hc_count":4,"hc_lowrank":320,"indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}"#,
    )
    .expect("a synthetic released family document parses")
}

#[test]
fn an_fp8_expert_is_a_run_of_whole_tiles() {
    // D91: five 128-wide tiles over four ranks are 2/1/1/1, never 1.25 each.
    let mut architecture = released_family();
    architecture.quantization.expert_format = WeightFormat::Fp8Tile {
        block_m: 128,
        block_n: 128,
        scale_dtype: ScaleDType::Bf16,
        dynamic_activations: true,
    };
    let widths = [256usize, 128, 128, 128];
    for (rank, expected) in widths.into_iter().enumerate() {
        let tp = TpPlan::new(4, rank).expect("a rank");
        let plan = ArchitectureTensorPlan::served(&architecture, tp).expect("the FP8 plan builds");
        let prefix = "model.language_model.layers.0.mlp.experts.0";
        let gate = plan
            .rank_spec(&format!("{prefix}.gate_proj.weight"))
            .expect("the gate is planned");
        assert_eq!(gate.shape, vec![expected, 2560], "rank {rank} gate");
        let down = plan
            .rank_spec(&format!("{prefix}.down_proj.weight"))
            .expect("the down projection is planned");
        assert_eq!(down.shape, vec![2560, expected], "rank {rank} down");
        let scale = plan
            .rank_spec(&format!("{prefix}.gate_proj.weight_scale_inv"))
            .expect("the tile grid is planned");
        assert_eq!(scale.shape[0], expected / 128, "rank {rank} scale rows");
    }
}

#[test]
fn a_released_family_plan_names_its_family() {
    let architecture = released_family();
    let tp = TpPlan::new(1, 0).expect("one rank");
    let tensors =
        ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the family plan builds");
    assert!(!tensors.is_empty());
    assert!(
        tensors
            .spec("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
            .is_some()
    );
    assert_eq!(
        tensors
            .spec("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
            .expect("the projection")
            .dtype,
        DType::BF16
    );
}

#[test]
fn a_sparse_indexer_launcher_records_pool_score_and_pick() {
    use sparkflash_ops::engine_plan::PartRequirement;
    use sparkflash_ops::launchers::SparseIndexerLauncher;
    use sparkflash_ops::plan::{PartGeometry, Precision};

    let requirement = PartRequirement {
        part: sparkflash_model::Part::SparseIndexer,
        geometry: PartGeometry {
            hidden: 2560,
            rows: 1,
            heads: 4,
            kv_heads: 1,
            head_dim: 128,
            top_k: 2048,
            ratio: 4,
            positions: 4096,
            world: 1,
            rank: 0,
            ..PartGeometry::new()
        },
        precision: Precision::Bf16,
    };
    let resolved = PlanContext::SM121_CATALOGUE
        .resolver()
        .resolve(&requirement)
        .expect("the Qwen indexer resolves");
    let launcher = SparseIndexerLauncher::new(&requirement, &resolved, 4, 4096)
        .expect("the indexer launcher binds");
    let names: Vec<&str> = launcher
        .symbols
        .iter()
        .map(|symbol| symbol.symbol)
        .collect();
    assert_eq!(
        names,
        vec!["q4e_qsa_pool", "q4e_qsa_scores", "q4e_qsa_pick"],
        "the three bodies are recorded in launch order"
    );
    // The last row (position 4096, 4 rows, pool 4) sees 1025 pools. The three bodies have
    // three different geometries: a pooled block a position at 128 threads, a (64-pool tile,
    // row) score grid at 256 threads, and one row a block for the pick.
    assert_eq!(launcher.symbols[0].launch.grid, (1025, 1, 1));
    assert_eq!(launcher.symbols[0].launch.block, (128, 1, 1));
    assert_eq!(launcher.symbols[1].launch.grid, (17, 4, 1));
    assert_eq!(launcher.symbols[1].launch.block, (256, 1, 1));
    assert_eq!(launcher.symbols[2].launch.grid, (4, 1, 1));
    assert_eq!(launcher.symbols[2].launch.block, (256, 1, 1));
    assert_ne!(
        launcher.symbols[0].launch.grid,
        launcher.symbols[1].launch.grid
    );
    assert_ne!(
        launcher.symbols[1].launch.grid,
        launcher.symbols[2].launch.grid
    );
    // The primary descriptor is the resolved symbol's own geometry, not the pool's.
    let primary = launcher
        .symbols
        .iter()
        .find(|symbol| symbol.symbol == launcher.kernel.symbol)
        .expect("the resolved symbol is in the sequence");
    assert_eq!(primary.launch, launcher.launch);
    assert_ne!(
        launcher.symbols[0].launch.kernel,
        launcher.symbols[2].launch.kernel
    );

    // GLM's indexer is the same three-body pass with its own block and score tile. Its pool
    // body runs a block a query row (256 threads), and its scores tile the same 64 pools.
    let glm_requirement = PartRequirement {
        part: sparkflash_model::Part::SparseIndexer,
        geometry: PartGeometry {
            hidden: 4096,
            rows: 1,
            heads: 8,
            kv_heads: 1,
            head_dim: 128,
            top_k: 2048,
            ratio: 4,
            positions: 4096,
            world: 1,
            rank: 0,
            ..PartGeometry::new()
        },
        precision: Precision::Bf16,
    };
    let resolved = PlanContext::SM121_CATALOGUE
        .resolver()
        .resolve(&glm_requirement)
        .expect("the GLM indexer resolves");
    let launcher = SparseIndexerLauncher::new(&glm_requirement, &resolved, 4, 4096)
        .expect("the GLM indexer launcher binds");
    let names: Vec<&str> = launcher
        .symbols
        .iter()
        .map(|symbol| symbol.symbol)
        .collect();
    assert_eq!(
        names,
        vec!["q4e_glm_pool", "q4e_glm_index_scores", "q4e_glm_pick"],
        "the GLM bodies are recorded in launch order"
    );
    assert_eq!(launcher.symbols[0].launch.grid, (4, 1, 1));
    assert_eq!(launcher.symbols[0].launch.block, (256, 1, 1));
    assert_eq!(launcher.symbols[1].launch.grid, (17, 4, 1));
    assert_eq!(launcher.symbols[1].launch.block, (256, 1, 1));
    assert_eq!(launcher.symbols[2].launch.grid, (4, 1, 1));
    assert_eq!(launcher.symbols[2].launch.block, (256, 1, 1));
    assert_eq!(launcher.launch.grid, (4, 1, 1));
}

#[test]
fn the_rank_runs_reconstruct_the_five_tile_expert() {
    // D91: 5 whole 128-wide tiles over 2 ranks are 3/2, over 4 are 2/1/1/1. The plan's rank
    // runs must be those element offsets/counts and concatenate back to 640 without a gap.
    let mut architecture = released_family();
    architecture.quantization.expert_format = WeightFormat::Fp8Tile {
        block_m: 128,
        block_n: 128,
        scale_dtype: ScaleDType::Bf16,
        dynamic_activations: true,
    };
    let prefix = "model.language_model.layers.0.mlp.experts.0";
    for (world, expected) in [(2usize, vec![384usize, 256]), (4, vec![256, 128, 128, 128])] {
        let mut runs = Vec::new();
        for (rank, width) in expected.iter().enumerate() {
            let tp = TpPlan::new(world, rank).expect("a rank");
            let plan =
                ArchitectureTensorPlan::served(&architecture, tp).expect("the FP8 plan builds");
            let weight = format!("{prefix}.gate_proj.weight");
            let run = plan
                .rank_run(&weight)
                .expect("the expert weight is sharded");
            assert_eq!(run.total, 640, "TP{world}/{rank} logical axis");
            assert_eq!(run.count, *width, "TP{world}/{rank} run");
            assert_eq!(
                plan.rank_spec(&weight)
                    .expect("the weight is planned")
                    .shape[0],
                *width,
                "TP{world}/{rank} rank shape is the run"
            );
            let scale = format!("{prefix}.gate_proj.weight_scale_inv");
            let scale_run = plan.rank_run(&scale).expect("the scale is sharded");
            assert_eq!(
                (scale_run.offset, scale_run.count),
                (run.offset / 128, *width / 128)
            );
            runs.push((run.offset, run.count));
        }
        runs.sort_unstable();
        let mut at = 0usize;
        for (offset, count) in &runs {
            assert_eq!(*offset, at, "TP{world}: runs are contiguous");
            at += count;
        }
        assert_eq!(at, 640, "TP{world}: the runs cover the logical axis");
        assert_eq!(at / 128, 5, "TP{world}: five whole tiles");
    }
}

#[test]
fn an_attention_launcher_describes_the_resolved_symbol() {
    use sparkflash_ops::engine_plan::PartRequirement;
    use sparkflash_ops::launchers::AttentionLauncher;
    use sparkflash_ops::plan::{PartGeometry, Precision, StreamId};

    let requirement = PartRequirement {
        part: sparkflash_model::Part::GqaAttention,
        geometry: PartGeometry {
            hidden: 2560,
            rows: 1,
            heads: 8,
            kv_heads: 2,
            head_dim: 128,
            positions: 4096,
            world: 1,
            rank: 0,
            ..PartGeometry::new()
        },
        precision: Precision::Bf16,
    };
    let resolved = PlanContext::SM121_CATALOGUE
        .resolver()
        .resolve(&requirement)
        .expect("the Qwen GQA body resolves");
    let launcher = AttentionLauncher::new(&requirement, &resolved, 4, 4096, StreamId(0))
        .expect("the attention launcher binds");
    assert_eq!(launcher.kernel, resolved.id());
    assert_eq!(
        launcher.launch.kernel,
        resolved.id(),
        "the descriptor's grid/block describe the kernel the rank resolved, not a second lookup"
    );
    // Four rows over a 4:1 query/KV group is 16 row-heads, four blocks of four.
    assert_eq!(launcher.launch.grid, (4, 1, 1));
    assert_eq!(launcher.launch.block, (256, 1, 1));
}
