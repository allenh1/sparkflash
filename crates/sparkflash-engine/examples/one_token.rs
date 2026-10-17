//! Phase 1 one-token harness (host half): run one token through the reference and print the
//! report that docs/build/roadmap-to-serving.md section 4.5 defines.
//!
//! Usage:
//!   cargo run -p sparkflash-engine --example one_token [--token N]
//!   cargo run -p sparkflash-engine --example one_token -- --checkpoint <snapshot dir> [--token N]
//!
//! Without a checkpoint the harness runs the synthetic mode: the released layer counts (40 backbone
//! layers and 3 stages) at the tiny shapes, seeded weights, no files. The logits line then says its
//! source is bytes rather than numbers.
//!
//! With --checkpoint the harness runs the released configuration against a real snapshot through
//! ReferenceWeights, which verifies the plan the way LoadedWeights does and then serves the stored
//! forms the reference pass decodes. It has to serve stored forms: LoadedWeights deliberately
//! converts `wo_a` to BF16 with its scale already applied for the serving path, while the reference
//! pass decodes the plan's F8E4M3 and applies the separate scale tensor itself. The two Engram
//! tables (98 GB of FP8 each) are read row by row through read_ranges, so the run does not allocate
//! them; the loader's byte limit (2 GiB) still bounds every other tensor. This is the reference table
//! the device half diffs against (docs/build/phase1-boundary-table.md), and it needs the node: the
//! checkpoint is not on this host.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::loader::ReferenceWeights;
use sparkflash_engine::report::{SMALL_SHAPE_LIMIT, WeightClass, one_token_report};
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;
use std::path::PathBuf;

/// The element limit for the released configuration. The largest non-paged planned tensor is
/// embed.weight with 661,913,600 elements (head.weight ties it exactly); the Engram tables are
/// paged and never see this limit.
const CHECKPOINT_ELEMENT_LIMIT: usize = 1 << 30;

/// The released layer counts at small shapes: 40 backbone layers and three MTP stages, with the
/// released DSpark targets, so the row rule can be checked without a checkpoint.
fn released_layer_counts_at_small_shapes() -> ModelConfig {
    ModelConfig {
        n_layers: 40,
        n_mtp_layers: 3,
        compress_ratios: vec![0; 43],
        kv_source_layers: Vec::new(),
        index_source_layers: Vec::new(),
        candidate_source_layer: 0,
        engram_layer_ids: Vec::new(),
        engram_num_embeddings: Vec::new(),
        dspark_target_layer_ids: vec![37, 38, 39],
        ..ModelConfig::tiny()
    }
}

fn main() {
    let mut token = 0u32;
    let mut checkpoint: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--token" => {
                token = args
                    .next()
                    .expect("--token needs a value")
                    .parse()
                    .expect("--token is a u32");
            }
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    args.next().expect("--checkpoint needs a directory"),
                ));
            }
            other => {
                panic!("unknown argument {other}; usage: one_token [--checkpoint DIR] [--token N]")
            }
        }
    }

    let (config, plan, class) = match &checkpoint {
        Some(_) => {
            let config =
                ModelConfig::v41_flash().expect("the released configuration is consistent");
            let plan = TensorPlan::language_model(&config);
            (config, plan, WeightClass::Numbers)
        }
        None => {
            let config = released_layer_counts_at_small_shapes();
            let plan = TensorPlan::language_model(&config);
            println!(
                "synthetic: released layer counts at tiny shapes ({} layers + {} stages), seed 1",
                config.n_layers, config.n_mtp_layers
            );
            (config, plan, WeightClass::Bytes)
        }
    };

    let (source, limit): (Box<dyn WeightSource>, usize) = match &checkpoint {
        Some(dir) => {
            let index_path = dir.join("model.safetensors.index.json");
            // ReferenceWeights verifies the checkpoint against the plan the way the loader does, then
            // serves the stored forms the reference pass decodes (wo_a as F8E4M3 plus its own scale
            // tensor, experts reinterpreted to the logical FP4 shape) and pages the Engram tables.
            let reference =
                ReferenceWeights::open(dir, &index_path, &plan).unwrap_or_else(|error| {
                    panic!(
                        "--checkpoint {} does not match the planned model: {error}",
                        dir.display()
                    )
                });
            println!(
                "checkpoint: {} tensors verified, {} Engram tensors paged",
                reference.plan().len(),
                reference
                    .plan()
                    .specs()
                    .iter()
                    .filter(|spec| {
                        spec.name.ends_with(".engram.embed.weight")
                            || spec.name.ends_with(".engram.embed.scale")
                    })
                    .count()
            );
            (Box::new(reference), CHECKPOINT_ELEMENT_LIMIT)
        }
        None => (
            Box::new(SyntheticWeights::from_plan(&plan, 1)),
            SMALL_SHAPE_LIMIT,
        ),
    };

    match one_token_report(&config, source, limit, token, class) {
        Ok(report) => print!("{report}"),
        Err(error) => {
            eprintln!("one-token report failed: {error}");
            std::process::exit(1);
        }
    }
}
