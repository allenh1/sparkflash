//! Phase 1 one-token harness (host half): run one token through the reference and print the
//! report that docs/build/roadmap-to-serving.md section 4.5 defines.
//!
//! Usage:
//!   cargo run -p sparkflash-engine --example one_token [--token N]
//!
//! This is the synthetic mode: the released layer counts (40 backbone + 3 stages) at the tiny
//! shapes, seeded weights, no checkpoint, so the report's row rule, seams and audit run on any
//! host. The logits line says its source is bytes rather than numbers.
//!
//! The released-checkpoint reference table is deliberately not a mode of this example yet. The
//! reference pass reads one tensor at a time under an element limit, and two Engram tables are
//! 98,308,270,592 and 98,305,579,008 elements - the loader refuses them by design and there is no
//! paging weight source for them in the engine. That is a follow-up task; the device half cannot
//! diff a real-weight host table until it lands. The released-geometry attention scratch bug this
//! harness first exposed is fixed in forward/mod.rs and pinned by
//! tests/forward.rs::a_geometry_with_wider_heads_than_the_residual_width_runs_and_records_the_narrow_row.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::report::{SMALL_SHAPE_LIMIT, WeightClass, one_token_report};
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::TensorPlan;

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
            other => panic!("unknown argument {other}; usage: one_token [--token N]"),
        }
    }

    let config = released_layer_counts_at_small_shapes();
    let plan = TensorPlan::language_model(&config);
    let weights = SyntheticWeights::from_plan(&plan, 1);
    println!(
        "synthetic: released layer counts at tiny shapes ({} layers + {} stages), seed 1",
        config.n_layers, config.n_mtp_layers
    );
    match one_token_report(
        &config,
        Box::new(weights),
        SMALL_SHAPE_LIMIT,
        token,
        WeightClass::Bytes,
    ) {
        Ok(report) => print!("{report}"),
        Err(error) => {
            eprintln!("one-token report failed: {error}");
            std::process::exit(1);
        }
    }
}
