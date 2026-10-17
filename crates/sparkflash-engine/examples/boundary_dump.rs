//! Phase 1 boundary dump: write one step in the version 1 manifest-plus-blob format.
//!
//! Spec: docs/build/phase1-boundary-table.md. The pair PREFIX.manifest and PREFIX.blob is what a
//! device run emits too, so boundary_diff can compare the two without a translation layer.
//!
//! Usage:
//!   cargo run --release -p sparkflash-engine --example boundary_dump -- --synthetic --out /tmp/host
//!   cargo run --release -p sparkflash-engine --example boundary_dump -- --checkpoint <snapshot> --out /tmp/host
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::loader::ReferenceWeights;
use sparkflash_engine::report::SMALL_SHAPE_LIMIT;
use sparkflash_engine::synthetic::SyntheticWeights;
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// The element limit for the released configuration, as in examples/one_token.rs.
const CHECKPOINT_ELEMENT_LIMIT: usize = 1 << 30;

fn main() {
    let mut token = 0u32;
    let mut checkpoint: Option<PathBuf> = None;
    let mut synthetic = false;
    let mut out: Option<PathBuf> = None;
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
            "--synthetic" => synthetic = true,
            "--out" => out = Some(PathBuf::from(args.next().expect("--out needs a prefix"))),
            other => panic!("unknown argument {other}"),
        }
    }
    let out = out
        .expect("usage: boundary_dump (--synthetic | --checkpoint DIR) [--token N] --out PREFIX");
    if synthetic == checkpoint.is_some() {
        panic!("give exactly one of --synthetic and --checkpoint DIR");
    }

    let config = if checkpoint.is_some() {
        ModelConfig::v41_flash().expect("the released configuration is consistent")
    } else {
        released_layer_counts_at_small_shapes()
    };
    let plan = TensorPlan::language_model(&config);
    let (source, limit): (Box<dyn WeightSource>, usize) = match &checkpoint {
        Some(dir) => {
            let index_path = dir.join("model.safetensors.index.json");
            let reference =
                ReferenceWeights::open(dir, &index_path, &plan).unwrap_or_else(|error| {
                    panic!(
                        "--checkpoint {} does not match the planned model: {error}",
                        dir.display()
                    )
                });
            (Box::new(reference), CHECKPOINT_ELEMENT_LIMIT)
        }
        None => (
            Box::new(SyntheticWeights::from_plan(&plan, 1)),
            SMALL_SHAPE_LIMIT,
        ),
    };

    let backbone = config.n_layers;
    let engine = InferenceEngine::new(config, source, limit).expect("the engine builds");
    let mut state = SequenceState::new(engine.config());
    let observed = engine
        .prefill_observed(&mut state, &[token])
        .expect("the step runs");
    let manifest_path = out.with_extension("manifest");
    let blob_path = out.with_extension("blob");
    let mut manifest = BufWriter::new(File::create(&manifest_path).expect("create the manifest"));
    let mut blob = BufWriter::new(File::create(&blob_path).expect("create the blob"));
    observed
        .boundary
        .write_v1(backbone, &mut manifest, &mut blob)
        .expect("write the v1 pair");
    manifest.flush().expect("flush the manifest");
    blob.flush().expect("flush the blob");
    drop(manifest);
    drop(blob);
    let blob_len = std::fs::metadata(&blob_path)
        .expect("the blob exists")
        .len();
    println!(
        "dumped {} rows, {} tensors each, {} bytes of blob: {} + {}",
        observed.boundary.len(),
        sparkflash_engine::forward::boundary::TENSOR_ORDER.len(),
        blob_len,
        manifest_path.display(),
        blob_path.display()
    );
}

/// The released layer counts at small shapes, exactly as examples/one_token.rs builds them.
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
