//! The batch-18 head row shard: a rank's vocab-row slice gathered into full logits.
//!
//! Option A of docs/engine/tp4-attention-sharding-design.md splits the LM head's stored rows by
//! rank and rebuilds the full vocab_size logit vector with an exact all-gather, then runs the
//! same greedy rule over it. This test pins the property that makes it safe: four engines, each
//! over its own row slice and a rank-order gather bus, produce bit-identical logits and tokens to
//! one engine over the whole head on the same hybrid step path the TP=4 host driver uses.
//!
//! It also pins the two guards: a source that serves a head slice without a gather seam is
//! refused rather than silently multiplying a partial weight, and the replicated hatch (whole head
//! plus a seam configured) still takes the replicated path unchanged.

mod common;

use std::sync::Arc;

use common::CodedSource;
use common::host_path::{Bus, BusGather, LIMIT, SEED, Step, bits, reference_steps, run_steps};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::InferenceEngine;
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::{WeightError, WeightSource};

/// One rank's view of a checkpoint whose head.weight is row-sharded: the same coded fixture,
/// with the rank's contiguous block of vocab rows served as the head slice.
struct HeadShardSource {
    inner: CodedSource,
    world: usize,
    rank: usize,
    vocab: usize,
    dim: usize,
}

impl HeadShardSource {
    fn new(plan: &TensorPlan, world: usize, rank: usize, vocab: usize, dim: usize) -> Self {
        Self {
            inner: CodedSource::from_plan(plan, SEED),
            world,
            rank,
            vocab,
            dim,
        }
    }

    /// This rank's head rows and the row they start at.
    fn slice(&self) -> Result<(usize, usize), WeightError> {
        if self.world == 0 || !self.vocab.is_multiple_of(self.world) {
            return Err(WeightError::Malformed {
                name: "head.weight".to_owned(),
                detail: format!("{} vocab rows do not split over {}", self.vocab, self.world),
            });
        }
        let local_rows = self.vocab / self.world;
        Ok((local_rows, self.rank * local_rows))
    }
}

impl WeightSource for HeadShardSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<sparkflash_engine::tensor::TensorSpec, WeightError> {
        let spec = self.inner.spec(name)?;
        if name != "head.weight" {
            return Ok(spec);
        }
        let (local_rows, _) = self.slice()?;
        Ok(sparkflash_engine::tensor::TensorSpec {
            name: spec.name,
            dtype: spec.dtype,
            shape: vec![local_rows, self.dim],
        })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let bytes = self.inner.read(name)?;
        if name != "head.weight" {
            return Ok(bytes);
        }
        let (local_rows, row_start) = self.slice()?;
        let row_bytes = self.dim * 2;
        let start = row_start * row_bytes;
        Ok(bytes[start..start + local_rows * row_bytes].to_vec())
    }
}

/// Four engines, one per head-row slice, gathering each step's logits through the bus.
fn sharded_steps(config: &ModelConfig, tokens: &[u32], world: usize) -> Vec<Vec<Step>> {
    let bus = Arc::new(Bus::new(world));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let bus = Arc::clone(&bus);
                let config = config.clone();
                let tokens = tokens.to_vec();
                scope.spawn(move || {
                    let plan = TensorPlan::language_model(&config);
                    let engine = InferenceEngine::new(
                        config.clone(),
                        Box::new(HeadShardSource::new(
                            &plan,
                            world,
                            rank,
                            config.vocab_size,
                            config.dim,
                        )),
                        LIMIT,
                    )
                    .expect("sharded engine")
                    .with_bit_gather(Box::new(BusGather { rank, world, bus }));
                    let provider =
                        HeadShardSource::new(&plan, world, rank, config.vocab_size, config.dim);
                    run_steps(&config, &engine, &provider, &tokens).expect("sharded steps")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("rank thread"))
            .collect()
    })
}

#[test]
fn the_head_row_shard_gathers_bit_identical_logits_on_the_hybrid_path() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = vec![3, 5, 7, 11];
    let reference = reference_steps(&config, &tokens);
    let sharded = sharded_steps(&config, &tokens, 4);
    for (rank, steps) in sharded.iter().enumerate() {
        assert_eq!(steps.len(), reference.len());
        for (position, (theirs, ours)) in reference.iter().zip(steps).enumerate() {
            assert_eq!(
                theirs.token, ours.token,
                "rank {rank} step {position}: token differs"
            );
            assert_eq!(
                bits(&theirs.logits),
                bits(&ours.logits),
                "rank {rank} step {position}: gathered logits are not bit-identical"
            );
        }
    }
}

#[test]
fn a_sharded_head_without_a_gather_seam_is_refused() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(HeadShardSource::new(
            &plan,
            4,
            0,
            config.vocab_size,
            config.dim,
        )),
        LIMIT,
    )
    .expect("engine");
    let provider = HeadShardSource::new(&plan, 4, 0, config.vocab_size, config.dim);
    let error = run_steps(&config, &engine, &provider, &[3])
        .expect_err("a partial head must not run as if it were whole");
    let text = format!("{error}");
    assert!(
        text.contains("gather seam"),
        "the refusal should name the missing seam, got: {text}"
    );
}

#[test]
fn the_replicated_head_with_a_seam_still_takes_the_replicated_path() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = vec![3, 5, 7, 11];
    let reference = reference_steps(&config, &tokens);
    let plan = TensorPlan::language_model(&config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(CodedSource::from_plan(&plan, SEED)),
        LIMIT,
    )
    .expect("engine")
    .with_bit_gather(Box::new(BusGather {
        rank: 0,
        world: 1,
        bus: Arc::new(Bus::new(1)),
    }));
    let provider = CodedSource::from_plan(&plan, SEED);
    let steps = run_steps(&config, &engine, &provider, &tokens).expect("hatch steps");
    for (position, (theirs, ours)) in reference.iter().zip(&steps).enumerate() {
        assert_eq!(
            theirs.token, ours.token,
            "hatch step {position}: token differs"
        );
        assert_eq!(
            bits(&theirs.logits),
            bits(&ours.logits),
            "hatch step {position}: logits differ"
        );
    }
}
