//! The batch-18 attention output-row shard: wq_b by head, wo_a by o-group, exact o_lora gather.
//!
//! Option A stage 2 of docs/engine/tp4-attention-sharding-design.md keeps every output element's
//! accumulation inside one rank: a rank computes its own head block of q, runs the sparse body on
//! those heads, computes its own o-groups of wo_a, gathers the o_lora vector in rank order and
//! runs the replicated wo_b. This test pins bit identity: four engines, each over its own
//! output-row slices and a rank-order gather bus, must produce exactly the logits and tokens of
//! one engine over the whole checkpoint, on the same hybrid step path the TP=4 host driver uses.
//!
//! The sharded path is detected from the source (wq_b serves fewer than n_heads * head_dim rows),
//! validated here once, and refused when a rank is not whole heads and whole o-groups. The
//! replicated hatch (whole tensors plus a seam) keeps the old path.

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use common::CodedSource;
use common::host_path::{Bus, BusGather, LIMIT, SEED, Step, bits, reference_steps, run_steps};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::InferenceEngine;
use sparkflash_engine::forward::attention::attention_shard_geometry;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// One rank's view of a checkpoint whose attention output rows are sharded: the coded fixture
/// with the rank's contiguous row block served for wq_b and wo_a (weight, and their scale grid
/// when the grid itself divides by the world).
struct AttentionShardSource {
    inner: CodedSource,
    world: usize,
    rank: usize,
    sharded: BTreeSet<String>,
}

impl AttentionShardSource {
    fn new(plan: &TensorPlan, world: usize, rank: usize) -> Self {
        let mut sharded = BTreeSet::new();
        for spec in plan.specs() {
            if !attention_projection_name(&spec.name) {
                continue;
            }
            if spec.name.ends_with(".scale") {
                let rows = spec.shape.first().copied().unwrap_or(0);
                if rows == 0 || !rows.is_multiple_of(world) {
                    continue;
                }
            }
            sharded.insert(spec.name.clone());
        }
        Self {
            inner: CodedSource::from_plan(plan, SEED),
            world,
            rank,
            sharded,
        }
    }

    fn is_sharded(&self, name: &str) -> bool {
        self.sharded.contains(name)
    }
}

/// The names the batch-18 stage-2 rule shards: the backbone wq_b and wo_a weight and scale.
fn attention_projection_name(name: &str) -> bool {
    name.starts_with("layers.")
        && (name.ends_with(".attn.wq_b.weight")
            || name.ends_with(".attn.wq_b.scale")
            || name.ends_with(".attn.wo_a.weight")
            || name.ends_with(".attn.wo_a.scale"))
}

impl WeightSource for AttentionShardSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        let spec = self.inner.spec(name)?;
        if !self.is_sharded(name) {
            return Ok(spec);
        }
        let rows = spec.shape.first().copied().unwrap_or(0);
        if self.world == 0 || rows == 0 || !rows.is_multiple_of(self.world) {
            return Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: format!("{rows} stored rows do not split over {}", self.world),
            });
        }
        let mut shape = spec.shape.clone();
        shape[0] = rows / self.world;
        Ok(TensorSpec {
            name: spec.name,
            dtype: spec.dtype,
            shape,
        })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let bytes = self.inner.read(name)?;
        if !self.is_sharded(name) {
            return Ok(bytes);
        }
        let rows = self.inner.spec(name)?.shape.first().copied().unwrap_or(0);
        if self.world == 0 || rows == 0 || !rows.is_multiple_of(self.world) {
            return Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: format!("{rows} stored rows do not split over {}", self.world),
            });
        }
        let local_rows = rows / self.world;
        if !bytes.len().is_multiple_of(rows) {
            return Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: format!("{} bytes are not whole rows of {rows}", bytes.len()),
            });
        }
        let row_bytes = bytes.len() / rows;
        let start = self.rank * local_rows * row_bytes;
        Ok(bytes[start..start + local_rows * row_bytes].to_vec())
    }
}

/// The tiny geometry changed to 4 heads of 8 so the rank split is whole heads and whole o-groups
/// at world 4: 1 head and 1 group per rank, and every scale grid is a single 32-row block.
fn shard_ready_config() -> ModelConfig {
    let mut config = ModelConfig::tiny();
    config.n_heads = 4;
    config.head_dim = 8;
    config.o_groups = 4;
    config
}

/// Four engines, one per attention output-row slice, gathering each layer's o_lora through the
/// bus.
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
                        Box::new(AttentionShardSource::new(&plan, world, rank)),
                        LIMIT,
                    )
                    .expect("sharded engine")
                    .with_bit_gather(Box::new(BusGather { rank, world, bus }));
                    let provider = AttentionShardSource::new(&plan, world, rank);
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
fn the_attention_output_row_shard_is_bit_identical_on_the_hybrid_path() {
    let config = shard_ready_config();
    let tokens: Vec<u32> = vec![3, 5, 7, 11];
    let reference = reference_steps(&config, &tokens);
    let sharded = sharded_steps(&config, &tokens, 4);
    assert_eq!(sharded.len(), 4);
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
                "rank {rank} step {position}: sharded attention is not bit-identical"
            );
        }
    }
}

#[test]
fn a_rank_that_is_not_whole_o_groups_is_refused() {
    // 4 heads in 2 groups makes a group 2 heads wide; world 4 would give each rank 1 head, which
    // is half a group, so the row blocks would not line up with the wo_a group blocks.
    let mut config = shard_ready_config();
    config.o_groups = 2;
    let plan = TensorPlan::language_model(&config);
    let source = AttentionShardSource::new(&plan, 4, 0);
    let error = attention_shard_geometry(&config, "layers.0", &source, 4)
        .expect_err("1 head per rank is not a whole o-group");
    let text = format!("{error}");
    assert!(
        text.contains("o-groups"),
        "the refusal should name the group alignment, got: {text}"
    );
}

#[test]
fn the_replicated_attention_with_a_seam_still_takes_the_replicated_path() {
    let config = shard_ready_config();
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
