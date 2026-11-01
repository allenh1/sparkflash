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

//! The Phase-1 TP seam, host-only: identity, shard equivalence and residency.
//!
//! These tests drive the reference forward over one rank's routed-expert shards through the
//! engine-owned [sparkflash_engine::parallel_tp] seam:
//!
//! * `identity`: with a one-rank adapter the sharded engine's logits are bit-identical to the
//!   unsharded reference, so inserting the seam changed no value;
//! * `exact wire`: over an exact f32 reduce, 2 and 4 in-process ranks match the reference within
//!   `EXACT_WIRE_REL_L2_BAR` (1e-5) per routed-FFN seam and end to end, so a slicing error cannot
//!   hide inside the bf16 wire's rounding (about 2e-3 per seam). On this fixture the routed FFN's
//!   reduced output has an L2 norm near 1e-7, so the logits do not move on either wire: the
//!   per-seam comparisons are the ones that see the shards;
//! * `bf16 wire`: over the transport's canonical arithmetic every rank's logits and seams are
//!   bit-equal to rank 0's. The distance from the reference is the wire's rounding, so it is
//!   printed, not barred (the relative-L2 bars task-54 declared were replaced by the exact-wire
//!   bar);
//! * `resident`: after `new_tp` returns, a per-token forward issues no whole-tensor read against
//!   the caller's source (a counting wrapper proves the count stops changing), while the Engram
//!   tables stay in the source and are paged per token.
//!
//! The collective is the seam's in-process bus ([sparkflash_engine::parallel_tp::LocalBus]) on its
//! default bf16 wire, which implements the trait's canonical arithmetic exactly: bf16-round each
//! payload, f32-accumulate in ascending rank order from 0.0, one bf16 round, widen. The node path
//! implements the same trait over TCP in the CLI.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::{DType, expert_scale_shape};
use sparkflash_engine::forward::{InferenceEngine, Sampling, SequenceState};
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::parallel_tp::{
    EXACT_WIRE_REL_L2_BAR, LocalBus, LocalRank, ResidentPlan, ResidentSource, TpLayout, Wire,
    is_engram_table,
};
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

mod common;

use common::FiniteSource;
use common::host_path::bits;

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    sparkflash_engine::linalg::relative_l2(a, b).expect("two vectors of the same length")
}

fn engine(config: &ModelConfig, seed: u64, limit: usize) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = EngramF8Source::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), limit)
        .expect("the tiny configuration is usable")
}

/// The leaf of a routed-expert tensor (`w1`, `w2`, `w3`), or `None` for every other tensor.
fn routed_expert_leaf(name: &str) -> Option<&str> {
    let base = name
        .strip_suffix(".weight")
        .or_else(|| name.strip_suffix(".scale"))?;
    if !base.contains(".ffn.experts.") {
        return None;
    }
    base.rsplit('.').next()
}

/// This rank's local spec for one served spec, in the local-spec rules agreed with the CLI:
/// w1/w3 slice rows, w2 slices input columns, scale grids follow their weight.
fn local_spec(
    served: &TensorSpec,
    world: usize,
    moe_inter_dim: usize,
) -> Result<TensorSpec, WeightError> {
    let Some(leaf) = routed_expert_leaf(&served.name) else {
        return Ok(served.clone());
    };
    if !moe_inter_dim.is_multiple_of(world) {
        return Err(WeightError::Malformed {
            name: served.name.clone(),
            detail: format!("moe_inter_dim {moe_inter_dim} does not split over {world}"),
        });
    }
    let local_inter = moe_inter_dim / world;
    let shape = match (leaf, served.name.ends_with(".scale")) {
        ("w2", false) => {
            let cols = served.shape[1];
            if cols != moe_inter_dim {
                return Err(WeightError::Malformed {
                    name: served.name.clone(),
                    detail: format!("w2 has {cols} input columns, expected {moe_inter_dim}"),
                });
            }
            vec![served.shape[0], local_inter]
        }
        ("w2", true) => expert_scale_shape(served.shape[0], local_inter).to_vec(),
        (_w1_w3, false) => {
            let rows = served.shape[0];
            if rows != moe_inter_dim {
                return Err(WeightError::Malformed {
                    name: served.name.clone(),
                    detail: format!("{leaf} has {rows} rows, expected {moe_inter_dim}"),
                });
            }
            vec![local_inter, served.shape[1]]
        }
        (_w1_w3, true) => vec![local_inter, served.shape[1]],
    };
    Ok(TensorSpec {
        name: served.name.clone(),
        dtype: served.dtype,
        shape,
    })
}

/// The tiny fixture stores the Engram tables as f32, but the released checkpoint stores them as
/// F8E4M3 rows with an E8M0 grid and the paged reader decodes that storage format directly. Serving
/// the plan's storage dtype keeps both Engram paths - the reference's whole-table read and the
/// resident source's per-row paging - on the same bytes, so the identity test still compares the
/// same model.
struct EngramF8Source {
    inner: FiniteSource,
    specs: BTreeMap<String, TensorSpec>,
}

impl EngramF8Source {
    fn from_plan(plan: &TensorPlan, seed: u64) -> Self {
        let inner = FiniteSource::from_plan(plan, seed);
        let mut specs = BTreeMap::new();
        for wanted in plan.specs() {
            let served = if is_engram_table(&wanted.name) {
                wanted.clone()
            } else {
                inner
                    .spec(&wanted.name)
                    .expect("the fixture serves its plan")
            };
            specs.insert(wanted.name.clone(), served);
        }
        Self { inner, specs }
    }
}

impl WeightSource for EngramF8Source {
    fn names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.get(name) {
            Some(spec) => Ok(spec.clone()),
            None => Err(WeightError::Missing {
                name: name.to_owned(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if !is_engram_table(name) {
            return self.inner.read(name);
        }
        let spec = self.spec(name)?;
        let elements = usize::try_from(spec.elements()).expect("a tiny table fits");
        if spec.dtype == DType::F8E8M0 {
            // E8M0 exponent 127 is 2^0: an all-ones grid, so the decoded value is the weight alone.
            return Ok(vec![127u8; elements]);
        }
        let f32_bytes = self.inner.read(name)?;
        let mut out = Vec::with_capacity(elements);
        for chunk in f32_bytes.as_chunks::<4>().0 {
            out.push(sparkflash_engine::quant::e4m3_encode(f32::from_le_bytes(
                *chunk,
            )));
        }
        Ok(out)
    }
}

/// Wraps the fixture with this rank's routed-expert slices (global names, local specs).
struct LocalSource {
    inner: EngramF8Source,
    specs: BTreeMap<String, TensorSpec>,
    world: usize,
    rank: usize,
    moe_inter_dim: usize,
}

impl LocalSource {
    fn from_plan(
        plan: &TensorPlan,
        seed: u64,
        world: usize,
        rank: usize,
        moe_inter_dim: usize,
    ) -> Self {
        let inner = EngramF8Source::from_plan(plan, seed);
        let mut specs = BTreeMap::new();
        for wanted in plan.specs() {
            let served = inner
                .spec(&wanted.name)
                .expect("the fixture serves its plan");
            specs.insert(
                wanted.name.clone(),
                local_spec(&served, world, moe_inter_dim).expect("the fixture shapes split"),
            );
        }
        Self {
            inner,
            specs,
            world,
            rank,
            moe_inter_dim,
        }
    }

    fn local_values(&self, name: &str, local: &TensorSpec) -> Result<Vec<f32>, WeightError> {
        let global = self.inner.spec(name)?;
        let bytes = self.inner.read(name)?;
        if global.dtype != DType::F32 {
            return Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: "the TP fixture slices f32 tensors only".to_owned(),
            });
        }
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let leaf = routed_expert_leaf(name).expect("only called for shards");
        let scale = name.ends_with(".scale");
        let local_inter = self.moe_inter_dim / self.world;
        let global_cols = global.shape[1];
        if !scale {
            match leaf {
                "w2" => {
                    // Collect this rank's input-column range from every row.
                    let mut out = Vec::with_capacity(global.shape[0] * local_inter);
                    for row in 0..global.shape[0] {
                        let start = row * global_cols + self.rank * local_inter;
                        out.extend_from_slice(&values[start..start + local_inter]);
                    }
                    return Ok(out);
                }
                _ => {
                    // w1/w3: the local rows are contiguous.
                    let start = self.rank * local_inter * global_cols;
                    return Ok(values[start..start + local_inter * global_cols].to_vec());
                }
            }
        }
        // Scales: w1/w3 slice rows; w2 slices column blocks, a local block k mapping to global
        // block (rank * local_inter + k * 32) / 32.
        let local_scale_cols = if leaf == "w2" {
            local_inter.div_ceil(32)
        } else {
            global_cols
        };
        if local.shape != [local.shape[0], local_scale_cols] {
            return Err(WeightError::Malformed {
                name: name.to_owned(),
                detail: format!("local scale {:?} does not match the grid", local.shape),
            });
        }
        let mut out = Vec::with_capacity(local.shape[0] * local_scale_cols);
        for row in 0..local.shape[0] {
            let global_row = if leaf == "w2" {
                row
            } else {
                self.rank * local_inter + row
            };
            for k in 0..local_scale_cols {
                let global_k = if leaf == "w2" {
                    (self.rank * local_inter + k * 32) / 32
                } else {
                    k
                };
                out.push(values[global_row * global_cols + global_k]);
            }
        }
        Ok(out)
    }
}

impl WeightSource for LocalSource {
    fn names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.get(name) {
            Some(spec) => Ok(spec.clone()),
            None => Err(WeightError::Missing {
                name: name.to_owned(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let local = self.spec(name)?;
        if routed_expert_leaf(name).is_some() {
            let values = self.local_values(name, &local)?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            return Ok(bytes);
        }
        self.inner.read(name)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        // The fixture tables are tiny: materialise the tensor and cut the ranges out of it. A real
        // source pages them from the file; the resident source's contract is only that it forwards.
        let bytes = match self.read(name) {
            Ok(bytes) => bytes,
            Err(error) => return Some(Err(error)),
        };
        let mut out = Vec::with_capacity(ranges.iter().map(|&(_, len)| len).sum());
        for &(offset, len) in ranges {
            let start = match usize::try_from(offset) {
                Ok(start) => start,
                Err(_) => {
                    return Some(Err(WeightError::Malformed {
                        name: name.to_owned(),
                        detail: format!("range offset {offset} does not fit a usize"),
                    }));
                }
            };
            match start.checked_add(len).and_then(|end| bytes.get(start..end)) {
                Some(slice) => out.extend_from_slice(slice),
                None => {
                    return Some(Err(WeightError::Malformed {
                        name: name.to_owned(),
                        detail: format!(
                            "range {start}+{len} is past the {} byte tensor",
                            bytes.len()
                        ),
                    }));
                }
            }
        }
        Some(Ok(out))
    }
}

/// Per-name counts of whole-tensor reads and paged range reads.
#[derive(Default)]
struct ReadCounts {
    reads: Mutex<BTreeMap<String, usize>>,
    ranges: Mutex<BTreeMap<String, usize>>,
}

impl ReadCounts {
    fn record(map: &Mutex<BTreeMap<String, usize>>, name: &str) {
        let mut map = map.lock().expect("the read counter is not poisoned");
        *map.entry(name.to_owned()).or_default() += 1;
    }

    fn read(&self, name: &str) {
        Self::record(&self.reads, name);
    }

    fn ranged(&self, name: &str) {
        Self::record(&self.ranges, name);
    }

    fn reads_of(&self, name: &str) -> usize {
        self.reads
            .lock()
            .expect("the read counter is not poisoned")
            .get(name)
            .copied()
            .unwrap_or(0)
    }

    fn ranges_of(&self, name: &str) -> usize {
        self.ranges
            .lock()
            .expect("the read counter is not poisoned")
            .get(name)
            .copied()
            .unwrap_or(0)
    }

    fn total_reads(&self) -> usize {
        self.reads
            .lock()
            .expect("the read counter is not poisoned")
            .values()
            .sum()
    }
}

/// A plan supplied as a plain name list, the shape the CLI implements for its own sharded plan.
struct PlanNames(Vec<String>);

impl ResidentPlan for PlanNames {
    fn resident_plan_names(&self) -> Vec<String> {
        self.0.clone()
    }
}

/// A source whose reads are counted per name, so a test can prove which tensors the forward
/// stopped reading and which ones it still pages.
struct CountingSource {
    inner: LocalSource,
    counts: Arc<ReadCounts>,
}

impl WeightSource for CountingSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.counts.read(name);
        self.inner.read(name)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        match self.inner.read_ranges(name, ranges) {
            Some(Ok(bytes)) => {
                self.counts.ranged(name);
                Some(Ok(bytes))
            }
            other => other,
        }
    }
}

fn tp_engines(
    config: &ModelConfig,
    seed: u64,
    limit: usize,
    world: usize,
    wire: Wire,
) -> (Vec<InferenceEngine>, Arc<LocalBus>) {
    let plan = TensorPlan::language_model(config);
    let bus = Arc::new(LocalBus::recording(world).with_wire(wire));
    let engines = (0..world)
        .map(|rank| {
            let source = LocalSource::from_plan(&plan, seed, world, rank, config.moe_inter_dim);
            InferenceEngine::new_tp(
                config.clone(),
                Box::new(source),
                limit,
                TpLayout {
                    rank,
                    world_size: world,
                },
                Box::new(LocalRank {
                    rank,
                    bus: Arc::clone(&bus),
                }),
            )
            .expect("the tiny fixture supports the TP layout")
        })
        .collect();
    (engines, bus)
}

/// Runs one rank's engine on a thread and returns its first-step logits.
fn run_logits(
    engine: InferenceEngine,
    config: &ModelConfig,
    tokens: &[u32],
) -> std::thread::JoinHandle<Vec<f32>> {
    let config = config.clone();
    let tokens = tokens.to_vec();
    std::thread::spawn(move || {
        let mut state = SequenceState::new(&config);
        let step = engine
            .prefill(&mut state, &tokens)
            .expect("the fixture forward runs");
        step.logits
    })
}

#[test]
fn the_one_rank_shard_path_is_bit_identical_to_the_reference() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let reference = engine(&config, 7, limit);
    let mut reference_state = SequenceState::new(&config);
    let expected = reference
        .prefill(&mut reference_state, &[3, 7, 11])
        .expect("the reference forward runs")
        .logits;

    let (engines, bus) = tp_engines(&config, 7, limit, 1, Wire::Bf16);
    let mut state = SequenceState::new(&config);
    let got = engines[0]
        .prefill(&mut state, &[3, 7, 11])
        .expect("the one-rank sharded forward runs")
        .logits;
    assert_eq!(got.len(), expected.len());
    for (index, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {index} differs: reference {a} vs one-rank {b}"
        );
    }
    // One chunk of three tokens is one block per layer, so one seam per layer.
    let recorded = bus.seams(0).expect("the bus records its seams");
    assert_eq!(recorded.len(), config.n_layers);
    assert!(recorded.iter().all(|row| row.len() == 3 * config.dim));
}

/// Every rank's first-step logits for `tokens` over one recording bus on `wire`, and the bus.
fn run_world(
    config: &ModelConfig,
    limit: usize,
    world: usize,
    wire: Wire,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, Arc<LocalBus>) {
    let (engines, bus) = tp_engines(config, 7, limit, world, wire);
    let handles: Vec<_> = engines
        .into_iter()
        .map(|engine| run_logits(engine, config, tokens))
        .collect();
    let logits = handles
        .into_iter()
        .map(|handle| handle.join().expect("a rank thread does not panic"))
        .collect();
    (logits, bus)
}

/// The reference's first-step logits for `tokens`.
fn reference_logits(config: &ModelConfig, limit: usize, tokens: &[u32]) -> Vec<f32> {
    engine(config, 7, limit)
        .prefill(&mut SequenceState::new(config), tokens)
        .expect("the reference forward runs")
        .logits
}

#[test]
fn two_and_four_ranks_shard_the_reference_over_an_exact_wire() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let tokens = [3, 7, 11];
    let expected = reference_logits(&config, limit, &tokens);
    let (_, one_bus) = run_world(&config, limit, 1, Wire::F32, &tokens);
    let one_seams = one_bus.seams(0).expect("the bus records its seams");
    for world in [2usize, 4] {
        let (logits, bus) = run_world(&config, limit, world, Wire::F32, &tokens);
        for (rank, logits) in logits.iter().enumerate() {
            let rel = rel_l2(logits, &expected);
            println!("measured: exact wire, world {world} rank {rank} logits rel L2 = {rel:e}");
            assert!(
                rel <= EXACT_WIRE_REL_L2_BAR,
                "rank {rank} of {world}: logits rel L2 {rel:e} above {EXACT_WIRE_REL_L2_BAR:e}"
            );
            let seams = bus.seams(rank).expect("the bus records its seams");
            assert_eq!(seams.len(), one_seams.len(), "rank {rank} of {world}");
            for (index, (ours, theirs)) in seams.iter().zip(&one_seams).enumerate() {
                let rel = rel_l2(ours, theirs);
                assert!(
                    rel <= EXACT_WIRE_REL_L2_BAR,
                    "rank {rank} of {world} seam {index}: rel L2 {rel:e} above \
                     {EXACT_WIRE_REL_L2_BAR:e}"
                );
            }
        }
    }
}

#[test]
fn two_and_four_ranks_agree_bit_for_bit_over_the_bf16_wire() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let tokens = [3, 7, 11];
    let expected = reference_logits(&config, limit, &tokens);
    let (_, one_bus) = run_world(&config, limit, 1, Wire::Bf16, &tokens);
    let one_seams = one_bus.seams(0).expect("the bus records its seams");
    for world in [2usize, 4] {
        let (logits, bus) = run_world(&config, limit, world, Wire::Bf16, &tokens);
        let seams = bus.seams(0).expect("the bus records its seams");
        assert_eq!(seams.len(), config.n_layers, "one seam per layer");
        for rank in 1..world {
            assert_eq!(
                bits(&logits[rank]),
                bits(&logits[0]),
                "rank {rank} of {world}"
            );
            let theirs = bus.seams(rank).expect("the bus records its seams");
            assert_eq!(theirs.len(), seams.len(), "rank {rank} of {world}");
            for (index, (ours, theirs)) in seams.iter().zip(&theirs).enumerate() {
                assert_eq!(
                    bits(theirs),
                    bits(ours),
                    "rank {rank} of {world} seam {index}"
                );
            }
        }
        let worst_seam = seams
            .iter()
            .zip(&one_seams)
            .map(|(ours, theirs)| rel_l2(ours, theirs))
            .fold(0.0f64, f64::max);
        println!(
            "measured: bf16 wire, world {world}: worst seam rel L2 = {worst_seam:e}, logits rel L2 \
             = {:e}",
            rel_l2(&logits[0], &expected)
        );
    }
}

#[test]
fn the_resident_source_is_not_read_again_per_token() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let plan = TensorPlan::language_model(&config);
    let counts = Arc::new(ReadCounts::default());
    let source = CountingSource {
        inner: LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim),
        counts: Arc::clone(&counts),
    };
    let engine = InferenceEngine::new_tp(
        config.clone(),
        Box::new(source),
        limit,
        TpLayout {
            rank: 0,
            world_size: 1,
        },
        Box::new(LocalRank {
            rank: 0,
            bus: Arc::new(LocalBus::new(1)),
        }),
    )
    .expect("the fixture supports the layout");
    let after_load = counts.total_reads();
    assert!(after_load > 0, "new_tp must read the rank's tensors once");
    let mut state = SequenceState::new(&config);
    engine
        .prefill(&mut state, &[3, 7])
        .expect("the resident forward runs");
    engine
        .prefill(&mut state, &[11])
        .expect("the resident forward runs again");
    assert_eq!(
        counts.total_reads(),
        after_load,
        "the per-token forward read the underlying source again"
    );
    assert!(engine.tp().is_some());
}

#[test]
fn the_resident_source_holds_shard_and_replicated_weights_and_pages_engram_tables() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let plan = TensorPlan::language_model(&config);
    let paged: Vec<String> = plan
        .specs()
        .iter()
        .filter(|spec| is_engram_table(&spec.name))
        .map(|spec| spec.name.clone())
        .collect();
    assert!(!paged.is_empty(), "the tiny fixture has an Engram layer");

    // Expected accounting from the rank's served (local) specs, computed on a throwaway fixture.
    let probe = LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim);
    let expected_resident_bytes: usize = plan
        .specs()
        .iter()
        .filter(|spec| !is_engram_table(&spec.name))
        .map(|spec| {
            probe
                .spec(&spec.name)
                .expect("the fixture serves its plan")
                .storage_bytes() as usize
        })
        .sum();
    let excluded_bytes: usize = plan
        .specs()
        .iter()
        .filter(|spec| is_engram_table(&spec.name))
        .map(|spec| spec.storage_bytes() as usize)
        .sum();

    // The engine path: construction reads every non-Engram tensor once, the forward reads none of
    // them again, and the Engram tables are paged during the forward.
    let counts = Arc::new(ReadCounts::default());
    let source = CountingSource {
        inner: LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim),
        counts: Arc::clone(&counts),
    };
    let engine = InferenceEngine::new_tp(
        config.clone(),
        Box::new(source),
        limit,
        TpLayout {
            rank: 0,
            world_size: 1,
        },
        Box::new(LocalRank {
            rank: 0,
            bus: Arc::new(LocalBus::new(1)),
        }),
    )
    .expect("the fixture supports the layout");
    for spec in plan.specs() {
        if is_engram_table(&spec.name) {
            assert_eq!(
                counts.reads_of(&spec.name),
                0,
                "{} was read whole at construction",
                spec.name
            );
        } else {
            assert_eq!(
                counts.reads_of(&spec.name),
                1,
                "{} was not read exactly once at construction",
                spec.name
            );
        }
    }
    let mut state = SequenceState::new(&config);
    engine
        .prefill(&mut state, &[3, 7])
        .expect("the resident forward runs");
    engine
        .prefill(&mut state, &[11])
        .expect("the resident forward runs again");
    for spec in plan.specs() {
        if is_engram_table(&spec.name) {
            assert!(
                counts.ranges_of(&spec.name) > 0,
                "{} was never paged by the forward",
                spec.name
            );
            assert_eq!(
                counts.reads_of(&spec.name),
                0,
                "{} was paged but also read whole",
                spec.name
            );
        } else {
            assert_eq!(
                counts.reads_of(&spec.name),
                1,
                "{} was read again after construction",
                spec.name
            );
        }
    }

    // The type on its own: the resident map holds the shard and replicated tensors, keeps the
    // Engram tables out of it, forwards their ranges, and reports the bytes it holds.
    let direct_counts = Arc::new(ReadCounts::default());
    let direct_source = CountingSource {
        inner: LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim),
        counts: Arc::clone(&direct_counts),
    };
    let resident = ResidentSource::load(Box::new(direct_source), &plan).expect("the fixture loads");
    assert_eq!(
        resident.paged_names(),
        paged.as_slice(),
        "exactly the Engram tables are paged"
    );
    assert_eq!(resident.resident_bytes(), expected_resident_bytes);
    assert!(resident.is_resident("layers.0.attn_norm.weight"));
    assert!(!resident.is_resident(&paged[0]));
    let ranges = [(0u64, 8usize)];
    let bytes = resident
        .read_ranges(&paged[0], &ranges)
        .expect("the Engram table is paged, not resident")
        .expect("the fixture pages the table");
    assert_eq!(bytes.len(), 8);
    assert_eq!(direct_counts.ranges_of(&paged[0]), 1);
    println!(
        "measured resident: {} tensors / {} bytes resident, {} paged tensors / {} bytes excluded (engine path saw {} whole-tensor reads at construction)",
        resident.len(),
        resident.resident_bytes(),
        resident.paged_names().len(),
        excluded_bytes,
        counts.total_reads(),
    );
}

#[test]
fn set_sampling_applies_a_request_sampler_without_consuming_the_engine() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let sampling = Sampling {
        temperature: 0.7,
        seed: 11,
    };
    let mut tp_engine = InferenceEngine::new_tp(
        config.clone(),
        Box::new(LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim)),
        1 << 22,
        TpLayout {
            rank: 0,
            world_size: 1,
        },
        Box::new(LocalRank {
            rank: 0,
            bus: Arc::new(LocalBus::new(1)),
        }),
    )
    .expect("the fixture supports the layout");
    tp_engine.set_sampling(sampling);
    assert!(
        tp_engine.tp().is_some(),
        "set_sampling must keep the TP seam state"
    );
    let reference = engine(&config, 7, 1 << 22).with_sampling(sampling);
    let mut state = SequenceState::new(&config);
    let mut reference_state = SequenceState::new(&config);
    let out = tp_engine
        .prefill(&mut state, &[3, 7, 11])
        .expect("the TP engine prefills");
    let want = reference
        .prefill(&mut reference_state, &[3, 7, 11])
        .expect("the reference engine prefills");
    assert_eq!(out.logits, want.logits);
    assert_eq!(out.next_token, want.next_token);
    assert!(
        out.next_token.is_some(),
        "a non-greedy sampler picks a token"
    );
}

#[test]
fn a_plan_name_list_drives_the_same_resident_set_as_the_tensor_plan() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let names = PlanNames(plan.names().map(str::to_owned).collect());
    let source = CountingSource {
        inner: LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim),
        counts: Arc::new(ReadCounts::default()),
    };
    let resident = ResidentSource::load(Box::new(source), &names).expect("the name list loads");
    let paged: Vec<String> = plan
        .specs()
        .iter()
        .filter(|spec| is_engram_table(&spec.name))
        .map(|spec| spec.name.clone())
        .collect();
    assert_eq!(resident.paged_names(), paged.as_slice());
    assert_eq!(resident.len(), plan.specs().len() - paged.len());
    assert!(resident.resident_bytes() > 0);
    assert!(resident.is_resident("layers.0.attn_norm.weight"));
}

#[test]
fn the_local_specs_split_the_routed_experts_only() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    for world in [2usize, 4] {
        for rank in 0..world {
            let source = LocalSource::from_plan(&plan, 7, world, rank, config.moe_inter_dim);
            let local_inter = config.moe_inter_dim / world;
            for spec in plan.specs() {
                let local = source
                    .spec(&spec.name)
                    .expect("every planned name is served");
                match routed_expert_leaf(&spec.name) {
                    Some("w2") => {
                        assert_eq!(local.shape[0], spec.shape[0]);
                        assert_eq!(
                            local.shape[1],
                            if spec.name.ends_with(".scale") {
                                local_inter.div_ceil(32)
                            } else {
                                local_inter
                            }
                        );
                    }
                    Some(_) => {
                        assert_eq!(local.shape[0], local_inter);
                        assert_eq!(local.shape[1], spec.shape[1]);
                    }
                    None => assert_eq!(local.shape, spec.shape, "{} is replicated", spec.name),
                }
            }
        }
    }
}

#[test]
fn a_layout_that_does_not_split_the_routed_intermediate_is_refused() {
    let config = ModelConfig::tiny();
    let plan = TensorPlan::language_model(&config);
    let source = LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim);
    let error = InferenceEngine::new_tp(
        config.clone(),
        Box::new(source),
        1 << 22,
        TpLayout {
            rank: 0,
            world_size: 3,
        },
        Box::new(LocalRank {
            rank: 0,
            bus: Arc::new(LocalBus::new(3)),
        }),
    )
    .err()
    .expect("3 does not divide the tiny moe_inter_dim");
    assert!(matches!(error, EngineError::Shape { .. }), "{error}");
}
