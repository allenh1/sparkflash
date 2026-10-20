//! The Phase-1 TP seam, host-only: identity, shard equivalence and residency.
//!
//! These tests drive the reference forward over one rank's routed-expert shards through the
//! engine-owned [sparkflash_engine::parallel_tp] seam:
//!
//! * `identity`: with a one-rank adapter the sharded engine's logits are bit-identical to the
//!   unsharded reference, so inserting the seam changed no value;
//! * `two-and-four-rank`: with 2 and 4 in-process ranks the logits stay inside the bars declared
//!   on task-54 (per-seam routed-FFN relative L2 <= 5.0e-3, end-to-end logits <= 2.0e-2);
//! * `resident`: after `new_tp` returns, a per-token forward issues no read against the caller's
//!   source (a counting wrapper proves the count stops changing).
//!
//! The collective is an in-process bus (two barriers, one shared payload slot per rank) that
//! implements the trait's canonical arithmetic exactly: bf16-round each payload, f32-accumulate in
//! ascending rank order from 0.0, one bf16 round, widen. It is test scaffolding only; the node path
//! implements the same trait over TCP in the CLI.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::{DType, expert_scale_shape};
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::parallel_tp::{TpCollective, TpLayout};
use sparkflash_engine::quant::bf16_round;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

mod common;

use common::FiniteSource;

/// The declared per-seam relative-L2 bar (task-54, written before the run).
const SEAM_REL_L2_BAR: f64 = 5.0e-3;
/// The declared end-to-end logits relative-L2 bar (task-54, written before the run).
const LOGITS_REL_L2_BAR: f64 = 2.0e-2;

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "two vectors of different lengths");
    let mut diff = 0.0f64;
    let mut lhs = 0.0f64;
    let mut rhs = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        diff += d * d;
        lhs += f64::from(x) * f64::from(x);
        rhs += f64::from(y) * f64::from(y);
    }
    diff.sqrt() / lhs.sqrt().max(rhs.sqrt()).max(f64::MIN_POSITIVE)
}

fn engine(config: &ModelConfig, seed: u64, limit: usize) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
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

/// Wraps the fixture with this rank's routed-expert slices (global names, local specs).
struct LocalSource {
    inner: FiniteSource,
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
        let inner = FiniteSource::from_plan(plan, seed);
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
}

/// A source whose reads are counted, so a test can prove the forward stopped reading it.
struct CountingSource {
    inner: LocalSource,
    reads: Arc<AtomicUsize>,
}

impl WeightSource for CountingSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(name)
    }
}

/// The in-process TP bus: one payload slot per rank and two barriers per seam.
struct Bus {
    slots: Mutex<Vec<Vec<f32>>>,
    recorded: Vec<Mutex<Vec<Vec<f32>>>>,
    published: Barrier,
    consumed: Barrier,
}

impl Bus {
    fn new(world: usize) -> Self {
        Self {
            slots: Mutex::new(vec![Vec::new(); world]),
            recorded: (0..world).map(|_| Mutex::new(Vec::new())).collect(),
            published: Barrier::new(world),
            consumed: Barrier::new(world),
        }
    }

    fn record(&self, rank: usize, values: &[f32]) {
        self.recorded[rank]
            .lock()
            .expect("the bus is not poisoned")
            .push(values.to_vec());
    }

    fn reduce_and_record(&self, rank: usize, values: &mut [f32]) {
        {
            let mut slots = self.slots.lock().expect("the bus is not poisoned");
            slots[rank] = values.to_vec();
        }
        self.published.wait();
        {
            let slots = self.slots.lock().expect("the bus is not poisoned");
            let mut accumulator = vec![0.0f32; values.len()];
            for payload in slots.iter() {
                for (sum, value) in accumulator.iter_mut().zip(payload.iter()) {
                    *sum += bf16_round(*value);
                }
            }
            for (out, sum) in values.iter_mut().zip(accumulator.iter()) {
                *out = bf16_round(*sum);
            }
        }
        self.record(rank, values);
        self.consumed.wait();
    }
}

struct BusCollective {
    rank: usize,
    world: usize,
    bus: Arc<Bus>,
}

impl TpCollective for BusCollective {
    fn world(&self) -> usize {
        self.world
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn label(&self) -> &'static str {
        "in-process bus"
    }
    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        if self.world == 1 {
            self.bus.record(self.rank, values);
            return Ok(());
        }
        self.bus.reduce_and_record(self.rank, values);
        Ok(())
    }
    fn all_gather_assign(&mut self, _local: &[f32], _out: &mut [f32]) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "the in-process bus".to_owned(),
            detail: "the Phase-1 seam has no gather".to_owned(),
        })
    }
}

fn tp_engines(
    config: &ModelConfig,
    seed: u64,
    limit: usize,
    world: usize,
) -> (Vec<InferenceEngine>, Arc<Bus>) {
    let plan = TensorPlan::language_model(config);
    let bus = Arc::new(Bus::new(world));
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
                Box::new(BusCollective {
                    rank,
                    world,
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

    let (engines, bus) = tp_engines(&config, 7, limit, 1);
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
    let recorded = bus.recorded[0].lock().expect("the bus is not poisoned");
    assert_eq!(recorded.len(), config.n_layers);
    assert!(recorded.iter().all(|row| row.len() == 3 * config.dim));
}

#[test]
fn two_and_four_rank_logits_stay_inside_the_declared_bars() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let reference = engine(&config, 7, limit);
    let mut reference_state = SequenceState::new(&config);
    let expected = reference
        .prefill(&mut reference_state, &[3, 7, 11])
        .expect("the reference forward runs")
        .logits;

    let (two, two_bus) = tp_engines(&config, 7, limit, 2);
    let handles: Vec<_> = two
        .into_iter()
        .map(|engine| run_logits(engine, &config, &[3, 7, 11]))
        .collect();
    let two_logits: Vec<Vec<f32>> = handles
        .into_iter()
        .map(|handle| handle.join().expect("a rank thread does not panic"))
        .collect();
    for (rank, logits) in two_logits.iter().enumerate() {
        let rel = rel_l2(logits, &expected);
        let differing = logits
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "measured: world 2 rank {rank} logits rel L2 = {rel:e}, {} of {} values differ",
            differing,
            expected.len()
        );
        assert!(
            rel <= LOGITS_REL_L2_BAR,
            "rank {rank} of 2: logits rel L2 {rel:e} above the declared {LOGITS_REL_L2_BAR:e}"
        );
    }

    let (four, four_bus) = tp_engines(&config, 7, limit, 4);
    let handles: Vec<_> = four
        .into_iter()
        .map(|engine| run_logits(engine, &config, &[3, 7, 11]))
        .collect();
    let four_logits: Vec<Vec<f32>> = handles
        .into_iter()
        .map(|handle| handle.join().expect("a rank thread does not panic"))
        .collect();
    for (rank, logits) in four_logits.iter().enumerate() {
        let rel = rel_l2(logits, &expected);
        let differing = logits
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "measured: world 4 rank {rank} logits rel L2 = {rel:e}, {} of {} values differ",
            differing,
            expected.len()
        );
        assert!(
            rel <= LOGITS_REL_L2_BAR,
            "rank {rank} of 4: logits rel L2 {rel:e} above the declared {LOGITS_REL_L2_BAR:e}"
        );
    }

    // The per-seam bar: compare each rank's reduced routed FFN output against the one-rank run's
    // f32 routed FFN output at the same (layer, token).
    let (one, one_bus) = tp_engines(&config, 7, limit, 1);
    let handle = run_logits(
        one.into_iter().next().expect("one engine"),
        &config,
        &[3, 7, 11],
    );
    handle.join().expect("the one-rank thread does not panic");
    let one_seams = one_bus.recorded[0].lock().expect("the bus is not poisoned");
    assert_eq!(one_seams.len(), config.n_layers);
    for world in [2usize, 4] {
        let bus = if world == 2 { &two_bus } else { &four_bus };
        for rank in 0..world {
            let seams = bus.recorded[rank].lock().expect("the bus is not poisoned");
            assert_eq!(
                seams.len(),
                config.n_layers,
                "rank {rank} of {world} recorded the wrong number of seams"
            );
            for (index, reference_seam) in one_seams.iter().enumerate() {
                let rel = rel_l2(&seams[index], reference_seam);
                println!("measured: world {world} rank {rank} seam {index} rel L2 = {rel:e}");
                assert!(
                    rel <= SEAM_REL_L2_BAR,
                    "rank {rank} of {world} seam {index} rel L2 {rel:e} above the declared {SEAM_REL_L2_BAR:e}"
                );
            }
        }
    }
}

#[test]
fn the_resident_source_is_not_read_again_per_token() {
    let config = ModelConfig::tiny();
    let limit = 1 << 22;
    let plan = TensorPlan::language_model(&config);
    let reads = Arc::new(AtomicUsize::new(0));
    let source = CountingSource {
        inner: LocalSource::from_plan(&plan, 7, 1, 0, config.moe_inter_dim),
        reads: Arc::clone(&reads),
    };
    let engine = InferenceEngine::new_tp(
        config.clone(),
        Box::new(source),
        limit,
        TpLayout {
            rank: 0,
            world_size: 1,
        },
        Box::new(BusCollective {
            rank: 0,
            world: 1,
            bus: Arc::new(Bus::new(1)),
        }),
    )
    .expect("the fixture supports the layout");
    let after_load = reads.load(Ordering::SeqCst);
    assert!(after_load > 0, "new_tp must read the rank's tensors once");
    let mut state = SequenceState::new(&config);
    engine
        .prefill(&mut state, &[3, 7])
        .expect("the resident forward runs");
    engine
        .prefill(&mut state, &[11])
        .expect("the resident forward runs again");
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after_load,
        "the per-token forward read the underlying source again"
    );
    assert!(engine.tp().is_some());
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
        Box::new(BusCollective {
            rank: 0,
            world: 3,
            bus: Arc::new(Bus::new(3)),
        }),
    )
    .err()
    .expect("3 does not divide the tiny moe_inter_dim");
    assert!(matches!(error, EngineError::Shape { .. }), "{error}");
}
