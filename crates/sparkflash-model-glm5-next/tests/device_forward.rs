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

//! The device arm against its host twin: [Glm5NextConfig::small] written as the release stores it,
//! 24 tokens through one, two and four ranks on one GPU, each world over a recording bus on the
//! exact f32 wire, against the host world of the same ranks. Every device rank must agree with
//! every other bit for bit. With the FP8 weights' inputs unquantised, rank 0's logits and every
//! reduce's seam sit within [EXACT_WIRE_REL_L2_BAR] of the host world's; with the release's
//! dynamic activation scheme, the median seam does, and the drift past it is printed. The device
//! runs the tokens [MAX_ROWS] a step, so a seam carries a step's rows, which are unpicked into the
//! host's one-token seams before they are compared.
//!
//! Rows batched in one step, from one sequence or several at different positions, must be the
//! bits of the same tokens run one a step. A world whose collective has a device leg, here an
//! in-process twin of the NCCL exchange, must be the bits of the host round trip. A speculative
//! verify's kept rows, and every step after its commit, must be the bits of the kept tokens run
//! one a step. Speculative greedy decoding, the MTP layer or a DFlash 2 block drafter drafting and
//! a verify keeping what greedy would, must be plain greedy decoding's tokens and logits, its
//! drafts the ones plain decoding drafts at the same positions. A greedy step's picks must be
//! the argmax of the logits the plain step returns, and the row it leaves on the device the plain
//! step's last kept row.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model::BackendError;
use sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint;
use sparkflash_model_glm5_next::convert::convert_rank;
use sparkflash_model_glm5_next::device::CudaHandle;
use sparkflash_model_glm5_next::{
    DeviceRank, DeviceSequence, Glm5NextConfig, Glm5NextWeights, MAX_ROWS,
};
use sparkflash_model_qwen4_exp::device::cubin_available;
use sparkflash_model_qwen4_exp::dflash::write_synthetic_drafter;
use sparkflash_weights::linalg::{self, EngineError, REFERENCE_TENSOR_LIMIT};
use sparkflash_weights::rank_file::rank_file_name;
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_weights::tp::{
    BitGatherHandle, EXACT_WIRE_REL_L2_BAR, LocalBus, LocalRank, TpCollective, TpLayout, TpRuntime,
    Wire,
};

const TOKENS: usize = 24;

fn tokens(config: &Glm5NextConfig) -> Vec<u32> {
    stride_tokens(config, TOKENS, 37, 11)
}

fn stride_tokens(config: &Glm5NextConfig, count: usize, stride: usize, first: usize) -> Vec<u32> {
    (0..count)
        .map(|index| ((index * stride + first) % config.vocab_size) as u32)
        .collect()
}

/// A CUDA context, or why this run cannot have one.
fn open() -> Option<Arc<CudaHandle>> {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return None;
    }
    match Cuda::load() {
        Ok(cuda) => Some(Arc::new(CudaHandle::new(cuda))),
        Err(error) => {
            eprintln!("skipped: no CUDA driver ({error})");
            None
        }
    }
}

fn source(directory: &Path) -> SafeTensorsSource {
    SafeTensorsSource::open(directory, &directory.join(INDEX_NAME)).expect("the fixture opens")
}

/// What a world's ranks load: the checkpoint, or the rank files converted from it.
#[derive(Clone, Copy)]
enum Load<'a> {
    Checkpoint(&'a Path),
    Converted(&'a Path),
}

impl Load<'_> {
    fn rank(
        self,
        cuda: &Arc<CudaHandle>,
        config: &Glm5NextConfig,
        layout: TpLayout,
    ) -> Result<DeviceRank, BackendError> {
        match self {
            Self::Checkpoint(directory) => {
                DeviceRank::load(Arc::clone(cuda), config, &source(directory), layout)
            }
            Self::Converted(directory) => {
                let file = directory.join(rank_file_name(layout.rank, layout.world_size));
                DeviceRank::load_converted(Arc::clone(cuda), config, &file, layout)
            }
        }
    }
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

/// The worst relative L2 over paired rows.
fn worst(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| linalg::relative_l2(a, b).unwrap())
        .fold(0.0, f64::max)
}

/// Rank 0's logits and seams once every rank's agree with rank 0's bit for bit.
fn agreed(
    world: usize,
    bus: &LocalBus,
    logits: Vec<Vec<Vec<f32>>>,
    arm: &str,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let seams = bus.seams(0).expect("the bus records");
    for rank in 1..world {
        assert_eq!(
            bits(&logits[rank]),
            bits(&logits[0]),
            "{arm} rank {rank} of {world}"
        );
        assert_eq!(
            bits(&bus.seams(rank).expect("the bus records")),
            bits(&seams),
            "{arm} rank {rank} of {world}'s seams"
        );
    }
    (logits.into_iter().next().expect("rank 0"), seams)
}

fn host(
    config: &Glm5NextConfig,
    directory: &Path,
    world: usize,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let source = source(directory);
    let ranks: Vec<Glm5NextWeights> = (0..world)
        .map(|rank| {
            let layout = TpLayout {
                rank,
                world_size: world,
            };
            Glm5NextWeights::load_tp(config, &source, REFERENCE_TENSOR_LIMIT, layout)
                .expect("the host splits the fixture")
        })
        .collect();
    let bus = Arc::new(LocalBus::recording(world).with_wire(Wire::F32));
    let logits = std::thread::scope(|scope| {
        let handles: Vec<_> = ranks
            .iter()
            .enumerate()
            .map(|(rank, weights)| {
                let adapter = LocalRank {
                    rank,
                    bus: Arc::clone(&bus),
                };
                scope.spawn(move || {
                    let tp = TpRuntime::new(weights.layout, Box::new(adapter.clone()))
                        .expect("the adapter is this rank");
                    let gather = BitGatherHandle::new(Box::new(adapter));
                    let mut state = weights.new_state();
                    weights
                        .forward_tp(&tp, &gather, &mut state, tokens)
                        .expect("the host rank runs")
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a host rank does not panic"))
            .collect()
    });
    agreed(world, &bus, logits, "host")
}

/// The in-process twin of the NCCL device leg: each rank publishes its payload's device address,
/// and every rank copies its peers' payloads into its slots on the device, the bytes the exchange
/// moves. The ranks share one context, so a peer's address is every rank's to read.
struct Exchange {
    sends: Mutex<Vec<u64>>,
    met: Barrier,
}

impl Exchange {
    fn new(world: usize) -> Arc<Self> {
        Arc::new(Self {
            sends: Mutex::new(vec![0; world]),
            met: Barrier::new(world),
        })
    }
}

/// A rank of the bus whose device leg is [Exchange], summing on `wire`.
struct DeviceBusRank {
    host: LocalRank,
    exchange: Arc<Exchange>,
    wire: Wire,
}

impl TpCollective for DeviceBusRank {
    fn world(&self) -> usize {
        self.host.world()
    }

    fn rank(&self) -> usize {
        self.host.rank
    }

    fn label(&self) -> &'static str {
        "in-process device exchange"
    }

    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        self.host.all_reduce_assign(values)
    }

    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.host.all_gather_assign(local, out)
    }

    fn device_wire(&self) -> Option<Wire> {
        Some(self.wire)
    }

    fn exchange_on_device(
        &mut self,
        _stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        let failed = |detail: String| EngineError::Shape {
            what: "device exchange twin".to_owned(),
            detail,
        };
        let cuda = Cuda::load().map_err(failed)?;
        // This rank's payload is written before a peer reads it, ...
        cuda.synchronize().map_err(failed)?;
        let rank = self.host.rank;
        self.exchange.sends.lock().unwrap()[rank] = send;
        self.exchange.met.wait();
        let sends = self.exchange.sends.lock().unwrap().clone();
        for (peer, &from) in sends.iter().enumerate() {
            if peer != rank {
                let slot = Deviceptr(slots + (peer * bytes) as u64);
                cuda.copy_dtod_async(&slot, &Deviceptr(from), bytes)
                    .map_err(failed)?;
            }
        }
        cuda.synchronize().map_err(failed)?;
        // ... and not overwritten until every peer has read it.
        self.exchange.met.wait();
        Ok(())
    }
}

/// What a rank's sequences draft with.
#[derive(Clone, Copy)]
enum Drafting<'a> {
    /// Nothing: [DeviceRank::new_sequence].
    No,
    /// The MTP layer: [DeviceRank::new_drafting_sequence].
    Mtp,
    /// The DFlash 2 drafter in this directory, loaded onto the rank:
    /// [DeviceRank::new_block_drafting_sequence].
    Block(&'a Path),
}

/// Every rank on its own thread with its own context handle and stream, holding one fresh
/// sequence of each of `capacities`, drafting with `drafting`, and then running `run`, its
/// collective the bus or, given an `exchange`, the bus with that device leg. A rank that cannot
/// load says so before any rank enters a collective, so a failure cannot leave its peers waiting.
#[allow(clippy::too_many_arguments)]
fn ranks<T: Send>(
    config: &Glm5NextConfig,
    load: Load<'_>,
    world: usize,
    bus: &Arc<LocalBus>,
    exchange: Option<&Arc<Exchange>>,
    capacities: &[usize],
    drafting: Drafting<'_>,
    run: impl Fn(
        &mut DeviceRank,
        &mut [DeviceSequence],
        &TpRuntime,
        &BitGatherHandle,
    ) -> Result<T, BackendError>
    + Sync,
) -> Vec<T> {
    let loaded = Barrier::new(world);
    let failed = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let adapter = LocalRank {
                    rank,
                    bus: Arc::clone(bus),
                };
                let (loaded, failed, run) = (&loaded, &failed, &run);
                scope.spawn(move || {
                    let layout = TpLayout {
                        rank,
                        world_size: world,
                    };
                    let cuda = Cuda::load().map(|cuda| Arc::new(CudaHandle::new(cuda)));
                    let weights = cuda.as_ref().map_err(Clone::clone).and_then(|cuda| {
                        let mut weights = load
                            .rank(cuda, config, layout)
                            .map_err(|error| error.to_string())?;
                        if let Drafting::Block(directory) = drafting {
                            weights
                                .load_block_drafter(directory)
                                .map_err(|error| error.to_string())?;
                        }
                        Ok(weights)
                    });
                    let sequences = weights.as_ref().map_err(Clone::clone).and_then(|weights| {
                        capacities
                            .iter()
                            .map(|&capacity| match drafting {
                                Drafting::No => weights.new_sequence(capacity),
                                Drafting::Mtp => weights.new_drafting_sequence(capacity),
                                Drafting::Block(_) => weights.new_block_drafting_sequence(capacity),
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|error| error.to_string())
                    });
                    if weights.is_err() || sequences.is_err() {
                        failed.store(true, Ordering::SeqCst);
                    }
                    loaded.wait();
                    if failed.load(Ordering::SeqCst) {
                        return Err(format!(
                            "rank {rank} of {world}: {:?} / {:?}",
                            weights.as_ref().err(),
                            sequences.as_ref().err()
                        ));
                    }
                    let (mut weights, mut sequences) = (weights?, sequences?);
                    println!(
                        "measured: world {world} rank {rank}: {} device bytes",
                        weights.device_bytes()
                    );
                    let collective: Box<dyn TpCollective> = match exchange {
                        Some(exchange) => Box::new(DeviceBusRank {
                            host: adapter.clone(),
                            exchange: Arc::clone(exchange),
                            wire: Wire::F32,
                        }),
                        None => Box::new(adapter.clone()),
                    };
                    let tp = TpRuntime::new(layout, collective).expect("the adapter is this rank");
                    let gather = BitGatherHandle::new(Box::new(adapter));
                    run(&mut weights, &mut sequences, &tp, &gather)
                        .map_err(|error| format!("rank {rank} of {world}: {error}"))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("a device rank does not panic")
                    .unwrap_or_else(|error| panic!("{error}"))
            })
            .collect()
    })
}

/// A step's seams in the host's order: each step reduces `sublayers` seams of its rows, and the
/// host one seam a sublayer a token.
fn per_token(seams: &[Vec<f32>], sublayers: usize, hidden: usize) -> Vec<Vec<f32>> {
    let mut unpicked = Vec::with_capacity(seams.len());
    for step in seams.chunks(sublayers) {
        for row in 0..step[0].len() / hidden {
            for seam in step {
                unpicked.push(seam[row * hidden..(row + 1) * hidden].to_vec());
            }
        }
    }
    unpicked
}

fn device(
    config: &Glm5NextConfig,
    load: Load<'_>,
    world: usize,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let bus = Arc::new(LocalBus::recording(world).with_wire(Wire::F32));
    let logits = ranks(
        config,
        load,
        world,
        &bus,
        None,
        &[tokens.len()],
        Drafting::No,
        |weights, sequences, tp, gather| {
            let logits = weights.forward_tp(tp, gather, &mut sequences[0], tokens)?;
            assert_eq!(sequences[0].position(), tokens.len());
            Ok(logits)
        },
    );
    let (logits, seams) = agreed(world, &bus, logits, "device");
    let sublayers = 2 * config.num_hidden_layers;
    (logits, per_token(&seams, sublayers, config.hidden_size))
}

/// One world's distance from the host world's: every seam's relative L2, the worst token's
/// logits, and the tokens whose argmax differs.
struct Distance {
    seams: Vec<f64>,
    logits: f64,
    flips: usize,
}

/// Runs `config` through the host and device worlds of one, two and four ranks over one fixture,
/// and prints what it measures, each seam past the bar by name.
fn measure(config: &Glm5NextConfig) -> Vec<(usize, Distance)> {
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), config, 7, 2).expect("the fixture writes");
    let tokens = tokens(config);
    let sublayers = 2 * config.num_hidden_layers;
    [1, 2, 4]
        .into_iter()
        .map(|world| {
            let (reference, reference_seams) = host(config, directory.path(), world, &tokens);
            let (logits, seams) =
                device(config, Load::Checkpoint(directory.path()), world, &tokens);
            // An attention and an MLP reduce per layer per token.
            assert_eq!(seams.len(), tokens.len() * sublayers);
            assert_eq!(seams.len(), reference_seams.len());
            let distance = Distance {
                seams: seams
                    .iter()
                    .zip(&reference_seams)
                    .map(|(ours, theirs)| linalg::relative_l2(ours, theirs).unwrap())
                    .collect(),
                logits: worst(&logits, &reference),
                flips: logits
                    .iter()
                    .zip(&reference)
                    .filter(|(ours, theirs)| linalg::argmax(ours) != linalg::argmax(theirs))
                    .count(),
            };
            let past: Vec<String> = distance
                .seams
                .iter()
                .enumerate()
                .filter(|(_, seam)| **seam > EXACT_WIRE_REL_L2_BAR)
                .map(|(index, seam)| {
                    format!(
                        "token {} layer {} {} {seam:.1e}",
                        index / sublayers,
                        index % sublayers / 2,
                        if index % 2 == 0 { "attention" } else { "ffn" }
                    )
                })
                .collect();
            println!(
                "measured: world {world}: worst seam rel L2 {:e}, worst logits {:e}, {} of {} \
                 argmax flips, {} of {} seams past the bar {past:?}",
                distance.seams.iter().copied().fold(0.0, f64::max),
                distance.logits,
                distance.flips,
                tokens.len(),
                past.len(),
                distance.seams.len()
            );
            (world, distance)
        })
        .collect()
}

/// With the FP8 weights' inputs unquantised on both sides, nothing between the device and the host
/// is discontinuous, so every seam and every token's logits sit within the exact bar.
#[test]
fn with_unquantised_activations_every_device_world_matches_the_host_world() {
    // The context this test holds keeps the device's primary context alive across the ranks'.
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig {
        dynamic_activations: false,
        ..Glm5NextConfig::small()
    };
    for (world, distance) in measure(&config) {
        for (index, seam) in distance.seams.iter().enumerate() {
            assert!(
                *seam <= EXACT_WIRE_REL_L2_BAR,
                "world {world} seam {index}: {seam:e}"
            );
        }
        assert!(
            distance.logits <= EXACT_WIRE_REL_L2_BAR,
            "world {world} logits: {:e}",
            distance.logits
        );
        assert_eq!(distance.flips, 0, "world {world}");
    }
}

/// The release quantises each FP8 weight's input per 128 columns, and a value the two sides
/// associate one ulp apart can sit on either side of a code's rounding boundary: one code moves,
/// and that seam and what follows it drift. The quantiser itself is the host's bit for bit on the
/// same input (glm_kernels), so this run asserts what is exact - every device rank agrees with
/// every other bit for bit - and that most seams still sit within the bar, which a quantiser the
/// device skipped or misplaced would fail at every seam; it prints the drift.
#[test]
fn with_the_release_activation_scheme_the_device_ranks_agree_and_most_seams_are_exact() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    assert!(config.dynamic_activations);
    for (world, distance) in measure(&config) {
        let mut seams = distance.seams.clone();
        seams.sort_by(f64::total_cmp);
        let median = seams[seams.len() / 2];
        assert!(
            median <= EXACT_WIRE_REL_L2_BAR,
            "world {world}: the median seam is {median:e}"
        );
    }
}

/// Three sequences, one already five tokens in, stepped together eight rows a step, against the
/// same tokens one a step: every row the same bits, at every world, and every rank agreeing.
#[test]
fn batched_rows_are_the_bits_of_one_token_steps() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let streams = [
        tokens(&config),
        stride_tokens(&config, 13, 53, 5),
        stride_tokens(&config, 6, 29, 101),
    ];
    let lead = 5;
    // More rows than a step, so the steps mix sequences and cut them.
    assert!(streams.iter().map(Vec::len).sum::<usize>() - lead > 2 * MAX_ROWS);
    let capacities: Vec<usize> = streams.iter().chain(&streams).map(Vec::len).collect();
    for world in [1, 2, 4] {
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let runs = ranks(
            &config,
            Load::Checkpoint(directory.path()),
            world,
            &bus,
            None,
            &capacities,
            Drafting::No,
            |weights, sequences, tp, gather| {
                let (alone, together) = sequences.split_at_mut(streams.len());
                let mut single = Vec::new();
                for (sequence, tokens) in alone.iter_mut().zip(&streams) {
                    let mut rows = Vec::new();
                    for token in tokens {
                        rows.extend(weights.forward_tp(tp, gather, sequence, &[*token])?);
                    }
                    single.push(rows);
                }
                let mut batched =
                    vec![weights.forward_tp(tp, gather, &mut together[0], &streams[0][..lead])?];
                let [first, second, third] = together else {
                    unreachable!("three sequences");
                };
                let rest = weights.forward_batch(
                    tp,
                    gather,
                    &mut [
                        (first, &streams[0][lead..]),
                        (second, &streams[1]),
                        (third, &streams[2]),
                    ],
                )?;
                batched[0].extend(rest[0].iter().cloned());
                batched.extend(rest.into_iter().skip(1));
                for (sequence, tokens) in together.iter().zip(&streams) {
                    assert_eq!(sequence.position(), tokens.len());
                }
                Ok((single, batched))
            },
        );
        for (rank, (single, batched)) in runs.iter().enumerate() {
            for (stream, (alone, together)) in single.iter().zip(batched).enumerate() {
                assert_eq!(alone.len(), streams[stream].len());
                for (token, (a, b)) in alone.iter().zip(together).enumerate() {
                    assert!(
                        a.iter()
                            .map(|v| v.to_bits())
                            .eq(b.iter().map(|v| v.to_bits())),
                        "world {world} rank {rank} stream {stream} token {token}: batched rows \
                         are not the one-token bits"
                    );
                }
                let first = &runs[0].0[stream];
                assert_eq!(bits(alone), bits(first), "world {world} rank {rank}");
            }
        }
        println!(
            "measured: world {world}: {} rows over 3 streams, batched bit-equal to one-token steps",
            streams.iter().map(Vec::len).sum::<usize>()
        );
    }
}

/// Three sequences verify drafts together, up to [MAX_ROWS] rows a step, and each commit keeps
/// a prefix of its rows: one row, all of them, or some between. A verify's rows past the kept
/// ones are drafts the model would not have made (a token other than the stream's), so a commit
/// that kept them would show. Every kept row, and every row after a commit, is the bits of the
/// same stream run one token a step.
#[test]
fn a_verify_keeps_the_bits_of_its_kept_rows() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let streams = [
        tokens(&config),
        stride_tokens(&config, 13, 53, 5),
        stride_tokens(&config, 6, 29, 101),
    ];
    let lead = 2;
    // Each stream's verify widths in turn, every step's together at most MAX_ROWS.
    let widths = [[4, 3, 1], [2, 1, 4], [3, 2, 2]];
    assert!(
        widths
            .iter()
            .all(|step| step.iter().sum::<usize>() <= MAX_ROWS)
    );
    let capacities: Vec<usize> = streams.iter().chain(&streams).map(Vec::len).collect();
    let vocab = config.vocab_size as u32;
    for world in [1, 2, 4] {
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let runs = ranks(
            &config,
            Load::Checkpoint(directory.path()),
            world,
            &bus,
            None,
            &capacities,
            Drafting::No,
            |weights, sequences, tp, gather| {
                let (alone, verified) = sequences.split_at_mut(streams.len());
                let mut single = Vec::new();
                for (sequence, tokens) in alone.iter_mut().zip(&streams) {
                    let mut rows = Vec::new();
                    for token in tokens {
                        rows.extend(weights.forward_tp(tp, gather, sequence, &[*token])?);
                    }
                    single.push(rows);
                }
                let mut kept_rows: Vec<Vec<Vec<f32>>> = Vec::new();
                for (sequence, tokens) in verified.iter_mut().zip(&streams) {
                    kept_rows.push(weights.forward_tp(tp, gather, sequence, &tokens[..lead])?);
                }
                let mut kept_counts = Vec::new();
                for step in 0.. {
                    let mut chunks = Vec::new();
                    for (index, (sequence, tokens)) in verified.iter().zip(&streams).enumerate() {
                        let at = sequence.position();
                        let width = widths[step % widths.len()][index].min(tokens.len() - at);
                        let kept = 1 + (step + index) % width.max(1);
                        // The kept rows are the stream's; the rest are drafts it rejects.
                        let chunk: Vec<u32> = (0..width)
                            .map(|row| {
                                let token = tokens[(at + row).min(tokens.len() - 1)];
                                if row < kept {
                                    token
                                } else {
                                    (token + 1 + row as u32) % vocab
                                }
                            })
                            .collect();
                        chunks.push((chunk, kept));
                    }
                    if chunks.iter().all(|(chunk, _)| chunk.is_empty()) {
                        break;
                    }
                    let mut batch: Vec<(&mut DeviceSequence, &[u32])> = verified
                        .iter_mut()
                        .zip(&chunks)
                        .map(|(sequence, (chunk, _))| (sequence, chunk.as_slice()))
                        .collect();
                    let logits = weights.verify_batch(tp, gather, &mut batch)?;
                    drop(batch);
                    for (index, ((chunk, kept), rows)) in chunks.iter().zip(logits).enumerate() {
                        if chunk.is_empty() {
                            continue;
                        }
                        kept_rows[index].extend(rows.into_iter().take(*kept));
                        if chunk.len() > 1 {
                            weights.commit(&mut verified[index], *kept)?;
                        }
                        kept_counts.push((chunk.len(), *kept));
                    }
                }
                for (sequence, tokens) in verified.iter().zip(&streams) {
                    assert_eq!(sequence.position(), tokens.len());
                }
                Ok((single, kept_rows, kept_counts))
            },
        );
        let (_, _, counts) = &runs[0];
        for kind in [
            |(width, kept): (usize, usize)| width > 1 && kept == 1,
            |(width, kept): (usize, usize)| width > 1 && kept == width,
            |(width, kept): (usize, usize)| kept > 1 && kept < width,
        ] {
            assert!(counts.iter().copied().any(kind), "{counts:?}");
        }
        for (rank, (single, kept_rows, _)) in runs.iter().enumerate() {
            for (stream, (alone, verified)) in single.iter().zip(kept_rows).enumerate() {
                assert_eq!(alone.len(), streams[stream].len());
                assert_eq!(verified.len(), alone.len());
                for (token, (a, b)) in alone.iter().zip(verified).enumerate() {
                    assert!(
                        a.iter()
                            .map(|v| v.to_bits())
                            .eq(b.iter().map(|v| v.to_bits())),
                        "world {world} rank {rank} stream {stream} token {token}: a verified \
                         row is not the one-token bits"
                    );
                }
                assert_eq!(
                    bits(alone),
                    bits(&runs[0].0[stream]),
                    "world {world} rank {rank}"
                );
            }
        }
        println!(
            "measured: world {world}: {} verifies over 3 streams (width, kept) {counts:?}, kept \
             rows bit-equal to one-token steps",
            counts.len()
        );
    }
}

/// Greedy decoding of a stream: its logits at every position, its tokens past the prompt, the
/// first from the prompt's last row, and the drafts after each token but the last.
struct Greedy {
    logits: Vec<Vec<f32>>,
    tokens: Vec<u32>,
    drafts: Vec<Vec<u32>>,
}

fn greatest(row: &[f32]) -> u32 {
    linalg::argmax(row).expect("a row of logits") as u32
}

/// Each prompt prefilled alone and decoded greedily one token a step, `generate` steps, drafting
/// `count` tokens after each token it runs when given a count.
fn greedy(
    weights: &mut DeviceRank,
    sequences: &mut [DeviceSequence],
    (tp, gather): (&TpRuntime, &BitGatherHandle),
    prompts: &[Vec<u32>],
    generate: usize,
    count: Option<usize>,
) -> Result<Vec<Greedy>, BackendError> {
    let mut streams = Vec::new();
    for (sequence, prompt) in sequences.iter_mut().zip(prompts) {
        let mut logits = weights.forward_tp(tp, gather, sequence, prompt)?;
        let mut tokens = vec![greatest(logits.last().expect("a prompt row"))];
        let mut drafts = Vec::new();
        for _ in 0..generate {
            let next = *tokens.last().expect("a token");
            if let Some(count) = count {
                let mut drafted =
                    weights.draft(tp, gather, &mut [(&mut *sequence, next)], count)?;
                drafts.push(drafted.remove(0));
            }
            logits.extend(weights.forward_tp(tp, gather, sequence, &[next])?);
            tokens.push(greatest(logits.last().expect("a row")));
        }
        streams.push(Greedy {
            logits,
            tokens,
            drafts,
        });
    }
    Ok(streams)
}

/// Three streams decode greedily twice on sequences drafting with `drafting`, at each of
/// `worlds`, the drafting ranks' collective the bus or, `device_leg`, the bus with an [Exchange]:
/// one a token a step, drafting after every token, and speculatively, all three drafting together,
/// verifying and keeping what greedy keeps. The verify's rows are the stream's own tokens up to a
/// scheduled count and the drafts after it, so commits keep one row, every row and some between
/// however well the fixture drafts. Both must be the bits of greedy decoding on sequences that do
/// not draft, and each speculative draft the bits of the one-token draft at its position: the
/// drafter's caches after a commit are the ones plain decoding leaves. Every rank drafts the same
/// tokens. Returns each world's one-token drafts of each stream.
fn speculative_greedy(
    config: &Glm5NextConfig,
    checkpoint: &Path,
    drafting: Drafting<'_>,
    worlds: &[usize],
    device_leg: bool,
) -> Vec<Vec<Vec<Vec<u32>>>> {
    // A prefill of two steps, and a one-token prompt whose step fills no MTP row.
    let prompts = [
        stride_tokens(config, 9, 37, 11),
        stride_tokens(config, 5, 53, 5),
        stride_tokens(config, 1, 29, 101),
    ];
    const GENERATE: usize = 10;
    const COUNT: usize = 3;
    // Each stream's verify widths in turn, every step's together at most MAX_ROWS.
    let widths = [[4, 2, 2], [1, 4, 3], [3, 3, 2]];
    assert!(
        widths
            .iter()
            .all(|step| step.iter().sum::<usize>() <= MAX_ROWS
                && step.iter().all(|w| *w <= COUNT + 1))
    );
    let capacities: Vec<usize> = prompts
        .iter()
        .map(|prompt| prompt.len() + GENERATE + COUNT)
        .collect();
    let twice: Vec<usize> = capacities.iter().chain(&capacities).copied().collect();
    let mut every = Vec::new();
    for &world in worlds {
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let plain = ranks(
            config,
            Load::Checkpoint(checkpoint),
            world,
            &bus,
            None,
            &capacities,
            Drafting::No,
            |weights, sequences, tp, gather| {
                greedy(weights, sequences, (tp, gather), &prompts, GENERATE, None)
            },
        );
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let exchange = device_leg.then(|| Exchange::new(world));
        let runs = ranks(
            config,
            Load::Checkpoint(checkpoint),
            world,
            &bus,
            exchange.as_ref(),
            &twice,
            drafting,
            |weights, sequences, tp, gather| {
                let (alone, speculative) = sequences.split_at_mut(prompts.len());
                let single = greedy(
                    weights,
                    alone,
                    (tp, gather),
                    &prompts,
                    GENERATE,
                    Some(COUNT),
                )?;
                let [first, second, third] = speculative else {
                    unreachable!("three sequences");
                };
                let mut logits = weights.forward_batch(
                    tp,
                    gather,
                    &mut [
                        (&mut *first, &prompts[0]),
                        (&mut *second, &prompts[1]),
                        (&mut *third, &prompts[2]),
                    ],
                )?;
                let mut tokens: Vec<Vec<u32>> = logits
                    .iter()
                    .map(|rows| vec![greatest(rows.last().expect("a prompt row"))])
                    .collect();
                let mut drafted = Vec::new();
                let mut counts = Vec::new();
                for step in 0.. {
                    let mut active: Vec<(usize, &mut DeviceSequence)> =
                        [&mut *first, &mut *second, &mut *third]
                            .into_iter()
                            .enumerate()
                            .filter(|(index, sequence)| {
                                sequence.position() < prompts[*index].len() + GENERATE
                            })
                            .collect();
                    if active.is_empty() {
                        break;
                    }
                    let mut batch: Vec<(&mut DeviceSequence, u32)> = active
                        .iter_mut()
                        .map(|(index, sequence)| {
                            (&mut **sequence, *tokens[*index].last().expect("a token"))
                        })
                        .collect();
                    let drafts = weights.draft(tp, gather, &mut batch, COUNT)?;
                    drop(batch);
                    let mut chunks = Vec::new();
                    for ((index, sequence), drafts) in active.iter().zip(&drafts) {
                        let at = sequence.position() - prompts[*index].len();
                        let confidence: Vec<u32> = sequence
                            .draft_confidence()
                            .iter()
                            .map(|value| value.to_bits())
                            .collect();
                        drafted.push((*index, at, drafts.clone(), confidence));
                        let width = widths[step % widths.len()][*index].min(GENERATE - at);
                        let want = 1 + (step + index) % width;
                        // The next token, greedy's tokens up to `want`, then the drafts.
                        let chunk: Vec<u32> = (0..width)
                            .map(|row| match row {
                                0 => *tokens[*index].last().expect("a token"),
                                _ if row < want => single[*index].tokens[at + row],
                                _ => drafts[row - 1],
                            })
                            .collect();
                        chunks.push(chunk);
                    }
                    let mut batch: Vec<(&mut DeviceSequence, &[u32])> = active
                        .iter_mut()
                        .zip(&chunks)
                        .map(|((_, sequence), chunk)| (&mut **sequence, chunk.as_slice()))
                        .collect();
                    let verified = weights.verify_batch(tp, gather, &mut batch)?;
                    drop(batch);
                    for (((index, sequence), chunk), rows) in
                        active.iter_mut().zip(&chunks).zip(verified)
                    {
                        // Greedy keeps each row whose token the row before it picks.
                        let kept = 1
                            + (1..chunk.len())
                                .take_while(|&row| chunk[row] == greatest(&rows[row - 1]))
                                .count();
                        tokens[*index].extend(&chunk[1..kept]);
                        tokens[*index].push(greatest(&rows[kept - 1]));
                        logits[*index].extend(rows.into_iter().take(kept));
                        if chunk.len() > 1 {
                            weights.commit(sequence, kept)?;
                        }
                        counts.push((chunk.len(), kept));
                    }
                }
                Ok((single, logits, tokens, drafted, counts))
            },
        );
        let (_, _, _, _, counts) = &runs[0];
        for kind in [
            |(width, kept): (usize, usize)| width > 1 && kept == 1,
            |(width, kept): (usize, usize)| width > 1 && kept == width,
            |(width, kept): (usize, usize)| kept > 1 && kept < width,
        ] {
            assert!(counts.iter().copied().any(kind), "{counts:?}");
        }
        for (rank, (single, logits, tokens, drafted, _)) in runs.iter().enumerate() {
            for (stream, alone) in single.iter().enumerate() {
                let expected = &plain[0][stream];
                assert_eq!(
                    alone.tokens, expected.tokens,
                    "world {world} rank {rank} stream {stream}"
                );
                assert_eq!(
                    tokens[stream], expected.tokens,
                    "world {world} rank {rank} stream {stream}"
                );
                assert_eq!(
                    bits(&alone.logits),
                    bits(&expected.logits),
                    "world {world} rank {rank} stream {stream}: drafting changed the logits"
                );
                assert_eq!(
                    bits(&logits[stream]),
                    bits(&expected.logits),
                    "world {world} rank {rank} stream {stream}: speculative logits are not greedy's"
                );
                assert_eq!(
                    alone.drafts, runs[0].0[stream].drafts,
                    "world {world} rank {rank}"
                );
            }
            for (stream, at, drafts, confidence) in drafted {
                assert_eq!(
                    drafts, &single[*stream].drafts[*at],
                    "world {world} rank {rank} stream {stream} token {at}: a speculative draft is \
                     not the one-token draft"
                );
                // The block drafter's confidence in each draft, a probability; the MTP layer
                // keeps none.
                let block = matches!(drafting, Drafting::Block(_));
                assert_eq!(
                    confidence.len(),
                    if block { drafts.len() } else { 0 },
                    "world {world} rank {rank} stream {stream} token {at}"
                );
                assert!(
                    confidence
                        .iter()
                        .map(|&bits| f32::from_bits(bits))
                        .all(|value| value > 0.0 && value <= 1.0),
                    "world {world} rank {rank} stream {stream} token {at}: {confidence:?}"
                );
            }
            // Every rank decides a verify's depth from the same confidences.
            assert_eq!(drafted, &runs[0].3, "world {world} rank {rank}");
        }
        let accepted = runs[0]
            .3
            .iter()
            .map(|(stream, at, drafts, _)| {
                let greedy = &plain[0][*stream].tokens[at + 1..];
                drafts
                    .iter()
                    .zip(greedy)
                    .take_while(|(draft, token)| draft == token)
                    .count()
            })
            .sum::<usize>();
        // A drafter broken to one token would pass every check above, so the drafts must vary.
        let distinct = runs[0]
            .3
            .iter()
            .flat_map(|(_, _, drafts, _)| drafts)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            distinct > 1,
            "world {world}: every draft is one token: {:?}",
            runs[0].3
        );
        let leg = if device_leg { "device leg" } else { "bus" };
        println!(
            "measured: world {world} {leg}: {} verifies over 3 streams (width, kept) {counts:?}, \
             {} drafts checked ({distinct} distinct tokens, {accepted} greedy's), speculative \
             greedy bit-equal to greedy",
            counts.len(),
            runs[0].3.len()
        );
        every.push(runs[0].0.iter().map(|alone| alone.drafts.clone()).collect());
    }
    every
}

#[test]
fn speculative_greedy_is_the_bits_of_greedy() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    speculative_greedy(&config, directory.path(), Drafting::Mtp, &[1, 2, 4], false);
}

/// A DFlash 2 drafter for [Glm5NextConfig::small]: two layers capturing target layers 1 and 3, a
/// window of 8 so the streams outrun the ring and the step's commit skips positions no query can
/// see, the release's block, convolution and selector geometry at the fixture's widths, and its
/// two query heads to a key-value head, which four ranks shard as they do the release's.
const SMALL_DRAFTER: &str = r#"{
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": false,
    "dflash_config": {
        "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
        "mask_token_id": 255, "selector_rank": 32, "selector_top_k": 16,
        "target_layer_ids": [1, 3]
    },
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 256, "intermediate_size": 512,
    "is_causal": false,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "model_type": "qwen3", "num_attention_heads": 8, "num_hidden_layers": 2,
    "num_key_value_heads": 4, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "sliding_window": 8, "vocab_size": 256
}"#;

/// [speculative_greedy] with the block drafter sharded over the ranks, each summing the ranks'
/// partials at every residual add, scoring its slice of the head and merging the ranks'
/// candidates, on the bus and on the device leg: the device leg's sums and exchanged candidates
/// are the bus's, so its drafts are too.
#[test]
fn block_speculative_greedy_is_the_bits_of_greedy() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let drafter = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_drafter(drafter.path(), SMALL_DRAFTER, 11).expect("the drafter writes");
    let drafting = Drafting::Block(drafter.path());
    let bus = speculative_greedy(&config, directory.path(), drafting, &[1, 2, 4], false);
    let device = speculative_greedy(&config, directory.path(), drafting, &[2, 4], true);
    assert_eq!(
        device,
        bus[1..],
        "the device leg's drafts are not the bus's"
    );
}

/// Every rank of two and four with a device leg runs prefill and one-token steps of three
/// sequences to the bits of the host round trip: the peers' partials summed in q4e_mhc_post in
/// rank order are Wire::F32's sum, and the exchanged logits are the gathered ones.
#[test]
fn the_device_leg_is_the_bits_of_the_host_round_trip() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let streams = [
        tokens(&config),
        stride_tokens(&config, 13, 53, 5),
        stride_tokens(&config, 6, 29, 101),
    ];
    const DECODE: usize = 3;
    let capacities: Vec<usize> = streams.iter().map(|tokens| tokens.len() + DECODE).collect();
    for world in [2, 4] {
        let run = |exchange: Option<&Arc<Exchange>>| {
            let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
            ranks(
                &config,
                Load::Checkpoint(directory.path()),
                world,
                &bus,
                exchange,
                &capacities,
                Drafting::No,
                |weights, sequences, tp, gather| {
                    let [first, second, third] = sequences else {
                        unreachable!("three sequences");
                    };
                    let mut logits = weights.forward_batch(
                        tp,
                        gather,
                        &mut [
                            (&mut *first, &streams[0]),
                            (&mut *second, &streams[1]),
                            (&mut *third, &streams[2]),
                        ],
                    )?;
                    for step in 0..DECODE {
                        let next: Vec<[u32; 1]> = (0..3)
                            .map(|member| {
                                [((step * 17 + member * 5 + 3) % config.vocab_size) as u32]
                            })
                            .collect();
                        let rows = weights.forward_batch(
                            tp,
                            gather,
                            &mut [
                                (&mut *first, &next[0][..]),
                                (&mut *second, &next[1][..]),
                                (&mut *third, &next[2][..]),
                            ],
                        )?;
                        for (all, more) in logits.iter_mut().zip(rows) {
                            all.extend(more);
                        }
                    }
                    Ok(logits)
                },
            )
        };
        let host = run(None);
        let device = run(Some(&Exchange::new(world)));
        for (rank, logits) in device.iter().enumerate() {
            for (stream, rows) in logits.iter().enumerate() {
                assert_eq!(rows.len(), streams[stream].len() + DECODE);
                assert_eq!(
                    bits(rows),
                    bits(&host[0][stream]),
                    "world {world} rank {rank} stream {stream}: the device leg is not the host \
                     round trip's bits"
                );
            }
        }
        println!(
            "measured: world {world}: {} rows over 3 streams, the device leg bit-equal to the host \
             round trip",
            capacities.iter().sum::<usize>()
        );
    }
}

/// Each sequence with its tokens, as a batch.
fn pairs<'s, T: AsRef<[u32]>>(
    sequences: &'s mut [DeviceSequence],
    tokens: &'s [T],
) -> Vec<(&'s mut DeviceSequence, &'s [u32])> {
    sequences
        .iter_mut()
        .zip(tokens)
        .map(|(sequence, tokens)| (sequence, tokens.as_ref()))
        .collect()
}

/// A plain step's logits next to the greedy step of the same rows: its picks, the last row each
/// sequence left on the device, and which of the plain rows that is.
type Picked = (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<usize>);

#[test]
fn a_greedy_step_picks_its_logits_argmax_and_keeps_its_last_row() {
    let Some(_cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    // Prompts that share one step, so each one's last row is among the step's logits.
    let prompts = [
        stride_tokens(&config, 3, 37, 11),
        stride_tokens(&config, 2, 53, 5),
        stride_tokens(&config, 3, 29, 101),
    ];
    // A verify of three rows keeping two, one of one row, and one of two keeping both.
    let verifies = [
        (stride_tokens(&config, 3, 41, 17), 2),
        (stride_tokens(&config, 1, 43, 3), 1),
        (stride_tokens(&config, 2, 47, 61), 2),
    ];
    assert!(prompts.iter().map(Vec::len).sum::<usize>() <= MAX_ROWS);
    let capacities: Vec<usize> = prompts
        .iter()
        .chain(&prompts)
        .map(|prompt| prompt.len() + 4)
        .collect();
    let last = |weights: &DeviceRank, sequences: &[DeviceSequence]| {
        sequences
            .iter()
            .map(|sequence| weights.last_logits(sequence))
            .collect::<Result<Vec<_>, _>>()
    };
    for (world, device_leg) in [(1, false), (2, false), (4, false), (2, true), (4, true)] {
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let exchange = device_leg.then(|| Exchange::new(world));
        let runs = ranks(
            &config,
            Load::Checkpoint(directory.path()),
            world,
            &bus,
            exchange.as_ref(),
            &capacities,
            Drafting::No,
            |weights, sequences, tp, gather| {
                let (plain, greedy) = sequences.split_at_mut(prompts.len());
                let mut steps: Vec<Picked> = Vec::new();
                // Each step runs the plain sequences and then the greedy ones, whose LM head is
                // then this rank's last.
                let rows = weights.forward_batch(tp, gather, &mut pairs(plain, &prompts))?;
                let picks =
                    weights.forward_batch_greedy(tp, gather, &mut pairs(greedy, &prompts))?;
                let at = prompts.iter().map(|prompt| prompt.len() - 1).collect();
                steps.push((rows, picks, last(weights, greedy)?, at));
                let chunks: Vec<&[u32]> = verifies.iter().map(|(chunk, _)| &chunk[..]).collect();
                // A verify commits before its rank's next step, so each batch commits before the
                // other runs.
                let rows = weights.verify_batch(tp, gather, &mut pairs(plain, &chunks))?;
                for (index, (chunk, kept)) in verifies.iter().enumerate() {
                    if chunk.len() > 1 {
                        weights.commit(&mut plain[index], *kept)?;
                    }
                }
                let picks = weights.verify_batch_greedy(tp, gather, &mut pairs(greedy, &chunks))?;
                let pending = weights.last_logits(&greedy[0]).err();
                for (index, (chunk, kept)) in verifies.iter().enumerate() {
                    if chunk.len() > 1 {
                        weights.commit(&mut greedy[index], *kept)?;
                    }
                }
                let at = verifies.iter().map(|(_, kept)| kept - 1).collect();
                steps.push((rows, picks, last(weights, greedy)?, at));
                // A later LM head takes the logits of the sequences it does not run.
                let next = [[5u32]];
                let rows = weights.forward_batch(tp, gather, &mut pairs(&mut plain[..1], &next))?;
                let picks = weights.forward_batch_greedy(
                    tp,
                    gather,
                    &mut pairs(&mut greedy[..1], &next),
                )?;
                steps.push((rows, picks, last(weights, &greedy[..1])?, vec![0]));
                let refused = [
                    weights.last_logits(&greedy[1]).err(),
                    weights.last_logits(&plain[0]).err(),
                ];
                Ok((steps, pending, refused))
            },
        );
        for (rank, (steps, pending, refused)) in runs.iter().enumerate() {
            let who = format!("world {world} device leg {device_leg} rank {rank}");
            for (step, (rows, picks, last, at)) in steps.iter().enumerate() {
                assert_eq!(
                    bits(&rows.concat()),
                    bits(&runs[0].0[step].0.concat()),
                    "{who}"
                );
                for (stream, rows) in rows.iter().enumerate() {
                    let argmax: Vec<u32> = rows.iter().map(|row| greatest(row)).collect();
                    assert_eq!(
                        picks[stream], argmax,
                        "{who} step {step} stream {stream}: the picks are not the argmax"
                    );
                    assert_eq!(
                        bits(&last[stream..=stream]),
                        bits(&rows[at[stream]..=at[stream]]),
                        "{who} step {step} stream {stream}: the last logits are not the last row's"
                    );
                }
            }
            for (error, says) in [
                (pending, "not committed"),
                (&refused[0], "are gone"),
                (&refused[1], "not a greedy step"),
            ] {
                let error = error.as_ref().map(ToString::to_string);
                assert!(
                    error.as_ref().is_some_and(|error| error.contains(says)),
                    "{who}: wanted a refusal saying {says:?}, got {error:?}"
                );
            }
        }
    }
}

/// Every rank of one, two and four booted from its converted file runs the tokens to the same
/// logits and seams, bit for bit, as the rank uploaded from the checkpoint; a rank file is refused
/// by a configuration whose walk it does not hold.
#[test]
fn a_converted_rank_is_the_bits_of_the_checkpoint_rank() {
    let Some(cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let converted = tempfile::tempdir().expect("a temporary directory");
    let tokens = tokens(&config);
    for world in [1, 2, 4] {
        for rank in 0..world {
            let layout = TpLayout {
                rank,
                world_size: world,
            };
            convert_rank(
                &source(directory.path()),
                &config,
                layout,
                converted.path(),
                "fixture",
            )
            .expect("the rank converts");
        }
        let (logits, seams) = device(&config, Load::Checkpoint(directory.path()), world, &tokens);
        let (booted, booted_seams) =
            device(&config, Load::Converted(converted.path()), world, &tokens);
        assert_eq!(bits(&booted), bits(&logits), "world {world} logits");
        assert_eq!(bits(&booted_seams), bits(&seams), "world {world} seams");
        println!("measured: world {world}: converted ranks bit-equal to checkpoint ranks");
    }
    let narrower = Glm5NextConfig {
        moe_intermediate_size: config.moe_intermediate_size / 2,
        ..config.clone()
    };
    let whole = TpLayout {
        rank: 0,
        world_size: 1,
    };
    let file = converted.path().join(rank_file_name(0, 1));
    let error = DeviceRank::load_converted(Arc::clone(&cuda), &narrower, &file, whole)
        .err()
        .expect("narrower experts are another walk");
    assert!(error.to_string().contains("expected"), "{error}");
}

#[test]
fn a_device_rank_refuses_what_it_cannot_run() {
    let Some(cuda) = open() else {
        return;
    };
    let config = Glm5NextConfig::small();
    let directory = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(directory.path(), &config, 7, 2).expect("the fixture writes");
    let whole = TpLayout {
        rank: 0,
        world_size: 1,
    };
    let source = source(directory.path());
    let three = TpLayout {
        rank: 0,
        world_size: 3,
    };
    let error = DeviceRank::load(Arc::clone(&cuda), &config, &source, three)
        .err()
        .expect("three ranks do not split the fixture");
    assert!(error.to_string().contains("split"), "{error}");
    let mut weights =
        DeviceRank::load(Arc::clone(&cuda), &config, &source, whole).expect("one rank loads");
    let error = weights
        .new_sequence(config.max_position_embeddings + 1)
        .err()
        .expect("a sequence past the context window");
    assert!(error.to_string().contains("context window"), "{error}");
    let mut sequence = weights.new_sequence(2).expect("a sequence");
    let bus = Arc::new(LocalBus::new(1));
    let adapter = LocalRank { rank: 0, bus };
    let tp = TpRuntime::new(whole, Box::new(adapter.clone())).expect("one rank");
    let gather = BitGatherHandle::new(Box::new(adapter));
    let error = weights
        .forward_tp(&tp, &gather, &mut sequence, &[config.vocab_size as u32])
        .unwrap_err();
    assert!(error.to_string().contains("vocabulary"), "{error}");
    let error = weights
        .forward_tp(&tp, &gather, &mut sequence, &[1, 2, 3])
        .unwrap_err();
    assert!(error.to_string().contains("positions"), "{error}");
    assert_eq!(sequence.position(), 0, "a refused chunk advances nothing");
    let mut other = weights.new_sequence(4).expect("a sequence");
    let error = weights
        .forward_batch(
            &tp,
            &gather,
            &mut [(&mut other, &[1, 2]), (&mut sequence, &[1, 2, 3])],
        )
        .unwrap_err();
    assert!(error.to_string().contains("positions"), "{error}");
    assert_eq!(
        (other.position(), sequence.position()),
        (0, 0),
        "a batch with a refused member runs none of it"
    );
    let mut drafts = weights.new_sequence(MAX_ROWS + 1).expect("a sequence");
    let error = weights
        .verify_batch(&tp, &gather, &mut [(&mut drafts, &[1; MAX_ROWS + 1])])
        .unwrap_err();
    assert!(error.to_string().contains("one step"), "{error}");
    let error = weights.commit(&mut drafts, 1).unwrap_err();
    assert!(error.to_string().contains("no verify"), "{error}");
    weights
        .verify_batch(&tp, &gather, &mut [(&mut drafts, &[1, 2, 3])])
        .expect("a verify of three rows");
    let error = weights
        .forward_tp(&tp, &gather, &mut drafts, &[4])
        .unwrap_err();
    assert!(error.to_string().contains("not committed"), "{error}");
    for kept in [0, 4] {
        let error = weights.commit(&mut drafts, kept).unwrap_err();
        assert!(error.to_string().contains("keeps 1 to all"), "{error}");
    }
    assert_eq!(drafts.position(), 3, "a refused commit gives back nothing");
    weights
        .forward_tp(&tp, &gather, &mut other, &[1])
        .expect("another sequence's step");
    let error = weights.commit(&mut drafts, 2).unwrap_err();
    assert!(error.to_string().contains("next step"), "{error}");
    let error = weights
        .draft(&tp, &gather, &mut [(&mut other, 1)], 1)
        .unwrap_err();
    assert!(error.to_string().contains("keeps no MTP state"), "{error}");
    let mut drafting = weights
        .new_drafting_sequence(4)
        .expect("a drafting sequence");
    let error = weights
        .draft(&tp, &gather, &mut [(&mut drafting, 1)], 1)
        .unwrap_err();
    assert!(error.to_string().contains("run no step"), "{error}");
    weights
        .verify_batch(&tp, &gather, &mut [(&mut drafting, &[1, 2])])
        .expect("a verify of two rows");
    let error = weights
        .draft(&tp, &gather, &mut [(&mut drafting, 3)], 1)
        .unwrap_err();
    assert!(error.to_string().contains("not committed"), "{error}");
    weights
        .commit(&mut drafting, 2)
        .expect("a commit of both rows");
    let error = weights
        .draft(&tp, &gather, &mut [(&mut drafting, 3)], 4)
        .unwrap_err();
    assert!(error.to_string().contains("past the sequence's"), "{error}");
    let error = weights
        .draft(
            &tp,
            &gather,
            &mut [(&mut drafting, config.vocab_size as u32)],
            1,
        )
        .unwrap_err();
    assert!(error.to_string().contains("vocabulary"), "{error}");
    assert_eq!(
        weights
            .draft(&tp, &gather, &mut [(&mut drafting, 3)], 3)
            .expect("three drafts fill the last positions")[0]
            .len(),
        3
    );
    let mut many: Vec<DeviceSequence> = (0..=MAX_ROWS)
        .map(|_| {
            weights
                .new_drafting_sequence(2)
                .expect("a drafting sequence")
        })
        .collect();
    let mut batch: Vec<(&mut DeviceSequence, u32)> =
        many.iter_mut().map(|sequence| (sequence, 1)).collect();
    let error = weights.draft(&tp, &gather, &mut batch, 1).unwrap_err();
    assert!(error.to_string().contains("at most"), "{error}");
    drop(batch);
    let without = Glm5NextConfig {
        num_nextn_predict_layers: 0,
        ..config.clone()
    };
    let mut plain =
        DeviceRank::load(Arc::clone(&cuda), &without, &source, whole).expect("a rank without MTP");
    let error = plain.new_drafting_sequence(4).err().expect("no MTP layer");
    assert!(error.to_string().contains("no MTP layer"), "{error}");
    let error = plain
        .forward_tp(&tp, &gather, &mut drafting, &[1])
        .unwrap_err();
    assert!(error.to_string().contains("another model"), "{error}");
    let bf16 = DeviceBusRank {
        host: LocalRank {
            rank: 0,
            bus: Arc::new(LocalBus::new(1)),
        },
        exchange: Exchange::new(1),
        wire: Wire::Bf16,
    };
    let tp = TpRuntime::new(whole, Box::new(bf16)).expect("one rank");
    let error = weights
        .forward_tp(&tp, &gather, &mut sequence, &[1])
        .unwrap_err();
    assert!(error.to_string().contains("Wire::F32"), "{error}");
    assert_eq!(sequence.position(), 0, "a refused device leg runs nothing");
}
