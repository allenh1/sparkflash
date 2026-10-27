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

//! The measurement pipeline behind the rocenante-probe binary.
//!
//! The doctor in sparkflash-cli runs rocenante-probe --bytes SIZE --peer IP --json and reads one
//! JSON field, oneshot_allreduce_us, to decide whether the one-shot All-Reduce latency gate
//! passed. This module owns every part of that contract: argument validation, the timed one-shot
//! exchange, the statistic, and the JSON document the doctor parses.
//!
//! Three rules shape the code.
//!
//! 1. A probe never publishes a number it did not measure. Every exchange must succeed; the first
//!    failure aborts the run with the transport's error and nothing is written to stdout. There is
//!    no fallback transport, no retry, no synthetic or estimated timing, and a warmup failure is a
//!    failure: a probe that invents a latency is worse than a probe that fails.
//! 2. The fabric measurement cannot be stumbled into accidentally. The default transport is the
//!    RDMA path, the in-process loopback cluster is reachable only through an explicit
//!    --transport loopback, and its report is labelled fabric=false in the JSON, so no reader can
//!    mistake a pipeline number for a fabric number. When the fabric path cannot open, the probe
//!    exits 3 and stdout stays empty.
//!
//! The fabric path cannot open in two distinct ways. They are not the same failure and they name
//! different prerequisites, so they are stated separately rather than rolled into one sentence:
//!
//! - Built without the rdma feature: run_fabric returns Unavailable immediately and names the exact
//!   rebuild command, cargo build --release -p sparkflash-comm --features rdma --bin
//!   rocenante-probe. Nothing from rdma.rs is compiled into that build.
//! - Built with the rdma feature: the real libibverbs backend opens at run time. The probe builds
//!   this process's configuration with [`crate::rdma::RdmaConfig::from_env_with_peers`], whose rank
//!   table is the --peer list, and hands it to [`crate::rdma::RdmaTransport::open_with`]. That
//!   loads libibverbs, resolves and validates both rails, registers the staging memory, creates the
//!   RC QPs and then waits in the control plane until every rank in --world has joined. A host with
//!   no RDMA subsystem fails at rail resolution, a host whose rails resolve but whose library is
//!   missing fails in Verbs::load, and a host that is missing peers blocks until the deadline: each
//!   of those is wrapped below with the transport's own typed reason. The probe never invents a
//!   latency. If the backend refuses, the run exits 3 with an empty stdout, and that refusal is
//!   honest; it is not a measurement.
//! 3. Everything that does not need a device is a pure function with unit tests: argument parsing
//!    and validation (Args and ProbeRequest::from_args), the statistic
//!    (LatencyStats::from_samples) and the JSON emission (ProbeReport::to_json_string). The field
//!    name the doctor reads lives in DOCTOR_FIELD_ONESHOT_ALLREDUCE_US, and a unit test parses the
//!    emitted string the way the doctor does, with the doctor's literal, so a rename fails a test
//!    here instead of a live preflight.
//!
//! The statistic is the median (p50) of the post-warmup samples, and the report says so in both
//! the JSON (statistic and statistic_note) and the human line. Why the median: the doctor's gate
//! is a budget for a typical round trip. The mean is dragged upward by right-skewed tails that one
//! preemption, interrupt-coalescing burst or page fault produces, and the minimum is a best case
//! that a link jittering far above the budget can still hit once; the median is robust to both, is
//! what the p50 line of an RDMA latency histogram reports, and repeats from run to run. The report
//! also carries min_us, p90_us, p99_us and max_us so the spread sits next to the headline number, and
//! samples so a reader can tell a three-sample number from a thousand-sample one.

use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use serde_json::{Map, Value, json};

use crate::ROCENANTE_MAX_BYTES;
use crate::layout::CollectiveLayout;
use crate::transport::{LoopbackCluster, LoopbackConfig, LoopbackTransport, Transport};

/// The JSON field the doctor reads from stdout. The doctor matches this name literally, and the
/// unit test that reads the emitted string uses the doctor's literal, so renaming this constant
/// without renaming the reader fails that test rather than a live preflight.
pub const DOCTOR_FIELD_ONESHOT_ALLREDUCE_US: &str = "oneshot_allreduce_us";

/// Payload size of the doctor's latency gate, in bytes (doctor::ONESHOT_AR_BYTES).
pub const DEFAULT_BYTES: usize = 10 * 1024;

/// Timed samples taken after warmup when --samples is not given.
pub const DEFAULT_SAMPLES: usize = 50;

/// Exchanges whose timings are discarded before the timed samples.
pub const DEFAULT_WARMUP: usize = 5;

/// Ranks in the collective: the spec's four-node fabric.
pub const DEFAULT_WORLD: usize = 4;

/// Deadline in milliseconds for one exchange.
pub const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Name of the statistic reported as oneshot_allreduce_us.
pub const STATISTIC: &str = "median";

/// Why that statistic, in one sentence, carried in the JSON next to the number.
pub const STATISTIC_NOTE: &str = "median (p50) of the post-warmup samples: robust to the right-skewed tails that drag the mean upward and to the single lucky best case that would let a jittering link pass on its minimum";

/// Which path the probe measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TransportChoice {
    /// The fabric path: the crate's one-shot RDMA transport. The default, and the only path whose
    /// number is the doctor's gate.
    Rdma,
    /// The in-process reference cluster: real rank threads, the real slot handshake and the real
    /// canonical reduction, but no fabric. Used only when --transport loopback asks for it.
    Loopback,
}

impl TransportChoice {
    /// Name used on the command line and in the report.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rdma => "rdma",
            Self::Loopback => "loopback",
        }
    }

    /// Whether a measurement over this transport says anything about the fabric.
    #[must_use]
    pub const fn is_fabric(self) -> bool {
        matches!(self, Self::Rdma)
    }
}

impl fmt::Display for TransportChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Command line of the probe. clap rejects unknown flags before any measurement work starts.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "rocenante-probe",
    version,
    about = "Measure the one-shot RoCEnante All-Reduce and report the doctor's oneshot_allreduce_us field"
)]
pub struct Args {
    /// Bytes exchanged per rank (a bf16 payload, so it must be even). Defaults to the doctor's
    /// 10 KiB gate size.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BYTES)]
    pub bytes: usize,
    /// Peer node, repeatable; the doctor passes one. Required by the fabric transport.
    #[arg(long = "peer", value_name = "HOST", value_delimiter = ',')]
    pub peers: Vec<String>,
    /// Print exactly one JSON object on stdout: the doctor's machine-readable contract.
    #[arg(long)]
    pub json: bool,
    /// Timed samples collected after warmup and reported in the output.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_SAMPLES)]
    pub samples: usize,
    /// Exchanges whose timings are discarded so the measurement is steady state rather than
    /// connection setup.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_WARMUP)]
    pub warmup: usize,
    /// Transport to measure: rdma (fabric, the default) or loopback (explicit local pipeline run).
    #[arg(long, value_enum, default_value_t = TransportChoice::Rdma)]
    pub transport: TransportChoice,
    /// Ranks in the collective.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_WORLD)]
    pub world: usize,
    /// This rank's index in the collective.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub rank: usize,
    /// Per-exchange deadline in milliseconds.
    #[arg(long, value_name = "MS", default_value_t = DEFAULT_TIMEOUT_MS)]
    pub timeout_ms: u64,
}

/// Why the probe could not report a measurement.
#[derive(Debug)]
pub enum ProbeError {
    /// The command line or the request is not usable.
    Usage(String),
    /// The measurement path is missing on this host or in this build.
    Unavailable(String),
    /// The exchange ran and failed; no number may be reported from a failed run.
    Exchange(String),
}

impl ProbeError {
    /// Process exit code: 2 usage, 3 measurement unavailable, 4 exchange failure.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Unavailable(_) => 3,
            Self::Exchange(_) => 4,
        }
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(text) => write!(formatter, "usage error: {text}"),
            Self::Unavailable(text) => write!(formatter, "measurement unavailable: {text}"),
            Self::Exchange(text) => write!(formatter, "measurement failed: {text}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// A validated measurement request: every field is checked before a device is touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeRequest {
    /// Bytes exchanged per rank.
    pub bytes: usize,
    /// Peer nodes as given on the command line.
    pub peers: Vec<String>,
    /// Timed samples.
    pub samples: usize,
    /// Exchanges discarded before the timed samples.
    pub warmup: usize,
    /// Transport to measure.
    pub transport: TransportChoice,
    /// Ranks in the collective.
    pub world: usize,
    /// This rank.
    pub rank: usize,
    /// Per-exchange deadline.
    pub timeout: Duration,
}

impl ProbeRequest {
    /// Validates parsed arguments. Pure: no device, no clock, no I/O.
    ///
    /// # Errors
    /// Returns ProbeError::Usage naming the offending flag and the remedy.
    pub fn from_args(args: Args) -> Result<Self, ProbeError> {
        if args.bytes < 2 || !args.bytes.is_multiple_of(2) {
            return Err(ProbeError::Usage(format!(
                "--bytes must be an even number of at least 2 because the payload is bf16, two bytes per element; got {}",
                args.bytes
            )));
        }
        if args.bytes > ROCENANTE_MAX_BYTES {
            return Err(ProbeError::Usage(format!(
                "--bytes must be at most {ROCENANTE_MAX_BYTES} because the one-shot path is capped there; got {}",
                args.bytes
            )));
        }
        if args.samples == 0 {
            return Err(ProbeError::Usage(
                "--samples must be at least 1: a probe with no timed sample has no statistic to report".to_owned(),
            ));
        }
        if args.warmup >= args.samples {
            return Err(ProbeError::Usage(format!(
                "--warmup ({}) must be smaller than --samples ({}): the statistic is taken over the post-warmup samples only",
                args.warmup, args.samples
            )));
        }
        if args.world < 2 {
            return Err(ProbeError::Usage(format!(
                "--world must be at least 2 ranks for a collective; got {}",
                args.world
            )));
        }
        if args.rank >= args.world {
            return Err(ProbeError::Usage(format!(
                "--rank {} is out of range for --world {}",
                args.rank, args.world
            )));
        }
        if args.peers.iter().any(|peer| peer.trim().is_empty()) {
            return Err(ProbeError::Usage(
                "--peer names must not be empty".to_owned(),
            ));
        }
        if args.transport.is_fabric() && args.peers.is_empty() {
            return Err(ProbeError::Usage(
                "the rdma transport needs at least one --peer, which the doctor passes; for an explicitly local pipeline measurement use --transport loopback, whose number is not the fabric gate".to_owned(),
            ));
        }
        Ok(Self {
            bytes: args.bytes,
            peers: args.peers,
            samples: args.samples,
            warmup: args.warmup,
            transport: args.transport,
            world: args.world,
            rank: args.rank,
            timeout: Duration::from_millis(args.timeout_ms),
        })
    }

    /// bf16 elements in the per-rank payload.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.bytes / 2
    }
}

/// The measured distribution of the post-warmup samples, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    /// Number of timed samples the statistics were computed from.
    pub samples: usize,
    /// Fastest sample.
    pub min_us: f64,
    /// Median (p50), the reported statistic. For an even sample count it is the mean of the two
    /// central samples, which is also what nearest-rank p50 reports for that case only by accident.
    pub p50_us: f64,
    /// 90th percentile by nearest rank.
    pub p90_us: f64,
    /// 99th percentile by nearest rank: the tail the 20 us gate would miss if only the median were
    /// recorded, and the reason the report carries the full distribution rather than one number.
    pub p99_us: f64,
    /// Slowest sample.
    pub max_us: f64,
}

impl LatencyStats {
    /// Computes min, median, p90, p99 and max over the samples. Pure: no device, no clock.
    ///
    /// # Errors
    /// Returns ProbeError::Usage for an empty sample set, which has no statistic.
    pub fn from_samples(samples: &[Duration]) -> Result<Self, ProbeError> {
        if samples.is_empty() {
            return Err(ProbeError::Usage(
                "no samples were collected, so there is no statistic to report".to_owned(),
            ));
        }
        let mut micros: Vec<f64> = samples
            .iter()
            .map(|sample| sample.as_nanos() as f64 / 1_000.0)
            .collect();
        micros.sort_unstable_by(f64::total_cmp);
        Ok(Self {
            samples: micros.len(),
            min_us: micros[0],
            p50_us: median(&micros),
            p90_us: nearest_rank(&micros, 0.90),
            p99_us: nearest_rank(&micros, 0.99),
            max_us: micros[micros.len() - 1],
        })
    }

    /// The statistic the probe reports: the median.
    #[must_use]
    pub const fn reported_us(&self) -> f64 {
        self.p50_us
    }

    /// max minus min: a constant sample set has zero spread.
    #[must_use]
    pub fn spread_us(&self) -> f64 {
        self.max_us - self.min_us
    }
}

/// Median of an ascending slice: the central element, or the mean of the two central elements.
fn median(sorted: &[f64]) -> f64 {
    let count = sorted.len();
    if count % 2 == 1 {
        sorted[count / 2]
    } else {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    }
}

/// Percentile of an ascending slice by nearest rank.
fn nearest_rank(sorted: &[f64], quantile: f64) -> f64 {
    let count = sorted.len();
    let rank = ((quantile * count as f64).ceil() as usize).clamp(1, count);
    sorted[rank - 1]
}

/// One complete one-shot exchange, timed by the implementation that runs it.
///
/// An implementation owns its transport, so the warmup exchanges and the timed samples share one
/// connection: warmup exists to push connection setup, first-touch faults and lazy staging out of
/// the measurement, not to be averaged into it.
pub trait OneShotExchange {
    /// Runs one full exchange and returns its wall-clock duration.
    ///
    /// # Errors
    /// Returns ProbeError::Exchange when the exchange did not complete.
    fn exchange(&mut self) -> Result<Duration, ProbeError>;
}

/// Runs the warmup exchanges, then the timed samples, then the statistic.
///
/// Every exchange must succeed, and the first failure is returned as the probe's error: no report
/// is built from a partial run, so there is no way for a failed measurement to be published as a
/// number. There is no retry, no fallback transport and no synthetic timing. The statistic is the
/// median of the timed samples; the report carries the sample count so a reader can tell a
/// three-sample number from a thousand-sample one.
///
/// # Errors
/// Returns whatever the exchange returned, and ProbeError::Usage when no sample was collected.
pub fn measure(
    exchange: &mut impl OneShotExchange,
    request: &ProbeRequest,
) -> Result<ProbeReport, ProbeError> {
    for _ in 0..request.warmup {
        exchange.exchange()?;
    }
    let mut samples = Vec::with_capacity(request.samples);
    for _ in 0..request.samples {
        samples.push(exchange.exchange()?);
    }
    let stats = LatencyStats::from_samples(&samples)?;
    Ok(ProbeReport::new(request, stats))
}

/// The measurement, in the shape the doctor parses.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    /// The doctor's field: the median post-warmup one-shot All-Reduce latency in microseconds.
    pub oneshot_allreduce_us: f64,
    /// Name of the statistic in oneshot_allreduce_us.
    pub statistic: &'static str,
    /// Number of timed samples the statistic was computed from.
    pub samples: usize,
    /// Warmup exchanges whose timings were discarded.
    pub warmup_discarded: usize,
    /// Payload the collective moved, per rank.
    pub bytes: usize,
    /// Transport the number was measured on.
    pub transport: &'static str,
    /// Whether that transport says anything about the fabric. False means the number came from the
    /// in-process loopback cluster and must not be compared against the fabric gate.
    pub fabric: bool,
    /// Peers the run was given.
    pub peers: Vec<String>,
    /// Fastest timed sample, in microseconds.
    pub min_us: f64,
    /// 90th percentile of the timed samples, in microseconds.
    pub p90_us: f64,
    /// 99th percentile of the timed samples, in microseconds.
    pub p99_us: f64,
    /// Slowest timed sample, in microseconds.
    pub max_us: f64,
}

impl ProbeReport {
    /// Builds the report from a validated request and its statistics.
    #[must_use]
    pub fn new(request: &ProbeRequest, stats: LatencyStats) -> Self {
        Self {
            oneshot_allreduce_us: stats.reported_us(),
            statistic: STATISTIC,
            samples: stats.samples,
            warmup_discarded: request.warmup,
            bytes: request.bytes,
            transport: request.transport.as_str(),
            fabric: request.transport.is_fabric(),
            peers: request.peers.clone(),
            min_us: stats.min_us,
            p90_us: stats.p90_us,
            p99_us: stats.p99_us,
            max_us: stats.max_us,
        }
    }

    /// The JSON document the doctor parses. Pure: the same report always emits the same string.
    ///
    /// The doctor reads exactly one field, DOCTOR_FIELD_ONESHOT_ALLREDUCE_US, so that name comes
    /// from the constant here and from the doctor's literal in the tests that read the emitted
    /// string the way the doctor does.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert(
            DOCTOR_FIELD_ONESHOT_ALLREDUCE_US.to_owned(),
            json!(self.oneshot_allreduce_us),
        );
        object.insert("statistic".to_owned(), json!(self.statistic));
        object.insert("statistic_note".to_owned(), json!(STATISTIC_NOTE));
        object.insert("samples".to_owned(), json!(self.samples));
        object.insert("warmup_discarded".to_owned(), json!(self.warmup_discarded));
        object.insert("bytes".to_owned(), json!(self.bytes));
        object.insert("transport".to_owned(), json!(self.transport));
        object.insert("fabric".to_owned(), json!(self.fabric));
        object.insert("peers".to_owned(), json!(self.peers));
        object.insert("min_us".to_owned(), json!(self.min_us));
        object.insert("p90_us".to_owned(), json!(self.p90_us));
        object.insert("p99_us".to_owned(), json!(self.p99_us));
        object.insert("max_us".to_owned(), json!(self.max_us));
        Value::Object(object)
    }

    /// One compact line of JSON, which is what --json writes to stdout.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    /// The human-readable line: the statistic and the sample count are always named.
    #[must_use]
    pub fn to_human_string(&self) -> String {
        let mut text = format!(
            "one-shot All-Reduce of {} bytes over {}: {} {:.2} us across {} samples ({} warmup discarded), min {:.2} us, p90 {:.2} us, p99 {:.2} us, max {:.2} us",
            self.bytes,
            self.transport,
            self.statistic,
            self.oneshot_allreduce_us,
            self.samples,
            self.warmup_discarded,
            self.min_us,
            self.p90_us,
            self.p99_us,
            self.max_us
        );
        if !self.fabric {
            text.push_str(&format!(
                " [{} is the in-process pipeline, not the fabric: the JSON says fabric=false, so this number is not the doctor's latency gate]",
                self.transport
            ));
        }
        if !self.peers.is_empty() {
            text.push_str(&format!(" peers {}", self.peers.join(",")));
        }
        text
    }
}

/// Runs a validated request and returns the measurement.
///
/// The fabric transport is the default. When it cannot be opened, its error is returned unchanged:
/// the probe never falls back to the loopback cluster, which is reachable only through an explicit
/// --transport loopback.
///
/// # Errors
/// Returns ProbeError::Unavailable when the transport cannot be opened on this host or in this
/// build, ProbeError::Exchange when an exchange failed, and ProbeError::Usage for a request that
/// does not describe a legal collective.
pub fn run(request: &ProbeRequest) -> Result<ProbeReport, ProbeError> {
    match request.transport {
        TransportChoice::Rdma => run_fabric(request),
        TransportChoice::Loopback => {
            let mut exchange = LoopbackExchange::start(request)?;
            measure(&mut exchange, request)
        }
    }
}

#[cfg(feature = "rdma")]
fn run_fabric(request: &ProbeRequest) -> Result<ProbeReport, ProbeError> {
    let mut exchange = RdmaExchange::open(request)?;
    measure(&mut exchange, request)
}

#[cfg(not(feature = "rdma"))]
fn run_fabric(_request: &ProbeRequest) -> Result<ProbeReport, ProbeError> {
    Err(ProbeError::Unavailable(
        "sparkflash-comm was built without the rdma feature, so there is no fabric path to measure; rebuild the probe with: cargo build --release -p sparkflash-comm --features rdma --bin rocenante-probe. The loopback transport is not a substitute for this gate: use --transport loopback only for an explicitly local pipeline measurement".to_owned(),
    ))
}

#[cfg(feature = "rdma")]
struct RdmaExchange {
    transport: crate::rdma::RdmaTransport,
    buffer: Vec<u16>,
    layout: CollectiveLayout,
    seq: u64,
}

#[cfg(feature = "rdma")]
impl RdmaExchange {
    fn open(request: &ProbeRequest) -> Result<Self, ProbeError> {
        let layout =
            CollectiveLayout::one_shot(request.world, request.elements()).map_err(|error| {
                ProbeError::Usage(format!(
                    "world {} with {} bf16 elements is not a legal one-shot layout: {error}",
                    request.world,
                    request.elements()
                ))
            })?;
        let transport = crate::rdma::RdmaConfig::from_env_with_peers(
            request.world,
            request.rank,
            request.peers.clone(),
        )
        .and_then(crate::rdma::RdmaTransport::open_with)
        .map_err(|error| {
            ProbeError::Unavailable(format!(
                "the fabric path refused to open on rank {} of {} with peers {:?}: {error}. The ibverbs backend that the rdma feature loads at run time reported this reason, and a host with both rails ACTIVE and libibverbs present opens them, so this refusal names the prerequisite it stopped on. The probe will not report a number it could not measure",
                request.rank, request.world, request.peers
            ))
        })?;
        Ok(Self {
            transport,
            buffer: vec![0_u16; request.elements()],
            layout,
            seq: 0,
        })
    }
}

#[cfg(feature = "rdma")]
impl OneShotExchange for RdmaExchange {
    fn exchange(&mut self) -> Result<Duration, ProbeError> {
        self.seq += 1;
        let started = Instant::now();
        self.transport
            .all_reduce_bf16(&mut self.buffer, self.seq, &self.layout)
            .map_err(|error| {
                ProbeError::Exchange(format!("the one-shot exchange failed: {error}"))
            })?;
        Ok(started.elapsed())
    }
}

/// The in-process reference measurement: one real cluster, one real exchange per iteration.
///
/// The coordinator rank is this thread and every other rank is a worker thread. The coordinator
/// publishes a sequence number to the workers, runs its own exchange, then collects one report per
/// worker, so no rank starts iteration n+1 before every rank finished iteration n. The measured
/// interval is the coordinator's own all_reduce_bf16 call, which returns only after every peer has
/// posted: that is the end-to-end collective latency of the in-process pipeline. Thread creation
/// is outside the measurement because the workers live for the whole run.
struct LoopbackExchange {
    transport: LoopbackTransport,
    buffer: Vec<u16>,
    layout: CollectiveLayout,
    gates: Vec<Sender<u64>>,
    done: Receiver<(usize, Option<String>)>,
    workers: Vec<JoinHandle<()>>,
    timeout: Duration,
    seq: u64,
}

impl LoopbackExchange {
    fn start(request: &ProbeRequest) -> Result<Self, ProbeError> {
        let cluster = Arc::new(
            LoopbackCluster::new(LoopbackConfig {
                world_size: request.world,
                max_elements: request.elements(),
                timeout: request.timeout,
                send_delay: Duration::ZERO,
            })
            .map_err(|error| {
                ProbeError::Unavailable(format!("the in-process cluster refused to start: {error}"))
            })?,
        );
        Self::with_cluster(cluster, request)
    }

    fn with_cluster(
        cluster: Arc<LoopbackCluster>,
        request: &ProbeRequest,
    ) -> Result<Self, ProbeError> {
        let elements = request.elements();
        let layout = CollectiveLayout::one_shot(request.world, elements).map_err(|error| {
            ProbeError::Usage(format!(
                "world {} with {elements} bf16 elements is not a legal one-shot layout: {error}",
                request.world
            ))
        })?;
        let transport = cluster.handle(request.rank).map_err(|error| {
            ProbeError::Unavailable(format!(
                "rank {} has no transport in the cluster: {error}",
                request.rank
            ))
        })?;
        let (done_sender, done) = channel();
        let mut gates = Vec::with_capacity(request.world - 1);
        let mut workers = Vec::with_capacity(request.world - 1);
        for rank in 0..request.world {
            if rank == request.rank {
                continue;
            }
            let (gate_sender, gate_receiver) = channel::<u64>();
            gates.push(gate_sender);
            let cluster = Arc::clone(&cluster);
            let done_sender = done_sender.clone();
            let world = request.world;
            workers.push(thread::spawn(move || {
                let layout = match CollectiveLayout::one_shot(world, elements) {
                    Ok(layout) => layout,
                    Err(error) => {
                        let _ = done_sender.send((
                            rank,
                            Some(format!(
                                "world {world} with {elements} elements is not a legal layout: {error}"
                            )),
                        ));
                        return;
                    }
                };
                let mut buffer = vec![0_u16; elements];
                while let Ok(seq) = gate_receiver.recv() {
                    let outcome = match cluster.handle(rank) {
                        Ok(handle) => handle
                            .all_reduce_bf16(&mut buffer, seq, &layout)
                            .err()
                            .map(|error| error.to_string()),
                        Err(error) => Some(error.to_string()),
                    };
                    let failed = outcome.is_some();
                    let _ = done_sender.send((rank, outcome));
                    if failed {
                        break;
                    }
                }
            }));
        }
        drop(done_sender);
        Ok(Self {
            transport,
            buffer: vec![0_u16; elements],
            layout,
            gates,
            done,
            workers,
            timeout: request.timeout,
            seq: 0,
        })
    }
}

impl OneShotExchange for LoopbackExchange {
    fn exchange(&mut self) -> Result<Duration, ProbeError> {
        self.seq += 1;
        for gate in &self.gates {
            let _ = gate.send(self.seq);
        }
        let started = Instant::now();
        let own = self
            .transport
            .all_reduce_bf16(&mut self.buffer, self.seq, &self.layout);
        let elapsed = started.elapsed();
        own.map_err(|error| {
            ProbeError::Exchange(format!(
                "rank {} failed its one-shot exchange: {error}",
                self.transport.rank()
            ))
        })?;
        for _ in 0..self.gates.len() {
            match self.done.recv_timeout(self.timeout) {
                Ok((_, None)) => {}
                Ok((rank, Some(text))) => {
                    return Err(ProbeError::Exchange(format!(
                        "rank {rank} failed its one-shot exchange: {text}"
                    )));
                }
                Err(error) => {
                    return Err(ProbeError::Exchange(format!(
                        "a peer rank did not report inside {:?}: {error}",
                        self.timeout
                    )));
                }
            }
        }
        Ok(elapsed)
    }
}

impl Drop for LoopbackExchange {
    fn drop(&mut self) {
        // Dropping the gate senders ends every worker's recv loop; a worker still inside a
        // collective returns when its own deadline expires.
        self.gates.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(extra: &[&str]) -> Result<Args, ProbeError> {
        let mut argv = vec!["rocenante-probe".to_owned()];
        argv.extend(extra.iter().map(|value| (*value).to_owned()));
        Args::try_parse_from(argv).map_err(|error| ProbeError::Usage(error.to_string()))
    }

    fn request_of(extra: &[&str]) -> Result<ProbeRequest, ProbeError> {
        ProbeRequest::from_args(args_of(extra)?)
    }

    /// Mirrors the doctor's reader in sparkflash-cli: parse stdout, take the one field.
    fn doctor_reads_allreduce_us(raw: &str) -> Option<f64> {
        let value: Value = serde_json::from_str(raw).ok()?;
        value.get("oneshot_allreduce_us").and_then(Value::as_f64)
    }

    struct ScriptedExchange {
        micros: Vec<u64>,
        calls: usize,
        fail_on_call: Option<usize>,
    }

    impl ScriptedExchange {
        fn new(micros: &[u64]) -> Self {
            Self {
                micros: micros.to_vec(),
                calls: 0,
                fail_on_call: None,
            }
        }

        fn failing_on(micros: &[u64], call: usize) -> Self {
            Self {
                micros: micros.to_vec(),
                calls: 0,
                fail_on_call: Some(call),
            }
        }
    }

    impl OneShotExchange for ScriptedExchange {
        fn exchange(&mut self) -> Result<Duration, ProbeError> {
            self.calls += 1;
            if Some(self.calls) == self.fail_on_call {
                return Err(ProbeError::Exchange(format!(
                    "scripted transport failure on call {}",
                    self.calls
                )));
            }
            let index = (self.calls - 1).min(self.micros.len().saturating_sub(1));
            Ok(Duration::from_micros(self.micros[index]))
        }
    }

    #[test]
    fn the_doctors_invocation_parses_into_a_fabric_request() {
        let request = request_of(&["--bytes", "10240", "--peer", "198.51.100.85", "--json"])
            .unwrap_or_else(|error| panic!("the doctor's invocation must parse: {error}"));
        assert_eq!(request.bytes, DEFAULT_BYTES);
        assert_eq!(request.bytes, 10 * 1024);
        assert_eq!(request.peers, vec!["198.51.100.85".to_owned()]);
        assert_eq!(request.transport, TransportChoice::Rdma);
        assert!(request.transport.is_fabric());
        assert_eq!(request.samples, DEFAULT_SAMPLES);
        assert_eq!(request.warmup, DEFAULT_WARMUP);
        assert_eq!(request.world, DEFAULT_WORLD);
        assert_eq!(request.rank, 0);
        assert_eq!(request.elements(), 5 * 1024);
    }

    #[test]
    fn peers_are_repeatable_and_comma_delimited() {
        let repeated = request_of(&["--peer", "10.0.0.1", "--peer", "10.0.0.2"])
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(repeated.peers.len(), 2);
        let comma =
            request_of(&["--peer", "10.0.0.1,10.0.0.2"]).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(comma.peers, repeated.peers);
    }

    #[test]
    fn bad_requests_are_usage_errors() {
        let cases: &[(&[&str], &str)] = &[
            (&["--bytes", "3"], "even number of at least 2"),
            (&["--bytes", "0"], "even number of at least 2"),
            (&["--bytes", "2097154"], "at most 2097152"),
            (&["--samples", "0"], "at least 1"),
            (
                &["--samples", "5", "--warmup", "5"],
                "smaller than --samples",
            ),
            (&["--bytes", "1024"], "needs at least one --peer"),
            (&["--peer", "10.0.0.1", "--world", "1"], "at least 2 ranks"),
            (&["--peer", "10.0.0.1", "--rank", "4"], "out of range"),
            (&["--peer", " "], "must not be empty"),
        ];
        for (extra, needle) in cases {
            match request_of(extra) {
                Ok(request) => panic!("{extra:?} was accepted as {request:?}"),
                Err(ProbeError::Usage(text)) => {
                    assert!(text.contains(needle), "{extra:?} produced {text}");
                }
                Err(other) => panic!("{extra:?}: expected a usage error, got {other}"),
            }
        }
        assert!(
            matches!(args_of(&["--nope"]), Err(ProbeError::Usage(_))),
            "an unknown flag must be rejected before any work"
        );
    }

    #[test]
    fn the_statistic_is_the_documented_median() {
        let even = LatencyStats::from_samples(&[
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(30),
            Duration::from_micros(40),
        ])
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(even.samples, 4);
        assert_eq!(even.min_us, 10.0);
        assert_eq!(even.p50_us, 25.0);
        assert_eq!(even.p90_us, 40.0);
        assert_eq!(even.p99_us, 40.0);
        assert_eq!(even.max_us, 40.0);
        assert_eq!(even.reported_us(), even.p50_us);

        let odd = LatencyStats::from_samples(&[
            Duration::from_micros(5),
            Duration::from_micros(1),
            Duration::from_micros(3),
        ])
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(odd.samples, 3);
        assert_eq!(odd.min_us, 1.0);
        assert_eq!(odd.p50_us, 3.0);
        assert_eq!(odd.p90_us, 5.0);
        assert_eq!(odd.p99_us, 5.0);

        let single = LatencyStats::from_samples(&[Duration::from_micros(7)])
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(single.p50_us, 7.0);
        assert_eq!(single.p99_us, 7.0);
        assert_eq!(single.spread_us(), 0.0);

        assert!(
            LatencyStats::from_samples(&[]).is_err(),
            "an empty sample set has no statistic"
        );
    }

    #[test]
    fn the_statistic_separates_a_constant_sample_set_from_a_varying_one() {
        let constant_samples = vec![Duration::from_micros(7); 9];
        let constant =
            LatencyStats::from_samples(&constant_samples).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(constant.spread_us(), 0.0);
        assert_eq!(constant.min_us, constant.p50_us);
        assert_eq!(constant.p50_us, constant.p90_us);
        assert_eq!(constant.p90_us, constant.max_us);
        assert_eq!(constant.p99_us, constant.max_us);

        let varying_samples = [1, 2, 3, 4, 50, 60, 70].map(Duration::from_micros);
        let varying =
            LatencyStats::from_samples(&varying_samples).unwrap_or_else(|error| panic!("{error}"));
        assert!(varying.spread_us() > 0.0);
        assert_eq!(varying.p50_us, 4.0);
        assert_ne!(varying.p50_us, varying.min_us);
        assert_ne!(varying.p50_us, varying.max_us);
        assert_eq!(varying.p99_us, varying.max_us);
        let mean = (1.0 + 2.0 + 3.0 + 4.0 + 50.0 + 60.0 + 70.0) / 7.0;
        assert!(
            (mean - varying.p50_us).abs() > 1.0,
            "the median must not be the mean: mean {mean}, median {}",
            varying.p50_us
        );
    }

    #[test]
    fn the_json_carries_the_doctors_field_the_statistic_and_the_sample_count() {
        let request = request_of(&["--peer", "10.0.0.1", "--samples", "3", "--warmup", "2"])
            .unwrap_or_else(|error| panic!("{error}"));
        let mut exchange = ScriptedExchange::new(&[999, 999, 10, 20, 30]);
        let report = measure(&mut exchange, &request).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(report.samples, 3);
        assert_eq!(report.warmup_discarded, 2);
        assert_eq!(report.oneshot_allreduce_us, 20.0);

        let raw = report.to_json_string();
        assert!(raw.contains("\"oneshot_allreduce_us\":20.0"), "{raw}");
        assert_eq!(doctor_reads_allreduce_us(&raw), Some(20.0));
        let value: Value = serde_json::from_str(&raw).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(value["samples"], json!(3));
        assert_eq!(value["warmup_discarded"], json!(2));
        assert_eq!(value["statistic"], json!("median"));
        assert_eq!(value["fabric"], json!(true));
        assert_eq!(value["transport"], json!("rdma"));
        assert_eq!(value["bytes"], json!(DEFAULT_BYTES));
        assert_eq!(value["min_us"], json!(10.0));
        assert_eq!(value["p99_us"], json!(30.0));
        assert!(
            value["statistic_note"]
                .as_str()
                .unwrap_or_default()
                .contains("median"),
            "{raw}"
        );
    }

    #[test]
    fn a_renamed_field_breaks_the_doctors_reader() {
        let request = request_of(&["--peer", "10.0.0.1", "--samples", "1", "--warmup", "0"])
            .unwrap_or_else(|error| panic!("{error}"));
        let mut exchange = ScriptedExchange::new(&[12]);
        let report = measure(&mut exchange, &request).unwrap_or_else(|error| panic!("{error}"));
        let raw = report.to_json_string();
        assert_eq!(doctor_reads_allreduce_us(&raw), Some(12.0));
        let renamed = raw.replace(DOCTOR_FIELD_ONESHOT_ALLREDUCE_US, "oneshot_all_reduce_us");
        assert_eq!(
            doctor_reads_allreduce_us(&renamed),
            None,
            "a near-miss field name must not satisfy the doctor, so the contract test cannot be vacuous"
        );
    }

    #[test]
    fn the_sample_count_is_visible_so_three_cannot_look_like_a_thousand() {
        let small_request = request_of(&["--peer", "10.0.0.1", "--samples", "3", "--warmup", "1"])
            .unwrap_or_else(|error| panic!("{error}"));
        let mut small_exchange = ScriptedExchange::new(&[10, 10, 10]);
        let small =
            measure(&mut small_exchange, &small_request).unwrap_or_else(|error| panic!("{error}"));

        let large_request =
            request_of(&["--peer", "10.0.0.1", "--samples", "1000", "--warmup", "5"])
                .unwrap_or_else(|error| panic!("{error}"));
        let mut large_exchange = ScriptedExchange::new(&[10]);
        let large =
            measure(&mut large_exchange, &large_request).unwrap_or_else(|error| panic!("{error}"));

        assert_eq!(small.samples, 3);
        assert_eq!(large.samples, 1000);
        assert!(small.to_json_string().contains("\"samples\":3"));
        assert!(large.to_json_string().contains("\"samples\":1000"));
        assert!(small.to_human_string().contains("3 samples"));
        assert!(large.to_human_string().contains("1000 samples"));
    }

    #[test]
    fn a_failing_exchange_reports_an_error_and_no_number() {
        let request = request_of(&["--peer", "10.0.0.1", "--samples", "3", "--warmup", "2"])
            .unwrap_or_else(|error| panic!("{error}"));
        let mut late = ScriptedExchange::failing_on(&[10, 10, 10, 10, 10], 4);
        match measure(&mut late, &request) {
            Ok(report) => panic!("a failed exchange produced a report: {report:?}"),
            Err(ProbeError::Exchange(text)) => {
                assert!(
                    text.contains("scripted transport failure on call 4"),
                    "{text}"
                );
            }
            Err(other) => panic!("expected an exchange error, got {other}"),
        }
        assert_eq!(late.calls, 4, "the run must stop at the first failure");

        let mut during_warmup = ScriptedExchange::failing_on(&[10], 1);
        assert!(
            matches!(
                measure(&mut during_warmup, &request),
                Err(ProbeError::Exchange(_))
            ),
            "a warmup failure must abort rather than be retried or skipped"
        );
        assert_eq!(
            during_warmup.calls, 1,
            "a failing warmup must not be retried"
        );
    }

    #[test]
    fn the_fabric_path_is_the_default_and_loopback_must_be_asked_for() {
        let fabric = request_of(&["--peer", "10.0.0.1"]).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(fabric.transport, TransportChoice::Rdma);
        assert!(fabric.transport.is_fabric());
        let loopback =
            request_of(&["--transport", "loopback"]).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(loopback.transport, TransportChoice::Loopback);
        assert!(!loopback.transport.is_fabric());
        assert!(
            loopback.peers.is_empty(),
            "loopback needs no peer, and none may be required of it"
        );
    }

    #[cfg(not(feature = "rdma"))]
    #[test]
    fn the_fabric_path_refuses_without_the_feature_instead_of_falling_back() {
        let request =
            request_of(&["--peer", "10.0.0.1", "--json"]).unwrap_or_else(|error| panic!("{error}"));
        match run(&request) {
            Ok(report) => {
                panic!("the fabric path reported {report:?} on a build without the rdma feature")
            }
            Err(ProbeError::Unavailable(text)) => {
                assert!(text.contains("--features rdma"), "{text}");
                assert!(text.contains("--transport loopback"), "{text}");
            }
            Err(other) => panic!("expected an unavailable error, got {other}"),
        }
    }

    #[test]
    fn loopback_measures_a_real_collective_and_labels_it() {
        let request = request_of(&[
            "--transport",
            "loopback",
            "--world",
            "4",
            "--bytes",
            "2048",
            "--samples",
            "3",
            "--warmup",
            "1",
            "--timeout-ms",
            "2000",
        ])
        .unwrap_or_else(|error| panic!("{error}"));
        let report = run(&request).unwrap_or_else(|error| panic!("loopback run: {error}"));
        assert_eq!(report.samples, 3);
        assert_eq!(report.warmup_discarded, 1);
        assert_eq!(report.transport, "loopback");
        assert!(!report.fabric);
        assert!(report.oneshot_allreduce_us > 0.0, "{report:?}");
        assert!(report.min_us <= report.oneshot_allreduce_us);
        assert!(report.oneshot_allreduce_us <= report.max_us);
        assert!(report.to_json_string().contains("\"fabric\":false"));
        assert!(report.to_human_string().contains("loopback"));
    }

    #[test]
    fn a_killed_rank_fails_the_exchange_instead_of_producing_a_number() {
        let request = request_of(&[
            "--transport",
            "loopback",
            "--world",
            "3",
            "--bytes",
            "512",
            "--samples",
            "2",
            "--warmup",
            "0",
            "--timeout-ms",
            "300",
        ])
        .unwrap_or_else(|error| panic!("{error}"));
        let cluster = LoopbackCluster::new(LoopbackConfig {
            world_size: 3,
            max_elements: 256,
            timeout: Duration::from_millis(300),
            send_delay: Duration::ZERO,
        })
        .unwrap_or_else(|error| panic!("cluster: {error}"));
        cluster
            .kill(1)
            .unwrap_or_else(|error| panic!("kill: {error}"));
        let mut exchange = LoopbackExchange::with_cluster(Arc::new(cluster), &request)
            .unwrap_or_else(|error| panic!("{error}"));
        match exchange.exchange() {
            Ok(elapsed) => panic!("a killed rank still produced a duration: {elapsed:?}"),
            Err(ProbeError::Exchange(text)) => assert!(!text.is_empty()),
            Err(other) => panic!("expected an exchange error, got {other}"),
        }
    }

    /// The real fabric measurement, behind the hardware gate. It needs the two ACTIVE 200 Gb/s
    /// rails, a process on every rank started at the same time (the QP-endpoint control plane is
    /// the barrier), and the Lead's per-run permission. The assertions in the body are its positive
    /// control, so it cannot pass on a refusal: the report must be labelled fabric, carry the
    /// requested 50 samples, and hold a finite positive latency. From each node, all four at once:
    ///
    ///     SPARK_COMM_RANK=<0..3> SPARK_COMM_WORLD=4 \
    ///     SPARK_COMM_PEERS=198.51.100.80,198.51.100.84,198.51.100.68,198.51.100.59 \
    ///     cargo test -p sparkflash-comm --features rdma --lib --release \
    ///         probe::tests::the_fabric_gate_measures_a_real_one_shot_all_reduce -- --ignored --nocapture
    #[cfg(feature = "rdma")]
    #[ignore = "needs the live dual-rail RoCE fabric, one process per rank started together, and hardware permission"]
    #[test]
    fn the_fabric_gate_measures_a_real_one_shot_all_reduce() {
        let peers = std::env::var("SPARK_COMM_PEERS").unwrap_or_else(|_| {
            panic!(
                "set SPARK_COMM_PEERS to the rail-1 address table in rank order before running a hardware gate"
            )
        });
        let world = std::env::var("SPARK_COMM_WORLD").unwrap_or_else(|_| "4".to_owned());
        let rank = std::env::var("SPARK_COMM_RANK").unwrap_or_else(|_| "0".to_owned());
        let peer_list: Vec<String> = peers
            .split(',')
            .map(|peer| peer.trim().to_owned())
            .filter(|peer| !peer.is_empty())
            .collect();
        let mut args: Vec<&str> = vec![
            "--world",
            world.as_str(),
            "--rank",
            rank.as_str(),
            "--bytes",
            "10240",
            "--samples",
            "50",
            "--warmup",
            "5",
        ];
        for peer in &peer_list {
            args.push("--peer");
            args.push(peer);
        }
        let request = request_of(&args).unwrap_or_else(|error| panic!("{error}"));
        let report = run(&request).unwrap_or_else(|error| panic!("fabric probe: {error}"));
        println!("{}", report.to_human_string());
        assert!(
            report.fabric,
            "the hardware gate must measure the fabric path"
        );
        assert_eq!(report.samples, 50);
        assert!(
            report.oneshot_allreduce_us.is_finite() && report.oneshot_allreduce_us > 0.0,
            "{report:?}"
        );
    }
}
