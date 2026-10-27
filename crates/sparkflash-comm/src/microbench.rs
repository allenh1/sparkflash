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

//! Four-rank all-reduce microbenchmark: plan, statistics and report for the WS-B latency gate.
//!
//! This module measures nothing by itself; [`run`] drives a real transport and reports only
//! durations it actually observed. There is no synthetic timing, no estimate and no fallback:
//! **no number exists until a run completes on the four nodes**, and a loopback run is labelled
//! `fabric=false` so it can never be mistaken for the fabric gate.
//!
//! ## Why a separate harness from `rocenante-probe`
//!
//! `rocenante-probe` is the doctor's instrument: one size, one transport, one JSON field
//! (`oneshot_allreduce_us`). WS-B needs a sweep for the dispatch decision and the 25 us p50 gate:
//! every decode size from the spec's 10 KiB-360 KiB class plus the 2 MiB one-shot boundary,
//! cold and hot samples separated, p50/p90 and a gate verdict that applies only to fabric runs.
//! Changing the doctor's contract to carry a table would put a second reader on a field the live
//! preflight depends on, so the bench is its own binary
//! (`crates/sparkflash-comm/src/bin/ws-b-allreduce-bench.rs`) over this module.
//!
//! ## Cold and hot
//!
//! `cold_us` is the first exchange after the transport opened: connection setup, first-touch
//! page faults and lazy QP/MR state are inside it by construction. It is one sample and is
//! labelled that way. `p50_us` / `p90_us` / `p99_us` / `max_us` come from the post-warmup
//! samples exactly like the doctor's probe, and the report carries `samples` so a three-sample
//! run cannot look like a thousand-sample one.
//!
//! ## The gate
//!
//! `p50 <= 25 us` on the decode rows (payload at or below
//! [`DECODE_ONESHOT_MAX_BYTES`]) of a **fabric** run. The 25 us figure is the WS-B brief's gate,
//! not a measurement: the only measured numbers on this fabric are the 2026-10-20 bring-up's
//! 10 KiB medians (36.0-37.1 us dual rail, 29.5 us single rail), which miss both 25 us and the
//! spec's 20 us bar. A loopback run has `gate_applied=false` and `gate_passed=null`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use clap::ValueEnum;
use serde_json::{Map, Value, json};

use crate::dispatch::{
    CollectiveBackend, DECODE_HIDDEN_BYTES, DECODE_ONESHOT_MAX_BYTES, DispatchPolicy,
};
use crate::layout::CollectiveLayout;
use crate::probe::LatencyStats;
use crate::transport::{LoopbackCluster, LoopbackConfig, LoopbackTransport, Transport};

/// JSON schema string for the bench report. A new schema version means a reader change.
pub const SCHEMA: &str = "sparkflash-wsb-allreduce-bench/1";

/// The WS-B p50 gate, in microseconds, for decode-sized fabric all-reduces.
///
/// Not a measured value: the brief sets it. The measured 2026-10-20 bring-up medians were
/// 36.0-37.1 us dual rail and 29.5 us single rail at 10 KiB, so this gate is deliberately not
/// claimed as met.
pub const DEFAULT_GATE_P50_US: f64 = 25.0;

/// Timed samples taken after warmup when `--samples` is not given.
pub const DEFAULT_SAMPLES: usize = 200;

/// Exchanges discarded before the timed samples, per size.
pub const DEFAULT_WARMUP: usize = 20;

/// Ranks in the collective: the spec's four-node fabric.
pub const DEFAULT_WORLD: usize = 4;

/// The default sweep: the decode class plus the one-shot boundary.
///
/// * `decode_hidden_5120` is the expert-reduce payload the gate is written for.
/// * `decode_60kib` and `decode_360kib` are the rest of the spec's decode all-reduce range.
/// * `oneshot_2mib_boundary` is the inclusive top of the one-shot path; it is reported and is
///   deliberately excluded from the 25 us gate (it is not a decode collective).
pub const DEFAULT_SIZES: [(&str, usize); 4] = [
    ("decode_hidden_5120", DECODE_HIDDEN_BYTES),
    ("decode_60kib", 60 * 1024),
    ("decode_360kib", DECODE_ONESHOT_MAX_BYTES),
    ("oneshot_2mib_boundary", crate::ROCENANTE_MAX_BYTES),
];

/// One payload in the sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchSize {
    /// Human label carried into the report.
    pub label: String,
    /// Bytes per rank.
    pub bytes: usize,
}

impl BenchSize {
    /// Builds a size row.
    #[must_use]
    pub fn new(label: impl Into<String>, bytes: usize) -> Self {
        Self {
            label: label.into(),
            bytes,
        }
    }

    /// bf16 elements per rank.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.bytes / 2
    }
}

/// Which transport the bench drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchTransport {
    /// The four-node fabric: the one-shot RDMA transport (`--features rdma`).
    Rdma,
    /// The in-process reference cluster: real rank threads and the real canonical reduction, but
    /// no fabric. `fabric=false` in the report.
    Loopback,
    /// The patched NCCL ring: needs a host `libnccl.so.2` and device buffers; refused until the
    /// device owner can supply both.
    Nccl,
}

impl BenchTransport {
    /// Name used in the JSON and on the command line.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rdma => "rdma",
            Self::Loopback => "loopback",
            Self::Nccl => "nccl",
        }
    }

    /// Whether a run over this transport says anything about the fabric.
    #[must_use]
    pub const fn is_fabric(self) -> bool {
        matches!(self, Self::Rdma | Self::Nccl)
    }
}

impl std::fmt::Display for BenchTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why the bench could not report a measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    /// The plan or command line is not usable.
    Usage(String),
    /// The measurement path is missing on this host or in this build.
    Unavailable(String),
    /// The transport ran and failed; no number may be published from a failed run.
    Measurement(String),
}

impl BenchError {
    /// Process exit code: 2 usage, 3 measurement unavailable, 4 exchange failure.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Unavailable(_) => 3,
            Self::Measurement(_) => 4,
        }
    }
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(text) => write!(formatter, "usage error: {text}"),
            Self::Unavailable(text) => write!(formatter, "measurement unavailable: {text}"),
            Self::Measurement(text) => write!(formatter, "measurement failed: {text}"),
        }
    }
}

impl std::error::Error for BenchError {}

/// A validated benchmark plan.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchPlan {
    /// Transport to measure.
    pub transport: BenchTransport,
    /// Payload sweep.
    pub sizes: Vec<BenchSize>,
    /// Timed samples per size.
    pub samples: usize,
    /// Exchanges discarded before the timed samples, per size.
    pub warmup: usize,
    /// p50 gate, microseconds.
    pub gate_p50_us: f64,
    /// Ranks in the collective.
    pub world: usize,
    /// This rank.
    pub rank: usize,
    /// Control-plane addresses in rank order (fabric transports).
    pub peers: Vec<String>,
    /// Per-collective deadline.
    pub timeout: Duration,
}

impl BenchPlan {
    /// Validates every field before a transport is touched. Pure: no device, no clock.
    ///
    /// # Errors
    /// Returns [`BenchError::Usage`] naming the offending field.
    pub fn validate(&self) -> Result<(), BenchError> {
        if self.world < 2 {
            return Err(BenchError::Usage(format!(
                "--world must be at least 2 ranks for a collective; got {}",
                self.world
            )));
        }
        if self.rank >= self.world {
            return Err(BenchError::Usage(format!(
                "--rank {} is out of range for --world {}",
                self.rank, self.world
            )));
        }
        if self.sizes.is_empty() {
            return Err(BenchError::Usage(
                "the sweep needs at least one size; pass --bytes or use the default decode sizes"
                    .to_owned(),
            ));
        }
        let policy = DispatchPolicy::default();
        for size in &self.sizes {
            if size.bytes < 2 || !size.bytes.is_multiple_of(2) {
                return Err(BenchError::Usage(format!(
                    "size {} ({} B) must be an even number of at least 2 because the payload is bf16",
                    size.label, size.bytes
                )));
            }
            if !matches!(self.transport, BenchTransport::Nccl)
                && size.bytes > policy.one_shot_max_bytes()
            {
                return Err(BenchError::Usage(format!(
                    "size {} ({} B) is above the {} B one-shot cap, so it cannot run on the {} transport; use the nccl transport (device buffers required) or keep the sweep at or below the cap",
                    size.label,
                    size.bytes,
                    policy.one_shot_max_bytes(),
                    self.transport
                )));
            }
        }
        if self.samples == 0 {
            return Err(BenchError::Usage(
                "--samples must be at least 1: a bench with no timed sample has no statistic"
                    .to_owned(),
            ));
        }
        if self.warmup >= self.samples {
            return Err(BenchError::Usage(format!(
                "--warmup ({}) must be smaller than --samples ({}): the statistic is taken over the post-warmup samples only",
                self.warmup, self.samples
            )));
        }
        if !self.gate_p50_us.is_finite() || self.gate_p50_us <= 0.0 {
            return Err(BenchError::Usage(format!(
                "--gate-us must be a positive, finite number; got {}",
                self.gate_p50_us
            )));
        }
        if self.transport.is_fabric() && self.peers.len() != self.world {
            return Err(BenchError::Usage(format!(
                "the {} transport needs one --peer per rank ({} for world {}), got {}",
                self.transport,
                self.world,
                self.world,
                self.peers.len()
            )));
        }
        Ok(())
    }
}

/// The measured result of one payload.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRow {
    /// Sweep label.
    pub label: String,
    /// Bytes per rank.
    pub bytes: usize,
    /// bf16 elements per rank.
    pub elements: usize,
    /// Backend the dispatch policy selected.
    pub backend: CollectiveBackend,
    /// First exchange after the transport opened, microseconds (n=1).
    pub cold_us: f64,
    /// Post-warmup distribution, microseconds.
    pub hot: LatencyStats,
}

impl BenchRow {
    /// Builds a row from one cold sample and the post-warmup samples.
    ///
    /// # Errors
    /// Returns [`BenchError::Measurement`] for an empty hot sample set, because a row with no
    /// timed sample has no statistic.
    pub fn new(
        size: &BenchSize,
        backend: CollectiveBackend,
        cold: Duration,
        samples: &[Duration],
    ) -> Result<Self, BenchError> {
        let hot = LatencyStats::from_samples(samples)
            .map_err(|error| BenchError::Measurement(error.to_string()))?;
        Ok(Self {
            label: size.label.clone(),
            bytes: size.bytes,
            elements: size.elements(),
            backend,
            cold_us: micros(cold),
            hot,
        })
    }

    /// Whether this row is in the 25 us gate's class (decode payloads).
    #[must_use]
    pub fn is_gated(&self) -> bool {
        DispatchPolicy::is_decode_payload(self.bytes)
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_nanos() as f64 / 1_000.0
}

/// The gate's verdict over one report.
#[derive(Debug, Clone, PartialEq)]
pub struct GateVerdict {
    /// The gate value applied to p50.
    pub gate_p50_us: f64,
    /// Whether a fabric run had any decode row to gate.
    pub applied: bool,
    /// Whether every gated row's p50 met the gate.
    pub passed: bool,
    /// Labels of the rows the gate covered.
    pub gated_rows: Vec<String>,
    /// The slowest gated row and its p50, when there is one.
    pub worst: Option<(String, f64)>,
}

impl GateVerdict {
    /// One-line human verdict.
    #[must_use]
    pub fn to_human_string(&self) -> String {
        if !self.applied {
            return format!(
                "gate: p50 <= {:.2} us not applied (not a fabric run, or no decode row)",
                self.gate_p50_us
            );
        }
        let outcome = if self.passed { "PASS" } else { "FAIL" };
        let worst = self
            .worst
            .as_ref()
            .map(|(label, value)| format!("worst {label} p50 {value:.2} us"))
            .unwrap_or_else(|| "no gated row".to_owned());
        format!(
            "gate: p50 <= {:.2} us on [{}] -> {outcome} ({worst})",
            self.gate_p50_us,
            self.gated_rows.join(", ")
        )
    }
}

/// The complete benchmark report for one rank.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchReport {
    /// Transport the numbers came from.
    pub transport: BenchTransport,
    /// Whether that transport says anything about the fabric.
    pub fabric: bool,
    /// World size.
    pub world: usize,
    /// This rank.
    pub rank: usize,
    /// Peers the run was given.
    pub peers: Vec<String>,
    /// The gate value.
    pub gate_p50_us: f64,
    /// Timed samples per size.
    pub samples: usize,
    /// Warmup exchanges per size.
    pub warmup: usize,
    /// One row per payload, in plan order.
    pub rows: Vec<BenchRow>,
}

impl BenchReport {
    /// Builds a report from a completed run. Pure.
    #[must_use]
    pub fn new(plan: &BenchPlan, rows: Vec<BenchRow>) -> Self {
        Self {
            transport: plan.transport,
            fabric: plan.transport.is_fabric(),
            world: plan.world,
            rank: plan.rank,
            peers: plan.peers.clone(),
            gate_p50_us: plan.gate_p50_us,
            samples: plan.samples,
            warmup: plan.warmup,
            rows,
        }
    }

    /// The gate verdict: only a fabric run can pass, and only decode rows are gated.
    #[must_use]
    pub fn gate(&self) -> GateVerdict {
        let mut gated_rows: Vec<String> = Vec::new();
        let mut worst: Option<(String, f64)> = None;
        for row in &self.rows {
            if !row.is_gated() {
                continue;
            }
            gated_rows.push(row.label.clone());
            let value = row.hot.p50_us;
            if worst.as_ref().is_none_or(|(_, current)| value > *current) {
                worst = Some((row.label.clone(), value));
            }
        }
        let applied = self.fabric && !gated_rows.is_empty();
        let passed = applied
            && worst
                .as_ref()
                .is_some_and(|(_, value)| *value <= self.gate_p50_us);
        GateVerdict {
            gate_p50_us: self.gate_p50_us,
            applied,
            passed,
            gated_rows,
            worst,
        }
    }

    /// The report as a JSON value. Pure: the same report always emits the same document.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let gate = self.gate();
        let rows: Vec<Value> = self
            .rows
            .iter()
            .map(|row| {
                let mut object = Map::new();
                object.insert("label".to_owned(), json!(row.label));
                object.insert("bytes".to_owned(), json!(row.bytes));
                object.insert("elements".to_owned(), json!(row.elements));
                object.insert("backend".to_owned(), json!(row.backend.as_str()));
                object.insert("gated".to_owned(), json!(row.is_gated()));
                object.insert("cold_us".to_owned(), json!(row.cold_us));
                object.insert("cold_samples".to_owned(), json!(1));
                object.insert("min_us".to_owned(), json!(row.hot.min_us));
                object.insert("p50_us".to_owned(), json!(row.hot.p50_us));
                object.insert("p90_us".to_owned(), json!(row.hot.p90_us));
                object.insert("p99_us".to_owned(), json!(row.hot.p99_us));
                object.insert("max_us".to_owned(), json!(row.hot.max_us));
                object.insert("samples".to_owned(), json!(row.hot.samples));
                Value::Object(object)
            })
            .collect();
        let mut object = Map::new();
        object.insert("schema".to_owned(), json!(SCHEMA));
        object.insert("measured".to_owned(), json!(true));
        object.insert("transport".to_owned(), json!(self.transport.as_str()));
        object.insert("fabric".to_owned(), json!(self.fabric));
        object.insert("world".to_owned(), json!(self.world));
        object.insert("rank".to_owned(), json!(self.rank));
        object.insert("peers".to_owned(), json!(self.peers));
        object.insert("gate_p50_us".to_owned(), json!(self.gate_p50_us));
        object.insert("gate_applied".to_owned(), json!(gate.applied));
        object.insert(
            "gate_passed".to_owned(),
            if gate.applied {
                json!(gate.passed)
            } else {
                Value::Null
            },
        );
        object.insert("gate_rows".to_owned(), json!(gate.gated_rows));
        object.insert("samples".to_owned(), json!(self.samples));
        object.insert("warmup".to_owned(), json!(self.warmup));
        object.insert(
            "cold_note".to_owned(),
            json!("cold_us is the first exchange after the transport opened (n=1); hot_* come from the post-warmup samples"),
        );
        object.insert("rows".to_owned(), Value::Array(rows));
        Value::Object(object)
    }

    /// One compact line of JSON.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    /// Human-readable report.
    #[must_use]
    pub fn to_human_string(&self) -> String {
        let gate = self.gate();
        let mut text = format!(
            "WS-B four-rank all-reduce microbench [{}] transport={} fabric={} world={} rank={} samples={} warmup={}\n",
            SCHEMA, self.transport, self.fabric, self.world, self.rank, self.samples, self.warmup
        );
        for row in &self.rows {
            text.push_str(&format!(
                "  {:>24} bytes={:>8} elements={:>8} backend={:<12} cold={:>10.2} us p50={:>9.2} us p90={:>10.2} us p99={:>10.2} us max={:>10.2} us n={}\n",
                row.label,
                row.bytes,
                row.elements,
                row.backend.as_str(),
                row.cold_us,
                row.hot.p50_us,
                row.hot.p90_us,
                row.hot.p99_us,
                row.hot.max_us,
                row.hot.samples
            ));
        }
        text.push_str(&gate.to_human_string());
        if !self.fabric {
            text.push_str(&format!(
                "\n[{} is the in-process pipeline, not the fabric: fabric=false, gate_applied=false. This number is not the WS-B latency gate]",
                self.transport
            ));
        }
        text
    }
}

/// Runs a validated plan and returns the measurement.
///
/// # Errors
/// Returns [`BenchError::Unavailable`] when the transport cannot be opened (including the NCCL
/// device-buffer and host-library prerequisites), [`BenchError::Measurement`] when an exchange
/// fails, and [`BenchError::Usage`] for a plan that does not describe a legal collective.
pub fn run(plan: &BenchPlan) -> Result<BenchReport, BenchError> {
    plan.validate()?;
    let rows = match plan.transport {
        BenchTransport::Loopback => run_loopback(plan)?,
        BenchTransport::Rdma => run_rdma(plan)?,
        BenchTransport::Nccl => return Err(BenchError::Unavailable(nccl_unavailable_text())),
    };
    Ok(BenchReport::new(plan, rows))
}

fn nccl_unavailable_text() -> String {
    format!(
        "the NCCL bench path needs a host libnccl.so.2 (WS-B infra ticket B5: NCCL 2.28.3 is measured only inside the {image} image) and bf16 device buffers from the device owner. This host has no GPU, so the bench refuses instead of timing a host copy that is not the NCCL path. Use --transport rdma for the one-shot fabric path or --transport loopback for the harness self-test",
        image = crate::nccl::NCCL_OBSERVED_IMAGE
    )
}

/// One timed exchange: fill `buffer` for `seq` under `layout` on the open transport.
type Exchange<'a> = dyn FnMut(&mut [u16], u64, &CollectiveLayout) -> Result<(), BenchError> + 'a;

/// Measures every size on one already-open exchange, cold first then hot.
fn measure(plan: &BenchPlan, exchange: &mut Exchange<'_>) -> Result<Vec<BenchRow>, BenchError> {
    let policy = DispatchPolicy::default();
    let mut rows = Vec::with_capacity(plan.sizes.len());
    let mut seq: u64 = 0;
    for size in &plan.sizes {
        let elements = size.elements();
        let layout = CollectiveLayout::one_shot(plan.world, elements).map_err(|error| {
            BenchError::Usage(format!(
                "size {} ({} B) is not a legal one-shot layout for world {}: {error}",
                size.label, size.bytes, plan.world
            ))
        })?;
        let backend = policy.route(size.bytes);
        let mut buffer = vec![0u16; elements];

        seq += 1;
        let started = Instant::now();
        exchange(&mut buffer, seq, &layout)?;
        let cold = started.elapsed();

        for _ in 0..plan.warmup {
            seq += 1;
            exchange(&mut buffer, seq, &layout)?;
        }
        let mut samples = Vec::with_capacity(plan.samples);
        for _ in 0..plan.samples {
            seq += 1;
            let started = Instant::now();
            exchange(&mut buffer, seq, &layout)?;
            samples.push(started.elapsed());
        }
        rows.push(BenchRow::new(size, backend, cold, &samples)?);
    }
    Ok(rows)
}

#[cfg(feature = "rdma")]
fn run_rdma(plan: &BenchPlan) -> Result<Vec<BenchRow>, BenchError> {
    let config = crate::rdma::RdmaConfig::from_env_with_peers(
        plan.world,
        plan.rank,
        plan.peers.clone(),
    )
    .map_err(|error| {
        BenchError::Unavailable(format!(
            "the fabric configuration was refused before any exchange: {error}. Set SPARK_COMM_* or pass --peer in rank order"
        ))
    })?;
    let transport = crate::rdma::RdmaTransport::open_with(config).map_err(|error| {
        BenchError::Unavailable(format!(
            "the fabric path refused to open on rank {} of {} with peers {:?}: {error}. The bench reports no number it could not measure",
            plan.rank, plan.world, plan.peers
        ))
    })?;
    measure(plan, &mut |buffer, seq, layout| {
        transport
            .all_reduce_bf16(buffer, seq, layout)
            .map_err(|error| {
                BenchError::Measurement(format!("the one-shot exchange failed: {error}"))
            })
    })
}

#[cfg(not(feature = "rdma"))]
fn run_rdma(_plan: &BenchPlan) -> Result<Vec<BenchRow>, BenchError> {
    Err(BenchError::Unavailable(
        "sparkflash-comm was built without the rdma feature, so there is no fabric path to measure; rebuild the bench with: cargo build --release -p sparkflash-comm --features rdma --bin ws-b-allreduce-bench. The loopback transport is not a substitute for the fabric gate"
            .to_owned(),
    ))
}

/// One message from the coordinator to a loopback worker.
struct Gate {
    seq: u64,
    elements: usize,
}

/// The in-process coordinator side of the benchmark, mirroring probe's loopback cluster.
struct LoopbackHarness {
    transport: LoopbackTransport,
    gates: Vec<Sender<Gate>>,
    done: Receiver<(usize, Option<String>)>,
    workers: Vec<JoinHandle<()>>,
    timeout: Duration,
}

impl LoopbackHarness {
    fn start(plan: &BenchPlan, max_elements: usize) -> Result<Self, BenchError> {
        let cluster = Arc::new(
            LoopbackCluster::new(LoopbackConfig {
                world_size: plan.world,
                max_elements,
                timeout: plan.timeout,
                send_delay: Duration::ZERO,
            })
            .map_err(|error| {
                BenchError::Unavailable(format!("the in-process cluster refused to start: {error}"))
            })?,
        );
        let transport = cluster.handle(plan.rank).map_err(|error| {
            BenchError::Unavailable(format!(
                "rank {} has no transport in the cluster: {error}",
                plan.rank
            ))
        })?;
        let (done_sender, done) = channel();
        let mut gates = Vec::with_capacity(plan.world - 1);
        let mut workers = Vec::with_capacity(plan.world - 1);
        for peer in 0..plan.world {
            if peer == plan.rank {
                continue;
            }
            let (gate_sender, gate_receiver) = channel::<Gate>();
            gates.push(gate_sender);
            let cluster = Arc::clone(&cluster);
            let done_sender = done_sender.clone();
            let world = plan.world;
            workers.push(thread::spawn(move || {
                let handle = match cluster.handle(peer) {
                    Ok(handle) => handle,
                    Err(error) => {
                        let _ = done_sender
                            .send((peer, Some(format!("rank {peer} has no transport: {error}"))));
                        return;
                    }
                };
                let mut layouts: HashMap<usize, CollectiveLayout> = HashMap::new();
                let mut buffer: Vec<u16> = Vec::new();
                while let Ok(gate) = gate_receiver.recv() {
                    if buffer.len() != gate.elements {
                        buffer = vec![0u16; gate.elements];
                    }
                    let layout = match layouts.get(&gate.elements) {
                        Some(layout) => layout.clone(),
                        None => match CollectiveLayout::one_shot(world, gate.elements) {
                            Ok(layout) => {
                                layouts.insert(gate.elements, layout.clone());
                                layout
                            }
                            Err(error) => {
                                let _ = done_sender.send((
                                    peer,
                                    Some(format!(
                                        "world {world} with {} elements is not a legal layout: {error}",
                                        gate.elements
                                    )),
                                ));
                                return;
                            }
                        },
                    };
                    let outcome = handle
                        .all_reduce_bf16(&mut buffer, gate.seq, &layout)
                        .err()
                        .map(|error| error.to_string());
                    let failed = outcome.is_some();
                    let _ = done_sender.send((peer, outcome));
                    if failed {
                        return;
                    }
                }
            }));
        }
        drop(done_sender);
        Ok(Self {
            transport,
            gates,
            done,
            workers,
            timeout: plan.timeout,
        })
    }

    fn exchange(
        &mut self,
        buffer: &mut [u16],
        seq: u64,
        layout: &CollectiveLayout,
    ) -> Result<(), BenchError> {
        for gate in &self.gates {
            let _ = gate.send(Gate {
                seq,
                elements: layout.num_elements(),
            });
        }
        let own = self
            .transport
            .all_reduce_bf16(buffer, seq, layout)
            .map_err(|error| {
                BenchError::Measurement(format!(
                    "rank {} failed its one-shot exchange: {error}",
                    self.transport.rank()
                ))
            });
        for _ in 0..self.gates.len() {
            match self.done.recv_timeout(self.timeout) {
                Ok((_, None)) => {}
                Ok((peer, Some(text))) => {
                    return Err(BenchError::Measurement(format!(
                        "rank {peer} failed its one-shot exchange: {text}"
                    )));
                }
                Err(error) => {
                    return Err(BenchError::Measurement(format!(
                        "a peer rank did not report inside {:?}: {error}",
                        self.timeout
                    )));
                }
            }
        }
        own
    }
}

impl Drop for LoopbackHarness {
    fn drop(&mut self) {
        // Dropping the gate senders ends every worker's recv loop; a worker still inside a
        // collective returns when its own deadline expires.
        self.gates.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn run_loopback(plan: &BenchPlan) -> Result<Vec<BenchRow>, BenchError> {
    let max_elements = plan
        .sizes
        .iter()
        .map(BenchSize::elements)
        .max()
        .unwrap_or(0);
    let mut harness = LoopbackHarness::start(plan, max_elements)?;
    measure(plan, &mut |buffer, seq, layout| {
        harness.exchange(buffer, seq, layout)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(transport: BenchTransport, sizes: &[(&str, usize)]) -> BenchPlan {
        BenchPlan {
            transport,
            sizes: sizes
                .iter()
                .map(|(label, bytes)| BenchSize::new(*label, *bytes))
                .collect(),
            samples: 3,
            warmup: 1,
            gate_p50_us: DEFAULT_GATE_P50_US,
            world: 4,
            rank: 0,
            peers: vec![
                "198.51.100.80".to_owned(),
                "198.51.100.88".to_owned(),
                "198.51.100.59".to_owned(),
                "198.51.100.68".to_owned(),
            ],
            timeout: Duration::from_millis(2_000),
        }
    }

    fn measured_row(label: &str, bytes: usize, cold_us: u64, hot_us: &[u64]) -> BenchRow {
        let size = BenchSize::new(label, bytes);
        let samples: Vec<Duration> = hot_us.iter().copied().map(Duration::from_micros).collect();
        BenchRow::new(
            &size,
            DispatchPolicy::default().route(bytes),
            Duration::from_micros(cold_us),
            &samples,
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    fn fabric_report(rows: Vec<BenchRow>) -> BenchReport {
        let plan = plan(BenchTransport::Rdma, &[("decode_hidden_5120", 10_240)]);
        BenchReport::new(&plan, rows)
    }

    #[test]
    fn the_default_sweep_covers_the_decode_class_and_the_boundary() {
        assert_eq!(DEFAULT_SIZES[0].1, 10_240);
        assert_eq!(DEFAULT_SIZES[0].1, DECODE_HIDDEN_BYTES);
        assert!(DEFAULT_SIZES[0].1 <= DECODE_ONESHOT_MAX_BYTES);
        assert_eq!(
            DEFAULT_SIZES.last().map(|(_, bytes)| *bytes),
            Some(crate::ROCENANTE_MAX_BYTES)
        );
        for (label, bytes) in DEFAULT_SIZES {
            assert!(bytes >= 2 && bytes.is_multiple_of(2), "{label}");
        }
        let policy = DispatchPolicy::default();
        for (label, bytes) in DEFAULT_SIZES {
            assert_eq!(
                policy.route(bytes),
                CollectiveBackend::OneShotRdma,
                "{label} must be one-shot at the default threshold"
            );
        }
    }

    #[test]
    fn a_plan_is_validated_before_anything_is_opened() {
        let good = plan(BenchTransport::Loopback, &[("decode_hidden_5120", 10_240)]);
        assert!(good.validate().is_ok());

        let mut bad = good.clone();
        bad.sizes = vec![BenchSize::new("odd", 3)];
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.sizes = Vec::new();
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.rank = 4;
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.world = 1;
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.samples = 0;
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.warmup = bad.samples;
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.gate_p50_us = 0.0;
        assert!(matches!(bad.validate(), Err(BenchError::Usage(_))));

        let mut bad = good.clone();
        bad.peers = Vec::new();
        assert!(bad.validate().is_ok(), "loopback needs no peers");

        let mut fabric = plan(BenchTransport::Rdma, &[("decode_hidden_5120", 10_240)]);
        fabric.peers = Vec::new();
        assert!(matches!(fabric.validate(), Err(BenchError::Usage(_))));

        let mut too_big = plan(
            BenchTransport::Rdma,
            &[("above_cap", crate::ROCENANTE_MAX_BYTES + 2)],
        );
        assert!(
            matches!(too_big.validate(), Err(BenchError::Usage(_))),
            "a one-shot transport cannot carry a payload above its cap"
        );
        too_big.transport = BenchTransport::Nccl;
        assert!(
            too_big.validate().is_ok(),
            "the NCCL transport may size above the cap"
        );
    }

    #[test]
    fn a_row_separates_cold_from_the_post_warmup_distribution() {
        let row = measured_row("decode_hidden_5120", 10_240, 99, &[10, 20, 30]);
        assert_eq!(row.bytes, 10_240);
        assert_eq!(row.elements, 5_120);
        assert_eq!(row.backend, CollectiveBackend::OneShotRdma);
        assert_eq!(row.cold_us, 99.0);
        assert_eq!(row.hot.samples, 3);
        assert_eq!(row.hot.p50_us, 20.0);
        assert_eq!(row.hot.p90_us, 30.0);
        assert_eq!(row.hot.p99_us, 30.0);
        assert_eq!(row.hot.min_us, 10.0);
        assert_eq!(row.hot.max_us, 30.0);
        assert!(row.is_gated(), "the decode row must be gated");

        let boundary = measured_row(
            "oneshot_2mib_boundary",
            crate::ROCENANTE_MAX_BYTES,
            900,
            &[900],
        );
        assert!(!boundary.is_gated(), "the 2 MiB row is not a decode row");
        assert!(
            BenchRow::new(
                &BenchSize::new("empty", 2),
                CollectiveBackend::OneShotRdma,
                Duration::ZERO,
                &[]
            )
            .is_err()
        );
    }

    #[test]
    fn the_gate_applies_only_to_decode_rows_of_a_fabric_run() {
        let report = fabric_report(vec![
            measured_row("decode_hidden_5120", 10_240, 20, &[10, 20, 30]),
            measured_row("decode_60kib", 60 * 1024, 30, &[20, 24, 25]),
        ]);
        let gate = report.gate();
        assert!(gate.applied);
        assert!(gate.passed);
        assert_eq!(gate.gated_rows.len(), 2);
        assert_eq!(gate.worst, Some(("decode_60kib".to_owned(), 24.0)));

        let slow = fabric_report(vec![measured_row(
            "decode_hidden_5120",
            10_240,
            50,
            &[26, 30, 40],
        )]);
        let gate = slow.gate();
        assert!(gate.applied);
        assert!(!gate.passed);
        assert_eq!(gate.worst, Some(("decode_hidden_5120".to_owned(), 30.0)));

        // A slow 2 MiB row does not fail the decode gate.
        let boundary = fabric_report(vec![
            measured_row("decode_hidden_5120", 10_240, 20, &[10, 20, 30]),
            measured_row(
                "oneshot_2mib_boundary",
                crate::ROCENANTE_MAX_BYTES,
                900,
                &[900, 950, 980],
            ),
        ]);
        assert!(boundary.gate().passed);
        assert_eq!(
            boundary.gate().gated_rows,
            vec!["decode_hidden_5120".to_owned()]
        );

        // Loopback is measured but never gated.
        let in_process = BenchReport::new(
            &plan(BenchTransport::Loopback, &[("decode_hidden_5120", 10_240)]),
            vec![measured_row(
                "decode_hidden_5120",
                10_240,
                20,
                &[10, 20, 30],
            )],
        );
        let gate = in_process.gate();
        assert!(!gate.applied);
        assert!(!gate.passed);
        assert!(!in_process.fabric);
    }

    #[test]
    fn the_json_carries_the_schema_the_rows_and_the_gate() {
        let report = fabric_report(vec![measured_row(
            "decode_hidden_5120",
            10_240,
            99,
            &[10, 20, 30],
        )]);
        let value = report.to_json();
        assert_eq!(value["schema"], json!(SCHEMA));
        assert_eq!(value["transport"], json!("rdma"));
        assert_eq!(value["fabric"], json!(true));
        assert_eq!(value["measured"], json!(true));
        assert_eq!(value["gate_p50_us"], json!(DEFAULT_GATE_P50_US));
        assert_eq!(value["gate_applied"], json!(true));
        assert_eq!(value["gate_passed"], json!(true));
        assert_eq!(value["rows"][0]["backend"], json!("oneshot_rdma"));
        assert_eq!(value["rows"][0]["cold_us"], json!(99.0));
        assert_eq!(value["rows"][0]["cold_samples"], json!(1));
        assert_eq!(value["rows"][0]["p50_us"], json!(20.0));
        assert_eq!(value["rows"][0]["p90_us"], json!(30.0));
        assert_eq!(value["rows"][0]["samples"], json!(3));
        assert!(
            value["cold_note"]
                .as_str()
                .unwrap_or_default()
                .contains("n=1")
        );
        let raw = report.to_json_string();
        assert!(raw.contains(SCHEMA), "{raw}");
        assert!(raw.contains("\"fabric\":true"), "{raw}");

        let in_process = BenchReport::new(
            &plan(BenchTransport::Loopback, &[("decode_hidden_5120", 10_240)]),
            vec![measured_row(
                "decode_hidden_5120",
                10_240,
                10,
                &[10, 10, 10],
            )],
        );
        let value = in_process.to_json();
        assert_eq!(value["fabric"], json!(false));
        assert_eq!(value["gate_applied"], json!(false));
        assert!(value["gate_passed"].is_null());
        assert!(in_process.to_human_string().contains("not the fabric"));
    }

    #[test]
    fn the_loopback_harness_measures_a_real_collective_and_labels_it() {
        let mut plan = plan(
            BenchTransport::Loopback,
            &[("tiny", 2_048), ("small", 4_096)],
        );
        plan.samples = 2;
        plan.warmup = 1;
        let report = run(&plan).unwrap_or_else(|error| panic!("loopback run: {error}"));
        assert_eq!(report.rows.len(), 2);
        assert!(!report.fabric);
        for row in &report.rows {
            assert!(row.hot.p50_us > 0.0, "{row:?}");
            assert!(row.cold_us > 0.0, "{row:?}");
            assert_eq!(row.hot.samples, 2);
            assert_eq!(row.backend, CollectiveBackend::OneShotRdma);
        }
        assert!(!report.gate().applied);
        assert!(report.to_human_string().contains("loopback"));
    }

    #[test]
    fn nccl_is_refused_until_the_library_and_device_buffers_exist() {
        let mut plan = plan(BenchTransport::Nccl, &[("prefill_4mib", 4 * 1024 * 1024)]);
        plan.peers = vec!["198.51.100.80".to_owned(); 4];
        match run(&plan) {
            Err(BenchError::Unavailable(text)) => {
                assert!(text.contains("libnccl.so.2"), "{text}");
                assert!(text.contains("B5"), "{text}");
                assert!(text.contains("device buffers"), "{text}");
            }
            other => panic!("the NCCL path must refuse without a host library, got {other:?}"),
        }
        assert_eq!(BenchError::Unavailable("x".to_owned()).exit_code(), 3);
        assert_eq!(BenchError::Usage("x".to_owned()).exit_code(), 2);
        assert_eq!(BenchError::Measurement("x".to_owned()).exit_code(), 4);
    }

    #[cfg(not(feature = "rdma"))]
    #[test]
    fn the_fabric_transport_refuses_without_the_feature_instead_of_falling_back() {
        let plan = plan(BenchTransport::Rdma, &[("decode_hidden_5120", 10_240)]);
        match run(&plan) {
            Err(BenchError::Unavailable(text)) => {
                assert!(text.contains("--features rdma"), "{text}");
                assert!(text.contains("loopback"), "{text}");
            }
            other => panic!("expected an unavailable error, got {other:?}"),
        }
    }
}
