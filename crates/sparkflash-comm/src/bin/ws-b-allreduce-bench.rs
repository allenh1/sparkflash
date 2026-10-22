//! WS-B four-rank all-reduce microbenchmark binary.
//!
//! This is the executable form of docs/archive/build/ws-b-collectives-20261021.md section 6 and the
//! runner is scripts/ws-b-allreduce-bench.sh. It measures the one-shot fabric path at the decode
//! sizes the 25 us p50 gate is written for (10,240 B is the hidden_5120 bf16 expert reduce), plus
//! the 2 MiB one-shot boundary, cold and hot.
//!
//! No number exists until a run completes on the four nodes: a build here has no GPU and no
//! fabric, and the loopback transport is a harness self-test labelled fabric=false in the report.
//! The bench never falls back from the requested transport and never emits a timing from a failed
//! exchange.
//!
//! Exit codes: 0 measured (gate passed or not applicable), 2 usage, 3 measurement unavailable,
//! 4 exchange failure, 5 measured but the decode p50 gate failed.

use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use sparkflash_comm::microbench::{
    self, BenchError, BenchPlan, BenchSize, BenchTransport, DEFAULT_GATE_P50_US, DEFAULT_SAMPLES,
    DEFAULT_SIZES, DEFAULT_WARMUP, DEFAULT_WORLD,
};

/// Four-rank all-reduce microbenchmark for the WS-B latency gate.
#[derive(Debug, Parser)]
#[command(
    name = "ws-b-allreduce-bench",
    version,
    about = "Measure four-rank bf16 All-Reduce latency at the WS-B decode sizes (cold + hot, p50/p90/p99); no number is available until it runs on the four nodes",
    long_about = "Measure four-rank bf16 All-Reduce latency at the WS-B decode sizes (cold + hot, p50/p90/p99).\n\nThe default sweep is 10,240 B (hidden_5120 bf16), 60 KiB, 360 KiB and the 2 MiB one-shot boundary. The p50 <= 25 us gate applies only to the decode rows of a fabric run; loopback runs are labelled fabric=false and are never gated. Exit codes: 0 measured, 2 usage, 3 unavailable, 4 exchange failure, 5 measured but gate failed."
)]
struct Args {
    /// Transport to measure.
    #[arg(long, value_enum, default_value_t = BenchTransport::Rdma)]
    transport: BenchTransport,
    /// Bytes per rank, repeatable or comma separated. Defaults to the decode sweep.
    #[arg(long = "bytes", value_name = "N", value_delimiter = ',')]
    bytes: Vec<usize>,
    /// Timed samples per payload, taken after warmup.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_SAMPLES)]
    samples: usize,
    /// Exchanges discarded before the timed samples, per payload.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_WARMUP)]
    warmup: usize,
    /// p50 gate for the decode rows, microseconds.
    #[arg(long = "gate-us", value_name = "US", default_value_t = DEFAULT_GATE_P50_US)]
    gate_us: f64,
    /// Peer control-plane address, repeatable or comma separated, in rank order.
    #[arg(long = "peer", value_name = "HOST", value_delimiter = ',')]
    peers: Vec<String>,
    /// Ranks in the collective.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_WORLD)]
    world: usize,
    /// This rank's index in the collective.
    #[arg(long, value_name = "N", default_value_t = 0)]
    rank: usize,
    /// Per-collective deadline, milliseconds.
    #[arg(long = "timeout-ms", value_name = "MS", default_value_t = 5_000)]
    timeout_ms: u64,
    /// Print exactly one JSON object on stdout.
    #[arg(long)]
    json: bool,
    /// Load libnccl.so.2, report its version and exit. This is the WS-B ticket B5 acceptance
    /// probe; it needs no GPU and reports no latency.
    #[arg(long = "check-nccl")]
    check_nccl: bool,
}

impl Args {
    fn plan(&self) -> BenchPlan {
        let sizes = if self.bytes.is_empty() {
            DEFAULT_SIZES
                .iter()
                .map(|(label, bytes)| BenchSize::new(*label, *bytes))
                .collect()
        } else {
            self.bytes
                .iter()
                .map(|bytes| BenchSize::new(format!("custom_{bytes}"), *bytes))
                .collect()
        };
        BenchPlan {
            transport: self.transport,
            sizes,
            samples: self.samples,
            warmup: self.warmup,
            gate_p50_us: self.gate_us,
            world: self.world,
            rank: self.rank,
            peers: self.peers.clone(),
            timeout: Duration::from_millis(self.timeout_ms),
        }
    }
}

/// Ticket B5 acceptance: is a 2.28+ libnccl loadable on this host?
fn check_nccl() -> ExitCode {
    match sparkflash_comm::nccl::NcclApi::load() {
        Ok(api) => {
            println!(
                "nccl: library={} version={}",
                api.library_name(),
                api.version_string().unwrap_or_else(|| "unknown".to_owned())
            );
            println!(
                "nccl: minimum required version code is {} (2.28.0)",
                sparkflash_comm::nccl::NCCL_MIN_VERSION_CODE
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("ws-b-allreduce-bench: no usable NCCL: {error}");
            ExitCode::from(3)
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.check_nccl {
        return check_nccl();
    }
    let plan = args.plan();
    match microbench::run(&plan) {
        Ok(report) => {
            if args.json {
                println!("{}", report.to_json_string());
            } else {
                println!("{}", report.to_human_string());
            }
            let gate = report.gate();
            if gate.applied && !gate.passed {
                eprintln!("ws-b-allreduce-bench: decode p50 gate FAILED");
                return ExitCode::from(5);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("ws-b-allreduce-bench: {error}");
            if let BenchError::Unavailable(_) = error {
                eprintln!(
                    "ws-b-allreduce-bench: on this four-node cluster the fabric prerequisites are tracked in docs/archive/build/ws-b-infra-tickets-20261021.md (MTU 9000, nvidia-peermem, f0/f1 rail naming); no number was produced"
                );
            }
            ExitCode::from(error.exit_code() as u8)
        }
    }
}
