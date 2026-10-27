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

//! Isolated per-call measurement of the host collective reduce over TCP.
//!
//! This binary runs one rank of a real four-rank [TcpTransport::connect] cluster, performs the
//! served decode shape (5120 bf16 elements = the 10 KiB routed-row contribution, one-shot
//! all-to-all, [CollectiveLayout]) and prints per-call latencies plus, in async mode, the post
//! and wait halves separately.
//!
//! It exists to price one call of the reduce in isolation, without the forward loop: what the
//! probe attributes to moe_reduce is exactly one `all_reduce_bf16` of this shape, so the number
//! below is comparable with the row's per-layer block. It never touches a GPU and never changes
//! the canonical order: the result is compared against the local canonical reference on every
//! rank, so a mismatch exits non-zero.
//!
//! Local four-process run (no fabric, no nodes):
//!
//! ```text
//! cargo build -p sparkflash-comm --bin comm-reduce-bench
//! for r in 0 1 2 3; do ./target/debug/comm-reduce-bench --rank $r --base-port 19701 & done; wait
//! ```
//!
//! Four-node run: pass the address table instead of the loopback scheme, for example
//! `--rank 0 --addrs 198.51.100.80:19701,198.51.100.88:19701,198.51.100.59:19701,198.51.100.68:19701`.
//!
//! Set `SPARK_COMM_PROFILE=1` to get the phase breakdown line from the transport in addition to
//! this binary's latency line (the profile line is written to stderr, one window every 40 calls).

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
use sparkflash_comm::reduce::canonical_reduce_element;
use sparkflash_comm::tcp::{ReduceSlot, TcpConfig, TcpTransport};
use sparkflash_comm::transport::Transport;

const TIMEOUT: Duration = Duration::from_secs(20);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

struct Args {
    rank: usize,
    base_port: u16,
    addrs: Option<Vec<SocketAddr>>,
    elements: usize,
    samples: usize,
    warmup: usize,
    mode: Mode,
    stagger_us: u64,
    overlap_us: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sync,
    Async,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            rank: 0,
            base_port: 19_701,
            addrs: None,
            elements: 5_120,
            samples: 200,
            warmup: 20,
            mode: Mode::Sync,
            stagger_us: 0,
            overlap_us: 0,
        };
        let mut argv = std::env::args().skip(1);
        while let Some(flag) = argv.next() {
            let mut value = |name: &str| -> Result<String, String> {
                argv.next().ok_or_else(|| format!("{name} needs a value"))
            };
            match flag.as_str() {
                "--rank" => args.rank = value("--rank")?.parse().map_err(|e| format!("{e}"))?,
                "--base-port" => {
                    args.base_port = value("--base-port")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--addrs" => {
                    let list = value("--addrs")?;
                    let mut parsed = Vec::new();
                    for entry in list.split(',') {
                        parsed.push(
                            entry
                                .parse()
                                .map_err(|e| format!("bad address {entry}: {e}"))?,
                        );
                    }
                    args.addrs = Some(parsed);
                }
                "--elements" => {
                    args.elements = value("--elements")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--samples" => {
                    args.samples = value("--samples")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--warmup" => {
                    args.warmup = value("--warmup")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--mode" => {
                    args.mode = match value("--mode")?.as_str() {
                        "sync" => Mode::Sync,
                        "async" => Mode::Async,
                        other => return Err(format!("--mode must be sync or async, got {other}")),
                    };
                }
                "--stagger-us" => {
                    args.stagger_us = value("--stagger-us")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--overlap-us" => {
                    args.overlap_us = value("--overlap-us")?.parse().map_err(|e| format!("{e}"))?;
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: comm-reduce-bench --rank R [--addrs a,b,c,d | --base-port P] \
--elements 5120 --samples 200 --warmup 20 --mode sync|async --stagger-us 0 --overlap-us 0"
                            .to_owned(),
                    );
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }
        if args.samples == 0 {
            return Err("--samples must be positive".to_owned());
        }
        Ok(args)
    }
}

/// Deterministic per-rank contribution. Values are exact bf16 rounds of small f32s, so the
/// canonical reference below compares against the same bits the wire carries.
fn contribution(rank: usize, elements: usize) -> Vec<u16> {
    (0..elements)
        .map(|index| {
            let base = 0.001_953_125_f32;
            let pattern = (((index * 7 + rank * 13) % 19) as f32) - 9.0;
            let value = (rank as f32 + 1.0) * base * pattern;
            sparkflash_comm::f32_to_bf16_rn(value)
        })
        .collect()
}

fn expected(world: usize, elements: usize) -> Vec<u16> {
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| contribution(rank, elements))
        .collect();
    let mut out = vec![0u16; elements];
    for (slot, value) in out.iter_mut().enumerate() {
        let column: Vec<u16> = payloads.iter().map(|payload| payload[slot]).collect();
        *value = canonical_reduce_element(&column);
    }
    out
}

/// Spins without sleeping, so an overlap test measures the same kind of host work the forward
/// loop would do between issue and poll.
fn spin(micros: u64) {
    if micros == 0 {
        return;
    }
    let deadline = Instant::now() + Duration::from_micros(micros);
    let mut sink = 0u64;
    while Instant::now() < deadline {
        for value in 0..1024u64 {
            sink = sink.wrapping_add(value ^ sink);
        }
    }
    std::hint::black_box(sink);
}

fn percentile(sorted: &[Duration], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((percent / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)].as_secs_f64() * 1_000_000.0
}

fn report(name: &str, durations: &mut [Duration]) -> String {
    durations.sort_unstable();
    let sum: Duration = durations.iter().copied().sum();
    let mean = sum.as_secs_f64() * 1_000_000.0 / durations.len() as f64;
    format!(
        "{name}_p50_us={:.2} {name}_p90_us={:.2} {name}_p99_us={:.2} {name}_mean_us={:.2} {name}_min_us={:.2} {name}_max_us={:.2}",
        percentile(durations, 50.0),
        percentile(durations, 90.0),
        percentile(durations, 99.0),
        mean,
        durations
            .first()
            .map(|d| d.as_secs_f64() * 1e6)
            .unwrap_or(0.0),
        durations
            .last()
            .map(|d| d.as_secs_f64() * 1e6)
            .unwrap_or(0.0),
    )
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("comm-reduce-bench: {error}");
            std::process::exit(2);
        }
    };
    let world = args.addrs.as_ref().map_or(4, Vec::len);
    if args.rank >= world {
        eprintln!(
            "comm-reduce-bench: rank {} outside world {world}",
            args.rank
        );
        std::process::exit(2);
    }
    let config = match &args.addrs {
        Some(addrs) => TcpConfig::with_addresses(args.rank, addrs.clone())
            .map_err(|error| format!("address table: {error}"))
            .map(|config| {
                config
                    .with_timeout(TIMEOUT)
                    .with_connect_timeout(CONNECT_TIMEOUT)
            }),
        None => Ok(TcpConfig::loopback4(args.rank, args.base_port)
            .with_timeout(TIMEOUT)
            .with_connect_timeout(CONNECT_TIMEOUT)),
    };
    let config = match config {
        Ok(config) => config,
        Err(error) => {
            eprintln!("comm-reduce-bench: {error}");
            std::process::exit(2);
        }
    };
    let connect_started = Instant::now();
    let mut transport = match TcpTransport::connect(config) {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("comm-reduce-bench: rank {} connect: {error}", args.rank);
            std::process::exit(2);
        }
    };
    let connect_ms = connect_started.elapsed().as_secs_f64() * 1000.0;
    let layout = match CollectiveLayout::new(world, args.elements, FabricSchedule::OneShotAllToAll)
    {
        Ok(layout) => layout,
        Err(error) => {
            eprintln!("comm-reduce-bench: layout: {error}");
            std::process::exit(2);
        }
    };
    let mut buf = contribution(args.rank, args.elements);
    let want = expected(world, args.elements);
    let mut seq = 0u64;
    let mut check = "ok";
    let mut sync_us = Vec::with_capacity(args.samples);
    let mut post_us = Vec::with_capacity(args.samples);
    let mut wait_us = Vec::with_capacity(args.samples);

    let call = |transport: &mut TcpTransport,
                buf: &mut Vec<u16>,
                seq: u64|
     -> Result<Option<Duration>, String> {
        if args.stagger_us != 0 {
            let delay = Duration::from_micros(args.stagger_us * args.rank as u64);
            let until = Instant::now() + delay;
            while Instant::now() < until {}
        }
        match args.mode {
            Mode::Sync => {
                let started = Instant::now();
                transport
                    .all_reduce_bf16(buf, seq, &layout)
                    .map_err(|error| format!("all_reduce_bf16: {error}"))?;
                Ok(Some(started.elapsed()))
            }
            Mode::Async => {
                let post_started = Instant::now();
                let handle = transport
                    .post_reduce_bf16(ReduceSlot::new(0, 0), seq, buf)
                    .map_err(|error| format!("post_reduce_bf16: {error}"))?;
                let posted = post_started.elapsed();
                spin(args.overlap_us);
                let wait_started = Instant::now();
                transport
                    .wait_reduce(handle)
                    .map_err(|error| format!("wait_reduce: {error}"))?;
                let waited = wait_started.elapsed();
                // The total is post + independent work + wait; the work was inside the window.
                Ok(Some(
                    posted + Duration::from_micros(args.overlap_us) + waited,
                ))
            }
        }
    };

    for _ in 0..args.warmup {
        seq += 1;
        buf.copy_from_slice(&contribution(args.rank, args.elements));
        if let Err(error) = call(&mut transport, &mut buf, seq) {
            eprintln!("comm-reduce-bench: rank {} warmup: {error}", args.rank);
            std::process::exit(2);
        }
    }
    for sample in 0..args.samples {
        seq += 1;
        buf.copy_from_slice(&contribution(args.rank, args.elements));
        let started = Instant::now();
        if args.mode == Mode::Async {
            // Re-run the async sequence with separate clocks for the report, keeping exactly the
            // same post/work/wait split the closure executed during warmup.
            if args.stagger_us != 0 {
                let delay = Duration::from_micros(args.stagger_us * args.rank as u64);
                let until = Instant::now() + delay;
                while Instant::now() < until {}
            }
            let post_started = Instant::now();
            let handle = match transport.post_reduce_bf16(ReduceSlot::new(0, 0), seq, &mut buf) {
                Ok(handle) => handle,
                Err(error) => {
                    eprintln!("comm-reduce-bench: rank {} post: {error}", args.rank);
                    std::process::exit(2);
                }
            };
            post_us.push(post_started.elapsed());
            spin(args.overlap_us);
            let wait_started = Instant::now();
            if let Err(error) = transport.wait_reduce(handle) {
                eprintln!("comm-reduce-bench: rank {} wait: {error}", args.rank);
                std::process::exit(2);
            }
            wait_us.push(wait_started.elapsed());
            sync_us.push(started.elapsed());
        } else {
            match call(&mut transport, &mut buf, seq) {
                Ok(Some(elapsed)) => sync_us.push(elapsed),
                Ok(None) => {}
                Err(error) => {
                    eprintln!("comm-reduce-bench: rank {} sample: {error}", args.rank);
                    std::process::exit(2);
                }
            }
        }
        if sample < 8 && buf != want {
            check = "mismatch";
            eprintln!(
                "comm-reduce-bench: rank {} sample {sample} does not match the canonical reference",
                args.rank
            );
        }
    }
    // Shutdown barrier: every rank completes one more call before any transport drops. Without
    // it a peer that finishes first closes its socket and the still-draining demultiplexer reports
    // the peer unreachable even though every payload was already delivered.
    seq += 1;
    buf.copy_from_slice(&contribution(args.rank, args.elements));
    if let Err(error) = call(&mut transport, &mut buf, seq) {
        eprintln!("comm-reduce-bench: rank {} barrier: {error}", args.rank);
        std::process::exit(2);
    }
    std::thread::sleep(Duration::from_millis(300));
    if buf != want {
        check = "mismatch";
    }
    let mut line = format!(
        "bench rank={} mode={} elements={} world={} samples={} connect_ms={:.1} stagger_us={} overlap_us={} check={}",
        args.rank,
        if args.mode == Mode::Sync {
            "sync"
        } else {
            "async"
        },
        args.elements,
        world,
        args.samples,
        connect_ms,
        args.stagger_us,
        args.overlap_us,
        check,
    );
    if args.mode == Mode::Async {
        let mut post = post_us.clone();
        let mut wait = wait_us.clone();
        line.push(' ');
        line.push_str(&report("post", &mut post));
        line.push(' ');
        line.push_str(&report("wait", &mut wait));
        line.push(' ');
        line.push_str(&report("total", &mut sync_us));
    } else {
        line.push(' ');
        line.push_str(&report("total", &mut sync_us));
    }
    println!("{line}");
    if check != "ok" {
        std::process::exit(1);
    }
}
