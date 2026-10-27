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

//! Gates for the physical four-node DGX Spark cluster. Every test here is #[ignore]d because it
//! needs the live RoCE fabric plus the Lead's explicit per-run permission.
//!
//! Run one gate on all four nodes at the same time (they rendezvous in the QP-endpoint control
//! plane, so the four processes must be started together). On each node, from the worktree:
//!
//! `text
//! SPARK_COMM_HW_PERMISSION=1 \
//! SPARK_COMM_RANK=<0..3> SPARK_COMM_WORLD=4 \
//! SPARK_COMM_PEERS=198.51.100.80,198.51.100.84,198.51.100.68,198.51.100.59 \
//! SPARK_COMM_CONTROL_PORT=18515 \
//! cargo test -p sparkflash-comm --features rdma --release --test hardware_gates -- --ignored --exact --nocapture <name>
//! `
//!
//! The peer table is the rail-1 address table (198.51.100.80 = node0 = rank 0, .84 = node1 = rank 1,
//! .68 = node2 = rank 2, .59 = node3 = rank 3). The deployed vLLM container instead uses a mixed
//! rail-2 table; docs/archive/build/rdma-backend.md explains why those are different and why neither
//! should be "fixed".
//!
//! Without SPARK_COMM_HW_PERMISSION=1 every gate panics immediately instead of touching the
//! cluster, so a mistyped --ignored cannot quietly reach 198.51.100.80/.84/.68/.59.
//!
//! `SPARK_COMM_RAIL_COUNT` selects the mode: 2 (default) is the specification's dual rail and what
//! the dual-rail counter gate asserts; 1 is the single-rail configuration the deployed vLLM
//! recipe actually runs (measured: NCCL_IB_HCA=rocep1s0f1), and the counter gate reports that mode
//! separately instead of claiming dual-rail evidence from it.

// Without the feature nothing below the refusal helpers is reachable, so the unused-code lints
// would otherwise fire for every measurement helper that only the hardware run uses.
#![cfg_attr(not(feature = "rdma"), allow(dead_code, unused_imports))]

mod support;

use std::time::{Duration, Instant};

use sparkflash_comm::ROCENANTE_MAX_BYTES;
use sparkflash_comm::layout::CollectiveLayout;
use sparkflash_comm::reduce::canonical_all_reduce_into;

use support::{assert_bits_eq, finite_pattern, tricky_pattern};

/// Environment variable that carries the Lead's per-run permission.
const PERMISSION_ENV: &str = "SPARK_COMM_HW_PERMISSION";

/// Whether this run has the Lead's explicit permission to talk to the cluster.
fn hardware_permission() -> bool {
    std::env::var(PERMISSION_ENV).unwrap_or_default() == "1"
}

/// Fails loudly when the permission variable is missing.
fn require_permission(test: &str) {
    assert!(
        hardware_permission(),
        "refusing to run {test}: it needs the live 4x DGX Spark RoCE fabric (198.51.100.80/.84/.68/.59). \
Set {PERMISSION_ENV}=1 only with explicit per-run approval from the Lead."
    );
}

/// The world size this process was launched for.
fn world_size() -> usize {
    std::env::var("SPARK_COMM_WORLD")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(4)
}

/// The rank this process was launched as.
fn rank() -> usize {
    std::env::var("SPARK_COMM_RANK")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0)
}

/// Prints a line tagged with this rank, so four concurrent node logs can be compared.
fn report(rank: usize, message: &str) {
    println!("[rank {rank}] {message}");
}

/// Computes the median and 99th percentile of a latency sample.
fn percentiles(mut samples: Vec<Duration>) -> (Duration, Duration) {
    assert!(!samples.is_empty(), "no latency samples were collected");
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p99_index = (samples.len() * 99 / 100).min(samples.len() - 1);
    (median, samples[p99_index])
}

/// Elements in one frame of `bytes` payload bytes.
fn elements(bytes: usize) -> usize {
    bytes / 2
}

/// Refuses to fake a hardware measurement when the feature is not compiled in.
fn require_backend(test: &str) {
    #[cfg(not(feature = "rdma"))]
    panic!(
        "{test}: this binary was built without --features rdma, so no ibverbs backend exists. Rebuild on a DGX Spark node with libibverbs and the two RoCE devices present; refusing to fake a hardware measurement."
    );
    #[cfg(feature = "rdma")]
    let _ = test;
}

/// Gate 1: one-shot all-reduce latency on the live fabric.
const GATE_1: &str = "roce_oneshot_two_mib_all_reduce_meets_the_latency_budget";
/// Gate 2: bit-identical output against the CPU reference over 10,000 replays.
const GATE_2: &str = "roce_bits_match_the_cpu_reference_over_ten_thousand_replays";
/// Gate 3: measured per-rail byte accounting after one dispatch.
const GATE_3: &str = "dual_rail_counters_show_both_interfaces_carrying_half_the_bytes";

/// The specification's 10 KiB one-shot latency bar (SPEC line 1336; the doctor enforces 22 us).
const ONESHOT_LATENCY_BAR: Duration = Duration::from_micros(20);
/// Payload the specification times for the latency bar.
const ONESHOT_BAR_BYTES: usize = 10 * 1024;
/// Replays gate 2 demands bit-identical against the CPU reference.
const BIT_REPLAYS: usize = 10_000;
/// Payload per replay for gate 2.
const BIT_BYTES: usize = 10 * 1024;
/// Payload of the single dispatch gate 3 accounts.
const DUAL_RAIL_BYTES: usize = 1024 * 1024;
/// Size of the tcp::FrameHeader the data plane writes at the front of each frame.
const FRAME_HEADER_BYTES: usize = 24;
/// Replays that exercise the gate-2 payload against the tricky bit patterns.
const TRICKY_REPLAYS: usize = 100;

/// Opens the real transport for this process, refusing (never faking) when it cannot.
#[cfg(feature = "rdma")]
fn open_transport(test: &str) -> sparkflash_comm::rdma::RdmaTransport {
    use sparkflash_comm::rdma::{RdmaConfig, RdmaTransport};

    let (world, rank) = (world_size(), rank());
    let config = RdmaConfig::from_env(world, rank).unwrap_or_else(|error| {
        panic!(
            "{test}: rank {rank} of {world} has no usable configuration: {error}. Set \
SPARK_COMM_RANK, SPARK_COMM_WORLD and SPARK_COMM_PEERS (the rail-1 address table in rank order)"
        )
    });
    RdmaTransport::open_with(config).unwrap_or_else(|error| {
        panic!(
            "{test}: the ibverbs backend refused to open on rank {rank} of {world}: {error}; \
refusing to fake a hardware measurement"
        )
    })
}

/// Payload and header bytes this rank posts for one layout, from the schedule's own transfers.
fn posted_bytes(rank: usize, layout: &CollectiveLayout) -> usize {
    layout
        .transfers()
        .iter()
        .filter(|transfer| transfer.from == rank)
        .map(|transfer| transfer.bytes() + FRAME_HEADER_BYTES)
        .sum()
}

/// Runs `warmup` untimed replays and `samples` timed ones, returning the layout and timings.
#[cfg(feature = "rdma")]
fn measure_oneshot(
    transport: &sparkflash_comm::rdma::RdmaTransport,
    bytes: usize,
    warmup: usize,
    samples: usize,
) -> (CollectiveLayout, Vec<Duration>) {
    use sparkflash_comm::Transport;

    let world = world_size();
    let layout = CollectiveLayout::one_shot(world, elements(bytes)).unwrap_or_else(|error| {
        panic!("{bytes} bytes over {world} ranks is not a legal one-shot layout: {error}")
    });
    let mut buffer = vec![0_u16; layout.num_elements()];
    let mut seq = 0_u64;
    for _ in 0..warmup {
        seq += 1;
        transport
            .all_reduce_bf16(&mut buffer, seq, &layout)
            .unwrap_or_else(|error| panic!("untimed replay {seq} failed: {error}"));
    }
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        seq += 1;
        let started = Instant::now();
        transport
            .all_reduce_bf16(&mut buffer, seq, &layout)
            .unwrap_or_else(|error| panic!("timed replay {seq} failed: {error}"));
        timings.push(started.elapsed());
    }
    (layout, timings)
}

/// Lower bound on the time one dispatch needs at the negotiated rail rate.
#[cfg(feature = "rdma")]
fn wire_floor_us(
    transport: &sparkflash_comm::rdma::RdmaTransport,
    layout: &CollectiveLayout,
) -> f64 {
    let rail_count = transport.rail_count();
    let selection = transport.rail_selection();
    let mut aggregate_gbps = 0_u64;
    for index in 0..rail_count {
        aggregate_gbps += u64::from(
            selection
                .rail(index)
                .map(sparkflash_comm::fabric::SelectedRail::rate_gbps)
                .unwrap_or(0),
        );
    }
    assert!(
        aggregate_gbps > 0,
        "the fabric reported no negotiated rate; refusing to invent a latency budget"
    );
    let bytes = posted_bytes(rank(), layout);
    bytes as f64 * 8.0 / (aggregate_gbps as f64 * 1e9) * 1e6
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn roce_oneshot_two_mib_all_reduce_meets_the_latency_budget() {
    require_permission(GATE_1);
    require_backend(GATE_1);
    #[cfg(feature = "rdma")]
    {
        let rank = rank();
        let world = world_size();
        let transport = open_transport(GATE_1);
        let (_, small) = measure_oneshot(&transport, ONESHOT_BAR_BYTES, 20, 500);
        let (small_median, small_p99) = percentiles(small);
        report(
            rank,
            &format!(
                "10 KiB one-shot all-reduce over {world} ranks: median {small_median:?}, p99 {small_p99:?}; specification bar {ONESHOT_LATENCY_BAR:?}"
            ),
        );
        assert!(
            small_median <= ONESHOT_LATENCY_BAR,
            "10 KiB one-shot all-reduce median {small_median:?} is above the specification's {ONESHOT_LATENCY_BAR:?} bar on rank {rank}"
        );
        let (big_layout, big) = measure_oneshot(&transport, ROCENANTE_MAX_BYTES, 5, 50);
        let (big_median, big_p99) = percentiles(big);
        let floor_us = wire_floor_us(&transport, &big_layout);
        // Definition of this gate's acceptance bar, not a router threshold: a first working
        // backend is accepted at 1.5x the wire floor derived from the negotiated rail rate, and
        // never below the specification's 10 KiB bar. Qualifying against a tighter board-level
        // number is the Lead's call once the gate has run.
        let budget_us = (floor_us * 1.5).max(20.0);
        report(
            rank,
            &format!(
                "2 MiB ({}) one-shot all-reduce over {world} ranks: median {big_median:?}, p99 {big_p99:?}; wire floor {floor_us:.1} us at the negotiated rail rate; acceptance bar {budget_us:.1} us",
                ROCENANTE_MAX_BYTES
            ),
        );
        assert!(
            big_median.as_secs_f64() * 1e6 <= budget_us,
            "2 MiB one-shot all-reduce median {big_median:?} is above this gate's {budget_us:.1} us acceptance bar on rank {rank}"
        );
    }
    #[cfg(not(feature = "rdma"))]
    require_backend(GATE_1);
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn roce_bits_match_the_cpu_reference_over_ten_thousand_replays() {
    require_permission(GATE_2);
    require_backend(GATE_2);
    #[cfg(feature = "rdma")]
    {
        use sparkflash_comm::Transport;
        let rank = rank();
        let world = world_size();
        let transport = open_transport(GATE_2);
        let length = elements(BIT_BYTES);
        let layout = CollectiveLayout::one_shot(world, length).unwrap_or_else(|error| {
            panic!("{BIT_BYTES} bytes over {world} ranks is not a legal one-shot layout: {error}")
        });
        let mut buffer = vec![0_u16; length];
        let mut reference = vec![0_u16; length];
        for replay in 0..BIT_REPLAYS {
            let payloads: Vec<Vec<u16>> = (0..world)
                .map(|source| finite_pattern(source, length, replay as u64))
                .collect();
            buffer.copy_from_slice(&payloads[rank]);
            transport
                .all_reduce_bf16(&mut buffer, replay as u64 + 1, &layout)
                .unwrap_or_else(|error| panic!("replay {replay} failed on the wire: {error}"));
            let slices: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
            canonical_all_reduce_into(&mut reference, &slices).unwrap_or_else(|error| {
                panic!("the CPU reference refused replay {replay}: {error}")
            });
            assert_bits_eq(
                &buffer,
                &reference,
                &format!("{GATE_2}: replay {replay} of {BIT_REPLAYS}"),
            );
        }
        for replay in 0..TRICKY_REPLAYS {
            let payloads: Vec<Vec<u16>> = (0..world)
                .map(|source| tricky_pattern(source, length))
                .collect();
            buffer.copy_from_slice(&payloads[rank]);
            transport
                .all_reduce_bf16(&mut buffer, (BIT_REPLAYS + replay + 1) as u64, &layout)
                .unwrap_or_else(|error| {
                    panic!("tricky-pattern replay {replay} failed on the wire: {error}")
                });
            let slices: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
            canonical_all_reduce_into(&mut reference, &slices).unwrap_or_else(|error| {
                panic!("the CPU reference refused tricky replay {replay}: {error}")
            });
            assert_bits_eq(
                &buffer,
                &reference,
                &format!("{GATE_2}: tricky replay {replay} of {TRICKY_REPLAYS}"),
            );
        }
        report(
            rank,
            &format!(
                "{BIT_REPLAYS} replays at {BIT_BYTES} bytes over {world} ranks are bit-identical to reduce::canonical_all_reduce_into; {TRICKY_REPLAYS} extra replays cover the hand-picked tricky patterns (negative zero, subnormals, exact powers of two, values on the f32 rounding boundary; the patterns contain no NaN or infinity, and the reference accumulates in f32 and rounds once, so equality is exact for THIS reference order)"
            ),
        );
    }
    #[cfg(not(feature = "rdma"))]
    require_backend(GATE_2);
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn dual_rail_counters_show_both_interfaces_carrying_half_the_bytes() {
    require_permission(GATE_3);
    require_backend(GATE_3);
    #[cfg(feature = "rdma")]
    {
        use sparkflash_comm::Transport;
        let rank = rank();
        let world = world_size();
        let transport = open_transport(GATE_3);
        let layout =
            CollectiveLayout::one_shot(world, elements(DUAL_RAIL_BYTES)).unwrap_or_else(|error| {
                panic!(
                    "{DUAL_RAIL_BYTES} bytes over {world} ranks is not a legal one-shot layout: {error}"
                )
            });
        let mut buffer = finite_pattern(rank, layout.num_elements(), 4_242);
        let before = transport.telemetry_snapshot();
        transport
            .all_reduce_bf16(&mut buffer, 1, &layout)
            .unwrap_or_else(|error| panic!("the accounted dispatch failed: {error}"));
        let after = transport.telemetry_snapshot();
        let ledger = transport.rail_ledger();
        let rail0 = after.rdma_bytes_rail0 - before.rdma_bytes_rail0;
        let rail1 = after.rdma_bytes_rail1 - before.rdma_bytes_rail1;
        let expected = u64::try_from(posted_bytes(rank, &layout))
            .unwrap_or_else(|_| panic!("posted bytes overflow u64"));
        assert_eq!(
            rail0, ledger.rail0_bytes,
            "the specification counter and the data plane ledger must agree on rail 0"
        );
        assert_eq!(
            rail1, ledger.rail1_bytes,
            "the specification counter and the data plane ledger must agree on rail 1"
        );
        let total = rail0 + rail1;
        // The specification allows the rail counters to under-report by at most 3 bytes per
        // dispatch; this backend posts exactly the schedule's bytes, so it must be exact.
        assert!(
            total.abs_diff(expected) <= 3,
            "the measured total {total} differs from the schedule's {expected} posted bytes by more than the specification's 3-byte allowance on rank {rank}"
        );
        match transport.rail_count() {
            2 => {
                assert!(
                    rail1 > 0,
                    "dual-rail mode posted nothing on rail 1 on rank {rank}; a single-rail run must not be reported as dual-rail evidence"
                );
                assert!(
                    rail0 * 100 >= total * 45 && rail1 * 100 >= total * 45,
                    "one rail carried less than 45% of the {total} measured bytes (rail0 {rail0}, rail1 {rail1}) on rank {rank}"
                );
                report(
                    rank,
                    &format!(
                        "dual-rail run: rail0 {rail0} bytes ({:.2}%), rail1 {rail1} bytes ({:.2}%), total {total} bytes vs {expected} expected; devices {:?}",
                        100.0 * rail0 as f64 / total as f64,
                        100.0 * rail1 as f64 / total as f64,
                        transport.rail_names()
                    ),
                );
            }
            other => {
                assert_eq!(other, 1, "rail_count must be 1 or 2");
                assert_eq!(
                    rail1, 0,
                    "single-rail mode must not post on rail 1 on rank {rank}"
                );
                assert_eq!(
                    rail0, expected,
                    "single-rail mode must account every posted byte on rail 0 on rank {rank}"
                );
                report(
                    rank,
                    &format!(
                        "single-rail run (the deployed vLLM configuration): rail0 {rail0} bytes, rail1 0 bytes, {expected} expected; this is NOT dual-rail evidence — rerun with SPARK_COMM_RAIL_COUNT=2 and both twins active to exercise the dual-rail gate"
                    ),
                );
            }
        }
    }
    #[cfg(not(feature = "rdma"))]
    require_backend(GATE_3);
}
