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

//! The loopback host twin of the device-buffer one-shot path, exercised end to end.
//!
//! tests/device_seam.rs proves the seam's protocol contract with a hand-written four-thread
//! choreography. This file proves the production shape of the same protocol: the loopback device
//! transport is a Transport, its device form runs real four-rank exchanges through the slot ring,
//! it records a DispatchRecord for every call with zero host-staging bytes, it refuses the host
//! slice forms instead of falling back, and its arithmetic and ring shape are pinned so the window
//! run cannot renegotiate them.
//!
//! What none of this claims: latency, bandwidth or correctness across the four-node mesh. Those
//! need the nodes and are named unclaimed in every HostTwinReport.

use std::sync::{Arc, Barrier};

use sparkflash_comm::oneshot::loopback::{
    HostDeviceBuffer, LoopbackDeviceCluster, LoopbackDeviceConfig, kernel_reduce_into,
    run_host_twin, run_host_twin_prefill, twin_payload,
};
use sparkflash_comm::oneshot::{
    DeviceBuffer, ONESHOT_RAILS, ONESHOT_RANKS, ONESHOT_ROW_ELEMENTS, row_offset,
};
use sparkflash_comm::{CommError, Transport, canonical_all_reduce_into, f32_to_bf16_rn};

/// Runs sequences on a cluster with an explicit config, returning every rank's final buffer.
fn run_configured(
    elements: usize,
    sequences: usize,
    reverse_arrivals: bool,
) -> (Vec<Vec<u16>>, Vec<sparkflash_comm::oneshot::DispatchLog>) {
    let cluster = LoopbackDeviceCluster::new(LoopbackDeviceConfig {
        reverse_arrivals,
        ..LoopbackDeviceConfig::default()
    })
    .expect("cluster");
    let payloads: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
        .map(|rank| twin_payload(rank, 0, elements))
        .collect();
    let barrier = Arc::new(Barrier::new(ONESHOT_RANKS));
    let mut threads = Vec::new();
    for (rank, payload) in payloads.iter().enumerate() {
        let handle = cluster.handle(rank).expect("handle");
        let output = HostDeviceBuffer::allocate(&cluster, rank, elements).expect("buffer");
        let payload = payload.clone();
        let barrier = Arc::clone(&barrier);
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            for seq in 0..sequences as u64 {
                output.write(&payload).expect("stage");
                handle
                    .all_reduce_bf16_device(&output.device_buffer(), elements, seq)
                    .expect("device collective");
            }
            let log = handle.dispatch_log();
            (output.read(), log)
        }));
    }
    let mut outputs = Vec::new();
    let mut logs = Vec::new();
    for thread in threads {
        let (output, log) = thread.join().expect("rank thread");
        outputs.push(output);
        logs.push(log);
    }
    (outputs, logs)
}

fn canonical(payloads: &[Vec<u16>]) -> Vec<u16> {
    let mut expected = vec![0u16; payloads[0].len()];
    let rows: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
    canonical_all_reduce_into(&mut expected, &rows).expect("canonical twin");
    expected
}

#[test]
fn four_ranks_run_the_device_collective_and_match_the_canonical_twin() {
    let report = run_host_twin(5120, 16, ONESHOT_RAILS).expect("host twin");
    assert_eq!(report.mode, "oneshot");
    assert_eq!(report.steps, 16);
    assert_eq!(report.elements, 5120);
    assert_eq!(report.bytes, 10_240, "the gate's decode-size payload");
    assert!(report.bit_identical, "every rank must equal the CPU twin");
    assert!(
        report.device_path_only,
        "every record must be the device path"
    );
    assert_eq!(report.host_copy_bytes, 0);
    assert_eq!(report.dispatched, ONESHOT_RANKS * 16);
    // Each rank posts its row to three peers, so the NIC moves 4 * 16 * 3 * 10,240 bytes.
    assert_eq!(
        report.posted.total_bytes(),
        (ONESHOT_RANKS * 16 * (ONESHOT_RANKS - 1) * 10_240) as u64
    );
    assert!(!report.unclaimed.is_empty());
}

#[test]
fn arrival_order_never_changes_a_bit() {
    let elements = 4096;
    let (forward, forward_logs) = run_configured(elements, 8, false);
    let (reverse, reverse_logs) = run_configured(elements, 8, true);
    let payloads: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
        .map(|rank| twin_payload(rank, 0, elements))
        .collect();
    let expected = canonical(&payloads);
    for output in forward.iter().chain(reverse.iter()) {
        assert_eq!(*output, expected, "arrival order must not move a bit");
    }
    for log in forward_logs.iter().chain(reverse_logs.iter()) {
        assert_eq!(log.len(), 8);
        assert_eq!(log.host_copy_bytes(), 0);
        log.device_path_ok().expect("device path only");
    }
}

#[test]
fn the_twin_reduction_is_one_rounding_not_a_four_rounding_chain() {
    // rank 0 carries 1.0 and each other rank carries 2^-9 (quarter of a bf16 ulp at 1.0).
    // One rounding of the f32 sum 1 + 3 * 2^-9 lands above the bf16 midpoint and rounds up to
    // 1.0078125. Rounding after every rank instead loses each quarter ulp and stays at 1.0.
    let one = f32_to_bf16_rn(1.0);
    let quarter_ulp = f32_to_bf16_rn(2.0f32.powi(-9));
    let rows = [
        vec![one],
        vec![quarter_ulp],
        vec![quarter_ulp],
        vec![quarter_ulp],
    ];
    let row_slices: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
    let mut out = vec![0u16; 1];
    kernel_reduce_into(&mut out, &row_slices).expect("kernel twin");
    assert_eq!(out[0], f32_to_bf16_rn(1.0078125), "one rounding rounds up");

    let mut serial = rows[0][0];
    for row in &rows[1..] {
        let mut pair = [0u16, 0u16];
        let mut accumulated = vec![0u16; 1];
        canonical_all_reduce_into(&mut accumulated, &[&[serial], &row[..]])
            .expect("pair reduction");
        pair[0] = accumulated[0];
        serial = pair[0];
    }
    assert_eq!(
        serial, one,
        "the four-rounding chain loses the quarter ulps"
    );
    assert_ne!(
        out[0], serial,
        "one rounding and four roundings must differ here"
    );

    let mut canonical_out = vec![0u16; 1];
    canonical_all_reduce_into(&mut canonical_out, &row_slices).expect("canonical");
    assert_eq!(out, canonical_out, "the kernel twin is the canonical rule");
}

#[test]
fn an_all_negative_zero_payload_reduces_to_positive_zero() {
    let rows = [
        vec![0x8000u16],
        vec![0x8000u16],
        vec![0x8000u16],
        vec![0x8000u16],
    ];
    let row_slices: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
    let mut out = vec![0x8000u16; 1];
    kernel_reduce_into(&mut out, &row_slices).expect("kernel twin");
    assert_eq!(out, vec![0x0000u16], "the 0.0f seed decides the zero sign");
}

#[test]
fn the_ring_shape_is_pinned_for_single_and_chunked_runs() {
    let single = run_host_twin(5120, 5, ONESHOT_RAILS).expect("single run");
    assert_eq!(single.slot_indices, vec![0, 1, 0, 1, 0], "two-deep ring");
    assert!(single.bit_identical);

    let total = ONESHOT_ROW_ELEMENTS * 2 + 1024;
    let chunked = run_host_twin_prefill(total, ONESHOT_RAILS).expect("chunked run");
    assert_eq!(chunked.mode, "chunked");
    assert_eq!(chunked.steps, 3, "two full rows and the remainder");
    assert_eq!(
        chunked.slot_indices,
        vec![0, 1, 0],
        "chunk i uses slot i mod 2"
    );
    assert!(chunked.bit_identical);
    assert_eq!(chunked.host_copy_bytes, 0);
    assert_eq!(chunked.dispatched, ONESHOT_RANKS * 3);
}

#[test]
fn a_rank_records_the_slot_and_row_addresses_the_seam_pins() {
    let (_, logs) = run_configured(5120, 1, false);
    let record = logs[1].records()[0];
    assert_eq!(record.rank(), 1);
    assert_eq!(record.seq(), 0);
    assert_eq!(
        record.launch(),
        sparkflash_comm::oneshot::LaunchPath::DeviceKernel
    );
    assert_eq!(record.host_copy_bytes(), 0);
    record.device_path_ok().expect("device path");
    assert_eq!(
        record.local_row_address(),
        record.slot_address() + row_offset(1) as u64
    );
    let peers = record.peer_row_addresses();
    assert_eq!(peers.len(), ONESHOT_RANKS - 1);
    for (index, peer) in [0usize, 2, 3].iter().enumerate() {
        assert_eq!(
            peers[index],
            record.slot_address() + row_offset(*peer) as u64
        );
    }
}

#[test]
fn the_device_transport_refuses_the_host_slice_forms_and_foreign_buffers() {
    let cluster = LoopbackDeviceCluster::new(LoopbackDeviceConfig::default()).expect("cluster");
    let handle = cluster.handle(0).expect("handle");
    let mut buffer = vec![0u16; 5120];
    let layout = sparkflash_comm::CollectiveLayout::one_shot(ONESHOT_RANKS, 5120).expect("layout");
    assert!(matches!(
        handle.all_reduce_bf16(&mut buffer, 0, &layout),
        Err(CommError::Unsupported(_))
    ));
    let mut gathered = vec![0u16; ONESHOT_RANKS * 5120];
    assert!(matches!(
        handle.all_gather_bf16(&[0u16; 5120], &mut gathered, 0),
        Err(CommError::Unsupported(_))
    ));
    let unregistered = DeviceBuffer::new(0x1000, 10_240).expect("descriptor");
    assert!(matches!(
        handle.all_reduce_bf16_device(&unregistered, 5120, 0),
        Err(CommError::InvalidLayout(_))
    ));
    let other_rank = HostDeviceBuffer::allocate(&cluster, 3, 5120).expect("buffer");
    assert!(matches!(
        handle.all_reduce_bf16_device(&other_rank.device_buffer(), 5120, 0),
        Err(CommError::InvalidLayout(_))
    ));
}

#[test]
fn a_chunked_plan_refuses_an_output_of_the_wrong_size() {
    let total = ONESHOT_ROW_ELEMENTS * 2 + 1024;
    let cluster = LoopbackDeviceCluster::new(LoopbackDeviceConfig::default()).expect("cluster");
    let handle = cluster.handle(0).expect("handle");
    let small = HostDeviceBuffer::allocate(&cluster, 0, 1024).expect("small");
    let plan = sparkflash_comm::oneshot::ChunkPlan::new(total * 2, ONESHOT_RAILS).expect("plan");
    assert!(handle.run_pipeline(&small, &plan, 0).is_err());
}

#[test]
fn twin_payloads_are_finite_and_deterministic() {
    let first = twin_payload(2, 7, 1024);
    let second = twin_payload(2, 7, 1024);
    assert_eq!(first, second);
    assert_ne!(twin_payload(2, 8, 1024), first, "the sequence contributes");
    for value in &first {
        let exponent = (value >> 7) & 0xff;
        assert!(exponent > 0 && exponent < 0xff, "finite, non-zero exponent");
    }
}
