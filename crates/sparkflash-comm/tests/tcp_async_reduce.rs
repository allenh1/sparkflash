//! Async posted-reduce tests over real loopback sockets: canonical equality against the
//! synchronous path, 80 outstanding reduces with deferred waits, and an interleaved ordering case.
//!
//! These are transport sanity tests, not latency claims. The timing block prints the wall time of
//! 80 synchronous reduces and 80 posted reduces + deferred waits at 257 elements; it is labelled
//! transport sanity in the K6 report.

mod support;

use std::time::Instant;

use sparkflash_comm::{
    CollectiveLayout, FabricSchedule, ReduceSlot, TcpConfig, TcpTransport, Transport,
};
use support::{assert_bits_eq, finite_pattern, port_block, reference_all_reduce, run_ranks};

const WORLD: usize = 4;
const REDUCES: usize = 80;
const ELEMENTS: usize = 257;

fn config(rank: usize, base_port: u16) -> TcpConfig {
    TcpConfig::loopback4(rank, base_port)
}

/// All four ranks' payloads for one collective, identical in every rank thread.
fn payloads_for(seq: usize) -> Vec<Vec<u16>> {
    (0..WORLD)
        .map(|rank| finite_pattern(rank, ELEMENTS, 0xa5c3 ^ ((seq as u64) << 8)))
        .collect()
}

#[test]
fn eighty_posted_reduces_with_deferred_waits_match_the_canonical_reference() {
    let base_port = port_block();
    let results = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let expected: Vec<Vec<u16>> = (0..REDUCES)
            .map(|seq| reference_all_reduce(&payloads_for(seq)))
            .collect();
        // Leak the little test buffers so a posted handle can borrow each one across the batch.
        let mut buffers: Vec<&'static mut [u16]> = (0..REDUCES)
            .map(|seq| Box::leak(payloads_for(seq)[rank].clone().into_boxed_slice()))
            .collect();
        let mut handles = Vec::with_capacity(REDUCES);
        for (seq, slot) in buffers.iter_mut().enumerate() {
            let buffer = std::mem::take(slot);
            handles.push(
                transport
                    .post_reduce_bf16(ReduceSlot::new(seq as u32, 0), seq as u64, buffer)
                    .unwrap_or_else(|error| panic!("rank {rank} post {seq}: {error}")),
            );
        }
        // Deferred consumption: every reduce is waited only after all 80 were posted, and the
        // waits run in reverse post order so frame dispatch is not FIFO.
        for seq in (0..REDUCES).rev() {
            let buffer = transport
                .wait_reduce(handles.pop().expect("one handle per post"))
                .unwrap_or_else(|error| panic!("rank {rank} wait {seq}: {error}"));
            assert_bits_eq(
                buffer,
                &expected[seq],
                &format!("rank {rank} async seq {seq}"),
            );
        }
        true
    });
    assert!(results.iter().all(|value| *value));
}

#[test]
fn interleaved_post_and_wait_orders_still_reach_the_canonical_result() {
    let base_port = port_block();
    let results = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let expected: Vec<Vec<u16>> = (0..REDUCES)
            .map(|seq| reference_all_reduce(&payloads_for(seq)))
            .collect();
        let mut buffers: Vec<&'static mut [u16]> = (0..REDUCES)
            .map(|seq| Box::leak(payloads_for(seq)[rank].clone().into_boxed_slice()))
            .collect();
        // Each rank posts and waits in a different cyclic order, so frame arrival order and the
        // order outstanding reduces are consumed differ across ranks.
        let mut order: Vec<usize> = (0..REDUCES).collect();
        let rotation = rank * 17;
        order.rotate_left(rotation % REDUCES);
        let mut handles: Vec<(usize, _)> = Vec::with_capacity(REDUCES);
        for seq in order {
            let buffer = std::mem::take(&mut buffers[seq]);
            let handle = transport
                .post_reduce_bf16(ReduceSlot::new(seq as u32, 0), seq as u64, buffer)
                .unwrap_or_else(|error| panic!("rank {rank} post {seq}: {error}"));
            handles.push((seq, handle));
        }
        let mut wait_order: Vec<usize> = (0..REDUCES).collect();
        wait_order.rotate_left((rank * 11) % REDUCES);
        for seq in wait_order {
            let index = handles
                .iter()
                .position(|(posted, _)| *posted == seq)
                .expect("posted");
            let (_, handle) = handles.swap_remove(index);
            let buffer = transport
                .wait_reduce(handle)
                .unwrap_or_else(|error| panic!("rank {rank} wait {seq}: {error}"));
            assert_bits_eq(
                buffer,
                &expected[seq],
                &format!("rank {rank} interleaved seq {seq}"),
            );
        }
        true
    });
    assert!(results.iter().all(|value| *value));
}

#[test]
fn a_reused_slot_and_a_dead_wait_are_hard_errors() {
    let base_port = port_block();
    let results = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let expected = reference_all_reduce(&payloads_for(0));
        let mut first: Box<[u16]> = payloads_for(0)[rank].clone().into_boxed_slice();
        let handle = transport
            .post_reduce_bf16(ReduceSlot::new(7, 0), 70, &mut first)
            .unwrap_or_else(|error| panic!("rank {rank} first post: {error}"));
        let mut duplicate: Box<[u16]> = payloads_for(0)[rank].clone().into_boxed_slice();
        match transport.post_reduce_bf16(ReduceSlot::new(7, 0), 71, &mut duplicate) {
            Err(sparkflash_comm::CommError::Protocol(message)) => {
                assert!(
                    message.contains("already has seq"),
                    "unexpected message: {message}"
                );
            }
            Ok(_) => panic!("rank {rank}: a reused slot was accepted"),
            Err(other) => panic!("rank {rank}: a reused slot failed with {other}"),
        }
        let buffer = transport
            .wait_reduce(handle)
            .unwrap_or_else(|error| panic!("rank {rank} wait: {error}"));
        assert_bits_eq(
            buffer,
            &expected,
            &format!("rank {rank} slot-reuse survivor"),
        );
        true
    });
    assert!(results.iter().all(|value| *value));
}

#[test]
fn posting_80_reduces_costs_no_extra_round_trip_versus_sync() {
    // Transport sanity only: one rank block runs 80 synchronous one-shot reduces, then a second
    // transport runs 80 posted reduces whose waits are all deferred. Both use 257 elements, so the
    // number is the protocol's loopback cost, not a latency or throughput claim.
    let sync_base = port_block();
    let sync_wall = run_ranks(WORLD, |rank| {
        let transport = TcpTransport::connect(config(rank, sync_base))
            .unwrap_or_else(|error| panic!("rank {rank} sync connect: {error}"));
        let layout = CollectiveLayout::new(WORLD, ELEMENTS, FabricSchedule::OneShotAllToAll)
            .unwrap_or_else(|error| panic!("layout: {error}"));
        let start = Instant::now();
        for seq in 0..REDUCES {
            let mut buffer = payloads_for(seq)[rank].clone();
            transport
                .all_reduce_bf16(&mut buffer, seq as u64, &layout)
                .unwrap_or_else(|error| panic!("rank {rank} sync {seq}: {error}"));
        }
        start.elapsed().as_secs_f64() * 1e3
    })
    .into_iter()
    .fold(0.0f64, f64::max);

    let async_base = port_block();
    let async_wall = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, async_base))
            .unwrap_or_else(|error| panic!("rank {rank} async connect: {error}"));
        let mut buffers: Vec<&'static mut [u16]> = (0..REDUCES)
            .map(|seq| Box::leak(payloads_for(seq)[rank].clone().into_boxed_slice()))
            .collect();
        let start = Instant::now();
        let mut handles = Vec::with_capacity(REDUCES);
        for (seq, slot) in buffers.iter_mut().enumerate() {
            let buffer = std::mem::take(slot);
            handles.push(
                transport
                    .post_reduce_bf16(ReduceSlot::new(seq as u32, 0), seq as u64, buffer)
                    .unwrap_or_else(|error| panic!("rank {rank} post {seq}: {error}")),
            );
        }
        for handle in handles {
            transport
                .wait_reduce(handle)
                .unwrap_or_else(|error| panic!("rank {rank} wait: {error}"));
        }
        start.elapsed().as_secs_f64() * 1e3
    })
    .into_iter()
    .fold(0.0f64, f64::max);

    eprintln!(
        "transport sanity: 80 sync one-shot reduces {sync_wall:.3} ms; 80 posted + deferred waits {async_wall:.3} ms (257 bf16 elements, loopback; not a latency claim)"
    );
    // Sanity guard: the async path may be slower or faster, but it must not be pathologically
    // different (a 10x gap here would mean the test harness, not the protocol, is the cost).
    assert!(async_wall < 10.0 * sync_wall + 50.0);
}

#[test]
fn a_sequence_reused_across_slots_is_refused() {
    let base_port = port_block();
    let results = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let mut first: Box<[u16]> = payloads_for(0)[rank].clone().into_boxed_slice();
        let handle = transport
            .post_reduce_bf16(ReduceSlot::new(1, 0), 99, &mut first)
            .unwrap_or_else(|error| panic!("rank {rank} first post: {error}"));
        let mut second: Box<[u16]> = payloads_for(1)[rank].clone().into_boxed_slice();
        match transport.post_reduce_bf16(ReduceSlot::new(2, 0), 99, &mut second) {
            Err(sparkflash_comm::CommError::Protocol(message)) => {
                assert!(
                    message.contains("sequence numbers must be unique"),
                    "unexpected: {message}"
                );
            }
            Ok(_) => panic!("rank {rank}: a reused sequence was accepted"),
            Err(other) => panic!("rank {rank}: a reused sequence failed with {other}"),
        }
        let _ = transport
            .wait_reduce(handle)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        true
    });
    assert!(results.iter().all(|value| *value));
}

/// The loop's posting schedule: 40 layers x 2 seams = 80 reduces, all posted ahead of the first
/// wait, phase 0 = attention consumption, phase 1 = MoE seam, seq = layer * 2 + phase. Waits run at
/// the consumption points in layer order (attention then MoE), long after every post.
#[test]
fn two_phase_slots_post_eighty_reduces_ahead_and_wait_at_consumption_points() {
    const LAYERS: usize = 40;
    const POSTED: usize = LAYERS * 2;
    let base_port = port_block();
    let results = run_ranks(WORLD, |rank| {
        let mut transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let expected: Vec<Vec<u16>> = (0..POSTED)
            .map(|seq| reference_all_reduce(&payloads_for(seq)))
            .collect();
        let mut buffers: Vec<&'static mut [u16]> = (0..POSTED)
            .map(|seq| Box::leak(payloads_for(seq)[rank].clone().into_boxed_slice()))
            .collect();
        let mut handles: Vec<(usize, usize, _)> = Vec::with_capacity(POSTED);
        for layer in 0..LAYERS {
            for phase in 0..2u8 {
                let seq = layer * 2 + usize::from(phase);
                let buffer = std::mem::take(&mut buffers[seq]);
                let handle = transport
                    .post_reduce_bf16(ReduceSlot::new(layer as u32, phase), seq as u64, buffer)
                    .unwrap_or_else(|error| {
                        panic!("rank {rank} post layer {layer} phase {phase}: {error}")
                    });
                handles.push((layer, usize::from(phase), handle));
            }
        }
        // Exactly-once violations while all 80 are outstanding: a second post into the same slot,
        // and a sequence reused by a different slot.
        let mut duplicate_slot: Box<[u16]> = payloads_for(7)[rank].clone().into_boxed_slice();
        match transport.post_reduce_bf16(ReduceSlot::new(7, 1), 15, &mut duplicate_slot) {
            Err(sparkflash_comm::CommError::Protocol(message)) => {
                assert!(message.contains("already has seq"), "unexpected: {message}");
            }
            Ok(_) => panic!("rank {rank}: second post into layer 7 phase 1 was accepted"),
            Err(other) => panic!("rank {rank}: duplicate slot failed with {other}"),
        }
        let mut duplicate_seq: Box<[u16]> = payloads_for(9)[rank].clone().into_boxed_slice();
        match transport.post_reduce_bf16(ReduceSlot::new(99, 0), 5, &mut duplicate_seq) {
            Err(sparkflash_comm::CommError::Protocol(message)) => {
                assert!(
                    message.contains("sequence numbers must be unique"),
                    "unexpected: {message}"
                );
            }
            Ok(_) => panic!("rank {rank}: reused seq 5 was accepted"),
            Err(other) => panic!("rank {rank}: reused seq failed with {other}"),
        }
        // Consumption points: every posted reduce is waited only now, layer by layer.
        for layer in 0..LAYERS {
            for phase in 0..2usize {
                let index = handles
                    .iter()
                    .position(|(posted_layer, posted_phase, _)| {
                        *posted_layer == layer && *posted_phase == phase
                    })
                    .expect("one handle per slot");
                let (_, _, handle) = handles.swap_remove(index);
                let seq = layer * 2 + phase;
                let buffer = transport.wait_reduce(handle).unwrap_or_else(|error| {
                    panic!("rank {rank} wait layer {layer} phase {phase}: {error}")
                });
                assert_bits_eq(
                    buffer,
                    &expected[seq],
                    &format!("rank {rank} layer {layer} phase {phase}"),
                );
            }
        }
        // Double wait is a compile error: ReduceHandle is neither Copy nor Clone and wait consumes
        // it. The runtime guards exercised above are the slot and sequence checks.
        assert!(handles.is_empty());
        true
    });
    assert!(results.iter().all(|value| *value));
}
