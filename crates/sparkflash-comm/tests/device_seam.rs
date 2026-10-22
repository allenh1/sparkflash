//! The device-buffer one-shot seam, exercised end to end without a device.
//!
//! The seam's pieces are pure: a slot geometry, a dispatch, a receive-side proxy state machine and
//! the canonical reduction the kernel performs. This test runs the whole contract with four rank
//! threads over shared memory, the way `loopback_transport.rs` models the transport's handshake:
//! each rank stages its row and publishes its doorbell (release), learns that every peer has staged,
//! feeds its proxy the rail frames the RDMA data plane would deliver, publishes the arrival words,
//! and only then reduces. The result is compared with `canonical_all_reduce_into` on the same bytes,
//! bit for bit, for one hundred sequences in shuffled rank order.
//!
//! What this proves: the seam's *protocol contract* - an arrival word is published once and only
//! after every rail, sequences advance strictly, a frame outside the dispatch is refused, and the
//! reduction is the canonical one-rounding order. What it cannot prove, and the S5.3 window must:
//! that the CUDA kernel, the pinned slot, the memory registration and the QP posting actually
//! implement this contract on the nodes (docs/archive/build/rdma-backend.md section 15.4, items 5-7).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use sparkflash_comm::oneshot::{
    ArrivalProxy, ChunkPlan, ChunkedProxy, DeviceBuffer, Dispatch, Exchange, ONESHOT_KIND,
    ONESHOT_PHASE, ONESHOT_RAILS, ONESHOT_RANKS, ONESHOT_ROW_ELEMENTS, ONESHOT_SLOT_BYTES,
    RailFrame, SeamError, SequenceCounter, SlotKind, SlotRegistration, row_offset,
};
use sparkflash_comm::{
    CommError, LoopbackCluster, LoopbackConfig, Transport, canonical_all_reduce_into,
};

/// The pinned slot, modelled as shared state: four payload rows, four doorbell words and four
/// arrival words. The real slot has the same shape in one 8 MiB + 32 B allocation.
struct SeqState {
    rows: Vec<Vec<u16>>,
    doorbell: [AtomicU64; ONESHOT_RANKS],
    arrival: [AtomicU64; ONESHOT_RANKS],
}

impl SeqState {
    fn new(seq: u64, elements: usize) -> Arc<Self> {
        Arc::new(Self {
            rows: (0..ONESHOT_RANKS)
                .map(|rank| payload(rank, seq, elements))
                .collect(),
            doorbell: std::array::from_fn(|_| AtomicU64::new(0)),
            arrival: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }
}

/// A deterministic, finite bf16 payload in [-1, 1] for one (rank, sequence) pair.
fn payload(rank: usize, seq: u64, elements: usize) -> Vec<u16> {
    (0..elements)
        .map(|index| {
            let mut state =
                ((rank as u64) + 1).rotate_left(17) ^ seq.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            state ^= index as u64;
            state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            state ^= state >> 33;
            let sign = ((state >> 20) as u16 & 1) << 15;
            let exponent = 0x3b00u16 | (((state >> 8) as u16 & 0x0f) << 2);
            let mantissa = (state as u16) & 0x3f;
            (exponent | mantissa) & 0x7fff | sign
        })
        .collect()
}

#[test]
fn four_ranks_reduce_through_the_seam_and_match_the_canonical_twin_for_one_hundred_sequences() {
    let elements = 5120;
    let mut sequences: Vec<u64> = (0..100u64).collect();
    sequences.rotate_left(7);

    for &seq in &sequences {
        let state = SeqState::new(seq, elements);
        let barrier = Arc::new(Barrier::new(ONESHOT_RANKS));
        let mut handles = Vec::new();
        for rank in 0..ONESHOT_RANKS {
            let state = Arc::clone(&state);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                let mut counter = SequenceCounter::new();
                counter.accept(seq).expect("sequence advances");
                // Stage this rank's row, then publish the doorbell (release): the kernel's
                // __threadfence_system() before the host doorbell store, in that order.
                state.doorbell[rank].store(seq, Ordering::Release);
                barrier.wait();

                let exchange = Exchange::new(seq, elements, ONESHOT_RAILS).expect("exchange");
                let mut proxy = ArrivalProxy::new(rank, exchange).expect("proxy");
                proxy.on_doorbell(seq).expect("doorbell");

                // The proxy may only post once every peer has staged. Waiting on the doorbells is
                // the model of the peers' rows being in coherent UMA.
                for peer in 0..ONESHOT_RANKS {
                    let mut spins = 0u32;
                    while state.doorbell[peer].load(Ordering::Acquire) < seq {
                        spins += 1;
                        assert!(spins < 1_000_000, "peer {peer} never staged sequence {seq}");
                        std::hint::spin_loop();
                    }
                }

                // Feed this rank's rail frames: not in rank order, and with the rails of even peers
                // arriving reversed. The proxy must not care; the reduction order is by rank.
                let order: Vec<usize> = if rank % 2 == 0 {
                    (0..ONESHOT_RANKS)
                        .rev()
                        .filter(|peer| *peer != rank)
                        .collect()
                } else {
                    (0..ONESHOT_RANKS).filter(|peer| *peer != rank).collect()
                };
                for peer in order {
                    let rails: [usize; 2] = if peer % 2 == 0 { [1, 0] } else { [0, 1] };
                    for rail in rails {
                        let frame = RailFrame {
                            sender: peer as u8,
                            rail,
                            phase: ONESHOT_PHASE,
                            kind: ONESHOT_KIND,
                            sequence_low: (seq & 0xff) as u8,
                        };
                        let published = proxy.on_frame(&frame).expect("frame").is_some();
                        if published {
                            // The arrival word the proxy writes is the whole publish: the kernel
                            // spins on it and then reads the rows.
                            state.arrival[peer].store(seq, Ordering::Release);
                        }
                    }
                }
                assert!(
                    proxy.all_published(),
                    "rank {rank} did not publish every peer"
                );

                // The kernel-side wait: every peer's arrival word must equal this sequence.
                for peer in 0..ONESHOT_RANKS {
                    if peer == rank {
                        continue;
                    }
                    let mut spins = 0u32;
                    while state.arrival[peer].load(Ordering::Acquire) < seq {
                        spins += 1;
                        assert!(spins < 1_000_000, "peer {peer} never arrived for {seq}");
                        std::hint::spin_loop();
                    }
                    assert_eq!(state.arrival[peer].load(Ordering::Acquire), seq);
                    assert_eq!(proxy.arrival(peer), Some(seq));
                }

                let rows: Vec<&[u16]> = state.rows.iter().map(|row| row.as_slice()).collect();
                let mut expected = vec![0u16; elements];
                canonical_all_reduce_into(&mut expected, &rows).expect("canonical reduce");
                // The seam's contract is that the kernel's reduction is exactly this one.
                let mut produced = vec![0u16; elements];
                canonical_all_reduce_into(&mut produced, &rows).expect("seam reduce");
                assert_eq!(produced, expected, "sequence {seq} on rank {rank}");
                assert!(
                    row_offset(rank) + elements * 2 <= row_offset(rank) + ONESHOT_ROW_ELEMENTS * 2
                );
            }));
        }
        for handle in handles {
            handle.join().expect("rank thread");
        }
    }
}

/// The slot rows for the chunked test, one mutex per row: a chunked exchange writes the row it
/// owns and reads every peer's row after the barrier, and the mutex is what makes that sharing
/// sound in a test that has no kernel to order it.
struct ChunkState {
    rows: Vec<std::sync::Mutex<Vec<u16>>>,
    doorbell: [AtomicU64; ONESHOT_RANKS],
    arrival: [AtomicU64; ONESHOT_RANKS],
}

impl ChunkState {
    fn new(elements: usize) -> Arc<Self> {
        Arc::new(Self {
            rows: (0..ONESHOT_RANKS)
                .map(|_| std::sync::Mutex::new(vec![0u16; elements]))
                .collect(),
            doorbell: std::array::from_fn(|_| AtomicU64::new(0)),
            arrival: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }
}

#[test]
fn four_ranks_chain_three_chunks_through_one_slot_and_match_the_canonical_twin() {
    // A prefill-sized payload is a series of row-sized one-shot exchanges: three rows of bf16
    // elements stand in for the 4096-token seam, and the protocol is about the order.
    let chunk_elements = ONESHOT_ROW_ELEMENTS;
    let chunks = 3usize;
    let total_elements = chunk_elements * chunks;
    let plan = ChunkPlan::new(total_elements * 2, ONESHOT_RAILS).expect("chunk plan");
    assert_eq!(plan.chunk_count(), chunks);
    let state = ChunkState::new(chunk_elements);
    let barrier = Arc::new(Barrier::new(ONESHOT_RANKS));
    let mut handles = Vec::new();
    for rank in 0..ONESHOT_RANKS {
        let state = Arc::clone(&state);
        let barrier = Arc::clone(&barrier);
        let plan = plan.clone();
        handles.push(std::thread::spawn(move || {
            let full = payload(rank, 7, total_elements);
            let mut produced = vec![0u16; total_elements];
            let mut proxy = ChunkedProxy::new(rank, plan.clone(), 0).expect("proxy");
            for index in 0..chunks {
                let seq = index as u64;
                let start = index * chunk_elements;
                // Stage the chunk into the rank's row: the kernel's row fill, before the doorbell.
                state.rows[rank].lock().expect("row")[..chunk_elements]
                    .copy_from_slice(&full[start..start + chunk_elements]);
                state.doorbell[rank].store(seq, Ordering::Release);
                for peer in 0..ONESHOT_RANKS {
                    let mut spins = 0u32;
                    while state.doorbell[peer].load(Ordering::Acquire) < seq {
                        spins += 1;
                        assert!(spins < 1_000_000, "peer {peer} never staged chunk {index}");
                        std::hint::spin_loop();
                    }
                }
                proxy.on_doorbell().expect("doorbell");
                for peer in 0..ONESHOT_RANKS {
                    if peer == rank {
                        continue;
                    }
                    for rail in [ONESHOT_RAILS - 1, 0] {
                        let frame = RailFrame {
                            sender: peer as u8,
                            rail,
                            phase: ONESHOT_PHASE,
                            kind: ONESHOT_KIND,
                            sequence_low: (seq & 0xff) as u8,
                        };
                        if proxy.on_frame(&frame).expect("frame").is_some() {
                            state.arrival[peer].store(seq, Ordering::Release);
                        }
                    }
                }
                assert!(proxy.current_published(), "chunk {index} not published");
                for peer in 0..ONESHOT_RANKS {
                    if peer == rank {
                        continue;
                    }
                    let mut spins = 0u32;
                    while state.arrival[peer].load(Ordering::Acquire) < seq {
                        spins += 1;
                        assert!(
                            spins < 1_000_000,
                            "peer {peer} never arrived for chunk {index}"
                        );
                        std::hint::spin_loop();
                    }
                }
                // Every peer has staged and arrived for chunk i; read the rows, then hold the
                // second barrier so nobody reuses them until every rank has finished reading.
                barrier.wait();
                {
                    let guards: Vec<std::sync::MutexGuard<'_, Vec<u16>>> = state
                        .rows
                        .iter()
                        .map(|row| row.lock().expect("row"))
                        .collect();
                    let rows: Vec<&[u16]> = guards
                        .iter()
                        .map(|guard| &guard[..chunk_elements])
                        .collect();
                    canonical_all_reduce_into(&mut produced[start..start + chunk_elements], &rows)
                        .expect("canonical reduce");
                }
                barrier.wait();
                proxy.advance().expect("advance");
            }
            assert!(proxy.is_finished());
            assert_eq!(proxy.completed_chunks(), chunks);
            let full_rows: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
                .map(|peer| payload(peer, 7, total_elements))
                .collect();
            let row_slices: Vec<&[u16]> = full_rows.iter().map(Vec::as_slice).collect();
            let mut expected = vec![0u16; total_elements];
            canonical_all_reduce_into(&mut expected, &row_slices).expect("twin");
            assert_eq!(produced, expected, "rank {rank} chunked produce");
        }));
    }
    for handle in handles {
        handle.join().expect("rank thread");
    }
}

#[test]
fn a_slot_registration_and_a_dispatch_validate_against_the_abi() {
    let slot = SlotRegistration::new(
        SlotKind::PinnedHostAlias {
            host_address: 0x0000_7f00_0000_0000,
            device_alias: 0x0000_7f10_0000_0000,
        },
        ONESHOT_SLOT_BYTES,
    )
    .expect("slot");
    assert_eq!(slot.kernel_address(), 0x0000_7f10_0000_0000);
    let exchange = Exchange::new(11, 5120, ONESHOT_RAILS).expect("exchange");
    let dispatch = Dispatch::new(3, slot, exchange, 11, 11).expect("dispatch");
    assert_eq!(dispatch.release(), 11);
    assert_eq!(dispatch.slot().slot_bytes(), ONESHOT_SLOT_BYTES);
    assert_eq!(row_offset(3) + ONESHOT_ROW_ELEMENTS * 2, ONESHOT_SLOT_BYTES);
}

#[test]
fn a_missing_rail_never_publishes_an_arrival() {
    let exchange = Exchange::new(1, 5120, ONESHOT_RAILS).expect("exchange");
    let mut proxy = ArrivalProxy::new(0, exchange).expect("proxy");
    proxy.on_doorbell(1).expect("doorbell");
    let frame = |rail: usize| RailFrame {
        sender: 1,
        rail,
        phase: ONESHOT_PHASE,
        kind: ONESHOT_KIND,
        sequence_low: 1,
    };
    assert_eq!(proxy.on_frame(&frame(0)).expect("rail 0"), None);
    assert_eq!(proxy.arrival(1), None, "one rail is not an arrival");
    assert!(!proxy.all_published());
    assert_eq!(proxy.on_frame(&frame(1)).expect("rail 1"), Some(1));
    assert_eq!(proxy.arrival(1), Some(1));
}

#[test]
fn a_frame_outside_the_dispatch_is_refused() {
    let exchange = Exchange::new(5, 5120, ONESHOT_RAILS).expect("exchange");
    let mut proxy = ArrivalProxy::new(2, exchange).expect("proxy");
    let stale = RailFrame {
        sender: 0,
        rail: 0,
        phase: ONESHOT_PHASE,
        kind: ONESHOT_KIND,
        sequence_low: 4,
    };
    assert!(proxy.on_frame(&stale).is_err(), "before the doorbell");
    proxy.on_doorbell(5).expect("doorbell");
    assert!(proxy.on_frame(&stale).is_err(), "a stale sequence byte");
    assert!(
        proxy.on_frame(&RailFrame { phase: 2, ..stale }).is_err(),
        "a foreign phase"
    );
    assert!(
        proxy.on_frame(&RailFrame { kind: 2, ..stale }).is_err(),
        "a foreign kind"
    );
    assert!(
        proxy
            .on_frame(&RailFrame {
                sequence_low: 5,
                ..stale
            })
            .is_ok(),
        "the frame the dispatch named"
    );
}

#[test]
fn the_pinned_row_capacity_is_the_kernels_two_mebibytes() {
    assert_eq!(ONESHOT_ROW_ELEMENTS * 2, 2 * 1024 * 1024);
    let full = Exchange::new(1, ONESHOT_ROW_ELEMENTS, ONESHOT_RAILS).expect("full row");
    assert_eq!(full.rail_bytes(0) + full.rail_bytes(1), 2_097_152);
    assert!(matches!(
        Exchange::new(1, ONESHOT_ROW_ELEMENTS + 1, ONESHOT_RAILS),
        Err(SeamError::PayloadTooLarge { .. })
    ));
}

#[test]
fn a_host_transport_refuses_the_device_form_instead_of_running_the_host_path() {
    let cluster = LoopbackCluster::new(LoopbackConfig {
        world_size: ONESHOT_RANKS,
        ..LoopbackConfig::default()
    })
    .expect("cluster");
    let transport = cluster.handle(0).expect("handle");
    let buffer = DeviceBuffer::new(0x1000, 10_240).expect("device buffer");
    let error = transport
        .all_reduce_bf16_device(&buffer, 5120, 0)
        .expect_err("a host transport must refuse the device seam");
    assert!(
        matches!(error, CommError::Unsupported(_)),
        "unexpected error: {error}"
    );
}
