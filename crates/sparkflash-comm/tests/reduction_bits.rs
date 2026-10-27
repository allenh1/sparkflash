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

//! Headline requirement: the transport produces bits identical to the canonical rank-order
//! accumulation, and the independent CPU reference proves it.

mod support;

use std::time::Duration;

use support::{
    Lcg, assert_bits_eq, finite_pattern, reference_all_reduce, run_ranks, run_ranks_order,
    tricky_pattern,
};

use sparkflash_comm::{
    CollectiveLayout, DualPathCommunicator, FabricSchedule, LoopbackCluster, LoopbackConfig,
    ROCENANTE_MAX_BYTES, SparkFabricMode, Transport,
};

fn cluster(world: usize, send_delay: Duration) -> LoopbackCluster {
    LoopbackCluster::new(LoopbackConfig {
        world_size: world,
        timeout: support::CLUSTER_TIMEOUT,
        send_delay,
        ..LoopbackConfig::default()
    })
    .unwrap_or_else(|error| panic!("loopback cluster: {error}"))
}

fn reduce_on_loopback(
    cluster: &LoopbackCluster,
    world: usize,
    payloads: &[Vec<u16>],
    elements: usize,
    schedule: FabricSchedule,
) -> Vec<Vec<u16>> {
    let layout = CollectiveLayout::new(world, elements, schedule)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    run_ranks(world, |rank| {
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
        let mut buf = payloads[rank].clone();
        transport
            .all_reduce_bf16(&mut buf, 1, &layout)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        buf
    })
}

#[test]
fn both_schedules_match_the_independent_cpu_reference() {
    let world = 4;
    let elements = 64;
    for schedule in [
        FabricSchedule::OneShotAllToAll,
        FabricSchedule::HairpinRingReduceScatterAllGather,
    ] {
        let cluster = cluster(world, Duration::ZERO);
        let payloads: Vec<Vec<u16>> = (0..world)
            .map(|rank| tricky_pattern(rank, elements))
            .collect();
        let expected = reference_all_reduce(&payloads);
        let results = reduce_on_loopback(&cluster, world, &payloads, elements, schedule);
        for (rank, buf) in results.iter().enumerate() {
            assert_bits_eq(buf, &expected, &format!("rank {rank} {schedule:?}"));
        }
    }
}

#[test]
fn hairpin_ring_is_bit_identical_to_the_one_shot() {
    let world = 4;
    for elements in [1usize, 7, 8, 9, 64, 1000, 4097] {
        for pattern in 0..3 {
            let payloads: Vec<Vec<u16>> = (0..world)
                .map(|rank| match pattern {
                    0 => tricky_pattern(rank, elements),
                    1 => finite_pattern(rank, elements, 0xdead_beef),
                    _ => vec![0x8000; elements],
                })
                .collect();
            let expected = reference_all_reduce(&payloads);
            let one_shot = reduce_on_loopback(
                &cluster(world, Duration::ZERO),
                world,
                &payloads,
                elements,
                FabricSchedule::OneShotAllToAll,
            );
            let hairpin = reduce_on_loopback(
                &cluster(world, Duration::ZERO),
                world,
                &payloads,
                elements,
                FabricSchedule::HairpinRingReduceScatterAllGather,
            );
            for rank in 0..world {
                assert_bits_eq(&one_shot[rank], &expected, "one-shot");
                assert_bits_eq(&hairpin[rank], &expected, "hairpin");
                assert_bits_eq(&hairpin[rank], &one_shot[rank], "hairpin vs one-shot");
            }
        }
    }
}

#[test]
fn element_counts_that_do_not_divide_the_stride_still_match() {
    // The kernel's block/grid stride means the last block often handles a partial tile; the
    // transport must be correct for any element count, including counts below world_size.
    for world in [1usize, 2, 3, 4] {
        for elements in [
            0usize, 1, 2, 3, 5, 7, 8, 15, 16, 17, 33, 100, 1023, 1024, 1025,
        ] {
            for schedule in [
                FabricSchedule::OneShotAllToAll,
                FabricSchedule::HairpinRingReduceScatterAllGather,
            ] {
                let cluster = cluster(world, Duration::ZERO);
                let payloads: Vec<Vec<u16>> = (0..world)
                    .map(|rank| finite_pattern(rank, elements, 0x5eed + elements as u64))
                    .collect();
                let expected = reference_all_reduce(&payloads);
                let results = reduce_on_loopback(&cluster, world, &payloads, elements, schedule);
                for (rank, buf) in results.iter().enumerate() {
                    assert_bits_eq(
                        buf,
                        &expected,
                        &format!("world {world} elements {elements} rank {rank} {schedule:?}"),
                    );
                }
            }
        }
    }
}

#[test]
fn arrival_order_and_thread_count_do_not_change_the_bits() {
    let mut lcg = Lcg::new(0xc0ff_ee00);
    for iteration in 0..24u64 {
        let world = 2 + (iteration as usize % 3);
        let elements = 1 + (lcg.next() % 257) as usize;
        let delay = Duration::from_micros(20 + lcg.next() % 300);
        let schedule = if iteration % 2 == 0 {
            FabricSchedule::OneShotAllToAll
        } else {
            FabricSchedule::HairpinRingReduceScatterAllGather
        };
        let cluster = cluster(world, delay);
        let payloads: Vec<Vec<u16>> = (0..world)
            .map(|rank| finite_pattern(rank, elements, 0x1234_5678 + iteration))
            .collect();
        let expected = reference_all_reduce(&payloads);
        let layout = CollectiveLayout::new(world, elements, schedule)
            .unwrap_or_else(|error| panic!("layout: {error}"));

        // Spawn the rank threads in a shuffled order as well as staggering their sends.
        let mut order: Vec<usize> = (0..world).collect();
        for index in (1..world).rev() {
            let swap = (lcg.next() as usize) % (index + 1);
            order.swap(index, swap);
        }
        let results = run_ranks_order(&order, |rank| {
            let transport = cluster
                .handle(rank)
                .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
            let mut buf = payloads[rank].clone();
            transport
                .all_reduce_bf16(&mut buf, iteration + 1, &layout)
                .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
            buf
        });
        for (rank, buf) in results {
            assert_bits_eq(
                &buf,
                &expected,
                &format!("iteration {iteration} world {world} rank {rank}"),
            );
        }
    }
}

#[test]
fn two_mib_and_two_mib_minus_one_take_the_same_path_with_the_same_bits() {
    let communicator = DualPathCommunicator::new(0, 4, SparkFabricMode::Switched200G);
    // The binding threshold is on bytes: 2 MiB - 1 and 2 MiB both route to RoCEnante, 2 MiB + 1
    // goes to the NCCL ring.
    let below = communicator.dispatch_all_reduce_bf16(3, ROCENANTE_MAX_BYTES - 1, false);
    let at = communicator.dispatch_all_reduce_bf16(3, ROCENANTE_MAX_BYTES, false);
    let above = communicator.dispatch_all_reduce_bf16(3, ROCENANTE_MAX_BYTES + 1, false);
    assert!(below.is_rocenante());
    assert!(at.is_rocenante());
    assert!(!above.is_rocenante());
    let (below_seq, at_seq) = match (below, at) {
        (
            sparkflash_comm::CollectiveRoute::RoCEnanteOneShot { seq, .. },
            sparkflash_comm::CollectiveRoute::RoCEnanteOneShot { seq: second, .. },
        ) => {
            assert_eq!(seq, 1);
            assert_eq!(second, 2);
            (seq, second)
        }
        other => panic!("both dispatches must be RoCEnante: {other:?}"),
    };
    assert_eq!(below_seq + 1, at_seq);

    // And the bits: a 2 MiB buffer is exactly 1,048,576 bf16 elements.
    let world = 4;
    let big = ROCENANTE_MAX_BYTES / 2;
    for elements in [big - 1, big] {
        for schedule in [
            FabricSchedule::OneShotAllToAll,
            FabricSchedule::HairpinRingReduceScatterAllGather,
        ] {
            let cluster = cluster(world, Duration::ZERO);
            let payloads: Vec<Vec<u16>> = (0..world)
                .map(|rank| finite_pattern(rank, elements, 0xabba + elements as u64))
                .collect();
            let expected = reference_all_reduce(&payloads);
            let results = reduce_on_loopback(&cluster, world, &payloads, elements, schedule);
            for (rank, buf) in results.iter().enumerate() {
                assert_bits_eq(
                    buf,
                    &expected,
                    &format!("elements {elements} rank {rank} {schedule:?}"),
                );
            }
        }
    }
    // The two buffer sizes are one element apart, so their routes must differ only by seq.
    assert_eq!(ROCENANTE_MAX_BYTES - 2, (big - 1) * 2);
}

#[test]
fn all_gather_reassembles_every_segment_exactly() {
    let world = 4;
    for elements in [0usize, 1, 7, 1003] {
        let cluster = cluster(world, Duration::ZERO);
        let locals: Vec<Vec<u16>> = (0..world)
            .map(|rank| finite_pattern(rank, elements, 0xfeed_0000 + elements as u64))
            .collect();
        let expected = support::reference_all_gather(&locals);
        let results = run_ranks(world, |rank| {
            let transport = cluster
                .handle(rank)
                .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
            let mut out = vec![0u16; world * elements];
            transport
                .all_gather_bf16(&locals[rank], &mut out, 1)
                .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
            out
        });
        for (rank, out) in results.iter().enumerate() {
            assert_bits_eq(
                out,
                &expected,
                &format!("all-gather elements {elements} rank {rank}"),
            );
        }
    }
}
