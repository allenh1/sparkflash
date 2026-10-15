//! Loopback-transport behaviour: measured rail traffic, exchange counting, failure reporting
//! and a repeated-run stress that would expose any arrival-order dependence.

mod support;

use std::time::Duration;

use support::{
    Lcg, assert_bits_eq, finite_pattern, reference_all_reduce, run_ranks, run_ranks_order,
};

use sparkflash_comm::{
    CollectiveLayout, CommError, DualPathCommunicator, FabricSchedule, LoopbackCluster,
    LoopbackConfig, SparkFabricMode, Transport,
};

fn cluster(world: usize, timeout: Duration) -> LoopbackCluster {
    LoopbackCluster::new(LoopbackConfig {
        world_size: world,
        timeout,
        send_delay: Duration::ZERO,
        ..LoopbackConfig::default()
    })
    .unwrap_or_else(|error| panic!("loopback cluster: {error}"))
}

#[test]
fn one_shot_rail_ledger_matches_the_dispatch_telemetry() {
    let world = 4;
    let elements = 512;
    let num_bytes = elements * 2;
    let cluster = cluster(world, support::CLUSTER_TIMEOUT);
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| finite_pattern(rank, elements, 0x1ead))
        .collect();
    let layout = CollectiveLayout::new(world, elements, FabricSchedule::OneShotAllToAll)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    // Every rank dispatches the same collective through its own communicator, exactly as the
    // model forward pass would.
    let telemetry = run_ranks(world, |rank| {
        let communicator = DualPathCommunicator::new(rank, world, SparkFabricMode::Switched200G);
        let route = communicator.dispatch_all_reduce_bf16(7, num_bytes, true);
        assert!(route.is_rocenante());
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
        let mut buf = payloads[rank].clone();
        transport
            .all_reduce_bf16(&mut buf, 1, &layout)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        communicator.telemetry_snapshot()
    });

    let ledger = cluster.rail_ledger();
    // Every rank posted to its three peers, striped half per rail, so one rank accounts
    // 3 * num_bytes / 2 bytes on each rail and the cluster moves world_size times that.
    let per_rank_rail = 3 * (num_bytes / 2) as u64;
    for (rank, snapshot) in telemetry.iter().enumerate() {
        assert_eq!(
            snapshot.rdma_bytes_rail0, per_rank_rail,
            "rank {rank} rail 0"
        );
        assert_eq!(
            snapshot.rdma_bytes_rail1, per_rank_rail,
            "rank {rank} rail 1"
        );
        assert_eq!(snapshot.rocenante_allreduce_calls, 1, "rank {rank} calls");
        assert_eq!(
            snapshot.l2_prefetch_overlapped_calls, 1,
            "rank {rank} prefetch"
        );
    }
    assert_eq!(ledger.rail0_bytes, world as u64 * per_rank_rail);
    assert_eq!(ledger.rail1_bytes, world as u64 * per_rank_rail);
    assert_eq!(ledger.total_bytes(), 2 * world as u64 * per_rank_rail);
}

#[test]
fn hairpin_moves_exactly_half_the_one_shot_bytes() {
    let world = 4;
    let elements = 1024;
    for schedule in [
        FabricSchedule::OneShotAllToAll,
        FabricSchedule::HairpinRingReduceScatterAllGather,
    ] {
        let cluster = cluster(world, support::CLUSTER_TIMEOUT);
        let payloads: Vec<Vec<u16>> = (0..world)
            .map(|rank| finite_pattern(rank, elements, 0x2ead))
            .collect();
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
        });
        let ledger = cluster.rail_ledger();
        let expected_bytes = match schedule {
            FabricSchedule::OneShotAllToAll => world * (world - 1) * elements * 2,
            FabricSchedule::HairpinRingReduceScatterAllGather => 2 * (world - 1) * elements * 2,
        } as u64;
        assert_eq!(
            ledger.total_bytes(),
            expected_bytes,
            "{schedule:?} measured traffic"
        );
        // The hairpin moves half the bytes of the one-shot; both must land in the same place.
        assert_eq!(
            ledger.total_bytes(),
            match schedule {
                FabricSchedule::OneShotAllToAll => 12 * elements as u64 * 2,
                FabricSchedule::HairpinRingReduceScatterAllGather => 6 * elements as u64 * 2,
            },
            "{schedule:?} closed form"
        );
        // One doorbell per exchange phase per rank: 1 phase for the one-shot, 2 for the
        // hairpin. Counted after every rank has returned, so the totals are deterministic.
        let phases = match schedule {
            FabricSchedule::OneShotAllToAll => 1,
            FabricSchedule::HairpinRingReduceScatterAllGather => 2,
        };
        assert_eq!(cluster.doorbells(), (world * phases) as u64);
        assert_eq!(cluster.collectives(), world as u64);
    }
}

#[test]
fn a_dead_rank_fails_the_survivors_immediately() {
    let world = 4;
    let elements = 16;
    let cluster = cluster(world, Duration::from_millis(50));
    cluster
        .kill(3)
        .unwrap_or_else(|error| panic!("kill: {error}"));
    let layout = CollectiveLayout::new(world, elements, FabricSchedule::OneShotAllToAll)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    let outcomes = run_ranks_order(&[0, 1, 2], |rank| {
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
        let mut buf = vec![0u16; elements];
        transport.all_reduce_bf16(&mut buf, 1, &layout)
    });
    for (rank, outcome) in outcomes {
        assert!(
            matches!(outcome, Err(CommError::PeerClosed { peer: 3 })),
            "rank {rank} should have reported the dead peer 3, got {outcome:?}"
        );
    }
}

#[test]
fn a_missing_rank_hits_the_configured_deadline() {
    let world = 4;
    let cluster = cluster(world, Duration::from_millis(60));
    let layout = CollectiveLayout::new(world, 8, FabricSchedule::OneShotAllToAll)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    let outcomes = run_ranks_order(&[0, 1], |rank| {
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
        let mut buf = vec![0u16; 8];
        transport.all_reduce_bf16(&mut buf, 1, &layout)
    });
    for (rank, outcome) in outcomes {
        match outcome {
            Err(CommError::Timeout { timeout_ms, .. }) => assert_eq!(timeout_ms, 60),
            other => panic!("rank {rank} should have timed out, got {other:?}"),
        }
    }
}

#[test]
fn ranks_report_their_own_identity() {
    let world = 3;
    let cluster = cluster(world, support::CLUSTER_TIMEOUT);
    let identities = run_ranks(world, |rank| {
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
        (transport.rank(), transport.world_size())
    });
    for (rank, identity) in identities.iter().enumerate() {
        assert_eq!(*identity, (rank, world));
    }
    assert_eq!(cluster.world_size(), world);
}

#[test]
fn repeated_collectives_stay_bit_identical_under_random_delays() {
    let world = 4;
    let elements = 128;
    let mut lcg = Lcg::new(0x51ab_c0de);
    for iteration in 0..40u64 {
        let delay = Duration::from_micros(lcg.next() % 400);
        let schedule = if iteration % 2 == 0 {
            FabricSchedule::OneShotAllToAll
        } else {
            FabricSchedule::HairpinRingReduceScatterAllGather
        };
        // A fresh cluster per iteration, with this iteration's arrival stagger.
        let cluster = LoopbackCluster::new(LoopbackConfig {
            world_size: world,
            timeout: support::CLUSTER_TIMEOUT,
            send_delay: delay,
            ..LoopbackConfig::default()
        })
        .unwrap_or_else(|error| panic!("loopback cluster: {error}"));
        let payloads: Vec<Vec<u16>> = (0..world)
            .map(|rank| finite_pattern(rank, elements, 0x900d + iteration))
            .collect();
        let expected = reference_all_reduce(&payloads);
        let layout = CollectiveLayout::new(world, elements, schedule)
            .unwrap_or_else(|error| panic!("layout: {error}"));
        let results = run_ranks(world, |rank| {
            let transport = cluster
                .handle(rank)
                .unwrap_or_else(|error| panic!("handle {rank}: {error}"));
            let mut buf = payloads[rank].clone();
            transport
                .all_reduce_bf16(&mut buf, iteration + 1, &layout)
                .unwrap_or_else(|error| panic!("rank {rank} iteration {iteration}: {error}"));
            buf
        });
        for (rank, buf) in results.iter().enumerate() {
            assert_bits_eq(
                buf,
                &expected,
                &format!("iteration {iteration} rank {rank} {schedule:?}"),
            );
        }
    }
}
