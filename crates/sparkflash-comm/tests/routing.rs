//! Dual-path dispatch semantics: the 2 MiB threshold, sequence allocation, rail accounting and
//! the telemetry counters the API layer reads.

use sparkflash_comm::{
    CollectiveRoute, CommError, DualPathCommunicator, FabricSchedule, L2_PREFETCH_BYTES,
    ROCENANTE_MAX_BYTES, SparkFabricMode,
};

fn communicator(rank: usize, mode: SparkFabricMode) -> DualPathCommunicator {
    DualPathCommunicator::new(rank, 4, mode)
}

#[test]
fn routes_by_the_two_mib_threshold() {
    let communicator = communicator(0, SparkFabricMode::Switched200G);
    for num_bytes in [
        0usize,
        1,
        4096,
        ROCENANTE_MAX_BYTES - 1,
        ROCENANTE_MAX_BYTES,
    ] {
        let route = communicator.dispatch_all_reduce_bf16(0, num_bytes, false);
        assert!(route.is_rocenante(), "{num_bytes} bytes must use RoCEnante");
    }
    for num_bytes in [
        ROCENANTE_MAX_BYTES + 1,
        2 * ROCENANTE_MAX_BYTES,
        64 * 1024 * 1024,
    ] {
        let route = communicator.dispatch_all_reduce_bf16(0, num_bytes, false);
        assert_eq!(
            route,
            CollectiveRoute::NcclBidirectionalRing,
            "{num_bytes} bytes must use the NCCL ring"
        );
    }
    // The threshold is a byte count, not an element count.
    assert_eq!(ROCENANTE_MAX_BYTES, 2 * 1024 * 1024);
    assert_eq!(L2_PREFETCH_BYTES, 6 * 1024 * 1024);
}

#[test]
fn sequence_numbers_are_monotonic_and_are_not_consumed_by_nccl() {
    let communicator = communicator(1, SparkFabricMode::Switched200G);
    let first = communicator.dispatch_all_reduce_bf16(0, 1024, false);
    let second = communicator.dispatch_all_reduce_bf16(1, 2048, true);
    let nccl = communicator.dispatch_all_reduce_bf16(2, ROCENANTE_MAX_BYTES + 1, false);
    let third = communicator.dispatch_all_reduce_bf16(3, 4096, false);
    let gather = communicator.dispatch_all_gather_bf16(4, 512);
    assert_eq!(nccl, CollectiveRoute::NcclBidirectionalRing);
    let seqs: Vec<u64> = [first, second, third, gather]
        .iter()
        .map(|route| match route {
            CollectiveRoute::RoCEnanteOneShot { seq, .. } => *seq,
            CollectiveRoute::NcclBidirectionalRing => {
                panic!("a RoCEnante branch returned the ring route")
            }
        })
        .collect();
    assert_eq!(seqs, vec![1, 2, 3, 4]);
    // The prefetch flag travels with the route.
    match second {
        CollectiveRoute::RoCEnanteOneShot {
            l2_prefetch_6mb, ..
        } => assert!(l2_prefetch_6mb),
        other => panic!("expected RoCEnante, got {other:?}"),
    }
}

#[test]
fn rail_bytes_follow_the_binding_formula_and_truncate_odd_byte_counts() {
    let communicator = communicator(2, SparkFabricMode::Switched200G);
    let prefetch = communicator.dispatch_all_reduce_bf16(0, 8192, true);
    assert!(prefetch.is_rocenante());
    let snapshot = communicator.telemetry_snapshot();
    // Each rank posts its half to three peers on each rail: 3 * num_bytes / 2.
    assert_eq!(snapshot.rdma_bytes_rail0, 3 * 4096);
    assert_eq!(snapshot.rdma_bytes_rail1, 3 * 4096);
    assert_eq!(snapshot.rocenante_allreduce_calls, 1);
    assert_eq!(snapshot.l2_prefetch_overlapped_calls, 1);
    assert_eq!(snapshot.nccl_collective_calls, 0);
    assert_eq!(snapshot.total_rdma_bytes(), 2 * 3 * 4096);

    // Odd byte counts keep the binding integer division, which drops the odd byte per peer.
    // 101 bytes over three peers is 150 bytes per rail instead of the exact 151.5.
    let odd = communicator.dispatch_all_reduce_bf16(1, 101, false);
    assert!(odd.is_rocenante());
    let snapshot = communicator.telemetry_snapshot();
    assert_eq!(snapshot.rdma_bytes_rail0, 3 * 4096 + 3 * 50);
    assert_eq!(snapshot.rdma_bytes_rail1, 3 * 4096 + 3 * 50);
    // The exact accounting would be 3 * 50.5 = 151.5 bytes per rail; the binding formula
    // truncates to 150.
    let exact_per_rail = 3.0f64 * 50.5;
    assert!((exact_per_rail - 151.5).abs() < f64::EPSILON);
    let accounted = snapshot.rdma_bytes_rail0 - 3 * 4096;
    assert_eq!(accounted, 150);
    assert!(accounted < exact_per_rail as u64);
}

#[test]
fn prefetch_overlap_is_counted_only_when_the_caller_asks_for_it() {
    let communicator = communicator(3, SparkFabricMode::Switched200G);
    for layer in 0..5u32 {
        communicator.dispatch_all_reduce_bf16(layer, 64 * 1024, layer % 2 == 0);
    }
    let snapshot = communicator.telemetry_snapshot();
    assert_eq!(snapshot.rocenante_allreduce_calls, 5);
    assert_eq!(snapshot.l2_prefetch_overlapped_calls, 3);
    // The NCCL branch never claims the prefetch overlap.
    communicator.dispatch_all_reduce_bf16(9, ROCENANTE_MAX_BYTES + 1, true);
    let snapshot = communicator.telemetry_snapshot();
    assert_eq!(snapshot.nccl_collective_calls, 1);
    assert_eq!(snapshot.l2_prefetch_overlapped_calls, 3);
}

#[test]
fn all_gather_uses_its_own_counter_and_never_claims_prefetch() {
    let communicator = communicator(0, SparkFabricMode::SwitchlessRing4Hairpin);
    let small = communicator.dispatch_all_gather_bf16(1, 2048);
    let large = communicator.dispatch_all_gather_bf16(2, ROCENANTE_MAX_BYTES + 1);
    assert!(small.is_rocenante());
    assert_eq!(large, CollectiveRoute::NcclBidirectionalRing);
    let snapshot = communicator.telemetry_snapshot();
    assert_eq!(snapshot.rocenante_allgather_calls, 1);
    assert_eq!(snapshot.rocenante_allreduce_calls, 0);
    assert_eq!(snapshot.nccl_collective_calls, 1);
    assert_eq!(snapshot.l2_prefetch_overlapped_calls, 0);
    assert_eq!(snapshot.rdma_bytes_rail0, 3 * 1024);
}

#[test]
fn layout_for_follows_the_fabric_mode() {
    let switched = communicator(0, SparkFabricMode::Switched200G);
    let layout = switched
        .layout_for(4096)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    assert_eq!(layout.schedule(), FabricSchedule::OneShotAllToAll);
    assert_eq!(layout.num_elements(), 4096);
    assert_eq!(layout.num_bytes(), 8192);
    assert_eq!(layout.world_size(), 4);
    assert_eq!(layout.transfers().len(), 12);

    let ring = communicator(0, SparkFabricMode::SwitchlessRing4Hairpin);
    let layout = ring
        .layout_for(4096)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    assert_eq!(
        layout.schedule(),
        FabricSchedule::HairpinRingReduceScatterAllGather
    );
    assert_eq!(layout.transfers().len(), 24);
    assert_eq!(layout.phases().len(), 2);
    assert_eq!(
        SparkFabricMode::Switched200G.schedule(),
        FabricSchedule::OneShotAllToAll
    );
    assert!(
        SparkFabricMode::SwitchlessRing4Hairpin
            .as_str()
            .contains("Hairpin")
    );
    assert!(
        SparkFabricMode::Switched200G
            .as_str()
            .contains("Switched200G")
    );
}

#[test]
fn layout_for_reports_a_rank_outside_the_collective() {
    let communicator = DualPathCommunicator::new(4, 4, SparkFabricMode::Switched200G);
    assert!(matches!(
        communicator.layout_for(16),
        Err(CommError::RankOutOfRange {
            rank: 4,
            world_size: 4
        })
    ));
}

#[test]
fn an_untouched_communicator_snapshots_to_zero() {
    let communicator = communicator(0, SparkFabricMode::Switched200G);
    let snapshot = communicator.telemetry_snapshot();
    assert_eq!(snapshot.rocenante_allreduce_calls, 0);
    assert_eq!(snapshot.rocenante_allgather_calls, 0);
    assert_eq!(snapshot.nccl_collective_calls, 0);
    assert_eq!(snapshot.rdma_bytes_rail0, 0);
    assert_eq!(snapshot.rdma_bytes_rail1, 0);
    assert_eq!(snapshot.l2_prefetch_overlapped_calls, 0);
    assert_eq!(snapshot.total_rdma_bytes(), 0);
}
