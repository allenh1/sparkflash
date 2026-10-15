//! Gates for the physical four-node DGX Spark cluster. Every test here is #[ignore]d because it
//! needs the live RoCE fabric plus the Lead's explicit per-run permission. Run one with:
//!
//!   SPARK_COMM_HW_PERMISSION=1 cargo test -p sparkflash-comm --test hardware_gates -- --ignored --exact <name>
//!
//! Without SPARK_COMM_HW_PERMISSION=1 they panic immediately instead of touching the cluster, so
//! a mis-typed --ignored cannot quietly reach 192.0.2.80/.84/.68/.59.

/// Whether this run has the Lead's explicit permission to talk to the cluster.
fn hardware_permission() -> bool {
    std::env::var("SPARK_COMM_HW_PERMISSION").unwrap_or_default() == "1"
}

/// Fails loudly when the permission variable is missing.
fn require_permission(test: &str) {
    assert!(
        hardware_permission(),
        "refusing to run {test}: it needs the live 4x DGX Spark RoCE fabric (192.0.2.80/.84/.68/.59). \
Set SPARK_COMM_HW_PERMISSION=1 only with explicit per-run approval from the Lead."
    );
}

/// Fails loudly when the ibverbs backend is not compiled in, which is always the case today.
fn require_rdma_backend(test: &str) {
    #[cfg(feature = "rdma")]
    {
        match sparkflash_comm::rdma::RdmaTransport::open(4, 0) {
            Ok(_transport) => {}
            Err(error) => panic!("{test}: the ibverbs backend refused to open: {error}"),
        }
    }
    #[cfg(not(feature = "rdma"))]
    panic!(
        "{test}: this binary was built without --features rdma, so no ibverbs backend exists. \
Rebuild on a DGX Spark node with libibverbs and the two RoCE interfaces present."
    );
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn roce_oneshot_two_mib_all_reduce_meets_the_latency_budget() {
    require_permission("roce_oneshot_two_mib_all_reduce_meets_the_latency_budget");
    require_rdma_backend("roce_oneshot_two_mib_all_reduce_meets_the_latency_budget");
    panic!(
        "not implemented on this host: the gate needs 4 nodes, a 2 MiB payload, 10,000 timed \
replays and the RoCEnante board's own counters."
    );
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn roce_bits_match_the_cpu_reference_over_ten_thousand_replays() {
    require_permission("roce_bits_match_the_cpu_reference_over_ten_thousand_replays");
    require_rdma_backend("roce_bits_match_the_cpu_reference_over_ten_thousand_replays");
    panic!(
        "not implemented on this host: the gate compares real hardware output against reduce::canonical_all_reduce_into."
    );
}

#[test]
#[ignore = "needs the live RoCE fabric and explicit per-run permission; run with --ignored"]
fn dual_rail_counters_show_both_interfaces_carrying_half_the_bytes() {
    require_permission("dual_rail_counters_show_both_interfaces_carrying_half_the_bytes");
    require_rdma_backend("dual_rail_counters_show_both_interfaces_carrying_half_the_bytes");
    panic!(
        "not implemented on this host: the gate reads the two rail counters after one dispatch."
    );
}
