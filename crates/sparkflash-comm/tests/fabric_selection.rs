//! Public-surface checks for the startup rail selection: an operator must see a loud failure
//! instead of a silently DOWN or 40 Gb/s rail, and the selected devices must be visible in the
//! collective telemetry.

use sparkflash_comm::{
    DeviceState, DualPathCommunicator, FabricView, MIN_RAIL_RATE_GBPS, RAIL_OVERRIDE_ENV,
    RailOverride, RailSelection, RailSelectionSource, RdmaDevice, SPEC_RAIL0_INTERFACE,
    SPEC_RAIL1_INTERFACE, SparkFabricMode,
};

/// The four RDMA functions the live four-node cluster reports (read-only observation).
fn live_like_view() -> FabricView {
    FabricView::new(vec![
        RdmaDevice {
            name: "rocep1s0f0".to_owned(),
            netdev: "enp1s0f0np0".to_owned(),
            state: DeviceState::Down,
            rate_gbps: 40,
        },
        RdmaDevice {
            name: "rocep1s0f1".to_owned(),
            netdev: "enp1s0f1np1".to_owned(),
            state: DeviceState::Active,
            rate_gbps: 200,
        },
        RdmaDevice {
            name: "roceP2p1s0f0".to_owned(),
            netdev: "enP2p1s0f0np0".to_owned(),
            state: DeviceState::Down,
            rate_gbps: 40,
        },
        RdmaDevice {
            name: "roceP2p1s0f1".to_owned(),
            netdev: "enP2p1s0f1np1".to_owned(),
            state: DeviceState::Active,
            rate_gbps: 200,
        },
    ])
}

#[test]
fn the_active_f1_rails_are_selected_and_flagged_as_a_spec_deviation() {
    let selection = RailSelection::resolve(&live_like_view(), None)
        .unwrap_or_else(|error| panic!("resolution failed: {error}"));
    assert_eq!(selection.source, RailSelectionSource::Discovered);
    assert_eq!(selection.names(), ["rocep1s0f1", "roceP2p1s0f1"]);
    assert!(selection.deviates_from_spec());
    assert_eq!(selection.rails[0].spec_device, SPEC_RAIL0_INTERFACE);
    assert_eq!(selection.rails[1].spec_device, SPEC_RAIL1_INTERFACE);
    assert_eq!(selection.rails[0].rate_gbps(), MIN_RAIL_RATE_GBPS);
    assert_eq!(selection.rails[0].state(), DeviceState::Active);
    assert_eq!(selection.rails[1].netdev(), "enP2p1s0f1np1");
    assert!(selection.summary().contains("rocep1s0f1"));
    assert!(selection.summary().contains("DEVIATES"));
    assert_eq!(RAIL_OVERRIDE_ENV, "SPARKFLASH_RAIL_DEVICES");
}

#[test]
fn a_spec_named_rail_that_is_down_is_a_loud_startup_failure() {
    let pinned = RailOverride::new(SPEC_RAIL0_INTERFACE, "roceP2p1s0f1");
    match RailSelection::resolve(&live_like_view(), Some(&pinned)) {
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains(SPEC_RAIL0_INTERFACE), "{text}");
            assert!(text.contains("DOWN"), "{text}");
            assert!(text.contains("40Gb/s"), "{text}");
        }
        Ok(selection) => panic!("a DOWN rail must never be selected: {selection:?}"),
    }
    // A host whose only rails are DOWN has no valid pair at all.
    let dead = FabricView::new(vec![RdmaDevice {
        name: "rocep1s0f0".to_owned(),
        netdev: "enp1s0f0np0".to_owned(),
        state: DeviceState::Down,
        rate_gbps: 40,
    }]);
    assert!(RailSelection::resolve(&dead, None).is_err());
}

#[test]
fn the_selected_rails_travel_with_the_collective_telemetry() {
    let selection = RailSelection::resolve(&live_like_view(), None)
        .unwrap_or_else(|error| panic!("resolution failed: {error}"));
    let communicator = DualPathCommunicator::new(0, 4, SparkFabricMode::Switched200G);
    communicator.attach_fabric(&selection);
    assert!(
        communicator
            .dispatch_all_reduce_bf16(0, 1024, false)
            .is_rocenante()
    );

    let fabric = communicator
        .telemetry
        .fabric_snapshot()
        .unwrap_or_else(|| panic!("the rails must be visible in telemetry"));
    assert_eq!(fabric.rails[0].device, "rocep1s0f1");
    assert_eq!(fabric.rails[0].state, DeviceState::Active);
    assert_eq!(fabric.rails[0].rate_gbps, 200);
    assert_eq!(fabric.rails[1].device, "roceP2p1s0f1");
    assert!(fabric.rails[0].deviates_from_spec);

    // The counter snapshot keeps exactly the specification's field names.
    let counters = communicator.telemetry_snapshot();
    assert_eq!(counters.rocenante_allreduce_calls, 1);
    assert_eq!(counters.rdma_bytes_rail0, 1536);
    let json = serde_json::to_value(counters)
        .unwrap_or_else(|error| panic!("counters must serialize: {error}"));
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap_or_else(|| panic!("the counter snapshot must be a JSON object"))
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "l2_prefetch_overlapped_calls",
            "nccl_collective_calls",
            "rdma_bytes_rail0",
            "rdma_bytes_rail1",
            "rocenante_allgather_calls",
            "rocenante_allreduce_calls",
        ]
    );

    // The rail report serializes the state, rate and deviation for the cluster stats endpoint.
    let json = serde_json::to_value(&fabric)
        .unwrap_or_else(|error| panic!("fabric must serialize: {error}"));
    assert_eq!(json["source"], "Discovered");
    assert_eq!(json["rails"][0]["state"], "Active");
    assert_eq!(json["rails"][0]["rate_gbps"], 200);
    assert_eq!(json["rails"][0]["netdev"], "enp1s0f1np1");
    assert_eq!(json["rails"][0]["spec_device"], SPEC_RAIL0_INTERFACE);
    assert_eq!(json["rails"][0]["deviates_from_spec"], true);
    assert_eq!(json["rails"][1]["device"], "roceP2p1s0f1");
}

/// A device that is ACTIVE at 200 Gb/s.
fn active(name: &str, netdev: &str) -> RdmaDevice {
    RdmaDevice {
        name: name.to_owned(),
        netdev: netdev.to_owned(),
        state: DeviceState::Active,
        rate_gbps: 200,
    }
}

#[test]
fn an_override_that_collapses_the_two_rails_is_refused() {
    // The same device twice is the classic operator typo: it must not become a one-rail
    // collective, because that is exactly the silent bandwidth downgrade D12 forbids.
    match RailSelection::resolve(
        &live_like_view(),
        Some(&RailOverride::new("rocep1s0f1", "rocep1s0f1")),
    ) {
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("same device twice"), "{text}");
            assert!(text.contains("rocep1s0f1"), "{text}");
            assert!(text.contains("Gb/s"), "{text}");
        }
        Ok(selection) => panic!("a repeated rail must be refused: {selection:?}"),
    }

    // Two functions of one card share a PCIe root, so they are not two independent rails.
    let one_card = FabricView::new(vec![
        active("rocep1s0f1", "enp1s0f1np1"),
        active("rocep1s0f2", "enp1s0f2np2"),
        active("roceP2p1s0f1", "enP2p1s0f1np1"),
    ]);
    match RailSelection::resolve(
        &one_card,
        Some(&RailOverride::new("rocep1s0f1", "rocep1s0f2")),
    ) {
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("two functions of the same card"), "{text}");
            assert!(text.contains("rocep1s0f2"), "{text}");
        }
        Ok(selection) => panic!("two functions of one card must be refused: {selection:?}"),
    }

    // The same view still resolves a genuine two-card pin, so the rule is not over-broad.
    let ok = RailSelection::resolve(
        &one_card,
        Some(&RailOverride::new("rocep1s0f2", "roceP2p1s0f1")),
    )
    .unwrap_or_else(|error| panic!("a two-card pin must still resolve: {error}"));
    assert_eq!(ok.names(), ["rocep1s0f2", "roceP2p1s0f1"]);
    assert_eq!(ok.source, RailSelectionSource::Override);
}
