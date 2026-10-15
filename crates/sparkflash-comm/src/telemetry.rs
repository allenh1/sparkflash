//! RoCEnante / NCCL dispatch counters.
//!
//! The counters are relaxed atomics so a decode step never contends on them; a snapshot is
//! taken at a step or request boundary and is what /v1/cluster/stats reports.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::fabric::{FabricTelemetrySnapshot, RailSelection};

/// Collective dispatch counters for one communicator.
///
/// Field names are part of the specification (section 5.4) and are consumed by the API layer's
/// `RoceTransportStats`, so never rename them.
#[derive(Debug, Default)]
pub struct CommTelemetry {
    /// RoCEnante one-shot all-reduce calls (`num_bytes <= ROCENANTE_MAX_BYTES`).
    pub rocenante_allreduce_calls: AtomicU64,
    /// RoCEnante one-shot compact all-gather calls.
    pub rocenante_allgather_calls: AtomicU64,
    /// Patched-NCCL bidirectional-ring calls (payloads above the one-shot threshold).
    pub nccl_collective_calls: AtomicU64,
    /// Bytes posted on rail 0, `rocep1s0f0` (PCIe Gen5 x4 root 0).
    pub rdma_bytes_rail0: AtomicU64,
    /// Bytes posted on rail 1, `roceP2p1s0f0` (PCIe Gen5 x4 root 2).
    pub rdma_bytes_rail1: AtomicU64,
    /// Collective calls that overlapped the 6 MiB L2 weight prefetch with the wait.
    pub l2_prefetch_overlapped_calls: AtomicU64,
    /// Selected dual-rail devices, installed at startup by set_fabric.
    fabric: RwLock<Option<FabricTelemetrySnapshot>>,
}

/// Point-in-time copy of [`CommTelemetry`], safe to serialize into an HTTP response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommTelemetrySnapshot {
    /// RoCEnante one-shot all-reduce calls.
    pub rocenante_allreduce_calls: u64,
    /// RoCEnante one-shot compact all-gather calls.
    pub rocenante_allgather_calls: u64,
    /// Patched-NCCL bidirectional-ring calls.
    pub nccl_collective_calls: u64,
    /// Bytes posted on rail 0, `rocep1s0f0`.
    pub rdma_bytes_rail0: u64,
    /// Bytes posted on rail 1, `roceP2p1s0f0`.
    pub rdma_bytes_rail1: u64,
    /// Collective calls overlapped with the 6 MiB L2 weight prefetch.
    pub l2_prefetch_overlapped_calls: u64,
}

impl CommTelemetrySnapshot {
    /// Total bytes the dispatcher accounted across both rails.
    #[must_use]
    pub fn total_rdma_bytes(&self) -> u64 {
        self.rdma_bytes_rail0 + self.rdma_bytes_rail1
    }
}

impl CommTelemetry {
    /// Reads every counter once, relaxed.
    #[must_use]
    pub fn snapshot(&self) -> CommTelemetrySnapshot {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        CommTelemetrySnapshot {
            rocenante_allreduce_calls: load(&self.rocenante_allreduce_calls),
            rocenante_allgather_calls: load(&self.rocenante_allgather_calls),
            nccl_collective_calls: load(&self.nccl_collective_calls),
            rdma_bytes_rail0: load(&self.rdma_bytes_rail0),
            rdma_bytes_rail1: load(&self.rdma_bytes_rail1),
            l2_prefetch_overlapped_calls: load(&self.l2_prefetch_overlapped_calls),
        }
    }

    /// Installs the rails chosen at startup so their state, rate and speed are reported beside
    /// the counters.
    ///
    /// CommTelemetrySnapshot keeps exactly the specification's counter fields, so the device
    /// report is read back through fabric_snapshot instead of added to that struct.
    pub fn set_fabric(&self, selection: &RailSelection) {
        *self.fabric.write() = Some(FabricTelemetrySnapshot::from_selection(selection));
    }

    /// Forgets an installed rail report.
    pub fn clear_fabric(&self) {
        *self.fabric.write() = None;
    }

    /// Per-rail device state, rate and speed, when a selection has been installed.
    #[must_use]
    pub fn fabric_snapshot(&self) -> Option<FabricTelemetrySnapshot> {
        self.fabric.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_untouched_communicator_snapshots_to_zero() {
        let snapshot = CommTelemetry::default().snapshot();
        assert_eq!(
            snapshot,
            CommTelemetrySnapshot {
                rocenante_allreduce_calls: 0,
                rocenante_allgather_calls: 0,
                nccl_collective_calls: 0,
                rdma_bytes_rail0: 0,
                rdma_bytes_rail1: 0,
                l2_prefetch_overlapped_calls: 0,
            }
        );
        assert_eq!(snapshot.total_rdma_bytes(), 0);
    }

    #[test]
    fn snapshot_serializes_with_the_spec_field_names() {
        let telemetry = CommTelemetry::default();
        telemetry
            .rocenante_allreduce_calls
            .store(7, Ordering::Relaxed);
        telemetry.rdma_bytes_rail0.store(1024, Ordering::Relaxed);
        let json = serde_json::to_value(telemetry.snapshot())
            .unwrap_or_else(|error| panic!("snapshot must serialize: {error}"));
        for key in [
            "rocenante_allreduce_calls",
            "rocenante_allgather_calls",
            "nccl_collective_calls",
            "rdma_bytes_rail0",
            "rdma_bytes_rail1",
            "l2_prefetch_overlapped_calls",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(json["rocenante_allreduce_calls"], 7);
        assert_eq!(json["rdma_bytes_rail0"], 1024);
    }

    #[test]
    fn fabric_snapshot_reports_the_rails_without_touching_the_counters() {
        use crate::fabric::{
            DeviceState, FabricView, RailSelection, RailSelectionSource, RdmaDevice,
        };

        let view = FabricView::new(vec![
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "roceP2p1s0f1".to_owned(),
                netdev: "enP2p1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
        ]);
        let selection = RailSelection::resolve(&view, None)
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        let telemetry = CommTelemetry::default();
        assert!(telemetry.fabric_snapshot().is_none());
        telemetry.set_fabric(&selection);
        let fabric = telemetry
            .fabric_snapshot()
            .unwrap_or_else(|| panic!("the fabric report must be installed"));
        assert_eq!(fabric.source, RailSelectionSource::Discovered);
        assert_eq!(fabric.rails[0].device, "rocep1s0f1");
        assert_eq!(fabric.rails[0].state, DeviceState::Active);
        assert_eq!(fabric.rails[0].rate_gbps, 200);
        assert_eq!(fabric.rails[1].netdev, "enP2p1s0f1np1");
        assert!(fabric.rails[0].deviates_from_spec);

        // The counters keep the specification's shape and are untouched by the device report.
        assert_eq!(telemetry.snapshot().rdma_bytes_rail0, 0);
        let json = serde_json::to_value(&fabric)
            .unwrap_or_else(|error| panic!("fabric must serialize: {error}"));
        assert_eq!(json["rails"][0]["state"], "Active");
        assert_eq!(json["rails"][0]["rate_gbps"], 200);
        assert_eq!(json["rails"][0]["deviates_from_spec"], true);
        telemetry.clear_fabric();
        assert!(telemetry.fabric_snapshot().is_none());
    }
}
