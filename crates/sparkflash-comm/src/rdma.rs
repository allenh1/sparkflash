//! Skeleton of the real dual-rail ibverbs / GPUNetIO backend.
//!
//! Milestone 3 runs on a box with no RDMA hardware and no permission to touch the live
//! DeepSeek-V4.1-Flash cluster, so this module exists to fix the shape of the eventual backend
//! and to fail loudly. It compiles only with --features rdma, and every entry point returns
//! CommError::RdmaNotBuilt naming exactly what is missing.

use crate::error::CommError;
use crate::fabric::{
    FabricView, RailOverride, RailSelection, SPEC_RAIL0_INTERFACE, SPEC_RAIL1_INTERFACE,
    SYSFS_INFINIBAND_ROOT,
};
use crate::layout::CollectiveLayout;
use crate::transport::Transport;

/// The device names the specification declares for the two rails.
///
/// Documentation and preference only: on the live nodes these f0 functions are DOWN at 40 Gb/s,
/// so registration must go through RailSelection, which validates ACTIVE state and the 200 Gb/s
/// rate before any device is opened.
pub const RDMA_RAIL_INTERFACES: [&str; 2] = [SPEC_RAIL0_INTERFACE, SPEC_RAIL1_INTERFACE];

/// Reads the live sysfs fabric and resolves the rails this host should register.
///
/// The rail override environment variable, when set, pins the pair; the pinned devices are still
/// validated for ACTIVE state and the 200 Gb/s minimum.
///
/// # Errors
/// Returns CommError::FabricUnavailable when the fabric cannot be enumerated, when a pinned
/// device is missing, DOWN or too slow, or when fewer than two usable rails on distinct cards
/// exist.
pub fn select_rails_from_live_fabric() -> Result<RailSelection, CommError> {
    let view = FabricView::from_sysfs(SYSFS_INFINIBAND_ROOT)?;
    let pinned = RailOverride::from_env()?;
    RailSelection::resolve(&view, pinned.as_ref())
}

/// Why the skeleton cannot serve a collective on this host.
pub const RDMA_NOT_BUILT: &str = "the libibverbs/GPUNetIO backend is a Milestone 3 skeleton: build with \
--features rdma on a DGX Spark host that exposes the ACTIVE 200 Gb/s rails and link libibverbs";

/// Placeholder for the eventual dual-rail RoCEnante transport.
pub struct RdmaTransport {
    rank: usize,
    world_size: usize,
    reason: String,
}

impl RdmaTransport {
    /// Would register the 2 MiB one-shot slots on both rails and program the hairpin.
    ///
    /// The rails are resolved and validated first, so a host whose rails are DOWN or slower than
    /// 200 Gb/s fails with CommError::FabricUnavailable instead of quietly registering a slow
    /// device.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange for a rank outside the collective,
    /// CommError::FabricUnavailable when the fabric cannot supply two usable rails, and
    /// CommError::RdmaNotBuilt on this host because the ibverbs backend is still a skeleton.
    pub fn open(world_size: usize, rank: usize) -> Result<Self, CommError> {
        if world_size == 0 || rank >= world_size {
            return Err(CommError::RankOutOfRange { rank, world_size });
        }
        let selection = select_rails_from_live_fabric()?;
        Err(CommError::RdmaNotBuilt(format!(
            "{RDMA_NOT_BUILT}; rails resolved as {selection}"
        )))
    }

    /// Whether a real backend is compiled and usable.
    #[must_use]
    pub fn is_built() -> bool {
        false
    }
}

impl Transport for RdmaTransport {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_bf16(
        &self,
        _buf: &mut [u16],
        _seq: u64,
        _layout: &CollectiveLayout,
    ) -> Result<(), CommError> {
        Err(CommError::RdmaNotBuilt(self.reason.clone()))
    }

    fn all_gather_bf16(
        &self,
        _local: &[u16],
        _out: &mut [u16],
        _seq: u64,
    ) -> Result<(), CommError> {
        Err(CommError::RdmaNotBuilt(self.reason.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_point_reports_not_built() {
        assert!(!RdmaTransport::is_built());
        // A host without the live fabric reports the fabric problem; a host with the rails
        // present reports the missing backend. Either way open() must fail loudly.
        assert!(
            matches!(
                RdmaTransport::open(4, 0),
                Err(CommError::RdmaNotBuilt(_) | CommError::FabricUnavailable(_))
            ),
            "open() must fail on a host without the ibverbs backend"
        );
        assert!(matches!(
            RdmaTransport::open(4, 4),
            Err(CommError::RankOutOfRange { .. })
        ));
        let skeleton = RdmaTransport {
            rank: 1,
            world_size: 4,
            reason: RDMA_NOT_BUILT.to_owned(),
        };
        assert_eq!(skeleton.rank(), 1);
        assert_eq!(skeleton.world_size(), 4);
        let layout = CollectiveLayout::one_shot(4, 8).unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(
            skeleton.all_reduce_bf16(&mut [0u16; 8], 1, &layout),
            Err(CommError::RdmaNotBuilt(_))
        ));
        assert!(matches!(
            skeleton.all_gather_bf16(&[0u16; 2], &mut [0u16; 8], 1),
            Err(CommError::RdmaNotBuilt(_))
        ));
    }
}
