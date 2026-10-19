//! SparkFlash-RS RoCEnante one-shot RDMA + patched-NCCL dual-path collective transport.
//!
//! Two collective shapes exist for four DGX Spark ranks:
//!
//! * payloads of at most ROCENANTE_MAX_BYTES (2 MiB) go through the RoCEnante one-shot path,
//!   whose reduction is bit-identical across ranks for every finite value and for both
//!   infinities, by construction; NaN is the one stated exclusion, and only when a rank's result
//!   is compared with the CPU reference rather than with another rank (see below);
//! * larger payloads (chunked prefill) go through the patched bidirectional NCCL ring.
//!
//! Bit-identical means every rank computes, for every element, the f32 accumulation
//! 0 + rank0 + rank1 + rank2 + rank3 in ascending rank order seeded with 0.0f, rounded exactly
//! once to bf16 with round-to-nearest-even. Arrival order, thread count and the block/grid stride
//! cannot change the result, because the accumulator is ordered by rank index rather than by
//! arrival time. The canonical CPU formulation lives in the reduce module, and every transport is
//! tested against it.
//!
//! The finite-value guarantee is the verified one and it is not conditional. The NaN exclusion has
//! two scopes and they differ. Rank against rank: every rank applies the identical deterministic
//! conversion in the identical canonical order, so all four agree even on NaN, on the Rust paths by
//! construction and on the device path by symmetry (unmeasurable on a host with no GPU, and stated
//! as such rather than claimed). Rank against the CPU reference: the Rust reference and every Rust
//! transport preserve a NaN bit-for-bit and agree with each other, but the CUDA host conversion routine
//! __float2bfloat16_rn returned 0x7fff for every NaN the M3 reviewer fed it, so the two sides are
//! not bit-identical on NaN payloads, and the device instruction's NaN behaviour cannot be
//! measured on this host. roce_reduce.rs records the same exclusion on the kernel side. An
//! inference reduction does not depend on a NaN payload.
//!
//! The transport abstraction is what makes all of this testable without RDMA hardware:
//! LoopbackTransport models the RoCEnante slot/doorbell handshake with real rank threads, and
//! TcpTransport runs the same protocol over real 127.0.0.1 sockets with the same framing,
//! partial reads, peer death and timeouts a NIC path would see. Both execute the identical
//! schedule driver and the identical canonical reduction.
//!
//! Dual-rail device identity is discovered, never hardcoded. The fabric module reads each RDMA
//! port's state and negotiated rate, keeps only ACTIVE links of at least 200 Gb/s, prefers the
//! spec's card order and reports any deviation. The spec names rocep1s0f0 and roceP2p1s0f0; on
//! the live four-node cluster those f0 functions are DOWN at 40 Gb/s, so the ACTIVE f1 functions
//! of the same two cards are selected (see DECISIONS.md D12). Startup must call
//! RailSelection::resolve and fail on Err, then hand the selection to
//! DualPathCommunicator::attach_fabric so the per-rail state, netdev and rate travel with the
//! collective telemetry.

use std::sync::atomic::{AtomicU64, Ordering};

use sparkflash_trace::{TraceScope, spark_trace_span};

pub mod bf16;
pub mod error;
mod exchange;
pub mod fabric;
pub mod layout;
pub mod oneshot;
pub mod probe;
pub mod rail;
#[cfg(feature = "rdma")]
pub mod rdma;
pub mod reduce;
pub mod tcp;
pub mod telemetry;
pub mod transport;

pub use bf16::{bf16_to_f32, f32_to_bf16_rn};
pub use error::CommError;
pub use fabric::{
    DeviceState, FabricTelemetrySnapshot, FabricView, MIN_RAIL_RATE_GBPS, RAIL_COUNT,
    RAIL_OVERRIDE_ENV, RailOverride, RailSelection, RailSelectionSource, RailTelemetrySnapshot,
    RdmaDevice, SPEC_RAIL0_INTERFACE, SPEC_RAIL1_INTERFACE, SYSFS_INFINIBAND_ROOT, SelectedRail,
    parse_rate_gbps, spec_device,
};
pub use layout::{
    CollectiveLayout, FabricLink, FabricSchedule, MAX_FRAME_ELEMENTS, REFERENCE_WORLD_SIZE,
    ScheduleTransfer, TransferPhase,
};
pub use oneshot::{
    ArrivalProxy, DeviceBuffer, Dispatch, DispatchLog, DispatchRecord, Exchange, LaunchPath,
    ONESHOT_ARRIVAL_BYTES, ONESHOT_KERNEL_NAME, ONESHOT_KIND, ONESHOT_PHASE, ONESHOT_RAILS,
    ONESHOT_RANKS, ONESHOT_ROW_BYTES, ONESHOT_ROW_ELEMENTS, ONESHOT_SLOT_ALIGN, ONESHOT_SLOT_BYTES,
    RailFrame, SeamError, SequenceCounter, SlotKind, SlotRegistration, arrival_offset, row_offset,
};
pub use rail::{RailAudit, RailLedger, RailPlan, RailSplit};
pub use reduce::{canonical_all_reduce_into, canonical_reduce_element};
pub use tcp::{
    DEFAULT_MAX_FRAME_BYTES, FRAME_HEADER_BYTES, FRAME_MAGIC, FRAME_VERSION, FrameHeader,
    FrameKind, TcpConfig, TcpTransport, decode_bf16, encode_bf16, read_frame, write_frame,
};
pub use telemetry::{CommTelemetry, CommTelemetrySnapshot};
pub use transport::{LoopbackCluster, LoopbackConfig, LoopbackTransport, Transport};

/// Payload size at or below which the RoCEnante one-shot path is used (spec section 5.4).
pub const ROCENANTE_MAX_BYTES: usize = 2 * 1024 * 1024;

/// Bytes of the next sub-layer's weights the L2 prefetch stream touches during the RDMA wait.
pub const L2_PREFETCH_BYTES: usize = 6 * 1024 * 1024;

/// Physical fabric the four ranks are cabled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparkFabricMode {
    /// 4x DGX Spark on a lossless 200GbE RoCEv2 switch.
    Switched200G,
    /// 4x DGX Spark in a switchless ring (0-1-2-3-0) with ConnectX-7 ASIC hairpin diagonals.
    SwitchlessRing4Hairpin,
}

impl SparkFabricMode {
    /// Exchange schedule this fabric runs.
    #[must_use]
    pub fn schedule(self) -> FabricSchedule {
        match self {
            Self::Switched200G => FabricSchedule::OneShotAllToAll,
            Self::SwitchlessRing4Hairpin => FabricSchedule::HairpinRingReduceScatterAllGather,
        }
    }

    /// Human readable fabric name used in telemetry payloads.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Switched200G => "Switched200G_DualRail_RoCEnante",
            Self::SwitchlessRing4Hairpin => "SwitchlessRing4Hairpin_RoCEnante",
        }
    }
}

/// Where the dispatcher sent one collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveRoute {
    /// RoCEnante one-shot path, with the sequence number and prefetch flag to carry into it.
    RoCEnanteOneShot {
        /// Monotonic step sequence allocated for this collective, starting at 1.
        seq: u64,
        /// Whether the 6 MiB L2 weight prefetch runs during the RDMA wait.
        l2_prefetch_6mb: bool,
    },
    /// Patched bidirectional dual-rail NCCL ring.
    NcclBidirectionalRing,
}

impl CollectiveRoute {
    /// Whether this route uses the RoCEnante one-shot path.
    #[must_use]
    pub fn is_rocenante(&self) -> bool {
        matches!(self, Self::RoCEnanteOneShot { .. })
    }
}

/// Routes collectives between RoCEnante and NCCL and accounts their cost.
pub struct DualPathCommunicator {
    /// This rank's index.
    pub rank: usize,
    /// Number of ranks in the collective.
    pub world_size: usize,
    /// Physical fabric the ranks are cabled on.
    pub fabric_mode: SparkFabricMode,
    /// Monotonic sequence allocator handed to the RoCEnante path.
    pub step_seq: AtomicU64,
    /// Per-route counters and rail byte accounting.
    pub telemetry: CommTelemetry,
}

impl DualPathCommunicator {
    /// Creates a communicator for one rank.
    ///
    /// rank must be below world_size; the operations that consume the result report
    /// RankOutOfRange otherwise, so a bad configuration fails loudly at the first collective
    /// instead of panicking here.
    #[must_use]
    pub fn new(rank: usize, world_size: usize, fabric_mode: SparkFabricMode) -> Self {
        Self {
            rank,
            world_size,
            fabric_mode,
            step_seq: AtomicU64::new(0),
            telemetry: CommTelemetry::default(),
        }
    }

    /// Routes an All-Reduce to the RoCEnante one-shot path at or below 2 MiB and to the patched
    /// bidirectional NCCL ring above it.
    ///
    /// The RoCEnante branch counts one collective call, accounts three half-buffer writes per
    /// rail (the one-shot posts to each of the three peers, striped half per rail, which is
    /// 3 * num_bytes / 2 per rail), records the L2 prefetch overlap when requested and allocates
    /// the sequence number with fetch_add(1, SeqCst) + 1. The NCCL branch counts a collective
    /// call and returns the ring route.
    ///
    /// The per-rail counter applies the specification's integer division literally, so an odd
    /// num_bytes would drop up to three bytes from the sum. Every collective this model
    /// dispatches carries bf16 elements, so num_bytes is always even and the truncation is
    /// unreachable in practice; it is kept (decision D13) because these counters are the
    /// conformance surface an operator compares against the paper, and it is pinned by the
    /// rail_bytes_follow_the_binding_formula_and_truncate_odd_byte_counts test. RailPlan is what
    /// conserves bytes exactly on the wire.
    pub fn dispatch_all_reduce_bf16(
        &self,
        layer_id: u32,
        num_bytes: usize,
        enable_l2_weight_prefetch: bool,
    ) -> CollectiveRoute {
        if num_bytes <= ROCENANTE_MAX_BYTES {
            spark_trace_span!(
                TraceScope::ROCENANTE_RDMA,
                "dispatch_rocenante_oneshot",
                layer_id
            );
            self.telemetry
                .rocenante_allreduce_calls
                .fetch_add(1, Ordering::Relaxed);
            // Specification formula 3 * (num_bytes / 2) per rail. Integer division drops up to
            // three bytes only for an odd byte count, which a bf16 payload never has (D13).
            let half_bytes = (num_bytes / 2) as u64;
            self.telemetry
                .rdma_bytes_rail0
                .fetch_add(half_bytes * 3, Ordering::Relaxed);
            self.telemetry
                .rdma_bytes_rail1
                .fetch_add(half_bytes * 3, Ordering::Relaxed);
            if enable_l2_weight_prefetch {
                self.telemetry
                    .l2_prefetch_overlapped_calls
                    .fetch_add(1, Ordering::Relaxed);
            }
            CollectiveRoute::RoCEnanteOneShot {
                seq: self.step_seq.fetch_add(1, Ordering::SeqCst) + 1,
                l2_prefetch_6mb: enable_l2_weight_prefetch,
            }
        } else {
            spark_trace_span!(TraceScope::NCCL_COLLECTIVE, "dispatch_nccl_ring", layer_id);
            self.telemetry
                .nccl_collective_calls
                .fetch_add(1, Ordering::Relaxed);
            CollectiveRoute::NcclBidirectionalRing
        }
    }

    /// Routes the compact all-gather sibling of the all-reduce.
    ///
    /// The RoCEnante one-shot compact gather covers at most 2 MiB with the same rail accounting
    /// as the all-reduce, including its integer division; see dispatch_all_reduce_bf16 for why
    /// that truncation is unreachable for bf16 payloads (decision D13). Larger gathers fall back
    /// to the NCCL ring.
    pub fn dispatch_all_gather_bf16(&self, layer_id: u32, num_bytes: usize) -> CollectiveRoute {
        if num_bytes <= ROCENANTE_MAX_BYTES {
            spark_trace_span!(
                TraceScope::ROCENANTE_RDMA,
                "dispatch_rocenante_allgather",
                layer_id
            );
            self.telemetry
                .rocenante_allgather_calls
                .fetch_add(1, Ordering::Relaxed);
            // Same specification formula and the same unreachable-for-bf16 truncation (D13).
            let half_bytes = (num_bytes / 2) as u64;
            self.telemetry
                .rdma_bytes_rail0
                .fetch_add(half_bytes * 3, Ordering::Relaxed);
            self.telemetry
                .rdma_bytes_rail1
                .fetch_add(half_bytes * 3, Ordering::Relaxed);
            CollectiveRoute::RoCEnanteOneShot {
                seq: self.step_seq.fetch_add(1, Ordering::SeqCst) + 1,
                l2_prefetch_6mb: false,
            }
        } else {
            spark_trace_span!(
                TraceScope::NCCL_COLLECTIVE,
                "dispatch_nccl_ring_allgather",
                layer_id
            );
            self.telemetry
                .nccl_collective_calls
                .fetch_add(1, Ordering::Relaxed);
            CollectiveRoute::NcclBidirectionalRing
        }
    }

    /// Builds the schedule layout this fabric uses for a buffer of num_elements bf16 elements.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange for a rank outside the collective and
    /// CommError::InvalidLayout for an element count that cannot fit a frame.
    pub fn layout_for(&self, num_elements: usize) -> Result<CollectiveLayout, CommError> {
        if self.rank >= self.world_size {
            return Err(CommError::RankOutOfRange {
                rank: self.rank,
                world_size: self.world_size,
            });
        }
        CollectiveLayout::new(self.world_size, num_elements, self.fabric_mode.schedule())
    }

    /// Reads the dispatch counters once, relaxed.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CommTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Records the rails chosen at startup so their state, rate and speed are reported beside
    /// the dispatch counters.
    ///
    /// The counter snapshot keeps exactly the specification's field names; the device report is
    /// read back with CommTelemetry::fabric_snapshot.
    pub fn attach_fabric(&self, selection: &RailSelection) {
        self.telemetry.set_fabric(selection);
    }
}
