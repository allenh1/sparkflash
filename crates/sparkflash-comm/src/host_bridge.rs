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

//! The declared S1 host-bridge reduce: the one permitted PCIe copy path, typed and counted.
//!
//! S1's steady-state invariant is that the two per-layer collectives (the attention delta,
//! `all_reduce`, and the routed MoE delta, `all_reduce_moe`) are enqueued on the device and their
//! rows never come back to the host. This module is the single, declared exception to that rule,
//! for an S1 run whose node cannot start the device arm: a device->host copy of this rank's row, the
//! host reduce the lean loop already implements, and a host->device copy into the caller's output
//! span.
//!
//! The exception is deliberately not a path any code can drift into:
//!
//! * **Opt-in, twice.** The module is behind the off-by-default `s1-host-bridge` cargo feature
//!   ([`HOST_BRIDGE_FEATURE`]); a build without it refuses [`HostBridgeReduce::open`] with a typed
//!   [`NcclError::ArmUnavailable`]. And a run must construct this type explicitly - no default arm,
//!   no environment switch and no device-API fallback reaches it.
//! * **Typed.** [`HostBridgeReduce::label`] is `host-bridge` ([`HOST_BRIDGE_LABEL`]),
//!   [`HostBridgeReduce::arm`] is [`ReduceArm::HostBridge`], and the device collector refuses that
//!   arm by name in [`ReduceArm::require_available`], so a run labelled `host-bridge` can never
//!   have enqueued NCCL underneath it, and an NCCL run can never claim the host-bridge label.
//! * **Only the two layer collectives.** [`BridgeStage`] has exactly two variants and
//!   [`BridgeStage::parse`] refuses every other stage name; a seam or an observation point cannot
//!   ask for the bridge because there is no value to ask with.
//! * **Counted.** A completed bridge reduce moves the shared collective/element/byte counters and
//!   records `bridge_collectives`, `bridge_d2h_bytes` and `bridge_h2d_bytes` in
//!   [`DeviceReduceStats`]. A bridge that fails at any phase is not counted: the run is aborting and
//!   a partial bridge must not look like completed work.
//!
//! # The caller's contract
//!
//! * One [`HostBridgeReduce`] per run, opened before the first layer, exactly like the device
//!   communicator. [`HostBridgeReduce::open`] allocates the one host staging buffer; the reduce call
//!   allocates nothing, so a bridge that appears in the timed region pays only its two copies and
//!   the blocking host exchange.
//! * The [`BridgeIo`] implementation performs the two copies with the run's own CUDA handles. This
//!   crate has no CUDA driver dependency and does not grow one for the bridge; the caller already
//!   owns the device addresses and the stream order. Both copies must be complete (stream-ordered)
//!   before the reduce returns - the bridge is a synchronous exception, not an enqueued collective.
//! * Every rank passes the same `world`, `rank`, `count`, the same [`BridgeStage`] and the same
//!   `sequence` for the same collective, in the same order. A skipped or reordered bridge hangs the
//!   host transport exactly like a skipped device collective; the run's collective-count check
//!   catches the per-rank divergence before a number is reported.
//! * `input` and `output` are device addresses the caller guarantees are valid for `count` bf16
//!   values each on the current device, with `output` writable. They are out-of-place here: the
//!   bridge never assumes in-place aliasing.
//!
//! # What it is not
//!
//! It is not a performance path. The host reduce is the canonical rank-order f32-accumulate-then-
//! round path, so its arithmetic is the strict-mode contract (D54's tolerance is over-satisfied by
//! construction), and its cost is two PCIe copies plus the blocking exchange - exactly the
//! machinery the device port removes. Its only reason to exist is an S1 run that cannot start the
//! device arm; a run that has the device arm must not use it.

use std::mem::size_of;

use crate::device_reduce::{DeviceReduceStats, ReduceArm};
use crate::layout::{CollectiveLayout, MAX_FRAME_ELEMENTS};
use crate::nccl::NcclError;
use crate::transport::Transport;

/// The cargo feature that compiles the declared S1 host bridge in.
pub const HOST_BRIDGE_FEATURE: &str = "s1-host-bridge";

/// The stable row label: `reduce=host-bridge`.
pub const HOST_BRIDGE_LABEL: &str = "host-bridge";

/// The only stage values the bridge accepts.
///
/// Exactly the layer's two collectives: the attention delta and the routed MoE delta. The enum is
/// the permission - an observation point, a seam, the logits readback or the RDMA arm's stage has no
/// value here, so it cannot request the exception.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BridgeStage {
    /// `all_reduce`: the attention sublayer's delta.
    AttentionDelta,
    /// `all_reduce_moe`: the routed MoE delta.
    MoeDelta,
}

impl BridgeStage {
    /// Both variants, in layer order. Pinned by a test against [`crate::layer_device`]'s stage
    /// names so a rename cannot silently detach the bridge from the collectives it serves.
    pub const ALL: [Self; 2] = [Self::AttentionDelta, Self::MoeDelta];

    /// The layer stage name this bridge call stands in for.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttentionDelta => "all_reduce",
            Self::MoeDelta => "all_reduce_moe",
        }
    }

    /// Parses the stage name a runner passes on the command line.
    ///
    /// Every spelling that is not one of the two layer collectives is refused with a
    /// [`NcclError::Config`] that lists the permitted stages, so a wiring mistake is a named refusal
    /// at setup rather than a bridge that copies a row it should not have touched.
    ///
    /// # Errors
    /// Returns NcclError::Config for any other name.
    pub fn parse(name: &str) -> Result<Self, NcclError> {
        Self::ALL
            .into_iter()
            .find(|stage| stage.as_str() == name)
            .ok_or_else(|| {
                NcclError::Config(format!(
                    "host-bridge: stage {name:?} is not one of the two permitted layer collectives ({}); the host bridge is not a general copy path",
                    Self::ALL
                        .iter()
                        .map(|stage| stage.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

/// The caller's two copies, performed with the run's own CUDA handles.
///
/// The bridge owns the sequence and the counters; the caller owns the device and the copies. An
/// implementation must complete each copy before returning (the bridge is the declared synchronous
/// exception), and must not allocate per call beyond what its own copy needs.
pub trait BridgeIo {
    /// Copies `count` bf16 values from device address `device` into `host`.
    ///
    /// # Errors
    /// A copy failure, in the caller's own words; the bridge reports it as the `device-to-host`
    /// phase and does not count the collective.
    fn device_to_host(&mut self, device: u64, host: &mut [u16]) -> Result<(), String>;

    /// Copies `host` into device address `device`.
    ///
    /// # Errors
    /// A copy failure, in the caller's own words; the bridge reports it as the `host-to-device`
    /// phase and does not count the collective.
    fn host_to_device(&mut self, host: &[u16], device: u64) -> Result<(), String>;
}

/// The declared S1 host-bridge reduce: device->host, host reduce, host->device, counted.
///
/// Open it once per run (feature `s1-host-bridge`); every call blocks, because a host reduce is a
/// blocking exchange. The type is intentionally not `Clone` and not constructible from a device
/// communicator: one run, one bridge, one label.
pub struct HostBridgeReduce<'transport> {
    transport: &'transport dyn Transport,
    world: usize,
    rank: usize,
    layout: CollectiveLayout,
    /// The one host staging buffer, allocated at open. Both halves of every bridge use it, so the
    /// reduce call cannot allocate.
    staging: Vec<u16>,
    stats: DeviceReduceStats,
    count: usize,
}

impl<'transport> HostBridgeReduce<'transport> {
    /// Opens the run's host bridge and allocates its one staging buffer.
    ///
    /// # Errors
    /// Returns NcclError::ArmUnavailable when the `s1-host-bridge` feature is off (the bridge is
    /// not built into this binary), NcclError::Config for a bad world/rank/count, a count the host
    /// transport cannot carry in one frame (there is no silent chunking), or a transport whose
    /// world/rank is not the run's.
    pub fn open(
        transport: &'transport dyn Transport,
        world: usize,
        rank: usize,
        count: usize,
    ) -> Result<Self, NcclError> {
        if !cfg!(feature = "s1-host-bridge") {
            return Err(NcclError::ArmUnavailable {
                arm: HOST_BRIDGE_LABEL,
                detail: format!(
                    "this build has the {HOST_BRIDGE_FEATURE} feature off, so the declared S1 host-bridge fallback is not compiled in; a run that needs it must build with --features sparkflash-comm/{HOST_BRIDGE_FEATURE} and select it explicitly (the bridge is never a silent fallback)"
                ),
            });
        }
        validate_spec(world, rank)?;
        if count == 0 {
            return Err(NcclError::Config(
                "host-bridge: a bridge over zero bf16 elements is not a collective".to_owned(),
            ));
        }
        if count > MAX_FRAME_ELEMENTS {
            return Err(NcclError::Config(format!(
                "host-bridge: {count} bf16 elements exceed the host transport's one-frame limit of {MAX_FRAME_ELEMENTS}; the bridge does not chunk and does not fall back to the device arm"
            )));
        }
        if transport.world_size() != world {
            return Err(NcclError::Config(format!(
                "host-bridge: the transport is a {}-rank collective but the run's world is {world}",
                transport.world_size()
            )));
        }
        if transport.rank() != rank {
            return Err(NcclError::Config(format!(
                "host-bridge: the transport is rank {} but the run's rank is {rank}",
                transport.rank()
            )));
        }
        let layout = CollectiveLayout::one_shot(world, count)
            .map_err(|error| NcclError::Config(format!("host-bridge: {error}")))?;
        Ok(Self {
            transport,
            world,
            rank,
            layout,
            staging: vec![0u16; count],
            stats: DeviceReduceStats::default(),
            count,
        })
    }

    /// The arm label this handle is: `host-bridge`.
    #[must_use]
    pub const fn arm(&self) -> ReduceArm {
        ReduceArm::HostBridge
    }

    /// The stable row label: `reduce=host-bridge`.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        HOST_BRIDGE_LABEL
    }

    /// Ranks in the collective.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// bf16 elements per bridge call.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The counters as of this call: the shared collective ones plus the bridge copy account.
    #[must_use]
    pub const fn stats(&self) -> DeviceReduceStats {
        self.stats
    }

    /// Runs one declared bridge: D2H, host reduce, H2D, then count it.
    ///
    /// The bridge is synchronous by construction; it returns after the host reduce has returned and
    /// the H2D copy has completed in the caller's implementation. It does not allocate.
    ///
    /// # Errors
    /// Returns NcclError::HostBridge naming the failed phase (`device-to-host`, `host-reduce`,
    /// `host-to-device`) and the stage; a failed call is not counted, so the run cannot report a
    /// collective that did not complete. Returns NcclError::Config for a count that no longer
    /// matches the handle (the buffer was sized at open and is not resized).
    pub fn reduce(
        &mut self,
        io: &mut dyn BridgeIo,
        stage: BridgeStage,
        input: u64,
        output: u64,
        sequence: u64,
    ) -> Result<(), NcclError> {
        if self.staging.len() != self.count {
            return Err(NcclError::Config(format!(
                "host-bridge: the staging buffer holds {} elements but the handle was opened for {}",
                self.staging.len(),
                self.count
            )));
        }
        io.device_to_host(input, &mut self.staging)
            .map_err(|error| NcclError::HostBridge {
                phase: "device-to-host",
                detail: format!("{stage}: {error}", stage = stage.as_str()),
            })?;
        self.transport
            .all_reduce_bf16(&mut self.staging, sequence, &self.layout)
            .map_err(|error| NcclError::HostBridge {
                phase: "host-reduce",
                detail: format!("{stage}: {error}", stage = stage.as_str()),
            })?;
        io.host_to_device(&self.staging, output)
            .map_err(|error| NcclError::HostBridge {
                phase: "host-to-device",
                detail: format!("{stage}: {error}", stage = stage.as_str()),
            })?;
        // Counted only now: two completed copies and one completed host reduce.
        let bytes = (self.count as u64).saturating_mul(size_of::<u16>() as u64);
        self.stats.record_bridged(bytes);
        Ok(())
    }
}

impl std::fmt::Debug for HostBridgeReduce<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostBridgeReduce")
            .field("arm", &self.arm())
            .field("world", &self.world)
            .field("rank", &self.rank)
            .field("count", &self.count)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Checks a world/rank spec before any copy or exchange, identically to the device arm's check.
fn validate_spec(world: usize, rank: usize) -> Result<(), NcclError> {
    if world == 0 {
        return Err(NcclError::Config(
            "host-bridge: a bridge needs at least one rank".to_owned(),
        ));
    }
    if rank >= world {
        return Err(NcclError::Config(format!(
            "host-bridge: rank {rank} is out of range for world {world}"
        )));
    }
    Ok(())
}

/// Compiles the module even when the feature is off (the refusal path is production code).
const _: () = ();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_two_layer_collectives_are_representable() {
        assert_eq!(BridgeStage::ALL.len(), 2);
        let names: Vec<&'static str> = BridgeStage::ALL
            .iter()
            .map(|stage| stage.as_str())
            .collect();
        assert_eq!(names, vec!["all_reduce", "all_reduce_moe"]);
        for (stage, name) in BridgeStage::ALL.iter().zip(&names) {
            assert_eq!(BridgeStage::parse(name), Ok(*stage));
        }
    }

    #[test]
    fn the_stage_parser_refuses_every_other_name() {
        // The exception is the two collectives, not a copy primitive: a seam, an observation point or
        // the RDMA arm's stage has no bridge value and is refused by name.
        for name in [
            "logits",
            "residual_out",
            "attn_pre_mix_out",
            "moe_router",
            "all_reduce_rdma_arm",
            "all_reduce_moe ",
            "",
        ] {
            let error = BridgeStage::parse(name)
                .expect_err("only the two layer collectives may use the bridge");
            assert!(matches!(error, NcclError::Config(_)));
            let text = error.to_string();
            assert!(text.contains("all_reduce"), "{text}");
            assert!(text.contains("all_reduce_moe"), "{text}");
            assert!(text.contains("not a general copy path"), "{text}");
        }
    }

    #[test]
    fn the_label_is_the_declared_row_value() {
        assert_eq!(HOST_BRIDGE_LABEL, "host-bridge");
        assert_eq!(HOST_BRIDGE_FEATURE, "s1-host-bridge");
        assert_eq!(ReduceArm::HostBridge.as_str(), HOST_BRIDGE_LABEL);
    }

    #[cfg(not(feature = "s1-host-bridge"))]
    #[test]
    fn the_bridge_refuses_when_the_feature_is_off() {
        // What "opt-in" means in a default build: the path exists as a typed refusal, not as
        // silently-available machinery.
        assert!(!ReduceArm::HostBridge.is_built());
        let cluster = crate::transport::LoopbackCluster::new(crate::transport::LoopbackConfig {
            world_size: 1,
            ..crate::transport::LoopbackConfig::default()
        })
        .expect("one-rank loopback cluster");
        let transport = cluster.handle(0).expect("rank 0 transport");
        let error = HostBridgeReduce::open(&transport, 1, 0, 5_120)
            .expect_err("a default build must refuse the host bridge");
        match error {
            NcclError::ArmUnavailable { arm, detail } => {
                assert_eq!(arm, "host-bridge");
                assert!(detail.contains("s1-host-bridge"), "{detail}");
                assert!(detail.contains("never a silent fallback"), "{detail}");
            }
            other => panic!("expected ArmUnavailable, got {other:?}"),
        }
    }

    #[cfg(feature = "s1-host-bridge")]
    mod enabled {
        use super::*;
        use crate::error::CommError;
        use crate::layout::FabricSchedule;
        use crate::reduce::canonical_reduce_element;
        use crate::transport::{LoopbackCluster, LoopbackConfig};
        use std::time::Duration;

        /// The fake device addresses the tests' [`BridgeIo`] recognises.
        const DEVICE_INPUT: u64 = 0x1000;
        const DEVICE_OUTPUT: u64 = 0x2000;

        /// A device model backed by host vectors: the copies are real, the "device" is a Vec.
        struct FakeIo {
            input: Vec<u16>,
            output: Vec<u16>,
            fail_d2h: bool,
            fail_h2d: bool,
        }

        impl FakeIo {
            fn new(input: Vec<u16>, count: usize) -> Self {
                Self {
                    input,
                    output: vec![0u16; count],
                    fail_d2h: false,
                    fail_h2d: false,
                }
            }
        }

        impl BridgeIo for FakeIo {
            fn device_to_host(&mut self, device: u64, host: &mut [u16]) -> Result<(), String> {
                if self.fail_d2h {
                    return Err("injected device-to-host fault".to_owned());
                }
                if device != DEVICE_INPUT {
                    return Err(format!("fake io: unknown device address {device:#x}"));
                }
                if host.len() != self.input.len() {
                    return Err(format!(
                        "fake io: host staging holds {} elements, the row has {}",
                        host.len(),
                        self.input.len()
                    ));
                }
                host.copy_from_slice(&self.input);
                Ok(())
            }

            fn host_to_device(&mut self, host: &[u16], device: u64) -> Result<(), String> {
                if self.fail_h2d {
                    return Err("injected host-to-device fault".to_owned());
                }
                if device != DEVICE_OUTPUT {
                    return Err(format!("fake io: unknown device address {device:#x}"));
                }
                if host.len() != self.output.len() {
                    return Err("fake io: length mismatch on the writeback".to_owned());
                }
                self.output.copy_from_slice(host);
                Ok(())
            }
        }

        /// A transport whose host reduce always fails: the bridge must report the phase and count
        /// nothing.
        struct RefusingTransport {
            world: usize,
            rank: usize,
        }

        impl Transport for RefusingTransport {
            fn world_size(&self) -> usize {
                self.world
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
                Err(CommError::Unsupported(
                    "test: the host reduce is unavailable".to_owned(),
                ))
            }
            fn all_gather_bf16(
                &self,
                _local: &[u16],
                _out: &mut [u16],
                _seq: u64,
            ) -> Result<(), CommError> {
                Err(CommError::Unsupported("test: no gather".to_owned()))
            }
        }

        fn cluster(world_size: usize) -> LoopbackCluster {
            LoopbackCluster::new(LoopbackConfig {
                world_size,
                max_elements: 64,
                timeout: Duration::from_secs(5),
                send_delay: Duration::ZERO,
            })
            .expect("loopback cluster")
        }

        fn contribution(rank: usize, count: usize) -> Vec<u16> {
            (0..count)
                .map(|index| {
                    let value = (rank as f32 + 1.0) * 0.25 * ((index % 7) as f32 + 1.0);
                    crate::f32_to_bf16_rn(value)
                })
                .collect()
        }

        fn expected(world: usize, count: usize) -> Vec<u16> {
            let columns: Vec<Vec<u16>> = (0..world).map(|rank| contribution(rank, count)).collect();
            (0..count)
                .map(|slot| {
                    let column: Vec<u16> = columns.iter().map(|payload| payload[slot]).collect();
                    canonical_reduce_element(&column)
                })
                .collect()
        }

        #[test]
        fn the_bridge_reduce_counts_one_collective_and_two_copies() {
            let count = 8usize;
            let cluster = cluster(1);
            let transport = cluster.handle(0).expect("rank 0 transport");
            let mut bridge = HostBridgeReduce::open(&transport, 1, 0, count).expect("bridge opens");
            assert_eq!(bridge.arm(), ReduceArm::HostBridge);
            assert_eq!(bridge.label(), "host-bridge");
            assert_eq!(bridge.count(), count);
            assert_eq!(bridge.stats(), DeviceReduceStats::default());

            let mut io = FakeIo::new(contribution(0, count), count);
            bridge
                .reduce(
                    &mut io,
                    BridgeStage::AttentionDelta,
                    DEVICE_INPUT,
                    DEVICE_OUTPUT,
                    1,
                )
                .expect("the bridge must reduce");
            let stats = bridge.stats();
            assert_eq!(stats.collectives(), 1);
            assert_eq!(stats.elements(), count as u64);
            assert_eq!(stats.bytes(), (count * 2) as u64);
            assert_eq!(stats.bridge_collectives(), 1);
            assert_eq!(stats.bridge_copies(), 2);
            assert_eq!(stats.bridge_d2h_bytes(), (count * 2) as u64);
            assert_eq!(stats.bridge_h2d_bytes(), (count * 2) as u64);
            // One rank's host reduce is the canonical single-rank reduction, not a copy.
            assert_eq!(io.output, expected(1, count));

            let mut second = FakeIo::new(contribution(0, count), count);
            bridge
                .reduce(
                    &mut second,
                    BridgeStage::MoeDelta,
                    DEVICE_INPUT,
                    DEVICE_OUTPUT,
                    2,
                )
                .expect("the second collective reaches the same bridge");
            assert_eq!(bridge.stats().collectives(), 2);
            assert_eq!(bridge.stats().bridge_collectives(), 2);
            assert_eq!(bridge.stats().bridge_copies(), 4);
            assert_eq!(bridge.stats().bridge_d2h_bytes(), (count * 4) as u64);
        }

        #[test]
        fn the_bridge_reduce_matches_the_canonical_host_reduce_across_ranks() {
            let world = 4usize;
            let count = 16usize;
            let cluster = cluster(world);
            let transports: Vec<_> = (0..world)
                .map(|rank| cluster.handle(rank).expect("rank transport"))
                .collect();
            let results: Vec<(Vec<u16>, DeviceReduceStats)> = std::thread::scope(|scope| {
                let joins: Vec<_> = transports
                    .into_iter()
                    .enumerate()
                    .map(|(rank, transport)| {
                        scope.spawn(move || {
                            let mut bridge = HostBridgeReduce::open(&transport, world, rank, count)
                                .expect("bridge opens");
                            let mut io = FakeIo::new(contribution(rank, count), count);
                            bridge
                                .reduce(
                                    &mut io,
                                    BridgeStage::AttentionDelta,
                                    DEVICE_INPUT,
                                    DEVICE_OUTPUT,
                                    1,
                                )
                                .expect("the bridge must reduce");
                            (io.output, bridge.stats())
                        })
                    })
                    .collect();
                joins
                    .into_iter()
                    .map(|join| join.join().expect("rank thread"))
                    .collect()
            });
            let want = expected(world, count);
            for (rank, (output, stats)) in results.iter().enumerate() {
                assert_eq!(*output, want, "rank {rank} bridge output");
                assert_eq!(stats.collectives(), 1);
                assert_eq!(stats.bridge_collectives(), 1);
                assert_eq!(stats.bridge_copies(), 2);
            }
        }

        #[test]
        fn a_refused_copy_is_not_counted() {
            let count = 4usize;
            let cluster = cluster(1);
            let transport = cluster.handle(0).expect("rank 0 transport");
            let mut bridge = HostBridgeReduce::open(&transport, 1, 0, count).expect("bridge opens");

            let mut d2h = FakeIo::new(contribution(0, count), count);
            d2h.fail_d2h = true;
            let error = bridge
                .reduce(
                    &mut d2h,
                    BridgeStage::AttentionDelta,
                    DEVICE_INPUT,
                    DEVICE_OUTPUT,
                    1,
                )
                .expect_err("a refused device-to-host copy must abort the bridge");
            match error {
                NcclError::HostBridge { phase, detail } => {
                    assert_eq!(phase, "device-to-host");
                    assert!(detail.contains("all_reduce"), "{detail}");
                }
                other => panic!("expected HostBridge, got {other:?}"),
            }
            assert_eq!(bridge.stats(), DeviceReduceStats::default());

            let mut h2d = FakeIo::new(contribution(0, count), count);
            h2d.fail_h2d = true;
            let error = bridge
                .reduce(
                    &mut h2d,
                    BridgeStage::MoeDelta,
                    DEVICE_INPUT,
                    DEVICE_OUTPUT,
                    1,
                )
                .expect_err("a refused host-to-device copy must abort the bridge");
            match error {
                NcclError::HostBridge { phase, .. } => assert_eq!(phase, "host-to-device"),
                other => panic!("expected HostBridge, got {other:?}"),
            }
            assert_eq!(
                bridge.stats(),
                DeviceReduceStats::default(),
                "a partly-run bridge is not completed work"
            );
        }

        #[test]
        fn a_refused_host_reduce_is_not_counted() {
            let count = 4usize;
            let transport = RefusingTransport { world: 1, rank: 0 };
            let mut bridge = HostBridgeReduce::open(&transport, 1, 0, count).expect("bridge opens");
            let mut io = FakeIo::new(contribution(0, count), count);
            let error = bridge
                .reduce(
                    &mut io,
                    BridgeStage::AttentionDelta,
                    DEVICE_INPUT,
                    DEVICE_OUTPUT,
                    1,
                )
                .expect_err("a failed host reduce must abort the bridge");
            match error {
                NcclError::HostBridge { phase, detail } => {
                    assert_eq!(phase, "host-reduce");
                    assert!(detail.contains("all_reduce"), "{detail}");
                }
                other => panic!("expected HostBridge, got {other:?}"),
            }
            assert_eq!(bridge.stats(), DeviceReduceStats::default());
        }

        #[test]
        fn the_bridge_refuses_a_transport_for_another_world_or_rank() {
            let cluster = cluster(2);
            let transport = cluster.handle(0).expect("rank 0 transport");
            assert!(matches!(
                HostBridgeReduce::open(&transport, 4, 0, 4),
                Err(NcclError::Config(_))
            ));
            assert!(matches!(
                HostBridgeReduce::open(&transport, 2, 1, 4),
                Err(NcclError::Config(_))
            ));
        }

        #[test]
        fn the_bridge_refuses_a_bad_spec_before_any_copy() {
            let cluster = cluster(1);
            let transport = cluster.handle(0).expect("rank 0 transport");
            for (world, rank, count) in [
                (0, 0, 4),
                (1, 1, 4),
                (1, 0, 0),
                (1, 0, MAX_FRAME_ELEMENTS + 1),
            ] {
                let error = HostBridgeReduce::open(&transport, world, rank, count)
                    .expect_err("an impossible bridge must refuse");
                assert!(
                    matches!(error, NcclError::Config(_)),
                    "{world}/{rank}/{count}"
                );
            }
        }

        #[test]
        fn the_declared_layout_is_the_host_path_s_one_shot_schedule() {
            // The lean loop's host reduce uses OneShotAllToAll; the bridge must not invent a
            // different exchange schedule for the same payload.
            let layout = CollectiveLayout::one_shot(4, 5_120).expect("layout");
            assert_eq!(layout.schedule(), FabricSchedule::OneShotAllToAll);
            assert_eq!(layout.num_elements(), 5_120);
            assert_eq!(layout.world_size(), 4);
        }
    }
}
