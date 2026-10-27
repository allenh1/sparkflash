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

//! The real dual-rail libibverbs backend.
//!
//! This module is compiled only with --features rdma and links nothing: libibverbs is opened at run
//! time by [`verbs::Verbs::load`], and every ABI struct is a hand-written mirror asserted against
//! rdma-core's layout in [`abi`]. The bring-up is the sequence a RoCEv2 RC all-to-all needs and
//! nothing more:
//!
//! 1. resolve the two rails from sysfs through [`select_rails_from_live_fabric`], which refuses a
//!    port that is not ACTIVE at 200 Gb/s or faster, and lets `SPARKFLASH_RAIL_DEVICES` pin the
//!    f1 twins on a host whose f0 functions are DOWN;
//! 2. dlopen libibverbs, open the rail devices by their measured names, query port 1 (state, link
//!    layer, negotiated MTU) and GID index 3 (the RoCEv2 IPv4 GID the deployed vLLM recipe uses);
//! 3. allocate one PD, one CQ, one MR over a bounded staging buffer and one RC QP per peer per rail;
//! 4. exchange the per-(rail, peer) QP numbers, GIDs, staging addresses and remote keys over the
//!    TCP control plane in [`qpinfo`];
//! 5. transition every QP INIT, RTR, RTS with the peer's numbers and the smaller of the two path
//!    MTUs;
//! 6. hand the resources to [`dataplane::DataPlane`], which implements the schedule driver's
//!    [`crate::exchange::Exchange`] trait.
//!
//! Rail count is explicit: the deployed vLLM reference measures with a single rail
//! (`NCCL_IB_HCA=rocep1s0f1`, read from the running container), so `rail_count == 1` is a
//! supported mode that posts whole frames on rail 0. `rail_count == 2` is the specification's
//! dual-rail requirement (QUALIFICATION 3.3) and the mode the hardware gates assert.
//!
//! The staging buffer is host memory. The specification's slot design is device memory reached by
//! GPUNetIO; posting from registered host memory is the part of the backend that is real today, and
//! docs/archive/build/rdma-backend.md states that limit instead of implying a GPU path exists.

mod abi;
mod dataplane;
mod qpinfo;
mod verbs;

use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;

use self::abi::{IbvQpCap, LinkLayer, access, mtu, port_state};
use crate::error::CommError;
use crate::exchange;
use crate::fabric::{FabricView, RailOverride, RailSelection, SYSFS_INFINIBAND_ROOT};
use crate::layout::CollectiveLayout;
use crate::rail::RailLedger;
use crate::telemetry::{CommTelemetry, CommTelemetrySnapshot};
use crate::transport::Transport;
use dataplane::{DataPlane, ReadyRail, RemoteRail, SlotLayout};
pub use verbs::{ShimMode, shim_mode};

use qpinfo::{ControlConfig, RailEndpoint, RankEndpoints, exchange_endpoints};
use verbs::Verbs;

/// The device names the specification declares for the two rails.
///
/// Documentation and preference only: on the live nodes these f0 functions are DOWN, so the rails
/// are resolved through [`RailSelection`], which validates ACTIVE state and the 200 Gb/s minimum
/// (falling back to the ACTIVE f1 twins of the same two cards) before any device is opened.
pub const RDMA_RAIL_INTERFACES: [&str; 2] = [
    crate::fabric::SPEC_RAIL0_INTERFACE,
    crate::fabric::SPEC_RAIL1_INTERFACE,
];

/// Default TCP port rank 0 listens on for the QP-endpoint control plane.
pub const DEFAULT_CONTROL_PORT: u16 = 18_515;
/// Default GID table index: 3 is the RoCEv2 IPv4 GID the deployed vLLM recipe uses.
pub const DEFAULT_GID_INDEX: u8 = 3;
/// Every RoCE device in this fleet exposes port 1.
pub const DEFAULT_RDMA_PORT: u8 = 1;

/// Configuration of one rank of the RDMA backend.
#[derive(Debug, Clone)]
pub struct RdmaConfig {
    /// Number of ranks in the collective.
    pub world_size: usize,
    /// Rank this process serves.
    pub rank: usize,
    /// Control-plane addresses in rank order. The live cluster uses the rail-1 address table:
    /// 198.51.100.80 (node0), .84 (node1), .68 (node2), .59 (node3). The deployed vLLM container
    /// instead sees a mixed rail-2 table; see docs/archive/build/rdma-backend.md before "fixing" either.
    pub peers: Vec<String>,
    /// Port rank 0 listens on and every rank connects to.
    pub control_port: u16,
    /// GID table index used for the RoCEv2 IPv4 GID.
    pub gid_index: u8,
    /// HCA port number.
    pub port: u8,
    /// 1 for the deployed single-rail reference, 2 for the specification's dual rail.
    pub rail_count: usize,
    /// Deadline for control-plane setup and for each collective's completions.
    pub timeout: Duration,
    /// Largest bf16 payload one frame may carry.
    pub max_frame_bytes: usize,
}

impl RdmaConfig {
    /// Reads the configuration for one rank from the environment.
    ///
    /// `SPARK_COMM_PEERS` is a comma-separated address table in rank order (required when
    /// `world_size > 1`); `SPARK_COMM_CONTROL_PORT`, `SPARK_COMM_GID_INDEX`,
    /// `SPARK_COMM_RAIL_COUNT`, `SPARK_COMM_TIMEOUT_MS` and `SPARK_COMM_MAX_FRAME_BYTES`
    /// override the defaults.
    ///
    /// # Errors
    /// Returns [`CommError::RankOutOfRange`] for a rank outside the world and
    /// [`CommError::InvalidLayout`] for a missing or malformed peer table or an unparsable
    /// override.
    pub fn from_env(world_size: usize, rank: usize) -> Result<Self, CommError> {
        let peers = match std::env::var("SPARK_COMM_PEERS") {
            Ok(value) if !value.trim().is_empty() => value
                .split(',')
                .map(|peer| peer.trim().to_owned())
                .filter(|peer| !peer.is_empty())
                .collect(),
            _ => Vec::new(),
        };
        Self::from_env_and_peers(world_size, rank, peers)
    }

    /// Reads every knob from the environment, but takes the rank table from the caller.
    ///
    /// The probe's `--peer` flags are the authoritative address table in that process, so it uses
    /// this entry point instead of requiring SPARK_COMM_PEERS to be set as well.
    ///
    /// # Errors
    /// See [`Self::from_env`]; a table of the wrong length is rejected here with the rail-1 table
    /// hint the operator needs.
    pub fn from_env_with_peers(
        world_size: usize,
        rank: usize,
        peers: Vec<String>,
    ) -> Result<Self, CommError> {
        let peers = peers
            .into_iter()
            .map(|peer| peer.trim().to_owned())
            .filter(|peer| !peer.is_empty())
            .collect();
        Self::from_env_and_peers(world_size, rank, peers)
    }

    /// Shared body of the two environment entry points.
    fn from_env_and_peers(
        world_size: usize,
        rank: usize,
        peers: Vec<String>,
    ) -> Result<Self, CommError> {
        if world_size == 0 || rank >= world_size {
            return Err(CommError::RankOutOfRange { rank, world_size });
        }
        if world_size > 1 && peers.len() != world_size {
            return Err(CommError::InvalidLayout(format!(
                "SPARK_COMM_PEERS must name one control-plane address per rank for world {world_size}, got {} ({peers:?}). The rail-1 table is 198.51.100.80,198.51.100.84,198.51.100.68,198.51.100.59 in rank order",
                peers.len()
            )));
        }
        let config = Self {
            world_size,
            rank,
            peers,
            control_port: env_parsed("SPARK_COMM_CONTROL_PORT", DEFAULT_CONTROL_PORT)?,
            gid_index: env_parsed("SPARK_COMM_GID_INDEX", DEFAULT_GID_INDEX)?,
            port: env_parsed("SPARK_COMM_RDMA_PORT", DEFAULT_RDMA_PORT)?,
            rail_count: std::env::var("SPARK_COMM_RAIL_COUNT")
                .ok()
                .map(|value| {
                    value.trim().parse::<usize>().map_err(|error| {
                        CommError::InvalidLayout(format!(
                            "SPARK_COMM_RAIL_COUNT is not a number: {value:?} ({error})"
                        ))
                    })
                })
                .transpose()?
                .unwrap_or(2),
            timeout: Duration::from_millis(env_parsed::<u64>("SPARK_COMM_TIMEOUT_MS", 5_000)?),
            max_frame_bytes: env_parsed("SPARK_COMM_MAX_FRAME_BYTES", crate::ROCENANTE_MAX_BYTES)?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Checks the fields the bring-up cannot recover from.
    ///
    /// # Errors
    /// Returns [`CommError::RankOutOfRange`], [`CommError::InvalidLayout`] for a rail count
    /// outside 1..=2, a peer table of the wrong length, a zero frame cap or a zero timeout.
    pub fn validate(&self) -> Result<(), CommError> {
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(CommError::RankOutOfRange {
                rank: self.rank,
                world_size: self.world_size,
            });
        }
        if !matches!(self.rail_count, 1 | 2) {
            return Err(CommError::InvalidLayout(format!(
                "rail_count must be 1 or 2, got {}",
                self.rail_count
            )));
        }
        if self.world_size > 1 && self.peers.len() != self.world_size {
            return Err(CommError::InvalidLayout(format!(
                "the control plane needs one address per rank for world {}, got {}",
                self.world_size,
                self.peers.len()
            )));
        }
        if self.max_frame_bytes == 0 || !self.max_frame_bytes.is_multiple_of(2) {
            return Err(CommError::InvalidLayout(format!(
                "max_frame_bytes must be a positive even byte count, got {}",
                self.max_frame_bytes
            )));
        }
        if self.max_frame_bytes > crate::ROCENANTE_MAX_BYTES {
            return Err(CommError::InvalidLayout(format!(
                "max_frame_bytes {} is above the RoCEnante one-shot ceiling of {} bytes",
                self.max_frame_bytes,
                crate::ROCENANTE_MAX_BYTES
            )));
        }
        if self.timeout.is_zero() {
            return Err(CommError::InvalidLayout(
                "the collective timeout must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Parses an environment override, falling back to the default when unset.
fn env_parsed<T: std::str::FromStr>(name: &str, default: T) -> Result<T, CommError>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => value.trim().parse::<T>().map_err(|error| {
            CommError::InvalidLayout(format!("{name}={value:?} does not parse: {error}"))
        }),
        _ => Ok(default),
    }
}

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

/// The real dual-rail RoCEnante transport.
///
/// One handle belongs to one rank; [`crate::transport::Transport`] requires it to be shareable, so the
/// QPs live behind a mutex and one collective is in flight per handle.
pub struct RdmaTransport {
    selection: RailSelection,
    rail_count: usize,
    rank: usize,
    world_size: usize,
    max_frame_bytes: usize,
    /// Declared before the loader so every QP, MR, CQ, PD and device context is destroyed before
    /// libibverbs is dlclosed.
    data: Mutex<DataPlane>,
    telemetry: CommTelemetry,
    /// Declared last: the library must outlive every object it created.
    _verbs: Verbs,
}

impl RdmaTransport {
    /// Opens the backend for one rank, reading the environment for its configuration.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::open_with`], plus a configuration error when the
    /// environment does not describe this rank.
    pub fn open(world_size: usize, rank: usize) -> Result<Self, CommError> {
        Self::open_with(RdmaConfig::from_env(world_size, rank)?)
    }

    /// Brings up the rails and every QP described by `config`.
    ///
    /// # Errors
    /// Returns [`CommError::RankOutOfRange`] or [`CommError::InvalidLayout`] for a bad
    /// configuration, [`CommError::FabricUnavailable`] when the rails or devices are not usable,
    /// [`CommError::Unsupported`] when libibverbs cannot be loaded or its ABI does not export the
    /// symbols this backend calls, [`CommError::Verbs`] when the library refuses a call,
    /// [`CommError::Timeout`] when the control plane misses its deadline, and
    /// [`CommError::Protocol`] when a peer publishes an inconsistent endpoint record.
    pub fn open_with(config: RdmaConfig) -> Result<Self, CommError> {
        config.validate()?;
        let selection = select_rails_from_live_fabric()?;
        let verbs = Verbs::load()?;
        let slot = SlotLayout::new(config.world_size, config.max_frame_bytes)?;
        let mut ready: Vec<ReadyRail> = Vec::with_capacity(config.rail_count);
        let mut owned: [Option<RailEndpoint>; 2] = [None, None];
        for (rail_index, owned_slot) in owned.iter_mut().enumerate().take(config.rail_count) {
            let selected = selection.rail(rail_index).ok_or_else(|| {
                CommError::FabricUnavailable(format!(
                    "rail {rail_index} was requested but the fabric resolved only {} rails",
                    selection.rails.len()
                ))
            })?;
            let name = selected.name().to_owned();
            let netdev = selected.netdev().to_owned();
            let context = verbs.open_device(&name).map_err(|error| {
                CommError::FabricUnavailable(format!("rail {rail_index} device {name}: {error}"))
            })?;
            let port_info = context.query_port(config.port)?;
            if port_info.state != port_state::ACTIVE {
                return Err(CommError::FabricUnavailable(format!(
                    "rail {rail_index} device {name} port {} is not ACTIVE (enum {}); refusing to register a rail that cannot carry traffic",
                    config.port, port_info.state
                )));
            }
            if port_info.link_layer != LinkLayer::Ethernet as u8 {
                return Err(CommError::FabricUnavailable(format!(
                    "rail {rail_index} device {name} reports link layer {} instead of Ethernet: the backend speaks RoCEv2 only",
                    port_info.link_layer
                )));
            }
            let _mtu_bytes = mtu::bytes(port_info.active_mtu).ok_or_else(|| {
                CommError::FabricUnavailable(format!(
                    "rail {rail_index} device {name} reports unknown active_mtu enum {}",
                    port_info.active_mtu
                ))
            })?;
            let gid = context.query_gid(config.port, i32::from(config.gid_index))?;
            match gid.kind() {
                abi::GidKind::Unspecified => {
                    return Err(CommError::FabricUnavailable(format!(
                        "rail {rail_index} device {name} has an all-zero GID at index {}: the RoCEv2 GID table entry is not configured (the deployed vLLM recipe uses GID index 3)",
                        config.gid_index
                    )));
                }
                abi::GidKind::RoceV2Ipv4(_octets) => {}
                abi::GidKind::OtherIpv6 => {
                    // Refused rather than guessed at: the backend builds RoCEv2 IPv4 AH
                    // attributes (hop limit, is_global, sgid_index), and a different GID family
                    // would produce a QP that connects but never carries a packet.
                    return Err(CommError::FabricUnavailable(format!(
                        "rail {rail_index} device {name} GID index {} is an IPv6 GID ({}); this backend builds RoCEv2 IPv4 address vectors only (the deployed vLLM recipe uses GID index 3, the ::ffff:a.b.c.d form)",
                        config.gid_index,
                        gid.describe()
                    )));
                }
            }
            let pd = context.alloc_pd()?;
            let cqe = i32::try_from(config.world_size * 4 + 16).map_err(|_| {
                CommError::InvalidLayout(
                    "world size overflows the completion-queue depth".to_owned(),
                )
            })?;
            let cq = context.create_cq(cqe)?;
            let mut staging = vec![0u8; slot.total_bytes()].into_boxed_slice();
            let mr = pd.reg_mr(
                staging.as_mut_ptr().cast(),
                staging.len(),
                access::LOCAL_WRITE | access::REMOTE_WRITE | access::REMOTE_READ,
            )?;
            if mr.addr() != staging.as_ptr() as u64 {
                return Err(CommError::Unsupported(format!(
                    "rail {rail_index} registered MR address {:#x} does not cover the staging buffer at {:#x}",
                    mr.addr(),
                    staging.as_ptr() as u64
                )));
            }
            let cap = IbvQpCap {
                max_send_wr: 64,
                max_recv_wr: 64,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 0,
            };
            let mut qps = Vec::with_capacity(config.world_size);
            for _ in 0..config.world_size {
                qps.push(pd.create_qp(&cq, &cq, cap)?);
            }
            let qpns: Vec<u32> = qps.iter().map(verbs::Qp::qp_num).collect();
            *owned_slot = Some(RailEndpoint {
                gid,
                gid_index: config.gid_index,
                active_mtu: u16::try_from(port_info.active_mtu).map_err(|_| CommError::Verbs {
                    call: "ibv_query_port",
                    detail: format!(
                        "active_mtu {} does not fit the two-byte field of the control-plane record",
                        port_info.active_mtu
                    ),
                })?,
                mr_addr: staging.as_ptr() as u64,
                rkey: mr.rkey(),
                qpns,
            });
            ready.push(ReadyRail {
                name,
                netdev,
                gid,
                gid_index: config.gid_index,
                active_mtu: port_info.active_mtu,
                qps,
                staging,
                mr,
                cq,
                _pd: pd,
                remotes: Vec::new(),
                _context: context,
            });
        }
        // Single-rail mode publishes a zeroed placeholder for rail 1 and says so in rail_count:
        // the wire format keeps one shape, the decoder validates that an unopened rail is all
        // zeros, and no peer can mistake rail 1 for a second usable path.
        if owned[1].is_none() {
            owned[1] = Some(RankEndpoints::unused_rail(config.world_size));
        }
        let own = RankEndpoints {
            rank: config.rank,
            frame_cap: u32::try_from(config.max_frame_bytes).map_err(|_| {
                CommError::InvalidLayout(format!(
                    "frame cap {} does not fit the control-plane geometry field",
                    config.max_frame_bytes
                ))
            })?,
            slot_bytes: u64::try_from(slot.slot_bytes()).map_err(|_| {
                CommError::InvalidLayout(
                    "slot size does not fit the control-plane geometry field".to_owned(),
                )
            })?,
            rail_count: u8::try_from(config.rail_count).map_err(|_| {
                CommError::InvalidLayout(
                    "rail count does not fit the control-plane geometry field".to_owned(),
                )
            })?,
            rails: [
                owned[0]
                    .take()
                    .ok_or_else(|| CommError::InvalidLayout("no rail was opened".to_owned()))?,
                owned[1].take().ok_or_else(|| {
                    CommError::InvalidLayout("rail 1 endpoint was not built".to_owned())
                })?,
            ],
        };
        let control = ControlConfig {
            rank: config.rank,
            world: config.world_size,
            peers: config.peers.clone(),
            port: config.control_port,
            timeout: config.timeout,
        };
        let table = exchange_endpoints(&control, own)?;
        // The remote offsets are derived from the peer's own slot geometry, so a disagreement
        // would be silent corruption: refuse the table instead of writing at a guessed offset.
        if table.frame_cap() != u32::try_from(config.max_frame_bytes).unwrap_or(u32::MAX)
            || table.slot_bytes() != u64::try_from(slot.slot_bytes()).unwrap_or(u64::MAX)
            || table.rail_count() != u8::try_from(config.rail_count).unwrap_or(u8::MAX)
        {
            return Err(CommError::Protocol(format!(
                "the cluster disagrees on the staging geometry: this rank has frame cap {} / slot {} / {} rail(s), the table says frame cap {} / slot {} / {} rail(s)",
                config.max_frame_bytes,
                slot.slot_bytes(),
                config.rail_count,
                table.frame_cap(),
                table.slot_bytes(),
                table.rail_count()
            )));
        }
        for (rail_index, rail) in ready.iter_mut().enumerate() {
            let mut remotes = Vec::with_capacity(config.world_size);
            for dest in 0..config.world_size {
                if dest == config.rank {
                    remotes.push(RemoteRail {
                        gid: rail.gid,
                        gid_index: rail.gid_index,
                        active_mtu: rail.active_mtu,
                        mr_addr: 0,
                        rkey: 0,
                    });
                    continue;
                }
                let endpoint = table
                    .rank(dest)?
                    .rail(rail_index)
                    .ok_or_else(|| {
                        CommError::Protocol(format!(
                            "rank {dest} published no endpoint for rail {rail_index}"
                        ))
                    })?
                    .clone();
                remotes.push(RemoteRail {
                    gid: endpoint.gid,
                    gid_index: endpoint.gid_index,
                    active_mtu: i32::from(endpoint.active_mtu),
                    mr_addr: endpoint.mr_addr,
                    rkey: endpoint.rkey,
                });
            }
            rail.remotes = remotes;
        }
        for (rail_index, rail) in ready.iter().enumerate().take(config.rail_count) {
            let my_gid_index = rail.gid_index;
            let my_mtu = rail.active_mtu;
            for dest in 0..config.world_size {
                if dest == config.rank {
                    continue;
                }
                let endpoint = table.rank(dest)?.rail(rail_index).ok_or_else(|| {
                    CommError::Protocol(format!(
                        "rank {dest} published no endpoint for rail {rail_index}"
                    ))
                })?;
                let peer_qpn = endpoint.qpns.get(config.rank).copied().ok_or_else(|| {
                    CommError::Protocol(format!(
                        "rank {dest} published no QP number for rank {}",
                        config.rank
                    ))
                })?;
                let path_mtu = dataplane::min_path_mtu(my_mtu, i32::from(endpoint.active_mtu))?;
                let (mut attr, mask) = dataplane::init_attr(config.port);
                rail.qps[dest].modify(&mut attr, mask)?;
                let (mut attr, mask) = dataplane::rtr_attr(
                    config.port,
                    peer_qpn,
                    endpoint.gid,
                    my_gid_index,
                    path_mtu,
                );
                rail.qps[dest].modify(&mut attr, mask)?;
                let (mut attr, mask) = dataplane::rts_attr();
                rail.qps[dest].modify(&mut attr, mask)?;
            }
        }
        let data = DataPlane::new(
            config.world_size,
            config.rank,
            slot,
            config.rail_count,
            config.timeout,
            ready,
        )?;
        let telemetry = CommTelemetry::default();
        telemetry.set_fabric(&selection);
        Ok(Self {
            selection,
            rail_count: config.rail_count,
            rank: config.rank,
            world_size: config.world_size,
            max_frame_bytes: config.max_frame_bytes,
            data: Mutex::new(data),
            telemetry,
            _verbs: verbs,
        })
    }

    /// Whether the real backend is compiled in. It is: this module only exists with the feature.
    #[must_use]
    pub fn is_built() -> bool {
        true
    }

    /// Number of rails this handle was brought up with.
    #[must_use]
    pub fn rail_count(&self) -> usize {
        self.rail_count
    }

    /// Largest bf16 payload one frame may carry.
    #[must_use]
    pub fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    /// The validated rail selection this handle registered.
    #[must_use]
    pub fn rail_selection(&self) -> &RailSelection {
        &self.selection
    }

    /// Names of the opened rail devices.
    #[must_use]
    pub fn rail_names(&self) -> Vec<String> {
        self.data.lock().rail_names()
    }

    /// Bytes actually posted per rail during the most recent collective.
    ///
    /// This is measured from the writes this backend posted, not predicted from the layout, so a
    /// gate can compare it against the schedule's expectation.
    #[must_use]
    pub fn rail_ledger(&self) -> RailLedger {
        self.data.lock().ledger()
    }

    /// Counters that follow the specification's field names.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CommTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// The telemetry object itself, for a caller that wants to install it in a communicator.
    #[must_use]
    pub fn telemetry(&self) -> &CommTelemetry {
        &self.telemetry
    }

    /// Accounts one completed collective's measured bytes into the specification's counters.
    fn account(&self, ledger: RailLedger, kind: &str) {
        self.telemetry
            .rdma_bytes_rail0
            .fetch_add(ledger.rail0_bytes, Ordering::Relaxed);
        self.telemetry
            .rdma_bytes_rail1
            .fetch_add(ledger.rail1_bytes, Ordering::Relaxed);
        match kind {
            "all-reduce" => {
                self.telemetry
                    .rocenante_allreduce_calls
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.telemetry
                    .rocenante_allgather_calls
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Runs one exchange with the data plane locked, resetting the measured ledger first.
    fn run_exchange(
        &self,
        kind: &str,
        run: impl FnOnce(&mut DataPlane) -> Result<(), CommError>,
    ) -> Result<(), CommError> {
        let mut plane = self.data.lock();
        plane.reset_ledger();
        let outcome = run(&mut plane);
        let ledger = plane.ledger();
        drop(plane);
        self.account(ledger, kind);
        outcome
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
        buf: &mut [u16],
        seq: u64,
        layout: &CollectiveLayout,
    ) -> Result<(), CommError> {
        let rank = self.rank;
        self.run_exchange("all-reduce", move |plane| {
            exchange::all_reduce(plane, rank, buf, seq, layout)
        })
    }

    fn all_gather_bf16(&self, local: &[u16], out: &mut [u16], seq: u64) -> Result<(), CommError> {
        let (rank, world) = (self.rank, self.world_size);
        self.run_exchange("all-gather", move |plane| {
            exchange::all_gather(plane, rank, local, out, seq, world)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backend_is_built_and_refuses_on_a_host_without_a_usable_configuration() {
        assert!(RdmaTransport::is_built());
        // With no SPARK_COMM_PEERS for a four-rank world, the configuration is refused before any
        // device is touched. On a host that does have a peer table, the next refusal is the fabric
        // or the missing library, so all three are acceptable ways for this box to refuse.
        match RdmaTransport::open(4, 0) {
            Ok(_) => panic!("this host cannot serve four ranks; open() must not succeed"),
            Err(
                CommError::InvalidLayout(_)
                | CommError::FabricUnavailable(_)
                | CommError::Unsupported(_),
            ) => {}
            Err(other) => panic!("unexpected refusal: {other}"),
        }
        assert!(matches!(
            RdmaTransport::open(4, 4),
            Err(CommError::RankOutOfRange { .. })
        ));
        assert!(matches!(
            RdmaTransport::open(0, 0),
            Err(CommError::RankOutOfRange { .. })
        ));
    }

    #[test]
    fn the_spec_rail_names_are_the_documented_ones() {
        assert_eq!(RDMA_RAIL_INTERFACES, ["rocep1s0f0", "roceP2p1s0f0"]);
    }

    #[test]
    fn a_config_that_cannot_be_served_is_refused_before_bring_up() {
        let good = RdmaConfig {
            world_size: 4,
            rank: 0,
            peers: vec!["198.51.100.80".to_owned(); 4],
            control_port: DEFAULT_CONTROL_PORT,
            gid_index: DEFAULT_GID_INDEX,
            port: DEFAULT_RDMA_PORT,
            rail_count: 2,
            timeout: Duration::from_secs(5),
            max_frame_bytes: crate::ROCENANTE_MAX_BYTES,
        };
        good.validate().unwrap_or_else(|error| panic!("{error}"));

        let mut bad = good.clone();
        bad.rail_count = 3;
        assert!(matches!(bad.validate(), Err(CommError::InvalidLayout(_))));
        let mut bad = good.clone();
        bad.peers.pop();
        assert!(matches!(bad.validate(), Err(CommError::InvalidLayout(_))));
        let mut bad = good.clone();
        bad.max_frame_bytes = 1023;
        assert!(matches!(bad.validate(), Err(CommError::InvalidLayout(_))));
        let mut bad = good.clone();
        bad.max_frame_bytes = crate::ROCENANTE_MAX_BYTES * 2;
        assert!(matches!(bad.validate(), Err(CommError::InvalidLayout(_))));
        let mut bad = good.clone();
        bad.timeout = Duration::ZERO;
        assert!(matches!(bad.validate(), Err(CommError::InvalidLayout(_))));
        let mut bad = good;
        bad.rank = 9;
        assert!(matches!(
            bad.validate(),
            Err(CommError::RankOutOfRange { .. })
        ));

        // World 1 needs no peer table and is the mode a single node can exercise.
        let single = RdmaConfig {
            world_size: 1,
            rank: 0,
            peers: Vec::new(),
            rail_count: 1,
            ..bad
        };
        single.validate().unwrap_or_else(|error| {
            panic!("a single-rank, single-rail config must be legal: {error}")
        });
    }
}
