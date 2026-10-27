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

//! Control-plane exchange of the per-(rail, peer) queue-pair endpoints.
//!
//! A reliable-connected QP cannot be brought up by one side alone: the RTR transition needs the
//! peer's QP number and GID, the peer's RTR needs this side's, and the RDMA writes need the peer's
//! registered-memory base address and remote key. This module carries exactly that information
//! over the TCP framing utilities, once per process, before any collective runs.
//!
//! Topology: a star with rank 0 at the centre. Every rank sends its endpoint record to rank 0;
//! rank 0 assembles the table in rank order and broadcasts the assembled table to all ranks, so
//! every rank ends the call holding the same table and the call is a barrier as well as an
//! exchange. A full mesh of TCP connections would need no coordinator, but the star keeps one
//! listener and one deterministic assembly order, which is what makes the merged table verifiable:
//! the decoder rejects a table that is not in ascending rank order, has a repeated or missing rank,
//! or carries a zero QP number, an unknown MTU or an all-zero GID.
//!
//! The wire format is little-endian and versioned with its own magic, so a mismatched build on one
//! node fails at connect time with a protocol error instead of modifying a QP with nonsense.

use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use super::abi::{IbvGid, mtu};
use crate::error::CommError;

/// Magic ("SQPI" little-endian) opening every control message.
pub(crate) const QPINFO_MAGIC: u32 = 0x5351_5049;
/// Format version; a mismatched node is rejected rather than reinterpreted.
pub(crate) const QPINFO_VERSION: u16 = 1;
/// Fixed part of the control header: magic, version, kind, reserved, body length.
pub(crate) const QPINFO_HEADER_BYTES: usize = 12;
/// Control message kind carried in the header.
pub(crate) const QPINFO_KIND_ENDPOINTS: u8 = 1;
/// Largest control message this codec will allocate for (the table is small; the cap is a guard).
const MAX_CONTROL_MESSAGE_BYTES: usize = 1 << 20;

/// One rank's endpoint on one rail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RailEndpoint {
    /// The GID this rank listens on for this rail (RoCEv2 IPv4 on the live cluster).
    pub(crate) gid: IbvGid,
    /// Index in this rank's GID table that `gid` came from.
    pub(crate) gid_index: u8,
    /// `enum ibv_mtu` the port negotiated; both sides take its byte count and connect at the
    /// smaller of the two.
    pub(crate) active_mtu: u16,
    /// Virtual address of this rank's registered staging buffer on this rail.
    pub(crate) mr_addr: u64,
    /// Remote key authorising this rank's staging buffer.
    pub(crate) rkey: u32,
    /// QP number, indexed by destination rank: `qpns[d]` is the QP on this rail and rank that
    /// this rank posts to when it sends to rank `d`. A rank's own index is unused.
    pub(crate) qpns: Vec<u32>,
}

/// Everything one rank publishes about itself, including the geometry its staging buffer uses.
///
/// The geometry is per record rather than global on purpose: a decoder can then reject a table in
/// which two ranks disagree about the frame cap or slot size, which would otherwise put every
/// remote write at the wrong offset (a silent corruption, not a visible failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RankEndpoints {
    /// The publishing rank.
    pub(crate) rank: usize,
    /// Largest frame payload this rank's staging layout accepts.
    pub(crate) frame_cap: u32,
    /// Bytes one staging slot occupies: the frame header plus the whole frame.
    pub(crate) slot_bytes: u64,
    /// Rails this rank actually opened (1 or 2). Unused rail records are all-zero.
    pub(crate) rail_count: u8,
    /// Rail 0 and rail 1; entries at or above `rail_count` are placeholders and must be zero.
    pub(crate) rails: [RailEndpoint; 2],
}

impl RankEndpoints {
    /// The endpoint for an opened rail index; `None` when the index is above `rail_count`.
    pub(crate) fn rail(&self, index: usize) -> Option<&RailEndpoint> {
        if index < usize::from(self.rail_count) {
            self.rails.get(index)
        } else {
            None
        }
    }

    /// A placeholder endpoint for a rail this rank did not open (all fields zero).
    pub(crate) fn unused_rail(world: usize) -> RailEndpoint {
        RailEndpoint {
            gid: IbvGid::default(),
            gid_index: 0,
            active_mtu: 0,
            mr_addr: 0,
            rkey: 0,
            qpns: vec![0; world],
        }
    }
}

impl RailEndpoint {
    /// Whether this endpoint is the all-zero placeholder for an unopened rail.
    pub(crate) fn is_unused(&self) -> bool {
        self.gid == IbvGid::default()
            && self.gid_index == 0
            && self.active_mtu == 0
            && self.mr_addr == 0
            && self.rkey == 0
            && self.qpns.iter().all(|qpn| *qpn == 0)
    }
}

/// The merged table: one entry per rank, in ascending rank order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QpTable {
    /// Number of ranks this table describes.
    pub(crate) world: usize,
    /// One entry per rank; `entries[rank]` describes that rank.
    pub(crate) entries: Vec<RankEndpoints>,
}

impl QpTable {
    /// The endpoints published by `rank`.
    ///
    /// # Errors
    /// Returns [`CommError::RankOutOfRange`] when `rank` is outside the table.
    pub(crate) fn rank(&self, rank: usize) -> Result<&RankEndpoints, CommError> {
        self.entries.get(rank).ok_or(CommError::RankOutOfRange {
            rank,
            world_size: self.world,
        })
    }

    /// Bytes one rank's record occupies on the wire.
    pub(crate) fn rank_record_bytes(world: usize) -> usize {
        // rank, frame cap, slot bytes, rail count + reserved, then two rail records.
        4 + 4 + 8 + 1 + 3 + 2 * (16 + 1 + 2 + 8 + 4 + 4 * world)
    }

    /// Largest frame payload every rank in this table published.
    pub(crate) fn frame_cap(&self) -> u32 {
        self.entries.first().map_or(0, |entry| entry.frame_cap)
    }

    /// Staging slot size every rank in this table published.
    pub(crate) fn slot_bytes(&self) -> u64 {
        self.entries.first().map_or(0, |entry| entry.slot_bytes)
    }

    /// Number of rails every rank in this table opened.
    pub(crate) fn rail_count(&self) -> u8 {
        self.entries.first().map_or(0, |entry| entry.rail_count)
    }

    /// Encodes this table as one control message.
    ///
    /// # Errors
    /// Returns [`CommError::Protocol`] when the table's shape does not match its records.
    pub(crate) fn encode(&self) -> Result<Vec<u8>, CommError> {
        encode_message(self.world, &self.entries)
    }

    /// Decodes a full table, requiring exactly `expected_world` ranks in ascending order.
    ///
    /// # Errors
    /// Returns [`CommError::Protocol`] for a short message, wrong magic, version or kind, a world
    /// size other than `expected_world`, a record count other than `expected_world`, ranks that are
    /// not `0..world` in order, a zero QP number, an unknown MTU, or an all-zero GID.
    pub(crate) fn decode(bytes: &[u8], expected_world: usize) -> Result<Self, CommError> {
        let (world, entries) = decode_message(bytes, expected_world)?;
        if entries.len() != world {
            return Err(CommError::Protocol(format!(
                "QP-info message carries {} rank records but declares world {world}",
                entries.len()
            )));
        }
        Ok(Self { world, entries })
    }
}

/// Encodes one rank's record as a single-record control message, for the star exchange.
///
/// # Errors
/// Returns [`CommError::Protocol`] when the record does not match `world`.
pub(crate) fn encode_rank_message(
    world: usize,
    entry: &RankEndpoints,
) -> Result<Vec<u8>, CommError> {
    encode_message(world, std::slice::from_ref(entry))
}

/// Decodes a single-record control message from rank 0's peers.
///
/// # Errors
/// Returns [`CommError::Protocol`] when the message carries anything other than exactly one
/// record, when the record's rank is outside `expected_world`, or for any body-level violation.
pub(crate) fn decode_rank_message(
    bytes: &[u8],
    expected_world: usize,
) -> Result<RankEndpoints, CommError> {
    let (world, mut entries) = decode_message(bytes, expected_world)?;
    if entries.len() != 1 {
        return Err(CommError::Protocol(format!(
            "a rank record message must carry exactly one record, got {}",
            entries.len()
        )));
    }
    let entry = entries.remove(0);
    if entry.rank >= world {
        return Err(CommError::Protocol(format!(
            "rank record claims rank {} but the cluster has world {world}",
            entry.rank
        )));
    }
    Ok(entry)
}

/// Encodes a header plus `count` records for `world` ranks.
fn encode_message(world: usize, entries: &[RankEndpoints]) -> Result<Vec<u8>, CommError> {
    if world == 0 {
        return Err(CommError::Protocol(
            "a QP-info message cannot describe world 0".to_owned(),
        ));
    }
    let record_bytes = QpTable::rank_record_bytes(world);
    let body_bytes = 4 + 4 + entries.len() * record_bytes;
    let mut out = Vec::with_capacity(QPINFO_HEADER_BYTES + body_bytes);
    out.extend_from_slice(&QPINFO_MAGIC.to_le_bytes());
    out.extend_from_slice(&QPINFO_VERSION.to_le_bytes());
    out.push(QPINFO_KIND_ENDPOINTS);
    out.push(0);
    out.extend_from_slice(&(body_bytes as u32).to_le_bytes());
    out.extend_from_slice(&(world as u32).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    if let Some(first) = entries.first() {
        if first.frame_cap == 0
            || first.slot_bytes < u64::from(first.frame_cap)
            || !matches!(first.rail_count, 1 | 2)
        {
            return Err(CommError::Protocol(format!(
                "rank {} publishes an unusable staging geometry: frame cap {}, slot {} bytes, {} rails",
                first.rank, first.frame_cap, first.slot_bytes, first.rail_count
            )));
        }
        for entry in entries {
            if entry.frame_cap != first.frame_cap
                || entry.slot_bytes != first.slot_bytes
                || entry.rail_count != first.rail_count
            {
                return Err(CommError::Protocol(format!(
                    "rank {} publishes geometry (cap {}, slot {}, {} rails) but rank {} publishes ({}, {}, {}): every rank must agree on the staging layout",
                    entry.rank,
                    entry.frame_cap,
                    entry.slot_bytes,
                    entry.rail_count,
                    first.rank,
                    first.frame_cap,
                    first.slot_bytes,
                    first.rail_count
                )));
            }
        }
    }
    for entry in entries {
        if entry.rank >= world {
            return Err(CommError::RankOutOfRange {
                rank: entry.rank,
                world_size: world,
            });
        }
        out.extend_from_slice(&(entry.rank as u32).to_le_bytes());
        out.extend_from_slice(&entry.frame_cap.to_le_bytes());
        out.extend_from_slice(&entry.slot_bytes.to_le_bytes());
        out.push(entry.rail_count);
        out.extend_from_slice(&[0u8; 3]);
        for rail_index in 0..2 {
            let rail = entry.rails.get(rail_index).ok_or_else(|| {
                CommError::Protocol(format!("rank {} is missing rail {rail_index}", entry.rank))
            })?;
            if rail_index >= usize::from(entry.rail_count) {
                if !rail.is_unused() || rail.qpns.len() != world {
                    return Err(CommError::Protocol(format!(
                        "rank {} publishes a non-zero record for unopened rail {rail_index}",
                        entry.rank
                    )));
                }
                // The placeholder is zeros on the wire; no need to read its fields.
                out.extend_from_slice(&[0u8; 16]);
                out.push(0);
                out.extend_from_slice(&0u16.to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
                for _ in 0..world {
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
                continue;
            }
            if rail.is_unused() {
                return Err(CommError::Protocol(format!(
                    "rank {} opened rail {rail_index} but publishes an all-zero endpoint",
                    entry.rank
                )));
            }
            if rail.qpns.len() != world {
                return Err(CommError::Protocol(format!(
                    "rank {} rail {rail_index} publishes {} QP numbers, but world is {world}",
                    entry.rank,
                    rail.qpns.len()
                )));
            }
            out.extend_from_slice(&rail.gid.raw);
            out.push(rail.gid_index);
            out.extend_from_slice(&rail.active_mtu.to_le_bytes());
            out.extend_from_slice(&rail.mr_addr.to_le_bytes());
            out.extend_from_slice(&rail.rkey.to_le_bytes());
            for qpn in &rail.qpns {
                out.extend_from_slice(&qpn.to_le_bytes());
            }
        }
    }
    if out.len() - QPINFO_HEADER_BYTES != body_bytes {
        return Err(CommError::Protocol(
            "internal QP-info encoding length mismatch".to_owned(),
        ));
    }
    Ok(out)
}

/// Decodes a control message down to its declared world and its records.
fn decode_message(
    bytes: &[u8],
    expected_world: usize,
) -> Result<(usize, Vec<RankEndpoints>), CommError> {
    if bytes.len() < QPINFO_HEADER_BYTES {
        return Err(CommError::Protocol(format!(
            "QP-info message needs a {QPINFO_HEADER_BYTES}-byte header, got {} bytes",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != QPINFO_MAGIC {
        return Err(CommError::Protocol(format!(
            "bad QP-info magic 0x{magic:08x}, expected 0x{QPINFO_MAGIC:08x}"
        )));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != QPINFO_VERSION {
        return Err(CommError::Protocol(format!(
            "unsupported QP-info version {version}, expected {QPINFO_VERSION}"
        )));
    }
    if bytes[6] != QPINFO_KIND_ENDPOINTS {
        return Err(CommError::Protocol(format!(
            "unexpected QP-info kind {}",
            bytes[6]
        )));
    }
    let body_bytes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if bytes.len() != QPINFO_HEADER_BYTES + body_bytes {
        return Err(CommError::Protocol(format!(
            "QP-info header declares {body_bytes} body bytes but {} were supplied",
            bytes.len() - QPINFO_HEADER_BYTES
        )));
    }
    if body_bytes > MAX_CONTROL_MESSAGE_BYTES {
        return Err(CommError::Protocol(format!(
            "QP-info body of {body_bytes} bytes exceeds the {MAX_CONTROL_MESSAGE_BYTES}-byte guard"
        )));
    }
    let body = &bytes[QPINFO_HEADER_BYTES..];
    let mut cursor = 0;
    let world = read_u32(body, &mut cursor)? as usize;
    if world != expected_world {
        return Err(CommError::Protocol(format!(
            "QP-info message describes world {world} but this process was configured for {expected_world}"
        )));
    }
    let count = read_u32(body, &mut cursor)? as usize;
    let record_bytes = QpTable::rank_record_bytes(world);
    let wanted = 8 + count * record_bytes;
    if body.len() != wanted {
        return Err(CommError::Protocol(format!(
            "QP-info body for world {world} and {count} records must be {wanted} bytes, got {}",
            body.len()
        )));
    }
    let mut entries = Vec::with_capacity(count);
    let mut geometry: Option<(u32, u64, u8)> = None;
    for index in 0..count {
        let rank = read_u32(body, &mut cursor)? as usize;
        if count == world && rank != index {
            return Err(CommError::Protocol(format!(
                "QP-info table lists rank {rank} where rank {index} must appear: the table must be in ascending rank order with no gaps"
            )));
        }
        if rank >= world {
            return Err(CommError::Protocol(format!(
                "QP-info record claims rank {rank} but the world is {world}"
            )));
        }
        let frame_cap = read_u32(body, &mut cursor)?;
        let slot_bytes = read_u64(body, &mut cursor)?;
        let rail_count = read_bytes(body, &mut cursor, 1)?[0];
        let reserved = read_bytes(body, &mut cursor, 3)?;
        if reserved != [0u8; 3] {
            return Err(CommError::Protocol(format!(
                "rank {rank} record has non-zero reserved bytes in its geometry header"
            )));
        }
        if !matches!(rail_count, 1 | 2) {
            return Err(CommError::Protocol(format!(
                "rank {rank} publishes rail_count {rail_count}; the backend opens 1 or 2 rails"
            )));
        }
        if frame_cap == 0 || slot_bytes < u64::from(frame_cap) {
            return Err(CommError::Protocol(format!(
                "rank {rank} publishes frame cap {frame_cap} and slot size {slot_bytes}: a slot must hold the whole frame"
            )));
        }
        match geometry {
            None => geometry = Some((frame_cap, slot_bytes, rail_count)),
            Some(expected) if expected != (frame_cap, slot_bytes, rail_count) => {
                return Err(CommError::Protocol(format!(
                    "rank {rank} publishes geometry (frame cap {frame_cap}, slot {slot_bytes}, {rail_count} rails) but an earlier record publishes ({}, {}, {}): every rank must agree on the staging layout",
                    expected.0, expected.1, expected.2
                )));
            }
            Some(_) => {}
        }
        let mut rails: [Option<RailEndpoint>; 2] = [None, None];
        for (rail_index, rail_slot) in rails.iter_mut().enumerate() {
            let mut raw_gid = [0u8; 16];
            raw_gid.copy_from_slice(read_bytes(body, &mut cursor, 16)?);
            let gid = IbvGid { raw: raw_gid };
            let gid_index = read_bytes(body, &mut cursor, 1)?[0];
            let active_mtu = read_u16(body, &mut cursor)?;
            let mr_addr = read_u64(body, &mut cursor)?;
            let rkey = read_u32(body, &mut cursor)?;
            let mut qpns = Vec::with_capacity(world);
            for _dest in 0..world {
                qpns.push(read_u32(body, &mut cursor)?);
            }
            let endpoint = RailEndpoint {
                gid,
                gid_index,
                active_mtu,
                mr_addr,
                rkey,
                qpns,
            };
            if rail_index >= usize::from(rail_count) {
                if !endpoint.is_unused() {
                    return Err(CommError::Protocol(format!(
                        "rank {rank} published data for rail {rail_index} although it opened only {rail_count} rail(s)"
                    )));
                }
            } else {
                if endpoint.gid.kind() == super::abi::GidKind::Unspecified {
                    return Err(CommError::Protocol(format!(
                        "rank {rank} rail {rail_index} published an all-zero GID"
                    )));
                }
                if mtu::bytes(c_int_of(active_mtu)).is_none() {
                    return Err(CommError::Protocol(format!(
                        "rank {rank} rail {rail_index} published unknown MTU enum {active_mtu}"
                    )));
                }
                if mr_addr == 0 {
                    return Err(CommError::Protocol(format!(
                        "rank {rank} rail {rail_index} published a null staging address"
                    )));
                }
                if endpoint.qpns.contains(&0) {
                    return Err(CommError::Protocol(format!(
                        "rank {rank} rail {rail_index} published QP number 0"
                    )));
                }
            }
            *rail_slot = Some(endpoint);
        }
        let [Some(rail0), Some(rail1)] = rails else {
            return Err(CommError::Protocol(format!(
                "rank {rank} is missing a rail endpoint"
            )));
        };
        entries.push(RankEndpoints {
            rank,
            frame_cap,
            slot_bytes,
            rail_count,
            rails: [rail0, rail1],
        });
    }
    if cursor != body.len() {
        return Err(CommError::Protocol(format!(
            "QP-info body has {} trailing bytes",
            body.len() - cursor
        )));
    }
    Ok((world, entries))
}

/// Widens an unsigned MTU enum stored on the wire back to the C int the ABI uses.
fn c_int_of(value: u16) -> std::ffi::c_int {
    std::ffi::c_int::from(value)
}

/// Reads a little-endian u16 from a cursor.
fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, CommError> {
    let slice = read_bytes(bytes, cursor, 2)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Reads a little-endian u32 from a cursor.
fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, CommError> {
    let slice = read_bytes(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Reads a little-endian u64 from a cursor.
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, CommError> {
    let slice = read_bytes(bytes, cursor, 8)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

/// Reads exactly `len` bytes from a cursor, advancing it.
fn read_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], CommError> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| CommError::Protocol("QP-info cursor overflow".to_owned()))?;
    if end > bytes.len() {
        return Err(CommError::Protocol(format!(
            "QP-info body ended after {} bytes while {} more were needed",
            bytes.len(),
            end - bytes.len()
        )));
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

/// Configuration of the control-plane star.
#[derive(Debug, Clone)]
pub(crate) struct ControlConfig {
    /// Rank this process serves.
    pub(crate) rank: usize,
    /// Number of ranks in the cluster.
    pub(crate) world: usize,
    /// Control-plane addresses in rank order; non-zero ranks connect to `peers[0]`.
    pub(crate) peers: Vec<String>,
    /// TCP port rank 0 listens on, and every rank connects to.
    pub(crate) port: u16,
    /// Deadline for connecting and for each control message.
    pub(crate) timeout: Duration,
}

/// How the star exchange retries a peer that has not reached its listener yet.
const CONNECT_RETRY: Duration = Duration::from_millis(25);

/// Runs the star exchange and returns the merged endpoint table.
///
/// Rank 0 listens, collects one record per peer, merges them in rank order and broadcasts the
/// table; every other rank connects to rank 0, sends its record and reads the table. The call
/// returns only after the table is complete on this rank, so it is a barrier as well as an
/// exchange.
///
/// # Errors
/// Returns [`CommError::RankOutOfRange`] for a rank outside the world,
/// [`CommError::InvalidLayout`] for a peer table without a usable root address,
/// [`CommError::Timeout`] when a peer does not appear before the deadline, and
/// [`CommError::Protocol`] for a malformed or inconsistent message.
pub(crate) fn exchange_endpoints(
    config: &ControlConfig,
    own: RankEndpoints,
) -> Result<QpTable, CommError> {
    if config.world == 0 || config.rank >= config.world {
        return Err(CommError::RankOutOfRange {
            rank: config.rank,
            world_size: config.world,
        });
    }
    if own.rank != config.rank {
        return Err(CommError::Protocol(format!(
            "this process is rank {} but its endpoint record claims rank {}",
            config.rank, own.rank
        )));
    }
    if config.world == 1 {
        return Ok(QpTable {
            world: 1,
            entries: vec![own],
        });
    }
    if config.peers.len() != config.world {
        return Err(CommError::InvalidLayout(format!(
            "the control plane needs one address per rank for world {}, got {}: {:?}",
            config.world,
            config.peers.len(),
            config.peers
        )));
    }
    let root = resolve_address(&config.peers[0], config.port)?;
    let deadline = Instant::now() + config.timeout;
    if config.rank == 0 {
        collect_and_broadcast(config, own, root, deadline)
    } else {
        join_as_peer(config, own, root, deadline)
    }
}

/// Rank 0: accepts every peer, merges the records and broadcasts the table.
fn collect_and_broadcast(
    config: &ControlConfig,
    own: RankEndpoints,
    root: SocketAddr,
    deadline: Instant,
) -> Result<QpTable, CommError> {
    let listener = TcpListener::bind(root)?;
    listener.set_nonblocking(true)?;
    let mut streams = Vec::with_capacity(config.world - 1);
    while streams.len() < config.world - 1 {
        let stream = accept_retry(&listener, deadline, config.timeout)?;
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(config.timeout))?;
        stream.set_write_timeout(Some(config.timeout))?;
        streams.push(stream);
    }
    let mut entries: Vec<Option<RankEndpoints>> = (0..config.world).map(|_| None).collect();
    entries[0] = Some(own);
    for stream in &mut streams {
        let message = read_message(stream, config.timeout).map_err(|error| {
            control_error(error, "rank 0 failed to read a peer record".to_owned())
        })?;
        let record = decode_rank_message(&message, config.world)?;
        if record.rank == 0 {
            return Err(CommError::Protocol(
                "a peer claimed rank 0 in the control plane".to_owned(),
            ));
        }
        if entries[record.rank].replace(record.clone()).is_some() {
            return Err(CommError::Protocol(format!(
                "rank {} sent two control-plane records",
                record.rank
            )));
        }
    }
    let table = QpTable {
        world: config.world,
        entries: entries
            .into_iter()
            .enumerate()
            .map(|(rank, entry)| {
                entry.ok_or_else(|| {
                    CommError::Protocol(format!(
                        "rank 0 never received a control-plane record from rank {rank}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let encoded = table.encode()?;
    for stream in &mut streams {
        write_message(stream, &encoded)?;
    }
    Ok(table)
}

/// Rank r > 0: connects to rank 0, sends this rank's record and reads the merged table.
fn join_as_peer(
    config: &ControlConfig,
    own: RankEndpoints,
    root: SocketAddr,
    deadline: Instant,
) -> Result<QpTable, CommError> {
    let mut stream = connect_retry(root, deadline, config.timeout)?;
    stream.set_read_timeout(Some(config.timeout))?;
    stream.set_write_timeout(Some(config.timeout))?;
    let message = encode_rank_message(config.world, &own)?;
    write_message(&mut stream, &message)?;
    let table_bytes = read_message(&mut stream, config.timeout).map_err(|error| {
        control_error(
            error,
            format!(
                "rank {} failed to read the merged table from rank 0",
                config.rank
            ),
        )
    })?;
    let table = QpTable::decode(&table_bytes, config.world)?;
    let echoed = table.rank(config.rank)?;
    if echoed != &own {
        return Err(CommError::Protocol(format!(
            "the broadcast table does not contain this rank's own record: sent {own:?}, got {echoed:?}"
        )));
    }
    Ok(table)
}

/// Resolves a control-plane address (host name or IP) plus the port.
fn resolve_address(text: &str, port: u16) -> Result<SocketAddr, CommError> {
    let mut addresses = (text, port).to_socket_addrs().map_err(|error| {
        CommError::InvalidLayout(format!(
            "control-plane address {text:?} cannot be resolved: {error}"
        ))
    })?;
    addresses.next().ok_or_else(|| {
        CommError::InvalidLayout(format!(
            "control-plane address {text:?} resolves to no socket address"
        ))
    })
}

/// Connects to rank 0, retrying until the deadline because node startup order is not fixed.
fn connect_retry(
    address: SocketAddr,
    deadline: Instant,
    timeout: Duration,
) -> Result<TcpStream, CommError> {
    loop {
        match TcpStream::connect_timeout(&address, timeout.min(CONNECT_RETRY + CONNECT_RETRY)) {
            Ok(stream) => return Ok(stream),
            Err(error) => {
                if Instant::now() >= deadline {
                    return Err(CommError::timeout(
                        timeout,
                        format!("connecting to the control-plane root at {address}: {error}"),
                    ));
                }
                std::thread::sleep(CONNECT_RETRY);
            }
        }
    }
}

/// Accepts one connection, retrying until the deadline.
fn accept_retry(
    listener: &TcpListener,
    deadline: Instant,
    timeout: Duration,
) -> Result<TcpStream, CommError> {
    loop {
        match listener.accept() {
            Ok((stream, peer)) => {
                let _ = peer;
                return Ok(stream);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(CommError::timeout(
                        timeout,
                        "waiting for a control-plane peer to reach rank 0".to_owned(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Writes one length-prefixed control message.
fn write_message(stream: &mut TcpStream, bytes: &[u8]) -> Result<(), CommError> {
    let length = u32::try_from(bytes.len()).map_err(|_| {
        CommError::Protocol(format!(
            "control message of {} bytes cannot be length-prefixed",
            bytes.len()
        ))
    })?;
    stream.write_all(&length.to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}

/// Reads one length-prefixed control message, refusing absurd lengths.
///
/// A socket read that times out becomes [`CommError::Timeout`] with the control-plane deadline,
/// not [`CommError::Io`]: the caller promised a timeout at its API boundary.
fn read_message(stream: &mut TcpStream, timeout: Duration) -> Result<Vec<u8>, CommError> {
    let mut length_bytes = [0u8; 4];
    read_exact(stream, &mut length_bytes, timeout)?;
    let length = u32::from_le_bytes(length_bytes) as usize;
    if length == 0 || length > MAX_CONTROL_MESSAGE_BYTES {
        return Err(CommError::Protocol(format!(
            "control message length {length} is outside 1..={MAX_CONTROL_MESSAGE_BYTES}"
        )));
    }
    let mut bytes = vec![0u8; length];
    read_exact(stream, &mut bytes, timeout)?;
    Ok(bytes)
}

/// Reads exactly `bytes`, mapping a socket timeout onto the transport's timeout vocabulary.
fn read_exact(
    stream: &mut TcpStream,
    bytes: &mut [u8],
    timeout: Duration,
) -> Result<(), CommError> {
    use std::io::Read as _;
    stream
        .read_exact(bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                CommError::timeout(timeout, "reading a control-plane message")
            }
            _ => CommError::Io(error),
        })
}

/// Keeps a timeout a timeout while adding context to every other control-plane error.
fn control_error(error: CommError, context: String) -> CommError {
    match error {
        CommError::Timeout { .. } => error,
        other => CommError::Protocol(format!("{context}: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid endpoint record for a rank in a world.
    fn endpoint(rank: usize, world: usize) -> RankEndpoints {
        let rail = |rail_index: usize| {
            let mut raw = [0u8; 16];
            raw[10] = 0xff;
            raw[11] = 0xff;
            raw[12] = 192;
            raw[13] = 168;
            raw[14] = 1;
            raw[15] = 80 + rank as u8;
            RailEndpoint {
                gid: IbvGid { raw },
                gid_index: 3,
                active_mtu: mtu::MTU_1024 as u16,
                mr_addr: 0x7f00_0000_0000 + (rank as u64) * 0x10_0000 + rail_index as u64,
                rkey: 0x4000 + (rank as u32) * 2 + rail_index as u32,
                qpns: (0..world)
                    .map(|dest| 0x100 + (rank * world + dest) as u32)
                    .collect(),
            }
        };
        RankEndpoints {
            rank,
            frame_cap: 2 * 1024 * 1024,
            slot_bytes: 24 + 2 * 1024 * 1024,
            rail_count: 2,
            rails: [rail(0), rail(1)],
        }
    }

    /// A complete table for a world of that size.
    fn table(world: usize) -> QpTable {
        QpTable {
            world,
            entries: (0..world).map(|rank| endpoint(rank, world)).collect(),
        }
    }

    #[test]
    fn round_trip_preserves_every_endpoint() {
        let original = table(4);
        let bytes = original
            .encode()
            .unwrap_or_else(|error| panic!("encode: {error}"));
        assert_eq!(
            bytes.len() - QPINFO_HEADER_BYTES,
            8 + 4 * QpTable::rank_record_bytes(4)
        );
        let decoded = QpTable::decode(&bytes, 4).unwrap_or_else(|error| panic!("decode: {error}"));
        assert_eq!(decoded, original);
        assert_eq!(
            decoded.rank(2).unwrap_or_else(|error| panic!("{error}")),
            &endpoint(2, 4)
        );
        assert!(matches!(
            decoded.rank(9),
            Err(CommError::RankOutOfRange { .. })
        ));
    }

    #[test]
    fn a_single_record_round_trips_through_the_star_message() {
        let own = endpoint(2, 4);
        let bytes = encode_rank_message(4, &own).unwrap_or_else(|error| panic!("encode: {error}"));
        let decoded =
            decode_rank_message(&bytes, 4).unwrap_or_else(|error| panic!("decode: {error}"));
        assert_eq!(decoded, own);
        // The same bytes are not a complete table: rank 0's decoder must refuse them.
        assert!(matches!(
            QpTable::decode(&bytes, 4),
            Err(CommError::Protocol(_))
        ));
    }

    #[test]
    fn decode_refuses_a_table_that_is_damaged_or_mismatched() {
        let good = table(4)
            .encode()
            .unwrap_or_else(|error| panic!("encode: {error}"));

        let mut truncated = good.clone();
        truncated.pop();
        assert!(matches!(
            QpTable::decode(&truncated, 4),
            Err(CommError::Protocol(_))
        ));

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xff;
        match QpTable::decode(&bad_magic, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains("magic"), "{text}"),
            other => panic!("bad magic must be refused, got {other:?}"),
        }

        let mut bad_version = good.clone();
        bad_version[4] = 9;
        match QpTable::decode(&bad_version, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains("version"), "{text}"),
            other => panic!("bad version must be refused, got {other:?}"),
        }

        match QpTable::decode(&good, 8) {
            Err(CommError::Protocol(text)) => assert!(text.contains("world"), "{text}"),
            other => panic!("a world mismatch must be refused, got {other:?}"),
        }

        // The all-zero GID is refused. A record is: header 12 + world/count 8 + rank 4 +
        // frame cap 4 + slot bytes 8 + rail count 1 + reserved 3, so rank 0 rail 0's GID
        // occupies bytes 40..56 (the ::ffff:198.51.100.80 form).
        assert_eq!(good[50], 0xff);
        let mut zero_gid = good.clone();
        for byte in &mut zero_gid[40..56] {
            *byte = 0;
        }
        match QpTable::decode(&zero_gid, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains("GID"), "{text}"),
            other => panic!("an all-zero GID must be refused, got {other:?}"),
        }

        // The first QP number of rank 0 rail 0 starts at byte 71 and is refused when zeroed.
        let first_qpn = 12 + 8 + 4 + 4 + 8 + 1 + 3 + 16 + 1 + 2 + 8 + 4;
        let mut zero_qpn = good.clone();
        for byte in &mut zero_qpn[first_qpn..first_qpn + 4] {
            *byte = 0;
        }
        match QpTable::decode(&zero_qpn, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains("QP number"), "{text}"),
            other => panic!("a zero QP number must be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_single_rail_record_round_trips_and_publishes_its_geometry() {
        let mut own = endpoint(1, 4);
        own.rail_count = 1;
        own.rails[1] = RankEndpoints::unused_rail(4);
        let bytes = encode_rank_message(4, &own).unwrap_or_else(|error| panic!("encode: {error}"));
        let decoded =
            decode_rank_message(&bytes, 4).unwrap_or_else(|error| panic!("decode: {error}"));
        assert_eq!(decoded, own);
        assert_eq!(decoded.rail_count, 1);
        assert!(decoded.rail(0).is_some());
        assert!(decoded.rail(1).is_none());
        // A non-zero record for an unopened rail is refused before it reaches the wire.
        let mut bad = own.clone();
        bad.rails[1] = endpoint(1, 4).rails[0].clone();
        assert!(matches!(
            encode_rank_message(4, &bad),
            Err(CommError::Protocol(_))
        ));
    }

    #[test]
    fn decode_refuses_a_table_whose_ranks_disagree_on_geometry() {
        // Encoding the mismatched table is already refused.
        let mut mismatched = table(4);
        mismatched.entries[2].frame_cap += 2;
        assert!(matches!(mismatched.encode(), Err(CommError::Protocol(_))));
        // Patch the wire form of a good table and check the decoder refuses it too.
        let good = table(4)
            .encode()
            .unwrap_or_else(|error| panic!("encode: {error}"));
        let record = QpTable::rank_record_bytes(4);
        let frame_cap_offset = 12 + 8 + 2 * record + 4;
        let mut patched = good.clone();
        patched[frame_cap_offset..frame_cap_offset + 4]
            .copy_from_slice(&(2 * 1024 * 1024u32 + 2).to_le_bytes());
        match QpTable::decode(&patched, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains("geometry"), "{text}"),
            other => panic!("a geometry mismatch must be refused, got {other:?}"),
        }
    }

    #[test]
    fn decode_refuses_invalid_geometry_and_data_for_an_unopened_rail() {
        // Offsets inside rank 0's record: header 12 + world/count 8, then rank (4), frame cap
        // (4), slot bytes (8), rail count (1) + reserved (3), rail 0 (47 bytes for world 4:
        // GID 16, gid index 1, MTU 2, MR address 8, rkey 4, four QP numbers), then rail 1.
        let good = table(4)
            .encode()
            .unwrap_or_else(|error| panic!("encode: {error}"));
        let base = 12 + 8;
        let patch = |bytes: &mut Vec<u8>, at: usize, value: &[u8]| {
            bytes[at..at + value.len()].copy_from_slice(value);
        };
        let refuse = |bytes: &[u8], needle: &str| match QpTable::decode(bytes, 4) {
            Err(CommError::Protocol(text)) => assert!(text.contains(needle), "{text}"),
            other => panic!("a damaged record must be refused, got {other:?}"),
        };

        let mut frame_cap_zero = good.clone();
        patch(&mut frame_cap_zero, base + 4, &0u32.to_le_bytes());
        // The needle must be the rank-0 record's own message: the later "every rank must agree"
        // error also contains the words "frame cap", so a loose needle stays green even when the
        // frame_cap == 0 half alone is disabled.
        refuse(&frame_cap_zero, "publishes frame cap 0 and slot size");

        let mut slot_too_small = good.clone();
        patch(
            &mut slot_too_small,
            base + 8,
            &(2 * 1024 * 1024u64 - 1).to_le_bytes(),
        );
        refuse(
            &slot_too_small,
            "publishes frame cap 2097152 and slot size 2097151: a slot must hold",
        );

        let mut zero_rails = good.clone();
        patch(&mut zero_rails, base + 16, &[0u8]);
        refuse(&zero_rails, "rail_count 0");

        let mut unknown_mtu = good.clone();
        patch(&mut unknown_mtu, base + 37, &0u16.to_le_bytes());
        refuse(&unknown_mtu, "unknown MTU");

        let mut null_mr = good.clone();
        patch(&mut null_mr, base + 39, &0u64.to_le_bytes());
        refuse(&null_mr, "null staging address");

        // A rail-1 byte published by a rank that opened only rail 0 is refused.
        let mut single_table = table(4);
        for entry in &mut single_table.entries {
            entry.rail_count = 1;
            entry.rails[1] = RankEndpoints::unused_rail(4);
        }
        let single = single_table
            .encode()
            .unwrap_or_else(|error| panic!("encode: {error}"));
        let mut polluted = single.clone();
        patch(&mut polluted, base + 67, &[1u8]);
        refuse(&polluted, "rail 1");
    }

    #[test]
    fn the_star_exchange_gives_every_rank_the_merged_table() {
        // Find a free port by binding one, then release it for rank 0 to take.
        let probe = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|error| panic!("{error}"));
        let port = probe
            .local_addr()
            .unwrap_or_else(|error| panic!("{error}"))
            .port();
        drop(probe);

        let mut handles = Vec::new();
        for rank in 0..4_usize {
            handles.push(std::thread::spawn(move || {
                let config = ControlConfig {
                    rank,
                    world: 4,
                    peers: vec!["127.0.0.1".to_owned(); 4],
                    port,
                    timeout: Duration::from_secs(10),
                };
                let own = endpoint(rank, 4);
                let table = exchange_endpoints(&config, own.clone())
                    .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
                (rank, own, table)
            }));
        }
        for handle in handles {
            let (rank, own, table) = handle.join().unwrap_or_else(|_| panic!("a rank panicked"));
            assert_eq!(table.world, 4);
            assert_eq!(table.entries.len(), 4);
            assert_eq!(
                table.rank(rank).unwrap_or_else(|error| panic!("{error}")),
                &own
            );
            for peer in 0..4 {
                assert_eq!(
                    table.rank(peer).unwrap_or_else(|error| panic!("{error}")),
                    &endpoint(peer, 4)
                );
            }
        }
    }
}
