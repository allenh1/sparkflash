//! Slots, work-request plans and the schedule-driver half of the dual-rail RDMA backend.
//!
//! This module is split so the part that can be tested without hardware is separate from the part
//! that talks to libibverbs:
//!
//! * the slot layout, the rail half split, the immediate-data codec, the work-request tag codec,
//!   the write plan for one frame and the three QP state-transition plans are pure functions over
//!   integers. Their unit tests pin the arithmetic and the ABI field assignment;
//! * [`DataPlane`] owns the real CQs, MRs and QPs and implements the [`Exchange`] trait the schedule
//!   driver calls. It cannot run on the x86 development box (no libibverbs, no rails); it is
//!   exercised only by the hardware gates.
//!
//! Wire layout of one frame in a slot (both directions use the same layout):
//!
//! `text
//! slot base +  0: 24-byte tcp::FrameHeader, written by rail 0 only
//! slot base + 24: first rail half of the bf16 payload (little-endian), rail 0
//! slot + 24 + rail0_bytes: second rail half, rail 1
//! `
//!
//! The header carries the full frame length, so the receiver validates the split instead of
//! trusting it: rail 0's completion must cover exactly `24 + rail0_bytes` and rail 1's exactly
//! `rail1_bytes`, and the sum must equal the header's declared payload. The immediate value carries
//! a short copy of (sender, phase, kind, sequence) so a completion can be attributed before its
//! slot is read; the header is the authority and a disagreement is a protocol error.
//!
//! Slot reuse: every (peer, phase) has its own inbox slot and its own send-staging slot. A send
//! slot is reused only after its signalled completion drained in [`Exchange::collect`], and an inbox
//! slot only after the receive completion was consumed, so a slow peer can never overwrite a frame
//! the schedule still expects.

use std::ffi::c_int;
use std::time::{Duration, Instant};

use super::abi::{
    IbvQpAttr, IbvQpTypeUnion, IbvRdmaWr, IbvRecvWr, IbvSendWr, IbvSge, IbvWc, IbvWrUnion,
    attr_mask, qp_state, send_flags, wc, wr_opcode,
};
use crate::error::CommError;
use crate::exchange::{Exchange, ExchangeKind, ExchangeTag, Received};
use crate::rail::RailLedger;
use crate::tcp::{FRAME_HEADER_BYTES, FrameHeader, FrameKind};

/// Bit 63 of a work-request id marks a receive completion; sends leave it clear.
pub(crate) const RECEIVE_TAG_BIT: u64 = 1 << 63;
/// Phases per collective: 0 is the one-shot or reduce-scatter phase, 1 is the hairpin all-gather.
pub(crate) const PHASES: usize = 2;
/// Sequence number both sides of every QP start from. A constant is legal because each QP pair is
/// used for exactly one direction of this backend's traffic and both ends agree on it.
pub(crate) const PSN: u32 = 0x0000_0001;

/// Byte split of one frame across the rails (payload only; the header is always rail 0's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RailHalves {
    /// Payload bytes carried on rail 0.
    pub(crate) rail0_bytes: usize,
    /// Payload bytes carried on rail 1.
    pub(crate) rail1_bytes: usize,
    /// bf16 elements carried on rail 0.
    pub(crate) rail0_elements: usize,
    /// bf16 elements carried on rail 1.
    pub(crate) rail1_elements: usize,
}

/// Splits `elements` bf16 elements half/half, or puts all of them on rail 0 for single-rail mode.
///
/// # Errors
/// Returns [`CommError::InvalidLayout`] when `rail_count` is not 1 or 2 or the byte count
/// overflows.
pub(crate) fn rail_halves(elements: usize, rail_count: usize) -> Result<RailHalves, CommError> {
    let bytes = elements.checked_mul(2).ok_or_else(|| {
        CommError::InvalidLayout(format!("{elements} bf16 elements overflow a byte count"))
    })?;
    match rail_count {
        1 => Ok(RailHalves {
            rail0_bytes: bytes,
            rail1_bytes: 0,
            rail0_elements: elements,
            rail1_elements: 0,
        }),
        2 => {
            // Split whole elements, so each rail's byte count is even: a bare byte split could
            // cut one bf16 element in half, and then neither rail's completion would decode.
            let rail0_elements = elements - elements / 2;
            let rail1_elements = elements / 2;
            Ok(RailHalves {
                rail0_bytes: rail0_elements * 2,
                rail1_bytes: rail1_elements * 2,
                rail0_elements,
                rail1_elements,
            })
        }
        other => Err(CommError::InvalidLayout(format!(
            "rail_count must be 1 or 2, got {other}"
        ))),
    }
}

/// Byte layout of one rail's registered staging buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotLayout {
    world: usize,
    max_frame_bytes: usize,
    slot_bytes: usize,
    inbox_base: usize,
    send_base: usize,
    total_bytes: usize,
}

impl SlotLayout {
    /// Builds the layout for a world and a maximum frame payload.
    ///
    /// # Errors
    /// Returns [`CommError::InvalidLayout`] for world 0, a zero or oversized frame, or an
    /// allocation that would overflow.
    pub(crate) fn new(world: usize, max_frame_bytes: usize) -> Result<Self, CommError> {
        if world == 0 {
            return Err(CommError::InvalidLayout(
                "an RDMA staging layout needs at least one rank".to_owned(),
            ));
        }
        if max_frame_bytes == 0 {
            return Err(CommError::InvalidLayout(
                "an RDMA staging layout needs a positive frame size".to_owned(),
            ));
        }
        if u32::try_from(max_frame_bytes).is_err() {
            return Err(CommError::InvalidLayout(format!(
                "a {max_frame_bytes}-byte frame does not fit a 32-bit scatter/gather length"
            )));
        }
        // The slot holds the header plus the WHOLE frame, not the larger rail half: a
        // single-rail run puts the entire payload on rail 0, so a half-sized slot would overflow
        // fill_send_slot and cut the receive SGE short. The layout is rail-count independent on
        // purpose, so both ends of the connection derive the same slot size.
        let slot_bytes = FRAME_HEADER_BYTES
            .checked_add(max_frame_bytes)
            .ok_or_else(|| CommError::InvalidLayout("slot size overflow".to_owned()))?;
        let inbox_base = 0;
        let inbox_bytes = world
            .checked_mul(PHASES)
            .and_then(|count| count.checked_mul(slot_bytes))
            .ok_or_else(|| CommError::InvalidLayout("inbox size overflow".to_owned()))?;
        let send_base = inbox_base + inbox_bytes;
        let total_bytes = send_base
            .checked_add(inbox_bytes)
            .ok_or_else(|| CommError::InvalidLayout("staging size overflow".to_owned()))?;
        Ok(Self {
            world,
            max_frame_bytes,
            slot_bytes,
            inbox_base,
            send_base,
            total_bytes,
        })
    }

    /// Bytes reserved for one slot (header plus the whole frame).
    pub(crate) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Total bytes the layout needs, including both slot regions.
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Largest frame payload the layout accepts.
    pub(crate) fn max_frame_bytes(&self) -> usize {
        self.max_frame_bytes
    }

    /// Offset of the inbox slot a frame from `sender` in `phase` lands in.
    pub(crate) fn inbox_offset(&self, sender: usize, phase: u8) -> usize {
        self.inbox_base + (sender * PHASES + usize::from(phase % 2)) * self.slot_bytes
    }

    /// Offset of the staging slot used to send to `peer` in `phase`.
    pub(crate) fn send_offset(&self, peer: usize, phase: u8) -> usize {
        self.send_base + (peer * PHASES + usize::from(phase % 2)) * self.slot_bytes
    }

    /// Whether `peer` and `phase` name an existing slot.
    pub(crate) fn holds(&self, peer: usize, phase: u8) -> bool {
        peer < self.world && usize::from(phase) < PHASES
    }
}

/// Slowest path-MTU byte count out of two negotiated `enum ibv_mtu` values.
///
/// # Errors
/// Returns [`CommError::Verbs`] when either side reports a value this ABI does not know.
pub(crate) fn min_path_mtu(local: c_int, peer: c_int) -> Result<c_int, CommError> {
    let local_bytes = super::abi::mtu::bytes(local).ok_or_else(|| {
        CommError::verbs(
            "ibv_query_port",
            format!("active_mtu {} is not an enum ibv_mtu value", local),
        )
    })?;
    let peer_bytes = super::abi::mtu::bytes(peer).ok_or_else(|| {
        CommError::verbs(
            "ibv_query_port",
            format!(
                "the peer's active_mtu {} is not an enum ibv_mtu value",
                peer
            ),
        )
    })?;
    let smallest = local_bytes.min(peer_bytes);
    [
        super::abi::mtu::MTU_256,
        super::abi::mtu::MTU_512,
        super::abi::mtu::MTU_1024,
        super::abi::mtu::MTU_2048,
        super::abi::mtu::MTU_4096,
    ]
    .into_iter()
    .find(|value| super::abi::mtu::bytes(*value) == Some(smallest))
    .ok_or_else(|| CommError::verbs("ibv_modify_qp", "no MTU enum matches the negotiated size"))
}

/// Encodes the short copy of (sender, phase, kind, sequence) carried as the immediate value.
pub(crate) fn imm_encode(from: usize, phase: u8, kind: ExchangeKind, seq: u64) -> u32 {
    let kind_bit: u32 = match kind {
        ExchangeKind::Reduce => 0,
        ExchangeKind::Gather => 1,
    };
    ((from as u32 & 0xff) << 24)
        | (u32::from(phase & 1) << 16)
        | (kind_bit << 8)
        | ((seq & 0xff) as u32)
}

/// Decodes the immediate value into (sender, phase, kind, low sequence byte).
pub(crate) fn imm_decode(imm: u32) -> (usize, u8, u8, u8) {
    (
        ((imm >> 24) & 0xff) as usize,
        ((imm >> 16) & 0xff) as u8,
        ((imm >> 8) & 0xff) as u8,
        (imm & 0xff) as u8,
    )
}

/// Checks the immediate a completed frame carries against the frame the phase expects.
///
/// `wire_imm` is the value in the CQE, i.e. the posted immediate in network byte order. The
/// comparison happens in host order and performs exactly ONE byte-order conversion: the plan
/// stores `imm_encode(..).to_be()`, so byte-swapping that stored value again would compare network
/// bytes with host bytes and reject every non-palindromic immediate (for example rank 2, phase 0,
/// Reduce, seq 1: host `0x0200_0001` travels as `0x0100_0002`).
fn check_immediate(
    wire_imm: u32,
    sender: usize,
    phase: u8,
    kind: ExchangeKind,
    seq: u64,
) -> Result<(), CommError> {
    let host_imm = u32::from_be(wire_imm);
    let expected_host_imm = imm_encode(sender, phase, kind, seq);
    let (imm_from, imm_phase, imm_kind, imm_seq) = imm_decode(host_imm);
    let expected_kind_bit = u8::from(kind == ExchangeKind::Gather);
    if host_imm != expected_host_imm
        || imm_from != sender
        || imm_phase != phase
        || imm_kind != expected_kind_bit
        || imm_seq != (seq & 0xff) as u8
    {
        return Err(CommError::Protocol(format!(
            "frame from rank {sender} carries immediate {:#x} (from {imm_from}, phase {imm_phase}, kind {imm_kind}, seq {imm_seq}), expected {expected_host_imm:#x}",
            host_imm
        )));
    }
    Ok(())
}

/// Which slot a completion refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WrTag {
    /// An inbox slot on one rail, for a frame from `peer` in `phase`.
    Receive {
        /// Rail index.
        rail: usize,
        /// Sending rank.
        peer: usize,
        /// Collective phase.
        phase: u8,
    },
    /// A send-staging slot on one rail.
    Send {
        /// Rail index.
        rail: usize,
    },
}

/// Encodes a work-request id.
pub(crate) fn tag_encode(tag: WrTag) -> u64 {
    match tag {
        WrTag::Receive { rail, peer, phase } => {
            RECEIVE_TAG_BIT
                | (u64::from(phase) << 48)
                | (u64::from(rail as u8) << 56)
                | ((peer as u64) & 0xff)
        }
        WrTag::Send { rail } => u64::from(rail as u8) << 56,
    }
}

/// Decodes a work-request id. `peer` is taken from the low byte.
pub(crate) fn tag_decode(wr_id: u64) -> WrTag {
    let rail = ((wr_id >> 56) & 0x0f) as usize;
    if wr_id & RECEIVE_TAG_BIT != 0 {
        WrTag::Receive {
            rail,
            peer: (wr_id & 0xff) as usize,
            phase: ((wr_id >> 48) & 0xff) as u8,
        }
    } else {
        WrTag::Send { rail }
    }
}

/// One planned RDMA write for one rail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WritePlan {
    /// Rail index.
    pub(crate) rail: usize,
    /// Peer's staging-buffer address plus the inbox slot offset.
    pub(crate) remote_addr: u64,
    /// Offset inside this rank's staging buffer the write reads from.
    pub(crate) local_offset: usize,
    /// Bytes posted: the header plus the rail half on rail 0, the second half on rail 1.
    pub(crate) bytes: usize,
    /// Remote key the peer published for its staging buffer.
    pub(crate) rkey: u32,
    /// Immediate value carried by the write.
    pub(crate) imm: u32,
}

/// Plans the writes for one frame: rail 0 first, rail 1 only when it carries bytes.
///
/// Eight arguments is what the frame identity and the peer's two keys are; the alternative would be
/// a wrapper struct that exists only to satisfy the lint, so the lint is allowed here instead.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_writes(
    slot: &SlotLayout,
    halves: &RailHalves,
    rank: usize,
    phase: u8,
    tag: ExchangeTag,
    local_send_offset: usize,
    peer_mr: [u64; 2],
    peer_rkey: [u32; 2],
) -> Vec<WritePlan> {
    // The C union is __be32: the immediate travels in network byte order.
    let imm = imm_encode(rank, phase, tag.kind, tag.seq).to_be();
    let inbox = slot.inbox_offset(rank, phase);
    let rail0_bytes = FRAME_HEADER_BYTES + halves.rail0_bytes;
    let mut plans = vec![WritePlan {
        rail: 0,
        remote_addr: peer_mr[0] + inbox as u64,
        local_offset: local_send_offset,
        bytes: rail0_bytes,
        rkey: peer_rkey[0],
        imm,
    }];
    if halves.rail1_bytes > 0 {
        plans.push(WritePlan {
            rail: 1,
            remote_addr: peer_mr[1] + (inbox + rail0_bytes) as u64,
            local_offset: local_send_offset + rail0_bytes,
            bytes: halves.rail1_bytes,
            rkey: peer_rkey[1],
            imm,
        });
    }
    plans
}

/// A zeroed `ibv_qp_attr`: every field is an integer or an integer array, so zero is a valid
/// value and the transitions below set only the fields their mask announces.
fn zeroed_attr() -> IbvQpAttr {
    // SAFETY: IbvQpAttr contains only integers, arrays of integers, and the AhAttr/Grh structs,
    // all of which are valid when zeroed; the layout is asserted in abi.rs.
    unsafe { std::mem::zeroed() }
}

/// The INIT transition: state, P_Key index, port and the access flags a peer's RDMA write needs.
#[must_use]
pub(crate) fn init_attr(port: u8) -> (IbvQpAttr, c_int) {
    let mut attr = zeroed_attr();
    attr.qp_state = qp_state::INIT;
    attr.pkey_index = 0;
    attr.port_num = port;
    attr.qp_access_flags = super::abi::access::LOCAL_WRITE
        | crate::rdma::abi::access::REMOTE_WRITE
        | super::abi::access::REMOTE_READ;
    (
        attr,
        attr_mask::STATE | attr_mask::PKEY_INDEX | attr_mask::PORT | attr_mask::ACCESS_FLAGS,
    )
}

/// The RTR transition: peer QP number, peer GID, path MTU, PSN and RNR retry budget.
#[must_use]
pub(crate) fn rtr_attr(
    port: u8,
    peer_qpn: u32,
    peer_gid: super::abi::IbvGid,
    gid_index: u8,
    path_mtu: c_int,
) -> (IbvQpAttr, c_int) {
    let mut attr = zeroed_attr();
    attr.qp_state = qp_state::RTR;
    attr.path_mtu = path_mtu;
    attr.dest_qp_num = peer_qpn;
    attr.rq_psn = PSN;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.grh.dgid = peer_gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.static_rate = 0;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = port;
    (
        attr,
        attr_mask::STATE
            | attr_mask::PATH_MTU
            | attr_mask::DEST_QPN
            | attr_mask::RQ_PSN
            | attr_mask::MAX_DEST_RD_ATOMIC
            | attr_mask::MIN_RNR_TIMER
            | attr_mask::AV,
    )
}

/// The RTS transition: send PSN, ACK timeout and transport/RNR retry counts.
#[must_use]
pub(crate) fn rts_attr() -> (IbvQpAttr, c_int) {
    let mut attr = zeroed_attr();
    attr.qp_state = qp_state::RTS;
    attr.sq_psn = PSN;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    (
        attr,
        attr_mask::STATE
            | attr_mask::SQ_PSN
            | attr_mask::TIMEOUT
            | attr_mask::RETRY_CNT
            | attr_mask::RNR_RETRY
            | attr_mask::MAX_QP_RD_ATOMIC,
    )
}

/// Builds one send work request for a planned write.
fn send_wr(plan: &WritePlan, sge: &mut IbvSge) -> IbvSendWr {
    IbvSendWr {
        wr_id: tag_encode(WrTag::Send { rail: plan.rail }),
        next: std::ptr::null_mut(),
        sg_list: sge as *mut IbvSge,
        num_sge: 1,
        opcode: wr_opcode::RDMA_WRITE_WITH_IMM,
        send_flags: send_flags::SIGNALED,
        imm_data: plan.imm,
        wr: IbvWrUnion {
            rdma: IbvRdmaWr {
                remote_addr: plan.remote_addr,
                rkey: plan.rkey,
            },
        },
        qp_type: IbvQpTypeUnion { raw: [0u8; 4] },
        padding: [0u8; 4],
        tail: super::abi::IbvSendWrTail { raw: [0u8; 48] },
    }
}

/// Builds the single-SGE receive work request that arms one inbox slot.
fn recv_wr(wr_id: u64, sge: &mut IbvSge) -> IbvRecvWr {
    IbvRecvWr {
        wr_id,
        next: std::ptr::null_mut(),
        sg_list: sge as *mut IbvSge,
        num_sge: 1,
        reserved: 0,
    }
}

/// A peer's endpoint on one rail, as published by the control-plane exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RemoteRail {
    /// The peer's GID on this rail.
    pub(crate) gid: super::abi::IbvGid,
    /// The peer's GID table index.
    pub(crate) gid_index: u8,
    /// The peer's negotiated `enum ibv_mtu`.
    pub(crate) active_mtu: c_int,
    /// Base address of the peer's staging buffer on this rail.
    pub(crate) mr_addr: u64,
    /// Remote key authorising writes into the peer's staging buffer.
    pub(crate) rkey: u32,
}

/// One rail's resources, built by the bring-up in rdma.rs and consumed here.
///
/// Field order is load-bearing: Rust drops fields in declaration order, so the QPs are destroyed
/// before the CQs and PD they reference, the MR before its PD, and the device context last.
pub(crate) struct ReadyRail {
    /// `@ibv_get_device_name` of the opened device.
    pub(crate) name: String,
    /// Netdev behind the device, for the bring-up report.
    pub(crate) netdev: String,
    /// This rank's GID on this rail.
    pub(crate) gid: super::abi::IbvGid,
    /// GID table index this rank uses.
    pub(crate) gid_index: u8,
    /// This port's negotiated `enum ibv_mtu`.
    pub(crate) active_mtu: c_int,
    /// Bounded posted QP numbers: `qps[d]` connects to rank `d`.
    pub(crate) qps: Vec<super::verbs::Qp>,
    /// One MR over `staging`. Declared BEFORE `staging` so the MR is deregistered before the
    /// memory it covers is freed: Rust drops fields in declaration order.
    pub(crate) mr: super::verbs::Mr,
    /// The registered staging buffer; the MR above covers exactly this memory.
    pub(crate) staging: Box<[u8]>,
    /// One CQ for every QP of this rail.
    pub(crate) cq: super::verbs::Cq,
    /// Protection domain of the MR and the QPs.
    pub(crate) _pd: super::verbs::Pd,
    /// Every peer's endpoint on this rail, indexed by rank (this rank's own entry is unused).
    pub(crate) remotes: Vec<RemoteRail>,
    /// Open device context; declared last so everything above is destroyed first.
    pub(crate) _context: super::verbs::Context,
}

/// The state of one collective phase.
struct PhaseState {
    /// The tag the schedule driver announced.
    tag: ExchangeTag,
    /// (sender, elements) for every expected frame, in the driver's order.
    expected: Vec<(usize, usize)>,
    /// Receive byte counts per expected frame and rail.
    arrivals: Vec<[Option<u32>; 2]>,
    /// Frames already assembled.
    frames: Vec<Option<Received>>,
}

impl PhaseState {
    fn assembled_count(&self) -> usize {
        self.frames.iter().filter(|frame| frame.is_some()).count()
    }
}

/// The live half of the RDMA backend: CQs, MRs, QPs and the state of the current phase.
pub(crate) struct DataPlane {
    world: usize,
    rank: usize,
    rail_count: usize,
    slot: SlotLayout,
    timeout: Duration,
    rails: Vec<ReadyRail>,
    phase: Option<PhaseState>,
    pending_sends: usize,
    ledger: RailLedger,
}

// SAFETY: every field is either a plain integer, a Box, or a libibverbs object. The backend wraps
// the whole data plane in a Mutex, so exactly one thread drives these QPs at a time; libibverbs
// documents that distinct QPs may be used concurrently and a single QP must not be, which the
// mutex guarantees. The raw pointers are never used after the owning wrappers drop.
unsafe impl Send for DataPlane {}

impl DataPlane {
    /// Validates a set of ready rails and builds the data plane.
    ///
    /// # Errors
    /// Returns [`CommError::InvalidLayout`] when the rail count, world, slot size or resource
    /// counts disagree, and [`CommError::Unsupported`] when a registered MR does not cover the
    /// staging buffer it was created for (which would make every remote address wrong).
    pub(crate) fn new(
        world: usize,
        rank: usize,
        slot: SlotLayout,
        rail_count: usize,
        timeout: Duration,
        rails: Vec<ReadyRail>,
    ) -> Result<Self, CommError> {
        if world == 0 || rank >= world {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: world,
            });
        }
        if rails.len() != rail_count || !matches!(rail_count, 1 | 2) {
            return Err(CommError::InvalidLayout(format!(
                "the data plane was given {} rails but rail_count is {rail_count}",
                rails.len()
            )));
        }
        for (index, rail) in rails.iter().enumerate() {
            if rail.qps.len() != world || rail.remotes.len() != world {
                return Err(CommError::InvalidLayout(format!(
                    "rail {index} has {} QPs and {} peer endpoints for world {world}",
                    rail.qps.len(),
                    rail.remotes.len()
                )));
            }
            if rail.staging.len() != slot.total_bytes() {
                return Err(CommError::InvalidLayout(format!(
                    "rail {index} has a {} -byte staging buffer but the slot layout needs {}",
                    rail.staging.len(),
                    slot.total_bytes()
                )));
            }
            let registered = rail.mr.addr();
            let expected = rail.staging.as_ptr() as u64;
            if registered != expected {
                return Err(CommError::Unsupported(format!(
                    "rail {index} registered MR address {registered:#x} but its staging buffer is at {expected:#x}: every peer would write to the wrong address"
                )));
            }
        }
        Ok(Self {
            world,
            rank,
            rail_count,
            slot,
            timeout,
            rails,
            phase: None,
            pending_sends: 0,
            ledger: RailLedger::default(),
        })
    }

    /// Measured posted-byte ledger, accumulated across collectives since the last reset.
    pub(crate) fn ledger(&self) -> RailLedger {
        self.ledger
    }

    /// Clears the measured ledger, for a gate that wants one dispatch's bytes.
    pub(crate) fn reset_ledger(&mut self) {
        self.ledger = RailLedger::default();
    }

    /// One description per rail: the device name and the netdev behind it.
    pub(crate) fn rail_names(&self) -> Vec<String> {
        self.rails
            .iter()
            .map(|rail| format!("{} ({})", rail.name, rail.netdev))
            .collect()
    }

    /// Writes the frame header and payload halves into a send slot.
    fn fill_send_slot(
        &mut self,
        offset: usize,
        tag: ExchangeTag,
        payload: &[u16],
        halves: &RailHalves,
    ) -> Result<(), CommError> {
        let bytes = payload.len() * 2;
        let header = FrameHeader {
            kind: match tag.kind {
                ExchangeKind::Reduce => FrameKind::ReducePayload,
                ExchangeKind::Gather => FrameKind::GatherPayload,
            },
            phase: tag.phase,
            seq: tag.seq,
            rank: self.rank as u32,
            payload_bytes: bytes as u32,
        };
        let staging = &mut self.rails[0].staging;
        let header_end = offset + FRAME_HEADER_BYTES;
        if header_end + halves.rail0_bytes > staging.len() {
            return Err(CommError::InvalidLayout(format!(
                "send slot offset {offset} plus {} rail-0 bytes does not fit a {}-byte staging buffer",
                FRAME_HEADER_BYTES + halves.rail0_bytes,
                staging.len()
            )));
        }
        staging[offset..header_end].copy_from_slice(&header.encode());
        let mut cursor = header_end;
        for value in &payload[..halves.rail0_elements] {
            staging[cursor..cursor + 2].copy_from_slice(&value.to_le_bytes());
            cursor += 2;
        }
        if halves.rail1_bytes > 0 {
            // The second half lives on rail 1's staging buffer at the same relative position.
            let staging = &mut self.rails[1].staging;
            let start = offset + FRAME_HEADER_BYTES + halves.rail0_bytes;
            if start + halves.rail1_bytes > staging.len() {
                return Err(CommError::InvalidLayout(format!(
                    "send slot offset {offset} plus {} rail-1 bytes does not fit a {}-byte staging buffer",
                    FRAME_HEADER_BYTES + halves.rail0_bytes + halves.rail1_bytes,
                    staging.len()
                )));
            }
            let mut cursor = start;
            for value in
                &payload[halves.rail0_elements..halves.rail0_elements + halves.rail1_elements]
            {
                staging[cursor..cursor + 2].copy_from_slice(&value.to_le_bytes());
                cursor += 2;
            }
        }
        Ok(())
    }
}

/// A zeroed completion entry: every field is an integer, so zero is a valid value.
fn zeroed_wc() -> IbvWc {
    // SAFETY: IbvWc contains only integers and arrays of integers.
    unsafe { std::mem::zeroed() }
}

impl DataPlane {
    /// Polls one rail's CQ once, folding completions into `phase`.
    ///
    /// Returns (completions seen, send completions drained).
    fn poll_rail(&self, phase: &mut PhaseState, rail: usize) -> Result<(usize, usize), CommError> {
        let mut wcs = [zeroed_wc(); 16];
        let polled = self.rails[rail].cq.poll(&mut wcs)?;
        let mut sends = 0;
        for completion in &wcs[..polled] {
            if completion.status != wc::SUCCESS {
                return Err(CommError::verbs(
                    "ibv_poll_cq",
                    format!(
                        "completion failed with {} (opcode {}, qp {}, wr_id {:#x})",
                        wc::status_name(completion.status),
                        completion.opcode,
                        completion.qp_num,
                        completion.wr_id
                    ),
                ));
            }
            match tag_decode(completion.wr_id) {
                WrTag::Send { rail: send_rail } => {
                    if send_rail != rail {
                        return Err(CommError::Protocol(format!(
                            "a send completion on rail {rail} carries a rail {send_rail} tag"
                        )));
                    }
                    if completion.opcode != wc::RDMA_WRITE {
                        return Err(CommError::Protocol(format!(
                            "a WRITE_WITH_IMM send on rail {rail} completed with opcode {}, expected IBV_WC_RDMA_WRITE ({}); send opcodes are below 128",
                            completion.opcode,
                            wc::RDMA_WRITE
                        )));
                    }
                    sends += 1;
                }
                WrTag::Receive {
                    rail: recv_rail,
                    peer,
                    phase: recv_phase,
                } => {
                    if completion.opcode != wc::RECV_RDMA_WITH_IMM {
                        return Err(CommError::Protocol(format!(
                            "a WRITE_WITH_IMM receive on rail {rail} completed with opcode {}, expected IBV_WC_RECV_RDMA_WITH_IMM ({})",
                            completion.opcode,
                            wc::RECV_RDMA_WITH_IMM
                        )));
                    }
                    if recv_rail != rail || recv_phase != phase.tag.phase {
                        return Err(CommError::Protocol(format!(
                            "a completion on rail {rail} carries tag (rail {recv_rail}, phase {recv_phase}) but phase {} is open",
                            phase.tag.phase
                        )));
                    }
                    let index = phase
                        .expected
                        .iter()
                        .position(|(sender, _)| *sender == peer)
                        .ok_or_else(|| {
                            CommError::Protocol(format!(
                                "received a frame from rank {peer}, which this phase does not expect"
                            ))
                        })?;
                    let (sender, elements) = phase.expected[index];
                    check_immediate(
                        completion.imm_data,
                        sender,
                        phase.tag.phase,
                        phase.tag.kind,
                        phase.tag.seq,
                    )?;
                    let halves = rail_halves(elements, self.rail_count)?;
                    let expected_bytes = if rail == 0 {
                        FRAME_HEADER_BYTES + halves.rail0_bytes
                    } else {
                        halves.rail1_bytes
                    };
                    if completion.byte_len as usize != expected_bytes {
                        return Err(CommError::Protocol(format!(
                            "frame from rank {sender} on rail {rail} completed {} bytes, but the schedule expects {expected_bytes}",
                            completion.byte_len
                        )));
                    }
                    if phase.arrivals[index][rail]
                        .replace(completion.byte_len)
                        .is_some()
                    {
                        return Err(CommError::Protocol(format!(
                            "rank {sender} completed twice on rail {rail} in phase {}",
                            phase.tag.phase
                        )));
                    }
                }
            }
        }
        Ok((polled, sends))
    }

    /// Assembles one complete frame from its inbox slots.
    fn assemble_frame(&self, phase: &PhaseState, index: usize) -> Result<Received, CommError> {
        let (peer, elements) = phase.expected[index];
        let bytes = elements * 2;
        let halves = rail_halves(elements, self.rail_count)?;
        let base = self.slot.inbox_offset(peer, phase.tag.phase);
        let staging0 = &self.rails[0].staging;
        let header_bytes = staging0
            .get(base..base + FRAME_HEADER_BYTES)
            .ok_or_else(|| CommError::Protocol("inbox header slice out of bounds".to_owned()))?;
        let header = FrameHeader::decode(header_bytes).map_err(|error| {
            CommError::Protocol(format!("frame from rank {peer} has a bad header: {error}"))
        })?;
        if header.rank as usize != peer
            || header.seq != phase.tag.seq
            || header.phase != phase.tag.phase
            || header.payload_bytes as usize != bytes
        {
            return Err(CommError::Protocol(format!(
                "frame from rank {peer} header {header:?} disagrees with the schedule (phase {}, seq {}, {bytes} bytes)",
                phase.tag.phase, phase.tag.seq
            )));
        }
        let start0 = base + FRAME_HEADER_BYTES;
        let mut payload = Vec::with_capacity(elements);
        let first = staging0
            .get(start0..start0 + halves.rail0_bytes)
            .ok_or_else(|| CommError::Protocol("inbox rail 0 slice out of bounds".to_owned()))?;
        for chunk in first.as_chunks::<2>().0 {
            payload.push(u16::from_le_bytes(*chunk));
        }
        if halves.rail1_bytes > 0 {
            let staging1 = &self.rails[1].staging;
            let start1 = start0 + halves.rail0_bytes;
            let second = staging1
                .get(start1..start1 + halves.rail1_bytes)
                .ok_or_else(|| {
                    CommError::Protocol("inbox rail 1 slice out of bounds".to_owned())
                })?;
            for chunk in second.as_chunks::<2>().0 {
                payload.push(u16::from_le_bytes(*chunk));
            }
        }
        if payload.len() != elements {
            return Err(CommError::Protocol(format!(
                "assembled {} elements from rank {peer} but the schedule expects {elements}",
                payload.len()
            )));
        }
        Ok(Received {
            from: peer,
            payload,
        })
    }
}

impl Exchange for DataPlane {
    fn prepare(&mut self, tag: ExchangeTag, expected: &[(usize, usize)]) -> Result<(), CommError> {
        if usize::from(tag.phase) >= PHASES {
            return Err(CommError::Protocol(format!(
                "exchange phase {} does not exist (only {PHASES} phases)",
                tag.phase
            )));
        }
        if self.phase.is_some() {
            return Err(CommError::Protocol(
                "prepare was called while another phase is still open".to_owned(),
            ));
        }
        if self.pending_sends != 0 {
            return Err(CommError::Protocol(format!(
                "prepare was called with {} send completions still outstanding",
                self.pending_sends
            )));
        }
        let mut arrivals = Vec::with_capacity(expected.len());
        for (peer, elements) in expected {
            if *peer >= self.world {
                return Err(CommError::RankOutOfRange {
                    rank: *peer,
                    world_size: self.world,
                });
            }
            if *peer == self.rank {
                return Err(CommError::Protocol(format!(
                    "the schedule expects a frame from this rank ({}) itself",
                    self.rank
                )));
            }
            if !self.slot.holds(*peer, tag.phase) {
                return Err(CommError::Protocol(format!(
                    "the schedule expects a frame from rank {peer} in phase {}, which has no inbox slot",
                    tag.phase
                )));
            }
            let bytes = elements.checked_mul(2).ok_or_else(|| {
                CommError::InvalidLayout(format!("{elements} expected elements overflow bytes"))
            })?;
            if bytes > self.slot.max_frame_bytes() {
                return Err(CommError::InvalidLayout(format!(
                    "the schedule expects {bytes} bytes from rank {peer}, above the {}-byte frame cap",
                    self.slot.max_frame_bytes()
                )));
            }
            arrivals.push([None, None]);
        }
        for (peer, _) in expected {
            let offset = self.slot.inbox_offset(*peer, tag.phase);
            let length = u32::try_from(self.slot.slot_bytes()).map_err(|_| {
                CommError::InvalidLayout("slot does not fit a scatter/gather length".to_owned())
            })?;
            for rail in 0..self.rail_count {
                let mut sge = IbvSge {
                    addr: self.rails[rail].mr.addr() + offset as u64,
                    length,
                    lkey: self.rails[rail].mr.lkey(),
                };
                let mut wr = recv_wr(
                    tag_encode(WrTag::Receive {
                        rail,
                        peer: *peer,
                        phase: tag.phase,
                    }),
                    &mut sge,
                );
                self.rails[rail].qps[*peer].post_recv(&mut wr)?;
            }
        }
        self.phase = Some(PhaseState {
            tag,
            expected: expected.to_vec(),
            arrivals,
            frames: vec![None; expected.len()],
        });
        Ok(())
    }

    fn send(
        &mut self,
        to: usize,
        elements: std::ops::Range<usize>,
        payload: &[u16],
    ) -> Result<(), CommError> {
        let tag = self
            .phase
            .as_ref()
            .ok_or_else(|| CommError::Protocol("send was called before prepare".to_owned()))?
            .tag;
        if to >= self.world {
            return Err(CommError::RankOutOfRange {
                rank: to,
                world_size: self.world,
            });
        }
        if to == self.rank {
            return Err(CommError::Protocol(
                "the schedule asked this rank to send to itself".to_owned(),
            ));
        }
        if payload.len() != elements.len() {
            return Err(CommError::InvalidLayout(format!(
                "send covers {} elements but carries {}",
                elements.len(),
                payload.len()
            )));
        }
        let bytes = payload.len() * 2;
        if bytes > self.slot.max_frame_bytes() {
            return Err(CommError::InvalidLayout(format!(
                "send of {bytes} bytes is above the {}-byte frame cap",
                self.slot.max_frame_bytes()
            )));
        }
        let halves = rail_halves(payload.len(), self.rail_count)?;
        let send_offset = self.slot.send_offset(to, tag.phase);
        self.fill_send_slot(send_offset, tag, payload, &halves)?;
        let mut peer_mr = [0u64; 2];
        let mut peer_rkey = [0u32; 2];
        for rail in 0..self.rail_count {
            peer_mr[rail] = self.rails[rail].remotes[to].mr_addr;
            peer_rkey[rail] = self.rails[rail].remotes[to].rkey;
        }
        let plans = plan_writes(
            &self.slot,
            &halves,
            self.rank,
            tag.phase,
            tag,
            send_offset,
            peer_mr,
            peer_rkey,
        );
        let mut posted = 0;
        for plan in &plans {
            let rail = &self.rails[plan.rail];
            let mut sge = IbvSge {
                addr: rail.mr.addr() + plan.local_offset as u64,
                length: u32::try_from(plan.bytes).map_err(|_| {
                    CommError::InvalidLayout("a planned write does not fit a u32 length".to_owned())
                })?,
                lkey: rail.mr.lkey(),
            };
            let mut wr = send_wr(plan, &mut sge);
            rail.qps[to].post_send(&mut wr)?;
            posted += 1;
            if plan.rail == 0 {
                self.ledger.rail0_bytes += plan.bytes as u64;
            } else {
                self.ledger.rail1_bytes += plan.bytes as u64;
            }
        }
        self.pending_sends += posted;
        self.ledger.transfers += 1;
        self.ledger.element_writes += payload.len() as u64;
        Ok(())
    }

    fn collect(&mut self, expected: &[(usize, usize)]) -> Result<Vec<Received>, CommError> {
        let mut phase = self
            .phase
            .take()
            .ok_or_else(|| CommError::Protocol("collect was called before prepare".to_owned()))?;
        if phase.expected != expected {
            return Err(CommError::Protocol(format!(
                "collect was given {:?} but prepare announced {:?}",
                expected, phase.expected
            )));
        }
        let deadline = Instant::now() + self.timeout;
        let mut empty_polls = 0u32;
        loop {
            let mut polled = 0usize;
            let mut sends_drained = 0usize;
            for rail in 0..self.rail_count {
                let (seen, sends) = self.poll_rail(&mut phase, rail)?;
                polled += seen;
                sends_drained += sends;
            }
            if sends_drained > self.pending_sends {
                return Err(CommError::Protocol(format!(
                    "{sends_drained} send completions drained but only {} were outstanding",
                    self.pending_sends
                )));
            }
            self.pending_sends -= sends_drained;
            for index in 0..phase.expected.len() {
                if phase.frames[index].is_some() {
                    continue;
                }
                let rail0_done = phase.arrivals[index][0].is_some();
                let rail1_done = self.rail_count == 1 || phase.arrivals[index][1].is_some();
                if rail0_done && rail1_done {
                    let assembled = self.assemble_frame(&phase, index)?;
                    phase.frames[index] = Some(assembled);
                }
            }
            if phase.assembled_count() == phase.expected.len() && self.pending_sends == 0 {
                let mut frames = Vec::with_capacity(phase.frames.len());
                for frame in phase.frames {
                    frames.push(frame.ok_or_else(|| {
                        CommError::Protocol("a completed phase is missing a frame".to_owned())
                    })?);
                }
                return Ok(frames);
            }
            if Instant::now() >= deadline {
                return Err(CommError::timeout(
                    self.timeout,
                    format!(
                        "waiting for RDMA completions: {} of {} frames assembled, {} send completions outstanding",
                        phase.assembled_count(),
                        phase.expected.len(),
                        self.pending_sends
                    ),
                ));
            }
            if polled == 0 {
                empty_polls += 1;
                if empty_polls.is_multiple_of(64) {
                    std::thread::yield_now();
                }
            } else {
                empty_polls = 0;
            }
        }
    }

    fn finish(&mut self) -> Result<(), CommError> {
        if let Some(phase) = self.phase.take() {
            return Err(CommError::Protocol(format!(
                "finish was called with phase {} still holding {} of {} frames",
                phase.tag.phase,
                phase.assembled_count(),
                phase.expected.len()
            )));
        }
        if self.pending_sends != 0 {
            return Err(CommError::Protocol(format!(
                "finish was called with {} send completions outstanding",
                self.pending_sends
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdma::abi::{GidKind, IbvGid, mtu};

    fn gid(octet: u8) -> IbvGid {
        let mut raw = [0u8; 16];
        raw[10] = 0xff;
        raw[11] = 0xff;
        raw[12] = 10;
        raw[13] = 0;
        raw[14] = 1;
        raw[15] = octet;
        IbvGid { raw }
    }

    #[test]
    fn slot_layout_keeps_every_slot_inside_the_buffer() {
        let slot = SlotLayout::new(4, 2 * 1024 * 1024).unwrap_or_else(|error| panic!("{error}"));
        // The slot carries the header plus the WHOLE frame, so a single-rail send of the full
        // cap fits. 2 MiB cap -> 24 + 2 MiB per slot.
        assert_eq!(slot.slot_bytes(), FRAME_HEADER_BYTES + 2 * 1024 * 1024);
        assert!(slot.slot_bytes() >= slot.max_frame_bytes() + FRAME_HEADER_BYTES);
        let mut offsets = Vec::new();
        for peer in 0..4 {
            for phase in 0..2 {
                let inbox = slot.inbox_offset(peer, phase);
                let send = slot.send_offset(peer, phase);
                assert!(inbox + slot.slot_bytes() <= slot.total_bytes());
                assert!(send + slot.slot_bytes() <= slot.total_bytes());
                assert!(inbox < send);
                offsets.push(inbox);
                offsets.push(send);
            }
        }
        offsets.sort_unstable();
        let unique: Vec<usize> = offsets
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(unique.len(), 16, "every slot must be distinct");
        assert!(SlotLayout::new(0, 16).is_err());
        assert!(SlotLayout::new(4, 0).is_err());
        assert!(slot.holds(3, 1));
        assert!(!slot.holds(4, 0));
        assert!(!slot.holds(0, 2));
    }

    #[test]
    fn a_frame_split_keeps_whole_elements_on_both_rails() {
        for elements in [1_usize, 2, 3, 5, 5120, 1024 * 1024] {
            let halves = rail_halves(elements, 2).unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(halves.rail0_elements + halves.rail1_elements, elements);
            assert_eq!(halves.rail0_bytes, halves.rail0_elements * 2);
            assert_eq!(halves.rail1_bytes, halves.rail1_elements * 2);
            assert_eq!(halves.rail0_bytes % 2, 0);
            assert_eq!(halves.rail1_bytes % 2, 0);
            assert!(halves.rail0_bytes >= halves.rail1_bytes);
        }
        let single = rail_halves(5, 1).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(single.rail0_elements, 5);
        assert_eq!(single.rail1_bytes, 0);
        assert!(rail_halves(5, 3).is_err());
    }

    #[test]
    fn immediate_data_round_trips_sender_phase_kind_and_sequence() {
        for (from, phase, kind, seq) in [
            (0_usize, 0_u8, ExchangeKind::Reduce, 1_u64),
            (3, 1, ExchangeKind::Gather, 0x1_23),
            (255, 0, ExchangeKind::Reduce, u64::MAX),
        ] {
            let imm = imm_encode(from, phase, kind, seq);
            let (decoded_from, decoded_phase, decoded_kind, decoded_seq) = imm_decode(imm);
            assert_eq!(decoded_from, from & 0xff);
            assert_eq!(decoded_phase, phase & 1);
            assert_eq!(
                decoded_kind,
                if kind == ExchangeKind::Reduce { 0 } else { 1 }
            );
            assert_eq!(decoded_seq, (seq & 0xff) as u8);
        }
    }

    #[test]
    fn work_request_tags_survive_the_round_trip() {
        for tag in [
            WrTag::Receive {
                rail: 0,
                peer: 2,
                phase: 0,
            },
            WrTag::Receive {
                rail: 1,
                peer: 3,
                phase: 1,
            },
            WrTag::Send { rail: 0 },
            WrTag::Send { rail: 1 },
        ] {
            assert_eq!(tag_decode(tag_encode(tag)), tag);
        }
        assert_ne!(
            tag_encode(WrTag::Send { rail: 0 }) & RECEIVE_TAG_BIT,
            RECEIVE_TAG_BIT
        );
    }

    #[test]
    fn the_write_plan_splits_a_frame_between_the_two_rails() {
        let slot = SlotLayout::new(4, 2 * 1024 * 1024).unwrap_or_else(|error| panic!("{error}"));
        let elements = 1024;
        let halves = rail_halves(elements, 2).unwrap_or_else(|error| panic!("{error}"));
        let tag = ExchangeTag {
            seq: 7,
            kind: ExchangeKind::Reduce,
            phase: 1,
        };
        let send_offset = slot.send_offset(2, 1);
        let plans = plan_writes(
            &slot,
            &halves,
            1,
            1,
            tag,
            send_offset,
            [0x1_0000, 0x2_0000],
            [0xa, 0xb],
        );
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].rail, 0);
        assert_eq!(plans[0].bytes, FRAME_HEADER_BYTES + halves.rail0_bytes);
        assert_eq!(plans[0].local_offset, send_offset);
        assert_eq!(
            plans[0].remote_addr,
            0x1_0000 + slot.inbox_offset(1, 1) as u64
        );
        assert_eq!(plans[0].rkey, 0xa);
        assert_eq!(plans[1].rail, 1);
        assert_eq!(plans[1].bytes, halves.rail1_bytes);
        assert_eq!(
            plans[1].local_offset,
            send_offset + FRAME_HEADER_BYTES + halves.rail0_bytes
        );
        assert_eq!(
            plans[1].remote_addr,
            0x2_0000 + (slot.inbox_offset(1, 1) + FRAME_HEADER_BYTES + halves.rail0_bytes) as u64
        );
        assert_eq!(plans[1].rkey, 0xb);
        assert_eq!(plans[0].imm, plans[1].imm);
        assert_eq!(imm_decode(u32::from_be(plans[0].imm)).0, 1);

        let single = rail_halves(elements, 1).unwrap_or_else(|error| panic!("{error}"));
        let plans = plan_writes(
            &slot,
            &single,
            1,
            0,
            tag,
            send_offset,
            [0x1_0000, 0],
            [0xa, 0],
        );
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].bytes, FRAME_HEADER_BYTES + elements * 2);
        assert_eq!(imm_decode(u32::from_be(plans[0].imm)).0, 1);
    }

    #[test]
    fn the_receive_immediate_is_compared_in_exactly_one_byte_order() {
        // Sender 2, phase 0, Reduce, seq 1: imm_encode is 0x0200_0001 on the host and travels as
        // 0x0100_0002. A receive path that byte-swaps the expected value a second time rejects
        // every such frame with a Protocol error, which is what ranks 2 and 3 would have hit on
        // the first gate run.
        let slot = SlotLayout::new(4, 2 * 1024 * 1024).unwrap_or_else(|error| panic!("{error}"));
        let halves = rail_halves(1024, 2).unwrap_or_else(|error| panic!("{error}"));
        let tag = ExchangeTag {
            seq: 1,
            kind: ExchangeKind::Reduce,
            phase: 0,
        };
        let plans = plan_writes(
            &slot,
            &halves,
            2,
            0,
            tag,
            0,
            [0x1_0000, 0x2_0000],
            [0xa, 0xb],
        );
        let wire_imm = plans[0].imm;
        assert_eq!(
            u32::from_be(wire_imm),
            imm_encode(2, 0, ExchangeKind::Reduce, 1)
        );
        check_immediate(wire_imm, 2, 0, ExchangeKind::Reduce, 1)
            .unwrap_or_else(|error| panic!("the posted immediate must be accepted: {error}"));
        // The same frame with any part of its identity changed is still refused.
        for (sender, phase, kind, seq) in [
            (3, 0, ExchangeKind::Reduce, 1),
            (2, 1, ExchangeKind::Reduce, 1),
            (2, 0, ExchangeKind::Gather, 1),
            (2, 0, ExchangeKind::Reduce, 2),
        ] {
            assert!(
                check_immediate(wire_imm, sender, phase, kind, seq).is_err(),
                "an immediate for sender {sender} phase {phase} kind {kind:?} seq {seq} must be refused"
            );
        }
    }

    #[test]
    fn the_transition_plans_set_the_fields_their_masks_announce() {
        let (attr, mask) = init_attr(1);
        assert_eq!(attr.qp_state, qp_state::INIT);
        assert_eq!(attr.port_num, 1);
        assert_eq!(attr.pkey_index, 0);
        for bit in [
            attr_mask::STATE,
            attr_mask::PKEY_INDEX,
            attr_mask::PORT,
            attr_mask::ACCESS_FLAGS,
        ] {
            assert_ne!(mask & bit, 0);
        }
        assert_ne!(
            attr.qp_access_flags & crate::rdma::abi::access::REMOTE_WRITE,
            0
        );

        let peer_gid = gid(84);
        assert_eq!(peer_gid.kind(), GidKind::RoceV2Ipv4([10, 0, 1, 84]));
        let (attr, mask) = rtr_attr(1, 0x1234, peer_gid, 3, mtu::MTU_1024);
        assert_eq!(attr.qp_state, qp_state::RTR);
        assert_eq!(attr.dest_qp_num, 0x1234);
        assert_eq!(attr.rq_psn, PSN);
        assert_eq!(attr.path_mtu, mtu::MTU_1024);
        assert_eq!(attr.ah_attr.grh.dgid, peer_gid);
        assert_eq!(attr.ah_attr.grh.sgid_index, 3);
        assert_eq!(attr.ah_attr.is_global, 1);
        assert_eq!(attr.ah_attr.port_num, 1);
        for bit in [
            attr_mask::STATE,
            attr_mask::PATH_MTU,
            attr_mask::DEST_QPN,
            attr_mask::RQ_PSN,
            attr_mask::AV,
        ] {
            assert_ne!(mask & bit, 0);
        }

        let (attr, mask) = rts_attr();
        assert_eq!(attr.qp_state, qp_state::RTS);
        assert_eq!(attr.sq_psn, PSN);
        assert_eq!(attr.retry_cnt, 7);
        assert_eq!(attr.rnr_retry, 7);
        for bit in [attr_mask::STATE, attr_mask::SQ_PSN, attr_mask::RETRY_CNT] {
            assert_ne!(mask & bit, 0);
        }
    }

    #[test]
    fn the_path_mtu_is_the_smaller_of_the_two_ports() {
        assert_eq!(
            min_path_mtu(mtu::MTU_4096, mtu::MTU_1024).unwrap_or_else(|error| panic!("{error}")),
            mtu::MTU_1024
        );
        assert_eq!(
            min_path_mtu(mtu::MTU_1024, mtu::MTU_4096).unwrap_or_else(|error| panic!("{error}")),
            mtu::MTU_1024
        );
        assert_eq!(
            min_path_mtu(mtu::MTU_4096, mtu::MTU_4096).unwrap_or_else(|error| panic!("{error}")),
            mtu::MTU_4096
        );
        assert!(matches!(
            min_path_mtu(99, mtu::MTU_1024),
            Err(CommError::Verbs { .. })
        ));
    }
}
