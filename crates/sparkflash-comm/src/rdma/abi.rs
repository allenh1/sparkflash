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

//! Hand-written `repr(C)` mirror of the libibverbs ABI subset this backend calls.
//!
//! The DGX Spark nodes carry rdma-core headers, but this workspace must build on the x86_64
//! development box, which has neither `/usr/include/infiniband/verbs.h` nor libibverbs. Generating
//! bindings at build time needs those headers, and linking `-libverbs` would make every workspace
//! build depend on an RDMA install. So the structs below are written by hand against rdma-core's
//! public `verbs.h` and the runtime library is opened with `dlopen` (verbs.rs), exactly as the
//! CUDA driver is reached in crates/sparkflash-kernels/src/cuda_driver.rs.
//!
//! Three rules keep a hand-written ABI honest:
//!
//! * every struct is `#[repr(C)]`, so Rust does not reorder fields;
//! * every field the backend reads or writes has its byte offset pinned by a `const` static
//!   assertion below, and every struct's size is pinned too. If a future rdma-core changes a
//!   layout the assertion fails at compile time, which is the only way this crate can notice on a
//!   host with no headers;
//! * the library is loaded with `RTLD_NOW` and every symbol is resolved by name at load, so a library
//!   whose ABI differs from this subset is rejected before a pointer is used.
//!
//! The assertions are written against the LP64 layout rdma-core has used for the lifetime of these
//! structs (both aarch64 and x86_64 Linux). The mirror is deliberately not minimal: the public
//! `ibv_qp` and `ibv_mr` prefixes let the backend read a QP's `qp_num` and an MR's
//! `lkey`/`rkey` without `ibv_query_qp` (which cannot report a QP's own number), and
//! `ibv_qp_attr`/`ibv_send_wr` carry trailing space because rdma-core appended fields after the ones
//! this backend sets. The library reads a pointer, so a smaller library struct is safe and a larger
//! one must still fit the sizes asserted here.
//!
//! Fields that exist only to complete the mirror are marked `// ABI`; the module-level
//! `allow(dead_code)` is for them, not for a shortcut around using a field.

#![allow(dead_code)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

// ---------------------------------------------------------------------------------------------
// Opaque library-owned handles. The crate never dereferences these; it only stores the pointers
// the library returns, so a zero-sized opaque type is the faithful Rust spelling.
// ---------------------------------------------------------------------------------------------

/// Opaque `struct ibv_device`.
#[repr(C)]
pub struct IbvDevice {
    _private: [u8; 0],
}

/// Opaque `struct ibv_context` (one open device).
#[repr(C)]
pub struct IbvContext {
    _private: [u8; 0],
}

/// Opaque `struct ibv_pd` (protection domain).
#[repr(C)]
pub struct IbvPd {
    _private: [u8; 0],
}

/// Opaque `struct ibv_cq` (completion queue).
#[repr(C)]
pub struct IbvCq {
    _private: [u8; 0],
}

/// Opaque `struct ibv_srq`; the backend creates no shared receive queue.
#[repr(C)]
pub struct IbvSrq {
    _private: [u8; 0],
}

/// Opaque `struct ibv_comp_channel`; the backend polls CQs instead of using completion channels.
#[repr(C)]
pub struct IbvCompChannel {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------------------------
// Data structures.
// ---------------------------------------------------------------------------------------------

/// `struct ibv_gid`: a 16-byte GID, 8-byte aligned because the C union contains u64 members.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IbvGid {
    /// Raw 16 GID bytes, big-endian on the wire.
    pub raw: [u8; 16],
}

/// What a GID address family means for this backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GidKind {
    /// The GID table entry is all zero: the index is not configured, so it cannot be used.
    Unspecified,
    /// A RoCEv2 IPv4 GID, `::ffff:a.b.c.d`. This is the family an IPv4 RoCEv2 GID index reports
    /// and the family the live cluster uses (vLLM sets `NCCL_IB_GID_INDEX=3`).
    RoceV2Ipv4([u8; 4]),
    /// Any other 128-bit GID (IPv6 link-local or global, or a vendor form). Reported, not assumed.
    OtherIpv6,
}

impl IbvGid {
    /// Classifies the GID by its address family.
    #[must_use]
    pub fn kind(&self) -> GidKind {
        let raw = self.raw;
        if raw.iter().all(|byte| *byte == 0) {
            return GidKind::Unspecified;
        }
        if raw[..10].iter().all(|byte| *byte == 0) && raw[10] == 0xff && raw[11] == 0xff {
            return GidKind::RoceV2Ipv4([raw[12], raw[13], raw[14], raw[15]]);
        }
        GidKind::OtherIpv6
    }

    /// Human-readable form for bring-up reports and errors.
    #[must_use]
    pub fn describe(&self) -> String {
        match self.kind() {
            GidKind::Unspecified => "unspecified (all-zero GID)".to_owned(),
            GidKind::RoceV2Ipv4(octets) => format!(
                "RoCEv2 IPv4 {}.{}.{}.{}",
                octets[0], octets[1], octets[2], octets[3]
            ),
            GidKind::OtherIpv6 => {
                let mut text = String::with_capacity(16 * 3);
                for (index, byte) in self.raw.iter().enumerate() {
                    if index != 0 && index % 2 == 0 {
                        text.push(':');
                    }
                    text.push_str(&format!("{byte:02x}"));
                }
                text
            }
        }
    }
}

/// `struct ibv_sge`: one scatter/gather entry.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvSge {
    /// Local address of this segment (inside a registered MR).
    pub addr: u64,
    /// Segment length in bytes.
    pub length: u32,
    /// Local key of the MR containing `addr`.
    pub lkey: u32,
}

/// `struct ibv_send_wr`'s RDMA union arm.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvRdmaWr {
    /// Remote virtual address the operation targets.
    pub remote_addr: u64,
    /// Remote key authorising the target MR.
    pub rkey: u32,
}

/// `struct ibv_send_wr`'s atomic union arm. The backend posts no atomics; mirrored for layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvAtomicWr {
    /// Remote virtual address.
    pub remote_addr: u64,
    /// Compare value.
    pub compare_add: u64,
    /// Swap value.
    pub swap: u64,
    /// Remote key.
    pub rkey: u32,
}

/// `struct ibv_send_wr`'s UD union arm. Mirrored for layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvUdWr {
    /// Address handle.
    pub ah: *mut c_void,
    /// Remote QP number.
    pub remote_qpn: u32,
    /// Remote QKey.
    pub remote_qkey: u32,
}

/// The anonymous union at `ibv_send_wr.wr`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvWrUnion {
    /// RDMA read/write arm, the only one this backend uses.
    pub rdma: IbvRdmaWr,
    /// Atomic arm.
    pub atomic: IbvAtomicWr,
    /// Datagram arm.
    pub ud: IbvUdWr,
    /// The union's raw bytes; 32 bytes in rdma-core (the atomic arm is the largest).
    pub raw: [u8; 32],
}

/// `struct ibv_send_wr`'s XRC arm payload.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvXrc {
    /// Remote SRQ number; this backend posts no XRC traffic.
    pub remote_srqn: u32,
}

/// `struct ibv_send_wr`'s QP-type union.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvQpTypeUnion {
    /// XRC fields.
    pub xrc: IbvXrc,
    /// The union's raw bytes.
    pub raw: [u8; 4],
}

/// `struct ibv_mw_bind_info`: the bind-info arm of ibv_send_wr's trailing union.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvMwBindInfo {
    /// Memory region.
    pub mr: *mut c_void,
    /// Address inside the region.
    pub addr: u64,
    /// Number of bytes.
    pub length: u64,
    /// Memory-window flags.
    pub mw_flags: u32,
    /// Access flags.
    pub access_flags: u32,
}

/// `struct ibv_send_wr`'s `bind_mw` union arm. The backend posts no BIND_MW; mirrored for layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvMwBind {
    /// Memory window.
    pub mw: *mut c_void,
    /// Remote key.
    pub rkey: u32,
    /// Padding to the 8-byte alignment of `bind_info`.
    pub padding: u32,
    /// Bind info.
    pub bind_info: IbvMwBindInfo,
}

/// `struct ibv_send_wr`'s `tso` union arm. Mirrored for layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvTso {
    /// Header buffer.
    pub hdr: *mut c_void,
    /// Header size.
    pub hdr_sz: u16,
    /// MSS.
    pub mss: u16,
}

/// The anonymous union rdma-core 39 appends after the QP-type union.
///
/// The backend posts no BIND_MW and no TSO, but the real C struct is 128 bytes: a provider that
/// copies the work request, or a future BIND_MW path, reads these bytes. Carrying the union and
/// asserting its size is what makes the mirror safe instead of merely prefix-compatible.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendWrTail {
    /// Memory-window bind arm.
    pub bind_mw: IbvMwBind,
    /// TCP segmentation-offload arm.
    pub tso: IbvTso,
    /// The union's raw bytes; 48 bytes in rdma-core 39.
    pub raw: [u8; 48],
}

/// `struct ibv_send_wr`.
#[repr(C)]
pub struct IbvSendWr {
    /// Caller tag echoed in the completion.
    pub wr_id: u64,
    /// Next work request in the chain, or null for a single WR.
    pub next: *mut IbvSendWr,
    /// Scatter/gather list.
    pub sg_list: *mut IbvSge,
    /// Number of SGEs.
    pub num_sge: c_int,
    /// `enum ibv_wr_opcode`; this backend uses `IBV_WR_RDMA_WRITE_WITH_IMM`.
    pub opcode: c_int,
    /// `enum ibv_send_flags` bits.
    pub send_flags: c_int,
    /// Immediate value for `*_WITH_IMM` opcodes, in network byte order (the C union is `__be32`).
    pub imm_data: u32,
    /// Operation-specific union.
    pub wr: IbvWrUnion,
    /// QP-type-specific union.
    pub qp_type: IbvQpTypeUnion,
    /// ABI padding between `qp_type` and the trailing union.
    ///
    /// rdma-core 39 has no `comp_mask` field here (older releases did); these four bytes are the
    /// alignment padding before the anonymous union at offset 80.
    pub padding: [u8; 4],
    /// The anonymous union rdma-core 39 appends (bind_mw / tso), 48 bytes at offset 80.
    pub tail: IbvSendWrTail,
}

/// `struct ibv_recv_wr`.
#[repr(C)]
pub struct IbvRecvWr {
    /// Caller tag echoed in the completion.
    pub wr_id: u64,
    /// Next work request in the chain, or null.
    pub next: *mut IbvRecvWr,
    /// Scatter/gather list describing where the incoming message lands.
    pub sg_list: *mut IbvSge,
    /// Number of SGEs.
    pub num_sge: c_int,
    /// ABI padding to the C struct's 8-byte alignment.
    pub reserved: c_int,
}

/// `struct ibv_wc`: one completion queue entry.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IbvWc {
    /// Work-request tag set by the poster.
    pub wr_id: u64,
    /// `enum ibv_wc_status`; zero is success.
    pub status: c_int,
    /// `enum ibv_wc_opcode`.
    pub opcode: c_int,
    /// Provider error details.
    pub vendor_err: u32,
    /// Bytes transferred.
    pub byte_len: u32,
    /// Immediate value for a receive-with-immediate completion.
    pub imm_data: u32,
    /// QP number the completion belongs to.
    pub qp_num: u32,
    /// Source QP for a receive completion.
    pub src_qp: u32,
    /// `enum ibv_wc_flags` bits.
    pub flags: c_int,
    /// P_Key index.
    pub pkey_index: u16,
    /// Source LID.
    pub slid: u16,
    /// Service level.
    pub sl: u8,
    /// Destination LID path bits.
    pub dlid_path_bits: u8,
    /// ABI padding to the C struct's 48-byte size (the C struct pads 46 to 48).
    pub reserved: [u8; 2],
}

/// `struct ibv_port_attr`, exactly as rdma-core 39 lays it out: 52 bytes.
///
/// The C struct has been extended over time; the offsets this backend reads (state, active_mtu,
/// gid_tbl_len, link_layer) and the version-sensitive tail (47 `flags`, 48 `port_cap_flags2`) are
/// pinned by the static assertions below, so a library with a different shape is a build failure
/// here rather than a silent read of the wrong field.
#[repr(C)]
pub struct IbvPortAttr {
    /// `enum ibv_port_state`: 4 is ACTIVE.
    pub state: c_int,
    /// `enum ibv_mtu` the port is capable of.
    pub max_mtu: c_int,
    /// `enum ibv_mtu` currently negotiated.
    pub active_mtu: c_int,
    /// Number of entries in the GID table.
    pub gid_tbl_len: c_int,
    /// Capability flags.
    pub port_cap_flags: u32,
    /// Maximum message size.
    pub max_msg_sz: u32,
    /// Bad P_Key counter.
    pub bad_pkey_cntr: u32,
    /// QKey violation counter.
    pub qkey_viol_cntr: u32,
    /// Number of P_Key table entries.
    pub pkey_tbl_len: u16,
    /// Local LID (zero on a RoCE/Ethernet port).
    pub lid: u16,
    /// SM LID (zero on a RoCE/Ethernet port).
    pub sm_lid: u16,
    /// LMC.
    pub lmc: u8,
    /// Maximum VL number.
    pub max_vl_num: u8,
    /// SM service level.
    pub sm_sl: u8,
    /// Subnet timeout.
    pub subnet_timeout: u8,
    /// Init type reply.
    pub init_type_reply: u8,
    /// Active width.
    pub active_width: u8,
    /// `enum ibv_port_speed` truncated to 8 bits by rdma-core's struct; not used for the rate
    /// decision here (the validated sysfs rate is).
    pub active_speed: u8,
    /// Physical state.
    pub phys_state: u8,
    /// Link layer (2 is Ethernet, so the port is RoCE).
    pub link_layer: u8,
    /// Port capability flags, high half (rdma-core 39: offset 47).
    pub flags: u8,
    /// Extended port capability flags (rdma-core 39: offset 48, u16; the struct ends at 52).
    pub port_cap_flags2: u16,
}

/// `struct ibv_qp_cap`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IbvQpCap {
    /// Maximum outstanding send WRs.
    pub max_send_wr: u32,
    /// Maximum outstanding receive WRs.
    pub max_recv_wr: u32,
    /// Maximum send SGEs per WR.
    pub max_send_sge: u32,
    /// Maximum receive SGEs per WR.
    pub max_recv_sge: u32,
    /// Maximum inline data.
    pub max_inline_data: u32,
}

/// `struct ibv_global_route`: the RoCE routing header a QP needs to reach a peer GID.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IbvGlobalRoute {
    /// Destination GID.
    pub dgid: IbvGid,
    /// Flow label.
    pub flow_label: u32,
    /// Source GID table index.
    pub sgid_index: u8,
    /// Hop limit (the RoCEv2 TTL).
    pub hop_limit: u8,
    /// Traffic class / DSCP.
    pub traffic_class: u8,
}

/// `struct ibv_ah_attr`: address vector for a connected QP.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IbvAhAttr {
    /// Global routing header.
    pub grh: IbvGlobalRoute,
    /// Destination LID (zero on an Ethernet link layer).
    pub dlid: u16,
    /// Service level.
    pub sl: u8,
    /// Source path bits.
    pub src_path_bits: u8,
    /// Static rate.
    pub static_rate: u8,
    /// Non-zero when the global routing header is used (always 1 for RoCEv2).
    pub is_global: u8,
    /// Port number.
    pub port_num: u8,
}

/// `struct ibv_qp_init_attr`: QP creation parameters.
#[repr(C)]
pub struct IbvQpInitAttr {
    /// Opaque caller context stored on the QP.
    pub qp_context: *mut c_void,
    /// Send completion queue.
    pub send_cq: *mut IbvCq,
    /// Receive completion queue.
    pub recv_cq: *mut IbvCq,
    /// Shared receive queue, or null.
    pub srq: *mut IbvSrq,
    /// Queue capacities.
    pub cap: IbvQpCap,
    /// `enum ibv_qp_type`; 2 is RC.
    pub qp_type: c_int,
    /// Whether every send WR is signalled (0: only `IBV_SEND_SIGNALED` WRs are).
    pub sq_sig_all: c_int,
    /// ABI padding to the C struct's 64-byte size.
    pub reserved: [u8; 4],
}

/// `struct ibv_qp_attr`: the attributes `ibv_modify_qp` transitions a QP through.
///
/// The C struct's anonymous sub-structs are flattened here in the same field order, so the byte
/// offsets match; the trailing space covers `rate_limit` and `rate_limit_attr`, which newer
/// rdma-core versions appended.
#[repr(C)]
pub struct IbvQpAttr {
    /// `enum ibv_qp_state` to transition to (1 INIT, 2 RTR, 3 RTS).
    pub qp_state: c_int,
    /// Current state (unused for a fresh QP transition).
    pub cur_qp_state: c_int,
    /// `enum ibv_mtu` path MTU.
    pub path_mtu: c_int,
    /// `enum ibv_mig_state`.
    pub path_mig_state: c_int,
    /// QKey.
    pub qkey: u32,
    /// Receive PSN.
    pub rq_psn: u32,
    /// Send PSN.
    pub sq_psn: u32,
    /// Destination QP number for RTR.
    pub dest_qp_num: u32,
    /// Access flags for incoming RDMA operations.
    pub qp_access_flags: c_int,
    /// Queue capacities.
    pub cap: IbvQpCap,
    /// Primary address vector.
    pub ah_attr: IbvAhAttr,
    /// Alternate address vector.
    pub alt_ah_attr: IbvAhAttr,
    /// P_Key index.
    pub pkey_index: u16,
    /// Alternate P_Key index.
    pub alt_pkey_index: u16,
    /// Do not generate a completion on SQ drain.
    pub en_sqd_async_notify: u8,
    /// SQ draining.
    pub sq_draining: u8,
    /// Max outstanding RDMA/atomic reads.
    pub max_rd_atomic: u8,
    /// Max outstanding RDMA/atomic reads expected from the peer.
    pub max_dest_rd_atomic: u8,
    /// Minimum RNR NAK timer.
    pub min_rnr_timer: u8,
    /// Port number.
    pub port_num: u8,
    /// Local ACK timeout.
    pub timeout: u8,
    /// Transport retry count.
    pub retry_cnt: u8,
    /// RNR retry count.
    pub rnr_retry: u8,
    /// Alternate port number.
    pub alt_port_num: u8,
    /// Alternate local ACK timeout.
    pub alt_timeout: u8,
    /// Rate limit (appended by newer rdma-core).
    pub rate_limit: u32,
    /// ABI padding to the C struct's 144-byte size (rdma-core 39 ends after `rate_limit` at 140).
    pub reserved: [u8; 4],
}

/// The 64-byte public prefix of `struct ibv_qp` (160 bytes on glibc).
///
/// The library owns the object; this mirror exists so `ibv_create_qp`'s result can be read for its
/// `qp_num` (which defines the connection) and the state the backend last set. Everything after the
/// prefix is provider-private and is never touched. This type is a read-only *view*, never allocated
/// or copied; the static assertion pins the 64-byte prefix, not the C struct's 160-byte size.
#[repr(C)]
pub struct IbvQp {
    /// Owning device context.
    pub context: *mut IbvContext,
    /// Opaque caller context.
    pub qp_context: *mut c_void,
    /// Protection domain.
    pub pd: *mut IbvPd,
    /// Send CQ.
    pub send_cq: *mut IbvCq,
    /// Receive CQ.
    pub recv_cq: *mut IbvCq,
    /// Shared receive queue, or null.
    pub srq: *mut IbvSrq,
    /// Provider handle.
    pub handle: u32,
    /// QP number: the value the peer's RTR transition needs.
    pub qp_num: u32,
    /// `enum ibv_qp_state`.
    pub state: c_int,
    /// `enum ibv_qp_type`.
    pub qp_type: c_int,
}

/// The public prefix of `struct ibv_mr`.
///
/// `ibv_reg_mr` has no accessor for `lkey`/`rkey`, and both are needed: the local key authorises
/// every local SGE and the remote key authorises peer RDMA writes into the slot.
#[repr(C)]
pub struct IbvMr {
    /// Owning device context.
    pub context: *mut IbvContext,
    /// Protection domain.
    pub pd: *mut IbvPd,
    /// Registered address.
    pub addr: *mut c_void,
    /// Registered length.
    pub length: usize,
    /// Provider handle.
    pub handle: u32,
    /// Local key for SGEs.
    pub lkey: u32,
    /// Remote key for peer RDMA operations.
    pub rkey: u32,
    /// ABI padding to the C struct's 48-byte size.
    pub reserved: [u8; 4],
}

// ---------------------------------------------------------------------------------------------
// Constants. Values are the rdma-core `enum` values, spelled as `c_int` because the C enum is
// an int in every function signature this backend calls.
// ---------------------------------------------------------------------------------------------

/// `enum ibv_port_state` values used by the backend.
pub mod port_state {
    use std::ffi::c_int;

    /// Port is not active.
    pub const DOWN: c_int = 1;
    /// Port link is active.
    pub const ACTIVE: c_int = 4;
}

/// `enum ibv_mtu` values.
pub mod mtu {
    use std::ffi::c_int;

    /// 256-byte MTU.
    pub const MTU_256: c_int = 1;
    /// 512-byte MTU.
    pub const MTU_512: c_int = 2;
    /// 1024-byte MTU: what a 1500-byte IP MTU negotiates, and what the live cluster reports today.
    pub const MTU_1024: c_int = 3;
    /// 2048-byte MTU.
    pub const MTU_2048: c_int = 4;
    /// 4096-byte MTU: what a 9000-byte IP MTU negotiates.
    pub const MTU_4096: c_int = 5;

    /// MTU in bytes for an `enum ibv_mtu`, or None for an unknown value.
    #[must_use]
    pub fn bytes(value: c_int) -> Option<usize> {
        match value {
            MTU_256 => Some(256),
            MTU_512 => Some(512),
            MTU_1024 => Some(1024),
            MTU_2048 => Some(2048),
            MTU_4096 => Some(4096),
            _ => None,
        }
    }

    /// Name for an `enum ibv_mtu`, for reports.
    #[must_use]
    pub fn name(value: c_int) -> String {
        bytes(value).map_or_else(|| format!("unknown MTU enum {value}"), |b| format!("{b} B"))
    }
}

/// `enum ibv_qp_type`.
pub mod qp_type {
    use std::ffi::c_int;

    /// Reliable connected: the QP type an RC all-to-all needs.
    pub const RC: c_int = 2;
}

/// `enum ibv_qp_state`.
pub mod qp_state {
    use std::ffi::c_int;

    /// QP reset.
    pub const RESET: c_int = 0;
    /// QP initialised; port and access flags set.
    pub const INIT: c_int = 1;
    /// Ready to receive; peer QP number and address vector set.
    pub const RTR: c_int = 2;
    /// Ready to send.
    pub const RTS: c_int = 3;
}

/// `enum ibv_access_flags` bits.
pub mod access {
    use std::ffi::c_int;

    /// Local write access (needed for a receive buffer).
    pub const LOCAL_WRITE: c_int = 1 << 0;
    /// Remote write access (needed for a peer's RDMA WRITE target).
    pub const REMOTE_WRITE: c_int = 1 << 1;
    /// Remote read access.
    pub const REMOTE_READ: c_int = 1 << 2;
}

/// `enum ibv_send_flags` bits.
pub mod send_flags {
    use std::ffi::c_int;

    /// Completion is posted for this WR.
    pub const SIGNALED: c_int = 1 << 1;
    /// Fence this WR behind previous RDMA reads on the same QP.
    pub const FENCE: c_int = 1 << 0;
}

/// `enum ibv_wr_opcode` values.
pub mod wr_opcode {
    use std::ffi::c_int;

    /// RDMA write without an immediate value.
    pub const RDMA_WRITE: c_int = 0;
    /// RDMA write with an immediate value: what the RoCEnante slot write uses.
    pub const RDMA_WRITE_WITH_IMM: c_int = 1;
    /// Send.
    pub const SEND: c_int = 2;
}

/// `enum ibv_wc_status` and `enum ibv_wc_opcode` values used by the backend.
pub mod wc {
    use std::ffi::c_int;

    /// Completion succeeded.
    pub const SUCCESS: c_int = 0;
    /// A send-side send completion.
    pub const SEND: c_int = 0;
    /// An RDMA write completion (send side).
    pub const RDMA_WRITE: c_int = 1;
    /// A receive completion (base value; receive opcodes are 128 and above).
    pub const RECV: c_int = 128;
    /// A receive completion for an RDMA write with immediate.
    pub const RECV_RDMA_WITH_IMM: c_int = 129;

    /// Human-readable name of a completion status.
    #[must_use]
    pub fn status_name(status: c_int) -> String {
        let name = match status {
            0 => "IBV_WC_SUCCESS",
            1 => "IBV_WC_LOC_LEN_ERR",
            2 => "IBV_WC_LOC_QP_OP_ERR",
            3 => "IBV_WC_LOC_EEC_OP_ERR",
            4 => "IBV_WC_LOC_PROT_ERR",
            5 => "IBV_WC_WR_FLUSH_ERR",
            6 => "IBV_WC_MW_BIND_ERR",
            7 => "IBV_WC_BAD_RESP_ERR",
            8 => "IBV_WC_LOC_ACCESS_ERR",
            9 => "IBV_WC_REM_INV_REQ_ERR",
            10 => "IBV_WC_REM_ACCESS_ERR",
            11 => "IBV_WC_REM_OP_ERR",
            12 => "IBV_WC_RETRY_EXC_ERR",
            13 => "IBV_WC_RNR_RETRY_EXC_ERR",
            14 => "IBV_WC_LOC_RDD_VIOL_ERR",
            15 => "IBV_WC_REM_INV_RD_REQ_ERR",
            16 => "IBV_WC_REM_ABORT_ERR",
            17 => "IBV_WC_INV_EECN_ERR",
            18 => "IBV_WC_INV_EEC_STATE_ERR",
            19 => "IBV_WC_FATAL_ERR",
            20 => "IBV_WC_RESP_TIMEOUT_ERR",
            21 => "IBV_WC_GENERAL_ERR",
            22 => "IBV_WC_TM_ERR",
            23 => "IBV_WC_TM_RNDV_INCOMPLETE",
            _ => return format!("unknown ibv_wc_status {status}"),
        };
        name.to_owned()
    }

    /// Whether a completion opcode is a receive-side completion.
    #[must_use]
    pub fn is_receive(opcode: c_int) -> bool {
        opcode >= RECV
    }
}

/// `enum ibv_qp_attr_mask` bits for `ibv_modify_qp`.
pub mod attr_mask {
    use std::ffi::c_int;

    /// `qp_state` is set.
    pub const STATE: c_int = 1 << 0;
    /// `cur_qp_state` is set.
    pub const CUR_STATE: c_int = 1 << 1;
    /// `en_sqd_async_notify` is set.
    pub const EN_SQD_ASYNC_NOTIFY: c_int = 1 << 2;
    /// `qp_access_flags` is set.
    pub const ACCESS_FLAGS: c_int = 1 << 3;
    /// `pkey_index` is set.
    pub const PKEY_INDEX: c_int = 1 << 4;
    /// `port_num` is set.
    pub const PORT: c_int = 1 << 5;
    /// `qkey` is set.
    pub const QKEY: c_int = 1 << 6;
    /// `ah_attr` is set.
    pub const AV: c_int = 1 << 7;
    /// `path_mtu` is set.
    pub const PATH_MTU: c_int = 1 << 8;
    /// `timeout` is set.
    pub const TIMEOUT: c_int = 1 << 9;
    /// `retry_cnt` is set.
    pub const RETRY_CNT: c_int = 1 << 10;
    /// `rnr_retry` is set.
    pub const RNR_RETRY: c_int = 1 << 11;
    /// `rq_psn` is set.
    pub const RQ_PSN: c_int = 1 << 12;
    /// `max_rd_atomic` is set.
    pub const MAX_QP_RD_ATOMIC: c_int = 1 << 13;
    /// `alt_port_num`/`alt_timeout`/`alt_ah_attr` are set.
    pub const ALT_PATH: c_int = 1 << 14;
    /// `min_rnr_timer` is set.
    pub const MIN_RNR_TIMER: c_int = 1 << 15;
    /// `sq_psn` is set.
    pub const SQ_PSN: c_int = 1 << 16;
    /// `max_dest_rd_atomic` is set.
    pub const MAX_DEST_RD_ATOMIC: c_int = 1 << 17;
    /// `path_mig_state` is set.
    pub const PATH_MIG_STATE: c_int = 1 << 18;
    /// `cap` is set.
    pub const CAP: c_int = 1 << 19;
    /// `dest_qp_num` is set.
    pub const DEST_QPN: c_int = 1 << 20;
    /// `rate_limit` is set.
    pub const RATE_LIMIT: c_int = 1 << 25;
}

/// `enum ibv_link_layer`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkLayer {
    /// InfiniBand.
    InfiniBand = 1,
    /// Ethernet: the link layer a RoCE port reports.
    Ethernet = 2,
}

// ---------------------------------------------------------------------------------------------
// Static assertions: the whole reason this file can be reviewed without headers.
// ---------------------------------------------------------------------------------------------

const _: () = {
    // ibv_sge
    assert!(size_of::<IbvSge>() == 16);
    assert!(offset_of!(IbvSge, length) == 8);
    assert!(offset_of!(IbvSge, lkey) == 12);

    // ibv_send_wr: the whole 128-byte rdma-core 39 object, not only the prefix.
    assert!(size_of::<IbvSendWr>() == 128);
    assert!(offset_of!(IbvSendWr, next) == 8);
    assert!(offset_of!(IbvSendWr, sg_list) == 16);
    assert!(offset_of!(IbvSendWr, num_sge) == 24);
    assert!(offset_of!(IbvSendWr, opcode) == 28);
    assert!(offset_of!(IbvSendWr, send_flags) == 32);
    assert!(offset_of!(IbvSendWr, imm_data) == 36);
    assert!(offset_of!(IbvSendWr, wr) == 40);
    assert!(offset_of!(IbvSendWr, qp_type) == 72);
    assert!(offset_of!(IbvSendWr, padding) == 76);
    assert!(offset_of!(IbvSendWr, tail) == 80);
    assert!(size_of::<IbvSendWrTail>() == 48);
    assert!(size_of::<IbvMwBindInfo>() == 32);
    assert!(size_of::<IbvMwBind>() == 48);
    assert!(size_of::<IbvTso>() == 16);
    assert!(offset_of!(IbvMwBind, bind_info) == 16);

    // The operation union: rdma.remote_addr/rkey overlay the atomic arm's first fields.
    assert!(size_of::<IbvWrUnion>() == 32);
    assert!(size_of::<IbvRdmaWr>() == 16);
    assert!(size_of::<IbvAtomicWr>() == 32);
    assert!(offset_of!(IbvAtomicWr, compare_add) == 8);
    assert!(offset_of!(IbvAtomicWr, swap) == 16);
    assert!(offset_of!(IbvAtomicWr, rkey) == 24);

    // ibv_recv_wr
    assert!(size_of::<IbvRecvWr>() == 32);
    assert!(offset_of!(IbvRecvWr, sg_list) == 16);
    assert!(offset_of!(IbvRecvWr, num_sge) == 24);

    // ibv_wc
    assert!(size_of::<IbvWc>() == 48);
    assert!(offset_of!(IbvWc, status) == 8);
    assert!(offset_of!(IbvWc, opcode) == 12);
    assert!(offset_of!(IbvWc, byte_len) == 20);
    assert!(offset_of!(IbvWc, imm_data) == 24);
    assert!(offset_of!(IbvWc, qp_num) == 28);
    assert!(offset_of!(IbvWc, src_qp) == 32);
    assert!(offset_of!(IbvWc, pkey_index) == 40);
    assert!(offset_of!(IbvWc, sl) == 44);
    assert!(offset_of!(IbvWc, dlid_path_bits) == 45);

    // ibv_port_attr: rdma-core 39 is 52 bytes: 47 flags, 48 port_cap_flags2, no active_speed_ex.
    assert!(size_of::<IbvPortAttr>() == 52);
    assert!(offset_of!(IbvPortAttr, state) == 0);
    assert!(offset_of!(IbvPortAttr, max_mtu) == 4);
    assert!(offset_of!(IbvPortAttr, active_mtu) == 8);
    assert!(offset_of!(IbvPortAttr, gid_tbl_len) == 12);
    assert!(offset_of!(IbvPortAttr, pkey_tbl_len) == 32);
    assert!(offset_of!(IbvPortAttr, active_speed) == 44);
    assert!(offset_of!(IbvPortAttr, link_layer) == 46);
    assert!(offset_of!(IbvPortAttr, flags) == 47);
    assert!(offset_of!(IbvPortAttr, port_cap_flags2) == 48);

    // ibv_qp_cap / ibv_qp_init_attr
    assert!(size_of::<IbvQpCap>() == 20);
    assert!(size_of::<IbvQpInitAttr>() == 64);
    assert!(offset_of!(IbvQpInitAttr, send_cq) == 8);
    assert!(offset_of!(IbvQpInitAttr, cap) == 32);
    assert!(offset_of!(IbvQpInitAttr, qp_type) == 52);
    assert!(offset_of!(IbvQpInitAttr, sq_sig_all) == 56);

    // ibv_global_route / ibv_ah_attr
    assert!(size_of::<IbvGlobalRoute>() == 24);
    assert!(offset_of!(IbvGlobalRoute, flow_label) == 16);
    assert!(offset_of!(IbvGlobalRoute, sgid_index) == 20);
    assert!(offset_of!(IbvGlobalRoute, hop_limit) == 21);
    assert!(offset_of!(IbvGlobalRoute, traffic_class) == 22);
    assert!(size_of::<IbvAhAttr>() == 32);
    assert!(offset_of!(IbvAhAttr, dlid) == 24);
    assert!(offset_of!(IbvAhAttr, sl) == 26);
    assert!(offset_of!(IbvAhAttr, is_global) == 29);
    assert!(offset_of!(IbvAhAttr, port_num) == 30);

    // ibv_qp_attr
    assert!(size_of::<IbvQpAttr>() == 144);
    assert!(offset_of!(IbvQpAttr, cur_qp_state) == 4);
    assert!(offset_of!(IbvQpAttr, path_mtu) == 8);
    assert!(offset_of!(IbvQpAttr, qkey) == 16);
    assert!(offset_of!(IbvQpAttr, rq_psn) == 20);
    assert!(offset_of!(IbvQpAttr, sq_psn) == 24);
    assert!(offset_of!(IbvQpAttr, dest_qp_num) == 28);
    assert!(offset_of!(IbvQpAttr, qp_access_flags) == 32);
    assert!(offset_of!(IbvQpAttr, cap) == 36);
    assert!(offset_of!(IbvQpAttr, ah_attr) == 56);
    assert!(offset_of!(IbvQpAttr, alt_ah_attr) == 88);
    assert!(offset_of!(IbvQpAttr, pkey_index) == 120);
    assert!(offset_of!(IbvQpAttr, en_sqd_async_notify) == 124);
    assert!(offset_of!(IbvQpAttr, max_rd_atomic) == 126);
    assert!(offset_of!(IbvQpAttr, min_rnr_timer) == 128);
    assert!(offset_of!(IbvQpAttr, port_num) == 129);
    assert!(offset_of!(IbvQpAttr, timeout) == 130);
    assert!(offset_of!(IbvQpAttr, retry_cnt) == 131);
    assert!(offset_of!(IbvQpAttr, rnr_retry) == 132);
    assert!(offset_of!(IbvQpAttr, alt_timeout) == 134);
    assert!(offset_of!(IbvQpAttr, rate_limit) == 136);

    // Public prefixes the backend reads back from library-owned objects.
    assert!(size_of::<IbvQp>() == 64);
    assert!(offset_of!(IbvQp, pd) == 16);
    assert!(offset_of!(IbvQp, srq) == 40);
    assert!(offset_of!(IbvQp, handle) == 48);
    assert!(offset_of!(IbvQp, qp_num) == 52);
    assert!(offset_of!(IbvQp, state) == 56);
    assert!(offset_of!(IbvQp, qp_type) == 60);

    assert!(size_of::<IbvMr>() == 48);
    assert!(offset_of!(IbvMr, addr) == 16);
    assert!(offset_of!(IbvMr, length) == 24);
    assert!(offset_of!(IbvMr, handle) == 32);
    assert!(offset_of!(IbvMr, lkey) == 36);
    assert!(offset_of!(IbvMr, rkey) == 40);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gid_classification_names_the_three_families() {
        let zero = IbvGid::default();
        assert_eq!(zero.kind(), GidKind::Unspecified);
        assert!(zero.describe().contains("unspecified"));

        // ::ffff:198.51.100.80
        let mut ipv4 = [0u8; 16];
        ipv4[10] = 0xff;
        ipv4[11] = 0xff;
        ipv4[12] = 10;
        ipv4[13] = 0;
        ipv4[14] = 1;
        ipv4[15] = 80;
        let gid = IbvGid { raw: ipv4 };
        assert_eq!(gid.kind(), GidKind::RoceV2Ipv4([10, 0, 1, 80]));
        assert_eq!(gid.describe(), "RoCEv2 IPv4 198.51.100.80");

        // fe80::1: an IPv6 link-local GID.
        let mut ipv6 = [0u8; 16];
        ipv6[0] = 0xfe;
        ipv6[1] = 0x80;
        ipv6[15] = 1;
        let gid = IbvGid { raw: ipv6 };
        assert_eq!(gid.kind(), GidKind::OtherIpv6);
        assert!(gid.describe().starts_with("fe80"));
    }

    #[test]
    fn mtu_names_and_byte_counts_round_trip() {
        for (value, bytes) in [
            (mtu::MTU_256, 256),
            (mtu::MTU_512, 512),
            (mtu::MTU_1024, 1024),
            (mtu::MTU_2048, 2048),
            (mtu::MTU_4096, 4096),
        ] {
            assert_eq!(mtu::bytes(value), Some(bytes));
            assert_eq!(mtu::name(value), format!("{bytes} B"));
        }
        assert_eq!(mtu::bytes(0), None);
        assert_eq!(mtu::name(99), "unknown MTU enum 99");
    }

    #[test]
    fn completion_decoding_separates_receive_from_send() {
        assert!(!wc::is_receive(wc::SEND));
        assert!(!wc::is_receive(wc::RDMA_WRITE));
        assert!(wc::is_receive(wc::RECV));
        assert!(wc::is_receive(wc::RECV_RDMA_WITH_IMM));
        assert_eq!(wc::status_name(wc::SUCCESS), "IBV_WC_SUCCESS");
        assert_eq!(wc::status_name(12), "IBV_WC_RETRY_EXC_ERR");
        assert!(wc::status_name(9999).contains("unknown"));
    }

    #[test]
    fn the_send_wr_rdma_arm_overlays_the_c_offsets() {
        let wr = IbvSendWr {
            wr_id: 7,
            next: std::ptr::null_mut(),
            sg_list: std::ptr::null_mut(),
            num_sge: 1,
            opcode: wr_opcode::RDMA_WRITE_WITH_IMM,
            send_flags: send_flags::SIGNALED,
            imm_data: 0x0102_0304,
            wr: IbvWrUnion {
                rdma: IbvRdmaWr {
                    remote_addr: 0x1122_3344_5566_7788,
                    rkey: 0x99aa_bbcc,
                },
            },
            qp_type: IbvQpTypeUnion { raw: [0u8; 4] },
            padding: [0u8; 4],
            tail: IbvSendWrTail { raw: [0u8; 48] },
        };
        // SAFETY: the union is initialised by the rdma arm; the byte offsets are asserted above.
        unsafe {
            let raw = wr.wr.raw;
            assert_eq!(&raw[..8], &0x1122_3344_5566_7788u64.to_ne_bytes());
            assert_eq!(&raw[8..12], &0x99aa_bbccu32.to_ne_bytes());
            assert_eq!(wr.wr.rdma.rkey, 0x99aa_bbcc);
        }
    }
}
