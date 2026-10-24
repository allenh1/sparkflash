//! The layer port's cross-rank all-reduce: one communicator at setup, one enqueue per collective,
//! no host readback of the result.
//!
//! This is the drop-in API the whole-layer device port calls where the lean loop currently runs its
//! spawn/join machinery (moe_reduce, and the reduce inside the attention-output block). The lean
//! loop's host reduce is a blocking exchange: the host posts, waits and copies the reduced row back
//! before the next block can run, which is why its per-call cost is the whole exchange. This wrapper
//! changes the shape of the call, not its arithmetic:
//!
//! * Setup opens one communicator and hands the run an opaque handle (RunReduce::open). The run
//!   stores one word; the library and the raw NCCL communicator live behind it and are destroyed in
//!   Drop. DeviceReduce is the same collector in borrowing form for a caller that already owns the
//!   loaded library (the bench does), so both forms share the arm rule and the counters.
//! * Each collective is one enqueue on the caller's CUDA stream (RunReduce::reduce_bf16, or the
//!   free function reduce on the handle). It does not synchronise the stream and it does not copy
//!   the result to the host: the reduced row stays where the caller's device kernels read it, and a
//!   later host readback is ordered after the same stream by the caller's existing sync points.
//! * The counters in DeviceReduceStats are what the port reports instead of a per-position D2H
//!   observation point: how many collectives ran and over how many elements, plus the communicator's
//!   init-time device reserve (arena_bytes) so the run's memory model charges the arm honestly.
//!
//! # Layer-port call-site contract
//!
//! The tp4_forward.rs half of the port codes against exactly this sequence:
//!
//! 1. Setup, once per run, on every rank: rank 0 asks for the unique id (unique_id) and broadcasts
//!    its 128 bytes over the run's own bootstrap channel, then each rank calls RunReduce::open(world,
//!    rank, unique_id). The library is loaded inside open, so a host without libnccl refuses with a
//!    typed LibraryAbsent instead of producing a handle.
//! 2. Setup warm-up: RunReduce::warm_up(stream, scratch, count) enqueues WARM_UP_COLLECTIVES dummy
//!    reduces on the caller's stream, then the port takes its one setup sync. NCCL pays its
//!    connection setup in the first collectives, so layer 0 is ramping and is not steady state.
//! 3. Per collective: reduce(handle, stream, input, output, count) or the method form, once per
//!    collective, in the same order on every rank. No synchronise and no host readback. A host sync
//!    here to close a timer is a port bug, not a slow path.
//! 4. Run boundary: the step's own sync (the only one), an NCCL async-failure check (failure
//!    detection, not completion), then the stats read, then close(handle) exactly once on the
//!    path that also closes the transport.
//!
//! The async-failure check is [`RunReduce::check_async_failure`]. It is optional at the symbol
//! level: a library that does not export `ncclCommGetAsyncError` loads and runs, the check returns
//! `Ok` and [`RunReduce::async_failure_check_available`] is false, and the port's step sync is then
//! the only detection (the run record must say so rather than claim a check it could not perform).
//! When the symbol is present, `ncclSuccess` and `ncclInProgress` are "no failure" and every other
//! communicator status is [`NcclError::AsyncFailure`] - a failure detected after completion has
//! already been observed, so the run must abort on every rank.
//!
//! Invariants: the reduced row never crosses PCIe in steady state; the only legitimate readback is
//! the logits at the position boundary; every rank enqueues the same collectives in the same order,
//! so no rank-conditional path may skip a reduce - a skip hangs the communicator instead of failing
//! a check. The arm is chosen once at open and never switched mid-run. Neither arm allocates per
//! call: the communicator's device buffers are init-time (arena_bytes) and the reduce_ws span the
//! layer reserves is the RDMA arm's tree scratch, not NCCL's.
//!
//! ## Arithmetic, stated honestly
//!
//! The reduce is NCCL's ncclSum over ncclBfloat16. Under D54 that is the declared tolerance arm: a
//! four-rank sum with the same bf16 output rounding as the host reduce, up to the order of the
//! additions inside NCCL's ring. It is NOT the canonical (bit-identical) path, which exists in
//! crate::reduce and is what crate::tcp and the one-shot RDMA path replay. The measured deviation of
//! the same payload against the canonical f32-accumulate reduce is what scripts/gpu-reduce-bench.sh
//! reports (differing elements and max ULP), and the D54 logits bars are the acceptance criterion,
//! not this wrapper's opinion.
//!
//! ## Tolerance-arm declaration (D54), written before any number exists
//!
//! The arm is: a four-rank bf16 Sum all-reduce on NCCL - one f32 accumulation per element and one
//! bf16 round on the result - over the same exchange payload, between the same four peers, in the
//! same order of collectives per layer as the host path. The canonical reference is the current
//! rank-order f32-accumulate-then-round path in crate::reduce, which remains the strict-mode
//! contract. The measured deviation of this arm is the bench's differing-element count and max ULP
//! on the 10,240 B payload, and the row-level proof is the numerical gate on a whole-layer row:
//! greedy tokens identical, top-1 1.00, top-5 at least 0.80, top-50 at least 0.96, max
//! |dlogprob| at most 2.0e-2, mean at most 2.0e-3, KL at most 1.0e-4, all finite - with the
//! fixture's tightest top1-top2 margin, 0.0038 nats at position 4, as the position that decides it.
//!
//! NCCL's reduce order inside its ring is not the canonical rank order. That is the arm, not a
//! defect: D54 admits it, the bench measures its deviation, and any claim beyond the declared bars
//! would have to go back through the canonical path instead.
//!
//! ## The declared S1 host-bridge exception (typed, counted, opt-in, never a fallback)
//!
//! S1 ships one declared exception to "the reduced row never crosses PCIe in steady state": when the
//! device arm cannot run on a node, the two per-layer collectives MAY run through the declared
//! host-bridge arm - a device->host copy of this rank's row, the host reduce the lean loop already
//! implements ([`crate::tcp::TcpTransport`] over the run's own transport), and a host->device copy
//! back into the caller's output span. The declaration is these five clauses:
//!
//! 1. **It is the only permitted copy besides the final logits.** No other stage, seam, observation
//!    point or intermediate row may acquire a D2H/H2D pair; the bridge is not a general escape hatch.
//! 2. **Only the two per-layer collectives are representable.**
//!    [`crate::host_bridge::BridgeStage`] has exactly two variants, the attention delta (`all_reduce`)
//!    and the routed MoE delta (`all_reduce_moe`), and [`crate::host_bridge::BridgeStage::parse`]
//!    refuses every other stage name. A stage cannot ask for the bridge and get one.
//! 3. **It is opt-in, twice.** The implementation is behind the off-by-default `s1-host-bridge`
//!    cargo feature, and a run must construct [`crate::host_bridge::HostBridgeReduce`] for it
//!    explicitly. No default path, inherited arm or environment variable reaches it implicitly.
//! 4. **It is typed.** It is not a device communicator: [`ReduceArm::HostBridge`] exists so the row
//!    can say `reduce=host-bridge`, and [`ReduceArm::require_available`] refuses it for
//!    [`RunReduce`]/[`DeviceReduce`] before any library or communicator is touched. There is no code
//!    path that opens NCCL when the host bridge was selected - a silent fallback in either direction
//!    is the failure this type exists to make impossible.
//! 5. **It is counted.** A completed bridge reduce records the same collective/element/byte counters
//!    as a device reduce, plus `bridge_collectives`, `bridge_d2h_bytes` and `bridge_h2d_bytes` in
//!    [`DeviceReduceStats`], so the run record can print `reduce=host-bridge` and the extra copies
//!    without a new observation point. A bridge that fails at either copy or at the host reduce is
//!    not counted: the run is aborting, and a partial bridge must not appear as completed work.
//!
//! Arithmetic: the host reduce is the canonical rank-order f32-accumulate-then-round path
//! ([`crate::reduce`]), so this arm is the strict-mode arithmetic rather than the NCCL arm's ring
//! order; D54's declared tolerance is over-satisfied by construction, and the cost is the two PCIe
//! copies plus the blocking exchange. It exists for an S1 run that cannot start the device arm, not
//! as a performance path, and it is never enabled by default.
//!
//! ## Why device addresses, not buffers
//!
//! The port's rows live in the arena and are handed around as Deviceptr; this crate has no CUDA
//! driver dependency and should not grow one just to say "pointer". The API therefore takes the
//! device addresses as u64 and the stream as a u64, exactly the values the caller already holds. The
//! caller is responsible for the same lifetime and ordering contract any device kernel has: both
//! addresses on the current device, count bf16 values reachable at each, and the stream valid.

use std::ffi::c_void;
use std::mem::size_of;

use crate::nccl::{
    self, CudaStream, NCCL_UNIQUE_ID_BYTES, NcclApi, NcclComm, NcclError, NcclRedOp, NcclUniqueId,
};
use crate::transport::Transport;

/// Which transport the layer's cross-rank all-reduce uses.
///
/// Both arms sit behind the same DeviceReduce API and the same D54 tolerance declaration; only the
/// wire changes. The arm is chosen once at init and never switched mid-run, so a run's numbers
/// always belong to one transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceArm {
    /// NCCL over its own transport: the vendored library, dlopen'd at run time.
    Nccl,
    /// GPU-initiated RDMA with GPUDirect: the reduce kernels write peer device memory over the
    /// fabric. Designed but not implemented while nvidia-peermem is root-blocked.
    GpuRdma,
    /// The declared S1 exception: device->host, host reduce, host->device. This is NOT a device
    /// collector arm - it is the stable label (`host-bridge`) for the one permitted copy path, and
    /// every device API refuses it by name. Open it only as
    /// [`crate::host_bridge::HostBridgeReduce`] under the off-by-default `s1-host-bridge` feature;
    /// a run that selects this label can never fall through to NCCL.
    HostBridge,
}

impl ReduceArm {
    /// Stable label for the run record and the typed refusals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nccl => "nccl",
            Self::GpuRdma => "gpu_rdma",
            Self::HostBridge => "host-bridge",
        }
    }

    /// Whether this arm is a device communicator (the two reduced rows never cross PCIe).
    ///
    /// The host-bridge arm is the declared S1 exception and is deliberately not one: it copies both
    /// ways, so a caller that holds a device handle cannot mistake it for a device arm.
    #[must_use]
    pub const fn is_device(self) -> bool {
        !matches!(self, Self::HostBridge)
    }

    /// Whether this build contains the arm's implementation.
    #[must_use]
    pub const fn is_built(self) -> bool {
        match self {
            Self::Nccl => true,
            Self::GpuRdma => cfg!(feature = "gpu-rdma-reduce"),
            Self::HostBridge => cfg!(feature = "s1-host-bridge"),
        }
    }

    /// Refuses with a typed error when the arm cannot run in this build or on this host.
    ///
    /// This is the no-silent-fallback rule: a caller that asked for gpu_rdma never gets NCCL's
    /// transport without asking again, because a run that fell back silently would publish numbers
    /// under the wrong arm. The same rule makes the host-bridge label refuse every device call site
    /// unconditionally - it is only ever opened as
    /// [`crate::host_bridge::HostBridgeReduce`], never as a device communicator.
    ///
    /// # Errors
    /// Returns NcclError::ArmUnavailable naming the arm and why it cannot run.
    pub fn require_available(self) -> Result<(), NcclError> {
        match self {
            Self::Nccl => Ok(()),
            Self::GpuRdma => Err(NcclError::ArmUnavailable {
                arm: self.as_str(),
                detail: if cfg!(feature = "gpu-rdma-reduce") {
                    "the arm is compiled in but not implemented: GPU-initiated RDMA needs nvidia-peermem loaded, which is root-blocked on the four nodes; selecting the feature does not start it".to_owned()
                } else {
                    "this build has the gpu-rdma-reduce feature off; rebuild with the feature to arm it (no silent fallback to nccl)".to_owned()
                },
            }),
            Self::HostBridge => Err(NcclError::ArmUnavailable {
                arm: self.as_str(),
                detail: format!(
                    "the host-bridge arm is the declared S1 exception (device->host, host reduce, host->device), not a device collector: RunReduce/DeviceReduce refuse it so a run can never enqueue NCCL under the host-bridge label (no silent fallback in either direction); open it as HostBridgeReduce under the {} feature, which is {} in this build",
                    crate::host_bridge::HOST_BRIDGE_FEATURE,
                    if cfg!(feature = "s1-host-bridge") {
                        "enabled"
                    } else {
                        "off"
                    }
                ),
            }),
        }
    }
}

/// Checks a world/rank spec before any NCCL call, so a bad configuration is a typed refusal rather
/// than a bootstrap hang. A one-rank world is legal (NCCL supports it) and is the degenerate shape
/// the tests pin alongside the zero-element payload.
fn validate_spec(world: usize, rank: usize) -> Result<(), NcclError> {
    if world == 0 {
        return Err(NcclError::Config(
            "a device all-reduce needs at least one rank".to_owned(),
        ));
    }
    if rank >= world {
        return Err(NcclError::Config(format!(
            "rank {rank} is out of range for world {world}"
        )));
    }
    Ok(())
}

/// Total bytes of one bf16 all-reduce payload, if it is representable.
///
/// Returns NcclError::Config for a zero-element collective (a collective over nothing is a bug, not
/// a fast path) and for a count whose byte size overflows.
fn payload_bytes(count: usize) -> Result<u64, NcclError> {
    if count == 0 {
        return Err(NcclError::Config(
            "a device all-reduce needs at least one element".to_owned(),
        ));
    }
    let elements = u64::try_from(count)
        .map_err(|_| NcclError::Config(format!("count {count} does not fit in a u64")))?;
    elements
        .checked_mul(size_of::<u16>() as u64)
        .ok_or_else(|| NcclError::Config(format!("count {count} overflows the byte size")))
}

/// What the layer reports about its cross-rank collectives instead of a per-position D2H record.
///
/// Every field is a monotone counter of work actually completed; a failed enqueue is not counted, so
/// a run that falls back to the host reduce reports fewer device collectives rather than a lie. The
/// `bridge_*` fields are the declared S1 exception's account: how many of those collectives went
/// through the device->host / host reduce / host->device path and how many bytes each direction
/// carried. They stay zero on a device-only run, which is what makes "no PCIe copy in steady state"
/// a checkable statement in the record rather than a comment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceReduceStats {
    collectives: u64,
    elements: u64,
    bytes: u64,
    arena_bytes: u64,
    bridge_collectives: u64,
    bridge_d2h_bytes: u64,
    bridge_h2d_bytes: u64,
    registered_buffers: u64,
    registered_bytes: u64,
}

impl DeviceReduceStats {
    /// Collectives this process has enqueued.
    #[must_use]
    pub const fn collectives(&self) -> u64 {
        self.collectives
    }

    /// bf16 elements carried by those collectives (counted once, not per rank).
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    /// Payload bytes carried by those collectives (bf16 elements x 2).
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Device bytes the communicator reserves at open, for the run's arena accounting.
    ///
    /// NCCL allocates its internal channel buffers at init, never per collective. The port adds this
    /// number to the arena it charges for the reduce arm, so the memory model does not pretend the
    /// communicator is free. Zero on a default-constructed value.
    #[must_use]
    pub const fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    /// Collectives of the total that ran through the declared S1 host bridge.
    ///
    /// Non-zero only under `reduce=host-bridge`; a device-only run reports zero here, which is the
    /// in-record proof that the steady-state loop took no D2H/H2D copy.
    #[must_use]
    pub const fn bridge_collectives(&self) -> u64 {
        self.bridge_collectives
    }

    /// Bytes copied device->host by the host bridge (one payload per bridged collective).
    #[must_use]
    pub const fn bridge_d2h_bytes(&self) -> u64 {
        self.bridge_d2h_bytes
    }

    /// Bytes copied host->device by the host bridge (one payload per bridged collective).
    #[must_use]
    pub const fn bridge_h2d_bytes(&self) -> u64 {
        self.bridge_h2d_bytes
    }

    /// Copies per bridged collective: the D2H half and the H2D half. Fixed at two by the bridge's
    /// contract, so a caller cannot invent a one-way bridge.
    #[must_use]
    pub const fn bridge_copies(&self) -> u64 {
        self.bridge_collectives.wrapping_mul(2)
    }

    /// Buffers the run registered with the communicator at open (`ncclCommRegister`).
    ///
    /// Zero means nothing was registered: the record must then say so rather than claim the S2-a
    /// rule held. Registration is setup, not layer work, so it never moves the collective counters.
    #[must_use]
    pub const fn registered_buffers(&self) -> u64 {
        self.registered_buffers
    }

    /// Bytes covered by those registrations, summed over the buffers as the caller passed them.
    #[must_use]
    pub const fn registered_bytes(&self) -> u64 {
        self.registered_bytes
    }

    fn set_arena_bytes(&mut self, bytes: u64) {
        self.arena_bytes = bytes;
    }

    fn set_registered(&mut self, buffers: u64, bytes: u64) {
        self.registered_buffers = buffers;
        self.registered_bytes = bytes;
    }

    fn record(&mut self, bytes: u64) {
        self.record_completed(bytes);
    }

    /// Records one completed host-bridge collective: the same collective/element/byte counters a
    /// device collective moves, plus the two copy counters. A failed bridge is not recorded, so a
    /// run whose bridge aborted cannot claim the collective.
    pub(crate) fn record_bridged(&mut self, bytes: u64) {
        self.record_completed(bytes);
        self.bridge_collectives = self.bridge_collectives.wrapping_add(1);
        self.bridge_d2h_bytes = self.bridge_d2h_bytes.wrapping_add(bytes);
        self.bridge_h2d_bytes = self.bridge_h2d_bytes.wrapping_add(bytes);
    }

    fn record_completed(&mut self, bytes: u64) {
        let elements = bytes / size_of::<u16>() as u64;
        self.collectives = self.collectives.wrapping_add(1);
        self.elements = self.elements.wrapping_add(elements);
        self.bytes = self.bytes.wrapping_add(bytes);
    }
}

/// One NCCL communicator for the run, borrowed from a loaded library.
///
/// Create it once during setup and keep it for the whole run: the communicator is the expensive part
/// (it bootstraps the four ranks), while a per-layer reduce is one enqueue on an existing stream.
/// The wrapper borrows the NcclApi that created it, so the library outlives the communicator by
/// construction and NcclComm's destructor runs while the library is still loaded.
pub struct DeviceReduce<'api> {
    api: &'api NcclApi,
    comm: NcclComm<'api>,
    arm: ReduceArm,
    world: usize,
    rank: usize,
    stats: DeviceReduceStats,
}

impl<'api> DeviceReduce<'api> {
    /// Creates this rank's communicator from the run's shared unique id.
    ///
    /// # Errors
    /// Returns NcclError::Config for a rank outside the world or a zero world, and NcclError::Call
    /// when NCCL refuses the communicator (a missing peer, a wrong device, a library that cannot
    /// bootstrap).
    pub fn init(
        api: &'api NcclApi,
        world: usize,
        rank: usize,
        unique_id: NcclUniqueId,
    ) -> Result<Self, NcclError> {
        Self::init_with_arm(api, ReduceArm::Nccl, world, rank, unique_id)
    }

    /// Creates this rank's communicator for the arm named at setup.
    ///
    /// The arm is refused before any NCCL call when it cannot run (a build without the feature, or
    /// the designed-but-unimplemented RDMA arm), so selecting an arm is never a silent fallback.
    ///
    /// # Errors
    /// Returns NcclError::ArmUnavailable when the arm cannot run, NcclError::Config for a rank
    /// outside the world or a zero world, and NcclError::Call when NCCL refuses the communicator.
    pub fn init_with_arm(
        api: &'api NcclApi,
        arm: ReduceArm,
        world: usize,
        rank: usize,
        unique_id: NcclUniqueId,
    ) -> Result<Self, NcclError> {
        arm.require_available()?;
        validate_spec(world, rank)?;
        let comm = api.comm_init_rank_config(world, rank, unique_id)?;
        Ok(Self {
            api,
            comm,
            arm,
            world,
            rank,
            stats: DeviceReduceStats::default(),
        })
    }

    /// The arm this communicator was created for.
    #[must_use]
    pub const fn arm(&self) -> ReduceArm {
        self.arm
    }

    /// Ranks in the collective.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// This process's rank in the collective.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// The loaded library's version string, for the run record.
    #[must_use]
    pub fn library_version(&self) -> Option<String> {
        self.api.version_string()
    }

    /// The counters as of this call.
    #[must_use]
    pub const fn stats(&self) -> DeviceReduceStats {
        self.stats
    }

    /// Enqueues one bf16 AllReduce(Sum) from send into recv on stream.
    ///
    /// send and recv are device addresses (the u64 inside a Deviceptr), count is bf16 elements, and
    /// stream is the CUDA stream handle the surrounding device block already uses. The call returns
    /// once the collective is enqueued; the result is on the device and the caller orders its
    /// consumers after the stream. Nothing is read back to the host here.
    ///
    /// # Safety
    /// The caller must guarantee that send and recv point to at least count bf16 values each in
    /// memory NCCL can access on the current device, that recv is writable, that stream is a valid
    /// stream on that device, and that every rank enqueues the same sequence of collectives.
    /// Violating any of those is undefined behaviour inside NCCL, which this crate cannot check from
    /// a host without a GPU.
    ///
    /// # Errors
    /// Returns NcclError::Config for a zero or overflowing count, and NcclError::Call when NCCL
    /// refuses the enqueue.
    pub unsafe fn reduce_bf16(
        &mut self,
        send: u64,
        recv: u64,
        count: usize,
        stream: u64,
    ) -> Result<(), NcclError> {
        let bytes = payload_bytes(count)?;
        // SAFETY: the caller upholds the pointer, device and stream contract documented above.
        unsafe {
            self.comm.all_reduce_bf16(
                send as *const std::ffi::c_void,
                recv as *mut std::ffi::c_void,
                count,
                NcclRedOp::Sum,
                stream as CudaStream,
            )?;
        }
        self.stats.record(bytes);
        Ok(())
    }

    /// Enqueues one in-place bf16 AllReduce(Sum) over buffer.
    ///
    /// NCCL accepts the same address as send and receive for AllReduce; the layer port uses this
    /// when the delta row is the buffer that already holds this rank's contribution.
    ///
    /// # Safety
    /// Same contract as reduce_bf16, with buffer used for both directions.
    ///
    /// # Errors
    /// Returns NcclError::Config for a zero or overflowing count, and NcclError::Call when NCCL
    /// refuses the enqueue.
    pub unsafe fn reduce_bf16_in_place(
        &mut self,
        buffer: u64,
        count: usize,
        stream: u64,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the pointer, device and stream contract of reduce_bf16.
        unsafe { self.reduce_bf16(buffer, buffer, count, stream) }
    }

    /// The raw communicator, for a caller that needs to pass it to another NCCL entry point.
    #[must_use]
    pub fn as_raw(&self) -> nccl::NcclCommHandle {
        self.comm.as_raw()
    }

    /// Queries the communicator's asynchronous error status at the run boundary.
    ///
    /// Failure detection, not completion: call it after the step's sync and before reading
    /// [`Self::stats`]. A library without `ncclCommGetAsyncError` returns `Ok` and
    /// [`Self::async_failure_check_available`] is false (a documented no-op, not a failure).
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when the query itself fails and [`NcclError::AsyncFailure`] when
    /// the communicator reports an asynchronous failure; either one is an all-rank abort.
    pub fn check_async_failure(&self) -> Result<(), NcclError> {
        // SAFETY: the communicator is live for as long as self borrows the library.
        unsafe { self.api.check_async_failure(self.comm.as_raw()) }
    }

    /// Whether the loaded library exports the run-boundary failure check.
    ///
    /// False means only the port's step sync detects a failure; the run record should state that
    /// instead of claiming this check ran.
    #[must_use]
    pub fn async_failure_check_available(&self) -> bool {
        self.api.has_async_failure_check()
    }
}

impl std::fmt::Debug for DeviceReduce<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceReduce")
            .field("arm", &self.arm)
            .field("world", &self.world)
            .field("rank", &self.rank)
            .field("stats", &self.stats)
            .finish()
    }
}

/// The run-owned communicator: an opaque handle the layer port holds as one raw pointer.
///
/// DeviceReduce borrows the loaded library, which a run context cannot store as an owned field
/// without a self-referential struct. RunReduce owns both the library and the raw NCCL communicator,
/// destroys the communicator in Drop, and is handed to the port as a pointer: the layer stores the
/// pointer and passes it back to reduce, warm_up, stats and close. Nothing self-referential appears
/// on either side.
pub struct RunReduce {
    api: NcclApi,
    comm: nccl::NcclCommHandle,
    arm: ReduceArm,
    world: usize,
    rank: usize,
    stats: DeviceReduceStats,
    /// NCCL's handles for the buffers registered at open, released before the communicator.
    registrations: Vec<*mut c_void>,
}

impl Drop for RunReduce {
    fn drop(&mut self) {
        // Drop runs before the fields, so the registrations are released and the communicator is
        // destroyed while the library is still loaded. A failed release or destroy still consumes
        // the handle, so the error is not actionable here.
        let _ = self.deregister_buffers();
        let _ = self.api.comm_destroy_handle(self.comm);
    }
}

/// Dummy collectives the setup warm-up enqueues.
///
/// NCCL pays its connection setup in the first collectives; three enqueues plus the port's one setup
/// sync put that cost before layer 0 instead of inside it.
pub const WARM_UP_COLLECTIVES: usize = 3;

/// Conservative device bytes to charge the communicator in the run's arena model.
///
/// NCCL allocates its internal channel buffers at init, and the exact size depends on the library
/// build and the topology. This is the default reserve; a port that measures its own pre/post
/// allocation delta passes that number to open_with_arm instead, and either way the number lands in
/// DeviceReduceStats::arena_bytes so the run record states what the arm costs.
pub const NCCL_ARENA_RESERVE_BYTES: u64 = 16 * 1024 * 1024;

impl RunReduce {
    /// Opens the communicator and returns the run's opaque handle.
    ///
    /// The library is loaded here, so a host without libnccl refuses with NcclError::LibraryAbsent
    /// instead of producing a handle. Nothing is enqueued or synchronised: warm_up does the setup
    /// collectives, and the port's own setup sync completes them.
    ///
    /// # Errors
    /// Returns NcclError::ArmUnavailable and NcclError::Config before any NCCL call,
    /// NcclError::LibraryAbsent when the library is missing, and NcclError::Call when NCCL refuses
    /// the communicator.
    pub fn open(
        world: usize,
        rank: usize,
        unique_id: NcclUniqueId,
    ) -> Result<*mut c_void, NcclError> {
        Self::open_with_arm(
            world,
            rank,
            unique_id,
            ReduceArm::Nccl,
            NCCL_ARENA_RESERVE_BYTES,
        )
    }

    /// Opens the communicator for a named arm, charging arena_bytes to the run's memory model.
    ///
    /// # Errors
    /// Same as open, plus the arm's own refusal when the arm cannot run in this build.
    pub fn open_with_arm(
        world: usize,
        rank: usize,
        unique_id: NcclUniqueId,
        arm: ReduceArm,
        arena_bytes: u64,
    ) -> Result<*mut c_void, NcclError> {
        arm.require_available()?;
        validate_spec(world, rank)?;
        let api = NcclApi::load()?;
        let comm = api.comm_init_rank_handle(world, rank, unique_id)?;
        let mut stats = DeviceReduceStats::default();
        stats.set_arena_bytes(arena_bytes);
        let run = Self {
            api,
            comm,
            arm,
            world,
            rank,
            stats,
            registrations: Vec::new(),
        };
        Ok(Box::into_raw(Box::new(run)).cast::<c_void>())
    }

    /// Enqueues one bf16 AllReduce(Sum) from input into output on the caller's stream.
    ///
    /// Same contract as DeviceReduce::reduce_bf16: device addresses, count in bf16 elements, no
    /// synchronise and no readback. The counters advance only after the enqueue succeeds.
    ///
    /// # Safety
    /// input and output must point to at least count bf16 values each in memory NCCL can access on
    /// the current device, output must be writable, stream must be a valid stream there, and every
    /// rank must enqueue the same sequence of collectives.
    ///
    /// # Errors
    /// Returns NcclError::Config for a zero or overflowing count, and NcclError::Call when NCCL
    /// refuses the enqueue.
    pub unsafe fn reduce_bf16(
        &mut self,
        stream: u64,
        input: u64,
        output: u64,
        count: usize,
    ) -> Result<(), NcclError> {
        let bytes = payload_bytes(count)?;
        // SAFETY: the caller upholds the device-address and stream contract documented above.
        unsafe {
            self.api.enqueue_all_reduce_bf16(
                self.comm,
                input as *const c_void,
                output as *mut c_void,
                count,
                NcclRedOp::Sum,
                stream as CudaStream,
            )?;
        }
        self.stats.record(bytes);
        Ok(())
    }

    /// Enqueues the setup warm-up: WARM_UP_COLLECTIVES in-place reduces over scratch.
    ///
    /// Use the layer's real payload count so the same path is warmed. The call returns after
    /// enqueuing and does not synchronise: this crate has no CUDA dependency, so the port's own setup
    /// sync is what completes the warm-up. Warm-up collectives are not counted in stats, because
    /// they are setup rather than layer work.
    ///
    /// # Safety
    /// scratch must be a valid device address of count writable bf16 values on the current device,
    /// stream must be a valid stream there, and every rank must call this with the same count.
    ///
    /// # Errors
    /// Returns NcclError::Config for a zero or overflowing count, and NcclError::Call when NCCL
    /// refuses an enqueue.
    pub unsafe fn warm_up(
        &mut self,
        stream: u64,
        scratch: u64,
        count: usize,
    ) -> Result<(), NcclError> {
        payload_bytes(count)?;
        for _ in 0..WARM_UP_COLLECTIVES {
            // SAFETY: the caller upholds the device-address and stream contract documented above.
            unsafe {
                self.api.enqueue_all_reduce_bf16(
                    self.comm,
                    scratch as *const c_void,
                    scratch as *mut c_void,
                    count,
                    NcclRedOp::Sum,
                    stream as CudaStream,
                )?;
            }
        }
        Ok(())
    }

    /// Registers the run's collective buffers with the communicator: once, at open (S2-a).
    ///
    /// `spans` are `(device address, bytes)` pairs: the layer's collective inputs and outputs. They
    /// are fixed for the whole run, so registering them before the first position means no step
    /// pays a registration and a later graph capture (S3) sees the same registered addresses.
    /// A second call while registrations are held is refused: the run registers once, and a caller
    /// that needs a new set (the bench, between payload sizes) releases the old one first with
    /// [`Self::deregister_buffers`]. On a failure part-way, the registrations already made are
    /// released before the error returns, so a refused open holds nothing.
    ///
    /// # Safety
    /// Every span must be device memory on the communicator's device that stays allocated until
    /// [`Self::deregister_buffers`] (or Drop) releases it, and no collective may be in flight on
    /// these buffers while they are released.
    ///
    /// # Errors
    /// Returns NcclError::Config for an empty span list, a repeated registration, or a null or empty
    /// span, and NcclError::Call when NCCL refuses a registration.
    pub unsafe fn register_buffers(&mut self, spans: &[(u64, usize)]) -> Result<(), NcclError> {
        if spans.is_empty() {
            return Err(NcclError::Config(
                "register_buffers needs at least one span; the run registers its collective buffers at open".to_owned(),
            ));
        }
        if !self.registrations.is_empty() {
            return Err(NcclError::Config(format!(
                "register_buffers called while {} registration(s) are held; the run registers its collective buffers once, at open",
                self.registrations.len()
            )));
        }
        let mut bytes = 0u64;
        for (address, len) in spans {
            // SAFETY: the caller upholds the span contract; the communicator is live until Drop.
            let registered = unsafe {
                self.api
                    .comm_register_handle(self.comm, *address as *mut c_void, *len)
            };
            match registered {
                Ok(registration) => {
                    self.registrations.push(registration);
                    bytes = bytes.saturating_add(*len as u64);
                }
                Err(error) => {
                    let _ = self.deregister_buffers();
                    return Err(error);
                }
            }
        }
        self.stats
            .set_registered(self.registrations.len() as u64, bytes);
        Ok(())
    }

    /// Releases every registration [`Self::register_buffers`] made, newest first.
    ///
    /// Call it after the last collective on those buffers has completed and before the buffers are
    /// freed; Drop calls it too, so a run that forgot still releases them before the destroy. The
    /// registered counters fall back to zero. Every registration is released even when one fails,
    /// and the first failure is the one returned.
    ///
    /// # Errors
    /// Returns NcclError::Call when NCCL refuses a release.
    pub fn deregister_buffers(&mut self) -> Result<(), NcclError> {
        let mut first_error = None;
        while let Some(registration) = self.registrations.pop() {
            // SAFETY: the registration was made on this live communicator by register_buffers and
            // is released exactly once here (it has just been popped).
            let released = unsafe { self.api.comm_deregister_handle(self.comm, registration) };
            if let Err(error) = released
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        self.stats.set_registered(0, 0);
        first_error.map_or(Ok(()), Err)
    }

    /// The arm this communicator was opened for.
    #[must_use]
    pub const fn arm(&self) -> ReduceArm {
        self.arm
    }

    /// Ranks in the collective.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// This process's rank in the collective.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// The loaded library's version string, for the run record.
    #[must_use]
    pub fn library_version(&self) -> Option<String> {
        self.api.version_string()
    }

    /// The loaded library's soname (`libnccl.so.2` on the nodes), for the run record.
    #[must_use]
    pub fn library_name(&self) -> &str {
        self.api.library_name()
    }

    /// The counters as of this call.
    #[must_use]
    pub const fn stats(&self) -> DeviceReduceStats {
        self.stats
    }

    /// Queries the communicator's asynchronous error status at the run boundary.
    ///
    /// Call it after the step's sync and before reading [`Self::stats`]: failure detection, not
    /// completion. A library that predates `ncclCommGetAsyncError` returns `Ok` and
    /// [`Self::async_failure_check_available`] is false - a documented no-op, not a failure; the run
    /// record must then state that the sync was the only detection.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when the query itself fails and [`NcclError::AsyncFailure`] when
    /// the communicator reports an asynchronous failure; either one means the run must abort on
    /// every rank.
    pub fn check_async_failure(&self) -> Result<(), NcclError> {
        // SAFETY: the communicator is live until Drop, which runs before the library is released.
        unsafe { self.api.check_async_failure(self.comm) }
    }

    /// Whether the loaded library exports the run-boundary failure check.
    ///
    /// False means only the port's step sync detects a failure; the run record should state that
    /// instead of claiming this check ran.
    #[must_use]
    pub fn async_failure_check_available(&self) -> bool {
        self.api.has_async_failure_check()
    }
}

/// The run's unique id, for rank 0 to broadcast over the caller's own bootstrap channel.
///
/// The port carries the 128 bytes (hex or raw) to every rank and passes the same value to open. This
/// crate does not invent a bootstrap: reusing the run's peer channel is the only way the id and the
/// communicator's peer set stay consistent.
///
/// # Errors
/// Returns NcclError::LibraryAbsent when the library is missing, and NcclError::Call when NCCL
/// refuses the id.
pub fn unique_id() -> Result<NcclUniqueId, NcclError> {
    NcclApi::load()?.unique_id()
}

/// The all-gather sequence the run bootstrap uses on the forward loop's TCP transport.
///
/// It sits in the far-above block the CLI's own exchange sequences use (tp4_forward.rs reserves
/// u64::MAX - 0x1_0000 through 0x9_0000 for the layer/reduce/token exchanges and u64::MAX - 7 for
/// the embedding exchange); this is the next free block below them, so a bootstrap frame can never
/// be mistaken for a walk reduce frame on a reused transport, and a later CLI sequence must not take
/// 0xA_0000 without bumping this one. The setup barrier is 0xB_0000, the next block down.
pub const BOOTSTRAP_GATHER_SEQ: u64 = u64::MAX - 0xA_0000;

/// The setup barrier's sequence: the bounded post-open exchange that turns a rank-local init,
/// warm-up or sync failure into an all-rank abort at the transport deadline instead of a hang to the
/// window's outer timeout.
pub const BOOTSTRAP_BARRIER_SEQ: u64 = u64::MAX - 0xB_0000;

/// Words one NCCL unique id occupies in the bootstrap gather: 128 opaque bytes, two per u16 word.
pub const BOOTSTRAP_ID_WORDS: usize = NCCL_UNIQUE_ID_BYTES / 2;

/// Word index of this rank's flags (library loaded, id supplied).
pub const BOOTSTRAP_FLAGS_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS;

/// Word index of this rank's arm word.
pub const BOOTSTRAP_ARM_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS + 1;

/// Word index of this rank's native-rounding word.
pub const BOOTSTRAP_ROUNDING_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS + 2;

/// Word index of this rank's behaviour-toggle word.
pub const BOOTSTRAP_ENV_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS + 3;

/// Word index of the low half of this rank's plan digest.
pub const BOOTSTRAP_DIGEST_LOW_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS + 4;

/// Word index of the high half of this rank's plan digest.
pub const BOOTSTRAP_DIGEST_HIGH_WORD_INDEX: usize = BOOTSTRAP_ID_WORDS + 5;

/// Words one rank contributes to the bootstrap gather: the id, then the agreement words.
///
/// The agreement words are what turns identical-on-all-ranks from an assumption into a check: a
/// rank that cannot load the library, an arm that disagrees, a different native-rounding selection,
/// a different behaviour toggle or a different plan digest is refused on EVERY rank before any
/// communicator is opened. One gather, no second channel.
pub const BOOTSTRAP_SEGMENT_WORDS: usize = BOOTSTRAP_ID_WORDS + 6;

/// Flag bit: this rank loaded the NCCL library before the gather.
pub const BOOTSTRAP_FLAG_LIBRARY: u16 = 1 << 0;

/// Flag bit: this rank supplies the id. Exactly rank 0 may set it.
pub const BOOTSTRAP_FLAG_ID_SUPPLIED: u16 = 1 << 1;

/// The arm word: the bootstrap runs only on the device arm.
pub const BOOTSTRAP_DEVICE_ARM: u16 = 1;

/// The per-run configuration every rank must agree on before NCCL is entered.
///
/// native_rounding packs the run's native-rounding selection, env_toggles packs the behaviour
/// switches that change arithmetic per process (the seam's host A/B arm and the device trace
/// switches), and plan_digest is the run plan's own hash. Two ranks with different values produce
/// different numbers with no failure, which is why they are checked here rather than trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    /// The run's native-rounding selection as one word.
    pub native_rounding: u16,
    /// The behaviour switches that change arithmetic per process, as one bit set.
    pub env_toggles: u16,
    /// The run plan's digest.
    pub plan_digest: u64,
}

/// This rank's contribution to the bootstrap gather: the id words, then the agreement words.
///
/// Rank 0 passes Some(id) and library_ready; every other rank passes None. The flags word carries
/// whether the library was loaded before the gather (so a rank that lacks it becomes a deterministic
/// all-rank refusal rather than a peer waiting forever inside NCCL init) and whether this rank
/// supplies the id.
#[must_use]
pub fn bootstrap_segment(
    id: Option<&NcclUniqueId>,
    library_ready: bool,
    config: BootstrapConfig,
) -> [u16; BOOTSTRAP_SEGMENT_WORDS] {
    let mut words = [0u16; BOOTSTRAP_SEGMENT_WORDS];
    let mut flags = 0u16;
    if library_ready {
        flags |= BOOTSTRAP_FLAG_LIBRARY;
    }
    if let Some(id) = id {
        flags |= BOOTSTRAP_FLAG_ID_SUPPLIED;
        let bytes = id.as_bytes();
        for (index, word) in words[..BOOTSTRAP_ID_WORDS].iter_mut().enumerate() {
            *word = u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
        }
    }
    words[BOOTSTRAP_FLAGS_WORD_INDEX] = flags;
    words[BOOTSTRAP_ARM_WORD_INDEX] = BOOTSTRAP_DEVICE_ARM;
    words[BOOTSTRAP_ROUNDING_WORD_INDEX] = config.native_rounding;
    words[BOOTSTRAP_ENV_WORD_INDEX] = config.env_toggles;
    words[BOOTSTRAP_DIGEST_LOW_WORD_INDEX] = config.plan_digest as u16;
    words[BOOTSTRAP_DIGEST_HIGH_WORD_INDEX] = (config.plan_digest >> 16) as u16;
    words
}

/// One synchronous all-gather of fixed-width rank segments on the run's own transport.
///
/// Every rank passes the same local segment length and the same sequence; the output is world
/// segments in ascending rank order. The generic form exists so the codec and the agreement checks
/// can be tested on loopback without a CUDA device or an NCCL library.
///
/// # Errors
/// NcclError::Config for a bad world/rank or an empty segment; NcclError::Bootstrap when the
/// transport gather fails.
pub fn bootstrap_exchange(
    transport: &crate::tcp::TcpTransport,
    world: usize,
    rank: usize,
    sequence: u64,
    local: &[u16],
) -> Result<Vec<u16>, NcclError> {
    validate_spec(world, rank)?;
    if local.is_empty() {
        return Err(NcclError::Config(
            "bootstrap: an empty gather segment is not an exchange".to_owned(),
        ));
    }
    let expected = world.checked_mul(local.len()).ok_or_else(|| {
        NcclError::Config(format!(
            "world {world} of {} words overflows the gather length",
            local.len()
        ))
    })?;
    let mut gathered = vec![0u16; expected];
    transport
        .all_gather_bf16(local, &mut gathered, sequence)
        .map_err(|error| {
            NcclError::Bootstrap(format!(
                "the bootstrap all-gather over the run TCP transport failed: {error}; every rank aborts the run here, before any communicator is opened"
            ))
        })?;
    Ok(gathered)
}

/// Reads rank 0's unique id out of the bootstrap gather's output.
///
/// A short output is a protocol error, not a partial id.
///
/// # Errors
/// NcclError::Config for a zero world or an output that is not exactly world segments.
pub fn bootstrap_id_from_gathered(
    gathered: &[u16],
    world: usize,
) -> Result<NcclUniqueId, NcclError> {
    validate_spec(world, 0)?;
    let expected = world
        .checked_mul(BOOTSTRAP_SEGMENT_WORDS)
        .ok_or_else(|| NcclError::Config(format!("world {world} overflows the gather length")))?;
    if gathered.len() != expected {
        return Err(NcclError::Config(format!(
            "the bootstrap gather returned {} words, {world} ranks of {BOOTSTRAP_SEGMENT_WORDS} need {expected}",
            gathered.len()
        )));
    }
    let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
    for (index, word) in gathered[..BOOTSTRAP_ID_WORDS].iter().enumerate() {
        let pair = word.to_le_bytes();
        bytes[2 * index] = pair[0];
        bytes[2 * index + 1] = pair[1];
    }
    Ok(NcclUniqueId::from_bytes(bytes))
}

/// Checks the agreement words of a bootstrap gather, refusing every rank identically on a mismatch.
///
/// The order is the order an operator needs: a missing library first (it is the one failure that is
/// an environment error, not a configuration split), then the id-supply layout, then the arm, the
/// native-rounding selection, the behaviour toggles and the plan digest.
///
/// # Errors
/// NcclError::Bootstrap naming the disagreeing ranks and the word that disagreed. The same gathered
/// frame produces the same message on every rank, so every rank refuses for the same reason.
pub fn bootstrap_agreement(gathered: &[u16], world: usize) -> Result<(), NcclError> {
    validate_spec(world, 0)?;
    let expected = world
        .checked_mul(BOOTSTRAP_SEGMENT_WORDS)
        .ok_or_else(|| NcclError::Config(format!("world {world} overflows the gather length")))?;
    if gathered.len() != expected {
        return Err(NcclError::Config(format!(
            "the bootstrap gather returned {} words, {world} ranks of {BOOTSTRAP_SEGMENT_WORDS} need {expected}",
            gathered.len()
        )));
    }
    let segment =
        |rank: usize| &gathered[rank * BOOTSTRAP_SEGMENT_WORDS..][..BOOTSTRAP_SEGMENT_WORDS];
    let mut no_library = Vec::new();
    let mut wrong_id_layout = Vec::new();
    for rank in 0..world {
        let flags = segment(rank)[BOOTSTRAP_FLAGS_WORD_INDEX];
        if flags & BOOTSTRAP_FLAG_LIBRARY == 0 {
            no_library.push(rank);
        }
        let supplies_id = flags & BOOTSTRAP_FLAG_ID_SUPPLIED != 0;
        if supplies_id != (rank == 0) {
            wrong_id_layout.push(rank);
        }
    }
    if !no_library.is_empty() {
        return Err(NcclError::Bootstrap(format!(
            "rank(s) {no_library:?} could not load the NCCL library before the bootstrap gather; refusing on every rank before any communicator is opened (a peer that entered NCCL init here would wait for a rank that never joins)"
        )));
    }
    if !wrong_id_layout.is_empty() {
        return Err(NcclError::Bootstrap(format!(
            "the bootstrap gather's id flags are not the rank-0 layout: rank(s) {wrong_id_layout:?} supply the id (rank 0 must, and only rank 0 may); refusing on every rank before any communicator is opened"
        )));
    }
    let mismatched = |index: usize, expected: u16| -> Vec<usize> {
        (0..world)
            .filter(|rank| segment(*rank)[index] != expected)
            .collect()
    };
    let arm = segment(0)[BOOTSTRAP_ARM_WORD_INDEX];
    let arm_mismatch = mismatched(BOOTSTRAP_ARM_WORD_INDEX, arm);
    if !arm_mismatch.is_empty() {
        return Err(NcclError::Bootstrap(format!(
            "the bootstrap gather's arm word is not the same on every rank (rank 0 sent {arm}, rank(s) {arm_mismatch:?} sent another value); a mixed-arm run would enqueue collectives from some ranks and not others, so every rank refuses here"
        )));
    }
    for (index, label) in [
        (BOOTSTRAP_ROUNDING_WORD_INDEX, "native-rounding"),
        (BOOTSTRAP_ENV_WORD_INDEX, "behaviour-toggle"),
    ] {
        let value = segment(0)[index];
        let differ = mismatched(index, value);
        if !differ.is_empty() {
            return Err(NcclError::Bootstrap(format!(
                "the bootstrap gather's {label} word differs by rank (rank 0 sent {value}, rank(s) {differ:?} differ); two ranks with different {label} words produce different numbers with no failure, so every rank refuses here"
            )));
        }
    }
    let digest = |rank: usize| -> u64 {
        let words = segment(rank);
        u64::from(words[BOOTSTRAP_DIGEST_LOW_WORD_INDEX])
            | (u64::from(words[BOOTSTRAP_DIGEST_HIGH_WORD_INDEX]) << 16)
    };
    let plan = digest(0);
    let digest_mismatch: Vec<usize> = (0..world).filter(|rank| digest(*rank) != plan).collect();
    if !digest_mismatch.is_empty() {
        return Err(NcclError::Bootstrap(format!(
            "the bootstrap gather's plan digest differs by rank (rank 0 sent {plan:#x}, rank(s) {digest_mismatch:?} differ); the ranks are not running the same plan, so every rank refuses here"
        )));
    }
    Ok(())
}

/// Exchanges the run's NCCL unique id and its per-run configuration over the run's own transport.
///
/// Rank 0 supplies the id (unique_id); every other rank supplies None. library_ready is what THIS
/// rank found when it loaded the NCCL library before calling this - the status crosses in the gather
/// so one rank's missing library refuses deterministically on all four instead of stranding peers in
/// NCCL init. Every rank calls this with the same world and the same sequence, and the call is one
/// synchronous all-gather on the transport the forward loop already holds (the loop's own port
/// range, base + 100 in the CLI wiring), sequence BOOTSTRAP_GATHER_SEQ.
///
/// Rank 0's 128 bytes come back to every rank unchanged, and the agreement words are checked before
/// the id is used, so a per-rank difference in arm, native rounding, behaviour toggles or plan digest
/// is an all-rank refusal BEFORE any communicator is opened.
///
/// This is the only rendezvous: there is no second channel, no shared-FS polling, and no host
/// fallback. A gather error - a timeout, a short frame, an unresponsive peer - is a bootstrap failure
/// and the whole run must abort on every rank.
///
/// # Errors
/// NcclError::Bootstrap when the transport gather fails or the agreement words disagree;
/// NcclError::Config for a bad world/rank or an id supplied by the wrong rank.
pub fn exchange_unique_id(
    transport: &crate::tcp::TcpTransport,
    world: usize,
    rank: usize,
    local: Option<NcclUniqueId>,
    library_ready: bool,
    config: BootstrapConfig,
) -> Result<NcclUniqueId, NcclError> {
    validate_spec(world, rank)?;
    let segment = bootstrap_segment(local.as_ref(), library_ready, config);
    let gathered = bootstrap_exchange(transport, world, rank, BOOTSTRAP_GATHER_SEQ, &segment)?;
    bootstrap_agreement(&gathered, world)?;
    bootstrap_id_from_gathered(&gathered, world)
}

/// The bounded setup barrier: every rank that opened and completed setup meets here, on the run's
/// own transport, before any layer is entered.
///
/// A rank whose open, warm-up or setup sync failed returns Err from open_run_reduce and never
/// reaches this exchange, so its peers wait here for the transport's own deadline instead of
/// proceeding into the timed region and hanging to the window's outer timeout. This is setup, not
/// the in-loop abort handshake rule 2 forbids: it runs once, outside the timed region, and never per
/// stage.
///
/// # Errors
/// NcclError::Bootstrap when the gather fails or a participant reports it did not finish setup.
pub fn setup_barrier(
    transport: &crate::tcp::TcpTransport,
    world: usize,
    rank: usize,
) -> Result<(), NcclError> {
    let gathered = bootstrap_exchange(transport, world, rank, BOOTSTRAP_BARRIER_SEQ, &[1u16])?;
    if gathered.iter().any(|word| *word != 1) {
        return Err(NcclError::Bootstrap(format!(
            "the setup barrier returned words {gathered:?}; every rank must report a completed setup"
        )));
    }
    Ok(())
}

/// Borrows the run's communicator from the opaque handle, refusing a null pointer.
///
/// # Safety
/// handle must come from RunReduce::open and must not have been closed.
unsafe fn run_ref<'a>(handle: *mut c_void) -> Result<&'a RunReduce, NcclError> {
    if handle.is_null() {
        return Err(NcclError::Config(
            "null communicator handle; open it with RunReduce::open".to_owned(),
        ));
    }
    // SAFETY: the caller guarantees the handle is live and came from RunReduce::open.
    Ok(unsafe { &*handle.cast::<RunReduce>() })
}

/// Mutably borrows the run's communicator from the opaque handle, refusing a null pointer.
///
/// # Safety
/// handle must come from RunReduce::open, must not have been closed, and must not alias another
/// borrow.
unsafe fn run_mut<'a>(handle: *mut c_void) -> Result<&'a mut RunReduce, NcclError> {
    if handle.is_null() {
        return Err(NcclError::Config(
            "null communicator handle; open it with RunReduce::open".to_owned(),
        ));
    }
    // SAFETY: the caller guarantees the handle is live, from RunReduce::open, and unaliased.
    Ok(unsafe { &mut *handle.cast::<RunReduce>() })
}

/// Enqueues one reduce on the run's opaque handle: the layer port's per-collective call.
///
/// # Safety
/// Same contract as RunReduce::reduce_bf16, plus handle must be live.
///
/// # Errors
/// Same as RunReduce::reduce_bf16, plus NcclError::Config for a null handle.
pub unsafe fn reduce(
    handle: *mut c_void,
    stream: u64,
    input: u64,
    output: u64,
    count: usize,
) -> Result<(), NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    unsafe { run_mut(handle)?.reduce_bf16(stream, input, output, count) }
}

/// Enqueues the setup warm-up on the run's opaque handle.
///
/// # Safety
/// Same contract as RunReduce::warm_up, plus handle must be live.
///
/// # Errors
/// Same as RunReduce::warm_up, plus NcclError::Config for a null handle.
pub unsafe fn warm_up(
    handle: *mut c_void,
    stream: u64,
    scratch: u64,
    count: usize,
) -> Result<(), NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    unsafe { run_mut(handle)?.warm_up(stream, scratch, count) }
}

/// Registers buffers with the run's opaque handle: the bench's form of RunReduce::register_buffers.
///
/// # Safety
/// Same contract as RunReduce::register_buffers, plus handle must be live.
///
/// # Errors
/// Same as RunReduce::register_buffers, plus NcclError::Config for a null handle.
pub unsafe fn register_buffers(
    handle: *mut c_void,
    spans: &[(u64, usize)],
) -> Result<(), NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    unsafe { run_mut(handle)?.register_buffers(spans) }
}

/// Releases the registrations held by the run's opaque handle.
///
/// # Safety
/// handle must be live, and no collective may be in flight on the registered buffers.
///
/// # Errors
/// Same as RunReduce::deregister_buffers, plus NcclError::Config for a null handle.
pub unsafe fn deregister_buffers(handle: *mut c_void) -> Result<(), NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    unsafe { run_mut(handle)?.deregister_buffers() }
}

/// Closes the run's communicator: the one teardown call, on the same path as the transport close.
///
/// A null handle is a no-op, so a teardown path can call this unconditionally.
///
/// # Safety
/// handle must come from RunReduce::open and must not have been closed already.
pub unsafe fn close(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: the caller guarantees the handle came from RunReduce::open exactly once.
    drop(unsafe { Box::from_raw(handle.cast::<RunReduce>()) });
}

/// The run record's counters for the opaque handle.
///
/// # Safety
/// handle must be live.
///
/// # Errors
/// Returns NcclError::Config for a null handle.
pub unsafe fn stats(handle: *mut c_void) -> Result<DeviceReduceStats, NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    Ok(unsafe { run_ref(handle)?.stats() })
}

/// The arm the opaque handle was opened for.
///
/// # Safety
/// handle must be live.
///
/// # Errors
/// Returns NcclError::Config for a null handle.
pub unsafe fn arm(handle: *mut c_void) -> Result<ReduceArm, NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    Ok(unsafe { run_ref(handle)?.arm() })
}

/// The communicator's init-time device reserve, for the run's arena accounting.
///
/// # Safety
/// handle must be live.
///
/// # Errors
/// Returns NcclError::Config for a null handle.
pub unsafe fn arena_bytes(handle: *mut c_void) -> Result<u64, NcclError> {
    // SAFETY: the caller upholds the handle's contract.
    Ok(unsafe { run_ref(handle)?.stats().arena_bytes() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_bytes_rejects_an_empty_collective() {
        let error = payload_bytes(0).expect_err("an empty collective must be refused");
        assert!(matches!(error, NcclError::Config(_)));
    }

    #[test]
    fn payload_bytes_counts_two_bytes_per_bf16() {
        assert_eq!(payload_bytes(5_120).expect("valid payload"), 10_240);
        assert_eq!(payload_bytes(1).expect("valid payload"), 2);
    }

    #[test]
    fn stats_accumulate_work_that_was_enqueued() {
        let mut stats = DeviceReduceStats::default();
        stats.record(10_240);
        stats.record(20_480);
        assert_eq!(stats.collectives(), 2);
        assert_eq!(stats.elements(), 15_360);
        assert_eq!(stats.bytes(), 30_720);
    }

    #[test]
    fn the_nccl_arm_is_always_built_and_available() {
        assert!(ReduceArm::Nccl.is_built());
        assert!(ReduceArm::Nccl.require_available().is_ok());
    }

    #[test]
    fn the_rdma_arm_refuses_typed_and_never_falls_back() {
        let arm = ReduceArm::GpuRdma;
        assert_eq!(arm.is_built(), cfg!(feature = "gpu-rdma-reduce"));
        match arm.require_available() {
            Err(NcclError::ArmUnavailable {
                arm: reported,
                detail,
            }) => {
                assert_eq!(reported, "gpu_rdma");
                if cfg!(feature = "gpu-rdma-reduce") {
                    assert!(detail.contains("not implemented"), "{detail}");
                    assert!(detail.contains("peermem"), "{detail}");
                } else {
                    assert!(detail.contains("feature off"), "{detail}");
                }
            }
            other => panic!("expected ArmUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn the_host_bridge_arm_is_never_a_device_collector_arm() {
        // The declared S1 exception exists as a label and a counted path, never as something a
        // device handle can open: this is the no-silent-fallback property, on CPU, with no NCCL.
        let arm = ReduceArm::HostBridge;
        assert_eq!(arm.as_str(), "host-bridge");
        assert!(!arm.is_device());
        assert!(ReduceArm::Nccl.is_device());
        assert_eq!(arm.is_built(), cfg!(feature = "s1-host-bridge"));
        match arm.require_available() {
            Err(NcclError::ArmUnavailable {
                arm: reported,
                detail,
            }) => {
                assert_eq!(reported, "host-bridge");
                assert!(detail.contains("HostBridgeReduce"), "{detail}");
                assert!(detail.contains("s1-host-bridge"), "{detail}");
                assert!(detail.contains("no silent fallback"), "{detail}");
            }
            other => panic!("expected ArmUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn opening_the_host_bridge_through_the_device_api_refuses_before_any_library() {
        // Host-bridge selected for a device collector is an arm refusal, not a run: no NCCL load,
        // no communicator, no fallback to nccl underneath the host-bridge label.
        let id = NcclUniqueId::from_bytes([0u8; nccl::NCCL_UNIQUE_ID_BYTES]);
        let error = RunReduce::open_with_arm(4, 0, id, ReduceArm::HostBridge, 0)
            .expect_err("the host-bridge arm must refuse the device collector");
        assert!(matches!(error, NcclError::ArmUnavailable { .. }));
        // And the label is not silently rewritten to nccl in the refusal.
        let text = error.to_string();
        assert!(text.contains("host-bridge"), "{text}");
        assert!(!text.contains("NCCL library"), "{text}");
    }

    #[test]
    fn a_zero_rank_world_is_refused() {
        assert!(matches!(validate_spec(0, 0), Err(NcclError::Config(_))));
    }

    #[test]
    fn a_rank_outside_the_world_is_refused() {
        assert!(matches!(validate_spec(4, 4), Err(NcclError::Config(_))));
        assert!(matches!(validate_spec(1, 7), Err(NcclError::Config(_))));
    }

    #[test]
    fn a_one_rank_world_is_a_legal_degenerate_shape() {
        assert!(validate_spec(1, 0).is_ok());
        assert!(validate_spec(4, 0).is_ok());
        assert!(validate_spec(4, 3).is_ok());
    }

    #[test]
    fn an_oversized_payload_is_refused_without_touching_a_device() {
        assert!(matches!(
            payload_bytes(usize::MAX),
            Err(NcclError::Config(_))
        ));
    }

    #[test]
    fn a_rejected_payload_counts_nothing() {
        // The enqueue path only records after an enqueue succeeds; a refused payload must leave the
        // counters untouched so the run record cannot claim work that never ran.
        let mut stats = DeviceReduceStats::default();
        let outcome = payload_bytes(0).map(|bytes| stats.record(bytes));
        assert!(outcome.is_err());
        assert_eq!(stats.collectives(), 0);
        assert_eq!(stats.bytes(), 0);
    }

    #[test]
    fn stats_default_to_zero() {
        assert_eq!(
            DeviceReduceStats::default(),
            DeviceReduceStats {
                collectives: 0,
                elements: 0,
                bytes: 0,
                arena_bytes: 0,
                bridge_collectives: 0,
                bridge_d2h_bytes: 0,
                bridge_h2d_bytes: 0,
                registered_buffers: 0,
                registered_bytes: 0,
            }
        );
        assert_eq!(DeviceReduceStats::default().bridge_copies(), 0);
    }

    #[test]
    fn the_registration_account_is_setup_and_never_moves_the_collective_counters() {
        // The four collective spans of the layer (two inputs, two outputs, 10,240 B each).
        let mut stats = DeviceReduceStats::default();
        stats.set_registered(4, 4 * 10_240);
        assert_eq!(stats.registered_buffers(), 4);
        assert_eq!(stats.registered_bytes(), 40_960);
        assert_eq!(stats.collectives(), 0);
        assert_eq!(stats.bytes(), 0);
        stats.record(10_240);
        assert_eq!(stats.registered_buffers(), 4);
        stats.set_registered(0, 0);
        assert_eq!(stats.registered_buffers(), 0);
        assert_eq!(stats.collectives(), 1);
    }

    #[test]
    fn a_bridged_collective_moves_the_shared_counters_and_the_copy_account() {
        // One completed host-bridge collective of the layer payload: same collective/element/byte
        // accounting as the device arm, plus exactly one D2H half and one H2D half.
        let mut stats = DeviceReduceStats::default();
        stats.record_bridged(10_240);
        assert_eq!(stats.collectives(), 1);
        assert_eq!(stats.elements(), 5_120);
        assert_eq!(stats.bytes(), 10_240);
        assert_eq!(stats.bridge_collectives(), 1);
        assert_eq!(stats.bridge_d2h_bytes(), 10_240);
        assert_eq!(stats.bridge_h2d_bytes(), 10_240);
        assert_eq!(stats.bridge_copies(), 2);
    }

    #[test]
    fn a_device_only_run_reports_no_bridge_copies() {
        // The steady-state assertion the record relies on: device collectives never move the bridge
        // counters, so `bridge_collectives() == 0` is a checkable "no PCIe copy" statement.
        let mut stats = DeviceReduceStats::default();
        stats.record(10_240);
        assert_eq!(stats.collectives(), 1);
        assert_eq!(stats.bridge_collectives(), 0);
        assert_eq!(stats.bridge_copies(), 0);
        assert_eq!(stats.bridge_d2h_bytes(), 0);
        assert_eq!(stats.bridge_h2d_bytes(), 0);
    }

    #[test]
    fn a_mixed_run_keeps_the_device_and_bridge_accounts_separate() {
        let mut stats = DeviceReduceStats::default();
        stats.record(10_240);
        stats.record_bridged(10_240);
        assert_eq!(stats.collectives(), 2);
        assert_eq!(stats.elements(), 10_240);
        assert_eq!(stats.bytes(), 20_480);
        assert_eq!(stats.bridge_collectives(), 1);
        assert_eq!(stats.bridge_copies(), 2);
    }

    #[test]
    fn the_arena_reserve_is_part_of_the_stats() {
        // The port adds arena_bytes to its memory model when the arm is armed; keeping it in the
        // same struct as the payload counters is what stops the two accounts from drifting apart.
        let mut stats = DeviceReduceStats::default();
        stats.set_arena_bytes(NCCL_ARENA_RESERVE_BYTES);
        stats.record(10_240);
        assert_eq!(stats.arena_bytes(), NCCL_ARENA_RESERVE_BYTES);
        assert_eq!(stats.collectives(), 1);
        assert_eq!(stats.bytes(), 10_240);
    }

    #[test]
    fn a_null_handle_refuses_instead_of_dereferencing() {
        let null = std::ptr::null_mut::<c_void>();
        // SAFETY: null is the degenerate handle every accessor must refuse.
        unsafe {
            assert!(matches!(stats(null), Err(NcclError::Config(_))));
            assert!(matches!(arm(null), Err(NcclError::Config(_))));
            assert!(matches!(arena_bytes(null), Err(NcclError::Config(_))));
            assert!(matches!(
                reduce(null, 0, 0, 0, 5_120),
                Err(NcclError::Config(_))
            ));
            assert!(matches!(
                warm_up(null, 0, 0, 5_120),
                Err(NcclError::Config(_))
            ));
            assert!(matches!(
                register_buffers(null, &[(0x1000, 10_240)]),
                Err(NcclError::Config(_))
            ));
            assert!(matches!(
                deregister_buffers(null),
                Err(NcclError::Config(_))
            ));
            // close is the teardown path, so a null handle must be a no-op rather than a panic.
            close(null);
        }
    }

    #[test]
    fn the_setup_warm_up_and_the_arena_reserve_are_the_pinned_values() {
        assert_eq!(WARM_UP_COLLECTIVES, 3);
        assert_eq!(NCCL_ARENA_RESERVE_BYTES, 16 * 1024 * 1024);
    }

    #[test]
    fn opening_the_rdma_arm_refuses_before_any_library_or_device() {
        let id = NcclUniqueId::from_bytes([0u8; nccl::NCCL_UNIQUE_ID_BYTES]);
        let error = RunReduce::open_with_arm(4, 0, id, ReduceArm::GpuRdma, 0)
            .expect_err("the rdma arm must refuse");
        assert!(matches!(error, NcclError::ArmUnavailable { .. }));
    }

    #[test]
    fn opening_a_bad_spec_refuses_before_any_library() {
        let id = NcclUniqueId::from_bytes([0u8; nccl::NCCL_UNIQUE_ID_BYTES]);
        assert!(matches!(
            RunReduce::open_with_arm(0, 0, id, ReduceArm::Nccl, 0),
            Err(NcclError::Config(_))
        ));
        assert!(matches!(
            RunReduce::open_with_arm(4, 4, id, ReduceArm::Nccl, 0),
            Err(NcclError::Config(_))
        ));
    }

    fn sample_bootstrap_config() -> BootstrapConfig {
        BootstrapConfig {
            native_rounding: 0b011,
            env_toggles: 0b001,
            plan_digest: 0xfeed_face_dead_beef,
        }
    }

    #[test]
    fn the_bootstrap_words_round_trip_rank_zero_s_id() {
        let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(7).wrapping_add(3);
        }
        let id = NcclUniqueId::from_bytes(bytes);
        let config = sample_bootstrap_config();
        let rank_zero = bootstrap_segment(Some(&id), true, config);
        let others = bootstrap_segment(None, true, config);
        assert_eq!(
            rank_zero[BOOTSTRAP_FLAGS_WORD_INDEX],
            BOOTSTRAP_FLAG_LIBRARY | BOOTSTRAP_FLAG_ID_SUPPLIED
        );
        assert_eq!(others[BOOTSTRAP_FLAGS_WORD_INDEX], BOOTSTRAP_FLAG_LIBRARY);
        assert!(others[..BOOTSTRAP_ID_WORDS].iter().all(|word| *word == 0));
        assert_eq!(rank_zero[BOOTSTRAP_ARM_WORD_INDEX], BOOTSTRAP_DEVICE_ARM);
        let mut gathered = Vec::with_capacity(2 * BOOTSTRAP_SEGMENT_WORDS);
        gathered.extend_from_slice(&rank_zero);
        gathered.extend_from_slice(&others);
        bootstrap_agreement(&gathered, 2).expect("an agreed frame must pass");
        let seen = bootstrap_id_from_gathered(&gathered, 2).expect("the id must round-trip");
        assert_eq!(seen, id);
        assert_eq!(seen.as_bytes(), &bytes);
    }

    #[test]
    fn the_bootstrap_gather_refuses_a_short_segment_run() {
        let error = bootstrap_id_from_gathered(&[0u16; BOOTSTRAP_SEGMENT_WORDS - 1], 1)
            .expect_err("a short gather is a protocol error");
        assert!(matches!(error, NcclError::Config(_)));
        let error = bootstrap_agreement(&[0u16; BOOTSTRAP_SEGMENT_WORDS - 1], 1)
            .expect_err("the agreement check must not read a short frame");
        assert!(matches!(error, NcclError::Config(_)));
        let error = bootstrap_id_from_gathered(&[], 0).expect_err("a zero world is refused");
        assert!(matches!(error, NcclError::Config(_)));
    }

    #[test]
    fn the_bootstrap_refuses_a_world_or_rank_that_cannot_gather() {
        assert!(validate_spec(4, 5).is_err());
        assert!(validate_spec(0, 0).is_err());
        assert!(validate_spec(4, 0).is_ok());
    }
}
