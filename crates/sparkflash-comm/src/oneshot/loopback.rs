//! The loopback host twin of the device-buffer one-shot path.
//!
//! docs/build/rdma-backend.md section 15 leaves the device path unimplemented because every piece
//! of it needs a device: a pinned-host allocation with a device alias, an ibverbs memory region,
//! the proxy thread that posts work requests, and the CUDA kernel that reduces the slot rows.
//! This module implements the same protocol in process memory, behind the existing loopback
//! transport seam, so the path can be built and proven on a host with no GPU and no NIC:
//!
//! * a [LoopbackDeviceCluster] owns one pinned slot ring per rank. The slots are ordinary
//!   allocations whose addresses stand in for the device addresses the seam would register; the
//!   platform aliasing is simulated by the cluster, not claimed.
//! * a [LoopbackDeviceTransport] implements [crate::Transport::all_reduce_bf16_device] for real:
//!   it stages the rank's row, rings the doorbell, lets the receive-side proxy publish an arrival
//!   word only after every rail of every peer has landed, waits for the peers' arrival words, and
//!   reduces the slot rows into the output buffer. The host-slice forms are refused, exactly as a
//!   device transport should refuse them: this backend has one path and it is the device one.
//! * [kernel_reduce_into] spells out the CUDA kernel's arithmetic as a CPU twin: the accumulator
//!   starts at 0.0f and adds ranks 0, 1, 2, 3 in ascending order in f32, rounded once to bf16 with
//!   round-to-nearest-even. A test compares it bit for bit with
//!   [crate::reduce::canonical_all_reduce_into] on vectors chosen to separate one rounding from
//!   four, so a drift in either direction fails.
//! * the slot ring and the chunked pipeline are pinned: single calls cycle a two-deep ring, and a
//!   chunked plan reuses a ring slot only after every rank has finished reading its previous use
//!   (a per-slot done epoch). The reported [PipelineReport] carries the ring indices and the
//!   dispatch records, so the shape is a value a test can assert rather than a comment.
//!
//! What this does not do, and what no host run may claim: allocate device memory, register an MR,
//! post a work request or launch the kernel (docs/build/rdma-backend.md section 15.3 items 1, 3,
//! 4 and 5). It also produces no node number. [HostTwinReport] names latency, bandwidth and
//! four-node mesh correctness as unclaimed; the window is the only place they can be measured.
//!
//! The bytes a call moves are the bytes the NIC would move (each peer's whole rail share of the
//! row), and the dispatch record reports zero host-staging bytes because the twin reads and
//! writes the simulated device buffers directly - the same material the kernel would touch - not
//! a staging copy. Feeding a buffer between dispatches is the harness's job and is outside the
//! collective: it stands for the upstream kernel writing the next contribution, not for a
//! DtoH/HtoD copy.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::bf16::{bf16_to_f32, f32_to_bf16_rn};
use crate::error::CommError;
use crate::layout::CollectiveLayout;
use crate::rail::RailLedger;
use crate::reduce::canonical_all_reduce_into;
use crate::transport::Transport;

use super::{
    ArrivalProxy, ChunkPlan, ChunkedProxy, DeviceBuffer, DispatchAddresses, DispatchLog,
    DispatchRecord, Exchange, LaunchPath, ONESHOT_KIND, ONESHOT_PHASE, ONESHOT_RAILS,
    ONESHOT_RANKS, ONESHOT_SLOT_BYTES, RailFrame, SeamError, SequenceCounter, row_offset,
};

mod ring;

pub use ring::{
    RingPlan, RingTwinConfig, RingTwinReport, run_ring_twin, run_ring_twin_with_payloads,
};

mod orchestrate;

pub use orchestrate::{
    FailureKind, FailurePolicy, GuardPolicy, OrchestrationConfig, OrchestrationReport, RankFate,
    RequestOutcome, ServingRequest, StartupStatus, canonical_first_token, run_orchestration,
    shard_first_token,
};

/// Ring slots the twin allocates per rank by default: the double buffer the seam design names.
pub const HOST_TWIN_RING: usize = 2;

/// How the twin treats a ring slot's read-done epoch before it stages a new chunk into that slot.
///
/// [ReadEpochPolicy::Honor] is the only behaviour a device transport may have: the staging write
/// waits until every rank's kernel has finished reading the slot's previous use. [Ignore] exists
/// solely so a test can demonstrate the hazard the wait prevents - it is a deliberate contract
/// violation, never a mode a node run or the evidence path uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadEpochPolicy {
    /// Wait for the read-done epoch before reusing a slot (the real rule).
    #[default]
    Honor,
    /// Skip the wait and overwrite the slot before every reader is done (hazard demo only).
    Ignore,
}

/// A deliberately slow kernel read on one rank and chunk, used to widen the ring-reuse hazard
/// window so a test can prove the read-done epoch is load-bearing instead of describing it.
///
/// The host twin has no real kernel; this knob stands in for a kernel that is slower than the
/// proxy, which is exactly the scheduling the two-deep ring has to tolerate. It changes no
/// arithmetic and is not a performance model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelDelay {
    /// Rank whose kernel stalls.
    pub rank: usize,
    /// Chunk index the stall applies to.
    pub chunk: usize,
    /// How long the kernel stalls before reading the slot rows.
    pub duration: Duration,
}

/// Bytes a slot or output allocation adds so its first data byte is 8-byte aligned.
const ALLOC_PAD: usize = 8;

/// Configuration of one loopback device cluster.
#[derive(Debug, Clone)]
pub struct LoopbackDeviceConfig {
    /// Ranks in the cluster; the one-shot seam is always four.
    pub world_size: usize,
    /// Slots per rank: one or two (the pinned ring shape).
    pub slot_ring: usize,
    /// Rails every exchange posts: one or two.
    pub rails: usize,
    /// Deadline for doorbell, arrival and ring-reuse waits.
    pub timeout: Duration,
    /// Whether peers send their frames in reverse rank and rail order.
    pub reverse_arrivals: bool,
    /// How a ring slot's read-done epoch is treated before reuse.
    pub read_epoch_policy: ReadEpochPolicy,
    /// An optional, deliberate kernel-side stall (test knob; see [KernelDelay]).
    pub kernel_delay: Option<KernelDelay>,
}

impl Default for LoopbackDeviceConfig {
    fn default() -> Self {
        Self {
            world_size: ONESHOT_RANKS,
            slot_ring: HOST_TWIN_RING,
            rails: ONESHOT_RAILS,
            timeout: Duration::from_secs(5),
            reverse_arrivals: false,
            read_epoch_policy: ReadEpochPolicy::Honor,
            kernel_delay: None,
        }
    }
}

fn seam_error(error: SeamError) -> CommError {
    CommError::Protocol(format!("one-shot seam: {error}"))
}

fn invalid(reason: impl Into<String>) -> CommError {
    CommError::InvalidLayout(reason.into())
}

/// A byte allocation whose first data byte is 8-byte aligned, the seam's address requirement.
struct AlignedBytes {
    raw: Vec<u8>,
    pad: usize,
    bytes: usize,
}

impl AlignedBytes {
    fn new(bytes: usize) -> Self {
        let raw = vec![0u8; bytes + ALLOC_PAD];
        let pad = (ALLOC_PAD - raw.as_ptr() as usize % ALLOC_PAD) % ALLOC_PAD;
        Self { raw, pad, bytes }
    }

    fn address(&self) -> u64 {
        (self.raw.as_ptr() as usize + self.pad) as u64
    }

    fn as_slice(&self) -> &[u8] {
        &self.raw[self.pad..self.pad + self.bytes]
    }

    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.pad..self.pad + self.bytes]
    }
}

/// One rank's pinned slot in the cluster: arrival words then four payload rows.
struct Slot {
    bytes: AlignedBytes,
}

impl Slot {
    fn new() -> Self {
        Self {
            bytes: AlignedBytes::new(ONESHOT_SLOT_BYTES),
        }
    }

    fn address(&self) -> u64 {
        self.bytes.address()
    }
}

/// One rank's output allocation: the buffer a kernel launch would write the result into.
///
/// On the node this is device memory. Here it is a host allocation registered with the cluster by
/// address, and the twin writes it through the same owner it hands the dispatch record, so the
/// address the record carries is the allocation the result lands in.
struct OutputAllocation {
    data: Mutex<AlignedBytes>,
    address: u64,
    owner_rank: usize,
    elements: usize,
}

impl OutputAllocation {
    fn new(owner_rank: usize, elements: usize) -> Self {
        let bytes = AlignedBytes::new(elements * 2);
        let address = bytes.address();
        Self {
            data: Mutex::new(bytes),
            address,
            owner_rank,
            elements,
        }
    }

    fn bytes(&self, offset_bytes: usize, len: usize) -> Vec<u8> {
        let data = self.data.lock();
        data.as_slice()[offset_bytes..offset_bytes + len].to_vec()
    }

    fn store_bf16(&self, offset_bytes: usize, values: &[u16]) {
        let mut data = self.data.lock();
        let bytes = data.as_mut();
        for (index, value) in values.iter().enumerate() {
            let start = offset_bytes + index * 2;
            bytes[start..start + 2].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn snapshot_bf16(&self) -> Vec<u16> {
        let data = self.data.lock();
        data.as_slice()
            .chunks(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }
}

/// A host allocation standing in for the device output buffer of one rank.
///
/// The address the collective uses is [DeviceBuffer::address], the allocation's first data byte.
/// The allocation must belong to the cluster the transport comes from and to the rank that calls;
/// the transport refuses anything else instead of writing an unrelated address.
pub struct HostDeviceBuffer {
    cluster: LoopbackDeviceCluster,
    allocation: Arc<OutputAllocation>,
    buffer: DeviceBuffer,
}

impl HostDeviceBuffer {
    /// Allocates and registers an output buffer for one rank.
    ///
    /// # Errors
    /// Refuses a rank outside the cluster, zero elements, and a registration failure (the device
    /// buffer's alignment check is the seam's own). A chunked output holds the whole payload; the
    /// per-exchange row limit is enforced on each one-shot exchange, not on the buffer.
    pub fn allocate(
        cluster: &LoopbackDeviceCluster,
        rank: usize,
        elements: usize,
    ) -> Result<Self, CommError> {
        if rank >= cluster.world_size() {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: cluster.world_size(),
            });
        }
        if elements == 0 {
            return Err(invalid("a host device buffer needs at least one element"));
        }
        let allocation = Arc::new(OutputAllocation::new(rank, elements));
        let buffer = DeviceBuffer::new(allocation.address, elements * 2).map_err(seam_error)?;
        cluster
            .inner
            .outputs
            .lock()
            .insert(allocation.address, Arc::clone(&allocation));
        Ok(Self {
            cluster: cluster.clone(),
            allocation,
            buffer,
        })
    }

    /// The device buffer descriptor the collective takes.
    #[must_use]
    pub fn device_buffer(&self) -> DeviceBuffer {
        self.buffer
    }

    /// The rank that owns this buffer.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.allocation.owner_rank
    }

    /// The element count.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.allocation.elements
    }

    /// Writes one rank contribution, standing in for the upstream kernel filling the buffer.
    ///
    /// # Errors
    /// Refuses a slice whose length is not the buffer's element count.
    pub fn write(&self, values: &[u16]) -> Result<(), CommError> {
        if values.len() != self.elements() {
            return Err(invalid(format!(
                "the contribution has {} elements but the buffer holds {}",
                values.len(),
                self.elements()
            )));
        }
        self.allocation.store_bf16(0, values);
        Ok(())
    }

    /// Reads the buffer back for the verifier side of a test.
    ///
    /// This is the test's device-to-host read, outside the collective; the dispatch record's
    /// host-copy accounting covers the collective path only.
    #[must_use]
    pub fn read(&self) -> Vec<u16> {
        self.allocation.snapshot_bf16()
    }

    fn allocation(&self) -> Arc<OutputAllocation> {
        Arc::clone(&self.allocation)
    }

    fn cluster(&self) -> &LoopbackDeviceCluster {
        &self.cluster
    }
}

/// Shared state of one loopback device cluster.
struct Inner {
    config: LoopbackDeviceConfig,
    state: Mutex<State>,
    changed: Condvar,
    rail_ledger: Mutex<RailLedger>,
    doorbells: AtomicU64,
    collectives: AtomicU64,
    outputs: Mutex<HashMap<u64, Arc<OutputAllocation>>>,
}

struct State {
    alive: Vec<bool>,
    slots: Vec<Vec<Slot>>,
    doorbell: Vec<u64>,
    slot_seq: Vec<Vec<u64>>,
    arrival: Vec<Vec<u64>>,
    read_epoch: Vec<Vec<u64>>,
    inbox: Vec<Vec<RailFrame>>,
    slot_last_seq: Vec<Option<u64>>,
    slot_busy: Vec<Vec<bool>>,
    rank_in_flight: Vec<usize>,
    max_rank_in_flight: usize,
}

/// The loopback device cluster: the simulated slots, doorbells and NIC of one four-rank mesh.
pub struct LoopbackDeviceCluster {
    inner: Arc<Inner>,
}

impl Clone for LoopbackDeviceCluster {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for LoopbackDeviceCluster {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoopbackDeviceCluster")
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl LoopbackDeviceCluster {
    /// Creates a cluster with the pinned one-shot slot geometry.
    ///
    /// # Errors
    /// Refuses a world size other than four, a ring outside one or two, a rail count other than
    /// one or two, and a timeout of zero.
    pub fn new(config: LoopbackDeviceConfig) -> Result<Self, CommError> {
        if config.world_size != ONESHOT_RANKS {
            return Err(invalid(format!(
                "the one-shot seam is a {ONESHOT_RANKS}-rank collective, world_size {} was requested",
                config.world_size
            )));
        }
        if config.slot_ring == 0 || config.slot_ring > HOST_TWIN_RING {
            return Err(invalid(format!(
                "slot_ring is {} but the pinned ring shape is 1 or {HOST_TWIN_RING}",
                config.slot_ring
            )));
        }
        if config.rails != 1 && config.rails != ONESHOT_RAILS {
            return Err(invalid(format!(
                "rails is {} but the one-shot modes are 1 or {ONESHOT_RAILS}",
                config.rails
            )));
        }
        if config.timeout.is_zero() {
            return Err(invalid("the loopback device timeout must be non-zero"));
        }
        let world = config.world_size;
        let ring = config.slot_ring;
        let state = State {
            alive: vec![true; world],
            slots: (0..world)
                .map(|_| (0..ring).map(|_| Slot::new()).collect())
                .collect(),
            doorbell: vec![0; world],
            slot_seq: (0..world).map(|_| vec![0; ring]).collect(),
            arrival: vec![vec![0; ONESHOT_RANKS]; world],
            read_epoch: (0..world).map(|_| vec![0; ring]).collect(),
            inbox: (0..world).map(|_| Vec::new()).collect(),
            slot_last_seq: vec![None; ring],
            slot_busy: (0..world).map(|_| vec![false; ring]).collect(),
            rank_in_flight: vec![0; world],
            max_rank_in_flight: 0,
        };
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                state: Mutex::new(state),
                changed: Condvar::new(),
                rail_ledger: Mutex::new(RailLedger::default()),
                doorbells: AtomicU64::new(0),
                collectives: AtomicU64::new(0),
                outputs: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Ranks in the cluster.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.inner.config.world_size
    }

    /// Slots in each rank's ring.
    #[must_use]
    pub fn slot_ring(&self) -> usize {
        self.inner.config.slot_ring
    }

    /// The configured rail count.
    #[must_use]
    pub fn rails(&self) -> usize {
        self.inner.config.rails
    }

    /// Creates the transport handle for one rank.
    ///
    /// # Errors
    /// Returns [CommError::RankOutOfRange] for a rank outside the cluster.
    pub fn handle(&self, rank: usize) -> Result<LoopbackDeviceTransport, CommError> {
        if rank >= self.world_size() {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: self.world_size(),
            });
        }
        Ok(LoopbackDeviceTransport {
            cluster: self.clone(),
            rank,
            state: Mutex::new(RankState {
                next_slot: 0,
                sequence: SequenceCounter::new(),
                log: DispatchLog::new(),
            }),
        })
    }

    /// Bytes each rail has moved through the simulated NIC so far.
    #[must_use]
    pub fn rail_ledger(&self) -> RailLedger {
        *self.inner.rail_ledger.lock()
    }

    /// Doorbells rung so far, one per exchange.
    #[must_use]
    pub fn doorbells(&self) -> u64 {
        self.inner.doorbells.load(Ordering::Relaxed)
    }

    /// Exchanges dispatched so far.
    #[must_use]
    pub fn collectives(&self) -> u64 {
        self.inner.collectives.load(Ordering::Relaxed)
    }

    /// The most ring slots one rank held in flight at once during this cluster's life.
    ///
    /// One means the rank serialised its ring (a one-deep ring or a proxy that never ran ahead of
    /// its kernel); two means the two-deep pipeline actually overlapped two chunks - a staged chunk
    /// still awaiting its read-done epoch while the next chunk was being staged. It is a shape
    /// observation, not a latency measurement.
    #[must_use]
    pub fn max_rank_in_flight(&self) -> usize {
        self.inner.state.lock().max_rank_in_flight
    }

    fn allocation(&self, address: u64) -> Result<Arc<OutputAllocation>, CommError> {
        self.inner
            .outputs
            .lock()
            .get(&address)
            .cloned()
            .ok_or_else(|| {
                invalid(format!(
                    "the device buffer at {address:#x} is not a host allocation of this cluster; the loopback transport writes only registered allocations"
                ))
            })
    }

    fn deadline(&self) -> Instant {
        Instant::now() + self.inner.config.timeout
    }

    fn wait(
        &self,
        state: &mut parking_lot::MutexGuard<'_, State>,
        what: &str,
        deadline: Instant,
    ) -> Result<(), CommError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(CommError::timeout(
                self.inner.config.timeout,
                what.to_owned(),
            ));
        }
        self.inner.changed.wait_for(state, remaining);
        Ok(())
    }

    fn notify(&self) {
        self.inner.changed.notify_all();
    }
}

/// Per-rank mutable state behind the transport's shared reference.
struct RankState {
    next_slot: usize,
    sequence: SequenceCounter,
    log: DispatchLog,
}

/// The receive-side proxy contract, so single and chunked exchanges share one body.
trait ProxySink {
    fn begin(&mut self) -> Result<(), SeamError>;
    fn frame(&mut self, frame: &RailFrame) -> Result<Option<usize>, SeamError>;
    fn all_published(&self) -> bool;
}

impl ProxySink for ArrivalProxy {
    fn begin(&mut self) -> Result<(), SeamError> {
        self.on_doorbell(self.exchange().seq())
    }

    fn frame(&mut self, frame: &RailFrame) -> Result<Option<usize>, SeamError> {
        self.on_frame(frame)
    }

    fn all_published(&self) -> bool {
        ArrivalProxy::all_published(self)
    }
}

impl ProxySink for ChunkedProxy {
    fn begin(&mut self) -> Result<(), SeamError> {
        self.on_doorbell()
    }

    fn frame(&mut self, frame: &RailFrame) -> Result<Option<usize>, SeamError> {
        self.on_frame(frame)
    }

    fn all_published(&self) -> bool {
        self.current_published()
    }
}

/// What one chunked pipeline run did, as values a test or an evidence writer can assert.
#[derive(Debug, Clone, Default)]
pub struct PipelineReport {
    /// The dispatch record of every chunk, in chunk order.
    pub records: Vec<DispatchRecord>,
    /// The ring slot each chunk used, in chunk order.
    pub slot_indices: Vec<usize>,
    /// Bytes the simulated NIC moved during this run, per rail.
    pub posted: RailLedger,
    /// Host-staging bytes over every record: zero for the device path.
    pub host_copy_bytes: usize,
}

/// One rank's handle into a loopback device cluster.
///
/// It implements the device-buffer transport form and refuses the host-slice forms. The refusal is
/// the seam's no-silent-fallback rule: a caller holding a host slice must use a host transport,
/// not believe it ran a copy-elided collective.
pub struct LoopbackDeviceTransport {
    cluster: LoopbackDeviceCluster,
    rank: usize,
    state: Mutex<RankState>,
}

impl Transport for LoopbackDeviceTransport {
    fn world_size(&self) -> usize {
        self.cluster.world_size()
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
            "the loopback device transport runs only the device-buffer form; a host slice belongs on the host-staging transport, and the seam refuses to substitute it silently"
                .to_owned(),
        ))
    }

    fn all_gather_bf16(
        &self,
        _local: &[u16],
        _out: &mut [u16],
        _seq: u64,
    ) -> Result<(), CommError> {
        Err(CommError::Unsupported(
            "the loopback device transport has no all-gather form; the one-shot seam is an all-reduce seat"
                .to_owned(),
        ))
    }

    fn all_reduce_bf16_device(
        &self,
        buffer: &DeviceBuffer,
        elements: usize,
        seq: u64,
    ) -> Result<(), CommError> {
        let exchange = self.exchange(elements, seq)?;
        let slot_index = {
            let state = self.state.lock();
            state.next_slot % self.cluster.slot_ring()
        };
        let output = self.output(buffer, elements)?;
        let mut proxy = ArrivalProxy::new(self.rank, exchange).map_err(seam_error)?;
        self.run_exchange(&output, exchange, 0, slot_index, &mut proxy)?;
        let mut state = self.state.lock();
        state.next_slot = slot_index + 1;
        Ok(())
    }
}

impl LoopbackDeviceTransport {
    /// The rank this handle belongs to.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Every dispatch this handle has recorded, in sequence order.
    #[must_use]
    pub fn dispatch_log(&self) -> DispatchLog {
        self.state.lock().log.clone()
    }

    /// The number of dispatches this handle has recorded.
    #[must_use]
    pub fn dispatches(&self) -> usize {
        self.state.lock().log.len()
    }

    /// The most recent dispatch record, if any.
    #[must_use]
    pub fn last_dispatch(&self) -> Option<DispatchRecord> {
        self.state.lock().log.records().last().copied()
    }

    /// Runs a chunked payload through the same slot ring, one row per chunk, in strict order.
    ///
    /// Each chunk is a normal one-shot exchange whose output offset is the chunk's offset in the
    /// output buffer. The ring index is the chunk index modulo the ring length; a slot is reused
    /// only after every rank has finished reading its previous use, which the per-slot done epoch
    /// enforces before the next staging write. The returned [PipelineReport] records the slot
    /// indices and every dispatch, so the ring/pipeline shape is asserted, not assumed.
    ///
    /// # Errors
    /// Refuses an output buffer whose element count is not the plan's total, a plan on a different
    /// rail count, and any exchange, sequence or ring violation underneath.
    pub fn run_pipeline(
        &self,
        output: &HostDeviceBuffer,
        plan: &ChunkPlan,
        base_seq: u64,
    ) -> Result<PipelineReport, CommError> {
        self.check_output(output)?;
        if plan.total_elements() != output.elements() {
            return Err(invalid(format!(
                "the plan carries {} elements but the output buffer holds {}",
                plan.total_elements(),
                output.elements()
            )));
        }
        let before = self.cluster.rail_ledger();
        let mut proxy = ChunkedProxy::new(self.rank, plan.clone(), base_seq).map_err(seam_error)?;
        let mut report = PipelineReport::default();
        for index in 0..plan.chunk_count() {
            let seq = plan.chunk_seq(base_seq, index).map_err(seam_error)?;
            let exchange = plan.exchange(index, seq).map_err(seam_error)?;
            let chunk = plan.chunks()[index];
            let slot_index = index % self.cluster.slot_ring();
            let record = self.run_exchange(
                output,
                exchange,
                chunk.offset_bytes(),
                slot_index,
                &mut proxy,
            )?;
            report.slot_indices.push(slot_index);
            report.host_copy_bytes += record.host_copy_bytes();
            report.records.push(record);
            proxy.advance().map_err(seam_error)?;
        }
        let after = self.cluster.rail_ledger();
        report.posted = RailLedger {
            rail0_bytes: after.rail0_bytes - before.rail0_bytes,
            rail1_bytes: after.rail1_bytes - before.rail1_bytes,
            transfers: after.transfers - before.transfers,
            element_writes: after.element_writes - before.element_writes,
        };
        Ok(report)
    }

    fn exchange(&self, elements: usize, seq: u64) -> Result<Exchange, CommError> {
        Exchange::new(seq, elements, self.cluster.rails()).map_err(seam_error)
    }

    fn check_output(&self, output: &HostDeviceBuffer) -> Result<(), CommError> {
        if output.rank() != self.rank {
            return Err(invalid(format!(
                "rank {} was handed rank {}'s output buffer",
                self.rank,
                output.rank()
            )));
        }
        if output.cluster().inner.config.world_size != self.cluster.world_size() {
            return Err(invalid(
                "the output buffer belongs to a different loopback cluster",
            ));
        }
        let registered = self.cluster.allocation(output.device_buffer().address())?;
        if registered.owner_rank != self.rank {
            return Err(invalid(
                "the output allocation is registered to another rank",
            ));
        }
        Ok(())
    }

    fn output(
        &self,
        buffer: &DeviceBuffer,
        elements: usize,
    ) -> Result<HostDeviceBuffer, CommError> {
        if buffer.bytes() != elements * 2 {
            return Err(invalid(format!(
                "the device buffer is {} bytes but the exchange carries {elements} elements",
                buffer.bytes()
            )));
        }
        let allocation = self.cluster.allocation(buffer.address())?;
        if allocation.owner_rank != self.rank {
            return Err(invalid(format!(
                "rank {} was handed rank {}'s device buffer",
                self.rank, allocation.owner_rank
            )));
        }
        Ok(HostDeviceBuffer {
            cluster: self.cluster.clone(),
            allocation,
            buffer: *buffer,
        })
    }

    /// Runs one exchange through the cluster and the given receive-side proxy.
    fn run_exchange(
        &self,
        output: &HostDeviceBuffer,
        exchange: Exchange,
        output_offset_bytes: usize,
        slot_index: usize,
        proxy: &mut impl ProxySink,
    ) -> Result<DispatchRecord, CommError> {
        let started = Instant::now();
        let rank = self.rank;
        let world = self.cluster.world_size();
        let rails = exchange.rails();
        let seq = exchange.seq();
        let elements = exchange.elements();
        let bytes = exchange.bytes();
        if slot_index >= self.cluster.slot_ring() {
            return Err(invalid(format!(
                "slot {slot_index} is outside a ring of {}",
                self.cluster.slot_ring()
            )));
        }
        self.state.lock().sequence.accept(seq).map_err(seam_error)?;
        let allocation = output.allocation();
        if output_offset_bytes + bytes > output.device_buffer().bytes() {
            return Err(invalid("the chunk output offset exceeds the output buffer"));
        }
        let previous = {
            let state = self.cluster.inner.state.lock();
            state.slot_last_seq[slot_index]
        };
        if let Some(previous_seq) = previous {
            if previous_seq > seq {
                return Err(CommError::Protocol(format!(
                    "slot {slot_index} last carried sequence {previous_seq}, this exchange is {seq}"
                )));
            }
            // Equal is the other ranks staging the same exchange; only a strictly earlier use is a
            // ring reuse and needs the readers' done epoch.
            if previous_seq == seq {
                // Nothing to wait for: this is the concurrent staging of one exchange.
            } else {
                let epoch = previous_seq + 1;
                let deadline = self.cluster.deadline();
                let mut state = self.cluster.inner.state.lock();
                loop {
                    let pending: Vec<usize> = (0..world)
                        .filter(|peer| state.read_epoch[*peer][slot_index] < epoch)
                        .collect();
                    if pending.is_empty() {
                        break;
                    }
                    if !state.alive[rank] {
                        return Err(CommError::PeerClosed { peer: rank });
                    }
                    self.cluster.wait(
                    &mut state,
                    &format!(
                        "waiting for ranks {pending:?} to finish reading slot {slot_index} at sequence {previous_seq}"
                    ),
                    deadline,
                )?;
                }
            }
        }

        let input = allocation.bytes(output_offset_bytes, bytes);
        let mut state = self.cluster.inner.state.lock();
        if !state.alive[rank] {
            return Err(CommError::PeerClosed { peer: rank });
        }
        state.slot_last_seq[slot_index] = Some(seq);
        state.slot_seq[rank][slot_index] = seq + 1;
        state.doorbell[rank] = seq + 1;
        let row = row_offset(rank);
        state.slots[rank][slot_index].bytes.as_mut()[row..row + bytes].copy_from_slice(&input);
        let slot_address = state.slots[rank][slot_index].address();
        self.cluster.inner.doorbells.fetch_add(1, Ordering::Relaxed);
        self.cluster
            .inner
            .collectives
            .fetch_add(1, Ordering::Relaxed);
        self.cluster.notify();
        proxy.begin().map_err(seam_error)?;

        let doorbell_deadline = self.cluster.deadline();
        loop {
            let staged: Vec<usize> = (0..world)
                .filter(|peer| state.doorbell[*peer] < seq + 1)
                .collect();
            if staged.is_empty() {
                break;
            }
            self.cluster.wait(
                &mut state,
                &format!("waiting for ranks {staged:?} to stage sequence {seq}"),
                doorbell_deadline,
            )?;
        }
        for peer in 0..world {
            if peer != rank && state.slot_seq[peer][slot_index] != seq + 1 {
                return Err(CommError::Protocol(format!(
                    "rank {peer} staged sequence {} into slot {}, but this rank posts sequence {seq} into slot {slot_index}",
                    state.slot_seq[peer][slot_index] - 1,
                    peer
                )));
            }
        }

        let peers: Vec<usize> = (0..world).filter(|peer| *peer != rank).collect();
        let peer_order: Vec<usize> = if self.cluster.inner.config.reverse_arrivals {
            peers.iter().rev().copied().collect()
        } else {
            peers.clone()
        };
        for peer in &peer_order {
            let rail_order: Vec<usize> = if self.cluster.inner.config.reverse_arrivals {
                (0..rails).rev().collect()
            } else {
                (0..rails).collect()
            };
            for rail in rail_order {
                let len = exchange.rail_bytes(rail);
                if len == 0 {
                    continue;
                }
                let offset = if rail == 0 { 0 } else { exchange.rail_bytes(0) };
                let source = state.slots[rank][slot_index].bytes.as_slice()
                    [row + offset..row + offset + len]
                    .to_vec();
                state.slots[*peer][slot_index].bytes.as_mut()[row + offset..row + offset + len]
                    .copy_from_slice(&source);
                let frame = RailFrame {
                    sender: rank as u8,
                    rail,
                    phase: ONESHOT_PHASE,
                    kind: ONESHOT_KIND,
                    sequence_low: (seq & 0xff) as u8,
                };
                state.inbox[*peer].push(frame);
                let mut ledger = self.cluster.inner.rail_ledger.lock();
                match rail {
                    0 => ledger.rail0_bytes += len as u64,
                    _ => ledger.rail1_bytes += len as u64,
                }
                ledger.transfers += 1;
                ledger.element_writes += (len / 2) as u64;
                drop(ledger);
            }
        }
        self.cluster.notify();

        let expected_frames = (world - 1) * rails;
        let frame_deadline = self.cluster.deadline();
        let frames = loop {
            if state.inbox[rank].len() >= expected_frames {
                break std::mem::take(&mut state.inbox[rank]);
            }
            self.cluster.wait(
                &mut state,
                &format!("waiting for rail frames at sequence {seq}"),
                frame_deadline,
            )?;
        };
        drop(state);

        for frame in &frames {
            if let Some(peer) = proxy.frame(frame).map_err(seam_error)? {
                let mut state = self.cluster.inner.state.lock();
                state.arrival[rank][peer] = seq + 1;
                self.cluster.notify();
            }
        }
        if !proxy.all_published() {
            return Err(CommError::Protocol(
                "the receive-side proxy did not publish every peer after all rail frames landed"
                    .to_owned(),
            ));
        }

        let arrival_deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
        loop {
            let missing: Vec<usize> = (0..world)
                .filter(|peer| *peer != rank && state.arrival[rank][*peer] < seq + 1)
                .collect();
            if missing.is_empty() {
                break;
            }
            self.cluster.wait(
                &mut state,
                &format!("waiting for arrival words from ranks {missing:?} at sequence {seq}"),
                arrival_deadline,
            )?;
        }

        let rows: Vec<Vec<u16>> = (0..world)
            .map(|peer| {
                state.slots[rank][slot_index].bytes.as_slice()
                    [row_offset(peer)..row_offset(peer) + bytes]
                    .chunks(2)
                    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                    .collect()
            })
            .collect();
        let peer_rows: Vec<usize> = (0..world).filter(|peer| *peer != rank).collect();
        let mut addresses = [0u64; ONESHOT_RANKS - 1];
        for (index, peer) in peer_rows.iter().enumerate() {
            addresses[index] = slot_address + row_offset(*peer) as u64;
        }
        drop(state);

        let row_slices: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
        let mut reduced = vec![0u16; elements];
        kernel_reduce_into(&mut reduced, &row_slices)?;
        allocation.store_bf16(output_offset_bytes, &reduced);

        let mut state = self.cluster.inner.state.lock();
        state.read_epoch[rank][slot_index] = seq + 1;
        self.cluster.notify();
        drop(state);

        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let record = DispatchRecord::new(
            rank,
            seq,
            exchange,
            DispatchAddresses {
                slot: slot_address,
                local_row: slot_address + row_offset(rank) as u64,
                peer_rows: addresses,
            },
            LaunchPath::DeviceKernel,
            0,
            elapsed_ns,
        )
        .map_err(seam_error)?;
        self.state.lock().log.record(record).map_err(seam_error)?;
        Ok(record)
    }
}

/// The canonical reduction the CUDA kernel performs, written out as the CPU twin.
///
/// Every element accumulates from 0.0f through the ranks in ascending order in f32 and is rounded
/// to bf16 exactly once. The seed and the order are the contract: seeding from the first operand
/// would turn an all-negative-zero payload into negative zero, and rounding after each rank would
/// make the result depend on the serial chain rather than on the rank sum.
///
/// # Errors
/// Refuses a rank count other than four and rows whose lengths differ from the output.
pub fn kernel_reduce_into(out: &mut [u16], rows: &[&[u16]]) -> Result<(), CommError> {
    if rows.len() != ONESHOT_RANKS {
        return Err(invalid(format!(
            "the kernel reduces {ONESHOT_RANKS} rank rows, {} were passed",
            rows.len()
        )));
    }
    for (rank, row) in rows.iter().enumerate() {
        if row.len() != out.len() {
            return Err(invalid(format!(
                "rank {rank}'s row has {} elements but the output has {}",
                row.len(),
                out.len()
            )));
        }
    }
    for (element, value) in out.iter_mut().enumerate() {
        let mut accumulator = 0.0f32;
        for row in rows {
            accumulator += bf16_to_f32(row[element]);
        }
        *value = f32_to_bf16_rn(accumulator);
    }
    Ok(())
}

/// A deterministic finite bf16 payload in [-1, 1] for one (rank, sequence) pair.
#[must_use]
pub fn twin_payload(rank: usize, seq: u64, elements: usize) -> Vec<u16> {
    (0..elements)
        .map(|index| {
            let mut state =
                ((rank as u64) + 1).rotate_left(17) ^ seq.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            state ^= index as u64;
            state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            state ^= state >> 33;
            let sign = ((state >> 20) as u16 & 1) << 15;
            let exponent = 0x3b00u16 | (((state >> 8) as u16 & 0x0f) << 2);
            let mantissa = (state as u16) & 0x3f;
            (exponent | mantissa) & 0x7fff | sign
        })
        .collect()
}

/// What one loopback host-twin run proved, and which measurements stay unclaimed.
#[derive(Debug, Clone, Default)]
pub struct HostTwinReport {
    /// The collective form that ran: one-shot sequences or one chunked plan.
    pub mode: &'static str,
    /// Sequences run (one-shot) or chunks run (chunked).
    pub steps: usize,
    /// Elements per exchange.
    pub elements: usize,
    /// Bytes per exchange.
    pub bytes: usize,
    /// Rails every exchange posted.
    pub rails: usize,
    /// Slots in the ring the run used.
    pub slot_ring: usize,
    /// Ring slots used, in run order.
    pub slot_indices: Vec<usize>,
    /// Dispatches recorded across the four ranks.
    pub dispatched: usize,
    /// Host-staging bytes across every dispatch record (zero for the device path).
    pub host_copy_bytes: usize,
    /// Whether every rank's output equalled the canonical CPU twin bit for bit.
    pub bit_identical: bool,
    /// Whether every rank's own records took the device path.
    pub device_path_only: bool,
    /// Bytes the simulated NIC moved, per rail.
    pub posted: RailLedger,
    /// Wall time of the whole four-rank run (a host number, not a node latency).
    pub elapsed: Duration,
    /// Measurements this run cannot produce; the window run must claim them.
    pub unclaimed: Vec<&'static str>,
}

/// The unclaimed measurement list every host-twin report carries.
const UNCLAIMED: [&str; 5] = [
    "one-shot all-reduce p50 latency at 10 KiB against the 20 us gate",
    "per-rail bandwidth on the live 200 Gb/s rails",
    "correctness across the four-node mesh through the real kernel, MR and QP path",
    "10,000-replay determinism and two-run byte identity on the nodes",
    "device-side DtoH/HtoD absence, which a host allocation cannot demonstrate",
];

#[allow(clippy::too_many_arguments)]
fn twin_result(
    cluster: &LoopbackDeviceCluster,
    outputs: &[Vec<u16>],
    expected: &[u16],
    logs: Vec<DispatchLog>,
    mode: &'static str,
    steps: usize,
    slot_indices: Vec<usize>,
    elapsed: Duration,
) -> HostTwinReport {
    let mut bit_identical = true;
    let mut dispatched = 0;
    let mut host_copy_bytes = 0;
    let mut device_path_only = true;
    for (rank, log) in logs.iter().enumerate() {
        if outputs[rank] != expected {
            bit_identical = false;
        }
        dispatched += log.len();
        host_copy_bytes += log.host_copy_bytes();
        if log.device_path_ok().is_err() {
            device_path_only = false;
        }
    }
    let elements = outputs.first().map_or(0, Vec::len);
    HostTwinReport {
        mode,
        steps,
        elements,
        bytes: elements * 2,
        rails: cluster.rails(),
        slot_ring: cluster.slot_ring(),
        slot_indices,
        dispatched,
        host_copy_bytes,
        bit_identical,
        device_path_only,
        posted: cluster.rail_ledger(),
        elapsed,
        unclaimed: UNCLAIMED.to_vec(),
    }
}

/// Runs one-shot sequences with four rank threads and compares every output with the CPU twin.
///
/// The harness writes each rank's contribution into its simulated device buffer before each
/// dispatch, standing for the upstream kernel filling the next contribution; that write is outside
/// the collective and is not counted as a host-staging copy.
///
/// # Errors
/// Refuses an illegal element count, a cluster construction failure, or any exchange error.
pub fn run_host_twin(
    elements: usize,
    sequences: usize,
    rails: usize,
) -> Result<HostTwinReport, CommError> {
    if sequences == 0 {
        return Err(invalid("a host twin run needs at least one sequence"));
    }
    let config = LoopbackDeviceConfig {
        rails,
        ..LoopbackDeviceConfig::default()
    };
    let cluster = LoopbackDeviceCluster::new(config)?;
    let payloads: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
        .map(|rank| twin_payload(rank, 0, elements))
        .collect();
    let mut expected = vec![0u16; elements];
    let slices: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
    canonical_all_reduce_into(&mut expected, &slices)?;
    let started = Instant::now();
    let mut threads = Vec::with_capacity(ONESHOT_RANKS);
    for (rank, payload) in payloads.iter().enumerate() {
        let handle = cluster.handle(rank)?;
        let output = HostDeviceBuffer::allocate(&cluster, rank, elements)?;
        let payload = payload.clone();
        threads.push(std::thread::spawn(move || {
            for seq in 0..sequences as u64 {
                output.write(&payload)?;
                handle.all_reduce_bf16_device(&output.device_buffer(), elements, seq)?;
            }
            Ok::<(Vec<u16>, DispatchLog), CommError>((output.read(), handle.dispatch_log()))
        }));
    }
    let mut outputs = Vec::with_capacity(ONESHOT_RANKS);
    let mut logs = Vec::with_capacity(ONESHOT_RANKS);
    for thread in threads {
        let (output, log) = thread
            .join()
            .map_err(|_| CommError::Protocol("a host twin rank thread panicked".to_owned()))??;
        outputs.push(output);
        logs.push(log);
    }
    let elapsed = started.elapsed();
    let slot_indices = logs
        .first()
        .map(|log| {
            log.records()
                .iter()
                .map(|record| record.seq() as usize % cluster.slot_ring())
                .collect()
        })
        .unwrap_or_default();
    Ok(twin_result(
        &cluster,
        &outputs,
        &expected,
        logs,
        "oneshot",
        sequences,
        slot_indices,
        elapsed,
    ))
}

/// Runs one chunked plan through the ring and compares every chunk with the CPU twin.
///
/// # Errors
/// Refuses an illegal plan, a cluster construction failure, or any exchange error.
pub fn run_host_twin_prefill(
    total_elements: usize,
    rails: usize,
) -> Result<HostTwinReport, CommError> {
    let bytes = total_elements
        .checked_mul(2)
        .ok_or_else(|| invalid("the prefill plan is too large"))?;
    let plan = ChunkPlan::new(bytes, rails).map_err(seam_error)?;
    if plan.chunk_count() < 2 {
        return Err(invalid(format!(
            "{total_elements} elements fit one row; a chunked plan needs more than one"
        )));
    }
    let config = LoopbackDeviceConfig {
        rails,
        ..LoopbackDeviceConfig::default()
    };
    let cluster = LoopbackDeviceCluster::new(config)?;
    let payloads: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
        .map(|rank| twin_payload(rank, 0, total_elements))
        .collect();
    let mut expected = vec![0u16; total_elements];
    let slices: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
    canonical_all_reduce_into(&mut expected, &slices)?;
    let started = Instant::now();
    let mut threads = Vec::with_capacity(ONESHOT_RANKS);
    for (rank, payload) in payloads.iter().enumerate() {
        let handle = cluster.handle(rank)?;
        let output = HostDeviceBuffer::allocate(&cluster, rank, total_elements)?;
        let payload = payload.clone();
        let plan = plan.clone();
        threads.push(std::thread::spawn(move || {
            output.write(&payload)?;
            let report = handle.run_pipeline(&output, &plan, 0)?;
            Ok::<(Vec<u16>, DispatchLog, PipelineReport), CommError>((
                output.read(),
                handle.dispatch_log(),
                report,
            ))
        }));
    }
    let mut outputs = Vec::with_capacity(ONESHOT_RANKS);
    let mut logs = Vec::with_capacity(ONESHOT_RANKS);
    let mut first_report = PipelineReport::default();
    for (rank, thread) in threads.into_iter().enumerate() {
        let (output, log, report) = thread
            .join()
            .map_err(|_| CommError::Protocol("a host twin rank thread panicked".to_owned()))??;
        if rank == 0 {
            first_report = report;
        }
        outputs.push(output);
        logs.push(log);
    }
    let elapsed = started.elapsed();
    Ok(twin_result(
        &cluster,
        &outputs,
        &expected,
        logs,
        "chunked",
        plan.chunk_count(),
        first_report.slot_indices,
        elapsed,
    ))
}
