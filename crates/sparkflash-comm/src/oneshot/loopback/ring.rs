//! The ring/chunked host twin for prefill-sized collectives (task-36 host portion).
//!
//! Above the 2 MiB one-shot row, a prefill seam is an ordered series of row-sized exchanges that
//! reuse one slot ring. The one-shot host twin (the parent module) runs that series chunk by chunk
//! on one thread per rank, which proves the arithmetic and the ring bookkeeping but not the thing
//! the ring exists for: overlapping two chunks. This module adds the overlapped twin.
//!
//! Per rank, a proxy thread stages chunk i while the kernel thread is still reducing chunk i-1: the
//! proxy owns staging, the doorbell, the rail frames and the per-slot reuse wait; the kernel owns
//! the receive-side proxy, the arrival words, the four-row reduction and the read-done epoch. With
//! a two-deep ring the proxy can have chunk i staged while chunk i-1 is still being read, so the
//! cluster's per-rank in-flight slot count reaches two; with a one-deep ring the reuse wait forces
//! one, which is the serialised baseline.
//!
//! The read-done epoch is the load-bearing ordering rule, and the module can demonstrate that
//! rather than assert it: [ReadEpochPolicy::Ignore] (a deliberate contract violation, test-only)
//! and [super::KernelDelay] (a deliberate slow kernel read) together let a test overwrite a slot
//! before its readers are done and observe the corrupted output. The same run with the real policy
//! is bit-identical, so the hazard is shown, not described.
//!
//! What this cannot do: allocate device memory, register an MR, post a work request, launch a
//! kernel or measure a node. The bandwidth, latency, mesh correctness, determinism and device-side
//! copy-absence numbers are named unclaimed in [RingTwinReport::unclaimed]; none may be cited from
//! a host run.

use std::time::{Duration, Instant};

use crate::error::CommError;
use crate::oneshot::{
    ArrivalProxy, DispatchAddresses, DispatchLog, DispatchRecord, Exchange, LaunchPath,
    ONESHOT_KIND, ONESHOT_PHASE, ONESHOT_RANKS, ONESHOT_ROW_ELEMENTS, RailFrame, row_offset,
};
use crate::rail::RailLedger;
use crate::reduce::canonical_all_reduce_into;

use super::{
    HostDeviceBuffer, KernelDelay, LoopbackDeviceCluster, LoopbackDeviceConfig,
    LoopbackDeviceTransport, ReadEpochPolicy, invalid, seam_error, twin_payload,
};

/// One chunk of a ring plan: a whole number of bf16 elements and its offset in the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingChunk {
    elements: usize,
    offset_bytes: usize,
}

impl RingChunk {
    /// bf16 elements in this chunk.
    #[must_use]
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Byte offset of this chunk in the output payload.
    #[must_use]
    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    /// Bytes in this chunk.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.elements * 2
    }
}

/// An ordered split of a payload larger than one row into one-shot exchanges.
///
/// The last chunk may be shorter; every chunk is at most the pinned row. This is the test-side
/// plan for the ring twin; the production chunked contract stays in the parent module's ChunkPlan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingPlan {
    chunks: Vec<RingChunk>,
    total_elements: usize,
    rails: usize,
}

impl RingPlan {
    /// Splits a payload into row-sized chunks (the prefill path's natural shape).
    ///
    /// # Errors
    /// Refuses an empty payload, a rail count other than one or two, and overflow.
    pub fn rows(total_elements: usize, rails: usize) -> Result<Self, CommError> {
        Self::with_chunk_elements(total_elements, ONESHOT_ROW_ELEMENTS, rails)
    }

    /// Splits a payload into chunks of a given element count, with a possibly shorter last chunk.
    ///
    /// # Errors
    /// Refuses an empty payload, a zero or over-row chunk, a rail count other than one or two,
    /// and overflow.
    pub fn with_chunk_elements(
        total_elements: usize,
        chunk_elements: usize,
        rails: usize,
    ) -> Result<Self, CommError> {
        if total_elements == 0 {
            return Err(invalid("a ring plan needs at least one element"));
        }
        if chunk_elements == 0 || chunk_elements > ONESHOT_ROW_ELEMENTS {
            return Err(invalid(format!(
                "a ring chunk carries 1..={ONESHOT_ROW_ELEMENTS} elements, {chunk_elements} was requested"
            )));
        }
        if rails != 1 && rails != crate::oneshot::ONESHOT_RAILS {
            return Err(invalid(format!(
                "the one-shot modes are 1 or {} rails, {rails} was requested",
                crate::oneshot::ONESHOT_RAILS
            )));
        }
        let mut chunks = Vec::new();
        let mut offset_bytes = 0usize;
        let mut remaining = total_elements;
        while remaining > 0 {
            let elements = remaining.min(chunk_elements);
            chunks.push(RingChunk {
                elements,
                offset_bytes,
            });
            offset_bytes = offset_bytes
                .checked_add(elements * 2)
                .ok_or_else(|| invalid("the ring plan is too large"))?;
            remaining -= elements;
        }
        Ok(Self {
            chunks,
            total_elements,
            rails,
        })
    }

    /// Number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The chunks in order.
    #[must_use]
    pub fn chunks(&self) -> &[RingChunk] {
        &self.chunks
    }

    /// Total bf16 elements in the payload.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Total payload bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_elements * 2
    }

    /// Rails each exchange posts.
    #[must_use]
    pub fn rails(&self) -> usize {
        self.rails
    }

    /// The exchange of one chunk at a sequence.
    ///
    /// # Errors
    /// Refuses an out-of-range chunk or an exchange the seam would refuse.
    pub fn exchange(&self, index: usize, seq: u64) -> Result<Exchange, CommError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            invalid(format!(
                "chunk {index} is outside a plan of {} chunks",
                self.chunks.len()
            ))
        })?;
        Exchange::new(seq, chunk.elements, self.rails).map_err(seam_error)
    }

    /// The sequence of one chunk: strictly increasing with the chunk index.
    ///
    /// # Errors
    /// Refuses an out-of-range chunk or a sequence overflow.
    pub fn chunk_seq(&self, base_seq: u64, index: usize) -> Result<u64, CommError> {
        if index >= self.chunks.len() {
            return Err(invalid(format!(
                "chunk {index} is outside a plan of {} chunks",
                self.chunks.len()
            )));
        }
        base_seq
            .checked_add(u64::try_from(index).unwrap_or(u64::MAX))
            .ok_or_else(|| invalid("the ring sequence would overflow"))
    }
}

/// Configuration of one ring host-twin run.
#[derive(Debug, Clone)]
pub struct RingTwinConfig {
    /// Slots per rank: one (serialised baseline) or two (the two-deep pipeline).
    pub ring: usize,
    /// Rails per exchange: one or two.
    pub rails: usize,
    /// Chunk size in elements; None means row-sized chunks.
    pub chunk_elements: Option<usize>,
    /// A deliberate kernel-side stall (test knob; see [KernelDelay]).
    pub kernel_delay: Option<KernelDelay>,
    /// The read-done epoch policy; Honor is the only mode a device path may use.
    pub read_epoch_policy: ReadEpochPolicy,
}

impl Default for RingTwinConfig {
    fn default() -> Self {
        Self {
            ring: super::HOST_TWIN_RING,
            rails: crate::oneshot::ONESHOT_RAILS,
            chunk_elements: None,
            kernel_delay: None,
            read_epoch_policy: ReadEpochPolicy::Honor,
        }
    }
}

/// The measurements a host ring run cannot make; the window must claim them.
const UNCLAIMED_RING: [&str; 6] = [
    "per-chunk and total prefill latency at 4096 tokens against the S5.4 plan",
    "bandwidth of the 40 MiB hidden and 48 MiB engram-gather ring paths on the live rails",
    "correctness across the four-node mesh through the real kernel, MR and QP path",
    "the engram all-gather disk warm/cold latency and the zero-pad semantics on the nodes",
    "10,000-replay determinism and two-run byte identity at prefill size",
    "device-side DtoH/HtoD absence between chunks, which a host allocation cannot demonstrate",
];

/// What one chunked pipeline run through the overlapped ring did.
#[derive(Debug, Clone, Default)]
pub struct RingPipelineReport {
    /// The ring slot each chunk used, in chunk order.
    pub slots: Vec<usize>,
    /// The dispatch record of every chunk, in chunk order.
    pub records: Vec<DispatchRecord>,
    /// Host-staging bytes over every record: zero for the device path.
    pub host_copy_bytes: usize,
    /// Bytes the simulated NIC moved per rail during this run.
    pub posted: RailLedger,
    /// Wall time of the whole run on one rank.
    pub elapsed: Duration,
    /// The most slots this rank held in flight at once (two means the pipeline overlapped).
    pub max_rank_in_flight: usize,
}

/// What one four-rank ring-twin run proved, and what stays unclaimed.
#[derive(Debug, Clone, Default)]
pub struct RingTwinReport {
    /// Total bf16 elements in the payload.
    pub total_elements: usize,
    /// Total payload bytes.
    pub total_bytes: usize,
    /// Every chunk's element count, in order.
    pub chunk_elements: Vec<usize>,
    /// Every chunk's byte offset, in order.
    pub chunk_offsets: Vec<usize>,
    /// Number of chunks.
    pub chunks: usize,
    /// Slots per rank.
    pub ring: usize,
    /// Rails per exchange.
    pub rails: usize,
    /// Whether every rank's whole payload equalled the canonical one-rounding twin bit for bit.
    pub bit_identical: bool,
    /// Ranks whose output did not match (empty when bit_identical).
    pub mismatched_ranks: Vec<usize>,
    /// Whether every recorded dispatch took the device path.
    pub device_path_only: bool,
    /// Dispatches recorded across the four ranks.
    pub dispatched: usize,
    /// Host-staging bytes across every dispatch record (zero for the device path).
    pub host_copy_bytes: usize,
    /// The most slots one rank held in flight at once: one serialised, two overlapped.
    pub max_rank_in_flight: usize,
    /// Bytes the simulated NIC moved per rail.
    pub posted: RailLedger,
    /// Wall time of the whole four-rank run (a host number, not a node latency).
    pub elapsed: Duration,
    /// The read-done epoch policy the run used (Honor unless a hazard test opted out).
    pub read_epoch_policy: ReadEpochPolicy,
    /// Measurements this run cannot produce; the window run must claim them.
    pub unclaimed: Vec<&'static str>,
}

/// Runs a prefill-sized payload through the overlapped ring twin with generated payloads.
///
/// # Errors
/// Refuses an illegal plan, a cluster construction failure, or any exchange error.
pub fn run_ring_twin(
    total_elements: usize,
    config: RingTwinConfig,
) -> Result<RingTwinReport, CommError> {
    let payloads: Vec<Vec<u16>> = (0..ONESHOT_RANKS)
        .map(|rank| twin_payload(rank, 0, total_elements))
        .collect();
    run_ring_twin_with_payloads(total_elements, config, payloads)
}

/// Runs a prefill-sized payload through the overlapped ring twin with caller-supplied payloads.
///
/// # Errors
/// Refuses a payload set that is not four rows of the plan's length, an illegal plan, a cluster
/// construction failure, or any exchange error.
pub fn run_ring_twin_with_payloads(
    total_elements: usize,
    config: RingTwinConfig,
    payloads: Vec<Vec<u16>>,
) -> Result<RingTwinReport, CommError> {
    if payloads.len() != ONESHOT_RANKS {
        return Err(invalid(format!(
            "the ring twin needs {ONESHOT_RANKS} rank payloads, {} were passed",
            payloads.len()
        )));
    }
    for (rank, payload) in payloads.iter().enumerate() {
        if payload.len() != total_elements {
            return Err(invalid(format!(
                "rank {rank}'s payload has {} elements, the plan carries {total_elements}",
                payload.len()
            )));
        }
    }
    let plan = match config.chunk_elements {
        Some(chunk) => RingPlan::with_chunk_elements(total_elements, chunk, config.rails)?,
        None => RingPlan::rows(total_elements, config.rails)?,
    };
    if plan.chunk_count() < 2 {
        return Err(invalid(format!(
            "{total_elements} elements fit one {ops}element row; a ring plan needs more than one chunk",
            ops = String::new()
        )));
    }
    let cluster = LoopbackDeviceCluster::new(LoopbackDeviceConfig {
        slot_ring: config.ring,
        rails: config.rails,
        read_epoch_policy: config.read_epoch_policy,
        kernel_delay: config.kernel_delay,
        ..LoopbackDeviceConfig::default()
    })?;
    let mut expected = vec![0u16; total_elements];
    {
        let rows: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
        canonical_all_reduce_into(&mut expected, &rows)?;
    }
    let started = Instant::now();
    let mut threads = Vec::with_capacity(ONESHOT_RANKS);
    for (rank, payload) in payloads.iter().enumerate() {
        let handle = cluster.handle(rank)?;
        let output = HostDeviceBuffer::allocate(&cluster, rank, total_elements)?;
        let payload = payload.clone();
        let plan = plan.clone();
        threads.push(std::thread::spawn(move || {
            output.write(&payload)?;
            let report = handle.run_ring_pipeline(&output, &plan)?;
            Ok::<(Vec<u16>, DispatchLog, RingPipelineReport), CommError>((
                output.read(),
                handle.dispatch_log(),
                report,
            ))
        }));
    }
    let mut outputs = Vec::with_capacity(ONESHOT_RANKS);
    let mut logs = Vec::with_capacity(ONESHOT_RANKS);
    for thread in threads {
        let (output, log, _report) = thread
            .join()
            .map_err(|_| CommError::Protocol("a ring twin rank thread panicked".to_owned()))??;
        outputs.push(output);
        logs.push(log);
    }
    let elapsed = started.elapsed();
    let mut mismatched_ranks = Vec::new();
    for (rank, output) in outputs.iter().enumerate() {
        if output != &expected {
            mismatched_ranks.push(rank);
        }
    }
    let mut dispatched = 0usize;
    let mut host_copy_bytes = 0usize;
    let mut device_path_only = true;
    for log in &logs {
        dispatched += log.len();
        host_copy_bytes += log.host_copy_bytes();
        if log.device_path_ok().is_err() {
            device_path_only = false;
        }
    }
    let chunk_elements: Vec<usize> = plan.chunks().iter().map(RingChunk::elements).collect();
    let chunk_offsets: Vec<usize> = plan.chunks().iter().map(RingChunk::offset_bytes).collect();
    Ok(RingTwinReport {
        total_elements,
        total_bytes: total_elements * 2,
        chunk_elements,
        chunk_offsets,
        chunks: plan.chunk_count(),
        ring: cluster.slot_ring(),
        rails: cluster.rails(),
        bit_identical: mismatched_ranks.is_empty(),
        mismatched_ranks,
        device_path_only,
        dispatched,
        host_copy_bytes,
        max_rank_in_flight: cluster.max_rank_in_flight(),
        posted: cluster.rail_ledger(),
        elapsed,
        read_epoch_policy: config.read_epoch_policy,
        unclaimed: UNCLAIMED_RING.to_vec(),
    })
}

impl LoopbackDeviceTransport {
    /// Runs a prefill-sized payload through the overlapped two-thread pipeline on this rank.
    ///
    /// The proxy thread stages each chunk and sends its rail frames; this thread receives, publishes
    /// the arrival words, reduces the four slot rows into the output and publishes the read-done
    /// epoch. A chunk i+2 may only be staged into slot i once every rank's read-done epoch for the
    /// slot's previous use is published (unless the cluster was deliberately built with
    /// [ReadEpochPolicy::Ignore], which exists only to demonstrate the hazard).
    ///
    /// # Errors
    /// Refuses an output buffer that does not belong to this rank or cluster, a plan on a different
    /// rail count, and any exchange, arrival or ring violation underneath.
    pub fn run_ring_pipeline(
        &self,
        output: &HostDeviceBuffer,
        plan: &RingPlan,
    ) -> Result<RingPipelineReport, CommError> {
        self.check_output(output)?;
        if plan.total_elements() != output.elements() {
            return Err(invalid(format!(
                "the plan carries {} elements but the output buffer holds {}",
                plan.total_elements(),
                output.elements()
            )));
        }
        if plan.rails() != self.cluster.rails() {
            return Err(invalid(format!(
                "the plan posts {} rails but the cluster runs {}",
                plan.rails(),
                self.cluster.rails()
            )));
        }
        let before = self.cluster.rail_ledger();
        let started = Instant::now();
        let this = self;
        let records = std::thread::scope(|scope| -> Result<Vec<DispatchRecord>, CommError> {
            let proxy = scope.spawn(move || this.proxy_loop(output, plan));
            let kernel = this.kernel_loop(output, plan);
            let proxy_result = proxy
                .join()
                .map_err(|_| CommError::Protocol("the ring proxy thread panicked".to_owned()))?;
            proxy_result?;
            kernel
        })?;
        let after = self.cluster.rail_ledger();
        let posted = RailLedger {
            rail0_bytes: after.rail0_bytes - before.rail0_bytes,
            rail1_bytes: after.rail1_bytes - before.rail1_bytes,
            transfers: after.transfers - before.transfers,
            element_writes: after.element_writes - before.element_writes,
        };
        let host_copy_bytes = records.iter().map(DispatchRecord::host_copy_bytes).sum();
        Ok(RingPipelineReport {
            slots: (0..plan.chunk_count())
                .map(|index| index % self.cluster.slot_ring())
                .collect(),
            records,
            host_copy_bytes,
            posted,
            elapsed: started.elapsed(),
            max_rank_in_flight: self.cluster.max_rank_in_flight(),
        })
    }

    fn proxy_loop(&self, output: &HostDeviceBuffer, plan: &RingPlan) -> Result<(), CommError> {
        let ring = self.cluster.slot_ring();
        for (index, chunk) in plan.chunks().iter().enumerate() {
            let seq = plan.chunk_seq(0, index)?;
            let exchange = plan.exchange(index, seq)?;
            let slot = index % ring;
            self.state.lock().sequence.accept(seq).map_err(seam_error)?;
            let previous = self.cluster.inner.state.lock().slot_last_seq[slot];
            if let Some(previous_seq) = previous {
                if previous_seq > seq {
                    return Err(CommError::Protocol(format!(
                        "slot {slot} last carried sequence {previous_seq}, this chunk is {seq}"
                    )));
                }
                if previous_seq < seq
                    && self.cluster.inner.config.read_epoch_policy == ReadEpochPolicy::Honor
                {
                    self.wait_slot_drained(slot, previous_seq)?;
                }
            }
            let input = output
                .allocation()
                .bytes(chunk.offset_bytes(), chunk.bytes());
            self.stage_slot(slot, seq, &input)?;
            self.wait_doorbells(slot, seq)?;
            self.send_frames(slot, exchange)?;
        }
        Ok(())
    }

    fn kernel_loop(
        &self,
        output: &HostDeviceBuffer,
        plan: &RingPlan,
    ) -> Result<Vec<DispatchRecord>, CommError> {
        let ring = self.cluster.slot_ring();
        let mut records = Vec::with_capacity(plan.chunk_count());
        for (index, chunk) in plan.chunks().iter().enumerate() {
            let started = Instant::now();
            let seq = plan.chunk_seq(0, index)?;
            let exchange = plan.exchange(index, seq)?;
            let slot = index % ring;
            let mut proxy = ArrivalProxy::new(self.rank, exchange).map_err(seam_error)?;
            proxy.on_doorbell(seq).map_err(seam_error)?;
            let frames = self.take_frames(exchange)?;
            for frame in &frames {
                if let Some(peer) = proxy.on_frame(frame).map_err(seam_error)? {
                    self.publish_arrival(peer, seq);
                }
            }
            if !proxy.all_published() {
                return Err(CommError::Protocol(
                    "the receive-side proxy did not publish every peer after all rail frames landed"
                        .to_owned(),
                ));
            }
            self.wait_arrivals(seq)?;
            if let Some(delay) = self.cluster.inner.config.kernel_delay
                && delay.rank == self.rank
                && delay.chunk == index
            {
                // In the hazard demonstration the same knob waits for the premature reuse to
                // land instead of sleeping, so the corruption is deterministic rather than a
                // race between the proxy and the delayed kernel.
                let reuse_index = index + ring;
                if self.cluster.inner.config.read_epoch_policy == ReadEpochPolicy::Ignore
                    && reuse_index < plan.chunk_count()
                {
                    let reuse_seq = plan.chunk_seq(0, reuse_index)?;
                    self.wait_for_reuse_staged(slot, reuse_seq)?;
                } else {
                    std::thread::sleep(delay.duration);
                }
            }
            let rows = self.read_rows(slot, exchange.bytes());
            let mut reduced = vec![0u16; chunk.elements()];
            let row_slices: Vec<&[u16]> = rows.iter().map(Vec::as_slice).collect();
            super::kernel_reduce_into(&mut reduced, &row_slices)?;
            output
                .allocation()
                .store_bf16(chunk.offset_bytes(), &reduced);
            self.publish_read_done(slot, seq);
            let (slot_address, peer_rows) = self.slot_addresses(slot);
            let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let record = DispatchRecord::new(
                self.rank,
                seq,
                exchange,
                DispatchAddresses {
                    slot: slot_address,
                    local_row: slot_address + row_offset(self.rank) as u64,
                    peer_rows,
                },
                LaunchPath::DeviceKernel,
                0,
                elapsed_ns,
            )
            .map_err(seam_error)?;
            self.state.lock().log.record(record).map_err(seam_error)?;
            records.push(record);
        }
        Ok(records)
    }

    fn wait_slot_drained(&self, slot: usize, previous_seq: u64) -> Result<(), CommError> {
        let epoch = previous_seq + 1;
        let world = self.cluster.world_size();
        let deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
        loop {
            let pending: Vec<usize> = (0..world)
                .filter(|peer| state.read_epoch[*peer][slot] < epoch)
                .collect();
            if pending.is_empty() {
                return Ok(());
            }
            if !state.alive[self.rank] {
                return Err(CommError::PeerClosed { peer: self.rank });
            }
            self.cluster.wait(
                &mut state,
                &format!(
                    "waiting for ranks {pending:?} to finish reading slot {slot} at sequence {previous_seq}"
                ),
                deadline,
            )?;
        }
    }

    fn stage_slot(&self, slot: usize, seq: u64, input: &[u8]) -> Result<u64, CommError> {
        let rank = self.rank;
        let row = row_offset(rank);
        let mut state = self.cluster.inner.state.lock();
        if !state.alive[rank] {
            return Err(CommError::PeerClosed { peer: rank });
        }
        state.slot_last_seq[slot] = Some(seq);
        state.slot_seq[rank][slot] = seq + 1;
        state.doorbell[rank] = seq + 1;
        state.slots[rank][slot].bytes.as_mut()[row..row + input.len()].copy_from_slice(input);
        let address = state.slots[rank][slot].address();
        if !state.slot_busy[rank][slot] {
            state.slot_busy[rank][slot] = true;
            state.rank_in_flight[rank] += 1;
            state.max_rank_in_flight = state.max_rank_in_flight.max(state.rank_in_flight[rank]);
        }
        self.cluster
            .inner
            .doorbells
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cluster
            .inner
            .collectives
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cluster.notify();
        drop(state);
        Ok(address)
    }

    fn wait_doorbells(&self, slot: usize, seq: u64) -> Result<(), CommError> {
        let world = self.cluster.world_size();
        let deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
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
                deadline,
            )?;
        }
        for peer in 0..world {
            if peer != self.rank && state.slot_seq[peer][slot] != seq + 1 {
                return Err(CommError::Protocol(format!(
                    "rank {peer} staged sequence {} into slot {}, but this rank posts sequence {seq} into slot {slot}",
                    state.slot_seq[peer][slot] - 1,
                    peer
                )));
            }
        }
        Ok(())
    }

    fn send_frames(&self, slot: usize, exchange: Exchange) -> Result<(), CommError> {
        let rank = self.rank;
        let world = self.cluster.world_size();
        let rails = exchange.rails();
        let seq = exchange.seq();
        let row = row_offset(rank);
        let peers: Vec<usize> = (0..world).filter(|peer| *peer != rank).collect();
        let reverse = self.cluster.inner.config.reverse_arrivals;
        let peer_order: Vec<usize> = if reverse {
            peers.iter().rev().copied().collect()
        } else {
            peers.clone()
        };
        let mut state = self.cluster.inner.state.lock();
        for peer in &peer_order {
            let rail_order: Vec<usize> = if reverse {
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
                let source = state.slots[rank][slot].bytes.as_slice()
                    [row + offset..row + offset + len]
                    .to_vec();
                state.slots[*peer][slot].bytes.as_mut()[row + offset..row + offset + len]
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
            }
        }
        self.cluster.notify();
        Ok(())
    }

    fn take_frames(&self, exchange: Exchange) -> Result<Vec<RailFrame>, CommError> {
        let rank = self.rank;
        let expected = (self.cluster.world_size() - 1) * exchange.rails();
        let deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
        loop {
            if state.inbox[rank].len() >= expected {
                return Ok(state.inbox[rank].drain(0..expected).collect());
            }
            self.cluster.wait(
                &mut state,
                &format!("waiting for rail frames at sequence {}", exchange.seq()),
                deadline,
            )?;
        }
    }

    fn publish_arrival(&self, peer: usize, seq: u64) {
        let mut state = self.cluster.inner.state.lock();
        state.arrival[self.rank][peer] = seq + 1;
        self.cluster.notify();
    }

    fn wait_arrivals(&self, seq: u64) -> Result<(), CommError> {
        let rank = self.rank;
        let world = self.cluster.world_size();
        let deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
        loop {
            let missing: Vec<usize> = (0..world)
                .filter(|peer| *peer != rank && state.arrival[rank][*peer] < seq + 1)
                .collect();
            if missing.is_empty() {
                return Ok(());
            }
            self.cluster.wait(
                &mut state,
                &format!("waiting for arrival words from ranks {missing:?} at sequence {seq}"),
                deadline,
            )?;
        }
    }

    fn read_rows(&self, slot: usize, bytes: usize) -> Vec<Vec<u16>> {
        let rank = self.rank;
        let world = self.cluster.world_size();
        let state = self.cluster.inner.state.lock();
        (0..world)
            .map(|peer| {
                state.slots[rank][slot].bytes.as_slice()[row_offset(peer)..row_offset(peer) + bytes]
                    .chunks(2)
                    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                    .collect()
            })
            .collect()
    }

    fn wait_for_reuse_staged(&self, slot: usize, reuse_seq: u64) -> Result<(), CommError> {
        let rank = self.rank;
        let deadline = self.cluster.deadline();
        let mut state = self.cluster.inner.state.lock();
        loop {
            if state.slot_seq[rank][slot] > reuse_seq {
                return Ok(());
            }
            self.cluster.wait(
                &mut state,
                &format!(
                    "waiting for this rank to stage the premature reuse of slot {slot} at sequence {reuse_seq}"
                ),
                deadline,
            )?;
        }
    }

    fn publish_read_done(&self, slot: usize, seq: u64) {
        let rank = self.rank;
        let mut state = self.cluster.inner.state.lock();
        state.read_epoch[rank][slot] = seq + 1;
        if state.slot_busy[rank][slot] {
            state.slot_busy[rank][slot] = false;
            state.rank_in_flight[rank] -= 1;
        }
        self.cluster.notify();
    }

    fn slot_addresses(&self, slot: usize) -> (u64, [u64; ONESHOT_RANKS - 1]) {
        let rank = self.rank;
        let state = self.cluster.inner.state.lock();
        let address = state.slots[rank][slot].address();
        let peer_rows: Vec<usize> = (0..ONESHOT_RANKS).filter(|peer| *peer != rank).collect();
        let mut rows = [0u64; ONESHOT_RANKS - 1];
        for (index, peer) in peer_rows.iter().enumerate() {
            rows[index] = address + row_offset(*peer) as u64;
        }
        (address, rows)
    }
}
