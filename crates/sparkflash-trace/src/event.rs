//! Trace event records and the CPU/GPU ring buffers that hold them.
//!
//! Two rings exist:
//!
//! * A per-thread CPU ring, written by ScopedCpuSpan when its scope bit is set.
//! * A single in-graph GPU ring, whose ABI is shared with the sm_121a kernels through
//!   kernels/trace_globaltimer_sm121.cuh. Its first word (active_scope_mask) lives in
//!   coherent unified memory so the Rust admin path can arm GPU trace points inside an
//!   already-captured CUDA graph without recapturing anything.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use parking_lot::Mutex;

/// Default capacity of each per-thread CPU trace ring.
pub const DEFAULT_CPU_RING_CAPACITY: usize = 8192;

/// A completed CPU span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuTraceEvent {
    /// Static label of the instrumented region.
    pub label: &'static str,
    /// Scope bit that was active when the span was entered.
    pub scope: u64,
    /// Caller supplied metadata (row count, token count, layer id, ...).
    pub meta: u32,
    /// Monotonic thread id assigned by the registry.
    pub thread_id: u32,
    /// Span start, nanoseconds from the clock epoch (boot time on aarch64).
    pub start_ns: u64,
    /// Span end, nanoseconds from the same epoch as start_ns.
    pub end_ns: u64,
}

impl CpuTraceEvent {
    /// Span duration in nanoseconds (saturating).
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

/// Per-thread trace ring state.
#[derive(Debug)]
pub struct ThreadTraceState {
    thread_id: u32,
    capacity: usize,
    events: Mutex<Vec<CpuTraceEvent>>,
    dropped: AtomicU64,
    recorded: AtomicU64,
}

impl ThreadTraceState {
    fn new(thread_id: u32, capacity: usize) -> Self {
        Self {
            thread_id,
            capacity,
            events: Mutex::new(Vec::with_capacity(capacity.min(1024))),
            dropped: AtomicU64::new(0),
            recorded: AtomicU64::new(0),
        }
    }

    /// Monotonic thread id used as the Chrome trace `tid`.
    #[must_use]
    pub fn thread_id(&self) -> u32 {
        self.thread_id
    }

    /// Number of events recorded by this thread since the last clear.
    #[must_use]
    pub fn recorded(&self) -> u64 {
        self.recorded.load(Ordering::Relaxed)
    }

    /// Number of events dropped because the ring was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Appends an event, dropping it (and counting the loss) when the ring is full.
    pub fn record(&self, event: CpuTraceEvent) {
        let mut guard = self.events.lock();
        if guard.len() >= self.capacity {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        guard.push(event);
        self.recorded.fetch_add(1, Ordering::Relaxed);
    }

    /// Removes and returns this thread ring's events, decrementing the buffered count.
    pub fn take(&self) -> Vec<CpuTraceEvent> {
        let taken = std::mem::take(&mut *self.events.lock());
        self.recorded
            .fetch_sub(taken.len() as u64, Ordering::Relaxed);
        taken
    }

    /// Clones this thread ring's events.
    #[must_use]
    pub fn snapshot(&self) -> Vec<CpuTraceEvent> {
        self.events.lock().clone()
    }

    /// Clears the ring and the recorded counter.
    pub fn clear(&self) {
        self.events.lock().clear();
        self.recorded.store(0, Ordering::Relaxed);
    }
}

fn next_thread_id() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn registry() -> &'static Mutex<Vec<Arc<ThreadTraceState>>> {
    static REGISTRY: std::sync::OnceLock<Mutex<Vec<Arc<ThreadTraceState>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

thread_local! {
    static THREAD_STATE: Arc<ThreadTraceState> = {
        let state = Arc::new(ThreadTraceState::new(next_thread_id(), DEFAULT_CPU_RING_CAPACITY));
        registry().lock().push(Arc::clone(&state));
        state
    };
}

/// Returns the calling thread's trace ring, registering it on first use.
#[must_use]
pub fn current_thread_state() -> Arc<ThreadTraceState> {
    THREAD_STATE.with(Arc::clone)
}

/// Removes and returns every recorded CPU event from every registered thread,
/// sorted by start timestamp.
#[must_use]
pub fn drain_cpu_events() -> Vec<CpuTraceEvent> {
    let mut out = Vec::new();
    for state in registry().lock().iter() {
        out.extend(state.take());
    }
    out.sort_by_key(|e| (e.start_ns, e.thread_id));
    out
}

/// Clears every registered thread ring.
pub fn clear_cpu_events() {
    for state in registry().lock().iter() {
        state.clear();
    }
}

/// Aggregate CPU ring counters: (registered threads, recorded events, dropped events).
#[must_use]
pub fn cpu_ring_stats() -> (usize, u64, u64) {
    let guard = registry().lock();
    let recorded = guard.iter().map(|s| s.recorded()).sum();
    let dropped = guard.iter().map(|s| s.dropped()).sum();
    (guard.len(), recorded, dropped)
}

// ---------------------------------------------------------------------------------------
// In-graph GPU trace ring ABI (shared with kernels/trace_globaltimer_sm121.cuh)
// ---------------------------------------------------------------------------------------

/// Size in bytes of one GpuTraceEvent, matching the CUDA struct.
pub const GPU_TRACE_EVENT_BYTES: usize = 32;

/// Size in bytes of the GPU ring header, matching struct GpuTraceRing in the CUDA header.
///
/// The header is padded to 32 bytes even though only the first 16 carry data, because the
/// CUDA macro indexes events as `(char *)ring + sizeof(GpuTraceRing)` and a trace event is
/// declared `__align__(32)`: the event array must stay 32-byte aligned inside the ring, so
/// the header must be a whole 32-byte unit.
pub const GPU_RING_HEADER_BYTES: usize = 32;

/// Bytes of the GPU ring header that carry data; the rest is padding.
pub const GPU_RING_HEADER_USED_BYTES: usize = 16;

/// Alignment the CUDA header demands for a trace event: `struct __align__(32) GpuTraceEvent`.
pub const GPU_TRACE_EVENT_ALIGN: usize = 32;

/// A 32-byte in-graph GPU trace record. Layout matches GpuTraceEvent in the CUDA header.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuTraceEvent {
    /// %globaltimer nanoseconds at kernel entry.
    pub start_ns: u64,
    /// %globaltimer nanoseconds at kernel exit.
    pub end_ns: u64,
    /// In-kernel wait time (RDMA peer arrival, Engram stall, ...) in nanoseconds.
    pub sub_wait_ns: u32,
    /// (scope_bit_index << 8) | rank, where scope_bit_index is 1-based as produced by
    /// __ffsll(scope) in the CUDA macro.
    pub scope_and_rank: u32,
    /// Layer id the record belongs to.
    pub layer_id: u32,
    /// Payload metadata: bytes transferred or active rows M.
    pub payload_meta: u32,
}

impl GpuTraceEvent {
    /// Decodes the 1-based scope bit index back into a scope mask.
    ///
    /// Encoding matches the CUDA macro: (__ffsll(scope) << 8) | rank, so the bit index is
    /// 1-based and lives in the high byte.
    #[must_use]
    pub fn scope_mask(&self) -> u64 {
        let index = (self.scope_and_rank >> 8) & 0xFF;
        if index == 0 || index > 64 {
            0
        } else {
            1u64 << (index - 1)
        }
    }

    /// Decodes the rank embedded in the record.
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.scope_and_rank & 0xFF
    }

    /// Duration in nanoseconds (saturating).
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

/// Header of the coherent-UMA GPU trace ring. Allocated once per rank and mapped into the
/// GPU address space so the CUDA kernels can read the live scope mask.
#[repr(C, align(32))]
#[derive(Debug, Default)]
pub struct GpuTraceRingHeader {
    /// Live scope mask. Written by the Rust admin path, read by every instrumented kernel.
    pub active_scope_mask: AtomicU64,
    /// Monotonic event index; incremented by atomicAdd in the CUDA macro.
    pub write_idx: AtomicU32,
    /// Power-of-two minus one ring mask.
    pub capacity_mask: u32,
    /// Padding to 32 bytes so the event array stays 32-byte aligned. Must stay zero.
    pub reserved: [u64; 2],
}

impl GpuTraceRingHeader {
    /// Builds a header for a ring of `capacity` events. `capacity` must be a power of two.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "GPU trace ring capacity must be a power of two"
        );
        assert!(
            capacity <= MAX_RING_CAPACITY,
            "GPU trace ring capacity must fit the 32-bit header index"
        );
        Self {
            active_scope_mask: AtomicU64::new(0),
            write_idx: AtomicU32::new(0),
            capacity_mask: (capacity - 1) as u32,
            reserved: [0; 2],
        }
    }

    /// Ring capacity in events.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity_mask as usize + 1
    }

    /// Publishes a scope mask to the (possibly GPU-visible) header word.
    pub fn publish_mask(&self, mask: u64) {
        self.active_scope_mask.store(mask, Ordering::Release);
    }

    /// Reads the live scope mask.
    #[must_use]
    pub fn mask(&self) -> u64 {
        self.active_scope_mask.load(Ordering::Acquire)
    }

    /// Number of events written so far (including wrapped ones).
    #[must_use]
    pub fn written(&self) -> u32 {
        self.write_idx.load(Ordering::Acquire)
    }

    /// Total bytes required for a header plus `capacity` events.
    ///
    /// # Panics
    /// Panics for the same reasons as [GpuTraceRing::allocation_bytes]: "capacity" must be a power
    /// of two no larger than [MAX_RING_CAPACITY], and the byte count must fit in a "usize".
    #[must_use]
    pub fn layout_bytes(capacity: usize) -> usize {
        GpuTraceRing::allocation_bytes(capacity)
    }
}

/// Backing allocation for an owned ring.
struct AlignedAlloc {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedAlloc {
    fn zeroed(bytes: usize) -> Self {
        let size = bytes.max(GPU_TRACE_EVENT_ALIGN);
        let layout = Layout::from_size_align(size, GPU_TRACE_EVENT_ALIGN)
            .expect("ring layout is a valid power-of-two size");
        // SAFETY: size is non-zero because of the max() above.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, layout }
    }
}

impl Drop for AlignedAlloc {
    fn drop(&mut self) {
        // SAFETY: ptr came from the same layout via alloc_zeroed and is freed exactly once.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// A coherent-UMA GPU trace ring: one contiguous allocation with the 32-byte header
/// (GPU_RING_HEADER_BYTES = 32: 16 bytes of live words plus 16 of padding) at offset 0 and
/// "capacity" 32-byte event slots from offset 32, matching how the CUDA macro indexes it:
/// (GpuTraceEvent *)((char *)ring + sizeof(GpuTraceRing)) + idx. The padding is what keeps
/// every slot past the first 32-byte aligned.
///
/// Production code backs this with a cudaHostAllocMapped allocation through
/// GpuTraceRing::from_mapped_ptr; tests and the exporter use an owned allocation.
///
/// # Concurrency
/// The header words are atomics because the Grace CPU and the GPU share them live. The event
/// slots are plain memory written by one CUDA lane (block 0, thread 0, with the rank in
/// blockIdx.y) and read by the host only after the stream has completed, so every slot
/// accessor is unsafe: safe Rust cannot start a data race with an in-flight graph.
pub struct GpuTraceRing {
    ptr: NonNull<u8>,
    owned: Option<AlignedAlloc>,
    capacity: usize,
}

// SAFETY: the ring is a block of plain memory plus two atomic header words; a shared
// reference only exposes the atomics and the unsafe slot accessors, so safe code cannot race.
unsafe impl Send for GpuTraceRing {}
unsafe impl Sync for GpuTraceRing {}

impl std::fmt::Debug for GpuTraceRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTraceRing")
            .field("capacity", &self.capacity)
            .field("bytes", &self.bytes())
            .field("mapped", &self.owned.is_none())
            .finish()
    }
}

/// Largest ring the header can index: the 32-bit mask means capacity - 1 must fit in a u32.
pub const MAX_RING_CAPACITY: usize = 1 << 32;

impl GpuTraceRing {
    /// Allocates an owned ring of "capacity" events (a power of two).
    ///
    /// # Panics
    /// Panics when "capacity" is not a power of two.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let bytes = Self::allocation_bytes(capacity);
        let alloc = AlignedAlloc::zeroed(bytes);
        let ring = Self {
            ptr: alloc.ptr,
            owned: Some(alloc),
            capacity,
        };
        // SAFETY: the allocation is at least GPU_RING_HEADER_BYTES long and 32-byte aligned.
        unsafe { std::ptr::write(ring.header_ptr(), GpuTraceRingHeader::new(capacity)) };
        ring
    }

    /// Bytes of one owned ring holding "capacity" events: the header plus one 32-byte slot each.
    ///
    /// # Panics
    /// Panics when "capacity" is zero, is not a power of two, exceeds [MAX_RING_CAPACITY], or when
    /// the byte count would overflow "usize". The bound matters for soundness, not just tidiness:
    /// the header stores capacity - 1 in 32 bits, so an unchecked capacity would truncate the mask
    /// while the byte count wrapped, and the SAFE accessor [Self::slot] would then index outside
    /// the allocation.
    #[must_use]
    pub fn allocation_bytes(capacity: usize) -> usize {
        assert!(
            capacity.is_power_of_two(),
            "GPU trace ring capacity must be a power of two"
        );
        assert!(
            capacity <= MAX_RING_CAPACITY,
            "GPU trace ring capacity must fit the 32-bit header index"
        );
        capacity
            .checked_mul(GPU_TRACE_EVENT_BYTES)
            .and_then(|slots| slots.checked_add(GPU_RING_HEADER_BYTES))
            .expect("GPU trace ring allocation size must fit usize")
    }

    /// Wraps a cudaHostAllocMapped (or otherwise coherent) allocation produced by the caller.
    ///
    /// The returned ring does not free the memory.
    ///
    /// # Safety
    /// "ptr" must be 32-byte aligned, point to at least
    /// (GPU_RING_HEADER_BYTES + capacity * GPU_TRACE_EVENT_BYTES) bytes that stay valid for
    /// the ring's lifetime, be mapped into the GPU address space coherently, and not be
    /// aliased by another GpuTraceRing. "capacity" must be a power of two and match the
    /// capacity_mask the kernels expect.
    ///
    /// # Panics
    /// Panics when "capacity" is not a power of two.
    #[must_use]
    pub unsafe fn from_mapped_ptr(ptr: *mut u8, capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "GPU trace ring capacity must be a power of two"
        );
        let non_null = NonNull::new(ptr).expect("mapped ring pointer must be non-null");
        assert_eq!(
            non_null.as_ptr() as usize % GPU_TRACE_EVENT_ALIGN,
            0,
            "mapped ring pointer must be 32-byte aligned"
        );
        let ring = Self {
            ptr: non_null,
            owned: None,
            capacity,
        };
        // SAFETY: the caller guarantees the pointer is valid for a full header.
        unsafe { std::ptr::write(ring.header_ptr(), GpuTraceRingHeader::new(capacity)) };
        ring
    }

    fn header_ptr(&self) -> *mut GpuTraceRingHeader {
        self.ptr.as_ptr().cast::<GpuTraceRingHeader>()
    }

    /// Base pointer the kernels receive as their GpuTraceRing *.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Total bytes of the single allocation (header plus slots).
    #[must_use]
    pub fn bytes(&self) -> usize {
        GPU_RING_HEADER_BYTES + self.capacity * GPU_TRACE_EVENT_BYTES
    }

    /// Ring capacity in events.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// True when the ring owns its allocation rather than wrapping mapped memory.
    #[must_use]
    pub fn is_owned(&self) -> bool {
        self.owned.is_some()
    }

    /// Header access (the two words shared live with the device).
    #[must_use]
    pub fn header(&self) -> &GpuTraceRingHeader {
        // SAFETY: the constructor wrote a valid header at this address, which stays valid for
        // the ring's lifetime.
        unsafe { &*self.header_ptr() }
    }

    /// Pointer to slot "index", or None when out of range.
    #[must_use]
    pub fn slot_ptr(&self, index: usize) -> Option<*const GpuTraceEvent> {
        if index >= self.capacity {
            return None;
        }
        // SAFETY: the offset stays inside the allocation for index < capacity.
        Some(unsafe {
            self.ptr
                .as_ptr()
                .add(GPU_RING_HEADER_BYTES + index * GPU_TRACE_EVENT_BYTES)
                .cast::<GpuTraceEvent>()
        })
    }

    /// Reads one slot, or None when out of range.
    #[must_use]
    pub fn slot(&self, index: usize) -> Option<GpuTraceEvent> {
        let ptr = self.slot_ptr(index)?;
        // SAFETY: the device writes these words; volatile keeps the read honest.
        Some(unsafe { std::ptr::read_volatile(ptr) })
    }

    /// Appends an event exactly the way the CUDA macro does (atomic index, masked wrap).
    ///
    /// Returns the slot index that was written.
    ///
    /// # Safety
    /// The caller must guarantee no instrumented graph is concurrently writing the same slot,
    /// i.e. no capture is in flight on the device.
    pub unsafe fn push(&self, event: GpuTraceEvent) -> usize {
        let idx = self.header().write_idx.fetch_add(1, Ordering::AcqRel);
        let slot = (idx & self.header().capacity_mask) as usize;
        let ptr = self
            .slot_ptr(slot)
            .expect("masked slot index is always in range");
        // SAFETY: the caller upholds the no-concurrent-writer contract.
        unsafe { std::ptr::write_volatile(ptr.cast_mut(), event) };
        slot
    }

    /// Mirrors SPARK_GPU_TRACE_ENTER/EXIT: writes only while the scope is armed.
    ///
    /// # Safety
    /// Same contract as GpuTraceRing::push.
    pub unsafe fn push_if_armed(&self, scope: u64, event: GpuTraceEvent) -> bool {
        if self.header().mask() & scope == 0 {
            return false;
        }
        // SAFETY: the caller upholds the push contract.
        unsafe { self.push(event) };
        true
    }

    /// Returns every event written so far, in write order.
    ///
    /// # Safety
    /// The device must not be writing the ring concurrently.
    #[must_use]
    pub unsafe fn collect(&self) -> Vec<GpuTraceEvent> {
        let written = self.header().written() as usize;
        let start = written.saturating_sub(self.capacity);
        (start..written)
            .filter_map(|i| self.slot(i & self.header().capacity_mask as usize))
            .collect()
    }

    /// Resets the write index and zeroes every slot.
    ///
    /// # Safety
    /// The device must not be writing the ring concurrently.
    pub unsafe fn clear(&self) {
        self.header().write_idx.store(0, Ordering::Release);
        for index in 0..self.capacity {
            if let Some(ptr) = self.slot_ptr(index) {
                // SAFETY: the caller upholds the no-concurrent-writer contract.
                unsafe { std::ptr::write_volatile(ptr.cast_mut(), GpuTraceEvent::default()) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::TraceScope;

    #[test]
    fn gpu_abi_matches_cuda_header_layout() {
        assert_eq!(std::mem::size_of::<GpuTraceEvent>(), GPU_TRACE_EVENT_BYTES);
        assert_eq!(
            std::mem::size_of::<GpuTraceRingHeader>(),
            GPU_RING_HEADER_BYTES
        );
        // The CUDA struct is declared __align__(32); a mismatch would misalign every slot
        // after the first.
        assert_eq!(std::mem::align_of::<GpuTraceEvent>(), GPU_TRACE_EVENT_ALIGN);
        assert_eq!(
            std::mem::align_of::<GpuTraceRingHeader>(),
            GPU_TRACE_EVENT_ALIGN
        );
        assert_eq!(
            std::mem::offset_of!(GpuTraceRingHeader, active_scope_mask),
            0
        );
        assert_eq!(std::mem::offset_of!(GpuTraceRingHeader, write_idx), 8);
        assert_eq!(std::mem::offset_of!(GpuTraceRingHeader, capacity_mask), 12);
        assert_eq!(
            std::mem::offset_of!(GpuTraceRingHeader, reserved),
            GPU_RING_HEADER_USED_BYTES
        );
        assert_eq!(GpuTraceRingHeader::layout_bytes(1024), 32 + 1024 * 32);
        assert_eq!(GpuTraceRingHeader::new(1024).capacity(), 1024);
    }

    #[test]
    fn gpu_scope_and_rank_decode() {
        let event = GpuTraceEvent {
            scope_and_rank: (14u32 << 8) | 3,
            ..Default::default()
        };
        assert_eq!(event.rank(), 3);
        assert_eq!(event.scope_mask(), 1u64 << 13);
        assert_eq!(event.scope_mask(), TraceScope::ROCENANTE_RDMA);
        assert_eq!(GpuTraceEvent::default().scope_mask(), 0);
    }

    #[test]
    fn gpu_ring_wraps_like_the_cuda_macro() {
        let ring = GpuTraceRing::new(4);
        // SAFETY: nothing else touches this ring.
        unsafe {
            for i in 0..6u64 {
                ring.push(GpuTraceEvent {
                    start_ns: i,
                    end_ns: i + 1,
                    ..Default::default()
                });
            }
        }
        assert_eq!(ring.header().written(), 6);
        // SAFETY: nothing else touches this ring.
        let collected = unsafe { ring.collect() };
        assert_eq!(collected.len(), 4);
        assert_eq!(collected.first().unwrap().start_ns, 2);
        assert_eq!(collected.last().unwrap().start_ns, 5);
        // SAFETY: nothing else touches this ring.
        unsafe { ring.clear() };
        assert_eq!(ring.header().written(), 0);
        // SAFETY: nothing else touches this ring.
        assert!(unsafe { ring.collect() }.is_empty());
    }

    /// The CUDA macro computes the event array as (char*)ring + sizeof(GpuTraceRing): the
    /// slots must therefore follow the 32-byte header (GPU_RING_HEADER_BYTES, 16 live bytes
    /// plus padding) inside ONE allocation with 32-byte alignment, not live in a separate
    /// buffer.
    #[test]
    fn gpu_ring_slots_start_at_header_size_in_one_allocation() {
        let ring = GpuTraceRing::new(8);
        assert_eq!(
            ring.bytes(),
            GPU_RING_HEADER_BYTES + 8 * GPU_TRACE_EVENT_BYTES
        );
        let base = ring.as_ptr() as usize;
        assert_eq!(base % GPU_TRACE_EVENT_ALIGN, 0);
        let first = ring.slot_ptr(0).unwrap() as usize;
        let last = ring.slot_ptr(7).unwrap() as usize;
        assert_eq!(first - base, GPU_RING_HEADER_BYTES);
        assert_eq!(last - first, 7 * GPU_TRACE_EVENT_BYTES);
        assert_eq!(ring.slot_ptr(8), None);
        assert_eq!(ring.slot(8), None);
        assert_eq!(ring.capacity(), 8);
        assert!(ring.is_owned());
    }

    #[test]
    fn mapped_ring_wraps_caller_memory_without_freeing_it() {
        let buf = AlignedAlloc::zeroed(GPU_RING_HEADER_BYTES + 4 * GPU_TRACE_EVENT_BYTES);
        let raw = buf.ptr.as_ptr();
        // SAFETY: the buffer is large enough, 32-byte aligned and not aliased.
        let ring = unsafe { GpuTraceRing::from_mapped_ptr(raw, 4) };
        assert!(!ring.is_owned());
        assert_eq!(ring.capacity(), 4);
        assert_eq!(ring.header().capacity_mask, 3);
        assert_eq!(ring.header().written(), 0);
        assert_eq!(ring.as_ptr() as usize, raw as usize);
        // SAFETY: nothing else touches this ring.
        unsafe {
            ring.push(GpuTraceEvent {
                start_ns: 9,
                ..Default::default()
            })
        };
        assert_eq!(ring.slot(0).unwrap().start_ns, 9);
        // Dropping the ring must not release the caller's allocation.
        drop(ring);
        assert_eq!(buf.ptr.as_ptr(), raw);
    }

    #[test]
    fn push_if_armed_honours_the_scope_mask() {
        let _global_state = crate::test_support::lock();
        use crate::controller::arm_with_gpu;
        let ring = GpuTraceRing::new(4);
        let header = ring.header();
        assert_eq!(
            arm_with_gpu(header, TraceScope::ENGRAM_PREFETCH, 4),
            TraceScope::ENGRAM_PREFETCH
        );
        // SAFETY: nothing else touches this ring.
        unsafe {
            assert!(ring.push_if_armed(
                TraceScope::ENGRAM_PREFETCH,
                GpuTraceEvent {
                    scope_and_rank: (5u32 << 8) | 1,
                    ..Default::default()
                }
            ));
            assert!(!ring.push_if_armed(TraceScope::MOE_MXFP4, GpuTraceEvent::default()));
        }
        assert_eq!(ring.header().written(), 1);
        crate::controller::disarm_with_gpu(header);
    }

    #[test]
    fn cpu_ring_records_and_drains() {
        // Registers this thread in the process-global registry, which drain_cpu_events() walks in
        // full: without the crate-wide lock this test steals events from the span tests.
        let _global_state = crate::test_support::lock();
        let state = current_thread_state();
        state.clear();
        state.record(CpuTraceEvent {
            label: "unit_test_span",
            scope: TraceScope::ENGRAM_PREFETCH,
            meta: 7,
            thread_id: state.thread_id(),
            start_ns: 100,
            end_ns: 250,
        });
        assert_eq!(state.recorded(), 1);
        let drained = drain_cpu_events();
        assert!(
            drained
                .iter()
                .any(|e| e.label == "unit_test_span" && e.duration_ns() == 150)
        );
        assert_eq!(state.recorded(), 0);
    }

    #[test]
    #[should_panic(expected = "32-bit header index")]
    fn an_unaddressable_ring_capacity_is_rejected() {
        // In release an unchecked capacity * 32 wrapped: new(1 << 59) reported capacity 2^59 with
        // 32 bytes of storage, after which the safe slot() accessor indexed out of bounds.
        let _ = GpuTraceRing::new(1 << 59);
    }

    #[test]
    fn ring_allocation_bytes_are_bounded_and_checked() {
        assert_eq!(GpuTraceRing::allocation_bytes(4096), 32 + 4096 * 32);
        assert_eq!(GpuTraceRing::allocation_bytes(1 << 20), 32 + (1 << 25));
        assert_eq!(GpuTraceRing::allocation_bytes(MAX_RING_CAPACITY) % 32, 0);
    }

    #[test]
    fn cpu_ring_counts_drops_when_full() {
        let state = Arc::new(ThreadTraceState::new(99, 2));
        for i in 0..5 {
            state.record(CpuTraceEvent {
                label: "x",
                scope: 1,
                meta: i,
                thread_id: 99,
                start_ns: 0,
                end_ns: 1,
            });
        }
        assert_eq!(state.recorded(), 2);
        assert_eq!(state.dropped(), 3);
        assert_eq!(state.take().len(), 2);
        assert_eq!(state.recorded(), 0);
    }
}
