//! The error type shared by every part of the paged KV cache.

use std::path::PathBuf;

use crate::layout::PageId;

/// Everything that can go wrong in the paged KV cache.
///
/// The library never panics on a recoverable condition: an I/O failure, a short page, a
/// misaligned `O_DIRECT` buffer, an exhausted pool or a page that is not where its tier says
/// it is all surface as one of these variants.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    /// An I/O operation on a tier backend failed. The path identifies the backend.
    #[error("kv tier I/O at {path}: {source}")]
    Io {
        /// File the failed operation targeted.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// A tier backend was asked for a page it does not hold.
    #[error("page {0:?} is not resident in the requested tier")]
    PageNotResident(PageId),
    /// A page was inserted twice, which would silently alias two payloads.
    #[error("page {0:?} is already resident")]
    DuplicatePage(PageId),
    /// A payload had the wrong length for the pinned layout.
    #[error("{context}: expected {expected} bytes, got {actual}")]
    InvalidLength {
        /// What was being parsed or written.
        context: &'static str,
        /// Length the layout requires.
        expected: usize,
        /// Length actually supplied.
        actual: usize,
    },
    /// An `O_DIRECT` buffer, offset or length did not meet the device alignment.
    #[error("alignment violation: {value} is not a multiple of the device alignment {alignment}")]
    Misaligned {
        /// The offending value.
        value: u64,
        /// Required alignment, discovered from the device.
        alignment: u64,
    },
    /// The UMA pool, the DRAM staging tier or the NVMe tier cannot hold the request.
    #[error("kv capacity exceeded: {requested} pages requested, {available} available")]
    CapacityExceeded {
        /// Pages the caller wanted.
        requested: u64,
        /// Pages that could be made available.
        available: u64,
    },
    /// A node id was used after the node had been evicted.
    #[error("radix node {0:?} does not exist")]
    UnknownNode(crate::radix::NodeId),
    /// A sequence was released more times than it was attached.
    #[error("radix node {0:?} reference count underflow")]
    RefCountUnderflow(crate::radix::NodeId),
    /// The configured memory budget cannot be met.
    #[error("memory budget exceeded: {required} bytes required, {budget} bytes budget")]
    BudgetExceeded {
        /// Bytes the requested configuration needs.
        required: u64,
        /// Bytes available after the floor guard.
        budget: u64,
    },
    /// The host or the build does not provide a requested capability.
    #[error("unsupported on this host: {0}")]
    Unsupported(&'static str),
    /// A requested SWA tail has been evicted from the ring.
    #[error("swa tail {0:?} is not resident")]
    TailNotResident(crate::swa::TailId),
    /// A persistent tier index could not be trusted.
    #[error("corrupt kv cache metadata: {0}")]
    Corrupt(String),
    /// A global-KV page was published while a bounded replay was in scope on this thread.
    ///
    /// A replay regenerates SWA KV only; reaching the global-KV publish path would mean the
    /// replay could overwrite cached global KV, which the specification forbids outright. The
    /// store refuses the publish and counts it in
    /// [KvStoreTelemetry::replay_global_kv_writes](crate::telemetry::KvStoreTelemetry::replay_global_kv_writes).
    #[error("global KV page {0:?} was published from inside a bounded replay")]
    ReplayWroteGlobalKv(PageId),
}
