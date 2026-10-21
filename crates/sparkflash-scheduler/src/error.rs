//! The error type every scheduler plan returns.

/// Everything the scheduler can refuse.
///
/// The variants are deliberately decision-shaped: each one names a plan that was rejected
/// before any work was scheduled, because this crate decides and does not execute.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// A configuration value is outside the range the specification allows.
    #[error("invalid scheduler configuration: {0}")]
    InvalidConfig(String),
    /// A plan argument does not match the configuration.
    #[error("invalid plan argument: {0}")]
    InvalidArgument(String),
    /// A plan needs more memory than the static scratchpad it must live in.
    #[error("plan needs {needed} bytes but the static scratchpad is {capacity} bytes")]
    ScratchpadExhausted {
        /// Bytes the plan asked for.
        needed: usize,
        /// Bytes the static scratchpad has.
        capacity: usize,
    },
    /// The selected collective transport cannot carry a payload.
    #[error(transparent)]
    Transport(#[from] crate::transport::TransportError),
    /// The engine was driven out of order.
    #[error("engine step {requested} is not the next step ({next})")]
    StepOutOfOrder {
        /// Step the caller asked for.
        requested: u64,
        /// Step the engine was waiting for.
        next: u64,
    },
    /// The paged KV cache refused an operation the plan expected to be legal.
    #[error(transparent)]
    Paging(#[from] sparkflash_kv::PagingError),
    /// The batcher's own bookkeeping no longer holds: an internal inconsistency.
    #[error("batcher invariant: {0}")]
    Invariant(String),
}
