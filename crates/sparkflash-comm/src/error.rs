//! Error type shared by every collective transport.

use std::io;
use std::time::Duration;

use thiserror::Error;

/// Failure returned by a [`Transport`](crate::Transport) or by the RoCEnante dispatcher.
#[derive(Debug, Error)]
pub enum CommError {
    /// The requested collective layout cannot describe a legal collective.
    #[error("invalid collective layout: {0}")]
    InvalidLayout(String),
    /// A rank outside `0..world_size` was requested.
    #[error("rank {rank} is out of range for world_size {world_size}")]
    RankOutOfRange {
        /// The offending rank.
        rank: usize,
        /// The size of the collective the rank was used with.
        world_size: usize,
    },
    /// A peer rank is gone: its socket, thread or shared slot was closed.
    #[error("peer rank {peer} is no longer reachable")]
    PeerClosed {
        /// The rank that can no longer be reached.
        peer: usize,
    },
    /// The collective did not complete inside the transport timeout.
    #[error("collective timed out after {timeout_ms} ms while {what}")]
    Timeout {
        /// The configured timeout, in milliseconds.
        timeout_ms: u64,
        /// What the transport was waiting for when the deadline expired.
        what: String,
    },
    /// A frame on the wire violated the RoCEnante framing contract.
    #[error("protocol violation: {0}")]
    Protocol(String),
    /// Underlying socket or file error.
    #[error("transport I/O error: {0}")]
    Io(#[from] io::Error),
    /// The requested backend is not available on this host.
    #[error("collective backend unavailable: {0}")]
    Unsupported(String),
    /// The runtime libibverbs layer refused a call.
    ///
    /// This is the error every ibverbs entry point maps onto: the call's name, and the errno or
    /// ABI reason the library gave. It is separate from
    /// [`Unsupported`](Self::Unsupported), which means the library itself could not be loaded, and
    /// from [`Protocol`](Self::Protocol), which means a peer sent something this backend did not expect.
    #[error("ibverbs {call} failed: {detail}")]
    Verbs {
        /// The libibverbs entry point, for example "ibv_post_send".
        call: &'static str,
        /// What the library reported: errno text, or the ABI mismatch.
        detail: String,
    },
    /// The dual-rail RoCE fabric is not usable, or was pinned to devices that are not.
    #[error("dual-rail fabric unusable: {0}")]
    FabricUnavailable(String),
}

impl CommError {
    /// Builds the error for an expected frame that never arrived in time.
    pub(crate) fn timeout(timeout: Duration, what: impl Into<String>) -> Self {
        Self::Timeout {
            timeout_ms: duration_millis(timeout),
            what: what.into(),
        }
    }

    /// Builds the error for a libibverbs call that returned a non-zero code.
    ///
    /// Only the rdma feature's backend reports this variant, so the constructor is compiled with it.
    #[cfg(feature = "rdma")]
    pub(crate) fn verbs(call: &'static str, detail: impl Into<String>) -> Self {
        Self::Verbs {
            call,
            detail: detail.into(),
        }
    }

    /// Maps a socket error observed while talking to `peer` onto the transport vocabulary.
    ///
    /// A reset or a clean EOF means the peer is gone; a would-block or timed-out read means the
    /// collective missed its deadline. Everything else keeps its `Io` shape so the caller can
    /// still see the OS error.
    pub(crate) fn from_io_peer(
        peer: usize,
        timeout: Duration,
        what: &str,
        error: io::Error,
    ) -> Self {
        match error.kind() {
            io::ErrorKind::UnexpectedEof
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::NotConnected => Self::PeerClosed { peer },
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                Self::timeout(timeout, format!("{what} (peer {peer})"))
            }
            _ => Self::Io(error),
        }
    }
}

/// Milliseconds in `duration`, saturating instead of wrapping on absurd values.
pub(crate) fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
