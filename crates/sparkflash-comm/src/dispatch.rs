//! Size-based collective dispatch: one-shot RDMA for decode-sized payloads, NCCL for large ones.
//!
//! The fabric decision is a payload-size decision, and it is the only one the forward loop should
//! have to make:
//!
//! * payloads at or below [`ROCENANTE_MAX_BYTES`] (2 MiB) go through the RoCEnante one-shot path,
//!   which is the latency path and the bit-identical path (canonical rank-order reduction, see
//!   [`crate::reduce`]). Every decode collective is in this class: the expert reduce is
//!   5,120 bf16 elements = 10,240 B ([`DECODE_HIDDEN_BYTES`]), and the spec's decode all-reduce
//!   range tops out at 360 KiB ([`DECODE_ONESHOT_MAX_BYTES`]).
//! * payloads above 2 MiB go through the patched bidirectional NCCL ring, which is the bandwidth
//!   path; only chunked-prefill collectives are in this class.
//!
//! This module is pure: no transport, no library, no clock. It exists so the threshold, the
//! decode classification and the unavailable-backend behaviour are one tested definition rather
//! than a literal repeated at every call site. No latency number is claimed here; the gate is
//! measured by `crates/sparkflash-comm/src/bin/ws-b-allreduce-bench.rs` and is labelled
//! not-yet-measured until that run completes on the four nodes.
//!
//! ## No silent switches
//!
//! [`DispatchPolicy::route_with`] takes what is actually available and returns
//! [`DispatchError`] when the selected backend is not. A large collective on a host without
//! libnccl refuses with a message that names the ticket, instead of quietly running the one-shot
//! path (which caps at 2 MiB and would be the wrong transport anyway). This mirrors the crate's
//! existing rule: a transport refusal is reported, never papered over.

use crate::ROCENANTE_MAX_BYTES;
use crate::error::CommError;

/// bf16 elements in the decode hidden row: 5,120 (`hidden_size` of the deployed model).
pub const DECODE_HIDDEN_ELEMENTS: usize = 5_120;

/// Bytes of one decode hidden row in bf16: 5,120 elements x 2 B = 10,240 B, the gate payload.
pub const DECODE_HIDDEN_BYTES: usize = DECODE_HIDDEN_ELEMENTS * 2;

/// Upper edge of the specification's decode all-reduce payload range (10 KiB - 360 KiB).
///
/// This is a classification boundary, not a routing boundary: payloads between this and
/// [`ROCENANTE_MAX_BYTES`] are still one-shot, they are just no longer decode-sized.
pub const DECODE_ONESHOT_MAX_BYTES: usize = 360 * 1024;

/// Which transport a collective of a given size belongs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectiveBackend {
    /// RoCEnante one-shot RDMA: latency path and bit-identical canonical reduction.
    OneShotRdma,
    /// Patched bidirectional NCCL ring: bandwidth path for payloads above the one-shot cap.
    NcclRing,
}

impl CollectiveBackend {
    /// Stable label used in telemetry, evidence ledgers and benchmark rows.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneShotRdma => "oneshot_rdma",
            Self::NcclRing => "nccl_ring",
        }
    }

    /// Whether this backend carries the bit-identical canonical reduction contract.
    ///
    /// The one-shot path does; the NCCL ring is implementation- and topology-dependent and is not
    /// covered by that contract (see docs/build/ws-b-collectives-20261021.md).
    #[must_use]
    pub const fn is_bit_identical(self) -> bool {
        matches!(self, Self::OneShotRdma)
    }
}

/// A size-routing policy with one explicit threshold.
///
/// The default is the specification's 2 MiB one-shot slot. A non-default threshold exists for
/// tests and for a future fabric whose slot size changes; production code should use
/// [`DispatchPolicy::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPolicy {
    oneshot_max_bytes: usize,
}

impl Default for DispatchPolicy {
    fn default() -> Self {
        Self {
            oneshot_max_bytes: ROCENANTE_MAX_BYTES,
        }
    }
}

impl DispatchPolicy {
    /// Builds a policy with an explicit one-shot payload cap.
    ///
    /// # Errors
    /// Returns [`CommError::InvalidLayout`] for a zero cap: no collective could ever use the
    /// one-shot path, which is a configuration mistake rather than a policy.
    pub fn new(oneshot_max_bytes: usize) -> Result<Self, CommError> {
        if oneshot_max_bytes == 0 {
            return Err(CommError::InvalidLayout(
                "the one-shot dispatch cap must be at least one byte".to_owned(),
            ));
        }
        Ok(Self { oneshot_max_bytes })
    }

    /// The largest payload routed to the one-shot path, in bytes.
    #[must_use]
    pub const fn one_shot_max_bytes(&self) -> usize {
        self.oneshot_max_bytes
    }

    /// Routes a payload by size alone.
    ///
    /// At most `one_shot_max_bytes` is one-shot, anything larger is the NCCL ring. The comparison
    /// is inclusive at the boundary, matching the existing [crate::DualPathCommunicator] rule and
    /// `tests/routing.rs`.
    #[must_use]
    pub const fn route(&self, num_bytes: usize) -> CollectiveBackend {
        if num_bytes <= self.oneshot_max_bytes {
            CollectiveBackend::OneShotRdma
        } else {
            CollectiveBackend::NcclRing
        }
    }

    /// Routes a payload and checks that the selected backend is actually available.
    ///
    /// The NCCL availability flag only matters for payloads above the one-shot cap: a decode
    /// collective never needs NCCL, so a host without it can still serve decode.
    ///
    /// # Errors
    /// Returns [`DispatchError::OneShotUnavailable`] or [`DispatchError::NcclUnavailable`] when
    /// the selected backend is missing; the caller must then choose a documented fallback (host
    /// staging or chunking) rather than have this module switch silently.
    pub fn route_with(
        &self,
        num_bytes: usize,
        backends: BackendAvailability,
    ) -> Result<CollectiveBackend, DispatchError> {
        match self.route(num_bytes) {
            CollectiveBackend::OneShotRdma if !backends.oneshot_rdma => {
                Err(DispatchError::OneShotUnavailable { num_bytes })
            }
            CollectiveBackend::NcclRing if !backends.nccl => Err(DispatchError::NcclUnavailable {
                num_bytes,
                oneshot_max_bytes: self.oneshot_max_bytes,
            }),
            backend => Ok(backend),
        }
    }

    /// Whether a payload is in the spec's decode all-reduce class (10 KiB - 360 KiB).
    ///
    /// Decode is the latency-critical class the 25 us p50 gate is written for; the benchmark
    /// report gates exactly these rows and reports the larger one-shot rows beside them.
    #[must_use]
    pub const fn is_decode_payload(num_bytes: usize) -> bool {
        num_bytes > 0 && num_bytes <= DECODE_ONESHOT_MAX_BYTES
    }
}

/// Which backends a host can actually use.
///
/// `oneshot_rdma` means the RDMA data-plane is usable (the `rdma` feature is built and the
/// libibverbs bring-up succeeds); `nccl` means a loadable NCCL 2.28+ (`libnccl.so.2`) is present.
/// The host-staging TCP transport is deliberately not a flag: it is the correctness carrier, not
/// a drop-in substitute for either fabric backend, and choosing it must be explicit at the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendAvailability {
    /// The RoCEnante one-shot RDMA data-plane is usable.
    pub oneshot_rdma: bool,
    /// A loadable NCCL 2.28+ library is present.
    pub nccl: bool,
}

impl BackendAvailability {
    /// Both fabric backends are usable.
    pub const ALL: Self = Self {
        oneshot_rdma: true,
        nccl: true,
    };
    /// Neither fabric backend is usable (host staging only).
    pub const NONE: Self = Self {
        oneshot_rdma: false,
        nccl: false,
    };

    /// Builds an availability record from two flags.
    #[must_use]
    pub const fn new(oneshot_rdma: bool, nccl: bool) -> Self {
        Self { oneshot_rdma, nccl }
    }
}

/// Why the size-routed backend cannot run on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// The payload needs the one-shot RDMA path, which this host cannot open.
    OneShotUnavailable {
        /// Payload size that was routed.
        num_bytes: usize,
    },
    /// The payload is above the one-shot cap, so it needs NCCL, which this host cannot load.
    NcclUnavailable {
        /// Payload size that was routed.
        num_bytes: usize,
        /// The one-shot cap the payload exceeded.
        oneshot_max_bytes: usize,
    },
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OneShotUnavailable { num_bytes } => write!(
                formatter,
                "the one-shot RDMA path is unavailable, so the {num_bytes} B collective cannot run on it; this is the 2 MiB latency path and the fabric prerequisites are tracked in the WS-B infra tickets. Refusing instead of silently switching to a different transport"
            ),
            Self::NcclUnavailable {
                num_bytes,
                oneshot_max_bytes,
            } => write!(
                formatter,
                "the {num_bytes} B collective needs the NCCL ring because it is above the {oneshot_max_bytes} B one-shot cap, but libnccl.so.2 is not loadable on this host. NCCL exists only inside the vllm-node image (host install is WS-B infra ticket B5); chunk the payload to at most {oneshot_max_bytes} B per collective or wire the device-buffer NCCL comm. Refusing instead of silently routing a large collective to the one-shot path"
            ),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<DispatchError> for CommError {
    fn from(error: DispatchError) -> Self {
        Self::Unsupported(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_hidden_row_is_the_gate_payload() {
        assert_eq!(DECODE_HIDDEN_ELEMENTS, 5_120);
        assert_eq!(DECODE_HIDDEN_BYTES, 10_240);
        assert_eq!(DECODE_HIDDEN_BYTES, 5_120 * size_of::<u16>());
        assert!(
            DispatchPolicy::is_decode_payload(DECODE_HIDDEN_BYTES),
            "the decode hidden row must be in the decode class"
        );
        assert!(DispatchPolicy::is_decode_payload(60 * 1024));
        assert!(DispatchPolicy::is_decode_payload(DECODE_ONESHOT_MAX_BYTES));
        assert!(!DispatchPolicy::is_decode_payload(
            DECODE_ONESHOT_MAX_BYTES + 2
        ));
        assert!(
            !DispatchPolicy::is_decode_payload(0),
            "an empty payload is not a decode collective"
        );
    }

    #[test]
    fn the_default_threshold_is_the_spec_two_mib_slot() {
        let policy = DispatchPolicy::default();
        assert_eq!(policy.one_shot_max_bytes(), ROCENANTE_MAX_BYTES);
        assert_eq!(policy.one_shot_max_bytes(), 2 * 1024 * 1024);
        let cases = [
            (2usize, CollectiveBackend::OneShotRdma),
            (DECODE_HIDDEN_BYTES, CollectiveBackend::OneShotRdma),
            (60 * 1024, CollectiveBackend::OneShotRdma),
            (DECODE_ONESHOT_MAX_BYTES, CollectiveBackend::OneShotRdma),
            (ROCENANTE_MAX_BYTES - 1, CollectiveBackend::OneShotRdma),
            (ROCENANTE_MAX_BYTES, CollectiveBackend::OneShotRdma),
            (ROCENANTE_MAX_BYTES + 2, CollectiveBackend::NcclRing),
            (2 * ROCENANTE_MAX_BYTES, CollectiveBackend::NcclRing),
        ];
        for (bytes, expected) in cases {
            assert_eq!(policy.route(bytes), expected, "{bytes} B routed wrong");
        }
        assert_eq!(CollectiveBackend::OneShotRdma.as_str(), "oneshot_rdma");
        assert_eq!(CollectiveBackend::NcclRing.as_str(), "nccl_ring");
        assert!(CollectiveBackend::OneShotRdma.is_bit_identical());
        assert!(!CollectiveBackend::NcclRing.is_bit_identical());
    }

    #[test]
    fn a_custom_threshold_only_moves_the_boundary() {
        let policy = DispatchPolicy::new(4_096).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(policy.one_shot_max_bytes(), 4_096);
        assert_eq!(policy.route(4_096), CollectiveBackend::OneShotRdma);
        assert_eq!(policy.route(4_098), CollectiveBackend::NcclRing);
        assert!(matches!(
            DispatchPolicy::new(0),
            Err(CommError::InvalidLayout(_))
        ));
    }

    #[test]
    fn a_missing_nccl_only_refuses_the_large_class() {
        let policy = DispatchPolicy::default();
        let host_only = BackendAvailability::new(true, false);
        assert_eq!(
            policy
                .route_with(DECODE_HIDDEN_BYTES, host_only)
                .unwrap_or_else(|error| panic!("{error}")),
            CollectiveBackend::OneShotRdma,
            "decode must not need NCCL"
        );
        match policy.route_with(ROCENANTE_MAX_BYTES + 2, host_only) {
            Err(DispatchError::NcclUnavailable {
                num_bytes,
                oneshot_max_bytes,
            }) => {
                assert_eq!(num_bytes, ROCENANTE_MAX_BYTES + 2);
                assert_eq!(oneshot_max_bytes, ROCENANTE_MAX_BYTES);
            }
            other => panic!("expected NcclUnavailable, got {other:?}"),
        }
        let text = DispatchError::NcclUnavailable {
            num_bytes: ROCENANTE_MAX_BYTES + 2,
            oneshot_max_bytes: ROCENANTE_MAX_BYTES,
        }
        .to_string();
        assert!(text.contains("libnccl.so.2"), "{text}");
        assert!(text.contains("B5"), "{text}");
        assert!(text.contains("Refusing"), "{text}");
    }

    #[test]
    fn a_missing_rdma_refuses_the_small_class_with_a_named_error() {
        let policy = DispatchPolicy::default();
        let no_fabric = BackendAvailability::NONE;
        match policy.route_with(DECODE_HIDDEN_BYTES, no_fabric) {
            Err(DispatchError::OneShotUnavailable { num_bytes }) => {
                assert_eq!(num_bytes, DECODE_HIDDEN_BYTES);
            }
            other => panic!("expected OneShotUnavailable, got {other:?}"),
        }
        let text = DispatchError::OneShotUnavailable {
            num_bytes: DECODE_HIDDEN_BYTES,
        }
        .to_string();
        assert!(text.contains("10240 B"), "{text}");
        let comm: CommError = DispatchError::OneShotUnavailable {
            num_bytes: DECODE_HIDDEN_BYTES,
        }
        .into();
        assert!(matches!(comm, CommError::Unsupported(_)));
    }

    #[test]
    fn both_backends_available_routes_without_error() {
        let policy = DispatchPolicy::default();
        for (bytes, expected) in [
            (DECODE_HIDDEN_BYTES, CollectiveBackend::OneShotRdma),
            (ROCENANTE_MAX_BYTES + 2, CollectiveBackend::NcclRing),
        ] {
            assert_eq!(
                policy
                    .route_with(bytes, BackendAvailability::ALL)
                    .unwrap_or_else(|error| panic!("{error}")),
                expected
            );
        }
        assert_eq!(
            BackendAvailability::new(true, false),
            BackendAvailability {
                oneshot_rdma: true,
                nccl: false
            }
        );
    }
}
