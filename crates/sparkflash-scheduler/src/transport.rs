//! The collective seam the scheduler is written against.
//!
//! Spec 5.6.1 sends the compact INT32 Top-512 indices and the 2048 candidate block IDs
//! through the RoCEnante one-shot all-gather. `sparkflash-comm` owns that transport, but on
//! this branch it exposes only its arena geometry (`ROCENANTE_MAX_BYTES`); the trait below is
//! therefore declared here, in the crate that decides *what* to gather, and the RoCEnante
//! implementation plugs in when Milestone 3 lands. Tests and the CLI use
//! `LoopbackTransport`, which records shapes without moving bytes off the machine.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything a collective can refuse.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// The payload does not fit one one-shot arena slot.
    #[error("payload of {bytes} bytes needs {slots} of the {slot_bytes}-byte one-shot arena slots")]
    PayloadTooLarge {
        /// Bytes the caller asked to gather.
        bytes: usize,
        /// Arena slot size.
        slot_bytes: usize,
        /// Slots the payload would need.
        slots: usize,
    },
    /// The rank is outside the transport's world.
    #[error("rank {rank} is outside a {world_size}-rank transport")]
    UnknownRank {
        /// Rank the caller claimed.
        rank: usize,
        /// Ranks in the transport.
        world_size: usize,
    },
    /// The declared shape and the payload disagree.
    #[error("declared shape is {declared} bytes but the payload is {actual} bytes")]
    ShapeMismatch {
        /// Bytes the shape declares.
        declared: usize,
        /// Bytes actually handed over.
        actual: usize,
    },
    /// Not every rank arrived.
    #[error("transport rendezvous failed: {0}")]
    Rendezvous(String),
}

/// The shape of one collective payload: how a gather is described before it happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherShape {
    /// Rows the payload carries.
    pub rows: usize,
    /// Bytes one row carries.
    pub bytes_per_row: usize,
}

impl GatherShape {
    /// A shape with the given geometry.
    #[must_use]
    pub const fn new(rows: usize, bytes_per_row: usize) -> Self {
        Self {
            rows,
            bytes_per_row,
        }
    }

    /// Total payload bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.rows * self.bytes_per_row
    }

    /// One-shot slots this payload needs.
    #[must_use]
    pub const fn slots(&self, slot_bytes: usize) -> usize {
        self.total_bytes().div_ceil(slot_bytes)
    }
}

/// A collective transport.
///
/// The scheduler only ever calls `all_gather`; every decision a test checks (the payload
/// shape, the slot count, whether the draft path touches the transport at all) is therefore
/// observable from an implementation of this trait.
pub trait CollectiveTransport: Send + Sync + std::fmt::Debug {
    /// Ranks in the transport.
    fn world_size(&self) -> usize;
    /// This transport's rank.
    fn rank(&self) -> usize;
    /// Bytes one one-shot slot carries.
    fn slot_bytes(&self) -> usize;
    /// All-gathers one payload, returning every rank's bytes in rank order.
    ///
    /// # Errors
    /// Returns `TransportError` when the payload exceeds one one-shot slot, when the shape
    /// and the payload disagree, or when the rank is unknown.
    fn all_gather(
        &self,
        shape: GatherShape,
        payload: &[u8],
    ) -> Result<Vec<Vec<u8>>, TransportError>;
}

/// A single-process loopback: it records what was gathered and hands back the caller's own
/// payload plus zero-filled peer payloads of the same length.
///
/// A loopback cannot invent remote bytes, and pretending otherwise would make a test lie
/// about what it proved, so the peer payloads are explicitly zero-filled and only the
/// *shapes* are assertions in the test suite.
#[derive(Debug)]
pub struct LoopbackTransport {
    world_size: usize,
    rank: usize,
    slot_bytes: usize,
    calls: AtomicUsize,
    shapes: Mutex<Vec<GatherShape>>,
}

impl LoopbackTransport {
    /// A loopback for one rank.
    ///
    /// # Panics
    /// Panics when `rank >= world_size` or the world is empty, because the caller is
    /// building a fixture and wants to hear about it immediately.
    #[must_use]
    pub fn new(world_size: usize, rank: usize, slot_bytes: usize) -> Self {
        assert!(world_size > 0, "a loopback needs at least one rank");
        assert!(
            rank < world_size,
            "rank {rank} is outside a {world_size}-rank world"
        );
        Self {
            world_size,
            rank,
            slot_bytes,
            calls: AtomicUsize::new(0),
            shapes: Mutex::new(Vec::new()),
        }
    }

    /// How many gathers this rank performed. The sync-free test asserts this stays zero.
    #[must_use]
    pub fn gather_calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The shapes gathered so far, in order.
    #[must_use]
    pub fn recorded_shapes(&self) -> Vec<GatherShape> {
        self.shapes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Total bytes gathered so far.
    #[must_use]
    pub fn gathered_bytes(&self) -> usize {
        self.recorded_shapes()
            .iter()
            .map(GatherShape::total_bytes)
            .sum()
    }
}

impl CollectiveTransport for LoopbackTransport {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    fn all_gather(
        &self,
        shape: GatherShape,
        payload: &[u8],
    ) -> Result<Vec<Vec<u8>>, TransportError> {
        if payload.len() != shape.total_bytes() {
            return Err(TransportError::ShapeMismatch {
                declared: shape.total_bytes(),
                actual: payload.len(),
            });
        }
        let slots = shape.slots(self.slot_bytes);
        if slots > 1 {
            return Err(TransportError::PayloadTooLarge {
                bytes: shape.total_bytes(),
                slot_bytes: self.slot_bytes,
                slots,
            });
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.shapes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(shape);
        let mut gathered = Vec::with_capacity(self.world_size);
        for peer in 0..self.world_size {
            if peer == self.rank {
                gathered.push(payload.to_vec());
            } else {
                gathered.push(vec![0u8; shape.total_bytes()]);
            }
        }
        Ok(gathered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_payload_larger_than_one_slot_is_refused_at_the_boundary() {
        let transport = LoopbackTransport::new(4, 0, 2 * 1024 * 1024);
        let exact = GatherShape::new(2048, 1024);
        assert_eq!(exact.total_bytes(), 2 * 1024 * 1024);
        assert_eq!(exact.slots(transport.slot_bytes()), 1);
        let payload = vec![7u8; exact.total_bytes()];
        let gathered = transport
            .all_gather(exact, &payload)
            .expect("exactly one slot fits");
        assert_eq!(gathered.len(), 4);
        assert_eq!(transport.gather_calls(), 1);

        let too_big = GatherShape::new(2048, 1024 + 1);
        assert_eq!(too_big.slots(transport.slot_bytes()), 2);
        let error = transport
            .all_gather(too_big, &vec![0u8; too_big.total_bytes()])
            .expect_err("a two-slot payload must be refused by the one-shot path");
        assert!(matches!(
            error,
            TransportError::PayloadTooLarge { slots: 2, .. }
        ));
        assert_eq!(
            transport.gather_calls(),
            1,
            "a refused gather is not a call"
        );
    }

    #[test]
    fn the_loopback_returns_every_rank_and_records_the_shape() {
        let transport = LoopbackTransport::new(4, 2, 2 * 1024 * 1024);
        let shape = GatherShape::new(256, 2056);
        let payload = vec![3u8; shape.total_bytes()];
        let gathered = transport.all_gather(shape, &payload).expect("one slot");
        assert_eq!(gathered.len(), 4);
        assert_eq!(
            gathered[2], payload,
            "the caller's own rank carries real bytes"
        );
        assert!(gathered[0].iter().all(|byte| *byte == 0));
        assert_eq!(transport.recorded_shapes(), vec![shape]);
        assert_eq!(transport.gathered_bytes(), shape.total_bytes());
        assert_eq!(transport.world_size(), 4);
        assert_eq!(transport.rank(), 2);
    }

    #[test]
    fn a_shape_that_disagrees_with_the_payload_is_rejected() {
        let transport = LoopbackTransport::new(2, 0, 2 * 1024 * 1024);
        let error = transport
            .all_gather(GatherShape::new(10, 4), &[0u8; 39])
            .expect_err("39 bytes is not 10 rows of 4");
        assert!(matches!(
            error,
            TransportError::ShapeMismatch {
                declared: 40,
                actual: 39
            }
        ));
    }

    #[test]
    fn slots_round_up_and_a_tiny_payload_still_needs_one() {
        assert_eq!(GatherShape::new(0, 8).slots(16), 0);
        assert_eq!(GatherShape::new(1, 8).slots(16), 1);
        assert_eq!(GatherShape::new(2, 8).slots(16), 1);
        assert_eq!(GatherShape::new(3, 8).slots(16), 2);
    }

    #[test]
    #[should_panic(expected = "rank 4 is outside")]
    fn an_unknown_rank_cannot_be_built() {
        let _ = LoopbackTransport::new(4, 4, 1024);
    }
}
