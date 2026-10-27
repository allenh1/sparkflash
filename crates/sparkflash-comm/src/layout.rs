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

//! Decomposition of one collective into peer transfers.
//!
//! Two schedules produce the same bits on purpose:
//!
//! * FabricSchedule::OneShotAllToAll - the switched-200G schedule. Every rank posts its whole
//!   payload to all three peers and each rank reduces locally in canonical rank order.
//! * FabricSchedule::HairpinRingReduceScatterAllGather - the switchless schedule. The buffer is
//!   split into one chunk per rank; every rank sends its contribution of chunk c to the owner of
//!   c (its ring neighbour or, for the opposite rank, straight across the ConnectX-7 hairpin
//!   diagonal), the owner accumulates ranks 0 + 1 + 2 + 3 in order and rounds once, then
//!   all-gathers the reduced chunk back. This moves 2 * (world - 1) * num_elements element
//!   writes cluster-wide against world * (world - 1) * num_elements for the one-shot, i.e. half
//!   the traffic on four ranks, without ever changing the summation order.
//!
//! The hairpin ring layout, with rank r owning chunk r:
//!
//! ~~~text
//! ring edge            hairpin diagonal (0<->2, 1<->3)
//!    0 <---------------> 1
//!    ^ \               / ^
//!    |  \             /  |
//!    |   \           /   |
//!    |    v         v    |
//!    3 <---------------> 2
//!    ^                   ^
//!    |______ hairpin ____|
//! ~~~
//!
//! On the switchless fabric the diagonal is programmed into the ConnectX-7 eSwitch, so every
//! transfer above is a single hop; on the switched fabric every transfer is a switch hop.

use std::ops::Range;

use crate::error::CommError;

/// World size of the reference 4x DGX Spark cluster.
pub const REFERENCE_WORLD_SIZE: usize = 4;

/// Largest bf16 element count one frame can describe (a frame length is a u32 byte count).
pub const MAX_FRAME_ELEMENTS: usize = (u32::MAX as usize) / 2;

/// Which exchange schedule a collective uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FabricSchedule {
    /// Switched 200G: all-to-all one-shot with a local canonical reduction.
    OneShotAllToAll,
    /// Switchless ring with ConnectX-7 hairpin diagonals: reduce-scatter to chunk owners,
    /// canonical-ordered accumulation, then all-gather of the rounded chunks.
    HairpinRingReduceScatterAllGather,
}

/// Physical link class used by one peer transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FabricLink {
    /// A hop through the lossless 200G RoCEv2 switch.
    SwitchFabric,
    /// A direct DAC to the previous or next rank of the 0-1-2-3-0 ring.
    RingEdge,
    /// The direct diagonal (0<->2, 1<->3) forwarded inside the ConnectX-7 ASIC.
    HairpinDiagonal,
}

/// Which half of a schedule a transfer belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferPhase {
    /// One-shot all-to-all payload write.
    OneShot,
    /// A rank's contribution to a chunk it does not own.
    ReduceScatter,
    /// A reduced chunk sent from its owner to a peer.
    AllGather,
}

/// One peer-to-peer transfer in a schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleTransfer {
    /// Sending rank.
    pub from: usize,
    /// Receiving rank.
    pub to: usize,
    /// Physical link the transfer uses.
    pub link: FabricLink,
    /// Schedule half this transfer belongs to.
    pub phase: TransferPhase,
    /// Element range of the sender's buffer that travels.
    pub elements: Range<usize>,
}

impl ScheduleTransfer {
    /// Bytes on the wire for this transfer (bf16 is two bytes per element).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.elements.len() * 2
    }
}

/// Decomposition of one collective into its peer transfers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveLayout {
    world_size: usize,
    num_elements: usize,
    schedule: FabricSchedule,
}

impl CollectiveLayout {
    /// Builds a layout for world_size ranks and num_elements bf16 elements per rank.
    ///
    /// # Errors
    /// Returns CommError::InvalidLayout for world_size == 0 or for an element count that cannot
    /// fit a frame length.
    pub fn new(
        world_size: usize,
        num_elements: usize,
        schedule: FabricSchedule,
    ) -> Result<Self, CommError> {
        if world_size == 0 {
            return Err(CommError::InvalidLayout(
                "a collective needs at least one rank".to_owned(),
            ));
        }
        if num_elements > MAX_FRAME_ELEMENTS {
            return Err(CommError::InvalidLayout(format!(
                "{num_elements} elements cannot fit one frame (max {MAX_FRAME_ELEMENTS})"
            )));
        }
        Ok(Self {
            world_size,
            num_elements,
            schedule,
        })
    }

    /// Builds the switched-200G all-to-all one-shot layout.
    ///
    /// # Errors
    /// See CollectiveLayout::new.
    pub fn one_shot(world_size: usize, num_elements: usize) -> Result<Self, CommError> {
        Self::new(world_size, num_elements, FabricSchedule::OneShotAllToAll)
    }

    /// Builds the switchless hairpin reduce-scatter + all-gather layout.
    ///
    /// # Errors
    /// See CollectiveLayout::new.
    pub fn hairpin_ring(world_size: usize, num_elements: usize) -> Result<Self, CommError> {
        Self::new(
            world_size,
            num_elements,
            FabricSchedule::HairpinRingReduceScatterAllGather,
        )
    }

    /// Number of ranks in the collective.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// bf16 elements each rank contributes to the collective.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Bytes of payload each rank contributes.
    #[must_use]
    pub fn num_bytes(&self) -> usize {
        // Validated in new(): num_elements <= u32::MAX / 2.
        self.num_elements * 2
    }

    /// Schedule this layout executes.
    #[must_use]
    pub fn schedule(&self) -> FabricSchedule {
        self.schedule
    }

    /// Ordered phases this schedule runs.
    #[must_use]
    pub fn phases(&self) -> &'static [TransferPhase] {
        match self.schedule {
            FabricSchedule::OneShotAllToAll => &[TransferPhase::OneShot],
            FabricSchedule::HairpinRingReduceScatterAllGather => {
                &[TransferPhase::ReduceScatter, TransferPhase::AllGather]
            }
        }
    }

    /// Number of chunks the buffer is split into (one per rank).
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.world_size
    }

    /// Element range owned by chunk, which is also the rank that owns it.
    ///
    /// The split is balanced: the first num_elements % world_size chunks take one extra element,
    /// so the ranges always partition 0..num_elements with no gap and no overlap. An
    /// out-of-range chunk yields an empty range instead of panicking.
    #[must_use]
    pub fn chunk_range(&self, chunk: usize) -> Range<usize> {
        if chunk >= self.world_size {
            return 0..0;
        }
        let base = self.num_elements / self.world_size;
        let remainder = self.num_elements % self.world_size;
        let start = chunk * base + chunk.min(remainder);
        let len = base + usize::from(chunk < remainder);
        start..start + len
    }

    /// Rank that owns chunk.
    #[must_use]
    pub fn owner_of(&self, chunk: usize) -> usize {
        chunk
    }

    /// Physical link a transfer between two distinct ranks uses, or None when the pair is
    /// degenerate (same rank, or a rank outside the collective).
    #[must_use]
    pub fn link_between(&self, from: usize, to: usize) -> Option<FabricLink> {
        if from >= self.world_size || to >= self.world_size || from == to {
            return None;
        }
        match self.schedule {
            FabricSchedule::OneShotAllToAll => Some(FabricLink::SwitchFabric),
            FabricSchedule::HairpinRingReduceScatterAllGather => {
                let forward = (to + self.world_size - from) % self.world_size;
                let distance = forward.min(self.world_size - forward);
                if self.world_size.is_multiple_of(2) && distance == self.world_size / 2 {
                    Some(FabricLink::HairpinDiagonal)
                } else {
                    Some(FabricLink::RingEdge)
                }
            }
        }
    }

    /// Every transfer this schedule performs, in execution order (phase by phase).
    #[must_use]
    pub fn transfers(&self) -> Vec<ScheduleTransfer> {
        let mut transfers = Vec::new();
        match self.schedule {
            FabricSchedule::OneShotAllToAll => {
                for from in 0..self.world_size {
                    for to in 0..self.world_size {
                        if from == to {
                            continue;
                        }
                        transfers.push(ScheduleTransfer {
                            from,
                            to,
                            link: FabricLink::SwitchFabric,
                            phase: TransferPhase::OneShot,
                            elements: 0..self.num_elements,
                        });
                    }
                }
            }
            FabricSchedule::HairpinRingReduceScatterAllGather => {
                for from in 0..self.world_size {
                    for owner in 0..self.world_size {
                        if from == owner {
                            continue;
                        }
                        transfers.push(ScheduleTransfer {
                            from,
                            to: owner,
                            link: self
                                .link_between(from, owner)
                                .unwrap_or(FabricLink::RingEdge),
                            phase: TransferPhase::ReduceScatter,
                            elements: self.chunk_range(owner),
                        });
                    }
                }
                for owner in 0..self.world_size {
                    for to in 0..self.world_size {
                        if owner == to {
                            continue;
                        }
                        transfers.push(ScheduleTransfer {
                            from: owner,
                            to,
                            link: self.link_between(owner, to).unwrap_or(FabricLink::RingEdge),
                            phase: TransferPhase::AllGather,
                            elements: self.chunk_range(owner),
                        });
                    }
                }
            }
        }
        transfers
    }

    /// Number of transfers in the schedule.
    #[must_use]
    pub fn transfer_count(&self) -> usize {
        let peers = self.world_size * (self.world_size - 1);
        match self.schedule {
            FabricSchedule::OneShotAllToAll => peers,
            FabricSchedule::HairpinRingReduceScatterAllGather => 2 * peers,
        }
    }

    /// Closed-form element writes for this schedule, independent of Self::transfers.
    ///
    /// A write is one element of one rank's buffer landing in one destination, so the cluster
    /// total is what matters here: the one-shot writes every rank's full buffer into all
    /// world_size - 1 peers, and the hairpin writes every raw element once to its chunk owner
    /// (reduce-scatter) plus every reduced chunk once to all other ranks (all-gather).
    #[must_use]
    pub fn expected_element_writes(&self) -> usize {
        let per_rank = self.world_size - 1;
        match self.schedule {
            FabricSchedule::OneShotAllToAll => self.world_size * per_rank * self.num_elements,
            FabricSchedule::HairpinRingReduceScatterAllGather => 2 * per_rank * self.num_elements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_partition_the_buffer_for_every_remainder() {
        for world in 1usize..=5 {
            for elements in 0usize..=17 {
                let layout = CollectiveLayout::one_shot(world, elements)
                    .unwrap_or_else(|error| panic!("{error}"));
                let mut seen = vec![0u8; elements];
                let mut covered = 0usize;
                for chunk in 0..world {
                    let range = layout.chunk_range(chunk);
                    assert!(range.start <= range.end, "chunk {chunk} inverted");
                    assert!(range.end <= elements, "chunk {chunk} overruns the buffer");
                    for index in range.clone() {
                        assert_eq!(seen[index], 0, "element {index} covered twice");
                        seen[index] = 1;
                    }
                    covered += range.len();
                }
                assert_eq!(covered, elements, "world {world} elements {elements}");
                assert!(seen.iter().all(|slot| *slot == 1), "gap in the partition");
            }
        }
    }

    #[test]
    fn remainder_goes_to_the_lowest_chunks() {
        let layout =
            CollectiveLayout::hairpin_ring(4, 10).unwrap_or_else(|error| panic!("{error}"));
        let ranges: Vec<Range<usize>> = (0..4).map(|chunk| layout.chunk_range(chunk)).collect();
        assert_eq!(ranges, vec![0..3, 3..6, 6..8, 8..10]);
        assert_eq!(layout.chunk_range(4), 0..0);
        assert_eq!(layout.chunk_range(99), 0..0);
    }

    #[test]
    fn hairpin_diagonals_are_the_opposite_ranks() {
        let hairpin =
            CollectiveLayout::hairpin_ring(4, 8).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            hairpin.link_between(0, 2),
            Some(FabricLink::HairpinDiagonal)
        );
        assert_eq!(
            hairpin.link_between(2, 0),
            Some(FabricLink::HairpinDiagonal)
        );
        assert_eq!(
            hairpin.link_between(1, 3),
            Some(FabricLink::HairpinDiagonal)
        );
        assert_eq!(hairpin.link_between(0, 1), Some(FabricLink::RingEdge));
        assert_eq!(hairpin.link_between(3, 0), Some(FabricLink::RingEdge));
        assert_eq!(hairpin.link_between(0, 0), None);
        assert_eq!(hairpin.link_between(0, 4), None);

        let switched = CollectiveLayout::one_shot(4, 8).unwrap_or_else(|error| panic!("{error}"));
        for from in 0..4 {
            for to in 0..4 {
                if from != to {
                    assert_eq!(
                        switched.link_between(from, to),
                        Some(FabricLink::SwitchFabric)
                    );
                }
            }
        }
    }

    #[test]
    fn transfer_lists_match_their_closed_forms() {
        for world in 1usize..=4 {
            for elements in [0usize, 1, 7, 8, 9, 1024] {
                for schedule in [
                    FabricSchedule::OneShotAllToAll,
                    FabricSchedule::HairpinRingReduceScatterAllGather,
                ] {
                    let layout = CollectiveLayout::new(world, elements, schedule)
                        .unwrap_or_else(|error| panic!("{error}"));
                    let transfers = layout.transfers();
                    assert_eq!(transfers.len(), layout.transfer_count());
                    let writes: usize = transfers.iter().map(|t| t.elements.len()).sum();
                    assert_eq!(
                        writes,
                        layout.expected_element_writes(),
                        "world {world} elements {elements} schedule {schedule:?}"
                    );
                    for transfer in &transfers {
                        assert_ne!(transfer.from, transfer.to);
                        assert!(transfer.to < world);
                        assert!(transfer.elements.end <= elements);
                    }
                }
            }
        }
    }

    #[test]
    fn one_shot_moves_twice_the_hairpin_traffic_on_four_ranks() {
        let one_shot =
            CollectiveLayout::one_shot(4, 1024).unwrap_or_else(|error| panic!("{error}"));
        let hairpin =
            CollectiveLayout::hairpin_ring(4, 1024).unwrap_or_else(|error| panic!("{error}"));
        // Cluster-wide: the one-shot writes 4 * 3 full buffers; the hairpin writes 3 buffers
        // per phase over two phases.
        assert_eq!(one_shot.expected_element_writes(), 12 * 1024);
        assert_eq!(hairpin.expected_element_writes(), 6 * 1024);
        assert_eq!(
            one_shot.expected_element_writes(),
            2 * hairpin.expected_element_writes()
        );
    }

    #[test]
    fn rejects_impossible_layouts() {
        assert!(matches!(
            CollectiveLayout::new(0, 8, FabricSchedule::OneShotAllToAll),
            Err(CommError::InvalidLayout(_))
        ));
        assert!(matches!(
            CollectiveLayout::one_shot(4, MAX_FRAME_ELEMENTS + 1),
            Err(CommError::InvalidLayout(_))
        ));
        // The largest legal element count still describes a frame that fits a u32 length.
        let layout = CollectiveLayout::one_shot(4, MAX_FRAME_ELEMENTS)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(u32::try_from(layout.num_bytes()).is_ok());
    }
}
