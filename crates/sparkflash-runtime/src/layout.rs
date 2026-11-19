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

//! What a sequence's state is made of, and what may be rewound how.
//!
//! The runtime owns reuse and rewind, so an arm must state which of its regions grow with
//! position and which do not ([accounting a positional cache as a prefix][accounting]). The
//! distinction is the whole rewind strategy: a positional region can be truncated to any
//! earlier position, while a recurrent region has one live value and needs an arm checkpoint to
//! go back ([`LaneState::checkpoint`]).
//!
//! [accounting]: https://en.wikipedia.org/wiki/Attention_(machine_learning)

use sparkflash_model::backend::BackendError;

/// One region of sequence state, as the arm accounts for it.
///
/// Names are for the ledger and for error messages; they are not addresses. An arm may have
/// many regions of the same name (every attention layer's cache, say) and should.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// State addressed by position: a KV cache's rows, for example. Rewinding is a truncation
    /// to the position wanted and costs no copy.
    Positional {
        /// A human-readable name, such as `"kv"`.
        name: &'static str,
        /// The bytes one position occupies.
        bytes_per_row: usize,
    },
    /// State with one live value, advanced by every row: a recurrent mixer's state, for example.
    /// Rewinding needs a checkpoint the arm wrote at the position wanted.
    Recurrent {
        /// A human-readable name, such as `"gated-deltanet"`.
        name: &'static str,
        /// The bytes the region holds.
        bytes: usize,
    },
}

impl Region {
    /// The region's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Positional { name, .. } | Self::Recurrent { name, .. } => name,
        }
    }

    /// Whether this region grows with position.
    #[must_use]
    pub const fn is_positional(self) -> bool {
        matches!(self, Self::Positional { .. })
    }

    /// The bytes the region holds after `positions` positions.
    #[must_use]
    pub const fn bytes_at(self, positions: usize) -> usize {
        match self {
            Self::Positional { bytes_per_row, .. } => bytes_per_row.saturating_mul(positions),
            Self::Recurrent { bytes, .. } => bytes,
        }
    }
}

/// What a sequence's state looks like, in the order its arm declares it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateLayout {
    regions: Vec<Region>,
}

impl StateLayout {
    /// A layout of `regions`.
    #[must_use]
    pub fn new(regions: Vec<Region>) -> Self {
        Self { regions }
    }

    /// A sequence that holds nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// The regions, in declaration order.
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Every positional region, in declaration order.
    pub fn positional(&self) -> impl Iterator<Item = Region> + '_ {
        self.regions
            .iter()
            .copied()
            .filter(|region| region.is_positional())
    }

    /// Every recurrent region, in declaration order.
    pub fn recurrent(&self) -> impl Iterator<Item = Region> + '_ {
        self.regions
            .iter()
            .copied()
            .filter(|region| !region.is_positional())
    }

    /// Whether state changes only with position, so any rewind is a truncation.
    #[must_use]
    pub fn is_positional_only(&self) -> bool {
        !self.regions.is_empty() && self.regions.iter().all(|region| region.is_positional())
    }

    /// Whether at least one region needs an arm checkpoint to rewind.
    #[must_use]
    pub fn needs_checkpoint(&self) -> bool {
        self.regions.iter().any(|region| !region.is_positional())
    }

    /// Whether the sequence holds any state at all.
    #[must_use]
    pub fn has_state(&self) -> bool {
        !self.regions.is_empty()
    }

    /// The bytes one position occupies across every positional region.
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.positional()
            .map(|region| match region {
                Region::Positional { bytes_per_row, .. } => bytes_per_row,
                Region::Recurrent { .. } => 0,
            })
            .sum()
    }

    /// The bytes the whole lane holds after `positions` positions.
    #[must_use]
    pub fn bytes_at(&self, positions: usize) -> usize {
        self.regions
            .iter()
            .map(|region| region.bytes_at(positions))
            .sum()
    }
}

/// The broad shape of a layout, for capabilities and the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    /// Nothing is kept between requests.
    None,
    /// Only positional state: rewind is a truncation.
    Positional,
    /// Only recurrent state: rewind needs a checkpoint.
    Recurrent,
    /// Both kinds: positional state is truncated and recurrent state is restored.
    Both,
}

impl StateLayout {
    /// The layout's [StateKind].
    #[must_use]
    pub fn kind(&self) -> StateKind {
        match (
            self.regions.iter().any(|region| region.is_positional()),
            self.needs_checkpoint(),
        ) {
            (false, false) => StateKind::None,
            (true, false) => StateKind::Positional,
            (false, true) => StateKind::Recurrent,
            (true, true) => StateKind::Both,
        }
    }
}

/// Most rows one pass may hold, most members one batch may hold, and the context a sequence may
/// reach.
///
/// Rows and members are separate numbers because they are separate widths: a prompt chunk may be
/// four thousand rows while a token batch holds four members, and a verify pass of `members`
/// members is `members x (1 + depth)` rows. A drafting walk states the member count of its
/// verify batch here, so the engine caps members at its verify width and not at its prompt chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most rows one [crate::Walk::walk] may be given.
    pub rows: usize,
    /// The most members one batch may hold.
    pub members: usize,
    /// The most positions a sequence may hold.
    pub context: usize,
}

impl Limits {
    /// Limits of `rows` rows a pass and `context` positions a sequence, with as many members a
    /// batch as rows a pass.
    #[must_use]
    pub const fn new(rows: usize, context: usize) -> Self {
        Self {
            rows,
            members: rows,
            context,
        }
    }

    /// The same limits with `members` members a batch.
    #[must_use]
    pub const fn with_members(mut self, members: usize) -> Self {
        self.members = members;
        self
    }
}

/// The operations the runtime needs from one sequence's state.
///
/// A [crate::Walk] names its sequence's implementation; a missing one is a compile error at the
/// associated type. Every method is required: an all-positional arm still states its truncation,
/// and its checkpoint may be a visible no-op.
pub trait LaneState {
    /// The sequence's position: the rows it has forwarded and still holds.
    #[must_use]
    fn position(&self) -> usize;

    /// Copies the recurrent regions as they stand at `at`, so a later [Self::rewind] can return
    /// here. An arm with no recurrent region may do nothing. `at` is never past
    /// [Self::position].
    ///
    /// # Errors
    /// A device failure, or an arm that needs a checkpoint and cannot write one.
    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError>;

    /// Puts the state back to `to`: positional regions are truncated, recurrent regions are
    /// restored from the checkpoint at that position. A position of zero clears the state.
    ///
    /// # Errors
    /// `to` past [Self::position], a missing checkpoint, or a device failure.
    fn rewind(&mut self, to: usize) -> Result<(), BackendError>;

    /// Whether the sequence can grow to `positions`, or (when `reset`) start over and use the
    /// chunks a lane of that many positions needs. It answers the physical memory question; the
    /// runtime asks it before a job runs, so an idle sequence can be given up for a long one.
    fn room(&self, positions: usize, reset: bool) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kind_follows_the_regions() {
        let positional = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 16,
        }]);
        assert_eq!(positional.kind(), StateKind::Positional);
        assert_eq!(positional.bytes_at(4), 64);
        assert!(!positional.needs_checkpoint());

        let recurrent = StateLayout::new(vec![Region::Recurrent {
            name: "state",
            bytes: 4096,
        }]);
        assert_eq!(recurrent.kind(), StateKind::Recurrent);
        assert_eq!(recurrent.bytes_at(4), 4096);
        assert!(recurrent.needs_checkpoint());

        let both = StateLayout::new(vec![
            Region::Positional {
                name: "kv",
                bytes_per_row: 8,
            },
            Region::Recurrent {
                name: "state",
                bytes: 64,
            },
        ]);
        assert_eq!(both.kind(), StateKind::Both);
        assert_eq!(both.bytes_at(2), 80);
        assert_eq!(both.bytes_per_row(), 8);
        assert_eq!(StateLayout::empty().kind(), StateKind::None);
    }
}
