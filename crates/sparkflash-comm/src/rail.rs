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

//! Dual-rail ConnectX-7 striping and schedule byte accounting.
//!
//! Rail 0 and rail 1 are the two PCIe Gen5 x4 halves of the ConnectX-7 QSFP112 port. Which RDMA
//! device backs a rail is decided at startup by the crate::fabric discovery and never hardcoded:
//! the spec's f0 functions are DOWN at 40 Gb/s on the live four-node cluster, so the crate
//! registers the ACTIVE 200 Gb/s f1 functions and reports the deviation. This module only
//! stripes bytes: every transfer is split half and half across the two rails, and the remainder
//! (at most one byte per transfer) always goes to rail 0, which keeps the split deterministic
//! for odd byte counts.

use crate::error::CommError;
use crate::layout::{CollectiveLayout, FabricSchedule, TransferPhase};

/// Upper bound on the audit grid so a diagnostic call cannot allocate gigabytes.
const AUDIT_MAX_CELLS: usize = 1 << 22;

/// Deterministic half/half byte split between the two rails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RailSplit {
    /// Bytes assigned to rail 0.
    pub rail0_bytes: usize,
    /// Bytes assigned to rail 1.
    pub rail1_bytes: usize,
}

impl RailSplit {
    /// Bytes in the split; always exactly the split input.
    #[must_use]
    pub fn total(&self) -> usize {
        self.rail0_bytes + self.rail1_bytes
    }
}

/// Per-rail byte ledger, either predicted from a layout or measured by a transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RailLedger {
    /// Bytes assigned to rail 0.
    pub rail0_bytes: u64,
    /// Bytes assigned to rail 1.
    pub rail1_bytes: u64,
    /// Number of transfers accounted.
    pub transfers: u64,
    /// bf16 elements moved (a transfer of a 3-element chunk counts 3).
    pub element_writes: u64,
}

impl RailLedger {
    /// Total bytes across both rails.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.rail0_bytes + self.rail1_bytes
    }

    /// Accounts one transfer of that many bytes, striped half/half.
    pub fn add_transfer(&mut self, bytes: usize) {
        let split = RailPlan::split_bytes(bytes);
        self.rail0_bytes += split.rail0_bytes as u64;
        self.rail1_bytes += split.rail1_bytes as u64;
        self.transfers += 1;
        self.element_writes += (bytes / 2) as u64;
    }
}

/// Coverage and overlap report for a schedule, produced by RailPlan::audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RailAudit {
    /// Predicted per-rail and total byte counts.
    pub ledger: RailLedger,
    /// Distinct destination elements written, by phase and destination rank.
    pub coverage: Vec<Vec<usize>>,
    /// Maximum number of writes to any destination element, by phase and destination.
    pub overlap: Vec<Vec<u32>>,
    /// Elements each rank must stage locally (its own chunk, or the whole buffer for one-shot).
    pub local_elements: Vec<usize>,
}

/// Stateless rail striping and schedule accounting helpers.
pub struct RailPlan;

impl RailPlan {
    /// Splits a byte count between the rails; the odd byte, if any, goes to rail 0.
    #[must_use]
    pub fn split_bytes(num_bytes: usize) -> RailSplit {
        RailSplit {
            rail0_bytes: num_bytes - num_bytes / 2,
            rail1_bytes: num_bytes / 2,
        }
    }

    /// Splits a slice between the rails with the same rule as split_bytes.
    #[must_use]
    pub fn split_slice<T>(buffer: &[T]) -> (&[T], &[T]) {
        let split = Self::split_bytes(buffer.len());
        buffer.split_at(split.rail0_bytes)
    }

    /// Predicts the byte traffic of every transfer in the layout.
    #[must_use]
    pub fn account(layout: &CollectiveLayout) -> RailLedger {
        let mut ledger = RailLedger::default();
        for transfer in layout.transfers() {
            ledger.add_transfer(transfer.bytes());
        }
        ledger
    }

    /// Audits a layout: predicts the rail bytes and records where every element lands.
    ///
    /// The audit allocates one u32 counter per (phase, destination, element) cell, so it is a
    /// diagnostic for tests and bring-up, not a hot-path call.
    ///
    /// # Errors
    /// Returns CommError::Protocol for a malformed transfer and CommError::Unsupported when the
    /// audit grid would exceed the internal limit.
    pub fn audit(layout: &CollectiveLayout) -> Result<RailAudit, CommError> {
        let phases = layout.phases();
        let world = layout.world_size();
        let elements = layout.num_elements();
        let cells = phases
            .len()
            .checked_mul(world)
            .and_then(|value| value.checked_mul(elements))
            .unwrap_or(usize::MAX);
        if cells > AUDIT_MAX_CELLS {
            return Err(CommError::Unsupported(format!(
                "rail audit grid of {cells} cells exceeds the {AUDIT_MAX_CELLS} cell limit"
            )));
        }
        let mut counters = vec![vec![vec![0u32; elements]; world]; phases.len()];
        let mut ledger = RailLedger::default();
        for transfer in layout.transfers() {
            if transfer.from >= world || transfer.to >= world || transfer.from == transfer.to {
                return Err(CommError::Protocol(format!(
                    "malformed transfer {} -> {} for world {world}",
                    transfer.from, transfer.to
                )));
            }
            let phase = phase_index(layout, transfer.phase).ok_or_else(|| {
                CommError::Protocol(format!(
                    "transfer phase {:?} is not in the layout's phase list",
                    transfer.phase
                ))
            })?;
            ledger.add_transfer(transfer.bytes());
            for element in transfer.elements.clone() {
                if element >= elements {
                    return Err(CommError::Protocol(format!(
                        "transfer {} -> {} writes element {element} beyond {elements}",
                        transfer.from, transfer.to
                    )));
                }
                counters[phase][transfer.to][element] += 1;
            }
        }
        let mut coverage = Vec::with_capacity(phases.len());
        let mut overlap = Vec::with_capacity(phases.len());
        for phase in &counters {
            let mut phase_coverage = Vec::with_capacity(world);
            let mut phase_overlap = Vec::with_capacity(world);
            for destination in phase {
                phase_coverage.push(destination.iter().filter(|count| **count > 0).count());
                phase_overlap.push(destination.iter().copied().max().unwrap_or(0));
            }
            coverage.push(phase_coverage);
            overlap.push(phase_overlap);
        }
        let local_elements = match layout.schedule() {
            FabricSchedule::OneShotAllToAll => vec![elements; world],
            FabricSchedule::HairpinRingReduceScatterAllGather => (0..world)
                .map(|rank| layout.chunk_range(rank).len())
                .collect(),
        };
        Ok(RailAudit {
            ledger,
            coverage,
            overlap,
            local_elements,
        })
    }
}

/// Position of a phase inside CollectiveLayout::phases, or None when it is not scheduled.
#[must_use]
pub(crate) fn phase_index(layout: &CollectiveLayout, phase: TransferPhase) -> Option<usize> {
    layout
        .phases()
        .iter()
        .position(|candidate| *candidate == phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_conservative_and_puts_the_odd_byte_on_rail_zero() {
        for bytes in 0usize..4096 {
            let split = RailPlan::split_bytes(bytes);
            assert_eq!(split.total(), bytes, "split lost or invented bytes");
            assert_eq!(split.rail0_bytes, bytes.div_ceil(2));
            assert_eq!(split.rail1_bytes, bytes / 2);
        }
        let odd = RailPlan::split_bytes(7);
        assert_eq!(odd.rail0_bytes, 4);
        assert_eq!(odd.rail1_bytes, 3);
    }

    #[test]
    fn split_slice_matches_split_bytes() {
        let buffer: Vec<u16> = (0..9).collect();
        let (rail0, rail1) = RailPlan::split_slice(&buffer);
        assert_eq!(rail0, &buffer[..5]);
        assert_eq!(rail1, &buffer[5..]);
        let empty: [u16; 0] = [];
        let (left, right) = RailPlan::split_slice(&empty);
        assert!(left.is_empty() && right.is_empty());
    }

    #[test]
    fn one_shot_ledger_is_balanced_and_conservative() {
        let layout = CollectiveLayout::one_shot(4, 64).unwrap_or_else(|error| panic!("{error}"));
        let ledger = RailPlan::account(&layout);
        let expected_total = (layout.transfer_count() * 128) as u64;
        assert_eq!(ledger.total_bytes(), expected_total);
        assert_eq!(ledger.transfers, 12);
        assert_eq!(ledger.element_writes, 12 * 64);
        assert_eq!(ledger.rail0_bytes, ledger.rail1_bytes);
    }

    #[test]
    fn hairpin_ledger_moves_half_the_one_shot_bytes() {
        let one_shot = CollectiveLayout::one_shot(4, 64).unwrap_or_else(|error| panic!("{error}"));
        let hairpin =
            CollectiveLayout::hairpin_ring(4, 64).unwrap_or_else(|error| panic!("{error}"));
        let one_shot_ledger = RailPlan::account(&one_shot);
        let hairpin_ledger = RailPlan::account(&hairpin);
        assert_eq!(
            hairpin_ledger.total_bytes() * 2,
            one_shot_ledger.total_bytes()
        );
        assert_eq!(hairpin_ledger.rail0_bytes, hairpin_ledger.rail1_bytes);
        assert_eq!(
            hairpin_ledger.element_writes,
            hairpin.expected_element_writes() as u64
        );
    }

    #[test]
    fn audit_reports_exact_coverage_and_overlap() {
        let layout = CollectiveLayout::one_shot(4, 8).unwrap_or_else(|error| panic!("{error}"));
        let audit = RailPlan::audit(&layout).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(audit.coverage.len(), 1);
        for destination in 0..4 {
            // Every destination receives all three peers' full buffers.
            assert_eq!(audit.coverage[0][destination], 8);
            assert_eq!(audit.overlap[0][destination], 3);
        }
        assert_eq!(audit.local_elements, vec![8, 8, 8, 8]);

        let hairpin =
            CollectiveLayout::hairpin_ring(4, 10).unwrap_or_else(|error| panic!("{error}"));
        let audit = RailPlan::audit(&hairpin).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(audit.coverage.len(), 2);
        for destination in 0..4 {
            let own_chunk = hairpin.chunk_range(destination).len();
            // Reduce-scatter: every peer sends its contribution of my chunk.
            assert_eq!(audit.coverage[0][destination], own_chunk);
            assert_eq!(audit.overlap[0][destination], 3);
            // All-gather: the owners of every other chunk send me the reduced chunk.
            assert_eq!(audit.coverage[1][destination], 10 - own_chunk);
            assert_eq!(audit.overlap[1][destination], 1);
        }
        assert_eq!(
            audit.local_elements,
            (0..4)
                .map(|rank| hairpin.chunk_range(rank).len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn audit_refuses_an_oversized_grid() {
        let layout = CollectiveLayout::one_shot(4, 4 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(
            RailPlan::audit(&layout),
            Err(CommError::Unsupported(_))
        ));
        let hairpin = CollectiveLayout::hairpin_ring(4, 4 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(
            RailPlan::audit(&hairpin),
            Err(CommError::Unsupported(_))
        ));
        assert_eq!(phase_index(&hairpin, TransferPhase::AllGather), Some(1));
        assert_eq!(phase_index(&hairpin, TransferPhase::OneShot), None);
        assert_eq!(
            hairpin.schedule(),
            FabricSchedule::HairpinRingReduceScatterAllGather
        );
    }
}
