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

//! Where a prompt takes up a sequence's state, and what a rewind must restore.
//!
//! [`resume`] answers "may this prompt ride the state as it stands, or go back to the rewind
//! point?"; [`rewind_plan`] turns the layout's declaration into what the engine must do next.
//! Both are free functions, not methods on [`crate::Runtime`], so the tick engine of phase 6 can
//! call them unchanged when it replaces the fused engine.
//!
//! This module re-exports the prefix rules that used to live only in [`crate::fused`]; a caller
//! that imports from here and one that imports from there get the same function.

pub use crate::fused::{Resume, resume};

use crate::layout::{StateKind, StateLayout};

/// What a rewind has to restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restore {
    /// Nothing: the state is at the position already, or the sequence holds nothing.
    Nothing,
    /// Positional state only: truncate to the position.
    Truncate,
    /// At least one recurrent region: restore the arm's checkpoint at the position, and truncate
    /// the positional regions too.
    Checkpoint,
}

/// A rewind to `to`, and what it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rewind {
    /// The position the sequence goes back to.
    pub to: usize,
    /// What the arm must restore.
    pub restore: Restore,
}

/// The rewind plan for `layout` to position `to`.
#[must_use]
pub fn rewind_plan(layout: &StateLayout, to: usize) -> Rewind {
    let restore = match layout.kind() {
        StateKind::None => Restore::Nothing,
        StateKind::Positional => Restore::Truncate,
        StateKind::Recurrent | StateKind::Both => Restore::Checkpoint,
    };
    Rewind { to, restore }
}

/// Whether an arm with `layout` must write a checkpoint after a prompt ending at `position` for a
/// later rewind to it to succeed. A layout with a recurrent region must; a positional-only one
/// must not.
#[must_use]
pub fn checkpoint_after(layout: &StateLayout, position: usize) -> Option<usize> {
    layout.needs_checkpoint().then_some(position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Region;

    #[test]
    fn a_positional_layout_truncates_and_a_recurrent_one_checkpoints() {
        let positional = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 4,
        }]);
        assert_eq!(
            rewind_plan(&positional, 7),
            Rewind {
                to: 7,
                restore: Restore::Truncate
            }
        );
        assert_eq!(checkpoint_after(&positional, 7), None);

        let recurrent = StateLayout::new(vec![Region::Recurrent {
            name: "state",
            bytes: 16,
        }]);
        assert_eq!(
            rewind_plan(&recurrent, 7),
            Rewind {
                to: 7,
                restore: Restore::Checkpoint
            }
        );
        assert_eq!(checkpoint_after(&recurrent, 7), Some(7));

        assert_eq!(
            rewind_plan(&StateLayout::empty(), 3).restore,
            Restore::Nothing
        );
    }

    #[test]
    fn the_prefix_rules_are_the_fused_engines() {
        assert_eq!(
            resume(&[1, 2], true, Some(1), &[1, 2, 3]),
            Some(Resume::Extend(2))
        );
        assert_eq!(
            resume(&[1, 2], true, Some(1), &[1, 9]),
            Some(Resume::Rewind(1))
        );
    }
}
