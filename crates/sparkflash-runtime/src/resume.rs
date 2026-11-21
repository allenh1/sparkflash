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

//! Where a prompt takes up a sequence's state, what a rewind must restore, and the switch that
//! decides whether a rewind point is kept at all.
//!
//! [`resume`] answers "may this prompt ride the state as it stands, or go back to the rewind
//! point?"; [`rewind_plan`] turns the layout's declaration into what the engine must do next.
//! Both are free functions, not methods on [`crate::Runtime`], so the tick engine calls them
//! unchanged. They lived in the fused engine's module until it was retired.

use crate::layout::{StateKind, StateLayout};

/// Whether `prompt` extends `held` by at least one token: what a sequence whose prefill only
/// forwards past its history can take as it stands, and the rule a lane's room check uses to tell
/// a continued prompt from one that starts over.
#[must_use]
pub fn extends(held: &[u32], prompt: &[u32]) -> bool {
    held.len() < prompt.len() && prompt.starts_with(held)
}

/// Whether a sequence keeps a rewind point at the end of each prompt: unset, empty or `on` (the
/// default) keeps one, `off` keeps none, so a prompt that parts from the reply after the last
/// prompt starts over. Anything else is refused. Every arm that keeps a rewind point reads it.
pub const REWIND_ENV: &str = "SPARKFLASH_REWIND";

/// [REWIND_ENV] as set.
///
/// # Errors
/// A value other than `on` or `off`.
pub fn rewind() -> Result<bool, String> {
    rewind_request(std::env::var(REWIND_ENV).ok().as_deref())
}

/// What a [REWIND_ENV] of `value` asks for: `true` keeps a rewind point.
///
/// ```
/// use sparkflash_runtime::resume::rewind_request;
/// assert_eq!(rewind_request(None), Ok(true));
/// assert_eq!(rewind_request(Some("off")), Ok(false));
/// assert!(rewind_request(Some("0")).is_err());
/// ```
///
/// # Errors
/// A value other than `on` or `off`.
pub fn rewind_request(value: Option<&str>) -> Result<bool, String> {
    match value.map(str::trim) {
        None | Some("" | "on") => Ok(true),
        Some("off") => Ok(false),
        Some(other) => Err(format!(
            "{REWIND_ENV}={other} is not a switch: use on (the default: keep the end of each \
             prompt to go back to) or off (start over)"
        )),
    }
}

/// Where a prompt takes up a kept sequence ([`resume`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resume {
    /// The sequence's state ends at its history and the prompt extends it: forward the prompt
    /// past these many tokens.
    Extend(usize),
    /// The prompt keeps the rewind point and goes past it: put the state back to the point, then
    /// forward the prompt past these many tokens.
    Rewind(usize),
}

/// How `prompt` takes up a sequence that holds `history`, when it can: past the whole history
/// while the state ends there (`current`), or else past the first `point` tokens the sequence
/// can go back to. A sequence a verify left ahead of its history (answers its caller gave back)
/// is not current, so only its point serves. `None` says the prompt starts over.
///
/// ```
/// use sparkflash_runtime::resume::{Resume, resume};
/// let history = [1, 2, 3, 4, 5];
/// assert_eq!(resume(&history, true, Some(3), &[1, 2, 3, 4, 5, 6]), Some(Resume::Extend(5)));
/// assert_eq!(resume(&history, true, Some(3), &[1, 2, 3, 9]), Some(Resume::Rewind(3)));
/// assert_eq!(resume(&history, false, Some(3), &[1, 2, 3, 4, 5, 6]), Some(Resume::Rewind(3)));
/// assert_eq!(resume(&history, true, Some(3), &[1, 2, 3]), None);
/// assert_eq!(resume(&history, true, None, &[1, 2, 3, 9]), None);
/// assert_eq!(resume(&[], true, None, &[7]), Some(Resume::Extend(0)));
/// ```
#[must_use]
pub fn resume(
    history: &[u32],
    current: bool,
    point: Option<usize>,
    prompt: &[u32],
) -> Option<Resume> {
    if current && extends(history, prompt) {
        return Some(Resume::Extend(history.len()));
    }
    let point = point.filter(|&point| point <= history.len())?;
    extends(&history[..point], prompt).then_some(Resume::Rewind(point))
}

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
    fn the_prefix_rules_take_up_a_kept_sequence() {
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
