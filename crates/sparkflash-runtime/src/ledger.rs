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

//! Which of the runtime's strategies an arm runs, and how: the ledger printed when a fused engine
//! starts.
//!
//! Every strategy in this crate is a shape of walk ([docs/engine/03-runtime.md]'s segments). An
//! arm whose walk takes the shape runs the strategy natively; one that takes only a smaller shape
//! has the runtime loop it; one that cannot, or was told not to, has it off, with the reason. The
//! ledger says which for every strategy, so a deployment's log answers "is this arm drafting in
//! lanes?" without reading its code, and a strategy that silently fell back would show as one.
//!
//! The engine fills what it decides itself (how many lanes, whether prompts run in parts at all);
//! the arm fills what only it knows ([crate::FusedArm::ledger]): whether it splits its prompts,
//! drafts, verifies its lanes' rounds in one walk, hands back multi-token runs or keeps prefixes.
//!
//! [docs/engine/03-runtime.md]: ../../../docs/engine/03-runtime.md

use std::fmt;

/// One strategy the runtime offers an arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    /// Several requests' rows in one walk ([crate::FusedEngine]).
    Lanes,
    /// A long prompt a chunk at a time between the decoders' steps ([crate::FusedArm::begin_prompt]).
    PromptParts,
    /// Speculative drafting and a verify walk.
    Drafting,
    /// Every drafting lane's round verified in one walk ([crate::run_round]).
    JointRounds,
    /// A verify's accepted run handed to the request in one reply ([crate::FusedArm::ready]).
    Runs,
    /// A later prompt continuing what a slot holds ([crate::FusedArm::history]).
    PrefixReuse,
    /// Rank 0 decides and broadcasts, every other rank replays ([crate::GroupLanes]).
    Group,
}

impl Strategy {
    /// Every strategy, in the order the ledger prints them.
    pub const ALL: [Self; 7] = [
        Self::Lanes,
        Self::PromptParts,
        Self::Drafting,
        Self::JointRounds,
        Self::Runs,
        Self::PrefixReuse,
        Self::Group,
    ];

    /// The name the ledger prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lanes => "lanes",
            Self::PromptParts => "prompt parts",
            Self::Drafting => "drafting",
            Self::JointRounds => "joint rounds",
            Self::Runs => "multi-token runs",
            Self::PrefixReuse => "prefix reuse",
            Self::Group => "group",
        }
    }
}

/// How an arm runs one strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// The arm's walk takes the strategy's shape; the detail says how (`4 slots`, `DSpark, depth 7`).
    Native(String),
    /// The runtime runs it as a loop over a shape the walk does take; the detail says which.
    Looped(String),
    /// Not run; the detail says why (an opt-out, or what the arm lacks).
    Off(String),
}

impl Mode {
    /// Natively, with `detail` (which may be empty).
    #[must_use]
    pub fn native(detail: impl Into<String>) -> Self {
        Self::Native(detail.into())
    }

    /// Looped, as `detail` says.
    #[must_use]
    pub fn looped(detail: impl Into<String>) -> Self {
        Self::Looped(detail.into())
    }

    /// Off, because of `why`.
    #[must_use]
    pub fn off(why: impl Into<String>) -> Self {
        Self::Off(why.into())
    }

    /// Whether the strategy runs at all.
    #[must_use]
    pub const fn runs(&self) -> bool {
        !matches!(self, Self::Off(_))
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (word, detail) = match self {
            Self::Native(detail) => ("native", detail),
            Self::Looped(detail) => ("looped", detail),
            Self::Off(detail) => ("off", detail),
        };
        if detail.is_empty() {
            f.write_str(word)
        } else {
            write!(f, "{word} ({detail})")
        }
    }
}

/// Every strategy's mode for one arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ledger {
    arm: String,
    modes: Vec<(Strategy, Mode)>,
}

/// What an arm says of a strategy it never mentions.
const UNSTATED: &str = "the arm does not state it";

impl Ledger {
    /// A ledger for `arm` with nothing stated yet.
    #[must_use]
    pub fn new(arm: impl Into<String>) -> Self {
        Self {
            arm: arm.into(),
            modes: Vec::new(),
        }
    }

    /// The arm's name.
    #[must_use]
    pub fn arm(&self) -> &str {
        &self.arm
    }

    /// States `strategy`'s mode, replacing what was stated before.
    pub fn set(&mut self, strategy: Strategy, mode: Mode) -> &mut Self {
        match self
            .modes
            .iter_mut()
            .find(|(stated, _)| *stated == strategy)
        {
            Some((_, stated)) => *stated = mode,
            None => self.modes.push((strategy, mode)),
        }
        self
    }

    /// [Self::set], by value.
    #[must_use]
    pub fn with(mut self, strategy: Strategy, mode: Mode) -> Self {
        self.set(strategy, mode);
        self
    }

    /// `strategy`'s mode; a strategy nobody stated is off, and says so.
    #[must_use]
    pub fn mode(&self, strategy: Strategy) -> Mode {
        self.modes
            .iter()
            .find(|(stated, _)| *stated == strategy)
            .map_or_else(|| Mode::off(UNSTATED), |(_, mode)| mode.clone())
    }

    /// Whether `strategy` was stated.
    #[must_use]
    pub fn states(&self, strategy: Strategy) -> bool {
        self.modes.iter().any(|(stated, _)| *stated == strategy)
    }

    /// The drafting strategies a drafting arm states together: drafting itself, whether its
    /// lanes' rounds share one walk (`joint`, as [crate::joint_rounds] and the arm decided) and
    /// whether a round's run reaches the request in one reply (`runs`). `drafter` is `None` for
    /// an arm with no drafter configured, which turns all three off.
    pub fn drafting(&mut self, drafter: Option<String>, joint: bool, runs: bool) -> &mut Self {
        let Some(drafter) = drafter else {
            let none = "no drafter configured";
            self.set(Strategy::Drafting, Mode::off(none));
            self.set(Strategy::JointRounds, Mode::off(none));
            return self.set(Strategy::Runs, Mode::off(none));
        };
        self.set(Strategy::Drafting, Mode::Native(drafter));
        self.set(
            Strategy::JointRounds,
            if joint {
                Mode::native("")
            } else {
                Mode::looped(format!(
                    "{}=off: each lane's round walks alone",
                    crate::JOINT_ROUNDS_ENV
                ))
            },
        );
        self.set(
            Strategy::Runs,
            if runs {
                Mode::native("")
            } else {
                Mode::off("the arm holds no answers past a step")
            },
        )
    }

    /// The ledger as one boot line: `ledger <arm>: lanes native (4 slots); prompt parts ...`.
    #[must_use]
    pub fn line(&self) -> String {
        let modes: Vec<String> = Strategy::ALL
            .iter()
            .map(|&strategy| format!("{} {}", strategy.name(), self.mode(strategy)))
            .collect();
        format!("ledger {}: {}", self.arm, modes.join("; "))
    }
}

impl fmt::Display for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.line())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_every_strategy_in_order_and_an_unstated_one_is_off() {
        let ledger = Ledger::new("toy")
            .with(Strategy::Lanes, Mode::native("4 slots"))
            .with(
                Strategy::PromptParts,
                Mode::off("the arm runs its prompts whole"),
            );
        assert_eq!(
            ledger.line(),
            "ledger toy: lanes native (4 slots); prompt parts off (the arm runs its prompts \
             whole); drafting off (the arm does not state it); joint rounds off (the arm does not \
             state it); multi-token runs off (the arm does not state it); prefix reuse off (the \
             arm does not state it); group off (the arm does not state it)"
        );
    }

    #[test]
    fn a_later_statement_replaces_an_earlier_one() {
        let mut ledger = Ledger::new("toy");
        ledger.set(Strategy::PromptParts, Mode::native("4096-token chunks"));
        ledger.set(
            Strategy::PromptParts,
            Mode::off("SPARKFLASH_PROMPT_PARTS=off"),
        );
        assert_eq!(
            ledger.mode(Strategy::PromptParts),
            Mode::off("SPARKFLASH_PROMPT_PARTS=off")
        );
        assert!(!ledger.mode(Strategy::PromptParts).runs());
    }

    #[test]
    fn drafting_states_its_three_strategies_together() {
        let mut ledger = Ledger::new("toy");
        ledger.drafting(Some("MTP, depth 3".into()), false, true);
        assert_eq!(
            ledger.mode(Strategy::Drafting),
            Mode::native("MTP, depth 3")
        );
        assert!(matches!(
            ledger.mode(Strategy::JointRounds),
            Mode::Looped(_)
        ));
        assert_eq!(ledger.mode(Strategy::Runs), Mode::native(""));
        ledger.drafting(None, true, true);
        for strategy in [Strategy::Drafting, Strategy::JointRounds, Strategy::Runs] {
            assert_eq!(ledger.mode(strategy), Mode::off("no drafter configured"));
        }
    }
}
