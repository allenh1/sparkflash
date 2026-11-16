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

//! One speculative round for several sequences at once.
//!
//! Each member opens its round on the token it was fed: a member whose last verify already holds
//! that token's row answers it ([Opened::Held]); any other settles its last round and drafts its
//! next chain, reporting the chance each draft is accepted ([Opened::Round]). The rows of one walk
//! are then shared among the chains ([share_rows] over the arm's [StepWall]), and one walk
//! verifies every member's anchor and the drafts it was given. A member given no draft still
//! walks its anchor's row, so every member steps.
//!
//! The arm owns everything model-specific: its sequences' state, its drafter, and the walk over
//! segments (one sequence's anchor and drafts each). The output is the plain steps' whatever
//! depths the share picks, since each member's verify keeps only the rows its caller then feeds.

use sparkflash_model::backend::{BackendError, StepOutput};

use crate::share::{StepWall, share_rows};

/// Whether drafting lanes verify their rounds in one shared walk: unset or `on` shares, `off`
/// runs each lane's round alone, one after another (the arm's own single-sequence depth). Any
/// other value is refused.
pub const JOINT_ROUNDS_ENV: &str = "SPARKFLASH_JOINT_ROUNDS";

/// [JOINT_ROUNDS_ENV] as set.
///
/// # Errors
/// A value other than `on` or `off`.
pub fn joint_rounds() -> Result<bool, String> {
    match std::env::var(JOINT_ROUNDS_ENV) {
        Err(_) => Ok(true),
        Ok(value) => match value.trim() {
            "" | "on" => Ok(true),
            "off" => Ok(false),
            other => Err(format!("{JOINT_ROUNDS_ENV}={other} is neither on nor off")),
        },
    }
}

/// What a member's round opens to.
#[derive(Debug, Clone)]
pub enum Opened {
    /// The member's last verify already held the fed token's row: this is its step, and the
    /// member takes no row of the walk.
    Held(StepOutput),
    /// The member drafted its next chain: `chain[j]` is the chance its drafts `0..=j` are all
    /// accepted, falling with depth. An empty chain walks the anchor alone.
    Round(Vec<f64>),
}

/// An arm whose sequences draft and verify together.
pub trait RoundArm {
    /// One sequence.
    type Member;

    /// Opens `member`'s round on `token`, the token its caller fed: either the step its last
    /// verify already holds, or its next chain, drafted (the drafts stay with the member).
    ///
    /// # Errors
    /// A backend failure or an exhausted context.
    fn open(&self, member: &mut Self::Member, token: u32) -> Result<Opened, BackendError>;

    /// One walk: each member's anchor (the token it was fed) and the first `depth` drafts of the
    /// chain it opened, in the order given. Answers each member's anchor step; the member keeps
    /// its drafts' rows for the calls that feed them.
    ///
    /// # Errors
    /// A backend failure; every member of the walk fails with it.
    fn verify(
        &self,
        round: &mut [(&mut Self::Member, u32, usize)],
    ) -> Result<Vec<StepOutput>, BackendError>;

    /// What a walk of `rows` rows over `seqs` members costs, for the share.
    fn wall(&self) -> &dyn StepWall;

    /// The most drafts one walk of `seqs` members verifies in all.
    fn budget(&self, seqs: usize) -> usize;

    /// The least chance a draft must have to be given a row.
    fn floor(&self) -> f64 {
        0.0
    }
}

/// Steps every member one token: opens each round, shares the walk's rows among the chains, and
/// verifies them in one walk. Answers each member's step in the order given.
///
/// # Errors
/// The first member that fails to open, or the walk; the whole step fails with it.
pub fn run_round<A: RoundArm>(
    arm: &A,
    members: &mut [(A::Member, u32)],
) -> Result<Vec<StepOutput>, BackendError> {
    let mut answers: Vec<Option<StepOutput>> = vec![None; members.len()];
    let mut chains = Vec::with_capacity(members.len());
    let mut walking = Vec::with_capacity(members.len());
    for (index, (member, token)) in members.iter_mut().enumerate() {
        match arm.open(member, *token)? {
            Opened::Held(step) => answers[index] = Some(step),
            Opened::Round(chain) => {
                chains.push(chain);
                walking.push(index);
            }
        }
    }
    if !walking.is_empty() {
        let depths = share_rows(&chains, arm.budget(walking.len()), arm.floor(), arm.wall());
        let mut round: Vec<(&mut A::Member, u32, usize)> = members
            .iter_mut()
            .enumerate()
            .filter(|(index, _)| walking.contains(index))
            .zip(&depths)
            .map(|((_, (member, token)), &depth)| (member, *token, depth))
            .collect();
        let steps = arm.verify(&mut round)?;
        if steps.len() != walking.len() {
            return Err(BackendError::new(
                "rounds",
                format!(
                    "a walk of {} members answered {} steps",
                    walking.len(),
                    steps.len()
                ),
            ));
        }
        for (index, step) in walking.into_iter().zip(steps) {
            answers[index] = Some(step);
        }
    }
    Ok(answers.into_iter().map(Option::unwrap_or_default).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::share::ExpertUnionWall;
    use std::cell::RefCell;

    /// A toy member: the chain it drafts, the drafts it holds, and whether its next token is held.
    #[derive(Default)]
    struct Toy {
        chain: Vec<f64>,
        held: Option<u32>,
        verified: Option<usize>,
    }

    struct Arm {
        wall: ExpertUnionWall,
        walks: RefCell<Vec<Vec<(u32, usize)>>>,
    }

    impl RoundArm for Arm {
        type Member = Toy;

        fn open(&self, member: &mut Toy, token: u32) -> Result<Opened, BackendError> {
            if member.held == Some(token) {
                return Ok(Opened::Held(StepOutput {
                    next_token: Some(token + 100),
                    ..StepOutput::default()
                }));
            }
            Ok(Opened::Round(member.chain.clone()))
        }

        fn verify(
            &self,
            round: &mut [(&mut Toy, u32, usize)],
        ) -> Result<Vec<StepOutput>, BackendError> {
            self.walks.borrow_mut().push(
                round
                    .iter()
                    .map(|(_, token, depth)| (*token, *depth))
                    .collect(),
            );
            Ok(round
                .iter_mut()
                .map(|(member, token, depth)| {
                    member.verified = Some(*depth);
                    StepOutput {
                        next_token: Some(*token + 1),
                        ..StepOutput::default()
                    }
                })
                .collect())
        }

        fn wall(&self) -> &dyn StepWall {
            &self.wall
        }

        fn budget(&self, seqs: usize) -> usize {
            seqs * 3
        }
    }

    fn arm() -> Arm {
        Arm {
            wall: ExpertUnionWall {
                fixed_ms: 24.0,
                expert_ms: 1.0,
                row_ms: 0.5,
                seq_ms: 0.5,
                experts: 384,
                top_k: 8,
            },
            walks: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn one_walk_verifies_every_open_round_and_held_rows_take_none() {
        let arm = arm();
        let mut members = vec![
            (
                Toy {
                    chain: vec![0.95, 0.9, 0.85],
                    ..Toy::default()
                },
                10,
            ),
            (
                Toy {
                    held: Some(20),
                    ..Toy::default()
                },
                20,
            ),
            (
                Toy {
                    chain: vec![0.02],
                    ..Toy::default()
                },
                30,
            ),
        ];
        let steps = run_round(&arm, &mut members).unwrap();
        let tokens: Vec<_> = steps.iter().map(|step| step.next_token).collect();
        assert_eq!(tokens, [Some(11), Some(120), Some(31)]);
        let walks = arm.walks.borrow();
        assert_eq!(walks.len(), 1, "one walk for every open round");
        // The likely chain takes its drafts, the unlikely one walks its anchor alone, and the held
        // member is not in the walk.
        assert_eq!(walks[0], [(10, 3), (30, 0)]);
        assert_eq!(members[1].0.verified, None);
    }

    #[test]
    fn a_step_of_held_rows_walks_nothing() {
        let arm = arm();
        let mut members = vec![(
            Toy {
                held: Some(5),
                ..Toy::default()
            },
            5,
        )];
        let steps = run_round(&arm, &mut members).unwrap();
        assert_eq!(steps[0].next_token, Some(105));
        assert!(arm.walks.borrow().is_empty());
    }
}
