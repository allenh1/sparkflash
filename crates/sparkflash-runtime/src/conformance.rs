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

//! The generated arm checks, and the token that says an arm has them.
//!
//! [`conformance!`] expands to the checks every arm must have, lifted from Qwen's device tests:
//! whole prompt = chunked = token by token, alone = four at once, a continued turn = cold, a
//! rewound turn = cold, TP 1 = TP 2 = TP 4 within the arm's declared exactness, and no
//! unrecorded widening. The macro emits them `#[ignore]`d, because they need the arm's fixture
//! and a device, and a const list of the checks the arm claims. The token is a compile gate:
//! an arm that invokes it is on the conformance path; the list is checkable on a GPU-less host.
//!
//! Every check in [`ConformanceCheck::ALL`] must be supported by the arm's declared
//! [`WalkCapabilities`]; [`check_claims`] enforces that without booting anything. A check that
//! needs a device returns an operator-facing error from [`run_check`] until it is run under
//! `scripts/onboard-gate.sh`.

use sparkflash_model::backend::{BackendError, StepOutput};

use crate::layout::StateKind;
use crate::pool::LaneMemory;
use crate::runtime::Runtime;
use crate::walk::{Drafter, HeadRow, Hidden, Pass, PassMask, Segment, Segments, Walk};

/// One check the generated arm tests cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConformanceCheck {
    /// A prompt run whole, in chunks and a token at a time leaves the same state and tokens.
    WholePromptEqualsChunked,
    /// One request alone and four in one walk answer the same tokens.
    AloneEqualsFour,
    /// A continued turn's state equals a cold turn's.
    ContinuedEqualsCold,
    /// A rewound turn's state equals a cold turn's.
    RewoundEqualsCold,
    /// TP 1, TP 2 and TP 4 agree within the arm's declared exactness.
    TpWithinExactness,
    /// No format widened without a record, and bytes per token within the floor.
    NoUnrecordedWidening,
    /// A drafting arm answers the same tokens with and without a draft, including when the
    /// draft is rejected.
    DraftedEqualsPlain,
}

impl ConformanceCheck {
    /// Every check [`conformance!`] emits, in the order it emits them.
    pub const ALL: [Self; 7] = [
        Self::WholePromptEqualsChunked,
        Self::AloneEqualsFour,
        Self::ContinuedEqualsCold,
        Self::RewoundEqualsCold,
        Self::TpWithinExactness,
        Self::NoUnrecordedWidening,
        Self::DraftedEqualsPlain,
    ];

    /// The name the list and a failure print.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::WholePromptEqualsChunked => "whole prompt = chunked = token by token",
            Self::AloneEqualsFour => "alone = four at once",
            Self::ContinuedEqualsCold => "continued turn = cold",
            Self::RewoundEqualsCold => "rewound turn = cold",
            Self::TpWithinExactness => "TP 1 = TP 2 = TP 4 within exactness",
            Self::NoUnrecordedWidening => "no unrecorded widening",
            Self::DraftedEqualsPlain => "drafted = plain",
        }
    }
}

/// Every generated check, including the one only a drafting arm can claim.
pub const CONFORMANCE_CHECKS: &[ConformanceCheck] = &ConformanceCheck::ALL;

/// The checks an arm without a drafter claims: every check but [ConformanceCheck::DraftedEqualsPlain].
const NON_DRAFTING_CHECKS: [ConformanceCheck; 6] = [
    ConformanceCheck::WholePromptEqualsChunked,
    ConformanceCheck::AloneEqualsFour,
    ConformanceCheck::ContinuedEqualsCold,
    ConformanceCheck::RewoundEqualsCold,
    ConformanceCheck::TpWithinExactness,
    ConformanceCheck::NoUnrecordedWidening,
];

/// The checks `W` claims: every one, plus drafted-equals-plain when the arm declares a drafter.
///
/// This is the const the `conformance!` macro publishes, so a dense-style arm does not fail a
/// check it has no drafter to run, and a drafting arm cannot leave the check out.
#[must_use]
pub const fn checks_for<W: Walk + ?Sized>() -> &'static [ConformanceCheck] {
    if W::Drafter::PRESENT {
        CONFORMANCE_CHECKS
    } else {
        &NON_DRAFTING_CHECKS
    }
}

/// Whether an arm's declared capabilities can support every check it claims.
///
/// This reads [`Walk::CAPABILITIES`] only: no device, no fixture, no instance.
///
/// # Errors
/// A check the arm claims but its walk cannot take, naming the missing capability.
pub fn check_claims<W: Walk>(claims: &[ConformanceCheck]) -> Result<(), String> {
    let caps = W::CAPABILITIES;
    for &check in claims {
        let supported = match check {
            ConformanceCheck::WholePromptEqualsChunked => caps.passes.contains(PassMask::BULK),
            ConformanceCheck::AloneEqualsFour => caps.segments == Segments::Many,
            ConformanceCheck::ContinuedEqualsCold | ConformanceCheck::RewoundEqualsCold => {
                caps.state != StateKind::None
            }
            ConformanceCheck::TpWithinExactness | ConformanceCheck::NoUnrecordedWidening => true,
            ConformanceCheck::DraftedEqualsPlain => W::Drafter::PRESENT,
        };
        if !supported {
            return Err(format!(
                "{} claims {:?}, which its walk does not declare",
                std::any::type_name::<W>(),
                check
            ));
        }
    }
    Ok(())
}

/// Runs one generated check.
///
/// [`ConformanceCheck::DraftedEqualsPlain`] runs with no device when the walk supplies one
/// through [`Walk::cpu`]: it drives the same prompt through the runtime's drafted step and
/// through a plain token loop, and requires the same tokens, rejected draft included. The other
/// checks still need the arm's device fixture; the generated tests are `#[ignore]`d so a
/// GPU-less host still compiles and passes.
///
/// # Errors
/// The check needs the arm's device fixture, or the drafted run disagrees with the plain one.
pub fn run_check<W: Walk>(check: ConformanceCheck) -> Result<(), BackendError> {
    check_claims::<W>(&[check]).map_err(|why| BackendError::new("conformance", why))?;
    if check == ConformanceCheck::DraftedEqualsPlain
        && let Some(drafted) = W::cpu()
    {
        return drafted_equals_plain::<W>(drafted);
    }
    Err(BackendError::new(
        "conformance",
        format!(
            "{} needs {}'s device fixture; run scripts/onboard-gate.sh <arm>",
            check.name(),
            std::any::type_name::<W>()
        ),
    ))
}

/// How many tokens [`run_check`] compares before it calls a drafted run plain.
const CONFORMANCE_TOKENS: usize = 6;

/// A prompt the toy walks can start from.
const CONFORMANCE_PROMPT: [u32; 3] = [7, 9, 11];

/// The drafted-equals-plain check over a walk the runtime can drive with no device.
///
/// The drafted side is [`Runtime`]'s step: a prompt, then a fed token a step. The plain side is
/// the same walk driven a row at a time through [`Walk::walk`] and [`Walk::head`]. The two
/// token runs must be equal, which is the property a rejected draft can break.
fn drafted_equals_plain<W: Walk>(drafted: W) -> Result<(), BackendError> {
    let plain = W::cpu().ok_or_else(|| {
        BackendError::new("conformance", "the walk supplies only one CPU instance")
    })?;
    let expected = plain_run(&plain, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
    let runtime = Runtime::new(drafted, 0)?;
    let mut slot = runtime.open_slot()?;
    let first = runtime.prefill_slot(&mut slot, &CONFORMANCE_PROMPT)?;
    let mut actual = vec![first.next_token];
    let mut member = [(slot, first.next_token.unwrap_or_default())];
    while actual.len() < CONFORMANCE_TOKENS {
        let steps = runtime.step_slots(&mut member)?;
        let next = steps.first().and_then(|step| step.next_token);
        actual.push(next);
        let Some(token) = next else {
            break;
        };
        member[0].1 = token;
    }
    if actual != expected {
        return Err(BackendError::new(
            "conformance",
            format!(
                "drafted = plain failed for {}: drafted {actual:?}, plain {expected:?}",
                std::any::type_name::<W>()
            ),
        ));
    }
    Ok(())
}

/// The plain token run: one bulk prompt pass, then a token row a step, headed directly.
fn plain_run<W: Walk>(
    walk: &W,
    prompt: &[u32],
    tokens: usize,
) -> Result<Vec<Option<u32>>, BackendError> {
    let mut seq = walk.open(LaneMemory::new(0, 0))?;
    let hidden = walk.walk(
        &mut [Segment::new(&mut seq, prompt)],
        Pass::bulk(prompt.len()),
    )?;
    let step = head_one(
        walk,
        &mut seq,
        &hidden,
        hidden.rows.saturating_sub(1),
        None,
        prompt,
    )?;
    let mut run = vec![step.next_token];
    let mut token = step.next_token;
    while run.len() < tokens {
        let Some(fed) = token else {
            break;
        };
        let hidden = walk.walk(
            &mut [Segment::new(&mut seq, std::slice::from_ref(&fed))],
            Pass::token_rows(),
        )?;
        let step = head_one(walk, &mut seq, &hidden, 0, Some(fed), prompt)?;
        token = step.next_token;
        run.push(token);
    }
    Ok(run)
}

/// One head row, for the plain run.
fn head_one<W: Walk>(
    walk: &W,
    seq: &mut W::Seq,
    hidden: &Hidden,
    row: usize,
    token: Option<u32>,
    prompt: &[u32],
) -> Result<StepOutput, BackendError> {
    let mut rows = [match token {
        Some(token) => HeadRow::token(seq, row, token),
        None => HeadRow::prompt(seq, row, prompt),
    }];
    let mut steps = walk.head(hidden, &mut rows)?;
    steps
        .pop()
        .ok_or_else(|| BackendError::new("conformance", "the head answered no step"))
}

/// Emit the generated conformance checks for `Arm`: an `#[ignore]`d test a check, and the
/// const list [`check_claims`] can read on a CPU-only host.
///
/// The invocation is the compile gate. The module it creates is called `conformance`, so one
/// arm crate invokes it once, in a test module.
///
/// ```ignore
/// sparkflash_runtime::conformance!(MyWalk);
/// ```
#[macro_export]
macro_rules! conformance {
    ($arm:ty) => {
        /// The generated checks and the list this arm claims.
        pub mod conformance {
            use super::*;

            /// The arm this module's checks were generated for.
            pub const ARM: &str = stringify!($arm);

            /// The checks the arm claims; checkable without a device.
            pub const CHECKS: &[$crate::ConformanceCheck] = $crate::checks_for::<$arm>();

            #[test]
            fn the_arm_claims_every_generated_check() {
                $crate::check_claims::<$arm>(CHECKS).unwrap_or_else(|why| panic!("{}: {why}", ARM));
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn whole_prompt_equals_chunked() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::WholePromptEqualsChunked)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn alone_equals_four() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::AloneEqualsFour)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn continued_equals_cold() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::ContinuedEqualsCold)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn rewound_equals_cold() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::RewoundEqualsCold)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn tp_one_two_four() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::TpWithinExactness)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn no_unrecorded_widening() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::NoUnrecordedWidening)
                    .expect("the check needs a device");
            }

            #[test]
            #[ignore = "needs the arm's device fixture; run scripts/onboard-gate.sh <arm>"]
            fn drafted_equals_plain() {
                $crate::run_check::<$arm>($crate::ConformanceCheck::DraftedEqualsPlain)
                    .expect("the check needs a device");
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_check_has_a_name_and_there_are_seven() {
        assert_eq!(ConformanceCheck::ALL.len(), 7);
        for check in ConformanceCheck::ALL {
            assert!(!check.name().is_empty());
        }
        assert_eq!(CONFORMANCE_CHECKS, &ConformanceCheck::ALL);
    }
}
