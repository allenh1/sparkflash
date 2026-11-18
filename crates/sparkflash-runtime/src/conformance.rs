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

use sparkflash_model::backend::BackendError;

use crate::layout::StateKind;
use crate::walk::{PassMask, Segments, Walk};

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
}

impl ConformanceCheck {
    /// Every check [`conformance!`] emits, in the order it emits them.
    pub const ALL: [Self; 6] = [
        Self::WholePromptEqualsChunked,
        Self::AloneEqualsFour,
        Self::ContinuedEqualsCold,
        Self::RewoundEqualsCold,
        Self::TpWithinExactness,
        Self::NoUnrecordedWidening,
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
        }
    }
}

/// The const list `conformance!(Arm)` claims, checkable without a device.
pub const CONFORMANCE_CHECKS: &[ConformanceCheck] = &ConformanceCheck::ALL;

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

/// Runs one generated check. Until the fixture runner lands, this says what the check needs;
/// the generated tests are `#[ignore]`d so a GPU-less host still compiles and passes.
///
/// # Errors
/// The check needs the arm's device fixture; the message names it.
pub fn run_check<W: Walk>(check: ConformanceCheck) -> Result<(), BackendError> {
    check_claims::<W>(&[check]).map_err(|why| BackendError::new("conformance", why))?;
    Err(BackendError::new(
        "conformance",
        format!(
            "{} needs {}'s device fixture; run scripts/onboard-gate.sh <arm>",
            check.name(),
            std::any::type_name::<W>()
        ),
    ))
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
            pub const CHECKS: &[$crate::ConformanceCheck] = $crate::CONFORMANCE_CHECKS;

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
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_check_has_a_name_and_there_are_six() {
        assert_eq!(ConformanceCheck::ALL.len(), 6);
        for check in ConformanceCheck::ALL {
            assert!(!check.name().is_empty());
        }
        assert_eq!(CONFORMANCE_CHECKS, &ConformanceCheck::ALL);
    }
}
