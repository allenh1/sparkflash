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
//! unrecorded widening. The macro emits them `#[ignore]`d, because they boot the arm's fixture,
//! and a const list of the checks the arm claims. The token is a compile gate: an arm that
//! invokes it is on the conformance path; the list is checkable on a GPU-less host.
//!
//! # The fixture contract
//!
//! An arm supplies [`WalkFixture`]: `fixture(world)` boots a walk the checks can drive with no
//! served server. A CPU arm returns the toy walk it already has. A device arm returns its
//! synthetic checkpoint: GLM and Qwen write their tiny fixture, Inkling uses [`TinySource`],
//! dense reads the path `scripts/dense-synthetic-checkpoint.py` writes. A fixture the host
//! cannot supply is an error whose code is `fixture` and whose message names what is missing;
//! it never silently passes. A world with no group is an error whose code is `group`, which the
//! TP check reports as a named skip.
//!
//! [`TinySource`]: https://docs.rs/sparkflash-model-inkling2
//!
//! Every check in [`ConformanceCheck::ALL`] must be supported by the arm's declared
//! [`WalkCapabilities`]; [`check_claims`] enforces that without booting anything.

use std::any::Any;

use sparkflash_model::backend::{BackendError, StepOutput};

use crate::layout::StateKind;
use crate::pool::LaneMemory;
use crate::runtime::{Runtime, RuntimeSlot};
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

/// The checks an arm without a drafter claims: every check but [`ConformanceCheck::DraftedEqualsPlain`].
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

/// One stored-format class the fixture states, and the arithmetic the walk runs it in.
///
/// A class whose arithmetic differs from its stored format must carry a record; the check
/// refuses an unrecorded widening. The names are the ledger's, not a dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatRule {
    /// The format the checkpoint stores the class in.
    pub stored: &'static str,
    /// The format the walk multiplies it in.
    pub arithmetic: &'static str,
    /// Where a widening is recorded, or `None` when the two formats are equal.
    pub record: Option<&'static str>,
}

impl FormatRule {
    /// A class kept in its stored format.
    #[must_use]
    pub const fn stored(format: &'static str) -> Self {
        Self {
            stored: format,
            arithmetic: format,
            record: None,
        }
    }
}

/// What an arm's fixture is, values the checks read without opening it.
pub struct Fixture<W: Walk> {
    /// The walk the checks drive.
    pub walk: W,
    /// Where the walk came from, printed in the gate's report.
    pub source: String,
    /// The bytes of stored weight a token pass reads in this fixture.
    pub bytes_per_token: u64,
    /// The stored-bytes floor the fixture's plan states for a token pass.
    pub floor_bytes_per_token: u64,
    /// Every stored-format class the fixture's plan carries.
    pub formats: Vec<FormatRule>,
    /// Files, sources or a device handle the walk borrows from, kept alive beside it.
    pub keep: Option<Box<dyn Any>>,
}

impl<W: Walk> Fixture<W> {
    /// A fixture of `walk` from `source`, with the token-pass bytes and format classes.
    #[must_use]
    pub fn new(
        walk: W,
        source: impl Into<String>,
        bytes_per_token: u64,
        floor_bytes_per_token: u64,
        formats: Vec<FormatRule>,
        keep: Option<Box<dyn Any>>,
    ) -> Self {
        Self {
            walk,
            source: source.into(),
            bytes_per_token,
            floor_bytes_per_token,
            formats,
            keep,
        }
    }
}

/// The fixture contract: the arm supplies a walk the generated checks can boot with no server.
///
/// A fixture the host cannot supply is [`BackendError`] with code `fixture` and a message that
/// names what it needs. A fixture that exists only in a group is code `group`; the TP check
/// reports that as a named skip rather than a pass.
pub trait WalkFixture: Walk {
    /// Boots the arm at `world` ranks and returns the walk with what its checks need.
    ///
    /// # Errors
    /// Code `fixture` when the arm's fixture is not present on this host; code `group` when the
    /// fixture needs ranks this host does not have.
    fn fixture(world: usize) -> Result<Fixture<Self>, BackendError>
    where
        Self: Sized;
}

/// The answer one generated check gives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// The check ran and held.
    Passed,
    /// The check could not run on this host, for the named reason.
    Skipped(String),
}

/// Runs one generated check.
///
/// Every check boots the arm's [`WalkFixture`]. `drafted = plain` needs no device when the arm's
/// fixture is a CPU walk: it drives the same prompt through the runtime's drafted step and
/// through a plain token loop, and requires the same tokens, rejected draft included.
///
/// # Errors
/// The arm's fixture is missing (code `fixture`), or a check ran and the arm disagreed.
pub fn run_check<W: WalkFixture>(check: ConformanceCheck) -> Result<CheckOutcome, BackendError> {
    if check == ConformanceCheck::DraftedEqualsPlain && !W::Drafter::PRESENT {
        return Ok(CheckOutcome::Skipped(
            "the arm declares no drafter".to_owned(),
        ));
    }
    check_claims::<W>(&[check]).map_err(|why| BackendError::new("conformance", why))?;
    match check {
        ConformanceCheck::WholePromptEqualsChunked => whole_prompt_equals_chunked::<W>(),
        ConformanceCheck::AloneEqualsFour => alone_equals_four::<W>(),
        ConformanceCheck::ContinuedEqualsCold => continued_equals_cold::<W>(),
        ConformanceCheck::RewoundEqualsCold => rewound_equals_cold::<W>(),
        ConformanceCheck::TpWithinExactness => tp_within_exactness::<W>(),
        ConformanceCheck::NoUnrecordedWidening => no_unrecorded_widening::<W>(),
        ConformanceCheck::DraftedEqualsPlain => drafted_equals_plain::<W>(),
    }
}

/// Runs one check, prints its outcome in the gate's line format, and panics on a failure.
///
/// The generated tests call this, so `onboard-gate.sh` can report per check pass, fail, skip or
/// needs-fixture from the test output instead of treating an `#[ignore]`d test as a pass.
pub fn run_generated<W: WalkFixture>(check: ConformanceCheck) {
    let name = check.name();
    match run_check::<W>(check) {
        Ok(CheckOutcome::Passed) => {
            println!("conformance: {name}: PASS");
        }
        Ok(CheckOutcome::Skipped(reason)) => {
            println!("conformance: {name}: SKIP: {reason}");
        }
        Err(error) if error.code() == "fixture" => {
            println!("conformance: {name}: NEEDS-FIXTURE: {}", error.message());
            panic!("conformance: {name} needs a fixture: {}", error.message());
        }
        Err(error) => {
            println!("conformance: {name}: FAIL: {}", error.message());
            panic!("conformance: {name} failed: {}", error.message());
        }
    }
}

/// How many tokens a check compares.
const CONFORMANCE_TOKENS: usize = 5;

/// The prompt the checks start from.
const CONFORMANCE_PROMPT: [u32; 7] = [7, 9, 11, 13, 15, 17, 19];

/// A second prompt, for the four-at-once check.
const CONFORMANCE_OTHER: [u32; 7] = [23, 29, 31, 37, 41, 43, 47];

/// A third and fourth prompt, for the four-at-once check.
const CONFORMANCE_THIRD: [u32; 7] = [3, 5, 11, 17, 23, 29, 31];

/// A fourth prompt, for the four-at-once check.
const CONFORMANCE_FOURTH: [u32; 7] = [2, 3, 5, 7, 11, 13, 17];

/// The token run a prompt is fed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptMode {
    /// One bulk pass over every prompt row.
    Whole,
    /// Two bulk passes split in the middle.
    Chunked,
    /// One token row a pass.
    Token,
}

/// Whole prompt = chunked = token by token, with the same greedy tokens after the prompt.
fn whole_prompt_equals_chunked<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let fixture = W::fixture(1)?;
    let whole = direct_run(&fixture.walk, &CONFORMANCE_PROMPT, PromptMode::Whole)?;
    let chunked = direct_run(&fixture.walk, &CONFORMANCE_PROMPT, PromptMode::Chunked)?;
    let token = direct_run(&fixture.walk, &CONFORMANCE_PROMPT, PromptMode::Token)?;
    if whole != chunked {
        return Err(disagreement(
            "whole prompt = chunked",
            &whole,
            &chunked,
            &fixture.source,
        ));
    }
    if whole != token {
        return Err(disagreement(
            "chunked = token by token",
            &whole,
            &token,
            &fixture.source,
        ));
    }
    println!(
        "conformance: whole prompt = chunked = token by token: {} tokens over {}",
        whole.len(),
        fixture.source
    );
    Ok(CheckOutcome::Passed)
}

/// One request alone and four in one runtime answer the same tokens.
fn alone_equals_four<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let fixture = W::fixture(1)?;
    let prompts = [
        CONFORMANCE_PROMPT,
        CONFORMANCE_OTHER,
        CONFORMANCE_THIRD,
        CONFORMANCE_FOURTH,
    ];
    let runtime = Runtime::new(fixture.walk, 0)?;
    let mut alone = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        alone.push(runtime_run(&runtime, prompt, CONFORMANCE_TOKENS)?);
    }
    let together = runtime_four(&runtime, &prompts, CONFORMANCE_TOKENS)?;
    for (index, (single, four)) in alone.iter().zip(&together).enumerate() {
        if single != four {
            return Err(disagreement(
                &format!("alone = four at once (member {index})"),
                single,
                four,
                &fixture.source,
            ));
        }
    }
    println!(
        "conformance: alone = four at once: four members agree over {}",
        fixture.source
    );
    Ok(CheckOutcome::Passed)
}

/// A continued turn's tokens equal a cold turn's.
fn continued_equals_cold<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let fixture = W::fixture(1)?;
    let runtime = Runtime::new(fixture.walk, 0)?;
    let cold = runtime_run(&runtime, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
    let mut slot = runtime.open_slot()?;
    let prefix = CONFORMANCE_PROMPT[..4].to_vec();
    runtime.prefill_slot(&mut slot, &prefix)?;
    let first = runtime.prefill_slot(&mut slot, &CONFORMANCE_PROMPT)?;
    if slot.history() != CONFORMANCE_PROMPT {
        return Err(BackendError::new(
            "conformance",
            format!(
                "continued turn = cold: the runtime kept {:?}, not the prompt",
                slot.history()
            ),
        ));
    }
    let continued = decode_slot(&runtime, slot, first, CONFORMANCE_TOKENS)?;
    if cold != continued {
        return Err(disagreement(
            "continued turn = cold",
            &cold,
            &continued,
            &fixture.source,
        ));
    }
    println!(
        "conformance: continued turn = cold: {} tokens over {}",
        cold.len(),
        fixture.source
    );
    Ok(CheckOutcome::Passed)
}

/// A rewound turn's tokens equal a cold turn's.
fn rewound_equals_cold<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let fixture = W::fixture(1)?;
    let runtime = Runtime::new(fixture.walk, 0)?;
    let changed = [7, 9, 11, 13, 15, 17, 23];
    let cold = runtime_run(&runtime, &changed, CONFORMANCE_TOKENS)?;
    let mut slot = runtime.open_slot()?;
    let mut first_turn = CONFORMANCE_PROMPT;
    *first_turn.last_mut().expect("a prompt") = 29;
    runtime.prefill_slot(&mut slot, &first_turn)?;
    let first = runtime.prefill_slot(&mut slot, &changed)?;
    if slot.history() != changed {
        return Err(BackendError::new(
            "conformance",
            format!(
                "rewound turn = cold: the runtime kept {:?}, not the changed prompt",
                slot.history()
            ),
        ));
    }
    let rewound = decode_slot(&runtime, slot, first, CONFORMANCE_TOKENS)?;
    if cold != rewound {
        return Err(disagreement(
            "rewound turn = cold",
            &cold,
            &rewound,
            &fixture.source,
        ));
    }
    println!(
        "conformance: rewound turn = cold: {} tokens over {}",
        cold.len(),
        fixture.source
    );
    Ok(CheckOutcome::Passed)
}

/// TP 1 = TP 2 = TP 4, or a named skip when the host has no group.
fn tp_within_exactness<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let one = W::fixture(1)?;
    let runtime = Runtime::new(one.walk, 0)?;
    let baseline = runtime_run(&runtime, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
    let mut compared = vec![1usize];
    for world in [2usize, 4] {
        let fixture = match W::fixture(world) {
            Ok(fixture) => fixture,
            Err(error) if error.code() == "group" => {
                return Ok(CheckOutcome::Skipped(format!(
                    "TP {world}: {}",
                    error.message()
                )));
            }
            Err(error) => return Err(error),
        };
        let runtime = Runtime::new(fixture.walk, 0)?;
        let tokens = runtime_run(&runtime, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
        if tokens != baseline {
            return Err(disagreement(
                &format!("TP 1 = TP {world}"),
                &baseline,
                &tokens,
                &fixture.source,
            ));
        }
        compared.push(world);
    }
    println!("conformance: TP within exactness: world(s) {compared:?} agree");
    Ok(CheckOutcome::Passed)
}

/// No stored format widened without a record, and the token pass reads at least its floor.
fn no_unrecorded_widening<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let fixture = W::fixture(1)?;
    let stored = W::CAPABILITIES.compute.stored;
    if !stored.runs() {
        return Err(BackendError::new(
            "conformance",
            format!(
                "no unrecorded widening: {} declares stored-format arithmetic {stored}",
                std::any::type_name::<W>()
            ),
        ));
    }
    for rule in &fixture.formats {
        if rule.stored != rule.arithmetic && rule.record.is_none() {
            return Err(BackendError::new(
                "conformance",
                format!(
                    "no unrecorded widening: {} runs {} as {} with no record",
                    std::any::type_name::<W>(),
                    rule.stored,
                    rule.arithmetic
                ),
            ));
        }
    }
    if fixture.floor_bytes_per_token == 0 {
        return Err(BackendError::new(
            "conformance",
            format!(
                "no unrecorded widening: {}'s fixture states no bytes-per-token floor",
                std::any::type_name::<W>()
            ),
        ));
    }
    if fixture.bytes_per_token < fixture.floor_bytes_per_token {
        return Err(BackendError::new(
            "conformance",
            format!(
                "no unrecorded widening: a token reads {} bytes, under the {} floor, for {}",
                fixture.bytes_per_token, fixture.floor_bytes_per_token, fixture.source
            ),
        ));
    }
    println!(
        "conformance: no unrecorded widening: {} classes, {} B/token against a {} B floor over {}",
        fixture.formats.len(),
        fixture.bytes_per_token,
        fixture.floor_bytes_per_token,
        fixture.source
    );
    Ok(CheckOutcome::Passed)
}

/// The drafted-equals-plain check over the arm's fixture.
///
/// The drafted side is [`Runtime`]'s step: a prompt, then a fed token a step. The plain side is
/// a second fixture driven a row at a time through [`Walk::walk`] and [`Walk::head`]. The two
/// token runs must be equal, which is the property a rejected draft can break.
fn drafted_equals_plain<W: WalkFixture>() -> Result<CheckOutcome, BackendError> {
    let drafted = W::fixture(1)?;
    let plain = W::fixture(1)?;
    let expected = plain_run(&plain.walk, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
    let runtime = Runtime::new(drafted.walk, 0)?;
    let actual = runtime_run(&runtime, &CONFORMANCE_PROMPT, CONFORMANCE_TOKENS)?;
    if actual != expected {
        return Err(disagreement(
            "drafted = plain",
            &expected,
            &actual,
            &drafted.source,
        ));
    }
    println!(
        "conformance: drafted = plain: {} tokens over {}",
        actual.len(),
        drafted.source
    );
    Ok(CheckOutcome::Passed)
}

/// A failure that names the two token runs.
fn disagreement(
    what: &str,
    left: &[Option<u32>],
    right: &[Option<u32>],
    source: &str,
) -> BackendError {
    BackendError::new(
        "conformance",
        format!("{what} failed over {source}: left {left:?}, right {right:?}"),
    )
}

/// The plain token run: a prompt in `mode`, then a token row a step, headed directly.
fn direct_run<W: Walk>(
    walk: &W,
    prompt: &[u32],
    mode: PromptMode,
) -> Result<Vec<Option<u32>>, BackendError> {
    let mut seq = walk.open(LaneMemory::new(0, 0))?;
    let hidden = match mode {
        PromptMode::Whole => walk.walk(
            &mut [Segment::new(&mut seq, prompt)],
            Pass::bulk(prompt.len()).with_prompt_end(0),
        )?,
        PromptMode::Chunked => {
            let middle = (prompt.len() / 2).max(1).min(prompt.len());
            let (head, tail) = prompt.split_at(middle);
            let mut hidden = walk.walk(
                &mut [Segment::new(&mut seq, head)],
                Pass::bulk(head.len()).with_prompt_end(tail.len()),
            )?;
            if !tail.is_empty() {
                hidden = walk.walk(
                    &mut [Segment::new(&mut seq, tail)],
                    Pass::bulk(tail.len()).with_prompt_end(0),
                )?;
            }
            hidden
        }
        PromptMode::Token => {
            let mut hidden = Hidden::new(0);
            for (index, token) in prompt.iter().enumerate() {
                let last = index + 1 == prompt.len();
                hidden = walk.walk(
                    &mut [Segment::new(&mut seq, std::slice::from_ref(token))],
                    Pass::token_rows(),
                )?;
                if last {
                    break;
                }
            }
            hidden
        }
    };
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
    while run.len() < CONFORMANCE_TOKENS {
        let Some(fed) = token else {
            break;
        };
        let hidden = walk.walk(
            &mut [Segment::new(&mut seq, std::slice::from_ref(&fed))],
            Pass::token_rows(),
        )?;
        let step = head_one(walk, &mut seq, &hidden, 0, Some(fed), &[])?;
        token = step.next_token;
        run.push(token);
    }
    Ok(run)
}

/// The plain token run on a second walk, for the drafted comparison.
fn plain_run<W: Walk>(
    walk: &W,
    prompt: &[u32],
    tokens: usize,
) -> Result<Vec<Option<u32>>, BackendError> {
    let mut seq = walk.open(LaneMemory::new(0, 0))?;
    let hidden = walk.walk(
        &mut [Segment::new(&mut seq, prompt)],
        Pass::bulk(prompt.len()).with_prompt_end(0),
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
        let step = head_one(walk, &mut seq, &hidden, 0, Some(fed), &[])?;
        token = step.next_token;
        run.push(token);
    }
    Ok(run)
}

/// One head row, for a plain run.
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

/// One prompt through the runtime, then `tokens` greedy tokens.
fn runtime_run<W: Walk>(
    runtime: &Runtime<W>,
    prompt: &[u32],
    tokens: usize,
) -> Result<Vec<Option<u32>>, BackendError> {
    let mut slot = runtime.open_slot()?;
    let first = runtime.prefill_slot(&mut slot, prompt)?;
    decode_slot(runtime, slot, first, tokens)
}

/// `tokens` greedy tokens from an already prompted slot.
fn decode_slot<W: Walk>(
    runtime: &Runtime<W>,
    slot: RuntimeSlot<W>,
    first: StepOutput,
    tokens: usize,
) -> Result<Vec<Option<u32>>, BackendError> {
    let mut run = vec![first.next_token];
    let mut member = [(slot, first.next_token.unwrap_or_default())];
    while run.len() < tokens {
        let steps = runtime.step_slots(&mut member)?;
        let next = steps.first().and_then(|step| step.next_token);
        run.push(next);
        let Some(token) = next else {
            break;
        };
        member[0].1 = token;
    }
    Ok(run)
}

/// Four prompts prompted one after another, then stepped together, one token a round.
fn runtime_four<W: Walk>(
    runtime: &Runtime<W>,
    prompts: &[[u32; 7]],
    tokens: usize,
) -> Result<Vec<Vec<Option<u32>>>, BackendError> {
    let mut members = Vec::with_capacity(prompts.len());
    let mut runs: Vec<Vec<Option<u32>>> = vec![Vec::new(); prompts.len()];
    for (index, prompt) in prompts.iter().enumerate() {
        let mut slot = runtime.open_slot()?;
        let first = runtime.prefill_slot(&mut slot, prompt)?;
        runs[index].push(first.next_token);
        members.push((slot, first.next_token.unwrap_or_default()));
    }
    while runs[0].len() < tokens {
        let steps = runtime.step_slots(&mut members)?;
        for (index, step) in steps.iter().enumerate() {
            runs[index].push(step.next_token);
        }
        for ((_, token), step) in members.iter_mut().zip(&steps) {
            if let Some(next) = step.next_token {
                *token = next;
            }
        }
    }
    Ok(runs)
}

/// Emit the generated conformance checks for `Arm`: an `#[ignore]`d test a check, and the
/// const list [`check_claims`] can read on a CPU-only host.
///
/// The invocation is the compile gate. The module it creates is called `conformance`, so one
/// arm crate invokes it once, in a test module. The arm must implement [`WalkFixture`] beside
/// the invocation.
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
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn whole_prompt_equals_chunked() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::WholePromptEqualsChunked);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn alone_equals_four() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::AloneEqualsFour);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn continued_equals_cold() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::ContinuedEqualsCold);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn rewound_equals_cold() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::RewoundEqualsCold);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn tp_one_two_four() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::TpWithinExactness);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn no_unrecorded_widening() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::NoUnrecordedWidening);
            }

            #[test]
            #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh <arm>"]
            fn drafted_equals_plain() {
                $crate::run_generated::<$arm>($crate::ConformanceCheck::DraftedEqualsPlain);
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

    #[test]
    fn a_format_rule_names_an_unrecorded_widening() {
        let rule = FormatRule {
            stored: "FP8 E4M3",
            arithmetic: "BF16",
            record: None,
        };
        assert_ne!(rule.stored, rule.arithmetic);
        let recorded = FormatRule {
            record: Some("docs/engine/04-onboarding.md"),
            ..rule
        };
        assert!(recorded.record.is_some());
        assert_eq!(FormatRule::stored("BF16").arithmetic, "BF16");
    }
}
