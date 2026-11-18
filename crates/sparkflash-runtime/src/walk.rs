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

//! The one shape every model arm supplies.
//!
//! An arm states what its state is ([`Walk::layout`]), opens a sequence
//! ([`Walk::open`]), says how wide a pass may be ([`Walk::limits`]), runs rows
//! ([`Walk::walk`]) and turns the last rows into tokens ([`Walk::head`]). Every strategy
//! decisions - which rows share a pass, when a prompt is chunked, when state is reused or
//! rewound, how a group mirrors, what the ledger says - belongs to [`crate::Runtime`].
//!
//! [`Pass`] is fully specified here, including the shapes only DeepSeek's tick needs yet:
//! token rows, a bulk prompt chunk, the paired chunks of several prompts, and verify rows under
//! an output constraint. The mid-walk hook is part of the pass too, because its placement is the
//! one load-bearing detail of tick-ahead: a hook after the last layer and before the head has
//! the residual rows to decide with, and a hook anywhere else does not.
//!
//! The trait carries no `Send` or `Sync` bound. DeepSeek's sequence is single-threaded; an
//! engine that needs the bounds asks for them at its own implementation.

use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::ledger::State;

use crate::layout::{LaneState, Limits, StateKind, StateLayout};
use crate::pool::LaneMemory;

/// What a walk is for.
///
/// The rows themselves are in the [`Segment`]s; this says how the arm should treat them. A
/// walk may ignore the distinction (dense's does for rows), but it must not mistake a verify
/// pass for a bulk prompt: the output constraint is where an arm reads how many tokens the
/// caller will keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    /// One token a sequence, one sequence a member: a decode step, or a verify's anchor rows.
    TokenRows,
    /// One prompt's chunk, as many rows as the engine's chunk policy gave it.
    Bulk {
        /// The number of rows the chunk holds.
        rows: usize,
    },
    /// The chunks of several prompts in one pass: each segment is one prompt's chunk.
    Paired {
        /// The number of chunks (segments) in the pass.
        chunks: usize,
    },
    /// Draft rows verified against an output constraint: the caller has fixed some tokens and
    /// will keep at most [`OutputConstraint::max_rows`] of the pass's outputs.
    Verify {
        /// The number of rows the pass holds.
        rows: usize,
    },
}

/// What the caller already fixed about a verify pass's output.
///
/// A drafter's verify walk exists to accept a prefix of a draft and no more. An arm that can
/// enforce the constraint in the pass does; one that cannot still runs the rows and lets the
/// runtime drop the rest. The fields are deliberately values, not a callback: the ledger and a
/// replay must see the same constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputConstraint<'a> {
    /// Tokens the caller has already fixed, one a verify row, in row order.
    pub fixed: &'a [u32],
    /// The most rows whose tokens the caller will keep.
    pub max_rows: usize,
}

impl<'a> OutputConstraint<'a> {
    /// A constraint that keeps at most `max_rows` rows and fixes `fixed`.
    #[must_use]
    pub const fn new(fixed: &'a [u32], max_rows: usize) -> Self {
        Self { fixed, max_rows }
    }
}

/// A hook between the last layer and the head.
///
/// DeepSeek decides the next tick while a walk runs; this is where it reads the residual rows
/// the walk just made and records its decision. The hook runs once a walk, before
/// [`Walk::head`], with the rows still live. An arm that does not need one leaves
/// [`Pass::hook`] unset and need not call it; an arm that does must call it at the end of
/// [`Walk::walk`].
pub trait WalkHook {
    /// Reads `hidden` after the last layer. The rows must not be modified through this call.
    ///
    /// # Errors
    /// The hook's own failure; the walk fails with it.
    fn after_layers(&mut self, hidden: &Hidden) -> Result<(), BackendError>;
}

impl<F> WalkHook for F
where
    F: FnMut(&Hidden) -> Result<(), BackendError>,
{
    fn after_layers(&mut self, hidden: &Hidden) -> Result<(), BackendError> {
        self(hidden)
    }
}

/// One pass over some rows of some sequences.
pub struct Pass<'a> {
    /// What the pass is for.
    pub kind: PassKind,
    /// The output constraint of a [`PassKind::Verify`] pass, and `None` otherwise.
    pub constraint: Option<OutputConstraint<'a>>,
    /// The hook between the last layer and the head ([tick-ahead][`WalkHook`]).
    pub hook: Option<&'a mut dyn WalkHook>,
}

impl<'a> Pass<'a> {
    /// One token a sequence.
    #[must_use]
    pub const fn token_rows() -> Self {
        Self {
            kind: PassKind::TokenRows,
            constraint: None,
            hook: None,
        }
    }

    /// One prompt's chunk of `rows` rows.
    #[must_use]
    pub const fn bulk(rows: usize) -> Self {
        Self {
            kind: PassKind::Bulk { rows },
            constraint: None,
            hook: None,
        }
    }

    /// The paired chunks of `chunks` prompts.
    #[must_use]
    pub const fn paired(chunks: usize) -> Self {
        Self {
            kind: PassKind::Paired { chunks },
            constraint: None,
            hook: None,
        }
    }

    /// Verify `rows` rows under `constraint`.
    #[must_use]
    pub const fn verify(rows: usize, constraint: OutputConstraint<'a>) -> Self {
        Self {
            kind: PassKind::Verify { rows },
            constraint: Some(constraint),
            hook: None,
        }
    }

    /// The same pass with `hook` between the layers and the head.
    #[must_use]
    pub fn with_hook(mut self, hook: &'a mut dyn WalkHook) -> Self {
        self.hook = Some(hook);
        self
    }

    /// The number of rows the pass says it holds, when the kind names one.
    #[must_use]
    pub const fn rows(&self) -> Option<usize> {
        match self.kind {
            PassKind::TokenRows => None,
            PassKind::Bulk { rows } | PassKind::Verify { rows } => Some(rows),
            PassKind::Paired { chunks } => Some(chunks),
        }
    }
}

/// One sequence's rows in a walk, and the tokens they forward.
#[derive(Debug)]
pub struct Segment<'a, S> {
    /// The sequence whose state the rows advance.
    pub seq: &'a mut S,
    /// The tokens, one a row; never empty.
    pub rows: &'a [u32],
}

impl<'a, S> Segment<'a, S> {
    /// A segment of `rows` over `seq`.
    #[must_use]
    pub fn new(seq: &'a mut S, rows: &'a [u32]) -> Self {
        Self { seq, rows }
    }
}

/// The residual rows a walk left, and where in the arm's buffer they start.
///
/// The arm interprets the fields; the runtime only passes the value back to [`Walk::head`] and
/// reads [`Hidden::rows`] to know how many rows it may ask for. An arm that keeps its residual
/// on the device puts the device offset in `first`; one that copies it back to the host puts the
/// host offset there. No field is a pointer, so the value is `Send` and a trace can record it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hidden {
    /// The number of residual rows the walk produced.
    pub rows: usize,
    /// The index of the first of those rows in the arm's residual buffer.
    pub first: usize,
}

impl Hidden {
    /// A hidden of `rows` rows starting at zero.
    #[must_use]
    pub const fn new(rows: usize) -> Self {
        Self { rows, first: 0 }
    }

    /// A hidden of `rows` rows starting at `first`.
    #[must_use]
    pub const fn at(first: usize, rows: usize) -> Self {
        Self { rows, first }
    }
}

/// What a head row's token extends.
///
/// The runtime knows whether the row is the last of a prompt or a decoded token, and the arm's
/// sampler needs the same distinction to record the penalty history correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadInput<'a> {
    /// The row ends a prompt: the sampler resets and records `tokens` as the prompt.
    Prompt(&'a [u32]),
    /// The row consumes `token`.
    Token(u32),
}

/// One residual row to turn into a token, and the sequence whose sampler takes it.
#[derive(Debug)]
pub struct HeadRow<'a, S> {
    /// The sequence whose sampler chooses the token.
    pub seq: &'a mut S,
    /// The row of [`Hidden`] to read.
    pub row: usize,
    /// What the row extends.
    pub input: HeadInput<'a>,
}

impl<'a, S> HeadRow<'a, S> {
    /// A head row ending a prompt of `tokens`.
    #[must_use]
    pub fn prompt(seq: &'a mut S, row: usize, tokens: &'a [u32]) -> Self {
        Self {
            seq,
            row,
            input: HeadInput::Prompt(tokens),
        }
    }

    /// A head row consuming `token`.
    #[must_use]
    pub fn token(seq: &'a mut S, row: usize, token: u32) -> Self {
        Self {
            seq,
            row,
            input: HeadInput::Token(token),
        }
    }
}

/// Which pass shapes a walk accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassMask(u8);

impl PassMask {
    /// One token a sequence.
    pub const TOKEN_ROWS: Self = Self(1 << 0);
    /// One prompt's chunk.
    pub const BULK: Self = Self(1 << 1);
    /// The chunks of several prompts in one pass.
    pub const PAIRED: Self = Self(1 << 2);
    /// Verify rows under an output constraint.
    pub const VERIFY: Self = Self(1 << 3);
    /// Every pass shape this trait defines.
    pub const ALL: Self = Self(0b1111);
    /// No pass shape.
    pub const EMPTY: Self = Self(0);

    /// Whether every bit of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The raw bits, for a ledger or a test.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// How many segments one walk accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Segments {
    /// One segment a walk.
    One,
    /// Several sequences' rows in one segment list.
    Many,
}

/// What an arm declares about itself, without a device.
///
/// These are values on the type ([Walk::CAPABILITIES]), not answers from an instance, because
/// `sparkflash arms --ledger` prints every arm's gap table with no device booted. The ledger reads
/// this value; the plan's "computed from the types" is therefore "from the types and the
/// declared capabilities".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkCapabilities {
    /// The pass shapes the walk accepts.
    pub passes: PassMask,
    /// How many segments a walk accepts.
    pub segments: Segments,
    /// The broad shape of the arm's state, for the ledger's reuse and rewind rows.
    pub state: StateKind,
    /// What the arm declares about the strategies the runtime does not decide itself: the
    /// tensor-core bodies, the stored-format arithmetic and the rewind. The runtime owns the
    /// lanes, the prompt parts and the group, and derives those rows from the fields above.
    pub compute: ComputeDeclaration,
}

/// What an arm declares about the four strategies its own kernels decide.
///
/// Each field is a [`State`], so the declaration is static and readable with no device. A row
/// left out is refused at boot by [`crate::Ledger::require_complete`], which is why the fields
/// have no default constructor beyond the explicit [`ComputeDeclaration::none`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeDeclaration {
    /// Every GEMM-class prompt part on the tensor-core tile ([crate::Strategy::TensorCorePrompt]).
    pub prompt: State,
    /// The one-to-eight row step on the tensor-core tile ([crate::Strategy::TensorCoreDecode]).
    pub token: State,
    /// Every stored format kept in its own arithmetic ([crate::Strategy::StoredArithmetic]).
    pub stored: State,
    /// A changed turn going back to the last prompt's end ([crate::Strategy::Rewind]).
    pub rewind: State,
}

impl ComputeDeclaration {
    /// All four strategies off, each with the reason the walk gives.
    #[must_use]
    pub const fn none(
        prompt: &'static str,
        token: &'static str,
        stored: &'static str,
        rewind: &'static str,
    ) -> Self {
        Self {
            prompt: State::Off(prompt),
            token: State::Off(token),
            stored: State::Off(stored),
            rewind: State::Off(rewind),
        }
    }

    /// All four strategies native, with the details the arm states.
    #[must_use]
    pub const fn all_native(
        prompt: &'static str,
        token: &'static str,
        stored: &'static str,
        rewind: &'static str,
    ) -> Self {
        Self {
            prompt: State::Native(prompt),
            token: State::Native(token),
            stored: State::Native(stored),
            rewind: State::Native(rewind),
        }
    }
}

impl WalkCapabilities {
    /// Capabilities that accept every pass and many segments, with `state`.
    ///
    /// The compute rows are all native: a walk that declares every pass is the toy shape the
    /// runtime's tests use. An arm states its own, because only it knows which bodies it has.
    #[must_use]
    pub const fn all(state: StateKind) -> Self {
        Self {
            passes: PassMask::ALL,
            segments: Segments::Many,
            state,
            compute: ComputeDeclaration::all_native(
                "every prompt part on the tensor-core tile",
                "the token rows on the tensor-core tile",
                "stored formats kept in their own arithmetic",
                "rewind from the positional or recurrent region",
            ),
        }
    }

    /// Whether the walk accepts `pass`.
    #[must_use]
    pub const fn accepts(&self, pass: PassKind) -> bool {
        let mask = match pass {
            PassKind::TokenRows => PassMask::TOKEN_ROWS,
            PassKind::Bulk { .. } => PassMask::BULK,
            PassKind::Paired { .. } => PassMask::PAIRED,
            PassKind::Verify { .. } => PassMask::VERIFY,
        };
        self.passes.contains(mask)
    }
}

/// A draft source.
///
/// Leaving [`Walk::Drafter`] out is a compile error; an arm with no released draft head names
/// [`Absent`] with its reason, which the ledger prints. The marker's [`Drafter::PRESENT`] is a
/// const so a ledger is available without a device; a future drafting phase adds the methods
/// that actually draft here.
pub trait Drafter<W: Walk + ?Sized> {
    /// Whether the arm has a drafter at all.
    const PRESENT: bool;

    /// One line for the ledger.
    #[must_use]
    fn describe(&self) -> String {
        std::any::type_name::<Self>().to_owned()
    }
}

/// An image path.
///
/// Same terms as [`Drafter`]: the associated type is required, and an arm that serves text only
/// names [`NoVision`].
pub trait Vision<W: Walk + ?Sized> {
    /// Whether the arm has a vision path at all.
    const PRESENT: bool;

    /// One line for the ledger.
    #[must_use]
    fn describe(&self) -> String {
        std::any::type_name::<Self>().to_owned()
    }
}

/// The drafter of an arm that publishes none, with the reason it gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Absent<Reason = &'static str>(pub Reason);

impl<W, Reason> Drafter<W> for Absent<Reason>
where
    W: Walk + ?Sized,
    Reason: std::fmt::Display,
{
    const PRESENT: bool = false;

    fn describe(&self) -> String {
        format!("absent ({})", self.0)
    }
}

/// The image path of an arm that serves text only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoVision;

impl<W: Walk + ?Sized> Vision<W> for NoVision {
    const PRESENT: bool = false;
}

/// One model arm's arithmetic, as the runtime drives it.
///
/// Every item here is required except [`Walk::capabilities`], which reads the associated
/// constant. A missing `layout`, `open`, `limits`, `walk`, `head`, capability value,
/// sequence state, drafter or vision type stops the arm compiling.
pub trait Walk: 'static {
    /// One sequence's device state. It must expose checkpoint and rewind, because the runtime
    /// owns reuse and cannot restore an arm's state through a witness.
    type Seq: LaneState;

    /// The draft source. An arm without one names [`Absent`]; there is no default.
    type Drafter: Drafter<Self>;

    /// The image path. An arm without one names [`NoVision`]; there is no default.
    type Vision: Vision<Self>;

    /// What the arm declares without a device. Required: a forgotten declaration is a compile
    /// error, and the ledger never has to print "the arm does not state it".
    const CAPABILITIES: WalkCapabilities;

    /// Which state grows with position and which is recurrent.
    fn layout(&self) -> &StateLayout;

    /// Opens one sequence on `memory`. The reservation is for accounting; the arm owns the
    /// bytes.
    ///
    /// # Errors
    /// A device allocation, an unreadable checkpoint or an exhausted lane.
    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError>;

    /// The most rows one pass may hold, and the context.
    fn limits(&self) -> Limits;

    /// Applies a request's sampler to one of the arm's sequences before its prompt runs.
    ///
    /// The runtime owns when a request's sampler is set; an arm whose [Walk::head] chooses a
    /// token keeps the sampler with its sequence and applies it here. An arm that chooses
    /// without a sampler leaves the sequence as it is.
    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling);

    /// Runs these rows of these sequences through the layers, in one pass.
    ///
    /// A row must compute as it would alone: the arm's wide stages read the weights once for
    /// every row, and its per-sequence stages run on each row's own slice. The arm must call
    /// [`Pass::hook`] after the last layer and before returning the hidden rows, and must append
    /// each segment's rows to that sequence's position.
    ///
    /// # Errors
    /// A backend failure; the whole pass fails with it.
    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError>;

    /// Turns the requested residual rows into tokens, on the device.
    ///
    /// # Errors
    /// A backend failure, or a row outside `hidden`.
    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError>;

    /// What the arm declares, by value.
    #[must_use]
    fn capabilities(&self) -> WalkCapabilities {
        Self::CAPABILITIES
    }

    /// The most segments one walk may hold, when that is narrower than [Self::limits]' rows.
    ///
    /// The fused engine caps a batch here rather than handing a walk more members than its row
    /// buffers take. The default is the pass limit, which is the whole truth for an arm whose
    /// prompt chunk and token batch are the same width; an arm with a wide prompt chunk and a
    /// narrow token batch (Qwen's 4,096 and 4) states its own.
    #[must_use]
    fn max_segments(&self) -> usize {
        self.limits().rows
    }
}
