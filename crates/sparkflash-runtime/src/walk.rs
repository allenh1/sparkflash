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

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::vision::VisionImage;

use crate::layout::{LaneState, Limits, StateKind, StateLayout};
use crate::pool::LaneMemory;
use crate::share::StepWall;

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
    /// The rows of the prompt still to come after a [`PassKind::Bulk`] chunk; `0` is the
    /// prompt's last chunk and `None` is a pass that is not a prompt chunk.
    ///
    /// A drafter that commits a window of its prompt work needs the end: the old serving loops
    /// limited the commits to the prompt's end (`window_commits`), and one that ran to the
    /// chunk's own end would do the last chunk's work twice.
    pub prompt_end: Option<usize>,
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
            prompt_end: None,
            hook: None,
        }
    }

    /// One prompt's chunk of `rows` rows.
    #[must_use]
    pub const fn bulk(rows: usize) -> Self {
        Self {
            kind: PassKind::Bulk { rows },
            constraint: None,
            prompt_end: None,
            hook: None,
        }
    }

    /// The paired chunks of `chunks` prompts.
    #[must_use]
    pub const fn paired(chunks: usize) -> Self {
        Self {
            kind: PassKind::Paired { chunks },
            constraint: None,
            prompt_end: None,
            hook: None,
        }
    }

    /// Verify `rows` rows under `constraint`.
    #[must_use]
    pub const fn verify(rows: usize, constraint: OutputConstraint<'a>) -> Self {
        Self {
            kind: PassKind::Verify { rows },
            constraint: Some(constraint),
            prompt_end: None,
            hook: None,
        }
    }

    /// The same pass with `hook` between the layers and the head.
    #[must_use]
    pub fn with_hook(mut self, hook: &'a mut dyn WalkHook) -> Self {
        self.hook = Some(hook);
        self
    }

    /// The same bulk pass with `after` rows of the prompt still to come.
    #[must_use]
    pub const fn with_prompt_end(mut self, after: usize) -> Self {
        self.prompt_end = Some(after);
        self
    }

    /// Whether this is a prompt's last chunk; `None` when the pass is not a prompt chunk or the
    /// caller did not say where the prompt ends.
    #[must_use]
    pub const fn last_chunk(&self) -> Option<bool> {
        match self.prompt_end {
            Some(after) => Some(after == 0),
            None => None,
        }
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

/// One sequence's rows in a walk, the tokens they forward, and the images the rows take.
#[derive(Debug)]
pub struct Segment<'a, S> {
    /// The sequence whose state the rows advance.
    pub seq: &'a mut S,
    /// The tokens, one a row; never empty.
    pub rows: &'a [u32],
    /// The prompt's fresh images, in prompt order: the ones whose spans this prompt encodes.
    ///
    /// The runtime keeps them for the whole prompt and passes them with every chunk, because a
    /// chunk may carry only part of an image; the arm caches what it encoded for the prompt and
    /// injects the rows a chunk's positions take. Empty for a token or verify pass, and for an
    /// arm whose [`Vision`] is [`NoVision`].
    pub images: &'a [VisionImage],
}

impl<'a, S> Segment<'a, S> {
    /// A segment of `rows` over `seq`, taking no image rows.
    #[must_use]
    pub fn new(seq: &'a mut S, rows: &'a [u32]) -> Self {
        Self {
            seq,
            rows,
            images: &[],
        }
    }

    /// A segment of `rows` over `seq`, whose image positions take `images`' rows.
    #[must_use]
    pub fn with_images(seq: &'a mut S, rows: &'a [u32], images: &'a [VisionImage]) -> Self {
        Self { seq, rows, images }
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
    /// The row is one verify row of the member's `rows` (the anchor, then the drafts): `row`
    /// is its index. The head answers this row's token; the runtime compares it with the draft
    /// after it and stops the member's run at the first mismatch, so a later row's head is not
    /// asked for.
    Verify {
        /// The member's rows in order, anchor first.
        rows: &'a [u32],
        /// The index of this row in `rows`.
        row: usize,
    },
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

    /// A verify row `index` of the member's `rows`.
    #[must_use]
    pub fn verify(seq: &'a mut S, row: usize, rows: &'a [u32], index: usize) -> Self {
        Self {
            seq,
            row,
            input: HeadInput::Verify { rows, row: index },
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

/// What a member's round opens to.
///
/// A round starts on the token its caller fed. A member whose last verify already holds that
/// token's answer opens [`Draft::Held`] and costs no row; every other member drafts its chain
/// and opens [`Draft::Chain`].
#[derive(Debug, Clone)]
pub enum Draft {
    /// The member's last verify already holds the fed token's step: this is its answer, and it
    /// takes no row of the walk.
    Held(StepOutput),
    /// The member's next chain. `drafts[j]` is the token proposed after `drafts[0..=j]`, and
    /// `confidence[j]` is the chance the first `j + 1` drafts are all accepted, falling with
    /// depth; the runtime shares its verify rows by it. An empty chain verifies the anchor alone.
    Chain {
        /// The chance each leading run of `drafts` is accepted.
        confidence: Vec<f64>,
        /// The tokens proposed after the anchor, likeliest first.
        drafts: Vec<u32>,
    },
}

/// A draft source: the round path every drafting arm already had in its own fusion.
///
/// Leaving [`Walk::Drafter`] out is a compile error; an arm with no released draft head names
/// [`Absent`] with its reason, which the ledger prints. Every method is required. A defaulted
/// `ready` or `speculation` silently loses a multi-token run or reports nothing, which is the
/// class of silent default the shared runtime exists to remove; a defaulted `give_back` would
/// leave a rejected draft's rows in the caches.
///
/// The three policies a moved arm used to decide in its fusion are stated here, once:
///
/// * A lone member verifies the drafter's own depth with no row share ([crate::Runtime::step]
///   does not call `share_rows` for one member), so a single request keeps exactly the chain it
///   drafted.
/// * [`crate::JOINT_ROUNDS_ENV`] `=off` runs each lane's round alone, one after another; an
///   arm for which off means no drafting at all returns an empty [`Draft::Chain`].
/// * A drafting walk's engine member cap is the arm's verify width (`Limits::members`), not its
///   prompt chunk: a verify refuses more rows than the rank's verify batch.
pub trait Drafter<W: Walk + ?Sized> {
    /// Whether the arm has a drafter at all.
    const PRESENT: bool;

    /// Whether this rank's round drafts now.
    ///
    /// [`Drafter::PRESENT`] is a property of the type; an arm whose draft head is compiled in
    /// but configured off for this rank says so here, and the runtime runs one plain token pass
    /// over every member instead of one held decode a member. That is the old fused
    /// `Rows::Lanes` step: the tokens are the same, the launch is not.
    fn drafts(&self) -> bool {
        Self::PRESENT
    }

    /// The rows a verify answer belongs to the arm: whatever its settle needs, never the
    /// runtime's host logits. A row the runtime never interprets can be `()`.
    type Rows;

    /// What a shared walk costs, for the row share ([crate::share_rows]).
    fn wall(&self) -> &dyn StepWall;

    /// Opens `seq`'s round on `token`: the step its last verify already holds, or its next
    /// chain. The drafts stay with the sequence until [`Drafter::verify`] walks them.
    ///
    /// # Errors
    /// A backend failure, an exhausted context, or a chain the drafter cannot propose.
    fn open_round(&self, seq: &mut W::Seq, token: u32) -> Result<Draft, BackendError>;

    /// One verify pass over the members' rows: each segment is one member's anchor followed by
    /// the drafts the row share gave it, and `depths[j]` is how many of those rows are drafts.
    /// The runtime calls this once for a joint round and once a member when the lanes run alone.
    ///
    /// # Errors
    /// A backend failure; every member of the pass fails with it.
    fn verify(
        &self,
        walk: &W,
        segments: &mut [Segment<'_, W::Seq>],
        depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError>;

    /// Folds a verify answer back into `seq`'s step: the token the model chose at the anchor,
    /// the rows it keeps, and (in the arm) the queued answers of the drafts it accepted. `chunk`
    /// is the member's anchor followed by the drafts it was walked over.
    ///
    /// # Errors
    /// A backend failure while the rejected rows are given back.
    fn settle(
        &self,
        seq: &mut W::Seq,
        chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError>;

    /// The most drafts one walk of `seqs` members verifies in all, for [crate::share_rows].
    /// An arm whose verify refuses more than a fixed row count caps here.
    fn budget(&self, seqs: usize) -> usize;

    /// Whether `seq` already holds the answer to a step fed `token`. The engine keeps feeding
    /// a member the token its last step chose while this says yes, so an accepted run reaches the
    /// request in one reply.
    fn ready(&self, seq: &W::Seq, token: u32) -> bool;

    /// Puts `seq`'s state at `to`, the position the caller keeps, after the last answers went
    /// unused. An arm whose caches cannot go back may leave them where they are; its next prompt
    /// then goes back to the rewind point.
    ///
    /// # Errors
    /// An arm that cannot put its state at `to`.
    fn give_back(&self, seq: &mut W::Seq, to: usize) -> Result<(), BackendError>;

    /// The speculative counters of `seq`'s request, as the arm kept them.
    fn speculation(&self, seq: &W::Seq) -> Speculation;

    /// The least chance a draft must have to be given a row.
    fn floor(&self) -> f64 {
        0.0
    }

    /// One line for the ledger.
    #[must_use]
    fn describe(&self) -> String {
        std::any::type_name::<Self>().to_owned()
    }
}

/// An image path.
///
/// Same terms as [`Drafter`]: the associated type is required, and an arm that serves text only
/// names [`NoVision`]. The runtime owns when a prompt starts, resumes and ends; this contract is
/// what it calls at those points, so an arm's tower is encoded once a prompt and the rows are
/// dropped when the prompt's chunks are done. The methods are associated because the runtime has
/// only the walk, and the tower lives on it.
pub trait Vision<W: Walk + ?Sized> {
    /// Whether the arm has a vision path at all.
    const PRESENT: bool;

    /// Encodes `images` for `seq`, the fresh images of the prompt `tokens` whose first forwarded
    /// position is `seq`'s. The rows stay with the sequence, where [`Walk::walk`] injects them
    /// over the placeholders their spans name, and [Vision::clear] drops them.
    ///
    /// The runtime calls this once, before the prompt's first chunk, and never between chunks:
    /// an image that spans several chunks is encoded once.
    ///
    /// # Errors
    /// A span that does not fit the prompt, an image this rank has no tower for, or a device
    /// failure.
    fn encode(
        walk: &W,
        seq: &mut W::Seq,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<(), BackendError>;

    /// Drops what [Vision::encode] made, once the prompt's rows are no longer in flight.
    fn clear(walk: &W, seq: &mut W::Seq);

    /// One line for the ledger.
    #[must_use]
    fn describe() -> String {
        std::any::type_name::<Self>().to_owned()
    }
}

/// The drafter of an arm that publishes none, with the reason it gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Absent<Reason = &'static str>(pub Reason);

/// The wall of a walk that drafts nothing: no row share can be charged for it.
struct NoWall;

impl StepWall for NoWall {
    fn ms(&self, _seqs: usize, _rows: usize) -> f64 {
        0.0
    }
}

static NO_WALL: NoWall = NoWall;

impl<W, Reason> Drafter<W> for Absent<Reason>
where
    W: Walk + ?Sized,
    Reason: std::fmt::Display,
{
    const PRESENT: bool = false;
    type Rows = ();

    fn wall(&self) -> &dyn StepWall {
        &NO_WALL
    }

    /// An absent drafter has no round: [`crate::Runtime::step`] branches on
    /// [`Drafter::PRESENT`] before calling any of these.
    fn open_round(&self, _seq: &mut W::Seq, _token: u32) -> Result<Draft, BackendError> {
        Err(BackendError::new(
            "drafter",
            format!("{} drafts no round", self.0),
        ))
    }

    fn verify(
        &self,
        _walk: &W,
        _segments: &mut [Segment<'_, W::Seq>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        Err(BackendError::new(
            "drafter",
            format!("{} verifies no rows", self.0),
        ))
    }

    fn settle(
        &self,
        _seq: &mut W::Seq,
        _chunk: &[u32],
        _rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        Err(BackendError::new(
            "drafter",
            format!("{} has no round to settle", self.0),
        ))
    }

    fn budget(&self, _seqs: usize) -> usize {
        0
    }

    fn ready(&self, _seq: &W::Seq, _token: u32) -> bool {
        false
    }

    fn give_back(&self, seq: &mut W::Seq, to: usize) -> Result<(), BackendError> {
        seq.rewind(to)
    }

    fn speculation(&self, _seq: &W::Seq) -> Speculation {
        Speculation::default()
    }

    fn describe(&self) -> String {
        format!("absent ({})", self.0)
    }
}

/// The image path of an arm that serves text only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoVision;

impl<W: Walk + ?Sized> Vision<W> for NoVision {
    const PRESENT: bool = false;

    fn encode(
        _walk: &W,
        _seq: &mut W::Seq,
        _tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<(), BackendError> {
        debug_assert!(images.is_empty(), "a text-only walk was handed images");
        Ok(())
    }

    fn clear(_walk: &W, _seq: &mut W::Seq) {}
}

/// One model arm's arithmetic, as the runtime drives it.
///
/// Every item here is required except [`Walk::capabilities`], which reads the associated
/// constant, and [`Walk::cpu`], which is a conformance hook. A missing `layout`, `open`,
/// `limits`, `drafter`, `walk`, `head`, capability value, sequence state, drafter or vision
/// type stops the arm compiling.
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

    /// The most rows one pass may hold, the members one batch may hold, and the context.
    ///
    /// A prompt chunk and a token batch are different widths, and a verify pass is
    /// `members x (1 + depth)` rows: the two numbers are separate so a drafting walk's member
    /// cap is its verify width and not its prompt chunk ([Drafter]'s third policy).
    fn limits(&self) -> Limits;

    /// The arm's draft source. A drafting arm returns its configured drafter; an arm that names
    /// [`Absent`] returns the absent value it declares.
    fn drafter(&self) -> &Self::Drafter;

    /// A walk of this type the generated checks can drive with no device, or `None` when they
    /// need the arm's fixture. The toy walk in the runtime's tests supplies one so
    /// [`crate::ConformanceCheck::DraftedEqualsPlain`] runs at least on a CPU.
    fn cpu() -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

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

    /// The most segments one walk may hold, when that is narrower than [`Limits::members`].
    ///
    /// The fused engine caps a batch here rather than handing a walk more members than its row
    /// buffers take. The default is [`Limits::members`], which is the whole truth for an arm
    /// whose prompt chunk and token batch are the same width.
    #[must_use]
    fn max_segments(&self) -> usize {
        self.limits().members
    }

    /// How many positions past the rewind point the state can still go back from.
    ///
    /// A cache that slides holds its window and a widest chunk: once a reply has written more
    /// positions past the prompt's end than that, the rows the rewind reads are gone. An arm
    /// whose state keeps every position leaves the default; an arm whose cache slides states the
    /// reach, and the runtime starts a changed turn cold rather than rewind into overwritten
    /// rows.
    #[must_use]
    fn rewind_slack(&self) -> usize {
        usize::MAX
    }

    /// Where the rewind point of a prompt of `prompt` ids sits: its end, unless the arm says
    /// otherwise.
    ///
    /// A changed turn resumes at the point when it keeps every id before it. An arm whose chat
    /// template writes a past turn back with other ids at the old prompt's end states the
    /// earlier position those turns still share (Dory's prompt ends on ids the next turn
    /// replaces, so its point is one id short). Only an arm whose state is all positional may
    /// move the point: a recurrent region's checkpoint is taken at the prompt's end, and the
    /// runtime refuses a point anywhere else for such a layout.
    #[must_use]
    fn rewind_point(&self, prompt: usize) -> usize {
        prompt
    }
}
