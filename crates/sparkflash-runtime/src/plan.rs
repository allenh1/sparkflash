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

//! The walk every catalogue arm shares: [`PlanWalk`].
//!
//! A layer is a list of branches ([`PlanBranch`]), each one part ([`PartExec`]) inside its
//! frame: the host's entry from the residual into the part's input ([`PlanHost::enter`], a norm
//! for a plain residual) and its exit back ([`PlanHost::leave`]: the group's combine where the
//! part leaves a rank's partial, [`RankSplit::Summed`], then the add). A widened residual is a
//! frame, not a branch: its streams' mix in and mix out are the host's entry and exit
//! ([`PlanLayer::residual`]). This module owns that loop, the sequence around it ([`PlanSeq`]:
//! its ids, its sampler, its rewind bounds) and the head's record-then-choose
//! ([`HeadInput::record`]). The host supplies only what needs its device: the embed, the frames,
//! the head's logits and its device pick, and the per-sequence lane memory ([`PlanLane`]).
//!
//! An arm whose every layer is catalogue parts therefore writes no walk: it is a host and its
//! parts. A family with a part of its own adds one [`PartExec`] and registers it
//! ([`crate::PartRegistry`]); `crates/sparkflash-model-dense/tests/onboard_example.rs` is the
//! worked example.

use std::sync::{Arc, Mutex, MutexGuard};

use sparkflash_model::Sampler;
use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::compose::{LayerStep, StepRole};
use sparkflash_model::parts::Part as PartKind;

use crate::layout::{LaneState, Limits, Region, StateLayout, check_checkpoint, check_rewind};
use crate::part::{PartBody, PartExec, PartHost};
use crate::walk::{
    Drafter, HeadRow, Hidden, Pass, PassKind, Segment, Vision, Walk, WalkCapabilities,
};

/// A built part as a planned walk holds it: shared between the serving thread and its sessions.
pub type PlanPart<P> = Box<dyn PartExec<P> + Send + Sync>;

/// One branch of a layer: the host's frame around it, the part, whether the part's output rows
/// are a rank's partial, and which of a sequence's region lists is the part's.
pub struct PlanBranch<P: PartHost, F> {
    /// The host's entry and exit around the part: a norm's weights for a plain residual, a
    /// stream mixer's as well for a widened one, nothing for a part that lands elsewhere.
    pub frame: F,
    /// The part.
    pub part: PlanPart<P>,
    /// Whether the part's output rows are a rank's partial ([`crate::RankSplit::Summed`]): the
    /// host's exit sums them over the group before they land.
    pub summed: bool,
    /// Which of a sequence's region lists is this part's, in walk order.
    pub slot: usize,
}

/// A sequence's device state, as its host keeps it: the lane memory its parts' regions live in
/// and the rewind point of its recurrent ones.
pub trait PlanLane {
    /// The error code of the lane's refusals: the host's [`PlanHost::DEVICE`].
    const DEVICE: &'static str;

    /// Saves the rewind point at position `at` (the end of a prompt). A lane with no recurrent
    /// region has nothing to save.
    ///
    /// # Errors
    /// A device failure.
    fn save(&mut self, at: usize) -> Result<(), BackendError>;

    /// Goes back to the rewind point saved at `to`, which is above zero. Positional regions need
    /// nothing: the position moves.
    ///
    /// # Errors
    /// A device failure.
    fn restore(&mut self, to: usize) -> Result<(), BackendError>;

    /// Sets every recurrent region to the empty sequence's and forgets the rewind point.
    ///
    /// # Errors
    /// A device failure.
    fn clear(&mut self) -> Result<(), BackendError>;

    /// Whether the lane can hold `positions` more positions (from zero when `reset`).
    fn room(&self, positions: usize, reset: bool) -> bool;

    /// A new request on the lane (the runtime sets its sampling once a request): what the lane
    /// counts a request, such as a drafter's speculation counters or its queued answers, starts
    /// over. A lane that counts nothing does nothing.
    fn requested(&mut self) {}
}

/// What a pass is, as a planned walk's host sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanPass {
    /// What the pass is for.
    pub kind: PassKind,
    /// The rows of the prompt still to come after a prompt chunk ([`Pass::prompt_end`]).
    pub prompt_end: Option<usize>,
}

/// One sequence of a planned walk: the host's lane, the ids its state holds and its sampler.
pub struct PlanSeq<L> {
    /// The host's device state of the sequence.
    pub lane: L,
    /// The ids whose rows the state holds, in position order. Its length is the position.
    history: Vec<u32>,
    /// The sampler the head chooses with.
    pub sampler: Sampler,
    /// The request's sampling.
    pub sampling: Sampling,
}

impl<L> PlanSeq<L> {
    /// A sequence at position zero over `lane`.
    pub fn new(lane: L) -> Self {
        Self {
            lane,
            history: Vec::new(),
            sampler: Sampler::new(),
            sampling: Sampling::default(),
        }
    }

    /// The ids the sequence's state holds, in position order.
    #[must_use]
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// The sequence's position: the rows its state holds.
    #[must_use]
    pub fn position(&self) -> usize {
        self.history.len()
    }

    /// Keeps the first `rows` ids of the history and gives the rest back: a drafter's settle,
    /// once its lane kept the leading run of a verify the walk took whole. The sampler's history
    /// is the round's own and stays.
    ///
    /// # Errors
    /// More rows than the history holds.
    pub fn keep(&mut self, rows: usize) -> Result<(), BackendError>
    where
        L: PlanLane,
    {
        check_rewind(L::DEVICE, rows, self.history.len())?;
        self.history.truncate(rows);
        Ok(())
    }
}

impl<L: PlanLane> LaneState for PlanSeq<L> {
    fn position(&self) -> usize {
        self.history.len()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        check_checkpoint(L::DEVICE, at, self.history.len())?;
        self.lane.save(at)
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind(L::DEVICE, to, self.history.len())?;
        if to == 0 {
            self.lane.clear()?;
        } else {
            self.lane.restore(to)?;
        }
        self.history.truncate(to);
        self.sampler.truncate(to);
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.lane.room(positions, reset)
    }
}

/// The segments of one walk: each sequence and the ids it takes, rows laid one after another.
pub type PlanRows<'a, 'b, L> = [(&'a mut PlanSeq<L>, &'b [u32])];

/// One window of a pass, as [`PlanHost::begin_window`] sees it before the window runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanWindow {
    /// The pass the window is part of.
    pub pass: PlanPass,
    /// The window's place among the pass's windows, from zero.
    pub number: usize,
    /// The pass's windows.
    pub windows: usize,
    /// The pass row of the window's first row.
    pub first: usize,
    /// The window's rows.
    pub rows: usize,
    /// For each of the window's segments, in order, the rows of its sequence this pass takes
    /// after the window.
    pub after: Vec<usize>,
}

/// The device a [`PlanWalk`] drives. Every method is required.
///
/// The host's buffers are its own ([`Self::Scratch`]): the walk names rows by count, and every
/// method reads and writes the buffers the host laid out. [`Self::enter`] reads the residual
/// into the parts' input, a part writes its output, and [`Self::leave`] lands that output: over
/// the group where it is a partial, then into the residual (or wherever the frame says).
pub trait PlanHost: Send + Sync + Sized + 'static {
    /// The parts' host: the pass and open contexts the parts are written for.
    type Parts: PartHost;
    /// A branch's frame: what [`Self::enter`] and [`Self::leave`] read.
    type Frame: Send + Sync;
    /// A sequence's device state.
    type Lane: PlanLane + Send;
    /// The draft head: [`Absent`] for an arm without one.
    type Drafter: Drafter<PlanWalk<Self>>;
    /// The image path: [`crate::NoVision`] for an arm without one. Its encode leaves the rows
    /// with the sequence's lane, and the host's window reads them there: the walk hands the
    /// host no images of its own.
    type Vision: Vision<PlanWalk<Self>>;
    /// One walk's buffers.
    type Scratch: Send;
    /// What a pass hands its parts besides the rows: the host's kernel choice for a pass shape.
    type Mode: Copy;

    /// The error code of every refusal.
    const DEVICE: &'static str;
    /// The arm's capabilities, word for word its ledger row.
    const CAPABILITIES: WalkCapabilities;
    /// The roles beyond the token mixer and the feed-forward this host frames
    /// ([`plan_layer`]'s `hosted`); every other role is refused by name.
    const ROLES: &'static [StepRole] = &[];
    /// The roles this host's parts run inside another part, with no branch of their own
    /// ([`plan_layer_folding`]'s `folded`).
    const FOLDED: &'static [StepRole] = &[];
    /// Whether every head row hands its logits over, a greedy one too: a host that reads every
    /// row back whatever the pick, whose sessions have always answered the row's logits. A greedy
    /// row then takes the sampler's own argmax over them, the pick [`Self::choices`] would make.
    /// Off by default: a greedy row's pick is the host's [`Self::choices`].
    const KEEPS_LOGITS: bool = false;

    /// The number of layers.
    fn layers(&self) -> usize;

    /// The branches of `layer`, in forward order.
    fn branches(&self, layer: usize) -> &[PlanBranch<Self::Parts, Self::Frame>];

    /// The checkpoint's vocabulary.
    fn vocab(&self) -> usize;

    /// The most rows one walk takes.
    fn pass_rows(&self) -> usize;

    /// The pass and context limits the runtime schedules by.
    fn limits(&self) -> Limits;

    /// One walk's buffers, with logits for `rows` head rows.
    ///
    /// # Errors
    /// A device allocation the host cannot make.
    fn scratch(this: &Arc<Self>, rows: usize) -> Result<Self::Scratch, BackendError>;

    /// A fresh sequence's lane.
    ///
    /// # Errors
    /// No lane free, or a device failure.
    fn lane(this: &Arc<Self>) -> Result<Self::Lane, BackendError>;

    /// The draft head over this host.
    ///
    /// # Errors
    /// A draft head the host cannot build.
    fn drafter(this: &Arc<Self>) -> Result<Self::Drafter, BackendError>;

    /// The mode a window of `rows` rows of `pass` takes.
    fn mode(&self, pass: PlanPass, rows: usize) -> Self::Mode;

    /// How `pass` over `segments` splits into windows the walk runs one after another, as row
    /// counts in row order (each above zero, at most [`Self::pass_rows`], summing to the pass's
    /// rows). A window may end inside a segment: its sequence's later rows go in the next. One
    /// window by default.
    ///
    /// # Errors
    /// A host that cannot split the pass.
    fn windows(
        &self,
        pass: PlanPass,
        segments: &PlanRows<'_, '_, Self::Lane>,
    ) -> Result<Vec<usize>, BackendError> {
        let _ = pass;
        Ok(vec![segments.iter().map(|(_, ids)| ids.len()).sum()])
    }

    /// The host's start of a window, before its embed: what the window's rows need set up before
    /// any layer runs, over the window's segments (their histories do not hold the window's ids
    /// yet). Nothing by default.
    ///
    /// # Errors
    /// A window the host refuses, or a device failure.
    fn begin_window(
        &self,
        scratch: &Self::Scratch,
        segments: &mut PlanRows<'_, '_, Self::Lane>,
        window: &PlanWindow,
    ) -> Result<(), BackendError> {
        let _ = (scratch, segments, window);
        Ok(())
    }

    /// A window that [`Self::begin_window`] began failed before its [`Self::finish`] ended it:
    /// the host puts back what the begin set up. Nothing by default.
    fn abandon_window(&self, scratch: &Self::Scratch) {
        let _ = scratch;
    }

    /// The regions a sequence's lane holds, in walk order: the walk's layout. The state every
    /// part declares by default ([`plan_state`]); a host whose lane its own cache plan lays out
    /// states that plan's.
    fn regions(&self) -> Vec<Region> {
        plan_state(self)
    }

    /// How many of the layers, from the first, the walk runs in the window just begun
    /// ([`Self::begin_window`]): the host's [`Self::finish`] runs the rest itself, such as
    /// layers whose rows it splits in halves that overlap their collectives. The last walked
    /// branch's exit still names the next layer's first frame. Every layer by default.
    fn walked_layers(&self, scratch: &Self::Scratch) -> usize {
        let _ = scratch;
        self.layers()
    }

    /// The host's tail after a window's layers, over the window's segments (their histories do
    /// not hold the window's ids yet): a head the host runs eagerly, a draft head's caches, what
    /// a prompt's end commits. `last` is the pass's last window. Nothing by default.
    ///
    /// # Errors
    /// A device failure.
    fn finish(
        &self,
        scratch: &Self::Scratch,
        segments: &mut PlanRows<'_, '_, Self::Lane>,
        pass: PlanPass,
        last: bool,
    ) -> Result<(), BackendError> {
        let _ = (scratch, segments, pass, last);
        Ok(())
    }

    /// The answer `lane` already holds for its head row, once the row's input is recorded (`fed`:
    /// the token the row consumed, `None` for a prompt's end), such as a verified draft's queued
    /// answer. `None` by default: the head chooses.
    ///
    /// # Errors
    /// A lane whose held answers are out of order.
    fn answered(
        &self,
        lane: &mut Self::Lane,
        fed: Option<u32>,
    ) -> Result<Option<StepOutput>, BackendError> {
        let _ = (lane, fed);
        Ok(None)
    }

    /// Whether `token` ends the reply: the head answers `None` for it. No token by default.
    fn ends(&self, token: u32) -> bool {
        let _ = token;
        false
    }

    /// The token the head chooses from one row's `logits`, once the row's input is recorded in
    /// `sampler`: the sampler's own choice by default. A host whose draws are its own (a sampled
    /// request's noise a function of the row's position) chooses here; [`Self::ends`] still
    /// decides whether the token ends the reply.
    fn choose(&self, sampler: &mut Sampler, logits: &[f32], sampling: &Sampling) -> u32 {
        sampler.choose(logits, sampling)
    }

    /// The floats one row of [`Self::logits`] holds: the vocabulary by default. A host whose
    /// head trims a padded vocabulary states the trimmed width.
    fn logits_width(&self) -> usize {
        self.vocab()
    }

    /// How many positions past the rewind point a sequence's state can still go back from
    /// ([`Walk::rewind_slack`]): every position by default; a host whose cache slides states
    /// its reach.
    fn rewind_slack(&self) -> usize {
        usize::MAX
    }

    /// Where the rewind point of a prompt of `prompt` ids sits ([`Walk::rewind_point`]): its
    /// end by default; a host whose chat template rewrites a past turn's last ids states the
    /// earlier position the next turn still shares.
    fn rewind_point(&self, prompt: usize) -> usize {
        prompt
    }

    /// Backs the lane's positional regions up to `reach` positions.
    ///
    /// # Errors
    /// The lane memory's refusal.
    fn back(&self, lane: &mut Self::Lane, reach: usize) -> Result<(), BackendError>;

    /// The residual of `ids`, one row an id.
    ///
    /// # Errors
    /// A device failure.
    fn embed(&self, scratch: &Self::Scratch, ids: &[u32]) -> Result<(), BackendError>;

    /// The parts' input from the residual's `rows` rows, through `frame`: the norm for a plain
    /// residual, the streams' mix in then the norm for a widened one.
    ///
    /// # Errors
    /// A device failure.
    fn enter(
        &self,
        scratch: &Self::Scratch,
        frame: &Self::Frame,
        rows: usize,
    ) -> Result<(), BackendError>;

    /// The pass a part of `layer` runs over: the rows, each segment's regions by `slot`, the
    /// scratch and the mode.
    fn pass<'a>(
        &'a self,
        scratch: &'a Self::Scratch,
        segments: &'a [(&'a mut PlanSeq<Self::Lane>, &'a [u32])],
        layer: usize,
        slot: usize,
        rows: usize,
        mode: Self::Mode,
    ) -> <Self::Parts as PartHost>::Pass<'a>;

    /// Lands a part's `rows` output rows through `frame`: summed over the group first, in rank
    /// order, when `summed` (nothing to sum at one rank), then added to the residual, or mixed
    /// out into its streams for a widened one. `next` is the frame the walk enters next (`None`
    /// after the last layer's last branch), so an exit that fuses the next entry's first stage
    /// (a stream mixer's projection of the streams it just wrote) can; [`Self::enter`] then
    /// finds it done.
    ///
    /// # Errors
    /// A device or collective failure.
    fn leave(
        &self,
        scratch: &Self::Scratch,
        frame: &Self::Frame,
        next: Option<&Self::Frame>,
        summed: bool,
        rows: usize,
    ) -> Result<(), BackendError>;

    /// The residual's `rows` rows, copied back: a layer trace.
    ///
    /// # Errors
    /// A device failure.
    fn residual(&self, scratch: &Self::Scratch, rows: usize) -> Result<Vec<f32>, BackendError>;

    /// The logits of `count` residual rows from row `first`, row after row: the final norm and
    /// the head.
    ///
    /// # Errors
    /// More rows than the scratch holds logits for, or a device failure.
    fn logits(
        &self,
        scratch: &Self::Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError>;

    /// The greedy pick of `count` residual rows from row `first`, chosen on the device: the
    /// host sampler's argmax for every finite row.
    ///
    /// # Errors
    /// As [`Self::logits`].
    fn choices(
        &self,
        scratch: &Self::Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError>;
}

/// A layer as a planned walk runs it: its branches in run order, and the widened residual
/// that frames them, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanLayer {
    /// The residual's mixer ([`StepRole::Residual`]): it names no branch; the host's frames
    /// around the token mixer and the feed-forward run it. `None` for a plain residual.
    pub residual: Option<PartKind>,
    /// The branches, in run order: the side lookups on the layer input first, then the
    /// attention's compression and indexer (their output is what the token mixer reads, so they
    /// run before it), the token mixer, the feed-forward.
    pub branches: Vec<(StepRole, PartKind)>,
}

/// The layer a step runs as in a planned walk whose host frames `hosted` beyond the token mixer
/// and the feed-forward, or the first role it has no place for.
///
/// Every role is listed (no wildcard, so a new role leads the compiler here). A host that runs
/// none of the others passes `&[]` and is refused by name for a compressed cache, a sparse
/// indexer, a widened residual or a side table.
///
/// # Errors
/// The first role neither the walk nor the host runs.
pub fn plan_layer(step: &LayerStep, hosted: &[StepRole]) -> Result<PlanLayer, String> {
    plan_layer_folding(step, hosted, &[])
}

/// [`plan_layer`] for a host whose parts also run the roles in `folded` inside another part,
/// such as a sparse indexer its attention's part selects with: a folded role is planned like a
/// hosted one and then names no branch of its own.
///
/// Only the attention's own sublayers fold: a [`StepRole::Compression`] or a
/// [`StepRole::Indexer`] runs inside the token mixer's step. Any other folded role is refused by
/// name.
///
/// # Errors
/// A folded role that is not one of the attention's sublayers, or as [`plan_layer`].
pub fn plan_layer_folding(
    step: &LayerStep,
    hosted: &[StepRole],
    folded: &[StepRole],
) -> Result<PlanLayer, String> {
    for role in folded {
        match role {
            StepRole::Compression | StepRole::Indexer => {}
            StepRole::Attention
            | StepRole::Ffn
            | StepRole::Residual
            | StepRole::Side
            | StepRole::Message => {
                return Err(format!(
                    "a {role:?} step does not fold into another part: only the attention's \
                     compression and indexer run inside its step"
                ));
            }
        }
    }
    let runs: Vec<StepRole> = hosted.iter().chain(folded).copied().collect();
    let mut layer = PlanLayer {
        residual: None,
        branches: Vec::new(),
    };
    let mut before = Vec::new();
    for (role, part) in step.roles() {
        let framed = match role {
            StepRole::Attention | StepRole::Ffn => true,
            StepRole::Compression
            | StepRole::Indexer
            | StepRole::Residual
            | StepRole::Side
            | StepRole::Message => runs.contains(&role),
        };
        if !framed {
            return Err(format!(
                "layer {}'s part {} is a {role:?} step, and this planned walk's host runs a \
                 layer's token mixer and its feed-forward{}",
                step.index,
                part.name(),
                if runs.is_empty() {
                    String::new()
                } else {
                    format!(" and its {runs:?} steps")
                }
            ));
        }
        if folded.contains(&role) {
            continue;
        }
        match role {
            StepRole::Residual => layer.residual = Some(part),
            StepRole::Compression | StepRole::Indexer => before.push((role, part)),
            StepRole::Side | StepRole::Attention | StepRole::Ffn | StepRole::Message => {
                layer.branches.push((role, part));
            }
        }
    }
    // The attention's own sublayers follow it in forward order but feed its reads: they run
    // ahead of it.
    if !before.is_empty() {
        let at = layer
            .branches
            .iter()
            .position(|(role, _)| *role == StepRole::Attention)
            .unwrap_or(layer.branches.len());
        layer.branches.splice(at..at, before);
    }
    Ok(layer)
}

/// The branches of every layer of `host`, in walk order: the order of [`PlanBranch::slot`].
pub fn branches<H: PlanHost>(host: &H) -> impl Iterator<Item = &PlanBranch<H::Parts, H::Frame>> {
    (0..host.layers()).flat_map(|layer| host.branches(layer))
}

/// The state every part of `host`'s walk declares, in walk order: what a sequence's lane holds.
pub fn plan_state<H: PlanHost>(host: &H) -> Vec<Region> {
    branches(host)
        .flat_map(|branch| branch.part.state().iter().copied())
        .collect()
}

/// One walk over `segments`' rows, laid one after another: each sequence's ids at its own
/// position, in the host's windows ([`PlanHost::windows`]) one after another. Every window
/// embeds its ids and runs every layer's branches in run order: the host's entry, the part's
/// `body`, the host's exit (the group's combine where the part leaves a partial, then the
/// landing); then the host's tail ([`PlanHost::finish`]), and each sequence's history takes the
/// window's ids. The host begins each window ([`PlanHost::begin_window`]) before its embed, and
/// puts back one that fails before its tail ([`PlanHost::abandon_window`]). The residual of the
/// last window's rows is left in the host's buffer. Each
/// window takes `mode`, or the host's [`PlanHost::mode`] for it when `None`. With `trace`, each
/// layer's residual rows are appended to the layer's entry.
///
/// # Errors
/// No rows or more than the host's pass, an empty segment, an id outside the vocabulary, a
/// sequence past the context, windows that do not split the pass, or the host's failure.
pub fn walk_rows<H: PlanHost>(
    host: &H,
    scratch: &H::Scratch,
    segments: &mut PlanRows<'_, '_, H::Lane>,
    body: PartBody,
    pass: PlanPass,
    mode: Option<H::Mode>,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), BackendError> {
    let refuse = |message: String| Err(BackendError::new(H::DEVICE, message));
    let rows: usize = segments.iter().map(|(_, ids)| ids.len()).sum();
    let most = host.pass_rows();
    if rows == 0 || rows > most {
        return refuse(format!("a walk takes 1..={most} rows, not {rows}"));
    }
    let (vocab, context) = (host.vocab(), host.limits().context);
    for (sequence, tokens) in segments.iter_mut() {
        if tokens.is_empty() {
            return refuse("a segment has no tokens".to_owned());
        }
        if let Some(token) = tokens.iter().find(|token| **token as usize >= vocab) {
            return refuse(format!(
                "token {token} is outside the checkpoint vocabulary {vocab}"
            ));
        }
        let reach = sequence.position() + tokens.len();
        if reach > context {
            return refuse(format!("the sequence is past the served context {context}"));
        }
        host.back(&mut sequence.lane, reach)?;
    }
    let windows = host.windows(pass, segments)?;
    if windows.iter().any(|&count| count == 0 || count > most)
        || windows.iter().sum::<usize>() != rows
    {
        return refuse(format!(
            "the host split {rows} rows into windows {windows:?}, which do not cover them"
        ));
    }
    let layers = host.layers();
    // The frame the walk enters after branch `at` of `index`: the layer's next branch, else the
    // first branch of the next layer that has one.
    let next = |index: usize, at: usize| {
        host.branches(index)
            .get(at + 1)
            .or_else(|| (index + 1..layers).find_map(|later| host.branches(later).first()))
            .map(|branch| &branch.frame)
    };
    let mut start = 0;
    for (number, &count) in windows.iter().enumerate() {
        let end = start + count;
        // The window's slice of each segment it reaches, in row order, and the segment's rows
        // after it.
        let mut window: Vec<(&mut PlanSeq<H::Lane>, &[u32])> = Vec::new();
        let mut after = Vec::new();
        let mut at = 0;
        for (sequence, tokens) in segments.iter_mut() {
            let (low, high) = (at, at + tokens.len());
            at = high;
            let (from, to) = (low.max(start), high.min(end));
            if from < to {
                window.push((&mut **sequence, &tokens[from - low..to - low]));
                after.push(high - to);
            }
        }
        let ids: Vec<u32> = window
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        let mode = mode.unwrap_or_else(|| host.mode(pass, count));
        let planned = PlanWindow {
            pass,
            number,
            windows: windows.len(),
            first: start,
            rows: count,
            after,
        };
        host.begin_window(scratch, &mut window, &planned)?;
        let mut run = || {
            host.embed(scratch, &ids)?;
            for index in 0..host.walked_layers(scratch).min(layers) {
                for (at, branch) in host.branches(index).iter().enumerate() {
                    host.enter(scratch, &branch.frame, count)?;
                    let mut part = host.pass(scratch, &window, index, branch.slot, count, mode);
                    // One dynamic call a part a pass.
                    body.run::<H::Parts, _>(&*branch.part, &mut part)?;
                    drop(part);
                    host.leave(
                        scratch,
                        &branch.frame,
                        next(index, at),
                        branch.summed,
                        count,
                    )?;
                }
                if let Some(trace) = trace.as_deref_mut() {
                    let residual = host.residual(scratch, count)?;
                    trace[index].extend(residual);
                }
            }
            host.finish(scratch, &mut window, pass, number + 1 == windows.len())
        };
        let ran = run();
        if ran.is_err() {
            host.abandon_window(scratch);
        }
        ran?;
        for (sequence, tokens) in &mut window {
            sequence.history.extend_from_slice(tokens);
        }
        start = end;
    }
    Ok(())
}

/// The walk of a [`PlanHost`]: the runtime's [`Walk`] over the host's layers.
pub struct PlanWalk<H: PlanHost> {
    host: Arc<H>,
    layout: StateLayout,
    limits: Limits,
    scratch: Mutex<H::Scratch>,
    drafter: H::Drafter,
}

impl<H: PlanHost> PlanWalk<H> {
    /// A walk over `host`, with head scratch for `capacity` rows (at most a pass).
    ///
    /// # Errors
    /// A device allocation the scratch cannot make.
    pub fn new(host: Arc<H>, capacity: usize) -> Result<Self, BackendError> {
        let rows = capacity.clamp(1, host.pass_rows());
        let scratch = H::scratch(&host, rows)?;
        let layout = StateLayout::new(host.regions());
        let limits = host.limits();
        let drafter = H::drafter(&host)?;
        Ok(Self {
            host,
            layout,
            limits,
            scratch: Mutex::new(scratch),
            drafter,
        })
    }

    /// The host the walk drives.
    #[must_use]
    pub fn host(&self) -> &Arc<H> {
        &self.host
    }

    /// The logits of `count` rows of the last walk from row `first`, row after row: what a
    /// drafter's verify reads after it walks a verify pass.
    ///
    /// # Errors
    /// As [`PlanHost::logits`].
    pub fn logits(&self, first: usize, count: usize) -> Result<Vec<f32>, BackendError> {
        let scratch = self.scratch()?;
        self.host.logits(&scratch, first, count)
    }

    /// The greedy pick of `count` rows of the last walk from row `first`, chosen on the device:
    /// what a drafter's verify reads when every member takes its argmax.
    ///
    /// # Errors
    /// As [`PlanHost::choices`].
    pub fn choices(&self, first: usize, count: usize) -> Result<Vec<u32>, BackendError> {
        let scratch = self.scratch()?;
        self.host.choices(&scratch, first, count)
    }

    fn scratch(&self) -> Result<MutexGuard<'_, H::Scratch>, BackendError> {
        self.scratch
            .lock()
            .map_err(|_| BackendError::new(H::DEVICE, "the walk's scratch is poisoned"))
    }
}

/// The head of `rows`, from the residual rows the last walk left: each row's input recorded in
/// its sampler first and a lane's held answer taken as it stands ([`PlanHost::answered`]); then
/// one device pick a row when every other row takes its argmax and the host keeps no logits
/// ([`PlanHost::KEEPS_LOGITS`]), else the logits and the host's choice ([`PlanHost::choose`]).
/// A token the host ends the reply on ([`PlanHost::ends`]) answers `None`.
///
/// # Errors
/// The host's failure, or a host that answers another number of rows.
pub fn plan_head<H: PlanHost>(
    host: &H,
    scratch: &H::Scratch,
    rows: &mut [HeadRow<'_, PlanSeq<H::Lane>>],
) -> Result<Vec<StepOutput>, BackendError> {
    let asked = rows.len();
    let answered = |made: usize| {
        BackendError::new(
            H::DEVICE,
            format!("the head was asked for {asked} rows and made {made}"),
        )
    };
    // Record each row's token before asking whether its sampler is greedy: a prompt's record
    // resets and fills the history the penalty check reads.
    let mut steps: Vec<Option<StepOutput>> = Vec::with_capacity(asked);
    for row in rows.iter_mut() {
        let fed = row.input.record(&mut row.seq.sampler);
        steps.push(host.answered(&mut row.seq.lane, fed)?);
    }
    let open: Vec<usize> = (0..asked).filter(|&at| steps[at].is_none()).collect();
    let contiguous = open.len() == asked && rows.iter().enumerate().all(|(at, row)| row.row == at);
    let ends = |token: u32| (!host.ends(token)).then_some(token);
    // A greedy sampler whose penalties change nothing takes the row's argmax as it stands.
    // Choose it on the device and copy one u32 a row, not the vocabulary.
    if !H::KEEPS_LOGITS
        && open
            .iter()
            .all(|&at| rows[at].seq.sampler.takes_argmax(&rows[at].seq.sampling))
    {
        let picks: Vec<u32> = if contiguous {
            host.choices(scratch, 0, asked)?
        } else {
            let mut picks = Vec::with_capacity(open.len());
            for &at in &open {
                picks.extend(host.choices(scratch, rows[at].row, 1)?);
            }
            picks
        };
        if picks.len() != open.len() {
            return Err(answered(picks.len() + asked - open.len()));
        }
        for (&at, pick) in open.iter().zip(picks) {
            steps[at] = Some(StepOutput {
                logits: Vec::new(),
                next_token: ends(pick),
                draft: Vec::new(),
            });
        }
    } else {
        // A sampling request needs the row: temperature, top-k, top-p, penalties and the draw
        // stay on the host over the logits the step copies back.
        let width = host.logits_width();
        let logits: Vec<Vec<f32>> = if contiguous {
            host.logits(scratch, 0, asked)?
                .chunks(width)
                .map(<[f32]>::to_vec)
                .collect()
        } else {
            open.iter()
                .map(|&at| host.logits(scratch, rows[at].row, 1))
                .collect::<Result<Vec<_>, _>>()?
        };
        if logits.len() != open.len() {
            return Err(answered(logits.len() + asked - open.len()));
        }
        for (&at, logits) in open.iter().zip(logits) {
            let seq = &mut *rows[at].seq;
            // The row's token is already recorded above, once: a second record here would count
            // every sampled token twice in the penalties and leave the history a token a step
            // ahead of the position a rewind truncates it to.
            let chosen = host.choose(&mut seq.sampler, &logits, &seq.sampling);
            steps[at] = Some(StepOutput {
                logits,
                next_token: ends(chosen),
                draft: Vec::new(),
            });
        }
    }
    steps
        .into_iter()
        .map(|step| step.ok_or_else(|| answered(0)))
        .collect()
}

impl<H: PlanHost> Walk for PlanWalk<H> {
    type Seq = PlanSeq<H::Lane>;
    type Hidden = Hidden;
    type Drafter = H::Drafter;
    type Vision = H::Vision;

    const CAPABILITIES: WalkCapabilities = H::CAPABILITIES;

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        H::lane(&self.host).map(PlanSeq::new)
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn rewind_slack(&self) -> usize {
        self.host.rewind_slack()
    }

    fn rewind_point(&self, prompt: usize) -> usize {
        self.host.rewind_point(prompt)
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        seq.sampling = sampling;
        seq.sampler.reset();
        seq.lane.requested();
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let Pass {
            kind,
            constraint: _,
            prompt_end,
            hook,
        } = pass;
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        let scratch = self.scratch()?;
        let plan = PlanPass { kind, prompt_end };
        let mut pairs: Vec<(&mut Self::Seq, &[u32])> = segments
            .iter_mut()
            .map(|segment| (&mut *segment.seq, segment.rows))
            .collect();
        walk_rows(
            &*self.host,
            &scratch,
            &mut pairs,
            PartBody::of(kind),
            plan,
            None,
            None,
        )?;
        drop(pairs);
        let hidden = Hidden::new(rows);
        if let Some(hook) = hook {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        hidden: &mut Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        if hidden.rows == 0 {
            return Ok(Vec::new());
        }
        let scratch = self.scratch()?;
        plan_head(&*self.host, &scratch, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyper_step() -> LayerStep {
        LayerStep {
            index: 3,
            attention: PartKind::MlaAttention,
            compression: None,
            indexer: Some(PartKind::SparseIndexer),
            ffn: PartKind::Moe,
            residual: Some(PartKind::HyperConnection),
            sides: vec![PartKind::Engram],
            message: Vec::new(),
        }
    }

    #[test]
    fn a_host_that_frames_nothing_more_is_refused_by_role() {
        let error = plan_layer(&hyper_step(), &[]).unwrap_err();
        assert!(error.contains("layer 3's part"), "{error}");
        assert!(error.contains("Side"), "{error}");
    }

    #[test]
    fn a_widened_residual_frames_and_the_indexer_runs_ahead_of_its_mixer() {
        let hosted = [StepRole::Side, StepRole::Residual, StepRole::Indexer];
        let layer = plan_layer(&hyper_step(), &hosted).unwrap();
        assert_eq!(layer.residual, Some(PartKind::HyperConnection));
        assert_eq!(
            layer.branches,
            vec![
                (StepRole::Side, PartKind::Engram),
                (StepRole::Indexer, PartKind::SparseIndexer),
                (StepRole::Attention, PartKind::MlaAttention),
                (StepRole::Ffn, PartKind::Moe),
            ]
        );
        let partial = plan_layer(&hyper_step(), &[StepRole::Side, StepRole::Residual]);
        assert!(partial.unwrap_err().contains("Indexer"));
    }

    #[test]
    fn a_folded_indexer_is_planned_and_names_no_branch() {
        let hosted = [StepRole::Side, StepRole::Residual];
        let layer = plan_layer_folding(&hyper_step(), &hosted, &[StepRole::Indexer]).unwrap();
        assert_eq!(layer.residual, Some(PartKind::HyperConnection));
        assert_eq!(
            layer.branches,
            vec![
                (StepRole::Side, PartKind::Engram),
                (StepRole::Attention, PartKind::MlaAttention),
                (StepRole::Ffn, PartKind::Moe),
            ]
        );
        assert_eq!(
            plan_layer_folding(&hyper_step(), &hosted, &[]),
            plan_layer(&hyper_step(), &hosted)
        );
        for role in [StepRole::Side, StepRole::Residual, StepRole::Ffn] {
            let error = plan_layer_folding(&hyper_step(), &[], &[role]).unwrap_err();
            assert!(error.contains("does not fold"), "{error}");
        }
        let unframed =
            plan_layer_folding(&hyper_step(), &[StepRole::Residual], &[StepRole::Indexer]);
        assert!(unframed.unwrap_err().contains("Side"));
    }

    #[test]
    fn a_plain_layer_is_its_mixer_then_its_feed_forward() {
        let step = LayerStep {
            index: 0,
            attention: PartKind::GqaAttention,
            compression: None,
            indexer: None,
            ffn: PartKind::DenseMlp,
            residual: None,
            sides: Vec::new(),
            message: Vec::new(),
        };
        let layer = plan_layer(&step, &[]).unwrap();
        assert_eq!(layer.residual, None);
        assert_eq!(
            layer.branches,
            vec![
                (StepRole::Attention, PartKind::GqaAttention),
                (StepRole::Ffn, PartKind::DenseMlp),
            ]
        );
    }
}
