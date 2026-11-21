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
//! A layer is a list of branches ([`PlanBranch`]): the norm of the residual, one part
//! ([`PartExec`]) over the normed rows, the group's combine where the part leaves a rank's
//! partial ([`RankSplit::Summed`]), and the residual add. This module owns that loop, the
//! sequence around it ([`PlanSeq`]: its ids, its sampler, its rewind bounds) and the head's
//! record-then-choose ([`HeadInput::record`]). The host ([`PlanHost`]) supplies only what needs
//! its device: the embed, the norm, the combine, the add, the head's logits and its device pick,
//! and the per-sequence lane memory ([`PlanLane`]).
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
    Absent, HeadRow, Hidden, NoVision, Pass, PassKind, Segment, Walk, WalkCapabilities,
};

/// A built part as a planned walk holds it: shared between the serving thread and its sessions.
pub type PlanPart<P> = Box<dyn PartExec<P> + Send + Sync>;

/// One branch of a layer: the norm of the residual, the part over the normed rows, whether the
/// part's output rows are a rank's partial, and which of a sequence's region lists is the
/// part's.
pub struct PlanBranch<P: PartHost, N> {
    /// The norm's weights, as the host holds them.
    pub norm: N,
    /// The part.
    pub part: PlanPart<P>,
    /// Whether the part's output rows are a rank's partial ([`crate::RankSplit::Summed`]): the
    /// host's combine sums them before the residual add.
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

/// The device a [`PlanWalk`] drives. Every method is required.
///
/// The host's buffers are its own ([`Self::Scratch`]): the walk names rows by count, and every
/// method reads and writes the buffers the host laid out. The residual is one buffer of
/// `[rows][hidden]`; [`Self::norm`] reads it into the parts' input, a part writes its output,
/// [`Self::combine`] sums that output over the group and [`Self::add`] adds it to the residual.
pub trait PlanHost: Send + Sync + Sized + 'static {
    /// The parts' host: the pass and open contexts the parts are written for.
    type Parts: PartHost;
    /// A norm's weights.
    type Norm: Send + Sync;
    /// A sequence's device state.
    type Lane: PlanLane + Send;
    /// One walk's buffers.
    type Scratch: Send;
    /// What a pass hands its parts besides the rows: the host's kernel choice for a pass shape.
    type Mode: Copy;

    /// The error code of every refusal.
    const DEVICE: &'static str;
    /// The arm's capabilities, word for word its ledger row.
    const CAPABILITIES: WalkCapabilities;
    /// Why the arm has no draft head.
    const NO_DRAFTER: &'static str;

    /// The number of layers.
    fn layers(&self) -> usize;

    /// The branches of `layer`, in forward order.
    fn branches(&self, layer: usize) -> &[PlanBranch<Self::Parts, Self::Norm>];

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

    /// The mode a pass of `kind` takes.
    fn mode(&self, kind: PassKind) -> Self::Mode;

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

    /// The parts' input: `norm` of the residual's `rows` rows.
    ///
    /// # Errors
    /// A device failure.
    fn norm(
        &self,
        scratch: &Self::Scratch,
        norm: &Self::Norm,
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

    /// Sums a part's partial output rows over the group, in rank order. Nothing at one rank.
    ///
    /// # Errors
    /// A device or collective failure.
    fn combine(&self, scratch: &Self::Scratch, rows: usize) -> Result<(), BackendError>;

    /// Adds a part's output rows to the residual.
    ///
    /// # Errors
    /// A device failure.
    fn add(&self, scratch: &Self::Scratch, rows: usize) -> Result<(), BackendError>;

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

/// The branches a layer's step runs in a planned walk, in forward order, or the role the walk has
/// no place for.
///
/// A planned layer is a token mixer and a feed-forward, each a branch. The other roles a step can
/// hold are refused by name, each listed (no wildcard, so a new role leads the compiler here):
/// a compressed cache and a sparse indexer feed an attention's reads rather than the residual, a
/// widened residual replaces the add itself, and a side table adds a lookup beside the branch.
/// No catalogue part executes one of them as a branch yet.
///
/// # Errors
/// The first role the walk does not run.
pub fn plan_roles(step: &LayerStep) -> Result<Vec<(StepRole, PartKind)>, String> {
    step.roles()
        .into_iter()
        .map(|(role, part)| match role {
            StepRole::Attention | StepRole::Ffn => Ok((role, part)),
            StepRole::Compression | StepRole::Indexer | StepRole::Residual | StepRole::Side => {
                Err(format!(
                    "layer {}'s part {} is a {role:?} step, and a planned walk runs a layer's \
                     token mixer and its feed-forward",
                    step.index,
                    part.name()
                ))
            }
        })
        .collect()
}

/// The branches of every layer of `host`, in walk order: the order of [`PlanBranch::slot`].
pub fn branches<H: PlanHost>(host: &H) -> impl Iterator<Item = &PlanBranch<H::Parts, H::Norm>> {
    (0..host.layers()).flat_map(|layer| host.branches(layer))
}

/// The state every part of `host`'s walk declares, in walk order: what a sequence's lane holds.
pub fn plan_state<H: PlanHost>(host: &H) -> Vec<Region> {
    branches(host)
        .flat_map(|branch| branch.part.state().iter().copied())
        .collect()
}

/// One walk over `segments`' rows, laid one after another: each sequence's ids at its own
/// position. Every layer runs its branches in forward order: the norm, the part's `body`, the
/// group's combine where the part leaves a partial, and the residual add. The residual of every
/// row is left in the host's buffer; each sequence's history takes its ids. With `trace`, each
/// layer's residual rows are appended to the layer's entry.
///
/// # Errors
/// No rows or more than the host's pass, an empty segment, an id outside the vocabulary, a
/// sequence past the context, or the host's failure.
pub fn walk_rows<H: PlanHost>(
    host: &H,
    scratch: &H::Scratch,
    segments: &mut PlanRows<'_, '_, H::Lane>,
    body: PartBody,
    mode: H::Mode,
    mut trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(), BackendError> {
    let refuse = |message: String| Err(BackendError::new(H::DEVICE, message));
    let rows: usize = segments.iter().map(|(_, ids)| ids.len()).sum();
    let most = host.pass_rows();
    if rows == 0 || rows > most {
        return refuse(format!("a walk takes 1..={most} rows, not {rows}"));
    }
    let (vocab, context) = (host.vocab(), host.limits().context);
    let mut ids = Vec::with_capacity(rows);
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
        ids.extend_from_slice(tokens);
    }
    host.embed(scratch, &ids)?;
    for index in 0..host.layers() {
        for branch in host.branches(index) {
            host.norm(scratch, &branch.norm, rows)?;
            let mut pass = host.pass(scratch, &*segments, index, branch.slot, rows, mode);
            // One dynamic call a part a pass.
            body.run::<H::Parts, _>(&*branch.part, &mut pass)?;
            drop(pass);
            if branch.summed {
                host.combine(scratch, rows)?;
            }
            host.add(scratch, rows)?;
        }
        if let Some(trace) = trace.as_deref_mut() {
            let residual = host.residual(scratch, rows)?;
            trace[index].extend(residual);
        }
    }
    for (sequence, tokens) in segments.iter_mut() {
        sequence.history.extend_from_slice(tokens);
    }
    Ok(())
}

/// The walk of a [`PlanHost`]: the runtime's [`Walk`] over the host's layers.
pub struct PlanWalk<H: PlanHost> {
    host: Arc<H>,
    layout: StateLayout,
    limits: Limits,
    scratch: Mutex<H::Scratch>,
    drafter: Absent<&'static str>,
}

impl<H: PlanHost> PlanWalk<H> {
    /// A walk over `host`, with head scratch for `capacity` rows (at most a pass).
    ///
    /// # Errors
    /// A device allocation the scratch cannot make.
    pub fn new(host: Arc<H>, capacity: usize) -> Result<Self, BackendError> {
        let rows = capacity.clamp(1, host.pass_rows());
        let scratch = H::scratch(&host, rows)?;
        let layout = StateLayout::new(plan_state(&*host));
        let limits = host.limits();
        Ok(Self {
            host,
            layout,
            limits,
            scratch: Mutex::new(scratch),
            drafter: Absent(H::NO_DRAFTER),
        })
    }

    /// The host the walk drives.
    #[must_use]
    pub fn host(&self) -> &Arc<H> {
        &self.host
    }

    fn scratch(&self) -> Result<MutexGuard<'_, H::Scratch>, BackendError> {
        self.scratch
            .lock()
            .map_err(|_| BackendError::new(H::DEVICE, "the walk's scratch is poisoned"))
    }
}

/// The head of `rows`, from the residual rows the last walk left: each row's input recorded in
/// its sampler first, then one device pick a row when every row takes its argmax, else the
/// logits and the host sampler.
///
/// # Errors
/// The host's failure, or a host that answers another number of rows.
pub fn plan_head<H: PlanHost>(
    host: &H,
    scratch: &H::Scratch,
    rows: &mut [HeadRow<'_, PlanSeq<H::Lane>>],
) -> Result<Vec<StepOutput>, BackendError> {
    let contiguous = rows.iter().enumerate().all(|(index, row)| row.row == index);
    let asked = rows.len();
    let answered = |made: usize| {
        BackendError::new(
            H::DEVICE,
            format!("the head was asked for {asked} rows and made {made}"),
        )
    };
    // Record each row's token before asking whether its sampler is greedy: a prompt's record
    // resets and fills the history the penalty check reads.
    for row in rows.iter_mut() {
        row.input.record(&mut row.seq.sampler);
    }
    // A greedy sampler whose penalties change nothing takes the row's argmax as it stands.
    // Choose it on the device and copy one u32 a row, not the vocabulary.
    if rows
        .iter()
        .all(|row| row.seq.sampler.takes_argmax(&row.seq.sampling))
    {
        let picks: Vec<u32> = if contiguous {
            host.choices(scratch, 0, rows.len())?
        } else {
            let mut picks = Vec::with_capacity(rows.len());
            for row in rows.iter() {
                picks.extend(host.choices(scratch, row.row, 1)?);
            }
            picks
        };
        if picks.len() != rows.len() {
            return Err(answered(picks.len()));
        }
        return Ok(picks
            .into_iter()
            .map(|pick| StepOutput {
                logits: Vec::new(),
                next_token: Some(pick),
                draft: Vec::new(),
            })
            .collect());
    }
    // A sampling request needs the row: temperature, top-k, top-p, penalties and the draw stay
    // on the host over the logits the step copies back.
    let vocab = host.vocab();
    let logits: Vec<Vec<f32>> = if contiguous {
        host.logits(scratch, 0, rows.len())?
            .chunks(vocab)
            .map(<[f32]>::to_vec)
            .collect()
    } else {
        rows.iter()
            .map(|row| host.logits(scratch, row.row, 1))
            .collect::<Result<Vec<_>, _>>()?
    };
    if logits.len() != rows.len() {
        return Err(answered(logits.len()));
    }
    Ok(rows
        .iter_mut()
        .zip(logits)
        .map(|(row, logits)| {
            // The row's token is already recorded above, once: a second record here would count
            // every sampled token twice in the penalties and leave the history a token a step
            // ahead of the position a rewind truncates it to.
            let chosen = row.seq.sampler.choose(&logits, &row.seq.sampling);
            StepOutput {
                logits,
                next_token: Some(chosen),
                draft: Vec::new(),
            }
        })
        .collect())
}

impl<H: PlanHost> Walk for PlanWalk<H> {
    type Seq = PlanSeq<H::Lane>;
    type Hidden = Hidden;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

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

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        seq.sampling = sampling;
        seq.sampler.reset();
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let Pass {
            kind,
            constraint: _,
            prompt_end: _,
            hook,
        } = pass;
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        let scratch = self.scratch()?;
        let mode = self.host.mode(kind);
        let mut pairs: Vec<(&mut Self::Seq, &[u32])> = segments
            .iter_mut()
            .map(|segment| (&mut *segment.seq, segment.rows))
            .collect();
        walk_rows(
            &*self.host,
            &scratch,
            &mut pairs,
            PartBody::of(kind),
            mode,
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
