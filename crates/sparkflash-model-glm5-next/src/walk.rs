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

//! GLM-5.3-Flash as a Walk.
//!
//! The runtime drives an arm: it owns the prompt parts, the history, reuse, rewind, the lane
//! pool, give-back and the ledger, and this walk supplies only what GLM alone knows. GLM's
//! serving construction still opens its own [crate::serve::GlmFusion] in this pass, because the
//! runtime has no draft path yet (see [Walk::Drafter]); this is the arm that commits onto it
//! next, with its arithmetic already the same calls the fusion arm makes.
//!
//! * [Walk::layout] is what the rank's caches are: every sparse (DSA) layer holds a positional
//!   latent row and its indexer's rows a position, and every linear (KDA) layer holds a
//!   convolution history and a recurrent memory that are one live value. The bytes come from the
//!   rank's own plan ([DeviceRank::kv_plan]), not a compiled-in estimate.
//! * [Walk::limits] is the rank's bulk chunk width ([DeviceRank::prefill_rows], the profile's
//!   4096 by default) and the served context. The runtime's [sparkflash_runtime::ChunkPolicy]
//!   quantises the first, so a GLM prompt still runs the 4096-row chunks its kernels and scratch
//!   were sized for.
//! * [Walk::walk] runs the same calls the old serving loop ran: a bulk prompt through
//!   [DeviceRank::begin_prefill]/[DeviceRank::prefill_chunk] (the tensor-core chunk path) and a
//!   token row through [DeviceRank::forward_batch_greedy] (the row-GEMV decode path). It does
//!   not draft (see below).
//! * [Walk::head] is the arm's existing head: the host sampler over the row the greedy step left
//!   on the device, or a verified chain's [sparkflash_model::verdict] and [DeviceRank::commit].
//!
//! # The drafting gap
//!
//! [Walk::ready] and [Walk::speculation] are defaulted on [Walk] and [Drafter] is a marker, so
//! a drafting arm on the runtime silently loses its verified runs: the engine hands back one
//! token a step and the reported counters stay zero even though the arm drafted. GLM therefore
//! names [Absent] and its token path is a plain one-row-a-member step; [crate::serve::GlmFusion]
//! still serves, so nothing is slower in this pass.
//!
//! The round's arithmetic is here, ready for the contract: [GlmSequence::draft_after] (the MTP
//! layer's or the DFlash 2 block drafter's chain), [GlmSequence::settle] (the verdict, the
//! commit and the queue of kept answers), [GlmWalk::budget] with [GlmWalk::wall] and
//! [GlmWalk::joint_rounds] for the row share, and the defaulted [Walk::ready],
//! [Walk::give_back] and [Walk::speculation] overrides. When the contract lands, ready,
//! give-back and speculation belong on [Drafter] as required methods, and the arm's source is
//! the checkpoint's MTP layer or the loaded block drafter.

use std::sync::{Arc, Mutex, OnceLock};

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::{DepthPolicy, Queue, Speculator, kept_to_end, verdict};
use sparkflash_ops::sampler::Sampler;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, ExpertUnionWall, HeadInput, HeadRow, Hidden, LaneMemory, LaneState,
    Limits, NoVision, Pass, PassKind, PassMask, Region, Runtime, Segment, Segments, StateKind,
    StateLayout, Walk, WalkCapabilities,
};
use sparkflash_weights::tp::{BitGatherHandle, TpRuntime};

use crate::device::{DeviceRank, DeviceSequence, MAX_ROWS, Prefill};

/// How a session drafts, once the backend has been told to.
#[derive(Debug, Clone)]
pub(crate) struct SpecConfig {
    /// The most drafts a step verifies.
    pub(crate) drafts: usize,
    /// The depth policy, when the drafter says how sure it is of each draft (the block drafter).
    pub(crate) policy: Option<DepthPolicy>,
}

/// The backend's rank and the collectives every walk runs through.
pub(crate) struct Shared {
    pub(crate) rank: Mutex<DeviceRank>,
    pub(crate) tp: TpRuntime,
    pub(crate) gather: BitGatherHandle,
    pub(crate) eos: Vec<u32>,
    /// Set once, before the first sequence opens: a sequence's sequence is a drafting one or it
    /// is not, so the choice cannot change under a live one.
    pub(crate) spec: OnceLock<SpecConfig>,
}

/// A shared walk's cost at TP=4 on four GB10-class nodes, from the device harness: a plain step
/// 31.6 ms, about 0.75 ms a distinct expert, and the block drafter's 4.6-5 ms a stream for each
/// member past the first. The expert counts are the checkpoint's.
const JOINT_WALL: ExpertUnionWall = ExpertUnionWall {
    fixed_ms: 25.0,
    expert_ms: 0.75,
    row_ms: 0.3,
    seq_ms: 5.0,
    experts: 0,
    top_k: 0,
};

/// The rows a verified chain has, its anchor included.
type Chain = Vec<u32>;

/// A round the walk verified for one sequence, waiting for [Walk::head] to keep its leading run.
struct Verified {
    /// The anchor followed by the drafts the share left.
    chunk: Chain,
    /// The drafter's confidence in each draft it proposed.
    confidence: Vec<f32>,
    /// One row of full-vocabulary logits for a row of `chunk`.
    rows: Vec<Vec<f32>>,
}

/// One sequence's device state, as the runtime drives it.
///
/// The runtime owns the history; this owns the rank's cache and the request's sampler. A
/// speculative sequence also owns its queue of kept answers, because a verify produces more than
/// the one token a step answers with.
pub struct GlmSequence {
    shared: Arc<Shared>,
    sequence: DeviceSequence,
    sampler: Sampler,
    sampling: Sampling,
    /// The draft depth and its accounting, when this sequence drafts.
    spec: Option<Speculator>,
    /// Tokens a verify already produced, oldest first, each with the token that follows it.
    pending: Queue,
    /// The drafter's confidence in each of the last round's drafts.
    confidence: Vec<f32>,
    /// The last round the walk verified, waiting for its head.
    verified: Option<Verified>,
    /// The last bulk chunk's logits, which [DeviceRank::prefill_chunk] leaves on the host.
    prompt_logits: Option<Vec<f32>>,
    /// Whether a prompt is running a chunk at a time ([Self::bulk_chunk]).
    prefill_open: bool,
}

impl GlmSequence {
    fn new(shared: Arc<Shared>, sequence: DeviceSequence) -> Self {
        let spec = shared
            .spec
            .get()
            .map(|spec| Speculator::new(spec.drafts, spec.policy.clone()));
        Self {
            shared,
            sequence,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            spec,
            pending: Queue::new(),
            confidence: Vec::new(),
            verified: None,
            prompt_logits: None,
            prefill_open: false,
        }
    }

    /// Whether the sequence already holds the answer for a step fed `token`.
    #[must_use]
    pub fn holds(&self, token: u32) -> bool {
        self.pending
            .front()
            .is_some_and(|front| front.token == token)
    }

    /// The chain a round verifies after `anchor`: the anchor and the drafts the depth allows
    /// ([DeviceRank::draft_chain]), with the drafter's confidence in each draft it proposed,
    /// which the depth was chosen from and the round's verdict calibrates.
    ///
    /// The future drafter contract's round calls this; nothing does while the walk does not
    /// draft.
    #[allow(dead_code)]
    fn draft_after(
        &mut self,
        rank: &DeviceRank,
        anchor: u32,
    ) -> Result<(Chain, Vec<f32>), BackendError> {
        // A draft writes the MTP caches past what the sequence keeps, so a chain longer than the
        // room left would draft past the window.
        let room = self
            .sequence
            .capacity()
            .saturating_sub(self.sequence.position() + 1);
        let (drafts, confidence) = match self.spec.as_ref() {
            Some(spec) => rank.draft_chain(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                anchor,
                spec,
                room,
            )?,
            None => (Vec::new(), Vec::new()),
        };
        self.confidence.clone_from(&confidence);
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        Ok((chunk, confidence))
    }

    /// A verify of `verified` answered its rows: keep what the model chose, give the rest back
    /// before the rank's next step, and queue the kept rows for the calls that feed them. The
    /// future drafter contract's round calls this; [Walk::head] calls it for a
    /// [PassKind::Verify] pass.
    fn settle(
        &mut self,
        rank: &DeviceRank,
        verified: Verified,
    ) -> Result<StepOutput, BackendError> {
        let Verified {
            chunk,
            confidence,
            rows,
        } = verified;
        // Rows past the first disagreement draw nothing, so a seeded answer does not depend on
        // how many drafts the round verified.
        let picks =
            self.sampler
                .choose_verify(rows.iter().map(Vec::as_slice), &chunk[1..], &self.sampling);
        let kept = verdict(&chunk, &picks);
        let eos = &self.shared.eos;
        // No row past the reply's end stays in the caches, so a reply that ends inside the round
        // leaves its sequence at the history its caller saw, which the next turn extends.
        let held = kept_to_end(kept, &picks, eos);
        if chunk.len() > 1 {
            // Give back the rows the sequence does not keep, before this rank's next step.
            rank.commit(&mut self.sequence, held)?;
        }
        if let Some(spec) = self.spec.as_mut() {
            spec.observe(&confidence, chunk.len() - 1, kept);
        }
        self.pending.extend_round(&chunk, &picks, &rows, held, eos);
        let next_token = picks.first().copied().filter(|token| !eos.contains(token));
        Ok(StepOutput {
            logits: rows.first().cloned().unwrap_or_default(),
            next_token,
            draft: Vec::new(),
        })
    }

    /// Forwards one bulk prompt chunk and keeps its last row's logits for the head.
    ///
    /// The first chunk asks [DeviceRank::begin_prefill] which path the prompt takes: the token
    /// path whole (a prompt below [MIN_PREFILL_ROWS], or a rank with no chunk scratch), or the
    /// bulk chunks, which is the tensor-core prompt path. Every later chunk is a
    /// [DeviceRank::prefill_chunk] of the prompt [DeviceRank::begin_prefill] started, as the old
    /// serving loop's `prompt_part` ran it.
    fn bulk_chunk(&mut self, rank: &mut DeviceRank, rows: &[u32]) -> Result<(), BackendError> {
        let first = !self.prefill_open;
        if first {
            // A new prompt's tokens replace any chain the last one left queued, and no round of
            // the last turn's is still waiting for its head.
            self.pending.clear();
            self.confidence.clear();
            self.verified = None;
            self.prompt_logits = None;
        }
        let end = self.sequence.position() + rows.len();
        let logits = if first {
            match rank.begin_prefill(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                rows,
            )? {
                Prefill::Whole(logits) => logits,
                Prefill::Chunks { .. } => {
                    self.prefill_open = true;
                    rank.prefill_chunk(
                        &self.shared.tp,
                        &self.shared.gather,
                        &mut self.sequence,
                        rows,
                        end,
                    )?
                }
            }
        } else {
            // The lane must back the rows this chunk writes before it runs; the first chunk's
            // begin_prefill backed only its own.
            rank.back(&mut self.sequence, end)?;
            rank.prefill_chunk(
                &self.shared.tp,
                &self.shared.gather,
                &mut self.sequence,
                rows,
                end,
            )?
        };
        self.prompt_logits = Some(logits);
        Ok(())
    }
}

impl LaneState for GlmSequence {
    fn position(&self) -> usize {
        self.sequence.position()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        if at != self.sequence.position() {
            return Err(BackendError::new(
                "glm5_device",
                format!(
                    "the checkpoint is asked for position {at}, the sequence is at {}",
                    self.sequence.position()
                ),
            ));
        }
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        rank.keep_point(&mut self.sequence)
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        let from = self.sequence.position();
        if to > from {
            return Err(BackendError::new(
                "glm5_device",
                format!("the rewind is to position {to}, the sequence is at {from}"),
            ));
        }
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        if to == 0 {
            rank.reset(&mut self.sequence)?;
        } else {
            rank.rewind(&mut self.sequence, to)?;
        }
        drop(rank);
        self.pending.clear();
        self.confidence.clear();
        self.verified = None;
        self.prompt_logits = None;
        self.prefill_open = false;
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        // The runtime asks about the history's positions; the state is at least there, and a
        // drafting step also writes its chain's rows past it.
        let drafts = self.spec.as_ref().map_or(0, Speculator::max_depth);
        let wanted = positions.max(self.sequence.position() + 1 + drafts);
        self.sequence.room(wanted, reset)
    }
}

/// The GLM-5.3-Flash arm, exposed to the shared runtime as a Walk.
pub struct GlmWalk {
    shared: Arc<Shared>,
    context_limit: usize,
    layout: StateLayout,
    limits: Limits,
    /// Whether drafting members verify their rounds in one walk
    /// ([sparkflash_runtime::JOINT_ROUNDS_ENV]); off, each member's chain is verified whole
    /// without the row share. The future drafter contract's round reads this.
    #[allow(dead_code)]
    joint_rounds: bool,
    /// What a shared walk costs, for the row share. The future drafter contract's round reads
    /// this.
    #[allow(dead_code)]
    wall: ExpertUnionWall,
}

impl GlmWalk {
    /// The walk over `shared`'s rank and `context_limit` positions a sequence.
    ///
    /// # Errors
    /// A poisoned rank lock, or a [sparkflash_runtime::JOINT_ROUNDS_ENV] that is not a switch.
    pub(crate) fn new(shared: Arc<Shared>, context_limit: usize) -> Result<Self, BackendError> {
        let (plan, chunk, wall) = {
            let rank = shared
                .rank
                .lock()
                .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
            let config = rank.config();
            let plan = rank.kv_plan().map_err(|error| {
                BackendError::new("glm5_device", format!("the rank's cache plan: {error}"))
            })?;
            (
                plan,
                rank.prefill_rows(),
                ExpertUnionWall {
                    experts: config.n_routed_experts,
                    top_k: config.num_experts_per_tok,
                    ..JOINT_WALL
                },
            )
        };
        let joint_rounds = sparkflash_runtime::joint_rounds()
            .map_err(|error| BackendError::new("glm5_device", error))?;
        Ok(Self {
            shared,
            context_limit,
            layout: layout_of(plan),
            // The runtime's ChunkPolicy quantises this, so a profile's 4096 stays 4096 and the
            // rank's scratch can always hold the chunk it is handed.
            limits: Limits::new(chunk, context_limit),
            joint_rounds,
            wall,
        })
    }
}

/// The arm's state, as the runtime needs to see it: the sparse layers' latent and indexer rows
/// grow a position at a time, and the linear layers' convolution history and recurrent memory
/// are one live value a layer.
fn layout_of(plan: sparkflash_model::KvPlan) -> StateLayout {
    let mut regions = Vec::new();
    if plan.per_token() > 0 {
        regions.push(Region::Positional {
            name: "dlm-latent-and-indexer",
            bytes_per_row: usize::try_from(plan.per_token()).unwrap_or(usize::MAX),
        });
    }
    if plan.fixed() > 0 {
        regions.push(Region::Recurrent {
            name: "kda-conv-and-recurrent",
            bytes: usize::try_from(plan.fixed()).unwrap_or(usize::MAX),
        });
    }
    StateLayout::new(regions)
}

/// The runtime over a GLM walk: the serving arm the backend opens when it fuses its lanes, and
/// the arm a session outside the engine steps through.
///
/// [crate::serve::GlmFusion] still serves in this pass, because the runtime has no draft path
/// yet ([Walk::Drafter]); the next commit opens this instead.
///
/// # Errors
/// A device allocation the first slot cannot make, a [sparkflash_runtime::REWIND_ENV] that is
/// not a switch, or a [sparkflash_runtime::JOINT_ROUNDS_ENV] that is not one.
#[allow(dead_code)]
pub(crate) fn runtime(
    shared: Arc<Shared>,
    context_limit: usize,
) -> Result<Runtime<GlmWalk>, BackendError> {
    let walk = GlmWalk::new(shared, context_limit)?;
    // No free-memory figure is measured for the chunk: the rank's scratch decides the width,
    // which is what its profile's prefill.chunk sized.
    Runtime::new(walk, 0)
}

impl Walk for GlmWalk {
    type Seq = GlmSequence;
    /// The runtime has no draft path yet: [sparkflash_runtime::Runtime] steps one row a member
    /// and its [sparkflash_runtime::Runtime::verify] heads only each member's anchor, so an arm
    /// that named a drafter here would lose its rounds silently (see the module's drafting gap).
    /// GLM's source is the checkpoint's MTP layer or the loaded DFlash 2 block drafter; the
    /// contract that can carry it is the next pass's, shared with Qwen and Inkling.
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        // A prompt runs in bulk chunks and a step runs token rows; a verify pass is the round
        // the walk runs itself, for a runtime caller that has rows to check.
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same constant the profile check that refuses `decode.tensor_cores = true` reads.
        compute: ComputeDeclaration {
            prompt: State::Native("bulk GEMMs on the FP8 E4M3 MMAs; BF16 as head and remainder"),
            token: crate::TOKEN_DECODE,
            stored: State::Native(
                "FP8 E4M3 projections and experts run E4M3 x E4M3; BF16 stays BF16",
            ),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError> {
        let _ = memory;
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        // The walk does not draft in this pass, so it opens a plain sequence: the MTP layer's
        // and the block drafter's state are the future drafter contract's to ask for.
        let sequence = rank.new_sequence(self.context_limit)?;
        drop(rank);
        Ok(GlmSequence::new(Arc::clone(&self.shared), sequence))
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row this step
        // already copies back, so a non-greedy request is served rather than approximated.
        seq.sampling = sampling;
        seq.sampler.reset();
        // The adapter calls this once per request, and a kept sequence carries the last one's
        // counts: a request reports its own speculation, not the session's lifetime.
        if let Some(spec) = seq.spec.as_mut() {
            spec.reset();
        }
        seq.confidence.clear();
        seq.pending.clear();
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let mut rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        match pass.kind {
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                let mut rows = 0;
                for segment in segments.iter_mut() {
                    rows += segment.rows.len();
                    let words = segment.rows;
                    let seq = &mut *segment.seq;
                    seq.bulk_chunk(&mut rank, words)?;
                }
                Ok(Hidden::new(rows))
            }
            PassKind::Verify { .. } => {
                let mut rows = 0;
                let mut batch: Vec<(&mut DeviceSequence, &[u32])> = segments
                    .iter_mut()
                    .map(|segment| {
                        rows += segment.rows.len();
                        (&mut segment.seq.sequence, segment.rows)
                    })
                    .collect();
                let verified =
                    rank.verify_batch(&self.shared.tp, &self.shared.gather, &mut batch)?;
                drop(batch);
                for (segment, rows) in segments.iter_mut().zip(verified) {
                    segment.seq.verified = Some(Verified {
                        chunk: segment.rows.to_vec(),
                        confidence: Vec::new(),
                        rows,
                    });
                }
                Ok(Hidden::new(rows))
            }
            PassKind::TokenRows => self.token_rows(&mut rank, segments),
        }
    }

    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let rank = self
            .shared
            .rank
            .lock()
            .map_err(|_| BackendError::new("glm5_device", "the rank lock is poisoned"))?;
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            if row.row >= hidden.rows {
                return Err(BackendError::new(
                    "glm5_device",
                    format!("the head was asked for row {} of {}", row.row, hidden.rows),
                ));
            }
            let seq = &mut *row.seq;
            let fed = match &row.input {
                HeadInput::Prompt(tokens) => {
                    // The prompt's chunks are done: the next prompt starts a new one.
                    seq.prefill_open = false;
                    // Its sampler history is the whole prompt: the penalties count every token
                    // the request reads, however many chunks the prompt ran in.
                    seq.sampler.reset();
                    seq.sampler.record_prompt(tokens);
                    None
                }
                HeadInput::Token(token) => {
                    seq.sampler.record(*token);
                    Some(*token)
                }
            };
            // A verify already consumed this token and produced the answer: hand back the row it
            // sampled, exactly as the one-token step it stands in for would have.
            if let Some(token) = fed
                && seq
                    .pending
                    .front()
                    .is_some_and(|front| front.token == token)
            {
                let entry = seq.pending.pop(token).ok_or_else(|| {
                    BackendError::new("glm5_device", "the queue changed under its front")
                })?;
                steps.push(StepOutput {
                    logits: entry.logits,
                    next_token: entry.next_token,
                    draft: Vec::new(),
                });
                continue;
            }
            if let Some(verified) = seq.verified.take() {
                steps.push(seq.settle(&rank, verified)?);
                continue;
            }
            // The greedy path leaves each row's pick on the device and the last kept row's
            // logits behind it, so a step reads back one vocabulary row, not one a row. A bulk
            // prompt's last row came back from prefill_chunk instead.
            let logits = match seq.prompt_logits.take() {
                Some(logits) => logits,
                None => rank.last_logits(&seq.sequence)?,
            };
            let token = seq.sampler.choose(&logits, &seq.sampling);
            let next_token = if self.shared.eos.contains(&token) {
                None
            } else {
                Some(token)
            };
            steps.push(StepOutput {
                logits,
                next_token,
                draft: Vec::new(),
            });
        }
        Ok(steps)
    }

    fn ready(&self, seq: &Self::Seq, token: u32) -> bool {
        seq.holds(token)
    }

    /// The caches consumed the unused rows and cannot give them back, so the history is cut and
    /// the queued answers dropped: the sequence is ahead of its history and names none, and its
    /// next prompt goes back to its rewind point. This is the arm's old give-back, unchanged.
    fn give_back(&self, seq: &mut Self::Seq, _to: usize) -> Result<(), BackendError> {
        seq.pending.clear();
        seq.confidence.clear();
        seq.verified = None;
        Ok(())
    }

    fn speculation(&self, seq: &Self::Seq) -> Speculation {
        seq.spec
            .as_ref()
            .map_or_else(Speculation::default, Speculator::stats)
    }
}

impl GlmWalk {
    /// One token a segment: every member's row through the rank's batched greedy path, whose
    /// last kept row's logits [Walk::head] reads back.
    ///
    /// The walk does not draft in this pass (see [Walk::Drafter]): a round is the future
    /// drafter contract's, and the bodies it will use are [GlmSequence::draft_after],
    /// [GlmSequence::settle] and [Self::budget]. A batch wider than [MAX_ROWS] rows still runs
    /// as one call: [DeviceRank::forward_batch_greedy] splits it a step at a time.
    fn token_rows(
        &self,
        rank: &mut DeviceRank,
        segments: &mut [Segment<'_, GlmSequence>],
    ) -> Result<Hidden, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        for segment in segments.iter_mut() {
            // A step ends any prompt that was running a chunk at a time.
            segment.seq.prefill_open = false;
        }
        let mut batch: Vec<(&mut DeviceSequence, &[u32])> = segments
            .iter_mut()
            .map(|segment| (&mut segment.seq.sequence, segment.rows))
            .collect();
        rank.forward_batch_greedy(&self.shared.tp, &self.shared.gather, &mut batch)?;
        Ok(Hidden::new(rows))
    }

    /// The most drafts one walk of `seqs` members verifies in all: one row a member is its
    /// anchor, and a step is at most [MAX_ROWS] rows. The future drafter contract's row share
    /// reads this; nothing calls it while the walk does not draft.
    #[allow(dead_code)]
    fn budget(&self, seqs: usize) -> usize {
        let depth = self.shared.spec.get().map_or(0, |spec| spec.drafts);
        (seqs * depth).min(MAX_ROWS.saturating_sub(seqs))
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::ledger::Strategy;
    use sparkflash_runtime::{Mode, Walk, ledger_for};

    use super::*;

    /// The four compute rows the walk declares are the arm's static declaration, so the boot
    /// ledger and **sparkflash arms --ledger** cannot drift.
    #[test]
    fn the_declared_compute_rows_are_the_arm_ledger_rows() {
        use sparkflash_model::ledger::ledger_for_arm;
        let declared = ledger_for_arm("GLM").expect("GLM's declaration");
        let capabilities = <GlmWalk as Walk>::CAPABILITIES.compute;
        for (strategy, stated) in [
            (Strategy::TensorCorePrompt, capabilities.prompt),
            (Strategy::TensorCoreDecode, capabilities.token),
            (Strategy::StoredArithmetic, capabilities.stored),
            (Strategy::Rewind, capabilities.rewind),
        ] {
            let from_table = declared
                .rows
                .iter()
                .find(|(row, _)| *row == strategy)
                .map(|(_, state)| *state)
                .expect("the arm states every strategy");
            assert_eq!(stated, from_table, "{} drifted", strategy.name());
        }
    }

    /// The profile refusal and the ledger's token row are one constant, as dense's are: the
    /// switch cannot be refused for a reason the ledger does not state.
    #[test]
    fn the_token_row_and_the_profile_refusal_are_one_source() {
        let State::Off(why) = crate::TOKEN_DECODE else {
            panic!("GLM declares a tensor-core token body; delete the refusal with the row");
        };
        assert!(why.contains("row GEMV"));
        assert_eq!(
            <GlmWalk as Walk>::CAPABILITIES.compute.token,
            crate::TOKEN_DECODE
        );
        let mut ledger = ledger_for::<GlmWalk>();
        ledger.set(Strategy::Lanes, Mode::native("one segment a walk"));
        assert!(
            !ledger.line().contains("does not state it"),
            "the runtime derives every row GLM states: {}",
            ledger.line()
        );
    }

    /// The layout follows the plan: a positional latent and indexer region and a recurrent KDA
    /// one.
    #[test]
    fn the_layout_is_the_plans_two_kinds_of_state() {
        let positional = layout_of(sparkflash_model::KvPlan::new(1_920, 0));
        assert_eq!(positional.kind(), StateKind::Positional);
        let both = layout_of(sparkflash_model::KvPlan::new(20_160, 4_096));
        assert_eq!(both.kind(), StateKind::Both);
        assert!(both.needs_checkpoint());
        assert_eq!(both.bytes_per_row(), 20_160);
        assert_eq!(both.bytes_at(2), 40_320 + 4_096);
    }

    /// The walk declares no drafter while its token path is one plain row a member, so the
    /// ledger and the path cannot disagree about whether GLM drafts through the runtime.
    #[test]
    fn the_walk_declares_no_drafter_while_it_has_no_round_path() {
        assert!(
            !<<GlmWalk as Walk>::Drafter as sparkflash_runtime::Drafter<GlmWalk>>::PRESENT,
            "the token path does not draft; a drafter name here would be the loaded gun"
        );
        let ledger = ledger_for::<GlmWalk>();
        assert!(
            matches!(ledger.mode(Strategy::Drafting), Mode::Off(_)),
            "the runtime ledger states GLM's drafting off: {}",
            ledger.line()
        );
    }

    /// A profile's 4096-row chunk survives the runtime's ChunkPolicy quantization.
    #[test]
    fn the_runtime_chunks_a_glm_prompt_4096_rows_at_a_time() {
        let limits = Limits::new(crate::device::DEFAULT_PREFILL_ROWS, 1_048_576);
        assert_eq!(
            sparkflash_runtime::ChunkPolicy::for_limits(limits).rows(),
            crate::device::DEFAULT_PREFILL_ROWS
        );
    }

    sparkflash_runtime::conformance!(super::GlmWalk);
}
