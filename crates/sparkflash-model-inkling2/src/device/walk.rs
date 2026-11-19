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

//! Inkling-2-Small as a [Walk], with its DSpark round as the [Drafter].
//!
//! The runtime owns the prompt parts, history, resume, rewind, lane pool, give-back and the
//! ledger; this module supplies only what Inkling alone knows: the chunk schedule's rows, the
//! session that holds a sequence's caches, and the head that samples from them. Every pass is
//! the arm's own walk called with the same arguments:
//!
//! * [Walk::layout] is what a sequence's state is: every layer's key and value rows grow with
//!   position, and the four convolution histories a layer plus the drafter's Markov feature are
//!   one live value that a rewind restores from the prompt-boundary checkpoint.
//! * [Walk::limits] is [BATCH_ROWS] members a batch and the rank's widest prompt chunk
//!   ([Shared::prefill_chunk]) rows a pass. The runtime's [sparkflash_runtime::ChunkPolicy]
//!   returns that chunk, and [InklingWalk::walk_bulk] forwards it in the arm's own widths, so
//!   the prompt boundaries - and the bits that depend on them - are the ones the arm measured.
//!   All three widths (4,096, 8,192 and 65,535) are entries of
//!   [sparkflash_runtime::CHUNK_SIZES], so the policy can return any of them.
//! * [Walk::walk] runs one of Inkling's row sets: a prompt chunk ([Rows::Chunk]), one token a
//!   lane ([Rows::Lanes]), or an anchor and its drafts verified together ([Rows::Segments]).
//! * [Walk::head] is the arm's sampler over the row a walk left.
//!
//! The drafter is the arm's DSpark (or prompt-lookup) round with [Drafter]'s policies stated
//! once: a lone member verifies the depth the arm's own depth policy picked, and a joint round
//! shares rows through [Drafter::budget], which caps a verify at [BATCH_ROWS] rows
//! (members x (1 + depth)). With SPARKFLASH_INKLING_DRAFT unset the round is the plain step and
//! the runtime takes its answer without a walk ([Draft::Held]).
//!
//! The arm serves through this walk: the group's rank 0 and workers both drive
//! [sparkflash_runtime::Runtime] over it, and a single session is the runtime's one-slot
//! adapter. The image path is [InklingVision]: the runtime encodes a prompt's fresh images once
//! through rank 0's tower and passes them with each chunk, and this walk injects their rows.

use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, Sampling, Speculation, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::vision::VisionImage;
use sparkflash_runtime::{
    ComputeDeclaration, Draft, Drafter, ExpertUnionWall, HeadInput, HeadRow, Hidden, LaneMemory,
    LaneState, Limits, Opened, Pass, PassKind, PassMask, Region, Segment as RunSegment, Segments,
    StateKind, StateLayout, StepWall, Vision, Walk, WalkCapabilities,
};

use super::chunk::{Injection, Rows, Segment, exchange_sum, walk_rows};
use super::{
    BATCH_ROWS, Chunk, ChunkWidths, DEVICE, DraftState, Drafter as DraftKind, InklingDeviceSession,
    JOINT_WALL, Opening, PrefillWalk, Shared, choose, open_buffer,
};

/// One sequence's device state, as the runtime drives it.
///
/// The runtime owns the tokens it has fed; this owns the rank's caches and the request's
/// sampler. A pass's answers wait here for [Walk::head], because the runtime asks for them in a
/// second call.
pub struct InklingSequence {
    session: InklingDeviceSession,
    /// The one row the last token or prompt pass answered, waiting for its head.
    answered: Option<Vec<f32>>,
    /// A verify pass's answers, one a row, in the order the pass walked them.
    verified: Vec<Vec<f32>>,
    /// The first row of that verify pass in the walk's global row order.
    verified_from: usize,
}

impl InklingSequence {
    fn new(session: InklingDeviceSession) -> Self {
        Self {
            session,
            answered: None,
            verified: Vec::new(),
            verified_from: 0,
        }
    }

    fn clear_answers(&mut self) {
        self.answered = None;
        self.verified.clear();
        self.verified_from = 0;
    }
}

impl std::fmt::Debug for InklingSequence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InklingSequence")
            .field("position", &self.session.state.position)
            .finish_non_exhaustive()
    }
}

impl LaneState for InklingSequence {
    fn position(&self) -> usize {
        self.session.state.position
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        if at != self.session.state.position {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the checkpoint is asked for position {at}, the sequence is at {}",
                    self.session.state.position
                ),
            ));
        }
        self.session
            .save_boundary(at)
            .map_err(|error| BackendError::new(DEVICE, error))
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        let from = self.session.state.position;
        if to > from {
            return Err(BackendError::new(
                DEVICE,
                format!("the rewind is to position {to}, the sequence is at {from}"),
            ));
        }
        if to == from {
            return Ok(());
        }
        self.clear_answers();
        self.session.round = None;
        self.session.opening = None;
        if to == 0 {
            self.session
                .forget()
                .map_err(|error| BackendError::new(DEVICE, error.to_string()))?;
        } else {
            self.session
                .rewind_to(to)
                .map_err(|error| BackendError::new(DEVICE, error.to_string()))?;
        }
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.session.state.room(positions, reset)
    }
}

/// The Inkling-2-Small arm, exposed to the shared runtime as a [Walk].
pub struct InklingWalk {
    shared: Arc<Shared>,
    layout: StateLayout,
    limits: Limits,
    /// The rows one lane step and one prompt chunk walk in, shared by every sequence: the engine
    /// steps one pass at a time, so a buffer a lane - about 0.7 MiB a row - is lent rather than
    /// held per lane.
    buffer: Mutex<Option<Chunk>>,
    /// The shared walk's buffer when drafting members verify together: as wide as the widest
    /// round.
    joint: Mutex<Option<Chunk>>,
    /// The most members one batch and one draft round take: the rank's verify width.
    max_batch: usize,
    drafter: InklingDrafter,
}

impl InklingWalk {
    /// The walk over the shared rank, serving sequences of the rank's context.
    pub(super) fn new(shared: Arc<Shared>) -> Result<Self, BackendError> {
        let config = &shared.config;
        let drafter = InklingDrafter {
            present: shared.draft.is_some(),
            kind: shared.draft.map(|(kind, _)| kind),
            depth: shared.draft.map_or(0, |(_, depth)| depth),
            wall: ExpertUnionWall {
                experts: config.n_routed_experts,
                top_k: config.num_experts_per_tok,
                ..JOINT_WALL
            },
        };
        let layout = layout_of(&shared);
        let limits = Limits::new(shared.prefill_chunk, shared.context).with_members(BATCH_ROWS);
        Ok(Self {
            shared,
            layout,
            limits,
            buffer: Mutex::new(None),
            joint: Mutex::new(None),
            max_batch: BATCH_ROWS,
            drafter,
        })
    }

    /// A row buffer, locked for one pass.
    fn lock_buffer<'a>(
        &self,
        which: &'a Mutex<Option<Chunk>>,
        name: &'static str,
    ) -> Result<std::sync::MutexGuard<'a, Option<Chunk>>, BackendError> {
        which
            .lock()
            .map_err(|_| BackendError::new(DEVICE, format!("the {name} buffer lock is poisoned")))
    }

    /// One prompt chunk, forwarded in the widths the arm's chunk schedule picks, exactly as
    /// [InklingDeviceSession::forward_chunk] does it.
    ///
    /// The `after` argument is how many positions the prompt still has after this chunk: the
    /// widths are chosen from the positions remaining alone ([ChunkWidths::next]), so a runtime
    /// chunk that ends inside one of the arm's chunks does not move a boundary.
    fn walk_bulk(
        &self,
        seq: &mut InklingSequence,
        rows: &[u32],
        after: usize,
        images: &[VisionImage],
    ) -> Result<Hidden, BackendError> {
        let shared = &self.shared;
        let start = seq.session.state.position;
        if start + rows.len() > shared.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {} positions, {} remain",
                    rows.len(),
                    shared.context.saturating_sub(start)
                ),
            ));
        }
        seq.session
            .state
            .grow(shared, start + rows.len())
            .map_err(|error| BackendError::new("context", error))?;
        let widths = shared.chunk_widths;
        // The row walk forwards a position at a time whatever the widths are, as the arm's own
        // prompt loop does; every chunked walk takes the widths after the positions remaining.
        let splits: Vec<usize> = if shared.prefill_walk.chunked() {
            chunk_splits(widths, rows.len(), after)
        } else {
            vec![1; rows.len()]
        };
        let row_bytes = shared.hidden() * std::mem::size_of::<f32>();
        let injections = self.image_injections(seq, images, start, rows.len(), row_bytes)?;
        let mut walked = 0usize;
        for take in splits {
            let slice = &rows[walked..walked + take];
            let first = seq.session.state.position;
            // The chunk's own image rows, with the source moved past the rows earlier in the
            // same run: an image row is always a chunk's, never the decode step's.
            let pieces: Vec<Injection> = injections
                .iter()
                .filter_map(|run| {
                    let from = run.row.max(walked);
                    let to = (run.row + run.count).min(walked + take);
                    (from < to).then(|| Injection {
                        row: from - walked,
                        source: run.source.offset((from - run.row) * row_bytes),
                        count: to - from,
                    })
                })
                .collect();
            let logits =
                if take == 1 && shared.prefill_walk != PrefillWalk::Tensor && pieces.is_empty() {
                    seq.session.step(slice[0])?
                } else {
                    let walk = if shared.prefill_walk.chunked() {
                        shared.prefill_walk
                    } else {
                        PrefillWalk::Chunk
                    };
                    let taps = seq.session.drafter.as_ref().map(DraftState::sink);
                    let mut held = self.lock_buffer(&self.buffer, "the lanes' row")?;
                    open_buffer(shared, &mut held, take, 1)?;
                    let chunk = held.as_mut().expect("the row buffer was just opened");
                    walk_rows(
                        shared,
                        Rows::Chunk(&mut seq.session.state),
                        chunk,
                        slice,
                        walk,
                        &pieces,
                        taps,
                    )?
                    .pop()
                    .ok_or_else(|| BackendError::new(DEVICE, "a chunk answered no row"))?
                };
            seq.session
                .inject(take, first)
                .map_err(|error| BackendError::new(DEVICE, error.to_string()))?;
            seq.session.history.extend_from_slice(slice);
            seq.session.last_logits = logits.clone();
            seq.answered = Some(logits);
            walked += take;
        }
        Ok(Hidden::new(rows.len()))
    }

    /// The prompt's encoded image rows as chunk-relative runs, one a fresh image whose span
    /// overlaps this chunk. `first` is the chunk's first position and `rows` its width, so the
    /// same run list serves every chunk the runtime cuts: only the overlap moves.
    fn image_injections(
        &self,
        seq: &InklingSequence,
        images: &[VisionImage],
        first: usize,
        rows: usize,
        row_bytes: usize,
    ) -> Result<Vec<Injection>, BackendError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let buffer = seq.session.images.ok_or_else(|| {
            BackendError::new(
                DEVICE,
                "a prompt chunk carries images the rank never encoded",
            )
        })?;
        let mut runs = Vec::with_capacity(images.len());
        let mut at = 0usize;
        for image in images {
            let count = image.span.grid.image_rows();
            let from = image.span.start.max(first);
            let to = (image.span.start + count).min(first + rows);
            if from < to {
                runs.push(Injection {
                    row: from - first,
                    source: buffer
                        .offset(at * row_bytes)
                        .offset((from - image.span.start) * row_bytes),
                    count: to - from,
                });
            }
            at += count;
        }
        Ok(runs)
    }

    /// One token a lane ([Rows::Lanes]), the arm's fused token walk.
    fn walk_tokens(
        &self,
        segments: &mut [RunSegment<'_, InklingSequence>],
    ) -> Result<Hidden, BackendError> {
        let shared = &self.shared;
        for segment in segments.iter_mut() {
            if segment.rows.len() != 1 {
                return Err(BackendError::new(
                    DEVICE,
                    "a token pass holds one row a member",
                ));
            }
            let token = segment.rows[0];
            let session = &mut segment.seq.session;
            if session.state.position >= shared.context {
                return Err(BackendError::new(
                    "context",
                    "a lane's context is exhausted",
                ));
            }
            if token as usize >= shared.config.vocab_size {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "token {token} is outside the vocabulary {}",
                        shared.config.vocab_size
                    ),
                ));
            }
            let rows = session.state.position + 1;
            session
                .state
                .grow(shared, rows)
                .map_err(|error| BackendError::new("context", error))?;
        }
        let tokens: Vec<u32> = segments.iter().map(|segment| segment.rows[0]).collect();
        let rows = {
            let mut held = self.lock_buffer(&self.buffer, "the lanes' row")?;
            open_buffer(
                shared,
                &mut held,
                tokens.len(),
                tokens.len().next_power_of_two(),
            )?;
            let chunk = held.as_mut().expect("the lanes' buffer was just opened");
            let mut states: Vec<&mut super::State> = segments
                .iter_mut()
                .map(|segment| &mut segment.seq.session.state)
                .collect();
            walk_rows(
                shared,
                Rows::Lanes(&mut states),
                chunk,
                &tokens,
                PrefillWalk::Chunk,
                &[],
                None,
            )?
        };
        for (segment, logits) in segments.iter_mut().zip(rows) {
            segment.seq.clear_answers();
            segment.seq.answered = Some(logits);
        }
        Ok(Hidden::new(tokens.len()))
    }

    /// One walk of every segment's anchor and drafts ([Rows::Segments]), answering every row's
    /// logits: the body the drafter's verify runs.
    fn verify_rows(
        &self,
        segments: &mut [RunSegment<'_, InklingSequence>],
        depths: &[usize],
    ) -> Result<Vec<Vec<Vec<f32>>>, BackendError> {
        let shared = &self.shared;
        let mut tokens = Vec::new();
        let mut shapes = Vec::with_capacity(segments.len());
        for (index, segment) in segments.iter_mut().enumerate() {
            if segment.rows.is_empty() {
                return Err(BackendError::new(DEVICE, "a verify segment holds no rows"));
            }
            let session = &mut segment.seq.session;
            if session.opening.is_none() {
                // A caller that walks rows of its own without the arm's round: the segment's
                // rows are the anchor and its drafts.
                let (anchor, drafts) = segment
                    .rows
                    .split_first()
                    .expect("the segment was just checked");
                session.opening = Some(Opening {
                    start: session.state.position,
                    anchor: *anchor,
                    drafts: drafts.to_vec(),
                    confidence: Vec::new(),
                });
            }
            let opening = session
                .opening
                .as_ref()
                .ok_or_else(|| BackendError::new(DEVICE, "a lane walks a round it never opened"))?;
            let depth = segment.rows.len() - 1;
            let given = depths.get(index).copied().unwrap_or(depth);
            if opening.anchor != segment.rows[0] || given != depth {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "a verify segment of {} rows (anchor {}) was opened on anchor {} at depth {given}",
                        segment.rows.len(),
                        segment.rows[0],
                        opening.anchor
                    ),
                ));
            }
            let tails = session
                .verify
                .as_ref()
                .map(|verify| verify.tails)
                .ok_or_else(|| BackendError::new(DEVICE, "a lane walks with no verify buffer"))?;
            tokens.extend_from_slice(segment.rows);
            shapes.push((depth, tails, session.drafter.as_ref().map(DraftState::sink)));
        }
        let walked = {
            let mut joint = self.lock_buffer(&self.joint, "the joint walk's")?;
            if joint
                .as_ref()
                .is_some_and(|chunk| chunk.rows < tokens.len())
            {
                let old = joint.take().expect("the joint buffer was just checked");
                let pointers = old.buffers();
                shared.cuda.free_buffers(&[&pointers[0], &pointers[1]]);
            }
            if joint.is_none() {
                // As wide as every lane's deepest round, so it opens once.
                let widest = self.max_batch * (self.drafter.depth + 1);
                let width = widest.max(tokens.len());
                *joint = Some(
                    Chunk::allocate(shared, width, width, true)
                        .map_err(|error| BackendError::new(DEVICE, error))?,
                );
            }
            let chunk = joint.as_mut().expect("the joint buffer was just opened");
            let mut walked: Vec<Segment<'_>> = segments
                .iter_mut()
                .zip(&shapes)
                .map(|(segment, &(depth, tails, taps))| Segment {
                    state: &mut segment.seq.session.state,
                    tails,
                    rows: depth + 1,
                    taps,
                })
                .collect();
            walk_rows(
                shared,
                Rows::Segments(&mut walked),
                chunk,
                &tokens,
                PrefillWalk::Chunk,
                &[],
                None,
            )?
        };
        let mut rows = walked.into_iter();
        Ok(shapes
            .iter()
            .map(|(depth, _, _)| rows.by_ref().take(depth + 1).collect())
            .collect())
    }
}

/// The rows one prompt pass of `rows` positions walks, with `after` positions still to come.
///
/// This is [ChunkWidths::next]'s rule applied until the pass is consumed. It is a separate
/// function so a test can hold a runtime-chunked prompt to the boundaries the arm's own prompt
/// loop walks: the widths depend only on the positions remaining, so a pass the runtime cuts at
/// its own chunk size falls on the same boundaries.
fn chunk_splits(widths: ChunkWidths, rows: usize, after: usize) -> Vec<usize> {
    let mut splits = Vec::new();
    let mut walked = 0usize;
    while walked < rows {
        let left = after + rows - walked;
        let take = widths.next(left).min(rows - walked);
        splits.push(take);
        walked += take;
    }
    splits
}

/// The arm's state, as the runtime needs to see it.
///
/// Every layer's key and value rows grow with position; the four convolution histories a layer
/// and the drafter's Markov feature are one live value at a time, which is what the
/// prompt-boundary checkpoint copies.
fn layout_of(shared: &Shared) -> StateLayout {
    let layers = shared.plan.layers().len();
    let layer_row = shared.kv_width() * 2 * std::mem::size_of::<f32>();
    let histories = shared.conv_slot_bytes().iter().sum::<usize>() * layers;
    let feature = shared
        .dspark
        .as_ref()
        .map_or(0, |model| model.feature_bytes());
    layout_from(layer_row, layers, histories, feature)
}

/// [layout_of] from its numbers, so a test can hold it without a device.
fn layout_from(layer_row: usize, layers: usize, histories: usize, feature: usize) -> StateLayout {
    let mut regions = Vec::new();
    if layer_row > 0 && layers > 0 {
        regions.push(Region::Positional {
            name: "kv-cache",
            bytes_per_row: layer_row * layers,
        });
    }
    if histories + feature > 0 {
        regions.push(Region::Recurrent {
            name: "conv-histories-and-drafter",
            bytes: histories + feature,
        });
    }
    StateLayout::new(regions)
}

/// The arm's draft source: prompt lookup or the DSpark block drafter, over the arm's own round.
pub struct InklingDrafter {
    /// Whether the rank has a drafter at all (SPARKFLASH_INKLING_DRAFT).
    present: bool,
    /// Which one it is, for the ledger.
    kind: Option<DraftKind>,
    /// The most drafts a round verifies (SPARKFLASH_INKLING_DRAFT_DEPTH).
    depth: usize,
    wall: ExpertUnionWall,
}

impl Drafter<InklingWalk> for InklingDrafter {
    const PRESENT: bool = true;

    /// The rank drafts only when SPARKFLASH_INKLING_DRAFT named one. Off, the runtime runs one
    /// plain token pass over every lane, the arm's old fused `Rows::Lanes` step, rather than a
    /// held decode a lane.
    fn drafts(&self) -> bool {
        self.present
    }

    /// The arm's round already folded each member's rows into its step: a verify answers
    /// [InklingDeviceSession::close]'s step.
    type Rows = StepOutput;

    fn wall(&self) -> &dyn StepWall {
        &self.wall
    }

    fn open_round(&self, seq: &mut InklingSequence, token: u32) -> Result<Draft, BackendError> {
        if !self.present {
            // No drafter on this rank: the token's step is the plain one, and no row is left to
            // verify.
            return Ok(Draft::Held(seq.session.decode(token)?));
        }
        match seq.session.open_round(token)? {
            Opened::Held(step) => Ok(Draft::Held(step)),
            Opened::Round(chain) => {
                let opening =
                    seq.session.opening.as_ref().ok_or_else(|| {
                        BackendError::new(DEVICE, "a round opened with no opening")
                    })?;
                let room = seq.session.shared.context.saturating_sub(opening.start + 1);
                // A lone member verifies the depth the arm's own policy picked, so the chain the
                // walk sees is the one the arm drafted for; a joint round shares it.
                let chosen = seq
                    .session
                    .speculator
                    .depth(&opening.confidence, room)
                    .min(chain.len())
                    .min(opening.drafts.len());
                let mut confidence = chain;
                confidence.truncate(chosen);
                Ok(Draft::Chain {
                    confidence,
                    drafts: opening.drafts[..chosen].to_vec(),
                })
            }
        }
    }

    fn verify(
        &self,
        walk: &InklingWalk,
        segments: &mut [RunSegment<'_, InklingSequence>],
        depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let answered = walk.verify_rows(segments, depths)?;
        segments
            .iter_mut()
            .zip(answered)
            .map(|(segment, rows)| {
                let depth = segment.rows.len().saturating_sub(1);
                segment.seq.session.close(depth, rows)
            })
            .collect()
    }

    fn settle(
        &self,
        _seq: &mut InklingSequence,
        _chunk: &[u32],
        rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        // [InklingDeviceSession::close] settled the round inside the verify.
        Ok(rows)
    }

    fn budget(&self, seqs: usize) -> usize {
        // One verify is the members' anchors and their drafts, and the rank's verify batch
        // holds [BATCH_ROWS] rows: members x (1 + depth) never exceeds it.
        (seqs * self.depth).min(BATCH_ROWS.saturating_sub(seqs))
    }

    fn ready(&self, seq: &InklingSequence, token: u32) -> bool {
        seq.session.held(token)
    }

    fn give_back(&self, seq: &mut InklingSequence, to: usize) -> Result<(), BackendError> {
        let count = seq.session.history.len().saturating_sub(to);
        seq.session.give_back(count)
    }

    fn speculation(&self, seq: &InklingSequence) -> Speculation {
        seq.session.speculator.stats()
    }

    fn describe(&self) -> String {
        let name = match self.kind {
            Some(DraftKind::Lookup) => "prompt lookup",
            Some(DraftKind::Dspark) => "DSpark",
            None => {
                return "SPARKFLASH_INKLING_DRAFT is off: the anchor alone, its round the \
                        plain step"
                    .to_owned();
            }
        };
        format!("{name}, up to {} drafts", self.depth)
    }
}

/// Inkling's image path: rank 0's vision tower, encoded once a prompt.
///
/// The runtime calls [Vision::encode] when a prompt starts over or carries fresh images after
/// the reused prefix, and [Vision::clear] when its chunks are done. Rank 0 alone runs the
/// tower; every rank's rows arrive through the group's sum, so a worker's buffer carries rank
/// 0's rows exactly as its prompt path did ([InklingDeviceSession::encode_images] before this).
pub struct InklingVision;

impl Vision<InklingWalk> for InklingVision {
    const PRESENT: bool = true;

    fn encode(
        walk: &InklingWalk,
        seq: &mut InklingSequence,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<(), BackendError> {
        let shared = &walk.shared;
        shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        shared
            .span_check()
            .check(tokens, images)
            .map_err(|error| BackendError::new("request", error))?;
        seq.session.free_images();
        if images.is_empty() {
            return Ok(());
        }
        if shared.encodes() && shared.vision.is_none() {
            return Err(BackendError::new(
                "vision",
                format!(
                    "the request carries {} images and this rank holds no vision tower",
                    images.len()
                ),
            ));
        }
        // The images the reused prefix does not already hold: the arm encoded the others in
        // the prompt that wrote those positions.
        let start = seq.session.state.position;
        let fresh: Vec<&VisionImage> = images
            .iter()
            .filter(|image| image.span.start >= start)
            .collect();
        if fresh.is_empty() {
            return Ok(());
        }
        let hidden = shared.hidden();
        let row_bytes = hidden * std::mem::size_of::<f32>();
        let rows: usize = fresh.iter().map(|image| image.span.grid.image_rows()).sum();
        let mut total = 0usize;
        let buffer = shared
            .cuda
            .alloc(rows * row_bytes, &mut total)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let fill = || -> Result<(), BackendError> {
            shared
                .cuda
                .memset_d8_async(buffer, 0, rows * row_bytes)
                .map_err(|error| BackendError::new(DEVICE, error))?;
            if shared.encodes() {
                let tower = shared
                    .vision
                    .as_ref()
                    .ok_or_else(|| BackendError::new(DEVICE, "rank 0 holds no vision tower"))?;
                let mut tower = tower.lock().map_err(|_| {
                    BackendError::new(DEVICE, "the vision tower's lock is poisoned")
                })?;
                let mut at = 0usize;
                for image in &fresh {
                    let grid = &image.span.grid;
                    let made = tower
                        .encode(
                            &shared.cuda,
                            &image.pixels,
                            grid.best_height,
                            grid.best_width,
                            buffer.offset(at * row_bytes),
                        )
                        .map_err(|error| BackendError::new(DEVICE, error))?;
                    if made != grid.image_rows() {
                        return Err(BackendError::new(
                            DEVICE,
                            format!(
                                "the tower made {made} rows of a span that holds {}",
                                grid.image_rows()
                            ),
                        ));
                    }
                    at += made;
                }
            }
            let piece = shared.prefill_chunk.max(1);
            let mut done = 0usize;
            while done < rows {
                let count = piece.min(rows - done);
                exchange_sum(shared, buffer.offset(done * row_bytes), count * hidden)?;
                done += count;
            }
            Ok(())
        };
        match fill() {
            Ok(()) => {
                seq.session.images = Some(buffer);
                Ok(())
            }
            Err(error) => {
                let _ = shared.cuda.synchronize();
                shared.cuda.free_buffers(&[&buffer]);
                Err(error)
            }
        }
    }

    fn clear(walk: &InklingWalk, seq: &mut InklingSequence) {
        if let Err(error) = walk.shared.cuda.make_current() {
            debug_assert!(false, "the vision path's clear lost its context: {error}");
            return;
        }
        seq.session.free_images();
    }
}

impl Walk for InklingWalk {
    type Seq = InklingSequence;
    type Drafter = InklingDrafter;
    type Vision = InklingVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        // A prompt runs in the arm's chunks, a step runs one row a lane, and a round verifies an
        // anchor and its drafts in one walk.
        passes: PassMask::TOKEN_ROWS
            .union(PassMask::BULK)
            .union(PassMask::VERIFY),
        segments: Segments::Many,
        state: StateKind::Both,
        // The four rows the arm states in sparkflash_model::ARM_LEDGERS; a test below holds
        // them equal so the walk and the table cannot drift.
        compute: ComputeDeclaration {
            prompt: State::Native("dense projections and NVFP4/MXFP8 experts on the grouped MMAs"),
            token: State::Off("row token GEMV; decode.tensor_cores=true is refused"),
            stored: State::Native("NVFP4 and MXFP8 experts, BF16 dense, in their stored formats"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self, memory: LaneMemory) -> Result<Self::Seq, BackendError> {
        let _ = memory;
        InklingDeviceSession::open(&self.shared).map(InklingSequence::new)
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn max_segments(&self) -> usize {
        // A prompt pass is one segment; a step and a verify take the rank's verify width.
        self.max_batch
    }

    fn rewind_slack(&self) -> usize {
        // The sliding cache keeps its window and a widest chunk of rows past the prompt's end;
        // the arm's old rewind point also reserved the drafts a verify may add.
        self.shared.prefill_chunk.saturating_sub(self.drafter.depth)
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        seq.session.set_sampling(sampling);
        seq.clear_answers();
    }

    fn walk(
        &self,
        segments: &mut [RunSegment<'_, Self::Seq>],
        pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let hidden = match pass.kind {
            PassKind::Bulk { .. } => {
                if segments.len() != 1 {
                    return Err(BackendError::new(
                        DEVICE,
                        "a prompt chunk holds one segment",
                    ));
                }
                let after = pass.prompt_end.unwrap_or(0);
                let segment = &mut segments[0];
                let images = segment.images;
                let (seq, rows) = (&mut *segment.seq, segment.rows);
                self.walk_bulk(seq, rows, after, images)?
            }
            PassKind::TokenRows => self.walk_tokens(segments)?,
            PassKind::Verify { .. } => {
                let depths: Vec<usize> = segments
                    .iter()
                    .map(|segment| segment.rows.len().saturating_sub(1))
                    .collect();
                let answered = self.verify_rows(segments, &depths)?;
                let mut rows = 0;
                for (segment, answered) in segments.iter_mut().zip(answered) {
                    segment.seq.clear_answers();
                    segment.seq.verified_from = rows;
                    rows += answered.len();
                    segment.seq.verified = answered;
                }
                Hidden::new(rows)
            }
            PassKind::Paired { .. } => {
                return Err(BackendError::new(
                    DEVICE,
                    "the Inkling arm does not pair several prompts in one pass",
                ));
            }
        };
        if let Some(hook) = pass.hook {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new(DEVICE, error))?;
        let shared = &self.shared;
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            if row.row >= hidden.rows {
                return Err(BackendError::new(
                    DEVICE,
                    format!("the head was asked for row {} of {}", row.row, hidden.rows),
                ));
            }
            let seq = &mut *row.seq;
            let step = match &row.input {
                HeadInput::Prompt(tokens) => {
                    let logits = seq.answered.take().ok_or_else(|| {
                        BackendError::new(DEVICE, "a prompt head has no chunk's logits")
                    })?;
                    // The penalties count the whole prompt, the reused prefix included: it is
                    // one request, whatever the number of chunks it ran in.
                    seq.session.sampler.reset();
                    seq.session.sampler.record_prompt(tokens);
                    seq.session.last_logits = logits.clone();
                    seq.session.output(logits)
                }
                HeadInput::Token(token) => {
                    let logits = seq
                        .answered
                        .take()
                        .ok_or_else(|| BackendError::new(DEVICE, "a token head has no row"))?;
                    seq.session.decoded(*token, logits)
                }
                HeadInput::Verify { rows, row: index } => {
                    let token = rows.get(*index).copied().unwrap_or_default();
                    let at = row.row.checked_sub(seq.verified_from).ok_or_else(|| {
                        BackendError::new(DEVICE, "a verify row is before its pass")
                    })?;
                    let logits = seq.verified.get(at).cloned().ok_or_else(|| {
                        BackendError::new(DEVICE, format!("the verify pass has no row {at}"))
                    })?;
                    seq.session.sampler.record(token);
                    let next_token = choose(
                        &shared.config,
                        &mut seq.session.sampler,
                        &logits,
                        &seq.session.sampling,
                    );
                    StepOutput {
                        logits,
                        next_token,
                        draft: Vec::new(),
                    }
                }
            };
            steps.push(step);
        }
        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_model::ledger::Strategy;
    use sparkflash_runtime::{ChunkPolicy, Walk, ledger_for};

    use super::super::{
        DEFAULT_PREFILL_CHUNK, LONG_PREFILL_CHUNK, LONG_PREFILL_FROM, MAX_PREFILL_CHUNK,
    };
    use super::*;

    fn widths() -> ChunkWidths {
        ChunkWidths {
            base: DEFAULT_PREFILL_CHUNK,
            long: LONG_PREFILL_CHUNK,
            long_from: LONG_PREFILL_FROM,
        }
    }

    /// The four compute rows the walk declares are the arm's static declaration, so the boot
    /// ledger and the arms table cannot drift.
    #[test]
    fn the_declared_compute_rows_are_the_arm_ledger_rows() {
        use sparkflash_model::ledger::ledger_for_arm;
        let declared = ledger_for_arm("Inkling").expect("Inkling's declaration");
        let capabilities = <InklingWalk as Walk>::CAPABILITIES.compute;
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

    /// Every chunk width the arm can walk is a documented policy size, and the limits state the
    /// widest prompt chunk with the rank's verify width for members.
    #[test]
    fn the_arms_three_chunk_widths_are_policy_sizes() {
        for size in [DEFAULT_PREFILL_CHUNK, LONG_PREFILL_CHUNK, MAX_PREFILL_CHUNK] {
            assert_eq!(
                ChunkPolicy::for_limits(Limits::new(size, 1_048_576)).rows(),
                size,
                "{size} rows must be a documented ChunkPolicy size"
            );
        }
        let limits = Limits::new(LONG_PREFILL_CHUNK, 1_048_576).with_members(BATCH_ROWS);
        assert_eq!(limits.rows, LONG_PREFILL_CHUNK);
        assert_eq!(limits.members, BATCH_ROWS);
        assert_eq!(ChunkPolicy::for_limits(limits).rows(), LONG_PREFILL_CHUNK);
    }

    /// A prompt the runtime cuts at its policy width falls on the same boundaries as the arm's
    /// own prompt loop: the widths depend on the positions remaining alone.
    #[test]
    fn the_runtime_chunks_do_not_move_the_arms_prompt_boundaries() {
        let widths = widths();
        for total in [
            1, 4_095, 4_096, 4_097, 8_192, 10_000, 16_383, 16_384, 40_000, 65_535, 100_000,
        ] {
            let want = chunk_splits(widths, total, 0);
            let pass = LONG_PREFILL_CHUNK;
            let mut got = Vec::new();
            let mut start = 0usize;
            while start < total {
                let rows = pass.min(total - start);
                let after = total - start - rows;
                got.extend(chunk_splits(widths, rows, after));
                start += rows;
            }
            assert_eq!(got, want, "{total} positions split differently");
        }
        // The released walk's own examples: a long prompt widens until its tail.
        assert_eq!(chunk_splits(widths, 16_384, 0), [8_192, 4_096, 4_096]);
        assert_eq!(chunk_splits(widths, 10_000, 0), [4_096, 4_096, 1_808]);
    }

    /// A verify is the members' anchors plus their drafts, and the share's budget caps it at
    /// the rank's [BATCH_ROWS]-row verify width.
    #[test]
    fn the_verify_budget_caps_members_times_depth() {
        let drafter = InklingDrafter {
            present: true,
            kind: Some(DraftKind::Dspark),
            depth: 7,
            wall: ExpertUnionWall {
                experts: 0,
                top_k: 0,
                ..JOINT_WALL
            },
        };
        for seqs in 1..=BATCH_ROWS {
            let rows = seqs + drafter.budget(seqs);
            assert!(
                rows <= BATCH_ROWS,
                "{seqs} members and {} drafts is {rows} rows",
                drafter.budget(seqs)
            );
        }
        assert_eq!(drafter.budget(4), BATCH_ROWS - 4);
        assert_eq!(drafter.budget(8), 0);
        assert_eq!(drafter.describe(), "DSpark, up to 7 drafts");
        assert!(drafter.drafts());
        let off = InklingDrafter {
            present: false,
            kind: None,
            depth: 0,
            wall: ExpertUnionWall {
                experts: 0,
                top_k: 0,
                ..JOINT_WALL
            },
        };
        assert!(
            !off.drafts(),
            "an off drafter must take the plain lanes step"
        );
    }

    /// The layout is both kinds: the key and value rows are positional, the convolution
    /// histories and the drafter's feature need the prompt-boundary checkpoint.
    #[test]
    fn the_layout_is_positional_caches_and_recurrent_histories() {
        let layout = layout_from(64, 48, 1_024, 256);
        assert_eq!(layout.kind(), StateKind::Both);
        assert!(layout.needs_checkpoint());
        assert_eq!(layout.bytes_per_row(), 64 * 48);
        assert_eq!(layout.bytes_at(2), 64 * 48 * 2 + 1_280);
        assert_eq!(
            layout_from(0, 0, 0, 0).kind(),
            StateKind::None,
            "a rank with no state keeps nothing"
        );
    }

    /// The runtime's ledger states every row the walk declares, with no "does not state it".
    #[test]
    fn the_runtime_ledger_states_every_row_the_walk_declares() {
        let ledger = ledger_for::<InklingWalk>();
        assert!(
            !ledger.line().contains("does not state it"),
            "the runtime derives every row Inkling states: {}",
            ledger.line()
        );
    }

    /// The walk and its sequences are what a fused engine thread takes.
    #[test]
    #[allow(clippy::assertions_on_constants)] // the claim is the point: it is a type constant
    fn the_walk_and_its_sequences_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InklingWalk>();
        assert_send_sync::<InklingSequence>();
        assert_eq!(<InklingWalk as Walk>::CAPABILITIES.segments, Segments::Many);
        assert!(
            <<InklingWalk as Walk>::Vision as Vision<InklingWalk>>::PRESENT,
            "the arm serves images through its tower"
        );
    }

    sparkflash_runtime::conformance!(super::InklingWalk);
}
