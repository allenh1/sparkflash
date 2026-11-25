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

//! The seam between `sparkflash serve --tp4-device` and the device forward loop.
//!
//! The device forward runs inside the `device-chain` epoch: the arena, the bound weights, the
//! resident kernels and the NCCL communicator all live on that call's stack, and none of them is
//! `Send`. Serving therefore does not pull the forward out into a server; it hands the forward a
//! [`DeviceLink`] before the epoch starts ([`install`]), and the DSpark branch of the forward loop
//! takes it ([`take`]) and runs its request loop on the epoch's own thread instead of the one-shot
//! harness generation.
//!
//! The loop serves several sequences at once in scheduler ticks. Every rank holds a link. Rank 0's
//! link is fed by the HTTP driver and broadcasts, before each tick, the requests that join and the
//! sequences that end there (a stop string matched, or the client left) over the serve control
//! channel; a worker's link reads the same frames. Every rank runs the same scheduler on the same
//! decisions, so every rank runs the same steps with the same samplers, emits the same tokens (the
//! sampler's noise is a pure function of the seed and the position), and ends every sequence at
//! the same tick.
//!
//! A request's images (DECISIONS.md D59) travel with it: every rank gets each span's place, grid
//! and content key, and the rank that owns the image placeholder's embedding row ([`tower_rank`])
//! also gets the canvas and runs the vision tower; the span rows reach the other ranks through the
//! embedding exchange, like any token's.

use std::path::Path;
use std::sync::{Arc, Mutex};

/// The vision tower's cubin in the `--tp4-device` cubin directory: serving takes images when it is
/// there and SPARKFLASH_VISION is not 0.
pub const VISION_CUBIN: &str = "opt_vision_sm121.cubin";

/// Whether the device path serves images with the cubins in `cubins`: every rank asks the same
/// question of the same staged directory, so every rank gives the same answer.
#[must_use]
pub fn vision_enabled(cubins: &Path) -> bool {
    sparkflash_model::active_profile()
        .diagnostics
        .vision
        .unwrap_or(true)
        && cubins.join(VISION_CUBIN).is_file()
}

/// The rank that runs the vision tower: the owner of the image placeholder's embedding row.
#[must_use]
pub fn tower_rank() -> usize {
    sparkflash_vision::IMAGE_TOKEN_ID as usize / crate::plan::vocab_per_rank()
}

/// A structured-output constraint a request carries: the schema, and the table that turns it into
/// admissible vocabulary ids.
///
/// The table is process-wide (it is the checkpoint tokenizer's, built once), so two specs with the
/// same schema compare equal whatever Arc they hold: that is what keeps DeviceJob's derived
/// PartialEq meaningful across ranks.
#[derive(Debug, Clone)]
pub struct ConstraintSpec {
    /// The JSON Schema the document must satisfy.
    pub schema: serde_json::Value,
    /// The vocabulary's bytes.
    pub vocabulary: Arc<sparkflash_api::vocab::TokenBytes>,
    /// The end-of-sequence id, admissible only where the document may end.
    pub end: u32,
}

impl PartialEq for ConstraintSpec {
    fn eq(&self, other: &Self) -> bool {
        self.schema == other.schema && self.end == other.end
    }
}

impl ConstraintSpec {
    /// A spec for schema, or None when no vocabulary table was installed (a process that never
    /// loaded a tokenizer cannot enforce a constraint, and must not silently ignore one).
    #[must_use]
    pub fn new(schema: serde_json::Value, end: u32) -> Option<Self> {
        Some(Self {
            schema,
            vocabulary: vocabulary()?,
            end,
        })
    }
}

/// The process's vocabulary table, installed once the tokenizer is loaded.
static VOCABULARY: std::sync::OnceLock<Arc<sparkflash_api::vocab::TokenBytes>> =
    std::sync::OnceLock::new();

/// Installs the vocabulary table (the first call wins; every later one is a no-op).
pub fn install_vocabulary(table: Arc<sparkflash_api::vocab::TokenBytes>) {
    let _ = VOCABULARY.set(table);
}

/// The vocabulary table, when this process loaded a tokenizer.
#[must_use]
pub fn vocabulary() -> Option<Arc<sparkflash_api::vocab::TokenBytes>> {
    VOCABULARY.get().cloned()
}

/// One request as the device loop runs it: identical on every rank.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceJob {
    /// The rendered prompt ids (the chat template is already applied).
    pub prompt: Vec<u32>,
    /// The most completion tokens to emit.
    pub max_tokens: usize,
    /// 0 is greedy.
    pub temperature: f32,
    /// 1 keeps the whole distribution.
    pub top_p: f32,
    /// 0 keeps every token.
    pub top_k: u32,
    /// The sampler seed (rank 0 draws one for an unseeded request).
    pub seed: u64,
    /// Token ids that end the request (emitted, then stop): the end-of-sequence id unless the
    /// request ignores it.
    pub stop_ids: Vec<u32>,
    /// The prompt's images in prompt order: each span's place in `prompt` (every span position
    /// holds the placeholder id), grid and content key. The canvas is there on the
    /// [`tower_rank`] only and empty on the others.
    pub images: Vec<sparkflash_api::VisionImage>,
    /// The structured-output constraint, when the request asked for one. A constrained request is
    /// sampled greedily over the ids the schema admits; a drafted row is masked by the automaton
    /// state at its own position.
    pub constraint: Option<Arc<ConstraintSpec>>,
}

/// Why a device request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFinish {
    /// A stop id was emitted.
    Stop,
    /// `max_tokens` or the context ran out.
    Length,
    /// Rank 0 asked every rank to stop (a stop string matched, or the client went away).
    Aborted,
}

/// What one request did, measured on the rank that ran it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceOutcome {
    /// Completion tokens emitted (the first token from the prefill included).
    pub completion_tokens: usize,
    /// Why it ended.
    pub finish: DeviceFinish,
    /// The prefill's wall clock up to the first token, in ms.
    pub prefill_ms: f64,
    /// The decode steps' wall clock after the first token, in ms.
    pub decode_ms: f64,
    /// Verify steps run.
    pub steps: usize,
    /// Drafts verified, over all steps.
    pub drafts_verified: usize,
    /// Drafts accepted, over all steps.
    pub drafts_accepted: usize,
    /// Leading prompt tokens whose state the previous request left on the device (the shared
    /// prefix), so the prefill skipped them.
    pub cached_tokens: usize,
}

/// Rank 0's decisions for one scheduler tick: the same on every rank.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceTick {
    /// Requests that join, in arrival order, each with the sequence id rank 0 gave it.
    pub admit: Vec<(u64, DeviceJob)>,
    /// Sequences that end before the tick runs (a stop string matched, or the client left).
    pub cancel: Vec<u64>,
    /// The tick holds a long prefill back for a decode step: rank 0's scheduler decides on its
    /// clock (the decoders' share of a long prefill's time) and every rank holds the same ticks.
    pub hold_prefill: bool,
}

/// One rank's end of the serving loop.
pub trait DeviceLink: Send {
    /// The next tick's decisions, the same on every rank; `None` ends the loop. With `idle` (no
    /// sequence is on the device or waiting for it) it blocks until a request arrives; otherwise
    /// it returns at once. It admits at most `room` requests. `hold_prefill` is this rank's
    /// scheduler's answer for [`DeviceTick::hold_prefill`]: rank 0's is the one every rank gets.
    ///
    /// # Errors
    /// The control channel failed; the group cannot continue.
    fn tick(
        &mut self,
        idle: bool,
        room: usize,
        hold_prefill: bool,
    ) -> Result<Option<DeviceTick>, String>;

    /// Rank 0 decides the next tick now, under the running tick's device work, and sends it:
    /// the workers find it waiting when they finish this tick, and the next [`Self::tick`] returns
    /// it. `room` and `hold_prefill` are as for that call; it never blocks (a request that comes
    /// later waits for the tick after). A cancellation decided after it waits for the tick after
    /// too: the sequence runs one more step, its tokens dropped. A worker's link ignores it.
    ///
    /// # Errors
    /// The control channel failed; the group cannot continue.
    fn ahead(&mut self, _room: usize, _hold_prefill: bool) -> Result<(), String> {
        Ok(())
    }

    /// Sequence `seq`'s tokens from this tick (its first carries the prefill's token). `last` is
    /// true when the sequence ends with them. When rank 0's stream must end before that (a stop
    /// string, the client left), its next tick cancels the sequence.
    fn emit(&mut self, seq: u64, tokens: &[u32], last: bool);

    /// Sequence `seq` ended: its outcome, or the error that ended it. A step's error is fatal to
    /// the group (the ranks may have desynchronised): the loop ends every sequence with it and
    /// stops.
    ///
    /// # Errors
    /// The control channel failed.
    fn finish(&mut self, seq: u64, outcome: Result<&DeviceOutcome, &str>) -> Result<(), String>;

    /// The loop failed with `error` and stops; the scheduler is still up. Tearing the epoch down
    /// after a failed step may never return (a faulted context, a collective whose peers have
    /// gone), so a serving link ends the process from here rather than after it.
    fn fail(&mut self, _error: &str) {}
}

static LINK: Mutex<Option<Box<dyn DeviceLink>>> = Mutex::new(None);

/// Installs the link the next device forward epoch serves through.
pub fn install(link: Box<dyn DeviceLink>) {
    *LINK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(link);
}

/// Whether a link is installed (the next device epoch serves through it).
pub fn installed() -> bool {
    LINK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

/// Takes the installed link, if any: the forward loop serves through it instead of running the
/// one-shot generation.
pub fn take() -> Option<Box<dyn DeviceLink>> {
    LINK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

/// One command of the runtime's walk ([`crate::base_walk::BaseWalk`]) to the device forward.
///
/// The conformance fixture drives the base through the shared runtime's [`sparkflash_runtime::Walk`]
/// contract instead of the serving link: each command is one of the contract's operations on one
/// lane, run on the epoch's own thread (the forward's state is not `Send`) by
/// [`crate::tp4_forward`]'s walk server, which reuses the serving scheduler's prefill and token
/// rows bodies. Only a one-rank epoch takes it; a group's ranks would need the commands mirrored.
#[derive(Debug, Clone, PartialEq)]
pub enum WalkCommand {
    /// The epoch's limits ([`WalkDescription`]).
    Describe,
    /// A lane for a new sequence at position 0; answers [`WalkAnswer::Opened`].
    Open,
    /// The sequence on `lane` ends and the lane's memory goes back.
    Release {
        /// The lane the sequence held.
        lane: usize,
    },
    /// The sequence on `lane` goes back to position `to`: its counters to the resume point at
    /// or below it (a recurrent block's saved point, or the start), and the rows after that point
    /// run again.
    Rewind {
        /// The sequence's lane.
        lane: usize,
        /// The position it goes back to.
        to: usize,
    },
    /// Prompt rows on `lane` in one pass, the last row headed under `sampler`; answers one pick.
    /// `more` says more rows of the same prompt follow, so the next chunk continues this one's
    /// packed-key run; a chunk after one without it starts a run, as a placed prompt does.
    Chunk {
        /// The sequence's lane.
        lane: usize,
        /// The prompt's ids this pass walks.
        rows: Vec<u32>,
        /// The sampler the last row is headed under.
        sampler: sparkflash_model::gumbel::GumbelSampler,
        /// Whether more rows of the same prompt follow.
        more: bool,
    },
    /// One token row a lane, in one step, each headed under its own sampler; answers one pick a
    /// row, in order.
    Tokens {
        /// Each row's lane, id and sampler.
        rows: Vec<(usize, u32, sparkflash_model::gumbel::GumbelSampler)>,
    },
    /// Whether the sequence on `lane` can be backed for `positions` positions.
    Room {
        /// The sequence's lane.
        lane: usize,
        /// The positions it would reach.
        positions: usize,
    },
    /// The draft block after `anchor` on `lane`, its rows' picks under `sampler`; answers
    /// [`WalkAnswer::Chain`]. The anchor is the sequence's next row, not yet walked.
    Draft {
        /// The sequence's lane.
        lane: usize,
        /// The sequence's next row, the block drafts after.
        anchor: u32,
        /// The sampler the drafted rows' picks are taken under.
        sampler: sparkflash_model::gumbel::GumbelSampler,
    },
    /// One rows step verifying each member's lane and chunk (its drafted anchor, then the
    /// drafts kept for it), every member its own segment; each keeps its anchor and the drafts
    /// its picks agree with. Answers [`WalkAnswer::Verified`].
    Verify {
        /// Each member's lane and chunk: its anchor, then the drafts kept for it.
        members: Vec<(usize, Vec<u32>)>,
    },
}

/// What the walk server knows about the epoch, for [`sparkflash_runtime::Limits`] and the rewind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkDescription {
    /// The lanes the scheduler holds (one without the device exchange).
    pub lanes: usize,
    /// The most sequences one token pass steps.
    pub members: usize,
    /// The most positions a sequence may reach.
    pub context: usize,
    /// The widest prompt pass the walk takes: the epoch's bulk chunk ([crate::tp4_forward::prefill_chunk]).
    pub prompt_rows: usize,
    /// The resume boundaries' step: a sequence goes back only to a multiple of it.
    pub resume_step: usize,
    /// Whether the rows' embeddings and samples are exchanged on the device.
    pub exchange: bool,
    /// How many positions past a rewind point a sequence may write and still go back to it: the
    /// window rings a lane keeps behind its position, less the two windows a resume reads.
    pub slack: usize,
    /// A lane's bytes a position, for the runtime's chunk policy.
    pub row_bytes: usize,
    /// A lane's live recurrent state's bytes, or 0 when the stack has no recurrent block.
    pub recurrent_bytes: usize,
    /// The fewest prompt rows a chunk takes the bulk path with, or 0 when it never does.
    pub bulk_rows: usize,
    /// The drafts a block proposes, or 0 when the epoch drafts nothing.
    pub depth: usize,
    /// The most rows one verify step holds, its anchors included.
    pub verify_rows: usize,
    /// What a shared verify step costs, for the runtime's row share.
    pub wall: sparkflash_runtime::BandWall,
    /// The least chance a draft must have to be given a verify row.
    pub floor: f64,
}

/// The walk server's answer to one [`WalkCommand`].
#[derive(Debug, Clone, PartialEq)]
pub enum WalkAnswer {
    /// [`WalkCommand::Describe`].
    Described(WalkDescription),
    /// [`WalkCommand::Open`]: the lane.
    Opened(usize),
    /// A command with nothing to answer.
    Done,
    /// The picks of a [`WalkCommand::Chunk`] or [`WalkCommand::Tokens`].
    Picks(Vec<u32>),
    /// [`WalkCommand::Room`].
    Room(bool),
    /// [`WalkCommand::Draft`]: the drafts, and the chance each leading run is accepted.
    Chain {
        /// The drafted ids, in order.
        drafts: Vec<u32>,
        /// The chance each leading run of the drafts is accepted.
        confidence: Vec<f64>,
    },
    /// [`WalkCommand::Verify`]: each member's picks for its anchor and the drafts it kept, the
    /// last one the model's own next token.
    Verified(Vec<Vec<u32>>),
}

/// One command and where its answer goes.
pub struct WalkRequest {
    /// The command.
    pub command: WalkCommand,
    /// The answer, or the server's refusal.
    pub reply: std::sync::mpsc::Sender<Result<WalkAnswer, String>>,
}

static WALK: Mutex<Option<std::sync::mpsc::Receiver<WalkRequest>>> = Mutex::new(None);

/// Installs the walk the next device forward epoch serves ([`WalkCommand`]); answers the end the
/// walk sends its commands to.
pub fn install_walk() -> std::sync::mpsc::Sender<WalkRequest> {
    let (send, receive) = std::sync::mpsc::channel();
    *WALK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(receive);
    send
}

/// Takes the installed walk, if any: the forward loop serves its commands until every sender is
/// gone.
pub fn take_walk() -> Option<std::sync::mpsc::Receiver<WalkRequest>> {
    WALK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

/// Whether the next device epoch serves (a link or a walk is installed): it then holds lanes.
pub fn serving() -> bool {
    installed()
        || WALK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
}
