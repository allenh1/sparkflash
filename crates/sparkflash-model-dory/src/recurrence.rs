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

//! The latent-recurrence group: Dory's shared block run once a rank, with the latent passed on.
//!
//! This is not tensor parallelism: no rank holds a slice of a layer, and nothing is reduced.
//! Each rank is a [DoryStage], a run of recurrent loops over the whole hidden width. Rank 0 owns
//! the sequences, embeds the tokens and runs the input prefix and its loops; each later rank
//! receives the latent rows (`state` and the reinjected `input_h`), runs its loops on them and
//! passes them on; the last rank also runs the output suffix and the vocabulary head, chooses the
//! token with the shared [Sampler], and answers rank 0 with the token and its logits row. Every
//! rank keeps the KV caches of its own blocks and loops only.
//!
//! The group holds as many sequences as it has lanes, one a concurrent request. Every rank keeps
//! each lane's caches and position; the lanes on one rank share one scratch set, because a rank
//! runs one step at a time on one stream. While one lane's step is on rank 2, another's can be on
//! rank 1 and a third's on rank 0, so the ranks that sit idle behind one sequence's step work on
//! the others'.
//!
//! Decode steps travel in batches. A session's step joins rank 0's queue, and one session at a
//! time drives: it waits until rank 0 has run the last batch's rows, takes the steps waiting (at
//! most [LANE_BATCH]), and runs them as one pass on every rank ([StageCore::forward_lanes]),
//! which reads each projection's weights once for all of them. A stage's pass costs about the
//! same at one row as at eight, since it is the weight reads, so the fewer batches the lanes
//! travel in the sooner each comes round: a batch's answers come back as one message and wake
//! its sessions together, and the driver waits up to [RETURN_WINDOW] for the lanes whose answers
//! it has just read, rather than send the first one back alone and the rest a rank-0 pass later.
//! The steps that come back while rank 0 runs a batch join the next. The driver also holds a
//! batch while as many are on the ring as there are ranks after rank 0 (more when the lanes need
//! more than [LANE_BATCH] a batch): the slowest rank then always has one to run, and another
//! batch would only queue in front of it while its lanes could have ridden in a wider one. A
//! lane's row has the bits of a pass of its own whatever it shares a pass with.
//!
//! The ring is one-directional. A frame of [FRAME_WORDS] words heads every message, so a worker
//! learns what follows before it posts the payload's receive. Every frame but `Connect` and
//! `Shutdown` names its lane, or a `Step` each of its lanes:
//!
//! - `Connect`: the group's first frame, at open, before any request: rank 0's goes round the
//!   ring and the last rank returns it on the answers link, so every pair of ranks the group
//!   uses has made its first transfer. A transport may connect a pair on its first transfer (NCCL
//!   does), and a connect made once requests are in flight can wait on the ring's queued work,
//!   which waits on the connect.
//! - `Begin`: a new sequence on the lane. Every rank restarts the lane at position 0; the prompt
//!   ids follow, for the last rank's sampler, with the request's sampling in the frame.
//! - `Rows`: a chunk of a prompt's rows at a position; `state` then `input_h` follow, f32 rows of
//!   the hidden width. A chunk that wants output makes the last rank answer.
//! - `Step`: one decode row for each of up to [LANE_BATCH] lanes, with each lane's position and
//!   fed token; the rows follow as a chunk's do, one a lane in the frame's order. A rank that
//!   refuses a lane's row marks its entry with the rank and passes the row on unread, and the
//!   last rank answers every entry, in order, as one message of the entries' answer frames and
//!   one of their logits rows.
//! - `Output`: the last rank's answer to rank 0, the chosen token; the logits row follows (after
//!   the other entries' frames, for a `Step`).
//! - `Fault`: a rank refused a `Rows` frame. It stands in for that frame from there on, so the
//!   ring stays paired and rank 0 still gets one answer for each chunk that wanted one; as an
//!   answer (to a chunk, or to a refused `Step` entry), a row follows it too (whatever the lane's
//!   logits buffer holds), so every answer is frames then rows, read with one copy.
//! - `Shutdown`: rank 0 is closing the group.
//!
//! Answers come back on a link of their own ([RecurrenceLinks::answers]), on a stream of rank 0's
//! own, into page-locked memory, so rank 0 waits for one lane's answer while it queues another
//! lane's rows. (A copy into pageable memory would hold the driver's staging path for the whole
//! wait, and the other lanes' frame copies behind it: the steps would serialise.) Every rank
//! takes frames in the order they reach it and the ring keeps that order, so answers reach rank
//! 0 in the order it sent the chunks and steps that wanted them: each such chunk, and each
//! batch, takes a ticket, and the answers are read in ticket order.
//!
//! Rank 0 writes a whole prompt's frames before its first chunk, so its chunks queue without a
//! host round trip; chunk k+1 runs on rank 0 while chunk k runs on rank 1. It keeps at most two
//! of a prompt's chunks queued ahead, so another lane's decode step waits behind two chunks, not
//! the whole prompt. The group cuts a prompt into [recurrence_chunk]-row chunks, an eighth of one
//! process's or narrower, so a prompt keeps every stage busy rather than filling and draining a
//! four-stage pipe with three chunks. A chunk's rows run the same kernels on every rank as on one
//! process (one row the token kernels, a chunk the batched ones, whose rows do not depend on the
//! chunk's width: see `prompt_chunks`), so the group's logits are the single process's bits,
//! whatever the other lanes are doing.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, HostMapped, RunStream, StreamEvent};
use sparkflash_model::ModelDescriptor;
use sparkflash_model::backend::{
    BackendError, ModelBackend, ModelSession, Sampling, Speculation, StepOutput,
};
use sparkflash_ops::sampler::{Sampler, SamplerMark, argmax};

use crate::config::{DoryConfig, KvCacheMode};
use crate::device::{DeviceShared, DoryStage, LANE_BATCH, PREFILL_BATCH, StageCore, prompt_chunks};

/// The rows of a prompt's chunks on the ring: [RECURRENCE_CHUNK_SHORT] below
/// [RECURRENCE_LONG_PROMPT] tokens, [RECURRENCE_CHUNK] from there.
///
/// One process runs PREFILL_BATCH-row chunks; the group runs narrower ones so the stages
/// overlap. A prompt's time is about the last rank's work on all its rows plus the first
/// chunk's trip through the ranks before it, about 2.4 of the last rank's chunk times, so a
/// narrow chunk wins until its tensor-core tiles leave the GPU's 48 SMs idle. Chunks are whole
/// 128-row tiles (a partial tile pays a whole one's weight reads): at 384 rows q, k, v and g are
/// three waves and o_proj and down_proj one wave on 30 SMs; 512 rows keep o_proj and down_proj
/// in one wave and use it better, which pays once the fill is a small part of the prompt
/// (measured on four GB10s: 384 rows prefill 1,666 tokens 4% faster than 512, 512 rows 6,646 and
/// 13,285 tokens 3% faster than 384). A row's bits do not depend on the width (see
/// `prompt_chunks`).
pub(crate) fn recurrence_chunk(prompt: usize) -> usize {
    if prompt < RECURRENCE_LONG_PROMPT {
        RECURRENCE_CHUNK_SHORT
    } else {
        RECURRENCE_CHUNK
    }
}

/// The chunk rows of a prompt of at least [RECURRENCE_LONG_PROMPT] tokens.
pub(crate) const RECURRENCE_CHUNK: usize = 512;
/// The chunk rows of a shorter prompt, the narrowest [recurrence_chunk] gives.
pub(crate) const RECURRENCE_CHUNK_SHORT: usize = 384;
/// The prompt length from which [recurrence_chunk] takes the wider chunk.
pub(crate) const RECURRENCE_LONG_PROMPT: usize = 4096;
const _: () = assert!(
    RECURRENCE_CHUNK_SHORT.is_multiple_of(128)
        && RECURRENCE_CHUNK.is_multiple_of(128)
        && RECURRENCE_CHUNK_SHORT <= RECURRENCE_CHUNK
        && RECURRENCE_CHUNK <= PREFILL_BATCH
);

/// A prompt's chunks rank 0 keeps queued ahead of the one running, so another lane's step slots
/// in between them.
const PREFILL_AHEAD: usize = 2;

/// How long the driver holds a batch for a lane whose answer it has read and that has not
/// stepped again: a batch's sessions wake together and step again within about a millisecond of
/// each other, while a lane left for the next batch waits the whole of rank 0's next pass. A
/// session that does not step again in that time (its request ended, or its client is slow)
/// holds no one past it.
const RETURN_WINDOW: std::time::Duration = std::time::Duration::from_millis(2);

/// The host-facing name of a recurrence refusal.
const RECURRENCE: &str = "dory_recurrence";

/// Words in a frame: a `Step` frame's header and [LANE_BATCH] entries fit.
pub const FRAME_WORDS: usize = 32;

/// Bytes in a frame.
const FRAME_BYTES: usize = FRAME_WORDS * 4;

/// The first word of every frame: a desynchronised ring reads garbage, never a plausible frame.
const MAGIC: u32 = 0x444f_5259;

/// The word that names a frame's lane.
const LANE_WORD: usize = FRAME_WORDS - 1;

const BEGIN: u32 = 1;
const ROWS: u32 = 2;
const OUTPUT: u32 = 3;
const FAULT: u32 = 4;
const SHUTDOWN: u32 = 5;
const STEP: u32 = 6;
const CONNECT: u32 = 7;
const DRAFT: u32 = 8;

/// A `Rows` or `Fault` frame's chunk wants the last rank's answer.
const FLAG_OUTPUT: u32 = 1;

/// The word of a `Step` frame's first entry, and the words an entry takes.
const STEP_FIRST: usize = 3;
const STEP_ENTRY_WORDS: usize = 3;
const _: () = assert!(STEP_FIRST + LANE_BATCH * STEP_ENTRY_WORDS <= LANE_WORD);

/// An entry's first word: the lane in the low half, then the refusing rank plus one (0 for
/// none) in four bits, the flags, and the epoch in the top byte.
const ENTRY_LANE_BITS: u32 = 16;
const ENTRY_FAULT_MAX: usize = 0xe;
const ENTRY_DRAFT: u32 = 1 << 20;
const ENTRY_REWIND: u32 = 1 << 21;
const ENTRY_EPOCH_SHIFT: u32 = 24;

/// The rows a drafting lane may have started on guesses ahead of its session: about a full
/// ring's worth, since a guess is ready about a third of the way round.
const DRAFT_DEPTH: usize = 4;

/// How long the speculator holds a guess, or a drafting lane's row that missed, for the other
/// drafting lanes' next guesses, so that the drafting lanes go round the ring in one pass: a
/// little over one guess's round trip.
const GATHER_WAIT: std::time::Duration = std::time::Duration::from_millis(40);

/// The sampler marks the last rank keeps a lane, one for each row it has chosen on, newest
/// last: enough to rewind past every row a lane can have in flight.
const MARKS: usize = 2 * DRAFT_DEPTH + LANE_BATCH;

/// One lane's decode row in a `Step` frame, or one guess in a `Draft` frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct StepEntry {
    lane: usize,
    /// The row's position; in a guess, the position of the row the guessed token feeds.
    position: usize,
    /// The token rank 0 fed, for the last rank's sampler history; in a guess, the guess.
    fed: u32,
    /// The rank that refused the lane's row; every later rank passes the refusal on.
    fault: Option<usize>,
    /// The drafting rank guesses the token after this row from its draft exit.
    draft: bool,
    /// Every rank takes the lane back to this row's position before running it: the rows from
    /// there on that rank 0 started on a wrong guess are dropped.
    rewind: bool,
    /// Rank 0's rollback count for the lane when it started the row, echoed in the row's
    /// guess, so a guess from a row a rollback dropped is known stale.
    epoch: u8,
}

impl StepEntry {
    fn encode(&self, words: &mut [u32]) -> Result<(), String> {
        let fault = self.fault.map_or(0, |rank| rank + 1);
        if self.lane >= 1 << ENTRY_LANE_BITS || fault > ENTRY_FAULT_MAX + 1 {
            return Err(format!(
                "lane {} or rank {fault} does not fit its frame bits",
                self.lane
            ));
        }
        let mut first = self.lane as u32 | (fault as u32) << ENTRY_LANE_BITS;
        if self.draft {
            first |= ENTRY_DRAFT;
        }
        if self.rewind {
            first |= ENTRY_REWIND;
        }
        words[0] = first | u32::from(self.epoch) << ENTRY_EPOCH_SHIFT;
        words[1] = word(self.position, "position")?;
        words[2] = self.fed;
        Ok(())
    }

    fn decode(words: &[u32]) -> Self {
        let fault = ((words[0] >> ENTRY_LANE_BITS) & 0xf) as usize;
        Self {
            lane: (words[0] & ((1 << ENTRY_LANE_BITS) - 1)) as usize,
            position: words[1] as usize,
            fed: words[2],
            fault: fault.checked_sub(1),
            draft: words[0] & ENTRY_DRAFT != 0,
            rewind: words[0] & ENTRY_REWIND != 0,
            epoch: (words[0] >> ENTRY_EPOCH_SHIFT) as u8,
        }
    }
}

/// A transport between the ranks of one group.
///
/// Both calls queue behind the stage's earlier work on the link's stream, and the transfers
/// between one pair of ranks match in the order they were queued. The device arm launches on the
/// default stream, so a link that enqueues there needs no other ordering; rank 0's answers link
/// enqueues on the stream it was opened with. `cuda` is the stage's context.
pub trait RecurrenceLink: Send {
    /// Queues `bytes` of device memory at `buffer` for rank `peer`.
    ///
    /// # Errors
    /// The transport's refusal.
    fn send(
        &mut self,
        cuda: &Cuda,
        peer: usize,
        buffer: Deviceptr,
        bytes: usize,
    ) -> Result<(), String>;

    /// Queues the receive of `bytes` from rank `peer` into the device memory at `buffer`.
    ///
    /// # Errors
    /// The transport's refusal.
    fn recv(
        &mut self,
        cuda: &Cuda,
        peer: usize,
        buffer: Deviceptr,
        bytes: usize,
    ) -> Result<(), String>;
}

/// One rank's transports.
///
/// They must be independent channels (a communicator each, say): a receive pending on one never
/// holds up another.
pub struct RecurrenceLinks {
    /// Rank r to rank r + 1: the frames, the prompt ids and the latent rows, on the default
    /// stream.
    pub ring: Box<dyn RecurrenceLink>,
    /// The last rank to rank 0: the answers. Rank 0's end queues on [LinkStreams::answers];
    /// every other rank's on the default stream.
    pub answers: Box<dyn RecurrenceLink>,
    /// Rank 1 to rank 0: the draft exit's guesses, when the group drafts (and only then; every
    /// rank of a drafting group opens it). Rank 0's end queues on [LinkStreams::drafts], rank
    /// 1's on the default stream.
    pub drafts: Option<Box<dyn RecurrenceLink>>,
}

/// The streams rank 0's links queue on, handed to its link factory; a worker is given the
/// default stream, 0, for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkStreams {
    /// Where the answers link queues.
    pub answers: u64,
    /// Where the drafts link queues.
    pub drafts: u64,
}

/// One message header.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::large_enum_variant)] // a frame lives on the stack from one read to one write; boxing a Step's entries would allocate once a batch on every rank.
enum Frame {
    Begin {
        lane: usize,
        prompt: usize,
        sampling: Sampling,
    },
    Rows {
        lane: usize,
        rows: usize,
        position: usize,
        output: bool,
    },
    Step {
        count: usize,
        entries: [StepEntry; LANE_BATCH],
    },
    /// The drafting rank's guesses for some of a `Step`'s rows, in entry order.
    Draft {
        count: usize,
        entries: [StepEntry; LANE_BATCH],
    },
    Output {
        lane: usize,
        token: u32,
    },
    Fault {
        lane: usize,
        rank: usize,
        output: bool,
    },
    Connect,
    Shutdown,
}

fn word(value: usize, what: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{what} {value} does not fit a frame word"))
}

impl Frame {
    fn encode(&self) -> Result<[u32; FRAME_WORDS], String> {
        let mut words = [0u32; FRAME_WORDS];
        words[0] = MAGIC;
        match *self {
            Self::Begin {
                lane,
                prompt,
                sampling,
            } => {
                words[1] = BEGIN;
                words[2] = word(prompt, "a prompt of")?;
                words[3] = sampling.temperature.to_bits();
                words[4] = sampling.top_p.to_bits();
                words[5] = sampling.top_k as u32;
                words[6] = sampling.repetition_penalty.to_bits();
                words[7] = sampling.presence_penalty.to_bits();
                words[8] = sampling.frequency_penalty.to_bits();
                words[9] = sampling.seed as u32;
                words[10] = (sampling.seed >> 32) as u32;
                words[11] = word(sampling.prompt_len, "a sampling prompt length of")?;
                words[LANE_WORD] = word(lane, "lane")?;
            }
            Self::Rows {
                lane,
                rows,
                position,
                output,
            } => {
                words[1] = ROWS;
                words[2] = word(rows, "a chunk of")?;
                words[3] = word(position, "position")?;
                words[4] = if output { FLAG_OUTPUT } else { 0 };
                words[LANE_WORD] = word(lane, "lane")?;
            }
            Self::Step { count, entries } | Self::Draft { count, entries } => {
                if count == 0 || count > LANE_BATCH {
                    return Err(format!(
                        "a step of {count} lanes is outside 1..={LANE_BATCH}"
                    ));
                }
                words[1] = if matches!(self, Self::Step { .. }) {
                    STEP
                } else {
                    DRAFT
                };
                words[2] = count as u32;
                for (index, entry) in entries[..count].iter().enumerate() {
                    let at = STEP_FIRST + index * STEP_ENTRY_WORDS;
                    entry.encode(&mut words[at..at + STEP_ENTRY_WORDS])?;
                }
            }
            Self::Output { lane, token } => {
                words[1] = OUTPUT;
                words[2] = token;
                words[LANE_WORD] = word(lane, "lane")?;
            }
            Self::Fault { lane, rank, output } => {
                words[1] = FAULT;
                words[2] = word(rank, "rank")?;
                words[3] = if output { FLAG_OUTPUT } else { 0 };
                words[LANE_WORD] = word(lane, "lane")?;
            }
            Self::Connect => words[1] = CONNECT,
            Self::Shutdown => words[1] = SHUTDOWN,
        }
        Ok(words)
    }

    fn decode(words: &[u32; FRAME_WORDS]) -> Result<Self, String> {
        if words[0] != MAGIC {
            return Err(format!(
                "a recurrence frame opened with {:#010x}, not the ring's {MAGIC:#010x}: the ring \
                 is out of step",
                words[0]
            ));
        }
        let lane = words[LANE_WORD] as usize;
        Ok(match words[1] {
            BEGIN => Self::Begin {
                lane,
                prompt: words[2] as usize,
                sampling: Sampling {
                    temperature: f32::from_bits(words[3]),
                    top_p: f32::from_bits(words[4]),
                    top_k: words[5] as i32,
                    repetition_penalty: f32::from_bits(words[6]),
                    presence_penalty: f32::from_bits(words[7]),
                    frequency_penalty: f32::from_bits(words[8]),
                    seed: u64::from(words[9]) | (u64::from(words[10]) << 32),
                    prompt_len: words[11] as usize,
                },
            },
            ROWS => Self::Rows {
                lane,
                rows: words[2] as usize,
                position: words[3] as usize,
                output: words[4] & FLAG_OUTPUT != 0,
            },
            STEP | DRAFT => {
                let count = words[2] as usize;
                if count == 0 || count > LANE_BATCH {
                    return Err(format!(
                        "a Step frame of {count} lanes is outside 1..={LANE_BATCH}"
                    ));
                }
                let mut entries = [StepEntry::default(); LANE_BATCH];
                for (index, entry) in entries[..count].iter_mut().enumerate() {
                    let at = STEP_FIRST + index * STEP_ENTRY_WORDS;
                    *entry = StepEntry::decode(&words[at..at + STEP_ENTRY_WORDS]);
                }
                if words[1] == STEP {
                    Self::Step { count, entries }
                } else {
                    Self::Draft { count, entries }
                }
            }
            OUTPUT => Self::Output {
                lane,
                token: words[2],
            },
            FAULT => Self::Fault {
                lane,
                rank: words[2] as usize,
                output: words[3] & FLAG_OUTPUT != 0,
            },
            CONNECT => Self::Connect,
            SHUTDOWN => Self::Shutdown,
            other => return Err(format!("a recurrence frame of unknown kind {other}")),
        })
    }

    fn from_bytes(bytes: &[u8; FRAME_BYTES]) -> Result<Self, String> {
        let mut words = [0u32; FRAME_WORDS];
        for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*chunk);
        }
        Self::decode(&words)
    }
}

fn encode_frames(frames: &[Frame]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(frames.len() * FRAME_BYTES);
    for frame in frames {
        for word in frame.encode()? {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Writes `frames` at `buffer`, after the default stream's queued work.
fn write_frames(cuda: &Cuda, buffer: Deviceptr, frames: &[Frame]) -> Result<(), String> {
    let bytes = encode_frames(frames)?;
    cuda.make_current()?;
    cuda.copy_htod(&buffer, &bytes)
}

/// Reads the frame at `buffer`, after every transfer queued before it has landed.
fn read_frame(cuda: &Cuda, buffer: Deviceptr) -> Result<Frame, String> {
    let mut bytes = [0u8; FRAME_BYTES];
    cuda.make_current()?;
    cuda.copy_dtoh(&mut bytes, &buffer)?;
    Frame::from_bytes(&bytes)
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Device and pinned host allocations that free themselves.
struct Buffers {
    shared: Arc<DeviceShared>,
    device: Vec<Deviceptr>,
    host: Vec<HostMapped>,
}

impl Buffers {
    fn new(shared: &Arc<DeviceShared>) -> Self {
        Self {
            shared: Arc::clone(shared),
            device: Vec::new(),
            host: Vec::new(),
        }
    }

    fn device(&mut self, bytes: usize) -> Result<Deviceptr, String> {
        let mut total = 0usize;
        let pointer = self.shared.cuda().alloc(bytes, &mut total)?;
        self.device.push(pointer);
        Ok(pointer)
    }

    fn host(&mut self, bytes: usize) -> Result<HostMapped, String> {
        let mut total = 0usize;
        let mapped = self.shared.cuda().alloc_host_mapped(bytes, &mut total)?;
        self.host.push(mapped);
        Ok(mapped)
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        let cuda = self.shared.cuda();
        // The last owner may drop it on a thread that never made the context current.
        let _ = cuda.make_current();
        let references: Vec<&Deviceptr> = self.device.iter().collect();
        cuda.free_buffers(&references);
        for mapped in &self.host {
            cuda.free_host_mapped(mapped);
        }
    }
}

/// A worker's frame buffers: the frame it receives, the ones it composes (the last rank's
/// answers, one a lane of a step, or a fault; the drafting rank's guesses), and a prompt's ids
/// as a Begin frame carries them.
struct LinkBuffers {
    frame: Deviceptr,
    /// [LANE_BATCH] frame slots.
    replies: Deviceptr,
    /// The drafting rank's `Draft` frame.
    guesses: Deviceptr,
    ids: Deviceptr,
    _owned: Buffers,
}

impl LinkBuffers {
    fn allocate(shared: &Arc<DeviceShared>) -> Result<Self, String> {
        shared.cuda().make_current()?;
        let mut owned = Buffers::new(shared);
        Ok(Self {
            frame: owned.device(FRAME_BYTES)?,
            replies: owned.device(FRAME_BYTES * LANE_BATCH)?,
            guesses: owned.device(FRAME_BYTES)?,
            ids: owned.device(shared.context() * 4)?,
            _owned: owned,
        })
    }

    fn reply(&self, index: usize) -> Deviceptr {
        self.replies.offset(index * FRAME_BYTES)
    }
}

/// Rank 0's frames and prompt ids for one lane, staged through pinned host memory: writing them
/// queues a copy rather than waiting for the work other lanes have queued ahead of it.
///
/// A staging buffer is rewritten only once the copy that last read it has run (an event marks
/// each copy). That is at once for a lane whose last step has been answered; a drafting lane can
/// start its next row while the last one's copy is still queued behind other work.
struct LaneBuffers {
    /// Frame slots: a whole prompt's frames, or one decode step's.
    frames: Deviceptr,
    staged_frames: HostMapped,
    frames_copied: Fence,
    slots: usize,
    ids: Deviceptr,
    staged_ids: HostMapped,
    ids_copied: Fence,
    _owned: Buffers,
}

impl LaneBuffers {
    fn allocate(shared: &Arc<DeviceShared>, slots: usize) -> Result<Self, String> {
        shared.cuda().make_current()?;
        let mut owned = Buffers::new(shared);
        let ids = shared.context() * 4;
        Ok(Self {
            frames: owned.device(slots * FRAME_BYTES)?,
            staged_frames: owned.host(slots * FRAME_BYTES)?,
            frames_copied: Fence::new(shared)?,
            slots,
            ids: owned.device(ids)?,
            staged_ids: owned.host(ids)?,
            ids_copied: Fence::new(shared)?,
            _owned: owned,
        })
    }

    fn slot(&self, index: usize) -> Deviceptr {
        self.frames.offset(index * FRAME_BYTES)
    }

    /// Queues `frames` into the slots from 0, on the default stream.
    fn stage_frames(&self, cuda: &Cuda, frames: &[Frame]) -> Result<(), String> {
        if frames.len() > self.slots {
            return Err(format!(
                "{} frames do not fit the lane's {} slots",
                frames.len(),
                self.slots
            ));
        }
        let bytes = encode_frames(frames)?;
        cuda.make_current()?;
        cuda.host_wait(&self.frames_copied.event)?;
        self.staged_frames.write_at(0, &bytes)?;
        cuda.copy_htod_async(&self.frames, &self.staged_frames, 0, bytes.len())?;
        cuda.run_mark(&self.frames_copied.event)
    }

    /// Queues a prompt's ids, on the default stream.
    fn stage_ids(&self, cuda: &Cuda, tokens: &[u32]) -> Result<usize, String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        cuda.make_current()?;
        cuda.host_wait(&self.ids_copied.event)?;
        self.staged_ids.write_at(0, &bytes)?;
        cuda.copy_htod_async(&self.ids, &self.staged_ids, 0, bytes.len())?;
        cuda.run_mark(&self.ids_copied.event)?;
        Ok(bytes.len())
    }
}

/// The rank that runs the draft exit: the first after rank 0, so its guess comes from the latent
/// after rank 0's loops, as early as a guess can come.
const DRAFTER: usize = 1;

/// The held lanes up to which a group drafts unless asked otherwise: two passes' worth. The
/// drafting lanes go round the ring in passes of [LANE_BATCH], and a rank's pass costs about the
/// same at one lane as at eight, so up to two passes drafting outpaces the plain group (on four
/// GB10s, 26.7 against 12.7 tok/s at one stream, 186 against 94 at eight, 227 against 174 at
/// 16); at 24 streams the two are even, and at 32 the plain group's four full passes win (284
/// against 202), since the drafting rank's exit and the rows a wrong guess drops are then work
/// the ring has no idle pass for.
const DEFAULT_DRAFT_LANES: usize = 2 * LANE_BATCH;

/// The drafting a `world`-rank group runs: `asked`, or by default [DEFAULT_DRAFT_LANES] where the
/// group can draft and 0 where it cannot.
///
/// # Errors
/// Drafting asked of a group too small for it: the drafting rank must be neither the first nor
/// the last.
pub fn recurrence_draft_lanes(world: usize, asked: Option<usize>) -> Result<usize, String> {
    let can = world > DRAFTER + 1;
    match asked {
        Some(lanes) if lanes > 0 && !can => Err(format!(
            "drafting at {lanes} lane(s) needs at least {} ranks: rank {DRAFTER} drafts, and in a \
             group of {world} it is the last; pass 0",
            DRAFTER + 2
        )),
        Some(lanes) => Ok(lanes),
        None => Ok(if can { DEFAULT_DRAFT_LANES } else { 0 }),
    }
}

/// Opens the stage of `rank` and checks the group shape every rank must agree on. In a
/// drafting group, rank [DRAFTER] also holds the draft exit.
fn open_stage(
    directory: &std::path::Path,
    context: usize,
    rank: usize,
    world: usize,
    lanes: usize,
    drafting: bool,
) -> Result<Arc<DeviceShared>, String> {
    if world < 2 {
        return Err(format!(
            "a recurrence group needs at least two ranks, not {world}; one process serves the \
             whole model"
        ));
    }
    if lanes == 0 {
        return Err("a recurrence group needs at least one sequence lane".to_owned());
    }
    if drafting && world <= DRAFTER + 1 {
        return Err(format!(
            "a drafting recurrence group needs at least {} ranks, not {world}: the draft exit \
             runs on rank {DRAFTER}, which must not be the last rank",
            DRAFTER + 2
        ));
    }
    let shared = DeviceShared::open(directory, context, |config| {
        let stage = DoryStage::of_rank(config, rank, world)?;
        if drafting && rank == DRAFTER {
            stage.drafting()
        } else {
            Ok(stage)
        }
    })?;
    Ok(Arc::new(shared))
}

/// Rank 0's half of the `Connect` exchange: the frame goes out round the ring through `lane`'s
/// first slot and comes back from the last rank on the answers link into `answer`, and, in a
/// drafting group, from the drafting rank on the drafts link into its buffer.
fn connect_head(
    cuda: &Cuda,
    ring: &mut dyn RecurrenceLink,
    answers: &mut dyn RecurrenceLink,
    drafts: Option<&mut DraftLink>,
    lane: &LaneBuffers,
    answer: Deviceptr,
    world: usize,
) -> Result<(), String> {
    lane.stage_frames(cuda, &[Frame::Connect])?;
    ring.send(cuda, 1, lane.slot(0), FRAME_BYTES)?;
    answers.recv(cuda, world - 1, answer, FRAME_BYTES)?;
    let mut returned = vec![("the last recurrence rank", answer)];
    if let Some(drafts) = drafts {
        drafts.link.recv(cuda, DRAFTER, drafts.guess, FRAME_BYTES)?;
        returned.push(("the drafting recurrence rank", drafts.guess));
    }
    // The answers and drafts links may queue on streams of their own: wait for the whole
    // context.
    cuda.synchronize()?;
    for (from, buffer) in returned {
        match read_frame(cuda, buffer)? {
            Frame::Connect => {}
            other => {
                return Err(format!(
                    "{from} returned {other:?} for rank 0's Connect frame"
                ));
            }
        }
    }
    Ok(())
}

fn link_error(error: String) -> BackendError {
    BackendError::new(RECURRENCE, error)
}

fn poisoned<T>(_: T) -> BackendError {
    BackendError::new(RECURRENCE, "a recurrence group lock is poisoned")
}

/// The backend kind of a sequence a rank refused: the ring is still paired, so the group is not
/// broken, only this sequence.
const FAULT_KIND: &str = "dory_recurrence_fault";

/// Rank 0's half of the group: the model as a [ModelBackend], driving the ring for each step.
///
/// A session takes a free lane at its prefill and keeps it until it is dropped; a prefill that
/// finds every lane held is refused, so the caller serves at most `lanes` requests at once.
/// Sessions on different lanes may step from different threads at the same time.
pub struct DoryRecurrenceBackend {
    descriptor: ModelDescriptor,
    group: Arc<Group>,
    /// Rank 0's answer reader and, in a drafting group, its guess listener and speculator.
    threads: Vec<std::thread::JoinHandle<()>>,
}

/// Rank 0's state, shared by its sessions and its threads.
struct Group {
    shared: Arc<DeviceShared>,
    world: usize,
    /// A lane drafts while at most this many lanes are held; 0 in a group that never drafts.
    draft_lanes: usize,
    /// The ring and the lanes: held while a step's work is queued, never while it is awaited.
    head: Mutex<Head>,
    /// The answer messages on their way and the entries read: the reader thread fills it, the
    /// sessions take from it.
    board: Mutex<Board>,
    /// Signalled when the board changes.
    turn: Condvar,
    /// The guesses the listener has read and the rows waiting to join them, which the
    /// speculator has not yet started.
    guesses: Mutex<Guesses>,
    /// Signalled when a guess comes, a row comes to join the next guesses, or the drafts link
    /// closes.
    guessed: Condvar,
    /// Signalled when the speculator has started the joining rows it took.
    joined: Condvar,
    /// Decode steps waiting for a batch, and the batches' tickets.
    queue: Mutex<Queue>,
    /// Signalled when the queue changes hands or a batch has taken steps.
    queued: Condvar,
    /// Marked after a batch's rows on rank 0's stream; only the driving session touches it.
    batch_done: Mutex<Fence>,
    /// Why the ring can no longer be trusted, once a transfer failed mid-step.
    broken: Mutex<Option<String>>,
}

/// The drafting rank's guesses on their way from the listener to the speculator.
///
/// The listener only reads: it posts the next receive as soon as a frame is in, whatever the
/// head is doing. Were it to start the guessed rows itself, it would wait on the head's lock
/// while a session holding it waits on rank 0's stream, which can wait on the drafting rank's
/// next receive, which waits behind its send of the next guess to the listener.
///
/// The drafting lanes go round the ring in one pass, which costs each rank about what one row
/// does; lanes in passes of their own would take rank 0 a pass each. So a session whose drafting
/// lane missed does not start its corrected row alone while another lane's guess is on its way:
/// the row joins the pass that guess starts. And the speculator holds the guesses that have come
/// while another drafting lane's guess is still on its way, up to [GATHER_WAIT], so that lanes
/// started apart (a sequence's first row, a row that waited too long to join) come together.
#[derive(Default)]
struct Guesses {
    entries: VecDeque<StepEntry>,
    /// When the oldest of `entries` came.
    since: Option<std::time::Instant>,
    /// The rows waiting to join the next guesses, oldest first.
    joining: Vec<Joining>,
    /// What the speculator made of each joining row it took, by session: where its answer
    /// comes, or the refusal.
    joined: Vec<(u64, Result<Ticket, BackendError>)>,
    /// The drafting rank passed the group's `Shutdown` on: no more guesses come.
    closed: bool,
}

/// A session's decode row waiting to join the next guesses.
#[derive(Debug, Clone, Copy)]
struct Joining {
    session: u64,
    /// The row: its lane, position, token and the lane's epoch when it was queued.
    row: StepEntry,
    since: std::time::Instant,
}

/// The answer messages rank 0 expects, and the entries read for sessions that have not taken
/// them.
#[derive(Default)]
struct Board {
    /// The messages sent for and not yet read, in ticket order.
    sent: VecDeque<Sent>,
    /// The entries read for sessions that have not yet taken them.
    ready: Vec<Ready>,
    /// The entries no session will take, dropped as they are read: the rows a rollback dropped.
    discard: Vec<(u64, usize)>,
    /// The group is closing: the reader stops once every message sent for has been read.
    closed: bool,
}

/// One answer message rank 0 expects.
#[derive(Debug, Clone, Copy)]
struct Sent {
    number: u64,
    count: usize,
    /// A batch of the decode queue, whose lanes come back to it.
    queued: bool,
}

/// The decode steps rank 0 has not yet put in a batch.
#[derive(Default)]
struct Queue {
    /// Oldest first.
    waiting: Vec<Waiting>,
    /// A session is building and queueing a batch.
    driving: bool,
    /// What a batch made of each step it took, by session: where its answer comes, or the
    /// refusal.
    taken: Vec<(u64, Result<Ticket, BackendError>)>,
    /// The lanes whose answer has been read and that have not stepped again, with when.
    returning: Vec<(usize, std::time::Instant)>,
    /// The tickets of the decode batches sent whose answers have not been read.
    in_flight: Vec<u64>,
}

/// Where a chunk's or a step's answer comes back: the answer message's ticket and the entry in
/// it.
#[derive(Debug, Clone, Copy)]
struct Ticket {
    number: u64,
    entry: usize,
}

/// One entry of an answer message, read for its session.
struct Ready {
    number: u64,
    entry: usize,
    frame: Frame,
    row: Vec<u8>,
}

impl Ready {
    /// The answer, which must be `lane`'s; a token that ends the turn under `config` is none.
    fn output(self, lane: usize, config: &DoryConfig) -> Result<StepOutput, BackendError> {
        match self.frame {
            Frame::Output {
                lane: answered,
                token,
            } if answered == lane => Ok(StepOutput {
                next_token: config.next_token(token),
                logits: to_f32(&self.row),
                draft: Vec::new(),
            }),
            Frame::Fault {
                lane: answered,
                rank,
                ..
            } if answered == lane => Err(BackendError::new(
                FAULT_KIND,
                format!(
                    "recurrence rank {rank} refused this sequence (its log says why); the next \
                     request starts a new one"
                ),
            )),
            other => Err(link_error(format!(
                "the last recurrence rank answered lane {lane}'s step with {other:?}"
            ))),
        }
    }
}

/// One session's decode step.
#[derive(Debug, Clone, Copy)]
struct Waiting {
    session: u64,
    lane: usize,
    token: u32,
}

/// The ring and the lanes. `ring` is declared first so it closes before the lanes' caches go.
struct Head {
    ring: Box<dyn RecurrenceLink>,
    lanes: Vec<HeadLane>,
    /// The ticket the next chunk that wants an answer takes.
    tickets: u64,
    /// The id the next session takes, which marks the lane it holds.
    sessions: u64,
    /// The group has been shut down: nothing more goes on the ring.
    closed: bool,
}

struct HeadLane {
    core: StageCore,
    buffers: LaneBuffers,
    /// The session that holds the lane.
    owner: Option<u64>,
    /// The rows the lane started on the drafting rank's guesses.
    draft: LaneDraft,
}

/// A lane's rows started on guesses, ahead of its session.
///
/// The drafting rank guesses the token after each drafting row from the latent after rank 0's
/// loops, and rank 0 starts the row that token feeds as soon as the guess comes, while the
/// guessed-from row is still on its way round the ring. When the last rank's choice for that
/// row comes back it is the token the guessed row should have been fed: the same token, and the
/// guessed row is the session's next; another, and the guessed row and every row started after
/// it are dropped. Every emitted token's row has still run every loop on the token the model
/// chose, so the group's tokens and logits are the plain group's.
#[derive(Debug, Default)]
struct LaneDraft {
    /// Moves on at every rollback, every new sequence and every step through the queue: a
    /// guess from a row started before is stale.
    epoch: u8,
    /// The rows started on guesses past the session's next position, oldest first: each the
    /// guess after the one before, the last at the lane's position less one.
    path: VecDeque<Started>,
    /// The lane's next row tells every rank to drop the rows from its position on.
    rewind: bool,
    /// The guess the lane's newest drafting row will bring, until it comes: a row that misses
    /// on another lane meanwhile waits to join it, and guesses from another pass wait for it.
    awaiting: Option<Awaiting>,
}

/// The guess a lane's drafting row will bring.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Awaiting {
    /// The lane's epoch when the row started, which the guess echoes.
    epoch: u8,
    /// The position of the row the guess feeds.
    position: usize,
    /// The ticket of the row's pass: the lanes of one pass come back in one Draft frame.
    pass: u64,
}

/// A row started on a guess.
#[derive(Debug, Clone, Copy)]
struct Started {
    position: usize,
    /// The guessed token the row was fed.
    fed: u32,
    ticket: Ticket,
}

/// Rank 0's own end of a link, on a stream of its own, destroyed once it has drained.
struct LinkStream {
    shared: Arc<DeviceShared>,
    stream: RunStream,
}

impl LinkStream {
    fn new(shared: &Arc<DeviceShared>) -> Result<Self, String> {
        Ok(Self {
            shared: Arc::clone(shared),
            stream: shared.cuda().run_stream()?,
        })
    }
}

impl Drop for LinkStream {
    fn drop(&mut self) {
        let cuda = self.shared.cuda();
        let _ = cuda.make_current();
        let _ = cuda.synchronize_stream(&self.stream);
        cuda.destroy_run_stream(&self.stream);
    }
}

/// The answers link and where an answer message lands, owned by the reader thread. `link` is
/// declared first so it closes before the buffers go.
struct Answers {
    link: Box<dyn RecurrenceLink>,
    stream: LinkStream,
    /// An answer message as it arrives: [LANE_BATCH] frame slots, then as many rows.
    answer: Deviceptr,
    /// The message copied out, page-locked.
    staged: HostMapped,
    _owned: Buffers,
}

/// The drafts link and where a `Draft` frame lands, owned by the listener thread.
struct DraftLink {
    link: Box<dyn RecurrenceLink>,
    stream: LinkStream,
    guess: Deviceptr,
    staged: HostMapped,
    _owned: Buffers,
}

impl DoryRecurrenceBackend {
    /// Opens rank 0 of a `world`-rank group of `lanes` sequence lanes over `directory`, then its
    /// links on the stage's context; the factory is given the streams rank 0's ends of the
    /// answers and drafts links must queue on.
    ///
    /// The group drafts when `draft_lanes` is not 0: a lane's rows are then started on the
    /// drafting rank's guesses whenever at most `draft_lanes` lanes are held (see [LaneDraft]).
    /// Every rank must be opened with drafting on or off alike.
    ///
    /// # Errors
    /// No lane, a drafting group of fewer than three ranks, the stage's refusal (see
    /// [DoryStage::of_rank] and [crate::DoryDeviceBackend::open]), an allocation, or the links'.
    pub fn open(
        directory: &std::path::Path,
        context: usize,
        world: usize,
        lanes: usize,
        draft_lanes: usize,
        links: impl FnOnce(&Cuda, LinkStreams) -> Result<RecurrenceLinks, String>,
    ) -> Result<Self, String> {
        Self::open_with_descriptor(
            crate::descriptor(),
            directory,
            context,
            world,
            lanes,
            draft_lanes,
            links,
        )
    }

    /// Opens rank 0 under the descriptor the serving path resolved.
    ///
    /// # Errors
    /// As [Self::open].
    pub fn open_with_descriptor(
        mut descriptor: ModelDescriptor,
        directory: &std::path::Path,
        context: usize,
        world: usize,
        lanes: usize,
        draft_lanes: usize,
        links: impl FnOnce(&Cuda, LinkStreams) -> Result<RecurrenceLinks, String>,
    ) -> Result<Self, String> {
        let drafting = draft_lanes > 0;
        let shared = open_stage(directory, context, 0, world, lanes, drafting)?;
        // The narrowest chunk cuts any prompt of the context into the most chunks.
        let slots = 1 + prompt_chunks(KvCacheMode::PerLoop, context, RECURRENCE_CHUNK_SHORT).len();
        let mut entries = Vec::with_capacity(lanes);
        for core in StageCore::lanes(&shared, lanes)? {
            entries.push(HeadLane {
                core,
                buffers: LaneBuffers::allocate(&shared, slots)?,
                owner: None,
                draft: LaneDraft::default(),
            });
        }
        let mut owned = Buffers::new(&shared);
        let answer_bytes = LANE_BATCH * (FRAME_BYTES + shared.config().vocab_size * 4);
        let answer = owned.device(answer_bytes)?;
        let staged = owned.host(answer_bytes)?;
        let answer_stream = LinkStream::new(&shared)?;
        let mut draft_stream = drafting.then(|| LinkStream::new(&shared)).transpose()?;
        let RecurrenceLinks {
            mut ring,
            mut answers,
            drafts,
        } = links(
            shared.cuda(),
            LinkStreams {
                answers: answer_stream.stream.raw(),
                drafts: draft_stream
                    .as_ref()
                    .map_or(0, |stream| stream.stream.raw()),
            },
        )?;
        let mut drafts = match (drafts, draft_stream.take()) {
            (Some(link), Some(stream)) => {
                let mut owned = Buffers::new(&shared);
                Some(DraftLink {
                    link,
                    stream,
                    guess: owned.device(FRAME_BYTES)?,
                    staged: owned.host(FRAME_BYTES)?,
                    _owned: owned,
                })
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err(
                    "the link factory opened a drafts link for a group that does not draft"
                        .to_owned(),
                );
            }
            (None, Some(_)) => {
                return Err(
                    "a drafting recurrence group needs a drafts link from rank 1, and the link \
                     factory opened none"
                        .to_owned(),
                );
            }
        };
        connect_head(
            shared.cuda(),
            &mut *ring,
            &mut *answers,
            drafts.as_mut(),
            &entries[0].buffers,
            answer,
            world,
        )?;
        descriptor.context_tokens = context;
        let group = Arc::new(Group {
            world,
            draft_lanes,
            head: Mutex::new(Head {
                ring,
                lanes: entries,
                tickets: 0,
                sessions: 0,
                closed: false,
            }),
            board: Mutex::new(Board::default()),
            turn: Condvar::new(),
            guesses: Mutex::new(Guesses::default()),
            guessed: Condvar::new(),
            joined: Condvar::new(),
            queue: Mutex::new(Queue::default()),
            queued: Condvar::new(),
            batch_done: Mutex::new(Fence::new(&shared)?),
            broken: Mutex::new(None),
            shared,
        });
        let answers = Answers {
            link: answers,
            stream: answer_stream,
            answer,
            staged,
            _owned: owned,
        };
        let mut threads = Vec::with_capacity(2);
        let reader = Arc::clone(&group);
        threads.push(
            std::thread::Builder::new()
                .name("dory-answers".to_owned())
                .spawn(move || read_answers(&reader, answers))
                .map_err(|error| format!("the answer reader thread: {error}"))?,
        );
        if let Some(drafts) = drafts {
            let listener = Arc::clone(&group);
            threads.push(
                std::thread::Builder::new()
                    .name("dory-drafts".to_owned())
                    .spawn(move || listen(&listener, drafts))
                    .map_err(|error| format!("the guess listener thread: {error}"))?,
            );
            let speculator = Arc::clone(&group);
            threads.push(
                std::thread::Builder::new()
                    .name("dory-speculate".to_owned())
                    .spawn(move || speculate_guesses(&speculator))
                    .map_err(|error| format!("the speculator thread: {error}"))?,
            );
        }
        Ok(Self {
            descriptor,
            group,
            threads,
        })
    }

    /// The loops, prefix and suffix this rank runs.
    #[must_use]
    pub fn stage(&self) -> DoryStage {
        self.group.shared.stage().clone()
    }

    /// The device bytes this rank's weights claimed.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.group.shared.weight_bytes()
    }

    /// The sequences the group holds at once.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.group.head.lock().map_or(0, |head| head.lanes.len())
    }
}

impl Drop for DoryRecurrenceBackend {
    fn drop(&mut self) {
        // Best effort: a group whose ring already broke cannot be told anything, and its threads
        // may wait on transfers that never come, so they are left to the process's exit.
        let shut = self.group.broken().is_none() && self.group.shutdown().is_ok();
        self.group.close_board();
        if shut && self.group.broken().is_none() {
            for thread in self.threads.drain(..) {
                let _ = thread.join();
            }
        }
    }
}

impl ModelBackend for DoryRecurrenceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        let id = {
            let mut head = self.group.lock_head()?;
            head.sessions += 1;
            head.sessions
        };
        let cuda = self.group.shared.cuda();
        // The session's thread may never have touched the stage's context.
        cuda.make_current().map_err(link_error)?;
        let fence = || Fence::new(&self.group.shared).map_err(link_error);
        Ok(Box::new(DoryRecurrenceSession {
            group: Arc::clone(&self.group),
            id,
            lane: None,
            sampling: Sampling::default(),
            fences: [fence()?, fence()?],
            position: 0,
            speculation: Speculation::default(),
        }))
    }
}

impl Group {
    fn broken(&self) -> Option<String> {
        self.broken.lock().map_or_else(
            |_| Some("a recurrence group lock is poisoned".to_owned()),
            |b| b.clone(),
        )
    }

    /// The ring and lanes, unless the ring is broken.
    fn lock_head(&self) -> Result<MutexGuard<'_, Head>, BackendError> {
        if let Some(reason) = self.broken() {
            return Err(BackendError::new(RECURRENCE, reason));
        }
        self.head.lock().map_err(poisoned)
    }

    /// Runs `step`, which tells the ring something; a failure there leaves the other ranks
    /// waiting on transfers that never come, so the group is marked broken.
    fn guarded<T>(
        &self,
        step: impl FnOnce() -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let result = step();
        if let Err(error) = &result {
            self.mark_broken(error);
        }
        result
    }

    /// Never called with the board's lock held.
    fn mark_broken(&self, error: &BackendError) {
        if error.code() == FAULT_KIND {
            return;
        }
        if let Ok(mut broken) = self.broken.lock()
            && broken.is_none()
        {
            *broken = Some(format!(
                "the recurrence group stopped mid-step ({error}); restart every rank"
            ));
        }
        // Sessions waiting for an answer or a batch, and the reader waiting for a message,
        // learn the group is gone. Taking the board's lock first means no waiter is between its
        // check and its wait.
        drop(self.board.lock());
        self.turn.notify_all();
        self.queued.notify_all();
        drop(self.guesses.lock());
        self.guessed.notify_all();
        self.joined.notify_all();
    }

    fn validate(&self, tokens: &[u32]) -> Result<(), BackendError> {
        let config = self.shared.config();
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= config.vocab_size)
        {
            return Err(BackendError::new(
                RECURRENCE,
                format!(
                    "token {token} is outside the checkpoint vocabulary {}",
                    config.vocab_size
                ),
            ));
        }
        Ok(())
    }

    /// Expects answer message `number` of `count` entries, after every message expected
    /// before it. Called under the head's lock as the ticket is taken, so the reader reads the
    /// messages in the order the ring carries them.
    fn post(&self, number: u64, count: usize, queued: bool) -> Result<(), BackendError> {
        self.board.lock().map_err(poisoned)?.sent.push_back(Sent {
            number,
            count,
            queued,
        });
        self.turn.notify_all();
        Ok(())
    }

    /// Waits for the answer at `ticket`, to the chunk or step of `lane` that wanted one.
    fn answer(&self, ticket: Ticket, lane: usize) -> Result<StepOutput, BackendError> {
        let ready =
            {
                let mut board = self.board.lock().map_err(poisoned)?;
                loop {
                    if let Some(index) = board.ready.iter().position(|ready| {
                        ready.number == ticket.number && ready.entry == ticket.entry
                    }) {
                        break board.ready.swap_remove(index);
                    }
                    if let Some(reason) = self.broken() {
                        return Err(BackendError::new(RECURRENCE, reason));
                    }
                    board = self.turn.wait(board).map_err(poisoned)?;
                }
            };
        let result = ready.output(lane, self.shared.config());
        if let Err(error) = &result {
            self.mark_broken(error);
        }
        result
    }

    /// Drops the answers at `tickets`, read already or still to come: no session takes them.
    fn discard(&self, tickets: impl IntoIterator<Item = Ticket>) -> Result<(), BackendError> {
        let mut board = self.board.lock().map_err(poisoned)?;
        for ticket in tickets {
            match board
                .ready
                .iter()
                .position(|ready| ready.number == ticket.number && ready.entry == ticket.entry)
            {
                Some(index) => drop(board.ready.swap_remove(index)),
                None => board.discard.push((ticket.number, ticket.entry)),
            }
        }
        Ok(())
    }

    /// Lets the reader stop once the messages already expected have been read.
    fn close_board(&self) {
        if let Ok(mut board) = self.board.lock() {
            board.closed = true;
        }
        self.turn.notify_all();
    }

    /// Notes the answer message `number`, just read: its batch is no longer in flight, and its
    /// lanes are on their way back to the queue.
    fn returned(&self, number: u64, entries: &[Ready]) -> Result<(), BackendError> {
        let now = std::time::Instant::now();
        let mut queue = self.queue.lock().map_err(poisoned)?;
        queue.in_flight.retain(|sent| *sent != number);
        for ready in entries {
            if let Frame::Output { lane, .. } | Frame::Fault { lane, .. } = ready.frame {
                queue.returning.retain(|(returned, _)| *returned != lane);
                queue.returning.push((lane, now));
            }
        }
        self.queued.notify_all();
        Ok(())
    }

    /// Forgets `lane`'s return: its session stepped again, or is gone.
    fn forget_return(&self, lane: usize) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.returning.retain(|(returned, _)| *returned != lane);
            self.queued.notify_all();
        }
    }

    /// Whether a lane drafts now: the group drafts and few enough lanes are held that the
    /// stages a lane leaves idle are not another lane's.
    fn drafts(&self, head: &Head) -> bool {
        self.draft_lanes > 0 && !head.closed && head.held() <= self.draft_lanes
    }

    /// Drops `lane`'s rows started on guesses, from `position` (its session's next) on: rank 0
    /// goes back there at once, every other rank at the lane's next row, and a guess from a
    /// dropped row is stale.
    fn roll_back(&self, lane: &mut HeadLane, position: usize) -> Result<(), BackendError> {
        if lane.draft.path.is_empty() {
            return Ok(());
        }
        let stale: Vec<Ticket> = lane.draft.path.drain(..).map(|row| row.ticket).collect();
        lane.draft.epoch = lane.draft.epoch.wrapping_add(1);
        lane.draft.rewind = true;
        lane.core.rewind(position)?;
        self.discard(stale)
    }

    /// Queues one drafting row of `lane` alone on rank 0 and sends it on, marked for a guess,
    /// and returns its answer's ticket. The lane must have room for the row.
    fn start_direct(
        &self,
        head: &mut Head,
        lane: usize,
        token: u32,
    ) -> Result<Ticket, BackendError> {
        let step = Waiting {
            session: head.lanes[lane].owner.unwrap_or_default(),
            lane,
            token,
        };
        let number = self.start(head, &[step], true)?;
        Ok(Ticket { number, entry: 0 })
    }

    /// Queues `steps` (ascending, distinct lanes, each with room for its row) on rank 0 in one
    /// pass and sends them on, each marked for a guess when `draft`, and returns the ticket of
    /// their answer message.
    fn start(&self, head: &mut Head, steps: &[Waiting], draft: bool) -> Result<u64, BackendError> {
        self.guarded(|| head.step_lanes(self.shared.cuda(), steps, draft, None))?;
        let number = head.ticket();
        self.post(number, steps.len(), false)?;
        if draft {
            for step in steps {
                let lane = &mut head.lanes[step.lane];
                lane.draft.awaiting = Some(Awaiting {
                    epoch: lane.draft.epoch,
                    position: lane.core.position(),
                    pass: number,
                });
            }
        }
        Ok(number)
    }

    /// Queues `session`'s `row` on its drafting lane to join the next guesses, when another
    /// held lane's guess is on its way or another lane's row already waits for one: true when
    /// it did, false when the caller is to start the row at once.
    fn join(&self, head: &Head, session: u64, row: StepEntry) -> Result<bool, BackendError> {
        let awaited = head.lanes.iter().enumerate().any(|(index, other)| {
            index != row.lane && other.owner.is_some() && other.draft.awaiting.is_some()
        });
        let mut guesses = self.guesses.lock().map_err(poisoned)?;
        let others = guesses
            .joining
            .iter()
            .any(|joining| joining.row.lane != row.lane);
        if guesses.closed || !(awaited || others) {
            return Ok(false);
        }
        guesses.joining.push(Joining {
            session,
            row,
            since: std::time::Instant::now(),
        });
        drop(guesses);
        self.guessed.notify_all();
        Ok(true)
    }

    /// Waits for the speculator to start `session`'s joining row and returns its answer's
    /// ticket.
    fn joined_ticket(&self, session: u64) -> Result<Ticket, BackendError> {
        let mut guesses = self.guesses.lock().map_err(poisoned)?;
        loop {
            if let Some(index) = guesses.joined.iter().position(|(id, _)| *id == session) {
                return guesses.joined.swap_remove(index).1;
            }
            if let Some(reason) = self.broken() {
                guesses.joining.retain(|joining| joining.session != session);
                return Err(BackendError::new(RECURRENCE, reason));
            }
            guesses = self.joined.wait(guesses).map_err(poisoned)?;
        }
    }

    /// Waits until a guess has come or the oldest joining row has waited [GATHER_WAIT]: false
    /// once the drafts link has closed or the group is broken.
    fn await_guesses(&self) -> bool {
        let Ok(mut guesses) = self.guesses.lock() else {
            return false;
        };
        loop {
            if !guesses.entries.is_empty() {
                return true;
            }
            if guesses.closed || self.broken().is_some() {
                return false;
            }
            let waited = guesses
                .joining
                .first()
                .map(|joining| joining.since.elapsed());
            let next = match waited {
                Some(waited) if waited >= GATHER_WAIT => return true,
                Some(waited) => self
                    .guessed
                    .wait_timeout(guesses, GATHER_WAIT - waited)
                    .ok()
                    .map(|(next, _)| next),
                None => self.guessed.wait(guesses).ok(),
            };
            let Some(next) = next else {
                return false;
            };
            guesses = next;
        }
    }

    /// Holds the guesses that have come until another comes or `deadline` passes.
    fn hold_guesses(&self, deadline: std::time::Instant) {
        let Ok(guesses) = self.guesses.lock() else {
            return;
        };
        let count = guesses.entries.len();
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let _ = self.guessed.wait_timeout_while(guesses, left, |guesses| {
            guesses.entries.len() == count && !guesses.closed && self.broken().is_none()
        });
    }

    /// Refuses every row still waiting to join the guesses: the speculator has stopped.
    fn refuse_joining(&self) {
        let reason = self
            .broken()
            .unwrap_or_else(|| "the recurrence group is shut down".to_string());
        let Ok(mut guesses) = self.guesses.lock() else {
            return;
        };
        let refused: Vec<_> = guesses
            .joining
            .drain(..)
            .map(|joining| {
                let error = BackendError::new(RECURRENCE, reason.clone());
                (joining.session, Err(error))
            })
            .collect();
        guesses.joined.extend(refused);
        drop(guesses);
        self.joined.notify_all();
    }

    /// Whether the guesses that have come wait for another pass's: a held lane's guess from a
    /// row it still drafts on is on its way, from a pass whose lanes all fit in the room the
    /// guesses and the joining rows leave in their last pass.
    fn gathers(&self, head: &Head, mailbox: &Guesses) -> bool {
        let mut taken: Vec<usize> = mailbox
            .entries
            .iter()
            .filter(|guess| {
                head.lanes
                    .get(guess.lane)
                    .is_some_and(|lane| lane.draft.epoch == guess.epoch)
            })
            .map(|guess| guess.lane)
            .chain(mailbox.joining.iter().map(|joining| joining.row.lane))
            .collect();
        taken.sort_unstable();
        taken.dedup();
        if taken.is_empty() {
            return false;
        }
        let room = (LANE_BATCH - taken.len() % LANE_BATCH) % LANE_BATCH;
        let mut passes: Vec<(u64, usize)> = Vec::new();
        for (index, lane) in head.lanes.iter().enumerate() {
            let Some(awaiting) = lane.draft.awaiting else {
                continue;
            };
            if lane.owner.is_none()
                || awaiting.epoch != lane.draft.epoch
                || taken.binary_search(&index).is_ok()
            {
                continue;
            }
            match passes.iter_mut().find(|(pass, _)| *pass == awaiting.pass) {
                Some((_, count)) => *count += 1,
                None => passes.push((awaiting.pass, 1)),
            }
        }
        passes.iter().any(|(_, count)| *count <= room)
    }

    /// Starts the rows the guesses feed and the rows waiting to join them, in passes of at most
    /// [LANE_BATCH] lanes, the guessed rows in the order their guesses came.
    ///
    /// A guess starts its row when its lane still drafts and the guess comes from the lane's
    /// last row: a guess from a row a rollback dropped, from a sequence that has ended, that
    /// ends the turn, past [DRAFT_DEPTH] rows ahead, or for a lane whose session has a row
    /// waiting to join is let go. Joining rows start whether or not the group drafts now, in
    /// the room the guessed rows leave in their last pass; those it has no room for wait for the
    /// next guesses, until they have waited [GATHER_WAIT].
    ///
    /// While [Group::gathers] holds, the guesses wait until [GATHER_WAIT] after the oldest came:
    /// then nothing starts, and the time to look again is returned.
    fn speculate(&self) -> Result<Option<std::time::Instant>, BackendError> {
        let mut head = self.lock_head()?;
        let (guesses, joining) = {
            let mut mailbox = self.guesses.lock().map_err(poisoned)?;
            if let Some(since) = mailbox.since {
                let deadline = since + GATHER_WAIT;
                if std::time::Instant::now() < deadline && self.gathers(&head, &mailbox) {
                    return Ok(Some(deadline));
                }
            }
            mailbox.since = None;
            let guesses: Vec<StepEntry> = mailbox.entries.drain(..).collect();
            (guesses, std::mem::take(&mut mailbox.joining))
        };
        let config = self.shared.config();
        let draft = self.drafts(&head);
        let mut rows: Vec<(Waiting, Option<Started>)> = Vec::new();
        for guess in &guesses {
            let Some(lane) = head.lanes.get_mut(guess.lane) else {
                return Err(link_error(format!(
                    "the drafting rank guessed for lane {}, which rank 0 does not have",
                    guess.lane
                )));
            };
            if lane.draft.awaiting.is_some_and(|awaiting| {
                (awaiting.epoch, awaiting.position) == (guess.epoch, guess.position)
            }) {
                lane.draft.awaiting = None;
            }
            let follows = draft
                && lane.owner.is_some()
                && lane.draft.epoch == guess.epoch
                && lane.core.position() == guess.position
                && lane.draft.path.len() < DRAFT_DEPTH
                && lane.core.check_rows(1).is_ok()
                && (guess.fed as usize) < config.vocab_size
                && config.next_token(guess.fed).is_some()
                && !joining.iter().any(|joining| joining.row.lane == guess.lane)
                && !rows.iter().any(|(step, _)| step.lane == guess.lane);
            if follows {
                let step = Waiting {
                    session: lane.owner.unwrap_or_default(),
                    lane: guess.lane,
                    token: guess.fed,
                };
                let started = Started {
                    position: guess.position,
                    fed: guess.fed,
                    ticket: Ticket {
                        number: 0,
                        entry: 0,
                    },
                };
                rows.push((step, Some(started)));
            }
        }
        let mut room = match rows.len() % LANE_BATCH {
            0 if rows.is_empty() => LANE_BATCH,
            0 => 0,
            used => LANE_BATCH - used,
        };
        let mut joined = Vec::new();
        let mut waiting = Vec::new();
        for joining in joining {
            let row = joining.row;
            let lane = &head.lanes[row.lane];
            let checked = if head.closed {
                Err(BackendError::new(
                    RECURRENCE,
                    "the recurrence group is shut down",
                ))
            } else if lane.owner != Some(joining.session)
                || lane.draft.epoch != row.epoch
                || lane.core.position() != row.position
            {
                Err(BackendError::new(
                    RECURRENCE,
                    format!(
                        "lane {} moved on while its row at position {} waited to join the \
                         guesses",
                        row.lane, row.position
                    ),
                ))
            } else {
                lane.core.check_rows(1)
            };
            if let Err(error) = checked {
                joined.push((joining.session, Err(error)));
                continue;
            }
            if room == 0 && joining.since.elapsed() < GATHER_WAIT {
                waiting.push(joining);
                continue;
            }
            room = room.checked_sub(1).unwrap_or(LANE_BATCH - 1);
            let step = Waiting {
                session: joining.session,
                lane: row.lane,
                token: row.fed,
            };
            rows.push((step, None));
        }
        let mut batch = Vec::with_capacity(LANE_BATCH);
        for row in rows {
            if batch.len() == LANE_BATCH {
                self.start_batch(&mut head, &mut batch, draft, &mut joined)?;
            }
            batch.push(row);
        }
        self.start_batch(&mut head, &mut batch, draft, &mut joined)?;
        drop(head);
        if !joined.is_empty() || !waiting.is_empty() {
            let mut mailbox = self.guesses.lock().map_err(poisoned)?;
            mailbox.joined.extend(joined);
            waiting.append(&mut mailbox.joining);
            mailbox.joining = waiting;
            drop(mailbox);
            self.joined.notify_all();
        }
        Ok(None)
    }

    /// Starts `batch` (distinct lanes) in one pass and empties it: a joining row's ticket goes
    /// to its session in `joined`, a guessed row onto its lane's path.
    fn start_batch(
        &self,
        head: &mut Head,
        batch: &mut Vec<(Waiting, Option<Started>)>,
        draft: bool,
        joined: &mut Vec<(u64, Result<Ticket, BackendError>)>,
    ) -> Result<(), BackendError> {
        if batch.is_empty() {
            return Ok(());
        }
        // Every rank runs a batch's lanes in ascending order.
        batch.sort_by_key(|(step, _)| step.lane);
        let steps: Vec<Waiting> = batch.iter().map(|(step, _)| *step).collect();
        let number = self.start(head, &steps, draft)?;
        for (entry, (step, started)) in batch.drain(..).enumerate() {
            let ticket = Ticket { number, entry };
            match started {
                Some(started) => head.lanes[step.lane]
                    .draft
                    .path
                    .push_back(Started { ticket, ..started }),
                None => joined.push((step.session, Ok(ticket))),
            }
        }
        Ok(())
    }

    /// Queues `session`'s decode step of `token` on `lane` and returns its answer's ticket, once
    /// a batch has taken it: this session's own, when it is the one driving, or another's.
    fn submit(&self, session: u64, lane: usize, token: u32) -> Result<Ticket, BackendError> {
        let mut queue = self.queue.lock().map_err(poisoned)?;
        queue.waiting.push(Waiting {
            session,
            lane,
            token,
        });
        queue.returning.retain(|(returned, _)| *returned != lane);
        // A driver holding its batch for this lane goes on.
        self.queued.notify_all();
        loop {
            if let Some(index) = queue.taken.iter().position(|(id, _)| *id == session) {
                return queue.taken.swap_remove(index).1;
            }
            if let Some(reason) = self.broken() {
                queue.waiting.retain(|step| step.session != session);
                return Err(BackendError::new(RECURRENCE, reason));
            }
            if queue.driving {
                queue = self.queued.wait(queue).map_err(poisoned)?;
                continue;
            }
            queue.driving = true;
            drop(queue);
            let driven = self.drive();
            queue = self.queue.lock().map_err(poisoned)?;
            queue.driving = false;
            self.queued.notify_all();
            driven?;
        }
    }

    /// Takes the waiting steps into one batch and queues it on rank 0, after rank 0 has run the
    /// batch before: the steps that came back meanwhile are in this one. One session at a time.
    fn drive(&self) -> Result<(), BackendError> {
        let cuda = self.shared.cuda();
        let fence = self.batch_done.lock().map_err(poisoned)?;
        self.guarded(|| cuda.host_wait(&fence.event).map_err(link_error))?;
        self.gather()?;
        let mut head = self.lock_head()?;
        let steps: Vec<Waiting> = {
            let mut queue = self.queue.lock().map_err(poisoned)?;
            let count = queue.waiting.len().min(LANE_BATCH);
            queue.waiting.drain(..count).collect()
        };
        let mut outcomes = Vec::with_capacity(steps.len());
        let mut batch: Vec<Waiting> = Vec::with_capacity(steps.len());
        for step in steps {
            let checked = head
                .held_by(step.session, Some(step.lane))
                .and_then(|lane| {
                    if head.closed {
                        Err(BackendError::new(
                            RECURRENCE,
                            "the recurrence group is shut down",
                        ))
                    } else {
                        head.lanes[lane].core.check_rows(1)
                    }
                });
            match checked {
                Ok(()) => batch.push(step),
                Err(error) => outcomes.push((step.session, Err(error))),
            }
        }
        // Every rank runs a batch's lanes in ascending order.
        batch.sort_by_key(|step| step.lane);
        if !batch.is_empty() {
            match self.guarded(|| head.step_lanes(cuda, &batch, false, Some(&fence))) {
                Ok(()) => {
                    let number = head.ticket();
                    self.queue.lock().map_err(poisoned)?.in_flight.push(number);
                    self.post(number, batch.len(), true)?;
                    for (entry, step) in batch.iter().enumerate() {
                        outcomes.push((step.session, Ok(Ticket { number, entry })));
                    }
                }
                Err(error) => {
                    for step in &batch {
                        outcomes.push((step.session, Err(error.clone())));
                    }
                }
            }
        }
        drop((head, fence));
        let mut queue = self.queue.lock().map_err(poisoned)?;
        queue.taken.extend(outcomes);
        Ok(())
    }

    /// Holds the batch about to go until fewer than [Group::batches_in_flight] are in flight,
    /// then for the lanes whose answers were just read, until each has stepped again or
    /// [RETURN_WINDOW] has passed since its answer, or the batch is full.
    fn gather(&self) -> Result<(), BackendError> {
        loop {
            // The head's lock comes before the queue's: count the held lanes outside it. A lane
            // let go wakes this wait as well as an answer read.
            let cap = self.batches_in_flight(self.held_lanes());
            let queue = self.queue.lock().map_err(poisoned)?;
            if self.broken().is_some() || queue.in_flight.len() < cap {
                break;
            }
            drop(self.queued.wait(queue).map_err(poisoned)?);
        }
        let mut queue = self.queue.lock().map_err(poisoned)?;
        loop {
            if queue.waiting.len() >= LANE_BATCH || self.broken().is_some() {
                return Ok(());
            }
            let now = std::time::Instant::now();
            let Some(until) = queue
                .returning
                .iter()
                .map(|(_, at)| *at + RETURN_WINDOW)
                .filter(|until| *until > now)
                .max()
            else {
                return Ok(());
            };
            queue = self
                .queued
                .wait_timeout(queue, until - now)
                .map_err(poisoned)?
                .0;
        }
    }

    /// The decode batches rank 0 keeps in flight with `held` lanes held: one for each rank after
    /// it, so the slowest rank always has a batch to run, or more when the lanes need more than
    /// [LANE_BATCH] a batch. More than that and the batches queue in front of the slowest rank
    /// without its running any sooner, while the lanes behind them could have joined one wider
    /// batch at rank 0 for the same pass.
    fn batches_in_flight(&self, held: usize) -> usize {
        (self.world - 1).max(held.div_ceil(LANE_BATCH)).max(1)
    }

    /// The lanes a session holds.
    fn held_lanes(&self) -> usize {
        self.head.lock().map_or(0, |head| head.held())
    }

    fn shutdown(&self) -> Result<(), String> {
        let mut head = self.head.lock().map_err(|_| "the head lock is poisoned")?;
        let head = &mut *head;
        head.closed = true;
        let lane = head
            .lanes
            .first()
            .ok_or("a recurrence group without a lane")?;
        let cuda = self.shared.cuda();
        lane.buffers.stage_frames(cuda, &[Frame::Shutdown])?;
        head.ring.send(cuda, 1, lane.buffers.slot(0), FRAME_BYTES)?;
        cuda.synchronize()
    }
}

/// Rank 0's answer reader: reads each answer message as soon as it is expected, in ticket
/// order, onto the board, until the group closes or breaks.
///
/// The last rank's answer for a row no session will take (one a rollback dropped, or one a
/// session left in flight when it ended) is read all the same: an answer left unread would
/// hold the last rank's stream, and the ring behind it.
fn read_answers(group: &Group, mut answers: Answers) {
    if let Err(error) = group.shared.cuda().make_current() {
        group.mark_broken(&link_error(error));
        return;
    }
    loop {
        let sent = {
            let Ok(mut board) = group.board.lock() else {
                return;
            };
            loop {
                if let Some(sent) = board.sent.front().copied() {
                    break sent;
                }
                if board.closed || group.broken().is_some() {
                    return;
                }
                let Ok(next) = group.turn.wait(board) else {
                    return;
                };
                board = next;
            }
        };
        let entries = match answers.read(&group.shared, group.world - 1, sent) {
            Ok(entries) => entries,
            Err(error) => {
                group.mark_broken(&error);
                return;
            }
        };
        if sent.queued && group.returned(sent.number, &entries).is_err() {
            return;
        }
        let Ok(mut board) = group.board.lock() else {
            return;
        };
        board.sent.pop_front();
        for entry in entries {
            match board
                .discard
                .iter()
                .position(|stale| *stale == (entry.number, entry.entry))
            {
                Some(index) => {
                    board.discard.swap_remove(index);
                }
                None => board.ready.push(entry),
            }
        }
        drop(board);
        group.turn.notify_all();
    }
}

/// Rank 0's guess listener: reads the drafting rank's frames as they come, onto the guesses,
/// until it passes the group's `Shutdown` on.
fn listen(group: &Group, mut drafts: DraftLink) {
    if let Err(error) = group.shared.cuda().make_current() {
        group.mark_broken(&link_error(error));
        return;
    }
    loop {
        let frame = match drafts.read(&group.shared) {
            Ok(frame) => frame,
            Err(error) => {
                group.mark_broken(&error);
                return;
            }
        };
        let closed = match frame {
            Frame::Draft { count, entries } => {
                let Ok(mut guesses) = group.guesses.lock() else {
                    return;
                };
                if guesses.entries.is_empty() {
                    guesses.since = Some(std::time::Instant::now());
                }
                guesses.entries.extend(&entries[..count]);
                false
            }
            Frame::Shutdown => {
                let Ok(mut guesses) = group.guesses.lock() else {
                    return;
                };
                guesses.closed = true;
                true
            }
            other => {
                group.mark_broken(&link_error(format!(
                    "the drafting rank sent {other:?} on the drafts link"
                )));
                return;
            }
        };
        group.guessed.notify_all();
        if closed {
            return;
        }
    }
}

/// Rank 0's speculator: as Draft frames come, starts the rows their guesses feed with the rows
/// waiting to join them (see [Guesses]), until the drafts link closes or the group breaks. A
/// joining row that has waited [GATHER_WAIT] for a guess starts without one.
fn speculate_guesses(group: &Group) {
    if let Err(error) = group.shared.cuda().make_current() {
        group.mark_broken(&link_error(error));
    } else {
        while group.await_guesses() {
            match group.speculate() {
                Ok(None) => {}
                Ok(Some(deadline)) => group.hold_guesses(deadline),
                Err(error) => {
                    group.mark_broken(&error);
                    break;
                }
            }
        }
    }
    group.refuse_joining();
}

impl Head {
    /// The lane `session` holds, or a free one it takes now.
    fn claim(&mut self, session: u64, held: Option<usize>) -> Result<usize, BackendError> {
        if let Some(lane) = held
            && self.lanes[lane].owner == Some(session)
        {
            return Ok(lane);
        }
        let count = self.lanes.len();
        let lane = self
            .lanes
            .iter()
            .position(|lane| lane.owner.is_none())
            .ok_or_else(|| {
                BackendError::new(
                    RECURRENCE,
                    format!(
                        "all {count} recurrence lanes hold a sequence: the group serves \
                         --concurrency {count} requests at once"
                    ),
                )
            })?;
        self.lanes[lane].owner = Some(session);
        Ok(lane)
    }

    /// Refuses a step of `session` on a lane it no longer holds.
    fn held_by(&self, session: u64, lane: Option<usize>) -> Result<usize, BackendError> {
        lane.filter(|lane| self.lanes[*lane].owner == Some(session))
            .ok_or_else(|| {
                BackendError::new(
                    RECURRENCE,
                    "this sequence holds no recurrence lane: a decode step needs its prefill",
                )
            })
    }

    /// The lanes a session holds.
    fn held(&self) -> usize {
        self.lanes
            .iter()
            .filter(|lane| lane.owner.is_some())
            .count()
    }

    /// Queues `rows` rows of `lane` at its position on rank 0 and sends them on, after frame
    /// slot `slot`. The rows are the embedded `tokens`.
    fn step(
        &mut self,
        cuda: &Cuda,
        lane: usize,
        slot: usize,
        tokens: &[u32],
    ) -> Result<(), BackendError> {
        let Self { ring, lanes, .. } = self;
        let entry = &mut lanes[lane];
        entry.core.embed(tokens)?;
        entry.core.forward(tokens.len(), false)?;
        let (state, input_h) = entry.core.carry();
        let bytes = tokens.len() * entry.core.shared().config().hidden_size * 4;
        ring.send(cuda, 1, entry.buffers.slot(slot), FRAME_BYTES)
            .map_err(link_error)?;
        ring.send(cuda, 1, state, bytes).map_err(link_error)?;
        ring.send(cuda, 1, input_h, bytes).map_err(link_error)
    }

    /// Queues one decode row for each of `steps` (ascending, distinct lanes) on rank 0 in one
    /// pass, marks `fence` after it, and sends the Step frame and the rows on, each marked for a
    /// guess when `draft`. The frame goes through the first lane's slot 0.
    fn step_lanes(
        &mut self,
        cuda: &Cuda,
        steps: &[Waiting],
        draft: bool,
        fence: Option<&Fence>,
    ) -> Result<(), BackendError> {
        let Self { ring, lanes, .. } = self;
        let Some(first) = steps.first() else {
            return Ok(());
        };
        if steps.windows(2).any(|pair| pair[0].lane >= pair[1].lane) {
            return Err(BackendError::new(
                RECURRENCE,
                "a batch's lanes must be distinct and ascending",
            ));
        }
        let mut entries = [StepEntry::default(); LANE_BATCH];
        for (entry, step) in entries.iter_mut().zip(steps) {
            let lane = &mut lanes[step.lane];
            *entry = StepEntry {
                lane: step.lane,
                position: lane.core.position(),
                fed: step.token,
                fault: None,
                draft,
                rewind: std::mem::take(&mut lane.draft.rewind),
                epoch: lane.draft.epoch,
            };
        }
        let frame = Frame::Step {
            count: steps.len(),
            entries,
        };
        let slot = lanes[first.lane].buffers.slot(0);
        lanes[first.lane]
            .buffers
            .stage_frames(cuda, &[frame])
            .map_err(link_error)?;
        let tokens: Vec<u32> = steps.iter().map(|step| step.token).collect();
        let mut cores: Vec<&mut StageCore> = lanes
            .iter_mut()
            .enumerate()
            .filter(|(index, _)| steps.binary_search_by_key(index, |step| step.lane).is_ok())
            .map(|(_, lane)| &mut lane.core)
            .collect();
        StageCore::embed_lanes(&cores, &tokens)?;
        StageCore::forward_lanes(&mut cores, &vec![true; steps.len()], false)?;
        if let Some(fence) = fence {
            cuda.run_mark(&fence.event).map_err(link_error)?;
        }
        let (state, input_h) = cores[0].carry();
        let bytes = steps.len() * cores[0].shared().config().hidden_size * 4;
        ring.send(cuda, 1, slot, FRAME_BYTES).map_err(link_error)?;
        ring.send(cuda, 1, state, bytes).map_err(link_error)?;
        ring.send(cuda, 1, input_h, bytes).map_err(link_error)
    }

    fn ticket(&mut self) -> u64 {
        let ticket = self.tickets;
        self.tickets += 1;
        ticket
    }
}

impl Answers {
    /// Reads the answer message `sent`: its entries' frames, then their rows.
    fn read(
        &mut self,
        shared: &DeviceShared,
        last: usize,
        sent: Sent,
    ) -> Result<Vec<Ready>, BackendError> {
        let cuda = shared.cuda();
        let count = sent.count;
        if count == 0 || count > LANE_BATCH {
            return Err(link_error(format!(
                "an answer message of {count} entries is outside 1..={LANE_BATCH}"
            )));
        }
        let row_bytes = shared.config().vocab_size * 4;
        let frames = LANE_BATCH * FRAME_BYTES;
        self.link
            .recv(cuda, last, self.answer, count * FRAME_BYTES)
            .map_err(link_error)?;
        self.link
            .recv(cuda, last, self.answer.offset(frames), count * row_bytes)
            .map_err(link_error)?;
        cuda.copy_dtoh_mapped_on(
            &self.stream.stream,
            &self.staged,
            0,
            &self.answer,
            frames + count * row_bytes,
        )
        .map_err(link_error)?;
        (0..count)
            .map(|entry| {
                let mut bytes = [0u8; FRAME_BYTES];
                self.staged
                    .read_at(entry * FRAME_BYTES, &mut bytes)
                    .map_err(link_error)?;
                let mut row = vec![0u8; row_bytes];
                self.staged
                    .read_at(frames + entry * row_bytes, &mut row)
                    .map_err(link_error)?;
                Ok(Ready {
                    number: sent.number,
                    entry,
                    frame: Frame::from_bytes(&bytes).map_err(link_error)?,
                    row,
                })
            })
            .collect()
    }
}

impl DraftLink {
    /// Waits for the drafting rank's next frame.
    fn read(&mut self, shared: &DeviceShared) -> Result<Frame, BackendError> {
        let cuda = shared.cuda();
        self.link
            .recv(cuda, DRAFTER, self.guess, FRAME_BYTES)
            .map_err(link_error)?;
        cuda.copy_dtoh_mapped_on(
            &self.stream.stream,
            &self.staged,
            0,
            &self.guess,
            FRAME_BYTES,
        )
        .map_err(link_error)?;
        let mut bytes = [0u8; FRAME_BYTES];
        self.staged.read_at(0, &mut bytes).map_err(link_error)?;
        Frame::from_bytes(&bytes).map_err(link_error)
    }
}

/// An event marked on rank 0's stream: a session's prompt chunk, a batch, or a staging copy.
struct Fence {
    shared: Arc<DeviceShared>,
    event: StreamEvent,
}

impl Fence {
    fn new(shared: &Arc<DeviceShared>) -> Result<Self, String> {
        Ok(Self {
            shared: Arc::clone(shared),
            event: shared.cuda().stream_event()?,
        })
    }
}

// SAFETY: an event handle is an opaque driver object; whoever owns it records and waits on it
// from one thread at a time (a session's, or under the lock that holds it), with the context
// current.
unsafe impl Send for Fence {}

impl Drop for Fence {
    fn drop(&mut self) {
        let cuda = self.shared.cuda();
        if cuda.make_current().is_ok() {
            cuda.destroy_event(&self.event);
        }
    }
}

/// One request's sequence on rank 0.
struct DoryRecurrenceSession {
    group: Arc<Group>,
    id: u64,
    /// The lane this session holds, from its prefill.
    lane: Option<usize>,
    sampling: Sampling,
    fences: [Fence; PREFILL_AHEAD],
    /// The position of the sequence's next decode row: the prompt's length, then one more a
    /// step.
    position: usize,
    /// The guesses this sequence's rows were started on, and how many the model agreed with.
    speculation: Speculation,
}

impl Drop for DoryRecurrenceSession {
    fn drop(&mut self) {
        let Some(lane) = self.lane else {
            return;
        };
        if let Ok(mut head) = self.group.head.lock()
            && head.lanes[lane].owner == Some(self.id)
        {
            let entry = &mut head.lanes[lane];
            // The rows started on guesses are no one's now; their answers are dropped as read.
            let _ = self.group.roll_back(entry, self.position);
            entry.draft.epoch = entry.draft.epoch.wrapping_add(1);
            entry.owner = None;
        }
        self.group.forget_return(lane);
    }
}

impl DoryRecurrenceSession {
    /// Checks the row after the one just answered, when a guess started it: its guess must be
    /// the token the model chose. A wrong guess, or an end of turn, drops it and every row
    /// started after it.
    fn verify(&mut self, output: &StepOutput) -> Result<(), BackendError> {
        let group = Arc::clone(&self.group);
        let mut head = group.lock_head()?;
        let Some(lane) = self
            .lane
            .filter(|lane| head.lanes[*lane].owner == Some(self.id))
        else {
            return Ok(());
        };
        let entry = &mut head.lanes[lane];
        let Some(row) = entry.draft.path.front().copied() else {
            return Ok(());
        };
        self.speculation.verify_steps += 1;
        self.speculation.drafts_proposed += 1;
        if row.position == self.position && Some(row.fed) == output.next_token {
            self.speculation.drafts_accepted += 1;
            Ok(())
        } else {
            group.roll_back(entry, self.position)
        }
    }
}

/// How a session's decode row goes: already started, waiting to join the drafting lanes' next
/// guesses, or through the decode queue.
enum Start {
    Started(Ticket),
    Joining,
    Queue,
}

impl ModelSession for DoryRecurrenceSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        // The last rank chooses the token: the sampling travels to it in the Begin frame.
        self.sampling = sampling;
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        let group = Arc::clone(&self.group);
        let cuda = group.shared.cuda();
        cuda.make_current().map_err(link_error)?;
        if tokens.is_empty() {
            return Err(BackendError::new(RECURRENCE, "the prompt has no tokens"));
        }
        let context = group.shared.context();
        if tokens.len() > context {
            return Err(BackendError::new(
                RECURRENCE,
                format!(
                    "a prompt of {} tokens is past the served context {context}",
                    tokens.len()
                ),
            ));
        }
        group.validate(tokens)?;
        let chunks = prompt_chunks(
            KvCacheMode::PerLoop,
            tokens.len(),
            recurrence_chunk(tokens.len()),
        );
        let lane = {
            let mut head = group.lock_head()?;
            if head.closed {
                return Err(BackendError::new(
                    RECURRENCE,
                    "the recurrence group is shut down",
                ));
            }
            let lane = head.claim(self.id, self.lane)?;
            self.lane = Some(lane);
            let slots = head.lanes[lane].buffers.slots;
            if 1 + chunks.len() > slots {
                return Err(BackendError::new(
                    RECURRENCE,
                    format!(
                        "a prompt of {} chunks needs more than the {slots} frame slots",
                        chunks.len()
                    ),
                ));
            }
            let mut frames = Vec::with_capacity(1 + chunks.len());
            frames.push(Frame::Begin {
                lane,
                prompt: tokens.len(),
                sampling: self.sampling,
            });
            for (index, chunk) in chunks.iter().enumerate() {
                frames.push(Frame::Rows {
                    lane,
                    rows: chunk.len(),
                    position: chunk.start,
                    output: index + 1 == chunks.len(),
                });
            }
            // A sequence the session ran before on this lane may have rows started on guesses
            // still in flight: they are no one's, and every rank restarts the lane at Begin.
            let entry = &mut head.lanes[lane];
            let stale: Vec<Ticket> = entry.draft.path.drain(..).map(|row| row.ticket).collect();
            group.discard(stale)?;
            entry.draft.epoch = entry.draft.epoch.wrapping_add(1);
            entry.draft.rewind = false;
            entry.draft.awaiting = None;
            group.guarded(|| {
                let Head { ring, lanes, .. } = &mut *head;
                let entry = &mut lanes[lane];
                entry
                    .buffers
                    .stage_frames(cuda, &frames)
                    .map_err(link_error)?;
                let bytes = entry.buffers.stage_ids(cuda, tokens).map_err(link_error)?;
                entry.core.restart();
                ring.send(cuda, 1, entry.buffers.slot(0), FRAME_BYTES)
                    .map_err(link_error)?;
                ring.send(cuda, 1, entry.buffers.ids, bytes)
                    .map_err(link_error)
            })?;
            lane
        };
        self.position = tokens.len();
        self.speculation = Speculation::default();
        let mut ticket = None;
        for (index, chunk) in chunks.iter().enumerate() {
            let fence = &self.fences[index % PREFILL_AHEAD];
            if index >= PREFILL_AHEAD {
                // The chunk PREFILL_AHEAD before this one has left rank 0's stream.
                group.guarded(|| cuda.host_wait(&fence.event).map_err(link_error))?;
            }
            let mut head = group.lock_head()?;
            ticket = group.guarded(|| {
                head.step(cuda, lane, index + 1, &tokens[chunk.clone()])?;
                cuda.run_mark(&fence.event).map_err(link_error)?;
                if index + 1 < chunks.len() {
                    return Ok(None);
                }
                let number = head.ticket();
                group.post(number, 1, false)?;
                Ok(Some(Ticket { number, entry: 0 }))
            })?;
        }
        let ticket =
            ticket.ok_or_else(|| BackendError::new(RECURRENCE, "a prompt of no chunks"))?;
        group.answer(ticket, lane)
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        let group = Arc::clone(&self.group);
        let cuda = group.shared.cuda();
        cuda.make_current().map_err(link_error)?;
        group.validate(&[token])?;
        let lane = self.lane.ok_or_else(|| {
            BackendError::new(
                RECURRENCE,
                "this sequence holds no recurrence lane: a decode step needs its prefill",
            )
        })?;
        let position = self.position;
        let started = {
            let mut head = group.lock_head()?;
            head.held_by(self.id, Some(lane))?;
            let entry = &mut head.lanes[lane];
            match entry.draft.path.front().copied() {
                // A guess already started this row, on this very token.
                Some(row) if row.position == position && row.fed == token => {
                    entry.draft.path.pop_front();
                    Start::Started(row.ticket)
                }
                _ => {
                    // Rows started on guesses that are not this one (the caller fed another
                    // token than the group chose) are dropped.
                    group.roll_back(entry, position)?;
                    if group.drafts(&head) {
                        let entry = &head.lanes[lane];
                        entry.core.check_rows(1)?;
                        let row = StepEntry {
                            lane,
                            position,
                            fed: token,
                            epoch: entry.draft.epoch,
                            ..StepEntry::default()
                        };
                        if group.join(&head, self.id, row)? {
                            Start::Joining
                        } else {
                            Start::Started(group.start_direct(&mut head, lane, token)?)
                        }
                    } else {
                        // Through the queue, with the other lanes' steps and no guess: one
                        // from an earlier row of this lane is let go.
                        let entry = &mut head.lanes[lane];
                        entry.draft.epoch = entry.draft.epoch.wrapping_add(1);
                        Start::Queue
                    }
                }
            }
        };
        let ticket = match started {
            Start::Started(ticket) => ticket,
            Start::Joining => group.joined_ticket(self.id)?,
            Start::Queue => group.submit(self.id, lane, token)?,
        };
        let output = group.answer(ticket, lane)?;
        self.position += 1;
        self.verify(&output)?;
        Ok(output)
    }

    fn speculation(&self) -> Speculation {
        self.speculation
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        None
    }
}

/// One lane on a rank after 0.
struct WorkerLane {
    core: StageCore,
    sampler: Sampler,
    sampling: Sampling,
    /// This lane's sequence was refused here: its later chunks are drained and passed on as
    /// faults.
    faulted: bool,
    /// On the last rank, the sampler as it stood before each of the lane's latest decode rows,
    /// by the row's position, oldest first: a rewind to a row takes the sampler back to its mark.
    marks: VecDeque<(usize, SamplerMark)>,
}

impl WorkerLane {
    /// Takes the lane back to `position` for a row rank 0 marked to rewind: the rows from there
    /// on were started on a wrong guess. The last rank also takes the sampler back to where it
    /// stood before the row at `position`.
    fn rewind(&mut self, position: usize, last: bool) -> Result<(), String> {
        self.core
            .rewind(position)
            .map_err(|error| error.to_string())?;
        if last {
            let index = self
                .marks
                .iter()
                .position(|(at, _)| *at == position)
                .ok_or_else(|| {
                    format!(
                        "a rewind to position {position} is past the {} rows whose sampler the \
                         last rank keeps",
                        self.marks.len()
                    )
                })?;
            self.sampler.rewind(self.marks[index].1);
            self.marks.truncate(index);
        }
        Ok(())
    }

    /// Notes the sampler as it stands before the decode row at `position`.
    fn mark(&mut self, position: usize) {
        if self.marks.len() == MARKS {
            self.marks.pop_front();
        }
        self.marks.push_back((position, self.sampler.mark()));
    }
}

/// A rank after 0: its loops over the latent rows it receives, and on the last rank the suffix,
/// the head and the sampler, for every lane. In a drafting group rank [DRAFTER] also runs the
/// draft exit over every row it receives, and sends rank 0 its guess of the token after each
/// row rank 0 marked for one.
pub struct DoryRecurrenceStage {
    ring: Box<dyn RecurrenceLink>,
    answers: Box<dyn RecurrenceLink>,
    /// Every rank of a drafting group holds the drafts link; only the drafting rank sends on it.
    drafts: Option<Box<dyn RecurrenceLink>>,
    buffers: LinkBuffers,
    lanes: Vec<WorkerLane>,
    shared: Arc<DeviceShared>,
    rank: usize,
    world: usize,
}

impl DoryRecurrenceStage {
    /// Opens rank `rank` (1..world) of a `world`-rank group of `lanes` sequence lanes over
    /// `directory`, then its links on the stage's context (the factory is given the default
    /// stream, 0, for both of rank 0's streams). `draft_lanes` must be rank 0's: 0 for a group
    /// that never drafts, which opens no drafts link.
    ///
    /// # Errors
    /// Rank 0 (which is [DoryRecurrenceBackend]), no lane, a drafting group of fewer than three
    /// ranks, the stage's refusal, an allocation, or the links'.
    pub fn open(
        directory: &std::path::Path,
        context: usize,
        rank: usize,
        world: usize,
        lanes: usize,
        draft_lanes: usize,
        links: impl FnOnce(&Cuda, LinkStreams) -> Result<RecurrenceLinks, String>,
    ) -> Result<Self, String> {
        if rank == 0 {
            return Err(
                "recurrence rank 0 owns the sequences: open it as DoryRecurrenceBackend".to_owned(),
            );
        }
        let drafting = draft_lanes > 0;
        let shared = open_stage(directory, context, rank, world, lanes, drafting)?;
        let buffers = LinkBuffers::allocate(&shared)?;
        let entries = StageCore::lanes(&shared, lanes)?
            .into_iter()
            .map(|core| WorkerLane {
                core,
                sampler: Sampler::new(),
                sampling: Sampling::default(),
                faulted: false,
                marks: VecDeque::with_capacity(MARKS),
            })
            .collect();
        let RecurrenceLinks {
            ring,
            answers,
            drafts,
        } = links(
            shared.cuda(),
            LinkStreams {
                answers: 0,
                drafts: 0,
            },
        )?;
        if drafts.is_some() != drafting {
            return Err(format!(
                "recurrence rank {rank}'s link factory opened {} drafts link for a group that \
                 {}",
                if drafting { "no" } else { "a" },
                if drafting { "drafts" } else { "does not draft" }
            ));
        }
        let mut stage = Self {
            ring,
            answers,
            drafts,
            buffers,
            lanes: entries,
            shared,
            rank,
            world,
        };
        stage.connect()?;
        Ok(stage)
    }

    /// A worker's half of the `Connect` exchange: the frame from the rank before, on to the next
    /// rank, or from the last rank back to rank 0 on the answers link; the drafting rank also
    /// returns it to rank 0 on the drafts link.
    fn connect(&mut self) -> Result<(), String> {
        let shared = Arc::clone(&self.shared);
        let cuda = shared.cuda();
        let frame = self.buffers.frame;
        self.ring.recv(cuda, self.rank - 1, frame, FRAME_BYTES)?;
        match read_frame(cuda, frame)? {
            Frame::Connect => {}
            other => {
                return Err(format!(
                    "recurrence rank {} opened on {other:?}, not rank 0's Connect frame",
                    self.rank
                ));
            }
        }
        if self.last() {
            self.answers.send(cuda, 0, frame, FRAME_BYTES)?;
        } else {
            self.ring.send(cuda, self.rank + 1, frame, FRAME_BYTES)?;
        }
        if let Some(drafts) = self.drafter() {
            drafts.send(cuda, 0, frame, FRAME_BYTES)?;
        }
        cuda.synchronize()
    }

    /// The loops, prefix and suffix this rank runs.
    #[must_use]
    pub fn stage(&self) -> &DoryStage {
        self.shared.stage()
    }

    /// The device bytes this rank's weights claimed.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.shared.weight_bytes()
    }

    fn last(&self) -> bool {
        self.rank + 1 == self.world
    }

    /// The drafts link, on the rank that drafts.
    fn drafter(&mut self) -> Option<&mut Box<dyn RecurrenceLink>> {
        if self.shared.stage().draft {
            self.drafts.as_mut()
        } else {
            None
        }
    }

    /// Refuses a frame for a lane this rank does not have.
    fn check_lane(&self, lane: usize) -> Result<(), String> {
        if lane < self.lanes.len() {
            Ok(())
        } else {
            Err(format!(
                "a frame for lane {lane} reached recurrence rank {}, which has {} lanes: every \
                 rank must serve the same --concurrency",
                self.rank,
                self.lanes.len()
            ))
        }
    }

    /// Serves frames until rank 0 shuts the group down.
    ///
    /// # Errors
    /// A transfer or a device step that failed: the ring cannot be resumed past it, so the
    /// process should exit and the group restart.
    pub fn serve(mut self) -> Result<(), String> {
        let shared = Arc::clone(&self.shared);
        let cuda = shared.cuda();
        loop {
            let frame = self.buffers.frame;
            self.ring.recv(cuda, self.rank - 1, frame, FRAME_BYTES)?;
            match read_frame(cuda, frame)? {
                Frame::Begin {
                    lane,
                    prompt,
                    sampling,
                } => self.begin(lane, prompt, sampling)?,
                Frame::Rows {
                    lane,
                    rows,
                    position,
                    output,
                } => self.rows(lane, rows, position, output)?,
                Frame::Step { count, mut entries } => self.step(&mut entries[..count])?,
                Frame::Fault { lane, output, .. } => {
                    self.check_lane(lane)?;
                    self.pass_fault(frame, lane, output)?;
                }
                Frame::Shutdown => {
                    if !self.last() {
                        self.ring.send(cuda, self.rank + 1, frame, FRAME_BYTES)?;
                    }
                    // Rank 0's listener stops on it.
                    if let Some(drafts) = self.drafter() {
                        drafts.send(cuda, 0, frame, FRAME_BYTES)?;
                    }
                    cuda.synchronize()?;
                    return Ok(());
                }
                Frame::Output { .. } | Frame::Draft { .. } => {
                    return Err(format!(
                        "recurrence rank {} received an answer frame, which only rank 0 reads",
                        self.rank
                    ));
                }
                Frame::Connect => {
                    return Err(format!(
                        "recurrence rank {} received a Connect frame after the group opened",
                        self.rank
                    ));
                }
            }
        }
    }

    fn begin(&mut self, lane: usize, prompt: usize, sampling: Sampling) -> Result<(), String> {
        let context = self.shared.context();
        if prompt == 0 || prompt > context {
            return Err(format!(
                "a Begin frame for {prompt} prompt tokens is outside 1..={context}"
            ));
        }
        self.check_lane(lane)?;
        let last = self.last();
        let cuda = self.shared.cuda();
        let ids = self.buffers.ids;
        self.ring.recv(cuda, self.rank - 1, ids, prompt * 4)?;
        let entry = &mut self.lanes[lane];
        entry.core.restart();
        entry.faulted = false;
        entry.marks.clear();
        if last {
            let mut bytes = vec![0u8; prompt * 4];
            cuda.copy_dtoh(&mut bytes, &ids)?;
            let tokens: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect();
            entry.sampling = sampling;
            entry.sampler.reset();
            entry.sampler.record_prompt(&tokens);
        } else {
            self.ring
                .send(cuda, self.rank + 1, self.buffers.frame, FRAME_BYTES)?;
            self.ring.send(cuda, self.rank + 1, ids, prompt * 4)?;
        }
        Ok(())
    }

    fn rows(
        &mut self,
        lane: usize,
        rows: usize,
        position: usize,
        output: bool,
    ) -> Result<(), String> {
        if rows == 0 || rows > PREFILL_BATCH {
            // The payload cannot even be drained into this rank's buffers.
            return Err(format!(
                "a Rows frame of {rows} rows is outside 1..={PREFILL_BATCH}"
            ));
        }
        self.check_lane(lane)?;
        let last = self.last();
        let rank = self.rank;
        let cuda = self.shared.cuda();
        let config = self.shared.config();
        let bytes = rows * config.hidden_size * 4;
        let drafter = self.shared.stage().draft;
        let entry = &mut self.lanes[lane];
        let (state, input_h) = entry.core.carry();
        self.ring.recv(cuda, rank - 1, state, bytes)?;
        self.ring.recv(cuda, rank - 1, input_h, bytes)?;
        let refusal = if entry.faulted {
            Some(String::new())
        } else if position != entry.core.position() {
            Some(format!(
                "a chunk at position {position} of lane {lane} reached rank {rank} at position {}",
                entry.core.position()
            ))
        } else {
            entry
                .core
                .check_rows(rows)
                .err()
                .map(|error| error.to_string())
        };
        if let Some(reason) = refusal {
            if !reason.is_empty() {
                eprintln!("dory recurrence rank {rank}: {reason}");
            }
            entry.faulted = true;
            let reply = self.buffers.reply(0);
            write_frames(cuda, reply, &[Frame::Fault { lane, rank, output }])?;
            return self.pass_fault(reply, lane, output);
        }
        if drafter {
            // The draft exit's caches hold every row of the sequence, as the stage's own do; a
            // prompt's rows want no guess, so the exit runs once they have gone on.
            entry.core.keep(rows).map_err(|error| error.to_string())?;
        }
        let answer = last && output;
        entry
            .core
            .forward(rows, answer)
            .map_err(|error| error.to_string())?;
        if !last {
            let frame = self.buffers.frame;
            self.ring.send(cuda, rank + 1, frame, FRAME_BYTES)?;
            self.ring.send(cuda, rank + 1, state, bytes)?;
            self.ring.send(cuda, rank + 1, input_h, bytes)?;
            if drafter {
                entry
                    .core
                    .draft_kept(rows)
                    .map_err(|error| error.to_string())?;
            }
            return Ok(());
        }
        if answer {
            let logits = entry
                .core
                .read_logits()
                .map_err(|error| error.to_string())?;
            let token = entry.sampler.choose(&logits, &entry.sampling);
            let row = entry.core.logits_row(0);
            let reply = self.buffers.reply(0);
            write_frames(cuda, reply, &[Frame::Output { lane, token }])?;
            self.answers.send(cuda, 0, reply, FRAME_BYTES)?;
            self.answers.send(cuda, 0, row, config.vocab_size * 4)?;
        }
        Ok(())
    }

    /// One decode row for each of `entries`' lanes in one pass, then the Step on to the next rank,
    /// or from the last rank an answer for every entry, in order.
    ///
    /// A lane this rank refuses (a faulted sequence, a position out of step, the context full) is
    /// marked with this rank and its row passed on unread; one an earlier rank refused is passed
    /// on as it came. Either is answered with a `Fault`. A row marked to rewind first takes its
    /// lane back to the row's position.
    ///
    /// On the drafting rank the draft exit runs over every row: first, when a row is marked for a
    /// guess, so the guesses go to rank 0 before this rank's own loops run, and otherwise once the
    /// rows have gone on to the next rank, which then waits only on this rank's loops.
    fn step(&mut self, entries: &mut [StepEntry]) -> Result<(), String> {
        for entry in entries.iter() {
            self.check_lane(entry.lane)?;
        }
        if entries.windows(2).any(|pair| pair[0].lane >= pair[1].lane) {
            return Err(format!(
                "a Step frame reached recurrence rank {} with its lanes out of order: the ring is \
                 out of step",
                self.rank
            ));
        }
        let shared = Arc::clone(&self.shared);
        let cuda = shared.cuda();
        let config = shared.config();
        let rank = self.rank;
        let last = self.last();
        let drafter = shared.stage().draft;
        let count = entries.len();
        let bytes = count * config.hidden_size * 4;
        // Every lane's rows land in the lanes' one scratch set, a row a lane in entry order.
        let (state, input_h) = self.lanes[entries[0].lane].core.carry();
        self.ring.recv(cuda, rank - 1, state, bytes)?;
        self.ring.recv(cuda, rank - 1, input_h, bytes)?;
        let mut refused = false;
        for entry in entries.iter_mut().filter(|entry| entry.fault.is_none()) {
            let lane = &mut self.lanes[entry.lane];
            let refusal = if lane.faulted {
                Some(String::new())
            } else if let Some(Err(reason)) =
                entry.rewind.then(|| lane.rewind(entry.position, last))
            {
                Some(reason)
            } else if entry.position != lane.core.position() {
                Some(format!(
                    "a step at position {} of lane {} reached rank {rank} at position {}",
                    entry.position,
                    entry.lane,
                    lane.core.position()
                ))
            } else {
                lane.core.check_rows(1).err().map(|error| error.to_string())
            };
            if let Some(reason) = refusal {
                if !reason.is_empty() {
                    eprintln!("dory recurrence rank {rank}: {reason}");
                }
                lane.faulted = true;
                entry.fault = Some(rank);
                refused = true;
            }
        }
        let active: Vec<bool> = entries.iter().map(|entry| entry.fault.is_none()).collect();
        let guessing = entries
            .iter()
            .any(|entry| entry.fault.is_none() && entry.draft);
        let kept = drafter && !guessing && active.contains(&true);
        if active.contains(&true) {
            let mut cores = entry_cores(&mut self.lanes, entries);
            if drafter && guessing {
                StageCore::draft_lanes(&mut cores, &active, true)
                    .map_err(|error| error.to_string())?;
                let rows = cores[0]
                    .read_logits_rows(count)
                    .map_err(|error| error.to_string())?;
                let frame = guesses(entries, &rows);
                write_frames(cuda, self.buffers.guesses, &[frame])?;
                let drafts = self
                    .drafts
                    .as_mut()
                    .ok_or_else(|| format!("recurrence rank {rank} drafts with no drafts link"))?;
                drafts.send(cuda, 0, self.buffers.guesses, FRAME_BYTES)?;
            } else if kept {
                StageCore::keep_lanes(&cores, &active).map_err(|error| error.to_string())?;
            }
            StageCore::forward_lanes(&mut cores, &active, last)
                .map_err(|error| error.to_string())?;
        }
        if !last {
            let frame = if refused {
                let mut copy = [StepEntry::default(); LANE_BATCH];
                copy[..count].copy_from_slice(entries);
                let reply = self.buffers.reply(0);
                write_frames(
                    cuda,
                    reply,
                    &[Frame::Step {
                        count,
                        entries: copy,
                    }],
                )?;
                reply
            } else {
                self.buffers.frame
            };
            self.ring.send(cuda, rank + 1, frame, FRAME_BYTES)?;
            self.ring.send(cuda, rank + 1, state, bytes)?;
            self.ring.send(cuda, rank + 1, input_h, bytes)?;
            if kept {
                StageCore::draft_kept_lanes(&mut entry_cores(&mut self.lanes, entries), &active)
                    .map_err(|error| error.to_string())?;
            }
            return Ok(());
        }
        // Every lane's token is chosen before the first answer goes, so the answers reach rank 0
        // together and the lanes come back to it as one batch.
        let first = entries[0].lane;
        let rows = if active.contains(&true) {
            self.lanes[first]
                .core
                .read_logits_rows(count)
                .map_err(|error| error.to_string())?
        } else {
            Vec::new()
        };
        let mut answers = Vec::with_capacity(count);
        for (index, entry) in entries.iter().enumerate() {
            let lane = &mut self.lanes[entry.lane];
            answers.push(match entry.fault {
                None => {
                    lane.mark(entry.position);
                    lane.sampler.record(entry.fed);
                    Frame::Output {
                        lane: entry.lane,
                        token: lane.sampler.choose(&rows[index], &lane.sampling),
                    }
                }
                Some(rank) => Frame::Fault {
                    lane: entry.lane,
                    rank,
                    output: true,
                },
            });
        }
        // The answers are one message of the frames and one of the rows, both in entry order.
        write_frames(cuda, self.buffers.replies, &answers)?;
        let rows = self.lanes[first].core.logits_row(0);
        self.answers
            .send(cuda, 0, self.buffers.replies, count * FRAME_BYTES)?;
        self.answers
            .send(cuda, 0, rows, count * config.vocab_size * 4)?;
        Ok(())
    }

    /// Passes the fault frame at `frame` on: to the next rank, or from the last rank to rank 0
    /// when its chunk wanted an answer, with the lane's logits row behind it (its contents
    /// unread) so the answer has its one shape.
    fn pass_fault(&mut self, frame: Deviceptr, lane: usize, output: bool) -> Result<(), String> {
        let cuda = self.shared.cuda();
        if !self.last() {
            self.ring.send(cuda, self.rank + 1, frame, FRAME_BYTES)
        } else if output {
            let row = self.lanes[lane].core.logits_row(0);
            self.answers.send(cuda, 0, frame, FRAME_BYTES)?;
            self.answers
                .send(cuda, 0, row, self.shared.config().vocab_size * 4)
        } else {
            Ok(())
        }
    }
}

/// The drafting rank's `Draft` frame for a step: for each entry it ran that asked for a guess,
/// the draft exit's greedy token from the entry's logits row, as the token the lane's next row
/// is fed, with the entry's epoch. The guess is greedy whatever the request samples with: the
/// last rank's choice is what the guessed row is checked against, so a guess only decides how
/// often a row is started early, never a token.
fn guesses(entries: &[StepEntry], rows: &[Vec<f32>]) -> Frame {
    let mut guesses = [StepEntry::default(); LANE_BATCH];
    let mut count = 0;
    for (entry, row) in entries.iter().zip(rows) {
        if entry.fault.is_none() && entry.draft {
            guesses[count] = StepEntry {
                lane: entry.lane,
                position: entry.position + 1,
                fed: argmax(row),
                epoch: entry.epoch,
                ..StepEntry::default()
            };
            count += 1;
        }
    }
    Frame::Draft {
        count,
        entries: guesses,
    }
}

/// The cores of the lanes `entries` name, in lane order, which is the entries' order.
fn entry_cores<'a>(lanes: &'a mut [WorkerLane], entries: &[StepEntry]) -> Vec<&'a mut StageCore> {
    lanes
        .iter_mut()
        .enumerate()
        .filter(|(index, _)| {
            entries
                .binary_search_by_key(index, |entry| entry.lane)
                .is_ok()
        })
        .map(|(_, lane)| &mut lane.core)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frame_round_trips() {
        let sampling = Sampling {
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            repetition_penalty: 1.1,
            presence_penalty: 0.25,
            frequency_penalty: -0.5,
            seed: 0x0123_4567_89ab_cdef,
            prompt_len: 17,
        };
        for frame in [
            Frame::Begin {
                lane: 3,
                prompt: 4200,
                sampling,
            },
            Frame::Rows {
                lane: 0,
                rows: 4096,
                position: 104,
                output: true,
            },
            Frame::Rows {
                lane: 7,
                rows: 1,
                position: 9000,
                output: false,
            },
            step(1),
            step(LANE_BATCH),
            draft(1),
            draft(LANE_BATCH),
            Frame::Output {
                lane: 1,
                token: 131_071,
            },
            Frame::Fault {
                lane: 2,
                rank: 2,
                output: false,
            },
            Frame::Connect,
            Frame::Shutdown,
        ] {
            let words = frame.encode().expect("the frame encodes");
            assert_eq!(Frame::decode(&words).expect("the frame decodes"), frame);
        }
    }

    /// A step of `count` lanes, the last refused by rank 3; the entries past `count` are the
    /// default ones a decoded frame holds.
    fn step(count: usize) -> Frame {
        let mut entries = [StepEntry::default(); LANE_BATCH];
        for (index, entry) in entries[..count].iter_mut().enumerate() {
            *entry = StepEntry {
                lane: index * 9 + 1,
                position: 40_000 + index,
                fed: index as u32 * 1000,
                fault: (index + 1 == count).then_some(3),
                draft: index % 2 == 0,
                rewind: index % 3 == 1,
                epoch: (index * 37 + 250) as u8,
            };
        }
        Frame::Step { count, entries }
    }

    /// The drafting rank's guesses for `count` lanes.
    fn draft(count: usize) -> Frame {
        let mut entries = [StepEntry::default(); LANE_BATCH];
        for (index, entry) in entries[..count].iter_mut().enumerate() {
            *entry = StepEntry {
                lane: (1 << ENTRY_LANE_BITS) - 1 - index,
                position: 7 + index,
                fed: 131_071 - index as u32,
                epoch: u8::MAX - index as u8,
                ..StepEntry::default()
            };
        }
        Frame::Draft { count, entries }
    }

    #[test]
    fn a_step_entry_round_trips_its_flags_and_refuses_a_rank_past_its_bits() {
        let mut words = [0u32; STEP_ENTRY_WORDS];
        let entry = StepEntry {
            lane: 5,
            fault: Some(ENTRY_FAULT_MAX),
            draft: true,
            rewind: true,
            epoch: 0xff,
            ..StepEntry::default()
        };
        entry.encode(&mut words).expect("the entry encodes");
        assert_eq!(StepEntry::decode(&words), entry);
        let error = StepEntry {
            fault: Some(ENTRY_FAULT_MAX + 1),
            ..entry
        }
        .encode(&mut words)
        .expect_err("a rank past the fault bits");
        assert!(error.contains("does not fit"), "{error}");
    }

    #[test]
    fn a_guess_is_greedy_and_only_for_the_rows_that_asked() {
        let mut entries = [StepEntry::default(); 3];
        for (index, entry) in entries.iter_mut().enumerate() {
            *entry = StepEntry {
                lane: index,
                position: 100 + index,
                draft: index != 1,
                epoch: 9,
                ..StepEntry::default()
            };
        }
        let rows = vec![
            vec![0.0, 2.0, 1.0],
            vec![5.0, 0.0, 0.0],
            vec![0.0, 1.0, 3.0],
        ];
        let Frame::Draft { count, entries } = guesses(&entries, &rows) else {
            unreachable!()
        };
        assert_eq!(count, 2);
        assert_eq!(
            (
                entries[0].lane,
                entries[0].position,
                entries[0].fed,
                entries[0].epoch
            ),
            (0, 101, 1, 9)
        );
        assert_eq!(
            (entries[1].lane, entries[1].position, entries[1].fed),
            (2, 103, 2)
        );
        assert!(!entries[0].draft && !entries[0].rewind && entries[0].fault.is_none());
    }

    #[test]
    fn a_step_outside_its_lanes_is_refused() {
        let Frame::Step { entries, .. } = step(2) else {
            unreachable!()
        };
        for count in [0, LANE_BATCH + 1] {
            let error = Frame::Step { count, entries }
                .encode()
                .expect_err("a step of no lanes or too many");
            assert!(error.contains("outside"), "{error}");
        }
        let mut words = step(2).encode().expect("the frame encodes");
        words[2] = (LANE_BATCH + 1) as u32;
        let error = Frame::decode(&words).expect_err("a garbled count");
        assert!(error.contains("outside"), "{error}");
    }

    #[test]
    fn drafting_is_asked_for_or_follows_the_world() {
        assert_eq!(recurrence_draft_lanes(4, None), Ok(DEFAULT_DRAFT_LANES));
        assert_eq!(recurrence_draft_lanes(3, None), Ok(DEFAULT_DRAFT_LANES));
        assert_eq!(
            recurrence_draft_lanes(2, None),
            Ok(0),
            "no rank to draft on"
        );
        assert_eq!(
            recurrence_draft_lanes(4, Some(0)),
            Ok(0),
            "0 turns drafting off"
        );
        assert_eq!(recurrence_draft_lanes(4, Some(3)), Ok(3));
        assert_eq!(recurrence_draft_lanes(2, Some(0)), Ok(0));
        let error = recurrence_draft_lanes(2, Some(1)).expect_err("rank 1 is the last");
        assert!(error.contains("at least 3 ranks"), "{error}");
    }

    #[test]
    fn a_frame_without_the_magic_is_refused() {
        let mut words = Frame::Shutdown.encode().expect("the frame encodes");
        words[0] = 0;
        let error = Frame::decode(&words).expect_err("garbage is not a frame");
        assert!(error.contains("out of step"), "{error}");
    }
}
