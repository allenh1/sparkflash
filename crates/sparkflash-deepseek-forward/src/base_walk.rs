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

//! The DeepSeek base on the shared runtime's [`Walk`] contract.
//!
//! The base's state lives on its device epoch's thread (the forward's buffers are not `Send`),
//! so the walk is a client: each pass is one [`WalkCommand`] to the epoch's walk server
//! ([`crate::device_link::install_walk`]), which runs it through the serving scheduler's own
//! prefill and token-rows bodies. A prompt chunk takes whichever of the bulk chunks and the
//! exact rows steps the scheduler would pick for it, and its last row is headed on the device;
//! a token pass is the plain step's rows, every row headed. The head answers those picks, so
//! the walk keeps no logits. The conformance fixture in `sparkflash-cli` opens the epoch.

use std::sync::mpsc::{self, Sender};

use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::gumbel::GumbelSampler;
use sparkflash_model::ledger::State;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, HeadRow, LaneState, Limits, NoVision, Pass, PassKind, PassMask,
    Region, Residual, Segment, Segments, StateKind, StateLayout, Walk, WalkCapabilities,
    check_checkpoint, check_rewind,
};

use crate::device_link::{WalkAnswer, WalkCommand, WalkDescription, WalkRequest};

/// The code the walk's refusals carry.
pub const CODE: &str = "deepseek_base";

/// The widest prompt pass the walk takes: the scheduler's idle bulk tick, a chunk-policy size.
pub const PROMPT_ROWS: usize = 4096;

/// The end of the walk server's channel.
#[derive(Clone)]
struct Link(Sender<WalkRequest>);

impl Link {
    fn call(&self, command: WalkCommand) -> Result<WalkAnswer, BackendError> {
        let (reply, answer) = mpsc::channel();
        self.0
            .send(WalkRequest { command, reply })
            .map_err(|_| BackendError::new(CODE, "the walk server has ended"))?;
        answer
            .recv()
            .map_err(|_| BackendError::new(CODE, "the walk server ended before its answer"))?
            .map_err(|error| BackendError::new(CODE, error))
    }

    fn picks(&self, command: WalkCommand) -> Result<Vec<u32>, BackendError> {
        match self.call(command)? {
            WalkAnswer::Picks(picks) => Ok(picks),
            other => Err(unexpected(&other)),
        }
    }
}

fn unexpected(answer: &WalkAnswer) -> BackendError {
    BackendError::new(CODE, format!("the walk server answered {answer:?}"))
}

/// The base's walk: a client of one device epoch's walk server.
pub struct BaseWalk {
    link: Link,
    description: WalkDescription,
    layout: StateLayout,
    drafter: Absent<&'static str>,
}

impl BaseWalk {
    /// A walk over the epoch serving `sender`'s commands.
    ///
    /// # Errors
    /// The server has ended, or refuses to describe the epoch.
    pub fn new(sender: Sender<WalkRequest>) -> Result<Self, BackendError> {
        let link = Link(sender);
        let description = match link.call(WalkCommand::Describe)? {
            WalkAnswer::Described(description) => description,
            other => return Err(unexpected(&other)),
        };
        Ok(Self {
            link,
            description,
            layout: StateLayout::new(vec![Region::Positional {
                name: "kv",
                bytes_per_row: description.row_bytes,
            }]),
            drafter: Absent("the DeepSeek base walk drafts nothing yet"),
        })
    }

    /// What the server said about its epoch.
    #[must_use]
    pub const fn description(&self) -> WalkDescription {
        self.description
    }
}

/// One sequence: a lane of the epoch.
pub struct BaseSeq {
    link: Link,
    lane: usize,
    pos: usize,
    sampler: GumbelSampler,
    /// Why the request's sampling cannot be served, raised at its next pass.
    refused: Option<String>,
}

impl Drop for BaseSeq {
    fn drop(&mut self) {
        // A server that has ended holds no lane.
        let _ = self.link.call(WalkCommand::Release { lane: self.lane });
    }
}

impl LaneState for BaseSeq {
    fn position(&self) -> usize {
        self.pos
    }

    // Every state is positional: the counters move back, so there is nothing to save.
    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        check_checkpoint(CODE, at, self.pos)
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind(CODE, to, self.pos)?;
        self.link.call(WalkCommand::Rewind {
            lane: self.lane,
            to,
        })?;
        self.pos = to;
        Ok(())
    }

    fn room(&self, positions: usize, _reset: bool) -> bool {
        matches!(
            self.link.call(WalkCommand::Room {
                lane: self.lane,
                positions,
            }),
            Ok(WalkAnswer::Room(true))
        )
    }
}

/// A pass's rows and the picks the device headed.
pub struct BaseHidden {
    rows: usize,
    picks: Vec<(usize, u32)>,
}

impl Residual for BaseHidden {
    fn rows(&self) -> usize {
        self.rows
    }
}

impl Walk for BaseWalk {
    type Seq = BaseSeq;
    type Hidden = BaseHidden;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::TOKEN_ROWS.union(PassMask::BULK),
        segments: Segments::Many,
        state: StateKind::Positional,
        // Word for word the arm's rows in `sparkflash_model::ARM_LEDGERS`; the generated
        // profile test holds them equal.
        compute: ComputeDeclaration {
            prompt: State::Native("bulk chunks on the FP8 x FP4/FP8 MMAs"),
            token: State::Native("one-to-eight-row steps on the MMA (opt_gemv_mma)"),
            stored: State::Native("FP8 projections, FP4 and MXFP8 experts in their stored formats"),
            rewind: State::Native("positional state: a changed turn moves its counters back"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<BaseSeq, BackendError> {
        let lane = match self.link.call(WalkCommand::Open)? {
            WalkAnswer::Opened(lane) => lane,
            other => return Err(unexpected(&other)),
        };
        Ok(BaseSeq {
            link: self.link.clone(),
            lane,
            pos: 0,
            sampler: GumbelSampler::GREEDY,
            refused: None,
        })
    }

    fn limits(&self) -> Limits {
        Limits::new(PROMPT_ROWS, self.description.context).with_members(self.description.lanes)
    }

    fn drafter(&self) -> &Absent<&'static str> {
        &self.drafter
    }

    fn set_sampling(&self, seq: &mut BaseSeq, sampling: Sampling) {
        seq.refused = (sampling.repetition_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0)
            .then(|| {
                "repetition, presence and frequency penalties are not served on the DeepSeek \
                 base walk"
                    .to_owned()
            });
        seq.sampler = if sampling.temperature <= 0.0 {
            GumbelSampler::GREEDY
        } else {
            GumbelSampler {
                temperature: sampling.temperature,
                top_p: sampling.top_p.clamp(0.0, 1.0),
                top_k: u32::try_from(sampling.top_k).unwrap_or(0),
                seed: sampling.seed,
            }
        };
    }

    fn walk(
        &self,
        segments: &mut [Segment<'_, BaseSeq>],
        pass: Pass<'_>,
    ) -> Result<BaseHidden, BackendError> {
        if let Some(refused) = segments
            .iter()
            .find_map(|segment| segment.seq.refused.clone())
        {
            return Err(BackendError::new(CODE, refused));
        }
        let (rows, picks) = match pass.kind {
            PassKind::Bulk { .. } => {
                let [segment] = segments else {
                    return Err(BackendError::new(
                        CODE,
                        format!(
                            "a prompt chunk is one sequence's, and this pass holds {}",
                            segments.len()
                        ),
                    ));
                };
                let n = segment.rows.len();
                let picks = self.link.picks(WalkCommand::Chunk {
                    lane: segment.seq.lane,
                    rows: segment.rows.to_vec(),
                    sampler: segment.seq.sampler,
                    more: pass.prompt_end.is_some_and(|rest| rest > 0),
                })?;
                segment.seq.pos += n;
                (n, picks.into_iter().map(|pick| (n - 1, pick)).collect())
            }
            PassKind::TokenRows => {
                let mut rows = Vec::with_capacity(segments.len());
                for segment in segments.iter() {
                    let [token] = segment.rows else {
                        return Err(BackendError::new(
                            CODE,
                            format!(
                                "a token pass is one row a sequence, and a segment holds {}",
                                segment.rows.len()
                            ),
                        ));
                    };
                    rows.push((segment.seq.lane, *token, segment.seq.sampler));
                }
                let picks = self.link.picks(WalkCommand::Tokens { rows })?;
                for segment in segments.iter_mut() {
                    segment.seq.pos += 1;
                }
                (segments.len(), picks.into_iter().enumerate().collect())
            }
            kind => {
                return Err(BackendError::new(
                    CODE,
                    format!(
                        "the DeepSeek base walk runs a prompt chunk and a token row a sequence, \
                         not a {kind:?} pass"
                    ),
                ));
            }
        };
        if let Some(hook) = pass.hook {
            hook.after_layers(&sparkflash_runtime::Hidden::new(rows))?;
        }
        Ok(BaseHidden { rows, picks })
    }

    fn head(
        &self,
        hidden: &mut BaseHidden,
        rows: &mut [HeadRow<'_, BaseSeq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        rows.iter()
            .map(|row| {
                let pick = hidden
                    .picks
                    .iter()
                    .find(|(at, _)| *at == row.row)
                    .map(|(_, pick)| *pick)
                    .ok_or_else(|| {
                        BackendError::new(
                            CODE,
                            format!(
                                "row {} of {} was not headed: the base heads a chunk's last row \
                                 and every token row",
                                row.row, hidden.rows
                            ),
                        )
                    })?;
                Ok(StepOutput {
                    next_token: Some(pick),
                    ..StepOutput::default()
                })
            })
            .collect()
    }

    /// A rewind moves the position counters, which only a resume boundary leaves whole.
    fn rewind_point(&self, prompt: usize) -> usize {
        prompt - prompt % self.description.resume_step.max(1)
    }

    fn rewind_slack(&self) -> usize {
        self.description.slack
    }
}
