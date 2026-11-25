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

//! The walk server's operations ([crate::device_link::WalkCommand]) on the serving scheduler.
//!
//! Each operation is the scheduler's own body for one piece of a tick: a prompt chunk runs
//! [Batch::prefill_bulk] or [Batch::prefill_exact] exactly as a placed prompt does (whichever
//! [Batch::bulk_from] picks), and a token pass runs [Batch::token_rows], the plain step's rows.
//! The runtime owns the reuse and the chunk sizes, so the walk mode turns the scheduler's own
//! prefix reuse and decoder skip off; the sequences it keeps hold no request bookkeeping.

use super::*;
use crate::device_link::{WalkAnswer, WalkCommand, WalkDescription};

impl Batch<'_, '_, '_> {
    /// Puts the scheduler in walk mode: the runtime chooses the reuse, so the scheduler keeps no
    /// prefix of its own and every prompt row runs every layer.
    pub(in super::super) fn walk_mode(&mut self) {
        self.reuse = false;
        self.decoder_skip = false;
    }

    /// Runs one walk command; a refusal names what the epoch cannot do.
    ///
    /// # Errors
    /// The command's refusal, or a device failure.
    pub(in super::super) fn walk_command(
        &mut self,
        command: WalkCommand,
    ) -> Result<WalkAnswer, String> {
        match command {
            WalkCommand::Describe => self.walk_describe().map(WalkAnswer::Described),
            WalkCommand::Open => self.walk_open().map(WalkAnswer::Opened),
            WalkCommand::Release { lane } => self.walk_release(lane).map(|()| WalkAnswer::Done),
            WalkCommand::Rewind { lane, to } => {
                self.walk_rewind(lane, to).map(|()| WalkAnswer::Done)
            }
            WalkCommand::Chunk {
                lane,
                rows,
                sampler,
                more,
            } => self
                .walk_chunk(lane, &rows, sampler, more)
                .map(|pick| WalkAnswer::Picks(vec![pick])),
            WalkCommand::Tokens { rows } => self.walk_tokens(&rows).map(WalkAnswer::Picks),
            WalkCommand::Room { lane, positions } => {
                self.walk_room(lane, positions).map(WalkAnswer::Room)
            }
        }
    }

    fn walk_describe(&self) -> Result<WalkDescription, String> {
        // The walk declares positional state: a recurrent block's saved points would need the
        // runtime's checkpoint, which this server does not keep.
        if self.device.has_recurrent() {
            return Err(
                "the walk server keeps positional state only, and this stack has a recurrent block"
                    .to_owned(),
            );
        }
        Ok(WalkDescription {
            lanes: self.lanes.len(),
            context: self.room,
            resume_step: self.device.resume_step(),
            exchange: self.exchange.is_some(),
            slack: super::super::lanes::SLIDE_KEEP
                .saturating_sub(2 * sparkflash_kernels::flash_mla_csa2_seq_launch::SWA_WINDOW),
            row_bytes: self.device.lane_row_bytes(),
            bulk_rows: if self.inputs.bulk.is_some() && self.exchange.is_some() {
                bulk_min()
            } else {
                0
            },
        })
    }

    /// The scheduler's index of the sequence on `lane`.
    fn walk_seq(&self, lane: usize) -> Result<usize, String> {
        let id = self
            .lanes
            .get(lane)
            .and_then(|held| held.seq)
            .ok_or_else(|| format!("walk: lane {lane} holds no sequence"))?;
        self.seqs
            .iter()
            .position(|seq| seq.id == id)
            .ok_or_else(|| format!("walk: lane {lane}'s sequence {id} is not on the scheduler"))
    }

    fn walk_open(&mut self) -> Result<usize, String> {
        let lane = (0..self.lanes.len())
            .find(|&lane| self.lanes[lane].seq.is_none())
            .ok_or_else(|| {
                format!(
                    "walk: every one of the epoch's {} lanes holds a sequence",
                    self.lanes.len()
                )
            })?;
        self.forget(lane)?;
        self.ticks += 1;
        let id = self.ticks;
        let mut seq = Seq::waiting(
            id,
            Vec::new(),
            &[],
            0,
            usize::MAX,
            Sampler::GREEDY,
            None,
            &[],
            true,
            depth_policy(self.k),
        );
        seq.lane = Some(lane);
        seq.phase = Phase::Decode;
        self.seqs.push(seq);
        self.lanes[lane].seq = Some(id);
        self.enter(lane)?;
        self.device.rewind_rows(0);
        Ok(lane)
    }

    fn walk_release(&mut self, lane: usize) -> Result<(), String> {
        let index = self.walk_seq(lane)?;
        self.drop_reads(self.seqs[index].id)?;
        self.seqs.remove(index);
        self.lanes[lane].seq = None;
        self.lanes[lane].used = self.ticks;
        self.forget(lane)
    }

    /// Moves the sequence on `lane` back to `to`: the start, or a resume boundary its windows
    /// still hold, as a placed prompt resumes ([Batch::place]).
    fn walk_rewind(&mut self, lane: usize, to: usize) -> Result<(), String> {
        let index = self.walk_seq(lane)?;
        let pos = self.seqs[index].pos;
        if to > pos {
            return Err(format!("walk: a rewind to {to} is past position {pos}"));
        }
        if to == pos {
            return Ok(());
        }
        if to == 0 {
            self.forget(lane)?;
        } else if self.device.resume_boundary(to) != to {
            return Err(format!(
                "walk: a rewind to {to} is not a resume boundary (a multiple of {})",
                self.device.resume_step()
            ));
        } else if slid_past(self.device.lane_floor(lane), to) {
            return Err(format!(
                "walk: lane {lane}'s windows slid past a rewind to {to}"
            ));
        }
        self.enter(lane)?;
        self.device.rewind_rows(to);
        let seq = &mut self.seqs[index];
        seq.tokens.truncate(to);
        seq.prompt_len = to;
        seq.pos = to;
        seq.walk_bulk_end = None;
        Ok(())
    }

    /// Runs `rows` as prompt rows of the sequence on `lane` and answers the last row's pick;
    /// `more` keeps the chunk's packed-key run open for the prompt's next chunk.
    fn walk_chunk(
        &mut self,
        lane: usize,
        rows: &[u32],
        sampler: Sampler,
        more: bool,
    ) -> Result<u32, String> {
        let index = self.walk_seq(lane)?;
        if rows.is_empty() {
            return Err("walk: an empty prompt chunk".to_owned());
        }
        in_vocabulary(rows)?;
        self.servable(&sampler)?;
        let p = self.seqs[index].pos;
        let target = p + rows.len();
        if target > self.room {
            return Err(format!(
                "a prompt of {target} tokens does not fit the {}-token device context",
                self.room
            ));
        }
        let bulk = self.bulk_from(target, p);
        let end = target + draft::draft_block() + self.k + 2;
        if !self.device.lane_fits(end, self.tick_rows(bulk, false)) || !self.grow(lane, end)? {
            return Err(format!(
                "a sequence of {target} tokens does not fit the device's lane memory (SPARKFLASH_LANE_MEMORY_MB)"
            ));
        }
        let seq = &mut self.seqs[index];
        seq.sampler = sampler;
        seq.tokens.truncate(p);
        seq.tokens.extend_from_slice(rows);
        seq.prompt_len = target;
        // A bulk chunk that does not continue the same prompt's last one starts its packed-key
        // run, as a placed prompt's first chunk does.
        seq.reused = (seq.walk_bulk_end != Some(p)).then_some(p);
        self.enter(lane)?;
        self.device.rewind_rows(p);
        let mut pick = None;
        while self.seqs[index].pos < target {
            let first = if bulk {
                self.prefill_bulk(index, target, false)?
            } else {
                self.prefill_exact(index, target, false)?
            };
            pick = first.or(pick);
        }
        self.drop_reads(self.seqs[index].id)?;
        self.device.cuda.synchronize()?;
        self.seqs[index].walk_bulk_end = (bulk && more).then_some(target);
        pick.ok_or_else(|| "walk: the chunk's last row was not headed".to_owned())
    }

    /// One token row on each named lane, in one plain step; answers each row's pick in order.
    fn walk_tokens(&mut self, rows: &[(usize, u32, Sampler)]) -> Result<Vec<u32>, String> {
        if rows.is_empty() || rows.len() > R_MAX {
            return Err(format!(
                "walk: a token pass holds 1 to {R_MAX} rows, not {}",
                rows.len()
            ));
        }
        let mut group = Vec::with_capacity(rows.len());
        for &(lane, token, sampler) in rows {
            let index = self.walk_seq(lane)?;
            if group.contains(&index) {
                return Err(format!("walk: lane {lane} appears twice in one token pass"));
            }
            in_vocabulary(&[token])?;
            self.servable(&sampler)?;
            let next = self.seqs[index].pos + 1;
            if next > self.room || !self.grow(lane, next + draft::draft_block() + self.k + 2)? {
                return Err(format!(
                    "a sequence of {next} tokens does not fit the device's lane memory or its {}-token context",
                    self.room
                ));
            }
            group.push(index);
        }
        for (&index, &(_, token, sampler)) in group.iter().zip(rows) {
            let seq = &mut self.seqs[index];
            seq.sampler = sampler;
            seq.tokens.truncate(seq.pos);
            seq.tokens.push(token);
            seq.prompt_len = seq.tokens.len();
        }
        let (_, picks) = self.token_rows(&group, &mut |_, _| Ok(()))?;
        for &index in &group {
            let seq = &mut self.seqs[index];
            seq.pos += 1;
            seq.walk_bulk_end = None;
        }
        Ok(picks.into_iter().map(|(id, _)| id).collect())
    }

    fn walk_room(&self, lane: usize, positions: usize) -> Result<bool, String> {
        self.walk_seq(lane)?;
        let end = positions + draft::draft_block() + self.k + 2;
        Ok(positions <= self.room && self.device.lane_fits(end, self.tick_rows(true, false)))
    }
}
