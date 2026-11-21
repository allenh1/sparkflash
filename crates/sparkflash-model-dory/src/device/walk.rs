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

//! The single-process Dory arm as a [`Walk`].
//!
//! The runtime on the shared tick engine drives it: the prompt parts, the history, prefix
//! reuse, the rewind point, the lanes and the ledger are the runtime's, and this module supplies
//! the two things only the arm knows, the rows one pass runs and the token its head chooses.
//!
//! Every row is computed by the arm's own bodies ([`StageCore::embed`],
//! [`StageCore::forward`], [`StageCore::forward_lanes`] and [`super::head_rows`]), so a lane's
//! bits are the bits the arm's own prompt and decode loops gave it: a prompt chunk is one pass of
//! up to [`PREFILL_BATCH`] rows, and a step is one row a lane for at most [`LANE_BATCH`] lanes.
//!
//! Two properties of the arm's prompt kernels shape a pass, because a reused prompt forwards
//! only the ids it added and those rows must have the bits the whole prompt's chunks gave them.
//! A prompt projection of eight rows or fewer takes the token GEMV and a wider chunk the
//! tensor-core tile, whose per-row bits differ, so a per-loop pass of [`TOKEN_ROWS`] rows or
//! fewer runs [`PREFILL_Q`] rows, the extra ones repeats of its own tokens. And a prompt
//! attention block's key window is its first row's, so a row's bits depend on which query tile
//! it sits in, which is why a pass that does not start on a [`PREFILL_Q`] boundary re-runs the
//! rows of its first tile from the ids the sequence holds ([`DorySequence::forwarded`]) and hands
//! the runtime its own rows, at [`Hidden::first`]. A cache row the extra rows wrote sits past the
//! pass's last row, so the rows after the pass overwrite it before any attention reads it.

use std::sync::Arc;

use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::ledger::State;
use sparkflash_model::sampler::Sampler;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, HeadInput, HeadRow, Hidden, LaneState, Limits, NoVision, Pass,
    PassKind, PassMask, Region, Segment, Segments, StateKind, StateLayout, Walk, WalkCapabilities,
};

use crate::config::KvCacheMode;

use super::{
    DEVICE, DeviceShared, LANE_BATCH, OwnedScratch, PREFILL_BATCH, PREFILL_Q, Scratch, StageCore,
    TOKEN_ROWS, head_rows,
};

/// One Dory sequence: a stage's caches and position, and the request's sampler.
///
/// Every position is addressed by its own row of the KV caches, so a rewind is a truncation
/// and the arm has no recurrent region to checkpoint.
pub(crate) struct DorySequence {
    core: StageCore,
    sampler: Sampler,
    sampling: Sampling,
    /// The ids this sequence forwarded and still holds, one a position: a pass that does not
    /// start on a query tile re-runs the few rows before it, so every row keeps the tile
    /// boundaries the prompt's own chunks gave it.
    forwarded: Vec<u32>,
}

impl LaneState for DorySequence {
    fn position(&self) -> usize {
        self.core.position()
    }

    fn checkpoint(&mut self, at: usize) -> Result<(), BackendError> {
        if at != self.core.position() {
            return Err(BackendError::new(
                DEVICE,
                format!(
                    "the checkpoint is asked for position {at}, the sequence is at {}",
                    self.core.position()
                ),
            ));
        }
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        self.core.rewind(to)?;
        self.sampler.truncate(to);
        self.forwarded.truncate(to);
        Ok(())
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        let _ = reset;
        positions <= self.core.shared().context
    }
}

/// The single-process Dory arm, exposed to the shared runtime as a [`Walk`].
pub(crate) struct DoryWalk {
    shared: Arc<DeviceShared>,
    layout: StateLayout,
    limits: Limits,
    /// The row buffers every lane of this walk steps through: one set, so a pass over several
    /// lanes is one batched launch, and the runtime steps the lanes one pass at a time.
    scratch: Arc<OwnedScratch>,
    /// Whether a prompt runs loop by loop, a whole chunk at once, or one token at a time.
    per_loop: bool,
    drafter: Absent<&'static str>,
}

impl DoryWalk {
    /// A walk over `shared`'s weights, with the row buffers every lane it opens steps through.
    ///
    /// # Errors
    /// A scratch allocation the device refuses.
    pub(crate) fn new(shared: Arc<DeviceShared>) -> Result<Self, BackendError> {
        let scratch = Arc::new(OwnedScratch {
            buffers: Scratch::allocate(&shared)
                .map_err(|error| BackendError::new(DEVICE, error))?,
            shared: Arc::clone(&shared),
        });
        let per_loop = shared.config.recurrent_kv_cache_mode == KvCacheMode::PerLoop;
        let limits = walk_limits(per_loop, shared.context);
        Ok(Self {
            layout: layout_of(&shared),
            shared,
            limits,
            scratch,
            per_loop,
            // The loop-1 exit lives on a recurrence group's drafting rank, where a rank holds
            // the output suffix beside its loops; a single process has no second exit to draft
            // from, so the arm publishes none rather than a chain it cannot verify.
            drafter: Absent("the single-process Dory arm has no loop-1 exit"),
        })
    }
}

/// The limits the walk states at `context`.
///
/// A per-loop prompt takes a whole chunk a pass; last-loop's overwritten cache makes a row
/// depend on the tokens after it, so that mode keeps the token-at-a-time order.
fn walk_limits(per_loop: bool, context: usize) -> Limits {
    let rows = if per_loop { PREFILL_BATCH } else { 1 };
    Limits::new(rows, context).with_members(LANE_BATCH)
}

/// The arm's state, as the runtime needs to see it: one positional region a block's KV caches.
///
/// A block that runs every loop keeps one cache a loop, and a sliding block's cache is a ring
/// of its window and one prompt chunk; both are addressed by position, so a rewind truncates.
fn layout_of(shared: &DeviceShared) -> StateLayout {
    let config = &shared.config;
    let width = config.num_key_value_heads * config.head_dim;
    let mut regions = Vec::with_capacity(config.blocks.len());
    for block in &config.blocks {
        let slots = (0..block.cache_slots)
            .filter(|slot| shared.stage.keeps_slot(block, *slot))
            .count();
        if slots == 0 {
            continue;
        }
        regions.push(Region::Positional {
            name: "kv",
            // The keys and the values, BF16 a value, of every slot the block keeps.
            bytes_per_row: slots * width * 2 * 2,
        });
    }
    StateLayout::new(regions)
}

/// The tokens a narrow per-loop prompt pass runs: its own rows, then repeats of them up to
/// [`PREFILL_Q`] rows, so the pass takes the tensor-core tile a wider chunk takes.
///
/// The extra rows are a lane's own tokens, so nothing is out of range, and their cache rows sit
/// past the pass's last row: the rows after this pass overwrite them before any attention reads
/// them.
fn padded(tokens: &[u32], rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|index| tokens[index % tokens.len()])
        .collect()
}

impl Walk for DoryWalk {
    type Seq = DorySequence;
    type Hidden = Hidden;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::TOKEN_ROWS.union(PassMask::BULK),
        segments: Segments::Many,
        state: StateKind::Positional,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same one the profile check that refuses `decode.tensor_cores = true` reads.
        compute: ComputeDeclaration {
            prompt: State::Native(
                "prompt projections over eight rows on dory_mma_wide; attention on the MMA \
                 instances",
            ),
            token: State::Off("warp-dot row kernels; decode.tensor_cores=true is refused"),
            stored: State::Native("BF16 weights and a BF16 KV cache throughout"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        let core = StageCore::with_scratch(Arc::clone(&self.shared), Arc::clone(&self.scratch))
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(DorySequence {
            core,
            sampler: Sampler::new(),
            sampling: Sampling::default(),
            forwarded: Vec::new(),
        })
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn set_sampling(&self, seq: &mut Self::Seq, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the row every step
        // copies back, so a non-greedy request is served rather than approximated.
        seq.sampling = sampling;
        seq.sampler.reset();
    }

    /// Dory's template opens a thinking reply with ` thinking\n` and writes a past reply back as
    /// ` thinking</think>...`, so the next turn keeps the prompt only up to its last id.
    fn rewind_point(&self, prompt: usize) -> usize {
        prompt.saturating_sub(1)
    }

    /// A sliding block's cache holds its window and one prompt chunk, so the rows a rewind to
    /// the prompt's end reads survive a reply of at most [`PREFILL_BATCH`] positions.
    fn rewind_slack(&self) -> usize {
        PREFILL_BATCH
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
        if rows == 0 {
            return Err(BackendError::new(DEVICE, "a pass of no rows"));
        }
        // Where the pass's own rows start in the residual buffer: a prompt chunk that re-ran
        // the rows of its first tile leaves them in front of its own rows.
        let mut first = 0;
        match kind {
            PassKind::Bulk { .. } => {
                let [segment] = segments else {
                    return Err(BackendError::new(
                        DEVICE,
                        format!(
                            "a prompt chunk is one sequence's rows, not {} sequences'",
                            segments.len()
                        ),
                    ));
                };
                let tokens = segment.rows;
                if tokens.len() != rows {
                    return Err(BackendError::new(
                        DEVICE,
                        format!("a pass of {rows} rows names {} of them", tokens.len()),
                    ));
                }
                let seq = &mut *segment.seq;
                // A prompt row's bits are the tile the prompt's own chunks gave it, so a pass
                // that does not start on a query tile re-runs the few rows of the tile before
                // it, and a pass too narrow for the wide tile is padded to one.
                let start = if self.per_loop {
                    seq.core.position() % PREFILL_Q
                } else {
                    0
                };
                first = start;
                let kept = seq.forwarded.len();
                if kept < start || seq.core.position() != kept {
                    return Err(BackendError::new(
                        DEVICE,
                        format!(
                            "the pass starts at {} and the sequence holds {kept} ids",
                            seq.core.position()
                        ),
                    ));
                }
                let mut run: Vec<u32> = Vec::with_capacity(start + tokens.len() + PREFILL_Q);
                run.extend_from_slice(&seq.forwarded[kept - start..]);
                run.extend_from_slice(tokens);
                let real = run.len();
                let from = seq.core.position() - start;
                // A narrow pass takes the wide tile, as the arm's own prompt loop's chunks did;
                // the room left in the context caps the padding, and with less than a wide batch
                // left the pass runs its rows alone.
                let pad = if self.per_loop && real <= TOKEN_ROWS {
                    let room = seq.core.shared().context.saturating_sub(from);
                    Some(PREFILL_Q.min(room)).filter(|pad| *pad > real)
                } else {
                    None
                };
                let (batch, walked) = match pad {
                    Some(pad) => (padded(&run, pad), pad),
                    None => (run.clone(), real),
                };
                seq.core.rewind(from)?;
                seq.core.embed(&batch)?;
                seq.core.forward(walked, false)?;
                seq.core.rewind(seq.core.position() - walked + real)?;
                seq.forwarded.truncate(from);
                seq.forwarded.extend_from_slice(&run[..real]);
            }
            PassKind::TokenRows => {
                let mut tokens = Vec::with_capacity(segments.len());
                for segment in segments.iter() {
                    let [token] = segment.rows else {
                        return Err(BackendError::new(
                            DEVICE,
                            format!(
                                "a token pass is one row a sequence, and this segment holds {}",
                                segment.rows.len()
                            ),
                        ));
                    };
                    tokens.push(*token);
                }
                let mut lanes: Vec<&mut StageCore> = segments
                    .iter_mut()
                    .map(|segment| &mut segment.seq.core)
                    .collect();
                StageCore::embed_lanes(&lanes, &tokens)?;
                let active = vec![true; lanes.len()];
                StageCore::forward_lanes(&mut lanes, &active, false)?;
                for (segment, token) in segments.iter_mut().zip(&tokens) {
                    segment.seq.forwarded.push(*token);
                }
            }
            PassKind::Paired { .. } | PassKind::Verify { .. } => {
                return Err(BackendError::new(
                    DEVICE,
                    format!(
                        "the Dory device arm runs a prompt chunk and a token row a sequence, \
                         not a {kind:?} pass"
                    ),
                ));
            }
        }
        let hidden = Hidden::at(first, rows);
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
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            if row.row >= hidden.rows {
                return Err(BackendError::new(
                    DEVICE,
                    format!("the head is asked for row {} of {}", row.row, hidden.rows),
                ));
            }
            let seq = &mut *row.seq;
            let index = hidden.first + row.row;
            // The row's token is recorded before the choice, as the arm's own session did: a
            // prompt's record resets and fills the history the penalty check reads.
            head_rows(
                &seq.core.shared,
                &seq.core.scratch,
                hidden.first + hidden.rows,
                index..index + 1,
            )?;
            let logits = seq.core.read_logits()?;
            match &row.input {
                HeadInput::Prompt(tokens) => {
                    seq.sampler.reset();
                    seq.sampler.record_prompt(tokens);
                }
                HeadInput::Token(token) => seq.sampler.record(*token),
                HeadInput::Verify { rows, row: index } => {
                    seq.sampler
                        .record(rows.get(*index).copied().unwrap_or_default());
                }
            }
            let chosen = seq.sampler.choose(&logits, &seq.sampling);
            steps.push(StepOutput {
                next_token: seq.core.shared.config.next_token(chosen),
                logits,
                draft: Vec::new(),
            });
        }
        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{Fixture, WalkFixture};

    use super::*;
    use crate::DoryDeviceBackend;

    /// The Dory fixture: the crate's tiny ring checkpoint, written into a temporary directory
    /// the walk keeps alive. A check that boots it needs a CUDA device and the arm's cubins.
    impl WalkFixture for DoryWalk {
        fn fixture(world: usize) -> Result<Fixture<Self>, BackendError> {
            if world > 1 {
                return Err(BackendError::new(
                    "group",
                    "the Dory device arm is one process; TP 2 and TP 4 need a rank ring",
                ));
            }
            if !DoryDeviceBackend::available() {
                return Err(BackendError::new(
                    "fixture",
                    "the Dory conformance fixture needs a build with the arm's cubins",
                ));
            }
            let directory = tempfile::tempdir()
                .map_err(|error| BackendError::new("fixture", error.to_string()))?;
            crate::fixture::write(
                directory.path(),
                &crate::config::DoryConfig::tiny_ring_json(),
            )
            .map_err(|error| BackendError::new("fixture", error))?;
            let backend = DoryDeviceBackend::open(directory.path(), 5_000)
                .map_err(|error| BackendError::new("fixture", error))?;
            let walk = backend.walk_arm()?;
            let bytes = backend.weight_bytes() as u64;
            Ok(Fixture::new(
                walk,
                "the Dory tiny ring checkpoint",
                bytes,
                bytes,
                vec![sparkflash_runtime::FormatRule::stored("BF16")],
                Some(Box::new(directory)),
            ))
        }

        /// The single-process arm serves on the tick engine, so its generated checks drive it.
        fn tick_engine() -> bool {
            true
        }
    }

    /// The arm's state is position-addressed: a rewind truncates and needs no checkpoint.
    #[test]
    fn the_layout_is_positional() {
        let layout = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        assert_eq!(layout.kind(), StateKind::Positional);
        assert!(!layout.needs_checkpoint());
    }

    /// A narrow prompt pass runs the wide tile with its own rows repeated, in order and only
    /// its own ids, so nothing the pass reads is outside the prompt.
    #[test]
    fn a_narrow_pass_pads_with_its_own_rows() {
        assert_eq!(padded(&[4, 5, 6], 6), [4, 5, 6, 4, 5, 6]);
        let full = padded(&[4, 5, 6], PREFILL_Q);
        assert_eq!(full.len(), PREFILL_Q);
        assert_eq!(full[..9], [4, 5, 6, 4, 5, 6, 4, 5, 6]);
        assert_eq!(padded(&[9], PREFILL_Q), vec![9; PREFILL_Q]);
    }

    /// A row a narrow pass rewrites has the bits its wide chunk gave it, which is what lets a
    /// reused prompt forward only the ids it added.
    #[test]
    #[ignore = "boots the arm's fixture on a device"]
    fn a_rewritten_row_keeps_the_bits_of_its_wide_chunk() {
        use sparkflash_runtime::{Pass, Segment};

        let fixture = <DoryWalk as WalkFixture>::fixture(1).expect("the Dory fixture");
        let walk = fixture.walk;
        let prompt: Vec<u32> = (1..=40).collect();
        let mut whole = walk.open().expect("a sequence");
        let mut hidden = walk
            .walk(
                &mut [Segment::new(&mut whole, &prompt)],
                Pass::bulk(prompt.len()).with_prompt_end(0),
            )
            .expect("the whole prompt");
        let last = hidden.rows - 1;
        let cold = walk
            .head(
                &mut hidden,
                &mut [HeadRow::prompt(&mut whole, last, &prompt)],
            )
            .expect("the whole prompt's last row");
        let mut rewound = walk.open().expect("a second sequence");
        let (head, tail) = prompt.split_at(prompt.len() - 1);
        walk.walk(
            &mut [Segment::new(&mut rewound, head)],
            Pass::bulk(head.len()).with_prompt_end(tail.len()),
        )
        .expect("the prompt but its last id");
        let mut hidden = walk
            .walk(
                &mut [Segment::new(&mut rewound, tail)],
                Pass::bulk(tail.len()).with_prompt_end(0),
            )
            .expect("the last id on its own");
        let last = hidden.rows - 1;
        let warm = walk
            .head(
                &mut hidden,
                &mut [HeadRow::prompt(&mut rewound, last, &prompt)],
            )
            .expect("the rewritten last row");
        let bits = |row: &StepOutput| -> Vec<u32> {
            row.logits.iter().map(|value| value.to_bits()).collect()
        };
        assert_eq!(bits(&cold[0]), bits(&warm[0]));
    }

    /// A row's bits do not depend on where the runtime cut the prompt's chunks, which is what
    /// lets a reused prompt forward only the ids it added.
    #[test]
    #[ignore = "boots the arm's fixture on a device"]
    fn a_chunk_boundary_does_not_move_a_rows_bits() {
        use sparkflash_runtime::{Pass, Segment};

        let fixture = <DoryWalk as WalkFixture>::fixture(1).expect("the Dory fixture");
        let walk = fixture.walk;
        let prompt: Vec<u32> = (1..=40).collect();
        let logits = |pieces: &[&[u32]]| -> Vec<u32> {
            let mut seq = walk.open().expect("a sequence");
            let mut hidden = Hidden::new(0);
            let mut all: Vec<u32> = Vec::new();
            for (index, piece) in pieces.iter().enumerate() {
                let last = index + 1 == pieces.len();
                all.extend_from_slice(piece);
                hidden = walk
                    .walk(
                        &mut [Segment::new(&mut seq, piece)],
                        Pass::bulk(piece.len()).with_prompt_end(if last { 0 } else { 1 }),
                    )
                    .expect("a pass");
            }
            let last = hidden.rows - 1;
            let step = walk
                .head(&mut hidden, &mut [HeadRow::prompt(&mut seq, last, &all)])
                .expect("the head");
            step[0].logits.iter().map(|value| value.to_bits()).collect()
        };
        let whole40 = logits(&[&prompt]);
        let split_24_16 = logits(&[&prompt[..24], &prompt[24..]]);
        let split_39_1 = logits(&[&prompt[..39], &prompt[39..]]);
        let whole16 = logits(&[&prompt[..16]]);
        let split_8_8 = logits(&[&prompt[..8], &prompt[8..16]]);
        // A tile boundary at 32, a last row on its own (the reuse path), and a first chunk
        // narrower than the wide tile.
        assert_eq!(whole40, split_24_16, "40 rows in one pass, then 24 and 16");
        assert_eq!(whole40, split_39_1, "40 rows in one pass, then 39 and 1");
        assert_eq!(whole16, split_8_8, "16 rows in one pass, then 8 and 8");
    }

    /// What the arm states for the generated profile tests: its ledger row, one positional
    /// region, no drafter, and the chunks of each cache mode. A per-loop prompt of 5,000 tokens
    /// runs in a whole chunk and the rest; a last-loop prompt runs a token a pass.
    fn stated() -> ArmProfile {
        let layout = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        let per_loop = walk_limits(true, 8_192);
        let last_loop = walk_limits(false, 8_192);
        ArmProfile::new("Dory", layout, per_loop)
            .with_code(DEVICE)
            .prompt(per_loop, 5_000, &[PREFILL_BATCH, 5_000 - PREFILL_BATCH])
            .prompt(last_loop, 3, &[1, 1, 1])
    }

    sparkflash_runtime::conformance!(DoryWalk, profile: stated());
}
