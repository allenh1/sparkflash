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

//! What each serving arm states of the runtime's strategies: the gap table.
//!
//! The runtime prints a ledger when an engine starts ([sparkflash_runtime::Ledger]) and refuses
//! an arm that leaves a strategy unstated. The facts come from here, beside [MODELS](crate::MODELS),
//! so the boot path and `sparkflash arms --ledger` read one declaration and a missing row is a
//! boot error rather than a log line.
//!
//! Every arm states every [Strategy]. A row says whether the arm runs the strategy natively, has
//! the runtime loop it, or has it off with the reason. The detail is the arm's fixed statement;
//! a boot may refine a row with what it actually opened (lane counts, a drafter's depth), but it
//! cannot add or remove one.

use std::fmt;

/// One strategy the runtime offers an arm, in the order the ledger prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    /// Several requests' rows in one walk ([sparkflash_runtime::FusedEngine]).
    Lanes,
    /// A long prompt a chunk at a time between the decoders' steps.
    PromptParts,
    /// Every GEMM-class prompt part on the tensor-core tile.
    TensorCorePrompt,
    /// The one-to-eight row step on the tensor-core tile.
    TensorCoreDecode,
    /// Speculative drafting and a verify walk.
    Drafting,
    /// Every drafting lane's round verified in one walk.
    JointRounds,
    /// A verify's accepted run handed to the request in one reply.
    Runs,
    /// A later prompt continuing what a slot holds.
    PrefixReuse,
    /// A changed turn going back to the last prompt's end.
    Rewind,
    /// Every stored format kept in its own arithmetic: nothing widens unseen.
    StoredArithmetic,
    /// Rank 0 decides and broadcasts, every other rank replays.
    Group,
}

impl Strategy {
    /// Every strategy, in the order the ledger prints them.
    pub const ALL: [Self; 11] = [
        Self::Lanes,
        Self::PromptParts,
        Self::TensorCorePrompt,
        Self::TensorCoreDecode,
        Self::Drafting,
        Self::JointRounds,
        Self::Runs,
        Self::PrefixReuse,
        Self::Rewind,
        Self::StoredArithmetic,
        Self::Group,
    ];

    /// The name the ledger prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Lanes => "lanes",
            Self::PromptParts => "chunked prefill",
            Self::TensorCorePrompt => "tensor-core prompt",
            Self::TensorCoreDecode => "tensor-core decode",
            Self::Drafting => "drafting",
            Self::JointRounds => "joint rounds",
            Self::Runs => "multi-token runs",
            Self::PrefixReuse => "prefix reuse",
            Self::Rewind => "rewind",
            Self::StoredArithmetic => "stored-format arithmetic",
            Self::Group => "group",
        }
    }
}

/// How an arm runs one strategy, as the fixed declaration states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// The arm's walk takes the strategy's shape; the detail says how.
    Native(&'static str),
    /// The runtime runs it as a loop over a shape the walk does take; the detail says which.
    Looped(&'static str),
    /// Not run; the detail says why.
    Off(&'static str),
}

impl State {
    /// Whether the strategy runs at all.
    #[must_use]
    pub const fn runs(self) -> bool {
        !matches!(self, Self::Off(_))
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (word, detail) = match self {
            Self::Native(detail) => ("native", detail),
            Self::Looped(detail) => ("looped", detail),
            Self::Off(detail) => ("off", detail),
        };
        if detail.is_empty() {
            f.write_str(word)
        } else {
            write!(f, "{word} ({detail})")
        }
    }
}

/// One arm's complete statement, one row per [Strategy].
#[derive(Debug)]
pub struct ArmLedger {
    /// The name the ledger prints.
    pub arm: &'static str,
    /// The registry weight selectors this arm serves; empty for an arm no registry model uses.
    pub weights_ids: &'static [&'static str],
    /// Every strategy's mode. A [Strategy] missing here is refused at boot.
    pub rows: &'static [(Strategy, State)],
}

/// A ledger entry, by the arm's name.
#[must_use]
pub fn ledger_for_arm(arm: &str) -> Option<&'static ArmLedger> {
    ARM_LEDGERS.iter().find(|ledger| ledger.arm == arm)
}

/// A ledger entry, by a registry weight selector the arm serves.
#[must_use]
pub fn ledger_for_weights(weights_id: &str) -> Option<&'static ArmLedger> {
    ARM_LEDGERS
        .iter()
        .find(|ledger| ledger.weights_ids.contains(&weights_id))
}

/// Every serving arm's declaration.
pub static ARM_LEDGERS: &[ArmLedger] = &[
    ArmLedger {
        arm: "DeepSeek",
        weights_ids: &[crate::descriptor::DEFAULT_WEIGHTS_ID],
        rows: &[
            (
                Strategy::Lanes,
                State::Native("32 lanes in a 64-deep queue, rows gathered on the device"),
            ),
            (
                Strategy::PromptParts,
                State::Native("bulk chunks, 4096 rows a tick when idle and 2048 busy"),
            ),
            (
                Strategy::TensorCorePrompt,
                State::Native("bulk chunks on the FP8 x FP4/FP8 MMAs"),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Native("one-to-eight-row steps on the MMA (opt_gemv_mma)"),
            ),
            (Strategy::Drafting, State::Native("DSpark block drafter")),
            (
                Strategy::JointRounds,
                State::Native("up to 32 lanes share one speculative step"),
            ),
            (
                Strategy::Runs,
                State::Native("an accepted run reaches the request at once"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native(
                    "a lane keeps its sequence and resumes from the longest shared prefix",
                ),
            ),
            (
                Strategy::Rewind,
                State::Native("positional state: a changed turn moves its counters back"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("FP8 projections, FP4 and MXFP8 experts in their stored formats"),
            ),
            (
                Strategy::Group,
                State::Native("a tensor-parallel group: rank 0 decides each tick and broadcasts"),
            ),
        ],
    },
    ArmLedger {
        arm: "Qwen",
        weights_ids: &[
            crate::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID,
            crate::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID,
            crate::descriptor::QWEN36_FLASH_35B_A3B_FP8_WEIGHTS_ID,
        ],
        rows: &[
            (Strategy::Lanes, State::Native("four fused lanes")),
            (
                Strategy::PromptParts,
                State::Native("the prefill plan's chunks"),
            ),
            (
                Strategy::TensorCorePrompt,
                State::Native("prompt chunks on the wide tile and the grouped expert MMAs"),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Native("few-row twins of the tile at every row count"),
            ),
            (Strategy::Drafting, State::Native("DFlash 2 or MTP")),
            (
                Strategy::JointRounds,
                State::Native("lanes' rounds share one verify"),
            ),
            (
                Strategy::Runs,
                State::Native("an accepted run reaches the request at once"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native("a prompt that extends a lane's sequence"),
            ),
            (
                Strategy::Rewind,
                State::Native("a prompt that goes back to the last prompt's end"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("NVFP4 weights with BF16 activation pairs"),
            ),
            (
                Strategy::Group,
                State::Native(
                    "a tensor-parallel group on the tick engine: rank 0 decides each tick and the workers replay its rows",
                ),
            ),
        ],
    },
    ArmLedger {
        arm: "GLM",
        weights_ids: &[crate::descriptor::GLM53_FLASH_WEIGHTS_ID],
        rows: &[
            (Strategy::Lanes, State::Native("four fused lanes")),
            (
                Strategy::PromptParts,
                State::Native("bulk prefill chunks of 4096 rows"),
            ),
            (
                Strategy::TensorCorePrompt,
                State::Native("bulk GEMMs on the FP8 E4M3 MMAs; BF16 as head and remainder"),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Off("row GEMV; decode.tensor_cores=true is refused"),
            ),
            (Strategy::Drafting, State::Native("MTP or DFlash 2")),
            (
                Strategy::JointRounds,
                State::Native("lanes' rounds share one verify"),
            ),
            (
                Strategy::Runs,
                State::Native("an accepted run reaches the request at once"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native("a prompt that extends a lane's sequence"),
            ),
            (
                Strategy::Rewind,
                State::Native("a prompt that goes back to the last prompt's end"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("FP8 E4M3 projections and experts run E4M3 x E4M3; BF16 stays BF16"),
            ),
            (
                Strategy::Group,
                State::Native(
                    "a four-rank group on the tick engine: rank 0 decides each tick and the workers replay its rows",
                ),
            ),
        ],
    },
    ArmLedger {
        arm: "Inkling",
        weights_ids: &[crate::descriptor::INKLING2_SMALL_WEIGHTS_ID],
        rows: &[
            (Strategy::Lanes, State::Native("four fused lanes")),
            (
                Strategy::PromptParts,
                State::Native(
                    "bulk chunks of up to 8,192 tokens: 4,096 until 16,384 positions remain",
                ),
            ),
            (
                Strategy::TensorCorePrompt,
                State::Native("dense projections and NVFP4/MXFP8 experts on the grouped MMAs"),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Off("row token GEMV; decode.tensor_cores=true is refused"),
            ),
            (
                Strategy::Drafting,
                State::Native("prompt lookup or the DSpark stage"),
            ),
            (
                Strategy::JointRounds,
                State::Native("lanes' rounds share one verify"),
            ),
            (
                Strategy::Runs,
                State::Native("an accepted run reaches the request at once"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native("a prompt that extends a lane's sequence"),
            ),
            (
                Strategy::Rewind,
                State::Native("a prompt that goes back to the last prompt's end"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("NVFP4 and MXFP8 experts, BF16 dense, in their stored formats"),
            ),
            (
                Strategy::Group,
                State::Native(
                    "a four-rank group on the tick engine: rank 0 decides each tick and the workers replay its rows; a rank 0 that holds the vision tower keeps the fused group, whose request frame is ordered before its image encode",
                ),
            ),
        ],
    },
    ArmLedger {
        arm: "Dory",
        weights_ids: &[crate::descriptor::DORY1_0_4B_BF16_WEIGHTS_ID],
        rows: &[
            (
                Strategy::Lanes,
                State::Native("eight lanes, up to 32 rows a pass, a batch on every rank"),
            ),
            (
                Strategy::PromptParts,
                State::Native("chunks of up to 4096 rows between the other lanes' steps"),
            ),
            (
                Strategy::TensorCorePrompt,
                State::Native(
                    "prompt projections over eight rows on dory_mma_wide; attention on the MMA instances",
                ),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Off("warp-dot row kernels; decode.tensor_cores=true is refused"),
            ),
            (
                Strategy::Drafting,
                State::Native("loop-1 exit on the drafter rank while at most its lanes run"),
            ),
            (
                Strategy::JointRounds,
                State::Native("one verify walk over the ring's lanes"),
            ),
            (
                Strategy::Runs,
                State::Off("the ring hands back one token a lane"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native("a prompt that extends a lane's sequence"),
            ),
            (
                Strategy::Rewind,
                State::Native("a prompt that goes back to the last prompt's end"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("BF16 weights and a BF16 KV cache throughout"),
            ),
            (
                Strategy::Group,
                State::Native("a rank ring: each rank runs its loops, rank 0 schedules"),
            ),
        ],
    },
    ArmLedger {
        arm: "dense",
        weights_ids: &[],
        rows: &[
            (Strategy::Lanes, State::Native("four fused lanes")),
            (Strategy::PromptParts, State::Native("256-row chunks")),
            (
                Strategy::TensorCorePrompt,
                State::Native("prompt projections on the shared wide tile"),
            ),
            (
                Strategy::TensorCoreDecode,
                State::Off("row GEMV; decode.tensor_cores=true is refused"),
            ),
            (Strategy::Drafting, State::Off("no drafter")),
            (Strategy::JointRounds, State::Off("no drafter configured")),
            (
                Strategy::Runs,
                State::Off("the arm holds no answers past a step"),
            ),
            (
                Strategy::PrefixReuse,
                State::Native("a prompt that extends a lane's sequence"),
            ),
            (
                Strategy::Rewind,
                State::Native("a prompt that goes back to the last prompt's end"),
            ),
            (
                Strategy::StoredArithmetic,
                State::Native("BF16, FP8 E4M3 and NVFP4 parts run in their stored formats"),
            ),
            (
                Strategy::Group,
                State::Native("a tensor-parallel group over the shared RDMA exchange"),
            ),
        ],
    },
];

impl ArmLedger {
    /// The state this arm declares for `strategy`, if it declares one.
    #[must_use]
    pub fn state(&self, strategy: Strategy) -> Option<State> {
        self.rows
            .iter()
            .find(|(stated, _)| *stated == strategy)
            .map(|(_, state)| *state)
    }

    /// Every strategy this arm leaves out, in ledger order.
    #[must_use]
    pub fn unstated(&self) -> Vec<Strategy> {
        Strategy::ALL
            .iter()
            .copied()
            .filter(|strategy| self.state(*strategy).is_none())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arm_states_every_strategy_once() {
        for ledger in ARM_LEDGERS {
            assert!(
                ledger.unstated().is_empty(),
                "{} leaves {:?} unstated",
                ledger.arm,
                ledger.unstated()
            );
            assert_eq!(
                ledger.rows.len(),
                Strategy::ALL.len(),
                "{} states a row twice",
                ledger.arm
            );
        }
    }

    #[test]
    fn every_registered_model_has_an_arm_and_every_weights_id_resolves() {
        for model in crate::MODELS.iter() {
            assert!(
                ledger_for_weights(&model.weights_id).is_some(),
                "{} has no arm ledger",
                model.id
            );
        }
        for ledger in ARM_LEDGERS {
            for weights in ledger.weights_ids {
                assert!(
                    crate::find_model_by_weights(weights).is_some(),
                    "{} names {weights}, which no registry model uses",
                    ledger.arm
                );
            }
        }
    }

    #[test]
    fn a_dense_arm_states_what_it_lacks() {
        let dense = ledger_for_arm("dense").expect("the dense arm is declared");
        assert_eq!(
            dense.state(Strategy::TensorCoreDecode),
            Some(State::Off("row GEMV; decode.tensor_cores=true is refused"))
        );
        assert_eq!(
            dense.state(Strategy::Drafting),
            Some(State::Off("no drafter"))
        );
    }
}
