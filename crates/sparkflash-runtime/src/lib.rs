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

//! How rows run, written once for every model arm.
//!
//! An arm supplies arithmetic: its sequences, a walk over some rows of some of them, optionally a
//! drafter, and what a walk costs. Everything that decides *which rows run together* lives here,
//! so a strategy built once serves every arm that can take its walk:
//!
//! * [engine]: the tick engine, its scheduler thread and the sessions that post to it;
//! * [tick]: the scheduler's decisions, each one a function: the lane choice, the prompt chunks,
//!   the step's shape, and the trace that pins them;
//! * [share]: the step wall models and the row share, which gives each verify row to the draft
//!   likeliest accepted while it pays;
//! * [rounds]: one speculative round over several sequences - every chain drafted, rows shared,
//!   one walk, each sequence accepting its own;
//! * [group]: lanes across a tensor-parallel group - rank 0 decides and broadcasts, every
//!   other rank replays;
//! * [ledger]: which of these an arm runs natively, looped or not at all, printed when its engine
//!   starts;
//! * [mod@serve]: the one entry a walk arm's backend serves through, alone or in a group.
//!
//! The plan, and what is still to move here, is
//! [docs/engine/03-runtime.md](../../../docs/engine/03-runtime.md).

pub mod chunk;
pub mod conformance;
pub mod engine;
pub mod group;
pub mod layout;
pub mod ledger;
pub mod part;
pub mod plan;
pub mod pool;
pub mod record;
pub mod resume;
pub mod rounds;
pub mod runtime;
pub mod serve;
pub mod share;
pub mod testing;
pub mod tick;
pub mod walk;

pub use chunk::{CHUNK_SIZES, ChunkPolicy, quantise};
pub use conformance::{
    ArmFixture, CONFORMANCE_CHECKS, CheckOutcome, ConformanceCheck, Fixture, FormatRule,
    WalkFixture, check_claims, checks_for, local_group, run_check, run_generated, token_bytes,
};
pub use engine::{
    Kept as TickKept, PROMPT_PARTS_ENV, RequestId, TickConfig, TickEngine, TickEvent, TickServe,
    TickSession,
};
pub use group::{
    GroupWire, Split, TextWire, TickGroup, end_group, mirror_tick_lanes, mirror_tick_lanes_seen,
    mirror_ticks_until_the_group_ends, watched_hub,
};
pub use layout::{
    LaneState, Limits, Region, StateKind, StateLayout, check_checkpoint, check_rewind,
};
pub use ledger::{Ledger, Mode, Strategy};
pub use part::{PartBody, PartExec, PartHost, PartMaker, PartRegistry, RankSplit};
pub use plan::{
    PlanBranch, PlanHost, PlanLane, PlanLayer, PlanPart, PlanPass, PlanRows, PlanSeq, PlanWalk,
    PlanWindow, plan_head, plan_layer, plan_state, walk_rows,
};
pub use pool::{LaneMemory, LanePool};
pub use resume::{
    REWIND_ENV, Restore, Resume, Rewind, checkpoint_after, extends, resume, rewind, rewind_plan,
    rewind_request,
};
pub use rounds::{JOINT_ROUNDS_ENV, Opened, RoundArm, joint_rounds, run_round};
pub use runtime::{Part, Runtime, RuntimeSession, RuntimeSlot, StepMode, ledger_for};
pub use serve::{Place, Serving, WalkArm, lane_limit, mirror, serve, serve_group};
pub use share::{BandWall, DraftBudget, ExpertUnionWall, StepWall, share_rows};
pub use tick::{PREFILL_BYPASS_MAX_TOKENS, PREFILL_DECODE_INTERVAL};
pub use walk::{
    Absent, ComputeDeclaration, Draft, Drafter, HeadInput, HeadRow, Hidden, NoVision,
    OutputConstraint, Pass, PassKind, PassMask, Residual, Segment, Segments, Vision, Walk,
    WalkCapabilities, WalkHook,
};
