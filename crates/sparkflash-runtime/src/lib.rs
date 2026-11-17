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
//! * [fused]: slots on one engine thread, the prefix claim, park and reclaim, and multi-token
//!   runs (a member drains its round in one step);
//! * [share]: the step wall models and the row share, which gives each verify row to the draft
//!   likeliest accepted while it pays;
//! * [rounds]: one speculative round over several sequences - every chain drafted, rows shared,
//!   one walk, each sequence accepting its own;
//! * [group]: fused lanes across a tensor-parallel group - rank 0 decides and broadcasts, every
//!   other rank replays.
//!
//! The plan, and what is still to move here, is
//! [docs/engine/03-runtime.md](../../../docs/engine/03-runtime.md).

pub mod fused;
pub mod group;
pub mod rounds;
pub mod share;

pub use fused::{FusedArm, FusedEngine, FusedSession};
pub use group::{
    GroupLane, GroupLanes, GroupWire, Split, TextWire, end_group, mirror_lanes,
    mirror_until_the_group_ends, watched_hub,
};
pub use rounds::{JOINT_ROUNDS_ENV, Opened, RoundArm, joint_rounds, run_round};
pub use share::{BandWall, ExpertUnionWall, StepWall, share_rows};
