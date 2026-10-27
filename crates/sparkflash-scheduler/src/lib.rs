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

//! Milestone 5 scheduling: the bounded HSI indexer, the short-request prefill bypass, the
//! DSpark confidence policy (the engine's MTP stages own the draft tokens), the Causal Encoder
//! prefill bypass, sequence-parallel planning, L2 weight-prefetch planning and the trace-capture
//! CLI.
//!
//! Everything here is a decision engine. A plan computes what the runtime would do and
//! returns the numbers the tests assert, so the whole milestone is exercisable with no
//! weights, no GPU and no cluster.

#![deny(missing_docs)]

use serde::{Deserialize, Serialize};

/// Identifies one engine request across every step boundary it is scheduled in (D9).
///
/// The trace request id travels on the request itself and is re-bound at each step boundary,
/// so the scheduler passes this value to the admission decision and to the trace scope rather
/// than keeping a side table that would drift when a request is re-batched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestId(
    /// The engine's monotonic request counter.
    pub u64,
);

pub mod batching;
pub mod bypass;
pub mod config;
pub mod dspark;
pub mod engine;
pub mod error;
pub mod gates;
pub mod indexer;
pub mod philox;
pub mod prefill;
pub mod trace_cli;
pub mod transport;
