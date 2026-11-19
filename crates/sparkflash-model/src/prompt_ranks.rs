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

//! The merged prompt-logprob contract: one scored prompt position, after every rank's shard has
//! been merged.
//!
//! These are protocol values, not arithmetic: a rank's own contract (`sparkflash-model-deepseek-v41`
//! and `sparkflash-engine`) produces them, a served group's step adapter carries them, and the
//! HTTP layer renders them. They live here so a model crate's rank opener and the serving crate's
//! rank seam can both name them without one model crate depending on another.

/// One alternative in a merged prompt-logprob position.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedAlternative {
    /// The global vocabulary id.
    pub token_id: u32,
    /// The f32 log probability.
    pub logprob: f32,
}

/// One merged scored prompt position, before tokenizer decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedPosition {
    /// The zero-based index in the scored-position list (prompt index minus one for the TP=4 path).
    pub position: usize,
    /// The prompt token this position scores.
    pub own_token_id: u32,
    /// Its log probability.
    pub own_logprob: f32,
    /// Its exact 1-based competition rank (strictly greater count plus one).
    pub own_rank: u32,
    /// The requested alternatives, in serial top-k order.
    pub alternatives: Vec<MergedAlternative>,
}
