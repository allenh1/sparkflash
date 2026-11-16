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

//! The serving sampler on the device path: Gumbel-max sampling with counter-based noise.
//!
//! The sampler, its noise and the truncated cut live in [`sparkflash_model::gumbel`], shared with
//! the other arms that sample (Inkling's host pick takes the same cut over a whole row). The device
//! path uses three modes, by request:
//! - greedy (temperature 0): the head's argmax, the unchanged path;
//! - temperature only (top_p 1, top_k 0): the noise is added inside the head kernel and the
//!   cross-rank argmax picks the sample;
//! - truncated (top_k or top_p): the head writes raw logits, each rank keeps its slice's
//!   [`TOPC`] largest with its slice normaliser (`opt_rows_topc`), the slots are exchanged, and every
//!   rank cuts and samples on the host ([`Sampler::pick_truncated`]) from the same bytes.
//!
//! The truncated cut is exact while the nucleus fits in the merged candidate list (at least
//! [`TOPC`] tokens are guaranteed); a nucleus wider than that is cut at the list, which only a
//! near-flat distribution reaches.

pub use sparkflash_model::gumbel::GumbelSampler as Sampler;
#[allow(unused_imports)]
pub(super) use sparkflash_model::gumbel::{TOPC, TOPC_WORDS};
