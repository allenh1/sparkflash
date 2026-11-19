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

//! The host sampler, re-exported from its neutral home.
//!
//! The sampling a served arm applies to a logits row is request bookkeeping, not an operation
//! plan, so the type moved to [sparkflash_model::sampler] beside the other serving contracts. This
//! module keeps the `sparkflash_ops::sampler` path its callers already use; it is one type and one
//! set of arithmetic either way.

pub use sparkflash_model::sampler::*;

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::Sampling;

    /// The old path and the neutral home are one type: a fork would be two histories and two
    /// sets of arithmetic with the same name.
    #[test]
    fn the_ops_sampler_is_the_models_sampler() {
        let mut sampler: sparkflash_model::Sampler = Sampler::new();
        sampler.record_prompt(&[1, 2]);
        assert_eq!(sampler.history(), &[1, 2]);
        let greedy = Sampling::default();
        assert_eq!(sampler.choose(&[0.0, 3.0, 1.0], &greedy), 1);
        let mut ours = Sampler::new();
        ours.record_prompt(&[1, 2]);
        assert_eq!(ours.choose(&[0.0, 3.0, 1.0], &greedy), 1);
    }
}
