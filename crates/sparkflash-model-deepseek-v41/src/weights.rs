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

//! The weight source contract, re-exported from the generic weights crate.
//!
//! `check_source_against_plan` is the one V4.1-aware wrapper: it walks the model plan's specs and
//! delegates the comparison to the generic crate.

pub use sparkflash_weights::weights::*;

use crate::tensor::TensorPlan;

/// Check every tensor in `plan` against `source`.
///
/// This is the plan-shaped form of [sparkflash_weights::check_source_against_specs].
pub fn check_source_against_plan<S: WeightSource + ?Sized>(
    source: &S,
    plan: &TensorPlan,
) -> Conformance {
    sparkflash_weights::check_source_against_specs(source, plan.specs())
}
