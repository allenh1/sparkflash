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
