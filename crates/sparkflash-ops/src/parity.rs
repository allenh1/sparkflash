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

//! The bring-up parity harness: a layer's device dump against the reference, one case at a time.
//!
//! The harness is deliberately independent of any model crate. An arm builds a case from its
//! layer's OpPlan list and the two buffers, compares, and appends the result to a ParityReport;
//! a failure names the first element that left the tolerance, not just the layer. The reference
//! itself is whatever the model's host forward produced, so the same tool gives a new arm a gate
//! on day one.

use crate::error::OpError;
use crate::plan::{OpKind, OpPlan};

/// The absolute and relative allowance one comparison has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Largest allowed absolute difference.
    pub absolute: f32,
    /// Largest allowed difference relative to the reference magnitude.
    pub relative: f32,
}

impl Tolerance {
    /// No difference at all, for a bit-exact step.
    pub const EXACT: Self = Self {
        absolute: 0.0,
        relative: 0.0,
    };

    /// A comparison that accepts the f32 rounding of a different summation order.
    pub const F32_ROUNDING: Self = Self {
        absolute: 1e-6,
        relative: 1e-5,
    };

    /// A tolerance a case passes.
    #[must_use]
    pub const fn new(absolute: f32, relative: f32) -> Self {
        Self { absolute, relative }
    }

    /// Whether two finite values agree.
    #[must_use]
    pub fn accepts(self, reference: f32, actual: f32) -> bool {
        if reference == actual {
            return true;
        }
        if !reference.is_finite() || !actual.is_finite() {
            return false;
        }
        let difference = (reference - actual).abs();
        difference <= self.absolute + self.relative * reference.abs()
    }
}

/// One comparison's result.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityCase {
    /// The op or layer being checked.
    pub label: String,
    /// The plan the arm ran.
    pub kind: OpKind,
    /// Elements compared.
    pub elements: usize,
    /// The tolerance used.
    pub tolerance: Tolerance,
    /// The largest absolute difference seen.
    pub max_absolute: f32,
    /// The largest relative difference seen.
    pub max_relative: f32,
    /// The first element that left the tolerance, or None when the case passed.
    pub first_failure: Option<usize>,
}

impl ParityCase {
    /// Whether the case passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.first_failure.is_none()
    }
}

/// Compare one device buffer against its reference.
///
/// # Errors
/// Different lengths, a non-finite difference, or an exceeded tolerance. The error carries the
/// first failing index so a layer dump can be sliced at it.
pub fn compare(
    label: impl Into<String>,
    kind: OpKind,
    reference: &[f32],
    actual: &[f32],
    tolerance: Tolerance,
) -> Result<ParityCase, OpError> {
    let label = label.into();
    if reference.len() != actual.len() {
        return Err(OpError::shape(
            "parity",
            format!(
                "{label}: reference has {} values, device has {}",
                reference.len(),
                actual.len()
            ),
        ));
    }
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut first_failure = None;
    for (index, (&reference, &actual)) in reference.iter().zip(actual).enumerate() {
        if !reference.is_finite() || !actual.is_finite() {
            first_failure.get_or_insert(index);
            max_absolute = f32::INFINITY;
            continue;
        }
        let difference = (reference - actual).abs();
        max_absolute = max_absolute.max(difference);
        let relative = if reference == 0.0 {
            0.0
        } else {
            difference / reference.abs()
        };
        max_relative = max_relative.max(relative);
        if !tolerance.accepts(reference, actual) {
            first_failure.get_or_insert(index);
        }
    }
    let case = ParityCase {
        label,
        kind,
        elements: reference.len(),
        tolerance,
        max_absolute,
        max_relative,
        first_failure,
    };
    if let Some(index) = first_failure {
        return Err(OpError::Parity {
            index,
            message: format!(
                "{} at {index}: reference {}, device {}, |d| {} > abs {} + rel {}",
                case.label,
                reference[index],
                actual[index],
                case.max_absolute,
                tolerance.absolute,
                tolerance.relative
            ),
        });
    }
    Ok(case)
}

/// A layer's accumulated cases.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParityReport {
    /// The arm or layer the report covers.
    pub label: String,
    /// Every case, passed or failed.
    pub cases: Vec<ParityCase>,
}

impl ParityReport {
    /// A report for one layer.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            cases: Vec::new(),
        }
    }

    /// Compare and record a case. A mismatch is recorded and returned, so a caller that wants
    /// every stage's result can keep going.
    ///
    /// # Errors
    /// As compare.
    pub fn record(
        &mut self,
        label: impl Into<String>,
        kind: OpKind,
        reference: &[f32],
        actual: &[f32],
        tolerance: Tolerance,
    ) -> Result<(), OpError> {
        match compare(label, kind, reference, actual, tolerance) {
            Ok(case) => {
                self.cases.push(case);
                Ok(())
            }
            Err(error) => {
                if let OpError::Parity { index, .. } = &error {
                    self.cases.push(ParityCase {
                        label: format!("{} at {index}", self.label),
                        kind,
                        elements: reference.len(),
                        tolerance,
                        max_absolute: f32::INFINITY,
                        max_relative: f32::INFINITY,
                        first_failure: Some(*index),
                    });
                }
                Err(error)
            }
        }
    }

    /// Whether every recorded case passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.cases.is_empty() && self.cases.iter().all(ParityCase::passed)
    }

    /// The case labels that failed.
    #[must_use]
    pub fn failures(&self) -> Vec<&str> {
        self.cases
            .iter()
            .filter(|case| !case.passed())
            .map(|case| case.label.as_str())
            .collect()
    }
}

/// The plans an arm executes for one layer, in order, with the label each carries.
#[derive(Default)]
pub struct LayerPlan {
    /// The layer's name.
    pub label: String,
    /// The typed ops the layer launches.
    pub ops: Vec<Box<dyn OpPlan>>,
}

impl LayerPlan {
    /// A named layer plan.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ops: Vec::new(),
        }
    }

    /// Append an op, preserving launch order.
    pub fn push(&mut self, op: impl OpPlan + 'static) {
        self.ops.push(Box::new(op));
    }

    /// The scratch floats the whole layer needs, the sum of its ops.
    #[must_use]
    pub fn workspace_floats(&self) -> usize {
        self.ops.iter().map(|op| op.workspace_floats()).sum()
    }

    /// Validate every op before the first launch.
    ///
    /// # Errors
    /// The first invalid op's error, prefixed with the layer label.
    pub fn validate(&self) -> Result<(), OpError> {
        for op in &self.ops {
            op.validate().map_err(|error| {
                OpError::Binding(format!("{}: {}: {error}", self.label, op.kind().name()))
            })?;
        }
        Ok(())
    }
}
