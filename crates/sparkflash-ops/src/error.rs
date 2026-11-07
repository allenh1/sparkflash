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

//! The one error type every op plan and every launch reports.

/// Why a typed op could not be built, bound or launched.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpError {
    /// A tensor would read or write outside its declared shape.
    #[error("{what}: {message}")]
    Shape {
        /// The op or tensor that refused the shape.
        what: &'static str,
        /// The concrete mismatch.
        message: String,
    },
    /// The dtype or precision is not one this kernel implements.
    #[error("{op} does not take {dtype}")]
    Unsupported {
        /// The op's stable name.
        op: &'static str,
        /// The requested dtype.
        dtype: &'static str,
    },
    /// A pointer or a stride does not satisfy the kernel's alignment contract.
    #[error("{what} is not aligned for {requirement}: {address:#x}")]
    Alignment {
        /// The operand that failed.
        what: &'static str,
        /// The alignment the kernel states.
        requirement: &'static str,
        /// The offending address or stride.
        address: usize,
    },
    /// A plan was asked to bind an operand of the wrong shape or dtype.
    #[error("{0}")]
    Binding(String),
    /// The kernel catalogue has no implementation for the plan's part, shape, precision and
    /// pass.
    #[error(transparent)]
    Resolve(#[from] crate::resolver::ResolveError),
    /// A hand-written resolution failure that has no catalogue candidates.
    #[error("{0}")]
    Resolution(String),
    /// The device refused the load or launch.
    #[error("{0}")]
    Device(String),
    /// A parity comparison exceeded the case's tolerance.
    #[error("parity failed at {index}: {message}")]
    Parity {
        /// The first disagreeing element.
        index: usize,
        /// The absolute and relative error beside the tolerance.
        message: String,
    },
}

impl OpError {
    /// A shape refusal.
    #[must_use]
    pub fn shape(what: &'static str, message: impl Into<String>) -> Self {
        Self::Shape {
            what,
            message: message.into(),
        }
    }

    /// A device refusal, with the driver's explanation.
    #[must_use]
    pub fn device(message: impl Into<String>) -> Self {
        Self::Device(message.into())
    }
}
