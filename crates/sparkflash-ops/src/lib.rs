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

//! One typed op library over one kernel stack.
//!
//! Every model arm composes the same plans: a plan fixes shapes, precision, layout and stream at
//! load, launches with no allocation and no host synchronization, and names the kernel it runs in
//! the shared stack. A kernel improvement therefore lands in every arm, and a new arm is config,
//! a layer plan and op calls rather than a new CUDA tree.
//!
//! The crate is the seam the model arms meet at:
//!
//! * plan is the vocabulary: dtypes, layouts, streams, OpPlan and Launch.
//! * gemm, attention, scan, epilogue and collective are the typed plans.
//! * images exposes the shared cubins, so no model crate compiles CUDA.
//! * parity is the bring-up harness an arm runs against its host reference.
//! * gemv, gdn and tp are the shared launch helpers the Qwen, GLM and DeepSeek arms call.
//!
//! # Example
//!
//!     use sparkflash_ops::gemm::DenseGemmPlan;
//!     use sparkflash_ops::plan::{Matrix, OpPlan, Precision, Layout, StreamId};
//!
//!     let plain = DenseGemmPlan::new(
//!         Matrix::new(128, 4096),
//!         2560,
//!         Precision::MxFp8,
//!         Layout::BlockScaled { block: 32 },
//!         StreamId(7),
//!         1,
//!     )
//!     .expect("a valid GEMM");
//!     let split = DenseGemmPlan::new(
//!         Matrix::new(128, 4096),
//!         2560,
//!         Precision::MxFp8,
//!         Layout::BlockScaled { block: 32 },
//!         StreamId(7),
//!         4,
//!     )
//!     .expect("a valid split-K GEMM");
//!     assert!(!plain.is_split_k());
//!     assert!(split.is_split_k());
//!     assert_eq!(split.workspace_floats(), 4 * 128 * 4096);

pub mod attention;
pub mod codec;
pub mod collective;
pub mod dflash;
pub mod epilogue;
pub mod error;
pub mod gemm;
pub mod images;
pub mod parity;
pub mod plan;
pub mod sampler;
pub mod scan;

pub mod gdn;
pub mod gemv;
pub mod tp;

pub use error::OpError;
pub use plan::{
    DType, KernelId, Launch, Layout, Matrix, OpKind, OpPlan, Operand, Precision, StreamId,
};
