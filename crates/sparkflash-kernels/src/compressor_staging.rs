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

//! Staging the compressor projections for the host-weight GEMV, and the store-width rule.
//!
//! The compressor matrices are the checkpoint f32 (ratio > 1) or bf16 (ratio == 1) weights
//! (docs/archive/build/native-arithmetic-audit.md:153; model.py:446-449, 461-486), so the kernel that
//! reproduces their rows is b12x_hostw_gemv_m1_sm121, not the MXFP8 GEMV: the latter decodes
//! E4M3 + ue8m0 and cannot read either dtype. This module is the wrapper the stage body calls:
//! it permutes a weight into the kernel B12x order once per layer per run
//! (hostw_gemv_ref::stage_f32_weight / stage_bf16_weight), returns the bytes ready for one H2D,
//! and returns the store width it chose so the stage record can declare it.
//!
//! THE STORE-WIDTH RULE. The layer compress ratio decides. At ratio 1 the bf16 store is exact
//! enough, because the compressor output passes a mandatory narrow_bf16 before the store
//! (attention.rs:2310-2330 narrow_and_round_trip) and a bf16 store after a narrow is idempotent,
//! which tests/compressor_latent_proof.rs proves. Above ratio 1 the f32 store is required, because
//! pool_group (attention.rs:985-1020) reads the raw f32 rows before narrowing and a bf16 store
//! changes the pooled latent (the same proof file negative control).
//! The declaration is returned, not assumed: the caller records [StagedWeight::declared].

use crate::hostw_gemv_ref;

/// The dtype a compressor projection weight is staged in for the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightStore {
    /// The checkpoint f32 rows, staged as f32.
    F32,
    /// The checkpoint bf16 rows, staged as bf16 (ratio == 1 only).
    Bf16,
}

impl WeightStore {
    /// The word the stage record declares.
    #[must_use]
    pub fn declared(self) -> &'static str {
        match self {
            Self::F32 => "f32-store",
            Self::Bf16 => "bf16-store",
        }
    }

    /// Bytes one staged word occupies.
    #[must_use]
    pub fn word_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// The store width the layer compress ratio licenses.
#[must_use]
pub fn store_width_for_ratio(ratio: usize) -> WeightStore {
    if ratio == 1 {
        WeightStore::Bf16
    } else {
        WeightStore::F32
    }
}

/// Why a staging request was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StagingError {
    /// The weight has the wrong number of values.
    Shape {
        /// The matrix name.
        what: &'static str,
        /// The length it had.
        got: usize,
        /// The length cols * inner requires.
        want: usize,
    },
    /// The matrix is not a whole number of 32-column, 32-value micro-tiles.
    Tiles {
        /// The matrix name.
        what: &'static str,
        /// Matrix columns.
        cols: usize,
        /// Reduction length.
        inner: usize,
    },
}

impl core::fmt::Display for StagingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape { what, got, want } => write!(f, "{what} is {got} values, needs {want}"),
            Self::Tiles { what, cols, inner } => {
                write!(
                    f,
                    "{what} is {cols} by {inner}, not a whole number of 32-value tiles"
                )
            }
        }
    }
}

/// One compressor projection staged for the kernel, with the width declared.
#[derive(Clone, Debug, PartialEq)]
pub struct StagedWeight {
    /// The name of the checkpoint tensor this was staged from.
    pub name: &'static str,
    /// The width the kernel will read.
    pub store: WeightStore,
    /// Matrix columns (the kernel output width).
    pub cols: usize,
    /// Matrix inner dimension (the kernel reduction length).
    pub inner: usize,
    /// The staged f32 words, when the store is f32.
    pub f32_bytes: Vec<f32>,
    /// The staged bf16 words, when the store is bf16.
    pub bf16_bytes: Vec<u16>,
}

impl StagedWeight {
    /// Bytes the staged image occupies, the quantity the one H2D moves.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.f32_bytes.len() * 4 + self.bf16_bytes.len() * 2
    }
}

/// Stages one compressor projection in the width the layer ratio licenses.
///
/// A ratio == 1 layer is staged as bf16: the f32 source is rounded to bf16 first, which is the
/// same image a bf16 checkpoint carries and the store the proof showed unobservable. A ratio > 1
/// layer is staged as f32, unchanged. The name is carried into the declaration.
///
/// # Errors
/// [StagingError::Shape] unless the source is exactly cols * inner values.
pub fn stage_for_ratio(
    ratio: usize,
    name: &'static str,
    source: &[f32],
    cols: usize,
    inner: usize,
) -> Result<StagedWeight, StagingError> {
    let want = cols * inner;
    if source.len() != want {
        return Err(StagingError::Shape {
            what: name,
            got: source.len(),
            want,
        });
    }
    // The kernel permutes in whole 32-column, 32-value micro-tiles
    // (hostw_gemv_ref::b12x_hostw_weight_index:25-35); a matrix that is not a whole number of
    // tiles has no staging position and would read past its row, so it is refused here.
    if !cols.is_multiple_of(32) || !inner.is_multiple_of(32) {
        return Err(StagingError::Tiles {
            what: name,
            cols,
            inner,
        });
    }
    let store = store_width_for_ratio(ratio);
    match store {
        WeightStore::F32 => Ok(StagedWeight {
            name,
            store,
            cols,
            inner,
            f32_bytes: hostw_gemv_ref::stage_f32_weight(source, cols, inner),
            bf16_bytes: Vec::new(),
        }),
        WeightStore::Bf16 => Ok(StagedWeight {
            name,
            store,
            cols,
            inner,
            f32_bytes: Vec::new(),
            bf16_bytes: hostw_gemv_ref::stage_bf16_weight(&bf16_bits(source), cols, inner),
        }),
    }
}

/// Rounds f32 values to bf16 and returns the bit patterns, round to nearest even.
#[must_use]
pub fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            let round = 0x7FFF + ((bits >> 16) & 1);
            ((bits.wrapping_add(round)) >> 16) as u16
        })
        .collect()
}
