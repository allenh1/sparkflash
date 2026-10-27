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

//! Shared test-only weight sources.
//!
//! [FiniteSource] is the finite f32 source the reference tests use: plan names and shapes, every
//! tensor served as little-endian f32 with small finite values, deterministic per seed and name.
//! It is deliberately not [sparkflash_engine::synthetic::SyntheticWeights]: raw synthetic bytes can
//! decode to NaN or inf, and a comparison over NaNs is vacuous.

use sparkflash_engine::dtype::DType;
use sparkflash_engine::quant;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec, checkpoint_storage_dtype};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// The plan's own storage dtypes, filled with finite values: F8-E4M3 attention weights with an
/// all-ones E8M0 grid, F4-E2M1 expert weights with PerRow32 scales, BF16 plain weights and F32
/// biases. [FiniteSource] deliberately serves F32 everywhere, which exercises the fallback; this
/// source exercises the released coded kernels, so a chunk-vs-token comparison covers the fused
/// F4/F8 arithmetic instead of only the F32 path.
///
/// Not every test binary that includes this module uses both sources, so the unused half would trip
/// `-D warnings`; the module is shared and each binary draws what it needs.
#[allow(dead_code)]
pub struct CodedSource {
    specs: Vec<TensorSpec>,
    seed: u64,
}

#[allow(dead_code)]
impl CodedSource {
    /// The coded source for `plan`, at `seed`.
    pub fn from_plan(plan: &TensorPlan, seed: u64) -> CodedSource {
        CodedSource {
            specs: plan.specs().to_vec(),
            seed,
        }
    }
}

impl WeightSource for CodedSource {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(spec.clone()),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.spec(name)?;
        let elements = usize::try_from(spec.elements()).expect("a test tensor fits");
        let salt = name_salt(name);
        let mut bytes = Vec::new();
        match spec.dtype {
            DType::F32 => {
                bytes.reserve(elements * 4);
                for index in 0..elements {
                    bytes.extend_from_slice(&value_at(index, salt, self.seed).to_le_bytes());
                }
            }
            DType::BF16 => {
                bytes.reserve(elements * 2);
                for index in 0..elements {
                    let value = quant::bf16_round(value_at(index, salt, self.seed));
                    bytes.extend_from_slice(&quant::f32_to_bf16_bits(value).to_le_bytes());
                }
            }
            DType::F8E4M3 => {
                bytes.reserve(elements);
                for index in 0..elements {
                    bytes.push(quant::e4m3_encode(value_at(index, salt, self.seed)));
                }
            }
            DType::F4E2M1 => {
                // Wider than the f32 pattern so the nibbles span the table instead of rounding to
                // zero; E2M1's finite range is 0..6.
                bytes.resize(elements.div_ceil(2), 0);
                for index in 0..elements {
                    let value = 100.0 * value_at(index, salt, self.seed);
                    let nibble = quant::f4e2m1_encode(value);
                    if index.is_multiple_of(2) {
                        bytes[index / 2] = nibble;
                    } else {
                        bytes[index / 2] |= nibble << 4;
                    }
                }
            }
            DType::F8E8M0 => bytes.resize(elements, 127u8),
            other => panic!("the coded fixture has no pattern for {other:?}"),
        }
        Ok(bytes)
    }
}

/// A finite, plan-shaped weight source: the plan's names and shapes, every tensor as finite f32.
///
/// Only some integration tests include this module, and the ones that need the coded dtypes never
/// construct it; the allow keeps each test binary's dead-code pass independent of its siblings.
#[allow(dead_code)]
pub struct FiniteSource {
    specs: Vec<TensorSpec>,
    seed: u64,
}

#[allow(dead_code)]
impl FiniteSource {
    /// The source for `plan`, at `seed`.
    pub fn from_plan(plan: &TensorPlan, seed: u64) -> FiniteSource {
        FiniteSource {
            specs: plan.specs().to_vec(),
            seed,
        }
    }
}

/// A stable per-name offset, so two tensors of the same shape are not identical.
fn name_salt(name: &str) -> u64 {
    name.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u64::from(byte))
    })
}

/// The deterministic value of one element, shared by the f32 and the fp8 arms so the source stays
/// one pattern as its dtype changes.
fn value_at(index: usize, salt: u64, seed: u64) -> f32 {
    // A per-element mix, deliberately not a per-row one: with the high bits of the index the
    // pattern would barely change across a row, two rows could produce exactly equal dot
    // products, and the argmax would then be a tie that hides a mis-indexed read.
    let mixed = (index as u64)
        .wrapping_add(salt)
        .wrapping_add(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let hashed = (mixed ^ (mixed >> 13)).wrapping_mul(0x2545_f491_4f6c_dd1d);
    // Strictly positive and at least 0.01, so nothing divides by a zero scale, and 256 values deep
    // so neighbours differ.
    0.01 + 0.000_1 * (hashed >> 56) as f32
}

impl WeightSource for FiniteSource {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(TensorSpec {
                // f32 for every tensor except the Engram wkv projection, whose checkpoint storage
                // format is the only way to recover the codes (audit F4).
                dtype: checkpoint_storage_dtype(name).unwrap_or(DType::F32),
                ..spec.clone()
            }),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.spec(name)?;
        let elements = usize::try_from(spec.elements()).expect("a test tensor fits");
        let salt = name_salt(name);
        if spec.dtype != DType::F32 {
            // The Engram wkv projection: fp8 codes and an all-ones E8M0 grid (audit F4).
            let mut bytes = Vec::with_capacity(elements);
            match spec.dtype {
                DType::F8E4M3 => {
                    for index in 0..elements {
                        bytes.push(sparkflash_engine::quant::e4m3_encode(value_at(
                            index, salt, self.seed,
                        )));
                    }
                }
                DType::F8E8M0 => bytes.resize(elements, 127u8),
                other => {
                    unreachable!("the fixture serves only f32 and the Engram wkv pair: {other:?}")
                }
            }
            return Ok(bytes);
        }
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            bytes.extend_from_slice(&value_at(index, salt, self.seed).to_le_bytes());
        }
        Ok(bytes)
    }
}

// The helpers are shared across test binaries; a binary that draws only the plain sources would
// otherwise warn the whole module dead under -D warnings.
#[allow(dead_code)]
pub mod host_path;
