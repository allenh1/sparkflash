//! Shared test-only weight sources.
//!
//! [FiniteSource] is the finite f32 source the reference tests use: plan names and shapes, every
//! tensor served as little-endian f32 with small finite values, deterministic per seed and name.
//! It is deliberately not [sparkflash_engine::synthetic::SyntheticWeights]: raw synthetic bytes can
//! decode to NaN or inf, and a comparison over NaNs is vacuous.

use sparkflash_engine::dtype::DType;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// A finite, plan-shaped weight source: the plan's names and shapes, every tensor as finite f32.
pub struct FiniteSource {
    specs: Vec<TensorSpec>,
    seed: u64,
}

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

impl WeightSource for FiniteSource {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(TensorSpec {
                dtype: DType::F32,
                ..spec.clone()
            }),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let elements = usize::try_from(self.spec(name)?.elements()).expect("a test tensor fits");
        let salt = name_salt(name);
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            // A per-element mix, deliberately not a per-row one: with the high bits of the index the
            // pattern would barely change across a row, two rows could produce exactly equal dot
            // products, and the argmax would then be a tie that hides a mis-indexed read.
            let mixed = (index as u64)
                .wrapping_add(salt)
                .wrapping_add(self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let hashed = (mixed ^ (mixed >> 13)).wrapping_mul(0x2545_f491_4f6c_dd1d);
            // Strictly positive and at least 0.01, so nothing divides by a zero scale, and 256 values
            // deep so neighbours differ.
            let value = 0.01 + 0.000_1 * (hashed >> 56) as f32;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
}
