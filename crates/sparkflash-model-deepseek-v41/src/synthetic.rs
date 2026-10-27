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

//! A weight source that needs no checkpoint.
//!
//! The generator is a pure function of the seed and the tensor name, so two sources built from the
//! same plan and the same seed hand back byte-identical tensors without either storing anything.
//! That is what lets a test assert that an operation is deterministic: re-running it against a fresh
//! source is re-running it against the same input.
//!
//! The bytes are format-agnostic on purpose. What a byte *means* depends on the dtype, and the
//! decoders arrive with the arithmetic that consumes them; nothing in this module pretends to know
//! what a value is, so it cannot agree with arithmetic it was never checked against.

use std::collections::BTreeMap;

use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

/// A weight source backed by a seeded generator.
#[derive(Debug, Clone)]
pub struct SyntheticWeights {
    specs: BTreeMap<String, TensorSpec>,
    seed: u64,
    limit: u128,
}

impl SyntheticWeights {
    /// Build a source that produces every tensor in `plan`.
    pub fn from_plan(plan: &TensorPlan, seed: u64) -> Self {
        let specs = plan
            .specs()
            .iter()
            .map(|spec| (spec.name.clone(), spec.clone()))
            .collect();
        SyntheticWeights {
            specs,
            seed,
            limit: DEFAULT_LIMIT,
        }
    }

    /// Set the largest tensor this source will materialise, in bytes.
    ///
    /// The real model has single tensors of hundreds of megabytes; a source that will allocate
    /// whatever it is asked for is a source that can exhaust a machine by accident. The default
    /// limit is small enough to catch that and large enough for every test.
    pub fn with_limit(mut self, limit: u128) -> Self {
        self.limit = limit;
        self
    }

    /// The seed this source was built with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The largest tensor this source will materialise, in bytes.
    pub fn limit(&self) -> u128 {
        self.limit
    }

    /// The number of bytes every tensor in this source would occupy together.
    pub fn total_bytes(&self) -> u128 {
        self.specs.values().map(TensorSpec::storage_bytes).sum()
    }

    /// The bytes of one tensor, as a pure function of the seed and the name.
    fn generate(&self, name: &str, bytes: usize) -> Vec<u8> {
        let mut stream = SplitMix64(fnv1a(name.as_bytes()) ^ self.seed);
        let mut out = Vec::with_capacity(bytes);
        while out.len() < bytes {
            out.extend_from_slice(&stream.next_u64().to_le_bytes());
        }
        out.truncate(bytes);
        out
    }
}

/// The default per-tensor limit: 128 MiB, comfortably above every synthetic tensor a test builds and
/// far below the checkpoint's largest.
pub const DEFAULT_LIMIT: u128 = 128 * 1024 * 1024;

impl WeightSource for SyntheticWeights {
    fn names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.specs
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_string(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let spec = self.spec(name)?;
        let bytes = spec.storage_bytes();
        if bytes > self.limit {
            return Err(WeightError::TooLarge {
                name: name.to_string(),
                bytes,
                limit: self.limit,
            });
        }
        Ok(self.generate(name, bytes as usize))
    }
}

/// FNV-1a over bytes: the name-to-stream seed, chosen because it is short, stable and has no
/// dependency.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// SplitMix64: a small, fully specified bit generator.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
