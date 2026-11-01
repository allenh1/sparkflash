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

//! SplitMix64, and the per-tensor stream synthetic checkpoints draw their values from.
//!
//! [splitmix64] is Steele, Lea and Flood's finaliser with the golden-ratio increment: the draw the
//! n-gram hashes are seeded with, and the mixer behind [TensorRng]. A pack's fixture writer
//! decides what a tensor's values mean; this only hands it a stream that depends on nothing but the
//! seed and the tensor's name.

/// The SplitMix64 increment, floor(2^64 / golden ratio).
pub const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The first SplitMix64 mixing constant.
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;

/// The second SplitMix64 mixing constant.
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

/// One round of SplitMix64.
#[must_use]
pub fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

/// A deterministic per-tensor random stream.
///
/// Seeding from the tensor name (rather than from a running counter) means a fixture does not
/// depend on the order a plan happens to enumerate tensors in: adding a tensor cannot silently
/// change another tensor's values.
#[derive(Debug, Clone)]
pub struct TensorRng {
    state: u64,
}

impl TensorRng {
    /// The stream of tensor `name` under `seed`.
    #[must_use]
    pub fn new(seed: u64, name: &str) -> Self {
        let mut state = seed
            .wrapping_mul(SPLITMIX_GAMMA)
            .wrapping_add(0xA5A5_5A5A_1234_5678);
        for byte in name.as_bytes() {
            state = splitmix64(state.wrapping_add(u64::from(*byte)));
        }
        Self { state }
    }

    /// The next value in `[0, 1)`, on a 2^-24 grid.
    pub fn unit(&mut self) -> f32 {
        self.state = splitmix64(self.state);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// The next value in `[-1, 1)`.
    pub fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// The next four-bit value.
    pub fn nibble(&mut self) -> u8 {
        self.state = splitmix64(self.state);
        ((self.state >> 40) & 0x0F) as u8
    }
}
