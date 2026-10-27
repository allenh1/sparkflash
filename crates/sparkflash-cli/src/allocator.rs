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

//! Process-startup allocator tuning for the resident host path.
//!
//! The tp4-host resident path allocates and then frees the attention output-projection weights as
//! f32 on every layer: wo_a is 8192x4096 (134 MB) and wo_b 5120x8192 (168 MB). With the default
//! 128 KiB mmap threshold those buffers are mmap'd and munmap'd per layer, so a decode step pays
//! for page mapping and zero-fill instead of reusing pages. Measured on spark-0 (aarch64,
//! released checkpoint, 20 threads): one attention_forward block was 217 ms/layer and 86-96 ms/layer
//! after the thresholds were raised, about 5 s/token over the 40 backbone layers; a four-rank
//! resident profile step fell from 13.83 s to 7.90 s with identical tokens. This changes only
//! allocation behaviour, never arithmetic; a later buffer-reuse change can subsume it.
//!
//! Attribution matters more than the option names: on spark-0 only the pair of raises gives the
//! win (attention 214-219 ms/layer untuned, 223-248 ms with only the mmap threshold, 221-245 ms
//! with only the trim threshold, 82-94 ms with both); a 200 MB allocate-touch-free probe keeps
//! 205 MB of RSS after free only with the pair, so the mmap raise moves these buffers onto the heap
//! and the trim raise keeps the freed heap resident, while either option alone is insufficient.
//! That is the measured mechanism this comment records, not an assumption from the labels.
//!
//! It is deliberately a library function, not hidden in the binary's main: any embedder that runs
//! the same resident path (the CLI, the serve binary, a test) can call it, while opening a library
//! object never changes the process allocator behind the caller's back.

use std::ffi::OsStr;

/// Environment variable that disables the tune when set to exactly "0".
pub const KILL_SWITCH: &str = "SPARKFLASH_ALLOCATOR_TUNING";

/// The requested M_MMAP_THRESHOLD, one gibibyte. Public so an evidence header records the value
/// the tune actually asked for instead of a duplicated literal that could drift.
pub const REQUESTED_MMAP_THRESHOLD_BYTES: u64 = 1_073_741_824;

/// The requested M_TRIM_THRESHOLD, one gibibyte. See REQUESTED_MMAP_THRESHOLD_BYTES.
pub const REQUESTED_TRIM_THRESHOLD_BYTES: u64 = 1_073_741_824;

/// The value both glibc calls take, in c_int form.
const THRESHOLD: libc::c_int = REQUESTED_MMAP_THRESHOLD_BYTES as libc::c_int;

/// What [tune] did, so a caller (or an evidence header) can record it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tuning {
    /// Whether the kill switch allowed the tune to run.
    pub enabled: bool,
    /// Whether M_MMAP_THRESHOLD accepted the new value.
    pub mmap_threshold: bool,
    /// Whether M_TRIM_THRESHOLD accepted the new value.
    pub trim_threshold: bool,
}

/// Applies the allocator tune unless SPARKFLASH_ALLOCATOR_TUNING=0.
///
/// Safe on every target: the Linux arm is the only one that calls libc, and the other targets
/// leave the allocator alone. Call it once at process startup, before model work; calling it from
/// a library constructor or on every request is unnecessary and would make the kill switch order
/// dependent.
#[must_use]
pub fn tune() -> Tuning {
    let enabled = switch_allows(std::env::var_os(KILL_SWITCH).as_deref());
    if !enabled {
        return Tuning {
            enabled: false,
            mmap_threshold: false,
            trim_threshold: false,
        };
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: mallopt only changes process-wide glibc allocator tunables. The caller invokes
        // this at startup, before model allocations, thread pools or signal handlers exist.
        let mmap_threshold = unsafe { libc::mallopt(libc::M_MMAP_THRESHOLD, THRESHOLD) != 0 };
        let trim_threshold = unsafe { libc::mallopt(libc::M_TRIM_THRESHOLD, THRESHOLD) != 0 };
        Tuning {
            enabled: true,
            mmap_threshold,
            trim_threshold,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        Tuning {
            enabled: false,
            mmap_threshold: false,
            trim_threshold: false,
        }
    }
}

/// Whether the kill switch leaves tuning enabled for a given environment value.
fn switch_allows(value: Option<&OsStr>) -> bool {
    value != Some(OsStr::new("0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tune_is_on_unless_the_kill_switch_is_exactly_zero() {
        assert!(switch_allows(None));
        assert!(switch_allows(Some(OsStr::new("1"))));
        assert!(!switch_allows(Some(OsStr::new("0"))));
        assert!(switch_allows(Some(OsStr::new("00"))));
        assert!(switch_allows(Some(OsStr::new("false"))));
    }
}
