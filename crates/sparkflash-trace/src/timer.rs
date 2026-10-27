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

//! Nanosecond timestamp sources for CPU trace points.
//!
//! On aarch64 (Grace) this reads the ARMv9 generic timer directly in userspace
//! (mrs x0, cntvct_el0, about 1.2 ns, no VDSO or syscall). Everywhere else it falls back to
//! a monotonic process clock so the whole crate stays testable on x86 build hosts.

use std::sync::OnceLock;
#[cfg(not(target_arch = "aarch64"))]
fn fallback_epoch() -> &'static std::time::Instant {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now)
}

/// Frequency of the counter returned by read_counter, in Hz.
#[must_use]
#[inline]
pub fn counter_frequency() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        static FREQ: OnceLock<u64> = OnceLock::new();
        *FREQ.get_or_init(|| {
            let freq: u64;
            // SAFETY: reading CNTFRQ_EL0 is unprivileged on Linux/aarch64.
            unsafe {
                std::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack, preserves_flags));
            }
            if freq == 0 { 1_000_000_000 } else { freq }
        })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        1_000_000_000
    }
}

/// Raw free-running counter value.
#[must_use]
#[inline(always)]
pub fn read_counter() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let cnt: u64;
        // SAFETY: CNTVCT_EL0 is readable from EL0 on every ARMv9 Linux system.
        unsafe {
            std::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt, options(nomem, nostack, preserves_flags));
        }
        cnt
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_epoch().elapsed().as_nanos() as u64
    }
}

/// Monotonic nanoseconds from a fixed epoch, comparable across all threads of the process.
///
/// On aarch64 this is CNTVCT_EL0 scaled to nanoseconds, whose epoch is the boot of the
/// machine, not the start of the process - do not read the absolute value as uptime. Everywhere
/// else it is Instant::elapsed against a lazily cached epoch, i.e. process-relative. Both are
/// monotonic and safe to subtract, and the trace exporter rebases a capture onto its earliest
/// event so a viewer never mixes the two absolute scales.
#[must_use]
#[inline(always)]
pub fn read_arm_cntvct_ns() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        ((read_counter() as u128) * 1_000_000_000u128 / (counter_frequency() as u128)) as u64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        read_counter()
    }
}

/// Converts a raw counter value to nanoseconds.
#[must_use]
#[inline]
pub fn counter_to_ns(counter: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        ((counter as u128) * 1_000_000_000u128 / (counter_frequency() as u128)) as u64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_is_monotonic_and_nanosecond_scaled() {
        assert!(counter_frequency() > 0);
        let a = read_arm_cntvct_ns();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = read_arm_cntvct_ns();
        assert!(b > a, "expected monotonic clock");
        let delta_ms = (b - a) as f64 / 1e6;
        assert!(
            (1.0..200.0).contains(&delta_ms),
            "unexpected elapsed {delta_ms} ms"
        );
    }

    #[test]
    fn clock_is_comparable_across_threads() {
        let main = read_arm_cntvct_ns();
        let other = std::thread::spawn(read_arm_cntvct_ns).join().unwrap();
        assert!(
            other.abs_diff(main) < 5_000_000_000,
            "threads disagree wildly"
        );
    }
}
