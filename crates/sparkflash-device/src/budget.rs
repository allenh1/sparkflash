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

//! M3-b budget admission: refuse before the first device allocation.
//!
//! This is deliberately a pure arithmetic gate with a named floor, not an allocator. The caller
//! supplies the MemAvailable reading and the projected host and device bytes; [BudgetAdmission::admit]
//! either returns a receipt that states every number it used, or a typed refusal with the shortfall.
//! The real launcher must call it before it loads a driver or allocates a device buffer, and a test
//! with fake numbers pins both sides of the boundary.
//!
//! The floor is 1 GiB of remaining MemAvailable. It is a headroom rule, not the harnesses' older
//! absolute 32 GiB gate: the stage is admitted only when the projected host plus device bytes leave
//! at least 1 GiB of MemAvailable.
//! On the GB10 node the device allocations are charged against the same unified-memory
//! MemAvailable, so device bytes are projected here rather than treated as free.

use thiserror::Error;

/// The remaining-memory floor: 1 GiB.
pub const ADMISSION_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;

/// The numbers one admission decision uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetRequest {
    /// MemAvailable read immediately before the decision, in bytes.
    pub mem_available_bytes: u64,
    /// Host bytes the staging path will retain (weights, rows, cubin copies).
    pub projected_host_bytes: u64,
    /// Device allocation bytes the stage will request.
    pub projected_device_bytes: u64,
}

impl BudgetRequest {
    /// The total projected usage, or None on overflow.
    #[must_use]
    pub fn projected_bytes(self) -> Option<u64> {
        self.projected_host_bytes
            .checked_add(self.projected_device_bytes)
    }

    /// A request with no projection, for a caller that only wants to check the floor.
    #[must_use]
    pub fn new(mem_available_bytes: u64) -> Self {
        Self {
            mem_available_bytes,
            projected_host_bytes: 0,
            projected_device_bytes: 0,
        }
    }
}

/// A typed refusal. Every variant names the numbers that made it fail.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BudgetError {
    /// Host + device projection overflowed u64.
    #[error("projected budget overflows: {host} host + {device} device bytes cannot be added")]
    Overflow {
        /// The projected host bytes.
        host: u64,
        /// The projected device bytes.
        device: u64,
    },
    /// The projected usage leaves less than the 1 GiB floor.
    #[error(
        "budget refused: MemAvailable {available} B - projected {projected} B ({host} host + {device} device) leaves {remaining} B, below the {floor} B floor (short by {shortfall} B)"
    )]
    BelowFloor {
        /// The MemAvailable reading used.
        available: u64,
        /// The projected host bytes.
        host: u64,
        /// The projected device bytes.
        device: u64,
        /// The projected host + device bytes.
        projected: u64,
        /// MemAvailable - projected (saturated at zero).
        remaining: u64,
        /// The floor.
        floor: u64,
        /// floor - remaining.
        shortfall: u64,
    },
}

/// The accepted admission, with every number it used. Constructing one is only possible through
/// [BudgetAdmission::admit], so a caller cannot hand the launcher an un-admitted budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetAdmission {
    mem_available_bytes: u64,
    projected_host_bytes: u64,
    projected_device_bytes: u64,
    projected_bytes: u64,
    floor_bytes: u64,
    remaining_bytes: u64,
}

impl BudgetAdmission {
    /// The MemAvailable reading used.
    #[must_use]
    pub fn mem_available_bytes(&self) -> u64 {
        self.mem_available_bytes
    }

    /// The projected host bytes.
    #[must_use]
    pub fn projected_host_bytes(&self) -> u64 {
        self.projected_host_bytes
    }

    /// The projected device bytes.
    #[must_use]
    pub fn projected_device_bytes(&self) -> u64 {
        self.projected_device_bytes
    }

    /// Host + device projection.
    #[must_use]
    pub fn projected_bytes(&self) -> u64 {
        self.projected_bytes
    }

    /// The floor applied.
    #[must_use]
    pub fn floor_bytes(&self) -> u64 {
        self.floor_bytes
    }

    /// MemAvailable - projected.
    #[must_use]
    pub fn remaining_bytes(&self) -> u64 {
        self.remaining_bytes
    }

    /// Applies the 1 GiB headroom rule.
    ///
    /// # Errors
    /// [BudgetError::Overflow] or [BudgetError::BelowFloor], each carrying the numbers used.
    pub fn admit(request: BudgetRequest) -> Result<Self, BudgetError> {
        let projected = request.projected_bytes().ok_or(BudgetError::Overflow {
            host: request.projected_host_bytes,
            device: request.projected_device_bytes,
        })?;
        let remaining = request.mem_available_bytes.saturating_sub(projected);
        if remaining < ADMISSION_FLOOR_BYTES {
            return Err(BudgetError::BelowFloor {
                available: request.mem_available_bytes,
                host: request.projected_host_bytes,
                device: request.projected_device_bytes,
                projected,
                remaining,
                floor: ADMISSION_FLOOR_BYTES,
                shortfall: ADMISSION_FLOOR_BYTES - remaining,
            });
        }
        Ok(Self {
            mem_available_bytes: request.mem_available_bytes,
            projected_host_bytes: request.projected_host_bytes,
            projected_device_bytes: request.projected_device_bytes,
            projected_bytes: projected,
            floor_bytes: ADMISSION_FLOOR_BYTES,
            remaining_bytes: remaining,
        })
    }

    /// The report lines a stage run prints before any device call.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        vec![
            format!(
                "budget admission: MemAvailable {} B; projected host {} B + device {} B = {} B; floor {} B; remaining {} B - ADMITTED",
                self.mem_available_bytes,
                self.projected_host_bytes,
                self.projected_device_bytes,
                self.projected_bytes,
                self.floor_bytes,
                self.remaining_bytes
            ),
            "the admission decision ran before any device allocation".to_string(),
        ]
    }
}

/// Reads the live MemAvailable in bytes.
///
/// # Errors
/// The same condition as [sparkflash_kernels::cuda_driver::mem_available_kb]: no readable
/// /proc/meminfo or no MemAvailable line.
pub fn read_mem_available_bytes() -> Result<u64, String> {
    sparkflash_kernels::cuda_driver::mem_available_kb()
        .and_then(|kb| kb.checked_mul(1024))
        .ok_or_else(|| "cannot read MemAvailable from /proc/meminfo".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(available: u64, host: u64, device: u64) -> BudgetRequest {
        BudgetRequest {
            mem_available_bytes: available,
            projected_host_bytes: host,
            projected_device_bytes: device,
        }
    }

    #[test]
    fn admission_accepts_exactly_the_floor_and_refuses_one_byte_below() {
        let exact = ADMISSION_FLOOR_BYTES + 1_000 + 2_064_352;
        let accepted = BudgetAdmission::admit(request(exact, 1_000, 2_064_352))
            .expect("exactly the floor is admitted");
        assert_eq!(accepted.remaining_bytes, ADMISSION_FLOOR_BYTES);
        assert_eq!(accepted.projected_bytes, 2_065_352);

        let error = BudgetAdmission::admit(request(exact - 1, 1_000, 2_064_352))
            .expect_err("one byte below the floor is refused");
        match error {
            BudgetError::BelowFloor {
                available,
                projected,
                remaining,
                floor,
                shortfall,
                ..
            } => {
                assert_eq!(available, exact - 1);
                assert_eq!(projected, 2_065_352);
                assert_eq!(remaining, ADMISSION_FLOOR_BYTES - 1);
                assert_eq!(floor, ADMISSION_FLOOR_BYTES);
                assert_eq!(shortfall, 1);
            }
            other => panic!("expected BelowFloor, got {other:?}"),
        }
    }

    #[test]
    fn admission_refuses_when_projected_exceeds_available() {
        let error =
            BudgetAdmission::admit(request(100, 60, 50)).expect_err("projected exceeds available");
        match error {
            BudgetError::BelowFloor {
                remaining,
                shortfall,
                ..
            } => {
                assert_eq!(remaining, 0);
                assert_eq!(shortfall, ADMISSION_FLOOR_BYTES);
            }
            other => panic!("expected BelowFloor, got {other:?}"),
        }
    }

    #[test]
    fn admission_refuses_an_overflowing_projection() {
        let error = BudgetAdmission::admit(request(u64::MAX, u64::MAX, 1)).expect_err("overflow");
        assert!(matches!(error, BudgetError::Overflow { .. }));
        assert!(error.to_string().contains("overflows"));
    }

    #[test]
    fn the_printed_admission_states_every_number_it_used() {
        let admitted =
            BudgetAdmission::admit(request(ADMISSION_FLOOR_BYTES + 10_000, 4_000, 6_000))
                .expect("admitted");
        let text = admitted.report_lines().join("\n");
        assert!(text.contains(&format!(
            "MemAvailable {} B",
            ADMISSION_FLOOR_BYTES + 10_000
        )));
        assert!(text.contains("projected host 4000 B + device 6000 B = 10000 B"));
        assert!(text.contains(&format!("floor {ADMISSION_FLOOR_BYTES} B")));
        assert!(text.contains("before any device allocation"));
    }
}
