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

//! The one widening, the boot format ledger, and the bytes-per-token floor.
//!
//! [crate::formats] makes a weight's format a type, so a body cannot be handed the wrong one.
//! Types cannot see one thing: a load path may widen a stored tensor and hand the wider matrix
//! to a body of the same type, because by then the formats agree again. The plan's rule is that
//! there is **no conversion between formats except one** - [widen] - which takes a reason and
//! writes a [WidenRecord]. Boot builds a [FormatLedger] from each part's stored and arithmetic
//! formats; a part whose arithmetic differs from its stored format without a matching record is
//! refused, and a recorded one prints its reason.
//!
//! The ledger also states the bytes read per token for each part. A widening can be exact and
//! still slow, so [bytes_per_token_floor] compares that figure with the stored-bytes floor of the
//! [EnginePlan]: the bytes the rank's checkpoint actually stores. Reading more than that means a
//! part is read as a wider format than it is stored as, which is exactly the case the types
//! cannot catch.
//!
//! Neither type needs a device. A plan, a table of formats and a log are host values, so the
//! boot check and its tests run on the CPU.

use std::fmt;

use thiserror::Error;

use crate::engine_plan::{EnginePlan, PlanError};
use crate::formats::{Format, FormatKind};
use crate::tensor_plan::ArchitectureTensorPlan;

/// One recorded widening: a part stored as one format and read at another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WidenRecord {
    /// The part the record covers.
    pub part: String,
    /// The format the checkpoint stores.
    pub stored: FormatKind,
    /// The arithmetic format the boot path reads.
    pub arithmetic: FormatKind,
    /// Why the widening is allowed.
    pub reason: String,
}

/// The records a boot path has written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WidenLog {
    records: Vec<WidenRecord>,
}

impl WidenLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The records, in the order they were written.
    #[must_use]
    pub fn records(&self) -> &[WidenRecord] {
        &self.records
    }

    /// Whether a record covers this part and these two formats.
    #[must_use]
    pub fn has(&self, part: &str, stored: FormatKind, arithmetic: FormatKind) -> bool {
        self.records.iter().any(|record| {
            record.part == part && record.stored == stored && record.arithmetic == arithmetic
        })
    }

    fn push(&mut self, record: WidenRecord) {
        self.records.push(record);
    }
}

/// The one conversion between formats: a part stored as [Format] `F` is read at [Format] `G`'s
/// arithmetic.
///
/// It moves no bytes; it writes the record that lets boot accept the load path's decision. A
/// blank reason is refused, because the record exists to say why an exactness or a speed cost is
/// taken.
///
/// # Errors
/// A blank reason, or a widening from a format to itself.
pub fn widen<F: Format, G: Format>(
    part: &str,
    reason: &str,
    log: &mut WidenLog,
) -> Result<WidenRecord, WidenError> {
    if F::KIND == G::KIND {
        return Err(WidenError::SameFormat {
            part: part.to_owned(),
            format: F::KIND.name(),
        });
    }
    if reason.trim().is_empty() {
        return Err(WidenError::NoReason {
            part: part.to_owned(),
            stored: F::KIND.name(),
            arithmetic: G::KIND.name(),
        });
    }
    let record = WidenRecord {
        part: part.to_owned(),
        stored: F::KIND,
        arithmetic: G::KIND,
        reason: reason.to_owned(),
    };
    log.push(record.clone());
    Ok(record)
}

/// One part's row in the format ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerRow {
    /// The part's name, as boot and the checkpoint spell it.
    pub part: String,
    /// The format the checkpoint stores.
    pub stored: FormatKind,
    /// The arithmetic format the run reads.
    pub arithmetic: FormatKind,
    /// The stored bytes this part reads for one token.
    pub bytes_per_token: u128,
}

impl LedgerRow {
    /// One part's row.
    #[must_use]
    pub fn new(
        part: impl Into<String>,
        stored: FormatKind,
        arithmetic: FormatKind,
        bytes_per_token: u128,
    ) -> Self {
        Self {
            part: part.into(),
            stored,
            arithmetic,
            bytes_per_token,
        }
    }
}

/// The format ledger boot prints, after refusing every unrecorded widening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatLedger {
    rows: Vec<LedgerRow>,
    records: Vec<WidenRecord>,
}

impl FormatLedger {
    /// Build the ledger: every part whose arithmetic format differs from its stored format must
    /// have a matching [WidenRecord].
    ///
    /// # Errors
    /// The first part read at a format its checkpoint does not store, with no record.
    pub fn boot(rows: Vec<LedgerRow>, log: &WidenLog) -> Result<Self, WidenError> {
        for row in &rows {
            if row.stored != row.arithmetic && !log.has(&row.part, row.stored, row.arithmetic) {
                return Err(WidenError::Unrecorded {
                    part: row.part.clone(),
                    stored: row.stored.name(),
                    arithmetic: row.arithmetic.name(),
                });
            }
        }
        Ok(Self {
            rows,
            records: log.records.clone(),
        })
    }

    /// The ledger's rows.
    #[must_use]
    pub fn rows(&self) -> &[LedgerRow] {
        &self.rows
    }

    /// The widening records the ledger accepted.
    #[must_use]
    pub fn records(&self) -> &[WidenRecord] {
        &self.records
    }
}

impl fmt::Display for FormatLedger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "format ledger (part, stored, arithmetic, bytes/token):")?;
        for row in &self.rows {
            writeln!(
                f,
                "  {:<44} {:<8} {:<8} {}",
                row.part,
                row.stored.name(),
                row.arithmetic.name(),
                row.bytes_per_token
            )?;
            if row.stored != row.arithmetic
                && let Some(record) = self.records.iter().find(|record| {
                    record.part == row.part
                        && record.stored == row.stored
                        && record.arithmetic == row.arithmetic
                })
            {
                writeln!(
                    f,
                    "      widened {} -> {}: {}",
                    record.stored.name(),
                    record.arithmetic.name(),
                    record.reason
                )?;
            }
        }
        Ok(())
    }
}

/// The stored-bytes floor of a plan: the bytes this rank's checkpoint holds.
///
/// This is the least a read that touches no redundant copy can touch and the ceiling a read may
/// not exceed for one token without reading more than the checkpoint stores. It comes from the
/// same [ArchitectureTensorPlan] the boot path uploads, so it is the plan's own accounting
/// rather than a second estimate.
///
/// # Errors
/// A rank whose tensor plan does not build, as [ArchitectureTensorPlan::served_and_mtp].
pub fn stored_bytes_floor(plan: &EnginePlan) -> Result<u128, PlanError> {
    Ok(ArchitectureTensorPlan::served_and_mtp(plan.architecture(), plan.tp())?.rank_bytes())
}

/// One measured bytes-per-token figure against the plan's stored-bytes floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytesPerToken {
    /// The bytes the ledger says the run reads for a token.
    pub measured: u128,
    /// The plan's stored-bytes floor.
    pub floor: u128,
}

impl BytesPerToken {
    /// Whether the measured reads stay within the checkpoint's stored bytes.
    #[must_use]
    pub const fn within_floor(self) -> bool {
        self.measured <= self.floor
    }

    /// The bytes read above the floor (zero when within it).
    #[must_use]
    pub const fn excess(self) -> u128 {
        self.measured.saturating_sub(self.floor)
    }
}

/// Compare a ledger's bytes read per token with the plan's stored-bytes floor.
///
/// A recorded widening is exact but not free: it may read BF16 words where the checkpoint
/// stores E4M3. The comparison is the generated test's second half, after the types have
/// allowed the widening and the ledger has recorded it.
///
/// # Errors
/// A plan whose tensor plan does not build.
pub fn bytes_per_token_floor(
    plan: &EnginePlan,
    measured: u128,
) -> Result<BytesPerToken, PlanError> {
    Ok(BytesPerToken {
        measured,
        floor: stored_bytes_floor(plan)?,
    })
}

/// A boot ledger or widening that cannot be accepted.
#[derive(Debug, Error)]
pub enum WidenError {
    /// The widening gives no reason.
    #[error("widening `{part}` from {stored} to {arithmetic} needs a reason")]
    NoReason {
        /// The part being widened.
        part: String,
        /// The stored format.
        stored: &'static str,
        /// The arithmetic format.
        arithmetic: &'static str,
    },
    /// A widening names the same format twice.
    #[error("widening `{part}` from {format} to itself is not a conversion")]
    SameFormat {
        /// The part being widened.
        part: String,
        /// The format on both sides.
        format: &'static str,
    },
    /// Boot found a widening with no record.
    #[error(
        "`{part}` is stored as {stored} and read as {arithmetic}, and no `widen` record          says why"
    )]
    Unrecorded {
        /// The part read at another format.
        part: String,
        /// The stored format.
        stored: &'static str,
        /// The arithmetic format.
        arithmetic: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use sparkflash_model::Target;

    use super::*;
    use crate::formats::{Bf16, NvFp4};

    const GLM: &str = include_str!("../../../models/configs/glm53_flash.hf.json");

    #[test]
    fn a_widen_needs_a_reason_and_writes_a_record() {
        let mut log = WidenLog::new();
        let error = widen::<NvFp4, Bf16>("layers.0.mlp.up", "  ", &mut log)
            .expect_err("a blank reason is refused");
        assert!(error.to_string().contains("needs a reason"));
        assert!(log.records().is_empty());

        let record = widen::<NvFp4, Bf16>(
            "layers.0.mlp.up",
            "the released activation scale is f32",
            &mut log,
        )
        .expect("a recorded widening");
        assert_eq!(record.stored, FormatKind::NvFp4);
        assert_eq!(record.arithmetic, FormatKind::Bf16);
        assert!(log.has("layers.0.mlp.up", FormatKind::NvFp4, FormatKind::Bf16));
        assert_eq!(log.records().len(), 1);

        let same = widen::<Bf16, Bf16>("norm", "no-op", &mut log)
            .expect_err("a same-format widen is not a conversion");
        assert!(same.to_string().contains("is not a conversion"));
    }

    #[test]
    fn boot_refuses_an_unrecorded_widen_and_prints_a_recorded_one() {
        let rows = vec![LedgerRow::new(
            "layers.0.mlp.down",
            FormatKind::NvFp4,
            FormatKind::Bf16,
            4096,
        )];
        let error = FormatLedger::boot(rows.clone(), &WidenLog::new())
            .expect_err("an unrecorded widening is refused");
        let message = error.to_string();
        assert!(message.contains("layers.0.mlp.down"));
        assert!(message.contains("nvfp4"));
        assert!(message.contains("bf16"));

        let mut log = WidenLog::new();
        widen::<NvFp4, Bf16>(
            "layers.0.mlp.down",
            "the router bias is an exact bf16 widen",
            &mut log,
        )
        .expect("the recorded widening");
        let ledger = FormatLedger::boot(rows, &log).expect("a recorded widening boots");
        let printed = ledger.to_string();
        assert!(printed.contains("format ledger"));
        assert!(printed.contains("layers.0.mlp.down"));
        assert!(printed.contains("4096"));
        assert!(printed.contains("widened nvfp4 -> bf16"));
        assert!(printed.contains("the router bias is an exact bf16 widen"));
    }

    #[test]
    fn the_ledger_floor_is_the_plans_stored_bytes() {
        let plan = EnginePlan::from_json(GLM, Target::Sm121, 1, 0).expect("GLM TP1");
        let floor = stored_bytes_floor(&plan).expect("the tensor plan builds");
        assert!(floor > 0, "a released plan stores weights");

        let exact = bytes_per_token_floor(&plan, floor).expect("the plan");
        assert!(exact.within_floor());
        assert_eq!(exact.excess(), 0);

        let over = bytes_per_token_floor(&plan, floor + 1).expect("the plan");
        assert!(!over.within_floor());
        assert_eq!(over.excess(), 1);
    }
}
