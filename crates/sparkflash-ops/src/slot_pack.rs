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

//! A plan-order rank-file converter over [ArchitectureTensorPlan].
//!
//! This is **not** the arms' loader. The Qwen and GLM device walks write their own
//! [sparkflash_weights::rank_file::SlotIndex] files in walk order, including derived pointer
//! tables and fused slots; their loaders ([crate::slot_pack] is not one of them) replay that
//! walk. [SlotPack] is the plan-level converter: it places every planned tensor in plan order
//! and streams its bytes with the shared [SlotFileWriter]. It supports TP=1 only, because the
//! plan-order slots are the whole stored tensors; at TP>1 an arm walk's rank slices and derived
//! tables are not this plan order. [SlotPack::new] refuses a non-TP1 plan rather than writing
//! a file no arm loader can read.

use std::path::Path;

use sparkflash_weights::rank_file::{
    ConvertedPreamble, SlotFileWriter, SlotIndex, SlotPlan, SlotRecords, WrittenRank,
};
use sparkflash_weights::tp::TpLayout;
use sparkflash_weights::{WeightSource, check_length};

use crate::tensor_plan::ArchitectureTensorPlan;

/// One plan-ordered slot: a tensor by its released name and this rank's stored spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackSlot {
    /// The released tensor name.
    pub key: String,
    /// The rank-local spec. [SlotPack] only builds at TP=1, so this is the checkpoint spec.
    pub spec: sparkflash_weights::TensorSpec,
}

impl PackSlot {
    /// The slot's payload length in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.spec.storage_bytes() as usize
    }
}

/// The rank-file converter over one architecture's plan.
#[derive(Debug, Clone)]
pub struct SlotPack<X> {
    slots: Vec<PackSlot>,
    extra: X,
}

impl<X: SlotRecords + Clone> SlotPack<X> {
    /// Build the pack from a TP1 plan, in plan order, recording `extra` beside the slots.
    ///
    /// # Errors
    /// A plan at a world larger than one: plan-order slots are whole stored tensors, while an
    /// arm walk at TP>1 uploads rank slices and derived tables this converter does not model.
    pub fn new(plan: &ArchitectureTensorPlan, extra: X) -> Result<Self, String> {
        if plan.tp().world() != 1 {
            return Err(format!(
                "SlotPack is the TP1 plan-order converter; this plan is TP{}/rank{} and the \
                 arm's own walk loader must read it",
                plan.tp().world(),
                plan.tp().rank()
            ));
        }
        let slots = (0..plan.len())
            .map(|index| PackSlot {
                key: plan.spec_at(index).name.clone(),
                spec: plan.rank_spec_at(index).clone(),
            })
            .collect();
        Ok(Self { slots, extra })
    }

    /// The planned slots, in plan order.
    #[must_use]
    pub fn slots(&self) -> &[PackSlot] {
        &self.slots
    }

    /// The extra records the index carries.
    #[must_use]
    pub const fn extra(&self) -> &X {
        &self.extra
    }

    /// Write rank `layout.rank`'s file from `source` into `output_dir`.
    ///
    /// # Errors
    /// A source tensor the plan names and the checkpoint lacks, a length mismatch, or a file the
    /// container refuses.
    pub fn write(
        &self,
        source: &dyn WeightSource,
        output_dir: &Path,
        layout: TpLayout,
        label: &str,
    ) -> Result<WrittenRank, String> {
        let mut plan = SlotPlan::default();
        for slot in &self.slots {
            plan.place(&slot.key, slot.bytes());
        }
        let index = plan.into_index(layout, label, self.extra.clone());
        let mut writer = SlotFileWriter::create(output_dir, index)?;
        for slot in &self.slots {
            let bytes = source.read(&slot.key).map_err(|error| error.to_string())?;
            check_length(&slot.key, slot.bytes(), &bytes)?;
            writer.put(&slot.key, slot.bytes(), &bytes)?;
        }
        writer.finish()
    }

    /// Read a rank file's preamble and index, proving the layout.
    ///
    /// # Errors
    /// The container's read or verification refusal.
    pub fn read(
        path: &Path,
        layout: TpLayout,
    ) -> Result<(ConvertedPreamble, SlotIndex<X>), String> {
        SlotIndex::read(path, layout)
    }
}
