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

//! A tensor plan: every name a checkpoint carries, with its spec and the role a model pack gives it.
//!
//! A pack derives its plan from the configuration rather than transcribing it from the index, so
//! comparing the two is a real check: a shape the pack believes can disagree with the shard headers,
//! and the byte total of a complete plan pins the released index's `total_size`. The bookkeeping -
//! name lookup, role byte totals, and the two conformance checks - is the same for every pack; only
//! the names, shapes and the role enum are the pack's.

use std::collections::{BTreeMap, BTreeSet};

use crate::dtype::DType;
use crate::linalg::ScaleGrid;
use crate::nvfp4::NVFP4_GROUP;
use crate::spec::TensorSpec;

/// The planned tensors of one configuration, in construction order, each tagged with a role `R`.
#[derive(Debug, Clone)]
pub struct TensorPlan<R> {
    specs: Vec<TensorSpec>,
    roles: Vec<R>,
    by_name: BTreeMap<String, usize>,
}

impl<R> Default for TensorPlan<R> {
    fn default() -> Self {
        Self {
            specs: Vec::new(),
            roles: Vec::new(),
            by_name: BTreeMap::new(),
        }
    }
}

impl<R: Copy + PartialEq> TensorPlan<R> {
    /// Plan one tensor.
    ///
    /// # Panics
    /// When `name` is already planned: a pack that names a tensor twice has a bug, and the second
    /// spec would silently shadow the first.
    pub fn push(&mut self, name: &str, dtype: DType, shape: Vec<usize>, role: R) {
        assert!(
            !self.by_name.contains_key(name),
            "duplicate tensor name {name}"
        );
        self.by_name.insert(name.to_owned(), self.specs.len());
        self.specs.push(TensorSpec {
            name: name.to_owned(),
            dtype,
            shape,
        });
        self.roles.push(role);
    }

    /// Plan one ModelOpt NVFP4 weight of logical shape `[rows, columns]` under `prefix`: the packed
    /// codes, the E4M3 block scales, and the two per-tensor F32 scales ([crate::nvfp4]).
    ///
    /// The codes keep their logical shape, as every FP4 spec does; the container reports them as
    /// `U8` with the last dimension halved, and the loader maps that to this spec.
    pub fn push_nvfp4(&mut self, prefix: &str, rows: usize, columns: usize, role: R) {
        self.push(
            &format!("{prefix}.weight"),
            DType::F4E2M1,
            vec![rows, columns],
            role,
        );
        self.push(
            &format!("{prefix}.weight_scale"),
            DType::F8E4M3,
            vec![rows, columns.div_ceil(NVFP4_GROUP)],
            role,
        );
        self.push(
            &format!("{prefix}.weight_scale_2"),
            DType::F32,
            vec![],
            role,
        );
        self.push(&format!("{prefix}.input_scale"), DType::F32, vec![], role);
    }

    /// Plan one block-scaled FP8 weight of logical shape `[rows, columns]` under `prefix`: the E4M3
    /// codes and the `scale_dtype` `weight_scale_inv` grid of one scale per `block x block` tile
    /// ([ScaleGrid::Tile]).
    pub fn push_fp8_block(
        &mut self,
        prefix: &str,
        rows: usize,
        columns: usize,
        block: usize,
        scale_dtype: DType,
        role: R,
    ) {
        self.push(
            &format!("{prefix}.weight"),
            DType::F8E4M3,
            vec![rows, columns],
            role,
        );
        self.push(
            &format!("{prefix}.weight_scale_inv"),
            scale_dtype,
            ScaleGrid::Tile(block).shape(rows, columns).to_vec(),
            role,
        );
    }

    /// Every planned tensor, in construction order.
    #[must_use]
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    /// The role of the tensor at `index`.
    #[must_use]
    pub fn role(&self, index: usize) -> R {
        self.roles[index]
    }

    /// One tensor by its released name.
    #[must_use]
    pub fn spec(&self, name: &str) -> Option<&TensorSpec> {
        self.by_name.get(name).map(|&index| &self.specs[index])
    }

    /// Where [TensorPlan::specs] lists one tensor, by its released name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// The role of one tensor by its released name.
    #[must_use]
    pub fn role_of(&self, name: &str) -> Option<R> {
        self.by_name.get(name).map(|&index| self.roles[index])
    }

    /// Every planned name, in construction order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.iter().map(|spec| spec.name.as_str())
    }

    /// The number of planned tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the plan is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// The exact byte total of the plan as stored by the checkpoint.
    #[must_use]
    pub fn storage_bytes(&self) -> u128 {
        self.specs.iter().map(TensorSpec::storage_bytes).sum()
    }

    /// The byte total of one role.
    #[must_use]
    pub fn role_bytes(&self, role: R) -> u128 {
        self.specs
            .iter()
            .zip(&self.roles)
            .filter(|(_, r)| **r == role)
            .map(|(spec, _)| spec.storage_bytes())
            .sum()
    }

    /// The presence of every tensor a name list expects, and nothing outside it.
    ///
    /// # Errors
    /// A human-readable description of the first missing or unexpected name.
    pub fn conformance(&self, names: impl IntoIterator<Item = String>) -> Result<(), String> {
        let seen: BTreeSet<String> = names.into_iter().collect();
        for name in &seen {
            if !self.by_name.contains_key(name.as_str()) {
                return Err(format!(
                    "the checkpoint has tensor {name}, which the plan omits"
                ));
            }
        }
        self.require(&seen)
    }

    /// The presence of every tensor this plan needs, with extras accepted when another plan knows
    /// them, and the number of such extras.
    ///
    /// A released checkpoint is often a superset of what a pack serves: an MTP stage or a vision
    /// tower the pack plans but does not load. Planning them in `known` makes their names expected
    /// extras rather than mismatches - and a name neither plan knows is still refused, which is
    /// what keeps this a check on the checkpoint rather than on itself.
    ///
    /// # Errors
    /// A human-readable description of the first missing or unexpected name.
    pub fn conformance_with_unserved(
        &self,
        names: impl IntoIterator<Item = String>,
        known: &Self,
    ) -> Result<usize, String> {
        let seen: BTreeSet<String> = names.into_iter().collect();
        let mut unserved = 0_usize;
        for name in &seen {
            if self.by_name.contains_key(name.as_str()) {
                continue;
            }
            if known.by_name.contains_key(name.as_str()) {
                unserved += 1;
                continue;
            }
            return Err(format!(
                "the checkpoint has tensor {name}, which neither the served plan nor the \
                 whole-checkpoint plan names"
            ));
        }
        self.require(&seen)?;
        Ok(unserved)
    }

    fn require(&self, seen: &BTreeSet<String>) -> Result<(), String> {
        for name in self.names() {
            if !seen.contains(name) {
                return Err(format!(
                    "the plan requires tensor {name}, which the checkpoint lacks"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Part {
        Dense,
        Expert,
    }

    fn plan() -> TensorPlan<Part> {
        let mut plan = TensorPlan::default();
        plan.push("embed.weight", DType::BF16, vec![8, 4], Part::Dense);
        plan.push_nvfp4("experts.0.up", 4, 32, Part::Expert);
        plan
    }

    #[test]
    fn an_nvfp4_weight_is_four_tensors_of_the_modelopt_layout() {
        let plan = plan();
        assert_eq!(plan.len(), 5);
        let codes = plan
            .spec("experts.0.up.weight")
            .expect("the codes are planned");
        assert_eq!(
            (codes.dtype, codes.shape.clone()),
            (DType::F4E2M1, vec![4, 32])
        );
        let scales = plan
            .spec("experts.0.up.weight_scale")
            .expect("the scales are planned");
        assert_eq!(
            (scales.dtype, scales.shape.clone()),
            (DType::F8E4M3, vec![4, 2])
        );
        // 64 bytes of codes, 8 of scales and two F32 scalars.
        assert_eq!(plan.role_bytes(Part::Expert), 64 + 8 + 4 + 4);
        assert_eq!(plan.role_bytes(Part::Dense), 64);
        assert_eq!(plan.storage_bytes(), 64 + 80);
        assert_eq!(plan.role_of("experts.0.up.input_scale"), Some(Part::Expert));
    }

    #[test]
    fn a_block_fp8_weight_is_codes_and_a_tile_grid() {
        let mut plan = plan();
        plan.push_fp8_block("shared.down", 300, 256, 128, DType::F32, Part::Dense);
        let scales = plan
            .spec("shared.down.weight_scale_inv")
            .expect("the scales are planned");
        assert_eq!(
            (scales.dtype, scales.shape.clone()),
            (DType::F32, vec![3, 2])
        );
        // 64 bytes of embedding, 76,800 codes and six F32 scales.
        assert_eq!(plan.role_bytes(Part::Dense), 64 + 300 * 256 + 6 * 4);
    }

    #[test]
    #[should_panic(expected = "duplicate tensor name")]
    fn a_name_planned_twice_is_a_bug() {
        let mut plan = plan();
        plan.push("embed.weight", DType::BF16, vec![8, 4], Part::Dense);
    }

    #[test]
    fn conformance_refuses_extras_and_gaps() {
        let plan = plan();
        let names = || plan.names().map(str::to_owned).collect::<Vec<_>>();
        plan.conformance(names())
            .expect("the plan's own names conform");
        let mut extra = names();
        extra.push("vision.patch.weight".to_owned());
        assert!(plan.conformance(extra).is_err());
        let gap: Vec<String> = names().into_iter().skip(1).collect();
        assert!(plan.conformance(gap).unwrap_err().contains("embed.weight"));
    }

    #[test]
    fn unserved_extras_are_accepted_only_when_known() {
        let served = plan();
        let mut known = plan();
        known.push("mtp.fc.weight", DType::BF16, vec![4, 4], Part::Dense);
        let names: Vec<String> = known.names().map(str::to_owned).collect();
        assert_eq!(
            served.conformance_with_unserved(names.clone(), &known),
            Ok(1)
        );
        let mut unknown = names;
        unknown.push("not_a_tensor".to_owned());
        assert!(
            served
                .conformance_with_unserved(unknown, &known)
                .unwrap_err()
                .contains("not_a_tensor")
        );
    }
}
