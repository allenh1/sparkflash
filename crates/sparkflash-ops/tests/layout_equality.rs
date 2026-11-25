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

//! The equality harness every layout author reuses.
//!
//! A layout is accepted only when the plan it builds equals the plan the checkpoint's existing
//! description builds. [first_difference] names the first tensor that differs, by name or by
//! shape, so a failure is the one line an author has to fix rather than a diff of 76,000 rows.

use std::collections::BTreeMap;

use sparkflash_ops::ArchitectureTensorPlan;

/// The first tensor the layout plan and the reference plan disagree on.
///
/// The comparison is by tensor name, not declaration order: a layout may list a tensor where
/// it is convenient, and the checkpoint's order is the loader's business. The first difference
/// reports the missing or extra name, or the first common name whose logical shape, rank shape,
/// rank run, stored dtype or role differs. Returns `None` when the plans are equal.
#[must_use]
pub fn first_difference(
    reference: &ArchitectureTensorPlan,
    layout: &ArchitectureTensorPlan,
) -> Option<String> {
    let reference = table(reference);
    let layout = table(layout);
    for name in reference.keys() {
        if !layout.contains_key(name) {
            return Some(format!("tensor {name}: the layout does not claim it"));
        }
    }
    for name in layout.keys() {
        if !reference.contains_key(name) {
            return Some(format!(
                "tensor {name}: the layout claims it, the reference does not"
            ));
        }
    }
    for (name, expected) in &reference {
        let actual = &layout[name];
        if expected.0 != actual.0 {
            return Some(format!(
                "tensor {name}: expected stored dtype {}, layout built {}",
                expected.0, actual.0
            ));
        }
        if expected.1 != actual.1 {
            return Some(format!(
                "tensor {name}: expected shape {:?}, layout built {:?}",
                expected.1, actual.1
            ));
        }
        if expected.2 != actual.2 {
            return Some(format!(
                "tensor {name}: expected rank shape {:?}, layout built {:?}",
                expected.2, actual.2
            ));
        }
        if expected.3 != actual.3 {
            return Some(format!(
                "tensor {name}: expected rank run {:?}, layout built {:?}",
                expected.3, actual.3
            ));
        }
        if expected.4 != actual.4 {
            return Some(format!(
                "tensor {name}: expected role {}, layout built {}",
                expected.4, actual.4
            ));
        }
    }
    None
}

type Row = (
    String,
    Vec<usize>,
    Vec<usize>,
    Option<sparkflash_ops::RankRun>,
    String,
);

fn table(plan: &ArchitectureTensorPlan) -> BTreeMap<String, Row> {
    let mut out = BTreeMap::new();
    for index in 0..plan.len() {
        let spec = plan.spec_at(index);
        out.insert(
            spec.name.clone(),
            (
                format!("{:?}", spec.dtype),
                spec.shape.clone(),
                plan.rank_spec_at(index).shape.clone(),
                plan.rank_run_at(index),
                format!("{:?}", plan.plan().role(index)),
            ),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::{Architecture, TpPlan};

    #[test]
    fn a_changed_vocabulary_is_the_first_difference() {
        let text = include_str!("../../../models/configs/glm53_flash.hf.json");
        let architecture = Architecture::from_json(text).expect("the carried config");
        let tp = TpPlan::for_architecture(&architecture, 1, 0).expect("a TP plan");
        let reference =
            ArchitectureTensorPlan::checkpoint(&architecture, tp).expect("the reference plan");
        assert_eq!(first_difference(&reference, &reference), None);
        let mut changed = architecture;
        changed.vocab_size += 1;
        let layout = ArchitectureTensorPlan::checkpoint(&changed, tp).expect("the changed plan");
        let difference = first_difference(&reference, &layout).expect("a difference");
        assert!(difference.contains("lm_head.weight"), "{difference}");
        assert!(difference.contains("shape"), "{difference}");
    }
}
