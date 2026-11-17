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

//! Checkpoint tensor names for the dynamic (non-preset) path.
//!
//! The names are the one convention [sparkflash_ops::naming] spells for every reader of a family
//! no pack transcribes (the converter, this host backend, the web's fit and the shared tensor
//! plan). This module adds the host path's own gate: the set is the parts this base executes on
//! the host, and everything else the architecture names is reported by [unsupported_parts] so the
//! caller can refuse with the exact part list instead of failing at the first missing tensor.

use sparkflash_model::{Architecture, Part};

use sparkflash_weights::WeightSource;

use sparkflash_ops::naming::TensorNames;

/// The parts the dynamic host backend executes today.
///
/// The list is [sparkflash_ops::parts::HOST_REFERENCE_PARTS] so the CLI and the web control
/// plane judge a novel checkpoint by the same capability set.
pub(crate) const DYNAMIC_PARTS: &[Part] = sparkflash_ops::parts::HOST_REFERENCE_PARTS;

/// The parts `architecture` names that the dynamic host path does not execute.
///
/// MTP and vision are optional here, exactly as in
/// [sparkflash_ops::parts::host_reference_missing]: the architecture lists a stage or tower when
/// the document declares one, but the host forward never runs either and must still serve the
/// backbone.
#[must_use]
pub(crate) fn unsupported_parts(architecture: &Architecture) -> Vec<Part> {
    architecture
        .parts()
        .iter()
        .filter(|part| !matches!(part, Part::Mtp | Part::Vision) && !DYNAMIC_PARTS.contains(part))
        .collect()
}

/// Resolves every tensor the dynamic path reads for `architecture` from `source`, in the
/// spelling the checkpoint uses ([sparkflash_ops::naming::resolve]).
///
/// # Errors
///
/// A message naming the architecture part this base does not execute, or the first tensor the
/// checkpoint's spelling names that it does not hold.
pub(crate) fn resolve<S: WeightSource + ?Sized>(
    architecture: &Architecture,
    source: &S,
) -> Result<TensorNames, String> {
    let unsupported = unsupported_parts(architecture);
    if !unsupported.is_empty() {
        let names: Vec<String> = unsupported.iter().map(ToString::to_string).collect();
        return Err(format!(
            "the checkpoint names parts the dynamic host path does not execute: {}",
            names.join(", ")
        ));
    }
    sparkflash_ops::naming::resolve(architecture, source)
}
