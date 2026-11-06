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

//! The named parts a model is assembled from.
//!
//! A model arm is not a code tree. It is an [crate::architecture::Architecture] plus the parts
//! that architecture names; a part is implemented once per target and shared by every model that
//! uses it. This module is the vocabulary both sides agree on: the architecture reports the parts
//! it needs, a build reports the parts it carries, and the difference is the bring-up work.
//!
//! The name of a part is stable and appears in diagnostics and refusal messages. A kernel, an op
//! plan and a host reference live behind the same name for every model, so a new checkpoint that
//! names an existing part needs no new Rust.

use std::collections::BTreeSet;
use std::fmt;

/// A reusable piece of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Part {
    /// The token embedding gather.
    Embedding,
    /// RMSNorm.
    RmsNorm,
    /// The vocabulary projection.
    LmHead,
    /// Multi-head latent attention (MLA), compressed or not.
    MlaAttention,
    /// CSA2 compression: the pooled KV cache and its indexer-visible keys.
    Csa2Compression,
    /// A sparse indexer that selects the keys an attention reads.
    SparseIndexer,
    /// Grouped-query attention, including a sparse or windowed variant.
    GqaAttention,
    /// A Gated DeltaNet recurrence.
    GdnScan,
    /// A Kimi Delta Attention recurrence.
    KdaScan,
    /// A dense SwiGLU MLP.
    DenseMlp,
    /// Routed experts, their router and the shared expert.
    Moe,
    /// Hyper-connections: the widened residual stream and its data-dependent mix.
    HyperConnection,
    /// Manifold-constrained hyper-connections.
    ManifoldHyperConnection,
    /// An Engram lookup table and its projections.
    Engram,
    /// The n-gram PLE table and its convolution.
    Ple,
    /// A multi-token-prediction (MTP/DSpark) stage.
    Mtp,
    /// A vision tower and its aligner.
    Vision,
}

impl Part {
    /// Every part this vocabulary knows, in declaration order.
    pub const ALL: &'static [Self] = &[
        Self::Embedding,
        Self::RmsNorm,
        Self::LmHead,
        Self::MlaAttention,
        Self::Csa2Compression,
        Self::SparseIndexer,
        Self::GqaAttention,
        Self::GdnScan,
        Self::KdaScan,
        Self::DenseMlp,
        Self::Moe,
        Self::HyperConnection,
        Self::ManifoldHyperConnection,
        Self::Engram,
        Self::Ple,
        Self::Mtp,
        Self::Vision,
    ];

    /// The stable name used in diagnostics and configuration.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::RmsNorm => "rms_norm",
            Self::LmHead => "lm_head",
            Self::MlaAttention => "mla_attention",
            Self::Csa2Compression => "csa2_compression",
            Self::SparseIndexer => "sparse_indexer",
            Self::GqaAttention => "gqa_attention",
            Self::GdnScan => "gdn_scan",
            Self::KdaScan => "kda_scan",
            Self::DenseMlp => "dense_mlp",
            Self::Moe => "moe",
            Self::HyperConnection => "hyper_connection",
            Self::ManifoldHyperConnection => "manifold_hyper_connection",
            Self::Engram => "engram",
            Self::Ple => "ple",
            Self::Mtp => "mtp",
            Self::Vision => "vision",
        }
    }

    /// The part's name, so a refusal can print the missing piece.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|part| part.name() == name)
    }
}

impl fmt::Display for Part {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A device target a part can be implemented for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Target {
    /// Blackwell GB100 (sm_100).
    Sm100,
    /// Blackwell GB120 (sm_120).
    Sm120,
    /// Blackwell GB10, the DGX Spark part this tree runs on today (sm_121a).
    Sm121,
}

impl Target {
    /// Every target this vocabulary knows.
    pub const ALL: &'static [Self] = &[Self::Sm100, Self::Sm120, Self::Sm121];

    /// The target's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sm100 => "sm100",
            Self::Sm120 => "sm120",
            Self::Sm121 => "sm121",
        }
    }

    /// Parse a target name, accepting the common spellings.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sm100" | "sm_100" | "blackwell-gb100" => Some(Self::Sm100),
            "sm120" | "sm_120" | "blackwell-gb120" => Some(Self::Sm120),
            "sm121" | "sm_121" | "sm121a" | "sm_121a" | "blackwell-gb10" => Some(Self::Sm121),
            _ => None,
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A set of parts, used to compare a model's needs with a build's catalogue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PartSet(BTreeSet<Part>);

impl PartSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a part; returns whether it was new.
    pub fn insert(&mut self, part: Part) -> bool {
        self.0.insert(part)
    }

    /// Whether the set holds a part.
    #[must_use]
    pub fn contains(&self, part: Part) -> bool {
        self.0.contains(&part)
    }

    /// How many parts the set holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The parts in stable order.
    pub fn iter(&self) -> impl Iterator<Item = Part> + '_ {
        self.0.iter().copied()
    }

    /// The parts in this set that the other set does not hold, in stable order.
    #[must_use]
    pub fn missing_from(&self, available: &Self) -> Vec<Part> {
        self.0
            .iter()
            .copied()
            .filter(|part| !available.contains(*part))
            .collect()
    }

    /// Every part in either set.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self(self.0.union(&other.0).copied().collect())
    }
}

/// The parts this source tree implements for a target.
///
/// The kernels under `kernels/` are named `*_sm121`, so every part is built for [Target::Sm121]
/// and none is built for the other targets yet. This is the model-side claim; the per-kernel
/// truth is `sparkflash_ops::parts::CATALOGUE`, which names the image and symbols behind each
/// part and marks its canonical implementation.
///
/// Where several arms implement a part, DeepSeek's implementation is canonical: it is the most
/// exercised. GLM and Qwen variants may exist beside it and may be the only implementation of a
/// part DeepSeek lacks, but they do not replace it for a model that can use it.
#[must_use]
pub fn built(target: Target) -> PartSet {
    if target == Target::Sm121 {
        Part::ALL.iter().copied().collect()
    } else {
        PartSet::new()
    }
}

/// Whether `part` is built for `target`.
#[must_use]
pub fn is_built(part: Part, target: Target) -> bool {
    built(target).contains(part)
}

impl FromIterator<Part> for PartSet {
    fn from_iter<I: IntoIterator<Item = Part>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for PartSet {
    type Item = Part;
    type IntoIter = std::collections::btree_set::IntoIter<Part>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_part_has_a_unique_stable_name() {
        let names: BTreeSet<&str> = Part::ALL.iter().map(|part| part.name()).collect();
        assert_eq!(names.len(), Part::ALL.len());
        for part in Part::ALL {
            assert_eq!(Part::parse(part.name()), Some(*part));
        }
        assert_eq!(Part::parse("csa3_attention"), None);
    }

    #[test]
    fn a_target_accepts_the_spellings_a_build_uses() {
        assert_eq!(Target::parse("sm_121a"), Some(Target::Sm121));
        assert_eq!(Target::parse("SM120"), Some(Target::Sm120));
        assert_eq!(Target::parse("sm_100"), Some(Target::Sm100));
        assert_eq!(Target::parse("hopper"), None);
    }

    #[test]
    fn the_build_catalogue_names_the_targets_the_kernels_are_compiled_for() {
        assert_eq!(built(Target::Sm121).len(), Part::ALL.len());
        assert!(built(Target::Sm120).is_empty());
        for part in Part::ALL {
            assert!(is_built(*part, Target::Sm121));
            assert!(!is_built(*part, Target::Sm120));
        }
    }

    #[test]
    fn a_set_reports_the_parts_a_build_is_missing_in_order() {
        let needed: PartSet = [Part::Moe, Part::MlaAttention, Part::KdaScan]
            .into_iter()
            .collect();
        let present: PartSet = [Part::Moe].into_iter().collect();
        assert_eq!(
            needed.missing_from(&present),
            vec![Part::MlaAttention, Part::KdaScan]
        );
        assert!(!needed.is_empty());
        assert_eq!(needed.union(&present).len(), 3);
    }
}
