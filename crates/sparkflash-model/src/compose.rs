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

//! The generic layer walk.
//!
//! Every model's hand-written forward visits the same roles in the same order: the side lookups
//! that land on the layer input, the residual mix, attention, its compression and indexer
//! sublayers, and the feed-forward block. This module turns an
//! [Architecture](crate::architecture::Architecture) into that walk from typed fields alone, so a
//! new arm adds no Rust to the driver and the driver holds no family or name branches.
//!
//! The order is the one documented in
//! [docs/engine/01-forward-order.md](../../../docs/engine/01-forward-order.md): the Engram/PLE
//! side runs before the block (section 3.1 step 7a), the block mixes its residual input before
//! attention (section 3.2), the compressor and indexer run as part of the attention step
//! (sections 3.4 and 3.5), and the FFN closes the block. [StepRole::order] is the one place that
//! order is written down.
//!
//! A step names [Part] ids and nothing else. It chooses no kernel and no implementation: resolving
//! a part to its kernel is the catalogue's job ([crate::parts::built]), which serves one canonical
//! implementation per part across every family. DeepSeek's most-proven implementation therefore
//! stays the implementation wherever DeepSeek has one, and this walk cannot steer a DeepSeek model
//! to another family's arithmetic. There is no family-name branch anywhere in this module: every
//! decision is read from the typed [Architecture](crate::architecture::Architecture) fields.

use crate::architecture::{Architecture, AttentionKind, Ffn, Layer, Residual, Side};
use crate::parts::Part;

/// One decoder layer as an ordered walk of parts.
///
/// The fields are the typed decisions an architecture already made: the attention family, the
/// optional compression and indexer, the FFN, the residual mixer and the side lookups. The walk
/// reads them; it never inspects a family name or a config string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerStep {
    /// The layer's index in the checkpoint.
    pub index: usize,
    /// The attention family this layer runs.
    pub attention: Part,
    /// CSA2 compression, on layers that pool a compressed cache.
    pub compression: Option<Part>,
    /// The sparse indexer, on layers that run one.
    pub indexer: Option<Part>,
    /// The feed-forward part.
    pub ffn: Part,
    /// The residual mixer, when the layer has more than a plain additive path.
    pub residual: Option<Part>,
    /// Lookups that land on the layer's input, in declaration order.
    pub sides: Vec<Part>,
}

impl LayerStep {
    /// The ordered roles of this step, in forward order.
    ///
    /// This is [StepRole::order] on the step itself; it exists so a caller that already holds a
    /// step does not have to name the enum to read the order.
    #[must_use]
    pub fn roles(&self) -> Vec<(StepRole, Part)> {
        StepRole::order(self)
    }

    /// Build a step from a parsed layer, reading only its typed fields.
    fn from_layer(layer: &Layer) -> Self {
        let attention = match &layer.attention.kind {
            AttentionKind::Mla { .. } => Part::MlaAttention,
            AttentionKind::Gqa { .. } => Part::GqaAttention,
            AttentionKind::Gdn(_) => Part::GdnScan,
            AttentionKind::Kda(_) => Part::KdaScan,
        };
        let compression = layer
            .attention
            .compression
            .as_ref()
            .filter(|compression| compression.ratio > 0)
            .map(|_| Part::Csa2Compression);
        let indexer = layer
            .attention
            .indexer
            .as_ref()
            .map(|_| Part::SparseIndexer);
        let ffn = match &layer.ffn {
            Ffn::Dense(_) => Part::DenseMlp,
            Ffn::Moe(_) => Part::Moe,
        };
        let residual = match &layer.residual {
            Residual::Plain => None,
            Residual::Hyper(hyper) => Some(if hyper.manifold {
                Part::ManifoldHyperConnection
            } else {
                Part::HyperConnection
            }),
        };
        let sides = layer
            .sides
            .iter()
            .map(|side| match side {
                Side::Engram(_) => Part::Engram,
                Side::Ple(_) => Part::Ple,
            })
            .collect();
        Self {
            index: layer.index,
            attention,
            compression,
            indexer,
            ffn,
            residual,
            sides,
        }
    }
}

/// The role one ordered step plays in the forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StepRole {
    /// The attention family's main arithmetic.
    Attention,
    /// CSA2 compression inside the attention step.
    Compression,
    /// The sparse indexer inside the attention step.
    Indexer,
    /// The feed-forward block.
    Ffn,
    /// The residual-stream mixer.
    Residual,
    /// A lookup on the layer input, such as Engram or the n-gram PLE.
    Side,
}

impl StepRole {
    /// Map a step to its parts in forward order.
    ///
    /// The order follows
    /// [docs/engine/01-forward-order.md](../../../docs/engine/01-forward-order.md):
    ///
    /// 1. [StepRole::Side] first, because Engram and the n-gram PLE land on the *input* of
    ///    their layer, before the block runs (section 3.1 step 7a).
    /// 2. [StepRole::Residual] next, because the block mixes its residual input before either
    ///    sublayer (section 3.2). A plain additive residual names no part and contributes
    ///    nothing here.
    /// 3. [StepRole::Attention], the sublayer itself.
    /// 4. [StepRole::Compression] then [StepRole::Indexer], the attention's optional
    ///    sublayers (sections 3.4 and 3.5). They run only when the layer carries them.
    /// 5. [StepRole::Ffn] last, closing the block.
    #[must_use]
    pub fn order(step: &LayerStep) -> Vec<(Self, Part)> {
        let mut roles = Vec::with_capacity(
            step.sides.len()
                + 2
                + usize::from(step.residual.is_some())
                + usize::from(step.compression.is_some())
                + usize::from(step.indexer.is_some()),
        );
        roles.extend(step.sides.iter().map(|part| (Self::Side, *part)));
        if let Some(part) = step.residual {
            roles.push((Self::Residual, part));
        }
        roles.push((Self::Attention, step.attention));
        if let Some(part) = step.compression {
            roles.push((Self::Compression, part));
        }
        if let Some(part) = step.indexer {
            roles.push((Self::Indexer, part));
        }
        roles.push((Self::Ffn, step.ffn));
        roles
    }
}

impl Architecture {
    /// The decoder layers as ordered steps, in ascending index order.
    #[must_use]
    pub fn steps(&self) -> Vec<LayerStep> {
        self.layers.iter().map(LayerStep::from_layer).collect()
    }

    /// Visit every layer once, in ascending index order.
    pub fn walk(&self, mut visit: impl FnMut(&LayerStep)) {
        for step in self.steps() {
            visit(&step);
        }
    }

    /// Visit every ordered role of every layer, in forward order.
    ///
    /// The first argument is the layer index, so a caller can key a plan or a buffer by layer
    /// without reconstructing it.
    pub fn walk_roles(&self, mut visit: impl FnMut(usize, StepRole, Part)) {
        for step in self.steps() {
            for (role, part) in StepRole::order(&step) {
                visit(step.index, role, part);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEEPSEEK: &str =
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
    const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

    fn architecture(json: &str) -> Architecture {
        Architecture::from_json(json).expect("released config")
    }

    fn role_parts(steps: &[LayerStep], role: StepRole) -> Vec<(usize, Part)> {
        steps
            .iter()
            .flat_map(|step| {
                step.roles()
                    .into_iter()
                    .filter(|(candidate, _)| *candidate == role)
                    .map(|(_, part)| (step.index, part))
            })
            .collect()
    }

    /// The shared assertions every released schedule must satisfy.
    fn assert_walks_every_layer_once(steps: &[LayerStep], layer_count: usize) {
        assert_eq!(steps.len(), layer_count);
        assert_eq!(steps.first().map(|step| step.index), Some(0));
        assert_eq!(
            steps.last().map(|step| step.index),
            Some(layer_count.saturating_sub(1))
        );
        let indexes: Vec<usize> = steps.iter().map(|step| step.index).collect();
        assert_eq!(indexes, (0..layer_count).collect::<Vec<_>>());
    }

    #[test]
    fn the_three_released_configs_walk_every_layer_once_in_order() {
        for (name, json, count) in [
            ("DeepSeek", DEEPSEEK, 40),
            ("GLM", GLM, 45),
            ("Qwen", QWEN, 48),
        ] {
            let architecture = architecture(json);
            let steps = architecture.steps();
            assert_walks_every_layer_once(&steps, count);

            let mut visited = Vec::new();
            architecture.walk(|step| visited.push(step.index));
            assert_eq!(visited, indexes(count), "{name} walk()");

            let mut role_layers = Vec::new();
            architecture.walk_roles(|index, role, part| {
                if role == StepRole::Attention {
                    role_layers.push((index, part));
                }
            });
            assert_eq!(role_layers, role_parts(&steps, StepRole::Attention));
        }
    }

    fn indexes(count: usize) -> Vec<usize> {
        (0..count).collect()
    }

    #[test]
    fn deepseek_compresses_only_positive_ratios_and_indexes_the_eight_sources() {
        let architecture = architecture(DEEPSEEK);
        let steps = architecture.steps();
        assert_walks_every_layer_once(&steps, 40);

        for (step, layer) in steps.iter().zip(&architecture.layers) {
            let ratio = layer
                .attention
                .compression
                .as_ref()
                .map_or(0, |compression| compression.ratio);
            assert_eq!(
                step.compression.is_some(),
                ratio > 0,
                "layer {} compression",
                step.index
            );
        }
        let compressed = role_parts(&steps, StepRole::Compression);
        assert_eq!(compressed.len(), 38);
        assert!(
            compressed
                .iter()
                .all(|(_, part)| *part == Part::Csa2Compression)
        );

        let indexers = role_parts(&steps, StepRole::Indexer);
        assert_eq!(
            indexers.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![2, 8, 14, 20, 24, 28, 32, 36]
        );
        assert!(
            indexers
                .iter()
                .all(|(_, part)| *part == Part::SparseIndexer)
        );
        assert_eq!(architecture.indexer_layers().len(), 8);
    }

    #[test]
    fn glm_alternates_kda_and_mla_and_densifies_the_first_three_layers() {
        let architecture = architecture(GLM);
        let steps = architecture.steps();
        assert_walks_every_layer_once(&steps, 45);

        for (index, step) in steps.iter().enumerate() {
            let expected = if (index + 1) % 4 == 0 {
                Part::MlaAttention
            } else {
                Part::KdaScan
            };
            assert_eq!(step.attention, expected, "layer {index} attention");
            let expected_ffn = if index < 3 { Part::DenseMlp } else { Part::Moe };
            assert_eq!(step.ffn, expected_ffn, "layer {index} ffn");
        }
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::KdaScan)
                .count(),
            34
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::MlaAttention)
                .count(),
            11
        );
        assert_eq!(role_parts(&steps, StepRole::Ffn).len(), 45);
    }

    #[test]
    fn qwen_runs_gdn_and_gqa_moe_every_layer_and_ple_once() {
        let architecture = architecture(QWEN);
        let steps = architecture.steps();
        assert_walks_every_layer_once(&steps, 48);

        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::GdnScan)
                .count(),
            36
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.attention == Part::GqaAttention)
                .count(),
            12
        );
        assert!(steps.iter().all(|step| step.ffn == Part::Moe));
        for (index, step) in steps.iter().enumerate() {
            let expected = if (index + 1) % 4 == 0 {
                Part::GqaAttention
            } else {
                Part::GdnScan
            };
            assert_eq!(step.attention, expected, "layer {index} attention");
        }

        let ple = role_parts(&steps, StepRole::Side);
        assert_eq!(ple, vec![(1, Part::Ple)]);
        assert_eq!(
            steps
                .iter()
                .filter(|step| step.sides.contains(&Part::Ple))
                .count(),
            1
        );
    }

    #[test]
    fn a_side_runs_before_attention_and_ffn_and_residual_sits_between() {
        let qwen = architecture(QWEN);
        let qwen_step = qwen
            .steps()
            .into_iter()
            .find(|step| step.index == 1)
            .expect("layer 1");
        assert_eq!(
            qwen_step.roles(),
            vec![
                (StepRole::Side, Part::Ple),
                (StepRole::Residual, Part::HyperConnection),
                (StepRole::Attention, Part::GdnScan),
                (StepRole::Ffn, Part::Moe),
            ]
        );
        let side = qwen_step
            .roles()
            .iter()
            .position(|(role, _)| *role == StepRole::Side)
            .expect("side");
        let attention = qwen_step
            .roles()
            .iter()
            .position(|(role, _)| *role == StepRole::Attention)
            .expect("attention");
        let ffn = qwen_step
            .roles()
            .iter()
            .position(|(role, _)| *role == StepRole::Ffn)
            .expect("ffn");
        assert!(side < attention && attention < ffn);

        let deepseek = architecture(DEEPSEEK);
        let deepseek_step = deepseek
            .steps()
            .into_iter()
            .find(|step| step.index == 1)
            .expect("layer 1");
        assert_eq!(
            deepseek_step.roles(),
            vec![
                (StepRole::Side, Part::Engram),
                (StepRole::Residual, Part::HyperConnection),
                (StepRole::Attention, Part::MlaAttention),
                (StepRole::Ffn, Part::Moe),
            ]
        );
    }

    #[test]
    fn the_walk_visits_every_layer_exactly_once_in_ascending_order() {
        for (json, count) in [(DEEPSEEK, 40), (GLM, 45), (QWEN, 48)] {
            let architecture = architecture(json);
            let mut seen = vec![0usize; count];
            architecture.walk(|step| {
                assert!(step.index < count, "walk produced index {}", step.index);
                seen[step.index] += 1;
            });
            assert_eq!(seen, vec![1usize; count]);
        }
    }
}
