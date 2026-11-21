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

//! The model description: stacks of blocks of part instances.
//!
//! A checkpoint is not one flat list of parts. It is a language stack, and when it carries them
//! a vision stack, an audio stack and a draft head. Each stack is an ordered list of blocks, and
//! each block an ordered list of the part instances it runs. This module is the vocabulary the
//! scan (milestone 1) and the description-driven forward (milestone 2) share.
//!
//! Step 0 of that work lands the types and [Architecture::stacks] for the language stack only:
//! the decoder blocks as the shared walk already orders them. The vision, audio and draft stacks,
//! the global parts, and the tensor names and stored shapes a layout file refines arrive with
//! the layout reader. Nothing here names a model or reads a family string: the description is
//! built from the checkpoint's typed fields.

use std::fmt;

use crate::architecture::{Architecture, AttentionKind, Ffn, Layer, WeightFormat};
use crate::compose::LayerStep;
use crate::parts::Part;

/// Which stack of a model description a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StackKind {
    /// The decoder stack that produces the language model's hidden states.
    Language,
    /// The image tower and its aligner.
    Vision,
    /// The audio tower and its aligner.
    Audio,
    /// The speculative draft head.
    Draft,
}

impl StackKind {
    /// Every stack kind this vocabulary knows.
    pub const ALL: &'static [Self] = &[Self::Language, Self::Vision, Self::Audio, Self::Draft];

    /// The stable name used in diagnostics and the scan report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Language => "language",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Draft => "draft",
        }
    }
}

impl fmt::Display for StackKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A part's shape, as the description knows it.
///
/// An empty shape means the step that owns the part has not stated one yet: the layout reader
/// fills the stored tensor shapes, and the scan fills the rank-local execution shape. It is
/// never silently treated as a zero-sized shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// A shape no source has stated yet.
    #[must_use]
    pub const fn unknown() -> Self {
        Self { dims: Vec::new() }
    }

    /// A shape with the given dimensions, outermost first.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// The dimensions, outermost first.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Whether no source has stated this shape.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        self.dims.is_empty()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return formatter.write_str("?");
        }
        for (index, dim) in self.dims.iter().enumerate() {
            if index > 0 {
                formatter.write_str("x")?;
            }
            write!(formatter, "{dim}")?;
        }
        Ok(())
    }
}

/// One instance of a part in a block: what it is, the shape it runs at, how its weights are
/// stored, and the tensor names it takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartInstance {
    /// The part kind.
    pub part: Part,
    /// The shape the part runs at, when the description states one.
    pub shape: Shape,
    /// The stored weight format.
    pub format: WeightFormat,
    /// The checkpoint tensor names this instance takes, in the layout's order.
    pub tensors: Vec<String>,
}

impl PartInstance {
    /// An instance with no tensor names yet.
    #[must_use]
    pub fn new(part: Part, shape: Shape, format: WeightFormat) -> Self {
        Self {
            part,
            shape,
            format,
            tensors: Vec::new(),
        }
    }

    /// The instance with the tensor names it takes.
    #[must_use]
    pub fn with_tensors(mut self, tensors: Vec<String>) -> Self {
        self.tensors = tensors;
        self
    }
}

/// One block: the ordered part instances it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// The block's index in its stack.
    pub index: usize,
    /// The parts, in forward order.
    pub parts: Vec<PartInstance>,
}

/// One stack of a model description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stack {
    /// Which stack this is.
    pub kind: StackKind,
    /// The blocks, in forward order.
    pub blocks: Vec<Block>,
}

impl Stack {
    /// An empty stack of the given kind.
    #[must_use]
    pub fn new(kind: StackKind) -> Self {
        Self {
            kind,
            blocks: Vec::new(),
        }
    }

    /// Every part instance in the stack, in block then part order.
    pub fn parts(&self) -> impl Iterator<Item = &PartInstance> {
        self.blocks.iter().flat_map(|block| block.parts.iter())
    }
}

impl Architecture {
    /// Every stack this description names.
    ///
    /// Step 0 returns the language stack only, built from the decoder layers and the shared
    /// walk's part order. The other stacks and the global parts are added by the layout work;
    /// no caller may assume this list is complete yet.
    #[must_use]
    pub fn stacks(&self) -> Vec<Stack> {
        let blocks = self
            .layers
            .iter()
            .map(|layer| self.language_block(layer))
            .collect();
        vec![Stack {
            kind: StackKind::Language,
            blocks,
        }]
    }

    /// The stored format of a part, read from the checkpoint's typed quantization block.
    ///
    /// The embedding, the norms and the head are stored unquantized in every released
    /// checkpoint; the experts and the MTP stage may name a format of their own. Every other
    /// part follows the dense projections.
    #[must_use]
    pub fn stored_format(&self, part: Part) -> WeightFormat {
        match part {
            Part::Embedding | Part::RmsNorm | Part::LayerNorm | Part::LmHead => WeightFormat::Bf16,
            Part::Moe => self.quantization.expert_format,
            Part::Mtp => self
                .quantization
                .mtp_expert_format
                .unwrap_or(self.quantization.expert_format),
            _ => self.quantization.dense_format,
        }
    }

    fn language_block(&self, layer: &Layer) -> Block {
        let step = LayerStep::from_layer(layer);
        let parts = step
            .roles()
            .into_iter()
            .map(|(_, part)| {
                PartInstance::new(part, self.part_shape(part, layer), self.stored_format(part))
            })
            .collect();
        Block {
            index: layer.index,
            parts,
        }
    }

    /// The shape the config alone states for one decoder part.
    ///
    /// This is the part's principal matrix, not a substitute for the layout's per-tensor
    /// shapes: a part with several tensors gets them from the layout file, and this shape is
    /// the one the scan can state before that file exists.
    fn part_shape(&self, part: Part, layer: &Layer) -> Shape {
        let dims = match part {
            Part::RmsNorm | Part::LayerNorm => vec![self.hidden_size],
            Part::MlaAttention | Part::GqaAttention => {
                vec![layer.attention.heads, layer.attention.head_dim]
            }
            Part::GdnScan => match &layer.attention.kind {
                AttentionKind::Gdn(gdn) => vec![gdn.key_heads, gdn.key_dim],
                _ => Vec::new(),
            },
            Part::KdaScan => match &layer.attention.kind {
                AttentionKind::Kda(kda) => vec![kda.heads, kda.head_dim],
                _ => Vec::new(),
            },
            Part::DenseMlp => match &layer.ffn {
                Ffn::Dense(mlp) => vec![mlp.intermediate, self.hidden_size],
                Ffn::Moe(_) => Vec::new(),
            },
            Part::Moe => match &layer.ffn {
                Ffn::Moe(moe) => vec![moe.routed_experts, moe.intermediate, self.hidden_size],
                Ffn::Dense(_) => Vec::new(),
            },
            _ => Vec::new(),
        };
        Shape::new(dims.into_iter().filter(|dim| *dim > 0).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"model_type":"","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":8,"max_position_embeddings":16}"#;

    #[test]
    fn the_language_stack_walks_the_decoder_blocks_in_order() {
        let architecture = Architecture::from_json(CONFIG).expect("a synthetic config");
        let stacks = architecture.stacks();
        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0].kind, StackKind::Language);
        assert_eq!(stacks[0].blocks.len(), architecture.layers.len());
        assert_eq!(stacks[0].blocks[0].index, 0);
        assert_eq!(stacks[0].blocks[1].index, 1);
        let parts: Vec<Part> = stacks[0].parts().map(|instance| instance.part).collect();
        assert!(parts.contains(&Part::GqaAttention));
        assert!(parts.contains(&Part::DenseMlp));
    }

    #[test]
    fn a_part_instance_states_its_shape_and_format() {
        let architecture = Architecture::from_json(CONFIG).expect("a synthetic config");
        let stacks = architecture.stacks();
        let attention = stacks[0]
            .parts()
            .find(|instance| instance.part == Part::GqaAttention)
            .expect("the language stack's attention");
        assert_eq!(attention.shape, Shape::new([2, 4]));
        assert_eq!(attention.format, WeightFormat::Bf16);
        assert!(attention.tensors.is_empty());
    }

    #[test]
    fn an_unknown_shape_is_never_a_zero_shape() {
        assert!(Shape::unknown().is_unknown());
        assert_eq!(Shape::unknown().to_string(), "?");
        assert_eq!(Shape::new([8, 16]).to_string(), "8x16");
    }
}
