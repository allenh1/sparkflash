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

use std::collections::BTreeMap;
use std::fmt;

use crate::architecture::{
    Architecture, AttentionKind, Ffn, Layer, Residual, Score, Side, WeightFormat,
};
use crate::compose::LayerStep;
use crate::parts::Part;
use crate::vision_geometry::{VisionDownsample, VisionFact};

/// A value a part instance carries as a fact.
///
/// Facts are the values a backend opens from: a model-wide geometry value lands on the part that
/// consumes it, and a per-layer selection lands on that layer's part. They are deliberately
/// data, not a family configuration: a backend reads the fact by name and never re-parses the
/// checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum FactValue {
    /// A boolean switch.
    Bool(bool),
    /// A signed integer.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A string (a score function, a dtype name, an encoder side).
    Text(String),
    /// A list of integers (a layer list, a per-layer ratio array).
    Integers(Vec<i64>),
    /// An explicit null, used where the checkpoint stores one.
    Null,
}

impl FactValue {
    /// The value as an integer, when it is one.
    #[must_use]
    pub const fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// The value as a float, widening an integer.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// The value as a string.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            _ => None,
        }
    }
}

/// One checkpoint tensor a part instance claims, with the rank split the loader applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorClaim {
    /// The exact checkpoint tensor name.
    pub name: String,
    /// The logical stored shape.
    pub shape: Vec<usize>,
    /// The stored element class, as the layout spells it.
    pub dtype: String,
    /// The rank split (`whole`, `rows`, `columns`, `kv_rows`, `vector`, `conv`, `expert_rows`,
    /// `expert_columns`, `expert_axis`, `tiles` or `tile_rows`).
    pub split: String,
    /// The tile width a `tiles` or `tile_rows` split cuts on.
    pub tile: Option<usize>,
    /// The base forward's name for the tensor, when the checkpoint spells it otherwise (the
    /// layout's `slot`).
    pub slot: Option<String>,
}

impl TensorClaim {
    /// The name the base forward reads this tensor by: its slot, else its stored name.
    #[must_use]
    pub fn base_name(&self) -> &str {
        self.slot.as_deref().unwrap_or(&self.name)
    }
}

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
#[derive(Debug, Clone, PartialEq)]
pub struct PartInstance {
    /// The part kind.
    pub part: Part,
    /// The shape the part runs at, when the description states one.
    pub shape: Shape,
    /// The stored weight format, when the checkpoint's typed fields state it.
    ///
    /// `None` when the format belongs to a part whose tensors the config alone does not size
    /// (a tower, an Engram table, a draft stage with its own storage). The layout reader fills
    /// those; a guessed format is never reported as stored.
    pub format: Option<WeightFormat>,
    /// The checkpoint tensor names this instance takes, in the layout's order.
    pub tensors: Vec<String>,
    /// The tensors this instance claims, with the rank split each takes.
    pub claims: Vec<TensorClaim>,
    /// The facts this instance carries, by the name a backend reads.
    pub facts: BTreeMap<String, FactValue>,
}

impl PartInstance {
    /// An instance with no tensor names yet.
    #[must_use]
    pub fn new(part: Part, shape: Shape, format: Option<WeightFormat>) -> Self {
        Self {
            part,
            shape,
            format,
            tensors: Vec::new(),
            claims: Vec::new(),
            facts: BTreeMap::new(),
        }
    }

    /// The instance with the tensor names it takes.
    #[must_use]
    pub fn with_tensors(mut self, tensors: Vec<String>) -> Self {
        self.tensors = tensors;
        self
    }

    /// The instance with the tensor claims it takes.
    #[must_use]
    pub fn with_claims(mut self, claims: Vec<TensorClaim>) -> Self {
        self.claims = claims;
        self
    }

    /// The instance with one fact.
    #[must_use]
    pub fn with_fact(mut self, name: impl Into<String>, value: FactValue) -> Self {
        self.facts.insert(name.into(), value);
        self
    }
}

/// One block: the ordered part instances it runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// The block's index in its stack.
    pub index: usize,
    /// The parts, in forward order.
    pub parts: Vec<PartInstance>,
}

/// One stack of a model description.
#[derive(Debug, Clone, PartialEq)]
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
        let mut stacks = vec![Stack {
            kind: StackKind::Language,
            blocks,
        }];
        if let Some(vision) = self.vision_stack() {
            stacks.push(vision);
        }
        stacks
    }

    /// The vision tower as a stack of the components the checkpoint describes.
    ///
    /// One block holds one instance of each component: the depth a tower repeats its norm,
    /// attention and MLP is geometry on those parts, not a second description. A component the
    /// config and index leave undetermined is still an instance, so the scan can refuse it as
    /// `unknown layout` instead of dropping it.
    #[must_use]
    pub fn vision_stack(&self) -> Option<Stack> {
        let spec = self.vision_spec()?;
        let geometry = &spec.geometry;
        let head_dim = geometry.hidden_size.checked_div(geometry.num_heads)?;
        let position_shape = if spec.unstated.contains(&VisionFact::Position) {
            Shape::unknown()
        } else if spec.position.has_grid() {
            Shape::new([geometry.num_position_embeddings, geometry.hidden_size])
        } else {
            Shape::new([head_dim / 2])
        };
        let mut parts = vec![
            PartInstance::new(
                Part::VisionPatchEmbed,
                Shape::new([geometry.hidden_size, spec.patch_features()]),
                None,
            ),
            PartInstance::new(Part::VisionPosition, position_shape, None),
            PartInstance::new(Part::VisionNorm, Shape::new([geometry.hidden_size]), None),
            PartInstance::new(
                Part::VisionAttention,
                Shape::new([geometry.num_heads, head_dim]),
                None,
            ),
            PartInstance::new(
                Part::VisionMlp,
                Shape::new([geometry.intermediate_size, geometry.hidden_size]),
                None,
            ),
            PartInstance::new(
                Part::VisionMerger,
                Shape::new([geometry.out_hidden_size, spec.merged_width()]),
                None,
            ),
        ];
        if !matches!(
            spec.downsample,
            VisionDownsample::Merge(_) | VisionDownsample::None
        ) {
            let merger = parts.pop().expect("the merger is the last component");
            parts.push(PartInstance::new(
                Part::VisionDownsample,
                Shape::new([spec.merged_width(), geometry.hidden_size]),
                None,
            ));
            parts.push(merger);
        }
        Some(Stack {
            kind: StackKind::Vision,
            blocks: vec![Block { index: 0, parts }],
        })
    }

    /// The stored format of a part, read from the checkpoint's typed quantization block.
    ///
    /// The format is stated only for the parts the quantization block sizes: the embedding, the
    /// norms and the head (bf16 in every release), the routed experts, and the dense projections.
    ///
    /// A composite part (a vision tower, an Engram table, an indexer, a draft stage) keeps its
    /// tensors' formats in its own tensors, and the per-layer expert overrides a checkpoint may
    /// name are the layout reader's. `None` says so; it is never a default to the dense format.
    #[must_use]
    pub fn stored_format(&self, part: Part) -> Option<WeightFormat> {
        match part {
            Part::Embedding | Part::RmsNorm | Part::LayerNorm | Part::LmHead => {
                Some(WeightFormat::Bf16)
            }
            Part::Moe => {
                // A checkpoint may store experts per layer, naming one format a layer. The
                // single expert format is the stored format only when no layer overrides it.
                self.expert_format_is_uniform()
                    .then_some(self.quantization.expert_format)
            }
            Part::DenseMlp
            | Part::MlaAttention
            | Part::GqaAttention
            | Part::GdnScan
            | Part::KdaScan => Some(self.quantization.dense_format),
            _ => None,
        }
    }

    fn language_block(&self, layer: &Layer) -> Block {
        let step = LayerStep::from_layer(layer);
        let facts = self.layer_facts(layer);
        let parts = step
            .roles()
            .into_iter()
            .map(|(_, part)| {
                let mut instance =
                    PartInstance::new(part, self.part_shape(part, layer), self.stored_format(part));
                instance.facts = facts.clone();
                instance
            })
            .collect();
        Block {
            index: layer.index,
            parts,
        }
    }

    /// The facts one decoder layer carries, attached to every part instance of its block.
    ///
    /// A fact is put on every instance of the block, not on a single designated part, so a
    /// backend can read it from the part it is already holding. The values come from the typed
    /// architecture: the compression ratio and ownership from the attention, the index and
    /// candidate sources from the indexer, the Engram rows from the side, and the encoder split
    /// from the last layer that owns the compressed KV. No family string or layer literal is
    /// consulted.
    fn layer_facts(&self, layer: &Layer) -> BTreeMap<String, FactValue> {
        let mut facts = BTreeMap::new();
        let attention = &layer.attention;
        if let Some(compression) = &attention.compression {
            facts.insert(
                "compress_ratio".to_owned(),
                FactValue::Integer(compression.ratio as i64),
            );
            facts.insert(
                "kv_source".to_owned(),
                FactValue::Bool(compression.owns_keys),
            );
        } else {
            facts.insert("compress_ratio".to_owned(), FactValue::Integer(0));
            facts.insert("kv_source".to_owned(), FactValue::Bool(false));
        }
        if let Some(indexer) = &attention.indexer {
            facts.insert("index_source".to_owned(), FactValue::Bool(true));
            facts.insert(
                "index_heads".to_owned(),
                FactValue::Integer(indexer.heads as i64),
            );
            facts.insert(
                "index_head_dim".to_owned(),
                FactValue::Integer(indexer.head_dim as i64),
            );
            facts.insert(
                "index_topk".to_owned(),
                FactValue::Integer(indexer.topk as i64),
            );
            if let Some(pool) = &indexer.candidate_pool {
                facts.insert(
                    "candidate_source_layer".to_owned(),
                    FactValue::Integer(pool.source_layer as i64),
                );
                facts.insert(
                    "candidate_topk_blocks".to_owned(),
                    FactValue::Integer(pool.blocks as i64),
                );
                facts.insert(
                    "candidate_block_size".to_owned(),
                    FactValue::Integer(pool.block_size as i64),
                );
                facts.insert(
                    "candidate_source".to_owned(),
                    FactValue::Bool(
                        indexer
                            .candidate_pool
                            .is_some_and(|pool| pool.source_layer == layer.index),
                    ),
                );
            }
        } else {
            facts.insert("index_source".to_owned(), FactValue::Bool(false));
            facts.insert("candidate_source".to_owned(), FactValue::Bool(false));
        }
        if let Some(window) = attention.window {
            facts.insert("window_size".to_owned(), FactValue::Integer(window as i64));
        }
        facts.insert(
            "rope_head_dim".to_owned(),
            FactValue::Integer(attention.rope.rotated_dims as i64),
        );
        facts.insert(
            "rope_theta".to_owned(),
            FactValue::Float(attention.rope.theta),
        );
        let last_encoder = self
            .layers
            .iter()
            .filter(|other| {
                other
                    .attention
                    .compression
                    .as_ref()
                    .is_some_and(|compression| compression.owns_keys)
            })
            .map(|other| other.index)
            .max()
            .unwrap_or_else(|| self.layers.len().saturating_sub(1) / 2);
        facts.insert(
            "encoder_last_layer".to_owned(),
            FactValue::Integer(last_encoder as i64),
        );
        facts.insert(
            "encoder_side".to_owned(),
            FactValue::Text(
                if layer.index <= last_encoder {
                    "encoder"
                } else {
                    "decoder"
                }
                .to_owned(),
            ),
        );
        for side in &layer.sides {
            if let Side::Engram(engram) = side {
                let position = self
                    .layers
                    .iter()
                    .filter(|other| {
                        other
                            .sides
                            .iter()
                            .any(|side| matches!(side, Side::Engram(_)))
                    })
                    .position(|other| other.index == layer.index)
                    .unwrap_or(0);
                facts.insert("engram_member".to_owned(), FactValue::Bool(true));
                facts.insert(
                    "engram_position".to_owned(),
                    FactValue::Integer(position as i64),
                );
                facts.insert(
                    "engram_rows".to_owned(),
                    FactValue::Integer(engram.tables.get(position).copied().unwrap_or(0) as i64),
                );
                facts.insert(
                    "engram_num_embeddings".to_owned(),
                    FactValue::Integers(engram.tables.iter().map(|rows| *rows as i64).collect()),
                );
                facts.insert(
                    "engram_max_ngram_size".to_owned(),
                    FactValue::Integer(engram.max_ngram as i64),
                );
                facts.insert(
                    "engram_n_heads".to_owned(),
                    FactValue::Integer(engram.heads as i64),
                );
                facts.insert(
                    "engram_head_dim".to_owned(),
                    FactValue::Integer(engram.head_dim as i64),
                );
                facts.insert(
                    "engram_vocab_size".to_owned(),
                    FactValue::Integer(engram.vocab_size as i64),
                );
                facts.insert(
                    "engram_compressed_vocab_size".to_owned(),
                    FactValue::Integer(engram.compressed_vocab as i64),
                );
                facts.insert(
                    "engram_pad_id".to_owned(),
                    FactValue::Integer(i64::from(engram.pad_id)),
                );
            }
        }
        if !layer
            .sides
            .iter()
            .any(|side| matches!(side, Side::Engram(_)))
        {
            facts.insert("engram_member".to_owned(), FactValue::Bool(false));
        }
        if let Some(mtp) = &self.mtp {
            facts.insert(
                "tap_layers".to_owned(),
                FactValue::Integers(
                    mtp.target_layers
                        .iter()
                        .map(|layer| *layer as i64)
                        .collect(),
                ),
            );
        }
        if let Residual::Hyper(hyper) = &layer.residual {
            facts.insert(
                "hc_mult".to_owned(),
                FactValue::Integer(hyper.streams as i64),
            );
            facts.insert(
                "hc_sinkhorn_iters".to_owned(),
                FactValue::Integer(hyper.sinkhorn_iters as i64),
            );
            facts.insert("hc_eps".to_owned(), FactValue::Float(hyper.epsilon));
        }
        if let Ffn::Moe(moe) = &layer.ffn {
            facts.insert(
                "n_routed_experts".to_owned(),
                FactValue::Integer(moe.routed_experts as i64),
            );
            facts.insert(
                "n_shared_experts".to_owned(),
                FactValue::Integer(moe.shared_experts as i64),
            );
            facts.insert(
                "n_activated_experts".to_owned(),
                FactValue::Integer(moe.active_experts as i64),
            );
            facts.insert(
                "score_func".to_owned(),
                FactValue::Text(
                    match moe.score {
                        Score::Softmax => "softmax",
                        Score::Sigmoid => "sigmoid",
                        Score::SqrtSoftPlus => "sqrtsoftplus",
                    }
                    .to_owned(),
                ),
            );
            facts.insert(
                "route_scale".to_owned(),
                FactValue::Float(moe.routed_scaling),
            );
            facts.insert("norm_topk_prob".to_owned(), FactValue::Bool(moe.norm_topk));
            facts.insert(
                "swiglu_limit".to_owned(),
                moe.clamp.map_or(FactValue::Null, FactValue::Float),
            );
        }
        facts
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
        shape_from(dims)
    }
}

/// A shape from the config's dimensions, or unknown when any of them is unset.
///
/// A partially unset geometry (a head count of zero beside a real head width) must not become a
/// lower-rank shape: it is unknown.
fn shape_from(dims: Vec<usize>) -> Shape {
    if dims.is_empty() || dims.contains(&0) {
        Shape::unknown()
    } else {
        Shape::new(dims)
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
        assert_eq!(attention.format, Some(WeightFormat::Bf16));
        assert_eq!(architecture.stored_format(Part::VisionNorm), None);
        assert!(attention.tensors.is_empty());
    }

    #[test]
    fn an_unknown_shape_is_never_a_zero_shape() {
        assert!(Shape::unknown().is_unknown());
        assert_eq!(Shape::unknown().to_string(), "?");
        assert_eq!(Shape::new([8, 16]).to_string(), "8x16");
    }
}
