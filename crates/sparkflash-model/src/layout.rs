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

//! Layout files: the wiring a checkpoint's config does not state.
//!
//! A config.json gives sizes and switches. It does not say whether a block norms before or
//! after its branch, which activation a merger runs, or which stored tensor belongs to which
//! part. A layout file states those facts as data, so the description is not completed by a
//! Rust arm.
//!
//! The file has four sections:
//!
//! * match: the architectures and root model_type values the file describes. This matcher is
//!   the only place in the tree that compares a family string.
//! * stacks: each stack's block order and, per part, the wiring facts. A fact is a literal or
//!   a reference to a config key with an optional default.
//! * keys: every config key the layout reads and the fact it sets, plus keys that are
//!   ignorable, each with a written reason.
//! * tensor: name templates mapped to part roles. A template repeats over layers, experts,
//!   vision blocks or the MTP stage and names the rank split the loader uses.
//!
//! Shipped layouts live in models/layouts/ and are embedded at build time. This node's
//! ~/.cache/sparkflash/layouts/*.toml files are read as well; a cache file replaces a shipped
//! layout only by naming it in replaces. Two files that match one checkpoint are a refusal
//! that names both, never a silent precedence rule.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::architecture::{Architecture, AttentionKind, Ffn, Layer, Residual, Vision};
use crate::coverage::{ConfigCoverage, read_key};
use crate::parts::Part;
use crate::stack::{Block, PartInstance, Shape, Stack, StackKind};

/// The layouts shipped in the repository, embedded at build time.
const SHIPPED: &[(&str, &str)] = &[(
    "glm5-next.toml",
    include_str!("../../../models/layouts/glm5-next.toml"),
)];

/// The directory this node's own layouts are read from.
pub const CACHE_DIR: &str = "~/.cache/sparkflash/layouts";

/// A layout file, parsed and checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    /// Stable identifier; also the name a cache file replaces.
    pub id: String,
    /// The shipped layout this cache file replaces, if any.
    pub replaces: Option<String>,
    /// The architectures values this layout describes.
    pub architectures: Vec<String>,
    /// The root model_type values this layout describes.
    pub model_types: Vec<String>,
    /// Key declarations, in file order.
    pub keys: Vec<KeyDecl>,
    /// Stack declarations, keyed by stack name.
    pub stacks: BTreeMap<String, StackDecl>,
    /// Tensor declarations, in file order.
    pub tensors: Vec<TensorDecl>,
    /// Where this layout was read from, for a refusal that names both files.
    pub origin: String,
}

/// One config key a layout reads or ignores.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyDecl {
    /// The full path from the config root.
    pub path: String,
    /// The description fact this key sets.
    pub fact: Option<String>,
    /// Why this key can be ignored instead of read.
    pub ignore: Option<String>,
}

/// One stack's block order and part facts.
#[derive(Debug, Clone, PartialEq)]
pub struct StackDecl {
    /// The part kinds, in forward order.
    pub order: Vec<String>,
    /// Per-part wiring facts.
    pub parts: Vec<PartDecl>,
}

/// One part's wiring facts.
#[derive(Debug, Clone, PartialEq)]
pub struct PartDecl {
    /// The part kind's stable name.
    pub part: String,
    /// The facts this part needs.
    pub facts: Vec<FactDecl>,
}

/// One wiring fact.
#[derive(Debug, Clone, PartialEq)]
pub struct FactDecl {
    /// The fact's name within its part.
    pub name: String,
    /// A literal, or a reference to a config key with an optional default.
    pub value: FactValue,
}

/// A fact's value source.
#[derive(Debug, Clone, PartialEq)]
pub enum FactValue {
    /// A literal TOML value.
    Literal(toml::Value),
    /// A config key, and the value used when the key is absent.
    Key {
        /// The full config path.
        key: String,
        /// The fallback value.
        default: Option<toml::Value>,
    },
}

/// One tensor name template and the rank split it takes.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDecl {
    /// The name template, with {layer}, {block} and {expert} placeholders.
    pub name: String,
    /// The part kind that owns the tensor.
    pub part: String,
    /// The tensor role, as the ops library names it.
    pub role: String,
    /// The stored element type: bf16, f32, dense or expert.
    pub dtype: String,
    /// The logical shape, as integer literals or arithmetic expressions.
    pub shape: Vec<String>,
    /// The rank split: whole, rows, columns, kv_rows, vector, conv, expert_rows or expert_columns.
    pub split: String,
    /// How many instances the template describes.
    pub repeat: Repeat,
    /// Which layers or experts the template selects.
    pub when: When,
    /// When the tensor is stored but never loaded, why.
    pub unloaded: Option<String>,
}

/// Where one expanded tensor sits in the description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    block: Option<usize>,
    expert: Option<usize>,
    stack: StackKind,
}

/// How a tensor template repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repeat {
    /// One tensor.
    Once,
    /// One per decoder layer.
    Layers,
    /// One per routed expert of each selected MoE layer.
    Experts,
    /// One per routed expert of the MTP stage's selected MoE layer.
    MtpExperts,
    /// One per vision block.
    VisionBlocks,
    /// The MTP stage, using the first selected layer's wiring.
    Mtp,
}

/// The parts a tensor template selects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct When {
    /// The attention kind: kda, mla, gqa, gdn or inkling.
    pub attention: Option<String>,
    /// The feed-forward kind: dense or moe.
    pub ffn: Option<String>,
    /// Whether the layer owns an indexer (true) or reuses one (false).
    pub indexer: Option<bool>,
}

/// A resolved tensor of one checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutTensor {
    /// The checkpoint's exact tensor name.
    pub name: String,
    /// The part kind that owns it.
    pub part: Part,
    /// The tensor role.
    pub role: String,
    /// The logical shape.
    pub shape: Vec<usize>,
    /// The stored element class.
    pub dtype: LayoutDtype,
    /// The rank split.
    pub split: TensorSplit,
    /// The stack the tensor belongs to.
    pub stack: StackKind,
    /// The block index, when the tensor belongs to one block.
    pub block: Option<usize>,
    /// Why the tensor is stored but never loaded, when it is.
    pub unloaded: Option<String>,
}

/// The stored element class a layout declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDtype {
    /// BF16.
    Bf16,
    /// F32.
    F32,
    /// The checkpoint's dense projection format.
    Dense,
    /// The checkpoint's routed-expert format.
    Expert,
}

/// How a tensor is split across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorSplit {
    /// Replicated.
    Whole,
    /// Rows split by head count.
    Rows,
    /// Columns split by intermediate width.
    Columns,
    /// Rows split by key/value heads.
    KvRows,
    /// A 1-D vector split by rows.
    Vector,
    /// A 3-D convolution split by rows.
    Conv,
    /// An expert projection split on its output rows.
    ExpertRows,
    /// An expert projection split on its input columns.
    ExpertColumns,
}

/// A layout selected and applied to one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutDescription {
    /// The layout's identifier.
    pub layout: String,
    /// Every fact the layout read, by fact name.
    pub facts: BTreeMap<String, LayoutValue>,
    /// The tensors the layout describes, in declaration order.
    pub tensors: Vec<LayoutTensor>,
    /// The stacks, with each part instance's tensor names attached.
    pub stacks: Vec<Stack>,
}

/// A resolved fact value.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutValue {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Integer(i64),
    /// A float.
    Float(f64),
    /// A string.
    String(String),
    /// A list of integers.
    Integers(Vec<i64>),
    /// A list of strings.
    Strings(Vec<String>),
}

impl std::fmt::Display for LayoutValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(value) => write!(formatter, "{value}"),
            Self::Integer(value) => write!(formatter, "{value}"),
            Self::Float(value) => write!(formatter, "{value}"),
            Self::String(value) => formatter.write_str(value),
            Self::Integers(values) => write!(formatter, "{values:?}"),
            Self::Strings(values) => write!(formatter, "{values:?}"),
        }
    }
}

/// Why a layout could not be read, selected or applied.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// The file is not valid TOML or misses a field.
    #[error("layout {origin}: {detail}")]
    Invalid {
        /// The file's origin.
        origin: String,
        /// What is wrong.
        detail: String,
    },
    /// Two files describe one checkpoint.
    #[error(
        "checkpoint matches both layout {first} ({first_origin}) and layout {second} \
         ({second_origin}); remove one, or have the cache layout name {first} in replaces"
    )]
    Ambiguous {
        /// The first layout's id.
        first: String,
        /// The first layout's origin.
        first_origin: String,
        /// The second layout's id.
        second: String,
        /// The second layout's origin.
        second_origin: String,
    },
    /// A cache layout names a shipped layout that does not exist.
    #[error("cache layout {layout} replaces {replaces:?}, which no shipped layout names")]
    UnknownReplacement {
        /// The cache layout's id.
        layout: String,
        /// The name it replaces.
        replaces: String,
    },
    /// A fact or selector names something the reader does not know.
    #[error("layout {layout}: {detail}")]
    Refused {
        /// The layout's id.
        layout: String,
        /// What was refused and what would fix it.
        detail: String,
    },
}

impl LayoutError {
    fn invalid(origin: &str, detail: impl Into<String>) -> Self {
        Self::Invalid {
            origin: origin.to_owned(),
            detail: detail.into(),
        }
    }

    fn refused(layout: &str, detail: impl Into<String>) -> Self {
        Self::Refused {
            layout: layout.to_owned(),
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLayout {
    schema: u32,
    id: String,
    #[serde(default)]
    replaces: Option<String>,
    #[serde(rename = "match")]
    matcher: RawMatch,
    #[serde(default)]
    keys: Vec<RawKey>,
    #[serde(default)]
    stacks: BTreeMap<String, RawStack>,
    #[serde(default, rename = "tensor")]
    tensors: Vec<RawTensor>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMatch {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_types: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawKey {
    path: String,
    #[serde(default)]
    fact: Option<String>,
    #[serde(default)]
    ignore: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStack {
    #[serde(default)]
    order: Vec<String>,
    #[serde(default)]
    parts: Vec<RawPart>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPart {
    part: String,
    #[serde(default)]
    facts: Vec<RawFact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFact {
    name: String,
    value: toml::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    name: String,
    part: String,
    role: String,
    dtype: String,
    shape: Vec<RawDim>,
    split: String,
    #[serde(default)]
    repeat: Option<String>,
    #[serde(default)]
    when: Option<RawWhen>,
    #[serde(default)]
    unloaded: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawDim {
    Integer(usize),
    Expression(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWhen {
    #[serde(default)]
    attention: Option<String>,
    #[serde(default)]
    ffn: Option<String>,
    #[serde(default)]
    indexer: Option<bool>,
}
impl Layout {
    /// Parse one layout file.
    ///
    /// # Errors
    /// [LayoutError::Invalid] for malformed TOML, an unknown field, or a declaration that names
    /// a part or split the tree does not have.
    pub fn parse(origin: &str, text: &str) -> Result<Self, LayoutError> {
        let raw: RawLayout = toml::from_str(text)
            .map_err(|error| LayoutError::invalid(origin, error.to_string()))?;
        if raw.schema != 1 {
            return Err(LayoutError::invalid(
                origin,
                format!("schema {} is not 1", raw.schema),
            ));
        }
        if raw.id.is_empty() {
            return Err(LayoutError::invalid(origin, "id is empty"));
        }
        if raw.matcher.architectures.is_empty() && raw.matcher.model_types.is_empty() {
            return Err(LayoutError::invalid(
                origin,
                "match names no architectures or model_types",
            ));
        }
        let mut keys = Vec::new();
        for key in raw.keys {
            if key.fact.is_some() == key.ignore.is_some() {
                return Err(LayoutError::invalid(
                    origin,
                    format!("key {} must name exactly one of fact or ignore", key.path),
                ));
            }
            if let Some(reason) = &key.ignore
                && reason.trim().len() < 12
            {
                return Err(LayoutError::invalid(
                    origin,
                    format!("key {} needs a written ignore reason", key.path),
                ));
            }
            keys.push(KeyDecl {
                path: key.path,
                fact: key.fact,
                ignore: key.ignore,
            });
        }
        let mut stacks = BTreeMap::new();
        for (name, stack) in raw.stacks {
            if stack_kind(&name).is_none() {
                return Err(LayoutError::invalid(
                    origin,
                    format!("stack {name:?} is not language, vision, audio or draft"),
                ));
            }
            for part in &stack.order {
                if Part::parse(part).is_none() {
                    return Err(LayoutError::invalid(
                        origin,
                        format!("stack {name}: order names unknown part {part:?}"),
                    ));
                }
            }
            for part in &stack.parts {
                if Part::parse(&part.part).is_none() {
                    return Err(LayoutError::invalid(
                        origin,
                        format!("stack {name}: unknown part {:?}", part.part),
                    ));
                }
            }
            stacks.insert(
                name,
                StackDecl {
                    order: stack.order,
                    parts: stack
                        .parts
                        .into_iter()
                        .map(|part| PartDecl {
                            part: part.part,
                            facts: part
                                .facts
                                .into_iter()
                                .map(|fact| FactDecl {
                                    name: fact.name,
                                    value: fact_value(fact.value),
                                })
                                .collect(),
                        })
                        .collect(),
                },
            );
        }
        let mut tensors = Vec::new();
        for tensor in raw.tensors {
            if Part::parse(&tensor.part).is_none() {
                return Err(LayoutError::invalid(
                    origin,
                    format!(
                        "tensor {:?} names unknown part {:?}",
                        tensor.name, tensor.part
                    ),
                ));
            }
            let repeat = match tensor.repeat.as_deref().unwrap_or("once") {
                "once" => Repeat::Once,
                "layers" => Repeat::Layers,
                "experts" => Repeat::Experts,
                "mtp_experts" => Repeat::MtpExperts,
                "vision_blocks" => Repeat::VisionBlocks,
                "mtp" => Repeat::Mtp,
                other => {
                    return Err(LayoutError::invalid(
                        origin,
                        format!("tensor {:?}: unknown repeat {other:?}", tensor.name),
                    ));
                }
            };
            let split = match tensor.split.as_str() {
                "whole" => TensorSplit::Whole,
                "rows" => TensorSplit::Rows,
                "columns" => TensorSplit::Columns,
                "kv_rows" => TensorSplit::KvRows,
                "vector" => TensorSplit::Vector,
                "conv" => TensorSplit::Conv,
                "expert_rows" => TensorSplit::ExpertRows,
                "expert_columns" => TensorSplit::ExpertColumns,
                other => {
                    return Err(LayoutError::invalid(
                        origin,
                        format!("tensor {:?}: unknown split {other:?}", tensor.name),
                    ));
                }
            };
            if matches!(split, TensorSplit::ExpertRows | TensorSplit::ExpertColumns)
                && !matches!(repeat, Repeat::Experts | Repeat::MtpExperts)
            {
                return Err(LayoutError::invalid(
                    origin,
                    format!(
                        "tensor {:?}: an expert split needs repeat = experts",
                        tensor.name
                    ),
                ));
            }
            if !matches!(tensor.dtype.as_str(), "bf16" | "f32" | "dense" | "expert") {
                return Err(LayoutError::invalid(
                    origin,
                    format!("tensor {:?}: unknown dtype {:?}", tensor.name, tensor.dtype),
                ));
            }
            let shape: Vec<String> = tensor
                .shape
                .into_iter()
                .map(|dim| match dim {
                    RawDim::Integer(value) => value.to_string(),
                    RawDim::Expression(expression) => expression,
                })
                .collect();
            if shape.is_empty() || shape.iter().any(|dim| dim.trim().is_empty()) {
                return Err(LayoutError::invalid(
                    origin,
                    format!("tensor {:?} has an empty shape expression", tensor.name),
                ));
            }
            let when = tensor.when.map_or_else(When::default, |when| When {
                attention: when.attention,
                ffn: when.ffn,
                indexer: when.indexer,
            });
            if let Some(attention) = &when.attention
                && !matches!(
                    attention.as_str(),
                    "kda" | "mla" | "gqa" | "gdn" | "inkling"
                )
            {
                return Err(LayoutError::invalid(
                    origin,
                    format!(
                        "tensor {:?}: unknown attention selector {attention:?}",
                        tensor.name
                    ),
                ));
            }
            if let Some(ffn) = &when.ffn
                && !matches!(ffn.as_str(), "dense" | "moe")
            {
                return Err(LayoutError::invalid(
                    origin,
                    format!("tensor {:?}: unknown ffn selector {ffn:?}", tensor.name),
                ));
            }
            tensors.push(TensorDecl {
                name: tensor.name,
                part: tensor.part,
                role: tensor.role,
                dtype: tensor.dtype,
                shape,
                split: tensor.split,
                repeat,
                when,
                unloaded: tensor.unloaded,
            });
        }
        Ok(Self {
            id: raw.id,
            replaces: raw.replaces,
            architectures: raw.matcher.architectures,
            model_types: raw.matcher.model_types,
            keys,
            stacks,
            tensors,
            origin: origin.to_owned(),
        })
    }

    /// Whether this layout describes architecture.
    ///
    /// This is the only comparison against a family string outside this module.
    #[must_use]
    pub fn matches(&self, architecture: &Architecture) -> bool {
        self.architectures
            .iter()
            .any(|name| architecture.architectures.iter().any(|have| have == name))
            || self.model_types.contains(&architecture.family)
    }

    /// Read the layout's keys into coverage, apply its wiring facts to architecture, and return
    /// the facts and tensor description.
    ///
    /// # Errors
    /// [LayoutError::Refused] when a key path is malformed, a fact names an expression the
    /// reader cannot evaluate, or a tensor shape refers to an unknown fact.
    pub fn apply(
        &self,
        architecture: &mut Architecture,
        root: &Value,
        coverage: &mut ConfigCoverage,
    ) -> Result<LayoutDescription, LayoutError> {
        let mut facts = BTreeMap::new();
        for key in &self.keys {
            let value = read_path(root, &key.path, self)?;
            if let Some(reason) = &key.ignore {
                coverage.ignore_path(&key.path, reason);
                continue;
            }
            coverage.claim_path(&key.path);
            let Some(value) = value else {
                continue;
            };
            let Some(fact) = &key.fact else {
                continue;
            };
            facts.insert(fact.clone(), layout_value(value, self, fact)?);
        }
        for (stack_name, stack) in &self.stacks {
            for part in &stack.parts {
                for fact in &part.facts {
                    let (fact_path, value, key_path) =
                        self.part_fact(stack_name, part, fact, root)?;
                    if let Some(key_path) = &key_path {
                        coverage.claim_path(key_path);
                    }
                    facts.insert(fact_path, value);
                }
            }
        }
        self.apply_facts(architecture, &facts)?;
        let tensors = self.tensors(architecture)?;
        let stacks = self.describe_stacks(architecture, &tensors)?;
        Ok(LayoutDescription {
            layout: self.id.clone(),
            facts,
            tensors,
            stacks,
        })
    }

    fn part_fact(
        &self,
        stack_name: &str,
        part: &PartDecl,
        fact: &FactDecl,
        root: &Value,
    ) -> Result<(String, LayoutValue, Option<String>), LayoutError> {
        let (path, default) = match &fact.value {
            FactValue::Literal(value) => {
                let path = format!("{stack_name}.{}.{}", part.part, fact.name);
                let value = layout_toml_value(value, self, &path)?;
                return Ok((path, value, None));
            }
            FactValue::Key { key, default } => (key.as_str(), default.as_ref()),
        };
        let value = read_path(root, path, self)?;
        let value = match (value, default) {
            (Some(value), _) => layout_value(value, self, path)?,
            (None, Some(default)) => layout_toml_value(default, self, path)?,
            (None, None) => {
                return Err(LayoutError::refused(
                    &self.id,
                    format!(
                        "part {}.{} needs {path}, but the checkpoint does not state it and the \
                         layout gives no default",
                        part.part, fact.name
                    ),
                ));
            }
        };
        let fact_path = format!("{stack_name}.{}.{}", part.part, fact.name);
        Ok((fact_path, value, Some(path.to_owned())))
    }

    fn apply_facts(
        &self,
        architecture: &mut Architecture,
        facts: &BTreeMap<String, LayoutValue>,
    ) -> Result<(), LayoutError> {
        for (name, value) in facts {
            if !known_fact(name) {
                return Err(LayoutError::refused(
                    &self.id,
                    format!(
                        "fact {name:?} is not one this reader knows; name the part field it sets"
                    ),
                ));
            }
            match name.as_str() {
                "vision.position" | "vision.vision_position.position" => {
                    let Some(vision) = architecture.vision.as_mut() else {
                        return Err(LayoutError::refused(
                            &self.id,
                            "fact vision.position applies to a checkpoint with no vision_config"
                                .to_owned(),
                        ));
                    };
                    let text = value.to_string();
                    vision.position = Some(match text.as_str() {
                        "rotary" => crate::vision_geometry::VisionPosition::Rotary,
                        "grid" => crate::vision_geometry::VisionPosition::LearnedGrid,
                        "rotary_and_grid" => crate::vision_geometry::VisionPosition::RotaryAndGrid,
                        other => {
                            return Err(LayoutError::refused(
                                &self.id,
                                format!(
                                    "vision.position {other:?} is not rotary, grid or \
                                     rotary_and_grid"
                                ),
                            ));
                        }
                    });
                }
                "language.tie_word_embeddings" => {
                    if let LayoutValue::Bool(tied) = value {
                        architecture.embedding_tied_to_head = *tied;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Every tensor this layout describes for architecture.
    ///
    /// # Errors
    /// [LayoutError::Refused] for an unknown fact in a shape, an arithmetic overflow, or a
    /// selector that names a part kind this tree does not have.
    pub fn tensors(&self, architecture: &Architecture) -> Result<Vec<LayoutTensor>, LayoutError> {
        let mut tensors = Vec::new();
        for declaration in &self.tensors {
            let part = Part::parse(&declaration.part).ok_or_else(|| {
                LayoutError::refused(
                    &self.id,
                    format!(
                        "tensor {:?} names unknown part {:?}",
                        declaration.name, declaration.part
                    ),
                )
            })?;
            match declaration.repeat {
                Repeat::Once => {
                    let vars = self.global_vars(architecture);
                    let stack = stack_of_part(part);
                    self.push_tensor(
                        &mut tensors,
                        declaration,
                        part,
                        &vars,
                        Placement {
                            block: (stack == StackKind::Vision).then_some(0),
                            expert: None,
                            stack,
                        },
                    )?;
                }
                Repeat::Layers => {
                    for (index, layer) in architecture.layers.iter().enumerate() {
                        if !when_matches(&declaration.when, layer) {
                            continue;
                        }
                        let vars = self.layer_vars(architecture, layer, index);
                        self.push_tensor(
                            &mut tensors,
                            declaration,
                            part,
                            &vars,
                            Placement {
                                block: Some(index),
                                expert: None,
                                stack: StackKind::Language,
                            },
                        )?;
                    }
                }
                Repeat::Experts => {
                    for (index, layer) in architecture.layers.iter().enumerate() {
                        if !when_matches(&declaration.when, layer) {
                            continue;
                        }
                        let Ffn::Moe(moe) = &layer.ffn else {
                            return Err(LayoutError::refused(
                                &self.id,
                                format!(
                                    "tensor {:?} repeats over experts, but layer {index} is not MoE",
                                    declaration.name
                                ),
                            ));
                        };
                        for expert in 0..moe.routed_experts {
                            let mut vars = self.layer_vars(architecture, layer, index);
                            vars.insert("expert".to_owned(), expert as i64);
                            self.push_tensor(
                                &mut tensors,
                                declaration,
                                part,
                                &vars,
                                Placement {
                                    block: Some(index),
                                    expert: Some(expert),
                                    stack: StackKind::Language,
                                },
                            )?;
                        }
                    }
                }
                Repeat::MtpExperts => {
                    let Some(mtp) = &architecture.mtp else {
                        continue;
                    };
                    if mtp.stages == 0 {
                        continue;
                    }
                    let reference = architecture
                        .layers
                        .iter()
                        .find(|layer| when_matches(&declaration.when, layer))
                        .ok_or_else(|| {
                            LayoutError::refused(
                                &self.id,
                                format!(
                                    "tensor {:?} repeats over MTP experts, but no layer matches \
                                     its selector",
                                    declaration.name
                                ),
                            )
                        })?;
                    let Ffn::Moe(moe) = &reference.ffn else {
                        return Err(LayoutError::refused(
                            &self.id,
                            format!(
                                "tensor {:?} repeats over MTP experts, but its reference layer is \
                                 not MoE",
                                declaration.name
                            ),
                        ));
                    };
                    let index = architecture.layers.len();
                    for expert in 0..moe.routed_experts {
                        let mut vars = self.layer_vars(architecture, reference, index);
                        vars.insert("expert".to_owned(), expert as i64);
                        self.push_tensor(
                            &mut tensors,
                            declaration,
                            part,
                            &vars,
                            Placement {
                                block: Some(index),
                                expert: Some(expert),
                                stack: StackKind::Draft,
                            },
                        )?;
                    }
                }
                Repeat::VisionBlocks => {
                    let Some(vision) = &architecture.vision else {
                        continue;
                    };
                    for block in 0..vision.depth {
                        let mut vars = self.vision_vars(vision);
                        vars.insert("block".to_owned(), block as i64);
                        self.push_tensor(
                            &mut tensors,
                            declaration,
                            part,
                            &vars,
                            Placement {
                                block: Some(block),
                                expert: None,
                                stack: StackKind::Vision,
                            },
                        )?;
                    }
                }
                Repeat::Mtp => {
                    let Some(mtp) = &architecture.mtp else {
                        continue;
                    };
                    if mtp.stages == 0 {
                        continue;
                    }
                    let reference = architecture
                        .layers
                        .iter()
                        .find(|layer| when_matches(&declaration.when, layer))
                        .or_else(|| architecture.layers.first())
                        .ok_or_else(|| {
                            LayoutError::refused(
                                &self.id,
                                format!(
                                    "tensor {:?} describes the MTP stage, but the architecture \
                                     has no layer to take its wiring from",
                                    declaration.name
                                ),
                            )
                        })?;
                    let index = architecture.layers.len();
                    let vars = self.layer_vars(architecture, reference, index);
                    self.push_tensor(
                        &mut tensors,
                        declaration,
                        part,
                        &vars,
                        Placement {
                            block: Some(index),
                            expert: None,
                            stack: StackKind::Draft,
                        },
                    )?;
                }
            }
        }
        Ok(tensors)
    }

    fn push_tensor(
        &self,
        out: &mut Vec<LayoutTensor>,
        declaration: &TensorDecl,
        part: Part,
        vars: &BTreeMap<String, i64>,
        placement: Placement,
    ) -> Result<(), LayoutError> {
        let block = placement.block;
        let expert = placement.expert;
        let stack = placement.stack;
        let mut name = declaration.name.clone();
        if let Some(block) = block {
            name = name.replace("{layer}", &block.to_string());
            name = name.replace("{block}", &block.to_string());
        }
        if let Some(expert) = expert {
            name = name.replace("{expert}", &expert.to_string());
        }
        if name.contains('{') {
            return Err(LayoutError::refused(
                &self.id,
                format!(
                    "tensor {:?} leaves a placeholder after expansion; name the repeat it needs",
                    name
                ),
            ));
        }
        let shape = declaration
            .shape
            .iter()
            .map(|expression| evaluate(expression, vars))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|detail| {
                LayoutError::refused(&self.id, format!("tensor {name:?} shape: {detail}"))
            })?;
        let dtype = match declaration.dtype.as_str() {
            "bf16" => LayoutDtype::Bf16,
            "f32" => LayoutDtype::F32,
            "dense" => LayoutDtype::Dense,
            "expert" => LayoutDtype::Expert,
            other => {
                return Err(LayoutError::refused(
                    &self.id,
                    format!("tensor {name:?}: unknown dtype {other:?}"),
                ));
            }
        };
        let split = match declaration.split.as_str() {
            "whole" => TensorSplit::Whole,
            "rows" => TensorSplit::Rows,
            "columns" => TensorSplit::Columns,
            "kv_rows" => TensorSplit::KvRows,
            "vector" => TensorSplit::Vector,
            "conv" => TensorSplit::Conv,
            "expert_rows" => TensorSplit::ExpertRows,
            "expert_columns" => TensorSplit::ExpertColumns,
            other => {
                return Err(LayoutError::refused(
                    &self.id,
                    format!("tensor {name:?}: unknown split {other:?}"),
                ));
            }
        };
        out.push(LayoutTensor {
            name,
            part,
            role: declaration.role.clone(),
            shape,
            dtype,
            split,
            stack,
            block,
            unloaded: declaration.unloaded.clone(),
        });
        Ok(())
    }

    fn describe_stacks(
        &self,
        architecture: &Architecture,
        tensors: &[LayoutTensor],
    ) -> Result<Vec<Stack>, LayoutError> {
        let mut stacks = architecture.stacks();
        let draft: Vec<&LayoutTensor> = tensors
            .iter()
            .filter(|tensor| tensor.stack == StackKind::Draft)
            .collect();
        if !draft.is_empty() {
            let mut parts = Vec::new();
            let mut seen = BTreeSet::new();
            for tensor in &draft {
                if !seen.insert(tensor.part) {
                    continue;
                }
                let names: Vec<String> = draft
                    .iter()
                    .filter(|other| other.part == tensor.part)
                    .map(|other| other.name.clone())
                    .collect();
                parts.push(
                    PartInstance::new(tensor.part, Shape::new(tensor.shape.clone()), None)
                        .with_tensors(names),
                );
            }
            stacks.push(Stack {
                kind: StackKind::Draft,
                blocks: vec![Block {
                    index: architecture.layers.len(),
                    parts,
                }],
            });
        }
        for stack in &mut stacks {
            let Some(declaration) = self.stacks.get(stack.kind.name()) else {
                continue;
            };
            for block in &mut stack.blocks {
                block.parts.sort_by_key(|instance| {
                    declaration
                        .order
                        .iter()
                        .position(|name| name == instance.part.name())
                        .unwrap_or(usize::MAX)
                });
                attach_tensors(block, tensors, stack.kind);
            }
        }
        Ok(stacks)
    }

    fn global_vars(&self, architecture: &Architecture) -> BTreeMap<String, i64> {
        let mut vars = BTreeMap::new();
        insert(&mut vars, "hidden", architecture.hidden_size);
        insert(&mut vars, "vocab", architecture.vocab_size);
        insert(&mut vars, "layers", architecture.layers.len());
        insert(
            &mut vars,
            "max_position",
            architecture.max_position_embeddings,
        );
        if let Some(vision) = &architecture.vision {
            self.insert_vision_vars(&mut vars, vision);
        }
        vars
    }

    fn vision_vars(&self, vision: &Vision) -> BTreeMap<String, i64> {
        let mut vars = BTreeMap::new();
        insert(&mut vars, "hidden", vision.hidden_size);
        insert(&mut vars, "vocab", 0);
        self.insert_vision_vars(&mut vars, vision);
        vars
    }

    fn insert_vision_vars(&self, vars: &mut BTreeMap<String, i64>, vision: &Vision) {
        insert(vars, "vision_width", vision.hidden_size);
        insert(vars, "vision_depth", vision.depth);
        insert(vars, "vision_heads", vision.heads);
        insert(
            vars,
            "vision_head_dim",
            vision.hidden_size / vision.heads.max(1),
        );
        insert(vars, "vision_intermediate", vision.intermediate);
        insert(vars, "vision_in_channels", vision.in_channels);
        insert(vars, "vision_patch", vision.patch_size);
        insert(
            vars,
            "vision_frames",
            vision.patch_temporal.unwrap_or(vision.temporal_patch_size),
        );
        insert(vars, "vision_merge", vision.spatial_merge);
        insert(vars, "vision_out", vision.out_hidden_size);
        insert(vars, "vision_projection", vision.projection_intermediate);
    }

    fn layer_vars(
        &self,
        architecture: &Architecture,
        layer: &Layer,
        index: usize,
    ) -> BTreeMap<String, i64> {
        let mut vars = self.global_vars(architecture);
        insert(&mut vars, "layer", index);
        insert(&mut vars, "hidden", architecture.hidden_size);
        insert(&mut vars, "heads", layer.attention.heads);
        insert(&mut vars, "head_dim", layer.attention.head_dim);
        insert(&mut vars, "rope_dims", layer.attention.rope.rotated_dims);
        insert(
            &mut vars,
            "nope",
            layer
                .attention
                .head_dim
                .saturating_sub(layer.attention.rope.rotated_dims),
        );
        let (q_lora, kv_lora, v_head, kv_heads) = match &layer.attention.kind {
            AttentionKind::Mla {
                q_lora_rank,
                kv_lora_rank,
                v_head_dim,
                ..
            } => (*q_lora_rank, *kv_lora_rank, *v_head_dim, 0),
            AttentionKind::Gqa { kv_heads, .. } | AttentionKind::Inkling { kv_heads } => {
                (None, *kv_heads, layer.attention.head_dim, *kv_heads)
            }
            AttentionKind::Gdn(gdn) => (None, gdn.value_heads, gdn.value_dim, 0),
            AttentionKind::Kda(_) => (None, 0, 0, 0),
        };
        insert(&mut vars, "q_lora", q_lora.unwrap_or(0));
        insert(&mut vars, "kv_lora", kv_lora);
        insert(&mut vars, "v_head", v_head);
        insert(&mut vars, "kv_heads", kv_heads);
        let (kda_heads, kda_head_dim, kda_conv) = match &layer.attention.kind {
            AttentionKind::Kda(kda) => (kda.heads, kda.head_dim, kda.conv_kernel),
            _ => (0, 0, 0),
        };
        insert(&mut vars, "kda_heads", kda_heads);
        insert(&mut vars, "kda_head_dim", kda_head_dim);
        insert(&mut vars, "kda_width", kda_heads * kda_head_dim);
        insert(&mut vars, "kda_conv", kda_conv);
        if let Some(indexer) = &layer.attention.indexer {
            insert(&mut vars, "index_heads", indexer.heads);
            insert(&mut vars, "index_dim", indexer.head_dim);
            insert(&mut vars, "index_pool", indexer.pool);
        } else {
            insert(&mut vars, "index_heads", 0);
            insert(&mut vars, "index_dim", 0);
            insert(&mut vars, "index_pool", 0);
        }
        match &layer.ffn {
            Ffn::Dense(dense) => {
                insert(&mut vars, "intermediate", dense.intermediate);
                insert(&mut vars, "moe_intermediate", 0);
                insert(&mut vars, "routed", 0);
                insert(&mut vars, "shared", 0);
                insert(&mut vars, "shared_width", 0);
            }
            Ffn::Moe(moe) => {
                insert(&mut vars, "intermediate", moe.intermediate);
                insert(&mut vars, "moe_intermediate", moe.intermediate);
                insert(&mut vars, "routed", moe.routed_experts);
                insert(&mut vars, "shared", moe.shared_experts);
                insert(
                    &mut vars,
                    "shared_width",
                    moe.shared_intermediate * moe.shared_experts.max(1),
                );
            }
        }
        let streams = match &layer.residual {
            Residual::Hyper(hyper) => hyper.streams,
            _ => 0,
        };
        insert(&mut vars, "streams", streams);
        insert(&mut vars, "hyper_rows", (2 + streams) * streams);
        insert(&mut vars, "hyper_width", streams * architecture.hidden_size);
        vars
    }
}

fn insert(vars: &mut BTreeMap<String, i64>, name: &str, value: usize) {
    vars.insert(name.to_owned(), i64::try_from(value).unwrap_or(i64::MAX));
}

fn attach_tensors(block: &mut Block, tensors: &[LayoutTensor], stack: StackKind) {
    for instance in &mut block.parts {
        instance.tensors = tensors
            .iter()
            .filter(|tensor| {
                tensor.stack == stack
                    && tensor.part == instance.part
                    && tensor.block == Some(block.index)
            })
            .map(|tensor| tensor.name.clone())
            .collect();
    }
}

fn known_fact(name: &str) -> bool {
    matches!(
        name,
        "vision.position"
            | "vision.vision_position.position"
            | "vision.image_size"
            | "language.attention.head_dim"
            | "language.attention.rope"
            | "language.attention.kv_heads"
            | "language.moe.router_dtype"
            | "language.indexer.pool_compress"
            | "language.indexer.rope_interleave"
            | "language.draft.indexer_shared"
            | "language.linear_attention.kda_layers"
            | "language.linear_attention.full_attn_layers"
            | "language.tie_word_embeddings"
            | "match.text_model_type"
            | "match.vision_model_type"
    )
}

fn stack_kind(name: &str) -> Option<StackKind> {
    match name {
        "language" => Some(StackKind::Language),
        "vision" => Some(StackKind::Vision),
        "audio" => Some(StackKind::Audio),
        "draft" => Some(StackKind::Draft),
        _ => None,
    }
}

fn fact_value(value: toml::Value) -> FactValue {
    if let toml::Value::Table(table) = &value
        && let Some(toml::Value::String(key)) = table.get("key")
    {
        return FactValue::Key {
            key: key.clone(),
            default: table.get("default").cloned(),
        };
    }
    FactValue::Literal(value)
}

fn layout_toml_value(
    value: &toml::Value,
    layout: &Layout,
    what: &str,
) -> Result<LayoutValue, LayoutError> {
    match value {
        toml::Value::Boolean(value) => Ok(LayoutValue::Bool(*value)),
        toml::Value::Integer(value) => Ok(LayoutValue::Integer(*value)),
        toml::Value::Float(value) => Ok(LayoutValue::Float(*value)),
        toml::Value::String(value) => Ok(LayoutValue::String(value.clone())),
        toml::Value::Array(items) => {
            if items.iter().all(toml::Value::is_integer) {
                Ok(LayoutValue::Integers(
                    items.iter().filter_map(toml::Value::as_integer).collect(),
                ))
            } else if items.iter().all(toml::Value::is_str) {
                Ok(LayoutValue::Strings(
                    items
                        .iter()
                        .filter_map(toml::Value::as_str)
                        .map(str::to_owned)
                        .collect(),
                ))
            } else {
                Err(LayoutError::refused(
                    &layout.id,
                    format!("{what} is a list with mixed element kinds"),
                ))
            }
        }
        other => Err(LayoutError::refused(
            &layout.id,
            format!("{what} has unsupported value {other}"),
        )),
    }
}

fn layout_value(value: &Value, layout: &Layout, what: &str) -> Result<LayoutValue, LayoutError> {
    match value {
        Value::Bool(value) => Ok(LayoutValue::Bool(*value)),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                Ok(LayoutValue::Integer(value))
            } else if let Some(value) = number.as_f64() {
                Ok(LayoutValue::Float(value))
            } else {
                Err(LayoutError::refused(
                    &layout.id,
                    format!("{what} is not a number this reader can represent"),
                ))
            }
        }
        Value::String(value) => Ok(LayoutValue::String(value.clone())),
        Value::Array(items) => {
            if items.iter().all(Value::is_i64) {
                Ok(LayoutValue::Integers(
                    items.iter().filter_map(Value::as_i64).collect(),
                ))
            } else if items.iter().all(Value::is_string) {
                Ok(LayoutValue::Strings(
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect(),
                ))
            } else {
                Err(LayoutError::refused(
                    &layout.id,
                    format!("{what} is a list with mixed element kinds"),
                ))
            }
        }
        other => Err(LayoutError::refused(
            &layout.id,
            format!("{what} has unsupported value {other}"),
        )),
    }
}

/// Read a dotted config path, recording every member along it.
fn read_path<'a>(
    root: &'a Value,
    path: &str,
    layout: &Layout,
) -> Result<Option<&'a Value>, LayoutError> {
    let mut value = root;
    for segment in path.split('.') {
        if segment.is_empty() {
            return Err(LayoutError::refused(
                &layout.id,
                format!("key path {path:?} has an empty segment"),
            ));
        }
        let Some(child) = read_key(value, segment) else {
            return Ok(None);
        };
        value = child;
    }
    Ok(Some(value))
}

fn when_matches(when: &When, layer: &Layer) -> bool {
    if let Some(attention) = &when.attention {
        let have = match &layer.attention.kind {
            AttentionKind::Kda(_) => "kda",
            AttentionKind::Mla { .. } => "mla",
            AttentionKind::Gqa { .. } => "gqa",
            AttentionKind::Gdn(_) => "gdn",
            AttentionKind::Inkling { .. } => "inkling",
        };
        if attention != have {
            return false;
        }
    }
    if let Some(ffn) = &when.ffn {
        let have = match &layer.ffn {
            Ffn::Dense(_) => "dense",
            Ffn::Moe(_) => "moe",
        };
        if ffn != have {
            return false;
        }
    }
    if let Some(indexer) = when.indexer {
        let have = layer
            .attention
            .indexer
            .as_ref()
            .is_some_and(|indexer| !indexer.reuse);
        if indexer != have {
            return false;
        }
    }
    true
}

/// Evaluate a shape expression over the facts of one instance.
fn evaluate(expression: &str, vars: &BTreeMap<String, i64>) -> Result<usize, String> {
    let mut parser = Expr {
        bytes: expression.as_bytes(),
        at: 0,
        vars,
    };
    let value = parser.sum()?;
    parser.space();
    if parser.at != parser.bytes.len() {
        return Err(format!(
            "unexpected {:?} in {expression:?}",
            &expression[parser.at..]
        ));
    }
    usize::try_from(value).map_err(|_| format!("{expression:?} evaluates to {value}, not a size"))
}

struct Expr<'a> {
    bytes: &'a [u8],
    at: usize,
    vars: &'a BTreeMap<String, i64>,
}

impl Expr<'_> {
    fn space(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn sum(&mut self) -> Result<i64, String> {
        let mut value = self.product()?;
        loop {
            self.space();
            let Some(operator) = self.bytes.get(self.at).copied() else {
                return Ok(value);
            };
            if !matches!(operator, b'+' | b'-') {
                return Ok(value);
            }
            self.at += 1;
            let right = self.product()?;
            value = if operator == b'+' {
                value.checked_add(right)
            } else {
                value.checked_sub(right)
            }
            .ok_or_else(|| "arithmetic overflow".to_owned())?;
        }
    }

    fn product(&mut self) -> Result<i64, String> {
        let mut value = self.atom()?;
        loop {
            self.space();
            let Some(operator) = self.bytes.get(self.at).copied() else {
                return Ok(value);
            };
            if !matches!(operator, b'*' | b'/') {
                return Ok(value);
            }
            self.at += 1;
            let right = self.atom()?;
            if operator == b'*' {
                value = value
                    .checked_mul(right)
                    .ok_or_else(|| "arithmetic overflow".to_owned())?;
            } else {
                if right == 0 {
                    return Err("division by zero".to_owned());
                }
                value /= right;
            }
        }
    }

    fn atom(&mut self) -> Result<i64, String> {
        self.space();
        if self.bytes.get(self.at) == Some(&b'(') {
            self.at += 1;
            let value = self.sum()?;
            self.space();
            if self.bytes.get(self.at) != Some(&b')') {
                return Err("missing ')'".to_owned());
            }
            self.at += 1;
            return Ok(value);
        }
        if self.bytes.get(self.at) == Some(&b'-') {
            self.at += 1;
            return self.atom().map(|value| -value);
        }
        let start = self.at;
        while self.at < self.bytes.len()
            && (self.bytes[self.at].is_ascii_alphanumeric() || self.bytes[self.at] == b'_')
        {
            self.at += 1;
        }
        if start == self.at {
            return Err(format!("expected a number or fact at byte {}", self.at));
        }
        let token = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| "the expression is not UTF-8".to_owned())?;
        if let Ok(value) = token.parse::<i64>() {
            return Ok(value);
        }
        self.vars
            .get(token)
            .copied()
            .ok_or_else(|| format!("unknown fact {token:?}"))
    }
}

/// A set of layouts, shipped and cached.
#[derive(Debug, Clone, Default)]
pub struct LayoutSet {
    layouts: Vec<Layout>,
}

impl LayoutSet {
    /// The shipped layouts plus this node's cache.
    ///
    /// # Errors
    /// A malformed shipped or cache file; a cache file that replaces an unknown shipped id; a
    /// duplicate id.
    pub fn load() -> Result<Self, LayoutError> {
        let mut layouts = Vec::new();
        let mut ids = BTreeSet::new();
        for (name, text) in SHIPPED {
            let origin = format!("models/layouts/{name}");
            let layout = Layout::parse(&origin, text)?;
            if !ids.insert(layout.id.clone()) {
                return Err(LayoutError::invalid(
                    &origin,
                    format!("duplicate layout id {:?}", layout.id),
                ));
            }
            layouts.push(layout);
        }
        for path in cache_paths()? {
            let origin = path.display().to_string();
            let text = std::fs::read_to_string(&path)
                .map_err(|error| LayoutError::invalid(&origin, format!("read failed: {error}")))?;
            let layout = Layout::parse(&origin, &text)?;
            if !ids.insert(layout.id.clone()) {
                return Err(LayoutError::invalid(
                    &origin,
                    format!("duplicate layout id {:?}", layout.id),
                ));
            }
            layouts.push(layout);
        }
        Ok(Self { layouts })
    }

    /// A set from explicit sources, for tests.
    ///
    /// # Errors
    /// As [Layout::parse].
    pub fn from_sources(sources: &[(String, String)]) -> Result<Self, LayoutError> {
        let mut layouts = Vec::new();
        let mut ids = BTreeSet::new();
        for (origin, text) in sources {
            let layout = Layout::parse(origin, text)?;
            if !ids.insert(layout.id.clone()) {
                return Err(LayoutError::invalid(
                    origin,
                    format!("duplicate layout id {:?}", layout.id),
                ));
            }
            layouts.push(layout);
        }
        Ok(Self { layouts })
    }

    /// Every layout in the set, in load order.
    #[must_use]
    pub fn layouts(&self) -> &[Layout] {
        &self.layouts
    }

    /// The one layout that describes architecture, if any.
    ///
    /// # Errors
    /// [LayoutError::Ambiguous] when two files match and no replaces relation resolves them;
    /// [LayoutError::UnknownReplacement] when a cache file replaces an id that is not shipped.
    pub fn select(&self, architecture: &Architecture) -> Result<Option<&Layout>, LayoutError> {
        let mut shipped: Vec<&Layout> = self
            .layouts
            .iter()
            .filter(|layout| layout.replaces.is_none())
            .collect();
        let cache: Vec<&Layout> = self
            .layouts
            .iter()
            .filter(|layout| layout.replaces.is_some())
            .collect();
        for replacement in &cache {
            if let Some(id) = &replacement.replaces {
                let before = shipped.len();
                shipped.retain(|layout| layout.id != *id);
                if shipped.len() == before {
                    return Err(LayoutError::UnknownReplacement {
                        layout: replacement.id.clone(),
                        replaces: id.clone(),
                    });
                }
            }
        }
        let matching: Vec<&Layout> = shipped
            .into_iter()
            .filter(|layout| layout.matches(architecture))
            .chain(
                cache
                    .into_iter()
                    .filter(|layout| layout.matches(architecture)),
            )
            .collect();
        match matching.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some(one)),
            [first, second, ..] => Err(LayoutError::Ambiguous {
                first: first.id.clone(),
                first_origin: first.origin.clone(),
                second: second.id.clone(),
                second_origin: second.origin.clone(),
            }),
        }
    }
}

/// The one layout that describes architecture, if any.
///
/// # Errors
/// As [LayoutSet::load] and [LayoutSet::select].
pub fn matching(architecture: &Architecture) -> Result<Option<Layout>, LayoutError> {
    let set = LayoutSet::load()?;
    Ok(set.select(architecture)?.cloned())
}

/// The cache directory, expanded against this node's home.
#[must_use]
pub fn cache_dir() -> PathBuf {
    sparkflash_weights::home::expand_home(CACHE_DIR)
}

fn cache_paths() -> Result<Vec<PathBuf>, LayoutError> {
    let dir = cache_dir();
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&dir)
        .map_err(|error| LayoutError::invalid(&dir.display().to_string(), error.to_string()))?
    {
        let entry = entry
            .map_err(|error| LayoutError::invalid(&dir.display().to_string(), error.to_string()))?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "toml")
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"
schema = 1
id = "one"
[match]
model_types = ["synthetic"]
"#;

    #[test]
    fn a_file_is_parsed_and_matched_by_model_type() {
        let layout = Layout::parse("one.toml", SOURCE).expect("a layout");
        let architecture = Architecture::from_json(
            r#"{"model_type":"synthetic","hidden_size":8,"vocab_size":8,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8}"#,
        )
        .expect("a config");
        assert!(layout.matches(&architecture));
    }

    #[test]
    fn two_matching_files_are_refused_and_name_both() {
        let second = SOURCE.replace("id = \"one\"", "id = \"two\"");
        let set = LayoutSet::from_sources(&[
            ("one.toml".to_owned(), SOURCE.to_owned()),
            ("two.toml".to_owned(), second),
        ])
        .expect("two layouts");
        let architecture = Architecture::from_json(
            r#"{"model_type":"synthetic","hidden_size":8,"vocab_size":8,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8}"#,
        )
        .expect("a config");
        let error = set.select(&architecture).expect_err("an ambiguity");
        let text = error.to_string();
        assert!(text.contains("one.toml"), "{text}");
        assert!(text.contains("two.toml"), "{text}");
    }

    #[test]
    fn a_cache_file_replaces_a_shipped_id() {
        let replacement = SOURCE.replace("id = \"one\"", "id = \"two\"\nreplaces = \"one\"");
        let set = LayoutSet::from_sources(&[
            ("models/layouts/one.toml".to_owned(), SOURCE.to_owned()),
            ("cache/two.toml".to_owned(), replacement),
        ])
        .expect("two layouts");
        let architecture = Architecture::from_json(
            r#"{"model_type":"synthetic","hidden_size":8,"vocab_size":8,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8}"#,
        )
        .expect("a config");
        assert_eq!(
            set.select(&architecture)
                .expect("a selection")
                .expect("one")
                .id,
            "two"
        );
    }

    #[test]
    fn a_shape_expression_uses_the_key_and_refuses_an_unknown_fact() {
        let mut vars = BTreeMap::new();
        vars.insert("hidden".to_owned(), 8);
        assert_eq!(evaluate("2 * (hidden + 1) / 3", &vars).expect("a shape"), 6);
        let error = evaluate("width", &vars).expect_err("an unknown fact");
        assert!(error.contains("width"), "{error}");
    }
}
