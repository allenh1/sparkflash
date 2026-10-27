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

//! Guided decoding: the JSON-Schema-to-bytes core for OpenAI structured outputs (S1).
//!
//! OpenAI's `response_format: {"type": "json_schema", ...}` and `json_object` ask the server to make
//! the model emit a document that satisfies a schema. vLLM implements that with a logits processor:
//! at every step it asks a guided-decoding backend which tokens can extend the current prefix, then
//! masks every other token's logit to `-inf` before sampling. This module is the decoder-independent
//! half of that design: it compiles a JSON-Schema subset into an incremental byte automaton that
//! answers two questions for a partially generated document,
//!
//! * [`SchemaConstraint::allowed_bytes`] - which bytes may come next, as a 256-bit [`ByteMask`];
//! * [`SchemaConstraint::feed`] - consume one byte and advance the automaton,
//!
//! plus [`SchemaConstraint::admits`], the token-shaped query a logits processor makes ("does this
//! token's byte string keep a valid completion reachable?").
//!
//! # Why it is not wired into the sampling path yet
//!
//! Applying a mask needs a hook this crate does not own: the engine's sampler lives in
//! `sparkflash-engine` (`forward::sample`), whose `SampleOptions` has no logits-processor field, and
//! a token mask also needs the engine's token-to-bytes table. `crates/sparkflash-api` can only
//! refuse the request with a typed error today (see `refuse_unenforceable_response_format`), which
//! is what the router keeps doing. The exact seam is written down in
//! `docs/archive/build/ws-f-api-20261021.md`: an `EngineRequest` field carrying a token mask callback plus a
//! token-bytes table on the engine seam. This module is the part that seam would consume, and it is
//! fully unit-tested here so the hook can be wired without re-deriving JSON grammar rules.
//!
//! # Supported schema subset
//!
//! The automaton is deterministic by construction: every state has exactly one way to continue for
//! a given byte, so a mask never has to merge alternatives. That rules out the ambiguous keywords,
//! which are refused at compile time with a typed [`SchemaError::Unsupported`] instead of being
//! silently ignored. Supported keywords:
//!
//! * `type`: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null` (a union type is
//!   refused: it would need choice-state masking);
//! * objects: `properties`, `required`, `additionalProperties: false`. An object schema without
//!   `additionalProperties` is treated as closed, which can only over-constrain, never admit an
//!   invalid document;
//! * arrays: `items` (required), `minItems`, `maxItems`;
//! * `string`, `integer`, `number`, `boolean`, `null` scalars with the JSON grammar of the type.
//!
//! Refused keywords: `pattern`, `format`, `minLength`, `maxLength` (value automata, not structural),
//! `enum`, `const`, `anyOf`, `oneOf`, `allOf`, `not`, `if`/`then`/`else`, `$ref`/`$defs`,
//! `dependencies`, `dependentRequired`, `dependentSchemas`, `propertyNames`, `patternProperties`,
//! `additionalProperties: true`, numeric bounds (`minimum`, `maximum`, `exclusive*`, `multipleOf`),
//! `uniqueItems`, `contains`, `prefixItems`, `unevaluatedProperties`.
//!
//! # What a wired-in processor still has to do
//!
//! 1. Keep one [`SchemaConstraint`] per sequence, seeded from the request's schema.
//! 2. Map each vocabulary id to its byte string using the checkpoint tokenizer (the engine already
//!    owns it; `TokenizerSeam::token_str` shows the shape of that map).
//! 3. Before sampling, compute `mask = constraint.allowed_bytes()` and for every token whose bytes
//!    do not admit the mask, set its logit to `-inf`. End-of-sequence stays allowed only when the
//!    constraint is [`SchemaConstraint::is_complete`]: a truncated JSON document is not valid.
//! 4. After sampling, `constraint.feed_str(&token_bytes)`.
//!
//! Step 3 is a per-token scan over the vocabulary; a production version wants a byte-trie of the
//! vocabulary instead (walk the trie by the mask and mask whole subtrees). That is an engineering
//! step for the engine side, not a wire-format question.

use std::fmt;

use serde_json::{Map, Value};

/// A 256-bit set of permitted next bytes.
///
/// Bit `n` is set when byte `n` may be emitted next. [`SchemaConstraint::allowed_bytes`] returns one
/// of these and a logits processor turns it into a token mask; [`ByteMask::iter`] enumerates the
/// bytes and [`ByteMask::count`] reports how constrained the position is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ByteMask {
    words: [u64; 4],
}

impl ByteMask {
    /// The empty mask: no byte is permitted.
    pub const EMPTY: Self = Self { words: [0; 4] };

    /// The mask that contains every byte.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            words: [u64::MAX; 4],
        }
    }

    /// Builds a mask from an iterator of bytes.
    #[must_use]
    pub fn from_bytes(bytes: impl IntoIterator<Item = u8>) -> Self {
        let mut mask = Self::EMPTY;
        for byte in bytes {
            mask.insert(byte);
        }
        mask
    }

    /// Adds one byte to the mask.
    pub fn insert(&mut self, byte: u8) {
        self.words[usize::from(byte >> 6)] |= 1u64 << (byte & 63);
    }

    /// Removes one byte from the mask.
    pub fn remove(&mut self, byte: u8) {
        self.words[usize::from(byte >> 6)] &= !(1u64 << (byte & 63));
    }

    /// Adds every byte of `other` to this mask.
    pub fn union_with(&mut self, other: Self) {
        for (word, other) in self.words.iter_mut().zip(other.words) {
            *word |= other;
        }
    }

    /// Whether `byte` is permitted.
    #[must_use]
    pub fn contains(&self, byte: u8) -> bool {
        self.words[usize::from(byte >> 6)] & (1u64 << (byte & 63)) != 0
    }

    /// Whether the mask permits nothing at all: the partial document is a dead end.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// How many bytes the mask permits.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.words.iter().map(|word| word.count_ones()).sum()
    }

    /// The permitted bytes in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0u8..=255).filter(move |byte| self.contains(*byte))
    }
}

impl fmt::Debug for ByteMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("ByteMask(empty)");
        }
        let mut list = f.debug_list();
        for byte in self.iter() {
            list.entry(&format_args!("{byte:#04x}"));
        }
        list.finish()
    }
}

/// A schema the byte automaton cannot enforce, or a malformed supported schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The schema was not a JSON object (JSON Schema also allows the booleans `true`/`false`,
    /// which are refused below because neither constrains a document structurally).
    NotAnObject,
    /// A keyword this automaton does not implement. Refused rather than dropped, so a client
    /// cannot get an unconstrained answer for a schema it asked to be enforced.
    Unsupported {
        /// The offending keyword (or a short description when no keyword names it).
        keyword: String,
        /// Why the byte automaton cannot enforce it.
        reason: String,
    },
    /// A supported keyword with a malformed value.
    Invalid {
        /// The keyword.
        keyword: String,
        /// What is wrong with its value.
        detail: String,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnObject => f.write_str("a JSON-Schema response format must be a JSON object"),
            Self::Unsupported { keyword, reason } => {
                write!(f, "schema keyword '{keyword}' is not enforced: {reason}")
            }
            Self::Invalid { keyword, detail } => {
                write!(f, "schema keyword '{keyword}' is malformed: {detail}")
            }
        }
    }
}

impl std::error::Error for SchemaError {}

/// A byte that the automaton refused, with the position and the mask that was expected there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintError {
    /// How many bytes were accepted before the refused one.
    pub position: usize,
    /// The refused byte.
    pub byte: u8,
    /// The bytes that were permitted at that position.
    pub expected: ByteMask,
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "byte {:#04x} at offset {} is not allowed by the schema ({} permitted bytes)",
            self.byte,
            self.position,
            self.expected.count()
        )
    }
}

impl std::error::Error for ConstraintError {}

/// One node of the compiled schema.
///
/// Only the deterministic structural subset reaches this type: [`compile`] refuses everything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaNode {
    /// The JSON `null` value.
    Null,
    /// `true` or `false`.
    Bool,
    /// An integer literal (no fraction, no exponent).
    Integer,
    /// Any JSON number literal.
    Number,
    /// A JSON string literal.
    String,
    /// A JSON array whose elements must satisfy `items`.
    Array {
        /// The element schema.
        items: Box<SchemaNode>,
        /// Minimum element count (`minItems`, 0 when absent).
        min_items: usize,
        /// Maximum element count (`maxItems`, `None` when absent).
        max_items: Option<usize>,
    },
    /// A closed JSON object.
    Object {
        /// The permitted properties, in schema order.
        properties: Vec<(String, SchemaNode)>,
        /// Indices into `properties` that the document must contain (`required`).
        required: Vec<usize>,
    },
}

/// Keywords this automaton refuses, with the reason a client sees.
const UNSUPPORTED_KEYWORDS: &[(&str, &str)] = &[
    (
        "$ref",
        "references would have to be resolved and inlined first",
    ),
    (
        "$defs",
        "definitions would have to be resolved and inlined first",
    ),
    (
        "definitions",
        "definitions would have to be resolved and inlined first",
    ),
    (
        "enum",
        "a value set needs choice-state masking, which this deterministic automaton does not do",
    ),
    (
        "const",
        "a value set needs choice-state masking, which this deterministic automaton does not do",
    ),
    (
        "anyOf",
        "a choice between schemas needs branch-state masking",
    ),
    (
        "oneOf",
        "a choice between schemas needs branch-state masking",
    ),
    ("allOf", "schema intersection is not computed"),
    ("not", "schema exclusion is not computed"),
    ("if", "conditional schemas need branch-state masking"),
    ("then", "conditional schemas need branch-state masking"),
    ("else", "conditional schemas need branch-state masking"),
    (
        "pattern",
        "a regular expression needs a regex automaton rather than the structural one",
    ),
    (
        "format",
        "string formats need a value automaton rather than the structural one",
    ),
    ("minLength", "string lengths need a counter state"),
    ("maxLength", "string lengths need a counter state"),
    (
        "minimum",
        "numeric bounds need prefix arithmetic on the literal",
    ),
    (
        "maximum",
        "numeric bounds need prefix arithmetic on the literal",
    ),
    (
        "exclusiveMinimum",
        "numeric bounds need prefix arithmetic on the literal",
    ),
    (
        "exclusiveMaximum",
        "numeric bounds need prefix arithmetic on the literal",
    ),
    ("multipleOf", "numeric divisibility needs prefix arithmetic"),
    (
        "uniqueItems",
        "element uniqueness needs the whole array, not a byte prefix",
    ),
    ("contains", "element predicates need the whole array"),
    (
        "prefixItems",
        "tuple arrays need a per-position state machine, which this automaton does not do",
    ),
    ("propertyNames", "key predicates need a value automaton"),
    ("patternProperties", "key patterns need a regex automaton"),
    ("minProperties", "property counts need a counter state"),
    ("maxProperties", "property counts need a counter state"),
    (
        "dependentRequired",
        "cross-property constraints need a whole-object view",
    ),
    (
        "dependentSchemas",
        "cross-property constraints need a whole-object view",
    ),
    ("unevaluatedProperties", "schema evaluation is not computed"),
    ("additionalItems", "legacy tuple keyword is not supported"),
    (
        "dependencies",
        "cross-property constraints need a whole-object view",
    ),
];

/// Compiles a JSON-Schema document into the automaton's node tree.
///
/// # Errors
///
/// [`SchemaError::NotAnObject`] for a non-object schema, [`SchemaError::Unsupported`] for a keyword
/// outside the deterministic structural subset (see the module docs), and [`SchemaError::Invalid`]
/// for a malformed supported keyword.
pub fn compile(schema: &Value) -> Result<SchemaNode, SchemaError> {
    match schema {
        Value::Object(map) => compile_object_schema(map),
        Value::Bool(true) => Err(SchemaError::Unsupported {
            keyword: "true".to_owned(),
            reason: "an unconstrained value cannot be masked; the schema has to constrain a type"
                .to_owned(),
        }),
        Value::Bool(false) => Err(SchemaError::Unsupported {
            keyword: "false".to_owned(),
            reason: "a schema that admits nothing cannot be generated".to_owned(),
        }),
        _ => Err(SchemaError::NotAnObject),
    }
}

fn unsupported(keyword: &str, reason: &str) -> SchemaError {
    SchemaError::Unsupported {
        keyword: keyword.to_owned(),
        reason: reason.to_owned(),
    }
}

fn invalid(keyword: &str, detail: impl Into<String>) -> SchemaError {
    SchemaError::Invalid {
        keyword: keyword.to_owned(),
        detail: detail.into(),
    }
}

fn compile_object_schema(map: &Map<String, Value>) -> Result<SchemaNode, SchemaError> {
    // Keyword validation first: a schema that mentions something the automaton cannot enforce is
    // refused even when the offending keyword would not have changed this particular document.
    for (keyword, reason) in UNSUPPORTED_KEYWORDS {
        if map.contains_key(*keyword) {
            return Err(unsupported(keyword, reason));
        }
    }
    let declared = match map.get("type") {
        None => infer_type(map)?,
        Some(Value::String(kind)) => parse_type(kind)?,
        Some(Value::Array(_)) => {
            return Err(unsupported(
                "type",
                "a union type needs choice-state masking",
            ));
        }
        Some(other) => return Err(invalid("type", format!("expected a string, got {other}"))),
    };
    match declared {
        Declared::Null => Ok(SchemaNode::Null),
        Declared::Bool => Ok(SchemaNode::Bool),
        Declared::Integer => Ok(SchemaNode::Integer),
        Declared::Number => Ok(SchemaNode::Number),
        Declared::String => Ok(SchemaNode::String),
        Declared::Array => compile_array_schema(map),
        Declared::Object => compile_nested_object(map),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Declared {
    Null,
    Bool,
    Integer,
    Number,
    String,
    Array,
    Object,
}

fn infer_type(map: &Map<String, Value>) -> Result<Declared, SchemaError> {
    // An object schema is implied by its property keywords (and by `additionalProperties`), an
    // array schema by its item keywords: JSON Schema's own inference, restricted to the subset.
    let mentions = |keywords: &[&str]| keywords.iter().any(|keyword| map.contains_key(*keyword));
    if mentions(&["properties", "required", "additionalProperties"]) {
        return Ok(Declared::Object);
    }
    if mentions(&["items", "minItems", "maxItems"]) {
        return Ok(Declared::Array);
    }
    Err(unsupported(
        "schema",
        "the schema constrains no type; add \"type\" (an untyped value cannot be masked)",
    ))
}

fn parse_type(kind: &str) -> Result<Declared, SchemaError> {
    match kind {
        "null" => Ok(Declared::Null),
        "boolean" => Ok(Declared::Bool),
        "integer" => Ok(Declared::Integer),
        "number" => Ok(Declared::Number),
        "string" => Ok(Declared::String),
        "array" => Ok(Declared::Array),
        "object" => Ok(Declared::Object),
        other => Err(unsupported(
            "type",
            &format!("\"{other}\" is not a JSON type this automaton knows"),
        )),
    }
}

fn compile_array_schema(map: &Map<String, Value>) -> Result<SchemaNode, SchemaError> {
    let items = match map.get("items") {
        None => {
            return Err(unsupported(
                "items",
                "an unconstrained element schema cannot be masked; give items a schema",
            ));
        }
        Some(Value::Array(_)) => {
            return Err(unsupported(
                "items",
                "the legacy tuple form needs a per-position state machine",
            ));
        }
        Some(schema) => compile(schema)?,
    };
    let min_items = count_keyword(map, "minItems")?.unwrap_or(0);
    let max_items = count_keyword(map, "maxItems")?;
    if let Some(max) = max_items
        && max < min_items
    {
        return Err(invalid(
            "maxItems",
            format!("maxItems ({max}) is below minItems ({min_items})"),
        ));
    }
    Ok(SchemaNode::Array {
        items: Box::new(items),
        min_items,
        max_items,
    })
}

fn count_keyword(map: &Map<String, Value>, keyword: &str) -> Result<Option<usize>, SchemaError> {
    match map.get(keyword) {
        None => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .map(|value| Some(usize::try_from(value).unwrap_or(usize::MAX)))
            .ok_or_else(|| {
                invalid(
                    keyword,
                    format!("expected a non-negative integer, got {number}"),
                )
            }),
        Some(other) => Err(invalid(
            keyword,
            format!("expected an integer, got {other}"),
        )),
    }
}

fn compile_nested_object(map: &Map<String, Value>) -> Result<SchemaNode, SchemaError> {
    match map.get("additionalProperties") {
        None | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => {
            return Err(unsupported(
                "additionalProperties",
                "extra properties would carry unconstrained values; set it to false",
            ));
        }
        Some(other) => {
            return Err(invalid(
                "additionalProperties",
                format!("expected a boolean, got {other}"),
            ));
        }
    }
    let properties = match map.get("properties") {
        None => Vec::new(),
        Some(Value::Object(properties)) => {
            let mut compiled = Vec::with_capacity(properties.len());
            for (name, schema) in properties {
                compiled.push((name.clone(), compile(schema)?));
            }
            compiled
        }
        Some(other) => {
            return Err(invalid(
                "properties",
                format!("expected an object, got {other}"),
            ));
        }
    };
    let required = match map.get("required") {
        None => Vec::new(),
        Some(Value::Array(names)) => {
            let mut indices = Vec::with_capacity(names.len());
            for name in names {
                let name = name
                    .as_str()
                    .ok_or_else(|| invalid("required", format!("expected strings, got {name}")))?;
                let index = properties
                    .iter()
                    .position(|(property, _)| property == name)
                    .ok_or_else(|| {
                        invalid(
                            "required",
                            format!("\"{name}\" is not declared in properties"),
                        )
                    })?;
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
            indices
        }
        Some(other) => {
            return Err(invalid(
                "required",
                format!("expected an array, got {other}"),
            ));
        }
    };
    Ok(SchemaNode::Object {
        properties,
        required,
    })
}

/// One open container in the parse stack.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    Object(ObjectFrame),
    Array(ArrayFrame),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectFrame {
    properties: Vec<(String, SchemaNode)>,
    required: Vec<usize>,
    phase: ObjectPhase,
    seen: Vec<bool>,
    /// The property whose value is being parsed, set when the key closes.
    key_index: Option<usize>,
}

impl ObjectFrame {
    fn unseen_indices(&self) -> Vec<usize> {
        (0..self.properties.len())
            .filter(|index| !self.seen[*index] && Some(*index) != self.key_index)
            .collect()
    }

    fn has_unseen(&self) -> bool {
        !self.unseen_indices().is_empty()
    }

    fn all_required_seen(&self) -> bool {
        self.required
            .iter()
            .all(|index| self.seen[*index] || Some(*index) == self.key_index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectPhase {
    /// Just after `{`: a key or the closing brace.
    Start,
    /// After a comma: a key is mandatory (no trailing comma).
    Key,
    /// After a key: the colon.
    Colon,
    /// After the colon: the property's value.
    Value,
    /// After the value: a comma or the closing brace.
    AfterValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArrayFrame {
    items: SchemaNode,
    min_items: usize,
    max_items: Option<usize>,
    phase: ArrayPhase,
    count: usize,
}

impl ArrayFrame {
    fn can_add(&self) -> bool {
        self.max_items.is_none_or(|max| self.count < max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayPhase {
    /// Just after `[`: the first element or the closing bracket.
    Start,
    /// After a comma: an element is mandatory.
    Value,
    /// After an element: a comma or the closing bracket.
    AfterValue,
}

/// The scalar or literal currently being parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueState {
    /// An object key, matched against the unseen property names.
    Key {
        /// How many bytes matched so far.
        matched: usize,
        /// The unseen properties whose name still has that prefix.
        candidates: Vec<usize>,
    },
    /// A string value.
    Str {
        /// Where the string is inside an escape sequence.
        escape: StrEscape,
    },
    /// A number literal.
    Number {
        /// Whether the schema said `integer` or `number`.
        kind: NumberKind,
        /// Where the literal is in the JSON number grammar.
        phase: NumberPhase,
    },
    /// A `true`/`false`/`null` literal.
    Word {
        /// The literal being matched.
        word: &'static [u8],
        /// How many bytes matched so far.
        matched: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrEscape {
    /// Not in an escape sequence.
    None,
    /// Just after a backslash.
    AfterBackslash,
    /// Inside `\uXXXX`, with the number of hex digits consumed.
    Unicode(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberKind {
    /// `integer`: the literal may not carry a fraction or an exponent.
    Integer,
    /// `number`: the full JSON number grammar.
    Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    /// Just after the sign, before the integer part's first digit.
    Sign,
    /// After the leading `0`; no further integer digit is permitted.
    Zero,
    /// Inside the integer part.
    Integer,
    /// Just after the decimal point: a digit is mandatory.
    FractionStart,
    /// Inside the fraction.
    Fraction,
    /// Just after `e`/`E`: a sign or a digit.
    ExponentStart,
    /// Just after the exponent's sign: a digit is mandatory.
    ExponentSign,
    /// Inside the exponent.
    ExponentDigits,
}

impl NumberPhase {
    /// Whether the literal may end here (the next byte may be structural).
    fn can_end(self) -> bool {
        matches!(
            self,
            Self::Zero | Self::Integer | Self::Fraction | Self::ExponentDigits
        )
    }
}

/// What advancing the automaton by one byte produced.
enum ValueOutcome {
    /// The byte was consumed and the value continues.
    Continue(ValueState),
    /// The byte completed the value.
    ConsumedLast,
    /// The byte closed an object key.
    KeyClosed,
    /// The value ends before this byte, which the container then consumes.
    Ended,
}

/// An incremental JSON-Schema byte automaton.
///
/// Feed it the bytes of the document being generated, querying [`SchemaConstraint::allowed_bytes`]
/// between steps. A constraint is cheap to clone, which is what [`SchemaConstraint::admits`] uses to
/// test a candidate token without disturbing the live state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaConstraint {
    root: SchemaNode,
    stack: Vec<Frame>,
    value: Option<ValueState>,
    finished: bool,
    position: usize,
}

impl SchemaConstraint {
    /// Compiles `schema` and returns a constraint at the start of the document.
    ///
    /// # Errors
    ///
    /// The same errors as [`compile`].
    pub fn new(schema: &Value) -> Result<Self, SchemaError> {
        Ok(Self {
            root: compile(schema)?,
            stack: Vec::new(),
            value: None,
            finished: false,
            position: 0,
        })
    }

    /// Which bytes may come next.
    #[must_use]
    pub fn allowed_bytes(&self) -> ByteMask {
        if self.finished {
            return whitespace_mask();
        }
        if let Some(value) = &self.value {
            let mut mask = self.value_mask(value);
            if matches!(value, ValueState::Number { phase, .. } if phase.can_end()) {
                mask.union_with(self.after_value_mask());
            }
            return mask;
        }
        self.structural_mask()
    }

    /// Consumes one byte.
    ///
    /// # Errors
    ///
    /// [`ConstraintError`] when the byte cannot extend any document that satisfies the schema.
    pub fn feed(&mut self, byte: u8) -> Result<(), ConstraintError> {
        let expected = self.allowed_bytes();
        let result = self.feed_inner(byte);
        if result.is_err() {
            return Err(ConstraintError {
                position: self.position,
                byte,
                expected,
            });
        }
        Ok(())
    }

    /// Consumes a byte string, stopping at the first refused byte.
    ///
    /// # Errors
    ///
    /// The first [`ConstraintError`] the bytes produce.
    pub fn feed_str(&mut self, bytes: &[u8]) -> Result<(), ConstraintError> {
        for byte in bytes {
            self.feed(*byte)?;
        }
        Ok(())
    }

    /// Whether a whole candidate token's bytes keep a valid completion reachable.
    ///
    /// This is the query a logits processor makes for every vocabulary token; it works on a clone,
    /// so the live constraint is unchanged.
    #[must_use]
    pub fn admits(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        probe.feed_str(bytes).is_ok()
    }

    /// Whether the document has ended and satisfies the schema structurally.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.finished
    }

    /// Whether the document may end here, i.e. whether end-of-sequence is a legal continuation.
    ///
    /// A number literal is the one token whose end is only decided by the byte *after* it, so the
    /// automaton ends it lazily: this query and [`finish_document`](Self::finish_document) are what
    /// a logits processor uses to decide whether EOS is admissible.
    #[must_use]
    pub fn allows_end(&self) -> bool {
        if self.finished {
            return true;
        }
        let mut probe = self.clone();
        probe.finish_document()
    }

    /// Marks end-of-input and reports whether the whole document satisfies the schema.
    ///
    /// Idempotent: finishing a finished document returns `true` again.
    pub fn finish_document(&mut self) -> bool {
        let pending = match &self.value {
            Some(ValueState::Number { phase, .. }) if phase.can_end() => true,
            Some(_) => return false,
            None => true,
        };
        if pending && self.value.is_some() {
            self.value = None;
            self.finish_value();
        }
        self.finished
    }

    /// How many bytes have been accepted.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    fn feed_inner(&mut self, byte: u8) -> Result<(), ()> {
        if self.finished {
            if is_whitespace(byte) {
                self.position += 1;
                return Ok(());
            }
            return Err(());
        }
        if self.value.is_none() {
            return self.feed_structural(byte);
        }
        let state = self.value.take().expect("a value state was checked");
        match self.advance_value(state, byte) {
            Ok(ValueOutcome::Continue(state)) => {
                self.value = Some(state);
                self.position += 1;
                Ok(())
            }
            Ok(ValueOutcome::ConsumedLast) => {
                self.position += 1;
                self.finish_value();
                Ok(())
            }
            Ok(ValueOutcome::KeyClosed) => {
                self.position += 1;
                Ok(())
            }
            Ok(ValueOutcome::Ended) => {
                self.finish_value();
                self.feed_structural(byte)
            }
            Err(()) => Err(()),
        }
    }

    fn feed_structural(&mut self, byte: u8) -> Result<(), ()> {
        if is_whitespace(byte) {
            self.position += 1;
            return Ok(());
        }
        // A completed document cannot start a second value: the earlier value's last byte already
        // closed it, and only trailing whitespace may follow.
        if self.finished {
            return Err(());
        }
        match self.step(byte) {
            Some(step) => self.apply(step, byte),
            None => Err(()),
        }
    }

    /// Decides what a structural byte does without mutating the automaton.
    fn step(&self, byte: u8) -> Option<Step> {
        let Some(frame) = self.stack.last() else {
            return Some(Step::BeginRoot);
        };
        match frame {
            Frame::Object(object) => match object.phase {
                ObjectPhase::Start => {
                    if byte == b'}' && object.required.is_empty() {
                        Some(Step::EndContainer)
                    } else if byte == b'"' && !object.properties.is_empty() {
                        Some(Step::BeginKey)
                    } else {
                        None
                    }
                }
                ObjectPhase::Key => (byte == b'"' && object.has_unseen()).then_some(Step::BeginKey),
                ObjectPhase::Colon => (byte == b':').then_some(Step::Colon),
                ObjectPhase::Value => object.key_index.map(Step::ObjectValue),
                ObjectPhase::AfterValue => {
                    if byte == b',' && object.has_unseen() {
                        Some(Step::Comma)
                    } else if byte == b'}' && object.all_required_seen() {
                        Some(Step::EndContainer)
                    } else {
                        None
                    }
                }
            },
            Frame::Array(array) => match array.phase {
                ArrayPhase::Start => {
                    if byte == b']' {
                        (array.count >= array.min_items).then_some(Step::EndContainer)
                    } else {
                        Some(Step::ArrayValue)
                    }
                }
                ArrayPhase::Value => Some(Step::ArrayValue),
                ArrayPhase::AfterValue => {
                    if byte == b',' && array.can_add() {
                        Some(Step::Comma)
                    } else if byte == b']' && array.count >= array.min_items {
                        Some(Step::EndContainer)
                    } else {
                        None
                    }
                }
            },
        }
    }

    fn apply(&mut self, step: Step, byte: u8) -> Result<(), ()> {
        match step {
            Step::BeginRoot => {
                let node = self.root.clone();
                self.begin_value(node, byte)
            }
            Step::BeginKey => {
                let Some(Frame::Object(object)) = self.stack.last() else {
                    return Err(());
                };
                let candidates = object.unseen_indices();
                if candidates.is_empty() {
                    return Err(());
                }
                self.value = Some(ValueState::Key {
                    matched: 0,
                    candidates,
                });
                self.position += 1;
                Ok(())
            }
            Step::Colon => {
                let Some(Frame::Object(object)) = self.stack.last_mut() else {
                    return Err(());
                };
                object.phase = ObjectPhase::Value;
                self.position += 1;
                Ok(())
            }
            Step::ObjectValue(index) => {
                let Some(Frame::Object(object)) = self.stack.last() else {
                    return Err(());
                };
                let Some((_, node)) = object.properties.get(index) else {
                    return Err(());
                };
                let node = node.clone();
                self.begin_value(node, byte)
            }
            Step::ArrayValue => {
                let Some(Frame::Array(array)) = self.stack.last() else {
                    return Err(());
                };
                let node = array.items.clone();
                self.begin_value(node, byte)
            }
            Step::Comma => {
                match self.stack.last_mut() {
                    Some(Frame::Object(object)) => object.phase = ObjectPhase::Key,
                    Some(Frame::Array(array)) => array.phase = ArrayPhase::Value,
                    None => return Err(()),
                }
                self.position += 1;
                Ok(())
            }
            Step::EndContainer => {
                let closed = self.stack.pop().ok_or(())?;
                match (closed, byte) {
                    (Frame::Object(_), b'}') | (Frame::Array(_), b']') => {}
                    _ => return Err(()),
                }
                self.position += 1;
                self.finish_value();
                Ok(())
            }
        }
    }

    /// Starts a value of `node`, consuming its first byte.
    fn begin_value(&mut self, node: SchemaNode, byte: u8) -> Result<(), ()> {
        match node {
            SchemaNode::Object {
                properties,
                required,
            } => {
                if byte != b'{' {
                    return Err(());
                }
                let seen = vec![false; properties.len()];
                self.stack.push(Frame::Object(ObjectFrame {
                    properties,
                    required,
                    phase: ObjectPhase::Start,
                    seen,
                    key_index: None,
                }));
            }
            SchemaNode::Array {
                items,
                min_items,
                max_items,
            } => {
                if byte != b'[' {
                    return Err(());
                }
                self.stack.push(Frame::Array(ArrayFrame {
                    items: *items,
                    min_items,
                    max_items,
                    phase: ArrayPhase::Start,
                    count: 0,
                }));
            }
            SchemaNode::String => {
                if byte != b'"' {
                    return Err(());
                }
                self.value = Some(ValueState::Str {
                    escape: StrEscape::None,
                });
            }
            SchemaNode::Bool => {
                let word: &'static [u8] = match byte {
                    b't' => b"true",
                    b'f' => b"false",
                    _ => return Err(()),
                };
                self.value = Some(ValueState::Word { word, matched: 1 });
            }
            SchemaNode::Null => {
                if byte != b'n' {
                    return Err(());
                }
                self.value = Some(ValueState::Word {
                    word: b"null",
                    matched: 1,
                });
            }
            SchemaNode::Integer => self.begin_number(NumberKind::Integer, byte)?,
            SchemaNode::Number => self.begin_number(NumberKind::Number, byte)?,
        }
        self.position += 1;
        Ok(())
    }

    fn begin_number(&mut self, kind: NumberKind, byte: u8) -> Result<(), ()> {
        let phase = match byte {
            b'-' => NumberPhase::Sign,
            b'0' => NumberPhase::Zero,
            b'1'..=b'9' => NumberPhase::Integer,
            _ => return Err(()),
        };
        self.value = Some(ValueState::Number { kind, phase });
        Ok(())
    }

    /// Marks the current container as having consumed a value.
    fn finish_value(&mut self) {
        match self.stack.last_mut() {
            Some(Frame::Object(object)) => {
                if let Some(index) = object.key_index.take() {
                    object.seen[index] = true;
                }
                object.phase = ObjectPhase::AfterValue;
            }
            Some(Frame::Array(array)) => {
                array.count += 1;
                array.phase = ArrayPhase::AfterValue;
            }
            None => self.finished = true,
        }
    }

    fn advance_value(&mut self, state: ValueState, byte: u8) -> Result<ValueOutcome, ()> {
        match state {
            ValueState::Key {
                matched,
                candidates,
            } => {
                let names = match self.stack.last() {
                    Some(Frame::Object(object)) => object.properties.as_slice(),
                    _ => return Err(()),
                };
                if byte == b'"' {
                    let index = candidates
                        .iter()
                        .copied()
                        .find(|index| names[*index].0.len() == matched)
                        .ok_or(())?;
                    if let Some(Frame::Object(object)) = self.stack.last_mut() {
                        object.phase = ObjectPhase::Colon;
                        object.key_index = Some(index);
                    }
                    return Ok(ValueOutcome::KeyClosed);
                }
                let mut next = Vec::with_capacity(candidates.len());
                for index in candidates {
                    if names[index].0.as_bytes().get(matched) == Some(&byte) {
                        next.push(index);
                    }
                }
                if next.is_empty() {
                    return Err(());
                }
                Ok(ValueOutcome::Continue(ValueState::Key {
                    matched: matched + 1,
                    candidates: next,
                }))
            }
            ValueState::Str { escape } => {
                if escape == StrEscape::None && byte == b'"' {
                    return Ok(ValueOutcome::ConsumedLast);
                }
                let escape = advance_string(escape, byte)?;
                Ok(ValueOutcome::Continue(ValueState::Str { escape }))
            }
            ValueState::Word { word, matched } => {
                if word.get(matched) != Some(&byte) {
                    return Err(());
                }
                if matched + 1 == word.len() {
                    return Ok(ValueOutcome::ConsumedLast);
                }
                Ok(ValueOutcome::Continue(ValueState::Word {
                    word,
                    matched: matched + 1,
                }))
            }
            ValueState::Number { kind, phase } => match advance_number(kind, phase, byte) {
                Ok(phase) => Ok(ValueOutcome::Continue(ValueState::Number { kind, phase })),
                Err(()) if phase.can_end() => Ok(ValueOutcome::Ended),
                Err(()) => Err(()),
            },
        }
    }

    fn value_mask(&self, state: &ValueState) -> ByteMask {
        match state {
            ValueState::Key {
                matched,
                candidates,
            } => {
                let mut mask = ByteMask::EMPTY;
                let Some(Frame::Object(object)) = self.stack.last() else {
                    return mask;
                };
                for index in candidates {
                    let name = object.properties[*index].0.as_bytes();
                    if let Some(byte) = name.get(*matched) {
                        mask.insert(*byte);
                    }
                    if name.len() == *matched {
                        mask.insert(b'"');
                    }
                }
                mask
            }
            ValueState::Str { escape } => string_mask(*escape),
            ValueState::Word { word, matched } => {
                let mut mask = ByteMask::EMPTY;
                if let Some(byte) = word.get(*matched) {
                    mask.insert(*byte);
                }
                mask
            }
            ValueState::Number { kind, phase } => number_mask(*kind, *phase),
        }
    }

    /// Bytes that may follow the pending scalar value once it ends.
    fn after_value_mask(&self) -> ByteMask {
        let mut mask = whitespace_mask();
        match self.stack.last() {
            None => {}
            Some(Frame::Object(object)) => {
                if object.has_unseen() {
                    mask.insert(b',');
                }
                if object.all_required_seen() {
                    mask.insert(b'}');
                }
            }
            Some(Frame::Array(array)) => {
                let count = array.count + 1;
                if array.max_items.is_none_or(|max| count < max) {
                    mask.insert(b',');
                }
                if count >= array.min_items {
                    mask.insert(b']');
                }
            }
        }
        mask
    }

    fn structural_mask(&self) -> ByteMask {
        let mut mask = whitespace_mask();
        let Some(frame) = self.stack.last() else {
            start_mask(&self.root, &mut mask);
            return mask;
        };
        match frame {
            Frame::Object(object) => match object.phase {
                ObjectPhase::Start => {
                    if object.required.is_empty() {
                        mask.insert(b'}');
                    }
                    if !object.properties.is_empty() {
                        mask.insert(b'"');
                    }
                }
                ObjectPhase::Key => {
                    if object.has_unseen() {
                        mask.insert(b'"');
                    }
                }
                ObjectPhase::Colon => mask.insert(b':'),
                ObjectPhase::Value => {
                    if let Some(index) = object.key_index {
                        start_mask(&object.properties[index].1, &mut mask);
                    }
                }
                ObjectPhase::AfterValue => {
                    if object.has_unseen() {
                        mask.insert(b',');
                    }
                    if object.all_required_seen() {
                        mask.insert(b'}');
                    }
                }
            },
            Frame::Array(array) => match array.phase {
                ArrayPhase::Start | ArrayPhase::Value => {
                    start_mask(&array.items, &mut mask);
                    if array.phase == ArrayPhase::Start && array.count >= array.min_items {
                        mask.insert(b']');
                    }
                }
                ArrayPhase::AfterValue => {
                    if array.can_add() {
                        mask.insert(b',');
                    }
                    if array.count >= array.min_items {
                        mask.insert(b']');
                    }
                }
            },
        }
        mask
    }
}

/// The structural decision for one byte, taken before the automaton mutates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    BeginRoot,
    BeginKey,
    Colon,
    ObjectValue(usize),
    ArrayValue,
    Comma,
    EndContainer,
}

fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn whitespace_mask() -> ByteMask {
    ByteMask::from_bytes(*b" \t\n\r")
}

/// The bytes that can start a value of `node`, added to `mask`.
fn start_mask(node: &SchemaNode, mask: &mut ByteMask) {
    match node {
        SchemaNode::Null => mask.insert(b'n'),
        SchemaNode::Bool => {
            mask.insert(b't');
            mask.insert(b'f');
        }
        SchemaNode::Integer | SchemaNode::Number => {
            mask.insert(b'-');
            for digit in b'0'..=b'9' {
                mask.insert(digit);
            }
        }
        SchemaNode::String => mask.insert(b'"'),
        SchemaNode::Array { .. } => mask.insert(b'['),
        SchemaNode::Object { .. } => mask.insert(b'{'),
    }
}

fn string_mask(escape: StrEscape) -> ByteMask {
    match escape {
        // Every printable and non-ASCII byte, including the closing quote and the escape
        // introducer; raw control bytes are not legal in a JSON string.
        StrEscape::None => ByteMask::from_bytes(0x20..=0xFF),
        StrEscape::AfterBackslash => ByteMask::from_bytes(*b"\"\\/bfnrtu"),
        StrEscape::Unicode(_) => {
            ByteMask::from_bytes((b'0'..=b'9').chain(b'a'..=b'f').chain(b'A'..=b'F'))
        }
    }
}

fn number_mask(kind: NumberKind, phase: NumberPhase) -> ByteMask {
    let digits = || ByteMask::from_bytes(b'0'..=b'9');
    match phase {
        NumberPhase::Sign => digits(),
        NumberPhase::Zero => {
            let mut mask = ByteMask::EMPTY;
            if kind == NumberKind::Number {
                mask.insert(b'.');
                mask.insert(b'e');
                mask.insert(b'E');
            }
            mask
        }
        NumberPhase::Integer => {
            let mut mask = digits();
            if kind == NumberKind::Number {
                mask.insert(b'.');
                mask.insert(b'e');
                mask.insert(b'E');
            }
            mask
        }
        NumberPhase::FractionStart | NumberPhase::ExponentSign => digits(),
        NumberPhase::Fraction | NumberPhase::ExponentDigits => {
            let mut mask = digits();
            if phase == NumberPhase::Fraction {
                mask.insert(b'e');
                mask.insert(b'E');
            }
            mask
        }
        NumberPhase::ExponentStart => {
            let mut mask = digits();
            mask.insert(b'+');
            mask.insert(b'-');
            mask
        }
    }
}

fn advance_string(escape: StrEscape, byte: u8) -> Result<StrEscape, ()> {
    match escape {
        StrEscape::None => match byte {
            b'\\' => Ok(StrEscape::AfterBackslash),
            0x00..=0x1F => Err(()),
            _ => Ok(StrEscape::None),
        },
        StrEscape::AfterBackslash => match byte {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Ok(StrEscape::None),
            b'u' => Ok(StrEscape::Unicode(0)),
            _ => Err(()),
        },
        StrEscape::Unicode(consumed) => {
            if !byte.is_ascii_hexdigit() {
                return Err(());
            }
            if consumed == 3 {
                Ok(StrEscape::None)
            } else {
                Ok(StrEscape::Unicode(consumed + 1))
            }
        }
    }
}

fn advance_number(kind: NumberKind, phase: NumberPhase, byte: u8) -> Result<NumberPhase, ()> {
    match phase {
        NumberPhase::Sign => match byte {
            b'0' => Ok(NumberPhase::Zero),
            b'1'..=b'9' => Ok(NumberPhase::Integer),
            _ => Err(()),
        },
        NumberPhase::Zero => match (kind, byte) {
            (NumberKind::Number, b'.') => Ok(NumberPhase::FractionStart),
            (NumberKind::Number, b'e' | b'E') => Ok(NumberPhase::ExponentStart),
            _ => Err(()),
        },
        NumberPhase::Integer => match byte {
            b'0'..=b'9' => Ok(NumberPhase::Integer),
            b'.' if kind == NumberKind::Number => Ok(NumberPhase::FractionStart),
            b'e' | b'E' if kind == NumberKind::Number => Ok(NumberPhase::ExponentStart),
            _ => Err(()),
        },
        NumberPhase::FractionStart => match byte {
            b'0'..=b'9' => Ok(NumberPhase::Fraction),
            _ => Err(()),
        },
        NumberPhase::Fraction => match byte {
            b'0'..=b'9' => Ok(NumberPhase::Fraction),
            b'e' | b'E' => Ok(NumberPhase::ExponentStart),
            _ => Err(()),
        },
        NumberPhase::ExponentStart => match byte {
            b'+' | b'-' => Ok(NumberPhase::ExponentSign),
            b'0'..=b'9' => Ok(NumberPhase::ExponentDigits),
            _ => Err(()),
        },
        NumberPhase::ExponentSign => match byte {
            b'0'..=b'9' => Ok(NumberPhase::ExponentDigits),
            _ => Err(()),
        },
        NumberPhase::ExponentDigits => match byte {
            b'0'..=b'9' => Ok(NumberPhase::ExponentDigits),
            _ => Err(()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `document` and checks, at every position, that the mask and [`SchemaConstraint::feed`]
    /// agree byte for byte: a byte is in the mask exactly when feeding it succeeds.
    fn assert_mask_matches_feed(schema: &Value, document: &str) {
        let mut constraint = SchemaConstraint::new(schema).expect("the schema compiles");
        for (index, byte) in document.bytes().enumerate() {
            let allowed = constraint.allowed_bytes();
            for candidate in 0u8..=255 {
                let feeds = constraint.clone().feed(candidate).is_ok();
                assert_eq!(
                    allowed.contains(candidate),
                    feeds,
                    "offset {index}: mask says {} for {candidate:#04x}, feed says {feeds}",
                    allowed.contains(candidate)
                );
            }
            constraint.feed(byte).unwrap_or_else(|error| {
                panic!("offset {index}: document byte {byte:#04x} was refused: {error}")
            });
        }
        assert!(
            constraint.allows_end(),
            "the document may end at its last byte"
        );
        assert!(
            constraint.finish_document(),
            "the document satisfies the schema"
        );
        assert!(constraint.is_complete());
        let allowed = constraint.allowed_bytes();
        for candidate in 0u8..=255 {
            assert_eq!(
                allowed.contains(candidate),
                constraint.clone().feed(candidate).is_ok(),
                "after completion: mask and feed disagree on {candidate:#04x}"
            );
        }
    }

    fn object_schema(properties: Value, required: Value) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    }

    #[test]
    fn a_flat_object_document_is_accepted() {
        let schema = object_schema(
            serde_json::json!({"city": {"type": "string"}, "days": {"type": "integer"}}),
            serde_json::json!(["city"]),
        );
        assert_mask_matches_feed(&schema, r#"{"city":"Beijing","days":3}"#);
        assert_mask_matches_feed(&schema, r#"{"city":""}"#);
        assert_mask_matches_feed(&schema, "  { \"city\" : \"x\" }  ");
    }

    #[test]
    fn an_unknown_key_is_refused_before_it_is_written() {
        let schema = object_schema(
            serde_json::json!({"city": {"type": "string"}}),
            serde_json::json!([]),
        );
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"{\"").expect("the object starts");
        // No property starts with 't', so the byte cannot extend this key.
        let mut refused = constraint.clone();
        let error = refused.feed(b't').expect_err("no property starts with t");
        assert_eq!(error.position, 2);
        assert!(!error.expected.is_empty());
        constraint.feed_str(b"city\"").expect("the key matches");
        constraint.feed(b':').expect("the colon follows");
    }

    #[test]
    fn a_missing_required_property_keeps_the_object_open() {
        let schema = object_schema(
            serde_json::json!({"a": {"type": "boolean"}, "b": {"type": "boolean"}}),
            serde_json::json!(["b"]),
        );
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"{\"a\":true").expect("a is written");
        assert!(
            !constraint.allowed_bytes().contains(b'}'),
            "b is still required, so the object cannot close"
        );
        constraint.feed(b',').expect("a comma introduces b");
        constraint.feed_str(b"\"b\":false").expect("b is written");
        assert!(constraint.allowed_bytes().contains(b'}'));
        constraint.feed(b'}').expect("the object closes");
        assert!(constraint.allows_end());
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
    }

    #[test]
    fn duplicate_keys_and_trailing_commas_are_refused() {
        let schema = object_schema(
            serde_json::json!({"a": {"type": "null"}}),
            serde_json::json!([]),
        );
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"{\"a\":null").expect("a is written");
        assert!(!constraint.allowed_bytes().contains(b','));
        let mut closed = constraint.clone();
        closed.feed(b'}').expect("the object closes");
        assert!(closed.allows_end());
        assert!(closed.finish_document());
        assert!(closed.is_complete());
        let mut trailing = SchemaConstraint::new(&schema).expect("compiles");
        assert!(
            trailing.feed_str(b"{\"a\":null,").is_err(),
            "a trailing comma is not JSON"
        );
    }

    #[test]
    fn array_bounds_gate_the_comma_and_the_closing_bracket() {
        let schema = serde_json::json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 2,
        });
        assert_mask_matches_feed(&schema, "[1]");
        assert_mask_matches_feed(&schema, "[1,2]");
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed(b'[').expect("the array starts");
        assert!(
            !constraint.allowed_bytes().contains(b']'),
            "minItems keeps the empty array out"
        );
        constraint.feed(b'1').expect("the first element");
        constraint.feed(b',').expect("a comma");
        constraint.feed(b'2').expect("the second element");
        assert!(
            !constraint.allowed_bytes().contains(b','),
            "maxItems is reached, so no third element"
        );
        constraint.feed(b']').expect("the array closes");
        assert!(constraint.allows_end());
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
    }

    #[test]
    fn an_integer_schema_refuses_fractions_and_exponents() {
        let schema = serde_json::json!({"type": "integer"});
        assert_mask_matches_feed(&schema, "-12");
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed(b'1').expect("a digit starts the literal");
        assert!(!constraint.allowed_bytes().contains(b'.'));
        assert!(!constraint.allowed_bytes().contains(b'e'));
        assert!(constraint.clone().feed(b'.').is_err());
    }

    #[test]
    fn a_number_schema_accepts_the_full_json_number_grammar() {
        let schema = serde_json::json!({"type": "number"});
        for document in ["0", "-1.5", "1e10", "2.5E-3", "0.0"] {
            assert_mask_matches_feed(&schema, document);
        }
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed(b'2').expect("a digit");
        constraint.feed(b'.').expect("the point");
        assert!(
            !constraint.allowed_bytes().contains(b'.'),
            "a second point is not a number"
        );
    }

    #[test]
    fn a_leading_zero_cannot_be_followed_by_a_digit() {
        let schema = serde_json::json!({"type": "integer"});
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed(b'0').expect("zero");
        assert!(!constraint.allowed_bytes().contains(b'1'));
        assert!(constraint.clone().feed(b'1').is_err());
    }

    #[test]
    fn strings_allow_escapes_and_refuse_raw_control_bytes() {
        let schema = serde_json::json!({"type": "string"});
        assert_mask_matches_feed(&schema, r#""plain""#);
        assert_mask_matches_feed(&schema, r#""a \"quoted\" value""#);
        assert_mask_matches_feed(&schema, r#""\u00e9\ntab\t""#);
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed(b'"').expect("the string starts");
        assert!(constraint.clone().feed(0x01).is_err());
        constraint.feed(b'\\').expect("an escape starts");
        assert!(constraint.clone().feed(b'x').is_err());
        constraint.feed(b'u').expect("a unicode escape");
        assert!(constraint.clone().feed(b'z').is_err());
        constraint.feed_str(b"00e9").expect("four hex digits");
        assert!(
            constraint.allowed_bytes().contains(0xC3),
            "UTF-8 bytes are fine"
        );
        constraint.feed(b'"').expect("the string closes");
        assert!(constraint.allows_end());
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
    }

    #[test]
    fn booleans_and_null_are_word_literals() {
        assert_mask_matches_feed(&serde_json::json!({"type": "boolean"}), "true");
        assert_mask_matches_feed(&serde_json::json!({"type": "boolean"}), "false");
        assert_mask_matches_feed(&serde_json::json!({"type": "null"}), "null");
        let schema = serde_json::json!({"type": "boolean"});
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"tru").expect("a prefix");
        assert!(!constraint.allowed_bytes().contains(b't'));
        constraint.feed(b'e').expect("the last letter");
        assert!(constraint.allows_end());
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
    }

    #[test]
    fn trailing_bytes_after_the_root_value_are_refused() {
        let schema = serde_json::json!({"type": "null"});
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"null").expect("the document");
        assert!(constraint.allows_end());
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
        assert!(constraint.clone().feed(b'x').is_err());
        constraint.feed(b' ').expect("trailing whitespace is fine");
        constraint.feed(b'\n').expect("trailing whitespace is fine");
        assert!(constraint.clone().feed(b'x').is_err());
    }

    #[test]
    fn nested_schemas_compose() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false,
                    },
                },
            },
            "required": ["calls"],
            "additionalProperties": false,
        });
        assert_mask_matches_feed(&schema, r#"{"calls":[]}"#);
        assert_mask_matches_feed(&schema, r#"{"calls":[{"name":"one"},{"name":"two"}]}"#);
    }

    #[test]
    fn mask_operations_are_exact() {
        let mut mask = ByteMask::EMPTY;
        assert!(mask.is_empty());
        assert_eq!(mask.count(), 0);
        mask.insert(0x41);
        mask.insert(0xFF);
        assert!(mask.contains(0x41));
        assert!(mask.contains(0xFF));
        assert!(!mask.contains(0x40));
        assert_eq!(mask.count(), 2);
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0x41, 0xFF]);
        mask.remove(0x41);
        assert!(!mask.contains(0x41));
        mask.union_with(ByteMask::from_bytes([0x00, 0x7F]));
        assert_eq!(mask.count(), 3);
        assert_eq!(ByteMask::full().count(), 256);
    }

    #[test]
    fn admits_tests_a_whole_token_without_moving_the_state() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false,
        });
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        assert!(constraint.admits(b"{\"a\""));
        assert!(!constraint.admits(b"{\"b\""));
        constraint.feed_str(b"{\"a\":").expect("the value is due");
        assert!(constraint.admits(b"12"));
        assert!(!constraint.admits(b"1.5"));
        assert!(!constraint.admits(b"true"));
        // The probe did not consume anything.
        assert!(constraint.allowed_bytes().contains(b'1'));
    }

    #[test]
    fn unsupported_keywords_are_refused_typed() {
        for schema in [
            serde_json::json!({"enum": ["a", "b"]}),
            serde_json::json!({"const": 1}),
            serde_json::json!({"type": "string", "pattern": "^a"}),
            serde_json::json!({"type": "string", "format": "date-time"}),
            serde_json::json!({"type": "integer", "minimum": 0}),
            serde_json::json!({"anyOf": [{"type": "null"}]}),
            serde_json::json!({"$ref": "#/$defs/x"}),
            serde_json::json!({"type": ["string", "null"]}),
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": true,
            }),
            serde_json::json!({"type": "array", "items": [{"type": "null"}]}),
            serde_json::json!({}),
        ] {
            let error = SchemaConstraint::new(&schema)
                .expect_err("the schema is outside the supported subset");
            assert!(
                matches!(
                    error,
                    SchemaError::Unsupported { .. } | SchemaError::Invalid { .. }
                ),
                "{schema}: {error:?}"
            );
        }
    }

    #[test]
    fn malformed_supported_keywords_are_refused() {
        let error = SchemaConstraint::new(&serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "null"}},
            "required": ["b"],
        }))
        .expect_err("a required property must be declared");
        assert!(matches!(error, SchemaError::Invalid { .. }), "{error:?}");
        let error = SchemaConstraint::new(&serde_json::json!({
            "type": "array",
            "items": {"type": "null"},
            "minItems": 3,
            "maxItems": 1,
        }))
        .expect_err("the bounds are contradictory");
        assert!(matches!(error, SchemaError::Invalid { .. }), "{error:?}");
        let error = SchemaConstraint::new(&serde_json::json!({"type": "widget"}))
            .expect_err("widget is not a JSON type");
        assert!(
            matches!(error, SchemaError::Unsupported { .. }),
            "{error:?}"
        );
        assert!(matches!(
            SchemaConstraint::new(&serde_json::json!("nope")).expect_err("not an object"),
            SchemaError::NotAnObject
        ));
    }

    #[test]
    fn a_document_that_ends_early_is_not_complete() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "null"}},
            "required": ["a"],
            "additionalProperties": false,
        });
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(b"{\"a\":null").expect("the property");
        assert!(!constraint.allows_end(), "the object is still open");
        constraint.feed(b'}').expect("the object closes");
        assert!(constraint.finish_document());
        assert!(constraint.is_complete());
    }
}
