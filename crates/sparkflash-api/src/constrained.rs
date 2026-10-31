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
//! # Where it runs
//!
//! [`TokenConstraint`] drives the same automaton by vocabulary id, reading each id's bytes from
//! [`TokenBytes`]. The device path keeps one per constrained sequence on every rank: the masked
//! head takes the best admitted id inside each rank's vocabulary slice, and the picked id is fed
//! back with [`TokenConstraint::advance`] (DECISIONS.md D64). An engine without that mask reports
//! `constrained: false`, and the router refuses `response_format` with a typed 400 rather than
//! render the schema into the prompt and hope (see `refuse_unenforceable_response_format`).
//!
//! # Supported schema subset
//!
//! The automaton is deterministic by construction: every state has exactly one way to continue for
//! a given byte, so a mask never has to merge alternatives. That rules out the ambiguous keywords,
//! which are refused at compile time with a typed [`SchemaError::Unsupported`] instead of being
//! silently ignored. Supported keywords:
//!
//! * `type`: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null` (a union type is
//!   refused: it would need choice-state masking). Without `type`, the object and array keywords
//!   below imply it; a schema that implies no type is refused;
//! * objects: `properties`, `required`, `additionalProperties: false`. An object schema that names
//!   properties or required keys without `additionalProperties` is treated as closed, which can
//!   only over-constrain, never admit an invalid document. A bare `{"type": "object"}` is open: any
//!   keys with any JSON values, which is what `json_object` compiles to;
//! * arrays: `items` (required), `minItems`, `maxItems`;
//! * strings: `enum` with string values, matched on their JSON-escaped spelling;
//! * integers and numbers: inclusive `minimum` and `maximum`, checked exactly on the decimal digits
//!   (see [`NumberBounds`]). A bounded literal is written without an exponent, and a range that
//!   holds no value of the type (the minimum above the maximum, or no integer between them) is
//!   refused;
//! * `boolean` and `null` scalars with the JSON grammar of the type.
//!
//! Refused keywords: `pattern`, `format`, `minLength`, `maxLength` (value automata, not
//! structural), `const`, `enum` on anything but a string, `anyOf`, `oneOf`, `allOf`, `not`,
//! `if`/`then`/`else`, `$ref`/`$defs`, `dependencies`, `dependentRequired`, `dependentSchemas`,
//! `propertyNames`, `patternProperties`, `minProperties`, `maxProperties`,
//! `additionalProperties: true`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`,
//! `uniqueItems`, `contains`, `prefixItems`, `additionalItems`, `unevaluatedProperties`.
//!
//! # How a sampler uses it
//!
//! 1. Keep one [`TokenConstraint`] per sequence, seeded from the request's schema.
//! 2. Before sampling, admit only the ids [`TokenConstraint::for_each_allowed_in`] yields, which
//!    walks the first-byte index of the vocabulary by the byte mask. End-of-sequence is admitted
//!    only where the document may end: a truncated JSON document is not valid.
//! 3. After sampling, [`TokenConstraint::advance`] with the picked id. When no id is admitted, the
//!    sequence cannot go on and ends with an error of its own; the other sequences continue.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::vocab::TokenBytes;

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
    /// An integer literal within `minimum`/`maximum`.
    BoundedInteger(Arc<NumberBounds>),
    /// Any JSON number literal.
    Number,
    /// A number literal within `minimum`/`maximum`, written without an exponent.
    BoundedNumber(Arc<NumberBounds>),
    /// A JSON string literal.
    String,
    /// A JSON string literal restricted to a fixed set of values (`enum`), each held as it is
    /// spelled between the quotes (see `json_spelling`).
    StringEnum(Arc<[String]>),
    /// Any valid JSON value (used inside an open `{"type": "object"}`).
    Any,
    /// A JSON array whose elements must satisfy `items`.
    Array {
        /// The element schema.
        items: Arc<SchemaNode>,
        /// Minimum element count (`minItems`, 0 when absent).
        min_items: usize,
        /// Maximum element count (`maxItems`, `None` when absent).
        max_items: Option<usize>,
    },
    /// A JSON object (closed when `properties` or `additionalProperties: false` is given, open
    /// for bare `{"type": "object"}`).
    Object {
        /// The permitted properties, in schema order, each name held as it is spelled between the
        /// quotes (see `json_spelling`).
        properties: Arc<[(String, SchemaNode)]>,
        /// Indices into `properties` that the document must contain (`required`).
        required: Arc<[usize]>,
        /// Whether arbitrary keys with arbitrary JSON values are permitted.
        open: bool,
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
    if map.contains_key("enum") && declared != Declared::String {
        return Err(unsupported(
            "enum",
            "only string enums are supported by the deterministic byte automaton",
        ));
    }
    if (map.contains_key("minimum") || map.contains_key("maximum"))
        && !matches!(declared, Declared::Integer | Declared::Number)
    {
        return Err(invalid(
            "minimum",
            "numeric bounds are only valid on integer or number schemas",
        ));
    }
    match declared {
        Declared::Null => Ok(SchemaNode::Null),
        Declared::Bool => Ok(SchemaNode::Bool),
        Declared::Integer => compile_number_schema(map, true),
        Declared::Number => compile_number_schema(map, false),
        Declared::String => compile_string_schema(map),
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

/// How `text` is spelled between a JSON document's quotes. Enum values and property names are
/// matched byte for byte, so the automaton has to match the escaped form a parser reads back as
/// `text`: matched raw, a `"` inside one would close the string early and leave no admissible
/// byte, and a backslash or control byte would spell different text or invalid JSON.
fn json_spelling(text: &str) -> String {
    let quoted = Value::from(text).to_string();
    quoted[1..quoted.len() - 1].to_owned()
}

fn compile_string_schema(map: &Map<String, Value>) -> Result<SchemaNode, SchemaError> {
    let Some(raw) = map.get("enum") else {
        return Ok(SchemaNode::String);
    };
    let Some(items) = raw.as_array() else {
        return Err(invalid("enum", format!("expected an array, got {raw}")));
    };
    if items.is_empty() {
        return Err(invalid("enum", "enum array must not be empty"));
    }
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(unsupported(
                "enum",
                "only string enum values are supported by the byte automaton",
            ));
        };
        let spelled = json_spelling(text);
        if !values.contains(&spelled) {
            values.push(spelled);
        }
    }
    Ok(SchemaNode::StringEnum(values.into()))
}

fn compile_number_schema(
    map: &Map<String, Value>,
    integer: bool,
) -> Result<SchemaNode, SchemaError> {
    let bound = |keyword: &str| match map.get(keyword) {
        None => Ok(None),
        Some(Value::Number(n)) => Decimal::parse(&n.to_string())
            .map(Some)
            .ok_or_else(|| invalid(keyword, format!("{n} cannot be held as an exact decimal"))),
        Some(other) => Err(invalid(keyword, format!("expected a number, got {other}"))),
    };
    let bounds = NumberBounds {
        min: bound("minimum")?,
        max: bound("maximum")?,
    };
    if let (Some(min), Some(max)) = (&bounds.min, &bounds.max)
        && min.cmp(max) == Ordering::Greater
    {
        return Err(invalid(
            "maximum",
            format!("maximum ({max}) is below minimum ({min})"),
        ));
    }
    if bounds.min.is_none() && bounds.max.is_none() {
        return Ok(if integer {
            SchemaNode::Integer
        } else {
            SchemaNode::Number
        });
    }
    if integer && !bounds.has_integer() {
        return Err(invalid(
            "minimum",
            "no integer lies between minimum and maximum",
        ));
    }
    let bounds = Arc::new(bounds);
    Ok(if integer {
        SchemaNode::BoundedInteger(bounds)
    } else {
        SchemaNode::BoundedNumber(bounds)
    })
}

/// Inclusive `minimum`/`maximum` of a number schema, held as exact decimals.
///
/// A bounded literal is checked digit by digit: a byte is admitted only while some literal that
/// extends it still lies in bounds, so the mask never leads into a number that cannot end. That
/// check compares decimal digits, not floats, because a rounding error there would admit a prefix
/// with no valid completion, which is a state with no admissible token. Exponents are refused
/// under bounds: every value is still writable without one, and the digit check stays simple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberBounds {
    min: Option<Decimal>,
    max: Option<Decimal>,
}

impl NumberBounds {
    /// The magnitudes `lo..=hi` (`hi` None when unbounded) a literal of this sign may take, or
    /// None when no literal of that sign lies in bounds. A negative literal needs `minimum < 0`,
    /// which only refuses `-0` where `0` itself is allowed.
    fn magnitudes(&self, negative: bool) -> Option<(Magnitude, Option<Magnitude>)> {
        if negative {
            let hi = match &self.min {
                Some(min) if !min.negative => return None,
                Some(min) => Some(min.magnitude.clone()),
                None => None,
            };
            let lo = match &self.max {
                Some(max) if max.negative => max.magnitude.clone(),
                _ => Magnitude::default(),
            };
            Some((lo, hi))
        } else {
            let hi = match &self.max {
                Some(max) if max.negative => return None,
                Some(max) => Some(max.magnitude.clone()),
                None => None,
            };
            let lo = match &self.min {
                Some(min) if !min.negative => min.magnitude.clone(),
                _ => Magnitude::default(),
            };
            Some((lo, hi))
        }
    }

    /// Whether any integer lies in bounds.
    fn has_integer(&self) -> bool {
        [false, true].into_iter().any(|negative| {
            self.magnitudes(negative).is_some_and(|(lo, hi)| {
                hi.is_none_or(|hi| lo.ceil().cmp(&hi.floor()) != Ordering::Greater)
            })
        })
    }
}

/// An exact decimal number.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Decimal {
    /// Never set on zero, so `-0` and `0` are one value.
    negative: bool,
    magnitude: Magnitude,
}

impl Decimal {
    /// Parses a JSON number literal (exponent included) without rounding it.
    fn parse(text: &str) -> Option<Self> {
        let (negative, rest) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let (mantissa, exponent) = match rest.find(['e', 'E']) {
            Some(at) => (&rest[..at], rest[at + 1..].parse::<i64>().ok()?),
            None => (rest, 0),
        };
        // Far beyond any f64; the digits would only be zeros to allocate.
        if exponent.abs() > 1024 {
            return None;
        }
        let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if int.is_empty() || !int.bytes().chain(frac.bytes()).all(|b| b.is_ascii_digit()) {
            return None;
        }
        let digits: Vec<u8> = int.bytes().chain(frac.bytes()).collect();
        let point = int.len() as i64 + exponent;
        let magnitude = if point <= 0 {
            let mut frac = vec![b'0'; point.unsigned_abs() as usize];
            frac.extend_from_slice(&digits);
            Magnitude::new(b"", &frac)
        } else if point as usize >= digits.len() {
            let mut int = digits;
            int.resize(point as usize, b'0');
            Magnitude::new(&int, b"")
        } else {
            let (int, frac) = digits.split_at(point as usize);
            Magnitude::new(int, frac)
        };
        Some(Self {
            negative: negative && !magnitude.is_zero(),
            magnitude,
        })
    }

    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (false, false) => self.magnitude.cmp(&other.magnitude),
            (true, true) => other.magnitude.cmp(&self.magnitude),
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
        }
    }
}

impl std::fmt::Display for Decimal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let int = if self.magnitude.int.is_empty() {
            "0"
        } else {
            std::str::from_utf8(&self.magnitude.int).unwrap_or("?")
        };
        let sign = if self.negative { "-" } else { "" };
        if self.magnitude.frac.is_empty() {
            write!(f, "{sign}{int}")
        } else {
            let frac = std::str::from_utf8(&self.magnitude.frac).unwrap_or("?");
            write!(f, "{sign}{int}.{frac}")
        }
    }
}

/// A non-negative exact decimal: ASCII integer digits without leading zeros (empty for zero) and
/// fraction digits without trailing zeros.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Magnitude {
    int: Vec<u8>,
    frac: Vec<u8>,
}

impl Magnitude {
    fn new(int: &[u8], frac: &[u8]) -> Self {
        let lead = int.iter().take_while(|&&digit| digit == b'0').count();
        let trail = frac
            .iter()
            .rev()
            .take_while(|&&digit| digit == b'0')
            .count();
        Self {
            int: int[lead..].to_vec(),
            frac: frac[..frac.len() - trail].to_vec(),
        }
    }

    fn is_zero(&self) -> bool {
        self.int.is_empty() && self.frac.is_empty()
    }

    fn cmp(&self, other: &Self) -> Ordering {
        self.int
            .len()
            .cmp(&other.int.len())
            .then_with(|| self.int.cmp(&other.int))
            .then_with(|| {
                let len = self.frac.len().max(other.frac.len());
                let digit = |frac: &[u8], at: usize| frac.get(at).copied().unwrap_or(b'0');
                (0..len)
                    .map(|at| digit(&self.frac, at).cmp(&digit(&other.frac, at)))
                    .find(|order| order.is_ne())
                    .unwrap_or(Ordering::Equal)
            })
    }

    fn floor(&self) -> Self {
        self.truncate(0)
    }

    fn ceil(&self) -> Self {
        if self.frac.is_empty() {
            self.clone()
        } else {
            Self::new(&increment(&self.int), b"")
        }
    }

    /// Rounded down to `places` fraction digits.
    fn truncate(&self, places: usize) -> Self {
        Self::new(&self.int, &self.frac[..places.min(self.frac.len())])
    }
}

/// `digits + 1` for an ASCII digit string (empty for zero).
fn increment(digits: &[u8]) -> Vec<u8> {
    let mut out = digits.to_vec();
    for digit in out.iter_mut().rev() {
        if *digit == b'9' {
            *digit = b'0';
        } else {
            *digit += 1;
            return out;
        }
    }
    out.insert(0, b'1');
    out
}

/// A bounded number literal as far as it has been written.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundedLiteral {
    bounds: Arc<NumberBounds>,
    negative: bool,
    int: Vec<u8>,
    frac: Vec<u8>,
}

impl BoundedLiteral {
    /// Records `byte`, which moved the literal into `phase`.
    fn push(&mut self, phase: NumberPhase, byte: u8) {
        match (phase, byte) {
            (NumberPhase::Sign, _) => self.negative = true,
            (NumberPhase::Zero | NumberPhase::Integer, digit) => self.int.push(digit),
            (NumberPhase::Fraction, digit) => self.frac.push(digit),
            _ => {}
        }
    }

    /// Whether this literal, ended here, lies in bounds.
    fn contains(&self) -> bool {
        let Some((lo, hi)) = self.bounds.magnitudes(self.negative) else {
            return false;
        };
        let value = Magnitude::new(&self.int, &self.frac);
        value.cmp(&lo) != Ordering::Less && hi.is_none_or(|hi| value.cmp(&hi) != Ordering::Greater)
    }

    /// Whether some literal that starts with this one (this one included, where it may end) lies
    /// in bounds. `phase` is where the literal is in the number grammar.
    fn reachable(&self, kind: NumberKind, phase: NumberPhase) -> bool {
        let Some((lo, hi)) = self.bounds.magnitudes(self.negative) else {
            return false;
        };
        let integer = kind == NumberKind::Integer;
        // An integer literal can only take whole magnitudes.
        let (lo, hi) = if integer {
            (lo.ceil(), hi.map(|hi| hi.floor()))
        } else {
            (lo, hi)
        };
        let below = |value: &Magnitude| {
            hi.as_ref()
                .is_none_or(|hi| value.cmp(hi) != Ordering::Greater)
        };
        if !below(&lo) {
            return false;
        }
        match phase {
            // Every magnitude is still possible.
            NumberPhase::Sign => true,
            // `0` alone, or `0.F` in [0, 1) for a number.
            NumberPhase::Zero => lo.floor().is_zero(),
            // Digits I: I itself, I.F, or I followed by k more digits, so the magnitudes are the
            // union over k of [I * 10^k, (I + 1) * 10^k). The blocks move right as k grows, so
            // one meets the bounds exactly when the largest k whose block starts at or below the
            // maximum also ends above the minimum.
            NumberPhase::Integer => {
                let value = Magnitude::new(&self.int, b"");
                let Some(hi) = &hi else {
                    return true;
                };
                let top = hi.floor().int;
                if value.cmp(&hi.floor()) == Ordering::Greater {
                    return false;
                }
                let mut k = top.len() - value.int.len();
                let mut start = value.int.clone();
                start.resize(top.len(), b'0');
                if start > top {
                    k -= 1;
                }
                let mut end = increment(&value.int);
                end.resize(end.len() + k, b'0');
                let floor = if integer { lo } else { lo.floor() };
                Magnitude::new(&end, b"").cmp(&floor) == Ordering::Greater
            }
            // `I.` then at least one digit: [I, I + 1).
            NumberPhase::FractionStart => {
                let value = Magnitude::new(&self.int, b"");
                below(&value) && lo.floor().cmp(&value) != Ordering::Greater
            }
            // `I.F` with n fraction digits: [I.F, I.F + 10^-n).
            NumberPhase::Fraction => {
                let value = Magnitude::new(&self.int, &self.frac);
                below(&value) && lo.truncate(self.frac.len()).cmp(&value) != Ordering::Greater
            }
            NumberPhase::ExponentStart
            | NumberPhase::ExponentSign
            | NumberPhase::ExponentDigits => false,
        }
    }
}

/// The first byte of a number of `kind`, with its bounds when it has some.
fn start_number(
    kind: NumberKind,
    bounds: Option<&Arc<NumberBounds>>,
    byte: u8,
) -> Result<(NumberPhase, Option<Box<BoundedLiteral>>), ()> {
    let phase = match byte {
        b'-' => NumberPhase::Sign,
        b'0' => NumberPhase::Zero,
        b'1'..=b'9' => NumberPhase::Integer,
        _ => return Err(()),
    };
    let Some(bounds) = bounds else {
        return Ok((phase, None));
    };
    let mut literal = Box::new(BoundedLiteral {
        bounds: Arc::clone(bounds),
        negative: false,
        int: Vec::new(),
        frac: Vec::new(),
    });
    literal.push(phase, byte);
    if !literal.reachable(kind, phase) {
        return Err(());
    }
    Ok((phase, Some(literal)))
}

/// One more byte of a number literal.
fn step_number(
    kind: NumberKind,
    phase: NumberPhase,
    bounded: Option<&BoundedLiteral>,
    byte: u8,
) -> Result<(NumberPhase, Option<Box<BoundedLiteral>>), ()> {
    let Some(literal) = bounded else {
        return advance_number(kind, phase, byte).map(|phase| (phase, None));
    };
    if matches!(byte, b'e' | b'E') {
        return Err(());
    }
    let phase = advance_number(kind, phase, byte)?;
    let mut literal = Box::new(literal.clone());
    literal.push(phase, byte);
    if !literal.reachable(kind, phase) {
        return Err(());
    }
    Ok((phase, Some(literal)))
}

/// Whether a number literal may end here.
fn number_can_end(phase: NumberPhase, bounded: Option<&BoundedLiteral>) -> bool {
    phase.can_end() && bounded.is_none_or(BoundedLiteral::contains)
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
        items: Arc::new(items),
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
                compiled.push((json_spelling(name), compile(schema)?));
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
                let spelled = json_spelling(name);
                let index = properties
                    .iter()
                    .position(|(property, _)| *property == spelled)
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
    let open = map.get("properties").is_none()
        && map.get("additionalProperties").is_none()
        && map.get("required").is_none();
    Ok(SchemaNode::Object {
        properties: properties.into(),
        required: required.into(),
        open,
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
    properties: Arc<[(String, SchemaNode)]>,
    required: Arc<[usize]>,
    open: bool,
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
        self.open || !self.unseen_indices().is_empty()
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
    items: Arc<SchemaNode>,
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
    /// An open object key (any valid JSON string).
    AnyKey {
        /// Where the key string is inside an escape sequence.
        escape: StrEscape,
    },
    /// A string value.
    Str {
        /// Where the string is inside an escape sequence.
        escape: StrEscape,
    },
    /// A string enum value, matched against the allowed options.
    StrEnum {
        /// Allowed string values.
        options: Arc<[String]>,
        /// How many bytes matched so far.
        matched: usize,
        /// Indices into `options` whose value still has that prefix.
        candidates: Vec<usize>,
    },
    /// A number literal.
    Number {
        /// Whether the schema said `integer` or `number`.
        kind: NumberKind,
        /// Where the literal is in the JSON number grammar.
        phase: NumberPhase,
        /// The literal so far, when the schema bounds it.
        bounded: Option<Box<BoundedLiteral>>,
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
            if let ValueState::Number { phase, bounded, .. } = value
                && number_can_end(*phase, bounded.as_deref())
            {
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
            Some(ValueState::Number { phase, bounded, .. })
                if number_can_end(*phase, bounded.as_deref()) =>
            {
                true
            }
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

    /// Whether the automaton is currently inside an unescaped string or open object key.
    #[must_use]
    pub fn in_unescaped_string(&self) -> bool {
        matches!(
            self.value,
            Some(
                ValueState::Str {
                    escape: StrEscape::None
                } | ValueState::AnyKey {
                    escape: StrEscape::None
                }
            )
        )
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
                    } else if byte == b'"' && (object.open || !object.properties.is_empty()) {
                        Some(Step::BeginKey)
                    } else {
                        None
                    }
                }
                ObjectPhase::Key => (byte == b'"' && object.has_unseen()).then_some(Step::BeginKey),
                ObjectPhase::Colon => (byte == b':').then_some(Step::Colon),
                ObjectPhase::Value => match object.key_index {
                    Some(index) => Some(Step::ObjectValue(index)),
                    None if object.open => Some(Step::OpenObjectValue),
                    None => None,
                },
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
                if object.open && object.properties.is_empty() {
                    self.value = Some(ValueState::AnyKey {
                        escape: StrEscape::None,
                    });
                    self.position += 1;
                    return Ok(());
                }
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
            Step::OpenObjectValue => self.begin_value(SchemaNode::Any, byte),
            Step::ArrayValue => {
                let Some(Frame::Array(array)) = self.stack.last() else {
                    return Err(());
                };
                let node = SchemaNode::clone(&array.items);
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
            SchemaNode::Any => match byte {
                b'{' => {
                    return self.begin_value(
                        SchemaNode::Object {
                            properties: Arc::new([]),
                            required: Arc::new([]),
                            open: true,
                        },
                        byte,
                    );
                }
                b'[' => {
                    return self.begin_value(
                        SchemaNode::Array {
                            items: Arc::new(SchemaNode::Any),
                            min_items: 0,
                            max_items: None,
                        },
                        byte,
                    );
                }
                b'"' => return self.begin_value(SchemaNode::String, byte),
                b't' | b'f' => return self.begin_value(SchemaNode::Bool, byte),
                b'n' => return self.begin_value(SchemaNode::Null, byte),
                b'-' | b'0'..=b'9' => return self.begin_value(SchemaNode::Number, byte),
                _ => return Err(()),
            },
            SchemaNode::Object {
                properties,
                required,
                open,
            } => {
                if byte != b'{' {
                    return Err(());
                }
                let seen = vec![false; properties.len()];
                self.stack.push(Frame::Object(ObjectFrame {
                    properties,
                    required,
                    open,
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
                    items,
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
            SchemaNode::StringEnum(options) => {
                if byte != b'"' || options.is_empty() {
                    return Err(());
                }
                let candidates = (0..options.len()).collect();
                self.value = Some(ValueState::StrEnum {
                    options,
                    matched: 0,
                    candidates,
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
            SchemaNode::Integer => self.begin_number(NumberKind::Integer, None, byte)?,
            SchemaNode::BoundedInteger(bounds) => {
                self.begin_number(NumberKind::Integer, Some(&bounds), byte)?
            }
            SchemaNode::Number => self.begin_number(NumberKind::Number, None, byte)?,
            SchemaNode::BoundedNumber(bounds) => {
                self.begin_number(NumberKind::Number, Some(&bounds), byte)?
            }
        }
        self.position += 1;
        Ok(())
    }

    fn begin_number(
        &mut self,
        kind: NumberKind,
        bounds: Option<&Arc<NumberBounds>>,
        byte: u8,
    ) -> Result<(), ()> {
        let (phase, bounded) = start_number(kind, bounds, byte)?;
        self.value = Some(ValueState::Number {
            kind,
            phase,
            bounded,
        });
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
                    Some(Frame::Object(object)) => &object.properties[..],
                    _ => return Err(()),
                };
                // A `"` that a name spells (escaped, after a backslash) continues it; otherwise it
                // closes the key. The names share the matched prefix, so at most one of the two
                // readings applies.
                let next: Vec<usize> = candidates
                    .iter()
                    .copied()
                    .filter(|index| names[*index].0.as_bytes().get(matched) == Some(&byte))
                    .collect();
                if !next.is_empty() {
                    return Ok(ValueOutcome::Continue(ValueState::Key {
                        matched: matched + 1,
                        candidates: next,
                    }));
                }
                if byte != b'"' {
                    return Err(());
                }
                let index = candidates
                    .iter()
                    .copied()
                    .find(|index| names[*index].0.len() == matched)
                    .ok_or(())?;
                if let Some(Frame::Object(object)) = self.stack.last_mut() {
                    object.phase = ObjectPhase::Colon;
                    object.key_index = Some(index);
                }
                Ok(ValueOutcome::KeyClosed)
            }
            ValueState::AnyKey { escape } => {
                if escape == StrEscape::None && byte == b'"' {
                    if let Some(Frame::Object(object)) = self.stack.last_mut() {
                        object.phase = ObjectPhase::Colon;
                        object.key_index = None;
                    }
                    return Ok(ValueOutcome::KeyClosed);
                }
                let escape = advance_string(escape, byte)?;
                Ok(ValueOutcome::Continue(ValueState::AnyKey { escape }))
            }
            ValueState::Str { escape } => {
                if escape == StrEscape::None && byte == b'"' {
                    return Ok(ValueOutcome::ConsumedLast);
                }
                let escape = advance_string(escape, byte)?;
                Ok(ValueOutcome::Continue(ValueState::Str { escape }))
            }
            ValueState::StrEnum {
                options,
                matched,
                candidates,
            } => {
                // As for a key: an escaped `"` continues an option, an unescaped one closes it.
                let next: Vec<usize> = candidates
                    .iter()
                    .copied()
                    .filter(|&index| options[index].as_bytes().get(matched) == Some(&byte))
                    .collect();
                if !next.is_empty() {
                    return Ok(ValueOutcome::Continue(ValueState::StrEnum {
                        options,
                        matched: matched + 1,
                        candidates: next,
                    }));
                }
                if byte == b'"'
                    && candidates
                        .iter()
                        .any(|&index| options[index].len() == matched)
                {
                    return Ok(ValueOutcome::ConsumedLast);
                }
                Err(())
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
            ValueState::Number {
                kind,
                phase,
                bounded,
            } => match step_number(kind, phase, bounded.as_deref(), byte) {
                Ok((phase, bounded)) => Ok(ValueOutcome::Continue(ValueState::Number {
                    kind,
                    phase,
                    bounded,
                })),
                Err(()) if number_can_end(phase, bounded.as_deref()) => Ok(ValueOutcome::Ended),
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
            ValueState::AnyKey { escape } | ValueState::Str { escape } => string_mask(*escape),
            ValueState::StrEnum {
                options,
                matched,
                candidates,
            } => {
                let mut mask = ByteMask::EMPTY;
                for index in candidates {
                    let value = options[*index].as_bytes();
                    if let Some(byte) = value.get(*matched) {
                        mask.insert(*byte);
                    }
                    if value.len() == *matched {
                        mask.insert(b'"');
                    }
                }
                mask
            }
            ValueState::Word { word, matched } => {
                let mut mask = ByteMask::EMPTY;
                if let Some(byte) = word.get(*matched) {
                    mask.insert(*byte);
                }
                mask
            }
            ValueState::Number {
                kind,
                phase,
                bounded,
            } => {
                let mask = number_mask(*kind, *phase);
                match bounded {
                    None => mask,
                    Some(literal) => {
                        ByteMask::from_bytes(mask.iter().filter(|&byte| {
                            step_number(*kind, *phase, Some(literal), byte).is_ok()
                        }))
                    }
                }
            }
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
                    if object.open || !object.properties.is_empty() {
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
                    } else if object.open {
                        start_mask(&SchemaNode::Any, &mut mask);
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
    OpenObjectValue,
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
        SchemaNode::Any => {
            for byte in *b"{[\"tfn-" {
                mask.insert(byte);
            }
            for digit in b'0'..=b'9' {
                mask.insert(digit);
            }
        }
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
        SchemaNode::BoundedInteger(bounds) | SchemaNode::BoundedNumber(bounds) => {
            let kind = match node {
                SchemaNode::BoundedInteger(_) => NumberKind::Integer,
                _ => NumberKind::Number,
            };
            for byte in std::iter::once(b'-').chain(b'0'..=b'9') {
                if start_number(kind, Some(bounds), byte).is_ok() {
                    mask.insert(byte);
                }
            }
        }
        SchemaNode::String | SchemaNode::StringEnum(_) => mask.insert(b'"'),
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
            serde_json::json!({"type": "integer", "exclusiveMinimum": 0}),
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

    #[test]
    fn enum_values_and_property_names_match_their_escaped_spelling() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "say \"hi\"": {
                    "type": "string",
                    "enum": ["a \"b\"", "back\\slash", "line\nbreak", "a"],
                },
            },
            "required": ["say \"hi\""],
            "additionalProperties": false,
        });
        for (document, value) in [
            (r#"{"say \"hi\"":"a \"b\""}"#, "a \"b\""),
            (r#"{"say \"hi\"":"back\\slash"}"#, "back\\slash"),
            (r#"{"say \"hi\"":"line\nbreak"}"#, "line\nbreak"),
            (r#"{"say \"hi\"":"a"}"#, "a"),
        ] {
            assert_mask_matches_feed(&schema, document);
            let parsed: Value = serde_json::from_str(document).expect("the document is JSON");
            assert_eq!(parsed["say \"hi\""], value, "{document}");
        }
        // The raw spellings would end the name early or put a control byte in the string.
        let mut raw_key = SchemaConstraint::new(&schema).expect("compiles");
        assert!(raw_key.feed_str(b"{\"say \"").is_err());
        let mut raw_value = SchemaConstraint::new(&schema).expect("compiles");
        raw_value
            .feed_str(br#"{"say \"hi\"":"line"#)
            .expect("the shared prefix");
        assert!(raw_value.feed(b'\n').is_err());
    }

    /// Every literal of up to `len` bytes over the number alphabet is accepted exactly when it is a
    /// JSON number of the schema's kind within `[min, max]` (a negative literal also needs
    /// `min < 0`), and no prefix the automaton accepts is left without an admissible byte.
    fn assert_bounds_exact(integer: bool, min: Option<f64>, max: Option<f64>, len: usize) {
        let mut schema = serde_json::json!({"type": if integer { "integer" } else { "number" }});
        if let Some(min) = min {
            schema["minimum"] = serde_json::json!(min);
        }
        if let Some(max) = max {
            schema["maximum"] = serde_json::json!(max);
        }
        let root = SchemaConstraint::new(&schema).expect("the bounds compile");
        let alphabet = b"-0123456789.";
        // Refused prefixes are kept (as None) so that a valid literal behind a wrongly refused
        // prefix is still enumerated and caught.
        let mut frontier = vec![(Vec::new(), Some(root))];
        for _ in 0..len {
            let mut next = Vec::new();
            for (text, constraint) in &frontier {
                let mask = constraint.as_ref().map(SchemaConstraint::allowed_bytes);
                if let Some(mask) = &mask {
                    assert!(
                        !mask.is_empty(),
                        "{schema}: no byte is admissible after {:?}",
                        String::from_utf8_lossy(text)
                    );
                }
                for &byte in alphabet {
                    let mut probe = constraint.clone();
                    let fed = probe.as_mut().is_some_and(|probe| probe.feed(byte).is_ok());
                    if let Some(mask) = &mask {
                        assert_eq!(mask.contains(byte), fed, "{schema}: mask and feed disagree");
                    }
                    let mut longer = text.clone();
                    longer.push(byte);
                    let literal = String::from_utf8(longer.clone()).expect("ASCII");
                    let ends = fed
                        && probe
                            .clone()
                            .is_some_and(|mut probe| probe.finish_document());
                    let expected = serde_json::from_str::<serde_json::Number>(&literal)
                        .ok()
                        .filter(|_| !integer || !literal.contains('.'))
                        .and_then(|number| number.as_f64())
                        .is_some_and(|value| {
                            min.is_none_or(|min| value >= min)
                                && max.is_none_or(|max| value <= max)
                                && (!literal.starts_with('-') || min.is_none_or(|min| min < 0.0))
                        });
                    assert_eq!(ends, expected, "{schema}: literal {literal}");
                    next.push((longer, if fed { probe } else { None }));
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn numeric_bounds_are_enforced_exactly() {
        for integer in [true, false] {
            for (min, max) in [
                (Some(1.0), Some(5.0)),
                (Some(-10.0), Some(10.0)),
                (Some(-7.5), Some(-0.25)),
                (Some(0.3), Some(0.31)),
                (Some(100.0), Some(250.0)),
                (Some(0.0), None),
                (None, Some(42.0)),
                (Some(9.0), Some(9.0)),
                (Some(-0.001), Some(0.001)),
                (Some(0.5), Some(2.5)),
            ] {
                if integer && min == Some(-7.5) {
                    continue;
                }
                if integer && min == Some(0.3) {
                    continue;
                }
                assert_bounds_exact(integer, min, max, 5);
            }
        }
    }

    #[test]
    fn bounds_compile_exactly_or_are_refused() {
        // A bound written with an exponent is read exactly.
        let mut constraint = SchemaConstraint::new(
            &serde_json::json!({"type": "integer", "minimum": 1e2, "maximum": 1.5e2}),
        )
        .expect("compiles");
        constraint.feed_str(b"150").expect("the maximum itself");
        assert!(constraint.finish_document());
        let constraint = SchemaConstraint::new(
            &serde_json::json!({"type": "integer", "minimum": 1e2, "maximum": 1.5e2}),
        )
        .expect("compiles");
        assert!(!constraint.admits(b"151"));
        assert!(!constraint.admits(b"99"));
        // Bounds no integer lies between, or that cross, are a malformed schema.
        for schema in [
            serde_json::json!({"type": "integer", "minimum": 0.2, "maximum": 0.8}),
            serde_json::json!({"type": "number", "minimum": 2, "maximum": 1}),
        ] {
            let error = SchemaConstraint::new(&schema).expect_err("no value is in bounds");
            assert!(matches!(error, SchemaError::Invalid { .. }), "{error:?}");
        }
        // Inside an object the bound decides the digit, and the mask agrees with feed.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["n"],
        });
        assert_mask_matches_feed(&schema, r#"{"n":5}"#);
        let mut constraint = SchemaConstraint::new(&schema).expect("compiles");
        constraint.feed_str(br#"{"n":"#).expect("the key");
        assert!(!constraint.admits(b"0"));
        assert!(!constraint.admits(b"50"));
        assert!(!constraint.admits(b"-1"));
        assert!(constraint.admits(b"3}"));
    }
}

/// A schema constraint over vocabulary ids: the same automaton, driven by tokens.
///
/// [SchemaConstraint] speaks bytes and a sampler speaks ids, so this is the layer a logits
/// processor uses: it turns one into the other through [TokenBytes] and keeps the
/// end-of-sequence id admissible only where the document may end. A token is admitted when every
/// byte it appends keeps a valid completion reachable, which is the guarantee the automaton
/// already gives.
#[derive(Clone)]
pub struct TokenConstraint {
    schema: SchemaConstraint,
    vocab: Arc<TokenBytes>,
    end: Option<u32>,
}

impl TokenConstraint {
    /// Starts the document the schema describes.
    ///
    /// # Errors
    ///
    /// The same compile errors as [SchemaConstraint::new].
    pub fn new(
        schema: &Value,
        vocab: Arc<TokenBytes>,
        end: Option<u32>,
    ) -> Result<Self, SchemaError> {
        Ok(Self {
            schema: SchemaConstraint::new(schema)?,
            vocab,
            end,
        })
    }

    /// Whether the sampler may pick this id next.
    #[must_use]
    pub fn admits(&self, id: u32) -> bool {
        if Some(id) == self.end {
            return self.schema.allows_end();
        }
        match self.vocab.bytes(id) {
            Some(bytes) => self.schema.admits(bytes),
            None => false,
        }
    }

    /// Consumes the token the sampler picked.
    ///
    /// # Errors
    ///
    /// [ConstraintError] when the id appends bytes the schema forbids. The end id is not an error:
    /// it appends nothing, and the caller stops on it.
    pub fn advance(&mut self, id: u32) -> Result<(), ConstraintError> {
        if Some(id) == self.end {
            return Ok(());
        }
        let bytes = self.vocab.bytes(id);
        match bytes {
            Some(bytes) => self.schema.feed_str(bytes),
            None => Err(ConstraintError {
                position: self.schema.position(),
                byte: 0,
                expected: ByteMask::from_bytes(std::iter::empty::<u8>()),
            }),
        }
    }

    /// Calls `f` for every vocabulary id in `id_base..id_end` that the sampler may pick in this
    /// state.
    pub fn for_each_allowed_in(&self, id_base: u32, id_end: u32, mut f: impl FnMut(u32)) {
        let mask = self.schema.allowed_bytes();
        let in_plain_str = self.schema.in_unescaped_string();
        for byte in mask.iter() {
            let bucket = self.vocab.starting_with(byte);
            let start = bucket.partition_point(|&id| id < id_base);
            for &id in &bucket[start..] {
                if id >= id_end {
                    break;
                }
                if in_plain_str && self.vocab.is_plain_string(id) {
                    f(id);
                    continue;
                }
                let Some(bytes) = self.vocab.bytes(id) else {
                    continue;
                };
                if self.schema.admits(bytes) {
                    f(id);
                }
            }
        }
        if self.schema.allows_end()
            && let Some(end) = self.end
            && end >= id_base
            && end < id_end
        {
            f(end);
        }
    }

    /// Every vocabulary id the sampler may pick in this state.
    ///
    /// The first-byte index keeps this affordable: a JSON syntax state allows a handful of bytes,
    /// so only ids that start with one of them are tested against the automaton.
    #[must_use]
    pub fn allowed_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        self.for_each_allowed_in(0, u32::MAX, |id| ids.push(id));
        ids
    }

    /// Whether the document has ended and satisfies the schema.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.schema.is_complete()
    }

    /// Whether the document may end here, i.e. whether the end id is admissible.
    #[must_use]
    pub fn allows_end(&self) -> bool {
        self.schema.allows_end()
    }

    /// Bytes accepted so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.schema.position()
    }
}

#[cfg(test)]
mod token_constraint_tests {
    use super::*;
    use crate::vocab::TokenBytes;
    use serde_json::json;
    use std::sync::Arc;

    /// A small vocabulary that can spell an object one token per byte.
    fn vocabulary() -> Arc<TokenBytes> {
        let spellings = ["{", "}", "\"", "a", ":", "x", " "];
        Arc::new(TokenBytes::from_pairs(
            spellings
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, (*text).to_owned())),
        ))
    }

    /// The end-of-sequence id: outside [vocabulary], which occupies 0..7.
    const END: u32 = 7;

    fn object_schema() -> Value {
        json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": ["a"],
            "additionalProperties": false
        })
    }

    fn id_of(vocab: &TokenBytes, spelling: &str) -> u32 {
        (0..vocab.len() as u32)
            .find(|&id| vocab.bytes(id) == Some(spelling.as_bytes()))
            .expect("the vocabulary spells this token")
    }

    #[test]
    fn the_opening_brace_is_the_only_first_token() {
        let vocab = vocabulary();
        let constraint = TokenConstraint::new(&object_schema(), Arc::clone(&vocab), Some(END))
            .expect("the schema compiles");
        // A leading space is legal JSON, so it is admissible beside the brace; nothing else is.
        assert_eq!(
            constraint.allowed_ids(),
            vec![id_of(&vocab, " "), id_of(&vocab, "{")]
        );
        assert!(
            !constraint.allowed_ids().contains(&END),
            "the end id is not admissible yet"
        );
    }

    #[test]
    fn driving_the_automaton_with_tokens_spells_valid_json() {
        let vocab = vocabulary();
        let mut constraint = TokenConstraint::new(&object_schema(), Arc::clone(&vocab), Some(END))
            .expect("the schema compiles");
        for spelling in ["{", "\"", "a", "\"", ":", "\"", "x", "\"", "}"] {
            let id = id_of(&vocab, spelling);
            assert!(constraint.admits(id), "{spelling} must be admissible");
            constraint
                .advance(id)
                .expect("the token extends the document");
        }
        assert!(constraint.is_complete());
        assert!(constraint.allows_end());
        assert!(
            constraint.admits(END),
            "the end id is admissible once complete"
        );
    }

    #[test]
    fn a_token_that_would_break_the_json_is_not_admitted() {
        let vocab = vocabulary();
        let mut constraint = TokenConstraint::new(&object_schema(), Arc::clone(&vocab), Some(END))
            .expect("the schema compiles");
        // A closing brace before the object opens cannot extend any valid document.
        assert!(!constraint.admits(id_of(&vocab, "}")));
        constraint
            .advance(id_of(&vocab, "{"))
            .expect("the object opens");
        assert!(!constraint.admits(id_of(&vocab, "}")));
        // Whitespace between members is legal JSON, so it stays admissible.
        assert!(constraint.allowed_ids().contains(&id_of(&vocab, " ")));
    }

    #[test]
    fn the_end_id_is_refused_before_the_document_can_end() {
        let vocab = vocabulary();
        let mut constraint = TokenConstraint::new(&object_schema(), Arc::clone(&vocab), Some(END))
            .expect("the schema compiles");
        assert!(!constraint.admits(END));
        constraint
            .advance(END)
            .expect("the end id consumes no bytes");
        assert!(!constraint.is_complete());
    }

    #[test]
    fn multi_byte_tokens_crossing_states_are_in_allowed_ids() {
        let spellings = ["{\"", "a\":\"", "x\"}", "bad"];
        let vocab = Arc::new(TokenBytes::from_pairs(
            spellings
                .iter()
                .enumerate()
                .map(|(id, text)| (id as u32, (*text).to_owned())),
        ));
        let mut constraint = TokenConstraint::new(&object_schema(), Arc::clone(&vocab), Some(END))
            .expect("the schema compiles");
        assert_eq!(constraint.allowed_ids(), vec![0]);
        constraint.advance(0).expect("opens object and key quote");
        assert_eq!(constraint.allowed_ids(), vec![1]);
        constraint.advance(1).expect("key, colon, and value quote");
        assert_eq!(constraint.allowed_ids(), vec![3, 2, 0]);
        constraint
            .advance(2)
            .expect("value, closing quote, and closing brace");
        assert!(constraint.is_complete());
        assert_eq!(constraint.allowed_ids(), vec![END]);
    }

    #[test]
    fn string_enum_and_open_object_schemas_enforce_constraints() {
        let schema = json!({
            "type": "object",
            "properties": {
                "signal": {
                    "type": "string",
                    "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"]
                },
                "rating": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10
                }
            },
            "required": ["signal", "rating"],
            "additionalProperties": false
        });
        let mut c = SchemaConstraint::new(&schema).expect("compiles");
        c.feed_str(b"{\"signal\":\"buy\",\"rating\":9.5}")
            .expect("valid enum and number");
        assert!(c.is_complete());

        let mut bad_enum = SchemaConstraint::new(&schema).expect("compiles");
        assert!(bad_enum.feed_str(b"{\"signal\":\"hold_on\"").is_err());

        let mut bad_neg = SchemaConstraint::new(&schema).expect("compiles");
        assert!(
            bad_neg
                .feed_str(b"{\"signal\":\"buy\",\"rating\":-1")
                .is_err()
        );

        let open = json!({"type": "object"});
        let mut open_c = SchemaConstraint::new(&open).expect("open object compiles");
        open_c
            .feed_str(b"{\"any_key\":[1,true,null,{\"nested\":\"ok\"}]}")
            .expect("open object accepts arbitrary JSON properties");
        assert!(open_c.is_complete());
    }
}
