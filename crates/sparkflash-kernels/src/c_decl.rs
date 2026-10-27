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

//! A small parser for the CUDA source in kernels/.
//!
//! The M4 kernels cannot be executed on this host, so their ABI is pinned by reading their
//! declarations and re-deriving every number in Rust instead of restating a constant. This
//! module is deliberately a *text* parser: it never asks a C++ compiler, because the point is
//! to catch a drift between the kernel source and the Rust types on a machine where the kernel
//! would not run anyway.
//!
//! What it provides:
//!   * splice_lines, the translation-phase-2 backslash-newline splice, applied before anything is
//!     stripped or parsed so a split comment opener or a split identifier is never seen split;
//!   * strip_comments on the spliced text, so a commented-out copy of a declaration is never
//!     mistaken for the real one and a raw string's contents are never scanned for comments;
//!   * #define collection with continuation lines, and an integer expression evaluator that
//!     resolves identifiers through the table (so SPARK_M4_HC_PREDICTOR_ROWS is re-derived from
//!     SPARK_M4_HC_MULT rather than trusted);
//!   * static_assert extraction, so the tests can evaluate the kernel's own assertions in Rust;
//!   * __global__ declaration extraction with per-parameter type, name and qualifiers;
//!   * depth-aware comma splitting, which is what the payload[4][1024 * 1024] case in the M3
//!     kernel needed.

use std::collections::BTreeMap;

/// Applies translation phase 2 (line splicing) to `text`.
///
/// Every backslash-newline pair is deleted, in both the `\n` and `\r\n` spellings, so the two
/// physical lines become one logical line before any comment or token rule runs. This is the
/// phase-2 behaviour the parsers here need: `ret\<newline>urn` is the single keyword `return`,
/// and `/\<newline>/` opens a `//` comment.
///
/// Whitespace between the backslash and the newline is spliced too, and the whitespace is deleted
/// with the pair. That is not the letter of the C++17 phase-2 rule, but it is what every frontend
/// this milestone builds with does: clang 22.1.8, gcc 16.2.1 and nvcc 12.9.1 all accept
/// `ret\   \nurn` and join it to the single token `return` (clang and gcc warn
/// `-Wbackslash-newline-escape` and still exit 0; nvcc emits the joined identifier as well).
/// Because build.rs compiles with plain `-std=c++17` and no `-Werror`, a source the compiler
/// accepts is a source the guard has to read the same way, or an escape the compiler splices stays
/// invisible here. Two gaps of this kind were found this way: the round-4 A1 shape
/// (`ret\   \nurn`, tokens `ret | backslash | urn`), and the round-5 E1 shape, where the
/// whitespace is a vertical tab or a form feed - `ret\<0x0B>\nurn` and `ret\<0x0C>\nurn` splice
/// to `return` in all three frontends, and `clang -E -P` emits `return 1;` for
/// `ret\<0x0C>\nurn 1;`. Skipping only space and tab left that escape invisible.
///
/// Trigraphs are deliberately not translated: `??/` stopped being a backslash in C++17, which is
/// the language level this milestone builds with (`-std=c++17`), so a `??/` before a newline is
/// left as ordinary characters here too.
#[must_use]
pub fn splice_lines(text: &str) -> String {
    splice_lines_with_map(text).0
}

/// [`splice_lines`] plus, for every byte of its result, the byte offset in `text` it was copied from.
///
/// Splicing only deletes bytes, so the surviving bytes keep their order and the map is strictly
/// increasing. [`strip_comments_with_lines`] needs it because a parser that counts newlines in the
/// spliced text counts the newlines the splice removed, which is how the acquire guard came to report
/// the shipped helper at m4_common_sm121.cuh:484 while it sits on line 507.
#[must_use]
pub fn splice_lines_with_map(text: &str) -> (String, Vec<usize>) {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut map: Vec<usize> = Vec::with_capacity(text.len());
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            // Skip the whitespace the frontends accept between the backslash and the newline:
            // space, tab, vertical tab and form feed. The pair (with the whitespace) is deleted
            // whole, which is why `ret` and `urn` join with nothing between them and form the
            // single keyword. Vertical tab and form feed are here because clang, gcc and nvcc all
            // splice them (see the doc comment); leaving them out made their splices invisible.
            let mut newline = index + 1;
            while newline < bytes.len()
                && matches!(bytes[newline], b' ' | b'\t' | b'\x0b' | b'\x0c')
            {
                newline += 1;
            }
            let after = if bytes.get(newline) == Some(&b'\n') {
                newline + 1
            } else if bytes.get(newline) == Some(&b'\r') && bytes.get(newline + 1) == Some(&b'\n') {
                newline + 2
            } else {
                0
            };
            if after != 0 {
                out.push_str(&text[start..index]);
                map.extend(start..index);
                index = after;
                start = index;
                continue;
            }
        }
        index += 1;
    }
    out.push_str(&text[start..]);
    map.extend(start..text.len());
    (out, map)
}

/// The byte offset just past the raw string literal that starts at `start`, where `text[start..]`
/// begins with `R"`.
///
/// The delimiter between `R"` and its opening `(` is at most sixteen characters and excludes
/// space, `(`, `)` and backslash; the literal ends at the first `)` followed by that delimiter
/// and a `"`. The caller uses this to copy or to skip the literal without ever reading its
/// contents as comments or tokens.
///
/// # Panics
/// Panics when the opener has no `(` or the literal is unterminated: a raw string that does not
/// close is not valid C++ and must fail loudly rather than be read as code.
#[must_use]
pub fn raw_string_end(text: &str, start: usize) -> usize {
    let bytes = text.as_bytes();
    assert!(
        bytes.get(start) == Some(&b'R') && bytes.get(start + 1) == Some(&b'"'),
        "a raw string literal must start with R\""
    );
    let open = start + 2;
    let paren = text[open..]
        .find('(')
        .map(|offset| open + offset)
        .unwrap_or_else(|| panic!("raw string literal with no opening parenthesis"));
    let delimiter = &text[open..paren];
    let terminator = format!("){delimiter}\"");
    text[paren + 1..]
        .find(&terminator)
        .map(|offset| paren + 1 + offset + terminator.len())
        .unwrap_or_else(|| panic!("unterminated raw string literal"))
}

/// Removes // and /* */ comments after applying the translation-phase-2 line splice.
///
/// The splice is applied FIRST, exactly as C++ does: the text handed to the comment scanner is
/// already joined, so a comment opener split across two physical lines (`/\<newline>/`) really
/// opens a comment, a split identifier (`ret\<newline>urn`) is the keyword, and none of the
/// spliced text can reach a later parser in its split form. That ordering is the fix for the
/// round-3 escapes that planted a fence, a control keyword or a decoy `static_assert` behind a
/// split comment opener or a split identifier.
///
/// Character and string literals are tracked so a quote or an asterisk inside one cannot open a
/// comment, and C++ raw string literals (`R"delim(...)delim"`) are copied verbatim so their
/// contents are never scanned for comments either. The surviving newlines are preserved, so line
/// structure stays comparable except across a splice, where the two physical lines are one logical
/// line by construction. Every entry point in this module strips comments itself rather than
/// trusting its caller: a commented-out copy of a declaration, a #define or a static_assert placed
/// above the real one is the cheapest way to make a text parser agree with a reviewer who reads the
/// file, and the earlier version of this module walked into exactly that.
#[must_use]
pub fn strip_comments(text: &str) -> String {
    strip_comments_with_lines(text).0
}

/// Appends the byte `spliced[spliced_index]` to a comment-stripped copy, recording the PHYSICAL line
/// of the byte of the input it came from. A byte >= 0x80 becomes a two-byte UTF-8 character, and both
/// of its bytes carry the same line, so the map stays one entry per output byte.
fn push_mapped(
    out: &mut String,
    lines: &mut Vec<u32>,
    spliced_index: usize,
    byte: u8,
    line_of: &[u32],
    spliced_map: &[usize],
) {
    let physical = line_of[spliced_map[spliced_index]];
    out.push(byte as char);
    lines.push(physical);
    if (byte as char).len_utf8() == 2 {
        lines.push(physical);
    }
}

/// [`strip_comments`] plus, for every byte of the result, the PHYSICAL line of `text` it came from.
///
/// Two stages delete bytes - the phase-2 splice removes the backslash-newline pairs and the comment
/// scanner removes the comment bytes - so a line counted in the result is not a line in the file. The
/// acquire guard counted lines in the result and named the wrong ones: with 23 splices before the
/// helper in the shared header it reported the shipped helper at m4_common_sm121.cuh:484 while it
/// sits on line 507, and a diagnostic that sends a reader to the wrong line is worse than none. The
/// map is per byte rather than per line because a token can start on a continuation line, where its
/// own physical line is the one that matters.
///
/// # Panics
/// Panics when a raw string literal is unterminated, like [`strip_comments`].
#[must_use]
pub fn strip_comments_with_lines(text: &str) -> (String, Vec<u32>) {
    let (spliced, spliced_map) = splice_lines_with_map(text);
    // The physical line of every byte of the input, plus one past the end.
    let mut line_of = vec![0u32; text.len() + 1];
    let mut physical = 1u32;
    for (offset, byte) in text.bytes().enumerate() {
        line_of[offset] = physical;
        if byte == b'\n' {
            physical += 1;
        }
    }
    line_of[text.len()] = physical;
    let bytes = spliced.as_bytes();
    let mut out = String::with_capacity(spliced.len());
    let mut lines: Vec<u32> = Vec::with_capacity(spliced.len());
    let mut index = 0usize;
    let mut in_line = false;
    let mut in_block = false;
    let mut in_string = false;
    let mut in_char = false;
    while index < bytes.len() {
        let current = bytes[index];
        let next = if index + 1 < bytes.len() {
            bytes[index + 1]
        } else {
            0
        };
        if in_line {
            if current == b'\n' {
                in_line = false;
                push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
            }
        } else if in_block {
            if current == b'*' && next == b'/' {
                in_block = false;
                index += 1;
            } else if current == b'\n' {
                push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
            }
        } else if in_string || in_char {
            let terminator = if in_string { b'"' } else { b'\'' };
            if current == b'\\' {
                push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
                if index + 1 < bytes.len() && bytes[index + 1] != b'\n' {
                    push_mapped(
                        &mut out,
                        &mut lines,
                        index + 1,
                        bytes[index + 1],
                        &line_of,
                        &spliced_map,
                    );
                    index += 1;
                }
            } else {
                if current == terminator {
                    in_string = false;
                    in_char = false;
                }
                push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
            }
        } else if current == b'R' && next == b'"' {
            // A raw string is copied verbatim, so a // or /* inside it is content rather than a
            // comment opener, and a fence keyword inside it never reaches a token scanner.
            let end = raw_string_end(&spliced, index);
            out.push_str(&spliced[index..end]);
            for offset in index..end {
                lines.push(line_of[spliced_map[offset]]);
            }
            index = end;
            continue;
        } else if current == b'"' {
            in_string = true;
            push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
        } else if current == b'\'' {
            in_char = true;
            push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
        } else if current == b'/' && next == b'/' {
            in_line = true;
            index += 1;
        } else if current == b'/' && next == b'*' {
            in_block = true;
            index += 1;
        } else {
            push_mapped(&mut out, &mut lines, index, current, &line_of, &spliced_map);
        }
        index += 1;
    }
    (out, lines)
}

/// The object-like #define table: the constants an integer evaluator can resolve.
///
/// Function-like macros are deliberately absent because they are not values: they expand only when
/// followed by an argument list, and the constants the kernels use are all object-like. Macros with
/// an empty body (the ones whose value continues on the next line, like
/// SPARK_M4_STATIC_ASSERTS_FILE) are skipped here too; the tests that care about them parse the
/// body instead. The wider view the acquire guard needs is [`macro_bodies`], which is not
/// restricted to the object-like spelling.
///
/// Comments are stripped here, so a #define that has been commented out - on a line of its own or
/// inside a block comment - is not a definition. Passing already-stripped text is harmless.
#[must_use]
pub fn defines(text: &str) -> BTreeMap<String, String> {
    define_table(text, false)
}

/// Every #define of `text` whose body is on its first physical line, function-like macros included.
///
/// The preprocessor does not distinguish an object-like from a function-like macro at the point of
/// definition: both are replaced when used. A reachability analysis that wants to know whether a
/// macro body is a control keyword or a non-returning call therefore has to read both, or the
/// function-like half is invisible and `#define BAIL(x) return x` becomes an escape. The parameter
/// list of a function-like macro is stripped, so the body returned is the replacement text the
/// call would expand to (`return x` for `BAIL(x)`); a multi-line body (one that starts with a
/// backslash continuation, such as the trace macros) is still skipped because only the first
/// physical line is read. Comments are stripped here, so a commented-out definition is not one.
#[must_use]
pub fn macro_bodies(text: &str) -> BTreeMap<String, String> {
    define_table(text, true)
}

/// The shared #define walk. When `function_like` is false the function-like macros are dropped, so
/// the result is the constant table [`defines`] promises.
fn define_table(text: &str, function_like: bool) -> BTreeMap<String, String> {
    let text = strip_comments(text);
    let mut table = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("#define") else {
            continue;
        };
        let rest = rest.trim_start();
        let name_len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        if name_len == 0 {
            continue;
        }
        let name = &rest[..name_len];
        let remainder = &rest[name_len..];
        let value = if remainder.starts_with('(') {
            // Function-like. The parameter list is the balanced parenthesised prefix; the body is
            // what follows it. `matching_close` understands string literals, and this module
            // already depends on it for the same purpose elsewhere.
            if !function_like {
                continue;
            }
            let close = matching_close(remainder, 0, '(', ')');
            remainder[close + 1..].trim()
        } else {
            remainder.trim()
        };
        if value.is_empty() || value == "\\" {
            continue; // multi-line body, handled by the body parsers
        }
        table.insert(name.to_string(), value.to_string());
    }
    table
}

/// Integer constant expressions from the kernel sources, evaluated over a #define table.
///
/// The evaluator implements C's precedence for the operators the kernels actually use, strips
/// integer suffixes (u, ul, ull, l, ll), and understands sizeof on the fixed-width types and
/// float. Unknown identifiers are an error rather than a zero, so a renamed constant fails the
/// test instead of silently evaluating to something.
#[derive(Debug, Clone, Default)]
pub struct Consts {
    table: BTreeMap<String, String>,
}

impl Consts {
    /// Builds an evaluator over the #defines of one source text.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        Self {
            table: defines(text),
        }
    }

    /// Extends this table with the #defines of another source, first definition winning.
    pub fn extend(&mut self, text: &str) {
        for (name, value) in defines(text) {
            self.table.entry(name).or_insert(value);
        }
    }

    /// The raw text of a #define, if present.
    #[must_use]
    pub fn raw(&self, name: &str) -> Option<&str> {
        self.table.get(name).map(String::as_str)
    }

    /// Evaluates a constant expression, resolving identifiers through the table.
    ///
    /// # Panics
    /// Panics if the expression is malformed or names something not in the table: a test that
    /// silently accepted either would not be checking anything.
    #[must_use]
    pub fn eval(&self, expression: &str) -> i128 {
        let mut parser = ExprParser {
            chars: expression.chars().collect(),
            position: 0,
            table: &self.table,
            expanding: Vec::new(),
        };
        let value = parser.parse_expression();
        parser.skip_whitespace();
        assert!(
            parser.position == parser.chars.len(),
            "trailing input in constant expression {expression:?} at {}",
            parser.position
        );
        value
    }

    /// Evaluates a #define by name.
    ///
    /// # Panics
    /// Panics if the name is unknown or its body is not an integer expression.
    #[must_use]
    pub fn value(&self, name: &str) -> i128 {
        let body = self
            .table
            .get(name)
            .unwrap_or_else(|| panic!("no #define {name} in the parsed sources"));
        self.eval(body)
    }

    /// Whether a #define exists at all.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.table.contains_key(name)
    }
}

struct ExprParser<'a> {
    chars: Vec<char>,
    position: usize,
    table: &'a BTreeMap<String, String>,
    expanding: Vec<String>,
}

impl ExprParser<'_> {
    fn skip_whitespace(&mut self) {
        while self.position < self.chars.len() && self.chars[self.position].is_whitespace() {
            self.position += 1;
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_whitespace();
        let token: Vec<char> = token.chars().collect();
        if self.position + token.len() > self.chars.len() {
            return false;
        }
        if self.chars[self.position..self.position + token.len()] == token[..] {
            self.position += token.len();
            return true;
        }
        false
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_whitespace();
        self.chars.get(self.position).copied()
    }

    fn parse_expression(&mut self) -> i128 {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> i128 {
        let mut left = self.parse_logical_and();
        while self.eat("||") {
            let right = self.parse_logical_and();
            left = i128::from(left != 0 || right != 0);
        }
        left
    }

    fn parse_logical_and(&mut self) -> i128 {
        let mut left = self.parse_bit_or();
        while self.eat("&&") {
            let right = self.parse_bit_or();
            left = i128::from(left != 0 && right != 0);
        }
        left
    }

    fn parse_bit_or(&mut self) -> i128 {
        let mut left = self.parse_bit_xor();
        while self.eat("|") {
            let right = self.parse_bit_xor();
            left |= right;
        }
        left
    }

    fn parse_bit_xor(&mut self) -> i128 {
        let mut left = self.parse_bit_and();
        while self.eat("^") {
            let right = self.parse_bit_and();
            left ^= right;
        }
        left
    }

    fn parse_bit_and(&mut self) -> i128 {
        let mut left = self.parse_equality();
        while self.peek() == Some('&') && !self.starts_with("&&") {
            self.eat("&");
            let right = self.parse_equality();
            left &= right;
        }
        left
    }

    fn starts_with(&mut self, token: &str) -> bool {
        self.skip_whitespace();
        let token: Vec<char> = token.chars().collect();
        self.position + token.len() <= self.chars.len()
            && self.chars[self.position..self.position + token.len()] == token[..]
    }

    fn parse_equality(&mut self) -> i128 {
        let mut left = self.parse_relational();
        loop {
            if self.eat("==") {
                let right = self.parse_relational();
                left = i128::from(left == right);
            } else if self.eat("!=") {
                let right = self.parse_relational();
                left = i128::from(left != right);
            } else {
                return left;
            }
        }
    }

    fn parse_relational(&mut self) -> i128 {
        let mut left = self.parse_shift();
        loop {
            if self.starts_with("<=") {
                self.eat("<=");
                left = i128::from(left <= self.parse_shift());
            } else if self.starts_with(">=") {
                self.eat(">=");
                left = i128::from(left >= self.parse_shift());
            } else if self.peek() == Some('<') && !self.starts_with("<<") {
                self.eat("<");
                left = i128::from(left < self.parse_shift());
            } else if self.peek() == Some('>') && !self.starts_with(">>") {
                self.eat(">");
                left = i128::from(left > self.parse_shift());
            } else {
                return left;
            }
        }
    }

    fn parse_shift(&mut self) -> i128 {
        let mut left = self.parse_additive();
        loop {
            if self.starts_with("<<") {
                self.eat("<<");
                left <<= self.parse_additive();
            } else if self.starts_with(">>") {
                self.eat(">>");
                left >>= self.parse_additive();
            } else {
                return left;
            }
        }
    }

    fn parse_additive(&mut self) -> i128 {
        let mut left = self.parse_multiplicative();
        loop {
            if self.eat("+") {
                left += self.parse_multiplicative();
            } else if self.eat("-") {
                left -= self.parse_multiplicative();
            } else {
                return left;
            }
        }
    }

    fn parse_multiplicative(&mut self) -> i128 {
        let mut left = self.parse_unary();
        loop {
            if self.eat("*") {
                left *= self.parse_unary();
            } else if self.eat("/") {
                left /= self.parse_unary();
            } else if self.eat("%") {
                left %= self.parse_unary();
            } else {
                return left;
            }
        }
    }

    fn parse_unary(&mut self) -> i128 {
        if self.eat("-") {
            return -self.parse_unary();
        }
        if self.eat("~") {
            return !self.parse_unary();
        }
        if self.eat("!") {
            return i128::from(self.parse_unary() == 0);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> i128 {
        self.skip_whitespace();
        if self.eat("(") {
            let value = self.parse_expression();
            assert!(
                self.eat(")"),
                "unbalanced parenthesis in constant expression"
            );
            return value;
        }
        if self.peek() == Some('\'') {
            // A character literal is an integer in C; the kernels use them for byte constants.
            self.position += 1;
            let value = self.chars[self.position] as i128;
            self.position += 1;
            assert!(self.eat("'"), "unterminated character literal");
            return value;
        }
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            let start = self.position;
            while self.position < self.chars.len()
                && (self.chars[self.position].is_ascii_alphanumeric()
                    || self.chars[self.position] == '.'
                    || self.chars[self.position] == '_')
            {
                self.position += 1;
            }
            let literal: String = self.chars[start..self.position].iter().collect();
            return parse_integer_literal(&literal);
        }
        let start = self.position;
        while self.position < self.chars.len()
            && (self.chars[self.position].is_ascii_alphanumeric()
                || self.chars[self.position] == '_'
                || self.chars[self.position] == ':')
        {
            self.position += 1;
        }
        let mut name: String = self.chars[start..self.position].iter().collect();
        if name.is_empty() {
            panic!("empty primary in constant expression at {}", self.position);
        }
        if name == "sizeof" {
            assert!(self.eat("("), "sizeof without a parenthesised type");
            let type_start = self.position;
            while self.position < self.chars.len() && self.chars[self.position] != ')' {
                self.position += 1;
            }
            let ty: String = self.chars[type_start..self.position].iter().collect();
            assert!(self.eat(")"), "unterminated sizeof");
            return sizeof_type(ty.trim());
        }
        // Casts and qualifiers in a sizeof-free expression: skip over a leading type name.
        if let Some(rest) = name.strip_suffix("u64") {
            name = rest.to_string();
        }
        let body = self
            .table
            .get(&name)
            .unwrap_or_else(|| panic!("unknown identifier {name:?} in constant expression"))
            .clone();
        assert!(
            !self.expanding.contains(&name),
            "recursive #define while evaluating {name}"
        );
        self.expanding.push(name.clone());
        let mut nested = ExprParser {
            chars: body.chars().collect(),
            position: 0,
            table: self.table,
            expanding: self.expanding.clone(),
        };
        let value = nested.parse_expression();
        nested.skip_whitespace();
        assert!(
            nested.position == nested.chars.len(),
            "trailing input in the body of #define {name}"
        );
        self.expanding.pop();
        value
    }
}

/// The integer value of a C literal with an optional integer suffix, or of a float literal whose
/// value happens to be a whole number (the compiler folds those in array bounds).
///
/// # Panics
/// Panics on a literal this evaluator cannot represent, rather than guessing.
#[must_use]
pub fn parse_integer_literal(literal: &str) -> i128 {
    let digits: String = literal
        .chars()
        .take_while(|c| c.is_ascii_hexdigit() || *c == '.')
        .collect();
    let suffix: String = literal.chars().skip(digits.len()).collect();
    assert!(
        suffix
            .chars()
            .all(|c| matches!(c, 'u' | 'U' | 'l' | 'L' | 'f' | 'F')),
        "unsupported integer suffix {suffix:?} in literal {literal:?}"
    );
    if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        assert!(
            !hex.contains('.'),
            "hexadecimal float literal {literal:?} is unsupported"
        );
        return i128::from_str_radix(hex, 16)
            .unwrap_or_else(|e| panic!("bad hexadecimal literal {literal:?}: {e}"));
    }
    if digits.contains('.') {
        let value: f64 = digits
            .parse()
            .unwrap_or_else(|e| panic!("bad float literal {literal:?}: {e}"));
        assert!(
            value.fract() == 0.0,
            "float literal {literal:?} is not a whole number and cannot be an integer constant"
        );
        return value as i128;
    }
    digits
        .parse()
        .unwrap_or_else(|e| panic!("bad integer literal {literal:?}: {e}"))
}

/// sizeof of the types the kernels put in constant expressions.
///
/// # Panics
/// Panics for a type this table does not know, so a new type cannot slip through unchecked.
#[must_use]
pub fn sizeof_type(ty: &str) -> i128 {
    match ty.trim() {
        "char" | "signed char" | "unsigned char" | "uint8_t" | "int8_t" | "bool" => 1,
        "short" | "unsigned short" | "uint16_t" | "int16_t" | "__nv_bfloat16"
        | "__nv_bfloat16_raw" | "half" | "__half" => 2,
        "int" | "unsigned int" | "uint32_t" | "int32_t" | "float" | "__nv_fp8_storage_t"
        | "__nv_fp4_storage_t" => 4,
        "long" | "unsigned long" | "long long" | "unsigned long long" | "uint64_t" | "int64_t"
        | "double" | "size_t" | "uintptr_t" => 8,
        other => panic!("sizeof of unknown type {other:?}"),
    }
}

/// One static_assert from a source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticAssert {
    /// The condition, with whitespace squeezed so multi-line statements compare cleanly.
    pub condition: String,
    /// The human-readable message.
    pub message: String,
}

/// One bool per byte of `text`: true when the byte is inside a string, character or raw string
/// literal. A parser uses this to ignore a keyword that is only literal content.
fn literal_mask(text: &str) -> Vec<bool> {
    let bytes = text.as_bytes();
    let mut mask = vec![false; bytes.len()];
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'R' if bytes.get(index + 1) == Some(&b'"') => {
                let end = raw_string_end(text, index);
                mask[index..end].fill(true);
                index = end;
            }
            quote @ (b'"' | b'\'') => {
                let start = index;
                index += 1;
                while index < bytes.len() {
                    if bytes[index] == b'\\' {
                        index += 2;
                        continue;
                    }
                    let byte = bytes[index];
                    index += 1;
                    if byte == quote {
                        break;
                    }
                }
                let end = index.min(bytes.len());
                mask[start..end].fill(true);
            }
            _ => index += 1,
        }
    }
    mask
}

/// Extracts every static_assert(condition, message) from a source text, including the ones
/// inside a multi-line macro body. Comments are stripped here, so a commented-out assertion is
/// neither extracted nor able to hide a real one behind it, and a `static_assert` that appears
/// only inside a string, character or raw string literal is skipped, so a literal decoy cannot
/// stand in for a real assertion in the pinned per-file count.
///
/// # Panics
/// Panics on an unbalanced static_assert, because a malformed assertion is a real defect.
#[must_use]
pub fn static_asserts(text: &str) -> Vec<StaticAssert> {
    let stripped = strip_comments(text);
    let text: &str = &stripped;
    let literal = literal_mask(text);
    let mut found = Vec::new();
    let mut search = 0usize;
    while let Some(offset) = text[search..].find("static_assert") {
        let start = search + offset;
        if literal[start] {
            // The keyword is literal content, not an assertion. A message or a raw string that
            // spells it out must not count toward the pinned per-file assertion count.
            search = start + "static_assert".len();
            continue;
        }
        let open = text[start..]
            .find('(')
            .map(|index| start + index)
            .unwrap_or_else(|| panic!("static_assert without an open parenthesis"));
        let close = matching_close(text, open, '(', ')');
        let body = &text[open + 1..close];
        let parts = split_top_level(body, ',');
        assert!(
            parts.len() == 2,
            "static_assert with {} arguments: {:?}",
            parts.len(),
            &text[start..close]
        );
        found.push(StaticAssert {
            condition: squeeze_whitespace(&parts[0]),
            message: parts[1].trim().trim_matches('"').to_string(),
        });
        search = close + 1;
    }
    found
}

/// Squeezes runs of whitespace (including the line continuations of a macro) into single spaces.
#[must_use]
pub fn squeeze_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The index of the closing delimiter matching the one at open_index, ignoring string contents.
///
/// # Panics
/// Panics when the delimiters are unbalanced.
#[must_use]
pub fn matching_close(text: &str, open_index: usize, open: char, close: char) -> usize {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut chars = text.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        if index < open_index {
            continue;
        }
        if in_string {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return index;
            }
        }
    }
    panic!("unbalanced {open}...{close} starting at {open_index}");
}

/// Splits a parameter or argument list on commas that are not nested in parentheses, brackets or
/// braces. This is the depth-aware split the payload[4][1024 * 1024] case needed.
#[must_use]
pub fn split_top_level(text: &str, separator: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for c in text.chars() {
        if in_string {
            current.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                current.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(c);
            }
            _ if c == separator && depth == 0 => {
                parts.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

/// One parameter of a __global__ declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// The base type with the pointer and address-space qualifiers removed, e.g. "float".
    pub base: String,
    /// The parameter name.
    pub name: String,
    /// True when the declaration used a pointer.
    pub pointer: bool,
    /// True when the pointee is const.
    pub const_pointee: bool,
    /// True when the parameter carried __restrict__.
    pub restrict: bool,
    /// True for GpuTraceRing* and other by-value struct pointers needed verbatim.
    pub raw_type: String,
}

/// A __global__ kernel declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelDecl {
    /// The kernel name.
    pub name: String,
    /// True when declared extern "C".
    pub extern_c: bool,
    /// True when declared __global__.
    pub global: bool,
    /// Parameters in order.
    pub params: Vec<Param>,
}

/// Parses the declaration of one kernel by name. Comments are stripped here, so a commented-out
/// copy of the declaration above the real one cannot be the one that is parsed.
///
/// # Panics
/// Panics when the kernel is absent, or when the declaration cannot be parsed; both are defects
/// the test must not paper over.
#[must_use]
pub fn kernel_declaration(text: &str, name: &str) -> KernelDecl {
    let stripped = strip_comments(text);
    let text: &str = &stripped;
    let needle = format!("void {name}(");
    let offset = text
        .find(&needle)
        .unwrap_or_else(|| panic!("no declaration of {name} in the source"));
    let head = &text[..offset];
    let open = offset + needle.len() - 1;
    let close = matching_close(text, open, '(', ')');
    let params = split_top_level(&text[open + 1..close], ',')
        .into_iter()
        .map(|param| parse_param(&param))
        .collect();
    KernelDecl {
        name: name.to_string(),
        extern_c: head.contains("extern \"C\""),
        global: head.contains("__global__"),
        params,
    }
}

/// Parses one parameter declaration into its type, qualifiers and name.
///
/// # Panics
/// Panics when the parameter has no name or no recognisable type; an unnamed parameter would
/// make the ABI test meaningless.
#[must_use]
pub fn parse_param(declaration: &str) -> Param {
    let raw_type = squeeze_whitespace(&declaration.replace('\n', " "));
    assert!(!raw_type.is_empty(), "empty parameter declaration");
    // The original spelling is kept for the ABI comparison, and a qualifier-free copy is used to
    // find the name and the base type, so the name is the last word whether or not the parameter
    // is a pointer (a by-value const uint32_t layer_id has no star to split on).
    let stripped = squeeze_whitespace(
        &raw_type
            .replace("__restrict__", " ")
            .replace("__restrict", " ")
            .replace("__grid_constant__", " "),
    );
    let pointer = stripped.contains('*');
    let const_pointee = pointer && stripped.contains("const");
    let restrict = raw_type.contains("__restrict");
    let split = stripped
        .rfind(|c: char| c.is_whitespace())
        .map_or(0, |index| index + 1);
    let name = stripped[split..]
        .split('[')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    assert!(!name.is_empty(), "parameter {declaration:?} has no name");
    let head = stripped[..split].trim_end();
    let base = head
        .replace('*', " ")
        .split_whitespace()
        .filter(|word| *word != "const")
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        !base.is_empty(),
        "parameter {declaration:?} has no base type"
    );
    Param {
        base,
        name,
        pointer,
        const_pointee,
        restrict,
        raw_type,
    }
}

/// The (type, name) pairs of a struct or union body. Comments are stripped here, so a
/// commented-out struct or a decoy field cannot be read as real.
///
/// # Panics
/// Panics when the struct is absent or a field cannot be parsed.
#[must_use]
pub fn struct_fields(text: &str, name: &str) -> Vec<(String, String)> {
    let stripped = strip_comments(text);
    let text: &str = &stripped;
    let needle = format!("struct {name} {{");
    let offset = text
        .find(&needle)
        .or_else(|| text.find(&format!("struct {name}")))
        .unwrap_or_else(|| panic!("no struct {name} in the source"));
    // The brace offset is relative to the slice, so it has to be rebased before it indexes the
    // whole text: the previous version added and then subtracted the struct's own offset, which
    // happened to work only for a struct at the very start of a file.
    let brace = text[offset..].find('{').expect("struct without a body");
    let open = offset + brace;
    let close = matching_close(text, open, '{', '}');
    let body = &text[open + 1..close];
    let mut fields = Vec::new();
    for statement in split_top_level(body, ';') {
        let statement = squeeze_whitespace(&statement);
        if statement.is_empty() || statement.starts_with('{') {
            continue;
        }
        let mut words: Vec<&str> = statement.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let last = words.pop().expect("checked non-empty");
        let field = last.split('[').next().unwrap_or(last).to_string();
        let ty = words.join(" ");
        fields.push((ty, field));
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripping_preserves_the_line_structure() {
        let source = "/* one\ntwo */\n// three\nfour\n";
        let stripped = strip_comments(source);
        assert_eq!(
            stripped.lines().count(),
            source.lines().count(),
            "line numbers must stay comparable after stripping"
        );
        assert!(stripped.contains("four"));
        assert!(!stripped.contains("one") && !stripped.contains("three"));
    }

    #[test]
    fn a_commented_out_declaration_is_not_the_declaration() {
        // The reviewer's footgun: a text parser that trusts its caller finds the commented-out
        // copy first, parses its parameters, and reports an ABI that no compiler would produce.
        let source = r#"
// void spark_probe(const float* __restrict__ decoy, uint32_t n);
/* void spark_probe(const float* __restrict__ block_decoy); */
void spark_probe(const float* __restrict__ real, uint32_t n) {
}
"#;
        let declaration = kernel_declaration(source, "spark_probe");
        assert_eq!(declaration.params.len(), 2);
        assert_eq!(declaration.params[0].name, "real");
        assert!(declaration.params[0].restrict);
        assert_eq!(declaration.params[1].name, "n");
        // ...and the same text passed pre-stripped must give the same answer, because stripping
        // twice is idempotent rather than a second, different path.
        let pre_stripped = strip_comments(source);
        assert_eq!(
            kernel_declaration(&pre_stripped, "spark_probe"),
            declaration
        );
    }

    #[test]
    fn a_commented_out_define_is_not_a_define() {
        let source = "/*\n#define SPARK_M4_VOCAB 999u\n*/\n// #define SPARK_M4_VOCAB 111u\n#define SPARK_M4_VOCAB 129280u\n";
        let consts = Consts::parse(source);
        assert_eq!(consts.value("SPARK_M4_VOCAB"), 129_280);
        assert_eq!(consts.raw("SPARK_M4_VOCAB"), Some("129280u"));
        assert!(
            !defines(source).contains_key("SPARK_M4_DECOY"),
            "there is no decoy, and the parser must not invent one"
        );
    }

    #[test]
    fn a_commented_out_static_assert_is_not_extracted() {
        let source = "// static_assert(1, \"decoy\");\n/* static_assert(0, \"block decoy\"); */\nstatic_assert(SPARK_M4_VOCAB == 129280u, \"real\");\n#define SPARK_M4_VOCAB 129280u\n";
        let asserts = static_asserts(source);
        assert_eq!(asserts.len(), 1, "only the real assertion exists");
        assert_eq!(asserts[0].message, "real");
        assert!(asserts[0].condition.contains("SPARK_M4_VOCAB"));
        // The single extracted assertion really does evaluate, so the extraction above is not
        // passing because nothing was parsed.
        assert_eq!(Consts::parse(source).eval(&asserts[0].condition), 1);
    }

    #[test]
    fn a_quote_or_asterisk_in_a_character_literal_does_not_open_a_comment() {
        // Character literals are tracked, so the '/' and '*' here are not the start of a comment
        // and the string literal that follows is not mistaken for code.
        let source = r#"const char slash = '/'; const char star = '*'; const char quote = '"'; /* real comment */ static_assert(1 == 1, "after");"#;
        let stripped = strip_comments(source);
        assert!(stripped.contains("const char slash = '/'"));
        assert!(stripped.contains("after"));
        assert!(!stripped.contains("real comment"));
        assert_eq!(static_asserts(source).len(), 1);
    }

    #[test]
    fn a_line_comment_that_ends_in_a_backslash_splices_the_next_line_into_it() {
        // Translation phase 2 deletes a backslash-newline pair before comments are removed, so a //
        // comment swallows the next physical line. The reviewer's shape: a decoy and a "real"
        // assertion on the line after such a comment are BOTH still comment, so neither is parsed as
        // code - and an assertion after that line is. The spliced text has one fewer line because
        // the two physical lines really are one logical line.
        let source = "// a real assertion follows \\\nstatic_assert(1 == 1, \"decoy\"); static_assert(SPARK_X == 1u, \"real\");\nstatic_assert(SPARK_Y == 2u, \"later\");\n";
        let stripped = strip_comments(source);
        assert_eq!(
            stripped.lines().count(),
            source.lines().count() - 1,
            "the splice joins two physical lines into one"
        );
        assert!(
            !stripped.contains('\\'),
            "the splice marker is gone from the stripped text: {stripped:?}"
        );
        let asserts = static_asserts(source);
        assert_eq!(
            asserts.len(),
            1,
            "only the assertion after the spliced comment is code: {asserts:?}"
        );
        assert_eq!(asserts[0].condition.trim(), "SPARK_Y == 2u");

        // A comment WITHOUT the backslash does not swallow the next line, and it must still be
        // stripped: the two behaviours have to stay distinguishable.
        let no_splice = "// just this line\nstatic_assert(SPARK_X == 1u, \"real\");\n";
        assert_eq!(static_asserts(no_splice).len(), 1);
    }

    #[test]
    fn a_split_comment_opener_is_a_comment_before_the_parser_sees_it() {
        // The round-3 A7 shape: /<backslash><newline>/ makes the whole line a // comment to the
        // compiler, so the decoy static_assert on it is not in the compiled kernel. A parser that
        // stripped comments before splicing would count it, and a count that is one too high is how
        // a deleted real assertion stays invisible. The same shape re-anchors a poll (the acquire
        // guard's A4 case), which is why the splice lives in strip_comments rather than in a caller.
        let source = "static_assert(SPARK_X == 1u, \"real\");\n/\\\n/ static_assert(SPARK_M4_VOCAB == 129280u, \"decoy\");\n";
        let stripped = strip_comments(source);
        assert!(
            !stripped.contains("SPARK_M4_VOCAB"),
            "the split opener made the decoy line a comment: {stripped:?}"
        );
        let asserts = static_asserts(source);
        assert_eq!(
            asserts.len(),
            1,
            "only the real assertion is in the compiled kernel: {asserts:?}"
        );
        assert_eq!(asserts[0].condition.trim(), "SPARK_X == 1u");

        // The literal version of the same decoy: a static_assert spelled out inside a raw string is
        // content, not an assertion, so it cannot keep the pinned count up while a real one is gone.
        let raw = "const char *s = R\"x(// static_assert(SPARK_M4_VOCAB == 129280u, \"decoy\");)x\";\nstatic_assert(SPARK_Y == 2u, \"real\");\n";
        let raw_asserts = static_asserts(raw);
        assert_eq!(
            raw_asserts.len(),
            1,
            "the raw string is content and its static_assert does not count: {raw_asserts:?}"
        );
        assert_eq!(raw_asserts[0].condition.trim(), "SPARK_Y == 2u");
    }

    #[test]
    fn a_split_identifier_or_condition_is_joined() {
        // Translation phase 2 is applied in code as well as in comments: two identifier halves are
        // one identifier, and a condition continued across a splice is one expression the evaluator
        // can read. This is what closes the reviewer's ret<newline>urn shape.
        let split_identifier = "SPARK_M4_\\\nHIDDEN";
        assert_eq!(splice_lines(split_identifier), "SPARK_M4_HIDDEN");
        assert_eq!(strip_comments(split_identifier), "SPARK_M4_HIDDEN");

        let continued = "static_assert(SPARK_X == \\\n    1u, \"x\");\n";
        let stripped = strip_comments(continued);
        assert!(
            !stripped.contains('\\'),
            "the splice marker does not survive: {stripped:?}"
        );
        let asserts = static_asserts(continued);
        assert_eq!(asserts.len(), 1, "the assertion is still extracted");
        assert_eq!(
            squeeze_whitespace(&asserts[0].condition),
            "SPARK_X == 1u",
            "the joined condition is an expression the evaluator can read"
        );
    }

    #[test]
    fn a_backslash_separated_from_its_newline_by_horizontal_space_is_a_splice() {
        // clang 22 and nvcc 12.9.1 both splice backslash + spaces/tabs + newline, warn
        // -Wbackslash-newline-escape and still exit 0. build.rs compiles with plain -std=c++17 and
        // no -Werror, so a source the compiler accepts is a source the guard has to read the same
        // way. Round-4 A1: each fixture below splices to the single keyword `return`, and clang
        // -E -P really emits `return`.
        let splice = |gap: &str| {
            let mut text = String::from("ret");
            text.push('\\');
            text.push_str(gap);
            text.push('\n');
            text.push_str("urn");
            text
        };
        assert_eq!(splice_lines(&splice("   ")), "return");
        assert_eq!(splice_lines(&splice("\t\t")), "return");
        assert_eq!(splice_lines(&splice(" \t ")), "return");
        // Vertical tab and form feed are the same splice to every frontend this milestone builds
        // with, which is the round-5 E1 finding. Measured on the fixture `ret\<0x0C>\nurn 1;` and
        // its 0x0B twin: clang 22.1.8 and gcc 16.2.1 with -std=c++17 -E -P both emit `return 1;`
        // (each warning -Wbackslash-newline-escape and exiting 0), and nvcc 12.9.1 emits the joined
        // identifier `return` followed by `1;` as well. Skipping only space and tab left an escape
        // that all three splice invisible to this parser, which is a parser bug regardless of what
        // the guard does with it.
        assert_eq!(splice_lines(&splice("\u{b}")), "return");
        assert_eq!(splice_lines(&splice("\u{c}")), "return");
        assert_eq!(splice_lines(&splice(" \u{c}")), "return");
        assert_eq!(splice_lines(&splice("\t\u{b} ")), "return");
        // The same splice in front of a CRLF newline.
        let vt_crlf = {
            let mut text = String::from("ret");
            text.push('\\');
            text.push('\u{b}');
            text.push('\r');
            text.push('\n');
            text.push_str("urn");
            text
        };
        assert_eq!(splice_lines(&vt_crlf), "return");
        // The comment scanner routes through the same splice, so a comment opener split by a form
        // feed hides the rest of its line exactly as a split by spaces does.
        let split_comment = {
            let mut text = String::from("/");
            text.push('\\');
            text.push('\u{c}');
            text.push('\n');
            text.push_str("/ hidden code\nvisible code\n");
            text
        };
        let stripped = strip_comments(&split_comment);
        assert!(
            !stripped.contains("hidden") && stripped.contains("visible code"),
            "a form feed must not hide a // opener from the comment scanner: {stripped:?}"
        );
        // The comment scanner routes through the same splice, so the joined identifier is what the
        // later parsers see.
        assert_eq!(strip_comments(&splice("   ")), "return");
        // CRLF, with the whitespace in front of the pair.
        let crlf = {
            let mut text = String::from("ret");
            text.push('\\');
            text.push_str("  ");
            text.push('\r');
            text.push('\n');
            text.push_str("urn");
            text
        };
        assert_eq!(splice_lines(&crlf), "return");
        // A backslash followed by whitespace but NOT a newline is not a splice, and the text
        // survives unchanged.
        assert_eq!(splice_lines("ret\\   urn"), "ret\\   urn");
    }

    #[test]
    fn a_decoy_field_is_not_a_field() {
        let source = "// struct spark_probe { float decoy; };\n/* struct spark_probe { float block_decoy; }; */\nstruct spark_probe { float real; uint32_t n; };\n";
        assert_eq!(
            struct_fields(source, "spark_probe"),
            vec![
                ("float".to_string(), "real".to_string()),
                ("uint32_t".to_string(), "n".to_string())
            ]
        );
    }
}
