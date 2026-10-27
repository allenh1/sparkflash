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

/// One C token, the two lines it can be reported by, and where it started.
///
/// `line` is the line of the text `code_tokens` was handed, and the directive structure is read from
/// it: `end_of_directive` joins a backslash continuation by comparing token lines, so it has to stay
/// the line of the phase-2-spliced text.
///
/// `physical_line` is the line a reader of the file would find the token on. Every diagnostic uses
/// it, because the spliced line is not the file line: the shipped header has 23 splices before the
/// helper, so the helper was reported at m4_common_sm121.cuh:484 while it sits on line 507.
/// `code_tokens` sets it equal to `line` (there is no map for text that was not spliced here);
/// [`code_tokens_of`] replaces it with the mapped line. `offset` is the byte the token starts at in
/// that same text, which is what the map is keyed by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodeToken {
    line: usize,
    physical_line: usize,
    offset: usize,
    text: String,
}

/// Splits comment-free C into identifiers, numbers, string or character literals and punctuation,
/// keeping the line of each token so an error can point at it.
pub(crate) fn code_tokens(source: &str) -> Vec<CodeToken> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    let mut line = 1usize;
    while index < bytes.len() {
        let current = bytes[index];
        if current == b'\n' {
            line += 1;
            index += 1;
            continue;
        }
        if current.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if current.is_ascii_alphabetic() || current == b'_' {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            // A raw string opener may carry an encoding prefix (u8R"..", uR"..", UR"..", LR"..")
            // or none (R".."). The whole literal, prefix included, is one opaque token: its
            // contents are never code, so a fence or a control keyword spelled inside one cannot
            // be mistaken for the real thing.
            let word = &source[start..index];
            if matches!(word, "R" | "u8R" | "uR" | "UR" | "LR") && bytes.get(index) == Some(&b'"') {
                let end = crate::c_decl::raw_string_end(source, index - 1);
                let text = &source[start..end];
                let start_line = line;
                line += text.matches('\n').count();
                tokens.push(CodeToken {
                    line: start_line,
                    physical_line: start_line,
                    offset: start,
                    text: text.to_string(),
                });
                index = end;
                continue;
            }
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: start,
                text: word.to_string(),
            });
            continue;
        }
        if current.is_ascii_digit() {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || bytes[index] == b'.'
                    || bytes[index] == b'_')
            {
                index += 1;
            }
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: start,
                text: source[start..index].to_string(),
            });
            continue;
        }
        if current == b'"' || current == b'\'' {
            // A literal is one token, so a word inside a message can never be read as a keyword.
            let quote = current;
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
            let text = String::from_utf8_lossy(&bytes[start..end]).into_owned();
            let start_line = line;
            line += text.matches('\n').count();
            tokens.push(CodeToken {
                line: start_line,
                physical_line: start_line,
                offset: start,
                text,
            });
            continue;
        }
        if (current == b'&' || current == b'|') && bytes.get(index + 1) == Some(&current) {
            // C's translator has && and || as single tokens, so the guard keeps them whole: split
            // into two characters a short-circuit operator is indistinguishable from something
            // harmless, and is_control has to be able to name it.
            tokens.push(CodeToken {
                line,
                physical_line: line,
                offset: index,
                text: source[index..index + 2].to_string(),
            });
            index += 2;
            continue;
        }
        tokens.push(CodeToken {
            line,
            physical_line: line,
            offset: index,
            text: (current as char).to_string(),
        });
        index += 1;
    }
    tokens
}

/// Tokens of a source file, with every token's `line` set to the PHYSICAL line of `source`.
///
/// [`code_tokens`] counts newlines in the text it is handed, and the guard hands it the phase-2
/// spliced, comment-stripped text. Both stages delete bytes - the splice deletes the newline, the
/// comment scanner deletes the comment - so every token after the first splice in a file was reported
/// a few lines early: the shipped header has 23 splices before the helper, which put the acquiring
/// poll at 484 while `grep -n` finds it at 507. A diagnostic that names a line the developer cannot
/// find teaches them to distrust the guard, so this is what every check that reports a line uses.
pub(crate) fn code_tokens_of(source: &str) -> Vec<CodeToken> {
    let (stripped, lines) = crate::c_decl::strip_comments_with_lines(source);
    let mut tokens = code_tokens(&stripped);
    for token in &mut tokens {
        token.physical_line = lines[token.offset] as usize;
    }
    tokens
}

/// The directive keyword that follows the `#` at `index`, or `""` when the `#` ends the input.
pub(crate) fn directive_keyword(tokens: &[CodeToken], index: usize) -> &str {
    tokens.get(index + 1).map_or("", |next| next.text.as_str())
}

/// The compile-time state of one arm of a preprocessor conditional, from the point of view of a
/// text guard that cannot generally evaluate the condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arm {
    /// The arm is definitely compiled.
    Active,
    /// The arm is definitely not compiled.
    Disabled,
    /// The guard cannot resolve the condition, so the arm may or may not be compiled.
    Unknown,
}

/// One open preprocessor conditional.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conditional {
    /// The state of the arm currently being read.
    arm: Arm,
    /// Whether some previous arm of this conditional may have been taken.
    prior_maybe_taken: bool,
    /// Whether some previous arm of this conditional was definitely taken.
    prior_definitely_taken: bool,
}

/// The arm a `#if`/`#elif` condition denotes when that condition is a literal 0 or 1, and
/// [`Arm::Unknown`] for anything else.
///
/// The guard does not evaluate expressions, but it must not reject a region the compiler certainly
/// compiles either. `#if (1)`, `#if ((0))` and `#if 1u` are all literal: parentheses group, and an
/// integer suffix is part of the same preprocessing token. Reading only a bare `0`/`1` token
/// rejected those, which is the round-5 E5 finding, two rounds after the round-4 A3 false rejection
/// of `#if 1`. Anything else on the directive line (`#if 1 + 1`) stays unknown: folding further
/// would mean evaluating, which a text scan cannot do. That is stricter than the old bare-token read
/// in the other direction as well - `#if 1 - 1` used to be read as active because only its first
/// token was looked at, and is now unknown.
pub(crate) fn literal_arm(tokens: &[CodeToken], index: usize) -> Arm {
    let end = end_of_directive(tokens, index);
    let mut cursor = index + 2;
    let mut open = 0usize;
    while tokens.get(cursor).is_some_and(|token| token.text == "(") {
        open += 1;
        cursor += 1;
    }
    let Some(token) = tokens.get(cursor) else {
        return Arm::Unknown;
    };
    let digits_end = token
        .text
        .find(|byte: char| !byte.is_ascii_digit())
        .unwrap_or(token.text.len());
    let (digits, suffix) = token.text.split_at(digits_end);
    let arm = match digits {
        "0" => Arm::Disabled,
        "1" => Arm::Active,
        _ => return Arm::Unknown,
    };
    if !suffix
        .chars()
        .all(|byte| matches!(byte, 'u' | 'U' | 'l' | 'L'))
    {
        return Arm::Unknown;
    }
    cursor += 1;
    let mut close = 0usize;
    while close < open && tokens.get(cursor).is_some_and(|token| token.text == ")") {
        close += 1;
        cursor += 1;
    }
    if close != open || cursor != end {
        return Arm::Unknown;
    }
    arm
}

/// Applies one preprocessor directive to the open-conditional stack.
///
/// The guard cannot run the preprocessor, so it tracks a three-way state per level rather than the
/// boolean "disabled" it used before. A literal `#if 0` is definitely disabled, a literal `#if 1`
/// is definitely active, and anything else (`#ifdef`, `#if defined(...)`, an arithmetic condition)
/// is [`Arm::Unknown`]: it may hold, and the guard must not pretend either way. The previous
/// version collapsed `#if 1` and `#ifdef` into the same "not 0" state and then treated any open
/// level as not compiled, which rejected a poll or a fence that a literal `#if 1` really does
/// compile; that was the round-4 A3 false rejection. `#else` and `#elif` invert the arms that came
/// before, and a level whose earlier arm may have been taken stays [`Arm::Unknown`] even when its
/// own literal is `1`.
pub(crate) fn update_conditionals(
    tokens: &[CodeToken],
    index: usize,
    conditionals: &mut Vec<Conditional>,
) {
    match directive_keyword(tokens, index) {
        "if" => {
            let arm = literal_arm(tokens, index);
            conditionals.push(Conditional {
                arm,
                prior_maybe_taken: arm != Arm::Disabled,
                prior_definitely_taken: arm == Arm::Active,
            });
        }
        "ifdef" | "ifndef" => conditionals.push(Conditional {
            arm: Arm::Unknown,
            prior_maybe_taken: true,
            prior_definitely_taken: false,
        }),
        "elif" => {
            if let Some(level) = conditionals.last_mut() {
                level.arm = if level.prior_definitely_taken {
                    Arm::Disabled
                } else if level.prior_maybe_taken {
                    Arm::Unknown
                } else {
                    literal_arm(tokens, index)
                };
                level.prior_maybe_taken |= level.arm != Arm::Disabled;
                level.prior_definitely_taken |= level.arm == Arm::Active;
            }
        }
        "else" => {
            if let Some(level) = conditionals.last_mut() {
                level.arm = if level.prior_definitely_taken {
                    Arm::Disabled
                } else if level.prior_maybe_taken {
                    Arm::Unknown
                } else {
                    Arm::Active
                };
            }
        }
        "endif" => {
            conditionals.pop();
        }
        _ => {}
    }
}

/// Whether every enclosing conditional level is definitely compiled, so a token there is certainly
/// in the translation unit.
pub(crate) fn definitely_compiled(conditionals: &[Conditional]) -> bool {
    conditionals.iter().all(|level| level.arm == Arm::Active)
}

/// Whether some enclosing level is definitely disabled, so a token there is certainly not compiled.
pub(crate) fn definitely_disabled(conditionals: &[Conditional]) -> bool {
    conditionals.iter().any(|level| level.arm == Arm::Disabled)
}

/// The index just past a preprocessor directive's whole logical line, backslash continuations
/// included. A fence or a poll on a continuation line is part of the directive, not code.
pub(crate) fn end_of_directive(tokens: &[CodeToken], index: usize) -> usize {
    let mut directive_line = tokens[index].line;
    let mut cursor = index;
    loop {
        while cursor < tokens.len() && tokens[cursor].line == directive_line {
            cursor += 1;
        }
        let continued = cursor
            .checked_sub(1)
            .and_then(|previous| tokens.get(previous))
            .is_some_and(|previous| previous.text == "\\");
        if !continued {
            return cursor;
        }
        match tokens.get(cursor) {
            Some(next) => directive_line = next.line,
            None => return cursor,
        }
    }
}

/// Marks every token that is definitely compiled: not on a preprocessor directive line and in a
/// region where every enclosing conditional level is definitely active.
///
/// A poll is only a poll on such a token, because otherwise the compiler may never reach it. A
/// region the guard cannot resolve (`#ifdef`, an arithmetic `#if`) is NOT marked, so a poll that
/// lives only there is reported as no poll: that is the conservative half of the rule and it is
/// deliberate, not a limitation to be removed. A literal `#if 1` and the `#else` arm of a literal
/// `#if 0` ARE marked, because the compiler certainly compiles them; treating those as unreachable
/// was the round-4 A3 false rejection.
///
/// This is deliberately stricter than the post-poll fence walk, which accepts a fence whenever
/// every enclosing level is definitely active but rejects one in a may-not-hold region rather than
/// reading it as unconditional.
pub(crate) fn active_code_mask(tokens: &[CodeToken]) -> Vec<bool> {
    let mut active = vec![false; tokens.len()];
    let mut conditionals: Vec<Conditional> = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        if definitely_compiled(&conditionals) {
            active[index] = true;
        }
        index += 1;
    }
    active
}

/// The conditional stack in effect at the code token `stop`, recovered by scanning from the top of
/// the token list.
///
/// The fence walk starts at the poll, so without this it would begin with an empty stack and read
/// the `#else` of an enclosing `#if 1` as unconditional code. Recovering the stack at the poll
/// keeps the two scans of the same token list consistent.
pub(crate) fn conditional_stack_before(tokens: &[CodeToken], stop: usize) -> Vec<Conditional> {
    let mut conditionals = Vec::new();
    let mut index = 0usize;
    while index < stop && index < tokens.len() {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        index += 1;
    }
    conditionals
}

/// Whether a token ends the fall-through path, or makes the fence's execution conditional.
///
/// `throw` and `co_return` are here because they end the fall-through path where they are legal: a
/// device function may not throw (nvcc refuses `throw` in device code) and a device function may not
/// be a coroutine, so neither can actually ship a missing acquire - but the guard reads the shape of
/// the source, and a body that spells `throw 0u;` between the poll and the fence has an exit it
/// cannot see through, so it is treated like the other control tokens rather than left to a proof
/// about which constructs nvcc accepts.
pub(crate) fn is_control(text: &str) -> bool {
    matches!(
        text,
        "if" | "else"
            | "for"
            | "while"
            | "do"
            | "switch"
            | "case"
            | "default"
            | "return"
            | "break"
            | "continue"
            | "goto"
            | "throw"
            | "co_return"
            | "?"
            // A short-circuit operator makes what follows conditional, which is exactly the hazard
            // here: (ready == 0u) && ({ __threadfence(); 1; }) runs the fence only on one branch.
            // The single-character forms are rejected as well, because they are the halves of the
            // same operators if the tokenizer is ever changed again, and because a bitwise & or |
            // with a statement-expression operand is the same hazard. Rejecting them is deliberately
            // conservative: this guard is allowed to refuse a shape it cannot prove, and the kernel
            // it protects has neither operator between its poll and its fence.
            | "&&"
            | "||"
            | "&"
            | "|"
    )
}

/// Whether the control token is an OPERATOR, which is the only kind the statement scoping of
/// [`a_statement_completed_between`] may excuse. A control KEYWORD is never excused: `if`, `while`,
/// `for`, `switch` can have the poll inside their body, and `return`, `break`, `continue`, `goto`,
/// `throw` can leave the helper before it, so `if (ready) return 0u;` before the poll is still a
/// blocker even though that statement completed.
pub(crate) fn is_control_operator(text: &str) -> bool {
    matches!(text, "&&" | "||" | "&" | "|" | "?")
}

/// Whether a control keyword leaves the path rather than making it conditional. A `return`, `break`,
/// `continue`, `goto`, `throw` or `co_return` before the poll sends the path elsewhere, so the
/// acquire after it is UNREACHABLE, which is a different fact from "the acquire may not be reached"
/// (round-20 A6). The distinction is not cosmetic: `return poll(...); __threadfence();` calls the poll
/// unconditionally and the fence after it is dead code, and reporting that as a conditional evaluation
/// was false of it.
pub(crate) fn is_an_exit_keyword(text: &str) -> bool {
    matches!(
        text,
        "return" | "break" | "continue" | "goto" | "throw" | "co_return"
    )
}

/// Whether an exit keyword ends the HELPER's path, as the argument-span reading counts them: `return`,
/// `goto`, `throw` and `co_return`. `break` and `continue` are loop-scoped and cannot end the helper
/// from inside an argument list, and a `?` or a `&&` is not an exit at all (round-25 A3).
pub(crate) fn is_an_argument_exit(text: &str) -> bool {
    matches!(text, "return" | "goto" | "throw" | "co_return")
}

/// Whether the `{` at `index` opens a LAMBDA body rather than a block of the function being read. A
/// `return` inside it exits the lambda, not the helper, which the documentation has promised since
/// round 10, so the argument-span reading skips such a body (round-25 A3).
///
/// Round 30 rewrote the walk-back to read BALANCED GROUPS instead of guessing from the token before
/// the brace, because the guess produced two false verdicts. Working backwards from the body brace,
/// one piece at a time: a TRAILING RETURN TYPE is skipped by finding the `->` that introduces it,
/// so the `)` of `decltype(auto)` inside the type is a group rather than the end of the scan; then
/// the specifiers; then a balanced `(...)` parameter list; then a balanced `<...>` explicit template
/// parameter list; and finally the `]` of the CAPTURE LIST, which is what makes it a lambda. An
/// attribute is a `[[...]]` pair, so a `]` that closes one is skipped instead of being read as a
/// capture list: `[[maybe_unused]] { ... }` is a plain block and stays refused, while
/// `[x = (&deadline)[0]] { ... }` and `[&] [[maybe_unused]] { ... }` are lambdas and are accepted.
pub(crate) fn a_lambda_body_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if tokens.get(index).map(|token| token.text.as_str()) != Some("{") {
        return false;
    }
    // (1) The trailing return type, if the body has one: the brace follows `-> type`, and a type can
    // hold `(...)`, `[...]`, `{...}` and `<...>` groups as well as `*`, `const`, `::` and
    // identifiers. Round 31 A1: each bracket KIND gets its own counter, because a `<` or a `>`
    // written inside a parenthesised group is a comparison in a type - `std::enable_if_t<(1 < 2),
    // int>`, `decltype(1 << 2, int{})`, `decltype((1 <= 2), int{})` - and one shared counter let
    // it cancel the `)` of that group, so the scan never reached the arrow and the trailing-return
    // forms were refused with an unreachability sentence that was false of the compiled body. A
    // `;`, a top-level brace or a top-level opener ends the scan with no arrow.
    let mut cursor = index;
    let mut probe = index;
    let mut parens = 0usize;
    let mut squares = 0usize;
    let mut angles = 0usize;
    let mut braces = 0usize;
    let mut walked = 0usize;
    while probe > 0 {
        probe -= 1;
        walked += 1;
        if walked > 256 {
            return false;
        }
        let text = tokens[probe].text.as_str();
        // The tokenizer splits the arrow into `-` and `>`, so the pair is what introduces a
        // trailing return type. This is checked before the `>` arm below, which counts a `>` as
        // the close of a template group. An arrow is never inside a group.
        if text == ">"
            && parens == 0
            && squares == 0
            && braces == 0
            && probe > 0
            && tokens[probe - 1].text == "-"
        {
            cursor = probe - 1;
            break;
        }
        match text {
            // A `]` that closes no group of its own is the capture list, so the brace is the body
            // of a lambda (or of nothing this detector reads) and phase 1 is over.
            "]" if parens == 0 && squares == 0 && angles == 0 && braces == 0 => break,
            ")" => parens += 1,
            "]" => squares += 1,
            "}" => braces += 1,
            "(" => {
                if parens == 0 {
                    break;
                }
                parens -= 1;
            }
            "[" => {
                if squares == 0 {
                    break;
                }
                squares -= 1;
            }
            "{" => {
                if braces == 0 {
                    break;
                }
                braces -= 1;
            }
            // An angle bracket inside another kind of group is a comparison, not a delimiter of
            // the type: it is ignored so the group it sits in still closes on its own counter.
            // The `>` of a `>=` is a comparison as well (round 32 A1): counting it left the
            // group it sits in open, so the scan never reached the arrow.
            ">" if parens == 0 && squares == 0 && braces == 0 => {
                if tokens.get(probe + 1).map(|next| next.text.as_str()) != Some("=") {
                    angles += 1;
                }
            }
            // Round 32 A1: a top-level `<` is a delimiter only when a template NAME stands in
            // front of it. `std::enable_if_t<1 < 2, int>` and `enable_if_t<N < 2, int>` write a
            // COMPARISON there, and counting it closed the group the comparison sat in, so the
            // scan stopped at the real opener and the trailing-return forms were refused with
            // the unreachability sentence. A comparison neither opens nor closes anything, and
            // it does not end the scan either - the scan ends at a `;`, a top-level brace, the
            // beginning of the text or the step bound, so the walk still reaches the arrow.
            "<" if parens == 0 && squares == 0 && braces == 0 => {
                if a_template_name_stands_before(tokens, probe) && angles > 0 {
                    angles -= 1;
                }
            }
            ";" if parens == 0 && squares == 0 && angles == 0 && braces == 0 => break,
            _ => {}
        }
    }
    // (2) The pieces between the capture list and the body, in reverse order. The first `]` that is
    // NOT the close of an attribute is the capture list, so it is the one that answers the question.
    let mut pieces = 0usize;
    loop {
        pieces += 1;
        if pieces > 64 || cursor == 0 {
            return false;
        }
        match tokens[cursor - 1].text.as_str() {
            "mutable" | "noexcept" | "constexpr" | "consteval" => cursor -= 1,
            "]" => match an_attribute_closes_at(tokens, cursor - 1) {
                Some(opener) => cursor = opener,
                None => return true,
            },
            ")" => match the_matching_opener(tokens, cursor - 1, "(", ")") {
                Some(opener) => cursor = opener,
                None => return false,
            },
            // The angle matcher skips the groups nested inside the template argument or parameter
            // list, so a comparison there is not read as a delimiter (round 31 A1).
            ">" => match the_matching_angle_opener(tokens, cursor - 1) {
                Some(opener) => cursor = opener,
                None => return false,
            },
            _ => return false,
        }
    }
}

/// The index of the `<` that matches the template closer at `close`, walking backwards. Unlike
/// `the_matching_opener` this one tracks every bracket kind, because a template argument or
/// parameter list may CONTAIN groups - `std::enable_if_t<(1 < 2), int>`, `[&]<auto N = (1 << 2)>()`,
/// `decltype(int{})` - and a `<` or `>` written inside one of them is a comparison rather than a
/// delimiter, which is exactly the round-31 A1 defect: one shared counter let `(1 < 2)` cancel the
/// `)` it sits in, so the scan walked past the arrow. Inside a group the angle brackets here are
/// ignored; at group depth zero they nest as usual. The arrow's own `>` (`-` then `>` in this token
/// stream) is not a template close. `None` when the brackets do not balance.
pub(crate) fn the_matching_angle_opener(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut parens = 0usize;
    let mut squares = 0usize;
    let mut braces = 0usize;
    let mut angles = 1usize;
    let mut cursor = close;
    // Round 32 A1: `>` and `<` are comparisons as well, so the first `<` that would close the
    // count is not always the opener - in `[&]<auto N = M < 2>()` it is the comparison after
    // `M`, in front of which no `]` stands, so phase 2 cannot consume it. The walk therefore
    // REMEMBERS the leftmost opener it has matched and keeps looking: a comparison inside the
    // group must not stand in for its opener, which is the reading the standard gives when the
    // name in front of the inner `<` is a value rather than a template.
    let mut opener: Option<usize> = None;
    while cursor > 0 {
        cursor -= 1;
        let text = tokens[cursor].text.as_str();
        if text == ">"
            && parens == 0
            && squares == 0
            && braces == 0
            && cursor > 0
            && tokens[cursor - 1].text == "-"
        {
            continue;
        }
        match text {
            ")" => parens += 1,
            "]" => squares += 1,
            "}" => {
                if braces == 0 {
                    return opener;
                }
                braces -= 1;
            }
            "(" => {
                if parens == 0 {
                    return None;
                }
                parens -= 1;
            }
            "[" => {
                if squares == 0 {
                    return None;
                }
                squares -= 1;
            }
            "{" => {
                // The start of a block ends the statement, exactly as the `;` below does.
                if braces == 0 {
                    return opener;
                }
                braces -= 1;
            }
            // A `;` or a preprocessor line ends the statement, so no group can open before
            // it: the candidate found inside the statement is the answer.
            ";" | "#" if parens == 0 && squares == 0 && braces == 0 => return opener,
            // The `>` of a `>=` is a comparison, not a template close (round 32 A1).
            ">" if parens == 0 && squares == 0 && braces == 0 => {
                if tokens.get(cursor + 1).map(|next| next.text.as_str()) != Some("=") {
                    angles += 1;
                }
            }
            // Round 32 A1: only a `<` with a template NAME in front of it, or the one that
            // follows a generic lambda's capture list, opens a template group; `1 < 2` and
            // `1 <= 2` write a comparison there.
            "<" if parens == 0 && squares == 0 && braces == 0 => {
                let named = a_template_name_stands_before(tokens, cursor)
                    || tokens
                        .get(cursor.wrapping_sub(1))
                        .map(|before| before.text.as_str())
                        == Some("]");
                if named {
                    angles -= 1;
                    if angles == 0 {
                        opener = Some(cursor);
                        angles = 1;
                    }
                }
            }
            _ => {}
        }
    }
    opener
}
/// The index of the opener that matches the closer at `close`, walking backwards. `None` when the
/// tokens do not balance, which is also how a caller refuses to guess (round-30 A2).
pub(crate) fn the_matching_opener(
    tokens: &[CodeToken],
    close: usize,
    open: &str,
    shut: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close + 1;
    while cursor > 0 {
        cursor -= 1;
        let text = tokens[cursor].text.as_str();
        if text == shut {
            depth += 1;
        } else if text == open {
            if depth == 0 {
                return None;
            }
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        }
    }
    None
}

/// Whether the `]` at `close` ends an ATTRIBUTE-SPECIFIER, `[[...]]`, which is written with its two
/// brackets adjacent on both sides. An init-capture whose initialiser ends in a subscript ends `]]`
/// as well - `[x = (&deadline)[0]]` - but its group starts `[x`, so it is a capture list and this
/// returns `None`, which makes the walk-back answer `true`. Round 29 tested the two tokens before the
/// brace instead and false-refused that lambda and `[&] [[maybe_unused]] { ... }` (round-30 A2).
pub(crate) fn an_attribute_closes_at(tokens: &[CodeToken], close: usize) -> Option<usize> {
    if tokens.get(close).map(|token| token.text.as_str()) != Some("]") {
        return None;
    }
    let opener = the_matching_opener(tokens, close, "[", "]")?;
    if opener + 1 < close && tokens[opener + 1].text == "[" && tokens[close - 1].text == "]" {
        Some(opener)
    } else {
        None
    }
}

/// The last token before the attribute groups standing at `last`, if there are any.
///
/// C++ lets a balanced attribute group stand between a control statement's head and the brace of its
/// body: `if (deadline) [[likely]] { ... }`, and the same after `else`, `do`, `for`, `while` and
/// `switch`. It does NOT for `try` or `catch`: the grammar has no attribute between `try` and its
/// compound statement, nor between a catch clause's `)` and its compound statement, and both g++ 16 and
/// clang 22 reject `try [[likely]] {` and `catch (...) [[likely]] {` at parse in c++17, c++20, c++23 and
/// c++2c ("expected `{` before `[`"). Those two spellings are still inputs here because the readings
/// below run on a token stream, and the fixtures pin this walk with them. The token at `last` is then
/// the attribute's closing bracket, which is no control token, and a rule that tests it alone reads the
/// body as a bare block. What that hid: the refusal the plain spellings rest on is keyed on the token
/// that ends the head - the `)` of an if/for/while/switch/catch condition - or on `try`, `do` and
/// `requires` naming a body directly, so for `catch (...) [[likely]] {` it is the `)` of the parameter
/// list that the attribute bracket merely hid, and the attributed `try` and `catch` bodies were ACCEPTED
/// while the plain ones were refused (round-36 A1). Neither `try` nor `catch` is in [is_control], so no
/// later blocker rescues them.
/// [a_bare_block_opens_at] has stepped over the group since round-34 A1; this is the same walk, shared
/// so the three rules that need it cannot drift apart again.
pub(crate) fn the_token_before_the_attributes(
    tokens: &[CodeToken],
    last: Option<usize>,
) -> Option<usize> {
    let mut before = last;
    while let Some(close) = before {
        let Some(opener) = an_attribute_closes_at(tokens, close) else {
            break;
        };
        before = opener.checked_sub(1);
    }
    before
}

/// Whether a `;` at the same brace nesting as `from` stands between `from` and `to`, which means the
/// token at `from` belongs to a statement that was already complete before `to` began.
///
/// Round-19 A1: this is what keeps a control OPERATOR in an earlier, finished statement from being read
/// as a blocker of the poll's own statement. `const bool ran = (deadline > 0u) && true;` before the poll
/// ends at its `;`, so by the time the poll is evaluated nothing about that operator bounds the path -
/// and calling its evaluation conditional would be false. A control KEYWORD (`if`, `while`, `return`)
/// before the poll is NOT excused by this: its body can still bound the path, and the poll may be
/// inside it. A macro whose body carries only an operator obeys the same scoping as the literal
/// operator (round-20 A1), which is what makes `#define PRE (deadline > 0u) && true` before the poll
/// accepted while `#define BAIL return` before it stays refused.
pub(crate) fn a_statement_completed_between(tokens: &[CodeToken], from: usize, to: usize) -> bool {
    let mut depth = 0i32;
    for token in tokens.iter().take(to).skip(from + 1) {
        match token.text.as_str() {
            "{" => depth += 1,
            "}" => depth -= 1,
            ";" if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Whether the control token at `control` OWNS the poll at `poll` as the condition the poll is
/// evaluated in, rather than standing before it the way a blocker means.
///
/// Round-18 A2/A3: the poll may be the condition of `if`, `while`, `switch`, `for` or `catch`, or an
/// operand of the conditional operator `?`, and the fence after that whole statement runs
/// unconditionally - so a control token that owns the statement the poll stands in is not a blocker.
/// What is still a blocker is everything that can take the path away BEFORE the poll: `return
/// poll(...)` (a `return` is never a condition), `if (x) poll(...);` (the poll is in the body, and
/// `x` may be false), a short-circuit operator `ready && poll(...)` (the poll may not be evaluated)
/// and a `?` in an earlier, completed statement. The `(` is looked for between the keyword and the
/// poll, so a keyword whose condition has already closed does not qualify.
pub(crate) fn the_control_token_governs_the_poll(
    tokens: &[CodeToken],
    control: usize,
    poll: usize,
) -> bool {
    if control >= poll {
        return false;
    }
    let text = tokens[control].text.as_str();
    if text == "?" {
        // The conditional expression's own statement: a `;` or a brace between the `?` and the poll
        // puts the `?` in an earlier statement, where it cannot govern anything.
        return !tokens[control..poll]
            .iter()
            .any(|token| matches!(token.text.as_str(), ";" | "{" | "}"));
    }
    if !matches!(text, "if" | "while" | "switch" | "for" | "catch") {
        return false;
    }
    let Some(open) = (control + 1..poll).find(|&index| tokens[index].text == "(") else {
        return false;
    };
    matching_close_bracket(tokens, open).is_some_and(|close| close >= poll)
}

/// Calls that never return, so a fence after one is unreachable rather than unconditional. The
/// list names the standard spellings a kernel could reach: the compiler builtins
/// (`__builtin_trap`, `__builtin_unreachable`), the C/POSIX process exits (`abort`, `exit`,
/// `_Exit`, `_exit`, `quick_exit`), the C11 thread exit (`thrd_exit`), the POSIX thread exit
/// (`pthread_exit`) and `longjmp`. A name reached through a macro, a function pointer or an
/// inline-asm block is not covered, which is a stated limit.
pub(crate) const NON_RETURNING_CALLS: [&str; 11] = [
    "__builtin_trap",
    "__builtin_unreachable",
    "__trap",
    "abort",
    "exit",
    "_Exit",
    "_exit",
    "quick_exit",
    "thrd_exit",
    "pthread_exit",
    "longjmp",
];

/// Whether the name at `index` is used as a called designator, in every spelling the preprocessor
/// leaves alone: `f(...)`, `(f)(...)`, `(*f)(...)` and `(((f)))(...)`.
///
/// This is the same unwrap as [`call_argument_list`] without the "and a semicolon follows the
/// argument list" part, because a call inside an expression is still a call. It exists because the
/// round-5 review replaced the acquiring call with `(spark_wait_for_flag)(&hc_sync[1], deadline)`:
/// parenthesising a function designator does not stop it from being called, and the preprocessor does
/// not rewrite it, so a check that only looked at the token after the name saw no call at all.
pub(crate) fn designator_is_a_call(tokens: &[CodeToken], index: usize) -> bool {
    call_argument_list(tokens, index).is_some()
}

/// The indices of the raw poll designator being CALLED, in source order.
///
/// Both checks locate the poll by this one rule - the raw poll designator is called - so their
/// verdicts cannot diverge on the poll's spelling (round-13 A-2). The lint used to take a literal
/// token pattern as its `poll` argument, so `spark_wait_for_flag((flag), (deadline))`, a cast
/// argument and plain spacing differences were located by the structural check and missed by the
/// lint, which then reported "no poll of spark_wait_for_flag(flag, deadline) in this source's code"
/// for code nvcc compiles to the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e, measured).
pub(crate) fn raw_poll_call_indices(tokens: &[CodeToken]) -> Vec<usize> {
    let directives = directive_token_mask(tokens);
    // The macros in effect at each position, in source order: whether a name in front of an argument
    // list is a macro USE (round-4 A2: `BAIL poll(...)`) or the return TYPE of a declaration
    // (`#define uint32_t unsigned` then `uint32_t spark_wait_for_flag(...);`, round-15 A-1) is a
    // question about that macro, so the scope walks the text with this scan.
    let mut macros = MacroScope::from_tokens(tokens);
    let mut found = Vec::new();
    for index in 0..tokens.len() {
        macros.advance_to(index);
        if tokens[index].text == RAW_POLL
            && designator_is_a_call(tokens, index)
            && !is_a_declaration_of(tokens, index, &macros, &directives)
        {
            found.push(index);
        }
    }
    found
}

/// Whether the called designator at `index` is a DECLARATION of that function rather than a call of
/// it, whatever its return type: the argument list is followed by `;` (a prototype), by `=` (a
/// defaulted or pure-virtual declaration) or by the `{` of a body, or a declaration specifier
/// stands before the name (`void abort();`).
///
/// The `;` rule is what makes the return type irrelevant. Round-14 A-1: a plain forward
/// declaration `uint32_t spark_wait_for_flag(const volatile uint32_t *, uint64_t);` was read as a
/// CALL, because `uint32_t` is not in `DECLARATION_SPECIFIERS` and the declaration does not open a
/// body - so the lint anchored on plumbing and reported a `return` inside this very definition as
/// standing before the poll, for a header nvcc compiles to the baseline cubin.
///
/// A function-pointer PARAMETER is deliberately not covered: `(*f)(args)` is the same tokens
/// whether it declares a parameter or calls through a pointer, so the lint anchors on the helper's
/// own body rather than guessing from the following bracket.
/// Whether the mention at `index` is spelled with a VOID return type: `void (spark_wait_for_flag)
/// (flag, deadline);`. The parentheses around the designator are also the shape of a call through a
/// parenthesised designator, so the parentheses alone cannot decide it; the declaration specifier in
/// front of them can. The tokens redeclare the poll at block scope with a `void` return type, and nvcc
/// refuses the translation unit rather than compiling it: "error: incomplete type void is not allowed"
/// (round-23 finding 5, measured in the CUDA 13.0.3 container on sm_121a). Both checks refuse it, with
/// the same sentence, because a body that is not a translation unit has no cubin to compare with the
/// baseline.
pub(crate) fn a_void_type_precedes_the_designator(tokens: &[CodeToken], index: usize) -> bool {
    let Some(open) = index
        .checked_sub(1)
        .filter(|&previous| tokens[previous].text == "(")
    else {
        return false;
    };
    if matching_close_bracket(tokens, open) != Some(index + 1) {
        return false;
    }
    open.checked_sub(1)
        .is_some_and(|before| tokens[before].text == "void")
}

/// The shared message for that shape. The callers wrap it in the one constructor, so the two checks
/// answer the spelling with the same words.
pub(crate) fn void_typed_poll_message(line: usize) -> String {
    format!(
        "line {line}: {RAW_POLL} is spelled with a void return type here - `void ({RAW_POLL})(...)` \
         redeclares the observation instead of calling it, and nvcc refuses the translation unit \
         (\"error: incomplete type void is not allowed\"); the shared header writes the call as \
         `{CANONICAL_POLL_CALL}`"
    )
}

pub(crate) fn is_a_declaration_of(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    if a_void_type_precedes_the_designator(tokens, index) {
        return true;
    }
    if index > 0 && DECLARATION_SPECIFIERS.contains(&tokens[index - 1].text.as_str()) {
        return true;
    }
    let after_the_arguments = call_argument_list(tokens, index)
        .and_then(|(_, open)| matching_close_bracket(tokens, open))
        .map(|close| skip_noexcept_clause(tokens, close + 1))
        .and_then(|next| tokens.get(next))
        .map(|next| next.text.as_str());
    match after_the_arguments {
        // A body brace cannot follow a call's argument list, so this needs no type: it is the
        // definition of the poll itself.
        Some("{") => true,
        // A prototype or a defaulted/pure-virtual declaration. A bare call statement ends with `;`
        // too, so the token before the name has to say which one this is: a TYPE name means the name
        // is being declared, anything else (`=`, a bracket, an operator, a statement keyword) means
        // it is being called.
        Some(";" | "=") => a_type_stands_before(tokens, index, macros, directives),
        _ => false,
    }
}

/// The token after an optional `noexcept` or `noexcept(...)` clause. A declaration of the poll may
/// carry one - `uint32_t poll(...) noexcept;` - and it is still a declaration, not a call (round-15
/// A6).
pub(crate) fn skip_noexcept_clause(tokens: &[CodeToken], mut index: usize) -> usize {
    let is_noexcept = tokens
        .get(index)
        .is_some_and(|token| token.text == "noexcept");
    if is_noexcept {
        index += 1;
    }
    // The clause is either bare or parenthesised; the parentheses are the only part with a shape.
    let parenthesised = if is_noexcept && tokens.get(index).is_some_and(|token| token.text == "(") {
        matching_close_bracket(tokens, index)
    } else {
        None
    };
    if let Some(close) = parenthesised {
        index = close + 1;
    }
    index
}

/// Whether a type name stands before the called designator at `index`, through a pointer or
/// reference chain: the return type of a declaration whose type is a typedef or a class name
/// (`uint32_t spark_wait_for_flag(...)`), which `DECLARATION_SPECIFIERS` cannot see.
///
/// The token in front of a CALL is `=`, a bracket, an operator or a statement keyword
/// (`... = poll;`, `(poll)`, `return poll;`, `ready > poll(...)`), and the shapes that have
/// something else there are the return types a declaration can have:
///
///   * an identifier that is not a statement keyword - a typedef, a class name, an unknown name -
///     which is a TYPE unless the source defines it as a macro that can be INVOKED. A macro name can
///     be a return type (`#define uint32_t unsigned` then `uint32_t spark_wait_for_flag(...);` is
///     a declaration, round-15 A-1), and the shapes are otherwise identical, so the macro's own body
///     decides: a macro whose expansion contains a control keyword or a non-returning call is a
///     control STATEMENT here (`#define BAIL return` then `BAIL poll(...);` must stay a call,
///     round-4 A2), and any other macro is read as the type it stands for;
///   * a `>` that closes a template argument list (`wrapper<uint32_t> spark_wait_for_flag(...);`)
///     or a `)` that closes a `decltype(...)` - the two return types that end in punctuation.
///     Neither is accepted on a guess: the `<` has to be there (so a comparison, `ready >
///     poll(...)`, stays a call) and the token in front of the `<` must not make it an arrow
///     (`p->poll(...)`), and only `decltype` may open the parentheses.
///
/// A token of a `#define` line is excluded: it belongs to that directive (`#define CHAIN1 CHAIN0`
/// followed by the poll), not to a declaration.
pub(crate) fn a_type_stands_before(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    let mut cursor = index;
    loop {
        if cursor == 0 {
            return false;
        }
        let previous = cursor - 1;
        match tokens[previous].text.as_str() {
            // A pointer or reference declarator belongs to the type.
            "*" | "&" => cursor = previous,
            // An attribute-specifier between the type and the name belongs to the declaration:
            // `uint32_t [[maybe_unused]] poll(...);` declares the name (round-15 A2). The tokenizer
            // emits one token per punctuation character, so the closing pair is two `]` tokens and the
            // one in front of the name is a single `]` - matching `"]]"` here matched nothing, which
            // made this arm dead and let a declaration through as a call (round-17 A1).
            "]" => match attribute_specifier_open(tokens, previous) {
                Some(open) => cursor = open,
                None => return false,
            },
            // The opening parenthesis of a function-pointer declarator, `TYPE (*name)(args)`: the
            // name stands inside `(* ... )` and the type is what stands before the `(`. The same
            // tokens are a call THROUGH a function pointer, so this accepts the declarator and cannot
            // separate the two; that is stated as out of scope in the module documentation.
            "(" if tokens
                .get(previous + 1)
                .is_some_and(|next| matches!(next.text.as_str(), "*" | "&")) =>
            {
                cursor = previous;
            }
            _ => break,
        }
    }
    let previous = cursor - 1;
    if directives[previous] {
        return false;
    }
    let text = tokens[previous].text.as_str();
    if text == ">" {
        let not_an_arrow = previous == 0 || tokens[previous - 1].text != "-";
        return not_an_arrow && matching_open_angle(tokens, previous).is_some();
    }
    if text == ")" {
        return decltype_stands_before(tokens, previous);
    }
    if !is_identifier(text) || STATEMENT_KEYWORDS.contains(&text) {
        return false;
    }
    if macros.is_defined(text) {
        return !macros.is_suspicious(text);
    }
    true
}

/// The index of the `[[` that opens the attribute-specifier the `]]` at `close` ends, or None. A
/// pair of adjacent brackets is what makes this an attribute-specifier rather than `a[i]`.
pub(crate) fn attribute_specifier_open(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        match tokens[cursor].text.as_str() {
            "]" => depth += 1,
            "[" => {
                depth -= 1;
                if depth == 0 {
                    return tokens
                        .get(cursor + 1)
                        .is_some_and(|next| next.text == "[")
                        .then_some(cursor);
                }
            }
            _ => {}
        }
        cursor = cursor.checked_sub(1)?;
    }
}

/// The index of the `(` or `[` that matches the `)` or `]` at `close`, walking backwards. Bounded by
/// the statement, like the angle hunt it serves: an unbalanced group is not a group.
pub(crate) fn matching_open_bracket(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let (open, shut) = match tokens[close].text.as_str() {
        ")" => ("(", ")"),
        "]" => ("[", "]"),
        _ => return None,
    };
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        let text = tokens[cursor].text.as_str();
        if text == shut {
            depth += 1;
        } else if text == open {
            depth -= 1;
            if depth == 0 {
                return Some(cursor);
            }
        } else if text == ";" || text == "#" {
            return None;
        }
        cursor = cursor.checked_sub(1)?;
    }
}

/// Whether a template NAME stands in front of the `<` at `open`: an identifier, a `>`, a `::`,
/// the `template` keyword or a type keyword (every one of those is an identifier to
/// `is_identifier`). A `<` with none of them in front of it is a COMPARISON, and a comparison
/// neither opens a template group nor closes one. Round-18 A4 needed the test to tell
/// `0u < 1u > poll(...)` from `name<0u>`; round 32 A1 needs the SAME test in the two walks that
/// look for a lambda body, where counting `std::enable_if_t<1 < 2, int>` as an opener closed the
/// group the comparison sat in and refused a textbook nvcc compiles to the baseline. One
/// definition, three call sites: round 18 introduced it inside `matching_open_angle`, and round 32
/// put the two lambda walks that had their own angle handling onto the same definition.
pub(crate) fn a_template_name_stands_before(tokens: &[CodeToken], open: usize) -> bool {
    open.checked_sub(1).is_some_and(|before| {
        let text = tokens[before].text.as_str();
        is_identifier(text) || text == ">" || text == ":"
    })
}

/// The `<` that matches the `>` at `close`, or None when that `>` does not close a template
/// argument list. The tokenizer emits one token per punctuation character, so `>>` is two nested
/// `>` and a nested template id is counted correctly.
pub(crate) fn matching_open_angle(tokens: &[CodeToken], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut index = close;
    loop {
        match tokens[index].text.as_str() {
            ">" => depth += 1,
            "<" => {
                depth -= 1;
                if depth == 0 {
                    // A `<` opens a template argument list only when a template NAME stands before it
                    // (round-18 A4): the token in front of the `<` of `0u < 1u > poll(...)` is a
                    // numeric literal, so that `<` is a comparison and the `>` in front of the poll is
                    // not a template close. Reading it as one refused a call nvcc compiles to the same
                    // size as the shipped comparison.
                    return if a_template_name_stands_before(tokens, index) {
                        Some(index)
                    } else {
                        None
                    };
                }
            }
            // A balanced group is skipped rather than walked through (round-18 A4), so an earlier `<`
            // inside the poll's OWN statement - even inside parentheses, `(0u < 1u) > poll(...)` - is
            // not read as a template open. The statement bound below still ends the hunt.
            ")" | "]" => index = matching_open_bracket(tokens, index)?,
            // The hunt is bounded by the STATEMENT (round-17 A3): a `;`, a brace or a preprocessor
            // line ends it, because a template argument list cannot span one. Without the bound, the
            // walk reached an unmatched `<` anywhere earlier in the file - an include, a comparison in
            // another function - so the SAME comparison was a call in one source and a declaration in
            // another, and the documented claim "a comparison stays a call" was false.
            ";" | "{" | "}" | "#" => return None,
            _ => {}
        }
        if index == 0 {
            return None;
        }
        index -= 1;
    }
}

/// Whether the `)` at `close` closes a `decltype(...)` type: a type position in front of a name.
/// `sizeof(...)` is not a type and neither is a parenthesised expression, so those stay calls.
pub(crate) fn decltype_stands_before(tokens: &[CodeToken], close: usize) -> bool {
    let mut depth = 0usize;
    let mut index = close;
    loop {
        match tokens[index].text.as_str() {
            ")" => depth += 1,
            "(" => {
                depth -= 1;
                if depth == 0 {
                    return index > 0 && tokens[index - 1].text == "decltype";
                }
            }
            _ => {}
        }
        if index == 0 {
            return false;
        }
        index -= 1;
    }
}

/// The keywords that can stand where a type name cannot, immediately before a call: a declaration
/// does not start with any of them, and a statement very often does.
pub(crate) const STATEMENT_KEYWORDS: [&str; 32] = [
    "alignof",
    "case",
    "catch",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "delete",
    "do",
    "else",
    "false",
    "for",
    "goto",
    "if",
    "new",
    "noexcept",
    "nullptr",
    "operator",
    "requires",
    "return",
    "sizeof",
    "static_assert",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typename",
    "using",
    "while",
    "namespace",
];

/// The index of the bracket that closes the one at `open`, or None when the tokens are unbalanced.
/// The three bracket kinds share one counter, which is enough for a scanner that only needs the end
/// of a call's argument list.
pub(crate) fn matching_close_bracket(tokens: &[CodeToken], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        match token.text.as_str() {
            "(" | "[" | "{" => depth += 1,
            ")" | "]" | "}" => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// The callee designator and the argument list of the call whose name is at `index`: the index of
/// the designator's first token and the index of the `(` that opens the arguments. None when the
/// name is not called. This is the one place the designator spelling is unwrapped, so a check that
/// needs the whole call (the bare-call rule below, the statement-start rule in the reachability
/// lint, the structural helper check) cannot drift from [`designator_is_a_call`].
///
/// The designator may be dereferenced or have its address taken inside any nesting of parentheses:
/// `(*f)(...)`, `(&f)(...)`, `(*(&f))(...)` and `(&(*f))(...)` all call `f` through that
/// pointer, and nvcc compiles them all to the same call. The round-8 review compiled
/// `(*(&__threadfence))();` to the baseline cubin byte for byte while both checks rejected it, so
/// the unwrap walks the whole wrapper rather than one step.
pub(crate) fn call_argument_list(tokens: &[CodeToken], index: usize) -> Option<(usize, usize)> {
    // Forward: the wrapper's closing parentheses, then the argument list's `(`.
    let mut cursor = index + 1;
    let mut closes = 0usize;
    while tokens.get(cursor).is_some_and(|next| next.text == ")") {
        closes += 1;
        cursor += 1;
    }
    if !tokens.get(cursor).is_some_and(|next| next.text == "(") {
        return None;
    }
    let open = cursor;
    // Backward: the wrapper's parentheses, `*` and `&`, in any order and nesting.
    let mut needed = closes;
    let mut start = index;
    while needed > 0 {
        let previous = start.checked_sub(1)?;
        match tokens[previous].text.as_str() {
            "(" => needed -= 1,
            ")" => needed += 1,
            "*" | "&" => {}
            _ => return None,
        }
        start = previous;
    }
    Some((start, open))
}

/// Whether the name at `index` is a call with an empty argument list that ends the statement, in
/// every designator spelling: `f();`, `(f)();`, `((f))();`, `(*f)();`. This is the shape of the
/// fence itself - `__threadfence` takes no arguments - and the round-7 review found that both places
/// that used to spell it out as `name ( ) ;` rejected `(__threadfence)();`, which nvcc accepts and
/// compiles to the same PTX membar as the bare spelling. Rejecting valid C++ makes the guard a hazard
/// to the developer debugging it, so the spelling is accepted rather than documented as a limit.
pub(crate) fn is_bare_fence_call(tokens: &[CodeToken], index: usize) -> bool {
    let Some((_, open)) = call_argument_list(tokens, index) else {
        return false;
    };
    tokens.get(open + 1).is_some_and(|next| next.text == ")")
        && tokens.get(open + 2).is_some_and(|next| next.text == ";")
}

/// The index of the `__threadfence` name of a designator that starts at `index`, or None. The `&`
/// or `*` that takes the address of or dereferences the callee is PART OF THE DESIGNATOR, so
/// `(&__threadfence)();` and `(*__threadfence)();` are ordinary calls and not the short-circuit or
/// bitwise operator the control list rejects: both checks look past the designator instead of
/// refusing valid C++.
///
/// The wrapper is walked as a whole, not one step at a time, so `(*(&__threadfence))();` and
/// `(&(*__threadfence))();` are looked past as well. A genuine operator is still an operator: the
/// run stops at any other token, and a call the run reaches but that does not start its statement is
/// rejected by the statement-start rule (`x & (__threadfence)()` stays rejected).
pub(crate) fn fence_designator_starting_at(tokens: &[CodeToken], index: usize) -> Option<usize> {
    let mut cursor = index;
    while tokens
        .get(cursor)
        .is_some_and(|token| matches!(token.text.as_str(), "(" | "*" | "&"))
    {
        cursor += 1;
    }
    tokens
        .get(cursor)
        .is_some_and(|token| token.text == "__threadfence")
        .then_some(cursor)
}

/// The sentence every shape rejection ends with. The guard is a canonical-form checker rather than a
/// proof (see the module documentation): when it does not recognise a spelling it says so and names
/// the form to write, so a rejection of an exotic but valid spelling is a policy decision the
/// developer resolves by writing the canonical spelling instead of a bug to be worked around.
pub(crate) const CANONICAL_HINT: &str = concat!(
    "write the acquire the canonical way - a bare __threadfence(); statement after the poll - and ",
    "see the acquire-guard documentation for the complete list of spellings this check accepts"
);

/// The one constructor of a rejection. Every rejection this module builds goes through it, so the
/// canonical hint is attached by construction instead of by remembering: it is appended when the
/// message does not already end with it, and a debug build refuses a message that carries the hint
/// anywhere but at its end.
///
/// Round-14 A-2: the source scan beside the tests used to search an `Err(...)` span for the hint,
/// and four physical mutations escaped it (a rejection built by `ok_or_else`, a helper whose
/// if-branch carried the hint and whose else-branch did not, a hint followed by more message, and a
/// hint buried inside an if/else). With one constructor the hint cannot be forgotten by
/// construction, and the scan only has to check that every rejection site calls this function.
pub(crate) fn rejection(message: impl Into<String>) -> String {
    let message = message.into();
    debug_assert!(
        !message.contains(CANONICAL_HINT) || message.ends_with(CANONICAL_HINT),
        "a rejection message must not carry the canonical hint anywhere but at its end"
    );
    if message.ends_with(CANONICAL_HINT) {
        return message;
    }
    format!("{message}; {CANONICAL_HINT}")
}

/// What a using-declaration or using-directive names in front of one of the guarded names. The
/// reason a rejection is honest depends on which one it is, so the message is built from this rather
/// than from a bare `Option<namespace>` (round-11 A5, corrected in round-12 A5: the bare
/// `using __threadfence;` was claimed to still name the barrier, and nvcc refuses it).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum UsingTarget {
    /// `using X::__threadfence;` names a member of X, so an unqualified call can resolve to X's
    /// declaration and the meaning really does change. Measured: nvcc compiled that helper to 70264
    /// bytes / 2 PTB membar lines / sha 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline
    /// (round-10 A-2).
    Member(String),
    /// `using namespace X;` where a scanned source defines X with a member carrying a guarded name:
    /// the same re-binding, reached indirectly. Measured: nvcc refuses the ambiguous call the
    /// directive produces ("more than one instance of overloaded function __threadfence matches").
    Directive(String),
    /// `using ::__threadfence;` names the global barrier itself, so the name still denotes the
    /// barrier and nvcc compiles that body to the baseline cubin byte for byte. This one is a POLICY
    /// rejection: a text scanner does not resolve using-declarations.
    Global,
    /// `using __threadfence;` is not namespace-qualified, so it is not valid C++ here at all: nvcc
    /// refuses the translation unit with "namespace-qualified name is required" (round-12 A5). The
    /// shape cannot ship, so refusing it costs nothing, and the message says which spelling it was.
    Unqualified,
    /// `using X::template __threadfence;` names a function template rather than a function: nvcc
    /// refuses it with "no instance of function template X::__threadfence matches the argument list"
    /// (measured). The spelling is not a re-binding, so the message does not claim it is one.
    TemplateQualified(String),
    /// `using X::__threadfence<int>;` is a template-id, which nvcc refuses with "a template-id is not
    /// allowed" (measured). Like the template-qualified form it is a malformed using-declaration
    /// rather than a re-binding, so the message says that instead (round-13 A-3).
    TemplateId,
    /// `using __threadfence = X;` is an alias-declaration, not a using-declaration, and it RE-BINDS
    /// the unqualified name in its scope: nvcc refuses it when the right-hand side is not a type
    /// ("namespace X is not a type name", measured) and compiles the re-binding when it is, where the
    /// barrier is gone - 70264 bytes / 2 PTB membar lines / sha 42454ad1..., the same cubin as the
    /// `using X::__threadfence;` member re-bind, against the 70432 / 4 / 605321041ac5ec1e baseline
    /// (measured, round-13 A-3).
    Alias,
}

/// The rejection a using-declaration or using-directive earns, in one place so the three checks say
/// the same thing. Every case ends with the canonical hint, like every other rejection of a shape
/// this scan cannot certify.
pub(crate) fn using_declaration_rejection(
    position: &str,
    name: &str,
    target: &UsingTarget,
) -> String {
    match target {
        UsingTarget::Member(namespace) => format!(
            "{position}: a using-declaration that brings {namespace}'s {name} into scope changes what the unqualified {name} can denote, so this scan refuses the shape rather than certify it"
        ),
        UsingTarget::Directive(namespace) => format!(
            "{position}: a using-directive for {namespace}, which defines its own {name}, changes what the unqualified {name} can denote, so this scan refuses the shape rather than certify it"
        ),
        UsingTarget::Global => format!(
            "{position}: `using ::{name};` still names the barrier itself and nvcc compiles that body to the baseline cubin, but this scan does not resolve using-declarations, so it refuses the shape rather than certify it. This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::Unqualified => format!(
            "{position}: `using {name};` is not a namespace-qualified using-declaration, so nvcc refuses the translation unit with \"namespace-qualified name is required\" and this scan refuses the shape too. This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::TemplateQualified(namespace) => format!(
            "{position}: `using {namespace}::template {name};` names a function template rather than the barrier, and nvcc refuses it with \"no instance of function template {namespace}::{name} matches the argument list\" (measured). This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::TemplateId => format!(
            "{position}: `using X::{name}<...>;` is a template-id, which nvcc refuses with \"a template-id is not allowed\" (measured). This is a policy rejection, not a claim that {name} changed meaning"
        ),
        UsingTarget::Alias => format!(
            "{position}: `using {name} = ...;` is an alias-declaration that re-binds the unqualified {name} in its scope: nvcc refuses it when the right-hand side is not a type (\"namespace X is not a type name\") and compiles the re-binding when it is, which loses the barrier - 70264 bytes / 2 PTB membar lines / sha 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline (measured). This scan refuses the shape rather than certify it"
        ),
    }
}

/// Whether the text is a C++ identifier, which is what qualifies a designator as a member of a
/// namespace rather than the global name.
pub(crate) fn is_identifier(text: &str) -> bool {
    let mut characters = text.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// The declaration specifiers that make a name a declaration rather than a call. `void abort();`
/// declares a name; `(void)abort()`, `return abort()` and `f(abort())` call one. The difference
/// is the token immediately before the name (round-10 A-4).
pub(crate) const DECLARATION_SPECIFIERS: [&str; 11] = [
    "void", "bool", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "auto",
];

/// Whether the designator at `index` is qualified, so it names a member or another scope's function
/// rather than the library function this guard knows: `s.abort(1u)`, `x->abort()`, `X::_Exit(1)`
/// and `std::abort()` are not the non-returning function, and treating them as it was a false
/// rejection (round-10 A-4). A global `::abort` IS the global function and does count, which is why
/// a doubled colon only qualifies when an identifier or a template's `>` stands in front of it.
pub(crate) fn designator_is_qualified(tokens: &[CodeToken], index: usize) -> bool {
    match tokens
        .get(index.wrapping_sub(1))
        .map(|token| token.text.as_str())
    {
        Some(".") => true,
        Some(">") => tokens
            .get(index.wrapping_sub(2))
            .is_some_and(|token| token.text == "-"),
        Some(":") => {
            tokens
                .get(index.wrapping_sub(2))
                .is_some_and(|token| token.text == ":")
                && tokens
                    .get(index.wrapping_sub(3))
                    .is_some_and(|token| is_identifier(&token.text) || token.text == ">")
        }
        _ => false,
    }
}

/// Whether the name at `index` is the library non-returning function this guard knows, rather than
/// one of the four shapes that only look like it (round-10 A-4).
///
/// `s.abort(1u)` is a member function, `X::_Exit(1)` and `std::abort()` are other scopes',
/// `void abort();` is a declaration and a name this source `#define`s is a macro use, and none of
/// them is the function that never returns. The qualified case the scanner cannot resolve is a stated
/// policy: the documentation lists it, the compiled cubin baseline is what settles it, and the guard
/// is a text scanner rather than a name resolver, so `std::abort()` is deliberately not a blocker -
/// and that costs nothing, because nvcc refuses a host `abort` call in device code ("calling a
/// __host__ function abort from a __device__ function is not allowed"). The conservative reading was
/// refusing real code (a declaration, a member, another scope's function) while the permissive one
/// cannot ship a kernel without its fence.
pub(crate) fn is_the_library_non_returning_call(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
) -> bool {
    if index == 0 {
        return designator_is_a_call(tokens, index);
    }
    !designator_is_qualified(tokens, index)
        && !DECLARATION_SPECIFIERS.contains(&tokens[index - 1].text.as_str())
        && !macros.is_defined(&tokens[index].text)
        && designator_is_a_call(tokens, index)
}

/// Whether the brace that continues the statement in `history` opens a lambda's or a type's body
/// rather than a bare block.
///
/// A bare block runs unconditionally like the statements around it; a lambda's body does not run until
/// the closure is called, and a class, struct, namespace, enum or union body is not on the helper's
/// path at all. The round-10 reviewer compiled `auto l = [] { __threadfence(); }; (void)l;` into
/// 70264 bytes with 2 PTB membar lines (sha 42454ad1...), one barrier short of the 70432 / 4 /
/// 605321041ac5ec1e baseline, while the called version was byte-identical to it - so the fence in the closure
/// is a nested, conditional fence, not the acquire.
///
/// A `]` anywhere in the statement catches every lambda spelling, including `[&] { ... }` and
/// `[]() -> int { ... }`, and a type keyword catches the bodies whose last token before the brace is
/// the type's name rather than a keyword. A statement that reaches neither is the bare block the
/// round-8 review asked for.
pub(crate) fn opens_a_lambda_or_a_type_body(tokens: &[CodeToken], history: &[usize]) -> bool {
    let mut cursor = history.len();
    while cursor > 0 {
        cursor -= 1;
        match tokens[history[cursor]].text.as_str() {
            ";" | "{" | "}" => return false,
            "]" => {
                // An ATTRIBUTE in front of the brace ends in the same ] a capture list does, and it
                // introduces no body: `[[likely]] { __threadfence(); }` runs its fence exactly where
                // the same block without the attribute does, so refusing it because the ] looks like
                // a capture list is the compile-to-the-baseline direction this project calls the
                // worse error (round-35 A3). Step over the whole attribute group and keep looking;
                // anything else that ends in ] is still a capture list or a subscript, which is the
                // lambda body this test exists to find.
                if let Some(opener) = an_attribute_closes_at(tokens, history[cursor]) {
                    while cursor > 0 && history[cursor - 1] >= opener {
                        cursor -= 1;
                    }
                    continue;
                }
                return true;
            }
            "class" | "struct" | "namespace" | "enum" | "union" => return true,
            _ => {}
        }
    }
    false
}

/// Whether `index` sits inside a body the HELPER does not own, at any depth: a LAMBDA's body
/// (`[&]{ ... }` or `[&]<typename T>() -> T { ... }`, round 27 A2, round 28 A2) or a TYPE body
/// (`struct`, `class`, `union` - round 31 A2).
///
/// A `return` inside a lambda exits the LAMBDA, and a `return` inside a member function of a local
/// type exits that MEMBER. Neither is the helper's path - the helper reaches its acquire either way -
/// so every reading that looks for an exit standing on the helper's path skips both. nvcc compiles
/// `struct S { void run() { return; } };` before the poll to the baseline 70432 / 4 / 605321041ac5ec1e with
/// the acquire present, and round 31 A2 found that the class-body spelling was refused with the
/// unreachability sentence, which was false of it.
///
/// EVERY enclosing brace is examined rather than only the innermost one: an exit nested in a block,
/// an `if` or a loop inside the lambda still exits the lambda, and stopping at the innermost brace
/// refused those bodies with "the acquire after it is unreachable" while nvcc compiled them to the
/// baseline (round 28 A2). The walk stops at `floor`, the opening brace of the block the caller is
/// reading - the helper's own body in all three readings - so a lambda or a type defined outside the
/// helper cannot make this true, which is what keeps a bare block at statement level refused (it
/// compiles to 70464 / 2 / 2b56c510b5f4a945, and the reason it is refused with is then true).
///
/// Round 32 A3: the excuse is that the exit leaves the LAMBDA and the helper still reaches its
/// acquire. For a lambda the source does not call that holds, because the body never runs; for a
/// lambda the source CALLS right there it only holds if the body can RETURN.
/// `(void)([&]{ __trap(); return 0u; })();` before the poll is the counter-example: the trap ends
/// the helper, the acquire after it never runs, and nvcc compiles the text to
/// 69536 / 4 / 5826994128b34ed0 (37872 / 4 / 54cda5bc pre-change) - the cubin this file already
/// refuses as `__trap();` before the poll, so
/// the module's one-cubin-one-verdict rule applies: a CALLED lambda body that holds a
/// non-returning call of its own is not a body the helper does not own. A `return` does not make
/// the body non-returning - it returns to the caller - so `(void)([&]{ return; })();` stays
/// accepted, and a lambda nobody calls (`auto f = [&]{ __trap(); }; (void)f;`) stays accepted
/// too, which is what its baseline cubin measures.
pub(crate) fn inside_a_body_the_helper_does_not_own(
    tokens: &[CodeToken],
    index: usize,
    floor: usize,
) -> bool {
    let mut cursor = index;
    while cursor > floor {
        let Some(open) = the_enclosing_brace(tokens, cursor, floor) else {
            return false;
        };
        // Only the precise detectors, not the walk's looser one: an attribute's brackets would
        // make the looser test true for the whole helper.
        if a_lambda_body_opens_at(tokens, open) {
            return !(a_lambda_body_is_called_at(tokens, open)
                && a_called_lambda_body_cannot_return(tokens, open));
        }
        if a_type_body_opens_at(tokens, open) {
            return true;
        }
        // Neither: keep walking outward, because this brace may be a nested block INSIDE a lambda
        // or a type whose opener is further out.
        cursor = open;
    }
    false
}

/// Whether the lambda whose body starts at `open` is CALLED where it stands: `[&]{ ... }()` or
/// `([&]{ ... })()`. The tokenizer emits one token per punctuation character, so the wrap-around
/// form is `}` `)` `(`. Round 32 A3.
pub(crate) fn a_lambda_body_is_called_at(tokens: &[CodeToken], open: usize) -> bool {
    let Some(close) = matching_close_bracket(tokens, open) else {
        return false;
    };
    match tokens.get(close + 1).map(|token| token.text.as_str()) {
        Some("(") => true,
        Some(")") => tokens.get(close + 2).map(|token| token.text.as_str()) == Some("("),
        _ => false,
    }
}

/// Whether the body of a CALLED lambda at `open` cannot return: it calls one of the
/// `NON_RETURNING_CALLS` names on a path that runs when the body runs, so control never comes back
/// to the statement the call stands in and everything after that statement is unreachable (round
/// 32 A3). The same name test the two walks use decides it, so a declaration, a member, another
/// scope's function or a name this source `#define`s is not a non-returning call here either. A
/// trap reached through a macro, a function pointer or an inline-asm block is the same stated limit
/// the name list carries.
///
/// The scan descends through BARE BLOCKS and stops at every other brace: a lambda's or a type's
/// body, whose calls belong to another function, and a control statement's body, whose statements
/// run only if the statement does. `(void)([&]{ { __trap(); } return 0u; })();` before the poll is
/// the round-33 A1 counter-example to the narrower rule round 32 shipped: the brace is a bare
/// block, the trap runs exactly as it does unbraced, and the text is the same program as the
/// unbraced spelling this file already refuses as 69536 / 4 / 5826994128b34ed0 (37872 / 4 /
/// 54cda5bc pre-change). A trap behind a control
/// statement - `[&]{ if (c) { __trap(); } return 0u; }()` - is still not counted, because it is
/// conditional on `c` and the acquire after it still runs.
pub(crate) fn a_called_lambda_body_cannot_return(tokens: &[CodeToken], open: usize) -> bool {
    let Some(close) = matching_close_bracket(tokens, open) else {
        return false;
    };
    let mut macros = MacroScope::from_tokens(tokens);
    // One frame per open brace, innermost last: true where the brace opens a scope whose statements
    // this scan must not read as the lambda's own. `opaque` counts the true ones still open.
    let mut frames: Vec<bool> = Vec::new();
    let mut opaque = 0usize;
    let mut inner = open + 1;
    while inner < close {
        macros.advance_to(inner);
        let text = tokens[inner].text.as_str();
        match text {
            "{" => {
                let hides = !a_bare_block_opens_at(tokens, inner);
                if hides {
                    opaque += 1;
                }
                frames.push(hides);
            }
            "}" => {
                if frames.pop() == Some(true) {
                    opaque -= 1;
                }
            }
            _ => {
                if opaque == 0
                    && NON_RETURNING_CALLS.contains(&text)
                    && is_the_library_non_returning_call(tokens, inner, &macros)
                {
                    return true;
                }
            }
        }
        inner += 1;
    }
    false
}

/// Whether the `{` at `index` opens a BARE BLOCK: a scope that runs where it stands, so the
/// statements in it are on the path the code around them is on (round 33 A1).
///
/// This is the difference the round-32 fix turned on and got half right. `{ __trap(); }` inside an
/// immediately-called lambda runs in the same order as `__trap();` there - the brace adds no
/// condition and no function - so a scan that stops at the brace certifies a text that is the same
/// program as one it already refuses. Three kinds of brace are therefore NOT bare blocks:
///
///   * a lambda's or a type's body, which the precise detectors already identify, because a call in
///     there belongs to another function that may never run;
///   * a control statement's body, whose brace follows `)`, `else`, `do`, `try` or `requires`,
///     because a trap in there is conditional and refusing a conditional trap would refuse a text
///     nvcc compiles to the baseline - the direction this project treats as the worse error;
///   * a lambda's or a type's body behind a capture list or behind attributes, which is what the `]
///     case in those detectors covers - note that the attributes are stepped over first, so an
///     attribute standing on a bare block does not hide that block.
///
/// Everything else is a bare block. The rule is deliberately the narrow one: it descends only where
/// nothing can keep the statements from running, so it closes the false accept without opening a
/// false refusal. The limit that buys is stated rather than hidden: the rule reads braces, not
/// values, so `(void)([&]{ if (1) { __trap(); } return 0u; })();` before the poll stays accepted
/// even though the condition is constantly true, because deciding it needs the condition's value.
/// Round-32's narrower rule was the same kind of limit - it just put the boundary at the wrong
/// brace, refusing nothing and accepting a text whose cubin the file already refuses.
pub(crate) fn a_bare_block_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if a_lambda_body_opens_at(tokens, index) || a_type_body_opens_at(tokens, index) {
        return false;
    }
    // An ATTRIBUTE may stand between a control keyword and the brace of its body, which C++ allows on
    // a selection or iteration statement, as in: if (deadline) [[likely]] { ... }. The brace then
    // follows a ], which is not a control token, so an attribute-blind test read the body as a bare
    // block and refused a conditional trap that nvcc compiles to the baseline - the worse direction,
    // and the one this rule exists to avoid (round-34 A1). Step over every attribute standing there
    // and test the token before the first of them.
    !matches!(
        the_token_before_the_attributes(tokens, index.checked_sub(1))
            .map(|before| tokens[before].text.as_str()),
        Some(")" | "else" | "do" | "try" | "requires")
    )
}

/// The index of the `{` that opens the innermost block holding `index`, or `None` when the walk
/// reaches `floor` first - the caller's own block, which is the helper's body in every reading.
pub(crate) fn the_enclosing_brace(
    tokens: &[CodeToken],
    index: usize,
    floor: usize,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = index;
    while cursor > floor {
        cursor -= 1;
        match tokens[cursor].text.as_str() {
            "}" => depth += 1,
            "{" => {
                if depth == 0 {
                    return Some(cursor);
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    None
}

/// Whether the `{` at `index` opens a TYPE body, `struct S { ... }`, `class X : public A, public B
/// { ... }` or `union U { ... }`. The walk back stops at the first token that ends a declaration or
/// starts another construct, so an ordinary function body, a lambda body, a namespace and a control
/// statement all answer false (round 31 A2).
///
/// Round 32 A2: a balanced bracket group is SKIPPED instead of ending the walk, and a `,` no
/// longer ends it. A base clause (`struct S : A, B`, `class S : public A, public B`) and an
/// attribute specifier (`struct [[maybe_unused]] S`, `struct __attribute__((packed)) S`) both
/// stand between the keyword and the brace in declarations nvcc compiles to the baseline, and
/// stopping at their `,`, `]` or `)` refused those type bodies with the unreachability sentence.
/// The skip is bounded: `matching_open_bracket` answers `None` for an unbalanced group, so a
/// stray closer still ends the walk.
pub(crate) fn a_type_body_opens_at(tokens: &[CodeToken], index: usize) -> bool {
    if tokens.get(index).map(|token| token.text.as_str()) != Some("{") {
        return false;
    }
    let mut cursor = index;
    let mut looked = 0usize;
    while cursor > 0 && looked < 48 {
        cursor -= 1;
        looked += 1;
        match tokens[cursor].text.as_str() {
            "struct" | "class" | "union" => return true,
            ")" | "]" => match matching_open_bracket(tokens, cursor) {
                Some(opener) => cursor = opener,
                None => return false,
            },
            // A declaration, a body, an assignment, a namespace or a control statement:
            // everything before this point belongs to another construct.
            ";" | "{" | "}" | "=" | "namespace" | "enum" | "if" | "else" | "for" | "while"
            | "switch" | "do" | "try" | "catch" | "return" => return false,
            _ => {}
        }
    }
    false
}

/// Whether the operator at `index` is written inside a LAMBDA CAPTURE LIST, `[&]` or `[=]` or
/// `[&x]`, and therefore is not a path-bounding operator at all (round 26 A4). The capture list is
/// the one `[` ... `]` pair directly before a lambda's parameter list or body, so the name after
/// the closing bracket decides it.
pub(crate) fn inside_a_lambda_capture(tokens: &[CodeToken], index: usize) -> bool {
    // `[&]`, `[=]`, `[&, x]`: the operator sits between the capture's brackets, which is the test.
    if tokens
        .get(index.wrapping_sub(1))
        .is_some_and(|token| token.text == "[")
        && tokens
            .get(index + 1)
            .is_some_and(|token| token.text == "]" || is_identifier(&token.text))
    {
        return true;
    }
    let mut cursor = index;
    while cursor > 0 {
        cursor -= 1;
        match tokens[cursor].text.as_str() {
            ";" | "{" | "}" => return false,
            "[" => {
                let Some(close) = matching_close_bracket(tokens, cursor) else {
                    return false;
                };
                if close < index {
                    return false;
                }
                return matches!(
                    tokens.get(close + 1).map(|token| token.text.as_str()),
                    Some("(" | "{" | "mutable" | "noexcept" | "->")
                ) || tokens
                    .get(close + 1)
                    .is_some_and(|token| token.text == "->");
            }
            _ => {}
        }
    }
    false
}

/// Where the statement that starts with the designator begins, or None when the name is not the
/// global barrier at all.
///
/// `::__threadfence` is the global function, so its leading `::` - two `:` tokens to this
/// tokenizer - belongs to the designator and the statement starts before it. `X::__threadfence()`
/// and `x.__threadfence()` name a namespace's or an object's function, which is not this barrier:
/// reading the `::` as a label colon made a labelled statement out of `X::__threadfence();`, and
/// with a namespace member named `__threadfence` nvcc compiled that no-op, dropped two barriers and
/// the check stayed green (round-8 A3). For a qualified name the truthful verdict is that this body
/// contains no barrier, which is what None means to the caller.
pub(crate) fn global_fence_designator(
    tokens: &[CodeToken],
    visited: &[usize],
    designator: usize,
) -> Option<usize> {
    let Some(previous) = code_token_before(visited, designator) else {
        return Some(designator);
    };
    match tokens[previous].text.as_str() {
        // `x.__threadfence` and `x->__threadfence` name another object's function.
        "." => None,
        ">" if code_token_before(visited, previous)
            .is_some_and(|before| tokens[before].text == "-") =>
        {
            None
        }
        ":" => {
            // One colon before the name is a LABEL (`obs: __threadfence();`), which the
            // statement-boundary rule below understands; two are `::`, a scope qualifier.
            let Some(first) = code_token_before(visited, previous) else {
                return Some(designator);
            };
            if tokens[first].text != ":" {
                return Some(designator);
            }
            match code_token_before(visited, first) {
                // A name, a template's `>` or a `)` before the `::` qualifies the designator:
                // only `::__threadfence` with nothing before it is the global one.
                Some(before)
                    if tokens[before].text == ")"
                        || tokens[before].text == ">"
                        || is_identifier(&tokens[before].text) =>
                {
                    None
                }
                _ => Some(first),
            }
        }
        _ => Some(designator),
    }
}

/// The last code token visited before `start`, or None at the start of the block.
pub(crate) fn code_token_before(visited: &[usize], start: usize) -> Option<usize> {
    visited.iter().rev().copied().find(|&code| code < start)
}

/// Whether the colon at `colon` ends a label rather than the false branch of a conditional
/// expression. The two are told apart by the `?` a conditional expression must have at the same
/// parenthesis depth: `ready ? a : b` always has one before its colon, a label never does. Without
/// this, accepting `obs: __threadfence();` would also accept the fence as the false branch of a
/// conditional, which is exactly the conditional acquire the check exists to reject.
///
/// The search stops at the statement's start, because a conditional expression cannot span a `;` or a
/// brace: without that bound a `?` in an earlier statement would make a later label look like a
/// conditional expression. A colon passed on the way consumes one `?`, so `a ? b : c ? d : e` reads
/// correctly from either of its colons.
///
/// A colon that is part of `::` is not a label either: `X::__threadfence();` is a qualified call,
/// and reading its `::` as a label colon made the statement look like a labelled one (round-8 A3).
pub(crate) fn colon_is_a_label(tokens: &[CodeToken], visited: &[usize], colon: usize) -> bool {
    if code_token_before(visited, colon).is_some_and(|before| tokens[before].text == ":")
        || visited
            .iter()
            .copied()
            .find(|&code| code > colon)
            .is_some_and(|after| tokens[after].text == ":")
    {
        return false;
    }
    let mut depth = 0i32;
    let mut colons = 0i32;
    let mut cursor = colon;
    while let Some(previous) = code_token_before(visited, cursor) {
        match tokens[previous].text.as_str() {
            ")" => depth += 1,
            "(" => depth -= 1,
            ";" | "{" | "}" if depth == 0 => return true,
            ":" if depth == 0 => colons += 1,
            "?" if depth == 0 => {
                if colons == 0 {
                    return false;
                }
                colons -= 1;
            }
            _ => {}
        }
        cursor = previous;
    }
    true
}

/// Whether the code token at `start` starts a statement, given the code tokens visited before it.
/// `start` is the first token of a whole designator, so `(__threadfence)();` is decided by what
/// precedes its `(`, not by the `(` itself (round 7).
///
/// A labelled statement starts where its label starts, so `obs: __threadfence();` is accepted: the
/// label does not join the fence to an expression, and nvcc compiles the call unconditionally. The
/// reachability lint and the structural check must agree about that, so both call this function.
/// `case 1: __threadfence();` reaches the statement boundary too, and is then rejected by the
/// `case` token as the control it is.
pub(crate) fn starts_a_statement(tokens: &[CodeToken], visited: &[usize], start: usize) -> bool {
    let Some(mut previous) = code_token_before(visited, start) else {
        return true;
    };
    // An attribute-specifier is not an expression: `[[likely]] __threadfence();`, with the attribute
    // on the same line or the line above, is still the fence's own statement, and nvcc compiles that
    // helper to the baseline cubin byte for byte (round-12 A4). The whole `[[...]]` is walked over
    // before the boundary is decided, and both checks call this function, so they agree about it.
    while tokens[previous].text == "]" {
        let Some(first) = attribute_specifier_start(tokens, visited, previous) else {
            break;
        };
        let Some(before) = code_token_before(visited, first) else {
            return true;
        };
        previous = before;
    }
    match tokens[previous].text.as_str() {
        ";" | "{" | "}" => true,
        ":" if colon_is_a_label(tokens, visited, previous) => {
            // Walk back over the label's own name (and a `case`/`default` keyword, with the
            // constant of a `case`) to the boundary the whole label starts at.
            let mut cursor = previous;
            while let Some(name) = code_token_before(visited, cursor) {
                match tokens[name].text.as_str() {
                    ";" | "{" | "}" | "case" | "default" => return true,
                    _ => cursor = name,
                }
            }
            true
        }
        _ => false,
    }
}

/// The index of the opening `[` of the `[[...]]` attribute-specifier whose closing `]` is at `close`,
/// or None when those brackets are not an attribute.
///
/// An attribute is a double bracket, which is what tells `[[likely]] __threadfence();` from an array
/// subscript or a lambda capture: a single `[` is neither an attribute nor an expression the fence
/// can be part of, so this returns None for it and the statement-start rule keeps its usual answer.
pub(crate) fn attribute_specifier_start(
    tokens: &[CodeToken],
    visited: &[usize],
    close: usize,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = close;
    loop {
        match tokens[cursor].text.as_str() {
            "]" => depth += 1,
            "[" => {
                depth -= 1;
                if depth == 0 {
                    // The opening brackets are adjacent in the token stream: that pair is what makes
                    // this an attribute-specifier rather than `a[i]` or a lambda capture.
                    return tokens
                        .get(cursor + 1)
                        .is_some_and(|next| next.text == "[")
                        .then_some(cursor);
                }
            }
            _ => {}
        }
        cursor = code_token_before(visited, cursor)?;
    }
}

/// The range of a function declarator's parameter list: the indices of its opening `(` and matching
/// `)`, for the name at `index`. None when the name is not directly followed by a parameter list,
/// which is what a call through a parenthesised or dereferenced designator looks like.
pub(crate) fn parameter_list(tokens: &[CodeToken], index: usize) -> Option<(usize, usize)> {
    if tokens.get(index + 1).map(|token| token.text.as_str()) != Some("(") {
        return None;
    }
    let open = index + 1;
    let mut cursor = open;
    let mut depth = 0usize;
    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            "(" => depth += 1,
            ")" => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, cursor));
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

/// Whether the name at `index` sits where a callee designator can sit, which is what separates a call
/// of the acquiring poll from a declaration or a definition of the same name. A declarator is
/// preceded by its return type (`uint32_t`, `void`, `*`); a call is preceded by an operator, a
/// separator, a keyword that starts an expression, another call's parenthesis, or nothing.
pub(crate) fn looks_like_a_callee(tokens: &[CodeToken], index: usize) -> bool {
    match index
        .checked_sub(1)
        .and_then(|previous| tokens.get(previous))
    {
        None => true,
        Some(previous) => matches!(
            previous.text.as_str(),
            "=" | "("
                | ","
                | ";"
                | "{"
                | "}"
                | "return"
                | "?"
                | ":"
                | "&&"
                | "||"
                | "!"
                | ")"
                | "+"
                | "-"
        ),
    }
}

/// One macro definition of a source: its name, where it takes effect, and what its replacement list
/// says.
pub(crate) struct MacroDefinition {
    name: String,
    /// The token index of the `#` that starts the definition. A use before this position is not
    /// affected by it, because the preprocessor is positional.
    position: usize,
    /// The identifiers the replacement list mentions, which is where a chain of names shows up.
    mentions: Vec<String>,
    /// The replacement list's tokens, so a check can ask whether the body IS a name
    /// (`#define POLL_ALIAS spark_wait_for_flag`, round-17 A4) rather than merely mentioning it.
    body: Vec<String>,
    /// Whether the replacement list can end the fall-through path or never return: it carries a control
    /// KEYWORD (`#define BAIL return`) or a non-returning call. Operators are NOT here (round-20 A1),
    /// because a body that carries only an operator does not bound the path by itself.
    bounds_the_path: bool,
    /// Whether the replacement list carries an OPERATOR of the control list (`&&`, `||`, `&`, `|`, `?`)
    /// and no control keyword, so a USE of the macro is a blocker under the same completed-statement
    /// scoping as the literal operator (round-20 A1).
    carries_an_operator: bool,
}

/// The macros a source defines, in definition order, with the ones whose use cannot be trusted kept
/// up to date as a scan advances.
///
/// The preprocessor is positional, and the round-10 review was right to say so: in `int FLAG; ...
/// (void)FLAG; __threadfence(); #define FLAG return` the definition follows the use, so it does not
/// put a `return` in front of the fence; treating every definition as if it stood at the top of the
/// file rejected real code (round-10 A-5). [`MacroScope::advance_to`] therefore publishes a
/// definition only once a scan has passed its position, and the checks ask it about the token they are
/// looking at.
///
/// Suspicion still propagates through the names a replacement list mentions to a fixed point, exactly
/// as before - `#define OUTER INNER` beside `#define INNER if (observed != 0u)` is the round-8 shape,
/// which nvcc compiled to 38624 bytes with 3 PTX membar lines (sha e5b9d6...) (pre-change) while the baseline is
/// 70432 / 4 / 605321041ac5ec1e - but the reverse edges are indexed by name and each one is walked once. The
/// previous implementation re-scanned every body each time it learned a name, so a chain of N
/// definitions cost O(N^2) body scans over a growing name list: the round-10 review measured a
/// 400-link chain at 0.7-1.0 s and a 4000-link chain that did not finish inside 120 s. This one
/// visits each definition once when it takes effect and each edge once when a name turns suspicious,
/// so `a_long_macro_chain_does_not_stall_the_lint` pins a 4000-link chain at well under a second.
///
/// What is still not expanded, and is the honest limit of this reading: a macro defined in the
/// included header (invisible to this source), a keyword assembled by token pasting, and a
/// function-like macro whose ARGUMENT carries the control keyword (`#define WRAP(x) x` with
/// `WRAP(if (c))` is caught by the walk over the use, not by this body reading).
pub(crate) struct MacroScope {
    definitions: Vec<MacroDefinition>,
    /// A name a replacement list mentions, and the definitions that mention it, so marking one name
    /// walks its users exactly once instead of rescanning every body.
    users: std::collections::BTreeMap<String, Vec<usize>>,
    marked: Vec<bool>,
    /// How many definitions have taken effect at the scan's current position.
    in_effect: usize,
    defined: std::collections::BTreeSet<String>,
    suspicious: std::collections::BTreeSet<String>,
}

impl MacroScope {
    /// Reads the `#define`s of a source, in definition order.
    fn new(source: &str) -> Self {
        Self::from_tokens(&code_tokens_of(source))
    }

    /// The same, from tokens a check has already produced: a check that needs both the token stream
    /// and the macro scope (round-15 A-1) must not tokenize the source twice.
    fn from_tokens(tokens: &[CodeToken]) -> Self {
        let definitions = macro_definitions(tokens);
        let mut users: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            for name in &definition.mentions {
                users.entry(name.clone()).or_default().push(index);
            }
        }
        let marked = vec![false; definitions.len()];
        Self {
            definitions,
            users,
            marked,
            in_effect: 0,
            defined: std::collections::BTreeSet::new(),
            suspicious: std::collections::BTreeSet::new(),
        }
    }

    /// Publishes every definition that stands before `position`, and propagates suspicion through
    /// the definitions now in effect.
    fn advance_to(&mut self, position: usize) {
        while self.in_effect < self.definitions.len()
            && self.definitions[self.in_effect].position < position
        {
            let index = self.in_effect;
            self.in_effect += 1;
            let name = self.definitions[index].name.clone();
            self.defined.insert(name.clone());
            if self.definition_is_suspicious(index) {
                self.mark(index);
            } else {
                // A redefinition REPLACES the previous mark (round-15 A3): a name defined again with a
                // harmless body is no longer a macro that can hide control flow or a `return`, so a
                // stale mark must not make it read as one. Marks that other definitions took from the
                // name before the redefinition are not revisited; that is the stated limit of this
                // fixed-point walk, and it errs toward refusing.
                self.suspicious.remove(&name);
            }
        }
    }

    /// Whether a definition in effect is suspicious on its own or through a name it mentions.
    fn definition_is_suspicious(&self, index: usize) -> bool {
        let definition = &self.definitions[index];
        // Round-19 A5: an alias of the poll is not suspicious. Its body may contain a token the control
        // list names - `(&spark_wait_for_flag)` has an `&` - but that is the designator's address-of
        // operator, not a control operator in front of a statement, and nvcc compiles a helper that
        // calls through the alias to the baseline cubin. Marking it suspicious refused a call.
        // Round-20 A1: only a body that can BOUND the path is suspicious. A body that carries only an
        // operator keeps the operator's own rule instead, so `#define PRE (deadline > 0u) && true` before
        // the poll is not a blocker once its own statement has completed.
        (definition.bounds_the_path && !replacement_is_the_designator(&definition.body))
            || definition
                .mentions
                .iter()
                .any(|name| self.suspicious.contains(name))
    }

    /// Marks a definition suspicious, then the definitions that mention its name, once each, through
    /// an explicit worklist rather than a recursion: a chain of thousands of links is a shape the
    /// tests pin, and a recursion of that depth is a stack to fall off.
    ///
    /// Only definitions already in effect are marked: a later definition is judged when it takes
    /// effect, with this name already suspicious, which is what the preprocessor would do at a use
    /// after it.
    fn mark(&mut self, index: usize) {
        let mut queue = vec![index];
        while let Some(current) = queue.pop() {
            if self.marked[current] {
                continue;
            }
            self.marked[current] = true;
            let name = self.definitions[current].name.clone();
            self.suspicious.insert(name.clone());
            let users = self.users.get(&name).cloned().unwrap_or_default();
            for user in users {
                if user < self.in_effect && !self.marked[user] {
                    queue.push(user);
                }
            }
        }
    }

    /// Whether the source defines this name as a macro at the scan's current position.
    fn is_defined(&self, name: &str) -> bool {
        self.defined.contains(name)
    }

    /// Whether a macro of this name takes effect before the scan's current position and cannot be
    /// trusted: its expansion contains a control keyword, a non-returning call, or another such
    /// macro's name.
    fn is_suspicious(&self, name: &str) -> bool {
        self.suspicious.contains(name)
    }

    /// Whether a USE of `name` is an OPERATOR blocker: the definition in effect is a macro whose
    /// replacement list carries a control operator and no control keyword, so the use obeys the
    /// completed-statement scoping of the literal operator (round-20 A1).
    fn carries_only_operators(&self, name: &str) -> bool {
        self.definitions[..self.in_effect]
            .iter()
            .rev()
            .find(|definition| definition.name == name)
            .is_some_and(|definition| definition.carries_an_operator)
    }

    /// The replacement list of the definition of `name` that is in effect at the scan's position -
    /// the last definition of that name that has taken effect, because a redefinition replaces the
    /// one before it. `None` when the name is not in effect as a macro here.
    fn replacement_tokens(&self, name: &str) -> Option<&[String]> {
        self.definitions[..self.in_effect]
            .iter()
            .rev()
            .find(|definition| definition.name == name)
            .map(|definition| definition.body.as_slice())
    }
}

/// The `#define`s of a token stream, in definition order, with the position each one takes effect at
/// and the identifiers its replacement list mentions.
/// Whether `after` starts exactly where `before` ends, with no whitespace between them. The tokenizer
/// drops whitespace, so this is the only way to tell an object-like macro from a function-like one:
/// `#define P(x) ...` is function-like and `#define P (x) ...` is object-like, which is the
/// preprocessor's own rule (C 6.10.3p10) and what round-19 A5 needed to accept a parenthesised alias.
pub(crate) fn follows_without_space(before: &CodeToken, after: &CodeToken) -> bool {
    before.offset + before.text.len() == after.offset
}

pub(crate) fn macro_definitions(tokens: &[CodeToken]) -> Vec<MacroDefinition> {
    let mut definitions = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text != "#" || directive_keyword(tokens, index) != "define" {
            index += 1;
            continue;
        }
        let end = end_of_directive(tokens, index).min(tokens.len());
        if let Some(name) = tokens.get(index + 2) {
            let mut cursor = index + 3;
            // The `(` of a function-like macro has to follow the NAME with no space between them:
            // `#define P (spark_wait_for_flag)` is an OBJECT-like macro whose replacement list is the
            // parenthesised designator. Dropping the space made it function-like with an empty body, so
            // a call through it was not a mention of the poll and the alias was refused (round-19 A5).
            if tokens
                .get(cursor)
                .is_some_and(|token| token.text == "(" && follows_without_space(name, token))
            {
                // A function-like macro's parameter list; the replacement list starts after it.
                let mut depth = 0usize;
                while cursor < end {
                    match tokens[cursor].text.as_str() {
                        "(" => depth += 1,
                        ")" => {
                            depth -= 1;
                            if depth == 0 {
                                cursor += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    cursor += 1;
                }
            }
            let body = &tokens[cursor.min(end)..end];
            definitions.push(MacroDefinition {
                name: name.text.clone(),
                position: index,
                mentions: body
                    .iter()
                    .filter(|token| is_identifier(&token.text))
                    .map(|token| token.text.clone())
                    .collect(),
                body: body.iter().map(|token| token.text.clone()).collect(),
                bounds_the_path: macro_body_bounds_the_path(body),
                carries_an_operator: macro_body_carries_only_operators(body),
            });
        }
        index = end.max(index + 1);
    }
    definitions
}

/// Whether a replacement list is known to end the fall-through path or never return.
///
/// A control keyword is enough, and so is a non-returning call's name. The one recognised no-op
/// spelling is the `do { } while (0)` idiom: it runs its body once and falls through, so a fence
/// after its use still runs, while reading its `do`/`while` as a control token made a harmless
/// `#define abort() do {} while(0)` a false rejection (round-10 A-4). Every other control token
/// counts, including a `do`/`while` whose condition is not the literal zero, which may never
/// terminate.
/// Whether a replacement list NAMES A TYPE rather than control flow: the `?` of
/// `decltype(true ? 0u : 1u)` counted as a control keyword and turned a return-type macro into a macro
/// USE, which made a declaration of the poll look like a call (round-15 A3). Only the `decltype`-headed
/// spelling is exempt; a `?` anywhere else still counts.
pub(crate) fn macro_body_is_a_type(body: &[CodeToken]) -> bool {
    body.first().is_some_and(|token| token.text == "decltype")
        && body.get(1).is_some_and(|token| token.text == "(")
}

/// Whether a replacement list can end the fall-through path or never return, which is what makes a USE
/// of the macro a blocker whatever statement it stands in: a control KEYWORD or a non-returning call.
///
/// Round-20 A1 split this out of the old `macro_body_is_control`, which counted the operators too. A
/// body that carries only an operator does not bound the path by itself - `#define PRE (deadline > 0u)
/// && true` then `const bool ran = PRE;` before the poll compiles to the 70432 / 4 / 605321041ac5ec1e baseline,
/// while `#define BAIL return` then `BAIL;` before the poll compiles to 68576 / 2 / bfd9e8e177f4f76d - not the
/// baseline. The operators keep the rule below.
///
/// The one recognised no-op spelling is the `do { } while (0)` idiom: it runs its body once and falls
/// through, so a fence after its use still runs, while reading its `do`/`while` as a control token made
/// a harmless `#define abort() do {} while(0)` a false rejection (round-10 A-4). Every other control
/// token counts, including a `do`/`while` whose condition is not the literal zero, which may never
/// terminate.
pub(crate) fn macro_body_bounds_the_path(body: &[CodeToken]) -> bool {
    if macro_body_is_a_type(body) {
        return false;
    }
    if body
        .iter()
        .any(|token| NON_RETURNING_CALLS.contains(&token.text.as_str()))
    {
        return true;
    }
    body.iter()
        .any(|token| is_control(&token.text) && !is_control_operator(&token.text))
        && !the_do_while_zero_idiom(body)
}

/// Whether a replacement list carries a control OPERATOR and no control keyword, so a use of the macro
/// is a blocker under the same completed-statement scoping as the literal operator (round-20 A1):
/// `const bool ran = PRE;` before the poll is accepted because that statement completed, while
/// `MAYBE(spark_wait_for_flag(flag, deadline));` is refused because the poll is the macro's argument and
/// no `;` stands between the use and the observation.
pub(crate) fn macro_body_carries_only_operators(body: &[CodeToken]) -> bool {
    if macro_body_is_a_type(body) || macro_body_bounds_the_path(body) {
        return false;
    }
    body.iter().any(|token| is_control_operator(&token.text)) && !the_do_while_zero_idiom(body)
}

/// Whether the replacement list is exactly the `do { } while (0)` no-op, in either spelling of the
/// literal zero.
pub(crate) fn the_do_while_zero_idiom(body: &[CodeToken]) -> bool {
    let texts: Vec<&str> = body
        .iter()
        .map(|token| token.text.as_str())
        .filter(|text| *text != ";")
        .collect();
    matches!(
        texts.as_slice(),
        ["do", "{", "}", "while", "(", "0", ")"]
            | ["do", "{", "}", "while", "(", "0u", ")"]
            | ["do", "{", "}", "while", "(", "false", ")"]
    )
}

/// Whether each token belongs to a preprocessor directive's logical line.
///
/// A macro definition's replacement list can spell the macro's own name (`#define BAIL(x) ...`), so a
/// backward walk over the tokens has to know which of them are directive text and not a use.
pub(crate) fn directive_token_mask(tokens: &[CodeToken]) -> Vec<bool> {
    let mut mask = vec![false; tokens.len()];
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].text == "#" {
            let end = end_of_directive(tokens, index).min(tokens.len());
            for slot in &mut mask[index..end] {
                *slot = true;
            }
            index = end;
            continue;
        }
        index += 1;
    }
    mask
}

/// The name of a suspicious macro used on the same statement as the poll at `start`, if any.
///
/// A poll passed as a macro argument (`BAIL(spark_wait_for_flag(...))`) is expanded inside the macro
/// body, which can put a `return` or an `abort()` in front of the fence; the round-4 A2 shape. The
/// same hazard exists when an object-like macro stands in front of the poll as a control statement
/// (`#define BAIL return` then `BAIL spark_wait_for_flag(...);`). The walk after the poll starts at
/// the poll and never sees the macro name in either case, so the use has to be caught here: scan
/// back to the start of the poll's statement (`;`, `{`, `}` or the start of the source), skipping
/// directive lines so the definition's own spelling of the name is not mistaken for a use. A nested
/// spelling such as `BAIL((poll))` is found because every token of the statement is visited.
pub(crate) fn suspicious_macro_in_poll_statement(
    tokens: &[CodeToken],
    start: usize,
    macros: &MacroScope,
    directive: &[bool],
) -> Option<String> {
    let mut cursor = start;
    while cursor > 0 {
        cursor -= 1;
        if directive.get(cursor).copied().unwrap_or(false) {
            continue;
        }
        match tokens[cursor].text.as_str() {
            ";" | "{" | "}" => return None,
            name if macros.is_suspicious(name) => {
                return Some(name.to_string());
            }
            _ => {}
        }
    }
    None
}

/// A kernel that polls a flag a peer block wrote, and then reads what that flag released, needs the
/// acquire half of the release/acquire pair on the observing path.
///
/// This is the belt and braces beside the guarantee. The guarantee is the set of checks in the
/// module documentation above: [`observing_paths_call_the_acquiring_poll`] (no kernel names the raw
/// poll, in any spelling of the call), [`acquire_helper_fences_unconditionally`] (the helper's own
/// body is exactly one poll and then a fence, at one brace depth, with nothing that can intervene),
/// [`guarded_names_are_not_redefined`] and [`the_acquiring_poll_is_defined_exactly_once`]. Together
/// with the helper's body those make the fence part of the expansion every call site gets, in every
/// spelling of that call, and they name the rewrites that would replace the call instead of merely
/// assuming none exists. What follows is the reachability lint that the earlier rounds grew; it is
/// kept because it is the only check that reads the *text around* a poll and asks whether a fence
/// after it really runs, which is what protects the helper if its body is ever reshaped. It is not
/// the guarantee, and the list at the end of this comment is why: it reasons about source shape, so
/// it has to enumerate the tricks, and an enumeration is only as good as its list.
/// The Milestone 3 review defeated the earlier, line-based version of this guard six ways, and the
/// round-2 and round-3 reviews five more, all of them text tricks that a compiler ignores: a fence
/// hidden in a comment, a poll mention in a comment used to re-anchor the search before the release
/// fence, a fence wrapped in if (false), a fence inside an unrelated branch, an early return, a
/// control statement the line scan did not recognise (switch/case, do/while, a label), a
/// short-circuit && or || whose right operand holds the fence, a fence on a #define continuation
/// line, a fence behind a call that never returns, a keyword or comment opener split by a
/// translation-phase-2 splice, and a fence or a poll spelled inside a raw string literal. Round 4
/// added the horizontal-whitespace form of that splice (which clang and nvcc also make), a
/// function-like macro body, a poll passed as a suspicious macro's argument, and a literal `#if 1`
/// the earlier mask treated as not compiled. So this version works on tokens, after the phase-2
/// splice and comment stripping:
///
///   * the poll is located in CODE that is definitely compiled: a mention inside a comment, on a
///     preprocessor directive line, in a #define body, or in a region that is definitely disabled or
///     whose condition the guard cannot resolve (a literal #if 0, an #ifdef, an arithmetic #if) is
///     not a poll. A literal #if 1 and the #else arm of a literal #if 0 ARE compiled, so a poll
///     there counts. A poll whose text lives only in a macro body is still reported as no poll,
///     because the guard does not expand a macro that has no call site;
///   * the tokeniser makes each string, character and raw string literal one opaque token, so a
///     keyword inside literal content is never a keyword;
///   * the first __threadfence after the poll must be a bare call statement, not part of any larger
///     expression; the designator may be parenthesised, dereferenced or have its address taken in any
///     nesting (`(__threadfence)();`, `(*__threadfence)();`, `(&__threadfence)();`,
///     `(*(&__threadfence))();`), a fence alone in a bare block is that statement, and a label before
///     it is a statement boundary (`obs: __threadfence();`) rather than an expression, as long as the
///     colon is not the false branch of a `? :` and is not half of a `::`. Only the unqualified or
///     globally qualified name is this barrier: `X::__threadfence();` and `x.__threadfence();` name
///     a namespace's or an object's function, so the body is read as having no barrier;
///   * nothing that ends the fall-through path or makes execution conditional may stand between the
///     poll and the fence: if, else, for, while, do, switch, case, default, return, break, continue,
///     goto, throw, co_return, ?, &&, ||, and the single-character & and | (rejected because either
///     half of a short-circuit operator, and the bitwise operator with a statement-expression operand,
///     has the same hazard). The four operators are scoped to the fence's OWN statement: the `?` of a
///     completed `bool ready2 = ready ? true : false;` in front of the fence cannot make it
///     conditional and is accepted, while an operator with no `;` between it and the fence is in the
///     fence's statement and is rejected. The keywords keep the conservative reading at any position,
///     because nothing here bounds a branch or a loop in front of the fence;
///   * a call to a known non-returning function is rejected, since a fence after it cannot run, and
///     the test is whether the designator is CALLED rather than whether the call is a statement, so
///     `(abort)()`, `(__builtin_unreachable(), 0u)` inside a comma expression, an initialiser list
///     and a nested argument are all rejected too;
///   * a use of a macro defined in this source - object-like or function-like, with the parameter
///     list stripped - whose own body contains a control keyword, a non-returning call, or the name
///     of another such macro (read to a fixed point, so `#define OUTER INNER` cannot launder it) is
///     rejected. That covers a poll used on the same statement as such a macro, whether the poll is
///     its argument (`BAIL((poll))`) or follows it as a control statement (`BAIL poll;`), and a use
///     that stands BEFORE the poll anywhere in the poll's own block (round-13 A-2): the expansion can
///     return from the helper before the poll, so the lint reads the block rather than only the text
///     after the poll - and since round 20 A1 the structural check applies the same rule to the same
///     span, so such a text is refused by BOTH in the same words;
///   * a fence inside a preprocessor conditional that may not hold is rejected, while a fence inside
///     a level that is definitely active (a literal #if 1) is accepted; a region that is definitely
///     disabled (#if 0) is ignored rather than obeyed; and a whole logical directive line -
///     continuations included by phase 2 - is skipped instead of being scanned as code. Each of those
///     branches has its own case in `every_rejection_carries_the_canonical_hint`, with a needle only
///     that branch prints (round-14 A-3: the fence-in-a-conditional branch was claimed covered and was
///     not, because two branches shared the wording the case asserted);
///   * a text whose helper body holds more than one compiled poll, or whose helper body holds none
///     while the text defines the helper, is rejected instead of guessed at: the acquire pairs with
///     one observation, and the lint will not decide on the developer's behalf which call that is
///     (round-14 A-1; the two-poll refusal is conservative - a helper body that calls the poll twice
///     is valid C++ and nvcc compiles it, to 39112 bytes / 3 PTX membar lines / sha 07aa2c1d... (pre-change)
///     against the 70432 / 4 / 605321041ac5ec1e baseline, measured - the shipped helper observes once, and a
///     second observation is a change this guard will not certify on the developer's behalf). The
///     poll it follows is the HELPER's own call, not the first call of that name
///     in the text - a forward declaration, a function-pointer parameter and an earlier polling
///     function all name the raw poll without being the observation, and each of those headers
///     compiles to the baseline cubin (70432 / 4 / 605321041ac5ec1e, measured);
///
/// What this does NOT catch, stated so that the claim is no wider than the code: a non-returning
/// callee that is not on NON_RETURNING_CALLS, or one reached through a function pointer, a macro or
/// inline asm; a macro this source does not itself define (the included header's macros are invisible
/// here), or a control keyword assembled by token pasting - the names this source DOES define are
/// followed to a fixed point, so a nested expansion is caught; an unbounded loop before the fence; a
/// computed jump; and any reachability that depends on a value, because this is a check on the shape
/// of the source and not a dataflow analysis. Trigraphs are not a splice in C++17, the
/// level this milestone builds with, so `??/` before a newline is not treated as a comment opener;
/// the backslash-newline form is, and it is valid C++17.
///
/// It is also a guard on the SHAPE of the source in this repository, and it is not a proof of
/// anything about the hardware: only the release/acquire semantics of the fence itself are CUDA's
/// business, and on coherent UMA a reviewer may argue the acquire is unnecessary. The kernel says
/// in a comment that it runs the fence because a conditional one can be wrong while an
/// unconditional one cannot; the guard's job is to keep that claim true as the code changes.
///
/// The raw poll is located by the same rule the structural check uses - the helper's body must mention
/// the designator exactly once, and that mention must be a CALL of it (an argument list follows the
/// name, and nothing in front of it marks the same tokens as a declaration). Neither the arguments nor
/// the expression the call stands in are read, so every call is accepted whatever its prefix and
/// whatever the arguments are spelled like (round 17; the module documentation carries the accepted and
/// refused lists with the cubin each shape compiles to). A mention that is not a call - a declaration,
/// a function-pointer declarator, an attribute between the type and the name - is refused, and the
/// shipped spelling the message names is an illustration rather than a substitution for a helper whose
/// parameters are named differently. The `poll` argument is the label those messages print, not a
/// pattern to match (round-13 A-2).
/// Of those calls this lint follows the one inside the helper's own body, so plumbing that merely
/// names the raw poll cannot hijack it, and a rejection always names a line of that body rather than
/// a line of some other function (round-14 A-1).
///
/// # Errors
/// Returns the reason the fence is not reachable, so a test can assert on the specific evasions.
pub fn acquire_fence_after_poll(source: &str, poll: &str) -> Result<(), String> {
    let tokens = code_tokens_of(source);
    assert!(!poll.is_empty(), "the poll label must not be empty");
    // A using-declaration or using-directive that re-binds one of the guarded names is the same
    // rewrite as a #define, so this check refuses it too: `using X::__threadfence;` makes the bare
    // call below name X's no-op while its text is unchanged (round-10 A-2).
    if let Some((_, line, name, target)) =
        guarded_names_brought_into_scope(&[("this source", source)])
    {
        return Err(rejection(using_declaration_rejection(
            &format!("line {line}"),
            &name,
            &target,
        )));
    }
    // The macros of this source, in definition order. Their uses are rejected below when the
    // definition stands before the use, because the preprocessor is positional (round-10 A-5); a
    // named chain is followed to a fixed point, and a macro defined in the included header is not
    // expanded, which is the stated limit of the guard.
    let mut macros = MacroScope::new(source);
    let active = active_code_mask(&tokens);
    // The poll this lint follows is the HELPER's own observation, not the first raw-poll call in the
    // text. Round-14 A-1: a forward declaration, a function-pointer parameter and an earlier function
    // that polls all name the raw poll without being the observation, and anchoring on the first call
    // made the lint read plumbing and print a line that was not the helper's poll. The rule that
    // LOCATES the raw poll is still the one the structural check uses, so the two cannot diverge on
    // its spelling; this decides which of those calls is the observation.
    let calls: Vec<usize> = raw_poll_call_indices(&tokens)
        .into_iter()
        .filter(|&index| active[index])
        .collect();
    let helper_body = function_body_tokens(&tokens, ACQUIRE_HELPER);
    let start = match helper_body {
        Some((open, close)) => {
            // Inside the helper's own body this lint reads ONE rule (round 17, corrected in round 18):
            // the body must mention the poll exactly once, and that mention must be a CALL of it - the
            // designator followed by an argument list, however the designator is parenthesised,
            // dereferenced or addressed, and whatever expression the call stands in. What is refused
            // is a mention that is NOT a call: a declaration in any return-type spelling, a
            // function-pointer declarator, an attribute between the type and the name. Those bodies
            // omit the call and compile to something other than the baseline (70600 / 3 / 1f2503fd7ecaf189,
            // measured), while every call shape - including the ones an earlier rule refused as
            // "wrapped in a comparison" - is a translation unit whose cubin the module documentation
            // lists.
            let inside = compiled_raw_poll_mentions(&tokens, open, close);
            // Round-23 A3 (HIGH): a poll the preprocessor may not compile is not a poll this lint
            // follows - but it IS a poll the scan cannot place, and the structural check refuses the
            // body for it. Accepting the shape whenever a compiled poll also existed made a SILENT
            // second observation possible: `#if M4_FLAG` around a first poll, then an unconditional
            // poll and the fence, compiles under -DM4_FLAG=1 to two real polls (39072 / 95e0cf41 (pre-change))
            // whose one acquire pairs with the second only, while without the define it is the
            // baseline. Both checks now refuse it, and with the same sentence, so the agreement is
            // whole-message equality rather than a matching verdict.
            if let Some(&unplaced) = raw_poll_mentions(&tokens, open, close)
                .iter()
                .find(|mention| !inside.contains(mention))
            {
                return Err(rejection(format!(
                    "line {}: the poll in {ACQUIRE_HELPER} sits inside a preprocessor conditional, so it is not guaranteed to be compiled",
                    tokens[unplaced].physical_line
                )));
            }
            match inside.len() {
                1 => {
                    if a_void_type_precedes_the_designator(&tokens, inside[0]) {
                        return Err(rejection(void_typed_poll_message(
                            tokens[inside[0]].physical_line,
                        )));
                    }
                    let directives = directive_token_mask(&tokens);
                    let mut at_the_mention = MacroScope::from_tokens(&tokens);
                    at_the_mention.advance_to(inside[0]);
                    if !the_mention_is_a_call_of_the_poll(
                        &tokens,
                        inside[0],
                        &at_the_mention,
                        &directives,
                    ) {
                        return Err(rejection(format!(
                            "line {}: the only mention of {RAW_POLL} in {ACQUIRE_HELPER}'s body is not \
                             a CALL of it, so this lint cannot certify that the fence follows an \
                             observation; the shared header writes the call as \
                             `{CANONICAL_POLL_CALL}` (with the helper's own parameter names - the \
                             arguments are not read), and this scan reads a mention as the observation \
                             when the designator is called and nothing in front of it makes the same \
                             tokens a declaration",
                            tokens[inside[0]].physical_line
                        )));
                    }
                    inside[0]
                }
                0 => {
                    return Err(rejection(format!(
                        "no poll of {poll} in {ACQUIRE_HELPER}'s compiled body (a mention inside a \
                         comment is not a poll, and a poll the preprocessor may not compile is not one \
                         this lint follows)"
                    )));
                }
                count => {
                    return Err(rejection(format!(
                        // Round-29 A2: "this lint" used to stand here where the structural check
                        // says "this scan", which made the COUNT a family the two answered in their
                        // own voices - the module documentation claimed one family while this
                        // sentence acknowledged two. Both now say "this scan", so the two messages
                        // are byte-identical for the same text and the matrix can assert equality.
                        "line {} and line {}: {ACQUIRE_HELPER} mentions the poll {count} times, so \
                         this scan cannot tell which observation the fence pairs with",
                        tokens[inside[0]].physical_line,
                        tokens[inside[count - 1]].physical_line
                    )));
                }
            }
        }
        None => match calls.len() {
            1 => calls[0],
            0 => {
                return Err(rejection(format!(
                    "no poll of {poll} in this source's code (a mention inside a comment is not a \
                     poll)"
                )));
            }
            count => {
                return Err(rejection(format!(
                    "line {} and line {}: this text calls {poll} {count} times and does not define \
                     {ACQUIRE_HELPER}, so this lint cannot tell which call is the observation: run it \
                     on the helper's own text",
                    tokens[calls[0]].physical_line,
                    tokens[calls[count - 1]].physical_line
                )));
            }
        },
    };
    // A poll used with one of those macros is expanded inside it (or follows it as a control
    // statement), so the macro body can put a `return` or an `abort()` in front of the fence while
    // the walk below, which starts after the poll, never sees the macro name.
    let directive_tokens = directive_token_mask(&tokens);
    // Only the definitions in effect at the poll can be expanded by it. The poll wrapped in a
    // suspicious macro keeps its own reason, so this check runs before the pre-poll scan below: the
    // macro-use reason would otherwise shadow it (round-13 D1).
    macros.advance_to(start);
    if let Some(name) =
        suspicious_macro_in_poll_statement(&tokens, start, &macros, &directive_tokens)
    {
        return Err(rejection(format!(
            "line {}: the poll is used with {}, a macro whose body contains a control keyword, a \
             non-returning call, or another such macro, so this scan does not certify that the fence \
             runs after it",
            tokens[start].physical_line, name
        )));
    }
    // The poll's enclosing block, BEFORE the poll. The structural check reads the whole helper body,
    // so it refuses a control token or a non-returning call that stands before the poll; this walk
    // started AT the poll and accepted the same shape, which was the second disagreement in round-12
    // A1. The block is the poll's own, so an unrelated function's control flow is not read.
    let block_start = enclosing_block_start(&tokens, start);
    // The poll's own designator, from its first token to the name: `(poll)`, `(*poll)`, `(&poll)`.
    // Round-18 A1: the `&` of `(&poll)(...)` is part of the call, not a blocker in front of it, and
    // reading it as the bitwise operator refused a call nvcc compiles to the baseline. The fence has
    // had the same unwrap since round 8 (`fence_designator_starting_at`); this is the poll's.
    let designator_start =
        call_argument_list(&tokens, start).map_or(start, |(designator, _)| designator);
    // Round-18 A8: the poll has to be a top-level statement of the helper's own body. The structural
    // check refuses a poll at a nested brace depth (`poll_depth != 0`) and this walk accepted the same
    // text, so the two checks disagreed about a helper whose observation and fence sit inside a block
    // neither can relate to the helper's straight-line path. They answer the same way now, with the
    // structural check's own words.
    if helper_body.is_some_and(|(open, _)| block_start != open) {
        return Err(rejection(format!(
            "line {}: the poll in {ACQUIRE_HELPER} is not a top-level statement of its body, so the \
             helper's expansion is not the fixed poll-then-fence shape",
            tokens[start].physical_line
        )));
    }
    let mut earlier_conditionals = conditional_stack_before(&tokens, block_start);
    let mut earlier = block_start;
    while earlier < start {
        let text = tokens[earlier].text.as_str();
        if text == "#" {
            update_conditionals(&tokens, earlier, &mut earlier_conditionals);
            earlier = end_of_directive(&tokens, earlier);
            continue;
        }
        if definitely_disabled(&earlier_conditionals) {
            earlier += 1;
            continue;
        }
        if earlier >= designator_start && earlier < start {
            earlier += 1;
            continue;
        }
        macros.advance_to(earlier);
        if fence_designator_starting_at(&tokens, earlier).is_none() {
            // A macro whose body can BOUND the path - a control keyword or a non-returning call - is
            // the same blocker before the poll as after it, and BOTH checks read it now: the structural
            // check applies this rule to its own span (round-20 A1), which is why the verdicts agree. It
            // is how `BAIL(<poll>);` is caught: the poll is the macro's ARGUMENT, so the walk from the
            // poll never sees the name. A macro whose body carries only an OPERATOR keeps the operator's
            // own rule instead, in the branch below.
            // Round-28 A5: a macro use inside a LAMBDA's body is not this helper's path either, so
            // the macro branch skips a lambda body exactly as the literal branches do. `#define BAIL
            // return` used inside a lambda before the poll exits the LAMBDA and compiles to the
            // baseline; the macro path is read textually, and the documentation says why.
            if macros.is_suspicious(text)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
            {
                return Err(rejection(format!(
                    "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                     call, or another such macro, so this scan does not certify that the acquire after \
                     it runs",
                    tokens[earlier].physical_line, text
                )));
            }
            // Round-28 A1: the lambda guard the control-token branch below has had since round 27.
            // Without it this branch refused a non-returning call inside a lambda before the poll -
            // `auto f = [&]{ __trap(); }; (void)f;` - which nvcc compiles to the baseline, while the
            // structural check accepted it: a verdict split the lint lost.
            if NON_RETURNING_CALLS.contains(&text)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
                && is_the_library_non_returning_call(&tokens, earlier, &macros)
            {
                return Err(rejection(format!(
                    "line {}: {} does not return, so this scan does not certify that the acquire after it runs",
                    tokens[earlier].physical_line, text
                )));
            }
            // Round-18 A2/A3: a control token that GOVERNS the poll - the condition the poll is, or
            // the conditional expression it stands in - belongs to the poll's own statement, and the
            // fence after that statement still runs unconditionally, so it is not a blocker. This is
            // also what makes this walk agree with the structural check about a conditional
            // expression, which it used to refuse on the `?` alone.
            // Round-19 A1: an operator in an earlier statement that already completed is not a blocker.
            // `const bool ran = (deadline > 0u) && true;` before the poll compiles to the baseline, and
            // the operator cannot bound the poll's own statement because its own statement ended at its
            // semicolon. The structural check applies the same rule to its operator blockers.
            // Round-20 A1: a macro whose body carries only an operator is a blocker with the
            // operator's own rule, so the completed-statement scoping decides it exactly as it does for
            // the literal operator. That keeps `MAYBE(spark_wait_for_flag(flag, deadline));` refused
            // (the poll is the macro's argument and no `;` stands between the use and the observation)
            // while `const bool ran = PRE;` before the poll, whose own statement ended, is not a blocker
            // at all - nvcc compiles that text to the baseline cubin.
            let operator_blocker = (is_control_operator(text)
                || macros.carries_only_operators(text))
                && !inside_a_lambda_capture(&tokens, earlier);
            if (is_control(text) || operator_blocker)
                && !inside_a_lambda_capture(&tokens, earlier)
                && !inside_a_body_the_helper_does_not_own(&tokens, earlier, block_start)
                && !the_control_token_governs_the_poll(&tokens, earlier, start)
                && !(operator_blocker && a_statement_completed_between(&tokens, earlier, start))
            {
                if is_an_exit_keyword(text) {
                    return Err(rejection(format!(
                        "line {}: {:?} stands before the poll, so the acquire after it is unreachable",
                        tokens[earlier].physical_line, text
                    )));
                }
                return Err(rejection(format!(
                    "line {}: {:?} stands before the poll, so the acquire after it may not be reached",
                    tokens[earlier].physical_line, text
                )));
            }
        }
        earlier += 1;
    }

    // The conditional stack enclosing the poll. Recovering it here, rather than starting empty at
    // the poll, keeps the walk from reading the `#else` of an enclosing `#if 1` as unconditional
    // code.
    let mut conditionals = conditional_stack_before(&tokens, start);
    // The code tokens visited so far, in order. A preprocessor directive is not a statement, and
    // neither is a token inside a definitely-disabled region, so neither is recorded: a fence that
    // follows a #endif still follows whatever statement stood before the directive. The indices are
    // kept rather than the last token's text so the fence can look past the parentheses of its own
    // designator for the statement it starts.
    let mut visited_code: Vec<usize> = Vec::new();
    // Whether each open brace encloses a body that is not the helper's own straight-line path: a
    // lambda's or a type's body (round-10 A-3) and any branch or loop body. A fence inside one of
    // them is not the unconditional acquire, however unconditional it looks in isolation: the closure
    // of `auto l = [] { __threadfence(); };` runs only if it is called.
    let mut nested_bodies: Vec<bool> = Vec::new();
    // The first fence this walk saw inside a nested body, recorded rather than reported. The
    // structural check records the same thing and only reports it when the body has NO top-level
    // fence, and the two checks have to agree: `auto l = [] { __threadfence(); }; (void)l;
    // __threadfence();` is the baseline cubin byte for byte (70432 / 4 / 605321041ac5ec1e) and this walk used
    // to reject it on the lambda's fence before ever reaching the real acquire (round-12 A1).
    let mut nested_fence: Option<usize> = None;
    // Round-25 A2: a fence the preprocessor may not compile is a FALLBACK candidate, not an immediate
    // refusal. `poll; #if M4_FLAG __threadfence(); #endif __threadfence(); return observed;` has the
    // acquire - the unconditional fence - and preprocesses without the define to the shipped baseline,
    // so returning at the conditional one refused a body whose acquire is present. It is reported only
    // when no unconditional compiled fence followed it, exactly as the structural check does it.
    let mut conditional_fence: Option<usize> = None;
    // The first control token or non-returning call that stands between the poll and the fence,
    // recorded rather than reported at once. The structural check evaluates the fence first and only
    // then the blockers, so a body with no top-level fence is reported as exactly that; reporting the
    // blocker here as soon as it appeared made the two checks answer with different reasons for the
    // same body (round-12 A1). It is still an error - when a top-level fence is found, any recorded
    // blocker is between the poll and the acquire and is reported before the walk returns.
    let mut blockers: Vec<(usize, bool, bool, usize)> = Vec::new();
    // Every `;` with the brace depth it sits at. The structural check keeps exactly these two lists
    // and decides which blocker is relevant with them; the lint keeping them too is what makes the two
    // checks answer the same way about an operator in an earlier, completed statement (round-12 A1)
    // instead of the lint reading the fence inside a lambda as the operator's own statement.
    let mut semicolons: Vec<(usize, usize)> = Vec::new();
    // Round-24 A1: the walk starts after the poll's argument list - a fence written there runs BEFORE
    // the call and is never the acquire - but the arguments are still READ for control flow. A `return`,
    // a control keyword or a call that never returns inside them ends the path just as it would in front
    // of the poll: `spark_wait_for_flag((__trap(), flag), deadline); __threadfence(); return observed;`
    // compiles to 69536 / 4 / 5826994128b34ed0, byte-identical to the trap-before-the-poll statement body that
    // both checks refuse, and `({ return 0u; flag; })` as the first argument compiles to 70464 / 2 /
    // 2b56c510. Reading the span is what keeps the verdict a function of the compiled shape.
    let argument_span = call_argument_list(&tokens, start)
        .and_then(|(_, open)| matching_close_bracket(&tokens, open).map(|close| (open, close)));
    let mut index = call_argument_list(&tokens, start)
        .and_then(|(_, open)| matching_close_bracket(&tokens, open))
        .map_or(start + 1, |close| close + 1);
    if let Some((open, close)) = argument_span {
        // Round-25 A1/A3: the reading has to mean in the arguments what it means in front of the poll.
        // A macro whose body returns or traps is a blocker here too - the structural check read it and
        // the lint did not, so the two answered the same compiled body differently
        // (`spark_wait_for_flag(({ BAIL; flag; }), deadline);` with `#define BAIL return 0u` is
        // 70464 / 2 / 2b56c510b5f4a945, the cubin of the literal return-in-argument shape both refuse). A
        // LAMBDA body is skipped, because a `return` there exits the lambda (the docs have promised
        // that since round 10, and `spark_wait_for_flag(([&]{ return flag; })(), deadline);` compiles
        // to the baseline with the acquire present), and a bare `if`/`while`/`for`/`switch` is not a
        // blocker by itself: only the exit keyword inside it is, and the same reading finds that.
        let mut inner = open + 1;
        while inner < close {
            let text = tokens[inner].text.as_str();
            // Round-26 A2: a DEFINITELY DISABLED region in the arguments is not read at all, exactly
            // as the walks do not read it. `#if 0 return 0u; #endif` and a `#if 0`-disabled macro use
            // are removed by the compiler, so refusing them refused a body nvcc compiles to the
            // baseline. A conditional the scan cannot resolve is NOT disabled, and a `return` in one
            // still blocks - under the define it really runs.
            if text == "#" {
                update_conditionals(&tokens, inner, &mut conditionals);
                inner = end_of_directive(&tokens, inner);
                continue;
            }
            if definitely_disabled(&conditionals) {
                inner += 1;
                continue;
            }
            macros.advance_to(inner);
            // Round-26 A1: the narrow backward scan misses a TRAILING RETURN TYPE
            // (`[&]() -> const volatile uint32_t * { ... }`), so the walk's own detector decides a
            // body the narrow one does not recognise.
            if text == "{"
                && (a_lambda_body_opens_at(&tokens, inner)
                    || opens_a_lambda_or_a_type_body(
                        &tokens,
                        &(start..inner).collect::<Vec<usize>>(),
                    ))
            {
                inner = matching_close_bracket(&tokens, inner).map_or(inner + 1, |end| end + 1);
                continue;
            }
            let never_returns = NON_RETURNING_CALLS.contains(&text)
                && is_the_library_non_returning_call(&tokens, inner, &macros);
            if never_returns || is_an_argument_exit(text) {
                blockers.push((inner, never_returns, false, 0));
            } else if macros.is_suspicious(text) {
                return Err(rejection(format!(
                    "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                     call, or another such macro, so this scan does not certify that the fence runs after \
                     it",
                    tokens[inner].physical_line, text
                )));
            }
            inner += 1;
        }
    }
    while index < tokens.len() {
        let token = &tokens[index];
        if token.text == "#" {
            update_conditionals(&tokens, index, &mut conditionals);
            // The whole LOGICAL line belongs to the preprocessor, including a #define body or an
            // #else keyword, and none of it is a statement on this path. The backslash-newline
            // splice has already joined a continuation line into this logical line at translation
            // phase 2, so a fence on a continuation line is part of the directive and never runs
            // either: skipping only the line that carries the # is the reviewer's fourth shape.
            index = end_of_directive(&tokens, index);
            continue;
        }
        if definitely_disabled(&conditionals) {
            // Inside #if 0: nothing here executes, so neither a control nor a fence counts.
            index += 1;
            continue;
        }
        // Only the definitions that stand before this token are in effect, as the preprocessor would
        // have it at this point in the file (round-10 A-5).
        macros.advance_to(index);
        if token.text == "{" {
            let lambda_or_type = opens_a_lambda_or_a_type_body(&tokens, &visited_code);
            // A `)` is the end of an if/for/while/switch/catch condition, and `try`, `do` and
            // `requires` name a body directly. The structural check has always treated a `try` body
            // as a nested body while this walk stepped into it and accepted a fence there, which was
            // the one place the two checks disagreed (round-11 A1); `requires` was accepted by both
            // even though nothing inside a requires-expression is executed (round-11 A1).
            // The token read here can be an attribute's closing bracket, as in `try [[likely]] { ... }`
            // or `catch (...) [[likely]] { ... }`, so the read steps over any attribute group first
            // (round-36 A1). Reading the `]]` alone let BOTH checks accept those two spellings, which
            // the plain ones are refused for, and no later blocker rescues them: neither `try` nor
            // `catch` is in [is_control].
            let branch = the_token_before_the_attributes(&tokens, visited_code.last().copied())
                .is_some_and(|previous| {
                    matches!(
                        tokens[previous].text.as_str(),
                        ")" | "try" | "do" | "requires"
                    )
                });
            nested_bodies.push(lambda_or_type || branch);
            visited_code.push(index);
            index += 1;
            continue;
        }
        if token.text == "}" {
            nested_bodies.pop();
            visited_code.push(index);
            index += 1;
            continue;
        }
        if macros.is_suspicious(&token.text)
            && !inside_a_body_the_helper_does_not_own(&tokens, index, block_start)
        {
            return Err(rejection(format!(
                "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                 call, or another such macro, so this scan does not certify that the fence runs after \
                 it",
                token.physical_line, token.text
            )));
        }
        if NON_RETURNING_CALLS.contains(&token.text.as_str())
            && is_the_library_non_returning_call(&tokens, index, &macros)
            // Round-27 A2: a call inside a LAMBDA's body is not this helper's path either, exactly as
            // the structural check now reads it and as the argument-span reading already did.
            && !inside_a_body_the_helper_does_not_own(&tokens, index, block_start)
        {
            // A call that never returns makes everything after it unreachable, so the fence is not
            // an unconditional acquire on this path: the reviewer's fifth shape. The test is the
            // designator being called, NOT the call being a whole statement: `(__trap(), 0u)` is
            // called inside a comma expression, so requiring a `;` after the argument list missed a
            // whole class - the round-8 A1 hole, where nvcc removed two of the four barriers. The
            // verdict waits for the fence, so the reason matches the structural check's reason.
            blockers.push((index, true, false, nested_bodies.len()));
        }
        // The barrier this guard is about is the unqualified or globally qualified function:
        // `X::__threadfence()` and `x.__threadfence()` name a namespace's or an object's function,
        // which is not this barrier at all, and reading their `::` as a label colon is how a
        // namespaced no-op passed for the acquire (round-8 A3). None means "no barrier here", and the
        // loop then reports that no reachable fence follows the poll.
        let fence_designator = if token.text == "__threadfence" {
            let start = call_argument_list(&tokens, index).map_or(index, |(start, _)| start);
            global_fence_designator(&tokens, &visited_code, start)
        } else {
            None
        };
        if let Some(designator) = fence_designator {
            if nested_bodies.iter().any(|&nested| nested) {
                // A lambda's or a type's body (round-10 A-3), or a branch, loop, try block or
                // requires-expression: the fence runs only on some paths, so it is not the acquire
                // the helper owes its caller. It is NOT reported here, though - the walk keeps going
                // so a real top-level fence after it can still be found, which is what the structural
                // check does and what round-12 A1 requires of both.
                nested_fence.get_or_insert(index);
                visited_code.push(index);
                index += 1;
                continue;
            }
            if !is_bare_fence_call(&tokens, index) {
                return Err(rejection(format!(
                    "line {}: the __threadfence is not a bare call statement, so it may be guarded",
                    token.physical_line
                )));
            }
            if !definitely_compiled(&conditionals) {
                // Round-25 A2: record it and keep walking. Reporting here refused a body whose acquire
                // is present: with an unconditional fence later - or earlier, which the structural
                // check already accepted - the conditional one is irrelevant, and without the define
                // the body IS the shipped baseline. The verdict waits for the end of the walk.
                conditional_fence.get_or_insert(token.physical_line);
                visited_code.push(index);
                index += 1;
                continue;
            }
            // The statement this call starts. `__threadfence` and `(__threadfence)` are the same
            // call in C++, so the search starts before the whole designator: looking at the token
            // immediately before the name found a `(` and rejected valid C++ (round 7). A label is a
            // statement boundary as well (round 8) and a global `::` belongs to the designator, and
            // this is the same helper the structural check calls, so the two cannot disagree about a
            // spelling the way they did for `(*f)();`.
            if !starts_a_statement(&tokens, &visited_code, designator) {
                let previous = code_token_before(&visited_code, designator)
                    .map_or("the start of the block", |code| tokens[code].text.as_str());
                return Err(rejection(format!(
                    "line {}: the __threadfence is part of a larger expression that follows \
                     {previous:?}, so \
                     it may not run",
                    token.physical_line
                )));
            }
            // The first blocker that is not an operator in an earlier, completed statement. This is
            // the structural check's `relevant` predicate, on the same lists: the `;` that ends the
            // operator's statement is a `;` at the operator's own depth, so the `;` of `int z = 0;`
            // inside a statement expression or a lambda body does not end it (round-10 A-1).
            let fence_index = index;
            let relevant = |&(blocked, _, operator, blocker_depth): &(usize, bool, bool, usize)| {
                !operator
                    || !semicolons.iter().any(|&(code, semicolon_depth)| {
                        code > blocked && code < fence_index && semicolon_depth == blocker_depth
                    })
            };
            if let Some(&(blocked, never_returns, _, _)) =
                blockers.iter().find(|blocker| relevant(blocker))
            {
                let line = tokens[blocked].physical_line;
                let text = tokens[blocked].text.as_str();
                if never_returns {
                    return Err(rejection(format!(
                        "line {line}: {text} does not return and stands between the poll and the acquire in {ACQUIRE_HELPER}, so this scan does not certify that the acquire runs on every path"
                    )));
                }
                return Err(rejection(format!(
                    "line {line}: {text:?} stands between the poll and the acquire in {ACQUIRE_HELPER}, and this scan reads placement rather than a construct's body, so it does not certify the acquire on every path"
                )));
            }
            return Ok(());
        }
        if is_control(&token.text) {
            // The `&` or `*` of a fence designator is not the operator this rule is about, so
            // `(&__threadfence)();` and `(*__threadfence)();` fall through to the fence below. An
            // operator is recorded like any other blocker and scoped at the fence, by the same rule
            // the structural check uses: an operator with a `;` of its own depth between it and the
            // fence is in an earlier, completed statement, which is why a `&` in a lambda's capture
            // list does not block the real acquire after the lambda (round-12 A1).
            if fence_designator_starting_at(&tokens, index).is_none()
                && !(is_control(&token.text)
                    && inside_a_body_the_helper_does_not_own(&tokens, index, block_start))
            {
                let operator = matches!(token.text.as_str(), "?" | "&&" | "||" | "&" | "|");
                blockers.push((index, false, operator, nested_bodies.len()));
            }
            // A conditional OPERATOR only matters in the fence's own statement: the `?` of a
            // completed `bool b = ready ? x : y;` cannot make a later fence conditional, and
            // rejecting it was a false rejection (round-8 A6), while `ready && __threadfence()`
            // or `ready ? __threadfence() : ...` puts the fence in the same statement and stays
            // rejected. The keywords and the early exits keep the conservative reading, because
            // nothing here bounds a branch or a loop standing in front of the fence. The reason is a
            // limitation, not a claim about the path: `if (observed != 0u) { } __threadfence();`
            // compiles to the baseline cubin, so saying the fence is not guaranteed to run would be
            // false (round-11 A2). It is reported at the fence, or as the missing-fence verdict.
        }
        if token.text == ";" {
            semicolons.push((index, nested_bodies.len()));
        }
        visited_code.push(index);
        index += 1;
    }
    // No top-level fence. A nested one was seen, so the reason names it - the same reason the
    // structural check gives for the same body - and only then the plain "no fence" case.
    if let Some(line) = conditional_fence {
        return Err(rejection(format!(
            "line {line}: the fence sits inside a preprocessor conditional, so it is not guaranteed to run"
        )));
    }
    if let Some(nested) = nested_fence {
        return Err(rejection(format!(
            "line {}: the __threadfence in {ACQUIRE_HELPER} sits at a nested brace depth, inside a body that is not the helper's straight-line path - a branch, a loop, a try block, a requires-expression, a lambda or a type body - so a path through the helper can reach the caller without the fence; the acquire has to be a top-level statement after the poll",
            tokens[nested].physical_line
        )));
    }
    Err(rejection(format!(
        "no reachable __threadfence() follows the poll of {RAW_POLL}, so the observing path has no \
         acquire"
    )))
}

/// The index just after the `{` of the block that encloses the token at `index`, or 0 when the token
/// is not inside a brace.
///
/// The reachability lint reads the poll's OWN block before the poll, so control flow in an unrelated
/// function or block above it is not attributed to the poll (round-12 A1).
pub(crate) fn enclosing_block_start(tokens: &[CodeToken], index: usize) -> usize {
    let mut depth = 0i32;
    let mut cursor = index;
    while let Some(previous) = cursor.checked_sub(1) {
        match tokens[previous].text.as_str() {
            "}" => depth += 1,
            "{" if depth == 0 => return previous + 1,
            "{" => depth -= 1,
            _ => {}
        }
        cursor = previous;
    }
    0
}

/// The name of the helper that carries the acquire with the poll. This is where the guarantee
/// lives: an observing path names one function whose body never changes shape.
pub(crate) const ACQUIRE_HELPER: &str = "spark_wait_for_flag_acquire";
/// The raw poll. It stays available to the publisher side, which needs the bounded wait without an
/// acquire; a bare call to it in a kernel translation unit is exactly what half one below rejects.
pub(crate) const RAW_POLL: &str = "spark_wait_for_flag";

/// The shipped observation, spelled the way the shared header spells it. It is an ILLUSTRATION in the
/// messages and never a substitution: the arguments are not part of the rule (round-17 A2), so a helper
/// whose parameters are named differently is never told to write `flag` and `deadline`.
pub(crate) const CANONICAL_POLL_CALL: &str = "spark_wait_for_flag(flag, deadline)";

/// Whether the mention of the raw poll at `index` is a CALL of it. This is the whole rule inside the
/// helper's body, and it is the narrowest statement the scan can make honestly.
///
/// The scan certifies one thing: the designator is CALLED - an argument list follows it, and nothing
/// marks the same tokens as a declaration of something named like the poll. It does NOT read the
/// arguments (round-17 A2: `spark_wait_for_flag(f, d)` under renamed parameters,
/// `spark_wait_for_flag((flag), (deadline))` and a cast argument are the same call, and each compiles
/// the header to the baseline cubin), and it does NOT read the expression the call stands in
/// (round-17 A1: `!poll(...)`, `0u == poll(...)`, `(void) poll(...)` and a bare call each EXECUTE
/// the poll, and each is listed as accepted with its measured cubin in the module documentation).
///
/// Round 16 read a canonical spelling here instead, and the claim that it did so exactly was false
/// (round-17 root): a scan that starts at the name cannot see the prefix at all, so the prefix is not
/// part of the rule. What remains - is this a call rather than a declaration - is what
/// [`a_type_stands_before`] answers.
pub(crate) fn the_mention_is_a_call_of_the_poll(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
    directives: &[bool],
) -> bool {
    if designator_is_a_call(tokens, index)
        && !a_type_stands_before(tokens, index, macros, directives)
    {
        return true;
    }
    // Round-20 A3: a macro whose replacement list is the whole call `spark_wait_for_flag(flag,
    // deadline)` IS the observation at its own use - `const uint32_t observed = P;` calls the poll - so
    // the mention does not need an argument list of its own. The arguments are not read either way.
    macros
        .replacement_tokens(tokens[index].text.as_str())
        .is_some_and(replacement_is_the_call)
}

/// Whether the token at `index` names the raw poll: the designator itself, or a macro of this source
/// whose replacement list is exactly the designator (one level: `#define POLL_ALIAS
/// spark_wait_for_flag`, round-17 A4). An alias is a normal way to write the call and nvcc compiles it
/// to the baseline cubin, so counting it as a mention keeps the verdict about the text the compiler
/// sees rather than about the spelling the guard prefers.
pub(crate) fn token_names_the_raw_poll(
    tokens: &[CodeToken],
    index: usize,
    macros: &MacroScope,
) -> bool {
    let text = tokens[index].text.as_str();
    text == RAW_POLL
        || macros.replacement_tokens(text).is_some_and(|body| {
            replacement_is_the_designator(body) || replacement_is_the_call(body)
        })
}

/// The index just past a designator of the raw poll that may itself be parenthesised, or [`None`]:
/// `spark_wait_for_flag`, `(spark_wait_for_flag)`, `((spark_wait_for_flag))`. A pair counts only when
/// the designator is the WHOLE of what it wraps, so `(spark_wait_for_flag)(flag)` answers with the
/// index past the `)` and the argument list is what follows it.
pub(crate) fn poll_designator_end(body: &[String]) -> Option<usize> {
    if body.first().is_some_and(|first| first == RAW_POLL) {
        return Some(1);
    }
    if body.len() < 3 || body.first().is_none_or(|first| first != "(") {
        return None;
    }
    let mut depth = 0usize;
    for (position, token) in body.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return (poll_designator_end(&body[1..position]) == Some(position - 1))
                        .then_some(position + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether a macro's replacement list is the WHOLE canonical call of the poll
/// (`#define P spark_wait_for_flag(flag, deadline)`, round-20 A3). A use of it is the observation
/// itself, so `const uint32_t observed = P;` is a call of the poll rather than a mention with no
/// argument list. The arguments are not read, exactly as they are not read for a literal call; the rule
/// is that the replacement starts with the designator and its argument list closes at the end of the
/// replacement list, so the call is the whole of what the macro expands to.
///
/// Round-22 A2: the designator may be parenthesised on its own - `#define P
/// (spark_wait_for_flag)(flag, deadline)` - which is the same call as the bare name and compiles to the
/// same baseline cubin (70432 / 4 / 605321041ac5ec1e, measured). Neither rule read it: the designator rule
/// wants the name to be the WHOLE replacement list, and this rule wanted the bare name to come first,
/// so an ordinary spelling of the call fell between them and was refused.
/// The slice inside one pair of parentheses that wraps the WHOLE slice, or [`None`] when the slice is
/// not exactly such a pair: the first token opens and its matching `)` is the last token. A pair that
/// closes early (`(a) + (b)`) is not a wrapper, and a slice that starts with `)` or never balances is
/// rejected rather than unwrapped.
pub(crate) fn pair_wrapping(body: &[String]) -> Option<&[String]> {
    if body.len() < 3 || body.first().is_none_or(|first| first != "(") {
        return None;
    }
    let mut depth = 0usize;
    for (position, token) in body.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return (position + 1 == body.len()).then_some(&body[1..position]);
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn replacement_is_the_call(body: &[String]) -> bool {
    // Round-21 A3: any number of balanced pairs around the WHOLE replacement list are unwrapped first,
    // the way the designator rule below unwraps them, because
    // `#define P (spark_wait_for_flag(flag, deadline))` then `P;` is the same observation and nvcc
    // compiles it to the baseline cubin. This check refused it until round 21.
    let mut inner: &[String] = body;
    while let Some(unwrapped) = pair_wrapping(inner) {
        inner = unwrapped;
    }
    let Some(designator_end) = poll_designator_end(inner) else {
        return false;
    };
    let rest = &inner[designator_end..];
    if rest.first().is_none_or(|token| token != "(") {
        return false;
    }
    let mut depth = 0usize;
    for (position, token) in rest.iter().enumerate() {
        match token.as_str() {
            "(" => depth += 1,
            ")" => {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
                if depth == 0 {
                    return position == rest.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

/// Whether a macro's replacement list is the raw poll's DESIGNATOR, so a use of the macro is a mention
/// of the poll: the name itself, the name in ANY number of balanced pairs of parentheses
/// (`#define P (spark_wait_for_flag)`, `#define P ((spark_wait_for_flag))`), and any of those
/// dereferenced or addressed inside a pair (`(*spark_wait_for_flag)`, `(&spark_wait_for_flag)`).
///
/// Round-19 A5 introduced this for one pair of parentheses; round-20 A3 unwraps them all, because
/// `(( ... ))` is an ordinary thing to write and nvcc compiles it to the baseline cubin. The `*` or
/// `&` only counts when at least one pair of parentheses was there, because
/// `&spark_wait_for_flag(flag, deadline)` is the address of a call and is not the designator of one.
pub(crate) fn replacement_is_the_designator(body: &[String]) -> bool {
    let mut inner: &[String] = body;
    let mut parenthesised = false;
    while inner.len() >= 3
        && inner.first().is_some_and(|first| first == "(")
        && inner.last().is_some_and(|last| last == ")")
    {
        inner = &inner[1..inner.len() - 1];
        parenthesised = true;
    }
    let named = match inner {
        [wrapper, rest @ ..] if parenthesised && (wrapper == "*" || wrapper == "&") => rest,
        _ => inner,
    };
    named.len() == 1 && named[0] == RAW_POLL
}

/// The mentions of the raw poll inside `[open, close)` that are compiled code: every occurrence of the
/// name that is not a token of a preprocessor line and not inside a region the guard can see is
/// disabled. Comments are not tokens, so a mention in a comment is not one.
///
/// Both checks count mentions rather than calls (round 15): the body must mention the poll exactly
/// once and that mention must be canonical, so a declaration, a function-pointer declarator, an
/// attribute between the type and the name, a macro-defined return type or a wrapped call all count
/// as the one mention and are then refused as shapes the scan does not certify.
pub(crate) fn compiled_raw_poll_mentions(
    tokens: &[CodeToken],
    open: usize,
    close: usize,
) -> Vec<usize> {
    let active = active_code_mask(tokens);
    raw_poll_mentions(tokens, open, close)
        .into_iter()
        .filter(|&index| active[index])
        .collect()
}

/// The mentions of the raw poll inside `[open, close)` that are not in a region the guard can see is
/// disabled: what the STRUCTURAL check counts, so a mention inside a conditional it cannot resolve is
/// still counted and then refused by the conditional rule that names it. The lint reads the narrower
/// set above, which is the documented asymmetry: the structural check refuses that body and says the
/// observation may not be compiled, the lint reads no compiled poll at all.
pub(crate) fn raw_poll_mentions(tokens: &[CodeToken], open: usize, close: usize) -> Vec<usize> {
    let mut conditionals: Vec<Conditional> = conditional_stack_before(tokens, open);
    // The macro scope is advanced with the walk because a mention may be an ALIAS of the designator
    // (round-17 A4) and the preprocessor is positional: `#define POLL_ALIAS spark_wait_for_flag` only
    // counts after its own line.
    let mut macros = MacroScope::from_tokens(tokens);
    let mut mentions = Vec::new();
    let mut index = open;
    while index < close {
        if tokens[index].text == "#" {
            update_conditionals(tokens, index, &mut conditionals);
            index = end_of_directive(tokens, index);
            continue;
        }
        macros.advance_to(index);
        // A region the guard can see is disabled (#if 0) is not compiled code at all, so a mention
        // there is not one; a region it cannot resolve IS counted, and then refused by the caller's
        // own conditional rule, which says so in its message.
        if !definitely_disabled(&conditionals) && token_names_the_raw_poll(tokens, index, &macros) {
            mentions.push(index);
        }
        index += 1;
    }
    mentions
}

/// The structural half of the acquire guarantee: no kernel translation unit calls the raw poll.
///
/// This is the guarantee; [`acquire_fence_after_poll`] is the belt and braces beside it, and the
/// difference between them is the point. The reachability lint asks "is there a fence after this
/// poll, in this text", which is a question a call site can answer wrongly - and did, once per
/// review round - because when the fence is written at the call site the answer depends on the call
/// site's shape. This check removes the question: the observing path must name
/// `spark_wait_for_flag_acquire`, whose body is checked by
/// [`acquire_helper_fences_unconditionally`], so the fence is part of the expansion the call site
/// gets, in every spelling of that call - plain, parenthesised or dereferenced. Replacing the call
/// itself (a `#define` of the helper's name, or a second overload that wins overload resolution) is
/// what [`guarded_names_are_not_redefined`] and [`the_acquiring_poll_is_defined_exactly_once`]
/// reject; a text scan cannot see past a macro it does not read, and the module documentation
/// enumerates what is left.
///
/// The rule is deliberately about kernel translation units and not about the header: the raw poll is
/// correct on the publisher side, which only needs the wait.
///
/// A DECLARATION of the raw poll is accepted and ignored, because a declaration is not an observation
/// and cannot change what any body does: a plain prototype, a function-pointer declaration and one with
/// a `noexcept` specifier all pass - the first two shapes and the specifier are text nvcc accepts, and
/// round-15 A6 found the function-pointer and `noexcept` spellings reported as bare calls. What is
/// rejected is a CALL of the raw poll - and a definition of it would be one too, which
/// [`guarded_names_are_not_redefined`] refuses separately. A call through a function pointer,
/// `(*name)(args)`, is token for token the function-pointer declaration, so this scan cannot separate
/// the two; the module documentation states that limit, and the guarantee it could weaken is carried by
/// the helper's own body.
///
/// # Errors
/// Returns the file and line of the first bare poll.
pub fn observing_paths_call_the_acquiring_poll(kernels: &[(&str, &str)]) -> Result<(), String> {
    for (name, source) in kernels {
        let tokens = code_tokens_of(source);
        let directives = directive_token_mask(&tokens);
        let mut macros = MacroScope::from_tokens(&tokens);
        for (index, token) in tokens.iter().enumerate() {
            macros.advance_to(index);
            if token.text == RAW_POLL
                && designator_is_a_call(&tokens, index)
                && !is_a_declaration_of(&tokens, index, &macros, &directives)
            {
                return Err(rejection(format!(
                    "{name}:{}: a bare {RAW_POLL} call is the raw poll; an observing path must call \
                     {ACQUIRE_HELPER}, whose body carries the acquire fence in the same expansion; \
                    ",
                    token.physical_line
                )));
            }
        }
    }
    Ok(())
}

/// The token range of a function body's interior, by brace depth, or None when there is no such
/// definition with a body.
///
/// The definition is the occurrence of the name whose ARGUMENT LIST is followed by the body's `{`;
/// a call of the same name and a prototype are not bodies, and scanning forward from either to the
/// next brace anywhere in the file returned an unrelated range - which made the lint look for the
/// poll outside the helper it had found (round-14 A-1).
pub(crate) fn function_body_tokens(tokens: &[CodeToken], name: &str) -> Option<(usize, usize)> {
    for (index, token) in tokens.iter().enumerate() {
        if token.text != name || tokens.get(index + 1).map(|next| next.text.as_str()) != Some("(") {
            continue;
        }
        let mut cursor = index + 1;
        let mut parens = 0usize;
        while cursor < tokens.len() {
            match tokens[cursor].text.as_str() {
                "(" => parens += 1,
                ")" => {
                    parens -= 1;
                    if parens == 0 {
                        break;
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        cursor += 1;
        if cursor >= tokens.len() || tokens[cursor].text != "{" {
            continue;
        }
        let open = cursor;
        let mut depth = 0usize;
        while cursor < tokens.len() {
            match tokens[cursor].text.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((open + 1, cursor));
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        return None;
    }
    None
}

/// The structural half of the acquire guarantee, on the helper's own body: the body must mention the
/// raw poll exactly once, and that mention must be a CALL of it - an argument list follows the
/// designator and no type stands in front of it. The call's arguments are not read, and neither is the
/// expression it stands in (round 17; the module documentation lists what is accepted with each shape's
/// measured cubin). What is refused is a mention that is not a call: a declaration in any return-type
/// spelling, a function-pointer declarator, an attribute between the type and the name. An
/// unconditional `__threadfence();` must follow as a top-level statement of the same body.
///
/// Why this is stronger than the reachability lint: the lint re-derives a fence's reachability from
/// the text around a poll, so it has to enumerate every way a source can make a statement
/// conditional, and its comment has to admit what the enumeration misses. This check reads one body
/// that is written once, so no enumeration of shapes is needed: it refuses a control token or a
/// non-returning call at ANY brace depth between the poll and the fence, and it accepts a fence in a
/// bare block or behind a label, which are unconditional too. The two halves together state the
/// guarantee as: the observing path names the helper, and the helper's expansion is poll then fence
/// with nothing compiled between them that can intervene.
///
/// A definitely-disabled region (#if 0) between the poll and the fence is ignored rather than
/// obeyed, exactly as in the reachability lint, because none of it executes; a fence inside a
/// conditional the guard cannot resolve is rejected, because the compiler may not take it. A macro
/// whose body hides a control keyword IS analysed here too, in the lint's own words (round-20 A1): the
/// round-19 review found that the lint refused `#define BAIL return` then `BAIL;` before the poll while
/// this check accepted it, and nvcc compiles that text to 68576 / 2 / bfd9e8e177f4f76d rather than to the
/// 70432 / 4 / 605321041ac5ec1e baseline, so accepting it was the wrong verdict rather than a stricter one.
///
/// # Errors
/// Returns the line of the defect and the reason, so a test can assert on each mutation.
pub fn acquire_helper_fences_unconditionally(header: &str) -> Result<(), String> {
    let tokens = code_tokens_of(header);
    // A using-declaration that re-binds a guarded name is the same rewrite as a #define, and it has
    // to be refused here because the bare call in this body is the thing it redirects (round-10 A-2):
    // `using X::__threadfence;` beside `namespace X { void __threadfence() {} }` made the acquire
    // call X's no-op, and nvcc compiled that kernel to 70264 bytes / 2 PTB membar lines / sha
    // 42454ad1... against the 70432 / 4 / 605321041ac5ec1e baseline.
    if let Some((_, line, name, target)) =
        guarded_names_brought_into_scope(&[(COMMON_HEADER_NAME, header)])
    {
        return Err(rejection(using_declaration_rejection(
            &format!("line {line}"),
            &name,
            &target,
        )));
    }
    // The header's own macros, so a name the header defines is not read as the library function it
    // shadows (round-10 A-4), and so a use of a header macro with a control body is refused.
    let mut macros = MacroScope::new(header);
    let (open, close) = function_body_tokens(&tokens, ACQUIRE_HELPER).ok_or_else(|| {
        rejection(format!(
            "{ACQUIRE_HELPER} must be defined as a function body in the shared header: the acquiring poll is what makes the fence structural rather than a call-site promise"
        ))
    })?;
    let mut conditionals: Vec<Conditional> = conditional_stack_before(&tokens, open);
    let mut depth = 0usize;
    // Every poll in the body, with its brace depth. The count is part of the check: one observation
    // pairs with one acquire, and recording only the first poll let `poll; fence; poll;` through.
    let mut polls: Vec<(usize, usize)> = Vec::new();
    // The body's mentions of the raw poll, in every shape. The body must mention it exactly once and
    // that mention must be a CALL of it (round 17, corrected in round 18): a designator followed by an
    // argument list, whatever the designator's parentheses and whatever expression the call stands in.
    // A mention that is not a call - a declaration, a function-pointer declarator, an attribute
    // between the type and the name - is refused rather than read as a call, because those bodies
    // compile to something other than the baseline (70600 / 3 / 1f2503fd7ecaf189, measured).
    let mentions = raw_poll_mentions(&tokens, open, close);
    // The fence and the index its own statement starts at (a global `::` belongs to the designator).
    let mut top_fence: Option<(usize, usize)> = None;
    // The code tokens of the body in order. Directives are tokens in this stream but produce no
    // statements, so a fence that follows an #endif on its own line still counts as a bare statement;
    // keeping the visited indices also lets `(__threadfence)();` be read as the call it is rather
    // than as a fence preceded by a `(` (round 7), and lets the statement-boundary rule see a label
    // (round 8).
    let mut code_history: Vec<usize> = Vec::new();
    // For each open brace of the body, whether it opens a BARE block rather than the body of an if,
    // else, for, while, do, switch or try. A fence alone in a bare block runs unconditionally - the
    // round-8 review found `{ __threadfence(); }` rejected with the reason for a branch that was not
    // there - while a fence in a branch or a loop body does not.
    let mut bare_blocks: Vec<bool> = Vec::new();
    let mut nested_fence: Option<usize> = None;
    // A fence the preprocessor may not compile: a candidate only while nothing unconditional has been
    // found after the poll, because a later unconditional fence is the acquire and makes it irrelevant
    // (round-24 A2).
    let mut conditional_fence: Option<(usize, usize)> = None;
    // Everything that can keep the fence from running: the index, whether it never returns, whether it
    // is a conditional OPERATOR (scoped to the fence's own statement below), and the brace depth the
    // blocker sits at, because a `;` nested inside a brace does not end that statement (round-10
    // A-1). Every candidate is kept, because one of them may turn out to be an operator in an earlier,
    // completed statement that must not be reported.
    let mut blockers: Vec<(usize, bool, bool, usize)> = Vec::new();
    // Every USE of a macro that can bound the path before the poll - `#define BAIL return` then `BAIL;`.
    // Round-20 A1: the lint refused these and this check accepted them, which was the wrong verdict: nvcc
    // compiles that text to 68576 / 2 / bfd9e8e177f4f76d rather than to the 70432 / 4 / 605321041ac5ec1e baseline.
    let mut suspicious_macro_uses: Vec<usize> = Vec::new();
    // Every `;` of the body with its brace depth: the separator that ends the fence's statement is a
    // `;` at the same depth as the operator, and the `;` of `int z = 0;` inside a statement
    // expression is at a deeper one.
    let mut semicolons: Vec<(usize, usize)> = Vec::new();
    let mut cursor = open;
    while cursor < close {
        let text = tokens[cursor].text.as_str();
        macros.advance_to(cursor);
        if text == "#" {
            update_conditionals(&tokens, cursor, &mut conditionals);
            cursor = end_of_directive(&tokens, cursor);
            continue;
        }
        if text == "{" {
            // A `)` before the brace is the end of an if/for/while/switch condition and `else`,
            // `do` and `try` name a body directly; anything else opens a bare block, which executes
            // unconditionally like the statements around it. A lambda's or a type's body is not bare
            // either (round-10 A-3): the fence in `auto l = [] { __threadfence(); };` runs only if
            // the closure is called, and nvcc compiled that helper to 70264 bytes with 2 PTB membar
            // lines (sha 42454ad1...) while the called version was byte-identical to the 70432 / 4 /
            // 605321041ac5ec1e baseline.
            // `requires` is in the list because nothing inside a requires-expression is executed,
            // so a fence there is not the acquire even though it sits in braces (round-11 A1).
            // An attribute standing between a head and its brace does not change which body this
            // is, and its closing bracket is not one of the tokens below: the read steps over the group
            // first, so `try [[likely]] { ... }` is still not a bare block (round-36 A1).
            let bare = the_token_before_the_attributes(&tokens, code_history.last().copied())
                .is_none_or(|previous| {
                    !matches!(
                        tokens[previous].text.as_str(),
                        ")" | "else" | "do" | "try" | "requires"
                    )
                })
                && !opens_a_lambda_or_a_type_body(&tokens, &code_history);
            bare_blocks.push(bare);
            depth += 1;
            code_history.push(cursor);
            cursor += 1;
            continue;
        }
        if text == "}" {
            bare_blocks.pop();
            depth = depth.saturating_sub(1);
            code_history.push(cursor);
            cursor += 1;
            continue;
        }
        if definitely_disabled(&conditionals) {
            cursor += 1;
            continue;
        }
        if text == "__threadfence" {
            // Only the unqualified or globally qualified function is the barrier: `X::__threadfence()`
            // and `x.__threadfence()` name a namespace's or an object's function, and reading their
            // `::` as a label colon is how a namespaced no-op passed for the acquire (round-8 A3).
            // For those the branch is skipped and the body is reported as having no barrier, which is
            // the truthful verdict.
            let designator = call_argument_list(&tokens, cursor).map_or(cursor, |(start, _)| start);
            if let Some(designator) = global_fence_designator(&tokens, &code_history, designator) {
                // Round-24 A2: a fence the guard cannot resolve is a FALLBACK candidate, not a refusal.
                // `poll; __threadfence(); #if M4_FLAG __threadfence(); #endif; return observed;`
                // preprocesses without the define to exactly the shipped body - 70432 / 4 / 605321041ac5ec1e,
                // the committed baseline - so the unconditional fence is the acquire and the later
                // conditional one is irrelevant. Refusing on sight made this check refuse a baseline
                // body while the lint accepted it, with a reason that was simply false. This candidate
                // is reported only when no unconditional top-level fence is found after the poll.
                if !definitely_compiled(&conditionals) {
                    conditional_fence.get_or_insert((cursor, designator));
                    code_history.push(cursor);
                    cursor += 1;
                    continue;
                }
                // Every enclosing brace must be a bare block for the fence to run unconditionally: a
                // bare block around the fence is not a branch or a loop, so `{ __threadfence(); }`
                // qualifies, while anything inside `if (...) { ... }` does not.
                //
                // Round-22 A1: the acquire is the first fence AFTER the poll. A helper body may carry a
                // fence BEFORE the poll - it is not the acquire of that observation, and taking it made
                // this check refuse `__threadfence(); spark_wait_for_flag(flag, deadline);
                // __threadfence(); return observed;` with "precedes the poll" while the lint accepted
                // it. The reviewer measured that shape: the extra fence before the poll is the only
                // delta and it is 70448 bytes / 5 PTB membar lines / sha 488031bf, so the acquire after
                // the poll is present and the LINT is right. Fences before the poll are ignored here,
                // exactly as the lint ignores them.
                if bare_blocks.iter().all(|&bare| bare) {
                    if mentions.iter().any(|&mention| mention < cursor) {
                        top_fence.get_or_insert((cursor, designator));
                    }
                } else {
                    nested_fence.get_or_insert(cursor);
                }
                code_history.push(cursor);
                cursor += 1;
                continue;
            }
        }
        if mentions.contains(&cursor) {
            if !definitely_compiled(&conditionals) {
                return Err(rejection(format!(
                    "line {}: the poll in {ACQUIRE_HELPER} sits inside a preprocessor conditional, so it is not guaranteed to be compiled",
                    tokens[cursor].physical_line
                )));
            }
            polls.push((cursor, depth));
            code_history.push(cursor);
            // Round-23 root: the scan resumes after the poll's ARGUMENT LIST, exactly where the lint's
            // walk starts. It used to resume at the designator itself, so a fence inside the arguments -
            // which executes BEFORE the call - was read as the acquire of that observation, and the
            // verdict stopped being a function of the compiled shape: the statement-expression fence
            // `spark_wait_for_flag(({ __threadfence(); flag; }), deadline); return observed;` was
            // ACCEPTED here and refused by the lint, while it compiles to 70280 / 3 / 18b66cc5f312046c, the same
            // cubin as the plain fence-before-the-poll body both checks refuse. With the real acquire
            // after it, `spark_wait_for_flag((__threadfence(), flag), deadline); __threadfence();` was
            // REFUSED here and accepted by the lint, and it compiles to 70448 / 5 / 488031bfabdebb09, the cubin
            // of the accepted fence-before-and-after shape. A fence in the arguments is never the
            // acquire, and nothing inside the arguments stands between the poll and its fence.
            let (after_the_call, argument_span) = call_argument_list(&tokens, cursor)
                .and_then(|(_, open)| {
                    matching_close_bracket(&tokens, open)
                        .map(|close| (close + 1, Some((open, close))))
                })
                .unwrap_or((cursor + 1, None));
            if let Some((open, close)) = argument_span {
                // Round-24 A1: the span is not read for FENCES - a fence written in the arguments runs
                // before the call, so it is never the acquire of that observation - but it IS read for
                // control flow. Round 23 skipped it entirely, which made the scan stop reading a
                // `return`, a control keyword or a non-returning call inside the poll's own arguments:
                // `spark_wait_for_flag((__trap(), flag), deadline); __threadfence(); return observed;`
                // was ACCEPTED although it compiles to 69536 / 4 / 5826994128b34ed0, the cubin of the
                // trap-before-the-poll statement body both checks refuse, and `({ return 0u; flag; })`
                // as the first argument compiles to 70464 / 2 / 2b56c510b5f4a945. The lint reads the same span
                // now, so the two cannot diverge on the shape the compiler sees.
                // Round-25 A3: the reading is the lint's reading. A lambda body is skipped - a
                // `return` there exits the lambda, not the helper, and refusing
                // `spark_wait_for_flag(([&]{ return flag; })(), deadline);` contradicted the
                // documentation, which has promised since round 10 that a lambda's return is not a
                // blocker. A bare `if`/`while`/`for`/`switch` is not a blocker either: only an EXIT
                // (`return`, `goto`, `throw`, `co_return`) or a call that never returns is, so
                // `if (ready) return 0u;` still blocks while `if (0) { }` does not - and each of those
                // bodies compiles to the baseline with the acquire present.
                let mut inner = open + 1;
                while inner < close {
                    let text = tokens[inner].text.as_str();
                    // Round-26 A2: a DEFINITELY DISABLED region in the arguments is not read, exactly
                    // as the walks do not read one; a conditional this scan cannot resolve is not
                    // disabled, so a `return` under `#if M4_FLAG` still blocks - under the define it
                    // really runs and the fence after the poll is really dead.
                    if text == "#" {
                        update_conditionals(&tokens, inner, &mut conditionals);
                        inner = end_of_directive(&tokens, inner);
                        continue;
                    }
                    if definitely_disabled(&conditionals) {
                        inner += 1;
                        continue;
                    }
                    macros.advance_to(inner);
                    // Round-26 A1: the narrow backward scan misses a TRAILING RETURN TYPE, so the
                    // walk's own detector decides a body the narrow one does not recognise.
                    if text == "{"
                        && (a_lambda_body_opens_at(&tokens, inner)
                            || opens_a_lambda_or_a_type_body(
                                &tokens,
                                &(cursor..inner).collect::<Vec<usize>>(),
                            ))
                    {
                        inner =
                            matching_close_bracket(&tokens, inner).map_or(inner + 1, |end| end + 1);
                        continue;
                    }
                    let never_returns = NON_RETURNING_CALLS.contains(&text)
                        && is_the_library_non_returning_call(&tokens, inner, &macros);
                    if never_returns || is_an_argument_exit(text) {
                        blockers.push((inner, never_returns, false, depth));
                    } else if macros.is_suspicious(text) {
                        // Round-25 A4: one sentence, the same one the lint's argument-span reading and
                        // both walks use for this reason.
                        return Err(rejection(format!(
                            "line {}: {} is a macro whose body contains a control keyword, a non-returning \
                             call, or another such macro, so this scan does not certify that the fence runs \
                             after it",
                            tokens[inner].physical_line, text
                        )));
                    }
                    inner += 1;
                }
            }
            cursor = after_the_call;
            continue;
        }
        // Anything that ends the fall-through path, or makes execution conditional, stops the fence
        // from being an unconditional acquire. This is recorded at EVERY brace depth: `{ return 0u;
        // }`, `{ if (...) return 0u; }`, `{ abort(); }`, `{ __trap(); }` and `{ goto done; }`
        // between the poll and the fence all leave the helper without its acquire, and recording only
        // depth 0 was the round-8 A1 hole. A bare block that holds nothing but the fence has no such
        // token, so accepting it costs nothing. Every candidate is kept, because one of them may turn
        // out to be an operator in an earlier statement that must not be reported.
        if fence_designator_starting_at(&tokens, cursor).is_none() {
            // Called, not necessarily a statement of its own: `(__builtin_unreachable(), 0u)` and
            // `f(__trap())` never return either, and the fence after them never runs (round-8 A1).
            // Only the LIBRARY function counts: a member, a qualified name, a declaration and a name
            // this header #defines are not the function that never returns (round-10 A-4).
            let never_returns = NON_RETURNING_CALLS.contains(&text)
                && is_the_library_non_returning_call(&tokens, cursor, &macros);
            // Round-27 A2: an exit inside a LAMBDA's body is not this helper's path - it exits the
            // lambda - so it is not a blocker. The lint's pre-poll reading skips the same bodies;
            // refusing `const auto ran = [&]{ return flag; }();` before the poll refused a body nvcc
            // compiles to the baseline.
            if (never_returns || is_control(text))
                && !inside_a_body_the_helper_does_not_own(&tokens, cursor, open)
            {
                let operator = matches!(text, "?" | "&&" | "||" | "&" | "|");
                blockers.push((cursor, never_returns, operator, depth));
            } else if macros.carries_only_operators(text) {
                // Round-20 A1: a use of a macro whose body carries only an operator is a blocker with the
                // OPERATOR's own rule, so the `;` scoping in `relevant` decides it exactly as it does for
                // the literal operator: `MAYBE(spark_wait_for_flag(flag, deadline));` is refused because
                // the poll is the macro's argument and no `;` stands between the use and the observation,
                // while `const bool ran = PRE;` before the poll is not a blocker once that statement has
                // completed.
                blockers.push((cursor, false, true, depth));
            }
            // The same macro-suspicion rule the lint applies to this span, so the two checks refuse the
            // same texts in the same words (round-20 A1) - including the lambda bodies both skip since
            // round 28 A5, which is what keeps the two verdicts equal.
            if macros.is_suspicious(text)
                && !inside_a_body_the_helper_does_not_own(&tokens, cursor, open)
            {
                suspicious_macro_uses.push(cursor);
            }
        }
        if text == ";" {
            semicolons.push((cursor, depth));
        }
        code_history.push(cursor);
        cursor += 1;
    }
    let Some(&(poll, poll_depth)) = polls.first() else {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} does not mention {RAW_POLL} in its body, so there is no observation for the fence to pair with"
        )));
    };
    // Exactly one observation pairs with the acquire. Two polls with one fence between them is the
    // round-5 E2 shape: the old check recorded only the FIRST poll, so `poll; fence; poll;` was
    // green while the second observation was read with no fence after it.
    // The count comes from the SAME scan the lint counts with (round-23 root): `mentions`, not the
    // walk's `polls`. The walk resumes after the poll's argument list, so in
    // `#define P(spark_wait_for_flag) spark_wait_for_flag` + `P(spark_wait_for_flag)(flag, deadline)`
    // it visits only the macro-name mention and used to report one observation where this scan and the
    // lint both count two spellings - the alias `P` and the designator written as its argument. The
    // verdict was right either way, but the counted number has to be the same number the lint counts.
    if mentions.len() > 1 {
        return Err(rejection(format!(
            "line {} and line {}: {ACQUIRE_HELPER} mentions the poll {} times, so this scan cannot tell which observation the fence pairs with",
            tokens[mentions[0]].physical_line,
            tokens[mentions[1]].physical_line,
            mentions.len()
        )));
    }
    // The one mention has to be a CALL of the poll. What is refused is a mention that is not one - a
    // declaration in any of its return-type spellings, a function-pointer declarator, an attribute
    // between the type and the name - because a name that is only declared is not an observation. The
    // call's arguments and the expression around it are NOT read (round-17 A1/A2): every call executes
    // the poll, and the accepted list in the module documentation carries each shape's measured cubin.
    let directives = directive_token_mask(&tokens);
    let mut at_the_poll = MacroScope::from_tokens(&tokens);
    at_the_poll.advance_to(poll);
    if a_void_type_precedes_the_designator(&tokens, poll) {
        return Err(rejection(void_typed_poll_message(
            tokens[poll].physical_line,
        )));
    }
    if !the_mention_is_a_call_of_the_poll(&tokens, poll, &at_the_poll, &directives) {
        return Err(rejection(format!(
            "line {}: {ACQUIRE_HELPER}'s body mentions {RAW_POLL} in a shape this scan will not certify as a call of it - a name that is only declared is not an observation; the shared header writes the call as `{CANONICAL_POLL_CALL}` (with the helper's own parameter names, since the arguments are not read)",
            tokens[poll].physical_line
        )));
    }
    if poll_depth != 0 {
        return Err(rejection(format!(
            "line {}: the poll in {ACQUIRE_HELPER} is not a top-level statement of its body, so the helper's expansion is not the fixed poll-then-fence shape",
            tokens[poll].physical_line
        )));
    }
    let Some((fence, designator)) = top_fence else {
        if let Some((conditional, _)) = conditional_fence {
            return Err(rejection(format!(
                "line {}: the fence sits inside a preprocessor conditional, so it is not guaranteed to run",
                tokens[conditional].physical_line
            )));
        }
        if let Some(nested) = nested_fence {
            return Err(rejection(format!(
                "line {}: the __threadfence in {ACQUIRE_HELPER} sits at a nested brace depth, inside a body that is not the helper's straight-line path - a branch, a loop, a try block, a requires-expression, a lambda or a type body - so a path through the helper can reach the caller without the fence; the acquire has to be a top-level statement after the poll",
                tokens[nested].physical_line
            )));
        }
        return Err(rejection(format!(
            "no reachable __threadfence() follows the poll of {RAW_POLL}, so the observing path has no acquire"
        )));
    };
    // Round-20 A1 and round-21 A1: a use of a suspicious macro is refused in the LINT's own words,
    // because the lint refuses these texts and this check used to accept some of them. The macro can
    // leave the helper before the acquire runs, and every measured one differs from the 70432 / 4 /
    // 605321041ac5ec1e baseline: `BAIL;` before the poll is 68576 / 2 / bfd9e8e177f4f76d (round 20), and `BAIL;`
    // after it in a later, completed statement is 69072 / 2 / 18f93bdc42897348 (round 21, where this check
    // accepted the text and the lint refused it).
    //
    // The scope is the LINT's, exactly: every use that stands before the FENCE. The lint reads a use
    // before the poll in the poll's block, a use in the poll's own statement, and - in its forward walk
    // from the designator - every use between the poll and the fence. A use after the fence is read by
    // neither check, because the acquire has already run by then. Round 20 scoped this to "before the
    // poll, or in the poll's own statement"; that is what let `poll(...); BAIL; __threadfence();`
    // through while the lint refused it.
    //
    // The words are the lint's own for the position: its pre-poll scan says "the acquire after it
    // runs", its forward walk from the poll says "the fence runs after it". The rule stands BEFORE the
    // fence-placement rules below, because that is where the lint's forward walk meets a use: both then
    // answer a shape like `BAIL, __threadfence();` with the macro sentence rather than with the
    // fence-placement sentence.
    // The lint's poll-statement rule, in the lint's own words (round-22 finding 3). The lint refuses a
    // poll used WITH a suspicious macro - the macro is an argument of the poll, or it stands in the
    // poll's own statement before it - naming that macro and saying the scan does not certify that the
    // fence runs after it. This check answered the same shape with its own prefix and its own tail, so
    // the two messages agreed on the substring and on nothing else; they are whole-message equal now,
    // which is what the matrix asserts. The rule comes first for the lint's own reason (round-13 D1):
    // the macro-use reason would otherwise shadow the poll-statement one.
    if let Some(name) = suspicious_macro_in_poll_statement(&tokens, poll, &at_the_poll, &directives)
    {
        return Err(rejection(format!(
            "line {}: the poll is used with {}, a macro whose body contains a control keyword, a \
             non-returning call, or another such macro, so this scan does not certify that the fence \
             runs after it",
            tokens[poll].physical_line, name
        )));
    }
    if let Some(&index) = suspicious_macro_uses.iter().find(|&&index| index < fence) {
        let tail = if index < poll {
            "the acquire after it runs"
        } else {
            "the fence runs after it"
        };
        return Err(rejection(format!(
            "line {}: {} is a macro whose body contains a control keyword, a non-returning call, or \
             another such macro, so this scan does not certify that {tail}",
            tokens[index].physical_line, tokens[index].text
        )));
    }
    // "The fence follows the poll" is not checked here any more: round-22 A1 made the selection above
    // take only fences that have a poll mention before them, so a fence that precedes the poll is
    // never the candidate and there is nothing left to report about it. What a body with a fence
    // before the poll and none after it gets is the "no reachable __threadfence() follows the poll"
    // refusal below, which is the true reason for that shape.
    if !is_bare_fence_call(&tokens, fence) {
        return Err(rejection(format!(
            "line {}: the __threadfence is not a bare call statement, so it may be guarded",
            tokens[fence].physical_line
        )));
    }
    // The code token before the whole designator, not the raw previous token: a directive is a token
    // in this stream but produces no statement, so a fence on its own line after an #endif is a bare
    // statement like any other. The round-6 limit fixture found that the raw-previous-token version
    // rejected that shape as "part of a larger expression"; round 8 added the label case, and the
    // reachability lint answers it with the same helper so the two checks cannot disagree.
    if !starts_a_statement(&tokens, &code_history, designator) {
        let previous = code_token_before(&code_history, designator)
            .map_or("the start of the block", |code| tokens[code].text.as_str());
        return Err(rejection(format!(
            "line {}: the __threadfence is part of a larger expression that follows {previous:?}, so it may not run",
            tokens[fence].physical_line
        )));
    }
    // A conditional OPERATOR is scoped to the fence's own statement: the `?` of a completed
    // `bool b = ready ? x : y;` in front of a labelled fence cannot make the fence conditional, and
    // reporting it was a false rejection (round-8 A6), while an operator with no `;` between it and
    // the fence is in the fence's own statement and does block. The `;` that ends that statement is a
    // `;` at the operator's own brace depth: the `;` inside a statement expression or a lambda body
    // belongs to the inner statement, and reading it as the end of the operator's statement let
    // `(observed == 0u) && ({ int z = 0; __threadfence(); z; });` through (round-10 A-1, where nvcc
    // dropped a barrier: 38624 bytes / 3 PTB membar lines / sha d64426bf... (pre-change) against the 70432 / 4 /
    // 605321041ac5ec1e baseline). The keywords and the early exits keep the conservative reading, because
    // nothing here bounds a branch or a loop in front of the fence.
    // The poll's own designator, from its first token to the name: `(poll)`, `(*poll)`, `(&poll)` are
    // calls, and the `&` of the last one is not the bitwise operator a blocker scan would read it as
    // (round-18 A1). The lint skips the same span.
    let poll_designator =
        call_argument_list(&tokens, poll).map_or(poll, |(designator, _)| designator);
    let relevant = |&(index, _, operator, blocker_depth): &(usize, bool, bool, usize)| {
        index < fence
            && !(index >= poll_designator && index < poll)
            // Round-18 A2/A3: a control token that OWNS the poll - the `if`, `while`, `switch`, `for`
            // or `catch` whose condition the poll is, or the `?` of the conditional expression the
            // poll stands in - is not a blocker: the fence after that whole statement runs
            // unconditionally. Anything that can take the path away before the poll still counts, and
            // this is what keeps this check's verdict equal to the lint's for `true ? poll(...) : 0u`.
            && !the_control_token_governs_the_poll(&tokens, index, poll)
            // Round-18 A2/A3: a CONTROL token that stands BEFORE the poll is a blocker even when a
            // `;` separates it from the fence, because what it controls is whether the poll runs at
            // all - `if (ready) poll(...);`, `ready && poll(...)` and `return poll(...);` all compile
            // and none of them is the fixed shape. The `;` exclusion is for the operators that merely
            // wrap the call (`!poll(...)`, `0u == poll(...)`), which is why it is written as
            // `operator` rather than "anything". After the poll the `;` rule is untouched: a text
            // like `poll(...); &x;` is still the round-8 case. This makes the structural verdict equal
            // to the lint's for every shape in `measured_texts`, refused and accepted alike.
            && (!operator
                || (index < poll
                    && (is_control_operator(&tokens[index].text)
                        || macros.carries_only_operators(&tokens[index].text))
                    && !inside_a_lambda_capture(&tokens, index)
                    && !a_statement_completed_between(&tokens, index, poll))
                || !semicolons.iter().any(|&(code, semicolon_depth)| {
                    code > index && code < fence && semicolon_depth == blocker_depth
                }))
    };
    if let Some(&(index, never_returns, _, _)) = blockers.iter().find(|blocker| relevant(blocker)) {
        let line = tokens[index].physical_line;
        let text = tokens[index].text.as_str();
        if never_returns {
            if index > poll {
                return Err(rejection(format!(
                    "line {line}: {text} does not return and stands between the poll and the acquire in {ACQUIRE_HELPER}, so this scan does not certify that the acquire runs on every path"
                )));
            }
            return Err(rejection(format!(
                "line {line}: {text} does not return, so this scan does not certify that the acquire after it runs"
            )));
        }
        if index > poll {
            return Err(rejection(format!(
                "line {line}: {text:?} stands between the poll and the acquire in {ACQUIRE_HELPER}, and this scan reads placement rather than a construct's body, so it does not certify the acquire on every path"
            )));
        }
        if is_an_exit_keyword(text) {
            // Round-20 A6: the acquire after a `return` (or a `break`, `goto`, `throw`, ...) is
            // UNREACHABLE, which is not the same fact as "it may not be reached": `return poll(...);` is
            // an unconditional call whose fence is dead code.
            return Err(rejection(format!(
                "line {line}: {text:?} stands before the poll, so the acquire after it is unreachable"
            )));
        }
        return Err(rejection(format!(
            "line {line}: {text:?} stands before the poll, so the acquire after it may not be reached"
        )));
    }
    Ok(())
}

/// A `#define` of one of these three names rewrites the acquire out of the translation unit without
/// touching a single call site. The reviewer's round-5 mutations 1 and 2 are exactly the first and
/// the third of them.
pub(crate) const GUARDED_NAMES: [&str; 3] = [ACQUIRE_HELPER, RAW_POLL, "__threadfence"];

/// The file name of the shared header: the one source that may define the acquiring poll.
pub(crate) const COMMON_HEADER_NAME: &str = "m4_common_sm121.cuh";

/// The first `using` that re-binds one of the guarded names, as `(file, line, name, namespace)`.
///
/// A using-declaration rewrites what an unqualified name denotes exactly as a `#define` does, and
/// the round-10 A-2 hole shows why that matters: with `namespace X { __device__ void
/// __threadfence() {} }` and `using X::__threadfence;` beside the bare call, the helper's
/// `__threadfence();` calls X's no-op, and nvcc compiled the kernel to 70264 bytes with 2 PTB membar
/// lines (sha 42454ad1...) where the baseline is 70432 / 4 / 605321041ac5ec1e. Both checks accepted it,
/// because the text of the call is unchanged.
///
/// A using-DIRECTIVE cannot be resolved by a text scanner in general, so it is caught in the case the
/// scanner can see: `using namespace X;` where a scanned source defines `namespace X` with a member
/// carrying a guarded name. The documentation names the remaining case - a namespace from a header
/// outside the scan - as a limit, alongside the macro that comes from an included header.
pub(crate) fn guarded_names_brought_into_scope(
    sources: &[(&str, &str)],
) -> Option<(String, usize, String, UsingTarget)> {
    let mut namespaces: Vec<(String, String)> = Vec::new();
    for &(_, source) in sources {
        namespaces.extend(namespaces_with_guarded_members(source));
    }
    for &(file, source) in sources {
        let tokens = code_tokens_of(source);
        let mut index = 0usize;
        while index < tokens.len() {
            if tokens[index].text != "using" {
                index += 1;
                continue;
            }
            let mut cursor = index + 1;
            let directive = tokens
                .get(cursor)
                .is_some_and(|token| token.text == "namespace");
            if directive {
                cursor += 1;
            }
            let line = tokens[index].physical_line;
            while cursor < tokens.len() {
                let text = tokens[cursor].text.as_str();
                if matches!(text, ";" | "{" | "}" | "(" | ")") {
                    break;
                }
                if GUARDED_NAMES.contains(&text) {
                    // A using-DECLARATION reports which spelling it is, so the message can tell a
                    // re-bound name (using X::__threadfence;, the round-10 A-2 hole) from the global
                    // barrier itself (using ::__threadfence;, a policy rejection because nvcc still
                    // compiles it to the baseline cubin) from the bare spelling (using
                    // __threadfence;, which nvcc refuses - round-11 A5, corrected in round-12 A5).
                    let target = if directive {
                        UsingTarget::Directive(text.to_string())
                    } else {
                        using_declaration_target(&tokens, cursor)
                    };
                    return Some((file.to_string(), line, text.to_string(), target));
                }
                if let Some((_, member)) = directive
                    .then(|| namespaces.iter().find(|(namespace, _)| namespace == text))
                    .flatten()
                {
                    return Some((
                        file.to_string(),
                        line,
                        member.clone(),
                        UsingTarget::Directive(text.to_string()),
                    ));
                }
                cursor += 1;
            }
            index += 1;
        }
    }
    None
}

/// What a using-declaration names in front of a guarded name at `index`.
///
/// A doubled colon with an identifier in front of it is a member of that namespace (`using
/// X::__threadfence;`), the case where the meaning really changes. A doubled colon with nothing in
/// front of it is the global barrier itself (`using ::__threadfence;`), which nvcc still compiles to
/// the baseline cubin. No colon at all is the bare spelling (`using __threadfence;`), which nvcc
/// refuses outright (round-12 A5). `using` is an identifier to the tokenizer, so it is excluded from
/// the namespace slot: reporting the namespace "using" is the nonsense the first spelling of this
/// helper printed.
pub(crate) fn using_declaration_target(tokens: &[CodeToken], index: usize) -> UsingTarget {
    let after = tokens.get(index + 1).map(|token| token.text.as_str());
    // An alias-declaration is not a using-declaration at all, and it RE-BINDS the name in its scope,
    // so it is classified before the qualified forms (round-13 A-3).
    if after == Some("=") {
        return UsingTarget::Alias;
    }
    // `using X::template __threadfence;` - the qualified name is preceded by the `template`
    // keyword, which makes it a template rather than the barrier. This is checked first because the
    // name's own neighbour is the keyword, not the `::` (round-13 A-3).
    if index >= 4
        && tokens[index - 1].text == "template"
        && tokens[index - 2].text == ":"
        && tokens[index - 3].text == ":"
        && is_identifier(&tokens[index - 4].text)
        && !matches!(tokens[index - 4].text.as_str(), "using" | "namespace")
    {
        return UsingTarget::TemplateQualified(tokens[index - 4].text.clone());
    }
    if index >= 2 && tokens[index - 1].text == ":" && tokens[index - 2].text == ":" {
        if index >= 3
            && is_identifier(&tokens[index - 3].text)
            && !matches!(tokens[index - 3].text.as_str(), "using" | "namespace")
        {
            // `using X::__threadfence<int>;` names a specialization, which is not a valid
            // using-declaration, so it is not the member re-binding the next arm describes.
            if after == Some("<") {
                return UsingTarget::TemplateId;
            }
            return UsingTarget::Member(tokens[index - 3].text.clone());
        }
        if after == Some("<") {
            return UsingTarget::TemplateId;
        }
        return UsingTarget::Global;
    }
    if after == Some("<") {
        return UsingTarget::TemplateId;
    }
    UsingTarget::Unqualified
}

/// The namespaces this source defines whose body mentions one of the guarded names, as
/// `(namespace, guarded name)`. A using-directive for one of them can make the unqualified name
/// resolve to that member, which is the case a text scanner can see.
pub(crate) fn namespaces_with_guarded_members(source: &str) -> Vec<(String, String)> {
    let tokens = code_tokens_of(source);
    let mut found = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.text != "namespace" {
            continue;
        }
        let mut cursor = index + 1;
        // A nested namespace (`namespace a::b`) is read by its last part, and an unnamed namespace
        // has no name to import.
        let mut name: Option<&str> = None;
        while tokens.get(cursor).is_some_and(|next| next.text == ":") {
            cursor += 1;
        }
        if let Some(next) = tokens.get(cursor).filter(|next| is_identifier(&next.text)) {
            name = Some(next.text.as_str());
        }
        let (Some(name), Some(body)) = (name, namespace_body(&tokens, cursor)) else {
            continue;
        };
        if let Some(member) = tokens[body.0..body.1]
            .iter()
            .find(|slot| GUARDED_NAMES.contains(&slot.text.as_str()))
        {
            found.push((name.to_string(), member.text.clone()));
        }
    }
    found
}

/// The token range inside a namespace body at or after `from`, exclusive of the braces.
pub(crate) fn namespace_body(tokens: &[CodeToken], from: usize) -> Option<(usize, usize)> {
    let mut cursor = from;
    while cursor < tokens.len() {
        match tokens[cursor].text.as_str() {
            ";" => return None,
            "{" => {
                let open = cursor;
                let mut depth = 0usize;
                while cursor < tokens.len() {
                    match tokens[cursor].text.as_str() {
                        "{" => depth += 1,
                        "}" => {
                            depth -= 1;
                            if depth == 0 {
                                return Some((open + 1, cursor));
                            }
                        }
                        _ => {}
                    }
                    cursor += 1;
                }
                return None;
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

/// Rejects a `#define` of the acquiring poll, the raw poll or the fence in any scanned source.
///
/// This is the translation-unit half of the guarantee, and it closes a whole class of evasion that no
/// call-site check can see: `#define spark_wait_for_flag_acquire(f, d) (1u)` in a kernel and
/// `#define __threadfence() ((void)0)` above the include each leave every call site looking exactly
/// right, while the kernel reads the published payload with no acquire (the reviewer's round-5
/// mutations 1 and 2, both guard-green before this check existed). A macro *use* is not analysed
/// here - that is the reachability lint's job - but the definition of one of these three names is a
/// failure naming the file and the line.
///
/// A using-declaration or using-directive that brings one of the three names into scope is the same
/// rewrite and is rejected by the same rule (round-10 A-2).
///
/// `#undef` is deliberately not rejected: undefining a name cannot put a fence into code that has
/// none, and the fence this header defines is already inside the helper's body, whose shape is
/// checked separately.
///
/// # Errors
/// Returns the offending `file:line` and the name that was defined.
pub fn guarded_names_are_not_redefined(sources: &[(&str, &str)]) -> Result<(), String> {
    for &(name, source) in sources {
        let tokens = code_tokens_of(source);
        for (index, token) in tokens.iter().enumerate() {
            if token.text != "#" || directive_keyword(&tokens, index) != "define" {
                continue;
            }
            let Some(defined) = tokens.get(index + 2) else {
                continue;
            };
            if GUARDED_NAMES.contains(&defined.text.as_str()) {
                return Err(rejection(format!(
                    "{name}:{}: #define {} rewrites the acquire out of the translation unit; the raw poll, the acquiring poll and the fence are not macro names under kernels/",
                    defined.physical_line, defined.text
                )));
            }
        }
    }
    if let Some((file, line, name, target)) = guarded_names_brought_into_scope(sources) {
        return Err(rejection(using_declaration_rejection(
            &format!("{file}:{line}"),
            &name,
            &target,
        )));
    }
    Ok(())
}

/// Requires exactly one definition of the acquiring poll, and that it is in the shared header.
///
/// The helper is the guarantee, so a second spelling of it is a defect even when the first is
/// intact. An overload whose parameter is `uint32_t *` rather than `const volatile uint32_t *` wins
/// overload resolution for a non-volatile flag - the reviewer's round-5 mutation 3, whose selected
/// callee clang -S confirmed - and can carry a body with no fence; a redeclaration that changes the
/// signature does the same from a distance. A definition is recognised by the body brace that follows
/// its parameter list, and a parameter list followed by a semicolon is a declaration unless it sits
/// where a callee normally sits.
///
/// # Errors
/// Returns every offending `file:line`, and why it is one.
pub fn the_acquiring_poll_is_defined_exactly_once(sources: &[(&str, &str)]) -> Result<(), String> {
    let mut definitions: Vec<(&str, usize)> = Vec::new();
    for &(name, source) in sources {
        let tokens = code_tokens_of(source);
        for (index, token) in tokens.iter().enumerate() {
            if token.text != ACQUIRE_HELPER {
                continue;
            }
            let Some((_, close)) = parameter_list(&tokens, index) else {
                continue;
            };
            match tokens.get(close + 1).map(|next| next.text.as_str()) {
                Some("{") => definitions.push((name, token.physical_line)),
                Some(";") if !looks_like_a_callee(&tokens, index) => {
                    return Err(rejection(format!(
                        "{name}:{}: a declaration of {ACQUIRE_HELPER} outside the shared header can change which definition a call binds to; the helper is defined once, in {COMMON_HEADER_NAME}",
                        token.physical_line
                    )));
                }
                _ => {}
            }
        }
    }
    if definitions.len() != 1 {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} must be defined exactly once, in the shared header; found {} definition(s): {}",
            definitions.len(),
            definitions
                .iter()
                .map(|(name, line)| format!("{name}:{line}"))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if definitions[0].0 != COMMON_HEADER_NAME {
        return Err(rejection(format!(
            "{ACQUIRE_HELPER} is defined in {}, not in {COMMON_HEADER_NAME}",
            definitions[0].0
        )));
    }
    Ok(())
}
