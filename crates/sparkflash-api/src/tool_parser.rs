//! vLLM's `deepseek_v41` tool-call parser, generation side.
//!
//! The reference encoder renders tools into the V4.1 DSML prompt block
//! (`reference/encoding/encoding.py`: the block is wrapped in
//! `<｜DSML｜ calls>` with `<｜DSML｜ invoke name="...">` and
//! `<｜DSML｜ parameter name="..." string="true|false">` children), so the model answers in the
//! same format. vLLM runs that answer through `--tool-call-parser deepseek_v41` before it fills
//! the OpenAI `tool_calls` array; this module ports the generation half of the reference decoder
//! (`parse_message_from_completion_text` / `parse_tool_calls` in `reference/encoding/encoding.py`)
//! to the text the driver emits.
//!
//! Both shapes are parsed. [`parse_completion`] is the buffered (non-streaming) form that lifts the
//! whole block out of a complete answer. [`ToolCallStreamParser`] is the incremental streaming
//! form: it consumes content deltas as the engine emits them, suppresses the DSML text, and
//! produces the same `tool_calls` payload fragment by fragment. A chunk may split any token, JSON
//! escape or closing tag; the parser holds back only the bytes that could still become a marker.
//!
//! The two shapes share one contract: the visible content, the function names and the arguments
//! string concatenated from the streamed fragments are exactly the buffered payload (pinned by the
//! split-at-every-boundary tests below). A malformed or unterminated block is a typed
//! [`StreamToolRefusal`] on the streaming path; the buffered path keeps its documented
//! "unparsed text stays content" behaviour because a complete answer has no error channel mid-way.
//! The streaming parser expects one tool-call block per answer, like the buffered parser.
//!
//! Residual gap: the chat SSE path disables the parser when streaming logprobs are requested, so
//! an answer that both carries sample logprobs and offers tools still streams its raw DSML text.
//! That combination is recorded in the design note rather than silently mixed.

use serde_json::Value;

/// The V4.1 DSML marker shared with the prompt renderer.
const DSML: &str = "｜DSML｜";

fn calls_begin() -> String {
    format!("<{DSML} calls>")
}

fn calls_end() -> String {
    format!("</{DSML} calls>")
}

fn invoke_begin() -> String {
    format!("<{DSML} invoke name=\"")
}

fn invoke_close() -> String {
    format!("</{DSML} invoke>")
}

fn parameter_begin() -> String {
    format!("<{DSML} parameter name=\"")
}

fn parameter_end() -> String {
    format!("</{DSML} parameter>")
}

/// One tool call lifted out of a completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    /// Function name from the `invoke` tag.
    pub name: String,
    /// Arguments as a JSON object string, matching the reference decoder's reconstruction.
    pub arguments: String,
}

/// The parsed completion split into visible content and tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCompletion {
    /// The model's visible text before the tool-call block (the block itself is not content).
    pub content: String,
    /// Tool calls in the order the model emitted them.
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Parses a completion's text, or returns `None` when it carries no well-formed tool-call block.
///
/// A malformed block is not an error the client should see: the unparsed text stays content, which
/// is what a client without auto tool choice would show anyway. The caller only treats a
/// `Some` result as a parsed answer.
#[must_use]
pub fn parse_completion(text: &str) -> Option<ParsedCompletion> {
    let begin = calls_begin();
    // vLLM's reference decoder expects the block after a blank line; keep that exact shape but
    // also accept the block at the very start of the text (a model answer with no preamble).
    let (content, rest) = if let Some(index) = text.find(&format!("\n\n{begin}")) {
        (&text[..index], &text[index + 2 + begin.len()..])
    } else {
        let index = text.find(&begin)?;
        (&text[..index], &text[index + begin.len()..])
    };
    let rest = rest.strip_prefix('\n').unwrap_or(rest);
    let mut calls = Vec::new();
    let mut cursor = rest;
    loop {
        cursor = cursor.strip_prefix('\n').unwrap_or(cursor);
        if let Some(after) = cursor.strip_prefix(&calls_end()) {
            // The reference decoder requires the text to end after the block; tolerate only
            // trailing whitespace so a stray eos decode cannot turn a parsed answer into content.
            if after.trim().is_empty() {
                return Some(ParsedCompletion {
                    content: content.to_owned(),
                    tool_calls: calls,
                });
            }
            return None;
        }
        let after_invoke = cursor.strip_prefix(&invoke_begin())?;
        let name_end = after_invoke.find("\">\n")?;
        let name = &after_invoke[..name_end];
        cursor = &after_invoke[name_end + 3..];
        let mut parameters: Vec<(String, bool, String)> = Vec::new();
        loop {
            cursor = cursor.strip_prefix('\n').unwrap_or(cursor);
            if let Some(after_close) = cursor.strip_prefix(&invoke_close()) {
                cursor = after_close;
                break;
            }
            let after_parameter = cursor.strip_prefix(&parameter_begin())?;
            let name_end = after_parameter.find("\" string=\"")?;
            let parameter_name = &after_parameter[..name_end];
            let after_flag = &after_parameter[name_end + "\" string=\"".len()..];
            let (is_string, after_flag) =
                if let Some(after_true) = after_flag.strip_prefix("true\">") {
                    (true, after_true)
                } else {
                    (false, after_flag.strip_prefix("false\">")?)
                };
            let value_end = after_flag.find(&parameter_end())?;
            let value = &after_flag[..value_end];
            parameters.push((parameter_name.to_owned(), is_string, value.to_owned()));
            cursor = &after_flag[value_end + parameter_end().len()..];
        }
        calls.push(ParsedToolCall {
            name: name.to_owned(),
            arguments: arguments_json(&parameters)?,
        });
    }
}

/// Rebuilds the OpenAI `arguments` string exactly as the reference decoder does: a JSON object
/// with `"key": value` pairs in emission order, string parameters quoted and non-string values
/// inserted as JSON.
fn arguments_json(parameters: &[(String, bool, String)]) -> Option<String> {
    let mut pairs = Vec::with_capacity(parameters.len());
    for (name, is_string, value) in parameters {
        let key = serde_json::to_string(name).ok()?;
        let rendered = if *is_string {
            serde_json::to_string(value).ok()?
        } else if serde_json::from_str::<Value>(value).is_ok() {
            value.clone()
        } else {
            // The reference decoder inserts the raw text when it does not parse; a client that
            // sees a JSON object string still gets the argument value unchanged.
            serde_json::to_string(value).ok()?
        };
        pairs.push(format!("{key}: {rendered}"));
    }
    Some(format!("{{{}}}", pairs.join(", ")))
}

/// The OpenAI error code carried by a streaming DSML refusal.
pub const TOOL_STREAM_REFUSAL_CODE: &str = "tool_call_parse_error";

/// The byte bound on the partially-parsed tool-call block.
const MAX_TOOL_STREAM_BLOCK_BYTES: usize = 4 << 20;

/// One event from the incremental DSML parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamToolEvent {
    /// Visible answer text that is not part of a tool-call block.
    Content(String),
    /// The first delta of a tool call: its index in the emitted tool_calls array and the function
    /// name. The caller assigns the id and sends an empty arguments string with this delta.
    CallStart {
        /// Position of this call in the emitted tool_calls array.
        index: usize,
        /// The function name carried by the first delta.
        name: String,
    },
    /// A fragment of the OpenAI arguments JSON string. Concatenated in order, the fragments are
    /// exactly the buffered parser arguments value.
    ArgumentsDelta {
        /// Position of this call in the emitted tool_calls array.
        index: usize,
        /// A fragment of the OpenAI arguments JSON string.
        arguments_delta: String,
    },
}

/// A malformed or unterminated DSML block in a streamed answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamToolRefusal {
    /// The client-facing message; it names what the parser was expecting.
    pub message: String,
}

impl std::fmt::Display for StreamToolRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[derive(Debug)]
enum State {
    Content,
    SkipNewline(Box<State>),
    BetweenCalls,
    InvokeName,
    BetweenParameters {
        index: usize,
        first: bool,
    },
    ParameterName {
        index: usize,
        first: bool,
    },
    ParameterFlag {
        index: usize,
        name: String,
        first: bool,
    },
    ParameterValue {
        index: usize,
        name: String,
        is_string: bool,
        first: bool,
    },
    AfterBlock,
    Failed,
}

fn first_call_state() -> State {
    State::SkipNewline(Box::new(State::SkipNewline(Box::new(State::BetweenCalls))))
}

/// Holds back the longest suffix that could still become a calls marker.
fn hold_back(buffer: &str, patterns: &[String]) -> usize {
    let longest = patterns.iter().map(String::len).max().unwrap_or(0);
    let mut cut = buffer.len();
    for (offset, _) in buffer.char_indices().rev() {
        let suffix = &buffer[offset..];
        if suffix.len() >= longest {
            break;
        }
        if patterns.iter().any(|pattern| pattern.starts_with(suffix)) {
            cut = offset;
        }
    }
    cut
}

fn scan_content(buffer: &str) -> Option<(usize, String, State)> {
    let marker = calls_begin();
    let blank_marker = format!("\n\n{marker}");
    if let Some(index) = buffer.find(&blank_marker) {
        return Some((
            index + blank_marker.len(),
            buffer[..index].to_owned(),
            first_call_state(),
        ));
    }
    if let Some(index) = buffer.find(&marker) {
        return Some((
            index + marker.len(),
            buffer[..index].to_owned(),
            first_call_state(),
        ));
    }
    let cut = hold_back(buffer, &[marker, blank_marker]);
    if cut == 0 {
        None
    } else {
        Some((cut, buffer[..cut].to_owned(), State::Content))
    }
}

fn streaming_pair(name: &str, is_string: bool, value: &str) -> Result<String, String> {
    let key = serde_json::to_string(name)
        .map_err(|error| format!("tool-call stream: cannot encode parameter name: {error}"))?;
    let rendered = if is_string {
        serde_json::to_string(value)
            .map_err(|error| format!("tool-call stream: cannot encode string parameter: {error}"))?
    } else if serde_json::from_str::<Value>(value).is_ok() {
        value.to_owned()
    } else {
        serde_json::to_string(value)
            .map_err(|error| format!("tool-call stream: cannot quote parameter value: {error}"))?
    };
    Ok(format!("{key}: {rendered}"))
}

/// The incremental deepseek_v41 DSML parser for streamed content.
#[derive(Debug)]
pub struct ToolCallStreamParser {
    buffer: String,
    state: State,
    calls: usize,
    refusal: Option<StreamToolRefusal>,
}

impl Default for ToolCallStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallStreamParser {
    /// A parser outside any tool-call block.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            state: State::Content,
            calls: 0,
            refusal: None,
        }
    }

    /// How many tool calls the parser has started.
    #[must_use]
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Consumes one content delta and returns the ready events.
    ///
    /// # Errors
    ///
    /// A [StreamToolRefusal] for a malformed block or an exceeded byte bound; the refusal is
    /// sticky and every later call reports it.
    pub fn feed(&mut self, text: &str) -> Result<Vec<StreamToolEvent>, StreamToolRefusal> {
        if let Some(refusal) = &self.refusal {
            return Err(refusal.clone());
        }
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.buffer.push_str(text);
        self.advance()
    }

    /// Flushes the end of the stream.
    ///
    /// At end of stream an unterminated block is a [StreamToolRefusal], never a silent drop. When
    /// no calls marker was ever seen, any held-back bytes are ordinary content and are returned as
    /// [StreamToolEvent::Content].
    ///
    /// # Errors
    ///
    /// A [StreamToolRefusal] for a malformed or unterminated block.
    pub fn finish(&mut self) -> Result<Vec<StreamToolEvent>, StreamToolRefusal> {
        if let Some(refusal) = &self.refusal {
            return Err(refusal.clone());
        }
        let mut events = Vec::new();
        loop {
            let state = std::mem::replace(&mut self.state, State::Failed);
            match state {
                State::Content => {
                    let content = std::mem::take(&mut self.buffer);
                    if !content.is_empty() {
                        events.push(StreamToolEvent::Content(content));
                    }
                    return Ok(events);
                }
                State::AfterBlock => {
                    if self.buffer.chars().all(char::is_whitespace) {
                        self.buffer.clear();
                        return Ok(events);
                    }
                    return Err(self.refuse(
                        "tool-call stream: non-whitespace text after the tool-call block"
                            .to_owned(),
                    ));
                }
                State::SkipNewline(inner) => {
                    self.state = *inner;
                }
                State::Failed => {
                    return Err(self.refusal.clone().unwrap_or(StreamToolRefusal {
                        message: "tool-call stream: parser is in the failed state".to_owned(),
                    }));
                }
                State::BetweenCalls
                | State::InvokeName
                | State::BetweenParameters { .. }
                | State::ParameterName { .. }
                | State::ParameterFlag { .. }
                | State::ParameterValue { .. } => {
                    return Err(self.refuse(
                        "tool-call stream: unterminated DSML tool-call block at end of stream"
                            .to_owned(),
                    ));
                }
            }
        }
    }

    fn advance(&mut self) -> Result<Vec<StreamToolEvent>, StreamToolRefusal> {
        let mut events = Vec::new();
        loop {
            if self.buffer.len() > MAX_TOOL_STREAM_BLOCK_BYTES {
                let message = format!(
                    "tool-call stream: the tool-call block exceeds the {MAX_TOOL_STREAM_BLOCK_BYTES}-byte bound"
                );
                return Err(self.refuse(message));
            }
            let state = std::mem::replace(&mut self.state, State::Failed);
            match state {
                State::Content => {
                    if self.buffer.is_empty() {
                        self.state = State::Content;
                        return Ok(events);
                    }
                    let Some((consumed, content, next)) = scan_content(&self.buffer) else {
                        self.state = State::Content;
                        return Ok(events);
                    };
                    self.buffer.drain(..consumed);
                    if !content.is_empty() {
                        events.push(StreamToolEvent::Content(content));
                    }
                    self.state = next;
                }
                State::SkipNewline(inner) => {
                    if self.buffer.is_empty() {
                        self.state = State::SkipNewline(inner);
                        return Ok(events);
                    }
                    if self.buffer.starts_with('\n') {
                        self.buffer.drain(..1);
                    }
                    self.state = *inner;
                }
                State::BetweenCalls => {
                    let close = calls_end();
                    let invoke = invoke_begin();
                    if self.buffer.is_empty() {
                        self.state = State::BetweenCalls;
                        return Ok(events);
                    }
                    if self.buffer.starts_with(&close) {
                        self.buffer.drain(..close.len());
                        self.state = State::AfterBlock;
                    } else if self.buffer.starts_with(&invoke) {
                        self.buffer.drain(..invoke.len());
                        self.state = State::InvokeName;
                    } else if close.starts_with(&self.buffer) || invoke.starts_with(&self.buffer) {
                        self.state = State::BetweenCalls;
                        return Ok(events);
                    } else {
                        return Err(self.refuse(
                            "tool-call stream: malformed DSML tool-call block: expected an invoke or the calls end".to_owned(),
                        ));
                    }
                }
                State::InvokeName => {
                    let delimiter = "\">\n";
                    match self.buffer.find(delimiter) {
                        Some(index) => {
                            let name = self.buffer[..index].to_owned();
                            self.buffer.drain(..index + delimiter.len());
                            let call_index = self.calls;
                            self.calls += 1;
                            events.push(StreamToolEvent::CallStart {
                                index: call_index,
                                name,
                            });
                            self.state = State::SkipNewline(Box::new(State::BetweenParameters {
                                index: call_index,
                                first: true,
                            }));
                        }
                        None => {
                            self.state = State::InvokeName;
                            return Ok(events);
                        }
                    }
                }
                State::BetweenParameters { index, first } => {
                    let close = invoke_close();
                    let parameter = parameter_begin();
                    if self.buffer.is_empty() {
                        self.state = State::BetweenParameters { index, first };
                        return Ok(events);
                    }
                    if self.buffer.starts_with(&close) {
                        self.buffer.drain(..close.len());
                        events.push(StreamToolEvent::ArgumentsDelta {
                            index,
                            arguments_delta: if first {
                                "{}".to_owned()
                            } else {
                                "}".to_owned()
                            },
                        });
                        self.state = State::SkipNewline(Box::new(State::BetweenCalls));
                    } else if self.buffer.starts_with(&parameter) {
                        self.buffer.drain(..parameter.len());
                        self.state = State::ParameterName { index, first };
                    } else if close.starts_with(&self.buffer) || parameter.starts_with(&self.buffer)
                    {
                        self.state = State::BetweenParameters { index, first };
                        return Ok(events);
                    } else {
                        return Err(self.refuse(
                            "tool-call stream: malformed DSML tool-call block: expected a parameter or the invoke end".to_owned(),
                        ));
                    }
                }
                State::ParameterName { index, first } => {
                    let delimiter = "\" string=\"";
                    match self.buffer.find(delimiter) {
                        Some(name_end) => {
                            let name = self.buffer[..name_end].to_owned();
                            self.buffer.drain(..name_end + delimiter.len());
                            self.state = State::ParameterFlag { index, name, first };
                        }
                        None => {
                            self.state = State::ParameterName { index, first };
                            return Ok(events);
                        }
                    }
                }
                State::ParameterFlag { index, name, first } => {
                    if self.buffer.is_empty() {
                        self.state = State::ParameterFlag { index, name, first };
                        return Ok(events);
                    }
                    let flag = if self.buffer.starts_with("true\">") {
                        Some(true)
                    } else if self.buffer.starts_with("false\">") {
                        Some(false)
                    } else {
                        None
                    };
                    match flag {
                        Some(is_string) => {
                            let flag_length = if is_string {
                                "true\">".len()
                            } else {
                                "false\">".len()
                            };
                            self.buffer.drain(..flag_length);
                            self.state = State::ParameterValue {
                                index,
                                name,
                                is_string,
                                first,
                            };
                        }
                        None => {
                            let partial = "true\">".starts_with(&self.buffer)
                                || "false\">".starts_with(&self.buffer);
                            if partial {
                                self.state = State::ParameterFlag { index, name, first };
                                return Ok(events);
                            }
                            return Err(self.refuse(
                                "tool-call stream: malformed DSML tool-call block: expected a parameter string flag".to_owned(),
                            ));
                        }
                    }
                }
                State::ParameterValue {
                    index,
                    name,
                    is_string,
                    first,
                } => {
                    let end = parameter_end();
                    match self.buffer.find(&end) {
                        Some(value_end) => {
                            let pair =
                                match streaming_pair(&name, is_string, &self.buffer[..value_end]) {
                                    Ok(pair) => pair,
                                    Err(message) => return Err(self.refuse(message)),
                                };
                            self.buffer.drain(..value_end + end.len());
                            let arguments_delta = if first {
                                format!("{{{pair}")
                            } else {
                                format!(", {pair}")
                            };
                            events.push(StreamToolEvent::ArgumentsDelta {
                                index,
                                arguments_delta,
                            });
                            self.state = State::SkipNewline(Box::new(State::BetweenParameters {
                                index,
                                first: false,
                            }));
                        }
                        None => {
                            self.state = State::ParameterValue {
                                index,
                                name,
                                is_string,
                                first,
                            };
                            return Ok(events);
                        }
                    }
                }
                State::AfterBlock => {
                    if self.buffer.is_empty() {
                        self.state = State::AfterBlock;
                        return Ok(events);
                    }
                    if self
                        .buffer
                        .chars()
                        .any(|character| !character.is_whitespace())
                    {
                        return Err(self.refuse(
                            "tool-call stream: non-whitespace text after the tool-call block"
                                .to_owned(),
                        ));
                    }
                    self.buffer.clear();
                    self.state = State::AfterBlock;
                    return Ok(events);
                }
                State::Failed => {
                    return Err(self.refusal.clone().unwrap_or(StreamToolRefusal {
                        message: "tool-call stream: parser is in the failed state".to_owned(),
                    }));
                }
            }
        }
    }

    fn refuse(&mut self, message: String) -> StreamToolRefusal {
        let refusal = StreamToolRefusal { message };
        self.refusal = Some(refusal.clone());
        self.state = State::Failed;
        refusal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(inner: &str) -> String {
        format!("\n\n<{DSML} calls>\n{inner}\n</{DSML} calls>")
    }

    fn invoke(name: &str, inner: &str) -> String {
        format!("<{DSML} invoke name=\"{name}\">\n{inner}\n</{DSML} invoke>")
    }

    fn parameter(name: &str, is_string: bool, value: &str) -> String {
        format!(
            "<{DSML} parameter name=\"{name}\" string=\"{}\">{value}</{DSML} parameter>",
            if is_string { "true" } else { "false" }
        )
    }

    #[test]
    fn reference_style_call_is_lifted_out_of_the_content() {
        let text = format!(
            "Let me check.{}",
            block(&invoke(
                "get_weather",
                &format!(
                    "{}\n{}",
                    parameter("city", true, "Beijing"),
                    parameter("days", false, "3")
                )
            ))
        );
        let parsed = parse_completion(&text).expect("the block parses");
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"city": "Beijing", "days": 3}"#
        );
    }

    #[test]
    fn several_calls_keep_their_order() {
        let text = block(&format!(
            "{}\n{}",
            invoke("first", &parameter("a", true, "x")),
            invoke("second", &parameter("b", false, "true")),
        ));
        let parsed = parse_completion(&text).expect("the block parses");
        assert_eq!(parsed.content, "");
        let names: Vec<&str> = parsed
            .tool_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect();
        assert_eq!(names, ["first", "second"]);
        assert_eq!(parsed.tool_calls[1].arguments, r#"{"b": true}"#);
    }

    #[test]
    fn a_plain_answer_is_not_a_tool_call() {
        assert_eq!(parse_completion("no tools here"), None);
    }

    #[test]
    fn a_truncated_block_is_left_as_content() {
        let text = format!("pre\n\n<{DSML} calls>\n<{DSML} invoke name=\"x\">\n");
        assert_eq!(parse_completion(&text), None);
    }

    #[test]
    fn trailing_whitespace_after_the_block_is_tolerated() {
        let text = format!("{}\n", block(&invoke("ping", &parameter("n", false, "1"))));
        let parsed = parse_completion(&text).expect("the block parses");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"n": 1}"#);
    }

    #[test]
    fn duplicate_parameter_names_keep_both_entries() {
        // The reference decoder refuses duplicates; tolerating them preserves the model's answer
        // instead of dropping the call, and the last occurrence is what a JSON client reads.
        let text = block(&invoke(
            "dup",
            &format!(
                "{}\n{}",
                parameter("k", true, "one"),
                parameter("k", true, "two")
            ),
        ));
        let parsed = parse_completion(&text).expect("the block parses");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            r#"{"k": "one", "k": "two"}"#
        );
    }

    // ---- streaming parser ------------------------------------------------

    fn assemble(events: Vec<StreamToolEvent>) -> (String, Vec<(String, String)>) {
        let mut content = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                StreamToolEvent::Content(text) => content.push_str(&text),
                StreamToolEvent::CallStart { index, name } => {
                    assert_eq!(index, calls.len(), "call starts are in order");
                    calls.push((name, String::new()));
                }
                StreamToolEvent::ArgumentsDelta {
                    index,
                    arguments_delta,
                } => calls[index].1.push_str(&arguments_delta),
            }
        }
        (content, calls)
    }

    fn stream(
        text: &str,
        splits: &[usize],
    ) -> Result<(String, Vec<(String, String)>), StreamToolRefusal> {
        let mut parser = ToolCallStreamParser::new();
        let mut events = Vec::new();
        let mut start = 0usize;
        for &split in splits {
            events.extend(parser.feed(&text[start..split])?);
            start = split;
        }
        events.extend(parser.feed(&text[start..])?);
        events.extend(parser.finish()?);
        Ok(assemble(events))
    }

    fn expected_payload(text: &str) -> (String, Vec<(String, String)>) {
        match parse_completion(text) {
            Some(parsed) => (
                parsed.content,
                parsed
                    .tool_calls
                    .into_iter()
                    .map(|call| (call.name, call.arguments))
                    .collect(),
            ),
            None => (text.to_owned(), Vec::new()),
        }
    }

    fn fixed_texts() -> Vec<String> {
        vec![
            format!(
                "Let me check.{}",
                block(&invoke(
                    "get_weather",
                    &format!(
                        "{}\n{}",
                        parameter("city", true, "Beijing"),
                        parameter("days", false, "3")
                    )
                ))
            ),
            block(&format!(
                "{}\n{}",
                invoke("first", &parameter("a", true, "x")),
                invoke("second", &parameter("b", false, "true")),
            )),
            block(&invoke(
                "nested",
                &format!(
                    "{}\n{}",
                    parameter(
                        "payload",
                        false,
                        r#"{"nested": {"list": [1, 2.5, true, null], "text": "a\"b\\c"}}"#
                    ),
                    parameter("note", true, "北京 \"quoted\" \\ backslash"),
                ),
            )),
            format!("preamble{}", block(&invoke("ping", ""))),
            "A plain answer with no DSML.".to_owned(),
            "partial marker text ｜DSML｜ calls".to_owned(),
            format!(
                "{}\n{}  \n",
                block(&invoke("trailing", &parameter("n", false, "1"))),
                ""
            ),
        ]
    }

    #[test]
    fn a_streamed_answer_matches_the_buffered_payload() {
        for text in fixed_texts() {
            let want = expected_payload(&text);
            let got = stream(&text, &[]).expect("the fixed text streams");
            assert_eq!(got, want, "one-shot stream differs for {text:?}");
        }
    }

    #[test]
    fn every_split_boundary_reassembles_the_same_payload() {
        for text in fixed_texts() {
            let want = expected_payload(&text);
            let mut boundaries: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
            boundaries.push(text.len());
            for split in boundaries {
                let got = stream(&text, &[split])
                    .unwrap_or_else(|error| panic!("split at {split} refused: {error}"));
                assert_eq!(got, want, "split at byte {split} of {text:?}");
            }
        }
    }

    #[test]
    fn a_char_at_a_time_stream_reassembles_the_payload() {
        let text = format!(
            "Go {}",
            block(&invoke(
                "teleport",
                &format!(
                    "{}\n{}",
                    parameter("where", true, "the \"moon\""),
                    parameter("when", false, r#"{"at": [1, 2]}"#)
                )
            ))
        );
        let want = expected_payload(&text);
        let mut events = Vec::new();
        let mut parser = ToolCallStreamParser::new();
        for character in text.chars() {
            events.extend(parser.feed(&character.to_string()).expect("char feed"));
        }
        events.extend(parser.finish().expect("finish"));
        assert_eq!(assemble(events), want);
    }

    #[test]
    fn a_block_with_no_parameters_streams_an_empty_object() {
        let text = block(&invoke("ping", ""));
        let (_, calls) = stream(&text, &[]).expect("empty call streams");
        assert_eq!(calls, vec![("ping".to_owned(), "{}".to_owned())]);
    }

    #[test]
    fn a_malformed_literal_is_a_typed_refusal() {
        let text = format!(
            "\n\n<{DSML} calls>\n<{DSML} invoke name=\"x\">\n<{DSML} junk name=\"a\" string=\"true\">v</{DSML} parameter>\n</{DSML} invoke>\n</{DSML} calls>"
        );
        let mut parser = ToolCallStreamParser::new();
        let error = parser.feed(&text).expect_err("malformed block must refuse");
        assert!(
            error.message.contains("malformed DSML"),
            "refusal message: {}",
            error.message
        );
        // The refusal is sticky: later deltas report the same error instead of resyncing.
        assert!(parser.feed("more text").is_err());
        assert!(parser.finish().is_err());
    }

    #[test]
    fn an_unterminated_block_is_a_typed_refusal_at_finish() {
        let text = format!("\n\n<{DSML} calls>\n<{DSML} invoke name=\"x\">\n");
        let mut parser = ToolCallStreamParser::new();
        // The invoke name is already known, so the prefix may emit its CallStart; what matters is
        // that the end of the stream turns the unfinished block into a typed refusal.
        parser.feed(&text).expect("prefix streams");
        let error = parser.finish().expect_err("unterminated block must refuse");
        assert!(
            error.message.contains("unterminated"),
            "refusal message: {}",
            error.message
        );
    }

    #[test]
    fn text_after_the_block_is_a_typed_refusal() {
        let text = format!("{}oops", block(&invoke("x", &parameter("a", true, "1"))));
        let mut parser = ToolCallStreamParser::new();
        let error = parser.feed(&text).expect_err("trailing text must refuse");
        assert!(
            error.message.contains("after the tool-call block"),
            "refusal message: {}",
            error.message
        );
    }

    #[test]
    fn an_oversized_block_is_a_typed_refusal() {
        let huge = "x".repeat(MAX_TOOL_STREAM_BLOCK_BYTES + 16);
        let text = format!(
            "\n\n<{DSML} calls>\n<{DSML} invoke name=\"big\">\n<{DSML} parameter name=\"v\" string=\"true\">{huge}</{DSML} parameter>\n</{DSML} invoke>\n</{DSML} calls>"
        );
        let mut parser = ToolCallStreamParser::new();
        let error = parser.feed(&text).expect_err("the byte bound must refuse");
        assert!(
            error.message.contains("byte bound"),
            "refusal message: {}",
            error.message
        );
    }
}
