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

//! Dory's released ChatML template.
//!
//! This is the text surface from the checkpoint's chat_template.jinja, reduced the same way
//! the Qwen codec reduces its template: text turns, the tool catalogue and XML call format,
//! assistant reasoning, and the thinking-mode generation prompt. It is kept model-side because
//! Dory's template differs from Qwen's in the one place that changes every prompt: Dory opens an
//! empty system turn, never injects a reasoning-effort instruction, and uses its own thinking
//! markers on historical assistant turns.

use serde_json::Value;
use sparkflash_model::prompt::PromptError;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

const TOOLS_PROMPT: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

const TOOL_CALL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Render the released Dory chat template.
///
/// # Errors
/// A request without messages, a message that is not a text turn, or a tool definition the
/// template cannot render.
pub fn render_chat(request: &Value) -> Result<String, PromptError> {
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("messages must be an array"))?;
    if messages.is_empty() {
        return Err(invalid("messages must carry at least one turn"));
    }
    let kwargs = request.get("chat_template_kwargs");
    let thinking = bool_setting(request, kwargs, "enable_thinking").unwrap_or(true);
    let medium_effort = bool_setting(request, kwargs, "medium_effort").unwrap_or(false);
    let truncate_history_thinking =
        bool_setting(request, kwargs, "truncate_history_thinking").unwrap_or(true);
    let tools = request
        .get("tools")
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty());
    let first_is_system = messages[0].get("role").and_then(Value::as_str) == Some("system");
    let system_defined = first_is_system && messages[0].get("content").is_some();
    let (system, loop_messages) = if first_is_system {
        (content_text(&messages[0])?, &messages[1..])
    } else {
        (String::new(), messages.as_slice())
    };
    let last_user = loop_messages
        .iter()
        .rposition(|message| message.get("role").and_then(Value::as_str) == Some("user"));

    let mut text = String::new();
    if !first_is_system || system_defined || tools.is_some() {
        text.push_str(IM_START);
        text.push_str("system\n");
        text.push_str(&system);
        if let Some(tools) = tools {
            if !system.is_empty() {
                text.push_str("\n\n");
            }
            push_tools(&mut text, tools)?;
        }
        text.push_str(IM_END);
        text.push('\n');
    }

    for (index, message) in loop_messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("every message needs a role"))?;
        match role {
            "assistant" => {
                let content_value = message.get("content");
                let falsey = jina_falsey(content_value);
                let content_is_string = falsey
                    || matches!(
                        content_value,
                        Some(Value::String(_)) | Some(Value::Null) | None
                    );
                let content = if falsey {
                    String::new()
                } else {
                    content_text(message)?
                };
                let reasoning = message
                    .get("reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let has_reasoning = !reasoning.trim().is_empty();
                let truncated =
                    truncate_history_thinking && last_user.is_some_and(|last| index < last);
                let tool_calls = message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .filter(|calls| !calls.is_empty());
                let formatted = if has_reasoning {
                    format!("\x3cthink\x3e\n{reasoning}\x3c/think\x3e{content}")
                } else if !content_is_string
                    || content.contains("\x3cthink\x3e")
                    || content.contains("\x3c/think\x3e")
                {
                    content.clone()
                } else {
                    format!("\x3cthink\x3e\x3c/think\x3e{content}")
                };
                let formatted_is_string = content_is_string || has_reasoning;
                text.push_str(IM_START);
                text.push_str("assistant\n");
                if let Some(calls) = tool_calls {
                    if !formatted_is_string {
                        text.push_str("\x3cthink\x3e\x3c/think\x3e");
                    } else if formatted.trim().is_empty() && !has_reasoning {
                        text.push_str("\x3cthink\x3e\x3c/think\x3e");
                        text.push('\n');
                    } else if truncated {
                        let mut stripped = formatted.clone();
                        if let Some((_, after)) = stripped.rsplit_once("\x3c/think\x3e") {
                            stripped = after.to_owned();
                        } else if let Some((before, _)) = stripped.split_once("\x3cthink\x3e") {
                            stripped = before.to_owned();
                        }
                        let stripped = format!("\x3cthink\x3e\x3c/think\x3e{stripped}");
                        if !stripped.is_empty() {
                            text.push_str(&stripped);
                            text.push('\n');
                        }
                    } else {
                        text.push_str(formatted.trim());
                        text.push('\n');
                    }
                    for call in calls {
                        push_tool_call(&mut text, call)?;
                    }
                } else if truncated {
                    let mut stripped = formatted.clone();
                    if stripped.contains("\x3cthink\x3e")
                        && stripped.contains("\x3c/think\x3e")
                        && let Some((_, after)) = stripped.rsplit_once("\x3c/think\x3e")
                    {
                        stripped = format!("\x3cthink\x3e\x3c/think\x3e{after}");
                    }
                    let stripped = stripped.trim();
                    if !stripped.is_empty() {
                        text.push_str(stripped);
                    }
                } else {
                    text.push_str(formatted.trim());
                }
                text.push_str(IM_END);
                text.push('\n');
            }
            "user" | "system" | "developer" => {
                let content = if matches!(message.get("content"), Some(Value::Null)) {
                    "None".to_owned()
                } else {
                    content_text(message)?
                };
                text.push_str(IM_START);
                text.push_str(role);
                text.push('\n');
                if role == "user" && Some(index) == last_user && medium_effort {
                    text.push_str(&content);
                    text.push_str("\n\n{reasoning effort: efficient}");
                } else {
                    text.push_str(&content);
                }
                text.push_str(IM_END);
                text.push('\n');
            }
            "tool" => {
                let merged_before = index > 0
                    && loop_messages[index - 1].get("role").and_then(Value::as_str) == Some("tool");
                let merged_after = loop_messages
                    .get(index + 1)
                    .and_then(|next| next.get("role"))
                    .and_then(Value::as_str)
                    == Some("tool");
                if !merged_before && index > 0 {
                    text.push_str(IM_START);
                    text.push_str("user");
                    text.push('\n');
                }
                text.push_str("<tool_response>\n");
                if matches!(message.get("content"), Some(Value::Null)) {
                    text.push_str("None");
                } else {
                    text.push_str(&content_text(message)?);
                }
                text.push_str("\n</tool_response>\n");
                if !merged_after {
                    text.push_str(IM_END);
                    text.push('\n');
                }
            }
            other => {
                let content = if matches!(message.get("content"), Some(Value::Null)) {
                    "None".to_owned()
                } else {
                    content_text(message)?
                };
                text.push_str(IM_START);
                text.push_str(other);
                text.push('\n');
                text.push_str(&content);
                text.push_str(IM_END);
                text.push('\n');
            }
        }
    }

    let add_generation_prompt =
        bool_setting(request, kwargs, "add_generation_prompt").unwrap_or(true);
    if add_generation_prompt {
        text.push_str(IM_START);
        text.push_str("assistant\n");
        if thinking {
            text.push_str("<think>\n");
        } else {
            text.push_str("<think></think>");
        }
    }
    Ok(text)
}

fn bool_setting(request: &Value, kwargs: Option<&Value>, name: &str) -> Option<bool> {
    kwargs
        .and_then(|value| value.get(name))
        .and_then(Value::as_bool)
        .or_else(|| request.get(name).and_then(Value::as_bool))
}

fn jina_falsey(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Bool(flag)) => !*flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value == 0.0),
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Array(items)) => items.is_empty(),
        Some(Value::Object(map)) => map.is_empty(),
    }
}

fn invalid(message: impl Into<String>) -> PromptError {
    PromptError::invalid(message, Some("messages"), Some("invalid_request_error"))
}

/// One message's text: a string, or a list of text parts.
fn content_text(message: &Value) -> Result<String, PromptError> {
    match message.get("content") {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Null) | None => Ok(String::new()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            push_python_repr(&mut text, &Value::Array(parts.clone()));
            Ok(text)
        }
        Some(other) => {
            let mut text = String::new();
            push_python_repr(&mut text, other);
            Ok(text)
        }
    }
}

fn push_tools(text: &mut String, tools: &[Value]) -> Result<(), PromptError> {
    text.push_str(TOOLS_PROMPT);
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        let function_map = function
            .as_object()
            .ok_or_else(|| invalid("every tool needs an object"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("every tool needs a name"))?;
        text.push_str("\n<function>\n<name>");
        text.push_str(name);
        text.push_str("</name>");
        if let Some(description) = function.get("description").and_then(Value::as_str) {
            text.push_str("\n<description>");
            text.push_str(description.trim());
            text.push_str("</description>");
        }
        text.push_str("\n<parameters>");
        if let Some(properties) = function
            .pointer("/parameters/properties")
            .and_then(Value::as_object)
        {
            for (name, fields) in properties {
                let fields = fields
                    .as_object()
                    .ok_or_else(|| invalid("a tool parameter must be an object"))?;
                text.push_str("\n<parameter>\n<name>");
                text.push_str(name);
                text.push_str("</name>");
                if let Some(kind) = fields.get("type") {
                    text.push_str("\n<type>");
                    if kind.is_array() || kind.is_object() {
                        push_python_repr(text, kind);
                    } else {
                        push_python_str(text, kind);
                    }
                    text.push_str("</type>");
                }
                if let Some(description) = fields.get("description").and_then(Value::as_str) {
                    text.push_str("\n<description>");
                    text.push_str(description.trim());
                    text.push_str("</description>");
                }
                if let Some(values) = fields.get("enum") {
                    text.push_str("\n<enum>");
                    push_python_json(text, values);
                    text.push_str("</enum>");
                }
                push_extra_keys(text, fields, &["name", "type", "description", "enum"], true);
                text.push_str("\n</parameter>");
            }
        }
        if let Some(parameters) = function.get("parameters").and_then(Value::as_object) {
            push_extra_keys(text, parameters, &["type", "properties", "required"], true);
        }
        if let Some(required) = function.pointer("/parameters/required") {
            text.push_str("\n<required>");
            push_python_json(text, required);
            text.push_str("</required>");
        }
        text.push_str("\n</parameters>");
        push_extra_keys(
            text,
            function_map,
            &["type", "name", "description", "parameters"],
            true,
        );
        text.push_str("\n</function>");
    }
    text.push_str("\n</tools>");
    text.push_str(TOOL_CALL_INSTRUCTIONS);
    Ok(())
}

fn push_tool_call(text: &mut String, call: &Value) -> Result<(), PromptError> {
    let function = call.get("function").unwrap_or(call);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("every tool call needs a function name"))?;
    text.push_str("<tool_call>\n<function=");
    text.push_str(name);
    text.push_str(">\n");
    if let Some(arguments) = function.get("arguments")
        && let Some(arguments) = arguments_object(arguments)?
    {
        for (key, value) in arguments {
            text.push_str("<parameter=");
            text.push_str(&key);
            text.push_str(">\n");
            push_template_value(text, &value);
            text.push_str("\n</parameter>\n");
        }
    }
    text.push_str("</function>\n</tool_call>\n");
    Ok(())
}

fn arguments_object(value: &Value) -> Result<Option<serde_json::Map<String, Value>>, PromptError> {
    match value {
        Value::Object(map) => Ok(Some(map.clone())),
        Value::Null => Ok(None),
        Value::String(text) if text.trim().is_empty() => Ok(None),
        Value::String(text) => serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .map(Some)
            .ok_or_else(|| invalid("tool call arguments must be a JSON object")),
        _ => Err(invalid(
            "tool call arguments must be an object or a JSON object string",
        )),
    }
}

fn python_printable(character: char) -> bool {
    if character.is_control() {
        return false;
    }
    // Python's isprintable() rejects every separator except ASCII space and every format
    // character. List the ranges a chat prompt can realistically carry, including the soft
    // hyphen, bidi/isolate controls and the variation-selector supplements.
    !matches!(
        character,
        '\u{a0}'
            | '\u{ad}'
            | '\u{600}'..='\u{605}'
            | '\u{61c}'
            | '\u{6dd}'
            | '\u{70f}'
            | '\u{890}'..='\u{891}'
            | '\u{8e2}'
            | '\u{1680}'
            | '\u{180e}'
            | '\u{2000}'..='\u{200f}'
            | '\u{2028}'..='\u{202f}'
            | '\u{205f}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{3000}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
            | '\u{110bd}'
            | '\u{110cd}'
            | '\u{13430}'..='\u{1343f}'
            | '\u{1bca0}'..='\u{1bca3}'
            | '\u{1d173}'..='\u{1d17a}'
            | '\u{e0001}'
            | '\u{e0020}'..='\u{e007f}'
    )
}

fn push_python_escape(text: &mut String, character: char) {
    let code = character as u32;
    if code <= 0xff {
        text.push_str(&format!("\\x{code:02x}"));
    } else if code <= 0xffff {
        text.push_str(&format!("\\u{code:04x}"));
    } else {
        text.push_str(&format!("\\U{code:08x}"));
    }
}

/// Python's repr() for a value the template passes through `| string` as a list/dict.
fn push_python_repr(text: &mut String, value: &Value) {
    match value {
        Value::String(string) => {
            let quote = if string.contains('\'') && !string.contains('"') {
                '"'
            } else {
                '\''
            };
            text.push(quote);
            for character in string.chars() {
                match character {
                    '\\' => text.push_str("\\\\"),
                    c if c == quote => {
                        text.push('\\');
                        text.push(c);
                    }
                    '\n' => text.push_str("\\n"),
                    '\r' => text.push_str("\\r"),
                    '\t' => text.push_str("\\t"),
                    '\u{8}' => text.push_str("\\x08"),
                    '\u{b}' => text.push_str("\\x0b"),
                    '\u{c}' => text.push_str("\\x0c"),
                    c if !python_printable(c) => push_python_escape(text, c),
                    c => text.push(c),
                }
            }
            text.push(quote);
        }
        Value::Array(items) => {
            text.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                push_python_repr(text, item);
            }
            text.push(']');
        }
        Value::Object(map) => {
            text.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                push_python_repr(text, &Value::String(key.clone()));
                text.push_str(": ");
                push_python_repr(text, item);
            }
            text.push('}');
        }
        other => push_python_str(text, other),
    }
}

fn push_template_value(text: &mut String, value: &Value) {
    match value {
        Value::Object(_) | Value::Array(_) => push_python_json(text, value),
        other => push_python_str(text, other),
    }
}

/// Python's str() for a scalar, which is what the template's | string filter emits.
fn push_python_str(text: &mut String, value: &Value) {
    match value {
        Value::Null => text.push_str("None"),
        Value::Bool(true) => text.push_str("True"),
        Value::Bool(false) => text.push_str("False"),
        Value::Number(number) => push_number(text, number),
        Value::String(string) => text.push_str(string),
        other => push_python_json(text, other),
    }
}

/// The template's render_extra_keys macro: every unhandled key, in insertion order.
fn push_extra_keys(
    text: &mut String,
    map: &serde_json::Map<String, Value>,
    handled: &[&str],
    leading_newline: bool,
) -> bool {
    let mut wrote = false;
    for (key, value) in map {
        if handled.contains(&key.as_str()) {
            continue;
        }
        if leading_newline || wrote {
            text.push('\n');
        }
        wrote = true;
        text.push('<');
        text.push_str(key);
        text.push('>');
        match value {
            Value::Object(_) | Value::Array(_) => push_python_json(text, value),
            other => push_python_str(text, other),
        }
        text.push_str("</");
        text.push_str(key);
        text.push('>');
    }
    wrote
}

/// Python's scalar number rendering: integers exactly, floats as `repr`/`json.dumps`.
fn push_number(text: &mut String, number: &serde_json::Number) {
    if let Some(integer) = number.as_i64() {
        text.push_str(&integer.to_string());
    } else if let Some(integer) = number.as_u64() {
        text.push_str(&integer.to_string());
    } else if let Some(value) = number.as_f64() {
        push_float(text, value);
    }
}

fn push_float(text: &mut String, value: f64) {
    let mut repr = format!("{value:?}");
    if let Some((mantissa, exponent)) = repr.split_once(['e', 'E'])
        && let Ok(exponent) = exponent.parse::<i32>()
    {
        let sign = if exponent < 0 { '-' } else { '+' };
        repr = format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    text.push_str(&repr);
}

/// The template's `tojson` filter: sorted keys, ASCII-only JSON, and Jinja's HTML-safe
/// escaping (`<`, `>`, `&`, `'`), with Python's `json.dumps` spacing.
fn push_python_json(text: &mut String, value: &Value) {
    match value {
        Value::Null => text.push_str("null"),
        Value::Bool(true) => text.push_str("true"),
        Value::Bool(false) => text.push_str("false"),
        Value::Number(number) => push_number(text, number),
        Value::String(string) => push_json_string(text, string),
        Value::Array(items) => {
            text.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                push_python_json(text, item);
            }
            text.push(']');
        }
        Value::Object(map) => {
            text.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                push_json_string(text, key);
                text.push_str(": ");
                push_python_json(text, &map[key]);
            }
            text.push('}');
        }
    }
}

/// Python `json.dumps(ensure_ascii=True)` string escaping plus Jinja's HTML-safe replacements.
fn push_json_string(text: &mut String, string: &str) {
    text.push('"');
    for character in string.chars() {
        match character {
            '"' => text.push_str("\\\""),
            '\\' => text.push_str("\\\\"),
            '\n' => text.push_str("\\n"),
            '\r' => text.push_str("\\r"),
            '\t' => text.push_str("\\t"),
            '\u{8}' => text.push_str("\\b"),
            '\u{c}' => text.push_str("\\f"),
            '<' => text.push_str("\\u003c"),
            '>' => text.push_str("\\u003e"),
            '&' => text.push_str("\\u0026"),
            '\'' => text.push_str("\\u0027"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                text.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if c.is_ascii() => text.push(c),
            c => {
                let code = c as u32;
                if code <= 0xffff {
                    text.push_str(&format!("\\u{code:04x}"));
                } else {
                    let value = code - 0x10000;
                    let high = 0xd800 + (value >> 10);
                    let low = 0xdc00 + (value & 0x3ff);
                    text.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    text.push('"');
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::render_chat;

    #[test]
    fn a_default_turn_opens_the_empty_system_turn_and_no_reasoning_instruction() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e"
        );
    }

    #[test]
    fn a_system_turn_and_thinking_prompt_are_rendered_in_dory_order() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"}
            ]
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\n"
        );
    }

    #[test]
    fn historical_assistant_thinking_is_truncated_before_the_last_user() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "old answer", "reasoning_content": "old thought"},
                {"role": "user", "content": "two"}
            ],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert!(
            text.contains("<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3eold answer<|im_end|>"),
            "{text}"
        );
        assert!(!text.contains("old thought"), "{text}");
    }

    #[test]
    fn tools_use_the_released_catalogue_block() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City"}},
                        "required": ["city"]
                    }
                }
            }],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert!(text.starts_with("<|im_start|>system\n# Tools"), "{text}");
        assert!(text.contains("<name>weather</name>"), "{text}");
        assert!(text.contains("<required>[\"city\"]</required>"), "{text}");
        assert!(
            text.contains("<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e"),
            "{text}"
        );
    }

    #[test]
    fn tool_schemas_render_every_extra_key() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer", "default": 3, "minimum": 0}},
                        "required": ["x"],
                        "additionalProperties": false
                    }
                }
            }],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert!(text.contains(
            "<parameter>\n<name>x</name>\n<type>integer</type>\n<default>3</default>\n<minimum>0</minimum>\n</parameter>\n<additionalProperties>False</additionalProperties>\n<required>[\"x\"]</required>\n</parameters>\n</function>"
        ), "{text}");
    }

    #[test]
    fn scalar_tool_arguments_use_python_spellings() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {
                        "name": "f",
                        "arguments": {"flag": true, "nothing": null, "text": "hi", "list": [1, 2]}
                    }
                }]}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e\n<tool_call>\n<function=f>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=nothing>\nNone\n</parameter>\n<parameter=text>\nhi\n</parameter>\n<parameter=list>\n[1, 2]\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn tool_responses_keep_their_whitespace() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "  out  \n"}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>user\n<tool_response>\n  out  \n\n</tool_response>\n<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn reasoning_spaces_after_the_last_user_are_preserved() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "old", "reasoning_content": "  thought  "}
            ],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\n  thought  \x3c/think\x3eold<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e",
            "{text}"
        );
    }

    #[test]
    fn malformed_tool_arguments_are_refused() {
        let error = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "f", "arguments": "not json"}
                }]}
            ]
        }))
        .expect_err("malformed arguments are refused");
        assert!(format!("{error}").contains("arguments"), "{error}");
    }

    #[test]
    fn merged_tool_responses_do_not_gain_a_blank_line() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "a"},
                {"role": "tool", "content": "b"}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response>\n<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn a_tool_first_message_has_no_user_turn() {
        let text = render_chat(&json!({
            "messages": [{"role": "tool", "content": "a"}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<tool_response>\na\n</tool_response>\n<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn developer_and_null_user_content_render_like_the_template() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": null},
                {"role": "developer", "content": "d"}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nNone<|im_end|>\n<|im_start|>developer\nd<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn a_close_only_history_marker_is_not_rewritten() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a\x3c/think\x3eb"},
                {"role": "user", "content": "two"}
            ],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\na\x3c/think\x3eb<|im_end|>\n<|im_start|>user\ntwo<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e",
            "{text}"
        );
    }

    #[test]
    fn a_union_parameter_type_uses_python_repr() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": "q"}],
            "tools": [{"type": "function", "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
            }}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(text.contains("<type>['string', 'null']</type>"), "{text}");
    }

    #[test]
    fn multi_part_content_renders_as_python_repr() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"}
            ]}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n[{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn assistant_list_content_matches_the_template() {
        let text = render_chat(&json!({
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "x"}]}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n[{'type': 'text', 'text': 'x'}]<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn assistant_list_with_tool_calls_uses_the_marker_only() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "x"}],
                 "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e<tool_call>\n<function=f>\n</function>\n</tool_call>\n<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn repr_escapes_control_characters() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "a\nb\tc"}]}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(
            text.contains("<|im_start|>user\n[{'type': 'text', 'text': 'a\\nb\\tc'}]<|im_end|>"),
            "{text}"
        );
    }

    #[test]
    fn tojson_sorts_keys_and_html_escapes() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "f", "arguments": {"obj": {"z": 1, "a": "<z>&'café"}}}
                }]}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(text.contains("<parameter=obj>\n{\"a\": \"\\u003cz\\u003e\\u0026\\u0027caf\\u00e9\", \"z\": 1}\n</parameter>"), "{text}");
    }

    #[test]
    fn floats_match_python_and_tojson() {
        let call = |value: serde_json::Value| {
            render_chat(&json!({
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "function": {"name": "f", "arguments": {"x": value}}
                    }]}
                ],
                "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
            }))
            .expect("the request renders")
        };
        assert!(call(json!(1e-5)).contains("<parameter=x>\n1e-05\n</parameter>"));
        assert!(
            call(json!(1.2345678901234566e-9))
                .contains("<parameter=x>\n1.2345678901234566e-09\n</parameter>")
        );
    }

    #[test]
    fn repr_uses_double_quotes_when_the_string_has_an_apostrophe() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "don't panic"}]}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(
            text.contains("[{'type': 'text', 'text': \"don't panic\"}]"),
            "{text}"
        );
    }

    #[test]
    fn top_level_template_kwargs_are_honored() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": "q"}],
            "enable_thinking": false,
            "medium_effort": true,
            "add_generation_prompt": false
        }))
        .expect("the request renders");
        assert!(
            text.contains("q\n\n{reasoning effort: efficient}<|im_end|>"),
            "{text}"
        );
    }

    #[test]
    fn the_reasoning_field_is_not_reasoning_content() {
        let text = render_chat(&json!({
            "messages": [{"role": "assistant", "content": "c", "reasoning": "r"}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(
            text.contains("<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3ec<|im_end|>"),
            "{text}"
        );
    }

    #[test]
    fn scalar_content_and_a_missing_system_content_are_rendered() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "system"},
                {"role": "user", "content": 42}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(text, "<|im_start|>user\n42<|im_end|>\n", "{text}");
    }

    #[test]
    fn repr_escapes_non_printable_unicode() {
        for (raw, escaped) in [("a\u{a0}b", "a\\xa0b"), ("a\u{200b}b", "a\\u200bb")] {
            let text = render_chat(&json!({
                "messages": [{"role": "user", "content": [{"type": "text", "text": raw}]}],
                "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
            }))
            .expect("the request renders");
            assert!(text.contains(escaped), "{raw:?}: {text}");
        }
    }

    #[test]
    fn parsed_json_floats_keep_python_round_trip_text() {
        let request: Value = serde_json::from_str(
            r#"{
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "f", "arguments": {"x": 912068.5437784989}}}
                ]}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }"#,
        )
        .expect("the request parses");
        let text = render_chat(&request).expect("the request renders");
        assert!(
            text.contains("<parameter=x>\n912068.5437784989\n</parameter>"),
            "{text}"
        );
    }

    #[test]
    fn falsey_assistant_content_is_the_empty_string() {
        let text = render_chat(&json!({
            "messages": [{"role": "assistant", "content": []}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n\x3cthink\x3e\x3c/think\x3e<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn repr_escapes_format_characters() {
        for raw in ["\u{ad}", "\u{200e}", "\u{200f}", "\u{2066}", "\u{180e}"] {
            let text = render_chat(&json!({
                "messages": [{"role": "user", "content": [{"type": "text", "text": raw}]}],
                "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
            }))
            .expect("the request renders");
            assert!(!text.contains(raw), "{raw:?}: {text}");
        }
    }

    #[test]
    fn scalar_assistant_content_does_not_gain_a_thinking_marker() {
        let text = render_chat(&json!({
            "messages": [{"role": "assistant", "content": 7}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert_eq!(
            text, "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n7<|im_end|>\n",
            "{text}"
        );
    }

    #[test]
    fn historical_whitespace_before_tool_calls_is_preserved() {
        let text = render_chat(&json!({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": " ", "tool_calls": [
                    {"function": {"name": "f", "arguments": {}}}
                ]},
                {"role": "user", "content": "q2"}
            ],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(
            text.contains("\x3cthink\x3e\x3c/think\x3e \n<tool_call>"),
            "{text}"
        );
    }

    #[test]
    fn repr_escapes_non_bmp_non_printable_as_capital_u() {
        let text = render_chat(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "\u{e0001}"}]}],
            "chat_template_kwargs": {"enable_thinking": false, "add_generation_prompt": false}
        }))
        .expect("the request renders");
        assert!(text.contains("\\U000e0001"), "{text}");
    }
}
