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

//! Prompt rendering faithful to reference/encoding/encoding.py.
//!
//! The published deepseek-recipe 0.1 V4.1 renderer is close to, but not
//! byte-identical with, the official encoder. It differs in four observable
//! ways, so this module ports the reference renderer instead of wrapping it:
//!
//! 1. It always appends the assistant generation header, even when the last
//!    message is an assistant or latest_reminder turn.
//! 2. It injects the tool prompt into the first system message rather than the
//!    message that carries the tools (mid-conversation system messages).
//! 3. It never drops reasoning from earlier assistant turns (dropped unless
//!    tools are present).
//! 4. It has no representation for quick-instruction task tokens (action,
//!    query, title, ...).
//!
//! Recipe token constants and types are still used for wire compatibility.

use std::collections::HashMap;

use deepseek_recipe_core::util::json_formatter::stringify_python_style;
use deepseek_recipe_encoding::v4::{
    ASSISTANT_SP_TOKEN, BOS_TOKEN, DSML_SP_TOKEN, EOS_TOKEN, LATEST_REMINDER_SP_TOKEN,
    SYSTEM_SP_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN, USER_SP_TOKEN,
};
use serde_json::{Value, json};

use super::case::{CaseError, EncodingMessage, MessageContent, OpenAiToolCall, ThinkingMode};

/// V4.1 tool call block name (leading space is significant).
pub const TOOL_CALLS_BLOCK_NAME: &str = " calls";
/// V4.1 tool call tag name (leading space is significant).
pub const TOOL_CALL_TAG_NAME: &str = " invoke";
/// V4.1 tool parameter tag name (leading space is significant).
pub const TOOL_PARAMETER_TAG_NAME: &str = " parameter";

/// Default numeric budget the reference encoder uses when none is requested.
pub const DEFAULT_EFFORT_SCORE: u8 = 50;

const TOOLS_TEMPLATE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}{block}>" block like the following:

<{dsml}{block}>
<{dsml}{invoke} name="$TOOL_NAME">
<{dsml}{parameter} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}{parameter}>
...
</{dsml}{invoke}>
<{dsml}{invoke} name="$TOOL_NAME2">
...
</{dsml}{invoke}>
</{dsml}{block}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#;

const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

/// Render the numeric reasoning effort prefix (thinking mode, index 0 only).
pub(crate) fn render_reasoning_effort(
    index: usize,
    thinking_mode: ThinkingMode,
    effort_score: Option<u8>,
) -> String {
    if index == 0 && thinking_mode == ThinkingMode::Thinking {
        let score = effort_score.unwrap_or(DEFAULT_EFFORT_SCORE);
        format!(
            "Reasoning Effort: {score} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        )
    } else {
        String::new()
    }
}

/// Render the tool instruction block for already-serialized function schemas.
pub(crate) fn render_tools(tool_functions: &[Value]) -> String {
    let tool_schemas = tool_functions
        .iter()
        .map(stringify_python_style)
        .collect::<Vec<_>>()
        .join("\n");
    TOOLS_TEMPLATE
        .replace("{dsml}", DSML_SP_TOKEN)
        .replace("{block}", TOOL_CALLS_BLOCK_NAME)
        .replace("{invoke}", TOOL_CALL_TAG_NAME)
        .replace("{parameter}", TOOL_PARAMETER_TAG_NAME)
        .replace("{tool_schemas}", &tool_schemas)
}

fn tool_call_template(name: &str, arguments: &str) -> String {
    format!(
        "<{DSML_SP_TOKEN}{TOOL_CALL_TAG_NAME} name=\"{name}\">\n{arguments}\n</{DSML_SP_TOKEN}{TOOL_CALL_TAG_NAME}>"
    )
}

fn tool_calls_template(tool_calls: &str) -> String {
    format!(
        "<{DSML_SP_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{tool_calls}\n</{DSML_SP_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"
    )
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        other => stringify_python_style(other),
    }
}

fn type_label(value: &Value) -> String {
    match value.get("type") {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::String(label)) => label.clone(),
        Some(other) => python_repr(other),
    }
}

fn render_block(block: &Value) -> String {
    let object = block.as_object();
    let block_type = object
        .and_then(|object| object.get("type"))
        .and_then(Value::as_str);
    match block_type {
        Some("text") => object
            .and_then(|object| object.get("text"))
            .map(python_repr)
            .unwrap_or_default(),
        Some("tool_result") => {
            let content = object
                .and_then(|object| object.get("content"))
                .map(|content| {
                    if let Some(items) = content.as_array() {
                        items
                            .iter()
                            .map(|item| {
                                if item.get("type").and_then(Value::as_str) == Some("text") {
                                    item.get("text").map(python_repr).unwrap_or_default()
                                } else {
                                    format!("[Unsupported {}]", type_label(item))
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    } else {
                        python_repr(content)
                    }
                })
                .unwrap_or_default();
            format!("<tool_result>{content}</tool_result>")
        }
        Some(other) => format!("[Unsupported {other}]"),
        None => "[Unsupported None]".to_owned(),
    }
}

fn task_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some("<｜action｜>"),
        "query" => Some("<｜query｜>"),
        "authority" => Some("<｜authority｜>"),
        "domain" => Some("<｜domain｜>"),
        "title" => Some("<｜title｜>"),
        "read_url" => Some("<｜read_url｜>"),
        _ => None,
    }
}

pub(crate) fn encode_arguments_to_dsml(tool_call: &OpenAiToolCall) -> String {
    let mut arguments = tool_call.function.arguments.clone();
    if !arguments.is_object() {
        for _ in 0..2 {
            if let Value::String(text) = &arguments {
                match serde_json::from_str::<Value>(text) {
                    Ok(parsed) => arguments = parsed,
                    Err(_) => break,
                }
            } else {
                break;
            }
        }
        if !arguments.is_object() {
            arguments = json!({ "arguments": tool_call.function.arguments.clone() });
        }
    }
    let Some(object) = arguments.as_object() else {
        return String::new();
    };
    let mut parameters = Vec::with_capacity(object.len());
    for (key, value) in object {
        let rendered = match value.as_str() {
            Some(text) => format!(
                "<{DSML_SP_TOKEN}{TOOL_PARAMETER_TAG_NAME} name=\"{key}\" string=\"true\">{text}</{DSML_SP_TOKEN}{TOOL_PARAMETER_TAG_NAME}>"
            ),
            None => format!(
                "<{DSML_SP_TOKEN}{TOOL_PARAMETER_TAG_NAME} name=\"{key}\" string=\"false\">{}</{DSML_SP_TOKEN}{TOOL_PARAMETER_TAG_NAME}>",
                stringify_python_style(value)
            ),
        };
        parameters.push(rendered);
    }
    parameters.join("\n")
}

fn message_content_text(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(text)) => text.clone(),
        Some(MessageContent::Blocks(blocks)) => blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .map(|block| {
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        None => String::new(),
    }
}

fn find_last_user_index(messages: &[EncodingMessage]) -> i64 {
    for index in (0..messages.len()).rev() {
        let role = messages[index].role.as_str();
        if role == "user" || (role == "system" && index > 0) {
            return index as i64;
        }
    }
    -1
}

/// Merge tool messages into the preceding user message, mirroring
/// merge_tool_messages in the reference encoder.
pub(crate) fn merge_tool_messages(messages: &[EncodingMessage]) -> Vec<EncodingMessage> {
    let mut merged: Vec<EncodingMessage> = Vec::new();
    for message in messages {
        let mut message = message.clone();
        match message.role.as_str() {
            "tool" => {
                let tool_block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
                    "content": match &message.content {
                        Some(MessageContent::Text(text)) => Value::String(text.clone()),
                        Some(MessageContent::Blocks(blocks)) => Value::Array(blocks.clone()),
                        None => Value::String(String::new()),
                    },
                });
                let append = merged
                    .last()
                    .is_some_and(|last| last.role == "user" && last.content_blocks.is_some());
                if append {
                    if let Some(last) = merged.last_mut()
                        && let Some(blocks) = last.content_blocks.as_mut()
                    {
                        blocks.push(tool_block);
                    }
                } else {
                    merged.push(EncodingMessage {
                        role: "user".to_owned(),
                        content_blocks: Some(vec![tool_block]),
                        ..EncodingMessage::default()
                    });
                }
            }
            "user" => {
                let blocks = match message.content_blocks.take() {
                    Some(blocks) => blocks,
                    None => vec![json!({
                        "type": "text",
                        "text": message_content_text(&message.content),
                    })],
                };
                let append = merged.last().is_some_and(|last| {
                    last.role == "user" && last.content_blocks.is_some() && last.task.is_none()
                });
                if append {
                    if let Some(last) = merged.last_mut()
                        && let Some(existing) = last.content_blocks.as_mut()
                    {
                        existing.extend(blocks);
                    }
                } else {
                    message.content_blocks = Some(blocks);
                    merged.push(message);
                }
            }
            _ => merged.push(message),
        }
    }
    merged
}

/// Sort tool_result blocks by the order of the preceding assistant tool calls.
pub(crate) fn sort_tool_results_by_call_order(messages: &mut [EncodingMessage]) {
    let mut last_tool_call_order: HashMap<String, usize> = HashMap::new();
    for message in messages.iter_mut() {
        match message.role.as_str() {
            "assistant" => {
                if let Some(tool_calls) = message
                    .tool_calls
                    .as_ref()
                    .filter(|calls| !calls.is_empty())
                {
                    last_tool_call_order.clear();
                    for (index, tool_call) in tool_calls.iter().enumerate() {
                        let id = tool_call
                            .id
                            .clone()
                            .or_else(|| tool_call.function.id.clone())
                            .unwrap_or_default();
                        if !id.is_empty() {
                            last_tool_call_order.insert(id, index);
                        }
                    }
                }
            }
            "user" => {
                if let Some(blocks) = message.content_blocks.as_mut() {
                    let tool_indexes: Vec<usize> = blocks
                        .iter()
                        .enumerate()
                        .filter(|(_, block)| {
                            block.get("type").and_then(Value::as_str) == Some("tool_result")
                        })
                        .map(|(index, _)| index)
                        .collect();
                    if tool_indexes.len() > 1 && !last_tool_call_order.is_empty() {
                        let mut tools: Vec<Value> = tool_indexes
                            .iter()
                            .map(|&index| blocks[index].clone())
                            .collect();
                        tools.sort_by_key(|block| {
                            last_tool_call_order
                                .get(
                                    block
                                        .get("tool_use_id")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default(),
                                )
                                .copied()
                                .unwrap_or(0)
                        });
                        for (&index, tool) in tool_indexes.iter().zip(tools) {
                            blocks[index] = tool;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn drop_thinking_messages(messages: &[EncodingMessage]) -> Vec<EncodingMessage> {
    let last_user_index = find_last_user_index(messages);
    let mut result = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let role = message.role.as_str();
        let keep_role = matches!(
            role,
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
        );
        if keep_role || index as i64 >= last_user_index {
            result.push(message.clone());
        } else if role == "assistant" {
            let mut stripped = message.clone();
            stripped.reasoning_content = None;
            result.push(stripped);
        }
    }
    result
}

fn render_message(
    index: usize,
    messages: &[EncodingMessage],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    effort_score: Option<u8>,
) -> Result<String, CaseError> {
    let message = &messages[index];
    let last_user_index = find_last_user_index(messages);
    let role = message.role.as_str();
    let content = message_content_text(&message.content);
    let tool_functions: Vec<&Value> = message
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(|tool| &tool.function).collect())
        .unwrap_or_default();
    let tool_calls: &[OpenAiToolCall] = message.tool_calls.as_deref().unwrap_or_default();
    let reasoning_content = message.reasoning_content.clone().unwrap_or_default();
    let wo_eos = message.wo_eos;

    let effort_prompt = render_reasoning_effort(index, thinking_mode, effort_score);
    let mut prompt = if index == 0 && (!effort_prompt.is_empty() || role == "system") {
        SYSTEM_SP_TOKEN.to_owned()
    } else {
        String::new()
    };
    prompt.push_str(&effort_prompt);

    match role {
        "system" => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP_TOKEN);
            }
            prompt.push_str(&content);
            if !tool_functions.is_empty() {
                let functions: Vec<Value> = tool_functions
                    .iter()
                    .map(|value| (*value).clone())
                    .collect();
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(&functions));
            }
            if let Some(response_format) = &message.response_format {
                prompt.push_str("\n\n");
                prompt.push_str(
                    &RESPONSE_FORMAT_TEMPLATE
                        .replace("{schema}", &stringify_python_style(response_format)),
                );
            }
        }
        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            match message
                .content_blocks
                .as_ref()
                .filter(|blocks| !blocks.is_empty())
            {
                Some(blocks) => {
                    let parts: Vec<String> = blocks.iter().map(render_block).collect();
                    prompt.push_str(&parts.join("\n\n"));
                }
                None => prompt.push_str(&content),
            }
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(&content);
        }
        "tool" => return Err(CaseError::UnmergedToolMessage),
        "assistant" => {
            let mut thinking_part = String::new();
            let mut tool_calls_content = String::new();
            if !tool_calls.is_empty() {
                let rendered: Vec<String> = tool_calls
                    .iter()
                    .map(|tool_call| {
                        tool_call_template(
                            &tool_call.function.name,
                            &encode_arguments_to_dsml(tool_call),
                        )
                    })
                    .collect();
                tool_calls_content.push_str("\n\n");
                tool_calls_content.push_str(&tool_calls_template(&rendered.join("\n")));
            }
            let prev_has_task = index > 0 && messages[index - 1].task.is_some();
            if thinking_mode == ThinkingMode::Thinking
                && !prev_has_task
                && (!drop_thinking || index as i64 > last_user_index)
            {
                thinking_part.push_str(&reasoning_content);
                thinking_part.push_str(THINKING_END_TOKEN);
            }
            prompt.push_str(&thinking_part);
            prompt.push_str(&content);
            prompt.push_str(&tool_calls_content);
            if !wo_eos {
                prompt.push_str(EOS_TOKEN);
            }
        }
        other => return Err(CaseError::UnknownRole(other.to_owned())),
    }

    if index + 1 < messages.len() {
        let next_role = messages[index + 1].role.as_str();
        if next_role != "assistant" && next_role != "latest_reminder" {
            return Ok(prompt);
        }
    }

    if let Some(task) = message.task.as_deref() {
        let token = task_token(task).ok_or_else(|| CaseError::InvalidTask(task.to_owned()))?;
        if task != "action" {
            prompt.push_str(token);
        } else {
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if thinking_mode == ThinkingMode::Thinking {
                THINKING_START_TOKEN
            } else {
                THINKING_END_TOKEN
            });
            prompt.push_str(token);
        }
    } else if role == "user" || (role == "system" && index > 0) {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        let opens_thinking = thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || index as i64 >= last_user_index);
        prompt.push_str(if opens_thinking {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        });
    }

    Ok(prompt)
}

/// Render preprocessed messages into a prompt string, mirroring the reference
/// _encode_messages_text.
///
/// # Errors
///
/// Returns `CaseError` for unknown roles, unmerged tool messages, or invalid
/// task tokens.
pub(crate) fn encode_messages_text(
    messages: &[EncodingMessage],
    thinking_mode: ThinkingMode,
    context: Option<&[EncodingMessage]>,
    drop_thinking: bool,
    add_default_bos_token: bool,
    effort_score: Option<u8>,
) -> Result<String, CaseError> {
    let context = context.unwrap_or_default();

    let mut messages = merge_tool_messages(messages);
    {
        let mut combined: Vec<EncodingMessage> = context.to_vec();
        combined.extend(messages);
        sort_tool_results_by_call_order(&mut combined);
        messages = combined.split_off(context.len());
    }
    let merged_context = if context.is_empty() {
        Vec::new()
    } else {
        let mut merged_context = merge_tool_messages(context);
        sort_tool_results_by_call_order(&mut merged_context);
        merged_context
    };

    let mut full_messages: Vec<EncodingMessage> =
        Vec::with_capacity(merged_context.len() + messages.len());
    full_messages.extend(merged_context.iter().cloned());
    full_messages.extend(messages.iter().cloned());

    let mut prompt = if add_default_bos_token && merged_context.is_empty() {
        BOS_TOKEN.to_owned()
    } else {
        String::new()
    };

    let mut effective_drop_thinking = drop_thinking;
    if full_messages.iter().any(|message| {
        message
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    }) {
        effective_drop_thinking = false;
    }

    let (num_to_render, context_len);
    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        let dropped_context_len = drop_thinking_messages(&merged_context).len();
        full_messages = drop_thinking_messages(&full_messages);
        num_to_render = full_messages.len().saturating_sub(dropped_context_len);
        context_len = full_messages.len() - num_to_render;
    } else {
        num_to_render = messages.len();
        context_len = merged_context.len();
    }

    for index in 0..num_to_render {
        prompt.push_str(&render_message(
            index + context_len,
            &full_messages,
            thinking_mode,
            effective_drop_thinking,
            effort_score,
        )?);
    }

    Ok(prompt)
}

/// Render a single message for tests and callers that need one message only.
///
/// # Errors
///
/// Propagates `render_message` errors.
pub fn render_single_message(
    messages: &[EncodingMessage],
    index: usize,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    effort_score: Option<u8>,
) -> Result<String, CaseError> {
    render_message(index, messages, thinking_mode, drop_thinking, effort_score)
}
