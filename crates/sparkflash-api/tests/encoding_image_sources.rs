//! Pins the image-source semantics of render_case against the Python reference.
//!
//! Every expected prompt and image list in this file was produced by running
//! reference/encoding/encoding.py with return_multi_modal_data=True. The
//! reference returns the CURRENT TURN's images only
//! (processed_context, _ = process_image_messages(context)) and never renders
//! context messages into the prompt, so a context image contributes neither
//! tokens nor an entry to image_sources.

use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use serde_json::{Value, json};
use sparkflash_model_deepseek_v41::encoding::{SparkV41Encoding, parse_cases};

const BOS: &str = "<｜begin▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const IMAGE: &str = "<｜deepseek_image｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const EFFORT_50: &str =
    "Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

const CONTEXT_IMAGE_URL: &str = "https://example.com/a.png";
const CURRENT_IMAGE_URL: &str = "https://example.com/b.png";

fn context_with_image() -> Value {
    json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": "ctx text" },
            { "type": "image_url", "image_url": { "url": CONTEXT_IMAGE_URL } }
        ]
    }])
}

fn context_without_image() -> Value {
    json!([{ "role": "user", "content": "ctx text" }])
}

fn messages_with_image() -> Value {
    json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": "current text" },
            { "type": "image_url", "image_url": { "url": CURRENT_IMAGE_URL } }
        ]
    }])
}

fn messages_without_image() -> Value {
    json!([{ "role": "user", "content": "current text" }])
}

fn render(mode: &str, context: Option<Value>, messages: Value) -> (String, Vec<ImageSource>) {
    let mut case = json!({ "thinking_mode": mode, "messages": messages });
    if let Some(context) = context {
        case["context"] = context;
    }
    let cases = parse_cases(&case).unwrap_or_else(|error| panic!("case parses: {error}"));
    let rendered = SparkV41Encoding::new()
        .render_case(&cases[0], None)
        .unwrap_or_else(|error| panic!("case renders: {error}"));
    (rendered.prompt, rendered.image_sources)
}

fn current_image() -> Vec<ImageSource> {
    vec![ImageSource::Url {
        url: CURRENT_IMAGE_URL.to_owned(),
        detail: ImageDetail::High,
    }]
}

struct Shape {
    name: &'static str,
    mode: &'static str,
    context: fn() -> Value,
    context_is_some: bool,
    messages: fn() -> Value,
    expected_prompt: fn() -> String,
    expected_images: fn() -> Vec<ImageSource>,
}

fn chat_a() -> String {
    format!("{USER}current text\n\n{IMAGE}{ASSISTANT}{THINK_END}")
}
fn chat_b() -> String {
    format!("{USER}current text{ASSISTANT}{THINK_END}")
}
fn chat_c() -> String {
    format!("{BOS}{USER}current text\n\n{IMAGE}{ASSISTANT}{THINK_END}")
}
fn chat_d() -> String {
    format!("{BOS}{USER}current text{ASSISTANT}{THINK_END}")
}
fn thinking_a() -> String {
    format!("{USER}current text\n\n{IMAGE}{ASSISTANT}{THINK_START}")
}
fn thinking_b() -> String {
    format!("{USER}current text{ASSISTANT}{THINK_START}")
}
fn thinking_c() -> String {
    format!("{BOS}{SYSTEM}{EFFORT_50}{USER}current text\n\n{IMAGE}{ASSISTANT}{THINK_START}")
}
fn thinking_d() -> String {
    format!("{BOS}{SYSTEM}{EFFORT_50}{USER}current text{ASSISTANT}{THINK_START}")
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "a_context_image_and_current_image",
            mode: "chat",
            context: context_with_image,
            context_is_some: true,
            messages: messages_with_image,
            expected_prompt: chat_a,
            expected_images: current_image,
        },
        Shape {
            name: "b_context_image_only",
            mode: "chat",
            context: context_with_image,
            context_is_some: true,
            messages: messages_without_image,
            expected_prompt: chat_b,
            expected_images: Vec::new,
        },
        Shape {
            name: "c_current_image_only",
            mode: "chat",
            context: context_without_image,
            context_is_some: false,
            messages: messages_with_image,
            expected_prompt: chat_c,
            expected_images: current_image,
        },
        Shape {
            name: "d_no_images",
            mode: "chat",
            context: context_without_image,
            context_is_some: false,
            messages: messages_without_image,
            expected_prompt: chat_d,
            expected_images: Vec::new,
        },
        Shape {
            name: "a_context_image_and_current_image",
            mode: "thinking",
            context: context_with_image,
            context_is_some: true,
            messages: messages_with_image,
            expected_prompt: thinking_a,
            expected_images: current_image,
        },
        Shape {
            name: "b_context_image_only",
            mode: "thinking",
            context: context_with_image,
            context_is_some: true,
            messages: messages_without_image,
            expected_prompt: thinking_b,
            expected_images: Vec::new,
        },
        Shape {
            name: "c_current_image_only",
            mode: "thinking",
            context: context_without_image,
            context_is_some: false,
            messages: messages_with_image,
            expected_prompt: thinking_c,
            expected_images: current_image,
        },
        Shape {
            name: "d_no_images",
            mode: "thinking",
            context: context_without_image,
            context_is_some: false,
            messages: messages_without_image,
            expected_prompt: thinking_d,
            expected_images: Vec::new,
        },
    ]
}

#[test]
fn image_sources_are_the_current_turn_only() {
    for shape in shapes() {
        let context = (shape.context)();
        let (prompt, images) = render(
            shape.mode,
            if shape.context_is_some {
                Some(context)
            } else {
                None
            },
            (shape.messages)(),
        );
        let label = format!("{} / {}", shape.name, shape.mode);
        assert_eq!(
            prompt,
            (shape.expected_prompt)(),
            "{label}: prompt mismatch"
        );
        assert_eq!(
            images,
            (shape.expected_images)(),
            "{label}: image_sources mismatch"
        );
    }
}

#[test]
fn context_images_are_not_rendered_into_the_prompt() {
    // Shape (b): the context carries an image and the current turn does not.
    let (prompt, images) = render("chat", Some(context_with_image()), messages_without_image());
    assert_eq!(prompt, chat_b());
    assert!(
        !prompt.contains(IMAGE),
        "a context image must not be rendered: {prompt:?}"
    );
    assert!(images.is_empty());

    let (prompt, images) = render(
        "thinking",
        Some(context_with_image()),
        messages_without_image(),
    );
    assert_eq!(prompt, thinking_b());
    assert!(!prompt.contains(IMAGE));
    assert!(images.is_empty());
}

#[test]
fn context_image_url_never_reaches_image_sources() {
    for mode in ["chat", "thinking"] {
        let (_, images) = render(mode, Some(context_with_image()), messages_with_image());
        assert_eq!(images, current_image(), "{mode}");
        assert!(
            !images.iter().any(|source| matches!(
                source,
                ImageSource::Url { url, .. } | ImageSource::DataUrl { data_url: url, .. }
                    if url == CONTEXT_IMAGE_URL
            )),
            "{mode}: context image leaked into image_sources"
        );
    }
}
