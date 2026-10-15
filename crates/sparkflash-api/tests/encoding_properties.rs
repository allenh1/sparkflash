//! Property tests for the encoding layer: retarget idempotency, total effort
//! resolution, and rendering determinism.

use std::path::PathBuf;

use deepseek_recipe_core::conversation::ReasoningEffort;
use serde_json::json;
use sparkflash_api::encoding::{
    ReasoningEffortError, SparkV41Encoding, load_first_case, parse_cases, resolve_reasoning_effort,
    resolve_thinking_from_request, retarget_reasoning_effort,
};

const BOS: &str = "<｜begin▁of▁sentence｜>";
const SYSTEM: &str = "<｜System｜>";
const EFFORT_SUFFIX: &str =
    " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

fn effort_prompt(score: u8, body: &str) -> String {
    format!("{BOS}{SYSTEM}Reasoning Effort: {score}{EFFORT_SUFFIX}{body}")
}

#[test]
fn retarget_is_idempotent_and_preserves_the_body() {
    let bodies = [
        "",
        "<｜User｜>hi<｜Assistant｜><think>",
        "Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>later",
        "prefix Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nmore",
    ];
    for score in [1u8, 25, 42, 50, 75, 100, 255] {
        for body in bodies {
            let prompt = effort_prompt(50, body);
            let once = retarget_reasoning_effort(prompt.clone(), score);
            let twice = retarget_reasoning_effort(once.clone(), score);
            assert_eq!(once, twice, "retarget was not idempotent for score {score}");
            assert!(
                once.ends_with(body),
                "retarget corrupted text after the effort line"
            );
            assert_eq!(
                retarget_reasoning_effort(prompt.clone(), 50),
                prompt,
                "retarget changed a matching budget"
            );
        }
    }
}

#[test]
fn retarget_leaves_prompts_without_a_leading_effort_line_untouched() {
    let prompts = [
        format!("{BOS}<｜User｜>hello<｜Assistant｜><think>"),
        format!("{BOS}{SYSTEM}no effort here"),
        format!("{BOS}{SYSTEM}<｜User｜>Reasoning Effort: 75{EFFORT_SUFFIX}x"),
        String::new(),
    ];
    for prompt in prompts {
        for score in [1u8, 50, 100] {
            assert_eq!(retarget_reasoning_effort(prompt.clone(), score), prompt);
        }
    }
    // The bare-system prefix (no BOS) is still retargeted.
    let bare = format!("{SYSTEM}Reasoning Effort: 75{EFFORT_SUFFIX}body");
    let retargeted = retarget_reasoning_effort(bare, 42);
    assert!(retargeted.starts_with(&format!("{SYSTEM}Reasoning Effort: 42{EFFORT_SUFFIX}")));
}

#[test]
fn effort_resolution_is_total() {
    let values = [
        None,
        Some(json!(null)),
        Some(json!(1)),
        Some(json!(42)),
        Some(json!(100)),
        Some(json!(0)),
        Some(json!(-5)),
        Some(json!(101)),
        Some(json!(1000)),
        Some(json!(1.5)),
        Some(json!(true)),
        Some(json!("low")),
        Some(json!("high")),
        Some(json!("xhigh")),
        Some(json!("max")),
        Some(json!("medium")),
        Some(json!("bogus")),
        Some(json!([])),
        Some(json!({})),
        Some(json!({ "reasoning_effort": 30 })),
    ];
    for value in &values {
        for thinking in [true, false] {
            let spec = resolve_reasoning_effort(value.as_ref(), thinking);
            assert_eq!(spec.thinking, thinking);
            if let Some(score) = spec.score {
                assert!((1..=100).contains(&score), "score out of range: {score}");
            }
            if !thinking {
                assert!(spec.score.is_none());
                assert!(spec.effort.is_none());
            }
            assert_eq!(
                spec.renders_effort_line(),
                spec.thinking && spec.score.is_some()
            );
        }
    }
}

#[test]
fn effort_resolution_maps_documented_values() {
    let low = resolve_reasoning_effort(Some(&json!("low")), true);
    assert_eq!(low.score, Some(50));
    assert_eq!(low.effort, Some(ReasoningEffort::Low));
    assert!(low.error.is_none());

    let high = resolve_reasoning_effort(Some(&json!("high")), true);
    assert_eq!(high.score, Some(75));
    assert_eq!(high.effort, Some(ReasoningEffort::High));

    let xhigh = resolve_reasoning_effort(Some(&json!("xhigh")), true);
    assert_eq!(xhigh.score, Some(75));
    assert_eq!(xhigh.effort, Some(ReasoningEffort::Xhigh));

    let max = resolve_reasoning_effort(Some(&json!("max")), true);
    assert_eq!(max.score, Some(100));
    assert_eq!(max.effort, Some(ReasoningEffort::Max));

    let absent = resolve_reasoning_effort(None, true);
    assert_eq!(absent.score, Some(75));
    assert!(absent.error.is_none());

    let numeric = resolve_reasoning_effort(Some(&json!(63)), true);
    assert_eq!(numeric.score, Some(63));
    assert!(numeric.effort.is_none());

    let object = resolve_reasoning_effort(Some(&json!({ "reasoning_effort": 30 })), true);
    assert_eq!(object.score, Some(30));

    let disabled = resolve_reasoning_effort(
        Some(&json!({ "enable_thinking": false, "reasoning_effort": 30 })),
        true,
    );
    assert!(!disabled.thinking);
    assert_eq!(disabled.score, None);

    let disabled_by_kwargs = resolve_reasoning_effort(
        Some(&json!({ "chat_template_kwargs": { "thinking": false }, "reasoning_effort": "max" })),
        true,
    );
    assert!(!disabled_by_kwargs.thinking);
    assert_eq!(disabled_by_kwargs.score, None);

    let too_low = resolve_reasoning_effort(Some(&json!(0)), true);
    assert_eq!(too_low.score, Some(1));
    assert_eq!(
        too_low.error,
        Some(ReasoningEffortError::OutOfRange { value: 0 })
    );

    let too_high = resolve_reasoning_effort(Some(&json!(101)), true);
    assert_eq!(too_high.score, Some(100));
    assert_eq!(
        too_high.error,
        Some(ReasoningEffortError::OutOfRange { value: 101 })
    );

    let unknown = resolve_reasoning_effort(Some(&json!("bogus")), true);
    assert_eq!(unknown.score, Some(75));
    assert_eq!(
        unknown.error,
        Some(ReasoningEffortError::UnknownAlias("bogus".to_owned()))
    );
}

#[test]
fn thinking_resolution_honours_request_flags() {
    assert!(resolve_thinking_from_request(
        &json!({ "enable_thinking": true }),
        false
    ));
    assert!(!resolve_thinking_from_request(
        &json!({ "enable_thinking": false }),
        true
    ));
    assert!(resolve_thinking_from_request(
        &json!({ "chat_template_kwargs": { "thinking": true } }),
        false
    ));
    assert!(!resolve_thinking_from_request(
        &json!({ "chat_template_kwargs": { "thinking": false } }),
        true
    ));
    assert!(!resolve_thinking_from_request(
        &json!({ "thinking": { "type": "disabled" } }),
        true
    ));
    assert!(resolve_thinking_from_request(
        &json!({ "thinking": { "type": "enabled" } }),
        false
    ));
    assert!(!resolve_thinking_from_request(
        &json!({ "reasoning_effort": "none" }),
        true
    ));
    assert!(resolve_thinking_from_request(&json!({}), true));
    assert!(!resolve_thinking_from_request(&json!({}), false));
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/encoding/tests")
}

#[test]
fn rendering_real_vectors_is_deterministic() {
    let encoding = SparkV41Encoding::default();
    for case_id in 1..=5 {
        let path = fixtures_dir().join(format!("test_input_{case_id}.json"));
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let case = load_first_case(&body).expect("parse case");
        let first = encoding.render_case(&case, None).expect("render").prompt;
        let second = encoding.render_case(&case, None).expect("render").prompt;
        assert_eq!(first, second, "vector {case_id} rendered differently twice");
    }
}

#[test]
fn synthetic_case_rendering_is_deterministic_and_effort_retargets() {
    let value = json!({
        "thinking_mode": "thinking",
        "messages": [
            { "role": "system", "content": "system" },
            { "role": "user", "content": "question" }
        ]
    });
    let cases = parse_cases(&value).expect("parse");
    let case = &cases[0];
    let encoding = SparkV41Encoding::default();
    let base = encoding.render_case(case, None).expect("render").prompt;
    let again = encoding.render_case(case, None).expect("render").prompt;
    assert_eq!(base, again);
    assert!(
        base.contains(&format!("{BOS}{SYSTEM}Reasoning Effort: 50{EFFORT_SUFFIX}")),
        "python default effort must be 50: {base}"
    );
    let retargeted = encoding.render_case(case, Some(88)).expect("render").prompt;
    assert!(retargeted.contains(&format!("{BOS}{SYSTEM}Reasoning Effort: 88{EFFORT_SUFFIX}")));
    assert_eq!(
        retarget_reasoning_effort(retargeted.clone(), 88),
        retargeted
    );
}
