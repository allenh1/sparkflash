//! The one-pass prompt-logprobs seam: selected position rows are the prefix rows, bit for bit.
//!
//! The old honest implementation fed a prompt one token at a time and read each position's
//! distribution from the step that consumed it: prefill the first token, then decode every later
//! token, O(n) forward passes with a growing prefix. The one-pass seam asks one forward chunk for
//! the distributions of selected positions, so a caller that feeds the prompt in contiguous spans
//! pays one forward per token instead of one per prefix length.
//!
//! The rows must be the same values: this file compares the selected rows, and the final step
//! output, against the token-at-a-time reference over every chunk size from one token to the whole
//! prompt. "Same" is exact f32 bits, matching the existing chunked-prefill invariant
//! (`chunked_prefill.rs`).

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::tensor::TensorPlan;

mod common;

use common::FiniteSource;

fn engine(config: &ModelConfig, seed: u64) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), 1 << 22)
        .expect("the tiny configuration is usable")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The token-at-a-time reference: the rows the old prompt_logprobs path read, one per scored
/// position, and the step the last token left behind.
fn token_at_a_time(
    engine: &InferenceEngine,
    config: &ModelConfig,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, sparkflash_engine::forward::StepOutput) {
    let mut state = SequenceState::new(config);
    let mut feed = engine
        .prefill(&mut state, &tokens[..1])
        .expect("the first token prefills");
    let mut rows = vec![feed.logits.clone()];
    for &token in &tokens[1..] {
        feed = engine
            .decode(&mut state, token)
            .expect("the next token decodes");
        rows.push(feed.logits.clone());
    }
    rows.truncate(tokens.len() - 1);
    (rows, feed)
}

/// The one-pass reference: contiguous spans of `span` tokens, every span asked for the rows that
/// score the tokens after it, the last span asked for every row except the final generation
/// position.
fn one_pass(
    engine: &InferenceEngine,
    config: &ModelConfig,
    tokens: &[u32],
    span: usize,
) -> (Vec<Vec<f32>>, sparkflash_engine::forward::StepOutput) {
    let mut state = SequenceState::new(config);
    let mut rows = Vec::new();
    let mut step = None;
    let mut start = 0;
    while start < tokens.len() {
        let end = (start + span).min(tokens.len());
        let chunk = &tokens[start..end];
        let rows_needed = chunk.len() - usize::from(end == tokens.len());
        let positions: Vec<usize> = (0..rows_needed).collect();
        let produced = engine
            .prefill_with_position_logits(&mut state, chunk, &positions)
            .expect("the selected-row prefill runs");
        rows.extend(produced.rows);
        step = Some(produced.step);
        start = end;
    }
    (
        rows,
        step.expect("a non-empty prompt has at least one span"),
    )
}

fn assert_rows_equal(got: &[Vec<f32>], expected: &[Vec<f32>], what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: row count differs from the token-at-a-time reference"
    );
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            bits(got),
            bits(expected),
            "{what}: row {index} differs from the token-at-a-time reference"
        );
    }
}

#[test]
fn selected_position_rows_match_the_prefix_reforward_bit_for_bit() {
    let config = ModelConfig::tiny();
    let node_window = config.window_size;
    let lengths = [2usize, 3, 5, node_window, node_window + 1, node_window + 3];
    for len in lengths {
        let tokens: Vec<u32> = (1..=len as u32).collect();
        let (expected_rows, expected_step) = token_at_a_time(&engine(&config, 7), &config, &tokens);
        // Every span size from one token to the whole prompt, including sizes that cross the
        // attention window and sizes that leave a one-token final span.
        for span in 1..=len {
            let (got_rows, got_step) = one_pass(&engine(&config, 7), &config, &tokens, span);
            assert_rows_equal(
                &got_rows,
                &expected_rows,
                &format!("{len} tokens in spans of {span}"),
            );
            assert_eq!(
                bits(&got_step.logits),
                bits(&expected_step.logits),
                "{len} tokens in spans of {span}: the final logits differ"
            );
            assert_eq!(
                bits(&got_step.head_input),
                bits(&expected_step.head_input),
                "{len} tokens in spans of {span}: the final head input differs"
            );
            assert_eq!(
                got_step.next_token, expected_step.next_token,
                "{len} tokens in spans of {span}: the next token differs"
            );
            assert_eq!(
                got_step.consumed,
                len - ((len - 1) / span) * span,
                "the final span consumed only its own tokens"
            );
        }
    }
}

#[test]
fn a_selected_position_is_the_same_row_as_the_full_collection() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=6).collect();
    assert!(tokens.len() > 1);
    let (all_rows, _) = one_pass(&engine(&config, 7), &config, &tokens, tokens.len());
    let selected = [0usize, 2, 4];
    let engine = engine(&config, 7);
    let mut state = SequenceState::new(&config);
    let produced = engine
        .prefill_with_position_logits(&mut state, &tokens, &selected)
        .expect("the selected-row prefill runs");
    assert_eq!(produced.rows.len(), selected.len());
    for (row, &position) in produced.rows.iter().zip(&selected) {
        assert_eq!(
            bits(row),
            bits(&all_rows[position]),
            "asking for position {position} alone must return the same row"
        );
    }
    assert_eq!(state.tokens(), tokens, "the whole chunk was consumed");
}

#[test]
fn an_out_of_range_position_is_refused_before_the_pass() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=4).collect();
    let engine = engine(&config, 7);
    let mut state = SequenceState::new(&config);
    let error = engine
        .prefill_with_position_logits(&mut state, &tokens, &[0, tokens.len()])
        .expect_err("a position outside the chunk cannot be served from this pass");
    let message = error.to_string();
    assert!(
        message.contains("outside the 4 token prefill chunk"),
        "the refusal names the chunk and the position: {message}"
    );
    assert_eq!(
        state.tokens().len(),
        0,
        "the refusal happens before the pass advances the sequence"
    );
}

#[test]
fn an_empty_selection_returns_the_plain_prefill_step() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=4).collect();
    let plain = engine(&config, 7)
        .prefill(&mut SequenceState::new(&config), &tokens)
        .expect("prefill");
    let engine = engine(&config, 7);
    let mut state = SequenceState::new(&config);
    let produced = engine
        .prefill_with_position_logits(&mut state, &tokens, &[])
        .expect("an empty selection is a plain prefill");
    assert!(produced.rows.is_empty());
    assert_eq!(bits(&produced.step.logits), bits(&plain.logits));
    assert_eq!(bits(&produced.step.head_input), bits(&plain.head_input));
    assert_eq!(produced.step.next_token, plain.next_token);
}
