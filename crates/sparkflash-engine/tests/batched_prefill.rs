//! Batched/chunked prefill: one forward per token-budget chunk, every token written once.
//!
//! task-40(b) made chunked prefill the scheduler's policy and proved whole-prompt equivalence.
//! This file proves the *batched execution contract* the scheduler plans against: a prompt is
//! forwarded as one engine call per chunk (never one call per token on the batched path), the
//! outputs are bit-identical to both the whole-prompt call and the pre-batch-17 token loop, and
//! a forked state that already holds a cached prefix forwards only the uncached suffix.

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::batch::{BatchedPrefill, token_budget_ranges};
use sparkflash_engine::forward::{InferenceEngine, SequenceState, StepOutput};
use sparkflash_engine::tensor::TensorPlan;

mod common;

use common::FiniteSource;

/// An engine over the plan of config, driven by the finite source at seed.
fn engine(config: &ModelConfig, seed: u64) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), 1 << 22)
        .expect("the tiny configuration is usable")
}

/// The first difference between two step outputs, or None when they are bit equal.
///
/// Two fields are deliberately excluded, both by construction rather than by tolerance:
///
/// * consumed: the whole-prompt call consumes the prompt and the last chunk consumes only its
///   own range.
/// * draft and confidence: the reference drafts nothing for a forward that starts a sequence
///   (forward/mod.rs: the forward_spec anchor is None at position 0), so a whole-prompt prefill
///   has no draft while a batched or token-loop prefill's later chunks do. Logits, head input
///   and the sampled next token stay identical; the draft is a proposal that the verification
///   step owns. the_draft_tokens_differ_by_construction_when_the_chunk_starts_at_zero pins it.
fn output_difference(left: &StepOutput, right: &StepOutput) -> Option<String> {
    if left.next_token != right.next_token {
        return Some(format!(
            "next_token {:?} != {:?}",
            left.next_token, right.next_token
        ));
    }
    if left.logits.len() != right.logits.len() {
        return Some(format!(
            "logits length {} != {}",
            left.logits.len(),
            right.logits.len()
        ));
    }
    for (index, (a, b)) in left.logits.iter().zip(&right.logits).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!("logits[{index}] {a} != {b}"));
        }
    }
    if left.head_input.len() != right.head_input.len() {
        return Some(format!(
            "head_input length {} != {}",
            left.head_input.len(),
            right.head_input.len()
        ));
    }
    for (index, (a, b)) in left.head_input.iter().zip(&right.head_input).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Some(format!("head_input[{index}] {a} != {b}"));
        }
    }
    None
}

fn same_output(left: &StepOutput, right: &StepOutput) -> bool {
    output_difference(left, right).is_none()
}

#[test]
fn the_draft_tokens_differ_by_construction_when_the_chunk_starts_at_zero() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=6u32).collect();
    let handle = engine(&config, 5);
    let mut whole = SequenceState::new(&config);
    let whole_out = handle.prefill(&mut whole, &tokens).expect("whole prompt");
    assert!(
        whole_out.draft.is_empty(),
        "the reference drafts nothing for a forward that starts the sequence"
    );
    let mut batched = SequenceState::new(&config);
    let (batched_out, _) = BatchedPrefill::new(&handle)
        .prefill_token_budget(&mut batched, &tokens, 2, 8)
        .expect("batched prefill");
    assert!(
        !batched_out.draft.is_empty(),
        "a later chunk drafts after its last position"
    );
    assert_eq!(whole_out.next_token, batched_out.next_token);
    assert_eq!(
        whole_out
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        batched_out
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_batched_path_is_bit_identical_to_one_call_and_to_the_token_loop() {
    let config = ModelConfig::tiny();
    let window = config.window_size;
    for len in [
        1usize,
        2,
        3,
        5,
        window,
        window + 1,
        window + 3,
        2 * window + 1,
    ] {
        let tokens: Vec<u32> = (1..=len as u32).collect();
        let reference = engine(&config, 7)
            .prefill(&mut SequenceState::new(&config), &tokens)
            .expect("reference prefill");
        for chunk in 1..=len {
            let handle = engine(&config, 7);
            let mut state = SequenceState::new(&config);
            let (out, run) = BatchedPrefill::new(&handle)
                .prefill_token_budget(&mut state, &tokens, chunk, chunk)
                .unwrap_or_else(|error| panic!("batched prefill of {len} in {chunk}s: {error}"));
            assert!(
                same_output(&reference, &out),
                "a {chunk}-token chunk of a {len}-token prompt changed the output: {}",
                output_difference(&reference, &out).expect("a difference")
            );
            assert_eq!(run.chunks, len.div_ceil(chunk));
            assert_eq!(run.forwards, len.div_ceil(chunk), "one forward per chunk");
            assert_eq!(run.tokens_written, len, "every prompt token written once");
            assert_eq!(state.len(), len);

            let handle = engine(&config, 7);
            let mut loop_state = SequenceState::new(&config);
            let (loop_out, loop_run) = BatchedPrefill::new(&handle)
                .token_loop(&mut loop_state, &tokens)
                .expect("token-loop prefill");
            assert!(
                same_output(&reference, &loop_out),
                "the compatibility token loop must be bit-identical for {len} tokens"
            );
            assert_eq!(
                loop_run.forwards, len,
                "the token loop is one forward per token"
            );
            assert_eq!(loop_run.tokens_written, len);
        }
    }
}

#[test]
fn the_batched_plan_is_not_a_per_token_loop_even_with_short_chunks() {
    let config = ModelConfig::tiny();
    let handle = engine(&config, 11);
    let len = 11usize;
    let tokens: Vec<u32> = (1..=len as u32).collect();
    let mut state = SequenceState::new(&config);
    let plan = BatchedPrefill::new(&handle);
    let (_, run) = plan
        .prefill_token_budget(&mut state, &tokens, 4, 8)
        .expect("batched prefill");
    assert_eq!(
        run.forwards, 3,
        "11 tokens at 4 per chunk is three forwards"
    );
    assert_eq!(run.chunks, 3);
    assert_ne!(
        run.forwards, len,
        "the batched path must not forward per token"
    );

    let mut loop_state = SequenceState::new(&config);
    let (_, loop_run) = plan
        .token_loop(&mut loop_state, &tokens)
        .expect("token loop");
    assert_eq!(loop_run.forwards, len);

    assert_eq!(
        token_budget_ranges(0, len, 4, 8).expect("ranges"),
        vec![0..4, 4..8, 8..11]
    );
}

#[test]
fn a_forked_state_forwards_only_the_uncached_suffix_and_stays_bit_identical() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=9u32).collect();
    let reference_engine = engine(&config, 7);
    let mut whole_state = SequenceState::new(&config);
    let reference = reference_engine
        .prefill(&mut whole_state, &tokens)
        .expect("whole prompt prefill");

    // Prefix caching at the reference level: compute the cached prefix once, fork the KV, and
    // run only the suffix through the batched executor.
    let handle = engine(&config, 7);
    let mut cached = SequenceState::new(&config);
    handle
        .prefill(&mut cached, &tokens[..5])
        .expect("prefix prefill");
    assert_eq!(cached.len(), 5);
    let mut forked = cached.fork();
    assert_eq!(
        forked.tokens(),
        &tokens[..5],
        "the fork reuses the cached KV"
    );
    let (out, run) = BatchedPrefill::new(&handle)
        .prefill_token_budget(&mut forked, &tokens, 2, 8)
        .expect("suffix prefill");
    assert!(
        same_output(&reference, &out),
        "the forked suffix prefill differs from the whole prompt: {}",
        output_difference(&reference, &out).expect("a difference")
    );
    assert_eq!(run.forwards, 2, "9 - 5 suffix tokens at 2 per chunk");
    assert_eq!(run.tokens_written, 4, "the cached prefix is not rewritten");
    assert_eq!(cached.len(), 5, "the donor state is untouched");

    // The forked state and a one-shot state must decode identically afterwards.
    let mut other = whole_state;
    let left = handle.decode(&mut forked, 77).expect("decode on the fork");
    let right = handle
        .decode(&mut other, 77)
        .expect("decode on the whole-prompt state");
    assert!(same_output(&left, &right));
}

#[test]
fn a_state_that_is_not_the_prompt_prefix_is_refused() {
    let config = ModelConfig::tiny();
    let handle = engine(&config, 3);
    let plan = BatchedPrefill::new(&handle);
    let mut wrong = SequenceState::new(&config);
    handle
        .prefill(&mut wrong, &[90, 91, 92])
        .expect("a different prefix");
    let prompt: Vec<u32> = (1..=6u32).collect();
    assert!(
        plan.prefill_token_budget(&mut wrong, &prompt, 2, 8)
            .is_err()
    );
    let mut fresh = SequenceState::new(&config);
    assert!(plan.prefill_token_budget(&mut fresh, &[], 2, 8).is_err());
    let mut fresh = SequenceState::new(&config);
    assert!(
        plan.prefill_token_budget(&mut fresh, &[1], 0, 8).is_err(),
        "a zero chunk is refused"
    );
    let mut fresh = SequenceState::new(&config);
    assert!(
        plan.prefill_token_budget(&mut fresh, &[1], 4, 2).is_err(),
        "a chunk larger than the budget is refused, not silently split"
    );
}

#[test]
fn ranges_must_start_at_the_consumed_position_and_cover_the_prompt() {
    let config = ModelConfig::tiny();
    let handle = engine(&config, 3);
    let plan = BatchedPrefill::new(&handle);
    let prompt: Vec<u32> = (1..=8u32).collect();
    let mut state = SequenceState::new(&config);
    handle.prefill(&mut state, &prompt[..3]).expect("prefix");
    assert!(plan.prefill_ranges(&mut state, &prompt, &[]).is_err());
    assert!(
        plan.prefill_ranges(&mut state, &prompt, std::slice::from_ref(&(0..5)))
            .is_err()
    );
    assert!(
        plan.prefill_ranges(&mut state, &prompt, &[3..5, 6..8])
            .is_err()
    );
    assert!(
        plan.prefill_ranges(&mut state, &prompt, &[3..8, 8..8])
            .is_err()
    );
    assert!(
        plan.prefill_ranges(&mut state, &prompt, std::slice::from_ref(&(3..6)))
            .is_err()
    );
    let (_, run) = plan
        .prefill_ranges(&mut state, &prompt, &[3..5, 5..6, 6..8])
        .expect("contiguous coverage");
    assert_eq!(run.forwards, 3);
    assert_eq!(run.tokens_written, 5);
}

#[test]
fn the_range_planner_refuses_impossible_shapes() {
    assert!(token_budget_ranges(0, 0, 4, 8).is_err());
    assert!(token_budget_ranges(5, 5, 4, 8).is_err());
    assert!(token_budget_ranges(0, 9, 0, 8).is_err());
    assert!(token_budget_ranges(0, 9, 8, 4).is_err());
    assert!(token_budget_ranges(0, 9, 4, 0).is_err());
    assert_eq!(
        token_budget_ranges(3, 8, 2, 8).expect("ranges"),
        vec![3..5, 5..7, 7..8]
    );
}
