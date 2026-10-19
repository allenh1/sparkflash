//! Chunked prefill is the same function as unchunked prefill, bit for bit, on the host f32 path.
//!
//! task-40(b) makes chunked prefill the scheduler's prefill policy and needs the host invariant
//! before any device launch sequence is claimed: feeding a prompt to `InferenceEngine::prefill` in
//! contiguous chunks on one `SequenceState` must produce exactly the logits, head input and next
//! token of one whole-prompt prefill. "Exactly" is the requirement, not "close": if the chunk
//! boundaries changed the arithmetic at all, a device chunked-prefill launch could not be validated
//! against a whole-prompt host reference.
//!
//! The chunk sizes walk across the tiny configuration's attention window, so the comparison also
//! covers the window wrap inside a chunk and across a chunk boundary.

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};
use sparkflash_engine::tensor::TensorPlan;

mod common;

use common::FiniteSource;

/// An engine over the plan of `config`, driven by the finite source at `seed`.
fn engine(config: &ModelConfig, seed: u64) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), 1 << 22)
        .expect("the tiny configuration is usable")
}

#[test]
fn chunked_prefill_is_bitwise_identical_to_one_call() {
    let config = ModelConfig::tiny();
    let window = config.window_size;
    // Lengths below, at, and past the window: 3 tokens is one chunk for every size tested, the
    // others cross the ring wrap inside a chunk and across the last chunk boundary.
    let lengths = [
        1usize,
        2,
        3,
        5,
        window,
        window + 1,
        window + 3,
        2 * window + 1,
    ];
    for len in lengths {
        let tokens: Vec<u32> = (1..=len as u32).collect();
        let reference = engine(&config, 7)
            .prefill(&mut SequenceState::new(&config), &tokens)
            .expect("the reference prefill runs");
        // Every chunk size from one token to the whole prompt.
        for chunk in 1..=len {
            let engine = engine(&config, 7);
            let mut state = SequenceState::new(&config);
            let mut last = None;
            for part in tokens.chunks(chunk) {
                last =
                    Some(engine.prefill(&mut state, part).unwrap_or_else(|error| {
                        panic!("chunk prefill of {len} in {chunk}s: {error}")
                    }));
            }
            let got = last.expect("a non-empty prompt has at least one chunk");
            assert_eq!(
                got.logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "logits differ for {len} tokens prefilled in chunks of {chunk} (window {window})"
            );
            assert_eq!(
                got.head_input
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .head_input
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "head input differs for {len} tokens prefilled in chunks of {chunk}"
            );
            assert_eq!(
                got.next_token, reference.next_token,
                "next token differs for {len} tokens prefilled in chunks of {chunk}"
            );
            // The state itself has to agree for the next step too: same tokens, same position.
            assert_eq!(state.tokens(), tokens);
        }
    }
}

#[test]
fn a_single_chunk_is_the_un_chunked_path() {
    let config = ModelConfig::tiny();
    let tokens: Vec<u32> = (1..=config.window_size as u32 + 2).collect();
    let whole = engine(&config, 11)
        .prefill(&mut SequenceState::new(&config), &tokens)
        .expect("prefill");
    let engine = engine(&config, 11);
    let mut state = SequenceState::new(&config);
    let once = engine.prefill(&mut state, &tokens).expect("prefill");
    assert_eq!(
        once.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        whole.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        once.head_input
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        whole
            .head_input
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(once.next_token, whole.next_token);
}
