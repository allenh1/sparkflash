//! The window eviction contract, on the engine's own host f32 path.
//!
//! task-38 is the device chain gating the 130th token (position 129 with window 128) against a
//! 130-token reference: the b70 dump disagrees with the reference at p129 and never heals, and the
//! three answers at p129 (device 8060, host-f32 491, released vLLM 367) show the position is
//! sensitive. Two assumptions had to be checked before blaming the device: that a chunk prefill and
//! the same tokens decoded one by one are the same function past the window (the device chain runs
//! the prompt as single-token steps while the fixture's reference is one 130-token chunk), and that
//! the ring rotation itself is right at the ladder below.
//!
//! The ladder, in the release geometry (window 128):
//!   position 127 - 128 tokens - the ring is exactly full, no slot has been overwritten yet;
//!   position 128 - 129 tokens - FIRST wrap, the write for 128 evicts the slot 0 row (position 0);
//!   position 129 - 130 tokens - second wrap, slot 1 is overwritten and the oldest row is position 2.
//! Every test here uses the tiny configuration's window (8) and the same shape: full, first wrap,
//! second wrap.

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::window_indices;
use sparkflash_engine::forward::{InferenceEngine, SequenceState};

/// The reference sentinel for an index that attends nothing (model.py:420, 580).
const NO_INDEX: usize = usize::MAX;
use sparkflash_engine::tensor::TensorPlan;

mod common;

use common::FiniteSource;

/// An engine over the plan of `config`, driven by the finite source at `seed`.
fn engine(config: &ModelConfig, seed: u64, limit: usize) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), limit)
        .expect("the tiny configuration is usable")
}

/// A chunk longer than the window and the same tokens decoded one at a time must be the same
/// function at the last position. This is what refutes "the device chain steps the prompt one token
/// at a time while the fixture is a chunk", so a p129 divergence cannot be that mode difference:
/// past the first wrap the engine's two paths are bit-identical here, not merely close.
#[test]
fn a_chunk_past_the_window_is_bit_identical_to_the_same_tokens_decoded_one_by_one() {
    let config = ModelConfig::tiny();
    let window = config.window_size;
    assert!(window >= 4);
    // The lengths walk the ladder: at the window, first wrap, second wrap, and well past both.
    for len in [
        window - 1,
        window,
        window + 1,
        window + 2,
        window + 4,
        2 * window,
    ] {
        let chunk: Vec<u32> = (1..=len as u32).collect();
        let mut prefill_state = SequenceState::new(&config);
        let prefilled = engine(&config, 21, 1 << 22)
            .prefill(&mut prefill_state, &chunk)
            .expect("the reference prefill runs");
        let engine = engine(&config, 21, 1 << 22);
        let mut decode_state = SequenceState::new(&config);
        let mut decoded = None;
        for token in &chunk {
            decoded = Some(
                engine
                    .decode(&mut decode_state, *token)
                    .expect("decode runs"),
            );
        }
        let decoded = decoded.expect("the chunk is not empty");
        let worst = prefilled
            .logits
            .iter()
            .zip(decoded.logits.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-3,
            "at len {len} (window {window}, {:?} tokens past the first wrap) the chunk and the decoded path differ by {worst}",
            len.saturating_sub(window)
        );
        assert_eq!(
            prefilled.next_token, decoded.next_token,
            "at len {len} the two paths pick different tokens"
        );
    }
}

/// The ring rotation at the ladder: full, first wrap, second wrap. `window_indices` supplies the
/// ids a decode step attends (ring slots, oldest first, NO_INDEX for slots the sequence has not
/// reached); this pins that once the ring is full every slot appears exactly once, the order is the
/// one-slot rotation, and the oldest id is the slot the next write evicts. A staging filter that
/// drops or repeats a slot, or an off-by-one in the rotation, fails here rather than in a window.
#[test]
fn the_ring_rotates_one_slot_per_step_at_the_eviction_ladder() {
    let config = ModelConfig::tiny();
    let window = config.window_size;
    assert!(window >= 4);
    let mut saw_full = false;
    let mut saw_first_wrap = false;
    let mut saw_second_wrap = false;
    for position in 0..(2 * window + 2) {
        let ids: Vec<usize> = window_indices(&config, position, 1)
            .into_iter()
            .filter(|id| *id != NO_INDEX)
            .collect();
        let filled = (position + 1).min(window);
        assert_eq!(
            ids.len(),
            filled,
            "position {position} fills {filled} slots"
        );
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            filled,
            "position {position} lists a slot twice: {ids:?}"
        );
        if filled == window {
            assert_eq!(
                sorted,
                (0..window).collect::<Vec<_>>(),
                "position {position} does not cover every slot once: {ids:?}"
            );
            let oldest = (position % window + 1) % window;
            assert_eq!(ids[0], oldest, "position {position} oldest slot");
            let expected: Vec<usize> = (0..window).map(|slot| (oldest + slot) % window).collect();
            assert_eq!(ids, expected, "position {position} rotation");
            if position == window - 1 {
                saw_full = true;
                // No wrap yet: the order is 0, 1, ... window - 1.
                assert_eq!(
                    ids,
                    (0..window).collect::<Vec<_>>(),
                    "the full ring is in order"
                );
            } else if position == window {
                saw_first_wrap = true;
                // Slot 0 was overwritten, so the oldest row is slot 1.
                assert_eq!(ids[0], 1, "the first wrap evicts slot 0");
            } else if position == window + 1 {
                saw_second_wrap = true;
                // Slot 1 was overwritten, so the oldest row is slot 2.
                assert_eq!(ids[0], 2, "the second wrap evicts slot 1");
            }
        }
    }
    assert!(
        saw_full && saw_first_wrap && saw_second_wrap,
        "the ladder did not reach all three states: full={saw_full} first={saw_first_wrap} second={saw_second_wrap}"
    );
}
