//! The whole reference forward pass, end to end, with no weights and no GPU.
//!
//! Every property here is one the reference is supposed to have whatever the numbers are, and each
//! one is asserted at a configuration small enough to run in a test process: the shape flow, the
//! cache growing by exactly one token per decode step, attention never seeing a row it should not
//! (causality), the sliding window evicting what left it, greediness, determinism for a fixed seed,
//! and the element limit refusing a tensor the reference will not materialise.
//!
//! The weight source is [FiniteSource] below: the plan's own names and shapes, every tensor served
//! as little-endian f32 with small finite values. Nothing here needs the released checkpoint, and the
//! numbers are not the model's: a property that depends on the actual weights is out of reach on a
//! host without them and is not claimed.
//!
//! It is deliberately NOT [SyntheticWeights]. That source hands back raw bytes, which is honest for a
//! shape test - and documented as such - but raw bytes are not numbers: an E4M3 byte can decode to NaN
//! or to inf, and a pass fed those is a pass fed NaN. The engine test needs a source the engine can
//! actually do arithmetic on, while staying weight-free and deterministic (the seed chooses the
//! values, so a fixed seed is a fixed model and a different seed is a different one).

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::forward::{InferenceEngine, Sampling, SequenceState};
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// A finite, plan-shaped weight source: the plan's names and shapes, every tensor as finite f32.
///
/// The plan's dtype is reported as f32 for every tensor, which is exactly what [WeightSource::read]
/// writes, so a caller that decodes by the source's spec sees the values it was sent. Values are
/// small (0.01 ..= 0.09), deterministic, and vary with the seed and the tensor's name, so two seeds
/// are two different models and a mis-indexed read is visible rather than masked by uniform weights.
struct FiniteSource {
    specs: Vec<TensorSpec>,
    seed: u64,
}

impl FiniteSource {
    /// The source for `plan`, at `seed`.
    fn from_plan(plan: &TensorPlan, seed: u64) -> FiniteSource {
        FiniteSource {
            specs: plan.specs().to_vec(),
            seed,
        }
    }
}

/// A stable per-name offset, so two tensors of the same shape are not identical.
fn name_salt(name: &str) -> u64 {
    name.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u64::from(byte))
    })
}

impl WeightSource for FiniteSource {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(TensorSpec {
                dtype: DType::F32,
                ..spec.clone()
            }),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let elements = usize::try_from(self.spec(name)?.elements()).expect("a test tensor fits");
        let salt = name_salt(name);
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            // A per-element mix, deliberately not a per-row one: with the high bits of the index the
            // pattern would barely change across a row, two rows could produce exactly equal dot
            // products, and the argmax would then be a tie that hides a mis-indexed read.
            let mixed = (index as u64)
                .wrapping_add(salt)
                .wrapping_add(self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let hashed = (mixed ^ (mixed >> 13)).wrapping_mul(0x2545_f491_4f6c_dd1d);
            // Strictly positive and at least 0.01, so nothing divides by a zero scale, and 256 values
            // deep so neighbours differ.
            let value = 0.01 + 0.000_1 * (hashed >> 56) as f32;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// An engine over the plan of `config`, driven by the finite source at `seed`.
fn engine(config: &ModelConfig, seed: u64, limit: usize) -> InferenceEngine {
    let plan = TensorPlan::language_model(config);
    let source = FiniteSource::from_plan(&plan, seed);
    InferenceEngine::new(config.clone(), Box::new(source), limit)
        .expect("the tiny configuration is usable")
}

#[test]
fn a_prefill_step_returns_one_distribution_per_step_and_consumes_its_chunk() {
    let config = ModelConfig::tiny();
    let engine = engine(&config, 7, 1 << 22);
    let mut state = SequenceState::new(&config);
    let chunk: Vec<u32> = (1..=5).collect();
    let output = engine
        .prefill(&mut state, &chunk)
        .expect("the reference prefill runs");

    assert_eq!(
        output.logits.len(),
        config.vocab_size,
        "one distribution over the vocabulary"
    );
    assert_eq!(output.consumed, chunk.len());
    assert_eq!(
        state.len(),
        chunk.len(),
        "the sequence consumed the whole chunk"
    );
    assert_eq!(state.tokens(), chunk.as_slice());
    assert!(
        output.logits.iter().all(|value| value.is_finite()),
        "the logits are finite"
    );
    let next = output
        .next_token
        .expect("a step with logits chooses a token");
    assert!(
        (next as usize) < config.vocab_size,
        "{next} is inside the vocabulary"
    );
    // Greedy: the choice is the argmax of the distribution it returned.
    let argmax = output
        .logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).expect("finite"))
        .map(|(index, _)| index as u32);
    assert_eq!(Some(next), argmax, "the default sampler is greedy");
}

#[test]
fn the_same_seed_gives_the_same_logits_and_a_different_one_does_not() {
    let config = ModelConfig::tiny();
    let chunk: Vec<u32> = (1..=4).collect();
    let mut first = SequenceState::new(&config);
    let mut second = SequenceState::new(&config);
    let a = engine(&config, 11, 1 << 22)
        .prefill(&mut first, &chunk)
        .expect("runs");
    let b = engine(&config, 11, 1 << 22)
        .prefill(&mut second, &chunk)
        .expect("runs");
    assert_eq!(a.logits, b.logits, "a fixed seed is a fixed step");
    let mut third = SequenceState::new(&config);
    let c = engine(&config, 12, 1 << 22)
        .prefill(&mut third, &chunk)
        .expect("runs");
    assert_ne!(
        a.logits, c.logits,
        "a different seed is a different set of weights"
    );
}

#[test]
fn decode_consumes_exactly_one_token_and_grows_the_window_by_one() {
    let config = ModelConfig::tiny();
    let engine = engine(&config, 3, 1 << 22);
    let mut state = SequenceState::new(&config);
    let window = config.window_size;
    for step in 0..(2 * window + 3) {
        let before = state.len();
        let token = (step as u32) % config.vocab_size as u32;
        let output = engine
            .decode(&mut state, token)
            .expect("the reference decode runs");
        assert_eq!(
            state.len(),
            before + 1,
            "one decode step consumes one token"
        );
        assert_eq!(output.consumed, 1);
        assert_eq!(output.logits.len(), config.vocab_size);
        // Every layer wrote the position it just consumed, at the slot the ring maps it to.
        for layer in 0..config.n_layers {
            assert!(
                state.window_row(layer, state.len() - 1).is_some(),
                "layer {layer} holds the row it just wrote at position {}",
                state.len() - 1
            );
        }
    }
    // The window is bounded: a position that has left it is gone, not merely unused.
    let oldest = state.len() - 2 * window - 3;
    for layer in 0..config.n_layers {
        assert!(
            state.window_row(layer, oldest).is_none(),
            "layer {layer} evicted position {oldest}, which the window no longer covers"
        );
    }
}

#[test]
fn a_chunked_prefill_and_the_same_tokens_decoded_one_by_one_agree() {
    // The two paths are different code: one attends over the chunk it was handed, the other over the
    // rows already in the ring. Both are the same function of the tokens, so the last position must
    // reach the same distribution - up to the summation order of a float, which is why the tolerance
    // is not zero. This is the property that catches a mask or an offset that is only right in one of
    // the two paths, and it holds only while the chunk fits the window.
    let config = ModelConfig::tiny();
    assert!(config.n_layers > 0 && config.window_size >= 4);
    let chunk: Vec<u32> = vec![5, 6, 7, 8];
    let mut prefill_state = SequenceState::new(&config);
    let prefilled = engine(&config, 21, 1 << 22)
        .prefill(&mut prefill_state, &chunk)
        .expect("prefill runs");
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
    assert_eq!(prefill_state.tokens(), decode_state.tokens());
    let worst = prefilled
        .logits
        .iter()
        .zip(decoded.logits.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "the two paths differ by {worst} at the last position, which is more than float order explains"
    );
}

#[test]
fn the_element_limit_refuses_a_tensor_the_reference_will_not_materialise() {
    let config = ModelConfig::tiny();
    let mut state = SequenceState::new(&config);
    let error = engine(&config, 5, 8)
        .prefill(&mut state, &[1, 2])
        .expect_err("a limit of eight elements cannot hold the embedding table");
    match error {
        EngineError::TooLarge {
            name,
            elements,
            limit,
        } => {
            assert_eq!(limit, 8);
            assert!(elements > 8);
            assert!(!name.is_empty(), "the refusal names the tensor it refused");
        }
        other => panic!("the limit must refuse with TooLarge, got {other}"),
    }
}

#[test]
fn temperature_above_zero_draws_from_the_distribution_and_a_zero_one_is_greedy() {
    let config = ModelConfig::tiny();
    let engine = engine(&config, 9, 1 << 22).with_sampling(Sampling {
        temperature: 1.0,
        seed: 4,
    });
    let mut state = SequenceState::new(&config);
    let output = engine.prefill(&mut state, &[1, 2, 3]).expect("runs");
    let token = output.next_token.expect("a token is drawn");
    assert!((token as usize) < config.vocab_size);
    let greedy = sparkflash_engine::forward::sample_gumbel(&output.logits, 0.0, 4);
    let drawn_again = sparkflash_engine::forward::sample_gumbel(&output.logits, 1.0, 4);
    assert_eq!(
        drawn_again,
        Some(token),
        "the sampler is a function of the logits and the seed"
    );
    assert_eq!(
        greedy,
        output
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).expect("finite"))
            .map(|(i, _)| i as u32)
    );
}

/// The ATTENTION block's RMSNorm reads `config.norm_eps`, not `hc_eps` (M7).
///
/// Scope, stated plainly because the plural would be a claim this test does not make: the observable is
/// the window row the engine stores per layer, which is `attn_in` after the attention norm, so this pins
/// the attention block norm. The FFN block norm (the `ffn_norm` read in forward/mod.rs) is NOT pinned
/// here. Its scale is rescaled by the next layer's attention norm and by the hc mixing before it reaches
/// a stored row, so substituting hc_eps for it does not move the rows enough to cross the threshold:
/// mutating only that read leaves this test - and the rest of the suite - green. Pinning it needs a
/// white-box assertion on the FFN path inside forward/mod.rs, where `scratch.ffn_in` is visible.
///
/// The two fields are far apart in this configuration (norm_eps 1.0 against hc_eps 1e-6), and the normed
/// value is what the residual stream carries, so a large norm_eps shrinks every stored row by a wide
/// margin. hc_eps does not move those rows at all - changing it leaves them bit for bit equal - which is
/// exactly why a test that only looks at logits cannot see the difference: the final norm renormalises
/// whatever scale the layers produced, so the logits of the two configurations differ by a pure factor.
/// The stored window rows are the observable upstream of that final norm.
#[test]
fn the_attention_block_rms_norm_reads_the_configured_epsilon() {
    // Measured row RMS at the last consumed position, with this source and these tokens, layer 1:
    //   norm_eps 1e-20 (tiny), correct engine                        2.3356e-2
    //   norm_eps 1.0,   correct engine                               5.5755e-7  -> 41900x
    //   norm_eps 1.0,   the attention norm reading hc_eps instead    2.3331e-5  -> 1001x
    // Against the 5000x threshold that is about 8x of room and a 5x failure at layer 1, but layer 1 is
    // not the binding layer: the correct engine's layer 0 has about 21x of room (2.5655e-2 / 2.4324e-7)
    // and the mutated engine's layer 0 is 2544x (2.3356e-2 / 1.0082e-5), so a substitution fails first at
    // layer 0, with about 2x of margin. Every number here is a measurement.
    let sharp = ModelConfig {
        norm_eps: 1.0,
        ..ModelConfig::tiny()
    };
    let plain = ModelConfig::tiny();
    assert!(
        sharp.norm_eps > plain.norm_eps * 1e15,
        "the two epsilons must be far apart for this to measure anything"
    );
    assert_eq!(
        sharp.hc_eps, plain.hc_eps,
        "only norm_eps differs between the two runs"
    );

    let rows = |config: &ModelConfig| -> Vec<f32> {
        let mut state = SequenceState::new(config);
        engine(config, 7, 1 << 22)
            .prefill(&mut state, &[1, 2, 3])
            .expect("runs");
        let last = state.len() - 1;
        (0..config.n_layers)
            .map(|layer| {
                let row = state.window_row(layer, last).expect("a row per layer");
                let mean_square: f32 =
                    row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
                mean_square.sqrt()
            })
            .collect()
    };

    let shrunk = rows(&sharp);
    let reference = rows(&plain);
    for (layer, (small, large)) in shrunk.iter().zip(reference.iter()).enumerate() {
        assert!(
            *large > *small * 5_000.0,
            "layer {layer}: norm_eps 1.0 must shrink the stored row far below the 1e-20 row, got {small} against {large}"
        );
    }
}

#[test]
fn a_sequence_that_has_consumed_nothing_has_an_empty_cache_and_refuses_an_empty_prefill() {
    let config = ModelConfig::tiny();
    let engine = engine(&config, 1, 1 << 22);
    let mut state = SequenceState::new(&config);
    assert!(state.is_empty());
    assert_eq!(state.len(), 0);
    for layer in 0..config.n_layers {
        assert!(
            state.window_row(layer, 0).is_none(),
            "an empty cache holds nothing"
        );
    }
    let error = engine
        .prefill(&mut state, &[])
        .expect_err("an empty chunk is not a step");
    assert!(matches!(error, EngineError::Shape { .. }), "{error}");
}
