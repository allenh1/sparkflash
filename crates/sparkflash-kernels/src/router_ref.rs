//! The CPU reference twin of the device MoE router (kernels/moe_router_sm121.cu).
//!
//! The host path this twinned is crates/sparkflash-engine/src/forward/moe.rs (`Routing::run`,
//! model.py:809-827), and the arithmetic below is a transcription of it, not an approximation:
//! the tests compare the two bit for bit on the released configuration and on the engine's tiny
//! one, and the released score function is the only one this twin implements because it is the
//! only one the kernel implements.
//!
//! Released semantics, in order:
//!
//! 1. `z[e] = sum_k x[k] * gate[e][k] / gate_temp`, accumulated in ascending k with one f32
//!    rounding per multiply and one per add (the engine's `linalg::matmul` serial loop for this
//!    geometry). `gate_temp` is 1.0, so the division is exact but is still performed.
//! 2. `score[e] = sqrt(softplus(z[e]))`, with `softplus(z) = z > 20 ? z : exp(z).ln_1p()`.
//! 3. Selection by `score[e] + bias[e]`, six passes, strict `>` over ascending e, so a tie keeps
//!    the lower expert id. The bias decides only WHICH experts; the weights are the unbiased
//!    scores (model.py:818-823).
//! 4. `weights /= (sum + 1e-20)` when `norm_topk_prob` and top-k > 1, summing in selection order,
//!    then `weights *= route_scale` (1.5). Two roundings per weight, in that order.
//!
//! What this twin is for. The device kernel is compared against it on the released layer-0 gate
//! bytes by examples/m9_router.rs; using one reference for both the host comparison and the
//! device comparison is what makes a device-vs-host delta attributable to the device rather than
//! to two disagreeing host transcriptions. The twin's own fidelity is tested against the engine
//! module itself (`the_twin_matches_the_engine_on_the_released_configuration` and
//! `the_twin_matches_the_engine_at_tiny_shapes`).
//!
//! Equality claim: on any input, `router_ref::route_f32` produces the same selected ids and the
//! same weight bits as `sparkflash_engine::forward::moe::route` for a configuration with
//! `score_func == "sqrtsoftplus"` and the same n_routed/top_k/gate_temp/norm_topk_prob/
//! route_scale. That is tested, not assumed, at both the released and tiny shapes. The claim is
//! about the CPU arithmetic only; the device kernel's score function uses CUDA's `expf`/`log1pf`
//! and is therefore NOT bit-exact against this twin (examples/m9_router.rs declares the bar).

/// Routed experts in the released configuration (config.json n_routed_experts).
pub const N_ROUTED: usize = 384;
/// Hidden width in the released configuration (config.json hidden_size).
pub const HIDDEN: usize = 5120;
/// Activated experts per token in the released configuration (config.json num_experts_per_tok).
pub const TOP_K: usize = 6;
/// The released gate temperature. Public so a test can pin that the released config uses it.
pub const GATE_TEMP: f32 = 1.0;
/// PyTorch's F.softplus threshold (model.py:816-817).
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;
/// norm_topk_prob's deliberately-not-norm_eps denominator floor (model.py:825).
pub const NORM_EPS: f32 = 1e-20;
/// The routed scaling factor applied after normalisation (model.py:826).
pub const ROUTE_SCALE: f32 = 1.5;

/// The router parameters the twin reads. A struct rather than constants so a test can run the
/// engine's tiny configuration through the same code; the released values are [RouterParams::RELEASED].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterParams {
    /// Hidden width of one token and one gate row.
    pub hidden: usize,
    /// Number of routed experts.
    pub n_routed: usize,
    /// Experts activated per token.
    pub top_k: usize,
    /// The gate temperature: scores are divided by it before the score function.
    pub gate_temp: f32,
    /// Added to the top-k score sum before the division (1e-20, not the model's norm epsilon).
    pub norm_eps: f32,
    /// Multiplied into the normalised weights once, after normalisation.
    pub scale: f32,
    /// When true and top_k > 1, the weights are divided by their sum + norm_eps.
    pub norm_topk_prob: bool,
}

impl RouterParams {
    /// The released DeepSeek-V4.1-Flash router configuration.
    pub const RELEASED: Self = Self {
        hidden: HIDDEN,
        n_routed: N_ROUTED,
        top_k: TOP_K,
        gate_temp: GATE_TEMP,
        norm_eps: NORM_EPS,
        scale: ROUTE_SCALE,
        norm_topk_prob: true,
    };
}

/// One token's routing decision, with the unbiased score vector kept for diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterOutput {
    /// `[n_routed]` unbiased sqrtsoftplus scores, in expert order.
    pub scores: Vec<f32>,
    /// `[top_k]` selected expert ids, in descending selection-signal order.
    pub ids: Vec<u32>,
    /// `[top_k]` weights, already normalised and scaled by `route_scale`.
    pub weights: Vec<f32>,
}

/// `softplus(z)` with beta 1 and PyTorch's threshold 20 (model.py:816-817).
#[must_use]
pub fn softplus(z: f32) -> f32 {
    if z > SOFTPLUS_THRESHOLD {
        z
    } else {
        z.exp().ln_1p()
    }
}

/// The released score function: `sqrt(softplus(z))` (model.py:817).
#[must_use]
pub fn sqrtsoftplus(z: f32) -> f32 {
    softplus(z).sqrt()
}

/// Round-trips one BF16 bit pattern to the exact f32 value, without going through a float decode
/// that could round twice. `half::bf16` already widens exactly; this helper exists so both the
/// twin and the device harness spell the same widening and a test can pin it.
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

/// The gate projection for one token: `z[e] = sum_k x[k] * gate[e][k] / gate_temp`, ascending k.
///
/// # Errors
/// Returns an error when the slice lengths do not match the declared geometry.
pub fn gate_scores(x: &[f32], gate: &[f32], params: RouterParams) -> Result<Vec<f32>, String> {
    if x.len() != params.hidden {
        return Err(format!(
            "the router needs a {}-value token, got {}",
            params.hidden,
            x.len()
        ));
    }
    if gate.len() != params.n_routed * params.hidden {
        return Err(format!(
            "{} experts over {} values need {} gate values, got {}",
            params.n_routed,
            params.hidden,
            params.n_routed * params.hidden,
            gate.len()
        ));
    }
    let mut scores = Vec::with_capacity(params.n_routed);
    for expert in 0..params.n_routed {
        let row = &gate[expert * params.hidden..(expert + 1) * params.hidden];
        let mut acc = 0.0f32;
        for k in 0..params.hidden {
            acc += x[k] * row[k];
        }
        scores.push(acc / params.gate_temp);
    }
    Ok(scores)
}

/// `Gate.forward` for one token (model.py:809-827), against f32 operands.
///
/// `gate` is `[n_routed][hidden]` row-major, already widened from BF16. `bias` is the selection
/// bias the caller chose for this token: `gate.bias` for a text token, `gate.bias_vl` for an
/// image-span token (the caller owns the image mask, exactly as the engine's `route` does).
///
/// # Errors
/// Returns an error when the slice lengths do not match the declared geometry, when top-k exceeds
/// the expert count, or when the gate temperature is not positive and finite (the engine refuses
/// the same configuration).
pub fn route_f32(
    x: &[f32],
    gate: &[f32],
    bias: &[f32],
    params: RouterParams,
) -> Result<RouterOutput, String> {
    if params.top_k > params.n_routed {
        return Err(format!(
            "top-k {} picks more experts than the {} that exist",
            params.top_k, params.n_routed
        ));
    }
    if bias.len() != params.n_routed {
        return Err(format!(
            "{} experts need {} biases, got {}",
            params.n_routed,
            params.n_routed,
            bias.len()
        ));
    }
    if !params.gate_temp.is_finite() || params.gate_temp <= 0.0 {
        return Err(format!(
            "the gate temperature divides the scores, and {} is not usable",
            params.gate_temp
        ));
    }
    let z = gate_scores(x, gate, params)?;
    let scores: Vec<f32> = z.into_iter().map(sqrtsoftplus).collect();
    Ok(select(&scores, bias, params))
}

/// The same routing decision from BF16 operands, the form the device ABI consumes.
///
/// # Errors
/// As [route_f32], plus a length mismatch on `x_bf16` or `gate_bf16`.
pub fn route_bf16(
    x_bf16: &[u16],
    gate_bf16: &[u16],
    bias: &[f32],
    params: RouterParams,
) -> Result<RouterOutput, String> {
    if x_bf16.len() != params.hidden {
        return Err(format!(
            "the router needs a {}-value token, got {}",
            params.hidden,
            x_bf16.len()
        ));
    }
    if gate_bf16.len() != params.n_routed * params.hidden {
        return Err(format!(
            "{} experts over {} values need {} gate values, got {}",
            params.n_routed,
            params.hidden,
            params.n_routed * params.hidden,
            gate_bf16.len()
        ));
    }
    let x: Vec<f32> = x_bf16.iter().map(|bits| bf16_to_f32(*bits)).collect();
    let gate: Vec<f32> = gate_bf16.iter().map(|bits| bf16_to_f32(*bits)).collect();
    route_f32(&x, &gate, bias, params)
}

/// The selection and weighting half of `Gate.forward`, shared by both operand forms.
///
/// This is a transcription of the engine's loop, including its starting `best_signal` of
/// negative infinity: the first expert is accepted when `best` is still `None` regardless of its
/// signal, so an all-`-inf`/NaN score row is resolved the same way the engine resolves it.
fn select(scores: &[f32], bias: &[f32], params: RouterParams) -> RouterOutput {
    let mut ids: Vec<u32> = Vec::with_capacity(params.top_k);
    for _ in 0..params.top_k {
        let mut best: Option<usize> = None;
        let mut best_signal = f32::NEG_INFINITY;
        for expert in 0..params.n_routed {
            if ids.contains(&(expert as u32)) {
                continue;
            }
            let signal = scores[expert] + bias[expert];
            if best.is_none() || signal > best_signal {
                best = Some(expert);
                best_signal = signal;
            }
        }
        match best {
            Some(expert) => ids.push(expert as u32),
            None => break,
        }
    }
    let mut weights: Vec<f32> = ids.iter().map(|id| scores[*id as usize]).collect();
    if params.norm_topk_prob && weights.len() > 1 {
        let sum: f32 = weights.iter().sum();
        let denominator = sum + params.norm_eps;
        for weight in weights.iter_mut() {
            *weight /= denominator;
        }
    }
    for weight in weights.iter_mut() {
        *weight *= params.scale;
    }
    RouterOutput {
        scores: scores.to_vec(),
        ids,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random f32 in [-scale, scale): an LCG over the low bits, so a test
    /// does not depend on a rand crate and two runs on two hosts agree bit for bit.
    fn pseudo_random(seed: &mut u64, scale: f32) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((*seed >> 40) as f32) / ((1u64 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }

    fn released_inputs(tokens: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let mut seed = 0x5EED_1234_ABCD_0001u64;
        let gate: Vec<f32> = (0..N_ROUTED * HIDDEN)
            .map(|_| pseudo_random(&mut seed, 0.02))
            .collect();
        let bias: Vec<f32> = (0..N_ROUTED)
            .map(|_| pseudo_random(&mut seed, 0.5))
            .collect();
        let x: Vec<Vec<f32>> = (0..tokens)
            .map(|_| (0..HIDDEN).map(|_| pseudo_random(&mut seed, 1.0)).collect())
            .collect();
        (x, gate, bias)
    }

    fn engine_reference(
        cfg: &sparkflash_engine::config::ModelConfig,
        prefix: &str,
        x: &[f32],
        gate: &[f32],
        bias: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let mut selected = Vec::new();
        sparkflash_engine::forward::moe::route(cfg, prefix, x, gate, bias, None, &mut selected)
            .expect("the engine reference must accept the geometry");
        let ids = selected.iter().map(|(expert, _)| *expert as u32).collect();
        let weights = selected.iter().map(|(_, weight)| *weight).collect();
        (ids, weights)
    }

    #[test]
    fn the_twin_matches_the_engine_on_the_released_configuration() {
        let cfg = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let params = RouterParams {
            hidden: cfg.dim,
            n_routed: cfg.n_routed_experts,
            top_k: cfg.n_activated_experts,
            gate_temp: cfg.gate_temp as f32,
            norm_eps: NORM_EPS,
            scale: cfg.route_scale as f32,
            norm_topk_prob: cfg.norm_topk_prob,
        };
        // The released cached config must be exactly the router this module hard-codes, including
        // the defaults that config.json does not carry (gate_temp 1.0, norm_topk_prob true).
        assert_eq!(
            params,
            RouterParams::RELEASED,
            "the released config's router parameters must equal the released constants"
        );
        let (x, gate, bias) = released_inputs(2);
        for token in &x {
            let engine = engine_reference(cfg, "layers.0", token, &gate, &bias);
            let twin = route_f32(token, &gate, &bias, params)
                .expect("the twin must accept the same geometry");
            assert_eq!(
                (
                    twin.ids.clone(),
                    twin.weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
                ),
                (
                    engine.0,
                    engine.1.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
                ),
                "the twin and the engine host path must agree bit for bit"
            );
        }
    }

    #[test]
    fn the_twin_matches_the_engine_at_tiny_shapes() {
        let cfg = sparkflash_engine::config::ModelConfig::tiny();
        let hidden = cfg.dim;
        let n_routed = cfg.n_routed_experts;
        // tiny(): dim 64, 4 routed experts, 2 activated per token, sqrtsoftplus, route_scale 1.5.
        assert_eq!((hidden, n_routed), (64, 4));
        let params = RouterParams {
            hidden,
            n_routed,
            top_k: cfg.n_activated_experts,
            gate_temp: 1.0,
            norm_eps: NORM_EPS,
            scale: cfg.route_scale as f32,
            norm_topk_prob: cfg.norm_topk_prob,
        };
        let mut seed = 0x7135_A000_0001_0001u64;
        let gate: Vec<f32> = (0..n_routed * hidden)
            .map(|_| pseudo_random(&mut seed, 0.1))
            .collect();
        let bias: Vec<f32> = (0..n_routed)
            .map(|_| pseudo_random(&mut seed, 0.2))
            .collect();
        let x: Vec<f32> = (0..hidden).map(|_| pseudo_random(&mut seed, 1.0)).collect();
        let engine = engine_reference(&cfg, "layers.0", &x, &gate, &bias);
        let twin = route_f32(&x, &gate, &bias, params).expect("tiny geometry must fit");
        assert_eq!(twin.ids, engine.0);
        assert_eq!(
            twin.weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            engine.1.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_tie_keeps_the_lower_expert_id() {
        // x = 0 makes every score sqrtsoftplus(0) = sqrt(ln 2); a zero bias then ties all 384
        // experts, and the released top-6 must be experts 0..5 in ascending order.
        let params = RouterParams::RELEASED;
        let x = vec![0.0f32; params.hidden];
        let gate = vec![0.0f32; params.n_routed * params.hidden];
        let bias = vec![0.0f32; params.n_routed];
        let out = route_f32(&x, &gate, &bias, params).expect("the zero geometry must fit");
        assert_eq!(out.ids, vec![0, 1, 2, 3, 4, 5]);
        // softplus(0) = ln 2 through this crate's own function, and sqrt is IEEE exact-rounding.
        let expected = softplus(0.0).sqrt();
        for score in &out.scores {
            assert_eq!(*score, expected, "all 384 zero rows must score identically");
        }
        // The normaliser is a fold over the six selected scores in selection order, so the expected
        // weight is the same fold, not 6.0 * expected (which rounds differently).
        let sum = (0..params.top_k).fold(0.0f32, |acc, _| acc + expected);
        let expected_weight = expected / (sum + NORM_EPS) * ROUTE_SCALE;
        for weight in &out.weights {
            assert_eq!(
                *weight, expected_weight,
                "identical inputs must give identical weights"
            );
        }
    }

    #[test]
    fn the_bias_steers_selection_and_not_the_weights() {
        let params = RouterParams::RELEASED;
        let mut seed = 0xB1A5_0000_0001u64;
        let gate: Vec<f32> = (0..params.n_routed * params.hidden)
            .map(|_| pseudo_random(&mut seed, 0.02))
            .collect();
        let x: Vec<f32> = (0..params.hidden)
            .map(|_| pseudo_random(&mut seed, 1.0))
            .collect();
        let plain = vec![0.0f32; params.n_routed];
        let unbiased = route_f32(&x, &gate, &plain, params).expect("geometry fits");
        let far = (0..params.n_routed)
            .find(|expert| !unbiased.ids.contains(&(*expert as u32)))
            .expect("with 384 experts the unbiased selection leaves 378 of them out");
        let mut steered = plain.clone();
        steered[far] = 1.0e6;
        let biased = route_f32(&x, &gate, &steered, params).expect("geometry fits");
        assert_eq!(
            biased.ids[0], far as u32,
            "a +1e6 selection bias must win the first slot for an expert the scores left out"
        );
        // The weights are the unbiased scores: the same expert has the same raw score in both runs,
        // and both weight vectors are the same normalisation of those raw scores.
        assert_eq!(
            biased.scores[far], unbiased.scores[far],
            "the selection bias must not enter the score vector"
        );
        let at_plain = unbiased.ids.iter().position(|id| *id == far as u32);
        assert!(
            at_plain.is_none(),
            "the steered expert was already selected"
        );
        let raw = biased.scores[far];
        let sum = biased
            .ids
            .iter()
            .fold(0.0f32, |acc, id| acc + biased.scores[*id as usize]);
        let expected = raw / (sum + NORM_EPS) * ROUTE_SCALE;
        assert_eq!(
            biased.weights[0], expected,
            "the weight must be the raw score normalised and scaled, not the selection signal"
        );
    }

    #[test]
    fn norm_topk_prob_false_leaves_the_raw_scores_scaled() {
        // A tiny geometry keeps this test instant while exercising the same branch. Every expert
        // scores identically here, so each weight must be exactly raw_score * route_scale.
        let params = RouterParams {
            hidden: 64,
            n_routed: 4,
            top_k: 2,
            norm_topk_prob: false,
            ..RouterParams::RELEASED
        };
        let x = vec![0.1f32; params.hidden];
        let gate = vec![0.01f32; params.n_routed * params.hidden];
        let bias = vec![0.0f32; params.n_routed];
        let out = route_f32(&x, &gate, &bias, params).expect("geometry fits");
        let raw = out.scores[0];
        assert!(raw > 0.0);
        for weight in &out.weights {
            assert_eq!(
                *weight,
                raw * ROUTE_SCALE,
                "without normalisation the weight is the raw score times the route scale"
            );
        }
        // And the same geometry with normalisation on does not equal it: the branch matters.
        let normalised = route_f32(
            &x,
            &gate,
            &bias,
            RouterParams {
                norm_topk_prob: true,
                ..params
            },
        )
        .expect("geometry fits");
        assert_ne!(out.weights, normalised.weights);
    }

    #[test]
    fn the_geometry_is_refused_instead_of_silently_reshaped() {
        let params = RouterParams {
            hidden: 8,
            n_routed: 4,
            top_k: 2,
            ..RouterParams::RELEASED
        };
        let bias = vec![0.0f32; 4];
        assert!(route_f32(&[0.0; 7], &[0.0; 32], &bias, params).is_err());
        assert!(route_f32(&[0.0; 8], &[0.0; 31], &bias, params).is_err());
        assert!(route_f32(&[0.0; 8], &[0.0; 32], &[0.0; 3], params).is_err());
        let too_many = RouterParams { top_k: 5, ..params };
        assert!(route_f32(&[0.0; 8], &[0.0; 32], &bias, too_many).is_err());
        let bad_temp = RouterParams {
            gate_temp: 0.0,
            ..params
        };
        assert!(route_f32(&[0.0; 8], &[0.0; 32], &bias, bad_temp).is_err());
    }

    #[test]
    fn the_bf16_operand_form_is_the_exact_widening_of_the_f32_form() {
        let params = RouterParams {
            hidden: 16,
            n_routed: 4,
            top_k: 2,
            ..RouterParams::RELEASED
        };
        let mut seed = 0xBF16_0000_0001u64;
        let x: Vec<f32> = (0..params.hidden)
            .map(|_| pseudo_random(&mut seed, 1.0))
            .collect();
        let gate: Vec<f32> = (0..params.n_routed * params.hidden)
            .map(|_| pseudo_random(&mut seed, 0.1))
            .collect();
        let bias = vec![0.0f32; params.n_routed];
        let x_bits: Vec<u16> = x
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect();
        let gate_bits: Vec<u16> = gate
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect();
        // The f32 side of the comparison widens with an INDEPENDENT expression - the bit shift that
        // defines BF16 as the top half of an f32 - not with the twin's own bf16_to_f32, because
        // otherwise "the two entry points agree" would only restate that both call the same
        // widening. The helper's agreement with that expression is asserted over every one of the
        // 65,536 BF16 bit patterns, below.
        let x_widened: Vec<f32> = x_bits.iter().map(|bits| independent_widen(*bits)).collect();
        let gate_widened: Vec<f32> = gate_bits
            .iter()
            .map(|bits| independent_widen(*bits))
            .collect();
        let from_f32 = route_f32(&x_widened, &gate_widened, &bias, params).expect("fits");
        let from_bf16 = route_bf16(&x_bits, &gate_bits, &bias, params).expect("fits");
        assert_eq!(from_f32, from_bf16);
        // A spot check of the released score function's defining relation: sqrt(softplus(z)).
        let z = 1.0f32;
        let score = sqrtsoftplus(z);
        assert!((score * score - z.exp().ln_1p()).abs() < 1e-6);
    }

    /// The BF16 -> F32 widening written out from the bit layout: a BF16 is the top sixteen bits of
    /// an F32. This is deliberately NOT `half::bf16` and not the twin's `bf16_to_f32`: it is the
    /// independent expression the twin's helper is checked against.
    fn independent_widen(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    #[test]
    fn the_twins_widening_matches_the_independent_bit_shift_on_every_bfloat16_pattern() {
        // Exhaustive over the 16-bit pattern space, so this is not a sample of lucky rows. NaN
        // patterns are compared as NaN: the bit shift leaves a signalling NaN signalling, and a
        // NaN-aware comparison is the only correct one there.
        for bits in 0..=u16::MAX {
            let widened = bf16_to_f32(bits);
            let reference = independent_widen(bits);
            if reference.is_nan() {
                assert!(
                    widened.is_nan(),
                    "bits {bits:#06x} widened to {widened}, not NaN"
                );
            } else {
                assert_eq!(
                    widened.to_bits(),
                    reference.to_bits(),
                    "bits {bits:#06x} widened differently from the bit shift"
                );
            }
        }
        // A signalling-NaN pattern is the case a bit shift and a conversion library are most likely
        // to disagree on; both must still widen to a NaN.
        assert!(
            bf16_to_f32(0x7F81).is_nan(),
            "sNaN pattern must widen to NaN"
        );
        assert!(independent_widen(0x7F81).is_nan());
    }
}
