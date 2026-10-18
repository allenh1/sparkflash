//! CPU reference for the FlashMLA CSA2 kernel (kernels/flash_mla_csa2_sm121.cu).
//!
//! The kernel does two things, and this module defines both:
//!
//! 1. the HSI indexer: for each 8-token block of the 2,048-block pool, the dot product of the query
//!    against each of the block's 8 key vectors, keeping the best score and the token that scored it.
//!    Ties keep the earliest token, because the kernel compares with a strict greater-than;
//! 2. the attention itself: a three-pass softmax (maximum, normalising sum, weighted latent sum) over
//!    the selected candidates, with the sliding-window predicate applied exactly once per candidate.
//!
//! The window predicate is the load-bearing part. A candidate outside the 128-token window behind the
//! query is EXCLUDED - the kernel assigns it -FLT_MAX and the second pass skips it outright - rather
//! than being given a small weight. Its contribution is therefore exactly zero, and if every candidate
//! is out of the window the output is exactly zero rather than a division by a zero weight sum. Both
//! of those are asserted below, because "the window is applied" is exactly the kind of claim that
//! silently degrades into "the window is approximately applied".
//!
//! # The attention sink
//!
//! `layers.N.attn.attn_sink` is a real checkpoint tensor that the released attention uses: F32, one
//! learned logit per query head (`[64]`, so one element of it belongs to the single head this module
//! computes; the engine's own tensor plan pins that shape). The released implementation is not
//! vendored here - our architecture docs transcribe it: `sum_exp[i] += exp(attn_sink[i] -
//! scores_max[i])` after the online-softmax loop, then divide (docs/architecture/01-csa2-attention.md
//! section 7, from inference/kernel.py:382-383). Two decisions follow from that, and both are
//! deliberate:
//!
//! * the sink is a DENOMINATOR term only. It adds probability mass for a place attention can go that
//!   has no value vector, so it attenuates the candidate mixture instead of renormalising it; the
//!   direction of that mixture (the ratio between its channels) is untouched;
//! * it is added AFTER the candidate maximum rather than folded into it, because that is where the
//!   released code adds it. A sink far above the maximum therefore overflows the denominator to
//!   `+inf` and the output is exactly zero; that is the released arithmetic, and this reference does
//!   not re-stabilise the formula it exists to reproduce.
//!
//! The value is used as given: the checkpoint does not state its range, so nothing here clamps or
//! bounds it. [`attention_with_sink`] is the entry point, and [`attention`] is unchanged - it is
//! exactly [`attention_with_sink`] with a sink of negative infinity. The device side now carries the same
//! argument: kernels/flash_mla_csa2_sm121 takes a [heads] f32 sink and adds it to the denominator
//! after the maximum, so a device-vs-reference comparison runs with the sink included; the device
//! run of 2026-10-17 (stage 2, 5/5 positions) is that comparison.
//!
//! The SCORE SCALE is part of the contract, not a detail: a score is a dot product over the
//! 512-wide latent, multiplied by 1 / sqrt(512) = 0.04419417382415922 (MLA_SCORE_SCALE below).
//! Specification 2.3 states the sparse-selection rule but does not pin this constant, so it is
//! named once in kernels/m4_common_sm121.cuh and consumed by the kernel, and the same literal
//! appears here. A test compares the header's literal with this module's constant bit for bit, so
//! the two cannot drift apart silently, which is exactly the failure this reference had: it
//! softmaxed raw dot products while the kernel scaled them, and so it blessed a softmax over the
//! wrong scores and would have failed a correct kernel.
//!
//! What is NOT here: rope (the caller applies it, per the kernel's own signature), the selection of
//! the candidate list itself, and the merge of the per-block indexer results into one top-512 list.
//! Those are separate steps, and nothing in this file claims anything about them.

use crate::m4_quant::{bf16_bits, bf16_value, fp4_e2m1_decode, fp8_e4m3_decode, ue8m0_decode};

/// The MLA latent width. Mirrors SPARK_M4_MLA_LATENT.
pub const MLA_LATENT: usize = 512;
/// Latent channels per E4M3 scale. Mirrors SPARK_M4_KV_SCALE_CHANNELS.
pub const KV_SCALE_CHANNELS: usize = 16;
/// The sliding window, in tokens. Mirrors SPARK_M4_SWA_WINDOW.
pub const SWA_WINDOW: u32 = 128;
/// The indexer's per-head width. Mirrors SPARK_M4_INDEX_HEAD_DIM.
pub const INDEX_HEAD_DIM: usize = 128;
/// The high-speed index pool: 2,048 blocks of 8 tokens. Mirrors SPARK_M4_HSI_BLOCKS.
pub const HSI_BLOCKS: usize = 2048;
/// Tokens per HSI block. Mirrors SPARK_M4_HSI_BLOCK_TOKENS.
pub const HSI_BLOCK_TOKENS: usize = 8;
/// The largest candidate list the caller may pass. Mirrors SPARK_M4_MLA_MAX_SELECTED.
pub const MAX_SELECTED: usize = 512;
/// E2M1 values per byte.
pub const FP4_PER_BYTE: usize = 2;
/// E4M3 scales per latent row.
pub const ROW_SCALE_BYTES: usize = MLA_LATENT / KV_SCALE_CHANNELS;
/// Packed latent bytes per row.
pub const ROW_PACKED_BYTES: usize = MLA_LATENT / FP4_PER_BYTE;
/// Query heads. Mirrors SPARK_M4_HEADS.
pub const MLA_HEADS: usize = 64;
/// Indexer heads. Mirrors SPARK_M4_INDEX_HEADS.
pub const INDEX_HEADS: usize = 32;
/// Query heads served by one indexer head: 64 query heads over 32 indexer heads, so the two index
/// spaces cannot share an index. Mirrors SPARK_M4_MLA_HEADS_PER_INDEX_HEAD.
pub const HEADS_PER_INDEX_HEAD: usize = MLA_HEADS / INDEX_HEADS;
/// The MLA score scale, 1 / sqrt(512) over the latent width. Mirrors
/// SPARK_M4_MLA_SCORE_SCALE in kernels/m4_common_sm121.cuh; a test compares the two.
// The digits are deliberately the header's own spelling of 1/sqrt(512), and the parity test
// compares the two bit for bit, so the excess precision is the point rather than an accident.
#[allow(clippy::excessive_precision)]
pub const MLA_SCORE_SCALE: f32 = 0.04419417382415922;

/// The indexer head that serves a query head. This division is required, not cosmetic: the kernel
/// reads index_query[head * INDEX_HEAD_DIM] as if it were the query head's own index query, which
/// reads past the end of the 32-by-128 index query pool for half the blocks and uses the wrong
/// head's scores for
/// the rest. This function is the shared definition of the mapping; a test pins the kernel text to
/// it.
///
/// # Panics
/// Panics if the head is not a query head.
#[must_use]
pub fn index_head_of(head: usize) -> usize {
    assert!(head < MLA_HEADS, "the head must be a query head");
    head / HEADS_PER_INDEX_HEAD
}

/// The window predicate, transcribed from the kernel: a candidate is attendable when it is at or
/// behind the query and less than a window away from it.
#[must_use]
pub fn in_window(position: u32, candidate: u32) -> bool {
    position >= candidate && (position - candidate) < SWA_WINDOW
}

/// One latent channel of one KV row, dequantised: the packed E2M1 code times its 16-channel scale.
///
/// # Panics
/// Panics if the row or the scale bytes are too short.
#[must_use]
pub fn kv_latent(kv_fp4: &[u8], kv_scales: &[u8], row: usize, index: usize) -> f32 {
    let packed = kv_fp4[row * ROW_PACKED_BYTES + index / FP4_PER_BYTE];
    let nibble = if index.is_multiple_of(FP4_PER_BYTE) {
        packed & 0x0F
    } else {
        packed >> 4
    };
    fp4_e2m1_decode(nibble)
        * fp8_e4m3_decode(kv_scales[row * ROW_SCALE_BYTES + index / KV_SCALE_CHANNELS])
}

/// The attention output for one head, without a sink: three passes over the candidate list, exactly
/// as the kernel runs them.
///
/// The query q is one head's latent, rope already applied by the caller; the KV buffer is
/// \[positions\]\[packed\] with scales \[positions\][latent / 16]. Out-of-window candidates are skipped in
/// every pass, so they contribute exactly nothing; if no candidate is inside the window the result
/// is all zeros.
///
/// This is the no-sink path and it is unchanged by the sink work; [`attention_with_sink`] takes the
/// checkpoint's sink value and is the only entry point that can differ.
///
/// # Panics
/// Panics if the candidate list is longer than MAX_SELECTED, or the slices are too short.
#[must_use]
pub fn attention(
    q: &[f32],
    kv_fp4: &[u8],
    kv_scales: &[u8],
    selected: &[u32],
    position: u32,
) -> Vec<f32> {
    attention_impl(q, kv_fp4, kv_scales, selected, position, None)
}

/// The attention output for one head, with that head's attention sink.
///
/// The sink is one additive logit in the same space as the scaled scores - it is added as given and
/// is NOT multiplied by [`MLA_SCORE_SCALE`] again - and it enters the softmax denominator only:
///
/// ```text
/// out = sum_j exp(s_j - m) v_j / (exp(sink - m) + sum_j exp(s_j - m))
/// ```
///
/// with `m` the maximum over the in-window candidates, computed exactly as [`attention`] computes
/// it. Mass that goes to the sink contributes no value, so the output keeps the candidate mixture's
/// direction and shrinks by the sink's share of the denominator; if nothing is in the window the
/// result is all zeros with or without a sink. Following the released implementation, the sink is
/// added after `m` rather than folded into it, so a sink more than about 88 above `m` overflows the
/// denominator and the output is exactly zero.
///
/// # Panics
/// Panics if the candidate list is longer than MAX_SELECTED, or the slices are too short.
#[must_use]
pub fn attention_with_sink(
    q: &[f32],
    kv_fp4: &[u8],
    kv_scales: &[u8],
    selected: &[u32],
    position: u32,
    sink: f32,
) -> Vec<f32> {
    attention_impl(q, kv_fp4, kv_scales, selected, position, Some(sink))
}

/// The three passes, shared by the no-sink and sink entry points.
///
/// `sink = None` is the no-sink arithmetic exactly as it was before the sink existed, in the same
/// order, so its results are unchanged bit for bit; `Some(sink)` adds one term to the denominator of
/// pass two and touches nothing else.
///
/// # Panics
/// Panics if the candidate list is longer than MAX_SELECTED, or the slices are too short.
fn attention_impl(
    q: &[f32],
    kv_fp4: &[u8],
    kv_scales: &[u8],
    selected: &[u32],
    position: u32,
    sink: Option<f32>,
) -> Vec<f32> {
    assert_eq!(q.len(), MLA_LATENT, "the query is one latent wide");
    assert!(
        selected.len() <= MAX_SELECTED,
        "the candidate list is index_topk long"
    );

    // Pass one: the maximum score over the candidates that are inside the window. The sink is NOT
    // folded into this maximum: the released kernel adds it after the candidate max, so the
    // exponentials formed below are the ones it forms.
    let mut maximum = f32::NEG_INFINITY;
    for candidate in selected {
        if in_window(position, *candidate) {
            let score = score_candidate(q, kv_fp4, kv_scales, *candidate);
            if score > maximum {
                maximum = score;
            }
        }
    }

    // Pass two: the normalising sum, plus the sink's term when there is one. An all-excluded list
    // leaves the sum at zero and pass three's denominator guard turns that into a zero output; the
    // sink's term is skipped there rather than evaluated, because our maximum is -inf where the
    // released kernel's is the finite -1e30 floor, so exp(sink + 1e30) would be +inf for the
    // released code, and both reach the same all-zero answer.
    let mut weight_sum = 0.0f32;
    if maximum > f32::NEG_INFINITY {
        for candidate in selected {
            if in_window(position, *candidate) {
                weight_sum += (score_candidate(q, kv_fp4, kv_scales, *candidate) - maximum).exp();
            }
        }
        if let Some(sink) = sink {
            weight_sum += (sink - maximum).exp();
        }
    }

    // Pass three: the weighted latent sum.
    let mut out = vec![0.0f32; MLA_LATENT];
    if maximum > f32::NEG_INFINITY {
        for candidate in selected {
            if !in_window(position, *candidate) {
                continue;
            }
            let weight = (score_candidate(q, kv_fp4, kv_scales, *candidate) - maximum).exp();
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight * kv_latent(kv_fp4, kv_scales, *candidate as usize, index);
            }
        }
    }
    // The denominator is the (possibly sink-augmented) sum itself whenever something was attended;
    // the 1.0 guard exists only for the empty case, whose numerator is already all zeros. Tying the
    // guard to the empty case rather than to "weight_sum > 0" is what lets a non-finite sink behave
    // as IEEE arithmetic - a +inf denominator gives exactly zero, a NaN sink gives NaN - instead of
    // being silently swapped for an unnormalised sum.
    let denominator = if maximum > f32::NEG_INFINITY {
        weight_sum
    } else {
        1.0
    };
    for value in out.iter_mut() {
        *value /= denominator;
    }
    out
}

/// The score of one candidate: the query against its dequantised latent, in f32, times the MLA
/// score scale. Scaling here rather than at the call sites is deliberate: all three passes and the
/// indexer-adjacent helpers go through this one function, so no pass can be left unscaled - which
/// is precisely how the kernel and this reference came apart before.
fn score_candidate(q: &[f32], kv_fp4: &[u8], kv_scales: &[u8], candidate: u32) -> f32 {
    let mut score = 0.0f32;
    for (index, value) in q.iter().enumerate() {
        score += value * kv_latent(kv_fp4, kv_scales, candidate as usize, index);
    }
    score * MLA_SCORE_SCALE
}

/// One E8M0 scale per this many window-KV channels (model.py:27, kernel.py:98-124).
pub const WINDOW_QUANT_BLOCK: usize = 32;
/// One E8M0 scale byte per `WINDOW_QUANT_BLOCK` channels: sixteen for a 512-wide row.
pub const WINDOW_SCALE_BYTES: usize = MLA_LATENT / WINDOW_QUANT_BLOCK;

/// One channel of one engine-format window-KV row, dequantised as the engine stores it: the E4M3
/// code times its 32-channel E8M0 power-of-two scale, in f32, then rounded to bf16.
///
/// The bf16 round is the engine's storage: model.py:707 quantises in place with an `inplace=True`
/// round trip whose output dtype is the input dtype (BF16), so the released window cache holds
/// `bf16(dequant(code, scale))` and the attention reads those bf16 values. The device kernel applies
/// the same `__float2bfloat16_rn` round to the decoded value, so both sides consume the same
/// values; this module does not instead claim the engine's bf16 *arithmetic*, which the engine
/// reference forward deliberately does not reproduce.
///
/// # Panics
/// Panics if the row or the scale bytes are too short.
#[must_use]
pub fn window_kv_value(codes: &[u8], scales: &[u8], row: usize, index: usize) -> f32 {
    let code = fp8_e4m3_decode(codes[row * MLA_LATENT + index]);
    let scale = ue8m0_decode(scales[row * WINDOW_SCALE_BYTES + index / WINDOW_QUANT_BLOCK]);
    bf16_value(bf16_bits(code * scale))
}

/// Whether one candidate is inside the engine-format window: the ids are row indices into the
/// window rows the caller supplied, oldest first, so an id at or past `window_rows` is outside and
/// contributes exactly nothing.
#[must_use]
pub fn window_attendable(window_rows: usize, candidate: u32) -> bool {
    (candidate as usize) < window_rows
}

/// The window attention output for one head, without a sink.
///
/// The query is one head's latent with rope already applied. The window rows are the engine's
/// window format, oldest first: `window_codes` is [rows][512] E4M3 codes and `window_scales` is
/// [rows][16] E8M0 bytes. `selected` holds row indices; an id at or past `window_rows` is outside
/// the window and is skipped in every pass, so its contribution is exactly zero and an all-outside
/// list gives an all-zero output.
///
/// This is the engine-format counterpart of [`attention`]; it shares no decode or attention code with
/// the FP4 path, which is the point: the device kernel and this reference are two independent
/// implementations of the same format.
///
/// # Panics
/// Panics if the candidate list is longer than MAX_SELECTED, or the slices are too short for
/// `window_rows` rows.
#[must_use]
pub fn attention_window(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
    selected: &[u32],
) -> Vec<f32> {
    window_attention_impl(q, window_codes, window_scales, window_rows, selected, None)
}

/// The engine-format window attention output for one head, with that head's attention sink.
///
/// The sink is the same additive denominator logit as in [`attention_with_sink`]: one value added
/// after the maximum, in the denominator only. An all-outside candidate list therefore yields an
/// all-zero output even with a sink, because the numerator is empty.
///
/// # Panics
/// Panics if the candidate list is longer than MAX_SELECTED, or the slices are too short for
/// `window_rows` rows.
#[must_use]
pub fn attention_with_sink_window(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
    selected: &[u32],
    sink: f32,
) -> Vec<f32> {
    window_attention_impl(
        q,
        window_codes,
        window_scales,
        window_rows,
        selected,
        Some(sink),
    )
}

/// The score of one engine-format window candidate: the query against the row's stored bf16 values,
/// in f32, times the MLA score scale.
fn score_window_candidate(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    candidate: u32,
) -> f32 {
    let mut score = 0.0f32;
    for (index, value) in q.iter().enumerate() {
        score += value * window_kv_value(window_codes, window_scales, candidate as usize, index);
    }
    score * MLA_SCORE_SCALE
}

/// The three-pass window attention, in the same order as the FP4 three passes: candidate maximum,
/// normalising sum plus the sink term, weighted window-row sum.
fn window_attention_impl(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
    selected: &[u32],
    sink: Option<f32>,
) -> Vec<f32> {
    assert_eq!(q.len(), MLA_LATENT, "the query is one latent wide");
    assert!(
        selected.len() <= MAX_SELECTED,
        "the candidate list is index_topk long"
    );
    assert!(
        window_codes.len() >= window_rows * MLA_LATENT
            && window_scales.len() >= window_rows * WINDOW_SCALE_BYTES,
        "the window rows are shorter than the declared row count"
    );

    // Pass one: the maximum score over the in-window candidates. An out-of-window id is assigned
    // -FLT_MAX exactly as the kernel assigns it, so it cannot affect the maximum.
    let mut maximum = f32::NEG_INFINITY;
    for candidate in selected {
        if window_attendable(window_rows, *candidate) {
            let score = score_window_candidate(q, window_codes, window_scales, *candidate);
            if score > maximum {
                maximum = score;
            }
        }
    }

    // Pass two: the normalising sum and, when present, the sink's term - added after the maximum,
    // exactly where the released implementation adds it.
    let mut weight_sum = 0.0f32;
    if maximum > f32::NEG_INFINITY {
        for candidate in selected {
            if window_attendable(window_rows, *candidate) {
                weight_sum += (score_window_candidate(q, window_codes, window_scales, *candidate)
                    - maximum)
                    .exp();
            }
        }
        if let Some(sink) = sink {
            weight_sum += (sink - maximum).exp();
        }
    }

    // Pass three: the weighted sum of the stored window values. Out-of-window ids are skipped
    // outright, so their contribution is exactly zero rather than a small weight.
    let mut out = vec![0.0f32; MLA_LATENT];
    if maximum > f32::NEG_INFINITY {
        for candidate in selected {
            if !window_attendable(window_rows, *candidate) {
                continue;
            }
            let weight = (score_window_candidate(q, window_codes, window_scales, *candidate)
                - maximum)
                .exp();
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight
                    * window_kv_value(window_codes, window_scales, *candidate as usize, index);
            }
        }
    }
    let denominator = if maximum > f32::NEG_INFINITY {
        weight_sum
    } else {
        1.0
    };
    for value in out.iter_mut() {
        *value /= denominator;
    }
    out
}

/// The indexer: for each of the first blocks HSI blocks, the best query/key dot product and the token
/// that scored it. Ties keep the earliest token, as the kernel's strict comparison does.
///
/// # Panics
/// Panics if the slices are too short for the declared block count.
#[must_use]
pub fn index_block_best(
    index_query: &[f32],
    index_key: &[f32],
    blocks: usize,
) -> (Vec<f32>, Vec<u32>) {
    assert_eq!(
        index_query.len(),
        INDEX_HEAD_DIM,
        "the index query is one head's width"
    );
    assert!(
        index_key.len() >= blocks * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM,
        "the index key pool is too short"
    );
    let mut scores = vec![0.0f32; blocks];
    let mut best_tokens = vec![0u32; blocks];
    for block in 0..blocks {
        let mut best = f32::NEG_INFINITY;
        let mut best_token = 0u32;
        for slot in 0..HSI_BLOCK_TOKENS {
            let base = (block * HSI_BLOCK_TOKENS + slot) * INDEX_HEAD_DIM;
            let mut score = 0.0f32;
            for index in 0..INDEX_HEAD_DIM {
                score += index_query[index] * index_key[base + index];
            }
            if score > best {
                best = score;
                best_token = (block * HSI_BLOCK_TOKENS + slot) as u32;
            }
        }
        scores[block] = best;
        best_tokens[block] = best_token;
    }
    (scores, best_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A KV buffer with the given (row, channel, code) entries set: the scale byte 0x38 is E4M3 1.0,
    /// so a channel of a dequantised latent can be written directly with an E2M1 code.
    struct Kv {
        fp4: Vec<u8>,
        scales: Vec<u8>,
        rows: usize,
    }

    impl Kv {
        fn new(rows: usize) -> Self {
            Self {
                fp4: vec![0u8; rows * ROW_PACKED_BYTES],
                scales: vec![0u8; rows * ROW_SCALE_BYTES],
                rows,
            }
        }

        fn set(&mut self, row: usize, index: usize, code: u8) {
            assert!(row < self.rows);
            self.scales[row * ROW_SCALE_BYTES + index / KV_SCALE_CHANNELS] = 0x38; // E4M3 1.0
            let byte = row * ROW_PACKED_BYTES + index / FP4_PER_BYTE;
            if index.is_multiple_of(FP4_PER_BYTE) {
                self.fp4[byte] |= code & 0x0F;
            } else {
                self.fp4[byte] |= (code & 0x0F) << 4;
            }
        }
    }

    #[test]
    fn the_window_predicate_is_the_kernels_comparison() {
        assert!(in_window(0, 0), "the query attends to itself");
        assert!(in_window(200, 200));
        assert!(
            in_window(200, 200 - (SWA_WINDOW - 1)),
            "the window is inclusive at its far edge"
        );
        assert!(
            !in_window(200, 200 - SWA_WINDOW),
            "one token older is outside"
        );
        assert!(
            !in_window(200, 201),
            "a future candidate is never attendable"
        );
        assert!(!in_window(0, 1));
        // The predicate is a window, not a threshold: it must switch off again for old tokens.
        assert!(in_window(1024, 1000));
        assert!(!in_window(1024, 800));
    }

    #[test]
    fn golden_two_candidates_is_hand_computable() {
        // The query reads latent channel 0 only, and channel 0 of candidates 0 and 1 is E2M1 1.0 and
        // 2.0 (code 0x4) with a scale of 1.0, so the scores are 1 and 2. The softmax of (1, 2) weights
        // the two latents by 1/(1+e) and e/(1+e), and since only channel 0 is non-zero the output is
        // that weighted sum in channel 0 and zero everywhere else.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2); // 1.0
        kv.set(1, 0, 0x4); // 2.0
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let out = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);

        // The scores are the dot products TIMES the MLA score scale, so the softmax ratio is
        // exp(2 * scale) / exp(1 * scale) and the golden is the same shape with the scaled ratio.
        // Before the scale was applied here this test passed on raw scores while the kernel
        // computed the scaled ones, which is how a reference comes to bless the wrong answer.
        let ratio = MLA_SCORE_SCALE.exp();
        let expected = (1.0 + ratio * 2.0) / (1.0 + ratio);
        assert!((out[0] - expected).abs() < 1e-5, "channel 0 is {}", out[0]);
        // The scale is observable: the unscaled softmax is a different number, so this test cannot
        // pass by accident if the scale is dropped again.
        let unscaled = (1.0 + std::f32::consts::E * 2.0) / (1.0 + std::f32::consts::E);
        assert!(
            (expected - unscaled).abs() > 1e-3,
            "the scaled and unscaled goldens must differ"
        );
        for (index, value) in out.iter().enumerate().skip(1) {
            assert_eq!(*value, 0.0, "channel {index} must stay zero");
        }
        // Bit-for-bit determinism: the kernel's weighted sum accumulates per thread with no shared
        // atomics and its maximum is a fixed-order tree reduction (spark_block_max, not a racy
        // shared read-modify-write), so nothing in the kernel's result depends on warp scheduling
        // and the reference may be compared bit for bit rather than within a tolerance.
        let again = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);
        assert_eq!(
            out.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            again
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_out_of_window_candidate_contributes_exactly_nothing() {
        // The query is at position 0, so candidate 0 is attendable and candidate 1 is in the future.
        // The output must be exactly candidate 0's latent, not a softmax that merely down-weights the
        // future candidate. (At position 4 BOTH would be inside the window: it is 128 tokens deep.)
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2); // 1.0
        kv.set(1, 0, 0x7); // 6.0, and it must not be seen at all
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let out = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 0);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "the out-of-window candidate leaked: {}",
            out[0]
        );
    }

    #[test]
    fn an_empty_candidate_set_gives_zeros_and_not_a_nan() {
        // Every candidate is in the future, so nothing is attendable. The kernel guards its
        // denominator; a reference that divided by the zero weight sum would produce NaN and hide the
        // difference between "nothing to attend" and "broken".
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2);
        let q = vec![1.0f32; MLA_LATENT];
        let out = attention(&q, &kv.fp4, &kv.scales, &[5, 6], 0);
        assert!(out.iter().all(|value| *value == 0.0), "expected all zeros");
        // An empty list is the same case.
        let empty = attention(&q, &kv.fp4, &kv.scales, &[], 0);
        assert!(empty.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn a_large_sink_removes_the_attention_mass_from_the_only_candidate() {
        // One candidate whose latent is 1.0 in channel 0, against a query that reads channel 0, so
        // the score is MLA_SCORE_SCALE. Without a sink that candidate takes the whole softmax and the
        // output IS its latent. A sink of 10.0 - two orders of magnitude above the score - has to
        // take almost all of the mass, and because the sink has no value the output collapses towards
        // zero rather than being renormalised back to the candidate.
        let mut kv = Kv::new(1);
        kv.set(0, 0, 0x2); // 1.0
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let score = MLA_SCORE_SCALE;
        let sink = 10.0f32;

        let without = attention(&q, &kv.fp4, &kv.scales, &[0], 0);
        assert_eq!(without[0], 1.0, "one candidate takes all of the mass");

        let with = attention_with_sink(&q, &kv.fp4, &kv.scales, &[0], 0, sink);
        // Closed form, written in the other arrangement from the implementation's: the candidate's
        // weight is exp(score - max) = 1 and the sink's is exp(sink - max), so the candidate keeps
        // s / (1 + s) with s = exp(score - sink).
        let s = (score - sink).exp();
        let expected = s / (1.0 + s);
        assert!(
            (with[0] - expected).abs() < 1e-9,
            "got {} expected {expected}",
            with[0]
        );
        // The sink removes exactly the mass it takes: the difference is 1 - expected.
        let removed = without[0] - with[0];
        assert!(
            (removed - (1.0 - expected)).abs() < 1e-6,
            "the sink removed {removed}, expected {}",
            1.0 - expected
        );
        // And it is a visible collapse, so this test cannot pass with the sink ignored: a sink of
        // 10.0 leaves the candidate under a thousandth of its weight.
        assert!(
            with[0] < without[0] * 1e-3,
            "a sink of {sink} must collapse the candidate's weight: {} vs {}",
            with[0],
            without[0]
        );
        // The sink adds mass, not a value: no channel receives anything.
        for (index, value) in with.iter().enumerate().skip(1) {
            assert_eq!(*value, 0.0, "channel {index} must stay zero");
        }
    }

    #[test]
    fn the_sink_is_added_as_a_logit_and_not_scaled_again() {
        // The candidate's score already carries MLA_SCORE_SCALE; the sink is a logit in that same
        // additive space, so it is added as given. With score = MLA_SCORE_SCALE and sink = 1.0 the
        // candidate keeps s / (1 + s) with s = exp(MLA_SCORE_SCALE - 1), about 0.2777. A sink that
        // was multiplied by the score scale as well would give 0.5, so the two readings are more
        // than 0.2 apart and this test can tell them apart.
        let mut kv = Kv::new(1);
        kv.set(0, 0, 0x2);
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let sink = 1.0f32;
        let out = attention_with_sink(&q, &kv.fp4, &kv.scales, &[0], 0, sink);
        let s = (MLA_SCORE_SCALE - sink).exp();
        let expected = s / (1.0 + s);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {} expected {expected}",
            out[0]
        );
        let scaled_again_s = (MLA_SCORE_SCALE - sink * MLA_SCORE_SCALE).exp();
        let scaled_again = scaled_again_s / (1.0 + scaled_again_s);
        assert!(
            (out[0] - scaled_again).abs() > 0.2,
            "a sink scaled a second time must not fit this test: {} vs {scaled_again}",
            out[0]
        );
    }

    #[test]
    fn the_sink_only_scales_the_candidate_mixture() {
        // Two candidates whose latents point at different channels: row 0 is 1.0 in channel 0, row 1
        // is 2.0 in channel 1. The query reads channel 0, so candidate 0 scores MLA_SCORE_SCALE and
        // candidate 1 scores 0: their weights are 1 and exp(-MLA_SCORE_SCALE), and the no-sink answer
        // is that weighted mixture. A sink can only scale the mixture down - it cannot change the
        // ratio between its channels, because it contributes to the denominator and to nothing else.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2); // 1.0 in channel 0
        kv.set(1, 1, 0x4); // 2.0 in channel 1
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let sink = 3.0f32;

        let lower = (-MLA_SCORE_SCALE).exp();
        let candidate_weight_sum = 1.0 + lower;
        let without = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);
        assert!(
            (without[0] - 1.0 / candidate_weight_sum).abs() < 1e-6,
            "no sink, channel 0: {}",
            without[0]
        );
        assert!(
            (without[1] - 2.0 * lower / candidate_weight_sum).abs() < 1e-6,
            "no sink, channel 1: {}",
            without[1]
        );

        let with = attention_with_sink(&q, &kv.fp4, &kv.scales, &[0, 1], 8, sink);
        // The same numerator, over the candidate sum plus the sink's exp(sink - MLA_SCORE_SCALE).
        let sink_weight = (sink - MLA_SCORE_SCALE).exp();
        let denominator = candidate_weight_sum + sink_weight;
        assert!(
            (with[0] - 1.0 / denominator).abs() < 1e-6,
            "got {} expected {}",
            with[0],
            1.0 / denominator
        );
        assert!(
            (with[1] - 2.0 * lower / denominator).abs() < 1e-6,
            "got {} expected {}",
            with[1],
            2.0 * lower / denominator
        );
        // The ratio between the channels is untouched: the sink is not a mixture component.
        let expected_ratio = 2.0 * lower;
        assert!(
            (with[1] / with[0] - without[1] / without[0]).abs() < 1e-4,
            "the sink changed the mixture's shape: {} vs {}",
            with[1] / with[0],
            without[1] / without[0]
        );
        assert!(
            (with[1] / with[0] - expected_ratio).abs() < 1e-4,
            "the mixture's ratio must be the weight ratio: {} vs {expected_ratio}",
            with[1] / with[0]
        );
        // Both channels are strictly smaller, so the sink really did take mass.
        assert!(
            with[0] < without[0] && with[1] < without[1],
            "the sink must attenuate both channels: {with:?} vs {without:?}"
        );
    }

    #[test]
    fn an_unreachable_sink_reproduces_the_no_sink_output_bit_for_bit() {
        // Point 4 of the sink work: callers with no sink are not silently altered. A sink that can
        // never win - negative infinity, or so negative that its exponential underflows to zero -
        // must give the same bits as the no-sink entry point, not merely a close answer.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2);
        kv.set(1, 0, 0x4);
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let without = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);
        for sink in [f32::NEG_INFINITY, -1.0e30] {
            let with = attention_with_sink(&q, &kv.fp4, &kv.scales, &[0, 1], 8, sink);
            assert_eq!(
                without
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                with.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "a sink of {sink} must reproduce the no-sink output"
            );
        }
    }

    #[test]
    fn a_sink_with_nothing_in_the_window_is_still_zeros() {
        // The sink is a place for mass to go, not a value: when every candidate is outside the
        // window the output is all zeros whether or not a sink is given, and a sink must not turn
        // that into a NaN or an unnormalised number.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2);
        let q = vec![1.0f32; MLA_LATENT];
        let future = attention_with_sink(&q, &kv.fp4, &kv.scales, &[5, 6], 0, 5.0);
        assert!(
            future.iter().all(|value| *value == 0.0),
            "expected zeros, got {future:?}"
        );
        let empty = attention_with_sink(&q, &kv.fp4, &kv.scales, &[], 0, 5.0);
        assert!(
            empty.iter().all(|value| *value == 0.0),
            "expected zeros, got {empty:?}"
        );
    }

    #[test]
    fn a_sink_far_above_the_candidate_max_gives_the_released_zero() {
        // The released kernel adds the sink after the candidate maximum, so a sink more than about
        // 88 above it overflows exp() to +inf and the output is exactly zero - "attend to nothing",
        // with no value to show for it. That is the formula this reference reproduces; what must not
        // happen is a NaN or an unnormalised sum.
        let mut kv = Kv::new(1);
        kv.set(0, 0, 0x2);
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        let out = attention_with_sink(&q, &kv.fp4, &kv.scales, &[0], 0, 1.0e6);
        assert!(
            out.iter().all(|value| *value == 0.0),
            "expected an exact zero, got {out:?}"
        );
    }

    #[test]
    fn a_score_difference_moves_the_weight_the_way_a_softmax_does() {
        // Property: with two candidates whose latents differ in channel 0, the output is the
        // softmax-weighted mean of those two latent values. Checked against the closed form rather
        // than against the implementation.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2); // 1.0
        kv.set(1, 0, 0x2); // 1.0
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 1.0;
        // Both candidates score 1, so the weights are equal and the output is 1.0.
        let equal = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);
        assert!((equal[0] - 1.0).abs() < 1e-6);

        // Give candidate 1 the value 6.0 (code 0x7): its score is 6, the gap is 5, and the output is
        // the softmax-weighted mean of 1 and 6.
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2);
        kv.set(1, 0, 0x7);
        let out = attention(&q, &kv.fp4, &kv.scales, &[0, 1], 8);
        // The gap between the two scores is scaled too, so the ratio is exp(5 * scale), not exp(5).
        let ratio = (5.0 * MLA_SCORE_SCALE).exp();
        let expected = (1.0 + ratio * 6.0) / (1.0 + ratio);
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "got {} expected {expected}",
            out[0]
        );
        assert!(
            out[0] > equal[0],
            "the higher-scoring candidate must pull the output up"
        );
    }

    #[test]
    fn the_indexer_keeps_the_best_and_the_first_of_a_tie() {
        let mut keys = vec![0.0f32; HSI_BLOCK_TOKENS * INDEX_HEAD_DIM];
        let mut query = vec![0.0f32; INDEX_HEAD_DIM];
        query[0] = 1.0;
        // Slot 0 scores 1, slot 1 scores 3, slot 3 scores 3 (a tie with slot 1), the rest 0.
        keys[0] = 1.0;
        keys[INDEX_HEAD_DIM] = 3.0;
        keys[3 * INDEX_HEAD_DIM] = 3.0;
        let (scores, best) = index_block_best(&query, &keys, 1);
        assert_eq!(scores[0], 3.0);
        assert_eq!(best[0], 1, "a tie must keep the earlier token");
        // Raising the score of a later slot does move the winner, so the tie rule is not hiding a
        // comparison that never fires.
        keys[4 * INDEX_HEAD_DIM] = 4.0;
        let (scores, best) = index_block_best(&query, &keys, 1);
        assert_eq!(scores[0], 4.0);
        assert_eq!(best[0], 4);
        // A block whose keys are all zero scores zero rather than -inf, and keeps token 0.
        let zeros = vec![0.0f32; HSI_BLOCK_TOKENS * INDEX_HEAD_DIM];
        let (scores, best) = index_block_best(&query, &zeros, 1);
        assert_eq!(scores[0], 0.0);
        assert_eq!(best[0], 0);
    }

    /// The literal after a "#define NAME" in the shared header, as f32 bits. Panics if the define is
    /// missing, which is what a rename of the constant would do as well.
    fn header_float_bits(name: &str) -> u32 {
        let header = crate::m4_abi::COMMON_HEADER;
        let prefix = format!("#define {name} ");
        let line = header
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not defined in the shared header"));
        let literal = line[prefix.len()..]
            .trim()
            .trim_end_matches('f')
            .trim_end_matches('F');
        literal
            .parse::<f32>()
            .unwrap_or_else(|error| panic!("{name} is not a float literal: {error}"))
            .to_bits()
    }

    /// The right-hand side of a "#define NAME ..." in the shared header, trimmed.
    fn header_define(name: &str) -> String {
        let header = crate::m4_abi::COMMON_HEADER;
        let prefix = format!("#define {name} ");
        header
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not defined in the shared header"))[prefix.len()..]
            .trim()
            .to_string()
    }

    /// The integer literal after a "#define NAME" in the shared header.
    fn header_int(name: &str) -> i64 {
        let header = crate::m4_abi::COMMON_HEADER;
        let prefix = format!("#define {name} ");
        let line = header
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{name} is not defined in the shared header"));
        line[prefix.len()..]
            .trim()
            .trim_end_matches('u')
            .parse::<i64>()
            .unwrap_or_else(|error| panic!("{name} is not an integer literal: {error}"))
    }

    #[test]
    fn the_score_scale_is_the_kernels_literal_bit_for_bit() {
        // The kernel multiplies its scores by SPARK_M4_MLA_SCORE_SCALE and this reference multiplies
        // by MLA_SCORE_SCALE. A test that compared only the reference with itself would pass while
        // the two carried different constants, which is the defect that put a softmax over raw
        // scores in this module. Compare the header's literal with the constant, bit for bit.
        assert_eq!(
            header_float_bits("SPARK_M4_MLA_SCORE_SCALE"),
            MLA_SCORE_SCALE.to_bits(),
            "the kernel's score scale and the reference's must be the same float"
        );
        assert_ne!(
            header_float_bits("SPARK_M4_MLA_SCORE_SCALE"),
            1.0f32.to_bits(),
            "the parity check must not be able to pass with a different literal"
        );
        // 1 / sqrt(512) to the last bit, not merely close to it.
        assert_eq!(MLA_SCORE_SCALE, 1.0f32 / 512.0f32.sqrt());
    }

    #[test]
    fn the_index_head_mapping_covers_every_query_head_exactly_twice() {
        // 64 query heads, 32 indexer heads: each indexer head serves two query heads, every query
        // head has one, and the mapping never reaches the pool's end. A kernel that indexed
        // index_query with the query head would read entries 32..63 of a [32][128] pool, i.e. past
        // the end for the top half of the grid.
        assert_eq!(MLA_HEADS, 64);
        assert_eq!(INDEX_HEADS, 32);
        assert_eq!(HEADS_PER_INDEX_HEAD, 2);
        assert_eq!(header_int("SPARK_M4_HEADS"), MLA_HEADS as i64);
        assert_eq!(header_int("SPARK_M4_INDEX_HEADS"), INDEX_HEADS as i64);
        // The heads-per-index-head define is an expression, not a literal, so it is pinned as text:
        // with SPARK_M4_HEADS and SPARK_M4_INDEX_HEADS pinned above, the ratio follows.
        assert_eq!(
            header_define("SPARK_M4_MLA_HEADS_PER_INDEX_HEAD"),
            "(SPARK_M4_HEADS / SPARK_M4_INDEX_HEADS)",
            "the kernel's query-heads-per-indexer-head must stay the same ratio this module uses"
        );
        assert_eq!(MLA_HEADS / INDEX_HEADS, HEADS_PER_INDEX_HEAD);
        let mut served = vec![0usize; INDEX_HEADS];
        for head in 0..MLA_HEADS {
            let indexer = index_head_of(head);
            assert!(
                indexer < INDEX_HEADS,
                "head {head} escapes the indexer space"
            );
            served[indexer] += 1;
        }
        assert!(
            served.iter().all(|count| *count == HEADS_PER_INDEX_HEAD),
            "every indexer head must serve exactly {HEADS_PER_INDEX_HEAD} query heads"
        );
        assert_eq!(index_head_of(0), 0);
        assert_eq!(index_head_of(1), 0);
        assert_eq!(index_head_of(2), 1);
        assert_eq!(index_head_of(MLA_HEADS - 1), INDEX_HEADS - 1);
    }

    // ---- engine-format window KV (D4) -------------------------------------------

    /// The engine quantiser and the device-side oracle must produce the same bytes for the same
    /// row, and the reference must decode those bytes to exactly the bf16 values the engine stores.
    #[test]
    fn window_format_bytes_agree_with_the_engine_quantiser() {
        for seed in 0..8u64 {
            let row: Vec<f32> = (0..MLA_LATENT)
                .map(|index| {
                    let bits = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add((index as u64).wrapping_mul(1_442_695_040_888_963_407))
                        .wrapping_add(0x9E37_79B9_7F4A_7C15);
                    (((bits >> 40) & 0xFFFF) as f32 / 65_536.0 - 0.5) * 8.0
                })
                .collect();
            let engine = sparkflash_engine::quant::quantize_activation_fp8(&row)
                .expect("the row is a whole number of 32-blocks");
            let (codes, scales) = crate::m4_quant::act_quant_fp8_row(&row);
            assert_eq!(
                engine.codes(),
                codes.as_slice(),
                "the engine and the device oracle must emit the same E4M3 codes (seed {seed})"
            );
            assert_eq!(
                engine.scales(),
                scales.as_slice(),
                "the engine and the device oracle must emit the same E8M0 codes (seed {seed})"
            );
            let dequantised = engine.dequantize();
            for (index, &dequantised_value) in dequantised.iter().enumerate() {
                let stored = window_kv_value(&codes, &scales, 0, index);
                let expected = bf16_value(bf16_bits(dequantised_value));
                assert_eq!(
                    stored.to_bits(),
                    expected.to_bits(),
                    "the reference must consume the engine's bf16 storage (seed {seed}, channel {index})"
                );
            }
        }
    }

    /// Every one of the 256 E4M3 codes, and the two E8M0 scale extremes, decode exactly as the
    /// format says.
    #[test]
    fn window_decode_covers_every_e4m3_code_and_the_scale_edges() {
        // One row of all 256 codes, twice, at scale code 127 (2^0).
        let mut codes = Vec::with_capacity(MLA_LATENT);
        for _ in 0..2 {
            codes.extend(0u8..=255u8);
        }
        let scales = vec![127u8; WINDOW_SCALE_BYTES];
        for code in 0u8..=255u8 {
            let decoded = fp8_e4m3_decode(code);
            let expected = bf16_value(bf16_bits(decoded));
            let stored = window_kv_value(&codes, &scales, 0, usize::from(code));
            if decoded.is_nan() {
                assert!(stored.is_nan(), "code {code:#04x} must stay NaN");
            } else {
                assert_eq!(
                    stored.to_bits(),
                    expected.to_bits(),
                    "code {code:#04x} must decode to the bf16 rounding of its value"
                );
            }
        }
        // Scale code 0 is 2^-127, scale code 254 is 2^127; both with the E4M3 code for 1.0.
        let one = crate::m4_quant::fp8_e4m3_encode(1.0);
        let codes = vec![one; MLA_LATENT];
        for (scale_code, scale) in [(0u8, 2.0f32.powi(-127)), (254u8, 2.0f32.powi(127))] {
            let scales = vec![scale_code; WINDOW_SCALE_BYTES];
            let expected = bf16_value(bf16_bits(scale));
            let stored = window_kv_value(&codes, &scales, 0, 0);
            assert_eq!(
                stored.to_bits(),
                expected.to_bits(),
                "scale code {scale_code} must decode to its power of two"
            );
            assert_eq!(ue8m0_decode(scale_code), scale);
        }
        // Codes the engine quantiser cannot emit, pinned so the decoder's edge behaviour is a
        // tested contract: 0x7E is +448 and 0x7F is the E4M3 NaN. With scale code 254 (2^127) the
        // 448 value overflows f32 to +inf, and the NaN code stays a NaN. Real engine rows cannot
        // contain either (the quantiser clamps to +/-448 and never writes 0x7F), and the stage-2
        // harness scans the delivered rows for a NaN code or a non-finite decode before launch, so
        // this test describes the decoder and not reachable device input.
        let max_code = crate::m4_quant::fp8_e4m3_encode(448.0);
        assert_eq!(max_code, 0x7E);
        let mut codes = vec![max_code; MLA_LATENT];
        let scales = vec![254u8; WINDOW_SCALE_BYTES];
        let stored = window_kv_value(&codes, &scales, 0, 0);
        assert!(
            stored.is_infinite() && stored > 0.0,
            "448 x 2^127 must overflow to +inf"
        );
        codes.fill(0x7F);
        let stored = window_kv_value(&codes, &scales, 0, 0);
        assert!(stored.is_nan(), "the E4M3 NaN code must stay NaN");
    }

    /// An all-zero window row hits the reference's 1e-4 amax floor, so its scale stays finite and
    /// nonzero and every channel dequantises back to exactly zero (kernel.py:76, 108).
    #[test]
    fn a_zero_window_row_hits_the_scale_floor() {
        let row = vec![0.0f32; MLA_LATENT];
        let engine = sparkflash_engine::quant::quantize_activation_fp8(&row).expect("whole blocks");
        let (codes, scales) = crate::m4_quant::act_quant_fp8_row(&row);
        assert_eq!(engine.codes(), codes.as_slice());
        assert_eq!(engine.scales(), scales.as_slice());
        assert!(
            codes.iter().all(|&code| code == 0),
            "a zero row is all zero codes"
        );
        assert!(
            scales.iter().all(|&scale| scale == scales[0]) && scales[0] != 0,
            "every block gets the floor scale, not zero"
        );
        assert_eq!(
            ue8m0_decode(scales[0]),
            2.0f32.powi(-22),
            "the 1e-4 floor divided by 448 is 2^-22"
        );
        for index in 0..MLA_LATENT {
            assert_eq!(window_kv_value(&codes, &scales, 0, index), 0.0);
        }
    }

    /// The bf16 storage round is idempotent on every bf16 pattern: rounding a value that is already
    /// bf16 changes nothing, which is what lets the kernel and this reference agree bit for bit on
    /// the stored values.
    #[test]
    fn bf16_storage_round_is_idempotent_for_every_bf16_pattern() {
        for bits in 0u16..=u16::MAX {
            let value = bf16_value(bits);
            assert_eq!(
                bf16_value(bf16_bits(value)).to_bits(),
                value.to_bits(),
                "a bf16 pattern must round to itself"
            );
        }
    }

    /// The engine-format attention against a slow f64 reference, plus the out-of-window rules: an
    /// id at or past the row count changes nothing, its contribution is exactly zero, and an
    /// all-outside list gives an all-zero output even with a sink.
    #[test]
    fn window_attention_matches_a_slow_f64_reference_and_out_of_window_contributes_nothing() {
        let q: Vec<f32> = (0..MLA_LATENT)
            .map(|index| ((index as f32 * 0.017).sin() * 0.75) + ((index % 7) as f32 * 0.01))
            .collect();
        let rows = 3usize;
        let mut codes = Vec::with_capacity(rows * MLA_LATENT);
        let mut scales = Vec::with_capacity(rows * WINDOW_SCALE_BYTES);
        for row in 0..rows {
            let values: Vec<f32> = (0..MLA_LATENT)
                .map(|index| {
                    let angle = (row as f32 + 1.0) * 0.31 + index as f32 * 0.023;
                    (angle.sin() * 2.0) + (index % 5) as f32 * 0.05
                })
                .collect();
            let (row_codes, row_scales) = crate::m4_quant::act_quant_fp8_row(&values);
            codes.extend_from_slice(&row_codes);
            scales.extend_from_slice(&row_scales);
        }
        let sink = 0.25f32;
        let in_window = [0u32, 1, 2];
        let with_out_of_window = [0u32, 1, 2, 3, 7, u32::MAX];
        let without = attention_with_sink_window(&q, &codes, &scales, rows, &in_window, sink);
        let with = attention_with_sink_window(&q, &codes, &scales, rows, &with_out_of_window, sink);
        assert_eq!(
            without, with,
            "out-of-window ids must not change the output by even one bit"
        );

        // The slow f64 reference: the same stored values, the same score scale, the same sink.
        let mut values = Vec::with_capacity(rows * MLA_LATENT);
        for row in 0..rows {
            for index in 0..MLA_LATENT {
                values.push(f64::from(window_kv_value(&codes, &scales, row, index)));
            }
        }
        let mut scores = Vec::new();
        for row in 0..rows {
            let mut score = 0.0f64;
            for (index, value) in q.iter().enumerate() {
                score += f64::from(*value) * values[row * MLA_LATENT + index];
            }
            scores.push(score * f64::from(MLA_SCORE_SCALE));
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut denominator = (f64::from(sink) - maximum).exp();
        for score in &scores {
            denominator += (score - maximum).exp();
        }
        let mut expected = vec![0.0f64; MLA_LATENT];
        for (row, score) in scores.iter().enumerate() {
            let weight = (score - maximum).exp() / denominator;
            for index in 0..MLA_LATENT {
                expected[index] += weight * values[row * MLA_LATENT + index];
            }
        }
        for index in 0..MLA_LATENT {
            let delta = (f64::from(without[index]) - expected[index]).abs();
            assert!(
                delta <= 1e-5,
                "channel {index}: reference differs from the slow f64 result by {delta:e}"
            );
        }

        // An all-outside list is exactly zero, with and without the sink.
        let all_outside = [3u32, 7, u32::MAX];
        assert!(
            attention_with_sink_window(&q, &codes, &scales, rows, &all_outside, sink)
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            attention_window(&q, &codes, &scales, rows, &all_outside)
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(window_attendable(rows, 0) && !window_attendable(rows, 3));
    }
}
