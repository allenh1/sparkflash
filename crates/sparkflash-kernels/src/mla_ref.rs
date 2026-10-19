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
//! # CSA union mode
//!
//! attention_with_sink_union is the attention the released prefill/decode actually computes:
//! the engine-format window rows supplied by the caller (oldest first, all attended) concatenated
//! with the selected compressed candidates that are not in the query's future, one f32 softmax
//! over that union, and the sink added to the denominator after the maximum. The selected ids are
//! absolute token positions decoded from the FP4 latent; for a ratio-2 compressed source the
//! caller supplies a per-position expanded view (row p = store row p / ratio) so an id is a row
//! index again. Unlike the format-0/1 window paths above, an id outside the sliding window but at
//! or before the query is attended here - that is the Criterion-1 gap this mode closes. The
//! kernel counterpart is formats 1 and 2 of kernels/flash_mla_csa2_sm121.cu; format 2 is
//! CPU-pinned only, not yet device-verified (see docs/DECISIONS.md D24). The test
//! format_two_union_equals_the_engine_reference_on_the_attended_multiset compares this twin bit for
//! bit with sparkflash_engine::forward::attention::sparse_attention over the concatenated reference
//! KV (window rows then compressed store rows), so the union is pinned to the engine's attended
//! multiset and denominator order rather than to a second definition of our own.
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
/// The largest number of slots a CSA union attention call can attend: one compressed candidate list
/// plus one full sliding window. Mirrors SPARK_M4_MLA_MAX_ATTENDED.
pub const MAX_ATTENDED: usize = MAX_SELECTED + SWA_WINDOW as usize;
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

/// Whether one selected compressed candidate is attendable in CSA union mode.
///
/// The reference's reachability mask keeps the compressed positions strictly behind the query's own
/// position, which for a ratio-1 store is exactly `candidate <= position` (the engine computes
/// `reach = (start_pos + query + 1) / ratio` and masks `position >= reach`). A ratio-2 store is
/// presented to this function as a per-position expanded view (`row p = store row p / ratio`), so
/// an absolute candidate id maps to a row again and the same comparison stays the causal test.
#[must_use]
pub fn union_attendable(position: u32, candidate: u32) -> bool {
    candidate <= position
}

/// The CSA union attention output for one head, without a sink.
///
/// This is the attention the released prefill/decode path actually computes: the last
/// `window_rows` engine-format window rows (oldest first) and the selected compressed candidates,
/// concatenated in that order and softmaxed together. In the engine's forward that is
/// `sparse_attention(q, [window_kv; compressed_kv], sink, [window_indices..., compressed_ids...])`;
/// here `window_rows` is the number of window rows supplied and `selected` holds the compressed
/// candidates as absolute token positions.
///
/// A selected id is attended when [`union_attendable`] holds; an id at or past the query is
/// excluded exactly, and an id older than the sliding window is NOT excluded - attending it is the
/// point of this mode. Duplicate entries are attended once per entry, in list order. `kv_fp4` and
/// `kv_scales` are read by `id`; for a ratio-2 compressed source the caller supplies a
/// per-position view (`row p = store row p / ratio`) so the id is a row index again.
///
/// `window_rows` must be at most [`SWA_WINDOW`] and `window_rows + selected.len()` at most
/// [`MAX_ATTENDED`]. The window rows are the engine's format: `window_codes` is [rows][512] E4M3
/// codes and `window_scales` is [rows][16] E8M0 bytes, decoded to the bf16 value the engine stores.
///
/// # Panics
///
/// Panics if the query is not one latent wide, the candidate list is longer than [`MAX_SELECTED`],
/// the window is longer than [`SWA_WINDOW`], the union is longer than [`MAX_ATTENDED`] (exactly 640
/// slots run; 641 is refused rather than writing past the score array), the window slices are too
/// short, or an attended candidate has no row in the expanded KV view.
#[allow(clippy::too_many_arguments)]
// the engine's concatenated attention signature: the query, the window format, the compressed format, the candidate list and the causal position are distinct inputs.
#[must_use]
pub fn attention_union(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
    kv_fp4: &[u8],
    kv_scales: &[u8],
    selected: &[u32],
    position: u32,
) -> Vec<f32> {
    union_attention_impl(
        q,
        window_codes,
        window_scales,
        window_rows,
        kv_fp4,
        kv_scales,
        selected,
        position,
        None,
    )
}

/// The CSA union attention output for one head, with that head's attention sink.
///
/// The sink is the same additive denominator logit as in [`attention_with_sink`]: one value
/// added after the union's maximum, in the denominator only. It contributes no value, so a union
/// with nothing attendable still yields an all-zero output.
///
/// # Panics
///
/// As [`attention_union`].
#[allow(clippy::too_many_arguments)] // as attention_union, plus the per-head sink: the reference's own signature.
#[must_use]
pub fn attention_with_sink_union(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
    kv_fp4: &[u8],
    kv_scales: &[u8],
    selected: &[u32],
    position: u32,
    sink: f32,
) -> Vec<f32> {
    union_attention_impl(
        q,
        window_codes,
        window_scales,
        window_rows,
        kv_fp4,
        kv_scales,
        selected,
        position,
        Some(sink),
    )
}

/// The three-pass union attention: maximum over the union, normalising sum plus the sink, then the
/// weighted value sum. The slot order is the reference's concatenation - window rows
/// 0..window_rows first, then the selected list in the order given - so a duplicate changes the
/// arithmetic exactly as the engine's concatenated index list does.
#[allow(clippy::too_many_arguments)] // the twin's private form: the same inputs plus the optional sink.
fn union_attention_impl(
    q: &[f32],
    window_codes: &[u8],
    window_scales: &[u8],
    window_rows: usize,
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
    assert!(
        window_rows <= SWA_WINDOW as usize,
        "the union window is at most the sliding window"
    );
    assert!(
        window_rows + selected.len() <= MAX_ATTENDED,
        "the union attends at most window plus selected slots"
    );
    assert!(
        window_codes.len() >= window_rows * MLA_LATENT
            && window_scales.len() >= window_rows * WINDOW_SCALE_BYTES,
        "the window rows are shorter than the declared row count"
    );
    for candidate in selected {
        if union_attendable(position, *candidate) {
            let row = *candidate as usize;
            assert!(
                kv_fp4.len() >= (row + 1) * ROW_PACKED_BYTES
                    && kv_scales.len() >= (row + 1) * ROW_SCALE_BYTES,
                "an attended compressed id has no row in the expanded KV view"
            );
        }
    }

    // Pass one: the maximum over the union, window rows first and then the selected list.
    let mut maximum = f32::NEG_INFINITY;
    for row in 0..window_rows {
        let score = score_window_candidate(q, window_codes, window_scales, row as u32);
        if score > maximum {
            maximum = score;
        }
    }
    for candidate in selected {
        if union_attendable(position, *candidate) {
            let score = score_candidate(q, kv_fp4, kv_scales, *candidate);
            if score > maximum {
                maximum = score;
            }
        }
    }

    // Pass two: the sink's term and the normalising sum, in the released order - the sink is the
    // first denominator term, then the attended slots in slot order. The kernel's union branch and
    // sparkflash_engine::forward::attention::sparse_attention use this same order, which is what
    // makes the f32 sum bit-comparable rather than only close.
    let mut weight_sum = 0.0f32;
    if maximum > f32::NEG_INFINITY {
        if let Some(sink) = sink {
            weight_sum += (sink - maximum).exp();
        }
        for row in 0..window_rows {
            weight_sum += (score_window_candidate(q, window_codes, window_scales, row as u32)
                - maximum)
                .exp();
        }
        for candidate in selected {
            if union_attendable(position, *candidate) {
                weight_sum += (score_candidate(q, kv_fp4, kv_scales, *candidate) - maximum).exp();
            }
        }
    }

    // Pass three: the weighted value sum in the same order. An id past the query is skipped
    // outright, so its contribution is exactly zero rather than a small weight.
    let mut out = vec![0.0f32; MLA_LATENT];
    if maximum > f32::NEG_INFINITY {
        for row in 0..window_rows {
            let weight = (score_window_candidate(q, window_codes, window_scales, row as u32)
                - maximum)
                .exp();
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight * window_kv_value(window_codes, window_scales, row, index);
            }
        }
        for candidate in selected {
            if !union_attendable(position, *candidate) {
                continue;
            }
            let weight = (score_candidate(q, kv_fp4, kv_scales, *candidate) - maximum).exp();
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight * kv_latent(kv_fp4, kv_scales, *candidate as usize, index);
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

/// One lane's four products of the indexer dot, reduced inside the warp exactly as the kernel
/// reduces them: five shfl_down steps, 16, 8, 4, 2, 1, each lane adding the value of the lane that
/// many positions above it. Only lane 0's value is the block score, which is the value kept here.
fn indexer_warp_sum(per_lane: &[f32; 32]) -> f32 {
    let mut values = *per_lane;
    let mut offset = 16usize;
    while offset >= 1 {
        for lane in 0..offset {
            values[lane] += values[lane + offset];
        }
        offset /= 2;
    }
    values[0]
}

/// The position-causal indexer block scoring of the sequence entry
/// (kernels/flash_mla_csa2_seq_sm121.cu), for ONE indexer head.
///
/// Returns (best score, best token) per HSI block. A block whose first token is past the query, or
/// whose reachable tokens are all past it, publishes (-f32::MAX, 0) - the kernel's empty result. The
/// dot is 128 values wide, four per lane, and is reduced with the kernel's own warp tree, so the
/// arithmetic order matches; FMA contraction in the compiled kernel can still move the score by an
/// ULP, which is why the harness compares with the attention bar rather than bit-for-bit.
///
/// # Panics
/// Panics if the key or query slices do not have the pinned shape.
#[must_use]
pub fn indexer_block_scores(
    index_key: &[u16],
    index_query: &[u16],
    position: u32,
) -> Vec<(f32, u32)> {
    assert_eq!(
        index_key.len(),
        HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM,
        "the index key pool must be 2,048 x 8 x 128 bf16 values"
    );
    assert_eq!(
        index_query.len(),
        INDEX_HEAD_DIM,
        "one indexer query row is 128 bf16 values"
    );
    let query: [f32; INDEX_HEAD_DIM] = std::array::from_fn(|index| bf16_value(index_query[index]));
    let mut blocks = Vec::with_capacity(HSI_BLOCKS);
    for block in 0..HSI_BLOCKS {
        let first_token = block * HSI_BLOCK_TOKENS;
        let mut best = -f32::MAX;
        let mut best_token = 0u32;
        if (first_token as u64) <= u64::from(position) {
            for slot in 0..HSI_BLOCK_TOKENS {
                let token = first_token + slot;
                if (token as u64) > u64::from(position) {
                    break;
                }
                let key_row = &index_key[(block * HSI_BLOCK_TOKENS + slot) * INDEX_HEAD_DIM
                    ..(block * HSI_BLOCK_TOKENS + slot + 1) * INDEX_HEAD_DIM];
                let mut per_lane = [0.0f32; 32];
                for (lane, value) in per_lane.iter_mut().enumerate() {
                    let q0 = query[lane];
                    let q1 = query[lane + 32];
                    let q2 = query[lane + 64];
                    let q3 = query[lane + 96];
                    *value = q0 * bf16_value(key_row[lane])
                        + q1 * bf16_value(key_row[lane + 32])
                        + q2 * bf16_value(key_row[lane + 64])
                        + q3 * bf16_value(key_row[lane + 96]);
                }
                let score = indexer_warp_sum(&per_lane);
                if score > best {
                    best = score;
                    best_token = u32::try_from(token).expect("a token fits a u32");
                }
            }
        }
        blocks.push((best, best_token));
    }
    blocks
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

    // ---- CSA union attention (window rows + compressed candidates) -------------

    /// A query that reads one latent channel only, so a score is that channel's value times the
    /// MLA score scale.
    fn channel_query(channel: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; MLA_LATENT];
        q[channel] = 1.0;
        q
    }

    /// Engine-format bytes for a list of f32 rows, exactly as the harness builds them: the engine
    /// quantiser's E4M3 codes plus one E8M0 scale per 32 channels.
    fn engine_rows(rows: &[Vec<f32>]) -> (Vec<u8>, Vec<u8>) {
        let mut codes = Vec::with_capacity(rows.len() * MLA_LATENT);
        let mut scales = Vec::with_capacity(rows.len() * WINDOW_SCALE_BYTES);
        for row in rows {
            let (row_codes, row_scales) = crate::m4_quant::act_quant_fp8_row(row);
            codes.extend_from_slice(&row_codes);
            scales.extend_from_slice(&row_scales);
        }
        (codes, scales)
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn union_attendable_is_the_causal_comparison_not_the_window() {
        // The distinction the mode exists for: at position 129 the token at 0 is outside the
        // 128-token window but is still a legal compressed candidate, so the union attends it.
        assert!(union_attendable(129, 0));
        assert!(union_attendable(129, 129));
        assert!(!union_attendable(129, 130));
        assert!(union_attendable(0, 0));
        assert!(!union_attendable(0, 1));
        assert!(!in_window(129, 0), "the historical window excludes it");
        assert!(in_window(129, 129));
    }

    #[test]
    fn a_selected_id_outside_the_window_contributes_in_union_mode() {
        // Position 129, candidate 0: 129 tokens behind, so the window predicate excludes it, but
        // the union must attend it. The window row is all zeros and the candidate's channel 0 is
        // 1.0, so window-only gives 0 and the union gives 1 / (1 + exp(-scale)).
        let q = channel_query(0);
        let (codes, scales) = engine_rows(&[vec![0.0f32; MLA_LATENT]]);
        let mut kv = Kv::new(130);
        kv.set(0, 0, 0x2); // 1.0
        let window_only = attention_window(&q, &codes, &scales, 1, &[]);
        assert_eq!(window_only[0], 0.0, "a zero window row has zero value");
        let union = attention_union(&q, &codes, &scales, 1, &kv.fp4, &kv.scales, &[0], 129);
        let expected = 1.0 / (1.0 + (-MLA_SCORE_SCALE).exp());
        assert!(
            (union[0] - expected).abs() < 1e-6,
            "the out-of-window candidate must contribute: got {} expected {expected}",
            union[0]
        );
        assert!(
            union[0] > 0.5,
            "a window-only answer would be 0; got {}",
            union[0]
        );
    }

    #[test]
    fn a_selected_id_past_the_position_is_excluded_exactly() {
        let q = channel_query(0);
        let (codes, scales) = engine_rows(&[vec![0.0f32; MLA_LATENT]]);
        let mut kv = Kv::new(7);
        kv.set(6, 0, 0x7); // 6.0, and it must not be seen at position 5
        let future = attention_union(&q, &codes, &scales, 1, &kv.fp4, &kv.scales, &[6], 5);
        let without = attention_union(&q, &codes, &scales, 1, &kv.fp4, &kv.scales, &[], 5);
        assert_eq!(
            bits(&future),
            bits(&without),
            "a future id must not change the output by even one bit"
        );
        assert!(future.iter().all(|value| *value == 0.0));
        // The same id at the query's own position is attended, so the exclusion is the causal
        // comparison and not an off-by-one that drops everything.
        let mut kv5 = Kv::new(6);
        kv5.set(5, 0, 0x7);
        let attended = attention_union(&q, &codes, &scales, 1, &kv5.fp4, &kv5.scales, &[5], 5);
        assert!(attended[0] > 0.0);
    }

    #[test]
    fn duplicate_selected_ids_are_attended_once_per_entry() {
        // Window empty, two compressed rows: row 0 is 1.0 and row 1 is 2.0 in channel 0. With
        // [0, 1] the weights are (exp(-s), 1); with [0, 0, 1] row 0 has twice the weight, so the
        // output moves towards 1.0. A dedup would leave both calls identical.
        let q = channel_query(0);
        let mut kv = Kv::new(2);
        kv.set(0, 0, 0x2); // 1.0
        kv.set(1, 0, 0x4); // 2.0
        let single = attention_union(&q, &[], &[], 0, &kv.fp4, &kv.scales, &[0, 1], 10);
        let duplicated = attention_union(&q, &[], &[], 0, &kv.fp4, &kv.scales, &[0, 0, 1], 10);
        let lower = (-MLA_SCORE_SCALE).exp();
        let expected_single = (lower + 2.0) / (lower + 1.0);
        let expected_duplicated = (2.0 * lower + 2.0) / (2.0 * lower + 1.0);
        assert!((single[0] - expected_single).abs() < 1e-6);
        assert!((duplicated[0] - expected_duplicated).abs() < 1e-6);
        assert!(
            (single[0] - duplicated[0]).abs() > 1e-3,
            "a duplicate must change the mixture: {} vs {}",
            single[0],
            duplicated[0]
        );
    }

    #[test]
    fn a_token_in_the_window_and_also_selected_contributes_both_values() {
        // The window row stores 1.0 and the compressed row for the same token stores 2.0: they are
        // two different KV values in the reference's concatenated index space, and both are
        // attended. A dedup by token id would drop one of them.
        let q = channel_query(0);
        let (codes, scales) = engine_rows(&[vec![1.0f32; MLA_LATENT]]);
        let mut kv = Kv::new(1);
        kv.set(0, 0, 0x4); // 2.0 in the compressed row
        let union = attention_union(&q, &codes, &scales, 1, &kv.fp4, &kv.scales, &[0], 0);
        // Window score 1*s, compressed score 2*s; the weights are exp(-s) and 1.
        let lower = (-MLA_SCORE_SCALE).exp();
        let expected = (lower + 2.0) / (lower + 1.0);
        assert!(
            (union[0] - expected).abs() < 1e-6,
            "got {} expected {expected}",
            union[0]
        );
        assert!(union[0] > 1.0 && union[0] < 2.0);
    }

    #[test]
    fn an_empty_union_is_zero_even_with_a_sink() {
        let q = channel_query(0);
        assert!(
            attention_union(&q, &[], &[], 0, &[], &[], &[], 5)
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            attention_with_sink_union(&q, &[], &[], 0, &[], &[], &[], 5, 5.0)
                .iter()
                .all(|value| *value == 0.0)
        );
        // A future-only list is the same case; its rows need not exist because they are never read.
        let future = attention_union(&q, &[], &[], 0, &[], &[], &[6, 7], 5);
        assert!(future.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn union_with_no_selected_is_the_window_attention_bit_for_bit() {
        let q: Vec<f32> = (0..MLA_LATENT)
            .map(|index| ((index as f32 * 0.013).cos() * 0.5) + ((index % 11) as f32 * 0.02))
            .collect();
        let rows = 3usize;
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(
                (0..MLA_LATENT)
                    .map(|index| ((row as f32 + 1.0) * 0.7 + index as f32 * 0.019).sin())
                    .collect::<Vec<f32>>(),
            );
        }
        let (codes, scales) = engine_rows(&values);
        // In the engine-format window call the window rows contribute only when their row ids are
        // listed; the union mode attends every supplied window row, so the equivalent window call
        // lists rows 0..window_rows. No selected compressed candidates are involved.
        let listed: Vec<u32> = (0..rows as u32).collect();
        let window = attention_window(&q, &codes, &scales, rows, &listed);
        let union = attention_union(&q, &codes, &scales, rows, &[], &[], &[], 0);
        assert_eq!(bits(&window), bits(&union));
        let window_sink = attention_with_sink_window(&q, &codes, &scales, rows, &listed, 0.75);
        let union_sink =
            attention_with_sink_union(&q, &codes, &scales, rows, &[], &[], &[], 0, 0.75);
        assert_eq!(bits(&window_sink), bits(&union_sink));
    }

    #[test]
    fn an_unreachable_union_sink_reproduces_the_no_sink_bits() {
        let q = channel_query(0);
        let (codes, scales) = engine_rows(&[vec![0.5f32; MLA_LATENT]]);
        let mut kv = Kv::new(1);
        kv.set(0, 0, 0x4);
        let without = attention_union(&q, &codes, &scales, 1, &kv.fp4, &kv.scales, &[0], 0);
        for sink in [f32::NEG_INFINITY, -1.0e30] {
            let with = attention_with_sink_union(
                &q,
                &codes,
                &scales,
                1,
                &kv.fp4,
                &kv.scales,
                &[0],
                0,
                sink,
            );
            assert_eq!(
                bits(&without),
                bits(&with),
                "a sink of {sink} must not change the bits"
            );
        }
    }

    /// The union against a slow f64 reference built from the same decode rules: window rows first,
    /// then the causal selected ids, one sink term after the maximum. The selected list includes
    /// ids both inside and outside the window, a duplicate row (the same compressed id listed
    /// twice), an id past the position (excluded), and the id equal to the position (attended).
    #[test]
    fn union_attention_matches_a_slow_f64_reference() {
        let q: Vec<f32> = (0..MLA_LATENT)
            .map(|index| ((index as f32 * 0.017).sin() * 0.75) + ((index % 7) as f32 * 0.01))
            .collect();
        let window_rows = 2usize;
        let mut window_values = Vec::with_capacity(window_rows);
        for row in 0..window_rows {
            window_values.push(
                (0..MLA_LATENT)
                    .map(|index| {
                        let angle = (row as f32 + 1.0) * 0.31 + index as f32 * 0.023;
                        (angle.sin() * 2.0) + (index % 5) as f32 * 0.05
                    })
                    .collect::<Vec<f32>>(),
            );
        }
        let (codes, scales) = engine_rows(&window_values);
        let mut kv = Kv::new(201);
        for row in [0usize, 1, 127, 129, 199] {
            kv.set(row, 0, 0x2); // 1.0
            kv.set(row, 1, if row % 2 == 0 { 0x4 } else { 0x5 }); // 2.0 / 3.0
        }
        let position = 199u32;
        let selected = [0u32, 1, 1, 127, 129, 199, 200];
        let sink = 0.25f32;

        let mut scores = Vec::new();
        let mut values = Vec::new();
        for row in 0..window_rows {
            let mut score = 0.0f64;
            for (index, q_value) in q.iter().enumerate() {
                let value = f64::from(window_kv_value(&codes, &scales, row, index));
                values.push(value);
                score += f64::from(*q_value) * value;
            }
            scores.push(score * f64::from(MLA_SCORE_SCALE));
        }
        for candidate in selected {
            if !union_attendable(position, candidate) {
                continue;
            }
            let mut score = 0.0f64;
            for (index, q_value) in q.iter().enumerate() {
                let value = f64::from(kv_latent(&kv.fp4, &kv.scales, candidate as usize, index));
                values.push(value);
                score += f64::from(*q_value) * value;
            }
            scores.push(score * f64::from(MLA_SCORE_SCALE));
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut denominator = (f64::from(sink) - maximum).exp();
        for score in &scores {
            denominator += (score - maximum).exp();
        }
        let mut expected = vec![0.0f64; MLA_LATENT];
        for (slot, score) in scores.iter().enumerate() {
            let weight = (score - maximum).exp() / denominator;
            for index in 0..MLA_LATENT {
                expected[index] += weight * values[slot * MLA_LATENT + index];
            }
        }
        let union = attention_with_sink_union(
            &q,
            &codes,
            &scales,
            window_rows,
            &kv.fp4,
            &kv.scales,
            &selected,
            position,
            sink,
        );
        for index in 0..MLA_LATENT {
            let delta = (f64::from(union[index]) - expected[index]).abs();
            assert!(
                delta <= 1e-5,
                "channel {index}: union differs from the slow f64 result by {delta:e}"
            );
        }

        // The same union without a sink must agree with the same f64 reference with the sink term
        // removed, and the sink must attenuate every channel (it adds mass, not a value).
        let no_sink = attention_union(
            &q,
            &codes,
            &scales,
            window_rows,
            &kv.fp4,
            &kv.scales,
            &selected,
            position,
        );
        let candidate_sum: f64 = scores.iter().map(|score| (score - maximum).exp()).sum();
        let mut expected_no_sink = vec![0.0f64; MLA_LATENT];
        for (slot, score) in scores.iter().enumerate() {
            let weight = (score - maximum).exp() / candidate_sum;
            for index in 0..MLA_LATENT {
                expected_no_sink[index] += weight * values[slot * MLA_LATENT + index];
            }
        }
        for index in 0..MLA_LATENT {
            let delta = (f64::from(no_sink[index]) - expected_no_sink[index]).abs();
            assert!(
                delta <= 1e-5,
                "no-sink channel {index} differs by {delta:e}"
            );
            assert!(
                union[index].abs() <= no_sink[index].abs() + 1e-6,
                "the sink must only remove mass, channel {index}"
            );
        }
    }

    // ---- CSA union mode against the engine's own sparse_attention reference ----

    fn window_row(value: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; MLA_LATENT];
        row[0] = value;
        row
    }

    fn store_kv(row_codes: &[u8]) -> Kv {
        let mut kv = Kv::new(row_codes.len());
        for (row, code) in row_codes.iter().enumerate() {
            kv.set(row, 0, *code);
        }
        kv
    }

    /// The per-position view a ratio-N source hands union mode: row p decodes exactly like store row
    /// p / ratio. This is option A of the task-11 contract, and the equality test below is what
    /// proves it is not double-counting.
    fn expand_store(store: &Kv, ratio: usize) -> Kv {
        assert!(ratio >= 1, "a ratio is at least one token per store row");
        let mut expanded = Kv::new(store.rows * ratio);
        for row in 0..expanded.rows {
            let source = row / ratio;
            expanded.fp4[row * ROW_PACKED_BYTES..(row + 1) * ROW_PACKED_BYTES].copy_from_slice(
                &store.fp4[source * ROW_PACKED_BYTES..(source + 1) * ROW_PACKED_BYTES],
            );
            expanded.scales[row * ROW_SCALE_BYTES..(row + 1) * ROW_SCALE_BYTES].copy_from_slice(
                &store.scales[source * ROW_SCALE_BYTES..(source + 1) * ROW_SCALE_BYTES],
            );
        }
        expanded
    }

    /// The engine's own sparse_attention over the concatenated reference KV: window rows first, then
    /// the compressed store rows, with indices = window rows ++ compressed ids shifted by the window
    /// and usize::MAX for a future (masked) id. This mirrors attention_forward exactly, including
    /// the sink being the first denominator term. The caller passes the *store* (unexpanded) so a
    /// ratio-N case is compared against the engine's own compressed rows, not our expansion.
    #[allow(clippy::too_many_arguments)]
    fn reference_union_output(
        q: &[f32],
        window_codes: &[u8],
        window_scales: &[u8],
        window_rows: usize,
        store: &Kv,
        ratio: usize,
        selected: &[u32],
        position: u32,
        sink: f32,
    ) -> Vec<f32> {
        let mut kv = Vec::with_capacity((window_rows + store.rows) * MLA_LATENT);
        for row in 0..window_rows {
            for index in 0..MLA_LATENT {
                kv.push(window_kv_value(window_codes, window_scales, row, index));
            }
        }
        let compressed_base = window_rows;
        for row in 0..store.rows {
            for index in 0..MLA_LATENT {
                kv.push(kv_latent(&store.fp4, &store.scales, row, index));
            }
        }
        let mut indices: Vec<usize> = (0..window_rows).collect();
        for id in selected {
            if *id <= position {
                let id = *id as usize;
                assert!(
                    id.is_multiple_of(ratio),
                    "an absolute id must lie on the compressed source's token grid"
                );
                indices.push(compressed_base + id / ratio);
            } else {
                // The reference's NO_INDEX is usize::MAX, and sparse_attention masks it.
                indices.push(usize::MAX);
            }
        }
        let mut out = vec![0.0f32; MLA_LATENT];
        sparkflash_engine::forward::attention::sparse_attention(
            q,
            &kv,
            &[sink],
            &indices,
            1,
            MLA_LATENT,
            MLA_SCORE_SCALE,
            &mut out,
        )
        .expect("the reference sparse attention accepts whole 512-wide rows");
        out
    }

    struct UnionCase {
        label: &'static str,
        position: u32,
        window: Vec<Vec<f32>>,
        store: Vec<u8>,
        ratio: usize,
        selected: Vec<u32>,
    }

    /// The hard condition of the task-11 contract: format 2 must equal the ENGINE reference, not a
    /// hand-written union softmax of our own definition. Each case is compared bit for bit with
    /// sparkflash_engine::forward::attention::sparse_attention over the concatenated reference KV,
    /// and the case list covers the required multiset shapes: distinct compressed ids, a compressed
    /// id inside the window, a duplicate entry, an id at the query position, an id past it, and two
    /// ratio-2 expanded-view cases (one with a compressed id inside the window).
    #[test]
    fn format_two_union_equals_the_engine_reference_on_the_attended_multiset() {
        let q = channel_query(0);
        let sink = 0.75f32;
        let cases = vec![
            UnionCase {
                label: "distinct compressed ids outside the window, ratio 1",
                position: 129,
                window: vec![window_row(0.5), window_row(1.5)],
                store: (0..130)
                    .map(|row| [0x2u8, 0x4, 0x7, 0x1][row % 4])
                    .collect(),
                ratio: 1,
                selected: vec![0, 3, 100],
            },
            UnionCase {
                label: "a compressed id inside the sliding window, ratio 1",
                position: 10,
                window: vec![
                    window_row(2.0),
                    window_row(0.5),
                    window_row(1.0),
                    window_row(3.0),
                    window_row(0.5),
                    window_row(2.0),
                    window_row(1.5),
                    window_row(0.5),
                    window_row(2.5),
                ],
                store: (0..12).map(|row| [0x4u8, 0x7, 0x1, 0x2][row % 4]).collect(),
                ratio: 1,
                selected: vec![4, 7],
            },
            UnionCase {
                label: "duplicate selected entries, ratio 1",
                position: 9,
                window: vec![window_row(1.0), window_row(2.0)],
                store: (0..8).map(|row| [0x6u8, 0x1, 0x4, 0x2][row % 4]).collect(),
                ratio: 1,
                selected: vec![3, 3, 5],
            },
            UnionCase {
                label: "a compressed id at the query position, ratio 1",
                position: 7,
                window: vec![window_row(3.0), window_row(0.5)],
                store: (0..8).map(|row| [0x1u8, 0x7, 0x2, 0x4][row % 4]).collect(),
                ratio: 1,
                selected: vec![7, 2],
            },
            UnionCase {
                label: "a compressed id past the query position, ratio 1",
                position: 7,
                window: vec![window_row(0.5), window_row(3.0)],
                store: (0..9).map(|row| [0x2u8, 0x4, 0x1, 0x7][row % 4]).collect(),
                ratio: 1,
                selected: vec![8, 2],
            },
            UnionCase {
                label: "expanded per-position view, ratio 2",
                position: 5,
                window: vec![window_row(1.5), window_row(0.5), window_row(2.5)],
                store: vec![0x2, 0x4, 0x7, 0x1],
                ratio: 2,
                selected: vec![0, 2, 4],
            },
            UnionCase {
                label: "expanded per-position view with a compressed id inside the window, ratio 2",
                position: 5,
                window: vec![window_row(0.5), window_row(2.0), window_row(1.5)],
                store: vec![0x7, 0x1, 0x4, 0x2],
                ratio: 2,
                selected: vec![4],
            },
        ];
        for case in &cases {
            let (window_codes, window_scales) = engine_rows(&case.window);
            let store = store_kv(&case.store);
            let expanded = expand_store(&store, case.ratio);
            let window_rows = case.window.len();
            let got = attention_with_sink_union(
                &q,
                &window_codes,
                &window_scales,
                window_rows,
                &expanded.fp4,
                &expanded.scales,
                &case.selected,
                case.position,
                sink,
            );
            let want = reference_union_output(
                &q,
                &window_codes,
                &window_scales,
                window_rows,
                &store,
                case.ratio,
                &case.selected,
                case.position,
                sink,
            );
            assert_eq!(
                bits(&got),
                bits(&want),
                "{}: format 2 must reproduce sparse_attention on the concatenated reference KV",
                case.label
            );

            // The union must not be the window-only answer: at least one selected entry changes it.
            let window_only = attention_with_sink_union(
                &q,
                &window_codes,
                &window_scales,
                window_rows,
                &expanded.fp4,
                &expanded.scales,
                &[],
                case.position,
                sink,
            );
            assert_ne!(
                bits(&got),
                bits(&window_only),
                "{}: the selected ids must contribute to the union",
                case.label
            );

            // Future ids are masked exactly: dropping them must not change a bit. This is the
            // reference's NO_INDEX column and our union_attendable() exclusion.
            let attended: Vec<u32> = case
                .selected
                .iter()
                .copied()
                .filter(|id| *id <= case.position)
                .collect();
            if attended.len() != case.selected.len() {
                let without_future = attention_with_sink_union(
                    &q,
                    &window_codes,
                    &window_scales,
                    window_rows,
                    &expanded.fp4,
                    &expanded.scales,
                    &attended,
                    case.position,
                    sink,
                );
                assert_eq!(
                    bits(&got),
                    bits(&without_future),
                    "{}: a future id must be excluded exactly",
                    case.label
                );
            }

            // A duplicate is attended once per entry, in list order: de-duplicating changes the
            // result, and the engine reference above was given the duplicate index too.
            let mut unique = case.selected.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != case.selected.len() {
                let deduped = attention_with_sink_union(
                    &q,
                    &window_codes,
                    &window_scales,
                    window_rows,
                    &expanded.fp4,
                    &expanded.scales,
                    &unique,
                    case.position,
                    sink,
                );
                assert_ne!(
                    bits(&got),
                    bits(&deduped),
                    "{}: a duplicate selected entry is attended once per entry",
                    case.label
                );
            }

            // Option A's injective id mapping: compressed store row c is absolute position c*ratio,
            // and that expanded row decodes to exactly the store row.
            if case.ratio > 1 {
                assert!(
                    case.selected
                        .iter()
                        .all(|id| (*id as usize).is_multiple_of(case.ratio)),
                    "{}: ids are absolute positions on the compressed grid",
                    case.label
                );
                for row in 0..store.rows {
                    let id = row * case.ratio;
                    for index in 0..MLA_LATENT {
                        assert_eq!(
                            kv_latent(&store.fp4, &store.scales, row, index).to_bits(),
                            kv_latent(&expanded.fp4, &expanded.scales, id, index).to_bits(),
                            "{}: store row {row} must be expanded row {id}",
                            case.label
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ratio_two_expansion_cannot_double_count_a_store_row() {
        let q = channel_query(0);
        let (window_codes, window_scales) = engine_rows(&[window_row(0.5)]);
        let store = store_kv(&[0x2, 0x4, 0x7]);
        let expanded = expand_store(&store, 2);
        let position = 5u32;
        let sink = 0.25f32;
        // The correct absolute ids: compressed rows 0, 1, 2 are token positions 0, 2, 4.
        let correct = [0u32, 2, 4];
        let got = attention_with_sink_union(
            &q,
            &window_codes,
            &window_scales,
            1,
            &expanded.fp4,
            &expanded.scales,
            &correct,
            position,
            sink,
        );
        let want = reference_union_output(
            &q,
            &window_codes,
            &window_scales,
            1,
            &store,
            2,
            &correct,
            position,
            sink,
        );
        assert_eq!(
            bits(&got),
            bits(&want),
            "ids c*ratio must map exactly to compressed row c"
        );
        // The double-count hazard: using the compressed row index as the absolute id reads the
        // expanded view row 1, which is store row 0 again, so row 0 is counted twice and row 2 is
        // never attended. That answer must not equal the reference.
        let naive = [0u32, 1, 2];
        let wrong = attention_with_sink_union(
            &q,
            &window_codes,
            &window_scales,
            1,
            &expanded.fp4,
            &expanded.scales,
            &naive,
            position,
            sink,
        );
        assert_ne!(
            bits(&wrong),
            bits(&want),
            "compressed indices used as token ids double-count an expanded store row"
        );
        assert_eq!(
            kv_latent(&expanded.fp4, &expanded.scales, 0, 0).to_bits(),
            kv_latent(&expanded.fp4, &expanded.scales, 1, 0).to_bits(),
            "the expansion intentionally duplicates each store row"
        );
    }

    /// Compute the union output with the sink term in the requested denominator position, from the
    /// same gathered scores and values the implementation uses. This exists only to make the
    /// addition order observable in an assertion; the engine-reference test separately pins the
    /// actual value.
    #[allow(clippy::too_many_arguments)]
    fn ordered_union_output(
        q: &[f32],
        window_codes: &[u8],
        window_scales: &[u8],
        window_rows: usize,
        kv: &Kv,
        selected: &[u32],
        position: u32,
        sink: f32,
        sink_first: bool,
    ) -> Vec<f32> {
        let mut scores = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for row in 0..window_rows {
            scores.push(score_window_candidate(
                q,
                window_codes,
                window_scales,
                row as u32,
            ));
            for index in 0..MLA_LATENT {
                values.push(window_kv_value(window_codes, window_scales, row, index));
            }
        }
        for candidate in selected {
            if union_attendable(position, *candidate) {
                scores.push(score_candidate(q, &kv.fp4, &kv.scales, *candidate));
                for index in 0..MLA_LATENT {
                    values.push(kv_latent(&kv.fp4, &kv.scales, *candidate as usize, index));
                }
            }
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = if sink_first {
            (sink - maximum).exp()
        } else {
            0.0
        };
        for score in &scores {
            denominator += (score - maximum).exp();
        }
        if !sink_first {
            denominator += (sink - maximum).exp();
        }
        let mut out = vec![0.0f32; MLA_LATENT];
        for (slot, score) in scores.iter().enumerate() {
            let weight = (score - maximum).exp();
            for (index, value) in out.iter_mut().enumerate() {
                *value += weight * values[slot * MLA_LATENT + index];
            }
        }
        for value in out.iter_mut() {
            *value /= denominator;
        }
        out
    }

    #[test]
    fn the_union_adds_the_sink_term_first_in_the_denominator() {
        // critic46's review finding: moving the sink term from first to last denominator position
        // still left the engine-reference test passing, because that case never made the addition
        // order bit-visible. This case does. Seventeen near-2^-24 candidate weights plus a dominant
        // sink/maximum term make the two denominator accumulations differ by several f32 ULPs, and
        // the weighted output differs too, so a future reorder fails the assertion.
        let mut q = vec![0.0f32; MLA_LATENT];
        q[0] = 375.0;
        let (codes, scales) = engine_rows(&[window_row(1.0), window_row(1.0)]);
        let mut kv = Kv::new(16);
        for row in 0..15 {
            kv.set(row, 0, 0x2); // 1.0
        }
        kv.set(15, 0, 0x4); // 2.0, the maximum (set after the loop: set() ORs nibbles)
        let selected: Vec<u32> = (0..16).collect();
        let position = 20u32;
        // The maximum is the 2.0 row; the seventeen 1.0 rows contribute weights of
        // exp(-375*scale) ~ 2^-24 each, so their sum is large enough to move the denominator by
        // more than one ULP of the output when it is accumulated after the sink instead of before.
        let maximum = 2.0 * 375.0 * MLA_SCORE_SCALE;
        let mut observable = None;
        for step in -1000i32..=1000 {
            let sink = maximum + step as f32 * 1e-3;
            let first =
                ordered_union_output(&q, &codes, &scales, 2, &kv, &selected, position, sink, true);
            let last = ordered_union_output(
                &q, &codes, &scales, 2, &kv, &selected, position, sink, false,
            );
            if bits(&first) != bits(&last) {
                observable = Some((sink, first, last));
                break;
            }
        }
        let (sink, sink_first_expected, sink_last_expected) =
            observable.expect("the case must make the denominator addition order observable");
        assert_ne!(
            bits(&sink_first_expected),
            bits(&sink_last_expected),
            "the searched sink must make the two denominator orders differ"
        );
        let got = attention_with_sink_union(
            &q, &codes, &scales, 2, &kv.fp4, &kv.scales, &selected, position, sink,
        );
        assert_eq!(
            bits(&got),
            bits(&sink_first_expected),
            "the sink term must be added before the candidate weights"
        );
        assert_ne!(
            bits(&got),
            bits(&sink_last_expected),
            "this case must fail if the sink is moved to the last denominator term"
        );
    }

    #[test]
    fn the_union_capacity_accepts_exactly_640_slots() {
        // 128 window rows + 512 candidates = 640, the full bound. The kernel refuses 641 with a
        // NaN poison at its entry guard; this twin refuses it with a panic (the #[should_panic]
        // test below). Exactly 640 must run, not be refused.
        let q = channel_query(0);
        let window = vec![vec![0.0f32; MLA_LATENT]; 128];
        let (codes, scales) = engine_rows(&window);
        let mut store = Kv::new(512);
        for row in 0..512 {
            store.set(row, 0, 0x2);
        }
        let selected: Vec<u32> = (0..512).collect();
        let out = attention_with_sink_union(
            &q,
            &codes,
            &scales,
            128,
            &store.fp4,
            &store.scales,
            &selected,
            511,
            0.0,
        );
        assert_eq!(out.len(), MLA_LATENT);
        assert!(
            out.iter().all(|value| value.is_finite()),
            "exactly 640 slots must produce a finite output"
        );
    }

    #[test]
    #[should_panic(expected = "the union window is at most the sliding window")]
    fn the_union_capacity_refuses_641_slots() {
        let q = channel_query(0);
        let window = vec![vec![0.0f32; MLA_LATENT]; 129];
        let (codes, scales) = engine_rows(&window);
        let mut store = Kv::new(512);
        for row in 0..512 {
            store.set(row, 0, 0x2);
        }
        let selected: Vec<u32> = (0..512).collect();
        let _ = attention_with_sink_union(
            &q,
            &codes,
            &scales,
            129,
            &store.fp4,
            &store.scales,
            &selected,
            511,
            0.0,
        );
    }

    #[test]
    #[should_panic(expected = "the candidate list is index_topk long")]
    fn the_union_refuses_more_than_512_candidates() {
        let q = channel_query(0);
        let window = vec![vec![0.0f32; MLA_LATENT]; 1];
        let (codes, scales) = engine_rows(&window);
        let mut store = Kv::new(513);
        for row in 0..513 {
            store.set(row, 0, 0x2);
        }
        let selected: Vec<u32> = (0..513).collect();
        let _ = attention_with_sink_union(
            &q,
            &codes,
            &scales,
            1,
            &store.fp4,
            &store.scales,
            &selected,
            512,
            0.0,
        );
    }
    #[test]
    fn the_indexer_is_position_causal_and_publishes_the_empty_block_value() {
        let key = vec![0x3f80u16; HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM];
        let query = vec![0x3f80u16; INDEX_HEAD_DIM];
        let blocks = indexer_block_scores(&key, &query, 0);
        assert_eq!(blocks.len(), HSI_BLOCKS);
        assert_eq!(blocks[0].1, 0);
        assert!(blocks[0].0.is_finite());
        for (block, (score, token)) in blocks.iter().enumerate().skip(1) {
            assert_eq!(*score, -f32::MAX, "block {block} starts past the query");
            assert_eq!(*token, 0);
        }

        let blocks = indexer_block_scores(&key, &query, 9);
        assert!(blocks[0].0.is_finite());
        assert_eq!(blocks[0].1, 0);
        assert!(blocks[1].0.is_finite());
        assert_eq!(blocks[1].1, 8);
        for (block, (score, token)) in blocks.iter().enumerate().skip(2) {
            assert_eq!(*score, -f32::MAX, "block {block} starts past position 9");
            assert_eq!(*token, 0);
        }
    }

    #[test]
    fn the_indexer_keeps_the_lowest_token_on_an_equal_block_score() {
        // All keys and the query are 1.0, so every slot in a block scores 128.0 and the tie has to
        // resolve to the block's first token, not the last one written.
        let key = vec![0x3f80u16; HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM];
        let query = vec![0x3f80u16; INDEX_HEAD_DIM];
        let blocks = indexer_block_scores(&key, &query, 7);
        assert_eq!(blocks[0].1, 0);
        let blocks = indexer_block_scores(&key, &query, 1);
        assert_eq!(blocks[0].1, 0);
    }
}
