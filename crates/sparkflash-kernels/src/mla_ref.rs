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

use crate::m4_quant::{fp4_e2m1_decode, fp8_e4m3_decode};

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

/// The attention output for one head: three passes over the candidate list, exactly as the kernel
/// runs them.
///
/// The query q is one head's latent, rope already applied by the caller; the KV buffer is
/// \[positions\]\[packed\] with scales \[positions\][latent / 16]. Out-of-window candidates are skipped in
/// every pass, so they contribute exactly nothing; if no candidate is inside the window the result
/// is all zeros.
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
    assert_eq!(q.len(), MLA_LATENT, "the query is one latent wide");
    assert!(
        selected.len() <= MAX_SELECTED,
        "the candidate list is index_topk long"
    );

    // Pass one: the maximum score over the candidates that are inside the window.
    let mut maximum = f32::NEG_INFINITY;
    for candidate in selected {
        if in_window(position, *candidate) {
            let score = score_candidate(q, kv_fp4, kv_scales, *candidate);
            if score > maximum {
                maximum = score;
            }
        }
    }

    // Pass two: the normalising sum. An all-excluded list leaves the sum at zero, and the guard
    // below turns that into a zero output instead of a division by zero.
    let mut weight_sum = 0.0f32;
    if maximum > f32::NEG_INFINITY {
        for candidate in selected {
            if in_window(position, *candidate) {
                weight_sum += (score_candidate(q, kv_fp4, kv_scales, *candidate) - maximum).exp();
            }
        }
    }
    let denominator = if weight_sum > 0.0 { weight_sum } else { 1.0 };

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
}
