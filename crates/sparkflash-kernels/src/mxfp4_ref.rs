//! CPU reference for the W4A8 routed-expert MoE combine
//! (kernels/b12x_mxfp4_moe_sm121.cu).
//!
//! This module is the definition of that kernel's arithmetic, and it also counts what the kernel
//! spends. Three claims are pinned here:
//!
//! 1. the A operand's format (decision D14, docs/DECISIONS.md): FP8 E4M3, one byte per value, with
//!    one ue8m0 scale per 32 values, quantised by the reference's own act_quant (kernel.py:41-124)
//!    rather than by anything that merely resembles it. The per-32 block maximum (not a running
//!    maximum over the row), the 1e-4 amax floor, the round-up-to-a-power-of-two scale rule and the
//!    +-448 clamp are mirrored in crate::m4_quant::act_quant_fp8_row, and
//!    the_reference_quantiser_mirrors_act_quant checks each of them separately. The FP4 weights keep
//!    the checkpoint's \[out, in/32\] grid and the E2M1 packing.
//! 2. the arithmetic: out\[t\]\[j\] = residual + the sum over the top-k slots of that slot's FP8xFP8
//!    block sums, accumulated in f32 in ascending slot order and, inside a slot, over the padded
//!    width ascending in 32-value blocks, with both ue8m0 scales applied to each block's partial
//!    product the way kernel.py:552-553 applies them to the accumulator, and one rounding to bf16
//!    at the very end. The FP4 weight is read through the reference's cast to FP8
//!    (kernel.py:538-540), which is exact on all sixteen E2M1 codes (enumerated by a test here
//!    rather than assumed), and no routing weight is applied: it belongs to the activation operand,
//!    see routed_activation_operand.
//! 3. the 576 -> 640 padding: the kernel MASKS the padded region of the activation row to zero
//!    (index < MOE_WIDTH in its inner loop), and this reference masks identically. The invariant is
//!    therefore a property of the two implementations rather than a precondition on the caller who
//!    fills the activation buffer; a test fills the padding with nonzero garbage and shows the
//!    result does not move, with the garbage present in the fixture so the check is not vacuous.
//!    The 64 extra MACs per output per slot are still issued and still counted, because the kernel
//!    issues them, so the cost of the padding stays visible instead of being assumed away.
//!
//! What D14 changed in the cost, and therefore in the milestone's traffic claim: an activation is
//! one E4M3 byte per value plus one ue8m0 byte per 32, where this kernel used to read half a byte
//! per value. Per routed row that is 640 + 20 bytes against 320 + 20, so the activation traffic
//! nearly doubles (1.94x) while the weight traffic, the MAC count, the padding count and every
//! geometry proof are untouched. MoeCost reports the three byte classes separately so the change is
//! visible in a test rather than only in this comment.
//!
//! The packing is low nibble first, which is the only packing decision in the weight format and is
//! therefore pinned by a test of its own.

use crate::m4_quant::{
    QUANT_BLOCK, bf16_bits, fp4_e2m1_decode, fp8_e4m3_decode, fp8_e4m3_encode, ue8m0_decode,
};

/// E2M1 values per byte in a packed weight row.
pub const FP4_PER_BYTE: usize = 2;
/// The expert width per rank before padding. Mirrors SPARK_M4_MOE_WIDTH.
pub const MOE_WIDTH: usize = 576;
/// The expert width the kernel actually walks. Mirrors SPARK_M4_MOE_WIDTH_PADDED.
pub const MOE_WIDTH_PADDED: usize = 640;
/// Routed experts selected per token. Mirrors SPARK_M4_MOE_TOP_K.
pub const MOE_TOP_K: usize = 6;
/// Activation bytes per value: one, because the operand is FP8 E4M3 (D14; mirrors
/// SPARK_M4_MOE_ACT_BYTES_PER_VALUE in the kernel). A packed E2M1 operand would be half a byte.
pub const MOE_ACT_BYTES_PER_VALUE: usize = 1;
/// The activation row stride the kernel reads: one byte per padded width value, scales aside.
pub const MOE_ACT_BYTES_PER_ROW: usize = MOE_WIDTH_PADDED * MOE_ACT_BYTES_PER_VALUE;
/// The packed FP4 weight row stride.
pub const MOE_WEIGHT_BYTES_PER_ROW: usize = MOE_WIDTH_PADDED / FP4_PER_BYTE;
/// The ue8m0 scale row stride. Both operands use the same 32-value grid, so they share this.
pub const MOE_SCALE_BYTES_PER_ROW: usize = MOE_WIDTH_PADDED / QUANT_BLOCK;
/// The activation bytes one token's operand carries: one row per (token, slot), because the
/// routing weight is applied before quantisation (model.py:849-851).
pub const MOE_ACT_BYTES_PER_TOKEN: usize = MOE_TOP_K * MOE_ACT_BYTES_PER_ROW;

/// The pre-padded row strides the MoE kernel reads, as an explicit contract rather than an
/// implicit use of `MOE_SCALE_BYTES_PER_ROW`.
///
/// D14 fixes the activation operand as FP8 E4M3 on the padded width, so the kernel reduces
/// `K = MOE_WIDTH_PADDED = 640` and walks 20-byte scale rows. The reference instead shards whole
/// experts and keeps `K = 2304`, with `[out, 72]` scale grids for a whole expert or `[out, 18]`
/// for a 576-wide per-rank slice: a loader producing either would misindex every row after the
/// first, because the kernel always advances by 20 bytes per output row. The two extra columns are
/// masked, so this is an addressing contract, not a numeric one. Producing this layout is the
/// checkpoint loader's job, which is not part of milestone 4; it lands with the milestone-5
/// integration that loads real weights and prefetches them, and this type is the Rust-side
/// statement of what that loader must emit.
///
/// ONLY THE FOUR SCALAR STRIDES ARE PINNED. The row ordering is the loader's responsibility, and no
/// stride can see it: the FP4 per-output-row grid the kernel reads,
/// `spark_b12x_row_scale_index(n, 0, 640) = n * 20` (m4_common_sm121.cuh:563, used at
/// b12x_mxfp4_moe_sm121.cu:161-163), and the FP8 blocked grid
/// `spark_b12x_scale_index(n, 0, 640) = (n / 32) * 20` (m4_common_sm121.cuh:557) both present a
/// 20-byte row stride at this width. A loader that stored the blocked grid while reporting
/// `weight_scale_row_bytes = 20` passes `check_moe_layout`; the kernel then reads a different
/// output row's scales, which the stride check cannot detect. The check would have to be told which
/// grid the loader wrote, and the honesty of that declaration is not something a `const fn` can
/// enforce, so the limitation is stated here and pinned by
/// `a_stride_check_cannot_see_the_scale_grid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRowLayout {
    /// E4M3 bytes one routed (token, slot) activation row advances.
    pub activation_row_bytes: usize,
    /// ue8m0 bytes one activation scale row advances.
    pub activation_scale_row_bytes: usize,
    /// Packed FP4 bytes one output row of one expert advances.
    pub weight_row_bytes: usize,
    /// ue8m0 bytes one weight scale row advances.
    pub weight_scale_row_bytes: usize,
}

/// The only row layout `b12x_mxfp4_moe_sm121.cu` reads.
#[must_use]
pub const fn pre_padded_moe_layout() -> MoeRowLayout {
    MoeRowLayout {
        activation_row_bytes: MOE_ACT_BYTES_PER_ROW,
        activation_scale_row_bytes: MOE_SCALE_BYTES_PER_ROW,
        weight_row_bytes: MOE_WEIGHT_BYTES_PER_ROW,
        weight_scale_row_bytes: MOE_SCALE_BYTES_PER_ROW,
    }
}

/// # Errors
/// Returns a description of the first stride that does not match the kernel's pre-padded layout,
/// which is what turns a silent row misindex into a loud rejection.
///
/// The check is a stride check and nothing more: it does not verify the row ordering, so it cannot
/// tell the per-output-row weight scale grid the kernel reads from the blocked 32-row grid that
/// shares its 20-byte stride. See `MoeRowLayout`: the loader owns that part of the contract.
pub fn check_moe_layout(layout: MoeRowLayout) -> Result<(), String> {
    let expected = pre_padded_moe_layout();
    for (name, got, want) in [
        (
            "activation",
            layout.activation_row_bytes,
            expected.activation_row_bytes,
        ),
        (
            "activation scale",
            layout.activation_scale_row_bytes,
            expected.activation_scale_row_bytes,
        ),
        ("weight", layout.weight_row_bytes, expected.weight_row_bytes),
        (
            "weight scale",
            layout.weight_scale_row_bytes,
            expected.weight_scale_row_bytes,
        ),
    ] {
        if got != want {
            return Err(format!(
                "the {name} row stride is {got} bytes, but the pre-padded kernel walks {want}: a \
                 loader that shards whole experts (K = 2304, 18- or 72-byte scale rows) would \
                 misindex every row after the first"
            ));
        }
    }
    Ok(())
}

/// What one MoE combine costs, counted rather than estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MoeCost {
    /// Multiply-accumulates actually issued, padding included.
    pub macs: u64,
    /// The share of those MACs that read the zero padding.
    pub padding_macs: u64,
    /// Weight bytes read: the packed FP4 row, half a byte per value.
    pub weight_bytes: u64,
    /// Activation bytes read: the E4M3 row, a whole byte per value.
    pub activation_bytes: u64,
    /// ue8m0 scale bytes read, both operands' grids together.
    pub scale_bytes: u64,
}

/// The E2M1 value at one index of a packed weight row. The low nibble holds the even index.
///
/// # Panics
/// Panics if the row is too short.
#[must_use]
pub fn fp4_load(row: &[u8], index: usize) -> f32 {
    let byte = index / FP4_PER_BYTE;
    assert!(byte < row.len(), "the packed FP4 row is too short");
    let packed = row[byte];
    let nibble = if index.is_multiple_of(FP4_PER_BYTE) {
        packed & 0x0F
    } else {
        packed >> 4
    };
    fp4_e2m1_decode(nibble)
}

/// One routed slot's activation operand, quantised the way the reference quantises it.
///
/// The ordering is why this function exists, and why the kernel takes no routing weight:
/// model.py:841-851 computes the routed SwiGLU activation and then casts it to the activation dtype
/// (x.to(dtype), model.py:851); only after that does w2's quantiser read the tensor. So the routing
/// weight is inside the quantiser's input, and quantising an unrouted activation and scaling it
/// afterwards is a different arithmetic: the weight moves the per-32 amax and therefore the ue8m0
/// scale, and the E4M3 rounding then applies to a different value.
/// the_routing_weight_is_inside_the_activation_operand pins that difference, and the kernel's own
/// source is pinned against a routing term by src/m4_abi.rs.
///
/// The bf16 cast is part of the arithmetic rather than a storage detail: bf16 is the dtype the
/// quantiser reads, and it is applied here before the quantiser, exactly as model.py:851 does.
///
/// # Panics
/// Panics if the row is not a whole number of 32-value blocks.
#[must_use]
pub fn routed_activation_operand(activation: &[f32], routing: f32) -> (Vec<u8>, Vec<u8>) {
    let routed: Vec<f32> = activation
        .iter()
        .map(|value| crate::m4_quant::bf16_value(crate::m4_quant::bf16_bits(value * routing)))
        .collect();
    crate::m4_quant::act_quant_fp8_row(&routed)
}

/// One W4A8 MoE combine, exactly as b12x_mxfp4_moe_sm121.cu computes it, plus the cost count.
///
/// The hidden state is \[tokens\]\[hidden\] bf16 bits (the residual), the activations are
/// \[tokens * top_k\]\[padded\] E4M3 bytes with scales \[tokens * top_k\]\[padded / 32\] ue8m0, the
/// weights are \[experts\]\[hidden\]\[padded / 2\] packed E2M1 with scales
/// \[experts\]\[hidden\]\[padded / 32\] ue8m0, and the expert ids are \[tokens\]\[top_k\]. The
/// result is \[tokens\]\[hidden\] bf16 bits.
///
/// The strides are the pre-padded ones D14 fixes and `pre_padded_moe_layout` states. A loader that
/// produced the reference's whole-expert 72-wide or 576-wide per-rank 18-wide scale rows would
/// misindex every row after the first; `check_moe_layout` is the explicit rejection of that.
///
/// # Panics
/// Panics if the activation operand is sized like a packed E2M1 one rather than FP8 E4M3, if any
/// other buffer is too short for the declared shape, or if an expert index is out of range.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn moe_combine(
    hidden: &[u16],
    act_fp8: &[u8],
    act_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
    experts: usize,
) -> (Vec<u16>, MoeCost) {
    let routed_rows = tokens * MOE_TOP_K;
    assert!(
        act_fp8.len() >= routed_rows * MOE_ACT_BYTES_PER_ROW,
        "the activation operand is FP8 E4M3, one byte per value ({} bytes per routed row, D14), but \
         {} bytes were given for {} tokens x {} slots: that is the packed E2M1 operand this kernel \
         must not accept",
        MOE_ACT_BYTES_PER_ROW,
        act_fp8.len(),
        tokens,
        MOE_TOP_K
    );
    assert!(
        act_scales.len() >= routed_rows * MOE_SCALE_BYTES_PER_ROW,
        "the activation scale grid is {} ue8m0 bytes per routed row",
        MOE_SCALE_BYTES_PER_ROW
    );
    assert!(
        w.len() >= experts * hidden_dim * MOE_WEIGHT_BYTES_PER_ROW,
        "the weight buffer is too short for {experts} experts x {hidden_dim} outputs"
    );
    assert!(
        w_scales.len() >= experts * hidden_dim * MOE_SCALE_BYTES_PER_ROW,
        "the weight scale grid is {} ue8m0 bytes per output row",
        MOE_SCALE_BYTES_PER_ROW
    );
    let mut cost = MoeCost::default();
    let mut out = vec![0u16; tokens * hidden_dim];
    for token in 0..tokens {
        for column in 0..hidden_dim {
            let mut acc = 0.0f32;
            for slot in 0..MOE_TOP_K {
                let routed_row = token * MOE_TOP_K + slot;
                let expert = expert_ids[routed_row] as usize;
                assert!(expert < experts, "expert index {expert} is out of range");
                let act_row =
                    &act_fp8[routed_row * MOE_ACT_BYTES_PER_ROW..][..MOE_ACT_BYTES_PER_ROW];
                let act_scale_row =
                    &act_scales[routed_row * MOE_SCALE_BYTES_PER_ROW..][..MOE_SCALE_BYTES_PER_ROW];
                let w_row = &w[(expert * hidden_dim + column) * MOE_WEIGHT_BYTES_PER_ROW..]
                    [..MOE_WEIGHT_BYTES_PER_ROW];
                let w_scale_row = &w_scales
                    [(expert * hidden_dim + column) * MOE_SCALE_BYTES_PER_ROW..]
                    [..MOE_SCALE_BYTES_PER_ROW];
                for block in 0..MOE_SCALE_BYTES_PER_ROW {
                    let base = block * QUANT_BLOCK;
                    let act_scale = ue8m0_decode(act_scale_row[block]);
                    let w_scale = ue8m0_decode(w_scale_row[block]);
                    let mut partial = 0.0f32;
                    for i in 0..QUANT_BLOCK {
                        let index = base + i;
                        // The mask, exactly as the kernel applies it: the padded region of the
                        // ACTIVATION is zero whatever the caller left there, so the no-op claim does
                        // not depend on the caller zeroing anything. The MAC is still issued and
                        // still costed, because the kernel issues it.
                        let act_value = if index < MOE_WIDTH {
                            fp8_e4m3_decode(act_row[index])
                        } else {
                            0.0
                        };
                        // The FP4 weight enters through the reference's cast to FP8 through float
                        // (kernel.py:538-540). Writing the cast out is what keeps it a checked claim:
                        // the_fp4_to_fp8_cast_is_exact_on_every_e2m1_code enumerates all sixteen
                        // E2M1 codes and shows the round trip is the identity.
                        let w_value = fp8_e4m3_decode(fp8_e4m3_encode(fp4_load(w_row, index)));
                        partial += act_value * w_value;
                        cost.macs += 1;
                        if index >= MOE_WIDTH {
                            cost.padding_macs += 1;
                        }
                    }
                    // The two scales go on the BLOCK PARTIAL PRODUCT, as kernel.py:552-553 accumulates
                    // C_local * scale_a * scale_b. Because both scales are powers of two, scaling the
                    // partial product and scaling the operands are the same real arithmetic; what this
                    // ordering fixes is the accumulation order, one rounded add per 32-value block.
                    acc += (partial * act_scale) * w_scale;
                    // Counted per 32-value block, inside the loop that reads them, so the totals are
                    // the whole row the kernel walks: 320 weight bytes, 640 activation bytes and 40
                    // scale bytes per routed slot. (Before D14 these two counters were incremented
                    // once per slot with a single block's worth, so the earlier numbers were per-block
                    // units rather than the row's traffic.)
                    cost.weight_bytes += (QUANT_BLOCK / FP4_PER_BYTE) as u64;
                    cost.activation_bytes += QUANT_BLOCK as u64;
                    cost.scale_bytes += 2; // one activation ue8m0 byte and one weight ue8m0 byte
                }
            }
            let residual = crate::m4_quant::bf16_value(hidden[token * hidden_dim + column]);
            out[token * hidden_dim + column] = bf16_bits(residual + acc);
        }
    }
    (out, cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{
        E4M3_ELEMENT_EMAX, E4M3_MAX, act_quant_fp8_row, bf16_value, fp4_e2m1_encode,
        ue8m0_from_amax,
    };

    /// The code that decodes to 1.0 in E2M1: exponent field 1, mantissa 0.
    const ONE: u8 = 0x2;

    /// Packs E2M1 codes into bytes, two per byte, low nibble first.
    fn packed_row(codes: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; codes.len().div_ceil(FP4_PER_BYTE)];
        for (index, code) in codes.iter().enumerate() {
            let byte = index / FP4_PER_BYTE;
            if index % FP4_PER_BYTE == 0 {
                bytes[byte] |= code & 0x0F;
            } else {
                bytes[byte] |= (code & 0x0F) << 4;
            }
        }
        bytes
    }

    /// One activation operand row: the first count values are E4M3 value, the rest zero.
    fn e4m3_row(count: usize, value: f32) -> Vec<u8> {
        let mut row = vec![0u8; MOE_ACT_BYTES_PER_ROW];
        for byte in row.iter_mut().take(count) {
            *byte = fp8_e4m3_encode(value);
        }
        row
    }

    /// Repeat one activation row across every (token, slot) pair, with unit scales.
    fn repeated_operand(tokens: usize, count: usize, value: f32) -> (Vec<u8>, Vec<u8>) {
        let row = e4m3_row(count, value);
        let scales = vec![127u8; MOE_SCALE_BYTES_PER_ROW];
        let mut act = Vec::with_capacity(tokens * MOE_TOP_K * MOE_ACT_BYTES_PER_ROW);
        let mut act_scales = Vec::with_capacity(tokens * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW);
        for _ in 0..tokens * MOE_TOP_K {
            act.extend_from_slice(&row);
            act_scales.extend_from_slice(&scales);
        }
        (act, act_scales)
    }

    #[test]
    fn the_packing_is_low_nibble_first() {
        let codes = [ONE, 0x7u8]; // 1.0 then 6.0
        let row = packed_row(&codes);
        assert_eq!(row.len(), 1);
        assert_eq!(row[0], 0x72, "1.0 is the low nibble and 6.0 the high one");
        assert_eq!(fp4_load(&row, 0), 1.0);
        assert_eq!(fp4_load(&row, 1), 6.0);
    }

    #[test]
    fn golden_one_token_six_slots_is_hand_computable() {
        // One token, one output channel, and only the first 16 of the 576 real width values set to
        // E4M3 1.0 with unit ue8m0 scales on both sides. The padding is zero, so each slot
        // contributes 16 * 1.0 = 16.0 and six slots give 96.0. That is exactly representable in
        // bf16 - exponent 6, mantissa 0.5, so (6 + 127) << 7 = 0x4280 plus the mantissa half 0x40.
        let tokens = 1usize;
        let hidden_dim = 1usize;
        let experts = 2usize;

        let (act, act_scales) = repeated_operand(tokens, 16, 1.0);
        let mut codes = vec![0u8; MOE_WIDTH_PADDED];
        codes[..16].fill(ONE);
        let w = packed_row(&codes);
        let w_scales = vec![127u8; MOE_SCALE_BYTES_PER_ROW];
        let weights: Vec<u8> = (0..experts * hidden_dim).flat_map(|_| w.clone()).collect();
        let weight_scales: Vec<u8> = (0..experts * hidden_dim)
            .flat_map(|_| w_scales.clone())
            .collect();

        let (out, cost) = moe_combine(
            &[0u16],
            &act,
            &act_scales,
            &weights,
            &weight_scales,
            &[0, 0, 0, 0, 0, 0],
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(out[0], 0x42C0, "six slots of 16.0 is 96.0");
        assert_eq!(bf16_value(out[0]), 96.0);
        assert_eq!(
            cost.macs,
            (tokens * hidden_dim * MOE_TOP_K * MOE_WIDTH_PADDED) as u64
        );
        assert_eq!(
            cost.padding_macs,
            (tokens * hidden_dim * MOE_TOP_K * (MOE_WIDTH_PADDED - MOE_WIDTH)) as u64
        );
        // The D14 traffic, pinned rather than described: the activation now costs a whole byte per
        // value where the weight costs half of one, and the scales are counted separately.
        assert_eq!(
            cost.weight_bytes,
            (MOE_TOP_K * MOE_WEIGHT_BYTES_PER_ROW) as u64
        );
        assert_eq!(
            cost.activation_bytes,
            (MOE_TOP_K * MOE_ACT_BYTES_PER_ROW) as u64
        );
        assert_eq!(
            cost.scale_bytes,
            (2 * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW) as u64
        );
        assert_eq!(
            cost.activation_bytes,
            2 * cost.weight_bytes,
            "an E4M3 activation operand is twice the bytes of the packed FP4 weights it multiplies"
        );
    }

    #[test]
    fn the_padded_region_is_masked_so_caller_garbage_cannot_change_a_result() {
        // The kernel walks 640 values while the model has 576. Nonzero weights in the padding leave
        // the result alone AND nonzero activation bytes there leave it alone too, because both
        // implementations mask the region to zero. The padding is not a caller precondition.
        let tokens = 2usize;
        let hidden_dim = 2usize;
        let experts = 3usize;
        let rows = tokens * MOE_TOP_K;

        let mut act = vec![0u8; rows * MOE_ACT_BYTES_PER_ROW];
        for (row, slot) in act.chunks_mut(MOE_ACT_BYTES_PER_ROW).enumerate() {
            for (index, byte) in slot.iter_mut().enumerate() {
                if index < MOE_WIDTH {
                    *byte = fp8_e4m3_encode(((index + row) % 9) as f32);
                }
            }
        }
        let act_scales: Vec<u8> = (0..rows * MOE_SCALE_BYTES_PER_ROW)
            .map(|i| 120 + (i % 8) as u8)
            .collect();
        let weight_rows = experts * hidden_dim;
        // A uniform weight code, deliberately: E2M1's sixteen codes sum to exactly zero, so a
        // 16-periodic weight pattern would cancel over the 64-wide padding region and make this
        // test pass vacuously. That is not hypothetical - the first version of this fixture used
        // (i * 7) % 16 and both sides of the comparison came out zero.
        let weight_codes: Vec<u8> = vec![ONE; weight_rows * MOE_WIDTH_PADDED];
        let weight_scales: Vec<u8> = (0..weight_rows * MOE_SCALE_BYTES_PER_ROW)
            .map(|i| 118 + (i % 10) as u8)
            .collect();
        let w = packed_row(&weight_codes);
        let ids: Vec<u32> = (0..rows).map(|i| (i % experts) as u32).collect();

        let (baseline, cost) = moe_combine(
            &vec![0u16; tokens * hidden_dim],
            &act,
            &act_scales,
            &w,
            &weight_scales,
            &ids,
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(
            cost.padding_macs,
            (tokens * hidden_dim * MOE_TOP_K * (MOE_WIDTH_PADDED - MOE_WIDTH)) as u64,
            "the reference must count the padding MACs"
        );
        assert_ne!(
            baseline,
            vec![0u16; tokens * hidden_dim],
            "the fixture must produce a nonzero result, or the comparison below proves nothing"
        );

        // First, nonzero WEIGHTS in the padding are harmless: zeroing them must leave the result
        // identical, because the activation they multiply is zero either way.
        let mut zero_codes = weight_codes.clone();
        for row in 0..weight_rows {
            for index in MOE_WIDTH..MOE_WIDTH_PADDED {
                zero_codes[row * MOE_WIDTH_PADDED + index] = 0;
            }
        }
        let zero_w = packed_row(&zero_codes);
        let (zero_weight_padding, _) = moe_combine(
            &vec![0u16; tokens * hidden_dim],
            &act,
            &act_scales,
            &zero_w,
            &weight_scales,
            &ids,
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(
            baseline, zero_weight_padding,
            "nonzero weights in the padding must be multiplied by the zero activations"
        );

        // Second, and this is the point of A7: the padded region of the ACTIVATION is masked by the
        // kernel and by this reference, so garbage left there by a caller cannot move a result. The
        // fixture below really does carry nonzero bytes in that region.
        let mut padded = act.clone();
        for row in padded.chunks_mut(MOE_ACT_BYTES_PER_ROW) {
            for byte in row[MOE_WIDTH..].iter_mut() {
                *byte = fp8_e4m3_encode(6.0);
            }
        }
        assert!(
            padded
                .chunks(MOE_ACT_BYTES_PER_ROW)
                .all(|row| row[MOE_WIDTH..]
                    .iter()
                    .all(|byte| *byte == fp8_e4m3_encode(6.0))),
            "the garbage must actually be in the buffer the reference reads"
        );
        let (masked, masked_cost) = moe_combine(
            &vec![0u16; tokens * hidden_dim],
            &padded,
            &act_scales,
            &w,
            &weight_scales,
            &ids,
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(
            baseline, masked,
            "the padded region is masked, so nonzero bytes there cannot change a result"
        );
        assert_eq!(
            masked_cost.padding_macs, cost.padding_macs,
            "the masked MACs are still issued and still counted"
        );
        assert_eq!(masked_cost.macs, cost.macs);
        assert_eq!(masked_cost.activation_bytes, cost.activation_bytes);
        assert_eq!(masked_cost.weight_bytes, cost.weight_bytes);
        assert_eq!(masked_cost.scale_bytes, cost.scale_bytes);

        // The mask is load-bearing rather than decorative: the term it suppresses is NOT zero on
        // this fixture, so the equality above is not an accident of the data. Reproduce one
        // suppressed product by hand, unmasked.
        let block = MOE_WIDTH / QUANT_BLOCK;
        let suppressed = fp8_e4m3_decode(padded[MOE_WIDTH])
            * ue8m0_decode(act_scales[block])
            * fp4_load(&w[..MOE_WEIGHT_BYTES_PER_ROW], MOE_WIDTH)
            * ue8m0_decode(weight_scales[block]);
        assert!(
            suppressed.abs() > 0.0,
            "the suppressed term must be nonzero, or the mask would prove nothing"
        );
        // ...and a different garbage value must leave the result identical too, which is what
        // "masked" means: the content of the region is irrelevant, not merely zero on this run.
        let mut other = act.clone();
        for row in other.chunks_mut(MOE_ACT_BYTES_PER_ROW) {
            for byte in row[MOE_WIDTH..].iter_mut() {
                *byte = fp8_e4m3_encode(4.0);
            }
        }
        let (other_out, _) = moe_combine(
            &vec![0u16; tokens * hidden_dim],
            &other,
            &act_scales,
            &w,
            &weight_scales,
            &ids,
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(baseline, other_out);
        assert_eq!(baseline, masked);
    }

    #[test]
    fn the_reference_matches_a_naive_dequantized_loop() {
        let (tokens, hidden_dim, experts) = (2usize, 2usize, 2usize);
        let rows = tokens * MOE_TOP_K;
        // The activation fixture deliberately carries nonzero bytes in the padded region as well, so
        // the naive loop below has to get the mask right to agree: a fixture of zeros there would
        // let a missing mask pass.
        let mut act: Vec<u8> = (0..rows * MOE_ACT_BYTES_PER_ROW)
            .map(|i| fp8_e4m3_encode(((i % 9) as f32) - 4.0))
            .collect();
        for row in act.chunks_mut(MOE_ACT_BYTES_PER_ROW) {
            for byte in row[MOE_WIDTH..].iter_mut() {
                *byte = fp8_e4m3_encode(6.0);
            }
        }
        assert!(
            act.chunks(MOE_ACT_BYTES_PER_ROW)
                .all(|row| row[MOE_WIDTH..].iter().any(|byte| *byte != 0)),
            "the padded region must be nonzero for the mask to be exercised"
        );
        let act_scales: Vec<u8> = (0..rows * MOE_SCALE_BYTES_PER_ROW)
            .map(|i| 125 + (i % 5) as u8)
            .collect();
        let weight_codes: Vec<u8> = (0..experts * hidden_dim * MOE_WIDTH_PADDED)
            .map(|i| ((i * 3) % 13) as u8)
            .collect();
        let w = packed_row(&weight_codes);
        let weight_scales: Vec<u8> = (0..experts * hidden_dim * MOE_SCALE_BYTES_PER_ROW)
            .map(|i| 122 + (i % 6) as u8)
            .collect();
        let ids: Vec<u32> = (0..rows).map(|i| (i % experts) as u32).collect();

        let (reference, _) = moe_combine(
            &vec![0u16; tokens * hidden_dim],
            &act,
            &act_scales,
            &w,
            &weight_scales,
            &ids,
            tokens,
            hidden_dim,
            experts,
        );

        for token in 0..tokens {
            for column in 0..hidden_dim {
                let mut acc = 0.0f32;
                for slot in 0..MOE_TOP_K {
                    let routed_row = token * MOE_TOP_K + slot;
                    let expert = ids[routed_row] as usize;
                    let act_row =
                        &act[routed_row * MOE_ACT_BYTES_PER_ROW..][..MOE_ACT_BYTES_PER_ROW];
                    let act_scale_row = &act_scales[routed_row * MOE_SCALE_BYTES_PER_ROW..]
                        [..MOE_SCALE_BYTES_PER_ROW];
                    let w_row = &w[(expert * hidden_dim + column) * MOE_WEIGHT_BYTES_PER_ROW..]
                        [..MOE_WEIGHT_BYTES_PER_ROW];
                    let w_scale_row = &weight_scales
                        [(expert * hidden_dim + column) * MOE_SCALE_BYTES_PER_ROW..]
                        [..MOE_SCALE_BYTES_PER_ROW];
                    for block in 0..MOE_SCALE_BYTES_PER_ROW {
                        let base = block * QUANT_BLOCK;
                        let mut partial = 0.0f32;
                        for i in 0..QUANT_BLOCK {
                            let index = base + i;
                            // The same activation mask the kernel and the reference apply, spelled
                            // out here so this cross-check has no shared code with either.
                            let act_value = if index < MOE_WIDTH {
                                fp8_e4m3_decode(act_row[index])
                            } else {
                                0.0
                            };
                            partial += act_value
                                * fp8_e4m3_decode(fp8_e4m3_encode(fp4_load(w_row, index)));
                        }
                        acc += (partial * ue8m0_decode(act_scale_row[block]))
                            * ue8m0_decode(w_scale_row[block]);
                    }
                }
                assert_eq!(
                    reference[token * hidden_dim + column],
                    bf16_bits(acc),
                    "element ({token}, {column}) disagrees with the naive loop"
                );
            }
        }
    }

    #[test]
    fn the_residual_is_added_before_the_single_rounding() {
        // The kernel adds the residual in f32 and rounds once, so a zero expert has to leave the
        // residual's own bf16 value untouched rather than re-rounding it after a sum.
        let tokens = 1usize;
        let hidden_dim = 1usize;
        let experts = 1usize;
        let act = vec![0u8; tokens * MOE_TOP_K * MOE_ACT_BYTES_PER_ROW];
        let act_scales = vec![127u8; tokens * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW];
        // A whole padded weight row: the kernel always walks all 640 values.
        let mut codes = vec![0u8; MOE_WIDTH_PADDED];
        codes[0] = ONE;
        let w = packed_row(&codes);
        let w_scales = vec![127u8; MOE_SCALE_BYTES_PER_ROW];
        let residual = bf16_bits(1000.0);
        let (out, _) = moe_combine(
            &[residual],
            &act,
            &act_scales,
            &w,
            &w_scales,
            &[0u32; MOE_TOP_K],
            tokens,
            hidden_dim,
            experts,
        );
        assert_eq!(
            out[0],
            bf16_bits(bf16_value(residual)),
            "a zero expert must not move the residual"
        );
    }

    #[test]
    fn the_reference_quantiser_mirrors_act_quant() {
        // Rule 1, the amax floor (kernel.py:76): an all-zero block gets 2^-22, not a division by
        // zero and not 2^0.
        let (values, scales) = act_quant_fp8_row(&[0.0f32; QUANT_BLOCK]);
        assert_eq!(
            scales,
            vec![105u8],
            "1e-4 floored, then rounded up to 2^-22"
        );
        assert_eq!(ue8m0_decode(105), f32::from_bits(105u32 << 23));
        assert_eq!(values, vec![0u8; QUANT_BLOCK]);
        assert!(ue8m0_decode(scales[0]) > 0.0, "the scale must stay nonzero");

        // Rule 2, the round-up rule (kernel.py:36-37, 78). amax = 492.8 is not a power of two times
        // 448, so the scale is the NEXT power of two (2^1) and the block lands at 246.4, rounded to
        // the E4M3 grid at 240. The floor-of-log2 rule the weight grids use would have chosen 2^0 and
        // saturated instead - the assertion below spells that out with the weight-grid helper.
        let amax = 492.8f32;
        let (values, scales) = act_quant_fp8_row(&[amax; QUANT_BLOCK]);
        assert_eq!(scales, vec![128u8], "a scale of 2^1");
        assert_eq!(fp8_e4m3_decode(values[0]), 240.0);
        assert!(
            fp8_e4m3_decode(values[0]) <= E4M3_MAX,
            "the round-up rule keeps the block inside the format"
        );
        let weight_rule_scale = ue8m0_from_amax(amax, E4M3_ELEMENT_EMAX);
        assert_eq!(
            weight_rule_scale, 127,
            "the weight grid rule would pick 2^0 here"
        );
        assert_eq!(
            fp8_e4m3_decode(fp8_e4m3_encode(amax / ue8m0_decode(weight_rule_scale))),
            E4M3_MAX,
            "and that is exactly the saturation the activation rule avoids"
        );
        // The clamp is mirrored because the reference has it, but with this rule it cannot fire on a
        // block's own values: every block's amax fits under its own scale times 448.
        assert!(amax <= ue8m0_decode(scales[0]) * E4M3_MAX);

        // Rule 3, one maximum per block rather than a running maximum over the row (kernel.py:74):
        // the small first block keeps its own small scale while the large second block has its own.
        let mut row = vec![0.25f32; QUANT_BLOCK];
        row.extend(vec![amax; QUANT_BLOCK]);
        let (values, scales) = act_quant_fp8_row(&row);
        assert_eq!(scales, vec![117u8, 128u8]);
        assert_eq!(fp8_e4m3_decode(values[0]), 256.0, "0.25 / 2^-10 is exact");
        assert_eq!(fp8_e4m3_decode(values[QUANT_BLOCK]), 240.0);
        assert_eq!(values.len(), row.len());
        assert_eq!(scales.len(), row.len() / QUANT_BLOCK);

        // Rule 4, the product the ceiling is taken on: the reference's fp8_max_inv is a Python float
        // (kernel.py:45-46), and rounding it to f32 agrees with an f32 division, so the two spellings
        // of this module's E4M3_MAX reciprocal are not a source of drift.
        assert_eq!((1.0f64 / 448.0) as f32, 1.0f32 / crate::m4_quant::E4M3_MAX);
    }

    #[test]
    fn the_routing_weight_is_inside_the_activation_operand() {
        // model.py:849-851 multiplies the routing weight into the SwiGLU activation before the bf16
        // cast that the quantiser reads, so the operand is per (token, slot) and the weight moves the
        // per-32 scale. The fixture uses a routing weight that is NOT a power of two, because a
        // power-of-two weight would only shift the scales and the difference would be invisible.
        let row: Vec<f32> = (0..MOE_ACT_BYTES_PER_ROW)
            .map(|i| {
                if i < MOE_WIDTH {
                    0.5 + (i % 7) as f32 * 0.25
                } else {
                    0.0
                }
            })
            .collect();
        let routing = 0.3f32;
        let (folded, folded_scales) = routed_activation_operand(&row, routing);
        let (unfolded, unfolded_scales) = act_quant_fp8_row(&row);
        assert_eq!(folded.len(), MOE_ACT_BYTES_PER_ROW);
        assert_eq!(folded_scales.len(), MOE_SCALE_BYTES_PER_ROW);
        assert_ne!(
            folded_scales, unfolded_scales,
            "the routing weight changes the activation's ue8m0 scale, which is why it cannot be \
             applied after quantisation"
        );
        let folded_value = fp8_e4m3_decode(folded[0]) * ue8m0_decode(folded_scales[0]);
        let unfolded_value = fp8_e4m3_decode(unfolded[0]) * ue8m0_decode(unfolded_scales[0]);
        assert_ne!(
            folded_value,
            unfolded_value * routing,
            "quant(routing * x) is not routing * quant(x)"
        );
        assert!(folded_value > 0.0 && unfolded_value > 0.0);

        // ...and the operand this produces is the one the combine reads: one row per slot.
        let tokens = 1usize;
        let (act, act_scales) = repeated_operand(tokens, 16, 1.0);
        assert_eq!(act.len(), tokens * MOE_TOP_K * MOE_ACT_BYTES_PER_ROW);
        let (single, single_scales) = routed_activation_operand(&row, routing);
        let mut operand = Vec::new();
        let mut operand_scales = Vec::new();
        for _ in 0..MOE_TOP_K {
            operand.extend_from_slice(&single);
            operand_scales.extend_from_slice(&single_scales);
        }
        let (out, _) = moe_combine(
            &[0u16],
            &operand,
            &operand_scales,
            &packed_row(&[ONE; MOE_WIDTH_PADDED]),
            &[127u8; MOE_SCALE_BYTES_PER_ROW],
            &[0u32; MOE_TOP_K],
            1,
            1,
            1,
        );
        assert!(
            bf16_value(out[0]) > 0.0,
            "the routed operand must produce a positive combination"
        );
        assert_eq!(
            act_scales.len(),
            tokens * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW
        );
    }

    /// The sixteen E2M1 codes as literal f32 values, one per nibble, written from the format rather
    /// than read through `fp4_e2m1_decode`. A wrong decoder cannot cancel out of a test that
    /// compares it against this table.
    const E2M1_VALUES: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    #[test]
    fn the_e2m1_decoder_matches_the_hand_written_code_table() {
        // The decoder is pinned against the format table rather than against itself: mutating it to,
        // say, power * (1 + mantissa * 0.75) fails here even if every other test agrees with the
        // same wrong decoder.
        for (nibble, expected) in E2M1_VALUES.iter().enumerate() {
            assert_eq!(
                fp4_e2m1_decode(nibble as u8),
                *expected,
                "E2M1 code {nibble:#x} decodes to {expected}"
            );
        }
    }

    #[test]
    fn the_fp4_to_fp8_cast_is_exact_on_every_e2m1_code() {
        // The reference's fp4 GEMM casts B from FP4 to FP8 through float (kernel.py:538-540). That
        // cast is only harmless because every E2M1 value is exactly representable in E4M3, so this
        // enumerates all sixteen codes from the HAND-WRITTEN table, not from the decoder: a wrong
        // decoder would otherwise cancel on both sides of the equality.
        for (nibble, value) in E2M1_VALUES.iter().enumerate() {
            assert_eq!(
                fp8_e4m3_decode(fp8_e4m3_encode(*value)),
                *value,
                "E2M1 code {nibble:#x} ({value}) must survive the cast to E4M3 and back"
            );
        }
    }

    #[test]
    fn the_operand_format_is_not_interchangeable_with_e2m1() {
        assert_eq!(MOE_ACT_BYTES_PER_VALUE, 1);
        assert_eq!(MOE_ACT_BYTES_PER_ROW, 2 * MOE_WEIGHT_BYTES_PER_ROW);
        assert_eq!(MOE_ACT_BYTES_PER_TOKEN, MOE_TOP_K * MOE_ACT_BYTES_PER_ROW);
        assert_eq!(MOE_SCALE_BYTES_PER_ROW, 20);
        // An E2M1 operand cannot hold this value at all: E2M1's largest magnitude is 6, the
        // reference's activation is E4M3 with a maximum of 448, and the quantiser emits values above
        // 6 for ordinary activations. That is the substance of D14, not a naming preference.
        let (values, scales) = act_quant_fp8_row(&[30.0f32; QUANT_BLOCK]);
        let dequantised = fp8_e4m3_decode(values[0]) * ue8m0_decode(scales[0]);
        assert_eq!(dequantised, 30.0, "the E4M3 operand carries 30.0");
        assert_eq!(
            fp4_e2m1_decode(fp4_e2m1_encode(30.0)),
            6.0,
            "the same value in E2M1 saturates to 6"
        );
    }

    #[test]
    fn a_loader_layout_that_is_not_pre_padded_is_rejected() {
        // A8: the 640-wide kernel reads 20-byte scale rows. The reference's whole-expert grid
        // ([out, 72]) and its 576-wide per-rank slice ([out, 18]) both fit in a flat test buffer of
        // the same activation length, so only an explicit stride check rejects them instead of
        // silently misindexing the rows after the first.
        let expected = pre_padded_moe_layout();
        assert_eq!(expected.activation_row_bytes, 640);
        assert_eq!(expected.activation_scale_row_bytes, 20);
        assert_eq!(expected.weight_row_bytes, 320);
        assert_eq!(expected.weight_scale_row_bytes, 20);
        assert_eq!(check_moe_layout(expected), Ok(()));
        for (name, layout) in [
            (
                "the reference's whole-expert 72-wide scale rows",
                MoeRowLayout {
                    activation_scale_row_bytes: 72,
                    ..expected
                },
            ),
            (
                "the reference's 576-wide per-rank 18-wide scale rows",
                MoeRowLayout {
                    activation_scale_row_bytes: 18,
                    ..expected
                },
            ),
            (
                "an unpadded 576-wide activation",
                MoeRowLayout {
                    activation_row_bytes: 576,
                    ..expected
                },
            ),
        ] {
            let error =
                check_moe_layout(layout).expect_err("a layout that is not pre-padded is rejected");
            assert!(
                error.contains("row stride")
                    && (error.contains("scale") || error.contains("activation")),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn a_stride_check_cannot_see_the_scale_grid() {
        // A5 limit, pinned rather than implied. The kernel reads the FP4 per-output-row grid
        // spark_b12x_row_scale_index(n, 0, 640) = n * 20 (cuh:524); the FP8 blocked grid
        // spark_b12x_scale_index(n, 0, 640) = (n / 32) * 20 (cuh:518) is a different layout with
        // the same 20-byte row stride, so a loader that stored the blocked grid while reporting
        // weight_scale_row_bytes = 20 passes check_moe_layout. The two indices are computed here so
        // the collision is a pinned fact rather than a comment.
        let per_row_grid = |n: usize| n * MOE_SCALE_BYTES_PER_ROW;
        let blocked_grid = |n: usize| (n / 32) * MOE_SCALE_BYTES_PER_ROW;
        assert_eq!(per_row_grid(0), blocked_grid(0));
        assert_eq!(
            blocked_grid(31),
            0,
            "the blocked grid gives all 32 output rows of a tile the same scale row"
        );
        assert_eq!(
            per_row_grid(31),
            620,
            "the per-row grid advances every output row"
        );
        assert_eq!(blocked_grid(32), MOE_SCALE_BYTES_PER_ROW);
        assert_eq!(per_row_grid(32), 32 * MOE_SCALE_BYTES_PER_ROW);
        // The stride the loader reports is the same 20 for both, which is exactly why the layout
        // check cannot see the difference and the row ordering stays the loader's responsibility.
        assert_eq!(pre_padded_moe_layout().weight_scale_row_bytes, 20);
        assert_eq!(check_moe_layout(pre_padded_moe_layout()), Ok(()));
    }

    #[test]
    #[should_panic(expected = "the activation operand is FP8 E4M3")]
    fn the_e2m1_activation_operand_is_rejected_loudly() {
        // The operand this kernel read before D14: half a byte per value. It must fail at the
        // boundary rather than be reinterpreted as FP8, which would silently produce garbage.
        let tokens = 1usize;
        let act = vec![0u8; tokens * MOE_TOP_K * MOE_WEIGHT_BYTES_PER_ROW];
        let act_scales = vec![127u8; tokens * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW];
        let _ = moe_combine(
            &[0u16],
            &act,
            &act_scales,
            &[0u8; MOE_WEIGHT_BYTES_PER_ROW],
            &[127u8; MOE_SCALE_BYTES_PER_ROW],
            &[0u32; MOE_TOP_K],
            tokens,
            1,
            1,
        );
    }
}
