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
//! 2. the arithmetic, which is the released model's, not a per-rank chain: one expert's output is
//!    the f32 sum of the four tensor-parallel ranks' FP8xFP8 block sums (each rank owns a 576-value
//!    slice of the 2304-value intermediate width, padded to 640), computed with both ue8m0 scales
//!    applied to each 32-value block's partial product the way kernel.py:552-553 applies them to the
//!    accumulator. [moe_combine_rank] adds a rank's partials into a f32 per-(token, slot)
//!    accumulator and rounds nothing, because rounding a rank partial is a different arithmetic:
//!    the released model rounds each expert's w2 output to bf16 once, then adds the experts in
//!    ascending expert-id order into a f32 row, adds the shared expert's bf16 output, and rounds the
//!    MoE output to bf16 once (model.py:893-904). [moe_combine_finalize] implements that epilogue and
//!    [moe_combine] is the one-rank wrapper. The residual is NOT part of this arithmetic: the
//!    released model adds it later, in f32, inside hc_post (model.py:962-966). The FP4 weight is read
//!    through the reference's cast to FP8 (kernel.py:538-540), which is exact on all sixteen E2M1
//!    codes (enumerated by a test here rather than assumed), and no routing weight is applied: it
//!    belongs to the activation operand, see routed_activation_operand.
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

/// The stored per-rank code row: the released plan's real 576-value slice, two FP4 per byte.
pub const MOE_STORED_CODE_ROW: usize = MOE_WIDTH / FP4_PER_BYTE;
/// The stored per-rank scale row: one ue8m0 byte per 32 real values; a SEPARATE tensor from the
/// code rows, not a tail of a wider row.
pub const MOE_STORED_SCALE_ROW: usize = MOE_WIDTH / QUANT_BLOCK;
/// The MX blocks the stored kernel walks: 576 / 32, with no padding block.
pub const MOE_STORED_K_BLOCKS: usize = MOE_STORED_SCALE_ROW;

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

/// One TP rank's W4A8 down-projection partial, added into the native per-slot accumulator.
///
/// This is the kernel half of the native MoE arithmetic: the W4A8 GEMM is split across the four
/// tensor-parallel ranks over the intermediate width (each rank owns a 576-value slice, padded to
/// 640), so one expert's output is the f32 sum of all ranks' partials. The reference sums those
/// partials in f32 (model.py:900 has y[idx] += expert(...), where the rank all-reduce is f32 at
/// model.py:902) and rounds each expert's output to bf16 exactly once; there is no bf16 rounding
/// inside a GEMM. This function therefore performs NO rounding: it adds this rank's block-scaled
/// partials into slot_acc[token * MOE_TOP_K + slot] in f32, and [moe_combine_finalize] /
/// [moe_combine_routed_f32] round each slot once, after every rank has contributed.
///
/// slot_acc must be zero-initialised by the caller and has tokens * MOE_TOP_K entries. The
/// activation, weight, id and geometry inputs are the ones [moe_combine] documents.
///
/// # Panics
/// Panics if slot_acc is the wrong length, if the activation operand is sized like a packed E2M1
/// one rather than FP8 E4M3, if any other buffer is too short for the declared shape, or if an
/// expert index is out of range.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_rank(
    slot_acc: &mut [f32],
    act_fp8: &[u8],
    act_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
    experts: usize,
) -> MoeCost {
    let routed_rows = tokens * MOE_TOP_K;
    assert_eq!(
        slot_acc.len(),
        routed_rows * hidden_dim,
        "the slot accumulator holds one f32 per (routed row, hidden index)"
    );
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
    for token in 0..tokens {
        for column in 0..hidden_dim {
            let slot_base = token * MOE_TOP_K;
            for slot in 0..MOE_TOP_K {
                let routed_row = slot_base + slot;
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
                let mut acc = 0.0f32;
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
                    // scale bytes per routed slot.
                    cost.weight_bytes += (QUANT_BLOCK / FP4_PER_BYTE) as u64;
                    cost.activation_bytes += QUANT_BLOCK as u64;
                    cost.scale_bytes += 2; // one activation ue8m0 byte and one weight ue8m0 byte
                }
                slot_acc[routed_row * hidden_dim + column] += acc;
            }
        }
    }
    cost
}

/// The stored row layout b12x_mxfp4_moe_stored_sm121.cu reads: the released per-rank plan's real
/// 576-wide rows, with the activation operand left in the A8 chain layout.
///
/// The two weight tensors are separate: w2.weight is [experts][d][288] packed FP4 codes and
/// w2.scale is [experts][d][18] ue8m0 bytes. A row therefore costs 288 + 18 = 306 bytes across
/// the two tensors, but neither tensor has a 306-byte stride - the scale grid advances 18 bytes per
/// output row. The activation side is deliberately the A8 chain layout the incumbent kernel reads
/// (640 E4M3 bytes and a 20-byte scale grid per routed row): the chain quantises the activation
/// before the weight layout is known, so a switch to the stored kernel must not re-stage it, and the
/// stored kernel reads only the first 18 of the 20 scale blocks and only columns 0..575.
#[must_use]
pub const fn stored_moe_layout() -> MoeRowLayout {
    MoeRowLayout {
        activation_row_bytes: MOE_ACT_BYTES_PER_ROW,
        activation_scale_row_bytes: MOE_SCALE_BYTES_PER_ROW,
        weight_row_bytes: MOE_STORED_CODE_ROW,
        weight_scale_row_bytes: MOE_STORED_SCALE_ROW,
    }
}

/// # Errors
/// Returns a description of the first stride that does not match the stored layout, which is what
/// turns a silent row misindex into a loud rejection. In particular a caller that emits interleaved
/// 306-byte rows (the cost of one output row, not a stride) is rejected here: the stored kernel
/// reads the code rows and the scale rows from two different tensors.
pub fn check_stored_moe_layout(layout: MoeRowLayout) -> Result<(), String> {
    let expected = stored_moe_layout();
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
        (
            "stored weight code",
            layout.weight_row_bytes,
            expected.weight_row_bytes,
        ),
        (
            "stored weight scale",
            layout.weight_scale_row_bytes,
            expected.weight_scale_row_bytes,
        ),
    ] {
        if got != want {
            return Err(format!(
                "the {name} row stride is {got} bytes, but b12x_mxfp4_moe_stored_sm121 walks                  {want}: an interleaved 306-byte row or a padded 320/20 row would misindex every                  row after the first"
            ));
        }
    }
    Ok(())
}

/// One TP rank's W4A8 down-projection partial on the RELEASED STORED layout: the host twin of
/// b12x_mxfp4_moe_stored_sm121.cu.
///
/// The arithmetic is [moe_combine_rank]'s with the stored row contract: w is the rank's contiguous
/// [experts][d][288] code tensor, w_scales its separate [experts][d][18] ue8m0 tensor, and the loop
/// walks the 18 real MX blocks instead of the 20 padded ones. The activation operand is the SAME A8
/// operand [moe_combine_rank] reads (padded 640/20 rows, columns >= 576 masked by the incumbent);
/// this module handles no padding on the weight side, so the stored kernel needs no mask at all.
///
/// The two kernels are NOT bit-comparable against each other on one buffer: they read different K
/// widths. Exactness is stated against this twin on the same stored bytes, and the de-interleave
/// test proves the stride handling by rebuilding the padded rows and requiring both twins to agree
/// bit-for-bit. The f32 accumulator contract is unchanged: no rounding here, one f32 add per
/// 32-value block in ascending block order, and the caller finalises with [moe_combine_finalize].
///
/// # Panics
/// Panics if the accumulator length, an activation operand length or an expert index is wrong, or if
/// a weight buffer is too short for the declared stored shape.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_rank_stored(
    slot_acc: &mut [f32],
    act_fp8: &[u8],
    act_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
    experts: usize,
) -> MoeCost {
    let routed_rows = tokens * MOE_TOP_K;
    assert_eq!(
        slot_acc.len(),
        routed_rows * hidden_dim,
        "the slot accumulator holds one f32 per (routed row, hidden index)"
    );
    assert!(
        act_fp8.len() >= routed_rows * MOE_ACT_BYTES_PER_ROW,
        "the activation operand is the A8 chain layout: {} bytes per routed row",
        MOE_ACT_BYTES_PER_ROW
    );
    assert!(
        act_scales.len() >= routed_rows * MOE_SCALE_BYTES_PER_ROW,
        "the activation scale grid is {} ue8m0 bytes per routed row",
        MOE_SCALE_BYTES_PER_ROW
    );
    assert!(
        w.len() >= experts * hidden_dim * MOE_STORED_CODE_ROW,
        "the stored weight buffer is too short: {experts} experts x {hidden_dim} outputs x          {MOE_STORED_CODE_ROW} code bytes"
    );
    assert!(
        w_scales.len() >= experts * hidden_dim * MOE_STORED_SCALE_ROW,
        "the stored scale buffer is too short: {experts} experts x {hidden_dim} outputs x          {MOE_STORED_SCALE_ROW} scale bytes"
    );
    let mut cost = MoeCost::default();
    for token in 0..tokens {
        for column in 0..hidden_dim {
            let slot_base = token * MOE_TOP_K;
            for slot in 0..MOE_TOP_K {
                let routed_row = slot_base + slot;
                let expert = expert_ids[routed_row] as usize;
                assert!(expert < experts, "expert index {expert} is out of range");
                // The A operand is the chain layout; only the first MOE_WIDTH columns are real and
                // only the first MOE_STORED_K_BLOCKS of its 20-byte scale grid are read.
                let act_row =
                    &act_fp8[routed_row * MOE_ACT_BYTES_PER_ROW..][..MOE_ACT_BYTES_PER_ROW];
                let act_scale_row =
                    &act_scales[routed_row * MOE_SCALE_BYTES_PER_ROW..][..MOE_SCALE_BYTES_PER_ROW];
                let weight_row = expert * hidden_dim + column;
                let w_row = &w[weight_row * MOE_STORED_CODE_ROW..][..MOE_STORED_CODE_ROW];
                let w_scale_row =
                    &w_scales[weight_row * MOE_STORED_SCALE_ROW..][..MOE_STORED_SCALE_ROW];
                let mut acc = 0.0f32;
                for block in 0..MOE_STORED_K_BLOCKS {
                    let base = block * QUANT_BLOCK;
                    let act_scale = ue8m0_decode(act_scale_row[block]);
                    let w_scale = ue8m0_decode(w_scale_row[block]);
                    let mut partial = 0.0f32;
                    for i in 0..QUANT_BLOCK {
                        let index = base + i;
                        // No mask: base + i is a real K value here. The A operand still has the
                        // padded columns, but this loop never addresses them.
                        let act_value = fp8_e4m3_decode(act_row[index]);
                        let w_value = fp8_e4m3_decode(fp8_e4m3_encode(fp4_load(w_row, index)));
                        partial += act_value * w_value;
                        cost.macs += 1;
                    }
                    acc += (partial * act_scale) * w_scale;
                    cost.weight_bytes += (QUANT_BLOCK / FP4_PER_BYTE) as u64;
                    cost.activation_bytes += QUANT_BLOCK as u64;
                    cost.scale_bytes += 2;
                }
                slot_acc[routed_row * hidden_dim + column] += acc;
            }
        }
    }
    cost
}

/// The slots of one token in ascending expert-id order, as native adds them.
///
/// model.py:899-903 iterates the routed experts in ascending expert-id order (for i in range(...):
/// torch.where(indices == i)), and y[idx] += expert(...) therefore adds the bf16 expert outputs in
/// that order. FP32 addition is not associative, so the order is part of the arithmetic; the top-k
/// slot order the kernel produced them in is not the native order.
#[must_use]
pub fn moe_combine_slot_order(expert_ids: &[u32], token: usize) -> [usize; MOE_TOP_K] {
    let mut order = [0usize; MOE_TOP_K];
    let mut used = [false; MOE_TOP_K];
    // Selection sort by expert id: six entries, so the O(k^2) shape costs nothing, and taking the
    // first minimum in ascending slot order is stable on equal ids (the top-k cannot repeat an
    // expert, but the reference's stability is part of the order rule regardless).
    for (position, entry) in order.iter_mut().enumerate() {
        let mut best = usize::MAX;
        for slot in 0..MOE_TOP_K {
            if used[slot] {
                continue;
            }
            if best == usize::MAX
                || expert_ids[token * MOE_TOP_K + slot] < expert_ids[token * MOE_TOP_K + best]
            {
                best = slot;
            }
        }
        *entry = best;
        used[best] = true;
        let _ = position;
    }
    order
}

/// The routed part of the native combine epilogue, as f32:
/// sum over ascending expert id of bf16(w2 expert output).
///
/// Each slot's f32 rank sum is rounded to bf16 once (the expert's w2 output dtype,
/// model.py:851-852 returns self.w2(x.to(dtype))), and the bf16 values are added in ascending
/// expert-id order in f32. Native keeps the result in f32 and adds the shared expert before the
/// single MoE bf16 round (model.py:903-904), so this function deliberately does not round the total:
/// the chain adds its shared expert and rounds at that boundary. [moe_combine_finalize] is the
/// whole-token version with the shared add and the round included.
///
/// # Panics
/// Panics if the accumulator or the id list is the wrong length.
#[must_use]
pub fn moe_combine_routed_f32(
    slot_acc: &[f32],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(
        slot_acc.len(),
        tokens * MOE_TOP_K * hidden_dim,
        "the slot accumulator holds one f32 per (routed row, hidden index)"
    );
    let mut routed = vec![0.0f32; tokens * hidden_dim];
    for token in 0..tokens {
        let order = moe_combine_slot_order(expert_ids, token);
        for column in 0..hidden_dim {
            let mut total = 0.0f32;
            for slot in order {
                let expert_output = slot_acc[(token * MOE_TOP_K + slot) * hidden_dim + column];
                let rounded = crate::m4_quant::bf16_value(bf16_bits(expert_output));
                total += rounded;
            }
            routed[token * hidden_dim + column] = total;
        }
    }
    routed
}

/// The cross-rank f32 sum of every rank's slot accumulator: the operand of the native combine in
/// a tensor-parallel MoE.
///
/// Native sums the ranks in f32 before any bf16 rounding: the release's `y` is `float32`, every
/// expert output is added into it, and `dist.all_reduce(y)` runs on that f32 tensor
/// (model.py:893-900); the single MoE rounding is after the shared expert add (model.py:903-904).
/// A rank-partial bf16 round, or a bf16 all-reduce payload, is not that arithmetic. The ranks are
/// added in ascending rank order, the canonical order
/// `sparkflash_comm::reduce::canonical_all_reduce_into` uses; NCCL's own f32 reduce order is not
/// part of the release's source, so the order is declared rather than derived from it.
///
/// # Panics
/// Panics if `rank_slot_accs` is empty or if a rank's length is not
/// `tokens * MOE_TOP_K * hidden_dim`.
#[must_use]
pub fn moe_combine_rank_sum(
    rank_slot_accs: &[&[f32]],
    tokens: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert!(
        !rank_slot_accs.is_empty(),
        "the native combine needs at least one rank"
    );
    let expected = tokens * MOE_TOP_K * hidden_dim;
    for (rank, acc) in rank_slot_accs.iter().enumerate() {
        assert_eq!(
            acc.len(),
            expected,
            "rank {rank}'s slot accumulator is {} values, not {tokens} x {MOE_TOP_K} x {hidden_dim}",
            acc.len()
        );
    }
    let mut sum = vec![0.0f32; expected];
    for acc in rank_slot_accs {
        for (total, value) in sum.iter_mut().zip(acc.iter()) {
            *total += *value;
        }
    }
    sum
}

/// One complete native MoE output for a tensor-parallel rank set: the cross-rank f32 slot sum, the
/// per-expert bf16 rounds in ascending expert-id order, the shared expert's bf16 output added, and
/// the one MoE bf16 round (model.py:893-904, with the residual left to `hc_post`).
///
/// This is the combine the four-rank loop's native arm runs: [moe_combine_rank_sum] feeds
/// [moe_combine_finalize], and the result is the bf16 row `hc_post` then mixes into the f32
/// residual stream. The loop's current chain instead finalises each rank's partial and bf16-rounds
/// the payload before the canonical bf16 all-reduce, and adds the shared expert in f32 afterwards.
///
/// # Panics
/// Panics when [moe_combine_rank_sum] or [moe_combine_finalize] would.
#[must_use]
pub fn moe_combine_native(
    rank_slot_accs: &[&[f32]],
    shared: &[u16],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    let full = moe_combine_rank_sum(rank_slot_accs, tokens, hidden_dim);
    moe_combine_finalize(&full, shared, expert_ids, tokens, hidden_dim)
}

/// The full native MoE output for one token block: the routed f32 sum, the shared expert's bf16
/// output added, and one rounding to bf16.
///
/// This is model.py:893-904 with the residual left out: y is f32, each routed expert's w2 output is
/// bf16 before the add, the shared expert's bf16 output is added after the routed sum, and the MoE
/// output is rounded to bf16 once. The residual is added later by hc_post in f32
/// (model.py:962-966), which is why it does not appear here. shared is the shared expert's output
/// already rounded to bf16 bits (its own w2 GEMM output dtype); it is widened back for the add.
///
/// # Panics
/// Panics if the accumulator, the shared row or the id list is the wrong length.
#[must_use]
pub fn moe_combine_finalize(
    slot_acc: &[f32],
    shared: &[u16],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
) -> Vec<u16> {
    assert_eq!(
        shared.len(),
        tokens * hidden_dim,
        "the shared expert output is one bf16 value per token and hidden index"
    );
    let routed = moe_combine_routed_f32(slot_acc, expert_ids, tokens, hidden_dim);
    let mut out = vec![0u16; tokens * hidden_dim];
    for (index, &value) in routed.iter().enumerate() {
        let shared_value = crate::m4_quant::bf16_value(shared[index]);
        out[index] = bf16_bits(value + shared_value);
    }
    out
}

/// One complete W4A8 MoE combine for a single TP rank's operand data: accumulate the rank's
/// partials, finalize the routed sum and the shared add. The multi-rank chain calls
/// [moe_combine_rank] once per rank and [moe_combine_finalize] (or [moe_combine_routed_f32]) once,
/// because the rank partials must stay f32 until every rank has contributed.
///
/// The operands are the ones [moe_combine_rank] documents; shared replaces the old pre-combine
/// hidden row and is the shared expert's bf16 output (model.py:903).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn moe_combine(
    shared: &[u16],
    act_fp8: &[u8],
    act_scales: &[u8],
    w: &[u8],
    w_scales: &[u8],
    expert_ids: &[u32],
    tokens: usize,
    hidden_dim: usize,
    experts: usize,
) -> (Vec<u16>, MoeCost) {
    let mut slot_acc = vec![0.0f32; tokens * MOE_TOP_K * hidden_dim];
    let cost = moe_combine_rank(
        &mut slot_acc,
        act_fp8,
        act_scales,
        w,
        w_scales,
        expert_ids,
        tokens,
        hidden_dim,
        experts,
    );
    (
        moe_combine_finalize(&slot_acc, shared, expert_ids, tokens, hidden_dim),
        cost,
    )
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
        // Deliberately NOT ascending in slot order: with this fixture the slot order alternates, so
        // a naive loop that adds in slot order is not the same order as native's ascending
        // expert-id order, and the reference has to sort.
        let ids: Vec<u32> = (0..rows).map(|i| ((i + 1) % experts) as u32).collect();

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
            let mut order: Vec<usize> = (0..MOE_TOP_K).collect();
            order.sort_by_key(|slot| ids[token * MOE_TOP_K + slot]);
            for column in 0..hidden_dim {
                let mut total = 0.0f32;
                for slot in order.iter().copied() {
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
                    let mut acc = 0.0f32;
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
                    // Native rounds the expert's w2 output to bf16 once and adds the bf16 values in
                    // ascending expert-id order into f32 (model.py:899-903). The naive loop plays
                    // all four tensor-parallel ranks at once, so acc is already the full expert
                    // output.
                    total += bf16_value(bf16_bits(acc));
                }
                assert_eq!(
                    reference[token * hidden_dim + column],
                    bf16_bits(total),
                    "element ({token}, {column}) disagrees with the naive loop"
                );
            }
        }
    }

    #[test]
    fn the_shared_expert_is_added_after_the_per_expert_rounding() {
        // Native rounds each expert's f32 w2 output to bf16 before adding it to y, then adds the
        // shared expert's bf16 output and rounds the MoE output once (model.py:899-904). This
        // fixture separates that from the old rule (add the shared value to the unrounded routed
        // sum and round once): the bf16 codes differ.
        let slot_acc = [1.0f32 + 2.0f32.powi(-8), 0.0, 0.0, 0.0, 0.0, 0.0];
        let shared = [bf16_bits(2.0f32.powi(-8))];
        let out = moe_combine_finalize(&slot_acc, &shared, &[0u32; MOE_TOP_K], 1, 1);
        // bf16(1 + 2^-8) is a tie that rounds to 1.0; adding 2^-8 in f32 and rounding gives 1.0.
        assert_eq!(out[0], bf16_bits(1.0));
        // The old rule (shared added to the unrounded routed sum, one rounding) produces 1 + 2^-7.
        assert_ne!(
            out[0],
            bf16_bits(1.0 + 2.0f32.powi(-7)),
            "adding the shared expert before the per-expert rounding is a different arithmetic"
        );
        // A shared expert alone is not re-rounded: it is already bf16 and the single output
        // rounding cannot move it.
        let shared_only = [bf16_bits(1000.0f32)];
        let out =
            moe_combine_finalize(&[0.0f32; MOE_TOP_K], &shared_only, &[0u32; MOE_TOP_K], 1, 1);
        assert_eq!(out[0], bf16_bits(1000.0));
    }

    #[test]
    fn the_experts_are_added_in_ascending_expert_id_order() {
        // FP32 addition is not associative, and native adds the bf16 expert outputs in ascending
        // expert-id order (model.py:899-903). The large contribution sits in the slot whose expert
        // id is not the smallest, so slot order and ascending expert-id order differ: adding the
        // two unit experts first is exact in f32, adding the unit expert to 2^24 first is not.
        let ids = [3u32, 0, 1, 4, 5, 2];
        let slot_acc = [16777216.0f32, 1.0, 1.0, 0.0, 0.0, 0.0];
        let routed = moe_combine_routed_f32(&slot_acc, &ids, 1, 1);
        assert_eq!(
            routed[0], 16777218.0f32,
            "the small experts are added first, in ascending expert-id order"
        );
        assert_eq!(
            (slot_acc[0] + slot_acc[1]) + slot_acc[2],
            16777216.0f32,
            "adding in slot order really is a different f32 result on this fixture"
        );
        assert_eq!(
            moe_combine_slot_order(&ids, 0),
            [1usize, 2, 5, 0, 3, 4],
            "the order is by expert id, not by slot"
        );
        // Equal ids keep their slot order, so the sort is stable rather than arbitrary on a tie.
        let repeated = [2u32, 2, 0, 3, 1, 0];
        assert_eq!(
            moe_combine_slot_order(&repeated, 0),
            [2usize, 5, 4, 0, 1, 3]
        );
    }

    #[test]
    fn rank_partials_stay_f32_until_the_finalize() {
        // The four tensor-parallel rank launches add into the same f32 accumulator and NOTHING
        // rounds between them: an expert's output is the f32 sum of the four rank partials, and the
        // one bf16 rounding is in the finalize. One rank's partial here is 16 + 2^-16, which is not
        // a bf16 value (bf16(16 + 2^-16) is 16), so a per-rank rounding would change the
        // accumulator and this test would see it.
        let tokens = 1usize;
        let hidden_dim = 1usize;
        let experts = 1usize;
        let mut act_row = vec![0u8; MOE_ACT_BYTES_PER_ROW];
        for byte in act_row.iter_mut().take(16) {
            *byte = fp8_e4m3_encode(1.0);
        }
        // The one 2^-16 product: E4M3 2^-8 times the block's ue8m0 2^-8 scale, against a 1.0
        // weight with a unit weight scale.
        act_row[32] = fp8_e4m3_encode(2.0f32.powi(-8));
        let mut act_scale_row = vec![127u8; MOE_SCALE_BYTES_PER_ROW];
        act_scale_row[1] = 119; // 2^-8
        let act: Vec<u8> = std::iter::repeat_n(act_row, MOE_TOP_K).flatten().collect();
        let act_scales: Vec<u8> = std::iter::repeat_n(act_scale_row, MOE_TOP_K)
            .flatten()
            .collect();
        let mut w_codes = vec![0u8; MOE_WIDTH_PADDED];
        for code in w_codes.iter_mut().take(16) {
            *code = ONE;
        }
        w_codes[32] = ONE;
        let w = packed_row(&w_codes);
        let w_scales = vec![127u8; MOE_SCALE_BYTES_PER_ROW];
        let ids = [0u32; MOE_TOP_K];
        let mut slot_acc = vec![0.0f32; tokens * MOE_TOP_K * hidden_dim];
        for _ in 0..4 {
            let cost = moe_combine_rank(
                &mut slot_acc,
                &act,
                &act_scales,
                &w,
                &w_scales,
                &ids,
                tokens,
                hidden_dim,
                experts,
            );
            assert_eq!(
                cost.macs,
                (tokens * hidden_dim * MOE_TOP_K * MOE_WIDTH_PADDED) as u64
            );
        }
        let one_rank = 16.0f32 + 2.0f32.powi(-16);
        for value in &slot_acc {
            assert_eq!(
                *value,
                4.0 * one_rank,
                "four f32 rank partials add exactly; a per-rank bf16 rounding would give 64"
            );
        }
        // The one bf16 rounding happens in the finalize, where 64 + 2^-14 rounds back to 64:
        // six identical slots then add to 384.0.
        let routed = moe_combine_routed_f32(&slot_acc, &ids, tokens, hidden_dim);
        assert_eq!(routed[0], 384.0);
    }

    /// One captured-activation fixture committed for the native-rounding twin tests. The source
    /// paths and digests are in `tests/fixtures/native-rounding/m26-stage-input.json`.
    fn native_rounding_fixture(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native-rounding")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
    }

    /// The captured m26 expert ids, read from the committed stage-input record without a JSON
    /// dependency: the file is a checked-in fixture, not an untrusted input.
    fn captured_expert_ids() -> Vec<u32> {
        let text = String::from_utf8(native_rounding_fixture("m26-stage-input.json"))
            .expect("the fixture record is UTF-8");
        text.split_once("\"expert_ids\": [")
            .expect("the record names the expert ids")
            .1
            .split(']')
            .next()
            .expect("the id list is closed")
            .split(',')
            .map(|id| id.trim().parse::<u32>().expect("an integer expert id"))
            .collect()
    }

    /// The captured m26 MoE input row decoded to f32: E4M3 codes times the per-32 ue8m0 scales.
    fn captured_activation() -> Vec<f32> {
        let codes = native_rounding_fixture("m26-act_fp8.bin");
        let scales = native_rounding_fixture("m26-act_scales.bin");
        assert_eq!(codes.len(), scales.len() * 32);
        let mut values = Vec::with_capacity(codes.len());
        for block in 0..scales.len() {
            let scale = crate::m4_quant::ue8m0_decode(scales[block]);
            for code in &codes[block * 32..(block + 1) * 32] {
                values.push(crate::m4_quant::fp8_e4m3_decode(*code) * scale);
            }
        }
        values
    }

    /// The captured FFN input rounded to bf16: the dtype the released shared Expert returns
    /// (model.py:903). It plays the shared-expert output in the twin; m17b owns the real GEMM.
    fn captured_shared_stand_in(hidden_dim: usize) -> (Vec<u16>, Vec<f32>) {
        let bytes = native_rounding_fixture("m26-pre_ffn_hidden.f32");
        assert_eq!(bytes.len(), hidden_dim * 4);
        let bits: Vec<u16> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| bf16_bits(f32::from_le_bytes(*word)))
            .collect();
        let values: Vec<f32> = bits.iter().map(|bits| bf16_value(*bits)).collect();
        (bits, values)
    }

    /// Deterministic stand-ins for the four rank down-projection partials, driven by the captured
    /// activation values. The committed bundle holds the device run's `--input` bytes, not its
    /// rank accumulators or the checkpoint weights (the checkpoint is not on this host), so the
    /// row-to-partial map is a documented fixture map. The epilogue arithmetic under test does not
    /// depend on how the partials were produced.
    fn captured_rank_partials(activation: &[f32]) -> Vec<Vec<f32>> {
        let hidden_dim = activation.len();
        let mut ranks = Vec::with_capacity(4);
        for rank in 0..4usize {
            let mut acc = vec![0.0f32; MOE_TOP_K * hidden_dim];
            for slot in 0..MOE_TOP_K {
                for column in 0..hidden_dim {
                    let source = (column * 7 + rank * 13 + slot * 29) % hidden_dim;
                    let weight = 1.0 + ((column + rank * 17 + slot * 31) % 13) as f32 / 16.0;
                    acc[slot * hidden_dim + column] =
                        activation[source] * weight - 0.001 * rank as f32;
                }
            }
            ranks.push(acc);
        }
        ranks
    }

    #[test]
    fn the_native_combine_is_the_release_arithmetic_on_a_captured_activation() {
        let ids = captured_expert_ids();
        assert_eq!(
            ids.len(),
            MOE_TOP_K,
            "the captured router row has one id per slot"
        );
        let activation = captured_activation();
        let hidden_dim = activation.len();
        let ranks = captured_rank_partials(&activation);
        let slices: Vec<&[f32]> = ranks.iter().map(Vec::as_slice).collect();
        let (shared, shared_f32) = captured_shared_stand_in(hidden_dim);

        let native = moe_combine_native(&slices, &shared, &ids, 1, hidden_dim);

        // The independent release reference: model.py:893-904. Each expert's w2 output is the f32
        // sum of the ranks' partials rounded once to bf16; the bf16 experts are added in ascending
        // expert-id order; the bf16 shared output is added; the MoE output is rounded once.
        let order = moe_combine_slot_order(&ids, 0);
        let mut reference = vec![0u16; hidden_dim];
        for (column, expected) in reference.iter_mut().enumerate() {
            let mut experts = [0.0f32; MOE_TOP_K];
            for slot in 0..MOE_TOP_K {
                let mut sum = 0.0f32;
                for rank in &ranks {
                    sum += rank[slot * hidden_dim + column];
                }
                experts[slot] = bf16_value(bf16_bits(sum));
            }
            let mut routed = 0.0f32;
            for slot in order {
                routed += experts[slot];
            }
            *expected = bf16_bits(routed + shared_f32[column]);
        }
        assert_eq!(
            native, reference,
            "the native combine must be the release arithmetic bit for bit"
        );

        // The loop chain as it runs today (tp4_forward.rs run_layer_moe_routed + epilogue): each
        // rank finalises its own partial and bf16-rounds the payload before the canonical bf16
        // all-reduce, and the shared expert is added in f32 afterwards. The comparison rounds at
        // the same MoE-output boundary so the rank-partial/payload term is isolated from the
        // separate missing-final-round term.
        let mut reduced = vec![0.0f32; hidden_dim];
        for rank in &ranks {
            let routed = moe_combine_routed_f32(rank, &ids, 1, hidden_dim);
            for (total, value) in reduced.iter_mut().zip(routed.iter()) {
                *total += bf16_value(bf16_bits(*value));
            }
        }
        let current: Vec<u16> = reduced
            .iter()
            .zip(shared_f32.iter())
            .map(|(routed, shared)| bf16_bits(routed + shared))
            .collect();
        let mut mismatches = 0usize;
        let mut max_abs = 0.0f32;
        for (current, native) in current.iter().zip(native.iter()) {
            if current != native {
                mismatches += 1;
                let difference = bf16_value(*current) - bf16_value(*native);
                max_abs = max_abs.max(difference.abs());
            }
        }
        assert!(
            mismatches > 0,
            "the captured fixture must distinguish the loop chain from the native epilogue"
        );
        assert!(
            max_abs > 0.0,
            "a differing bf16 slot must carry a non-zero delta"
        );
        let mut delta_l2 = 0.0f64;
        let mut native_l2 = 0.0f64;
        for (current, native) in current.iter().zip(native.iter()) {
            let (current, native) = (
                f64::from(bf16_value(*current)),
                f64::from(bf16_value(*native)),
            );
            delta_l2 += (current - native) * (current - native);
            native_l2 += native * native;
        }
        eprintln!(
            "captured m26 activation: loop chain vs native combine {mismatches}/{hidden_dim} bf16 slots differ, max |delta| {max_abs:e}, relative L2 {:.6e}",
            (delta_l2 / native_l2).sqrt()
        );
    }

    #[test]
    fn the_rank_sum_adds_in_ascending_rank_order() {
        // The rank order is part of the arithmetic (f32 addition is not associative). Three ranks
        // with 1e20, -1e20 and 1.0: the ascending order keeps the 1.0 because it is added after the
        // two large values cancel; the reverse order loses it. If moe_combine_rank_sum ever
        // reassociates or reorders, this test moves.
        let mut top = vec![0.0f32; MOE_TOP_K];
        top[0] = 1e20;
        let mut bottom = vec![0.0f32; MOE_TOP_K];
        bottom[0] = -1e20;
        let mut one = vec![0.0f32; MOE_TOP_K];
        one[0] = 1.0;
        let ascending = moe_combine_rank_sum(&[&top, &bottom, &one], 1, 1);
        assert_eq!(ascending[0], 1.0, "ascending rank order keeps the 1.0");
        let reversed = moe_combine_rank_sum(&[&one, &bottom, &top], 1, 1);
        assert_eq!(
            reversed[0], 0.0,
            "the reverse order loses the 1.0, so the declared order is observable"
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

/// The stored-layout tests: the twin, the stride contract and the de-interleave proof.
///
/// These live in their own module because the incumbent tests above pin the A8 padded layout, and
/// the point here is that the two layouts are different contracts that must not be confused.
#[cfg(test)]
mod stored_tests {
    use super::*;
    use crate::m4_quant::{QUANT_BLOCK, bf16_bits};

    /// A deterministic pseudo-random byte stream (xorshift64), so the fixture needs no RNG crate.
    fn corpus(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    /// The released stored shape for a small fixture: [rows][288] codes and [rows][18] scales, with
    /// finite ue8m0 bytes (the decoder reserves 255) so a NaN cannot hide a row misindex.
    fn stored_rows(rows: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let codes = corpus(seed, rows * MOE_STORED_CODE_ROW);
        let scales: Vec<u8> = (0..rows * MOE_STORED_SCALE_ROW)
            .map(|index| 120 + (index % 9) as u8)
            .collect();
        (codes, scales)
    }

    /// Rebuild the padded A8 rows from the stored rows: the 288 real code bytes plus a tail of
    /// code_tail, and the 18 real scales plus a tail of scale_tail.
    fn deinterleave(
        codes: &[u8],
        scales: &[u8],
        rows: usize,
        code_tail: u8,
        scale_tail: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut padded_codes = Vec::with_capacity(rows * MOE_WEIGHT_BYTES_PER_ROW);
        let mut padded_scales = Vec::with_capacity(rows * MOE_SCALE_BYTES_PER_ROW);
        for row in 0..rows {
            padded_codes.extend_from_slice(
                &codes[row * MOE_STORED_CODE_ROW..(row + 1) * MOE_STORED_CODE_ROW],
            );
            padded_codes.extend(std::iter::repeat_n(
                code_tail,
                MOE_WEIGHT_BYTES_PER_ROW - MOE_STORED_CODE_ROW,
            ));
            padded_scales.extend_from_slice(
                &scales[row * MOE_STORED_SCALE_ROW..(row + 1) * MOE_STORED_SCALE_ROW],
            );
            padded_scales.extend(std::iter::repeat_n(
                scale_tail,
                MOE_SCALE_BYTES_PER_ROW - MOE_STORED_SCALE_ROW,
            ));
        }
        (padded_codes, padded_scales)
    }

    fn f32_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// The A operand: the incumbent's padded 640/20 layout, computed by the reference's own
    /// quantiser, so the stored twin is tested against the same operand the chain will feed it.
    fn operand(tokens: usize) -> (Vec<u8>, Vec<u8>) {
        let mut act = Vec::with_capacity(tokens * MOE_TOP_K * MOE_ACT_BYTES_PER_ROW);
        let mut act_scales = Vec::with_capacity(tokens * MOE_TOP_K * MOE_SCALE_BYTES_PER_ROW);
        for routed in 0..tokens * MOE_TOP_K {
            let values: Vec<f32> = (0..MOE_WIDTH_PADDED)
                .map(|index| ((routed * 37 + index * 11) % 19) as f32 / 8.0 - 1.0)
                .collect();
            let (row, scales) = routed_activation_operand(&values, 0.5);
            act.extend_from_slice(&row);
            act_scales.extend_from_slice(&scales);
        }
        (act, act_scales)
    }

    #[test]
    fn the_stored_twin_is_the_padded_twin_on_deinterleaved_rows() {
        // What this proves: the stored twin's row strides are the released layout's, by rebuilding
        // the padded A8 rows from the same stored bytes and requiring the two twins to produce
        // bit-identical f32 accumulators - the stride handling is the only difference between them.
        let tokens = 2usize;
        let hidden = 5usize;
        let experts = 3usize;
        let rows = experts * hidden;
        let (act, act_scales) = operand(tokens);
        let expert_ids: Vec<u32> = (0..tokens * MOE_TOP_K)
            .map(|index| ((index * 7 + 1) % experts) as u32)
            .collect();
        let (codes, scales) = stored_rows(rows, 0x9e37_79b9_7f4a_7c15);
        let shared: Vec<u16> = (0..tokens * hidden)
            .map(|index| bf16_bits(0.25 * (index as f32 + 1.0)))
            .collect();

        let mut stored_acc = vec![0.0f32; tokens * MOE_TOP_K * hidden];
        let stored_cost = moe_combine_rank_stored(
            &mut stored_acc,
            &act,
            &act_scales,
            &codes,
            &scales,
            &expert_ids,
            tokens,
            hidden,
            experts,
        );

        // Two de-interleaves: zeros in the padded tail (the canonical rebuild), and NONZERO garbage
        // there. The padded twin masks the activation at columns >= 576, so the padded tail's weight
        // bytes cannot change a term; requiring agreement in both cases proves the stored rows are
        // being read at their real strides rather than through the padded ones.
        for (name, code_tail, scale_tail) in [
            ("zero padding tail", 0x00u8, 0x00u8),
            ("nonzero padding tail", 0x11u8, 0x7Eu8),
        ] {
            let (padded_codes, padded_scales) =
                deinterleave(&codes, &scales, rows, code_tail, scale_tail);
            assert_eq!(stored_moe_layout().weight_row_bytes, 288);
            assert_eq!(stored_moe_layout().weight_scale_row_bytes, 18);
            let mut padded_acc = vec![0.0f32; tokens * MOE_TOP_K * hidden];
            let padded_cost = moe_combine_rank(
                &mut padded_acc,
                &act,
                &act_scales,
                &padded_codes,
                &padded_scales,
                &expert_ids,
                tokens,
                hidden,
                experts,
            );
            assert_eq!(
                f32_bits(&stored_acc),
                f32_bits(&padded_acc),
                "{name}: the stored twin's f32 accumulators must be bit-identical to the padded twin's"
            );
            let stored_rows_out =
                moe_combine_finalize(&stored_acc, &shared, &expert_ids, tokens, hidden);
            let padded_rows_out =
                moe_combine_finalize(&padded_acc, &shared, &expert_ids, tokens, hidden);
            assert_eq!(
                stored_rows_out, padded_rows_out,
                "{name}: the finalised rows must be bit-identical"
            );
            assert_eq!(
                stored_cost.macs,
                padded_cost.macs - padded_cost.padding_macs
            );
        }

        // The stored twin walks exactly the 18 real blocks: no padding MACs at all.
        assert_eq!(stored_cost.padding_macs, 0);
        assert_eq!(
            stored_cost.macs,
            (tokens * MOE_TOP_K * hidden * MOE_STORED_K_BLOCKS * QUANT_BLOCK) as u64
        );
        assert_eq!(
            stored_cost.weight_bytes,
            (tokens * MOE_TOP_K * hidden * MOE_STORED_K_BLOCKS * (QUANT_BLOCK / FP4_PER_BYTE))
                as u64
        );
    }

    #[test]
    fn the_stored_layout_is_a_different_contract_from_the_padded_one() {
        // 306 is the cost of one output row across the two tensors, not a stride: the code tensor
        // advances 288 bytes and the scale tensor 18, and each of those is enforced separately.
        assert_eq!(MOE_STORED_CODE_ROW + MOE_STORED_SCALE_ROW, 306);
        let stored = stored_moe_layout();
        assert_eq!(check_stored_moe_layout(stored), Ok(()));
        assert!(check_moe_layout(stored).is_err());
        assert!(check_stored_moe_layout(pre_padded_moe_layout()).is_err());
        assert_ne!(stored, pre_padded_moe_layout());

        // An interleaved 306-byte row - the layout the design page first assumed - is rejected on
        // both rows, so a loader cannot quietly write it and have the kernel read it as 288/18.
        let interleaved = MoeRowLayout {
            weight_row_bytes: 306,
            ..stored
        };
        assert!(check_stored_moe_layout(interleaved).is_err());
        // A padded 320/20 weight pair is rejected by the stored check as well.
        let padded_weights = MoeRowLayout {
            weight_row_bytes: MOE_WEIGHT_BYTES_PER_ROW,
            weight_scale_row_bytes: MOE_SCALE_BYTES_PER_ROW,
            ..stored
        };
        assert!(check_stored_moe_layout(padded_weights).is_err());
        // The two layouts differ only in the weight strides, not in the activation operand: that is
        // what keeps the A8 staging unchanged when the chain selects the stored cubin.
        assert_eq!(
            stored.activation_row_bytes,
            pre_padded_moe_layout().activation_row_bytes
        );
        assert_eq!(
            stored.activation_scale_row_bytes,
            pre_padded_moe_layout().activation_scale_row_bytes
        );
    }

    #[test]
    fn the_stored_twin_rejects_a_short_weight_buffer() {
        // The stride contract is also enforced by buffer length: a caller that staged the padded
        // arena but announced the stored shape is caught before any arithmetic, not after a partial
        // accumulation.
        let tokens = 1usize;
        let hidden = 2usize;
        let experts = 1usize;
        let (act, act_scales) = operand(tokens);
        let (codes, scales) = stored_rows(experts * hidden, 0xdead_beef_cafe_f00d);
        let ids = vec![0u32; tokens * MOE_TOP_K];
        let mut acc = vec![0.0f32; tokens * MOE_TOP_K * hidden];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            moe_combine_rank_stored(
                &mut acc,
                &act,
                &act_scales,
                &codes[..codes.len() - 1],
                &scales,
                &ids,
                tokens,
                hidden,
                experts,
            )
        }));
        assert!(result.is_err(), "a short code buffer must be rejected");
        let mut acc = vec![0.0f32; tokens * MOE_TOP_K * hidden];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            moe_combine_rank_stored(
                &mut acc,
                &act,
                &act_scales,
                &codes,
                &scales[..scales.len() - 1],
                &ids,
                tokens,
                hidden,
                experts,
            )
        }));
        assert!(result.is_err(), "a short scale buffer must be rejected");
    }
}
