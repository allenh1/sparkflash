// The four kv_path device kernels: kv_norm then rope on the MLA latent (fused, one bf16 rounding),
// rope on the stored row's tail, the whole-row window fp8 round-trip, and this position's k/v row
// pair appended to the device cache. The append implements
// the frozen DeviceKvCache geometry and visibility contract from
// crates/sparkflash-cli/src/layer_device.rs (win/layer-device @ ee0a0daa): one row is
// row_values = 2 * head_dim bf16 values (k then v), a layer's run is capacity_rows such rows, the
// row for position p lives at slot p % rows, and publication is stream order after the position's
// window read.
//
// WHY ONE FILE WITH FOUR SYMBOLS. The four kernels are one design piece and one cubin: build.rs
// probes the file, baseline/kernel_baseline.txt pins its cubin sha256, and the four symbols are
// loaded by name (crate::kv_path_launch) rather than through a KernelId, which names one symbol.
// The baseline was regenerated on this host with the pinned toolkit after the regen was verified to
// reproduce every pre-existing entry byte for byte, so the digest is the same kind of record as the
// other kernel files'. A CPU-only host still gets the clang CUDA frontend check against
// cuda_stub/cuda_stub.h through build.rs, exactly as for every other kernel.
//
// SEMANTICS (docs/archive/build/ws-kv-path-semantics-20261022.md, sections 2 and 3):
//
//   rope_rotate_bf16 - row and out MAY alias (the same buffer): each pair reads both of its inputs
//   before writing either, and no thread touches another pair's values, so in-place rotation needs no
//   scratch. The out-of-place form is the declared signature; the kernel does not assume it.
//
//   rope_rotate_bf16 - the engine's linalg::rope_rotate on the last rope_head_dim values of one
//   row: ADJACENT pairs (x[2i], x[2i+1]) -> (a cos - b sin, a sin + b cos), NOT the half-split
//   NeoX convention. The values are bf16 (the stored row), widened to f32 exactly, multiplied and
//   added with the rounding intrinsics so no FMA contraction can round once where the engine rounds
//   twice, and rounded back to bf16 once. The head before the tail is copied unchanged. cos/sin are
//   the per-position tables, rope_head_dim / 2 long, one element per pair.
//
//   window_quant_fp8_bf16 - the released window format applied to the WHOLE rotated row (the window
//   is a slice of already-quantised rows, never a second quantisation): one absolute maximum per
//   32-value block, floor it at 1e-4, divide by 448 in f32, round the exponent up (the activation
//   rule spark_act_quant_scale_code, which is not the weight-grid rule), E8M0 code, then each value
//   divided by the decoded power of two, clamped to +-448 and encoded E4M3 round-to-nearest-even.
//   The helpers are the activation quantiser's own header, so the format cannot drift from the
//   other consumers of it; the engine path asserts byte equality with the m4_quant oracle and the
//   Rust twin (src/kv_path_ref.rs) asserts the same, so a mismatch fails a test rather than a bar.
//
// There is no trace word in either signature: the pinned contract in KERNEL_CONTRACTS does not carry
// the ring/layer/step words, and the protocol's 16 scope bits are assigned with none describing a kv
// append.
//
// One launch handles one row: the kv path rotates and quantises the position's single 512-value
// latent row. The launchers in src/kv_path_launch.rs refuse a zero row, a tail that does not fit the
// row, an odd tail, and a block that is not the format's 32.

#include "m4_activation_quant_sm121.cuh"

// One block of threads per row; head_width is at most a few hundred values, so one block finishes
// the tail in a single wave and the head copy in a second loop of the same block.
#define SPARK_KV_ROW_THREADS 256u

extern "C" __global__ void rope_rotate_bf16(
    const __nv_bfloat16 *__restrict__ row,      // [head_width] the stored rotated row
    const float *__restrict__ cos,              // [rope_head_dim / 2] this position's cos
    const float *__restrict__ sin,              // [rope_head_dim / 2] this position's sin
    __nv_bfloat16 *__restrict__ out,            // [head_width] the rotated row
    const uint32_t head_width,                  // 512 for the MLA latent
    const uint32_t rope_head_dim,               // 64 in the released config
    const uint32_t position) {                  // ABI word; the tables are already this position's
    (void)position;
    const uint32_t start = head_width - rope_head_dim;
    const uint32_t pairs = rope_head_dim >> 1u;
    for (uint32_t index = threadIdx.x; index < start; index += blockDim.x) {
        out[index] = row[index];
    }
    for (uint32_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        const float a = __bfloat162float(row[start + 2u * pair]);
        const float b = __bfloat162float(row[start + 2u * pair + 1u]);
        // a cos - b sin as add(negate(b sin)): negation is exact, so this is the engine's
        // subtraction rounding for rounding, without needing a subtract intrinsic.
        const float first = __fadd_rn(__fmul_rn(a, cos[pair]), -__fmul_rn(b, sin[pair]));
        const float second = __fadd_rn(__fmul_rn(a, sin[pair]), __fmul_rn(b, cos[pair]));
        out[start + 2u * pair] = __float2bfloat16_rn(first);
        out[start + 2u * pair + 1u] = __float2bfloat16_rn(second);
    }
}

// kv_norm_rope_bf16 - the host's kv_norm then rope, fused so the device rounds to bf16 ONCE.
//
// HOST ORDER, which this kernel reproduces rather than approximates (tp4_forward.rs:3032-3038):
//
//     rms_norm(kv, kv_norm, 512, norm_eps)     // linalg::rms_norm, kv f32 in the host
//     rope_rotate(kv[512-64..], cos, sin)      // linalg::rope_rotate on the tail, still f32
//     bf16 once                                // bf16_le_bytes at the end
//
// so the arithmetic per row, ascending, is: sum = sum(x*x) in f32; mean = sum/512; inv =
// 1/sqrt(mean+eps); each value (x*inv)*w in f32; the tail's adjacent pairs (a cos - b sin,
// a sin + b cos) in f32; ONE bf16 rounding of the final f32 per output value. The rounding
// intrinsics keep nvcc from contracting the square-and-add or the pair multiply-add into an FMA,
// and the reciprocal is sqrt-then-divide rather than rsqrt, exactly as the engine rounds. One thread
// owns the whole 512-value row so the sum is ascending and sequential like the host's - a tree
// reduction would sum in another order and lose bit-exactness (kernels/rmsnorm_sm121.cu makes the
// same choice for the same reason).
//
// DECLARED DEVIATION, for the D54 tolerance arm: the device latent is ALREADY bf16 (the wkv GEMV
// writes bf16) where the host keeps it f32 from its own f32 GEMV, so the device widens bf16 inputs
// where the host had f32. That is one rounding the host does not have, and it is the only difference
// in the sequence; the reference twin's tests pin both facts (see src/kv_path_ref.rs: the twin is
// bit-identical to the host sequence given the same bf16 operand, and the f32-vs-bf16 latent gap is
// measured rather than assumed).
//
// norm is the released bf16 kv_norm.weight, whose bytes widen to the engine's f32 weight exactly.
// cos/sin are this position's f32 tables, rope_head_dim / 2 long, one element per pair.
extern "C" __global__ void kv_norm_rope_bf16(
    const __nv_bfloat16 *__restrict__ row,      // [head_width] the bf16 wkv latent
    const __nv_bfloat16 *__restrict__ norm,     // [head_width] the bf16 kv_norm weight
    const float eps,                            // config.norm_eps as f32
    const float *__restrict__ cos,              // [rope_head_dim / 2] this position's cos
    const float *__restrict__ sin,              // [rope_head_dim / 2] this position's sin
    __nv_bfloat16 *__restrict__ out,            // [head_width] normed + rotated, ONE rounding
    const uint32_t head_width,                  // 512 for the MLA latent
    const uint32_t rope_head_dim) {             // 64 in the released config
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    float sum = 0.0f;
    for (uint32_t index = 0; index < head_width; ++index) {
        const float value = __bfloat162float(row[index]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    const float mean = __fdiv_rn(sum, (float)head_width);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    const uint32_t start = head_width - rope_head_dim;
    for (uint32_t index = 0; index < start; ++index) {
        const float normed =
            __fmul_rn(__fmul_rn(__bfloat162float(row[index]), inv), __bfloat162float(norm[index]));
        out[index] = __float2bfloat16_rn(normed);
    }
    const uint32_t pairs = rope_head_dim >> 1u;
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        const uint32_t index = start + 2u * pair;
        const float a = __fmul_rn(__fmul_rn(__bfloat162float(row[index]), inv),
                                  __bfloat162float(norm[index]));
        const float b = __fmul_rn(__fmul_rn(__bfloat162float(row[index + 1u]), inv),
                                  __bfloat162float(norm[index + 1u]));
        // a cos - b sin as add(negate(b sin)): negation is exact, so this is the engine's
        // subtraction rounding without a subtract intrinsic.
        const float first = __fadd_rn(__fmul_rn(a, cos[pair]), -__fmul_rn(b, sin[pair]));
        const float second = __fadd_rn(__fmul_rn(a, sin[pair]), __fmul_rn(b, cos[pair]));
        out[index] = __float2bfloat16_rn(first);
        out[index + 1u] = __float2bfloat16_rn(second);
    }
}

extern "C" __global__ void window_quant_fp8_bf16(
    const __nv_bfloat16 *__restrict__ row,      // [cols] the rotated row, already bf16-stored
    uint8_t *__restrict__ codes,                // [cols] E4M3FN, one per value
    uint8_t *__restrict__ scales,               // [cols / block] E8M0, one per block
    const uint32_t cols,                        // 512 for the MLA latent
    const uint32_t block) {                     // 32: the released quantiser's block
    const uint32_t blocks = cols / block;
    for (uint32_t index = threadIdx.x; index < blocks; index += blockDim.x) {
        const uint32_t base = index * block;
        float amax = 0.0f;
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(row[base + i]);
            const float magnitude = value < 0.0f ? -value : value;
            amax = magnitude > amax ? magnitude : amax;
        }
        const uint8_t scale_code = spark_act_quant_scale_code(amax);
        scales[index] = scale_code;
        const float scale = spark_ue8m0_decode(scale_code);
        for (uint32_t i = 0; i < SPARK_M4_QUANT_BLOCK; ++i) {
            const float value = __bfloat162float(row[base + i]);
            float scaled = __fdiv_rn(value, scale);
            scaled = scaled > SPARK_M4_ACT_E4M3_MAX ? SPARK_M4_ACT_E4M3_MAX : scaled;
            scaled = scaled < -SPARK_M4_ACT_E4M3_MAX ? -SPARK_M4_ACT_E4M3_MAX : scaled;
            codes[base + i] = spark_fp8_e4m3_encode(scaled);
        }
    }
}


// kv_append_bf16 - this position's k/v row pair written into the layer's ring of the device cache.
//
// GEOMETRY, from DeviceKvCache (crates/sparkflash-cli/src/layer_device.rs): one row is
// row_values = 2 * head_dim bf16 values, the k row first and the v row immediately after it, and a
// layer's run is capacity_rows such rows (layer_bytes = capacity_rows * row_values * 2). The cache is
// a RING over positions: the row for position p lives at slot p % rows, and the window seam reads
// the last window_width positions through the same modulo, so a row older than the window may be
// overwritten by the ring and is never read (rows later than p are not read whatever the ring holds).
//
// VISIBILITY, the contract this kernel implements rather than implies: the stage queues this append
// on the stream AFTER the position's window read and BEFORE the next position's read (the host loop
// pushes at the bottom of the position, after the window slice). Publication is stream order: there
// is no synchronise in this kernel, no host-visible copy, and no cross-stream publication. The
// kernel only writes its own slot; it never reads the cache and never touches another row.
//
// The rows arrive already rotated and already bf16 (the rope/quantiser sequence runs before the
// append for the same position); this kernel copies, it does not compute. A caller that wants the
// quantised window bytes for the position gets them from window_quant_fp8_bf16 on the same rotated
// row - the cache itself stores the bf16 pair the attention body's slice-copy reads.
extern "C" __global__ void kv_append_bf16(
    const __nv_bfloat16 *__restrict__ k_row,    // [head_dim] this position's k row
    const __nv_bfloat16 *__restrict__ v_row,    // [head_dim] this position's v row
    __nv_bfloat16 *__restrict__ cache,          // [rows][2 * head_dim] the layer's ring
    const uint32_t rows,                        // ring capacity in rows (capacity_rows)
    const uint32_t head_dim,                    // values per row (the head width)
    const uint32_t position) {                  // the position being appended
    const uint32_t slot = position % rows;
    __nv_bfloat16 *const k_out = cache + (size_t)slot * 2u * head_dim;
    __nv_bfloat16 *const v_out = k_out + head_dim;
    for (uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        k_out[index] = k_row[index];
        v_out[index] = v_row[index];
    }
}

