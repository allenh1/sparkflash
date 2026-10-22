// The compressed-cache device form: the native FP4 append and the selected-list mapping.
//
// The rule is the one engine write_compressed_cache publishes (forward/attention.rs:2943-2992) and
// tp4_prefill::stage_attention calls per row: sparkflash_kv::fp4::quantize_main_kv
// (crates/sparkflash-kv/src/fp4.rs:411) - block 16, E4M3 scale of amax.max(6*2^-9)/6, E2M1 codes
// of value/scale clamped to +-6, nibbles packed low first, 256 code bytes and 32 scale bytes.
// The CPU twin is crates/sparkflash-kernels/src/compressed_cache_ref.rs; tests compare the twin to
// the pinned kv quantiser bit for bit and the launcher pins this file compiled.
//
// The append rule (see the twin header): a row is written exactly once, at store row
// (first_row + i) * ratio, because the body addresses the store by expanded (token) position, not
// by compressed row. The store is allocated zeroed once and only complete rows are written, so an
// untouched slot keeps its zeros - the body store_zero invariant. row_count == 0 is the skip: the
// kernel returns without touching a byte, and the host does not even launch it.
//
// The selected-list mapping is the arithmetic of tp4_prefill.rs:330-352: skip NO_INDEX
// (0xFFFFFFFF), subtract window_rows, multiply by ratio, keep the host order.
//
// ITS COST, declared: one thread keeps the order, so the fill is at most 512 dependent
// load-multiply-store iterations per call - a serial cost paid once per stage call and independent
// of how many ids the select actually chose (a few hundred nanoseconds to a microsecond scale on
// the device; a declared estimate, not a measurement - this box has no GPU). At 512 rows it is
// exactness for free against a parallel compaction that would have to reproduce the host order.

#include "m4_common_sm121.cuh"

#define REF_DIM 512u
#define REF_SCALE_GROUP 16u
#define REF_CODE_BYTES 256u
#define REF_SCALE_BYTES 32u
#define REF_E2M1_MAX 6.0f
#define REF_E4M3_MAX 448.0f
#define REF_AMAX_FLOOR 0.01171875f
#define REF_NO_INDEX 0xFFFFFFFFu

__device__ __forceinline__ float ref_exp2i(int n) {
    if (n >= -126) {
        return __uint_as_float((unsigned)((n + 127) << 23));
    }
    return (1.0f / 512.0f) * __uint_as_float((unsigned)((n + 9 + 127) << 23));
}

__device__ __forceinline__ float ref_e2m1_decode(unsigned code) {
    const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[code & 0x7u];
    return (code & 0x8u) != 0u ? -magnitude : magnitude;
}

__device__ __forceinline__ float ref_e4m3_decode(unsigned code) {
    const unsigned exp = (code >> 3) & 0xFu;
    const float mant = (float)(code & 0x7u);
    if (exp == 0xFu && mant == 7.0f) {
        return __uint_as_float(0x7FC00000u);
    }
    const float magnitude = (exp == 0u) ? mant * (1.0f / 512.0f)
                                        : (1.0f + mant / 8.0f) * ref_exp2i((int)exp - 7);
    return (code & 0x80u) != 0u ? -magnitude : magnitude;
}

__device__ __forceinline__ unsigned ref_e2m1_encode(float value) {
    if (value != value) {
        return 0u;
    }
    const unsigned sign = (__float_as_uint(value) >> 31) != 0u ? 0x8u : 0u;
    const float x = fabsf(value);
    if (x >= REF_E2M1_MAX) {
        return sign | 0x7u;
    }
    const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    unsigned best = 0u;
    float best_error = __uint_as_float(0x7F800000u);
    for (unsigned index = 0; index < 8u; ++index) {
        const float error = fabsf(x - magnitudes[index]);
        if (error < best_error || (error == best_error && (index % 2u) == 0u)) {
            best = index;
            best_error = error;
        }
    }
    return sign | best;
}

__device__ __forceinline__ unsigned ref_e4m3_encode(float value) {
    const unsigned max_code = 0x7Eu;
    if (value != value) {
        return 0x7Fu;
    }
    const unsigned sign = (__float_as_uint(value) >> 31) != 0u ? 0x80u : 0u;
    const float x = fabsf(value);
    if (x >= REF_E4M3_MAX) {
        return sign | max_code;
    }
    unsigned low = 0u;
    unsigned high = max_code;
    while (low < high) {
        const unsigned mid = low + (high - low + 1u) / 2u;
        if (fabsf(ref_e4m3_decode(mid)) <= x) {
            low = mid;
        } else {
            high = mid - 1u;
        }
    }
    if (low < max_code) {
        const unsigned up = low + 1u;
        const float down_error = x - fabsf(ref_e4m3_decode(low));
        const float up_error = fabsf(ref_e4m3_decode(up)) - x;
        if (up_error < down_error || (up_error == down_error && (up % 2u) == 0u)) {
            return sign | up;
        }
    }
    return sign | low;
}

// One block (256 threads) per completed compressed row. Thread 0..31 stage the 32 group scales;
// then thread t packs code bytes t (channels 2t and 2t+1), low nibble first.
extern "C" __global__ void compressed_append_sm121(
    const float* __restrict__ latent,
    unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales,
    unsigned first_row,
    unsigned row_count,
    unsigned ratio)
{
    if (row_count == 0u) {
        return;
    }
    const unsigned row = blockIdx.x;
    if (row >= row_count) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned store_row = (first_row + row) * ratio;
    const float* __restrict__ values = latent + (size_t)row * REF_DIM;
    __shared__ float s_scale[REF_SCALE_BYTES];

    if (tid < REF_SCALE_BYTES) {
        const float* group = values + tid * REF_SCALE_GROUP;
        float amax = 0.0f;
        for (unsigned offset = 0; offset < REF_SCALE_GROUP; ++offset) {
            amax = fmaxf(amax, fabsf(group[offset]));
        }
        const float floored = fmaxf(amax, REF_AMAX_FLOOR);
        s_scale[tid] = ref_e4m3_decode(ref_e4m3_encode(floored / REF_E2M1_MAX));
        scales[store_row * REF_SCALE_BYTES + tid] =
            (unsigned char)ref_e4m3_encode(floored / REF_E2M1_MAX);
    }
    __syncthreads();

    if (tid < REF_CODE_BYTES) {
        const unsigned first = tid * 2u;
        const float scale = s_scale[first / REF_SCALE_GROUP];
        const float a = values[first] / scale;
        const float b = values[first + 1u] / scale;
        const unsigned code_a =
            ref_e2m1_encode(fminf(fmaxf(a, -REF_E2M1_MAX), REF_E2M1_MAX));
        const unsigned code_b =
            ref_e2m1_encode(fminf(fmaxf(b, -REF_E2M1_MAX), REF_E2M1_MAX));
        codes[store_row * REF_CODE_BYTES + tid] =
            (unsigned char)((code_a & 0xFu) | ((code_b & 0xFu) << 4));
    }
}

// The selected list: one thread keeps the host order exactly (at most 512 ids), so the id vector
// the body reads is the host id vector mapped, not a reordered one.
extern "C" __global__ void compressed_selected_fill_sm121(
    const unsigned* __restrict__ ids,
    unsigned n_ids,
    unsigned window_rows,
    unsigned ratio,
    unsigned* __restrict__ out,
    unsigned* __restrict__ out_count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    unsigned picked = 0u;
    for (unsigned index = 0; index < n_ids; ++index) {
        const unsigned id = ids[index];
        if (id == REF_NO_INDEX) {
            continue;
        }
        out[picked] = (id - window_rows) * ratio;
        picked += 1u;
    }
    if (out_count != nullptr) {
        *out_count = picked;
    }
}
