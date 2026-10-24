// kernels/opt_moe_sm121.cu
//
// The routed-expert FFN in two launches instead of eight. Same arithmetic, same rounding points,
// same order of every rounded add as the staged chain (moe_expert_stage + routed_gate_up x2 +
// swiglu_mxfp8_quant_weighted + moe_expert_stage + b12x_mxfp4_moe_stored + moe_routed_combine), so
// the result is bit-identical to it; what changes is how the bytes move:
//
//   * no staging copy: both kernels read the expert slices in place through the pointer table;
//   * 16-byte coalesced weight loads (one lane = one 32-value MX block) instead of byte loads;
//   * the FP4 x E4M3 products are formed with the paired hardware converts and one f16x2 multiply.
//     Every such product is exact in f16 (|a| <= 448, |w| <= 6, at most 6 significant bits, never
//     below 2^-10), so widening it to f32 gives the same value the f32 multiply gave.
//
// opt_moe_gate_up_sm121: grid (576 / 32, rows), 128 threads. A block owns one routed row's 32
//   consecutive intermediate columns (one MX block of the SwiGLU output). Each warp computes 8
//   columns, gate and up together; each lane holds 5 of the 160 K blocks. The 160 scaled block terms
//   are folded in ascending block order with a shuffle chain (the reference's only ordered sum),
//   rounded to bf16, put through the weighted SwiGLU and the activation quantiser.
//   Output: codes [rows][576] E4M3, scales [rows][18] ue8m0.
//
// opt_moe_down_combine_sm121: grid (5120 / 32, tokens), 256 threads. A block owns 32 output columns
//   of one token for all six slots: 576 16-byte chunks per slot, read in address order, terms into
//   shared memory, an ascending 18-term fold per (slot, column), then the routed combine (slots in
//   ascending expert id, each rounded to bf16, f32 adds) writing delta_f32 and delta_bf16.

#include "opt_pdl_sm121.cuh"
#include "m4_common_sm121.cuh"
#include "m4_activation_quant_sm121.cuh"

#define OPT_MOE_HIDDEN 5120u
#define OPT_MOE_KB 160u
#define OPT_MOE_INTER 576u
#define OPT_MOE_GU_ROW 2560u
#define OPT_MOE_GU_SCALE_ROW 160u
#define OPT_MOE_D_ROW 288u
#define OPT_MOE_D_KB 18u
#define OPT_MOE_TOPK 6u
#define OPT_MOE_GU_THREADS 128u
#define OPT_MOE_GU_COLS_PER_WARP 8u
#define OPT_MOE_D_THREADS 256u
#define OPT_MOE_D_TILE 32u
#define OPT_MOE_D_CHUNKS (OPT_MOE_D_TILE * OPT_MOE_D_KB)

static_assert(OPT_MOE_KB == 5u * 32u, "five K blocks per lane");
static_assert(OPT_MOE_INTER % 32u == 0u, "the intermediate width tiles by 32");
static_assert((OPT_MOE_GU_THREADS / 32u) * OPT_MOE_GU_COLS_PER_WARP == 32u, "a block owns 32 columns");
static_assert(OPT_MOE_D_ROW == OPT_MOE_D_KB * 16u, "one 16-byte chunk per down K block");
static_assert(OPT_MOE_TOPK == SPARK_M4_MOE_TOP_K, "six routed slots");

__device__ __forceinline__ uint4 opt_ldg_stream(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// One 32-value block partial: sum over j of e4m3(a_j) * e2m1(w_j), sequential in j from +0.0, with
// the pair order (low nibble, high nibble) of every weight byte - the reference's loop exactly.
__device__ __forceinline__ float opt_block_partial(uint4 w, uint4 a0, uint4 a1) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t aa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float partial = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < 16u; ++k) {
        const __nv_fp4x2_storage_t wb = (__nv_fp4x2_storage_t)((ww[k >> 2] >> (8u * (k & 3u))) & 0xFFu);
        const __nv_fp8x2_storage_t ab = (__nv_fp8x2_storage_t)((aa[k >> 1] >> (16u * (k & 1u))) & 0xFFFFu);
        const __half2 wh = __half2(__nv_cvt_fp4x2_to_halfraw2(wb, __NV_E2M1));
        const __half2 ah = __half2(__nv_cvt_fp8x2_to_halfraw2(ab, __NV_E4M3));
        const float2 p = __half22float2(__hmul2(ah, wh));
        partial = __fadd_rn(partial, p.x);
        partial = __fadd_rn(partial, p.y);
    }
    return partial;
}

__device__ __forceinline__ uint32_t opt_round_slot_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (bits >> 16) | 0x0040u;
    }
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint16_t opt_narrow_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

template <uint32_t TOPK>
__device__ __forceinline__ void opt_moe_gate_up_body(
    const uint64_t *__restrict__ gate_table,  // [experts][2] (weight, scales) for this layer's w1
    const uint64_t *__restrict__ up_table,    // [experts][2] for this layer's w3
    const uint32_t *__restrict__ expert_ids,  // [rows] routed-row order
    const float *__restrict__ weights,        // [rows] routing weights
    const uint8_t *__restrict__ act_fp8,      // [tokens][5120] E4M3
    const uint8_t *__restrict__ act_scales,   // [tokens][160] ue8m0
    uint8_t *__restrict__ out_codes,          // [rows][576] E4M3
    uint8_t *__restrict__ out_scales,         // [rows][18] ue8m0
    const uint32_t rows,
    const uint32_t experts,
    const float limit) {
    const uint32_t row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t token = row / TOPK;
    const uint32_t id = expert_ids[row] < experts ? expert_ids[row] : 0u;
    const uint8_t *gw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * id]));
    const uint8_t *gs = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * id + 1u]));
    const uint8_t *uw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * id]));
    const uint8_t *us = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * id + 1u]));
    const float routing = weights[row];

    // The token's activation blocks this lane owns: K blocks lane + 32 k, k = 0..4.
    const uint4 *act = reinterpret_cast<const uint4 *>(act_fp8 + (uint64_t)token * OPT_MOE_HIDDEN);
    uint4 a[5][2];
    float as[5];
#pragma unroll
    for (uint32_t k = 0u; k < 5u; ++k) {
        const uint32_t block = lane + 32u * k;
        a[k][0] = __ldg(act + 2u * block);
        a[k][1] = __ldg(act + 2u * block + 1u);
        as[k] = spark_ue8m0_decode(act_scales[(uint64_t)token * OPT_MOE_GU_SCALE_ROW + block]);
    }

    __shared__ float s_val[32];
    const uint32_t col0 = blockIdx.x * 32u + warp * OPT_MOE_GU_COLS_PER_WARP;
    for (uint32_t c = 0u; c < OPT_MOE_GU_COLS_PER_WARP; ++c) {
        const uint32_t column = col0 + c;
        const uint4 *grow = reinterpret_cast<const uint4 *>(gw + (uint64_t)column * OPT_MOE_GU_ROW);
        const uint4 *urow = reinterpret_cast<const uint4 *>(uw + (uint64_t)column * OPT_MOE_GU_ROW);
        uint4 wg[5], wu[5];
        float sg[5], su[5];
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
            wg[k] = opt_ldg_stream(grow + lane + 32u * k);
            wu[k] = opt_ldg_stream(urow + lane + 32u * k);
        }
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
            sg[k] = spark_ue8m0_decode(__ldg(gs + (uint64_t)column * OPT_MOE_GU_SCALE_ROW + lane + 32u * k));
            su[k] = spark_ue8m0_decode(__ldg(us + (uint64_t)column * OPT_MOE_GU_SCALE_ROW + lane + 32u * k));
        }
        float tg[5], tu[5];
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
            tg[k] = __fmul_rn(__fmul_rn(opt_block_partial(wg[k], a[k][0], a[k][1]), as[k]), sg[k]);
            tu[k] = __fmul_rn(__fmul_rn(opt_block_partial(wu[k], a[k][0], a[k][1]), as[k]), su[k]);
        }
        // Ascending fold over blocks 0..159: block b lives in lane b % 32, register b / 32.
        float accg = 0.0f, accu = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
#pragma unroll
            for (uint32_t l = 0u; l < 32u; ++l) {
                accg = __fadd_rn(accg, __shfl_sync(0xFFFFFFFFu, tg[k], l));
                accu = __fadd_rn(accu, __shfl_sync(0xFFFFFFFFu, tu[k], l));
            }
        }
        if (lane == 0u) {
            const float g = __bfloat162float(__float2bfloat16_rn(accg));
            const float u = __bfloat162float(__float2bfloat16_rn(accu));
            const float clamped_gate = fminf(g, limit);
            const float clamped_up = fminf(fmaxf(u, -limit), limit);
            const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
            s_val[warp * OPT_MOE_GU_COLS_PER_WARP + c] =
                __bfloat162float(__float2bfloat16_rn(activated * routing));
        }
    }
    __syncthreads();
    if (warp == 0u) {
        const float value = s_val[lane];
        float amax = fabsf(value);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
        }
        const uint8_t scale = spark_act_quant_scale_code(amax);
        if (lane == 0u) {
            out_scales[(uint64_t)row * OPT_MOE_D_KB + blockIdx.x] = scale;
        }
        const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
        out_codes[(uint64_t)row * OPT_MOE_INTER + blockIdx.x * 32u + lane] =
            spark_fp8_e4m3_encode(value * inv_scale);
    }
}

#define OPT_MOE_GATE_UP_ENTRY(name, TOPK)                                                          \
    extern "C" __global__ void __launch_bounds__(OPT_MOE_GU_THREADS) name(                         \
        const uint64_t *__restrict__ gate_table, const uint64_t *__restrict__ up_table,           \
        const uint32_t *__restrict__ expert_ids, const float *__restrict__ weights,               \
        const uint8_t *__restrict__ act_fp8, const uint8_t *__restrict__ act_scales,              \
        uint8_t *__restrict__ out_codes, uint8_t *__restrict__ out_scales, const uint32_t rows,   \
        const uint32_t experts, const float limit) {                                              \
        OPT_PDL_ENTRY();                                                                          \
        opt_moe_gate_up_body<TOPK>(gate_table, up_table, expert_ids, weights, act_fp8,            \
                                   act_scales, out_codes, out_scales, rows, experts, limit);      \
    }

OPT_MOE_GATE_UP_ENTRY(opt_moe_gate_up_sm121, OPT_MOE_TOPK)
// The DSpark stages route top-3 of 128 experts with the backbone's expert geometry.
OPT_MOE_GATE_UP_ENTRY(opt_moe_gate_up_k3_sm121, 3u)

template <uint32_t TOPK>
__device__ __forceinline__ void opt_moe_down_combine_body(
    const uint64_t *__restrict__ down_table,  // [experts][2] (weight, scales) for this layer's w2
    const uint32_t *__restrict__ expert_ids,  // [tokens * 6]
    const uint8_t *__restrict__ h_codes,      // [tokens * 6][576] E4M3
    const uint8_t *__restrict__ h_scales,     // [tokens * 6][18] ue8m0
    float *__restrict__ delta_f32,            // [tokens][5120]
    uint16_t *__restrict__ delta_bf16,        // [tokens][5120]
    const uint32_t tokens,
    const uint32_t experts,
    const float *addend) {                 // [tokens][5120] f32 added last, or null
    const uint32_t token = blockIdx.y;
    if (token >= tokens) {
        return;
    }
    const uint32_t tid = threadIdx.x;
    const uint32_t tile = blockIdx.x;
    __shared__ uint4 s_h[TOPK][OPT_MOE_INTER / 16u];
    __shared__ float s_hs[TOPK][OPT_MOE_D_KB];
    __shared__ float s_terms[TOPK][OPT_MOE_D_TILE][OPT_MOE_D_KB + 1u];
    __shared__ float s_val[TOPK][OPT_MOE_D_TILE];
    __shared__ uint32_t s_ids[TOPK];

    const uint64_t row0 = (uint64_t)token * TOPK;
    const uint4 *hsrc = reinterpret_cast<const uint4 *>(h_codes + row0 * OPT_MOE_INTER);
    for (uint32_t i = tid; i < TOPK * (OPT_MOE_INTER / 16u); i += OPT_MOE_D_THREADS) {
        (&s_h[0][0])[i] = hsrc[i];
    }
    if (tid < TOPK * OPT_MOE_D_KB) {
        (&s_hs[0][0])[tid] = spark_ue8m0_decode(h_scales[row0 * OPT_MOE_D_KB + tid]);
    }
    if (tid < TOPK) {
        s_ids[tid] = expert_ids[row0 + tid];
    }
    __syncthreads();

    for (uint32_t s = 0u; s < TOPK; ++s) {
        const uint32_t id = s_ids[s] < experts ? s_ids[s] : 0u;
        const uint8_t *w = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * id]));
        const uint8_t *ws = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * id + 1u]));
        const uint4 *wtile = reinterpret_cast<const uint4 *>(w + (uint64_t)tile * OPT_MOE_D_TILE * OPT_MOE_D_ROW);
        const uint8_t *stile = ws + (uint64_t)tile * OPT_MOE_D_TILE * OPT_MOE_D_KB;
        uint4 wv[3];
        uint8_t sv[3];
#pragma unroll
        for (uint32_t it = 0u; it < 3u; ++it) {
            const uint32_t q = tid + it * OPT_MOE_D_THREADS;
            if (q < OPT_MOE_D_CHUNKS) {
                wv[it] = opt_ldg_stream(wtile + q);
                sv[it] = __ldg(stile + q);
            }
        }
#pragma unroll
        for (uint32_t it = 0u; it < 3u; ++it) {
            const uint32_t q = tid + it * OPT_MOE_D_THREADS;
            if (q < OPT_MOE_D_CHUNKS) {
                const uint32_t col = q / OPT_MOE_D_KB;
                const uint32_t blk = q % OPT_MOE_D_KB;
                const float partial = opt_block_partial(wv[it], s_h[s][2u * blk], s_h[s][2u * blk + 1u]);
                s_terms[s][col][blk] =
                    __fmul_rn(__fmul_rn(partial, s_hs[s][blk]), spark_ue8m0_decode(sv[it]));
            }
        }
    }
    __syncthreads();

    if (tid < TOPK * OPT_MOE_D_TILE) {
        const uint32_t s = tid / OPT_MOE_D_TILE;
        const uint32_t col = tid % OPT_MOE_D_TILE;
        float acc = 0.0f;
#pragma unroll
        for (uint32_t b = 0u; b < OPT_MOE_D_KB; ++b) {
            acc = __fadd_rn(acc, s_terms[s][col][b]);
        }
        // The staged chain added this into a zeroed f32 accumulator; keep that add (it only matters
        // for the sign of an exact zero).
        s_val[s][col] = __fadd_rn(0.0f, acc);
    }
    __syncthreads();

    if (tid < OPT_MOE_D_TILE) {
        uint32_t used = 0u;
        float total = 0.0f;
#pragma unroll
        for (uint32_t position = 0u; position < TOPK; ++position) {
            uint32_t best = TOPK;
#pragma unroll
            for (uint32_t slot = 0u; slot < TOPK; ++slot) {
                if (((used >> slot) & 1u) != 0u) {
                    continue;
                }
                if (best == TOPK || s_ids[slot] < s_ids[best]) {
                    best = slot;
                }
            }
            used |= 1u << best;
            total = __fadd_rn(total, __uint_as_float(opt_round_slot_bits(s_val[best][tid]) << 16));
        }
        const uint64_t out = (uint64_t)token * OPT_MOE_HIDDEN + (uint64_t)tile * OPT_MOE_D_TILE + tid;
        if (addend != nullptr) {
            // Read before the write below: the addend may alias delta_f32 (same index, same thread).
            total = __fadd_rn(total, addend[out]);
        }
        delta_f32[out] = total;
        delta_bf16[out] = opt_narrow_bits(total);
    }
}

#define OPT_MOE_DOWN_ENTRY(name, TOPK)                                                             \
    extern "C" __global__ void __launch_bounds__(OPT_MOE_D_THREADS) name(                          \
        const uint64_t *__restrict__ down_table, const uint32_t *__restrict__ expert_ids,         \
        const uint8_t *__restrict__ h_codes, const uint8_t *__restrict__ h_scales,                \
        float *__restrict__ delta_f32, uint16_t *__restrict__ delta_bf16, const uint32_t tokens,  \
        const uint32_t experts, const float *addend) {                                            \
        OPT_PDL_ENTRY();                                                                          \
        opt_moe_down_combine_body<TOPK>(down_table, expert_ids, h_codes, h_scales, delta_f32,     \
                                        delta_bf16, tokens, experts, addend);                     \
    }

OPT_MOE_DOWN_ENTRY(opt_moe_down_combine_sm121, OPT_MOE_TOPK)
OPT_MOE_DOWN_ENTRY(opt_moe_down_combine_k3_sm121, 3u)

// ---- Multi-row variants: one weight read per distinct expert.
//
// A verify step's rows route to overlapping experts (four rows touch ~17 distinct experts of their
// 24 slots). The per-row kernels above read an expert's slices once per routed row; these read them
// once per distinct expert and serve every row routed to it. Each row's arithmetic is the per-row
// kernel's, in the same order, so every output is bit-identical to it.
//
// The plan (grouping the routed rows by expert) is recomputed by every block in shared memory from
// the router's ids: rows <= OPT_MOE_PLAN_MAX, one thread per row, O(rows) per thread.
//
// opt_moe_gate_up_dedup: grid (576 / 32, rows), 128 threads; block y serves distinct expert y (blocks
//   past the distinct count exit). Same outputs as opt_moe_gate_up.
//   Block (0, 0) also writes the plan to plan_out for the down launch.
// opt_moe_down_dedup: grid (5120 / 64, rows), 256 threads; block y serves distinct expert y of that
//   plan over two 32-column tiles and writes each of its rows' slot values (the per-row kernel's
//   s_val) to slot_val [rows][5120].
// opt_moe_combine: grid (5120 / 256, tokens), 256 threads; the routed combine of the slot values
//   (ascending expert id, each rounded to bf16, f32 adds, then the addend), as opt_moe_down_combine.

#define OPT_MOE_PLAN_MAX 64u

struct OptMoePlan {
    uint32_t count;                       // distinct experts
    uint32_t expert[OPT_MOE_PLAN_MAX];    // per group
    uint32_t start[OPT_MOE_PLAN_MAX + 1]; // per group, into rows
    uint32_t rows[OPT_MOE_PLAN_MAX];      // routed rows, grouped
};

__device__ __forceinline__ void opt_moe_plan(const uint32_t *__restrict__ expert_ids, uint32_t n,
                                             uint32_t experts, OptMoePlan &plan) {
    __shared__ uint32_t s_id[OPT_MOE_PLAN_MAX];
    __shared__ uint32_t s_first[OPT_MOE_PLAN_MAX];
    const uint32_t i = threadIdx.x;
    if (i < n) {
        const uint32_t raw = expert_ids[i];
        s_id[i] = raw < experts ? raw : 0u;
    }
    __syncthreads();
    if (i < n) {
        uint32_t first = i;
        for (uint32_t j = 0u; j < i; ++j) {
            if (s_id[j] == s_id[i]) {
                first = j;
                break;
            }
        }
        s_first[i] = first;
    }
    __syncthreads();
    if (i < n) {
        const uint32_t f = s_first[i];
        uint32_t group = 0u, start = 0u, within = 0u;
        for (uint32_t j = 0u; j < n; ++j) {
            const uint32_t fj = s_first[j];
            group += (j < f && fj == j) ? 1u : 0u;
            start += fj < f ? 1u : 0u;
            within += (j < i && fj == f) ? 1u : 0u;
        }
        plan.rows[start + within] = i;
        if (f == i) {
            plan.expert[group] = s_id[i];
            plan.start[group] = start;
        }
        if (i == 0u) {
            uint32_t count = 0u;
            for (uint32_t j = 0u; j < n; ++j) {
                count += s_first[j] == j ? 1u : 0u;
            }
            plan.count = count;
            plan.start[count] = n;
        }
    }
    __syncthreads();
}

template <uint32_t TOPK>
__device__ __forceinline__ void opt_moe_gate_up_dedup_body(
    const uint64_t *__restrict__ gate_table, const uint64_t *__restrict__ up_table,
    const uint32_t *__restrict__ expert_ids, const float *__restrict__ weights,
    const uint8_t *__restrict__ act_fp8, const uint8_t *__restrict__ act_scales,
    uint8_t *__restrict__ out_codes, uint8_t *__restrict__ out_scales, const uint32_t rows,
    const uint32_t experts, const float limit, OptMoePlan *__restrict__ plan_out) {
    __shared__ OptMoePlan plan;
    __shared__ float s_val[OPT_MOE_PLAN_MAX][32];
    if (rows > OPT_MOE_PLAN_MAX) {
        return;
    }
    opt_moe_plan(expert_ids, rows, experts, plan);
    if (plan_out != nullptr && blockIdx.x == 0u && blockIdx.y == 0u) {
        const uint32_t *src = reinterpret_cast<const uint32_t *>(&plan);
        uint32_t *dst = reinterpret_cast<uint32_t *>(plan_out);
        for (uint32_t i = threadIdx.x; i < sizeof(OptMoePlan) / 4u; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    const uint32_t group = blockIdx.y;
    if (group >= plan.count) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t id = plan.expert[group];
    const uint32_t first = plan.start[group];
    const uint32_t members = plan.start[group + 1u] - first;
    const uint8_t *gw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * id]));
    const uint8_t *gs = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(gate_table[2u * id + 1u]));
    const uint8_t *uw = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * id]));
    const uint8_t *us = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(up_table[2u * id + 1u]));

    const uint32_t col0 = blockIdx.x * 32u + warp * OPT_MOE_GU_COLS_PER_WARP;
    for (uint32_t c = 0u; c < OPT_MOE_GU_COLS_PER_WARP; ++c) {
        const uint32_t column = col0 + c;
        const uint4 *grow = reinterpret_cast<const uint4 *>(gw + (uint64_t)column * OPT_MOE_GU_ROW);
        const uint4 *urow = reinterpret_cast<const uint4 *>(uw + (uint64_t)column * OPT_MOE_GU_ROW);
        uint4 wg[5], wu[5];
        float sg[5], su[5];
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
            wg[k] = opt_ldg_stream(grow + lane + 32u * k);
            wu[k] = opt_ldg_stream(urow + lane + 32u * k);
        }
#pragma unroll
        for (uint32_t k = 0u; k < 5u; ++k) {
            sg[k] = spark_ue8m0_decode(__ldg(gs + (uint64_t)column * OPT_MOE_GU_SCALE_ROW + lane + 32u * k));
            su[k] = spark_ue8m0_decode(__ldg(us + (uint64_t)column * OPT_MOE_GU_SCALE_ROW + lane + 32u * k));
        }
        for (uint32_t m = 0u; m < members; ++m) {
            const uint32_t row = plan.rows[first + m];
            const uint32_t token = row / TOPK;
            const uint4 *act = reinterpret_cast<const uint4 *>(act_fp8 + (uint64_t)token * OPT_MOE_HIDDEN);
            float tg[5], tu[5];
#pragma unroll
            for (uint32_t k = 0u; k < 5u; ++k) {
                const uint32_t block = lane + 32u * k;
                const uint4 a0 = __ldg(act + 2u * block);
                const uint4 a1 = __ldg(act + 2u * block + 1u);
                const float as = spark_ue8m0_decode(act_scales[(uint64_t)token * OPT_MOE_GU_SCALE_ROW + block]);
                tg[k] = __fmul_rn(__fmul_rn(opt_block_partial(wg[k], a0, a1), as), sg[k]);
                tu[k] = __fmul_rn(__fmul_rn(opt_block_partial(wu[k], a0, a1), as), su[k]);
            }
            float accg = 0.0f, accu = 0.0f;
#pragma unroll
            for (uint32_t k = 0u; k < 5u; ++k) {
#pragma unroll
                for (uint32_t l = 0u; l < 32u; ++l) {
                    accg = __fadd_rn(accg, __shfl_sync(0xFFFFFFFFu, tg[k], l));
                    accu = __fadd_rn(accu, __shfl_sync(0xFFFFFFFFu, tu[k], l));
                }
            }
            if (lane == 0u) {
                const float g = __bfloat162float(__float2bfloat16_rn(accg));
                const float u = __bfloat162float(__float2bfloat16_rn(accu));
                const float clamped_gate = fminf(g, limit);
                const float clamped_up = fminf(fmaxf(u, -limit), limit);
                const float activated = (clamped_gate / (1.0f + __expf(-clamped_gate))) * clamped_up;
                s_val[m][warp * OPT_MOE_GU_COLS_PER_WARP + c] =
                    __bfloat162float(__float2bfloat16_rn(activated * weights[row]));
            }
        }
    }
    __syncthreads();
    for (uint32_t m = warp; m < members; m += OPT_MOE_GU_THREADS / 32u) {
        const uint32_t row = plan.rows[first + m];
        const float value = s_val[m][lane];
        float amax = fabsf(value);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
        }
        const uint8_t scale = spark_act_quant_scale_code(amax);
        if (lane == 0u) {
            out_scales[(uint64_t)row * OPT_MOE_D_KB + blockIdx.x] = scale;
        }
        const float inv_scale = 1.0f / spark_ue8m0_decode(scale);
        out_codes[(uint64_t)row * OPT_MOE_INTER + blockIdx.x * 32u + lane] =
            spark_fp8_e4m3_encode(value * inv_scale);
    }
}

#define OPT_MOE_GATE_UP_DEDUP_ENTRY(name, TOPK)                                                    \
    extern "C" __global__ void __launch_bounds__(OPT_MOE_GU_THREADS) name(                         \
        const uint64_t *__restrict__ gate_table, const uint64_t *__restrict__ up_table,           \
        const uint32_t *__restrict__ expert_ids, const float *__restrict__ weights,               \
        const uint8_t *__restrict__ act_fp8, const uint8_t *__restrict__ act_scales,              \
        uint8_t *__restrict__ out_codes, uint8_t *__restrict__ out_scales, const uint32_t rows,   \
        const uint32_t experts, const float limit, OptMoePlan *__restrict__ plan_out) {           \
        OPT_PDL_ENTRY();                                                                          \
        opt_moe_gate_up_dedup_body<TOPK>(gate_table, up_table, expert_ids, weights, act_fp8,      \
                                         act_scales, out_codes, out_scales, rows, experts, limit, \
                                         plan_out);                                               \
    }

OPT_MOE_GATE_UP_DEDUP_ENTRY(opt_moe_gate_up_dedup_sm121, OPT_MOE_TOPK)
OPT_MOE_GATE_UP_DEDUP_ENTRY(opt_moe_gate_up_dedup_k3_sm121, 3u)

// The plan comes from opt_moe_gate_up_dedup's plan_out (the launch before this one). A block serves
// TILES consecutive 32-column tiles of one distinct expert: the next tile's weights are in flight
// while the current one computes, and the members' activations are staged once (up to DD_MB at a
// time), their terms folded together. Per (row, column) the terms and the ascending 18-term fold
// are the per-row kernel's.
#define OPT_MOE_DD_MB 8u
#define OPT_MOE_DD_TILES 2u

template <uint32_t TILES>
__device__ __forceinline__ void opt_moe_down_dedup_tiles_body(
    const uint64_t *__restrict__ down_table, const OptMoePlan *__restrict__ plan,
    const uint8_t *__restrict__ h_codes, const uint8_t *__restrict__ h_scales,
    float *__restrict__ slot_val) {
    __shared__ uint4 s_h[OPT_MOE_DD_MB][OPT_MOE_INTER / 16u];
    __shared__ float s_hs[OPT_MOE_DD_MB][OPT_MOE_D_KB];
    __shared__ float s_terms[OPT_MOE_DD_MB][OPT_MOE_D_TILE][OPT_MOE_D_KB + 1u];
    const uint32_t group = blockIdx.y;
    if (group >= plan->count) {
        return;
    }
    const uint32_t tid = threadIdx.x;
    const uint32_t id = plan->expert[group];
    const uint32_t first = plan->start[group];
    const uint32_t members = plan->start[group + 1u] - first;
    const uint8_t *w = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * id]));
    const uint8_t *ws = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(down_table[2u * id + 1u]));
    const uint32_t tile0 = blockIdx.x * TILES;
    uint4 wv[3];
    float sv[3];
#pragma unroll
    for (uint32_t it = 0u; it < 3u; ++it) {
        const uint32_t q = tid + it * OPT_MOE_D_THREADS;
        if (q < OPT_MOE_D_CHUNKS) {
            wv[it] = opt_ldg_stream(reinterpret_cast<const uint4 *>(w + (uint64_t)tile0 * OPT_MOE_D_TILE * OPT_MOE_D_ROW) + q);
            sv[it] = spark_ue8m0_decode(__ldg(ws + (uint64_t)tile0 * OPT_MOE_D_TILE * OPT_MOE_D_KB + q));
        }
    }
    auto stage = [&](uint32_t m0, uint32_t mb) {
        for (uint32_t i = tid; i < mb * (OPT_MOE_INTER / 16u); i += OPT_MOE_D_THREADS) {
            const uint32_t m = i / (OPT_MOE_INTER / 16u);
            const uint32_t j = i % (OPT_MOE_INTER / 16u);
            const uint32_t row = plan->rows[first + m0 + m];
            s_h[m][j] = reinterpret_cast<const uint4 *>(h_codes + (uint64_t)row * OPT_MOE_INTER)[j];
        }
        for (uint32_t i = tid; i < mb * OPT_MOE_D_KB; i += OPT_MOE_D_THREADS) {
            const uint32_t m = i / OPT_MOE_D_KB;
            const uint32_t b = i % OPT_MOE_D_KB;
            const uint32_t row = plan->rows[first + m0 + m];
            s_hs[m][b] = spark_ue8m0_decode(h_scales[(uint64_t)row * OPT_MOE_D_KB + b]);
        }
    };
    const bool once = members <= OPT_MOE_DD_MB;
    if (once) {
        stage(0u, members);
    }
    __syncthreads();
    for (uint32_t t = 0u; t < TILES; ++t) {
        const uint32_t tile = tile0 + t;
        uint4 nv[3];
        float ns[3];
        if (t + 1u < TILES) {
#pragma unroll
            for (uint32_t it = 0u; it < 3u; ++it) {
                const uint32_t q = tid + it * OPT_MOE_D_THREADS;
                if (q < OPT_MOE_D_CHUNKS) {
                    nv[it] = opt_ldg_stream(reinterpret_cast<const uint4 *>(w + (uint64_t)(tile + 1u) * OPT_MOE_D_TILE * OPT_MOE_D_ROW) + q);
                    ns[it] = spark_ue8m0_decode(__ldg(ws + (uint64_t)(tile + 1u) * OPT_MOE_D_TILE * OPT_MOE_D_KB + q));
                }
            }
        }
        for (uint32_t m0 = 0u; m0 < members; m0 += OPT_MOE_DD_MB) {
            const uint32_t mb = members - m0 < OPT_MOE_DD_MB ? members - m0 : OPT_MOE_DD_MB;
            if (!once) {
                stage(m0, mb);
                __syncthreads();
            }
            for (uint32_t m = 0u; m < mb; ++m) {
#pragma unroll
                for (uint32_t it = 0u; it < 3u; ++it) {
                    const uint32_t q = tid + it * OPT_MOE_D_THREADS;
                    if (q < OPT_MOE_D_CHUNKS) {
                        const uint32_t col = q / OPT_MOE_D_KB;
                        const uint32_t blk = q % OPT_MOE_D_KB;
                        const float partial = opt_block_partial(wv[it], s_h[m][2u * blk], s_h[m][2u * blk + 1u]);
                        s_terms[m][col][blk] = __fmul_rn(__fmul_rn(partial, s_hs[m][blk]), sv[it]);
                    }
                }
            }
            __syncthreads();
            if (tid < mb * OPT_MOE_D_TILE) {
                const uint32_t m = tid / OPT_MOE_D_TILE;
                const uint32_t col = tid % OPT_MOE_D_TILE;
                float acc = 0.0f;
#pragma unroll
                for (uint32_t b = 0u; b < OPT_MOE_D_KB; ++b) {
                    acc = __fadd_rn(acc, s_terms[m][col][b]);
                }
                const uint32_t row = plan->rows[first + m0 + m];
                slot_val[(uint64_t)row * OPT_MOE_HIDDEN + tile * OPT_MOE_D_TILE + col] = __fadd_rn(0.0f, acc);
            }
            __syncthreads();
        }
        if (t + 1u < TILES) {
#pragma unroll
            for (uint32_t it = 0u; it < 3u; ++it) {
                wv[it] = nv[it];
                sv[it] = ns[it];
            }
        }
    }
}

#define OPT_MOE_DOWN_TILES_ENTRY(name, TILES)                                                      \
    extern "C" __global__ void __launch_bounds__(OPT_MOE_D_THREADS, 3) name(                       \
        const uint64_t *__restrict__ down_table, const OptMoePlan *__restrict__ plan,             \
        const uint8_t *__restrict__ h_codes, const uint8_t *__restrict__ h_scales,                \
        float *__restrict__ slot_val) {                                                           \
        OPT_PDL_ENTRY();                                                                          \
        opt_moe_down_dedup_tiles_body<TILES>(down_table, plan, h_codes, h_scales, slot_val);      \
    }

OPT_MOE_DOWN_TILES_ENTRY(opt_moe_down_dedup_sm121, OPT_MOE_DD_TILES)

template <uint32_t TOPK>
__device__ __forceinline__ void opt_moe_combine_body(
    const uint32_t *__restrict__ expert_ids, const float *__restrict__ slot_val,
    float *__restrict__ delta_f32, uint16_t *__restrict__ delta_bf16, const uint32_t tokens,
    const float *addend) {
    const uint32_t token = blockIdx.y;
    if (token >= tokens) {
        return;
    }
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= OPT_MOE_HIDDEN) {
        return;
    }
    uint32_t ids[TOPK];
#pragma unroll
    for (uint32_t s = 0u; s < TOPK; ++s) {
        ids[s] = expert_ids[token * TOPK + s];
    }
    uint32_t used = 0u;
    float total = 0.0f;
#pragma unroll
    for (uint32_t position = 0u; position < TOPK; ++position) {
        uint32_t best = TOPK;
#pragma unroll
        for (uint32_t slot = 0u; slot < TOPK; ++slot) {
            if (((used >> slot) & 1u) != 0u) {
                continue;
            }
            if (best == TOPK || ids[slot] < ids[best]) {
                best = slot;
            }
        }
        used |= 1u << best;
        const float value = slot_val[(uint64_t)(token * TOPK + best) * OPT_MOE_HIDDEN + column];
        total = __fadd_rn(total, __uint_as_float(opt_round_slot_bits(value) << 16));
    }
    const uint64_t out = (uint64_t)token * OPT_MOE_HIDDEN + column;
    if (addend != nullptr) {
        total = __fadd_rn(total, addend[out]);
    }
    delta_f32[out] = total;
    delta_bf16[out] = opt_narrow_bits(total);
}

#define OPT_MOE_COMBINE_ENTRY(name, TOPK)                                                          \
    extern "C" __global__ void __launch_bounds__(256) name(                                        \
        const uint32_t *__restrict__ expert_ids, const float *__restrict__ slot_val,              \
        float *__restrict__ delta_f32, uint16_t *__restrict__ delta_bf16, const uint32_t tokens,  \
        const float *addend) {                                                                    \
        OPT_PDL_ENTRY();                                                                          \
        opt_moe_combine_body<TOPK>(expert_ids, slot_val, delta_f32, delta_bf16, tokens, addend);  \
    }

OPT_MOE_COMBINE_ENTRY(opt_moe_combine_sm121, OPT_MOE_TOPK)
OPT_MOE_COMBINE_ENTRY(opt_moe_combine_k3_sm121, 3u)
