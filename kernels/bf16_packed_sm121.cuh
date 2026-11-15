// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The packed BF16 form: a lossless layout of a BF16 matrix in about 12.25 bits a weight, whose
// words a warp rebuilds in registers and hands to the BF16 row walk's own sums. A packed GEMV is
// the BF16 GEMV's bits from three quarters of the bytes, and a decode step's dense projections
// are bound by those bytes.
//
// The layout follows the walk. q4e_row_dot_bf16 has lane `lane` read 16-byte chunks lane,
// lane + 32, ... in steps: quads of four chunks (c, c + 32, c + 64, c + 96), then perhaps a pair,
// then single chunks. A row whose columns are a multiple of 256 gives every lane the same steps.
// For each (step, lane) the form keeps
//   - plane A: each weight's sign and mantissa byte (sign << 7 | the low seven bits),
//   - plane B: each weight's exponent as a 4-bit offset under the step's largest, two a byte
//     (byte k % 4 of a chunk's word holds weight k % 4 low and weight k % 4 + 4 high),
//   - plane C: that largest exponent, one byte.
// A row is [A: columns bytes][B: columns / 2][C: 32 a step], padded to 16 bytes. A weight 15 or
// more binades under its step's largest is an escape: its offset reads 15 and its whole word sits
// in the matrix's escape list, `column << 16 | word`, the row's run of it named by
// offsets[row]..offsets[row + 1]. Real projections escape a few weights a row (zeros, mostly).
//
// Included by kernels/qwen4_exp_sm121.cu after q4e_dot_chunk, whose expressions the walks below
// repeat word for word; the host side (geometry, packing order, the round-trip check that
// refuses a matrix the form does not hold) is crates/sparkflash-model-inkling2/src/device/packed.rs.

#pragma once

// The offset that marks an escape.
#define Q4E_PACKED_ESCAPE 15u

// One packed row: its planes and its escapes.
struct Q4ePackedRow {
    const unsigned char* a;
    const unsigned char* b;
    const unsigned char* c;
    const unsigned int* entries;
    unsigned int begin;
    unsigned int end;
};

__device__ __forceinline__ Q4ePackedRow q4e_packed_row(const unsigned char* __restrict__ packed,
                                                       const unsigned int* __restrict__ offsets,
                                                       const unsigned int* __restrict__ entries,
                                                       int row, int columns,
                                                       unsigned long long row_bytes) {
    Q4ePackedRow out;
    out.a = packed + (unsigned long long)row * row_bytes;
    out.b = out.a + columns;
    out.c = out.b + columns / 2;
    out.entries = entries;
    out.begin = __ldg(offsets + row);
    out.end = __ldg(offsets + row + 1);
    return out;
}

// The eight BF16 words of one chunk from its plane-A bytes, its plane-B word and the step's top
// exponent in every byte of `top4`. An escape's word comes out wrong here and is patched after.
__device__ __forceinline__ uint4 q4e_packed_chunk(uint2 mant, unsigned int nib,
                                                  unsigned int top4) {
    const unsigned int e_lo = __vsub4(top4, nib & 0x0F0F0F0Fu);
    const unsigned int e_hi = __vsub4(top4, (nib >> 4) & 0x0F0F0F0Fu);
    const unsigned int h0 = (mant.x & 0x80808080u) | ((e_lo >> 1) & 0x7F7F7F7Fu);
    const unsigned int l0 = (mant.x & 0x7F7F7F7Fu) | ((e_lo << 7) & 0x80808080u);
    const unsigned int h1 = (mant.y & 0x80808080u) | ((e_hi >> 1) & 0x7F7F7F7Fu);
    const unsigned int l1 = (mant.y & 0x7F7F7F7Fu) | ((e_hi << 7) & 0x80808080u);
    return make_uint4(__byte_perm(l0, h0, 0x5140), __byte_perm(l0, h0, 0x7362),
                      __byte_perm(l1, h1, 0x5140), __byte_perm(l1, h1, 0x7362));
}

// Put back the stored word of every escape that falls in a step's chunks first + 32 j. The
// chunk and word indices are compile-time constants after unrolling, so the words stay in
// registers (a dynamic index sends them to local memory).
template <int N>
__device__ __forceinline__ void q4e_packed_patch(uint4 (&value)[N], int first,
                                                 const Q4ePackedRow& row) {
    for (unsigned int i = row.begin; i < row.end; ++i) {
        const unsigned int entry = __ldg(row.entries + i);
        const int column = (int)(entry >> 16);
        const int delta = (column >> 3) - first;
        if (delta < 0 || (delta & 31) != 0 || (delta >> 5) >= N) {
            continue;
        }
        const int j = delta >> 5;
        const int k = column & 7;
        const unsigned int word = entry & 0xFFFFu;
        const unsigned int keep = (k & 1) ? 0x0000FFFFu : 0xFFFF0000u;
        const unsigned int put = (k & 1) ? (word << 16) : word;
#pragma unroll
        for (int jj = 0; jj < N; ++jj) {
            if (jj == j) {
                if ((k >> 1) == 0) value[jj].x = (value[jj].x & keep) | put;
                if ((k >> 1) == 1) value[jj].y = (value[jj].y & keep) | put;
                if ((k >> 1) == 2) value[jj].z = (value[jj].z & keep) | put;
                if ((k >> 1) == 3) value[jj].w = (value[jj].w & keep) | put;
            }
        }
    }
}

// The N chunks lane `lane` reads at step `step` of a row, `first` its first chunk and `a_off`
// the step's offset into plane A (plane B's is half of it).
template <int N>
__device__ __forceinline__ void q4e_packed_step(const Q4ePackedRow& row, unsigned long long a_off,
                                                int step, int lane, int first,
                                                uint4 (&value)[N]) {
    const unsigned int top4 = (unsigned int)__ldg(row.c + step * 32 + lane) * 0x01010101u;
    if constexpr (N == 4) {
        const uint4 m01 = q4e_ldg(reinterpret_cast<const uint4*>(row.a + a_off + lane * 32));
        const uint4 m23 = q4e_ldg(reinterpret_cast<const uint4*>(row.a + a_off + lane * 32 + 16));
        const uint4 nb = q4e_ldg(reinterpret_cast<const uint4*>(row.b + a_off / 2 + lane * 16));
        value[0] = q4e_packed_chunk(make_uint2(m01.x, m01.y), nb.x, top4);
        value[1] = q4e_packed_chunk(make_uint2(m01.z, m01.w), nb.y, top4);
        value[2] = q4e_packed_chunk(make_uint2(m23.x, m23.y), nb.z, top4);
        value[3] = q4e_packed_chunk(make_uint2(m23.z, m23.w), nb.w, top4);
    } else if constexpr (N == 2) {
        const uint4 m01 = q4e_ldg(reinterpret_cast<const uint4*>(row.a + a_off + lane * 16));
        const uint2 nb = __ldg(reinterpret_cast<const uint2*>(row.b + a_off / 2 + lane * 8));
        value[0] = q4e_packed_chunk(make_uint2(m01.x, m01.y), nb.x, top4);
        value[1] = q4e_packed_chunk(make_uint2(m01.z, m01.w), nb.y, top4);
    } else {
        const uint2 m = __ldg(reinterpret_cast<const uint2*>(row.a + a_off + lane * 8));
        const unsigned int nb =
            __ldg(reinterpret_cast<const unsigned int*>(row.b + a_off / 2 + lane * 4));
        value[0] = q4e_packed_chunk(m, nb, top4);
    }
    if (row.begin != row.end) {
        q4e_packed_patch<N>(value, first, row);
    }
}

// q4e_row_dot_bf16's walk over R packed rows in step, each step's words rebuilt and handed to
// `visit`: visit.quad(v, chunk) with uint4 v[R][4], visit.pair(v, chunk) with v[R][2], and
// visit.single(v, chunk) with v[R][1]. `columns` is a multiple of 256.
template <int R, class Visit>
__device__ __forceinline__ void q4e_packed_walk(const Q4ePackedRow (&rows)[R], int columns,
                                                int lane, Visit& visit) {
    const int chunks = columns >> 3;
    int chunk = lane;
    int step = 0;
    unsigned long long a_off = 0;
    for (; chunk + 96 < chunks; chunk += 128, ++step, a_off += 32 * 32) {
        uint4 v[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q4e_packed_step<4>(rows[r], a_off, step, lane, chunk, v[r]);
        }
        visit.quad(v, chunk);
    }
    if (chunk + 32 < chunks) {
        uint4 v[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q4e_packed_step<2>(rows[r], a_off, step, lane, chunk, v[r]);
        }
        visit.pair(v, chunk);
        chunk += 64;
        ++step;
        a_off += 32 * 16;
    }
    for (; chunk < chunks; chunk += 32, ++step, a_off += 32 * 8) {
        uint4 v[R][1];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q4e_packed_step<1>(rows[r], a_off, step, lane, chunk, v[r]);
        }
        visit.single(v, chunk);
    }
}

// One row against one vector: q4e_row_dot_bf16's 16-byte walk and its grouping.
struct Q4ePackedDot {
    const float* __restrict__ x;
    float total;
    __device__ __forceinline__ void quad(const uint4 (&v)[1][4], int chunk) {
        total += q4e_dot_chunk(v[0][0], x, chunk * 8) + q4e_dot_chunk(v[0][1], x, (chunk + 32) * 8) +
                 q4e_dot_chunk(v[0][2], x, (chunk + 64) * 8) +
                 q4e_dot_chunk(v[0][3], x, (chunk + 96) * 8);
    }
    __device__ __forceinline__ void pair(const uint4 (&v)[1][2], int chunk) {
        total += q4e_dot_chunk(v[0][0], x, chunk * 8);
        total += q4e_dot_chunk(v[0][1], x, (chunk + 32) * 8);
    }
    __device__ __forceinline__ void single(const uint4 (&v)[1][1], int chunk) {
        total += q4e_dot_chunk(v[0][0], x, chunk * 8);
    }
};

// One warp's partial dot of a packed row against an f32 vector (16-byte aligned); reduce with
// q4e_warp_sum. The bits of q4e_row_dot_bf16 over the row's BF16 words.
__device__ __forceinline__ float q4e_packed_row_dot(const Q4ePackedRow& row,
                                                    const float* __restrict__ x, int columns,
                                                    int lane) {
    const Q4ePackedRow rows[1] = {row};
    Q4ePackedDot dot{x, 0.0f};
    q4e_packed_walk<1>(rows, columns, lane, dot);
    return dot.total;
}

// R rows against `batch` vectors: q4e_rows_dot_bf16_batch's grouping, so a (row, token) sum is
// the one-row walk's bits.
template <int R>
struct Q4ePackedBatch {
    const float* __restrict__ x;
    int columns;
    int batch;
    float (*totals)[Q4E_MAX_BATCH];
    __device__ __forceinline__ void quad(const uint4 (&v)[R][4], int chunk) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    totals[r][t] += q4e_dot_chunk(v[r][0], xt, chunk * 8) +
                                    q4e_dot_chunk(v[r][1], xt, (chunk + 32) * 8) +
                                    q4e_dot_chunk(v[r][2], xt, (chunk + 64) * 8) +
                                    q4e_dot_chunk(v[r][3], xt, (chunk + 96) * 8);
                }
            }
        }
    }
    __device__ __forceinline__ void pair(const uint4 (&v)[R][2], int chunk) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    totals[r][t] += q4e_dot_chunk(v[r][0], xt, chunk * 8);
                    totals[r][t] += q4e_dot_chunk(v[r][1], xt, (chunk + 32) * 8);
                }
            }
        }
    }
    __device__ __forceinline__ void single(const uint4 (&v)[R][1], int chunk) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            if (t < batch) {
                const float* xt = x + (unsigned long long)t * columns;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    totals[r][t] += q4e_dot_chunk(v[r][0], xt, chunk * 8);
                }
            }
        }
    }
};

template <int R>
__device__ __forceinline__ void q4e_packed_rows_dot_batch(const Q4ePackedRow (&rows)[R],
                                                          const float* __restrict__ x,
                                                          int columns, int lane, int batch,
                                                          float (*totals)[Q4E_MAX_BATCH]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Q4E_MAX_BATCH; ++t) {
            totals[r][t] = 0.0f;
        }
    }
    Q4ePackedBatch<R> dot{x, columns, batch, totals};
    q4e_packed_walk<R>(rows, columns, lane, dot);
}

// The rebuilt words, chunk by chunk: `out` receives (chunk, word) for every chunk of the row.
struct Q4ePackedWords {
    uint4* __restrict__ out;
    const uint4* __restrict__ expected;
    unsigned int differ;
    __device__ __forceinline__ void put(const uint4 value, int chunk) {
        if (out != nullptr) {
            out[chunk] = value;
        }
        if (expected != nullptr) {
            const uint4 want = expected[chunk];
            const unsigned int words[4] = {value.x ^ want.x, value.y ^ want.y, value.z ^ want.z,
                                           value.w ^ want.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                differ += ((words[i] & 0xFFFFu) != 0) + ((words[i] >> 16) != 0);
            }
        }
    }
    __device__ __forceinline__ void quad(const uint4 (&v)[1][4], int chunk) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            put(v[0][j], chunk + 32 * j);
        }
    }
    __device__ __forceinline__ void pair(const uint4 (&v)[1][2], int chunk) {
        put(v[0][0], chunk);
        put(v[0][1], chunk + 32);
    }
    __device__ __forceinline__ void single(const uint4 (&v)[1][1], int chunk) {
        put(v[0][0], chunk);
    }
};
