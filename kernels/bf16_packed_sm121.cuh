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
// words a warp rebuilds in registers and hands to a BF16 walk's own sums. A packed GEMV is the
// BF16 GEMV's bits from three quarters of the bytes, and a decode step's dense projections are
// bound by those bytes. Two kinds of walk read it, each from a layout of its own:
// q4e_row_dot_bf16's (q4e_gemv_packed, the Inkling, Dory and dense arms) from the row walk's,
// and the tiled readers (q4e_tiled_gemv with Q4eTgPacked, GLM's projections; Qwen's few-row
// tensor-core tile) from the tiled one below.
//
// The row walk's layout follows the row walk. q4e_row_dot_bf16 has lane `lane` read 16-byte chunks lane,
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
// A tiled reader - one that takes a row's chunks 32 at a time, lane l chunk 32 r + l, as the
// tiled GEMV body and the few-row tensor-core tile do - finds the row walk's layout strided: a
// quad step keeps lane l's four chunks together, so a round of the reader loads 8 bytes of every
// 32. The tiled layout keeps the same steps, tops and escapes and orders a step's plane bytes by
// chunk instead: chunk j of lane l's n in the step (chunk 32 j + l of it) at plane-A offset
// 256 j + 8 l and plane-B offset 128 j + 4 l, so a round's 32 chunks are 256 contiguous bytes of
// plane A and 128 of plane B. Same bytes, same bits; a matrix is packed in one layout or the
// other, and only its own kind of reader takes it (crates/sparkflash-ops/src/packed.rs refuses
// the rest).
//
// The walks repeat q4e_dot_chunk's and q4e_tg_fma8's expressions word for word (both in
// q4e_gemv_sm121.cuh); the host side (geometry, the arena, the round-trip check that refuses a
// matrix the form does not hold) is crates/sparkflash-ops/src/packed.rs.

#pragma once

#include "q4e_gemv_sm121.cuh"

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
template <int N, bool kTiled = false>
__device__ __forceinline__ void q4e_packed_step(const Q4ePackedRow& row, unsigned long long a_off,
                                                int step, int lane, int first,
                                                uint4 (&value)[N]) {
    const unsigned int top4 = (unsigned int)__ldg(row.c + step * 32 + lane) * 0x01010101u;
    if constexpr (kTiled) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const uint2 m = __ldg(reinterpret_cast<const uint2*>(row.a + a_off + j * 256 + lane * 8));
            const unsigned int nb = __ldg(
                reinterpret_cast<const unsigned int*>(row.b + a_off / 2 + j * 128 + lane * 4));
            value[j] = q4e_packed_chunk(m, nb, top4);
        }
    } else if constexpr (N == 4) {
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
template <int R, class Visit, bool kTiled = false>
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
            q4e_packed_step<4, kTiled>(rows[r], a_off, step, lane, chunk, v[r]);
        }
        visit.quad(v, chunk);
    }
    if (chunk + 32 < chunks) {
        uint4 v[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q4e_packed_step<2, kTiled>(rows[r], a_off, step, lane, chunk, v[r]);
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
            q4e_packed_step<1, kTiled>(rows[r], a_off, step, lane, chunk, v[r]);
        }
        visit.single(v, chunk);
    }
}

// Where chunk `chunk` of a row of `chunks` chunks (a multiple of 32) sits in the row walk's
// layout: its step, the step's chunks a lane `n` (4, 2 or 1), its index among them `j` and the
// step's offset into plane A (plane B's is half of it). Quads cover the first 128 * quads, then a
// pair takes 64 when more than 32 are left, then single chunks; the lane is chunk % 32 in every
// step, and a step holds 256 n bytes of plane A.
struct Q4ePackedAt {
    int step;
    int n;
    int j;
    int a_off;
};

// Chunk `at`'s plane-A and plane-B byte offsets in the tiled layout, `lane` its chunk % 32.
__device__ __forceinline__ int q4e_packed_tiled_a(const Q4ePackedAt& at, int lane) {
    return at.a_off + at.j * 256 + lane * 8;
}
__device__ __forceinline__ int q4e_packed_tiled_b(const Q4ePackedAt& at, int lane) {
    return at.a_off / 2 + at.j * 128 + lane * 4;
}

__device__ __forceinline__ Q4ePackedAt q4e_packed_at(int chunk, int chunks) {
    const int quads = chunks >> 7;
    if (chunk < 128 * quads) {
        return Q4ePackedAt{chunk >> 7, 4, (chunk >> 5) & 3, (chunk >> 7) * 1024};
    }
    const int rest = chunk - 128 * quads;
    const bool pair = chunks - 128 * quads > 32;
    if (pair && rest < 64) {
        return Q4ePackedAt{quads, 2, rest >> 5, quads * 1024};
    }
    const int single = pair ? (rest - 64) >> 5 : rest >> 5;
    return Q4ePackedAt{quads + (pair ? 1 : 0) + single, 1, 0,
                       quads * 1024 + (pair ? 512 : 0) + single * 256};
}

// A packed row for the tiled body: its plane-A bytes (null past the matrix) and its run of the
// escape list. The rest of the row is the matrix's, in the body's `row_of` (Q4eTgPackedRows),
// so four rows hold four registers each.
struct Q4eTgPackedRow {
    const unsigned char* a;
    unsigned int begin;
    unsigned int end;
};

// The tiled body's format over the packed form in the tiled layout: lane l of round r loads
// chunk 32 r + l's plane bytes (a round's are contiguous) and its (step, lane)'s top (`column` =
// 8 * that chunk), rebuilds the eight words and patches its escapes in `decode`, after the
// round's staging, and adds them as Q4eTgBf16 does, so a sum is the BF16 tiled GEMV's bits. The
// loaded uint4 holds the plane-A bytes, the plane-B word and the top.
struct Q4eTgPacked {
    static constexpr int kColumns = 8;
    static constexpr bool kBlockScaled = false;
    __device__ __forceinline__ static bool present(const Q4eTgPackedRow& row) {
        return row.a != nullptr;
    }
    template <typename RowOf>
    __device__ __forceinline__ static uint4 load(const RowOf& of, const Q4eTgPackedRow& row,
                                                 int column) {
        const int chunk = column >> 3;
        const int lane = chunk & 31;
        const int columns = of.columns;
        const Q4ePackedAt at = q4e_packed_at(chunk, columns >> 3);
        const unsigned char* b = row.a + columns;
        const unsigned char* c = b + columns / 2;
        const uint2 mant =
            __ldg(reinterpret_cast<const uint2*>(row.a + q4e_packed_tiled_a(at, lane)));
        const unsigned int nib =
            __ldg(reinterpret_cast<const unsigned int*>(b + q4e_packed_tiled_b(at, lane)));
        const unsigned int top = (unsigned int)__ldg(c + at.step * 32 + lane);
        return make_uint4(mant.x, mant.y, nib, top);
    }
    template <typename RowOf>
    __device__ __forceinline__ static uint4 decode(const RowOf& of, const Q4eTgPackedRow& row,
                                                   int column, uint4 loaded) {
        uint4 value[1] = {
            q4e_packed_chunk(make_uint2(loaded.x, loaded.y), loaded.z, loaded.w * 0x01010101u)};
        if (row.begin != row.end) {
            Q4ePackedRow escapes;
            escapes.entries = of.entries;
            escapes.begin = row.begin;
            escapes.end = row.end;
            q4e_packed_patch<1>(value, column >> 3, escapes);
        }
        return value[0];
    }
    __device__ __forceinline__ static float fma(const uint4 word4, const float4* x, float acc) {
        return Q4eTgBf16::fma(word4, x, acc);
    }
};

// Rows tile * 32 + warp * 4 + r of a packed [rows, columns] matrix.
struct Q4eTgPackedRows {
    const unsigned char* w;
    const unsigned int* offsets;
    const unsigned int* entries;
    int rows;
    int columns;
    unsigned long long row_bytes;
    __device__ __forceinline__ Q4eTgPackedRow operator()(int tile, int warp, int r) const {
        const int row = tile * Q4E_TG_TILE + warp * Q4E_TG_ROWS + r;
        if (row >= rows) {
            return Q4eTgPackedRow{nullptr, 0u, 0u};
        }
        return Q4eTgPackedRow{w + (unsigned long long)row * row_bytes, __ldg(offsets + row),
                              __ldg(offsets + row + 1)};
    }
};

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

// R rows against `batch` vectors (at most T): q4e_rows_dot_bf16_batch's grouping, so a (row,
// token) sum is the one-row walk's bits whichever tokens share the walk.
template <int R, int T = Q4E_MAX_BATCH>
struct Q4ePackedBatch {
    const float* __restrict__ x;
    int columns;
    int batch;
    float (*totals)[T];
    __device__ __forceinline__ void quad(const uint4 (&v)[R][4], int chunk) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
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
        for (int t = 0; t < T; ++t) {
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
        for (int t = 0; t < T; ++t) {
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

template <int R, int T = Q4E_MAX_BATCH>
__device__ __forceinline__ void q4e_packed_rows_dot_batch(const Q4ePackedRow (&rows)[R],
                                                          const float* __restrict__ x,
                                                          int columns, int lane, int batch,
                                                          float (*totals)[T]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            totals[r][t] = 0.0f;
        }
    }
    Q4ePackedBatch<R, T> dot{x, columns, batch, totals};
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
