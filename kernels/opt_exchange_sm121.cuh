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

// kernels/opt_exchange_sm121.cuh
//
// The direct-RDMA exchange's two device kernels, in one header so every image that runs the
// exchange compiles the same source: kernels/opt_misc_sm121.cu (the DeepSeek arm, which loads it
// from the cubin directory) and kernels/inkling_sm121.cu (the Inkling image) both include it. The
// host side is sparkflash_kernels::write_exchange over sparkflash_comm::rdma::WriteExchange.
#pragma once

#include <stdint.h>

#include "opt_pdl_sm121.cuh"

// opt_exchange_stage_bf16: the direct-RDMA exchange's device side, ahead of the fold. The host
//   launches it and opt_sum_ranks_bf16 back to back and waits for neither: this kernel does the
//   waiting, and the fold starts when this grid completes. `dst` is this rank's row in the round's
//   row set of the pinned, host-mapped slab (its device alias); `control` is the slab's control
//   words: per rail an arrival flag per rank, then the doorbell, then the abort word.
//
//   1. Copy this rank's row into the slab and ring the doorbell with seq. The host's pump is
//      watching that word; it posts the row to every peer and the flag behind it. The system
//      fence orders the row's stores before the doorbell for a reader outside the device, which
//      the NIC is. It costs about 6 us a round and it is not optional: with __threadfence() or no
//      fence in its place the NIC sends rows the stores have not reached yet, and the served
//      tokens change (measured: every c1 run of 1536 tokens came back different).
//   2. Wait until every peer's flag on every rail has reached seq, so the rows the fold reads
//      are this round's: a row travels in one part per rail, each with its own flag behind it.
//      The comparison is >= and never ==: a peer may already be a round ahead. The wait also ends
//      when the abort word is set: the host has given the round up, the fold behind this runs on
//      whatever is there, and the host reports the round once the device has drained.
//
//   One block, so "every byte copied" is a __syncthreads() rather than a grid barrier, and one
//   warp does the waiting, a lane to a flag. A look at every flag is then one load, where a
//   single thread reading them in turn made one after another out of host-mapped memory; that
//   measured 0.1 ms off a c1 step (docs/rdma-exchange.md). One warp is also deliberate in
//   the other direction: a waiter per block of the fold was measured, and that many polls of
//   host-mapped memory starve the copy they are waiting behind. The abort word is looked at
//   every 256th time; looking every time measured no different.
//   `world * rails` must be at most 32, which the caller checks.
//
//   The row is any bytes: a layer's bf16 rows, or one rank's slot of a gather (an argmax's few
//   words, an embedding row). The sixteen-byte copy needs both ends aligned for it, which a
//   layer's rows always are; a slot that is not goes a byte at a time, and it is small.
// The abort word's "a peer's row is not the length this round sent" tag, in the low two bytes.
// Bits 16-23 name the first lane that disagreed, which is one flag: peer = lane % world.
#define OPT_EXCHANGE_ABORT_LENGTH 0x4c45ull
#define OPT_EXCHANGE_ABORT_PEER_SHIFT 16

// Where a round's published word puts its sequence number; the low bits carry its row length.
// These must match LENGTH_SEQ_SHIFT in crates/sparkflash-comm/src/rdma/write_exchange.rs, which a
// unit test there checks by reading this file.
#define OPT_EXCHANGE_LENGTH_SEQ_SHIFT 20
#define OPT_EXCHANGE_LENGTH_BYTES_MASK ((1ull << OPT_EXCHANGE_LENGTH_SEQ_SHIFT) - 1ull)

extern "C" __global__ void __launch_bounds__(1024) opt_exchange_stage_bf16(
    const unsigned char *__restrict__ src, unsigned char *__restrict__ dst,
    const uint32_t bytes, volatile unsigned long long *control,
    const uint32_t rank, const uint32_t world, const unsigned long long seq,
    const uint32_t rails, const unsigned long long *lengths, const uint32_t lring) {
    OPT_PDL_ENTRY();
    const bool wide = ((reinterpret_cast<unsigned long long>(src) |
                        reinterpret_cast<unsigned long long>(dst)) & 15ull) == 0ull;
    const uint32_t chunks = wide ? bytes >> 4 : 0u;
    for (uint32_t i = threadIdx.x; i < chunks; i += blockDim.x) {
        reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
    }
    for (uint32_t i = (chunks << 4) + threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
    // Every thread that stored fences at system scope, then the block synchronises, and only then
    // does one lane ring the doorbell. The order matters: __syncthreads() is a *block*-scope
    // barrier, and a fence executed by one lane orders that lane's accesses, so a single fence
    // after the barrier is a release whose scope does not match its writers. This is the handoff
    // the whole module rests on - a peer that can see the doorbell rereads the row - and it is not
    // the place to rely on an argument about cumulativity that the memory model does not make.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x >= 32u) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t flags = world * rails;
    if (lane == 0u) {
        control[flags] = seq;
    }
    const bool watches = lane < flags && lane % world != rank;
    for (uint32_t look = 0u;; ++look) {
        const bool here = !watches || control[lane] >= seq;
        if (__all_sync(0xffffffffu, here)) {
            // Every flag has arrived. The flag says a row arrived; the length word says it is the
            // row this fold is about to read. Both ends derive the write offset and the fold's
            // addressing from their own row size, so a disagreement would be read in bounds and
            // silently summed. It is checked here and ends the round instead.
            bool same = true;
            if (watches && lring != 0u) {
                const uint32_t peer = lane % world;
                const volatile unsigned long long *published =
                    (const volatile unsigned long long *)&lengths[(uint64_t)peer * lring +
                                                                  (seq % (unsigned long long)lring)];
                // The word carries the publisher's sequence number in the high bits and its row
                // length in the low twenty. The size alone cannot tell two rounds apart: a rank
                // that inserts or drops a round of the same size shifts every later sequence and
                // the size check agrees. The sequence turns that into a refusal.
                const unsigned long long word = *published;
                same = (word >> OPT_EXCHANGE_LENGTH_SEQ_SHIFT) == seq &&
                       (word & OPT_EXCHANGE_LENGTH_BYTES_MASK) == (unsigned long long)bytes;
            }
            const unsigned bad = __ballot_sync(0xffffffffu, !same);
            if (bad == 0u) break;
            // Name the peer in the abort word so the host's report says which one. The peer goes
            // in bits 16-23: the tag itself occupies the low two bytes, and packing the peer at bit
            // 8 made the word unreadable for every peer but 0.
            if (lane == 0u) {
                const uint32_t first = __ffs((int)bad) - 1;
                control[flags + 1u] =
                    OPT_EXCHANGE_ABORT_LENGTH |
                    ((unsigned long long)(first % world) << OPT_EXCHANGE_ABORT_PEER_SHIFT);
            }
            break;
        }
        bool stop = false;
        if (lane == 0u && (look & 255u) == 255u) stop = control[flags + 1u] != 0ull;
        if (__any_sync(0xffffffffu, stop)) break;
    }
}

// opt_exchange_collect: a slot exchange's second half. After opt_exchange_stage_bf16 the round's
//   row set in the slab holds every rank's slot at rank * bytes, which is the layout of the
//   caller's gather buffer, so this copies the peers' slots across and leaves this rank's own,
//   which the buffer already holds (it was the source). Bits are moved and nothing is summed.
//   `unit` is 16 when both buffers and the slot size are aligned for the sixteen-byte copy and 1
//   otherwise; the caller decides, because it also sizes the grid: ceil(world * bytes / unit / 256)
//   blocks of 256.
extern "C" __global__ void __launch_bounds__(256) opt_exchange_collect(
    const unsigned char *__restrict__ set, unsigned char *__restrict__ slots,
    const uint32_t bytes, const uint32_t rank, const uint32_t world, const uint32_t unit) {
    OPT_PDL_ENTRY();
    const uint32_t per = bytes / unit;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)world * per) return;
    if ((uint32_t)(i / per) == rank) return;
    // __ldcg for the same reason the fold uses it: `set` is the slab, and the peer slots in it are
    // bytes this rank's NIC wrote. A plain load may be served from an L1 the NIC does not
    // invalidate, and the same row set is reused every second round.
    if (unit == 16u) {
        reinterpret_cast<uint4 *>(slots)[i] =
            __ldcg(reinterpret_cast<const uint4 *>(set) + i);
    } else {
        slots[i] = __ldcg(set + i);
    }
}
