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

// L2 prefetch hints any pack's kernels can issue.
//
// A bulk prefetch asks the memory system for a whole span in one instruction, so a block that knows
// which weights it will read can have DRAM fetching them while it does its own set-up (a lookup
// table, an activation staging, a barrier) instead of starting each load after the set-up and
// waiting out its latency. Nothing waits on a prefetch and it changes no value.
//
// Keep this header self-contained (no shared-memory declarations, no kernel symbols). The V4.1
// opt_rows_sm121.cu keeps its in-file copy until its sha256-pinned baseline is next refreshed
// (crates/sparkflash-kernels/baseline/kernel_baseline.txt).

#ifndef SPARKFLASH_PREFETCH_SM121_CUH
#define SPARKFLASH_PREFETCH_SM121_CUH

#include <cstdint>

// The largest span one bulk prefetch is issued for; longer spans take several.
#define SPARK_PREFETCH_CHUNK 32768u

// Prefetch `bytes` from `at` into L2 from the calling thread, rounded inward to 16 bytes (the bulk
// prefetch's granule) so no byte outside the span is touched.
__device__ __forceinline__ void spark_prefetch_l2(const void* at, uint64_t bytes) {
#ifndef SPARKFLASH_CUDA_SYNTAX_STUB
    const uint64_t start = reinterpret_cast<uint64_t>(at);
    const uint64_t begin = (start + 15ull) & ~15ull;
    const uint64_t end = (start + bytes) & ~15ull;
    for (uint64_t span = begin; span < end; span += SPARK_PREFETCH_CHUNK) {
        const uint32_t length =
            (uint32_t)(end - span < SPARK_PREFETCH_CHUNK ? end - span : SPARK_PREFETCH_CHUNK);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(span), "r"(length)
                     : "memory");
    }
#else
    (void)at;
    (void)bytes;
#endif
}

#endif  // SPARKFLASH_PREFETCH_SM121_CUH
