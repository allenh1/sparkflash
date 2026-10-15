// kernels/rocenante_oneshot_sm121.cuh
//
// Slot ABI and shared constants of the RoCEnante one-shot All-Reduce
// (kernels/rocenante_oneshot_sm121.cu, specification section 5.4, lines 646-649).
//
// A GPU writes these fields and a ConnectX-7 NIC DMAs into them, so a layout drift here
// corrupts a collective silently. Three independent gates keep the C side and its Rust mirror
// (crates/sparkflash-kernels/src/roce_abi.rs) in lockstep:
//
//   1. the static_asserts below, evaluated by the build-script probe and by the kernel object
//      itself on any machine with a CUDA frontend;
//   2. the unit tests in roce_abi.rs, which parse the DECLARATIONS in this file and in the
//      kernel, and compare them field by field with the Rust types;
//   3. the same Rust tests, which compare sizeof/alignof/offset_of of the Rust mirror with the
//      numbers those declarations imply.
//
// Nothing here may be reordered or repadded: the byte offsets are part of the wire protocol.
#pragma once

// The clang syntax probe in build.rs runs on hosts without a CUDA installation and compiles
// this header (plus the kernel) against a minimal stand-in for cuda_bf16.h. nvcc never sees
// that file, and the real header always takes the #elif branch.
#if defined(SPARKFLASH_CUDA_SYNTAX_STUB)
#include "cuda_stub.h"
#elif defined(__CUDACC__)
#include <cuda_bf16.h>
#endif

#include <stddef.h>
#include <stdint.h>

#include "trace_globaltimer_sm121.cuh"

// ---------------------------------------------------------------------------------------
// Slot geometry
// ---------------------------------------------------------------------------------------

// Ranks in the TP=4 one-shot group. arrival_seq and payload are both indexed by rank and the
// canonical reduction order is 0, 1, 2, 3.
#define SPARK_ROCE_SLOT_RANKS 4u

// bf16 elements in one rank's payload row: 1024 * 1024 elements * 2 bytes = 2 MiB, which is
// exactly the threshold the dual-path router sends to RoCEnante (ROCENANTE_MAX_BYTES in
// crates/sparkflash-comm). A buffer larger than this must be routed to NCCL instead; the
// kernel clamps and reports SPARK_ROCE_DIAG_ROW_OVERFLOW rather than writing out of bounds.
#define SPARK_ROCE_SLOT_ROW_ELEMS (1024u * 1024u)

// Derived byte counts, kept symbolic so the Rust mirror can re-derive them from its own
// constants; the static asserts below compare them against sizeof/offsetof.
#define SPARK_ROCE_SLOT_ROW_BYTES ((unsigned long long)(SPARK_ROCE_SLOT_ROW_ELEMS) * 2ull)
#define SPARK_ROCE_SLOT_ARRIVAL_BYTES ((unsigned long long)(SPARK_ROCE_SLOT_RANKS) * 8ull)
#define SPARK_ROCE_SLOT_BYTES                                                 \
    (SPARK_ROCE_SLOT_ARRIVAL_BYTES +                                          \
     (unsigned long long)(SPARK_ROCE_SLOT_RANKS) * SPARK_ROCE_SLOT_ROW_BYTES)

// The slot of specification 5.4, byte for byte:
//
//   struct RoceOneShotSlot {
//       volatile uint64_t arrival_seq[4];
//       __nv_bfloat16 payload[4][1024 * 1024];
//   };
//
// arrival_seq[r] is published by rank r - locally by its own kernel, remotely by the peer's
// NIC as an RDMA_WRITE_WITH_IMM - and payload[r] holds rank r's staged contribution in
// coherent UMA. The spec declares no alignment attribute and this header does not add one:
// sizeof is 8 MiB + 32 bytes and the natural alignment is 8, not 32. The sequence array is
// volatile because the spin reads it in a loop while a NIC writes it; "volatile" is a C
// qualifier with no Rust equivalent, so the mirror uses a plain [u64; 4].
struct RoceOneShotSlot {
    volatile uint64_t arrival_seq[4];      // per-rank step sequence number
    __nv_bfloat16 payload[4][1024 * 1024]; // up to 2 MiB per rank in coherent UMA
};

static_assert(sizeof(__nv_bfloat16) == 2, "__nv_bfloat16 must be 2 bytes in the slot payload");
static_assert(alignof(__nv_bfloat16) == 2, "__nv_bfloat16 must be 2-byte aligned");
static_assert(
    sizeof(RoceOneShotSlot) == SPARK_ROCE_SLOT_BYTES,
    "RoceOneShotSlot must be exactly 8 MiB + 32 bytes");
static_assert(
    alignof(RoceOneShotSlot) == 8,
    "RoceOneShotSlot is naturally 8-byte aligned; the spec adds no __align__ attribute");
static_assert(offsetof(RoceOneShotSlot, arrival_seq) == 0, "arrival_seq must start the slot");
static_assert(
    offsetof(RoceOneShotSlot, payload) == SPARK_ROCE_SLOT_ARRIVAL_BYTES,
    "payload must start at offset 32");
static_assert(offsetof(RoceOneShotSlot, payload) == 32, "payload must start at offset 32");

// ---------------------------------------------------------------------------------------
// Trace payload encoding for TRACE_SCOPE_ROCENANTE_RDMA
// ---------------------------------------------------------------------------------------
//
// SPARK_GPU_TRACE_EXIT_ROCE stores the transferred byte count in GpuTraceEvent::payload_meta.
// A one-shot transfer is at most 2 MiB, so bit 31 is free to mark a DIAGNOSTIC record: a peer
// that never arrived, a buffer that did not fit in the slot row, or both. When the flag is
// set the low bits carry SPARK_ROCE_DIAG_* and the missing-rank mask, and the record's
// sub_wait_ns carries the time spent waiting. On the success path payload_meta is the plain
// byte count and the flag is clear.
//
// The trace scope itself is shared with the kernel, so this encoding is decoded by
// sparkflash_kernels::roce_abi::RoceTracePayload and is checked against the Rust constants by
// roce_abi.rs.
#define SPARK_ROCE_PAYLOAD_DIAG_FLAG 0x80000000u
#define SPARK_ROCE_DIAG_MISSING_SHIFT 8u
#define SPARK_ROCE_DIAG_MISSING_MASK 0x0fu
#define SPARK_ROCE_DIAG_ROW_OVERFLOW 0x1u

// bf16 quiet NaN written into local_out when the reduction cannot be completed. A timeout
// means a peer's payload row may still hold the previous step's bytes; propagating NaN makes
// that failure loud instead of returning a plausible-looking wrong sum.
#define SPARK_ROCE_DIAG_NAN_BITS 0x7fc0u

// ---------------------------------------------------------------------------------------
// Bounded peer wait
// ---------------------------------------------------------------------------------------
//
// Specification 5.4 spins forever ("while (slot->arrival_seq[t] < step_seq)"), which a NIC
// firmware bug, a lost doorbell or a dead peer turns into a wedged GPU. The kernel instead
// waits until a deadline and then records a diagnostic trace record and NaNs the output.
//
// The wait is a %globaltimer deadline (nanoseconds; spark_read_globaltimer_ns is shared with
// the trace ABI) plus an exponential-free backoff: poll a few times, then __nanosleep, which
// lets the scheduler park the warp without giving up the low-latency path for the common case
// where the peers arrive in single-digit microseconds. The 1 ms default is ~45x the < 22 us
// one-shot latency gate of specification 6.1, so a healthy collective never reaches it, while
// a broken one recovers in bounded time. All three knobs are compile-time overridable and are
// mirrored by roce_abi.rs (which parses the defaults out of this file).
#ifndef SPARK_ROCENANTE_PEER_WAIT_NS
#define SPARK_ROCENANTE_PEER_WAIT_NS 1000000ull
#endif
#ifndef SPARK_ROCENANTE_SPIN_ITERS
#define SPARK_ROCENANTE_SPIN_ITERS 64u
#endif
#ifndef SPARK_ROCENANTE_SPIN_SLEEP_NS
#define SPARK_ROCENANTE_SPIN_SLEEP_NS 200u
#endif

// Canonical bf16 NaN for the diagnostic output path. __ushort_as_bfloat16 is the
// cuda_bf16.h constructor from raw bits, so no host/device math is involved.
__device__ __forceinline__ __nv_bfloat16 spark_roce_diag_nan_bf16() {
    return __ushort_as_bfloat16((unsigned short)SPARK_ROCE_DIAG_NAN_BITS);
}
