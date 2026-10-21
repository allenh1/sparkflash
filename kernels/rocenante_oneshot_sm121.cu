// kernels/rocenante_oneshot_sm121.cu
//
// One-Shot RDMA All-Reduce for 4x DGX Spark (GB10, sm_121a, coherent UMA).
// Specification 5.4 (lines 651-708). The slot ABI lives in rocenante_oneshot_sm121.cuh and
// the Rust mirror in crates/sparkflash-kernels/src/roce_abi.rs.
//
// Protocol, in the order the specification fixes it:
//
//   1. Every block grid-strides the local rank's input into slot->payload[my_rank].
//      __syncthreads() so no block is still writing when the doorbell rings.
//   2. Thread 0 of block 0 publishes slot->arrival_seq[my_rank] = step_seq, issues
//      __threadfence_system() so the staged payload bytes are globally visible, and only then
//      writes host_doorbell_seq = step_seq. The CPU RT proxy / GPUNetIO reads that word and
//      posts the RDMA_WRITE_WITH_IMM slice to the three peers across both CX-7 rails; each
//      peer's NIC writes the slice into this rank's slot->payload[peer] and its
//      arrival_seq[peer]. The timer read after the doorbell is the staging timestamp.
//   3. The first four threads of every block wait for arrival_seq[r] >= step_seq for r =
//      threadIdx.x, then __syncthreads() and stop the RDMA/straggler timer.
//   4. Every block grid-strides the reduction in strict canonical rank order: acc starts at
//      0.0f and accumulates rank 0, 1, 2 and 3 as f32, with exactly one rounding to bf16
//      (__float2bfloat16_rn, round-to-nearest-even). This order is the milestone's definition
//      of bit-identical; sparkflash_kernels::roce_reduce is the CPU reference for it.
//   5. Thread 0 of block 0 records the SCOPED trace event with num_elements * 2 bytes.
//
// Additions to the specification sketch, all defensive and all documented:
//
//   * The peer wait is bounded by a %globaltimer deadline instead of spinning forever
//     (SPARK_ROCENANTE_PEER_WAIT_NS, see the header). A timed-out rank is recorded as a
//     SPARK_ROCE_PAYLOAD_DIAG_FLAG diagnostic trace record and the output is filled with bf16
//     quiet NaN, because the corresponding payload row may still hold the previous step's
//     bytes. Silently returning a stale sum would be worse than a loud failure.
//   * num_elements is clamped to the 2 MiB slot row and an overflow is reported as
//     SPARK_ROCE_DIAG_ROW_OVERFLOW. The router guarantees <= 2 MiB, so this only fires on a
//     caller bug, but it must not scribble outside the slot.
//   * __threadfence() after the wait orders the payload loads behind the arrival observation.
//   * num_elements == 0 runs the whole handshake and records a zero-byte transfer: keeping the
//     protocol uniform is what makes the router's "every step is one seq" invariant hold.
//
// Caller contract, not enforced here:
//   * trace_ring must point at a valid GpuTraceRing whose active_scope_mask may be zero (the
//     SPARK_GPU_TRACE_* macros dereference it unconditionally to read the live mask).
//   * step_seq must be strictly increasing per rank and the slot must not be reused for
//     step_seq + 1 until every rank has finished reading it for step_seq; that double
//     buffering belongs to the communicator, not to this kernel.
//   * blockIdx.y carries the rank for the trace record, as for every instrumented kernel in
//     this tree; my_rank is the rank used for the slot indexing.
//
// Verification status: this file is compiled to PTX by the build-script probe when nvcc is
// present (static asserts, macro instantiations, full type checking) and parsed by the clang
// CUDA syntax probe when only clang is present. Neither can run the kernel; the numerics are
// carried by roce_reduce.rs's CPU reference and by the ABI conformance tests in roce_abi.rs.

#include "rocenante_oneshot_sm121.cuh"

extern "C" __global__ void rocenante_oneshot_allreduce_bf16_tp4(
    const __nv_bfloat16* __restrict__ local_in,
    __nv_bfloat16* __restrict__ local_out,
    RoceOneShotSlot* __restrict__ slot,
    volatile uint64_t* __restrict__ host_doorbell_seq,
    GpuTraceRing* __restrict__ trace_ring,
    const uint32_t layer_id,
    const uint32_t my_rank,
    const uint32_t num_elements,
    const uint64_t step_seq
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // A buffer above the RoCEnante threshold must have been routed to NCCL; clamp defensively
    // so a caller bug cannot write past payload[my_rank] and report it in the trace record.
    const uint32_t row_elems = (uint32_t)SPARK_ROCE_SLOT_ROW_ELEMS;
    const uint32_t row_overflow = (num_elements > row_elems) ? 1u : 0u;
    const uint32_t n = row_overflow ? row_elems : num_elements;

    uint64_t t_enter = 0ull;
    uint64_t t_staged = 0ull;
    uint64_t t_peers_ready = 0ull;

    // Per-rank timeout flags (one writer each) and the shared wait start.
    __shared__ uint32_t s_peer_timed_out[SPARK_ROCE_SLOT_RANKS];
    __shared__ uint64_t s_wait_start_ns;

    if (threadIdx.x < (uint32_t)SPARK_ROCE_SLOT_RANKS) {
        s_peer_timed_out[threadIdx.x] = 0u;
    }

    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_enter);

    // 1. Stage the local rank's contribution into its own coherent-UMA slot row.
    #pragma unroll 4
    for (uint32_t i = tid; i < n; i += stride) {
        slot->payload[my_rank][i] = local_in[i];
    }
    __syncthreads();

    // 2. Publish the sequence number, fence it system-wide, then ring the doorbell. The order
    //    is load bearing: the fence is what makes the staged payload visible to the local NIC
    //    and to the Grace CPU before the doorbell tells them to move it.
    if (tid == 0) {
        slot->arrival_seq[my_rank] = step_seq;
        __threadfence_system();
        *host_doorbell_seq = step_seq;
        SPARK_GPU_READ_TIMER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_staged);
    }
    __syncthreads();

    // Every block times its own bounded wait. Only global thread 0 rings the doorbell, but every
    // block polls the slot, and s_wait_start_ns is __shared__: a deadline written only by block
    // 0 would leave every other block comparing against uninitialised shared memory.
    if (threadIdx.x == 0u) {
        s_wait_start_ns = spark_read_globaltimer_ns();
    }
    __syncthreads();

    // 3. Bounded wait for the other ranks' arrival sequence numbers. Spec 5.4 first four
    //    threads, one rank each; every block waits because every block reduces.
    if (threadIdx.x < (uint32_t)SPARK_ROCE_SLOT_RANKS) {
        const uint32_t peer = threadIdx.x;
        if (slot->arrival_seq[peer] < step_seq) {
            const uint64_t deadline = s_wait_start_ns + (uint64_t)SPARK_ROCENANTE_PEER_WAIT_NS;
            uint32_t spins = 0u;
            while (slot->arrival_seq[peer] < step_seq) {
                if (spark_read_globaltimer_ns() >= deadline) {
                    s_peer_timed_out[peer] = 1u;
                    break;
                }
                if (++spins >= (uint32_t)SPARK_ROCENANTE_SPIN_ITERS) {
                    // Park the warp between polls; a healthy 10 KiB one-shot lands in single
                    // digit microseconds, so this only costs latency under contention.
                    __nanosleep((unsigned int)SPARK_ROCENANTE_SPIN_SLEEP_NS);
                    spins = 0u;
                }
            }
        }
        // Acquire fence, and deliberately UNCONDITIONAL. The check above may already have
        // observed the peer's arrival, so a wait-shaped branch would leave exactly that case
        // reading the payload without an acquire paired to the writer's release. On coherent
        // UMA the writer's __threadfence_system() may make this redundant, but a host with no
        // GPU cannot prove that, and one fence is free against the 22 us one-shot budget of
        // spec 6.1. The M3 review flagged the previous conditional placement.
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
        SPARK_GPU_READ_TIMER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_peers_ready);
    }

    // 4. Reduce in strict canonical rank order: one f32 accumulator starting at +0.0f, ranks
    //    0, 1, 2, 3 in ascending order, exactly one __float2bfloat16_rn (round-to-nearest-even)
    //    per element. The accumulation is per element, so the block/thread mapping cannot
    //    change the result.
    uint32_t missing = 0u;
    #pragma unroll
    for (uint32_t r = 0u; r < (uint32_t)SPARK_ROCE_SLOT_RANKS; ++r) {
        missing |= (s_peer_timed_out[r] != 0u ? 1u : 0u) << r;
    }
    const uint32_t failed =
        (row_overflow ? SPARK_ROCE_DIAG_ROW_OVERFLOW : 0u) |
        (missing << SPARK_ROCE_DIAG_MISSING_SHIFT);

    if (failed == 0u) {
        #pragma unroll 4
        for (uint32_t i = tid; i < n; i += stride) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < (uint32_t)SPARK_ROCE_SLOT_RANKS; ++r) {
                acc += __bfloat162float(slot->payload[r][i]);
            }
            local_out[i] = __float2bfloat16_rn(acc);
        }
    } else {
        // Incomplete collective: propagate quiet NaN rather than a stale partial sum.
        const __nv_bfloat16 nan_value = spark_roce_diag_nan_bf16();
        for (uint32_t i = tid; i < n; i += stride) {
            local_out[i] = nan_value;
        }
    }

    // 5. Scoped trace record: bytes on the success path, diagnostic payload otherwise. The
    //    sub_wait_ns field is t_peers_ready - t_staged, the RDMA/straggler wait, and
    //    payload_meta is num_elements * 2 bytes unless the diagnostic flag is set.
    const uint32_t payload_meta =
        (failed != 0u) ? (SPARK_ROCE_PAYLOAD_DIAG_FLAG | failed) : (uint32_t)(n * 2u);
    SPARK_GPU_TRACE_EXIT_ROCE(
        trace_ring,
        TRACE_SCOPE_ROCENANTE_RDMA,
        layer_id,
        payload_meta,
        t_enter,
        t_staged,
        t_peers_ready
    );
}
