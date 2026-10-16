// kernels/trace_globaltimer_sm121.cuh
// Zero-overhead in-graph GPU trace points for Blackwell sm_121a using %globaltimer in
// coherent unified memory. The Rust side of this ABI lives in sparkflash-trace
// (event::GpuTraceEvent / event::GpuTraceRingHeader); keep the two in lockstep.
//
// Arming protocol: the Rust admin path stores the live scope mask into
// GpuTraceRingHeader::active_scope_mask, which is allocated with cudaHostAllocMapped and is
// therefore visible to both the Grace CPU and the GPU. Flipping a bit activates the matching
// SCOPE in every kernel that is already captured in the CUDA graph - no recapture, no
// cudaEventRecord, no profiler process.
#pragma once
#include <stdint.h>

// Scope bits. MUST match sparkflash_trace::TraceScope.
#define TRACE_SCOPE_API_ENCODE      (1ULL << 0)
#define TRACE_SCOPE_SCHED_BATCH     (1ULL << 1)
#define TRACE_SCOPE_KV_RADIX        (1ULL << 2)
#define TRACE_SCOPE_SWA_REPLAY      (1ULL << 3)
#define TRACE_SCOPE_ENGRAM_PREFETCH (1ULL << 4)
#define TRACE_SCOPE_ENGRAM_IO_URING (1ULL << 5)
#define TRACE_SCOPE_ENGRAM_GPU_GATE (1ULL << 6)
#define TRACE_SCOPE_MEGA_MHC        (1ULL << 7)
#define TRACE_SCOPE_CSA2_INDEXER    (1ULL << 8)
#define TRACE_SCOPE_CSA2_ATTN       (1ULL << 9)
#define TRACE_SCOPE_DENSE_MXFP8     (1ULL << 10)
#define TRACE_SCOPE_MOE_MXFP4       (1ULL << 11)
#define TRACE_SCOPE_DSPARK_DRAFT    (1ULL << 12)
#define TRACE_SCOPE_ROCENANTE_RDMA  (1ULL << 13)
#define TRACE_SCOPE_NCCL_COLLECTIVE (1ULL << 14)
#define TRACE_SCOPE_L2_PREFETCH     (1ULL << 15)

// One 32-byte trace record. Mirrors sparkflash_trace::GpuTraceEvent.
struct __align__(32) GpuTraceEvent {
    uint64_t start_ns;        // %globaltimer nanoseconds at entry
    uint64_t end_ns;          // %globaltimer nanoseconds at exit
    uint32_t sub_wait_ns;     // in-kernel wait: RDMA peer arrival, Engram stall, ...
    uint32_t scope_and_rank;  // (scope_bit_index << 8) | rank, bit index 1-based
    uint32_t layer_id;
    uint32_t payload_meta;    // bytes transferred, or active rows M
};

// Ring header. Mirrors sparkflash_trace::GpuTraceRingHeader (32 bytes: 16 used, 16 padding).
//
// The padding is load bearing. Events are indexed as (char *)ring + sizeof(GpuTraceRing) and
// GpuTraceEvent is __align__(32), so sizeof(GpuTraceRing) must be a multiple of 32 or every
// event after the first is misaligned. Keep the two sides in lockstep; the Rust side asserts
// this layout in sparkflash-trace's event tests and in sparkflash-kernels' ABI test.
struct __align__(32) GpuTraceRing {
    volatile uint64_t active_scope_mask;  // toggled live from Rust through coherent UMA
    uint32_t write_idx;                   // atomicAdd-advanced by thread 0 of block 0
    uint32_t capacity_mask;               // power-of-two ring mask, e.g. (1u << 20) - 1
    uint64_t reserved[2];                 // padding to 32 bytes; must stay zero
    // GpuTraceEvent events[capacity_mask + 1];  // 32 MiB coherent UMA ring by default
};

// The rank a record belongs to is carried in blockIdx.y: launch every instrumented kernel on a
// grid whose y dimension is the rank, so no extra kernel argument is needed.

__device__ __forceinline__ uint64_t spark_read_globaltimer_ns() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
#define SPARK_GPU_TRACE_ENTER(ring, scope, t_var)                                        \
    do {                                                                                 \
        if (blockIdx.x == 0 && threadIdx.x == 0 && ((ring)->active_scope_mask & (scope))) { \
            (t_var) = spark_read_globaltimer_ns();                                        \
        }                                                                                \
    } while (0)

#define SPARK_GPU_READ_TIMER(ring, scope, t_var)      \
    do {                                              \
        if ((ring)->active_scope_mask & (scope)) {    \
            (t_var) = spark_read_globaltimer_ns();    \
        }                                             \
    } while (0)

#define SPARK_GPU_TRACE_EXIT(ring, scope, layer, meta, t_enter)                          \
    do {                                                                                 \
        if (blockIdx.x == 0 && threadIdx.x == 0 && ((ring)->active_scope_mask & (scope))) { \
            uint64_t t_end = spark_read_globaltimer_ns();                                 \
            uint32_t idx = atomicAdd((unsigned int *)&(ring)->write_idx, 1U)              \
                           & (ring)->capacity_mask;                                       \
            GpuTraceEvent *ev = (GpuTraceEvent *)((char *)(ring) + sizeof(GpuTraceRing))  \
                                + idx;                                                    \
            ev->start_ns = (t_enter);                                                     \
            ev->end_ns = t_end;                                                           \
            ev->sub_wait_ns = 0U;                                                         \
            ev->scope_and_rank = ((uint32_t)__ffsll((long long)(scope)) << 8)              \
                                 | (uint32_t)(blockIdx.y);                                \
            ev->layer_id = (uint32_t)(layer);                                             \
            ev->payload_meta = (uint32_t)(meta);                                          \
        }                                                                                \
    } while (0)

#define SPARK_GPU_TRACE_EXIT_ROCE(ring, scope, layer, bytes, t_enter, t_staged, t_ready)  \
    do {                                                                                 \
        if (blockIdx.x == 0 && threadIdx.x == 0 && ((ring)->active_scope_mask & (scope))) { \
            uint64_t t_end = spark_read_globaltimer_ns();                                 \
            uint32_t idx = atomicAdd((unsigned int *)&(ring)->write_idx, 1U)              \
                           & (ring)->capacity_mask;                                       \
            GpuTraceEvent *ev = (GpuTraceEvent *)((char *)(ring) + sizeof(GpuTraceRing))  \
                                + idx;                                                    \
            ev->start_ns = (t_enter);                                                     \
            ev->end_ns = t_end;                                                           \
            ev->sub_wait_ns = (uint32_t)((t_ready) - (t_staged));                          \
            ev->scope_and_rank = ((uint32_t)__ffsll((long long)(scope)) << 8)              \
                                 | (uint32_t)(blockIdx.y);                                \
            ev->layer_id = (uint32_t)(layer);                                             \
            ev->payload_meta = (uint32_t)(bytes);                                         \
        }                                                                                \
    } while (0)

#else
// Trace compiled out. Every argument is still consumed, because a kernel whose trace is disabled
// otherwise reports its timer variables and payload computations as "set but never used"
// (nvcc 550-D / 177-D), and a plain void cast does not satisfy that diagnostic. Passing the
// values through this identity function is a real use; both compilers inline it away.
__device__ __forceinline__ uint64_t spark_trace_keepalive(uint64_t value) { return value; }

#define SPARK_GPU_TRACE_ENTER(ring, scope, t_var)     \
    do {                                              \
        (void)(ring);                                 \
        (void)(scope);                                \
        (t_var) = spark_trace_keepalive(t_var);       \
    } while (0)
#define SPARK_GPU_READ_TIMER(ring, scope, t_var)      \
    do {                                              \
        (void)(ring);                                 \
        (void)(scope);                                \
        (t_var) = spark_trace_keepalive(t_var);       \
    } while (0)
#define SPARK_GPU_TRACE_EXIT(ring, scope, layer, meta, t_enter)          \
    do {                                                                 \
        (void)(ring);                                                    \
        (void)(scope);                                                   \
        (void)(layer);                                                   \
        (void)spark_trace_keepalive((uint64_t)(meta));                   \
        (void)spark_trace_keepalive(t_enter);                            \
    } while (0)
#define SPARK_GPU_TRACE_EXIT_ROCE(ring, scope, layer, bytes, t_enter, t_staged, t_ready) \
    do {                                                                                \
        (void)(ring);                                                                   \
        (void)(scope);                                                                  \
        (void)(layer);                                                                  \
        (void)spark_trace_keepalive((uint64_t)(bytes));                                 \
        (void)spark_trace_keepalive(t_enter);                                           \
        (void)spark_trace_keepalive(t_staged);                                          \
        (void)spark_trace_keepalive(t_ready);                                           \
    } while (0)
#endif
