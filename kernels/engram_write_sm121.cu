// kernels/engram_write_sm121.cu
//
// The Engram residual write, f32 in and f32 out: the final statement of the released Engram head
// (inference/model.py:355-365) and the `engram_write` stage of the whole-layer device port
// (crates/sparkflash-cli/src/layer_device.rs, STAGE_CONTRACTS entry 0, kernel `engram_write_bf16`).
//
// THE REFERENCE, per token (model.py:363-365):
//
//     out = (h + gate.unsqueeze(-1) * value).to(x.dtype)   // one bf16 cast, at the write
//
// `h` is the residual stream's hc copies, `gate` one coefficient per copy and `value` the shared
// value row of the Engram head's wkv output; `.to(x.dtype)` is the released stream's bf16, so the
// write rounds ONCE, after the add. The engine keeps that stream as a Vec<f32> whose elements are
// the bf16 values widened back exactly (crates/sparkflash-engine/src/forward/engram.rs, the
// `out_hc[at] = quant::bf16_round(out_hc[at] + row * col)` loop), so this kernel writes the same
// f32 values: the bf16 rounding happens on the device and the result is widened back to f32 - one
// cast on device, exactly where the released code has it, and no f32 value the host twin would not
// also hold.
//
// ARITHMETIC ORDER. One multiply and one add per element, each with one f32 rounding:
//
//     sum        = __fadd_rn(h, __fmul_rn(gate[copy], value[d]))
//     stream_out = __bfloat162float(__float2bfloat16_rn(sum))
//
// The explicit rounded intrinsics are the contract, not a hint: nvcc's default contraction would
// fuse the multiply and add into one FFMA, which rounds once where the host twin rounds twice, and
// the bf16 cast is the CUDA intrinsic the twin's `bf16_round` reproduces
// (crates/sparkflash-kernels/src/engram_native_ref.rs::bf16_round). No transcendental is involved,
// so unlike the gate's sigmoid this kernel has no F8-class tolerance: on finite operands it is
// bit-identical to the twin's write half.
//
// PARALLELISM. One block per 1024 columns of one copy: grid = (ceil(dim / 1024), hc). The copy is
// blockIdx.y because the stage's fixed ABI has no hc argument (`dim` is the last word; the contract
// is layer_device.rs:764-768). Within a block the 256 threads stride the 1024-column window, so
// consecutive lanes read consecutive f32 values of the stream, of `value` and write consecutive f32
// values of the output: one coalesced sector per 32 lanes per iteration. Blocks never reduce and
// never communicate, and every element is read and written once, so an in-place call
// (stream_out == stream) is exact: element i is loaded before element i is stored, and no other
// thread touches i.
//
// GEOMETRY. dim > 0, copies > 0 and copies <= 65535 (grid.y's bound) are checked by the typed
// launcher (crates/sparkflash-kernels/src/engram_write_launch.rs), together with the four span
// lengths; the kernel body trusts them. Every index is 64-bit so copy * dim cannot overflow for a
// dim that fits the declaration's `int`.
//
// TRACE. The signature carries the ring/layer/step words for ABI uniformity with the other seam
// kernels, but this kernel emits no event: the protocol's 16 scope bits are all assigned
// (kernels/trace_globaltimer_sm121.cuh) and none describes the Engram write, so an event here
// would be misattributed. The arguments are present so a future scope does not change the ABI, and
// the launcher passes 0 for the ring in the trace-disabled steady state (the same convention
// engram_gate_write_sm121.cu states).

#include "m4_common_sm121.cuh"

// The contract's block granularity: one block owns 1024 columns of one copy (`one block per 1024
// columns`, the stage's declared shape).
#define SPARK_ENGRAM_WRITE_COLUMNS 1024u

// The block that walks the window; 4 values per lane at the released 1024-column window.
#define SPARK_ENGRAM_WRITE_THREADS 256u

// The released write's one bf16 cast, as the f32 value the engine buffer holds. bf16 -> f32 is
// exact, so the widening adds no second rounding.
__device__ __forceinline__ float spark_engram_write_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

extern "C" __global__ void engram_write_bf16(
    const float *__restrict__ stream,      // [hc][dim] f32 residual stream, bf16-exact values
    const float *__restrict__ gate,        // [hc] f32 gate coefficients, one per copy
    const float *__restrict__ value,       // [dim] f32 the Engram head's shared value row
    float *__restrict__ stream_out,        // [hc][dim] f32 out: the stream with gate * value added
    const int dim,                         // columns per copy (the released dim, 5,120)
    GpuTraceRing *__restrict__ trace_ring, // ABI uniformity only; see TRACE above
    const uint32_t layer_id,
    const uint64_t step_seq) {
    (void)trace_ring;
    (void)layer_id;
    (void)step_seq;

    // The stage's declaration carries no hc word: the copy is the grid's y index.
    const uint32_t copy = blockIdx.y;
    const uint64_t columns = (uint64_t)(uint32_t)dim;
    const uint64_t start = (uint64_t)blockIdx.x * SPARK_ENGRAM_WRITE_COLUMNS;
    const uint64_t stop =
        start + SPARK_ENGRAM_WRITE_COLUMNS < columns ? start + SPARK_ENGRAM_WRITE_COLUMNS : columns;
    if (start >= stop) {
        return;
    }
    // One scalar load: the gate is uniform across the block and the copy, so it is read once.
    const float gate_value = gate[copy];
    const uint64_t copy_base = (uint64_t)copy * columns;
    for (uint64_t column = start + threadIdx.x; column < stop;
         column += SPARK_ENGRAM_WRITE_THREADS) {
        const uint64_t index = copy_base + column;
        const float sum = __fadd_rn(stream[index], __fmul_rn(gate_value, value[column]));
        stream_out[index] = spark_engram_write_round(sum);
    }
}
