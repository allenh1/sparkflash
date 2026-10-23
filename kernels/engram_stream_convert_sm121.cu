// kernels/engram_stream_convert_sm121.cu
//
// The two exact dtype conversions between the layer arm's f32 residual stream and the released
// Engram head's bf16 stream, in one cubin: `engram_narrow_bf16` (f32 -> bf16) and `engram_widen_f32`
// (bf16 -> f32).
//
// WHY THEY EXIST. The released Engram head reads and writes the residual stream as bf16
// (`model.py:311-365`: the stream operand of the gate, and the one bf16 cast at the write), while the
// S1 layer arm carries `hc_mult * dim` f32 values on device (`LayerResidual::stream`,
// crates/sparkflash-cli/src/layer_device.rs:372-382). The landed Engram kernels that touch the stream
// - `engram_gate_write_sm121` (bf16 stream in, bf16 out) and `engram_write_bf16` (the declared stage
// write, f32 stream in and out) - therefore meet at a bf16/f32 boundary. These two kernels are that
// boundary and nothing else: no arithmetic is added, removed or reordered, and no value is produced
// that the host twin would not also hold.
//
// EXACTNESS. Narrowing an f32 that is already a bf16 value is the identity under
// `__float2bfloat16_rn` (round to nearest even, on a value that is exactly representable), and
// widening a bf16 to f32 is exact by definition (`__bfloat162float` is a bit insertion, no rounding).
// The only case where the narrow is not the identity is an f32 stream value that is not bf16-exact;
// in the arm that value is impossible for the stream the gate sees, because the engine's host seam
// holds the released bf16 stream widened back exactly (crates/sparkflash-engine/src/forward/
// engram.rs, the `out_hc[at] = quant::bf16_round(...)` loop), and `--native-rounding mhc` keeps the
// device arm at the same bf16 boundaries (`native_bf16`, tp4_forward.rs:10568). Should the arm ever
// hold a non-bf16-exact f32 value there, the narrow is the released rounding - a documented class,
// not a surprise.
//
// GEOMETRY. Both kernels are grid-strided: the launcher passes a `uint32_t count` of ELEMENTS
// (2,0480 for the released stream = hc_mult * dim = 4 * 5,120, or 5,120 for a value row) and the grid
// is `ceil(count / 256)` blocks of 256 threads, capped at 65,535 blocks so the grid-stride loop also
// covers a count the grid cannot index directly. Every element is touched once, in index order, and
// the two kernels touch only their own spans: neither allocates, reads back, nor synchronises.
//
// ABI. Three argument words each - source pointer, destination pointer, element count. Unlike the
// seven layer-stage kernels, these two carry NO trace ring: they are not stages, and the repo's rule
// for a non-stage kernel is the one `layer_seams_launch.rs:20-23,41-43` states for
// `shared_swiglu_f32` - no trace ring, the stage that launches it records the trace. So a
// trace-disabled and a trace-armed arm launch these identically.
//
// PIN. No cubin bytes or digest live here: the artifact is pinned in baseline/kernel_baseline.txt and
// the Rust side resolves each symbol by name
// (crates/sparkflash-kernels/src/engram_convert_launch.rs). The declarations below are the pin the
// launcher parses.

#include "m4_common_sm121.cuh"

// Threads per block: the launcher's BLOCK and this constant must agree (the pin test reads both).
#define SPARK_ENGRAM_CONVERT_THREADS 256u

// f32 -> bf16, round to nearest even: the released stream's dtype at every write, and exactly
// `__float2bfloat16_rn`, which is the rounding `engram_native_ref::bf16_round` reproduces on the
// host (`half::bf16::from_f32` has the same rule).
extern "C" __global__ void engram_narrow_bf16(
    const float *__restrict__ src,
    __nv_bfloat16 *__restrict__ dst,
    const uint32_t count) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    dst[index] = __float2bfloat16_rn(src[index]);
  }
}

// bf16 -> f32: widening is a bit insertion (the low 16 mantissa bits are zeros), so this direction
// has no rounding at all. It is what the gate's bf16 stream values look like to f32 arithmetic.
extern "C" __global__ void engram_widen_f32(
    const __nv_bfloat16 *__restrict__ src,
    float *__restrict__ dst,
    const uint32_t count) {
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    dst[index] = __bfloat162float(src[index]);
  }
}
