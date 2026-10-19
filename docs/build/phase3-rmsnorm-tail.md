# The serving tail's standalone final RMSNorm (\`rmsnorm_sm121\`)

Status: the kernel, its pin, the host reference and the m18 harness are on branch task/c4c-rmsnorm (based on the m17c review-fix branch), and the kernel is **measured and wired**. m18 passed at 8 and 130 tokens with 0 mismatched BF16 bytes (81,920 and 1,331,200 byte bars), the odd width 513 case exact, the eps control exact and two launches byte-identical. c4c-norm then wired it as the engine final tail norm: b41 measured 0/10,240 BF16 mismatches and 1,167 device launches. The m17c review doc-only wording correction (the stage-4 operand handoff is a host re-upload of the stage-3 readback, with no host arithmetic) is in the same branch.

## 1. Why a standalone norm kernel

The model's tail is: the collapsed mHC residual \`[T, 5120]\` f32 -> final RMSNorm with the released
\`norm.weight\` -> LM head. The engine runs that norm on the host today:
\`crates/sparkflash-engine/src/forward/mod.rs:602\` calls
\`linalg::rms_norm(&mut tail, &norm, dim, config.norm_eps as f32)\`
(\`crates/sparkflash-engine/src/linalg.rs:318\`). It is the last host op before the head, so it is the
first thing c4c can move.

\`kernels/mega_mhc_sm121.cu\` computes an RMS-style normalisation internally, but it is not an
exported norm stage and its arithmetic belongs to the mHC residual seam; reusing it would mean
calling a fused mHC kernel to normalise a different operand. The LM head already reads a **BF16**
activation ABI (\`kernels/cert_lm_head_sm121.cu\`: hidden is \`const __nv_bfloat16 *\`), so this kernel
is BF16 in, BF16 out, with the released \`norm.weight\` bytes as BF16 \`[5120]\`.

## 2. The arithmetic, exactly

The kernel reproduces the engine's per-row operation order one for one. It is a bit-exactness
class, not a tolerance class:

| step | engine (\`linalg.rs:318-345\`) | kernel (\`kernels/rmsnorm_sm121.cu\`) |
| --- | --- | --- |
| sum | \`sum::<f32>()\` over ascending columns of \`x * x\` | \`sum = __fadd_rn(sum, __fmul_rn(value, value))\` in ascending column order |
| mean | \`sum / cols as f32\` | \`__fdiv_rn(sum, (float)cols)\` |
| inv | \`1.0 / (mean + eps).sqrt()\` | \`__fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)))\` |
| scale | \`*value = *value * inv * w\` (left-associated) | \`__fmul_rn(__fmul_rn(value, inv), weight)\` |
| round | the head's operand is BF16 | \`__float2bfloat16_rn\` (RN-even) |

The rounding intrinsics are deliberate: plain operators let nvcc contract \`x*x\` and the add into
one FMA (one rounding where the engine rounds twice) or use an approximate reciprocal square root
(not the engine's sqrt-then-reciprocal). The pinned cubin's PTX has **0** \`fma.rn\`, **2**
\`div.rn.f32\`, **1** \`sqrt.rn.f32\` and **0** \`rsqrt\` instructions.

**One thread owns one whole row**, walking it in ascending column order. A block or warp reduction
sums in a tree order that is not the engine's sequential sum, so it would not be bit-identical. The
row-serial order costs nothing that matters at the tail: the tail is the decode path and the row
count is the token count (8 or 130 in the window, 1 in serving). The kernel is one launch, grid
\`ceil(rows / 256)\`, block 256.

## 3. ABI

\`\`\`c
extern "C" __global__ void rmsnorm_sm121(
    const __nv_bfloat16 *x,       // [rows][cols] BF16
    const __nv_bfloat16 *weight,  // [cols] BF16 released norm.weight
    __nv_bfloat16 *out,           // [rows][cols] BF16
    uint32_t rows,
    uint32_t cols,
    float eps,                    // config rms_norm_eps cast to f32 (released: 1e-20)
    GpuTraceRing *trace_ring,     // ABI uniformity only; no event yet
    uint32_t layer_id,
    uint64_t step_seq);
\`\`\`

Nine argument words, declared by \`crates/sparkflash-kernels/src/rmsnorm_launch.rs\`
(\`ARG_WORDS = 9\`, \`BLOCK = (256, 1, 1)\`, \`RmsNormLaunchParams::words()\`). The launcher refuses a
zero row or column count instead of launching a zero grid.

**Trace.** The trace words are carried for ABI uniformity but the kernel emits no event: all
sixteen scope bits are assigned (\`kernels/trace_globaltimer_sm121.cuh\`, specification 5.8) and none
describes a tail norm, so recording under a dense or mHC scope would misattribute it. A norm scope
would be a protocol revision; the arguments mean that revision does not change this ABI. The pin
therefore records \`ptx_membar_lines=0\` as an expected property, not an accident.

## 4. Layout, loads and the pin

- \`cols % 8 == 0\` (5120 in the released config): the row is read in 16-byte chunks, four \`u32\`
  words = eight BF16 values. The load is explicit \`ld.global.nc.v4.u32\` because nvcc otherwise
  narrows it to eight 16-bit loads once it sees only BF16 halves used; the clang syntax probe (no
  nvcc) compiles the equivalent aligned \`uint4\` dereference under
  \`SPARKFLASH_CUDA_SYNTAX_STUB\`. The pinned PTX has the vector loads.
- Any other width: scalar loads, same arithmetic order. The harness exercises 513 columns.
- Output stores are scalar \`__nv_bfloat16\` writes; the rounding contract is the conversion
  intrinsic, and packing eight of them into a vector would need bit punning for a tail that is not
  bandwidth-bound.

Pin (nvcc V13.0.88 / CUDA 13.0.3, \`-arch=sm_121a\`, the pinned image):

\`\`\`text
kernel rmsnorm_sm121 cubin_bytes=31480 cubin_sha256=5f953bc0fb781afe3f9b42703b19b37f89ef45b74a4ad607938df2e2c3448579 ptx_membar_lines=0
\`\`\`

It is the eleventh entry in \`crates/sparkflash-kernels/baseline/kernel_baseline.txt:12-22\`;
\`build.rs\`'s \`PROBED_KERNELS\` now probes eleven sources (the combined probe still compiles every
kernel in one translation unit, so the helpers are \`spark_rmsnorm_\`-prefixed), and
\`cuda_driver::KernelId\` gained \`RmsNorm\` (ten variants). The frozen six-kernel \`m4_abi\` array is
untouched.

Build line:

\`\`\`bash
docker run --rm -v "$PWD":/w -w /w nvidia/cuda:13.0.3-devel-ubuntu22.04 \
  nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/rmsnorm_sm121.cu -o <dir>/rmsnorm_sm121.cubin
\`\`\`

## 5. The host reference is the engine, not a copy of the formula

\`crates/sparkflash-kernels/src/rmsnorm_ref.rs\` decodes the released BF16 bytes to the exact f32
values the engine holds (bf16 -> f32 is exact), calls
\`sparkflash_engine::linalg::rms_norm\` - the function the forward pass calls - and applies the
kernel's output rounding (\`m4_quant::bf16_bits\`, the same round-to-nearest-even as
\`__float2bfloat16_rn\`). If the engine's norm arithmetic changes, the m18 comparison shows a
mismatch instead of a twin that changed with it. Unit tests pin the real 5120 width, an odd width,
the analytic unit-row scaling, the source call to the engine (a review ratchet against a hand copy)
and the kernel's rounding/vector-load tokens.

## 6. m18 harness: what is measured

\`crates/sparkflash-kernels/examples/m18_final_norm.rs\`, one pinned cubin, released checkpoint bytes:

1. **released geometry**: \`config.json\` hidden 5120 and \`rms_norm_eps\` 1e-20 (cast to f32 exactly as
   the engine casts it); \`norm.weight\` is BF16 \`[5120]\`; the \`[T, 5120]\` activation rows are the
   released \`embed.weight\` rows 0..T - a declared stand-in for the collapsed tail hidden.
2. **real width, bit-exact**: device BF16 output must equal bf16(engine \`rms_norm\`) on the same
   released bytes at the configured epsilon, zero mismatched codes; two launches byte-identical.
3. **epsilon inside**: the same launch at eps = 1.0 is still bit-exact to the reference at 1.0 and
   differs from the released-epsilon output (proving the epsilon is applied, not ignored).
4. **odd width**: a synthetic \`[3, 513]\` BF16 case (not a multiple of eight, so the scalar path
   runs) is bit-exact to the reference; the operand is declared synthetic.

Window command used for the b41 c4c-norm record (the cubin was staged next to the binary):

\`\`\`bash
sparkflash_m18_final_norm_<rev>_aarch64 --checkpoint <snapshot dir> --cubin-dir <cubin dir> --tokens 8 --floor-gib 32
sparkflash_m18_final_norm_<rev>_aarch64 --checkpoint <snapshot dir> --cubin-dir <cubin dir> --tokens 130 --floor-gib 32
\`\`\`

Expected: \`m18: PASS (7 checks; tokens 8; one pinned cubin)\` and the same at 130 (the check count
and numbers go in the window record below).

## 7. Window record

**Measured and wired.** m18 recorded 0 mismatched BF16 bytes at 8 tokens (0/81,920) and 130 tokens (0/1,331,200), the odd 513 case exact, the eps control exact, and two launches byte-identical. The c4c-norm chain window b41 then measured final norm diag rows=1 cols=5120 bf16_mismatches=0 bytes=10240 and final_norm_launches=1, device_launches=1167. Staged evidence on the shared evidence tree /home/user/phase2-evidence/window-2026-10-18/:

- binary `m18_final_norm_e09e08c_aarch64`, 4,665,184 B, sha256
  `4e9ac91490639cb9d4e6adb9b6a938c2ad7540c7a89abb80c5650f159abf9566` (built from branch
  `task/c4c-rmsnorm` at `e09e08c`, `scripts/build-aarch64.sh --example=m18_final_norm`);
- cubin `cubins/rmsnorm_sm121.cubin`, 31,480 B, sha256 `5f953bc0...` (the pinned artifact above).

The m18 measurement window record is b40 (tokens 8 and 130, 7 checks each); this section is no longer pending. The c4c-norm chain window (b41) is the separate wiring evidence above.

## 8. What this is not

Not the served forward pass, not c4c engine integration (engine-dev wires it), not the LM head, not
a token/chain/per-layer claim, not host-time saved, not a performance or bandwidth claim. The input
rows are released \`embed.weight\` rows, a declared stand-in for the collapsed tail hidden, and the
odd-width operands are declared synthetic.

## 9. Integration note for c4c (engine-dev)

- Entry: \`rmsnorm_sm121(x_bf16, weight_bf16, out_bf16, rows, cols, eps_bits, trace_ring, layer_id,
  step_seq)\`; typed launcher \`rmsnorm_launch::{load, launch, RmsNormLaunchParams}\`; \`KernelId::RmsNorm\`.
- Pin: 31,480 B, \`5f953bc0...\`, \`ptx_membar_lines=0\` (baseline manifest line above).
- The host fallback stays \`sparkflash_engine::linalg::rms_norm\` on the f32 tail; the device path
  replaces the \`final_norm\` host seam (\`forward/mod.rs:602\`, \`StageEvent: "host final RMS norm;
  class-(b) seam, not device-executed"\`) with the launch and a declared fallback behind a flag.
- The REQUIRED_KERNELS catalog includes this kernel: it is one of the nine required
  chain pins, and the c4c final-norm launch is gated by it.
- The kernel emits no trace event (no norm scope among the sixteen assigned bits); do not record
  it under a dense/mHC scope.
