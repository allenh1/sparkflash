# Design and pin plan: a 576-K stored-layout combine kernel

**Status: implemented and pinned (task-6, branch task/m25-stored).** The kernel, its ABI pin, the
typed launcher, the host twin, the device harness and the twenty-entry baseline are committed; the
m25_moe_stored harness has NOT run on a GPU yet, so this page still makes no device measurement
claim. **The layout table below was corrected after the page was first written:** the released plan
stores the codes and the scales as two SEPARATE tensors (288-byte code rows, 18-byte scale rows), not
as interleaved 306-byte rows. 306 is the cost of one output row across the two tensors, not a
stride. Section 1 carries the corrected table and section 6 records the evidence and the
implementation. This page answers the launch-compatibility finding device-lead surfaced: the routed
down/combine kernel b12x_mxfp4_moe_sm121 reads the chain's A8 operand layout (640 padded K values
per row: 320 packed FP4 code bytes + 20 ue8m0 scale bytes, 20 MX blocks), while the released
checkpoint's per-rank plan stores the expert rows at their real width (576 K values: 288 code bytes +
18 scale bytes per row, 18 MX blocks). Binding the arena pointer is bytes-verified, but launching the
current combine kernel on the released plan would misindex every row after the first: its code row
stride would be 320 B where the plan says 288 B, its scale row stride 20 B where the plan says 18 B,
and its K loop would walk 20 blocks where the plan has 18.

The boot-repack remedy (materialise the 640-padded rows on every node at load) costs +25.2 GiB/rank
on a 121 GiB node and is rejected. The kernel-ABI remedy is a second combine kernel that reads the
stored 576-K layout. This page states its arithmetic, its layout contract and every pin that must
move; it is written so the next session can implement it against a GPU window instead of
reverse-engineering this decision.

## 1. The two layouts, stated exactly

| | A8 chain layout (today) | released per-rank plan (implemented) |
|---|---|---|
| K walked | 640 (576 real + 64 pad) | 576 |
| MX blocks (K / 32) | 20 | 18 |
| per-row packed code bytes | 320 | 288 |
| per-row ue8m0 scale bytes | 20 | 18 |
| per-row code stride | 320 | **288** |
| per-row scale stride | 20 (same tensor as the codes) | **18 (SEPARATE scale tensor)** |
| cost of one output row across its tensors | 340 | **306** (288 + 18) |
| scale grid position | row-adjacent, after the code bytes | separate tensor: scale byte s of output row r at w_scales + r * 18 + s |
| bytes per expert (d = 5120 rows) | 1,740,800 | **1,566,720** (1,474,560 code + 92,160 scale) |
| bytes per rank launch (6 experts) | 10,444,800 | **9,400,320** |

The layout change is **strides and K only**. It does not touch: the activation operand (E4M3, one
byte per value, with its own per-32 scale grid), the expert ids, the f32 slot accumulator, the top-6
slot ordering, or the arithmetic per block. CONFIRMED (task-6, from the released checkpoint and the
released weight loader): the scales are a separate tensor. The per-rank plan stores
w2.weight [experts][5120][288] I8 with a 288-byte code row and w2.scale [experts][5120][18]
F8_E8M0 with an 18-byte scale row; 288 + 18 = 306 is the cost of one output row across the two
tensors, not a row stride. The checkpoint's global tensors are [5120,1152] and [5120,72], and rank r's
slice is the leading 288 code bytes and 18 scale bytes of each row
(crates/sparkflash-cli/src/resident_weights.rs asserts the [5120,288] row shape and the 92,160-byte
per-rank scale shard). The kernel, the host twin and the pins implement that separate-tensor form; a
306-byte interleaved staging is rejected by mxfp4_ref::check_stored_moe_layout.

## 2. The kernel

New symbol b12x_mxfp4_moe_stored_sm121 in
kernels/b12x_mxfp4_moe_stored_sm121.cu: the same source shape as the incumbent after this session's
split-K rewrite (32 output columns x 8 parallel K groups, 256 threads, one reducer per (slot,
column), 13,824 B shared for 6 slots x 18 terms x 32 columns), with:

- SPARK_M4_STORED_K_VALUES = SPARK_M4_MOE_WIDTH (576), SPARK_M4_STORED_K_BLOCKS = 18u, and a
  static_assert(576u % 32u == 0u);
- SPARK_M4_STORED_CODE_ROW = 288u, SPARK_M4_STORED_SCALE_ROW = 18u, SPARK_M4_STORED_ROW_TOTAL =
  306u, and static_assert(288u + 18u == SPARK_M4_STORED_ROW_TOTAL);
- TWO weight pointers: the code row at w_fp4 + (uint64_t)row * 288u, the scale row at
  w_scales + (uint64_t)row * 18u, where row = expert * 5120 + output column;
- grid (tokens, 160), unchanged: 160 hidden tiles of 32 columns cover d = 5120;
- the same 10-word parameter list, so the ABI word count does not move. The two weight pointers
  keep their names and constness; only the layout they address changes, which is exactly what a
  second cubin pin is for.

**Why a second kernel, not a runtime layout flag.** A flag would add an argument (an ABI change to
the frozen declaration) and a branch inside the K-block loop of every thread. A separate symbol keeps
each cubin's compiled form pinned on its own line of the manifest and lets the chain select the
kernel by plan layout at load time, which is also what the bytes-verified arena binding needs: the
plan's layout is known before the first launch.

## 3. Arithmetic and exactness (the contract the rewrite already established)

Per output element and per slot the arithmetic is the incumbent's, with 18 ascending block terms
instead of 20:

t_b = __fmul_rn(__fmul_rn(partial_b, act_scale_b), w_scale_b) for b = 0..17 ascending, then
acc = __fadd_rn(acc, t_b) from +0.0f, then the unrounded add into the rank f32 slot accumulator.
The 32-value partial is the same FP8xFP8 block product; the FP4 weight decodes through the same
exact E2M1 -> E4M3 round trip. **This variant cannot be bit-compared against the incumbent on the
same operand bytes** (they read different K widths), so its exactness claim is stated against the
twin on the stored layout, exactly as the incumbent's is stated against the twin on the padded
layout:

- host twin: mxfp4_ref::moe_combine_rank_stored (same arithmetic, code stride 288 and separate
  scale stride 18, 18 blocks) plus stored_moe_layout()/check_stored_moe_layout(). A unit test takes
  the released stored shape for a small fixture, de-interleaves it to the padded layout (append 64
  code bytes + 4 scale bytes per row), and asserts the padded twin and the stored twin produce
  bit-identical f32 rank accumulators and bit-identical finalised rows; it repeats the proof with
  NONZERO padding tails, so the result is shown not to depend on what the padding holds.
- device harness (needs a GPU window): examples/m25_moe_stored.rs, modelled on
  examples/moe_slice.rs: stage the released stored layout unchanged, launch the stored cubin,
  compare the f32 slot accumulator bit-for-bit against the stored twin, launch a second time after
  re-zeroing the caller accumulator and require byte-identical output. Pass bar printed before the
  numbers, no timing claim on the first run, cubin size/sha256 pinned and checked before launching.

## 4. Pin and baseline plan (every file that moved, in order)

1. kernels/b12x_mxfp4_moe_stored_sm121.cu - new source.
2. crates/sparkflash-kernels/src/mxfp4_moe_stored_abi.rs - new ABI pin module: declaration
   (pinned by identity against the incumbent's parameter list, different symbol), constants
   (K_VALUES = 576, CODE_ROW_BYTES = 288, SCALE_ROW_BYTES = 18, ROW_BYTES_TOTAL = 306, K_BLOCKS =
   18, GRID_Y = 160, THREADS = 256), CUBIN_BYTES = 11,544, CUBIN_SHA256 = c452a9aa..., TOOLKIT_RELEASE
   = "V13.0.88".
3. crates/sparkflash-kernels/src/mxfp4_moe_stored_launch.rs - typed launcher; reuses
   mxfp4_moe_launch::Mxfp4MoeParams as its parameter type so the argument order cannot drift between
   the two kernels; KernelId::Mxfp4MoeStored resolves the symbol.
4. crates/sparkflash-kernels/baseline/kernel_baseline.txt - regenerated in the pinned image
   (nvidia/cuda:13.0.3-devel-ubuntu22.04, nvcc V13.0.88): the manifest gained one line, and its
   header count and text moved from nineteen to twenty. The original nineteen pins are byte-identical.
5. crates/sparkflash-kernels/scripts/regen-kernel-baseline.sh and check-kernel-baseline.sh - the
   kernel list and the header text moved together with (4); the check script's closing text is
   unchanged in shape.
6. crates/sparkflash-kernels/src/baseline.rs, router_abi.rs, routed_gate_up_abi.rs,
   tests/kernel_baseline.rs and the CLI's device_chain.rs - every count and phrase assertion updated
   in the same commit ("twenty baseline kernels", twenty pins, nineteen driver kernels, 20 baseline
   pins parsed).
7. crates/sparkflash-kernels/src/mxfp4_ref.rs - the stored twin, the layout contract and its unit
   tests (section 3).
8. crates/sparkflash-kernels/examples/m25_moe_stored.rs - the device harness (section 3),
   committed with the pass bar printed before the numbers and no timing claim.
9. Chain wiring (separate commit, only after the device harness passes in a window):
   crates/sparkflash-cli/src/moe_device.rs selects the stored cubin when the bound plan's layout is
   the released 576-K layout. The incumbent kernel and its pins stay for the A8 path; nothing in
   this plan removes or repins the current combine.

Baseline invariant: this adds a manifest entry, so it was one commit that moved the source, the
pins, the manifest (regenerated with the pinned image) and the tests that count it. Never hand-edit
a byte count; regenerate and review the diff, exactly as scripts/regen-kernel-baseline.sh says.

## 5. Not in scope / open questions for the implementer

- The +25.2 GiB/rank repack remains rejected; this design is the only alternative that does not need
  it.
- The plan's scale placement was confirmed to be a separate tensor (section 1); the design keeps it
  local to the two pointer expressions.
- No device measurement exists for this kernel until a GPU window runs m25_moe_stored; the variant
  must not be wired into the token path before that harness is green. Ordering is fixed: the next
  session's first GPU action is m25_moe_stored against the released stored layout, unchanged, with
  its printed bit-exact bar; chain wiring (item 9) is considered only after that run is green.

## 6. Task-6 implementation record

- Kernel: kernels/b12x_mxfp4_moe_stored_sm121.cu, symbol b12x_mxfp4_moe_stored_sm121. Pinned cubin:
  11,544 B, sha256 1dbfd764f11e7378ecc1790b8f5bca2e97871c6a6ad3092cd16c6a5df767ba08, nvcc V13.0.88
  (CUDA 13.0.3), sm_121a; cuobjdump: REG 39, STACK 0, SHARED 14,848 B, LOCAL 0; PTX membar lines 0.
- Host twin and de-interleave unit tests: crates/sparkflash-kernels/src/mxfp4_ref.rs (module
  stored_tests): the stored and padded twins produce bit-identical f32 accumulators and finalised
  rows on the same bytes, with zero AND nonzero padding tails; check_stored_moe_layout rejects the
  interleaved 306-byte row and the padded 320/20 contract; check_moe_layout rejects the stored
  contract.
- Baseline: baseline/kernel_baseline.txt has twenty entries and scripts/check-kernel-baseline.sh
  reports "all kernels match the recorded baseline (V13.0.88, ...)" in the pinned image.
- Harness: examples/m25_moe_stored.rs, committed; aarch64 release binary sha256
  e59a99a9adcfede19a25e494e4c6630d7213104f579e59e771116fdd1f7b1104 at
  target/aarch64-unknown-linux-gnu/release/examples/m25_moe_stored (3,458,160 B). NOT YET RUN on the
  node: no numerical result and no timing claim is made by this record.
- Items 1-8 are implemented and host-gate green (fmt, clippy -D warnings, workspace tests,
  examples build, baseline check). Item 9 (chain wiring) is deliberately not done.
