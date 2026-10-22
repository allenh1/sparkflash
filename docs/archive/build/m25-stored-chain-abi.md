# Chain handoff: the stored-layout combine ABI (task-6, for task-7 wiring)

Status: committed on branch task/m25-stored (pin commit 49a3389, harness commit 22c08a9; 49a3389 is
merged into integration). The kernel and its pins are host-gate green; the kernel has NOT run on a
GPU yet. This page is the exact ABI device-lead asked for, so the chain can select the stored cubin
behind a LayoutSelector without reading the kernel source. Where this page and the design page
disagree, this page and the kernel win - in particular the stored layout is TWO SEPARATE tensors
(288-byte code rows, 18-byte scale rows), not interleaved 306-byte rows; 306 = 288 + 18 is the cost
of one output row across the two tensors.

## 1. Symbol, cubin, harness

| | |
|---|---|
| kernel symbol | b12x_mxfp4_moe_stored_sm121 |
| cubin | b12x_mxfp4_moe_stored_sm121.cubin (11,544 B, sha256 1dbfd764f11e7378ecc1790b8f5bca2e97871c6a6ad3092cd16c6a5df767ba08) |
| pinned with | nvcc V13.0.88 (CUDA 13.0.3, image nvidia/cuda:13.0.3-devel-ubuntu22.04), arch sm_121a |
| resource usage | REG 39, STACK 0, SHARED 14,848 B, LOCAL 0; PTX membar lines 0 |
| ABI pin module | crates/sparkflash-kernels/src/mxfp4_moe_stored_abi.rs (KERNEL, SOURCE, CUBIN_BYTES, CUBIN_SHA256, CUBIN_MEMBAR_LINES, TOOLKIT_RELEASE, STORED_PARAMS, declaration()/declaration_matches()) |
| typed launcher | crates/sparkflash-kernels/src/mxfp4_moe_stored_launch.rs (KERNEL_ID = KernelId::Mxfp4MoeStored, grid(), load(), launch(); params type IS mxfp4_moe_launch::Mxfp4MoeParams) |
| device harness | examples/m25_moe_stored.rs -> aarch64 binary target/aarch64-unknown-linux-gnu/release/examples/m25_moe_stored (3,458,160 B, sha256 e59a99a9adcfede19a25e494e4c6630d7213104f579e59e771116fdd1f7b1104) |

The cubin must be built WITHOUT -DSPARKFLASH_ENABLE_GPU_TRACE for this harness/launcher path; with
the trace compiled out the trace_ring argument is not dereferenced, so the harness passes 0. If the
chain builds the cubin WITH the trace enabled, trace_ring must point at a real GpuTraceRing.

## 2. The argument list (10 words, in order)

It is the frozen incumbent combine declaration's list. The stored kernel is a drop-in for the same
call site; the pin module pins the two lists together BY IDENTITY (STORED_PARAMS =
mxfp4_moe_launch::PARAMS), so they cannot drift.

| word | name | C type | const/restrict | meaning |
|---|---|---|---|---|
| 0 | act_fp8 | const uint8_t * | const, __restrict__ | A8 activation operand base: [tokens*6][640] E4M3 bytes |
| 1 | act_scales | const uint8_t * | const, __restrict__ | A8 activation ue8m0 grid: [tokens*6][20] bytes |
| 2 | w_fp4 | const uint8_t * | const, __restrict__ | stored codes: [experts][5120][288] packed FP4 bytes |
| 3 | w_scales | const uint8_t * | const, __restrict__ | stored scales: [experts][5120][18] ue8m0 bytes |
| 4 | expert_ids | const uint32_t * | const, __restrict__ | [tokens*6] routed expert index per (token, slot) |
| 5 | slot_acc | float * | non-const, NO restrict | in/out f32 accumulator: [tokens*6][5120], the kernel ADDS into it |
| 6 | tokens | uint32_t | value | number of tokens; the grid is (tokens, 160, 1) |
| 7 | trace_ring | GpuTraceRing * | pointer | trace ring; ignored (null is fine) when the cubin was built without the trace |
| 8 | layer_id | uint32_t | value | layer index, trace only |
| 9 | step_seq | uint64_t | value | step sequence, trace only |

Launch: grid (tokens, 160, 1), block (256, 1, 1), no dynamic shared memory, no workspace, no
metadata/param struct. The launcher refuses tokens == 0 with a named error instead of launching a
zero-sized grid. Slot ordering of the A operand and expert_ids is the native top-6 ascending
expert-id order, exactly as the incumbent.

## 3. Stored layout constants and indexing

| constant | value | meaning |
|---|---|---|
| K_VALUES | 576 | real K of one TP rank: 2304 / 4 |
| QUANT_BLOCK | 32 | MX block width along K |
| K_BLOCKS | 18 | 576 / 32; the kernel walks exactly these, no padding block |
| FP4_PER_BYTE | 2 | packed FP4 |
| CODE_ROW_BYTES | 288 | 576 / 2, code stride per output row |
| SCALE_ROW_BYTES | 18 | one ue8m0 byte per MX block, scale stride per output row |
| ROW_BYTES_TOTAL | 306 | 288 + 18, the cost of one output row across the two tensors (not a stride) |
| ACT_ROW_BYTES | 640 | unchanged A8 activation row |
| ACT_SCALE_ROW_BYTES | 20 | unchanged A8 activation scale grid; blocks 0..17 read, 18..19 ignored |
| TILE_D | 32 | hidden columns one block writes |
| K_GROUPS | 8 | parallel K groups per block |
| THREADS | 256 | TILE_D * K_GROUPS |
| GRID_Y | 160 | 5120 / 32 |
| TOP_K | 6 | routed experts per token |

Indexing, stated exactly (row = expert * 5120 + column, column in 0..5119):

- code byte pair for K values 2*i and 2*i+1 of that row lives at w_fp4 + row * 288 + i;
- ue8m0 scale byte for K block b of that row lives at w_scales + row * 18 + b;
- the two base pointers are independent allocations; the kernel never computes w_fp4 + row * 306 and
  never reads a scale from the code tensor.

The activation side is deliberately unchanged A8: the chain quantises the activation before the
weight layout is known, so switching the weight layout must not re-stage it. The kernel reads only
act columns 0..575 and act scale blocks 0..17 of each 640/20 row.

## 4. Host-side preconditions to assert before launch

1. The staged arena announced layout equals mxfp4_ref::stored_moe_layout() - call
   check_stored_moe_layout() and refuse on Err; its message names the offending stride. This also
   rejects a padded 320/20 staging and an interleaved 306-byte staging.
2. The cubin bytes/digest equal CUBIN_BYTES/CUBIN_SHA256 from mxfp4_moe_stored_abi (the harness
   refuses before launching; the chain catalog should do the same once, at load).
3. w2.weight tensor is [5120,1152] I8 and w2.scale is [5120,72] F8_E8M0 in the checkpoint; the
   per-rank slice is the leading 288 code bytes and 18 scale bytes of each row. The harness asserts
   dtype and shape and slices rows without padding; the chain should stage the same way.
4. slot_acc is allocated and zeroed ONCE per step by the caller; each rank adds its partials. The
   kernel does not zero it.
5. tokens >= 1; tokens*6 routed rows in act_fp8/act_scales/expert_ids; slot_acc length
   tokens*6*5120. Expert ids are < experts and are the same ids the router produced.
6. If the cubin was built with the trace enabled, trace_ring must be a valid GpuTraceRing; with the
   default (trace compiled out) pass 0.

## 5. Refusal side if the cubin is absent or wrong

- KernelId::Mxfp4MoeStored resolves the symbol; a missing or stale cached cubin is a named
  load error (KernelId::cubin_names() -> ["b12x_mxfp4_moe_stored_sm121"], and the driver's
  cuModuleGetFunction failure names the symbol), never a silent fallback to the incumbent.
- The baseline manifest pins the cubin's bytes and sha256 on its own line; the baseline check
  script recompiles and compares all twenty kernels in the pinned image and prints
  "all kernels match the recorded baseline (V13.0.88, ...)".
- The ABI module pins the declaration and the geometry static_asserts at compile time (tests fail
  on drift); the launcher and the host twin cross-check the same constants at test time.
- The harness refuses a cubin whose size or sha256 differs from the pin before any device call.
- The chain-side LayoutSelector should key on the bound plan's layout, not on a runtime probe:
  select stored when check_stored_moe_layout(bound layout) is Ok, incumbent A8 when
  check_moe_layout is Ok, and refuse loudly when neither matches.

## 6. Expected pass lines from m25_moe_stored

Pass bar, printed BEFORE any number (default run: --tokens 1 --experts 64 --layer 0):

    pass bar: PASS requires the device rank accumulator bit-exact at 30720/30720 f32
             partials, the finalised rows bit-exact at 5120/5120 bf16 values, and a second launch
             byte-identical to the first

(the bar is one print with two embedded newlines; runner scripts should grep "pass bar:").

Then, with the path and device line varying but the measured lines exact for a green run:

    === stored combine slice ===
    layer 0 experts 64 tokens 1 top_k 6
    cubin: <path>/b12x_mxfp4_moe_stored_sm121.cubin (11544 B, sha256 1dbfd764f11e7378ecc1790b8f5bca2e97871c6a6ad3092cd16c6a5df767ba08)
    stored weight code bytes: 94371840 (64 experts x 5120 rows x 288 B, no padding)
    stored weight scale bytes: 5898240 (64 experts x 5120 rows x 18 B, separate tensor)
    cpu reference cost: MoeCost { macs: 17694720, padding_macs: 0, weight_bytes: 8847360, activation_bytes: 17694720, scale_bytes: 1105920 }
    rank accumulator: 30720/30720 f32 partials bit-exact against mxfp4_ref::moe_combine_rank_stored
    elements compared: 5120
    exact bf16 matches: 5120 (100.0000%)
    max abs delta: 0e0
    second launch byte-identical: yes
    m25_moe_stored: PASS (30720/30720 f32 partials, 5120/5120 finalised rows, second launch deterministic)

For --tokens 8 the same lines read 245760/245760 f32 partials, elements compared 40960,
exact bf16 matches 40960 (100.0000%), and PASS (245760/245760 f32 partials, 40960/40960 finalised
rows, second launch deterministic). On failure the last line is "m25_moe_stored: FAIL" and the
process exits nonzero.

## 7. How to run the first window action

    # build the cubin in the pinned image (on the dev host)
    docker run --rm --user "$(id -u):$(id -g)" -v <worktree>:/w -w /w \
      nvidia/cuda:13.0.3-devel-ubuntu22.04 sh -c 'export PATH=/usr/local/cuda/bin:$PATH; \
        nvcc -std=c++17 -cubin -arch=sm_121a -I kernels \
          kernels/b12x_mxfp4_moe_stored_sm121.cu -o out/b12x_mxfp4_moe_stored_sm121.cubin'

    # copy to the node (with the aarch64 harness) and run
    m25_moe_stored --checkpoint \
      /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
      --layer 0 --experts 64 --tokens 1 --cubin out/b12x_mxfp4_moe_stored_sm121.cubin

The harness aarch64 binary for this run is the one listed in section 1 (sha256 e59a99aa...). It
makes no timing claim; a measured run is a separate window protocol.

## 8. Not in scope here

Chain wiring (item 9 of the plan) is a separate commit on the device chain, after this harness is
green in a window. Do not flip the token path to the stored kernel on the strength of this page
alone: it documents the ABI and the pins, not a device result.
