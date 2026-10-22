# Phase 3 precision path: BF16-activation MXFP8 projection kernels (D37)

This is the build record of batch `batch/2026-10-19-25` (off main `7edda55`): the two D37
precision-first projection kernels, their pure-Rust twin, their typed launches, their pinned cubins
and the GPU harness `m21_bf16a_projections`. The kernels are **host-tested and pinned but UNWIRED**;
the GPU device-vs-twin comparison is **PENDING window b69** (no device number is recorded in this
document). No device-chain call site changes in this batch, so `REQUIRED_KERNELS` stays 9 and the
hybrid device chain is untouched.

What D37 chose is quoted, not re-decided here: [DECISIONS.md D37](../../DECISIONS.md) records that the
FP8-native path is stage-faithful but not token-parity, that the gap enters at the attention
projections' activation quantisation, and that a precision-first activation path is one of the two
options the user could choose. The user approved that path; this batch is its kernel half.

## 1. Why BF16 activations, stated as arithmetic

The five attention projections are block-scaled GEMMs. The checkpoint supplies the E4M3 weight codes
and one ue8m0 scale per 32x32 block; the device chain supplies the activations. In the current
MXFP8 path those activations are themselves quantised to E4M3 with one ue8m0 scale per 32 values
(`m4_activation_quant_sm121.cuh`, D23's ceil rule). That quantiser is the only difference between what
the device computes and what the f32 fixture reference computes: the fixture dequantizes the same FP8
weights and accumulates in f32, so **the weight codes are part of the accepted arithmetic and the
activation quantiser is not**.

The arithmetic cost of that quantiser is systematic, not white noise. A shared ue8m0 scale is
`2^(floor(log2(amax)) - emax)`, so every element of a 32-value block is first divided by a
power-of-two block scale and then rounded to E4M3, whose 3 mantissa bits give a relative step of
`2^-3` of the value's binade and a worst-case rounding of `2^-4` of the block maximum. Because one
scale covers all 32 products, the rounding errors correlate within a block; a 5120- or 4096-wide dot
product therefore sees a per-block gain error rather than errors that average away. D37 measured the
consequence at the router: the first expert-set flip is layer 3, 36/40 layers differ at position 0,
and the criterion run produces 0 of 9 tokens (8060 vs 491). D37's attribution measurement is the
reason to remove this quantiser rather than any weight handling: the MXFP8 kernel reproduced
`mxfp8_ref` exactly on the launch bytes (0/32,768, max_abs 0.0), while the `wq_b` input latent already
differed by 5.19e-2 and raw q by 2.94e-1 against an expected bf16 quantisation delta of 6.18e-2
(b58/b59, b60).

The precision-first path therefore changes exactly one operand's dtype. The contract of the two new
kernels is

```text
out[c] = bf16( sum_k bf16_a[k] * (e4m3(w[c,k]) * ue8m0(scale[c/32][k/32])) )
```

with **one f32 accumulator per output**, K ascending in 32-value blocks and ascending inside a block
(the order the existing MXFP8 kernel uses and `mxfp8_ref` already pins), and **one bf16 round at the
end** (`__float2bfloat16_rn`, round-to-nearest-even). bf16 -> f32 is exact, so the activation enters
the sum unrounded; each product is two f32 multiplies (weight decode x scale, then x activation), and
every add is f32. Weights, weight scales, tile geometry, grid, thread count and the trace ABI are
byte-for-byte the MXFP8 kernels' contract; only the activation buffer's dtype and the missing
activation scale buffer differ.

Why this is the "exact" precision path rather than merely a higher-precision one: the device chain
already holds these projection inputs as bf16 (embedding rows, RMSNorm outputs, the q latent, the
attention head outputs). Sending the same bf16 values through this GEMM removes a rounding that the
f32 reference never performs, so the projection's own arithmetic becomes the reference's
dequantize-and-accumulate arithmetic up to the reference's accumulation order and the single final
bf16 output round. The remaining device-vs-fixture difference is a chain/ordering question, not a
quantiser question; it is what the b69 criterion run has to measure after wiring.

## 2. What this batch adds

| file | what it is |
| :--- | :--- |
| `kernels/b12x_bf16a_mxfp8w_dense_sm121.cu` | BF16-activation dense projection; covers wq_a, wq_b, wkv, wo_b |
| `kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu` | BF16-activation grouped wo_a (8 groups x 1024 over 4096) |
| `crates/sparkflash-kernels/src/bf16a_mxfp8w_ref.rs` | the pure-Rust twin, i.e. the definition of the two kernels' arithmetic |
| `crates/sparkflash-kernels/src/bf16a_mxfp8w_launch.rs` | typed launch: pinned declaration, param array, 64x64 grid |
| `crates/sparkflash-kernels/src/bf16a_mxfp8w_wo_a_launch.rs` | typed launch: pinned declaration, param array, (groups, 4, 1) grid |
| `crates/sparkflash-kernels/examples/m21_bf16a_projections.rs` | GPU harness: pins, device-vs-twin exactness, determinism |
| `crates/sparkflash-kernels/baseline/kernel_baseline.txt` | 13 -> 15 entries: the two new cubins are pinned |
| `crates/sparkflash-kernels/build.rs` | `PROBED_KERNELS` 13 -> 15, so both kernels are clang/nvcc-probed on every build |
| `crates/sparkflash-kernels/src/cuda_driver.rs` | `KernelId::Bf16aMxfp8wDense`, `KernelId::Bf16aMxfp8wWoA`; `ALL` 12 -> 14 |
| `crates/sparkflash-kernels/scripts/{check,regen}-kernel-baseline.sh` | the list and header move to fifteen entries |
| `crates/sparkflash-kernels/tests/kernel_baseline.rs`, `src/baseline.rs` | the count ratchets 13 -> 15 and 12 -> 14 |

Both kernels are self-contained translation units: they include only `m4_common_sm121.cuh`, which
carries the B12x index helpers, the codecs and the trace ABI. No new header, no new layout, no change
to `m4_abi`'s frozen six-kernel array (`m4_abi.rs` is untouched, so D26's size ratchet is not
reopened).

## 3. ABI and geometry (identical to the MXFP8 twins)

Dense (`b12x_bf16a_mxfp8w_dense_sm121`, 10 argument words):

```text
const __nv_bfloat16 *a_bf16   [rows][inner]        (no activation scale buffer)
const uint8_t       *w_fp8    B12x tiled E4M3
const uint8_t       *w_scales B12x tiled ue8m0
__nv_bfloat16       *out      [rows][cols]
uint32 rows, cols, inner;  GpuTraceRing *trace_ring; uint32 layer_id; uint64 step_seq
grid (ceil(rows/64), ceil(cols/64), 1), block 256, one 64x64 tile/block, 4x4 per thread
```

wo_a (`b12x_bf16a_mxfp8w_wo_a_sm121`, 8 argument words):

```text
const __nv_bfloat16 *attn_bf16 [groups][group_inner]   (group_inner = 4096)
const uint8_t       *w_fp8     B12x [groups][o_lora_rank=1024][group_inner=4096]
const uint8_t       *w_scales  B12x, two group strides (weight bytes != scale bytes)
__nv_bfloat16       *lo_out    [groups][o_lora_rank]
uint32 groups; GpuTraceRing *trace_ring; uint32 layer_id; uint64 step_seq
grid (groups, 1024/256 = 4, 1), block 256, one 256-channel tile/block, one output per thread
```

Both use `TRACE_SCOPE_DENSE_MXFP8` with the MXFP8 kernels' `payload_meta` (tile elements x 2 bytes),
so a chain that already records these stages needs no trace change. The typed launch modules pin each
declaration against its own kernel source (`include_str!`) and check every parameter's name, base
type, pointer-ness, const-pointee and `__restrict__`, with a mutation test proving the pin is not
vacuous.

## 4. Pins

Toolkit and flags are the repository's: nvcc `V13.0.88` (CUDA 13.0.3), `-std=c++17 -arch=sm_121a
-cubin -I kernels`, compiled on the head node by
`scripts/regen-kernel-baseline.sh` with `SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc`.

| kernel | cubin bytes | cubin sha256 | PTX membar lines |
| :--- | ---: | :--- | ---: |
| `b12x_bf16a_mxfp8w_dense_sm121` | 11,736 | `a2e0a63c8b78e3fe9117d7c12b06353bbc46af7c16019cf71e572c93806eff7e` | 0 |
| `b12x_bf16a_mxfp8w_wo_a_sm121` | 8,976 | `d11a8610ff557617116c437b0bc7a0367f66a8069ccfc97373433ebd99ffd249` | 0 |

Every pre-existing pin is unchanged by this batch (the same toolkit recompiled it to the same
bytes), which is the check that the edit did not perturb a kernel that was already pinned.

Compile lines for a Lead-owned window (either the committed `-cubin` form or, for the harness, the
same command):

```sh
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_bf16a_mxfp8w_dense_sm121.cu \
    -o <cubin-dir>/b12x_bf16a_mxfp8w_dense_sm121.cubin
nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu \
    -o <cubin-dir>/b12x_bf16a_mxfp8w_wo_a_sm121.cubin
```

Measured on the head node in this batch, `SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc
crates/sparkflash-kernels/scripts/check-kernel-baseline.sh` printed all fifteen `ok` lines and ended
`all kernels match the recorded baseline (V13.0.88, cuda_13.0.r13.0/compiler.36424714_0)`, exit 0
(see the command and environment below).
The Rust-side ratchets were updated with it: `tests/kernel_baseline.rs` pins the script list, the
manifest, the on-disk kernel files and `build.rs`'s probe table to the same fifteen names, and
`src/baseline.rs` parses fifteen pins and requires each of the new names to be present.

## 5. The twin, and what "bit-exact" means here

`src/bf16a_mxfp8w_ref.rs` is the definition of the kernels' arithmetic, in the same sense
`mxfp8_ref` is the definition of the MXFP8 kernels'. It consumes the same bytes the kernels consume -
bf16 bit patterns for `a`, B12x-packed E4M3 codes and ue8m0 scale bytes for the weights - and it
writes bf16 bit patterns. Its loop is the kernel's loop: K ascending in 32-value blocks, ascending
inside a block, the weight decoded as `fp8(w) * ue8m0` before the multiply by the activation, and one
`bf16_bits` rounding at the end.

Its tests: a hand-computable 32-product golden case (2^-3 and 2^-4), a non-vacuity case proving a
bf16 activation that E4M3 cannot represent is not re-quantised (0.1 bf16 goes through unchanged), a
cross-check against a naive dequantize-and-accumulate loop on a ragged multi-tile shape, a test that
the pinned ascending accumulation order is observable and used, and a grouped-wo_a test proving a
group is the dense GEMM over its own slice with non-vacuity for both the weight and scale strides.
A layout test pins every index to `mxfp8_ref`'s helper, so the precision path cannot quietly grow a
second B12x formula.

## 6. The harness m21, its class and its bar

`examples/m21_bf16a_projections.rs` is the only thing in this batch that needs a GPU. It:

1. refuses both cubins unless size and sha256 equal the baseline pins, before any device allocation,
   and cross-checks its own constants against `baseline::pin`;
2. reads the released layer-0 weights and scales through the measured `m7_layer0_shapes` table
   (wq_a [1280, 5120], wq_b [32768, 1280], wkv [512, 5120], wo_a [8192, 4096]) and repacks them with
   `pack_b12x` / `pack_wo_a` - declared, value-preserving index permutations, with a unit test that
   every packed byte is the checkpoint byte at the B12x index;
3. builds the activation operands from released `embed.weight` BF16 rows (a declared stand-in at the
   real widths: full 5120-wide row for wq_a/wkv, first 1280 codes for wq_b, first 512 codes repeated
   eight times per group for wo_a - the K2->K3 and K4->K5 seams are explicitly *not* claimed);
4. launches the two kernels through the typed params and compares **every output BF16 code** with the
   twin on the same packed bytes.

Declared class and bar, printed by the harness before any number it measures: **exact arithmetic, not
a tolerance**. Bar: `exact == total` (0 mismatched codes of `rows x cols` for each dense projection,
`groups x o_lora_rank` for each wo_a position), maximum |device - twin| in BF16 value units = 0.0, and
a second launch byte-identical. A difference of one code is a finding.

**No GPU number is recorded in this document.** The measurement is window b69, owned by the Lead;
until its output is in hand the harness is built and staged, not run.

Build and run (Lead-owned window). The binary is built on the dev box by
`./scripts/build-aarch64.sh --example=m21_bf16a_projections`, copied to the node, and the cubins are
the two pinned compiles from section 4. Staged for the Lead in this batch:

```text
binary  /home/user/phase2-evidence/window-2026-10-18/bin/m21_bf16a_projections_cb4d6034_aarch64
        3,800,640 B  sha256 cb4d6034e8c609786c99ecc2f39cc2d9b722d2ffd5792623509bc707113f48be
cubins  /home/user/phase2-evidence/window-2026-10-18/cubins/b12x_bf16a_mxfp8w_dense_sm121.cubin
        /home/user/phase2-evidence/window-2026-10-18/cubins/b12x_bf16a_mxfp8w_wo_a_sm121.cubin
        (sizes and shas as pinned above; the harness refuses any other file)
```

Exact commands (checkpoint is the released 1,017 snapshot; the harness reads it through the
`b2_harness` `Snapshot` reader like m12/m13/m18):

```sh
BIN=/home/user/phase2-evidence/window-2026-10-18/bin/m21_bf16a_projections_cb4d6034_aarch64
SNAP=/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
CUB=/home/user/phase2-evidence/window-2026-10-18/cubins
"$BIN" --checkpoint "$SNAP" --cubin-dir "$CUB" --tokens 8
"$BIN" --checkpoint "$SNAP" --cubin-dir "$CUB" --tokens 130
```

The harness prints its shape contract before measuring: dense wq_a 1280x5120, wq_b 32768x1280,
wkv 512x5120 (out x reduction), and wo_a grouped as 8 groups x (1024 out x 4096 reduction) from
released [8192 x 4096] weights and [256 x 128] scales, one launch per position. Expected exact-code
totals at `--tokens N`: wq_a N*1280, wq_b N*32768, wkv N*512, and 8*1024 per wo_a position (so at
N=8: 10,240 / 262,144 / 4,096 per dense projection and 8,192 x 8 wo_a codes; at N=130: 166,400 /
4,259,840 / 66,560 and 8,192 x 130).

The expected pass line is `m21: PASS (<n> checks; tokens <t>; two pinned cubins)` with every check
`ok`; the wo_a loop runs one launch per token, so `--tokens 130` is 130 wo_a launches plus three
dense projections (each launched twice, because the determinism bar is checked on every projection). Re-run with `--tokens 8` first: it is the cheaper, more diagnostic window.

## 7. How it will replace the MXFP8 path in the device chain (planned, not done)

The wiring is deliberately a separate, Lead-reviewed change on the unmerged hybrid-sequence branch,
because it touches the chain and this batch must not move `REQUIRED_KERNELS`. The plan:

1. **Catalog and loader.** Add the two cubins to the device catalog with these pins; the
   `KernelId` variants already exist, so `cuda_driver::cubin_path`/the harness resolve them now.
2. **Call sites.** In the attention block, replace the four dense MXFP8 launches (K2 wq_a, K3 wq_b,
   K4 wkv, and the dense wo_b stage) with `bf16a_mxfp8w_launch` and the grouped K6 wo_a launch with
   `bf16a_mxfp8w_wo_a_launch`. The argument arrays are the MXFP8 ones minus the activation scale
   pointer; the grid/thread geometry is unchanged.
3. **Remove the five activation quantisers.** The D8 activation-quantise step in front of those five
   GEMMs disappears; the bf16 activation buffer that is already staged is passed directly. Weight
   packing, scale upload and output staging are unchanged.
4. **Pins and admission.** The chain's required-kernel list gains the two entries (9 -> 11) only in
   that wiring change, together with its own baseline/catalog tests.
5. **Measure, in order.** (a) the router diagnostic window (the `SPARKFLASH_ROUTER_TWIN=1`
   position-3 expert-set comparison that b68 used); (b) the full 130+8 criterion run against the f32
   fixture. Only (b) can move criterion 1.

This batch's kernel work is a prerequisite for all five steps and changes none of them.

## 8. Host measurements in this batch vs the pending GPU measurement

Measured here (host tests and compile only; reproducible on the dev box and, for the compile, on
the head node):

- both kernels compile under nvcc V13.0.88 for sm_121a to the pinned cubins above (15/15 baseline
  check passes on the node, exit 0);
- both kernels and the combined fifteen-kernel probe parse with clang's CUDA frontend against the
  stub headers on the dev box, including every `static_assert`;
- the typed declarations are pinned with non-vacuity mutation tests, and the launch param arrays are
  pinned to the declaration order;
- the new modules' tests pass (16: the twin's golden, re-quantisation-control, accumulation-order,
  grouped-stride and layout tests, plus the five declaration/grid tests in each launch module);
- the m21 harness compiles and its own tests pass without a GPU.

PENDING window b69 (no device number is recorded in this document until the Lead sends the m21
output):

- GPU execution of either kernel, and therefore device-vs-twin bit-exactness (m21's bar);
- determinism on device;
- the router-input drift reduction and the criterion run after wiring;
- latency and bandwidth of the precision path - its cost is the throughput/memory trade D37 names,
  and this batch makes no performance claim. A BF16 activation operand is twice the bytes of the
  E4M3 activation stream; the weight stream is unchanged.

## 9. Risks and honest limits

- **FMA contraction.** The twin assumes each product is rounded before the add. The MXFP8 twin made
  the same assumption and m12 measured K2 1,280/1,280, K3 32,768/32,768 and K4 512/512 exact on the
  device, so the same expression shape has compiled without contraction for this loop before. m21 in
  window b69 is the check for the new kernels; until its output lands, this is a plan, not a measured
  equality.
- **No tensor-core path.** Both kernels are the scalar reference path, as their MXFP8 twins are; the
  block-scaled FP8 MMA would accumulate in the hardware's order and would need a different declared
  comparison class.
- **The activation stand-in.** m21 measures arithmetic on released weights with declared BF16
  stand-ins at the real widths, not the released activations (the checkpoint stores none for this
  stage). The device chain's real bf16 activations are what the b69 criterion run will use.
- **The host-f32 experiment is complementary, not duplicated.** A separate task measures a host-f32
  projection mode in the chain to prove the attribution. This batch provides the production kernel
  form of the same arithmetic; neither result implies the other.
- **No decision edited.** D37 already records the user's choice; this document is the build/
  measurement record for the kernel half, and it deliberately does not restate or alter D37.

## 10. Reproduce the local checks

```sh
cargo test -p sparkflash-kernels --lib bf16a          # twin + launch tests
cargo test -p sparkflash-kernels --test kernel_baseline  # count/list ratchets (no nvcc needed for the walk test)
cargo test -p sparkflash-kernels --example m21_bf16a_projections  # harness tests, no GPU
cargo fmt --check && cargo clippy -p sparkflash-kernels --all-targets -- -D warnings
```

The node command that proves the pins is section 4's; it needs `SPARKFLASH_NVCC` set because the
node does not put nvcc on `PATH`.
