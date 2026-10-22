# Phase 3 m15: the mHC post-attention merge (hc_post, K8/MEGA_MHC) in isolation

**Status: window b20 PASSED on device at both shapes (8/8 and 130/130 positions) on binary
`m15_mhc_post_14a26701_aarch64` built from 4379166 - the recorded window for m15, and the one that
carries the corrected codes/scales twin. Window b19 passed earlier on the pre-correction twin and
stays recorded with that caveat; b15 and b16 failed at harness-side quantities, were fixed, and are
recorded verbatim below. Phase 2 exit criterion 1 remains OPEN - this is a stage measurement, not a
token or a chain.**
This is the m15 deliverable: the layer-0 post-attention mHC merge, the complement of m11's entry,
at the real layer-0 geometry, in the m10-m13 window-harness pattern. The subject is one fused
`mega_mhc_sm121` call at the post seam: it consumes the entry stage's residual streams X_0, the
attention block output Y_0 and the entry stage's 24 published coefficients B/A/C, and writes the
post-mix residual update X_1 = B X_0 + C Y_0, the next block-input capture X_hat_1 = A X_1, its FP8
quantisation and the 24 coefficients the fused kernel publishes from the updated streams.

The harness is [m15_mhc_post.rs](../../../crates/sparkflash-kernels/examples/m15_mhc_post.rs). The
chain stage it covers is K8 "post-mix mHC: `hc_post` (`mhc.rs:378`), kernel MEGA_MHC" in
[device-forward-chain.md](device-forward-chain.md) section 2.2 item 6. The chain wiring itself is
M3's and is not in this branch; see "The chain seam that does not exist yet" below.

## Subject and launch shape

| item | value |
| :--- | :--- |
| kernel | `mega_mhc_fused_bf16_sm121` (`kernels/mega_mhc_sm121.cu`) |
| pin | 70,312 B, sha256 `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618`, 4 PTX membar lines |
| grid / block | (4, 1, 1) blocks of (128, 1, 1) threads: one block per 1,280-value hidden tile |
| arguments | the pinned 19 words through `mega_mhc_launch::MegaMhcParams`: 16 device pointers in declaration order (x, y_prev, b_prev, c_prev, a_prev, predictor, hc_scale, hc_base, rms_weight, xhat_fp8, xhat_scales, hc_stats, hc_sync, hc_coeffs, hc_rms_out, xhat_normed), then trace_ring (0, trace-disabled cubin), layer_id (0) and step_seq |
| post call's operand roles | x = X_0 (the entry write), y_prev = Y_0 (the attention block output), b_prev/c_prev/a_prev = the entry stage's coefficient readback (comb/post/pre in the kernel's B/A/C row order) |
| geometry | hidden 5,120, four streams, 24 predictor rows, 16 FP8 codes per block, 160 ue8m0 scales per capture |
| launches | one launch per layer-0 position |

The kernel has ONE entry, so the post call is the same 16 non-null device pointers and the same 19
words as m11's entry call; what differs is which seam bytes the host puts in them. The harness
declares that shape rather than inferring it, and re-uploads y/b/c/a per position because the
coefficient readback and the attention output are per position.

## Provenance and the declared seams

Every weight, bias and norm byte is read verbatim from the released checkpoint
`layers.0.hc_attn_fn` (F32 [24, 20480], permuted into the kernel's B/A/C row order by
`released_predictor_into_kernel_rows`, no averaging), `layers.0.hc_attn_base` (F32 [24], into
kernel row order), `layers.0.hc_attn_scale` (F32 [3], released pre/post/comb order) and
`layers.0.attn_norm.weight` (F32 or BF16 [5120], the RMS pre-norm weight). The cubin is verified by
size and sha256 against the baseline manifest before the first DEVICE allocation (reading the cubin
into a Vec is itself an allocation), exactly as m11 does.

The mHC kernel is not an all-resident layer; this harness is a stage measurement, not a chain. Three
inputs are **host-provided and declared**, because the chain cannot produce them yet:

| seam | what it is here | status |
| :--- | :--- | :--- |
| X_0 | the entry stage's residual write, host twin on the released embed row t (four copies) through the HOST-BUILT layer-0 boundary seed (B = identity, C = 0, A = [1,0,0,0], Y_-1 = 0 - declared seam values, not bytes read from the checkpoint); the harness checks the bytes it passes against the kernel's own step-1 write, not against the reference's first return value | host seam, declared |
| B/A/C | the entry stage's 24 published coefficients, host twin of the entry call (`mega_mhc_ref::mega_mhc_layer_with_scale`) on the same X_0 and the same released tensors - the bytes a device-to-device coefficient handoff would carry | host seam, declared; the chain handoff is **not** assumed |
| Y_0 | the attention block output, a **declared stand-in** `bf16(0.5 * embed row t - 0.25 * embed row (t+1) % tokens)`, built from released embed bytes because no released attention output exists and this harness computes no attention | declared stand-in, NOT a measured attention output |

Device-produced (measured) outputs are the ones the kernel writes: the post-mix residual streams,
the 24 coefficients, the f32 block-input capture, the 160 ue8m0 scale bytes, the E4M3 codes and
1/rms(X_hat_1).

### The declared mapping from released bytes to the post call

The mapping is derived from the kernel's own header and argument list, and it is stated here as a
declaration the harness checks, not as an inference at window time. The kernel header says exactly
what its one entry does, in order: (1) `X_l <- B_{l-1} X_{l-1} + C_{l-1} Y_{l-1}`, updated in
registers and written back to the x buffer once; (2) `X_hat_l = A_{l-1} X_l`, from those same
registers; (3)-(4) the released H(X_l) coefficients, the input RMS pre-norm and the activation
quantisation of X_hat_l. The arguments are named accordingly in `mega_mhc_launch::PARAMS`:
`x` is the [n][d] residual streams (updated in place), `y_prev` the [d] previous block output,
`b_prev`/`c_prev`/`a_prev` the previous call's 24 coefficients.

* **Entry call** (the m11 stage): x = X_-1 (four copies of the released embed row t), y_prev = 0,
  b_prev/c_prev/a_prev = the released layer-0 boundary seed (B = identity, C = 0, A = [1,0,0,0]).
  Its write is therefore **X_0 = X_-1** - the identity on this seed - and it publishes H(X_0).
  The harness forms X_0 with its own `expected_update` on the boundary seed and refuses to run
  unless the bf16 bytes it passes equal that write (window b15's failure was a guard comparing the
  WRONG reference output, see below, not a disagreement about this write).
* **Post call** (this harness): x = X_0 (the same four-copy buffer the entry call would leave in
  place), y_prev = Y_0 (the attention block output, a declared stand-in here), b_prev/c_prev/a_prev
  = the entry call's H(X_0) readback. Its write is **X_1 = B_0 X_0 + C_0 Y_0** plus the next block
  input X_hat_1 = A_0 X_1, its capture/codes/scales and H(X_1).

The mapping is therefore (a) of the two possibilities the Lead's b15 note raises: the entry write is
what the post seam takes as x, and the attention output is what it takes as y_prev. Note that
`mega_mhc_layer_with_scale`'s FIRST return value is not that entry write: it returns the
reference's own `B X_l + C F(X_hat_l)` self-loop (F the dequantised codes it just wrote), a
traffic-accounting quantity built from the coefficients it just formed. The kernel's write for a call
uses the PREVIOUS call's b_prev/c_prev, so the harness must not compare that return to X_0. Window
b15 did, and the guard refused - correctly, for the wrong reason; the fix uses the declared write.

### The chain seam that does not exist yet

[crates/sparkflash-device/src/chain_device.rs](../../../crates/sparkflash-device/src/chain_device.rs)
refuses `ChainStage::MhcPost` with `NotStaged` before the driver, because "the post-mix needs a
coefficient handoff that does not exist yet" (the entry stage's 24 coefficients would have to cross
from the device readback into the post call's b_prev/c_prev/a_prev without a host round trip). This
harness does not assume that transport is solved: it copies B/A/C and Y_0 through the host and says
so. What it does establish is the call's arithmetic and its argument shape on released bytes; when
the handoff exists, the same 16 pointers and 19 words are what the chain stage will fill.

## Comparison class (declared before any number is printed)

The CPU twin is on the same f32 values the device updates from, formed at the kernel's own
INSTRUCTION level. The residual write, the 1/rms, the capture and the expected E4M3/ue8m0 bytes
come from this harness's own `expected_update`, `host_xhat` and `expected_quantisation`; the 24
coefficients come from `mega_mhc_ref::mega_mhc_layer_with_scale` (a bar quantity, because the
kernel reduces the projection in tile order and the reference sequentially). The reference's own
X_hat is a mul-then-add chain, not an fma chain, so the reference's codes/scales are deliberately
NOT the codes/scales twin: expected codes are built from the harness's fma X_hat and its own 1/rms,
in the kernel's order (normed = (xhat * w) * inv_rms, block amax, the released act-quant scale code,
then `fp8_e4m3_encode(normed * (1 / scale))` - a multiply by the reciprocal, as the kernel writes).
The source writes
`acc = c[s] * y; for t: acc += b[s][t] * x[t]`, but that is not what the pinned cubin executes:
nvcc contracts it, per stream, to

    q   = b[s][0] * x[0]          // the ONE plain mul.f32: the first B product is rounded
    acc = fma(c[s], y, q)         // the C Y term is the FUSED one
    acc = fma(b[s][1], x[1], acc)
    acc = fma(b[s][2], x[2], acc)
    acc = fma(b[s][3], x[3], acc)

Compiling this pinned kernel under the pinned toolkit gives 241 `fma.rn.f32` operations, no
`mul.rn.f32` and no `add.rn.f32`, and the arrangement above is the one in that PTX. The twin
reproduces it with one plain product and [f32::mul_add] for the fused terms (and for X_hat). This is
not pedantry: window b16 measured the shadow of getting it wrong - with the C Y term rounded first
and the first B term fused (the source's literal order) the bf16 write X_1 stayed bit-exact while
the f32 capture differed on every position, because bf16 rounding hides a last f32 bit but X_hat
sums those f32 values. The released projection/H/quantisation arithmetic then runs on those updated
f32 values. Per position:

| quantity | class |
| :--- | :--- |
| post-mix residual write X_1 (bf16) | **bit-identical** - both sides run the same arithmetical program on the same operands in the same order |
| block-input capture `xhat_normed` (f32) | **bit-identical**, computed from the same inputs and the device's own 1/rms, so it checks the capture arithmetic itself (b16's only failing quantity; see the window record for the instruction-arrangement cause and the fix) |
| ue8m0 scale bytes | **bit-identical** against the twin's own quantisation of its fma X_hat / 1/rms |
| E4M3 codes | declared bar: >= 99.5% exact and every code within 1, against the twin's own quantisation of its fma X_hat / 1/rms (the same twin as the capture, not the reference's mul-then-add X_hat) |
| 24 coefficients published from X_1 | declared bar: max abs delta <= 1e-4 against `mega_mhc_layer_with_scale`'s sequential projection |
| 1/rms(X_hat_1) | declared bar: <= 3e-4 |
| determinism | two device runs of every position byte-identical across all six outputs |
| f32 control | the kernel's own FP8 codes x scales against its own f32 capture: gap <= 10% of the capture maximum, reported separately and not part of the exactness claim |

The bars are not looseness. The kernel reduces the two sums of squares and the 26 projection
statistics in a fixed tile order (four tiles, a tree reduction per tile, a binary64 combination in
tile order), while the reference accumulates sequentially, so coefficients, 1/rms and the resulting
quantum boundaries are not the same floating-point program. This is the m7 stage-1/m11 class applied
to the post seam, unchanged; the write and the capture are exact where the arithmetic is the same
program.

The harness also prints the cause when E4M3 codes differ: with the scales bit-exact and every code
within one, the difference is the reduction-order 1/rms residual (the twin's sequential 1/rms that
the expected codes were built with, against the device's fixed tile-order 1/rms) moving a value
across an E4M3 boundary, not a quantiser-rule difference.

## Tokens, shapes and the floor

The kernel has no batch dimension - one position per launch, one launch group of the fixed (4,1,1)
grid - so `--tokens N` runs N independent layer-0 positions. N = 8 and N = 130 are both valid
shapes; 130 is the fixture prefill length and nothing about the kernel changes with N. The harness
refuses to run below 32 GiB MemAvailable (`--floor-gib`) at start and at end, and prints the
MemAvailable reading at both points. Timings are whole-launch on one node with vLLM stopped; they
are not a per-token derivation and are not a claim about chain time.

## Window command

Once the Lead window stages the binary, the two runs are (from the node, with vLLM stopped):

    ~/phase2-evidence/window-2026-10-18/m15_mhc_post_14a26701_aarch64 \
        --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
        --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins \
        --tokens 8 --repeat 100

and the same command with `--tokens 130` for the fixture prefill length. That binary is
4,069,864 B, sha256 `14a2670105cfac3819a284d9ce637202dbc6befba815f5127ce3351b2d4e3b5e`, built from
commit 4379166 - the binary window b20 ran. The cubin directory must
hold the pinned `mega_mhc_sm121.cubin` (70,312 B,
`e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618`); the harness verifies it before
any allocation.

## Window record

### Window b15: FAILED at the harness's own seam guard (no measured number)

Ran in a Lead window on 2026-10-19, start 01:36:02Z (host node0, node-only logs
`window-b15-20261019T013602Z.log` and `runs15/m15-tokens{8,130}.log` under
`/home/user/phase2-evidence/window-2026-10-18/`). Binary `m15_mhc_post_0699a78d_aarch64`
(4,035,936 B, sha256 `0699a78db06beeda96f95aa8769a60bd9e6519505046aef32570656679914dd1`); cubin
`mega_mhc_sm121` 70,312 B
`e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` verified by size and sha256
before allocation; vLLM was stopped for the window and restored afterwards.

Both shapes (8 and 130 positions) exited 1 at the same guard, before the first launch's comparison:

    m15_mhc_post: position 0: the entry twin's stream write is not the identity on the layer-0
    boundary seed, so X_0 cannot be the entry write; refusing to treat the post-seam input as the
    entry output

**Cause: a harness bug, not a kernel result.** The guard compared
`mega_mhc_layer_with_scale`'s first return value to X_0. That return is the reference's own
`B X_l + C F(X_hat_l)` self-loop (F the dequantised codes it just wrote), not the kernel's write,
which applies the previous call's coefficients; under the released H(X_0) coefficients it is a
different vector from X_0, so the guard fired. The declared mapping itself was never in doubt (the
kernel header and `mega_mhc_launch::PARAMS` say x is the [n][d] stream buffer and step 1 writes
`B X_{l-1} + C Y_{l-1}` into it), and no device launch produced a number: b15 measured nothing.
The fix replaces the guard with a check of the bytes actually passed against the kernel's own step-1
write on the boundary seed, and it also refuses a coefficient readback outside the released
activation ranges; it does not weaken the guard to make a window pass.

### Window b16: FAILED on the f32 capture only (no measured number claimed)

Ran in a Lead window on 2026-10-19, start 01:49:56Z (node-only logs
`window-b16-20261019T014956Z.log`, `runs16/m15-tokens8.log`, `runs16/m15-tokens130.log`). Binary
`m15_mhc_post_cf74a8df_aarch64` (4,051,752 B, sha256
`cf74a8df438d7481f49ffaf7a8b8a4db0f0c6c1418aacc75bc99a5bf0c9ba808`); cubin `mega_mhc_sm121`
70,312 B `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` verified before
allocation; vLLM stopped for the window and restored
afterwards.

Every position failed, and each failed on ONE quantity only. The verdict line was:

    FAIL position N: boundaries and control: fp8 within bar scales bit-exact X_1 bit-exact
    B/A/C <=1e-4 1/rms <=3e-4 capture DIFFER control within bar

8 of 8 positions at `--tokens 8` and 130 of 130 at `--tokens 130`: the post-mix write was
bit-exact, the 160 scales bit-exact, the E4M3 codes inside the 99.5%/within-1 bar, the 24
coefficients inside 1e-4, 1/rms inside 3e-4, the control gap inside its 10% bar - and the f32
block-input capture differed on every position.

**Cause: the twin's instruction sequence, not the kernel.** The twin formed the update in the
source's literal order: `round(c[s]*y)` first, then fuse the first B term. nvcc contracts the same
statement the other way round - the FIRST B product is the rounded `mul.f32` and the C Y term is the
fused `fma` (the PTX quoted under "Comparison class"). The two differ in the last f32 bit, which
bf16 rounding hides (so X_1 was bit-exact) but which X_hat sums directly (so the capture exposed it
on every position). The fix reproduces the PTX arrangement exactly in `expected_update`: one plain
product for `b[s][0]*x[0]`, `fma(c[s], y, .)`, then `fma` over t = 1..3. Two tests now pin the
arrangement (they distinguish it from the C-Y-first order and from a mul-then-add twin, and both
would fail under b16's twin). No bar was widened and no exactness claim was moved to a bar: the
capture stays declared bit-identical, and the next window measures whether that now holds.

### Window b19: PASSED (earlier record; pre-correction codes/scales twin)

Ran in a Lead window on 2026-10-19, start 02:13:26Z (node-only logs
`window-b19-20261019T021326Z.log`, `runs19/m15-tokens8.log`, `runs19/m15-tokens130.log` under
`/home/user/phase2-evidence/window-2026-10-18/`). Binary `m15_mhc_post_080222a7_aarch64`
(4,051,776 B, sha256 `080222a708f48730175fcec14b37244c4fa16e3c8fd38859227fb58da134c57e`, built
from commit a70b4ca); cubin `mega_mhc_sm121` 70,312 B
`e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` verified by size and sha256
before allocation; vLLM stopped for the window and restored afterwards; MemAvailable 117.0 GiB at
start and 116.8 GiB at end, floor 32 GiB ok.

Both shapes passed, exit 0:

* `--tokens 8`: `PASS (8 of 8 positions at every boundary)`.
* `--tokens 130`: `summary: 130 positions x 2 verdicts, 0 failed`, and
  `m15_mhc_post: PASS (130 of 130 positions at every boundary; the post-mix write and the
  block-input capture are exact where their arithmetic is the same program, and the
  coefficients/codes are within the declared bars where the reduction order differs)`.

So at both shapes: the post-mix residual write X_1 is bit-identical, the 5,120 f32 block-input
capture `xhat_normed` is bit-identical against the twin computed with the device's own 1/rms, the
160 ue8m0 scale bytes are bit-identical, the E4M3 codes are inside their 99.5%-exact/within-1 bar,
the 24 coefficients are within 1e-4, 1/rms is within 3e-4, both device runs of every position are
byte-identical, and the FP8 control gap is inside its 10%-of-capture bar. The per-position verdict
lines are in the node logs above. Determinism is checked separately (the same launch twice,
byte-identical across all six outputs); the timed launches' outputs are not compared, and they start
from the streams buffer the last determinism run left in place (the kernel updates x in place, so
the timed launch input has already become X_1) - the record above therefore claims no number from a
repeat.

Timing, from the `--tokens 130` run (100 launches after the 130 per-position comparisons; the
timings are of repeated launches of an already-verified call, i.e. whole-launch cost on this one node
with vLLM stopped, not a per-token derivation and not a claim about chain time): min 0.0330 ms,
p50 0.0338 ms, mean 0.0338 ms, max 0.0349 ms per launch; batched 0.0335 ms/launch.

The two harness-side failures that preceded this window are recorded above in full, including the
guard and instruction-order bugs they exposed; neither produced a device number.

### Post-b19 review correction to the comparison twin (re-windowed as b20)

The fresh review of the b19 record found that the code/scale twin was
`mega_mhc_ref::mega_mhc_layer_with_scale`, whose X_hat is a mul-then-add chain (Rust never
contracts `value += weight * element`), while the device's X_hat is an fma chain - so the harness's
declared twin (fma, used for the capture) was not the twin that produced the expected codes/scales,
and the "cause" line cited the twin's 1/rms while the codes had been built with the reference's
sequential 1/rms. The harness now builds the expected E4M3/ue8m0 bytes from its own fma `host_xhat`
and `expected_inv_rms` via `expected_quantisation` (the kernel's order, multiply by the reciprocal),
and the finding line cites that same rms. The reference function is still used for the coefficient
twin only, where the class is a bar either way.

This is a host-side comparison change, not a device-kernel or argument-shape change: the measured
device outputs, the binary that produced them and the timing record are untouched. The corrected
comparison was re-windowed as b20 below and passed at both shapes; the corrected harness is binary
`m15_mhc_post_14a26701_aarch64` (4,069,864 B, sha256
`14a2670105cfac3819a284d9ce637202dbc6befba815f5127ce3351b2d4e3b5e`), and the "Window command"
section above names it.

### Window b20: PASSED (the recorded window for m15, corrected twin)

Ran in a Lead window on 2026-10-19, start 02:57:37Z (node-only logs
`window-b20-20261019T025737Z.log`, `runs20/m15-tokens8.log`, `runs20/m15-tokens130.log` under
`/home/user/phase2-evidence/window-2026-10-18/`). Binary `m15_mhc_post_14a26701_aarch64`
(4,069,864 B, sha256 `14a2670105cfac3819a284d9ce637202dbc6befba815f5127ce3351b2d4e3b5e`, built from
commit 4379166); cubin `mega_mhc_sm121` 70,312 B
`e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618` verified by size and sha256
before the first device allocation; vLLM stopped for the window and restored afterwards.
MemAvailable 117.7 GiB at start and 117.5 GiB at end (both shapes), floor 32 GiB ok.

Both shapes passed, exit 0, with the corrected codes/scales twin (expected E4M3/ue8m0 bytes built
from the harness's own fma X_hat and its own 1/rms, not the reference's mul-then-add X_hat).
The pass lines below are reproduced exactly as the window printed them:

* `--tokens 8`: `summary: 8 positions x 2 verdicts, 0 failed`;
  `PASS (8 of 8 positions at every boundary; the post-mix write and the block-input capture are
  exact where their arithmetic is the same program, and the coefficients/codes are within the
  declared bars where the reduction order differs)`.
* `--tokens 130`: `summary: 130 positions x 2 verdicts, 0 failed`; the same PASS sentence with
  `130 of 130`.

So at both shapes: the post-mix residual write X_1 is bit-identical, the 5,120 f32 block-input
capture `xhat_normed` is bit-identical against the twin computed with the device's own 1/rms, the
160 ue8m0 scale bytes are bit-identical against the twin's own quantisation, the E4M3 codes are
inside their 99.5%-exact/within-1 bar, the 24 coefficients are within 1e-4, 1/rms is within 3e-4,
both device runs of every position are byte-identical, and the FP8 control gap is inside its
10%-of-capture bar. The per-position verdict lines are in the node logs above.

Timing, from the `--tokens 130` run (100 launches after the 130 per-position comparisons; the
timings are of repeated launches of an already-verified call - whole-launch cost on this one node
with vLLM stopped, not a per-token derivation and not a claim about chain time; the timed launches
start from the streams buffer the last determinism run left in place, i.e. X_1, and their outputs
are not compared): min 0.0330 ms, p50 0.0341 ms, mean 0.0342 ms, max 0.0395 ms per launch; batched
0.0339 ms/launch.

Window b19 above remains the earlier record: it passed on the pre-correction twin, whose expected
codes/scales side was the reference's mul-then-add X_hat rather than the fma twin this harness
declares. Its device-side measurements and timing stand; its codes/scales comparison is superseded
by b20.

### Criterion status

Phase 2 exit criterion 1 stays **OPEN**: m15 establishes the post seam's arithmetic, argument shape
and pass class on released layer-0 bytes, not a token, a chain pass or an end-to-end layer.

## Explicitly NOT claimed

* any attention work, inverse RoPE, wo_a/wo_b, or a measured attention output - Y_0 is a declared
  stand-in built from released embed rows;
* any chain pass, chain ordering, seam ledger, per-layer schedule or device-to-device coefficient
  handoff - the post call's B/A/C and Y_0 are host copies, declared as such;
* the 24 coefficients this call publishes as released LAYER-1 coefficients: the kernel applies this
  layer's hc tensors to the updated streams, so they are the H arithmetic on X_1 under layer-0's
  installed module, not another layer's checkpoint tensors;
* any token, any text, any end-to-end layer or model output;
* routing, SwiGLU, the MoE block, the shared expert, the combine;
* weight staging, device-budget admission, engine integration;
* a host-time saving of any kind.
