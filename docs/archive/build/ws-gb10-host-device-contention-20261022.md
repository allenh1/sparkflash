# GB10 host/device contention: three measured negatives on the way to 150

Status: decision record, 2026-10-22. Published number unchanged at **249.25 ms/step**
(`268f2ab1`). The rows below are all four-rank, identity-passing (tokens
`[491,92928,201,15,4893,16646,15,14265,86]`, logits sha `aed6054d...`), and each was staged
from its own worktree so the binaries differ only as described.

## 1. Per-routed-row slice tables (closed, MR !159)

Replaced the compact staged stacks with `[tokens*top_k]` u64 tables of arena slice addresses and
removed 24 of the 36 D2D copies per layer.

- Rows: chain 425.33, lean **348.53**, probe 355.47 against wsd6's 328.61 / **249.25** / 260.67.
- Probe: `moe_gate_up` kernel 0.4087 -> 2.4980 ms/layer, `moe_stored_combine` 0.1831 -> 1.2506,
  dtod 198,720 -> 0.
- Verdict: **+99.3 ms/step**. The copy had been buying layout. The compact stack is one contiguous
  `[top_k][width][hidden/2]` image read in slot order; a pointer chase into the interleaved arena
  lost ~84 % of the kernel's effective bandwidth. A copy is not free, but neither is coalescing.

## 2. Run-scope plumbing + gather kernel (closed, MR !160)

Plumbing (run-scope trace rings, kv-norm prewarm, packed small-operand uploads) plus a pinned
`staged_gather_sm121` kernel that copies the arena slices into the same compact stacks with one
launch per matrix instead of 12 copies.

- Rows: lean 344.42 (with a stray vLLM restore overlapping it), then a clean A/B: baseline 248.67,
  branch lean **461.00**, and probe 255.36 earlier in the same window - i.e. **unstable as well as
  slower**.
- Probe: kernel time 10.29 -> 14.21 s over 138 positions (+28 ms/step). Gather itself
  0.3630 ms/layer for 56.4 MB = ~155 GB/s against the 273 GB/s it was priced at; the packed
  uploads also cost kernel locality (`attention_output` kernel 0.4879 -> 0.6218,
  `moe_router` 0.1925 -> 0.2885).
- Call counts fell exactly as designed (dtod 198,720 -> 0; projections h2d 16,560 -> 5,520;
  router 27,600 -> 16,560; output 27,600 -> 11,040) and the wall did not follow.
- Verdict: **+25 ms/step** in probe terms, unstable. Driver-call count is not wall: with the async
  launch path the calls overlap device work, and only calls that force a drain are visible.

## 3. Host seam ILP (closed, MR !161)

Interleaved independent f32 accumulation chains in `mhc.rs` (8 coefficient chains for `hc_mixes`,
4/8 for `hc_pre`/`hc_post`) and in `linalg::matmul_checked` (4 output rows per column step,
scoped-thread chunking >=32 rows), with exact pre-change bit pins.

- Clean A/B: baseline 246.67 and 248.67 minutes either side; branch lean **291.78**, worst step 486
  vs 250.
- Probe: the seams did get faster - `mhc_entry` 0.3164 -> 0.1225 ms/layer, `ffn_seams`
  0.5151 -> 0.3225, `ref_seam` 0.4671 -> 0.4483, about **-17 ms/step of host wall** - and every
  device kernel got slower: `attention_output` 0.4879 -> 0.7140, `moe_gate_up` 0.4087 -> 0.6350,
  `moe_router` 0.1925 -> 0.3454, kernel total 10.29 -> 14.94 s (**+33.7 ms/step**).
- Verdict: **+43 ms/step net**.

## What this means

On GB10 the CPU and the GPU share the memory system and the power envelope. The signature in (3) is
unmistakable: identical device code and pins, every kernel slower by a similar factor, at the same
time as the host got faster. So **host work is not a pure serial cost that can be optimised in
place**; a faster host can make the step slower. Two rules follow, and they are now the frame for
the 150 gate:

1. Any host change must be measured as a whole-step four-rank delta. A function-local speedup is not
   evidence - three windows of it have now gone the wrong way.
2. The lever is to **remove** host work or the bytes it touches (device arms, fewer drains, fewer
   readbacks), not to execute it faster.

The device side is where the remaining headroom is uncontested: 71.2 ms/step of kernel time against
a measured 273 GB/s peak, with the one kernel audited properly (shared-expert GEMV) at 171 GB/s
(62.7 %). The routed GEMMs have never been audited that way; at 25-30 % of peak there is more than
30 ms/step there before any arithmetic changes.
## 4. Bisect of the host-seam ILP regression (both halves hurt)

The two halves were staged separately, each one file from `268f2ab1`, and each run in its own
four-rank lean row minutes from a clean baseline (246.7 / 246.9 / 248.7 for the baseline rows).

| arm | files | lean step | worst |
|---|---|---:|---:|
| baseline wsd6 | - | **246.7** | 250 |
| A: mHC interleave | `mhc.rs` only | 378.4 | 489 |
| B: linalg interleave | `linalg.rs` only | 321.7 | 496 |
| both (wsd10) | both | 291.8 | 486 |

The per-step samples show the mechanism is not a uniform slowdown: `bsa` ran
`[484, 489, 476, 474, 475, 271, 244, 246, 247]` - the first five positions at nearly double, then
exactly baseline - while the baseline rows are flat `[251, 246, 246, 245, 246, 245, 246, 251, 246]`.
So the host interleave does not add a constant cost; it makes the step bimodal for a while (or for
particular positions) with a ~230 ms stall, and it does so in both halves. That is the same shape a
rank-wide stall would have: one rank late by ~230 ms, everybody waits at the next reduce. The most
consistent reading is CPU/GPU contention on the shared GB10 package, but the useful fact for the
program is simpler and does not depend on the mechanism: **any host-side arithmetic rework has
produced a bimodal, slower four-rank step, in every arm measured, and the whole-step row - not the
function benchmark - is the only thing that catches it.**

## 5. wo_a on the decode GEMV: the first kernel win that behaves

`b12x_bf16a_mxfp8w_wo_a_sm121` was launching 8 blocks x 256 threads for the whole layer (2048
threads for 8.389 MB, one thread per output channel walking K=4096), which the measurement table
priced at 19.5 GB/s. The first attempt substituted the fused coded-activation GEMV and handed it a
f32 activation row: the row refused at position 0 layer 0 with `wo_a wrote a non-finite value` on
three ranks - the identity gate doing its job, and the useful lesson is that a word-identical
arithmetic contract is necessary but not sufficient, the *kernel symbol and operand layout* have to
match too. The fix launches the pinned `b12x_mxfp8_gemv_m1_nf_f32a_sm121` (non-fused f32-activation
mode), no repin, 22 pins unchanged.

Measured (wsd11b, probe): `attention_output` kernel **0.4879 -> 0.2022 ms/layer** (-0.286, the
largest single-kernel win of the program), block wall 1.0895 -> 0.8369. Lean step: samples
`[266, 246, 457, 267, 245, 249, 249, 243, 254]`, median 249 against the baseline's 246 - i.e. **flat**.

That flatness is the lesson, not a failure: in this block the host share was already 0.60 ms/layer
against a 0.49 kernel, so the device time was hidden behind host work and removing it exposed
nothing. **A device kernel win can only move the step when the block's host share is already below
its kernel time.** The table below is the same probe run, and it says every remaining block is in the
host-bound regime:

| block | kernel ms/layer | wall ms/layer | host share |
|---|---:|---:|---:|
| attention_output | 0.202 | 0.837 | 0.635 |
| moe_gate_up | 0.637 | 1.894 | 1.257 |
| moe_router | 0.352 | 0.423 | 0.071 |
| moe_stored_combine | 0.278 | 0.316 | 0.038 |
| attention_projections | 0.142 | 0.422 | 0.280 |
| ffn_shared | 0.238 | 0.324 | 0.086 |
| mhc_entry | - | 0.318 | 0.318 |
| ffn_seams | - | 0.521 | 0.521 |
| ref_seam | - | 0.501 | 0.501 |

So the 249 -> 150 plan has an ordering constraint that the last three windows have now established
by measurement: **kernel work alone cannot reach 150**, because the step is host-bound per block and
host-side arithmetic rework makes it worse. What remains is the pipelining question - the host and
the device currently take turns inside every layer, and the only measured-safe lever left is to make
them overlap (host seam of layer L+1 while the device runs layer L) or to remove host seams
entirely. Either is a design change, not a kernel tweak, and it needs its own plan before any more
rows are spent.
