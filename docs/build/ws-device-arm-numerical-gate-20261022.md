# The device arm's numerical gate: what it compares, what it is allowed to differ by, and how to attribute

The standing rule is unchanged: **the bit-exact CPU twin stays the reference, not the product; the product gate is
tolerance-based.** This note pre-registers the device arm's gate *before* any device row exists, because a
tolerance set after the numbers are known is not a gate.

## 1. What is compared

The frozen reference (`docs/build/reference/speed-gate/fab52c00a0dc6f6c/rank{0..3}.json`) is a host-arm row over
the frozen argv (nine prompt ids, `--forward-reference-ids 491,367,201,80,3302,201,54,201,4341`) - logits,
top-k and the gather evidence per rank. The device row is the same argv with `SPARKFLASH_LAYER_DEVICE=1`.

The gate is `scripts/numerical-gate.sh --run <run-dir>` against that reference, with the recorded tolerance set:

| bar | value | why this bar |
| --- | --- | --- |
| greedy tokens | **identical over N prompts x M tokens** (the frozen argv: one 9-token prompt per rank) | a different argmax is a different model, not a rounding difference |
| `max_abs` | 2.0e-2 | one bf16 rounding at logit scale; a port that widens twice will exceed it |
| `mean_abs` | 2.0e-3 | catches a systematic bias that a per-element max can hide |
| `kl` | 1.0e-4 | distribution-level agreement, not just the head |
| `top1` | 1.0 | no argmax may move |
| `top5_overlap` | 0.80 | allows reordering inside the head |
| `top50_overlap` | 0.96 | a shifted tail means the row is wrong even if top-1 is right |

`TOP_K` 1000, `COMPARE_K` 100 as recorded in the gate script. **The arm is declared before the run**
(`SPARKFLASH_TOLERANCE_ARM`); a row without one is not gate-eligible.

## 2. The declared deviation inventory (what the device arm is allowed to differ by, and why)

Every entry is a by-construction difference, declared in the stage's contract row and testable by a twin. They
are listed here so a failing gate can be attributed instead of argued:

| # | stage | deviation | expected size |
| --- | --- | --- | --- |
| D1 | `kv_path` | the device latent is **bf16** where the host keeps f32, so `kv_norm_rope` widens an already-rounded row (one widening the host does not have) | measured by twin: <= 2 % relative on the latent, bf16-operand scale |
| D2 | `moe_shared` (f32 arm) | device SwiGLU uses `expf` vs Rust `exp`, and true divide vs reciprocal-then-multiply | ulp-level per element; must not move an argmax |
| D3 | `query_path` | the latent prologue reads the **bf16** `q_normed` row and widens - this *matches* the reference (which quantises the bf16 tensor), so it is a non-deviation, listed to stop it being "fixed" twice | 0 |
| D4 | `mhc_entry`, `ffn_seams` | the coefficient tail and RMSNorm run in the pinned seam kernels; the two new mHC stages are exact per element given their table, so the residual arm class is the tail + norm only | 0 for the two stages; tail/norm measured by twin |
| D5 | `reference_seam` | compressor weights are host-dtype (bf16 at ratio 1, f32 above) and the staged permutation is a value-preserving reorder; store width selected per layer ratio | 0 if the permutation is value-preserving; measured by the compressor proof test |
| D6 | `ref_select` | candidate ordering and the ascending-id tie-break must match the host **on the selected index vector**; scores may round inside a tie, indices may not | 0 on the index vector |
| D7 | collectives | NCCL AllReduce(Sum) over bf16 changes summation order vs the host's transport reduction | ulp-level; expected to be the largest term at the logit level after D1 |

**Rule**: a deviation not on this list and not twin-measured is a defect. Adding one requires a twin test and a
line in the table, not a wider tolerance.

## 3. How a mismatching row is attributed

The device arm deliberately has **no per-stage observation read** (that is what makes it fast), so the final
logits are the only device-side output. Attribution therefore goes the other way:

1. **Twin first**: every stage's CPU twin is bit-exact against the host math on the same operands, so a stage
   whose twin passes cannot be a *sequence* error; it can only be an operand error (wrong span, wrong dtype,
   wrong index). The dtype audit and the open-time width asserts exist to make operand errors fail at open.
2. **Bisect by frontier**: a probe row stops at the first refusing stage, so a run with a lower frontier
   exercises a prefix. If a prefix run's logits still mismatch, the defect is in the prefix; if not, it is in
   the stages between. This is slow but it never guesses.
3. **Debug row** (offline, never quoted as a number): build with the per-stage D2H observation read enabled and
   compare each boundary row to the host arm's same row elementwise. This is the only way to localise a
   difference that is inside tolerance but not attributable; it costs the copies the fast arm exists to avoid,
   which is why it is a debug build, not a mode of the product arm.

## 4. What invalidates a device gate row

- The arm was not declared before the run.
- The row is not a clean window (any `sparkflash-*` process alive, or serving resident).
- The frontier is not `completed` (a prefix run cannot be compared to a full row - it has no final logits).
- Any declared deviation from §2 is missing from the stage's contract text: an undeclared deviation is not a
  deviation, it is an unidentified defect.
- The four ranks disagree on the frontier or on the collective count (`verify_collective_count == layers * 2`).

## 5. What the gate cannot tell us

It cannot tell us *why* a row is right. A passing tolerance row on one nine-token prompt is evidence that the
port is the same model on that prompt, not that every stage is numerically faithful; the twin suite and the
debug-row procedure are what carry that claim. Both are required before any speed number from the device arm is
published: **a fast wrong row is worse than a slow right one**, which is why the frontier probe and the gate come
before the clock.
