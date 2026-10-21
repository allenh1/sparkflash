# Deferred design note: f32 WKV dense variant (new f32a/f32-out dense kernel)

Status: **deferred, not cancelled. Nothing here is built or staged**: no kernel source, no cubin, no
pin-set edit, no host wiring. This note keeps the plan ready if a native-arm D41 measurement shows
the window-KV seed row still matters.

## Why it was deferred (Lead decision, current)

The D41 oracle is moving to native arithmetic (the device arm's attention projections run on the
native dense family now; the same is planned for the remaining non-native terms M2/M4/M3, see
`docs/build/native-rounding-arms.md`). The f32 capture reference therefore stops being the target:
the window-KV seed question becomes **native vs native (vLLM TD=4)**, not native vs our f32
reference. A variant whose only purpose is to reproduce the f32 capture bit-for-bit is no longer a
D41 row by itself.

## What the variant is, exactly

The eighth projection stage `wkv` currently runs `b12x_bf16a_mxfp8w_dense_sm121` (pinned,
`kernel_baseline.txt`): BF16 activation operand in, BF16 output. Its arithmetic differs from the
host f32 reference chain at two boundaries (activation narrowed to BF16 before the FP8 quantiser,
output rounded to BF16), which is the recorded attention-body deviation at row 0. The deferred
variant is **additive**:

1. A new dense kernel `b12x_f32a_mxfp8w_dense_sm121` (or a `f32-out` flag variant of the same
   family): activation operand read as f32, FP8 E4M3 block-32 ue8m0 quantisation applied by the
   kernel's own rule as today, f32 accumulate, **f32 output** (no bf16 output round).
2. One host line in the loop after the launch: `narrow_bf16` the returned f32 latent where the
   released path's first BF16 boundary is, so the downstream kv_norm/rope/store code keeps its
   current dtype contract.

The point is not "f32 is better": it is that the f32 arm then has an observation of the WKV seed
whose only difference from the f32 host reference is the reference itself, so the window-KV
attribution experiment (which seeds the sliding window with the window row's own latent) is not
confounded by the projection's own rounding.

## Expected size of the effect (estimate, not measured)

From the accepted attention-body analysis: the row-0 attention output deviation is
`attention_out` relative L2 ~1.98e-2 in the current chain, of which the WKV projection's two
boundaries are the arithmetic estimate below; removing them is expected to bring the projection
contribution to ~3-4e-3 relative on `attention_out` (**arithmetic estimate, not a measurement**).
That does not say the window row stops mattering; it only removes one confound from the measurement
that would decide it.

## Cost and A/B

- One new cubin (additive; the pinned set grows 20 -> 21) and a Lead-sanctioned pin-set regen.
  **No pinned cubin is mutated** - the existing 20 stay.
- Kernel source plus its host twin (`mxfp4_ref`-style dense reference reading f32 activation,
  f32 output), and a differential test against the existing bf16a kernel on one captured
  activation row: bit-equal f32 output vs the host f32 reference, and the BF16-narrowed output
  equals the existing kernel's output where the arithmetic coincides.
- Host wiring: read the f32 latent (one extra dtoh/sec of 512 f32 = 2 KiB per layer per position,
  or keep it device-side if the next consumer accepts f32), apply `narrow_bf16`, and record the
  latent sha in the evidence so the row is attributable.
- Smallest no-window A/B: the host twin differential test above. Smallest window A/B: one rank-0
  launch on the m1y15/m26-style captured seam (the WKV input latent is available from the loop's
  recorded attention seams), comparing `wkv_out` f32/f32 and `narrow_bf16(wkv_out)` against the
  f32 host reference - no full window needed to decide the arithmetic question.
- If it is later promoted to a D41 arm: one extra pinned kernel, and a per-layer dtoh/read change;
  no change to the collective protocol.

## Reopen trigger

Reopen this note when **all** hold: (1) the native attention projections are the runtime arm; (2) a
D41 (or native-arm diagnostic) measurement still shows a window-KV seed row movement that the
native-vs-native comparison attributes to the seed latent's arithmetic rather than to the window
store/selection; (3) the Lead wants a native-vs-f32 differential row as evidence for that
attribution. Until then this stays a plan.
