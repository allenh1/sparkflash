# Phase 3 m17a - the routed MoE branch at real layer-0 geometry

## Status

**PASS at both token counts (window b27).** The harness ran the routed branch's four released kernel
stages on released layer-0 bytes and all declared checks passed at tokens 8 and 130: 30 checks per
run, four pinned cubins. The F6 routing-weight host seam is declared and measured as such, and
DECISIONS D32 records it; nothing here claims a fully-device weighted routed block.

History: window b26 (the first binary) PASSed stages 1-3 at both token counts and then panicked inside
the harness's stage-4 operand assembly - a harness indexing bug, not a kernel. The assembly was fixed,
a regression test added, the binary rebuilt, and window b27 carried it to the full four-stage PASS.
The window record below holds both windows' evidence.

Source: `crates/sparkflash-kernels/examples/m17_moe_a.rs`, aarch64 binary
`m17_moe_a_cc1a1d01_aarch64` (4,040,960 B, sha256
`cc1a1d01f46d633a76bef14072a1a3645acd38b37a65c1f25f7ee0e4913f82e2`),
built by `scripts/build-aarch64.sh --example=m17_moe_a` from branch
`task/m17-moe-a` commit a9221b8 (based on `origin/main` 5416ccf). The four pinned cubins were already
exercised by earlier windows (router m9, gate/up window b6, the combine in the M7 stages, the SwiGLU
quantiser in the stage-8 B1 probe); what is new here is the four-stage composition at the released
geometry on released checkpoint bytes.

## Subject and launch

One harness runs the routed branch's four released kernel stages on one node, in the released order,
with the released checkpoint bytes:

| # | stage | kernel | cubin pin | launches |
| :-- | :--- | :--- | :--- | :--- |
| 1 | router | `moe_router_sm121` | 37,384 B / `5992cef9...` | 1 |
| 2 | gate/up | `routed_gate_up_sm121` | 9,752 B / `115200d2...` | 8 (4 ranks x 2 projections) |
| 3 | SwiGLU + MXFP8 quantise | `swiglu_mxfp8_quant_sm121` | 21,776 B / `4dbef63b...` | 4 (one per rank slice) |
| 4 | down/combine | `b12x_mxfp4_moe_sm121` | 10,048 B / `36a15a4f...` | 4 (serial rank chain) |

Every cubin is refused by size and sha256 before any device allocation; a mismatch is an error
before the path is measured, never a tolerance. The router and gate/up pins come from their typed
ABI modules; the quantiser and combine pins are stated in the harness against
`baseline/kernel_baseline.txt` because those kernels are reached through their launch
modules, not a typed ABI.

Geometry, from the released `config.json`: hidden 5120, `moe_intermediate_size` 2304,
384 routed experts, `num_experts_per_tok` 6, `n_shared_experts` 1, `swiglu_limit` 10.0,
`routed_scaling_factor` 1.5, no `first_k_dense_replace` key - layer 0 is a real MoE
layer. The routed intermediate is the four 576-wide rank slices of D30, padded to 640 by the kernels'
ABI. The shared expert and the shared-add epilogue are m17b, not this harness.

## Released operands (bytes only, no synthetic weights)

| operand | released tensor | dtype / shape | read as |
| :--- | :--- | :--- | :--- |
| router gate | `layers.<l>.ffn.gate.weight` | BF16 [384, 5120] | bytes, exact BF16 widening for the twin |
| router bias | `layers.<l>.ffn.gate.bias` | F32 [384] | bytes; `bias_vl` is NOT read |
| routed w1/w3 | `layers.<l>.ffn.experts.<e>.w1/w3.weight` | I8 [2304, 2560] packed FP4 | rank slice rows r*576.., verbatim |
| routed w1/w3 scales | `layers.<l>.ffn.experts.<e>.w1/w3.scale` | F8_E8M0 [2304, 160] | rank slice, verbatim |
| routed w2 | `layers.<l>.ffn.experts.<e>.w2.weight` | I8 [5120, 1152] packed FP4 | sliced to the rank's 288 bytes, padded to 320 |
| routed w2 scales | `layers.<l>.ffn.experts.<e>.w2.scale` | F8_E8M0 [5120, 72] | sliced to 18, padded to 20 |
| activation rows | `embed.weight` rows 0..tokens | BF16 [vocab, 5120] | **declared stand-in** for the post-attention hidden |
| residual hidden | the same `embed.weight` rows | BF16 | **declared stand-in** for the pre-FFN residual |

The activation row stand-in is the same choice `m9_router.rs` and `m10_gate_up.rs` already
declare: this tree does not have layer 0's post-attention hidden stream in an isolated harness. Every
claim is therefore about the routed arithmetic over released bytes, not about layer text.

The activation operand for gate/up is the D8 quantiser applied to the stand-in row
(`act_quant_fp8_row`), exactly as `mega_mhc_sm121` leaves the FFN block input. The
distinct experts are the ones the **device router selected**, staged once and indexed by a table
local id; the harness prints the local-to-released mapping and every later stage uses that table, so
the branch is routing-driven, not an id cycle.

## Comparison classes (declared before any number)

1. **Router.** ids EXACT against the CPU twin (`router_ref`, which the crate's tests prove
   bit-identical to the engine host path on this configuration). Scores max abs delta <= 1e-6 and
   max ULP <= 8; weights max abs delta <= 1e-6 and max ULP <= 16. The only known degree of freedom
   is that the device uses CUDA `expf`/`log1pf` and the host uses libm; `sqrtf` is correctly
   rounded. The selection is pinned exact rather than probabilistic: the kernel's strict-`>` scan
   over ascending expert ids is the released tie rule (lower id wins), and a gap that crossed would
   be a finding. Two launches byte-identical.
2. **Gate/up.** Device w1 and w3 outputs BIT-IDENTICAL (ULP 0) to `routed_gate_up_ref` on every
   `tokens * 6 * 640` bf16 value of both projections; columns 576..639 exactly zero; two
   launches byte-identical. The eight launches are **independent**: same activation bytes, different
   weight slices, disjoint outputs, no ordering between them, so a later chain may stream them. Only
   the stage-4 ranks are serial by construction.
3. **SwiGLU + quantise.** E4M3 codes and ue8m0 scales BIT-EXACT against
   `swiglu_ref::quantise_row` over the same bf16 values, per rank slice at columns 640 - the
   stage-8 B1 class. A difference is a finding, never a tolerance. Two launches byte-identical. The
   launch is unweighted (F6 below).
4. **Down/combine.** Device bf16 BIT-EXACT against the chained `mxfp4_ref::moe_combine` over the
   same bytes. Four serial rank launches: each rank's bf16 output is the next rank's residual
   hidden, so the chained rounding is part of both sides. Two full chains byte-identical.

## The F6 seam (named, not hidden)

The released arithmetic multiplies the SwiGLU output by the routing weight **before** the w2
activation quantiser reads it (model.py:841-851), and no pinned routed kernel takes a weight operand.
The seam is therefore explicit and its crossing values are printed by the harness:

- stage 3 measures the device quantiser on the **unweighted** gate/up product;
- stage 4 consumes an activation operand built on the **host** from the same device gate/up bf16
  values: host SwiGLU over those bytes, then
  `mxfp4_ref::routed_activation_operand(activation, routing_weight)` - the reference rule
  including its bf16 cast;
- the values that cross: device gate/up bf16 `[tokens*6][640]` per rank and projection
  (2 bytes each), one f32 routing weight per selected expert per token, and the quantised operand
  `[tokens*6][640]` E4M3 + `[tokens*6][20]` ue8m0 per rank.

The combine is therefore **weighted but its activation operand is host-quantised by the same rule
the reference uses**; the stage-3 device codes are not the combine's input, because
weight-multiplying already-quantised codes would quantise twice and is not the model's order. This is
recorded as the F6 decision entry D32 (it assumes m16's D31 lands first as batch 14, so the numbering does not collide)
and the NOT-claimed list excludes a fully-device weighted routed block.

Planned follow-up, per the Lead's decision: after m17a passes, the natural next step is a weighted
SwiGLU+quantise variant (same source with an optional per-expert weight operand, new cubin and a
deliberate re-pin with its own DECISIONS entry, exactly the D31 pattern used for the certified LM
head). It is not part of m17a and no kernel was changed here.

## Window command

```sh
cd ~/phase2-evidence/window-2026-10-18/bin
./m17_moe_a_cc1a1d01_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 8 --layer 0 --floor-gib 32
./m17_moe_a_cc1a1d01_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 130 --layer 0 --floor-gib 32
```

Measured wall time in b27: tokens 8 ran 04:47:59 -> 04:48:10Z (11 s, 37 distinct experts,
695,623,680 expert bytes read); tokens 130 ran 04:48:10 -> 04:50:52Z (162 s, 148 distinct experts,
2,782,494,720 expert bytes read) - the serial host twins and the checkpoint reads dominate. The 32-GiB
MemAvailable floor is checked at start, before/after each launch, before/after each chain and at end;
the lowest reading in the window was 113.8 GiB at the tokens-130 end, and the node needs vLLM stopped
as usual for a window.

## Host dry run and tests

`cargo test -p sparkflash-kernels --example m17_moe_a`: 19 tests pass (the shared b2_harness
suite plus five m17 tests: argument parsing of the documented command, the local expert table,
disjoint rank slices covering the stored rows, the padding check counting only columns after 576, and
the regression test for the stage-4 activation assembly that window b26 caught). `cargo clippy -p sparkflash-kernels --example m17_moe_a -- -D warnings` is clean. The
workspace gates are run before review. No device code path is executed by these tests.

## Window record

### Window b26 (partial, 2026-10-19): stages 1-3 PASS, stage-4 harness panic

Lead-run window b26 ran the first binary (`m17_moe_a_8b788b9c_aarch64`, sha256
`8b788b9cd439945d0ed68559bd43dfbc7f02887f4fd87df318042e56d1d7d2db`, verified by the window script
before the run) at tokens 8 then 130. Both runs reached stage 4 and then panicked inside the harness;
stages 1-3 had already PASSED in both runs. Verbatim evidence from
`runs26/m17a-tokens8.log` and `runs26/m17a-tokens130.log`:

```text
=== RUN m17a-tokens8 start 2026-10-19T04:35:02Z
...
--- stage 4: b12x_mxfp4_moe_sm121 down/combine, 4 serial rank launches (residual chain) ---
thread 'main' (2066858) panicked at crates/sparkflash-kernels/examples/m17_moe_a.rs:727:54:
index out of bounds: the len is 30720 but the index is 30720
=== RUN m17a-tokens8 exit=134 end 2026-10-19T04:35:11Z

=== RUN m17a-tokens130 start 2026-10-19T04:35:11Z
...
--- stage 4: b12x_mxfp4_moe_sm121 down/combine, 4 serial rank launches (residual chain) ---
thread 'main' (2066885) panicked at crates/sparkflash-kernels/examples/m17_moe_a.rs:727:54:
index out of bounds: the len is 499200 but the index is 499200
=== RUN m17a-tokens130 exit=134 end 2026-10-19T04:37:03Z
```

499,200 = 780 routed rows x 640 columns; 30,720 = 48 rows x 640. Both are the stage-4 activation
buffer length, and the bad index equals it at rank 3. The cause was in the harness only, no kernel
was involved: the stage-4 assembly read each rank's gate/up row at `row * 640 + rank * 576`,
treating the four rank outputs as windows into one 640-wide row. Each rank's output row is 640 wide
for that rank's own 576-column slice (columns 576..640 are that rank's zero padding), so the correct
base is `row * 640`. The fix is described below; stage 4 was unverified after b26.

### The b26 fix

`assemble_activation` is now a named function: per-rank row base `row * 640`, explicit bound
check, and `the_activation_reads_each_ranks_own_640_wide_row_not_a_window` gives every rank, row and
column a distinct value so a window-style index fails the test (the pre-fix indexing fails it). No
comparison class, pin, claim or kernel changed. The rebuilt binary is the one used in b27:
`m17_moe_a_cc1a1d01_aarch64`, 4,040,960 B, sha256
`cc1a1d01f46d633a76bef14072a1a3645acd38b37a65c1f25f7ee0e4913f82e2`, built from branch commit
a9221b8. The window script printed that sha256 before the b27 runs, so the binary and the record are
tied.

### Window b27 (2026-10-19): PASS at both token counts

Window `window-b27-20261019T044614Z.log`, after the window script verified the binary sha
`cc1a1d01...`:

```text
=== RUN m17a-tokens8 start 2026-10-19T04:47:59Z
=== RUN m17a-tokens8 exit=0 end 2026-10-19T04:48:10Z
=== RUN m17a-tokens130 start 2026-10-19T04:48:10Z
=== RUN m17a-tokens130 exit=0 end 2026-10-19T04:50:52Z
```

Both runs: 30 checks, 4 stages, four pinned cubins (each cubin's size and sha256 printed by the
harness before allocation: router 37,384 B / `5992cef9...`, gate/up 9,752 B / `115200d2...`, SwiGLU
quantiser 21,776 B / `4dbef63b...`, combine 10,048 B / `36a15a4f...`). The exact per-run records
from `runs27/m17a-tokens8.log` and `runs27/m17a-tokens130.log`:

#### tokens 8 (48 routed rows, 37 distinct experts, 695,623,680 released expert bytes read)

```text
MemAvailable at m17a start: 122232732 kB (116.6 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 48)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 1)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 2)
  check stage 1: two launches byte-identical: ok (12672 raw output bytes compared)
stage 1 selection: 48 routed rows; 37 distinct experts over the batch; ids/weights/scores read back
routed expert table: 37 distinct experts selected by the device router, staged as local ids 0..37; every later stage indexes this table, and the mapping is printed below
activation stand-in: 8 embed rows x 5120 bf16 values, D8-quantised to 40960 E4M3 bytes + 1280 ue8m0 bytes
expert bytes: 37 experts x (w1/w3/w2 weights + scales), 695623680 B read verbatim from released tensors
  check stage 2 rank 0: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 0: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 0: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  check stage 2 rank 1: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 1: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 1: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  check stage 2 rank 2: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 2: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 2: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  check stage 2 rank 3: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 3: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 3: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  check stage 3 rank 0: E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 0: ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 0: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  check stage 3 rank 1: E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 1: ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 1: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  check stage 3 rank 2: E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 2: ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 2: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  check stage 3 rank 3: E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 3: ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 3: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
F6 seam crossing: device gate/up bf16 [rows 48][640] per rank -> host SwiGLU (10) -> routing weight per routed row -> routed_activation_operand -> [rows 48][640] E4M3 + [rows 48][20] ue8m0 per rank
w2 rank slices: 4 ranks x (60620800 packed + 3788800 scale bytes) from the released w2 tensors, each stored row sliced to 576 and padded to 640
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 40960)
  check stage 4: two full chains byte-identical: ok (40960 bf16 codes compared)
stage 4 output: 8 token rows x 5120 bf16 codes (residual + weighted routed sum over 6 slots and 4 ranks)
MemAvailable at m17a end: 121135880 kB (115.5 GiB), floor 32 GiB - ok
m17a: PASS (30 checks; 4 stages; tokens 8; four pinned cubins)
```

#### tokens 130 (780 routed rows, 148 distinct experts, 2,782,494,720 released expert bytes read)

```text
MemAvailable at m17a start: 122016956 kB (116.4 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 780)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 2)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 4)
  check stage 1: two launches byte-identical: ok (205920 raw output bytes compared)
stage 1 selection: 780 routed rows; 148 distinct experts over the batch; ids/weights/scores read back
routed expert table: 148 distinct experts selected by the device router, staged as local ids 0..148; every later stage indexes this table, and the mapping is printed below
activation stand-in: 130 embed rows x 5120 bf16 values, D8-quantised to 665600 E4M3 bytes + 20800 ue8m0 bytes
expert bytes: 148 experts x (w1/w3/w2 weights + scales), 2782494720 B read verbatim from released tensors
  check stage 2 rank 0: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 0: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 0: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  check stage 2 rank 1: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 1: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 1: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  check stage 2 rank 2: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 2: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 2: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  check stage 2 rank 3: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 3: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 3: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  check stage 3 rank 0: E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 0: ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 0: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  check stage 3 rank 1: E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 1: ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 1: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  check stage 3 rank 2: E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 2: ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 2: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  check stage 3 rank 3: E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 3: ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 3: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
F6 seam crossing: device gate/up bf16 [rows 780][640] per rank -> host SwiGLU (10) -> routing weight per routed row -> routed_activation_operand -> [rows 780][640] E4M3 + [rows 780][20] ue8m0 per rank
w2 rank slices: 4 ranks x (242483200 packed + 15155200 scale bytes) from the released w2 tensors, each stored row sliced to 576 and padded to 640
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 665600)
  check stage 4: two full chains byte-identical: ok (665600 bf16 codes compared)
stage 4 output: 130 token rows x 5120 bf16 codes (residual + weighted routed sum over 6 slots and 4 ranks)
MemAvailable at m17a end: 119373184 kB (113.8 GiB), floor 32 GiB - ok
m17a: PASS (30 checks; 4 stages; tokens 130; four pinned cubins)
```

The record's numbers, stated plainly: the router's 48/780 selections are exact against the twin and
its scores/weights stay inside the declared bars (max abs delta 1.19e-7 / 5.96e-8, max ULP 1-2 / 2-4);
all four ranks' gate and up projections are bit-identical to the twin on every one of
61,440/998,400 bf16 values per projection with zero nonzero padding; the SwiGLU quantiser's E4M3
codes and ue8m0 scales are bit-exact on all 30,720/499,200 codes and 960/15,600 scales per rank; and
the four-rank combine chain is bit-exact against `mxfp4_ref::moe_combine` on all 40,960/665,600
output bf16 codes, with both full chains byte-identical. The F6 seam printed as declared (device
gate/up bf16 [rows][640] per rank -> host SwiGLU -> routing weight -> `routed_activation_operand` ->
[rows][640] E4M3 + [rows][20] ue8m0 per rank). Peak use stayed far under the 32-GiB floor
(MemAvailable 116.6 -> 115.5 GiB across tokens 8, 116.4 -> 113.8 GiB across tokens 130). The
decision record for the seam is DECISIONS D32.

## NOT claimed

- Not a fully-device weighted routed block: the F6 weight multiply and the activation quantiser for
  the combine operand are host steps (see above).
- Not a chain, token, per-layer or end-to-end MoE claim: one layer's routed branch, one declared
  stand-in activation, no shared expert, no shared-add epilogue, no residual-stream reproduction.
- Not layer text: the activation and residual are released `embed.weight` rows, a stand-in.
- Not engine integration and not a host-time-saving claim: no engine seam is changed, and a saving
  needs both sides measured on one window.
- Not the shared expert or the shared-add epilogue: m17b.
- Not a performance claim: the harness makes no timing claim; its per-stage launch counts are
  structure, not cost.
- Not a floating-point-equality claim about CUDA's `expf`/`log1pf` versus libm: the router
  class is the declared bar above, and the ids are pinned because a selection change would be a
  finding.
