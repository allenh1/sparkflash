# Phase 3 m17a - the routed MoE branch at real layer-0 geometry

## Status

**PASS at both token counts.** The harness ran the routed branch's four released kernel stages on
released layer-0 bytes and all declared checks passed at tokens 8 and 130: 30 checks per run, four
pinned cubins, in window b27 (binary `cc1a1d01...`). Batch 16 folds in two non-blocking review
findings (an output-buffer prefill and a staged-tensor-name assertion, below); that source adds a
31st check and window b30 measured it at both token counts (31 checks each, binary `f3476b5a...`).
b27 therefore stays the pre-hardening record, and the hardened binary's PASS is b30.

**Then m17a was rebuilt on the weighted routed epilogue (D33), and window b33 measured it: PASS at
both token counts.** Stage 3 now takes the router's per-row weight operand on the device and stage 4
consumes that device operand, closing the F6 host seam. `m17a_weighted_6443d5d0_aarch64` ran tokens 8
(start 05:59:29Z, `exit=0` end 05:59:41Z) and tokens 130 (start 05:59:41Z, `exit=0` end 06:02:24Z) in
window `window-b33-20261019T055723Z.log`: **35 checks each**, four pinned cubins, the weighted
epilogue's pin 13,000 B / `0e602daf...` verified by size and sha256 before any allocation; both
records are reproduced verbatim below. The F6 routing-weight host seam was declared and measured as
such for b27/b30 (D32); the weighted build closes it and b33 is the device measurement (D33).

Window numbering is the Lead's global numbering: b30 is the hardened m17a, b31 the c4a-1 record, b32
the c4a-2 first attempt and **b33 the c4a-2 corrected PASS plus this weighted m17a PASS** - this page
uses b33, not b31.

History: window b26 (the first binary) PASSed stages 1-3 at both token counts and then panicked inside
the harness's stage-4 operand assembly - a harness indexing bug, not a kernel. The assembly was fixed,
a regression test added, the binary rebuilt, and window b27 carried it to the full four-stage PASS.
The window record below holds both windows' evidence.

Source: `crates/sparkflash-kernels/examples/m17_moe_a.rs`, aarch64 binary
`m17_moe_a_cc1a1d01_aarch64` (4,040,960 B, sha256
`cc1a1d01f46d633a76bef14072a1a3645acd38b37a65c1f25f7ee0e4913f82e2`),
built by `scripts/build-aarch64.sh --example=m17_moe_a` from branch
`task/m17-moe-a` commit a9221b8 (based on `origin/main` 5416ccf) for b27, and rebuilt as
`m17_moe_a_f3476b5a_aarch64` (4,153,704 B, sha256
`f3476b5a22b39c9df39b6cedbeb36e74f3025f3729dcc9fb7140754f47cc8f84`) from `task/m17b-shared` rebased
onto the current main for the batch-16 hardening; window b30 measures the hardened binary. The
weighted build is `m17a_weighted_6443d5d0_aarch64` (4,142,000 B, sha256
`6443d5d0729650bdf571a3907c95ef5c169296b25a55050b42bd7174b16828cd`, D33), measured in window b33.
Build cwd for every binary on this page: `/home/user/wt/p3-m17` (`scripts/build-aarch64.sh`
is cwd-sensitive: the same source built from another directory gives different bytes and size, so
these shas are only reproducible from this exact cwd). The four
pinned cubins were already
exercised by earlier windows (router m9, gate/up window b6, the combine in the M7 stages, the SwiGLU
quantiser in the stage-8 B1 probe); what is new here is the four-stage composition at the released
geometry on released checkpoint bytes.

## Subject and launch

One harness runs the routed branch's four released kernel stages on one node, in the released order,
with the released checkpoint bytes:

| # | stage | kernel | cubin pin | launches (one pass) |
| :-- | :--- | :--- | :--- | :--- |
| 1 | router | `moe_router_sm121` | 37,384 B / `5992cef9...` | 1 |
| 2 | gate/up | `routed_gate_up_sm121` | 9,752 B / `115200d2...` | 8 (4 ranks x 2 projections) |
| 3 | weighted SwiGLU + MXFP8 quantise | `swiglu_mxfp8_quant_weighted_sm121` | 13,000 B / `0e602daf...` | 4 (one per rank slice) |
| 4 | down/combine | `b12x_mxfp4_moe_sm121` | 10,048 B / `36a15a4f...` | 4 (serial rank chain) |

Each stage runs its launches twice (determinism), so a full run makes 2/16/8/8 launches; the table
counts one pass. Every cubin is refused by size and sha256 before any device allocation; a mismatch is
an error before the path is measured, never a tolerance. The router and gate/up pins come from their typed
ABI modules; the weighted quantiser and combine pins are stated in the harness against
`baseline/kernel_baseline.txt` because those kernels are reached through their launch
modules, not a typed ABI. The weighted epilogue is a deliberate baseline re-pin (D33: 13,000 B /
`0e602daf...`, nvcc V13.0.88, `ptx_membar_lines=0`, the manifest's tenth entry); the unweighted
epilogue's pin (`21,776 B / 4dbef63b...`) is unchanged and remains m17b's shared-expert quantiser.

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
3. **Weighted SwiGLU + quantise.** E4M3 codes and ue8m0 scales BIT-EXACT against
   `swiglu_ref::quantise_row_weighted` over the same 640-wide padded gate/up rows and the same
   per-row routing weight, per rank slice - the stage-8 B1 class with the model's multiply-before-
   quantise order, including the single bf16 rounding of the product. Additionally the real 576-wide
   columns are BIT-EXACT against `mxfp4_ref::routed_activation_operand` over the same real values and
   weight (the operand the old host seam built, now only a comparison - built by calling that function
   itself over the real SwiGLU row, not an arithmetically equal re-implementation). A difference is a finding,
   never a tolerance. Two launches byte-identical. The weight operand is the router's device
   `weights_d` buffer, read in place.
4. **Down/combine.** Device bf16 BIT-EXACT against the chained `mxfp4_ref::moe_combine` over the
   SAME device-built weighted operand bytes from stage 3. Four serial rank launches: each rank's bf16
   output is the next rank's residual hidden, so the chained rounding is part of both sides. Two full
   chains byte-identical.

## The F6 seam closed by the weighted epilogue (D33)

The released arithmetic multiplies the SwiGLU output by the routing weight **before** the w2
activation quantiser reads it (model.py:841-851). D32 recorded that as a host seam for b27/b30: those
windows measured the device quantiser unweighted and the harness built the combine operand on the
host from the same device gate/up bf16 values with `mxfp4_ref::routed_activation_operand`.

The weighted epilogue removes that seam from the device path. `swiglu_mxfp8_quant_weighted_sm121`
takes the router's per-row `float *weights` operand and does the model's order on the device:
asymmetric-clamped SwiGLU, one bf16 rounding of the product, then the activation ceil-rule quantise.
Stage 3 reads `weights_d` in place (the router output is not copied or re-uploaded for the operand).
The operand stage 4's combine receives is device-produced, but the handoff is a host round trip: the
harness reads the stage-3 codes/scales back (to compare them against the twins) and then re-uploads
those same readback bytes into the chain's own buffers with `copy_htod`. No host arithmetic - no
multiply, no quantise - is applied between the device write and the chain launch, so the values the
combine consumes are exactly the device-built ones. The crossing shapes are:

- device router weights `[tokens*6]` f32 (the router's own buffer) -> weighted epilogue;
- device gate/up bf16 `[tokens*6][640]` per rank -> weighted epilogue;
- device operand `[tokens*6][640]` E4M3 + `[tokens*6][20]` ue8m0 per rank -> device combine;
- device combine -> device combine, four serial rank launches.

The bytes do round-trip through the host: the stage-3 readback is the source the chain re-uploads.
What is absent is host arithmetic on them - no host multiply and no host quantise in the operand
path - and the re-upload is a byte copy of the device write, not a recomputation. The crossing shapes
are printed above. The unweighted
epilogue remains pinned and is m17b's shared-expert quantiser; the six-kernel frozen m4 ABI array is
untouched (D33). Window b33 is the device measurement of this build: 35 checks passed at tokens 8
and 130 (record below), including the bit-exact operand and chain comparisons and the real-width
`routed_activation_operand` comparison.

## Window command

Window b27 ran binary `m17_moe_a_cc1a1d01_aarch64`; window b30 ran the hardened
`m17_moe_a_f3476b5a_aarch64` (4,153,704 B, sha256
`f3476b5a22b39c9df39b6cedbeb36e74f3025f3729dcc9fb7140754f47cc8f84`); the weighted build is
`m17a_weighted_6443d5d0_aarch64` (4,142,000 B, sha256
`6443d5d0729650bdf571a3907c95ef5c169296b25a55050b42bd7174b16828cd`) and window b33 measured it at both token counts (35 checks each, records below). All three binaries run
the same two commands and are built from `/home/user/wt/p3-m17`.

```sh
cd ~/phase2-evidence/window-2026-10-18/bin
./m17a_weighted_6443d5d0_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 8 --layer 0 --floor-gib 32
./m17a_weighted_6443d5d0_aarch64 --checkpoint ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins --tokens 130 --layer 0 --floor-gib 32
```

Measured wall time in b27: tokens 8 ran 04:47:59 -> 04:48:10Z (11 s, 37 distinct experts,
695,623,680 expert bytes read); tokens 130 ran 04:48:10 -> 04:50:52Z (162 s, 148 distinct experts,
2,782,494,720 expert bytes read) - the serial host twins and the checkpoint reads dominate. The 32-GiB
MemAvailable floor is checked at start, before/after each launch, before/after each chain and at end;
the lowest reading in the window was 113.8 GiB at the tokens-130 end, and the node needs vLLM stopped
as usual for a window.

Measured wall time in b30 (hardened binary): tokens 8 ran 05:14:35 -> 05:14:47Z (12 s, 37 distinct
experts) and tokens 130 ran 05:14:47 -> 05:17:30Z (163 s, 148 distinct experts); the same host twins and checkpoint reads run as in b27; the harness makes no timing claim, so the page
records the wall times without attributing them to a component. The lowest MemAvailable reading was
113.6 GiB during the tokens-130 stage-4 chain.

## Host dry run and tests

`cargo test -p sparkflash-kernels --example m17_moe_a`: 22 tests pass (the shared b2_harness
suite plus eight m17 tests: argument parsing of the documented command, the local expert table,
disjoint rank slices covering the stored rows, the padding check counting only columns after 576, the
staged-tensor-name format, the weighted epilogue pin against the embedded baseline, the source pin
that the real-width comparison calls `mxfp4_ref::routed_activation_operand` itself, and the
regression test for the stage-4 activation assembly that window b26 caught). The weighted run prints
35 checks (stage 1: 4, staged names: 1, stage 2: 12, stage 3: 16, stage 4: 2); window b33 measured
both runs at 35 checks, `exit=0`. `cargo clippy -p sparkflash-kernels --example m17_moe_a -- -D warnings` is clean. The
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

### Window b27 (2026-10-19): PASS at both token counts (pre-hardening binary `cc1a1d01`)

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
quantiser 21,776 B / `4dbef63b...`, combine 10,048 B / `36a15a4f...`). The blocks below are curated
excerpts of the per-run records from `runs27/m17a-tokens8.log` and `runs27/m17a-tokens130.log` (the
full logs hold every class text and MemAvailable line); the batch-16 hardening adds a 31st check and
is measured separately in window b30 below.

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

### Window b30 (2026-10-19): PASS at both token counts (hardened harness)

Window `window-b30-20261019T051153Z.log` (chained after b29; it stopped vLLM during b29's restart and
restarted it at the end). The window script verified the binary sha256 before the runs
(`f3476b5a22b39c9df39b6cedbeb36e74f3025f3729dcc9fb7140754f47cc8f84`), and each of the four cubins'
size and sha256 is printed before allocation.

```text
=== window b30 (m17a hardened rerun, chained after b29) start 2026-10-19T05:11:53Z host=node0 ===
=== RUN m17a-hardened-tokens8 start 2026-10-19T05:14:35Z
=== RUN m17a-hardened-tokens8 exit=0 end 2026-10-19T05:14:47Z
=== RUN m17a-hardened-tokens130 start 2026-10-19T05:14:47Z
=== RUN m17a-hardened-tokens130 exit=0 end 2026-10-19T05:17:30Z
=== all runs complete 2026-10-19T05:17:30Z ===
=== restarting vLLM 2026-10-19T05:17:30Z ===
started tmux session vllm-window; waiting for health
```

The restart was initiated at 05:17:30Z and the log tail ends at "waiting for health"; the window log
does not record a health exit, so this page does not claim one.

#### tokens 8 (hardened: 31 checks, 37 distinct experts, 695,623,680 released expert bytes read)

```text
MemAvailable at m17a start: 123187088 kB (117.5 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 48)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 1)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 2)
  check stage 1: two launches byte-identical: ok (12672 raw output bytes compared)
stage 1 selection: 48 routed rows; 37 distinct experts over the batch; ids/weights/scores read back
routed expert table: 37 distinct experts selected by the device router, staged as local ids 0..37; every later stage indexes this table, and the mapping is printed below
activation stand-in: 8 embed rows x 5120 bf16 values, D8-quantised to 40960 E4M3 bytes + 1280 ue8m0 bytes
  check stage 2: staged expert tensor names are the released global ids: ok (0 of 222 expected released expert tensor names missing from the 225 names actually read)
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
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 40960)
  check stage 4: two full chains byte-identical: ok (40960 bf16 codes compared)
MemAvailable at m17a end: 122005776 kB (116.4 GiB), floor 32 GiB - ok
m17a: PASS (31 checks; 4 stages; tokens 8; four pinned cubins)
```

#### tokens 130 (hardened: 31 checks, 780 routed rows, 148 distinct experts, 2,782,494,720 released expert bytes read)

```text
MemAvailable at m17a start: 122955328 kB (117.3 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 780)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 2)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 4)
  check stage 1: two launches byte-identical: ok (205920 raw output bytes compared)
stage 1 selection: 780 routed rows; 148 distinct experts over the batch; ids/weights/scores read back
routed expert table: 148 distinct experts selected by the device router, staged as local ids 0..148; every later stage indexes this table, and the mapping is printed below
activation stand-in: 130 embed rows x 5120 bf16 values, D8-quantised to 665600 E4M3 bytes + 20800 ue8m0 bytes
  check stage 2: staged expert tensor names are the released global ids: ok (0 of 888 expected released expert tensor names missing from the 891 names actually read)
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
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 665600)
  check stage 4: two full chains byte-identical: ok (665600 bf16 codes compared)
MemAvailable at m17a end: 119333484 kB (113.8 GiB), floor 32 GiB - ok
m17a: PASS (31 checks; 4 stages; tokens 130; four pinned cubins)
```

The new 31st check reads here as: `0 of 222 expected released expert tensor names missing from the 225
names actually read` at tokens 8 (37 staged experts x 6 names = 222) and the same at tokens 130; the
padding checks still read `0 nonzero padding values` on all four ranks now that the buffers started at
`0xFFFF`. Both runs: 31 checks, 4 stages, four pinned cubins, exit 0.

### Batch-16 hardening (post-b27, non-blocking review findings)

Two LOW findings from the adversarial review of the b27 harness are fixed in batch 16. Neither
changes a kernel, a class or a claim, but both tighten what a pass proves:

- **Padding prefill.** The gate/up output buffers are prefilled with `0xFFFF` before the first launch,
  because `cuMemAlloc` does not promise zeroed pages; the "columns 576..640 exactly zero" check can no
  longer pass on recycled or zeroed pages if a launch stopped writing the padding.
- **Staged tensor names.** `read_expert_tensors` builds its six names through
  `expert_tensor_names(prefix, global_id)`, and the run asserts `Snapshot::read_names()` contains every
  `layers.<l>.ffn.experts.<id>.{w1,w3,w2}.{weight,scale}` for every staged device id; a unit test pins
  the format. Swapping the global id for the local id in a copy previously left all tests green because
  the twin consumes whatever bytes were staged.

The hardened harness prints 31 checks (30 from b27 plus the staged-name check). It was rebuilt as
`m17_moe_a_f3476b5a_aarch64` (4,153,704 B, sha256
`f3476b5a22b39c9df39b6cedbeb36e74f3025f3729dcc9fb7140754f47cc8f84`) and window b30 measured it at
tokens 8 and 130: 31 checks each, both exit 0 (record below). The b27 record above belongs to binary
`cc1a1d01...` and is not extended to the batch-16 binary. DECISIONS D32 points at this record.

### Window b33 (2026-10-19): weighted build PASS at both token counts (D33)

Window `window-b33-20261019T055723Z.log` on host `node0` starts 2026-10-19T05:57:23Z. The window
script stopped vLLM at 05:57:23Z (four `vllm_node` processes), saw `gpu compute apps=0` at 05:58:35Z,
and printed the binary and cubin sha256s before any run:

```text
6443d5d0729650bdf571a3907c95ef5c169296b25a55050b42bd7174b16828cd  /home/user/phase2-evidence/window-2026-10-18/bin/m17a_weighted_6443d5d0_aarch64
6b237e6c20301ddeda7f47ff9add5385a19e789d818388c5eb22981f1af5377e  /home/user/phase2-evidence/window-2026-10-18/cubins/swiglu_mxfp8_quant_weighted_sm121.cubin
```

The same window first ran the c4a-2 corrected all-40-layer body (05:58:36 -> 05:59:29, recorded on its
own page); m17a's two runs followed:

```text
=== RUN m17a-weighted-tokens8 start 2026-10-19T05:59:29Z
=== RUN m17a-weighted-tokens8 exit=0 end 2026-10-19T05:59:41Z
=== RUN m17a-weighted-tokens130 start 2026-10-19T05:59:41Z
=== RUN m17a-weighted-tokens130 exit=0 end 2026-10-19T06:02:24Z
=== all runs complete 2026-10-19T06:02:24Z ===
=== restarting vLLM 2026-10-19T06:02:24Z ===
started tmux session vllm-window; waiting for health
```

The window log's last line is `started tmux session vllm-window; waiting for health`; the Lead
reported vLLM healthy after the window and restored (this page records the log's line, not a health
exit it does not contain). Wall times 12 s (tokens 8) and 163 s (tokens 130); lowest MemAvailable
reading during the runs 113.1 GiB (tokens 130, stage 2 rank 0 pre-launch), floor 32 GiB throughout.
Both runs: 35 checks, 4 stages, four pinned cubins (router 37,384 B / `5992cef9...`, gate/up
9,752 B / `115200d2...`, weighted epilogue 13,000 B / `0e602daf...`, combine 10,048 B /
`36a15a4f...`), `exit=0`. The records below are the complete per-run logs
`runs33/m17a-weighted-tokens8.log` and `runs33/m17a-weighted-tokens130.log`, verbatim.

The b33 binary (6443d5d0) was built before one verification path was rewired by review: in that
binary the real-width `routed_activation_operand` comparison was built with
`swiglu_ref::quantise_row_weighted`, which the crate's tests pin equal to
`routed_activation_operand` composed with SwiGLU. The follow-up commit makes the harness call
`mxfp4_ref::routed_activation_operand` itself (identical arithmetic, no device or launch change), so
the b33 PASS stands; the binary and the b33 record stay tied to the pre-fix source commit (main
`da8d491`, whose example source produced 6443d5d0) and were not rebuilt or re-run. No window is
needed for a comparison-code change that never touches the device path.

#### tokens 8 (weighted build, 35 checks; 48 routed rows, 37 distinct experts, 695,623,680 released expert bytes read)

```text
=== m17a: routed MoE branch at real layer-0 geometry (router, gate/up, SwiGLU+quantise, down/combine) ===
geometry: hidden 5120, intermediate 2304 = 4 rank slices of 576 (padded 640), routed experts 384, top-k 6; tokens 8
devices: four pinned cubins, each verified by size+sha256 before any allocation; each stage's comparison class is printed before the numbers it judges
what this is not: not the served forward pass, not a chain/token/per-layer claim, not a timing claim; the shared expert and the shared-add epilogue are m17b. The activation rows and the residual hidden are released embed.weight rows, a declared stand-in for layer 0's post-attention hidden.
MemAvailable at m17a start: 122316012 kB (116.6 GiB), floor 32 GiB - ok
checkpoint: /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
--- stage 1: moe_router_sm121 (released gate weight + noaux_tc bias) ---
comparison class (declared before any number): ids EXACT vs router_ref on all 8 x 6 selections; scores max abs delta <= 1e-6 and max ULP <= 8; weights max abs delta <= 1e-6 and max ULP <= 16; two launches byte-identical. The device's expf/log1pf are not libm's exp/ln_1p, which is the only known degree of freedom; ids are pinned because a tie-crossing selection would be a finding, not a tolerance (the tie rule is the kernel's strict-greater scan, lower expert id wins).
released operands: gate.weight BF16 3932160 B, gate.bias F32 1536 B (bias_vl not read)
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/moe_router_sm121.cubin (37384 B, sha256 5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b)
device 0: NVIDIA GB10 of 1 (total 124610 MiB, MemAvailable 119434 MiB)
stage 1 device bytes admitted: 4028288
MemAvailable at m17a stage 1 pre-launch: 122051508 kB (116.4 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 1 post-launch: 122051508 kB (116.4 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 48)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 1)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 2)
  check stage 1: two launches byte-identical: ok (12672 raw output bytes compared)
stage 1 selection: 48 routed rows; 37 distinct experts over the batch; ids/weights/scores read back
--- stage 2: routed_gate_up_sm121, 4 rank slices x 2 projections (independent launches) ---
comparison class (declared before any number): device w1 and w3 outputs BIT-IDENTICAL (ULP 0) to routed_gate_up_ref on every tokens*6*640 bf16 value of the projection, columns 576..639 exactly zero, two launches byte-identical. Twin is bit-identical to the shared b2_harness arithmetic by construction of the crate's tests; the 8 launches are independent (same activation, disjoint weights and outputs), so a later chain can stream them.
routed expert table: 37 distinct experts selected by the device router, staged as local ids 0..37; every later stage indexes this table, and the mapping is printed below
  local 0 = expert 250
  local 1 = expert 104
  local 2 = expert 64
  local 3 = expert 355
  local 4 = expert 304
  local 5 = expert 317
  local 6 = expert 16
  local 7 = expert 79
  local 8 = expert 360
  local 9 = expert 33
  local 10 = expert 117
  local 11 = expert 247
  local 12 = expert 295
  local 13 = expert 212
  local 14 = expert 157
  local 15 = expert 357
  ... 21 more entries
activation stand-in: 8 embed rows x 5120 bf16 values, D8-quantised to 40960 E4M3 bytes + 1280 ue8m0 bytes
  staged expert 250 (local 0) of 37
  staged expert 296 (local 36) of 37
  check stage 2: staged expert tensor names are the released global ids: ok (0 of 222 expected released expert tensor names missing from the 225 names actually read)
expert bytes: 37 experts x (w1/w3/w2 weights + scales), 695623680 B read verbatim from released tensors
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/routed_gate_up_sm121.cubin (9752 B, sha256 115200d26c3404b7169c476e350f5575dadc3fec9e8b0ec97113cd297fa6c1af)
  rank 0: device bytes admitted 116102592 (110.7 MiB); w window rows 0..576
MemAvailable at m17a stage 2 rank 0 pre-launch: 121154224 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 0 post-launch: 121162692 kB (115.5 GiB), floor 32 GiB - ok
  check stage 2 rank 0: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 0: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 0: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  rank 1: device bytes admitted 116102592 (110.7 MiB); w window rows 576..1152
MemAvailable at m17a stage 2 rank 1 pre-launch: 121170756 kB (115.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 1 post-launch: 121170756 kB (115.6 GiB), floor 32 GiB - ok
  check stage 2 rank 1: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 1: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 1: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  rank 2: device bytes admitted 116102592 (110.7 MiB); w window rows 1152..1728
MemAvailable at m17a stage 2 rank 2 pre-launch: 121187136 kB (115.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 2 post-launch: 121187136 kB (115.6 GiB), floor 32 GiB - ok
  check stage 2 rank 2: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 2: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 2: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
  rank 3: device bytes admitted 116102592 (110.7 MiB); w window rows 1728..2304
MemAvailable at m17a stage 2 rank 3 pre-launch: 121203516 kB (115.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 3 post-launch: 121203516 kB (115.6 GiB), floor 32 GiB - ok
  check stage 2 rank 3: both projections bit-identical: ok (0 mismatched bf16 values of 61440, first at None)
  check stage 2 rank 3: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 3: two launches byte-identical: ok (61440 gate + 61440 up bytes compared per projection)
--- stage 3: swiglu_mxfp8_quant_weighted_sm121 on the device gate/up bf16 values, with the device router's per-row weights as the weight operand ---
comparison class (declared before any number): E4M3 codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row_weighted over the same 640 bf16 values and the same per-row routing weight, for every routed row of every rank; the REAL 576-wide operand is additionally bit-exact against mxfp4_ref::routed_activation_operand over the same real values and weight (the model operand the combine consumes); two launches byte-identical. The weight operand is the router's device output buffer weights_d, read in place, so the operand is device-built and no host multiply or host quantise remains in the path (the F6 seam is closed).
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/swiglu_mxfp8_quant_weighted_sm121.cubin (13000 B, sha256 6b237e6c20301ddeda7f47ff9add5385a19e789d818388c5eb22981f1af5377e)
  rank 0: rows 48, cols 640, device bytes admitted 154560, weights_d reused in place
MemAvailable at m17a stage 3 rank 0 pre-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 0 post-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
  check stage 3 rank 0: weighted E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 0: weighted ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 0: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 28512 code+scale bytes in the real width)
  check stage 3 rank 0: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  rank 1: rows 48, cols 640, device bytes admitted 154560, weights_d reused in place
MemAvailable at m17a stage 3 rank 1 pre-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 1 post-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
  check stage 3 rank 1: weighted E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 1: weighted ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 1: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 28512 code+scale bytes in the real width)
  check stage 3 rank 1: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  rank 2: rows 48, cols 640, device bytes admitted 154560, weights_d reused in place
MemAvailable at m17a stage 3 rank 2 pre-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 2 post-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
  check stage 3 rank 2: weighted E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 2: weighted ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 2: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 28512 code+scale bytes in the real width)
  check stage 3 rank 2: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
  rank 3: rows 48, cols 640, device bytes admitted 154560, weights_d reused in place
MemAvailable at m17a stage 3 rank 3 pre-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 3 post-launch: 121317756 kB (115.7 GiB), floor 32 GiB - ok
  check stage 3 rank 3: weighted E4M3 codes bit-exact: ok (0 mismatched of 30720)
  check stage 3 rank 3: weighted ue8m0 scales bit-exact: ok (0 mismatched of 960)
  check stage 3 rank 3: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 28512 code+scale bytes in the real width)
  check stage 3 rank 3: two launches byte-identical: ok (30720 code + 960 scale bytes compared)
stage 3 output: device-built weighted operand per rank, 48 rows x 640 E4M3 codes + 20 ue8m0 scales (the real 576 columns equal the model's routed_activation_operand)
F6 seam closed: device router weights_d -> weighted epilogue -> device codes/scales -> combine; the host no longer multiplies the routing weight or quantises the operand
--- stage 4: b12x_mxfp4_moe_sm121 down/combine, 4 serial rank launches (residual chain) ---
comparison class (declared before any number): device bf16 BIT-EXACT against the chained mxfp4_ref::moe_combine over the SAME device-built weighted operand bytes (stage 3's codes/scales), with the same w2 slices and ids (four ranks, each rank's bf16 output is the next rank's residual hidden, so the chained rounding is part of both sides); two full chains byte-identical. Unlike b27, the activation operand is NOT host-quantised: stage 3's weighted epilogue produced it on the device from the router's weight buffer, so the F6 host seam is out of this path.
F6 seam crossing: device gate/up bf16 [rows 48][640] per rank -> device weighted epilogue (routing weight operand, no host round trip) -> device operand [rows 48][640] E4M3 + [rows 48][20] ue8m0 per rank -> device combine
w2 rank slices: 4 ranks x (60620800 packed + 3788800 scale bytes) from the released w2 tensors, each stored row sliced to 576 and padded to 640
reference: mxfp4_ref::moe_combine chained over the 4 rank slices fed by the SAME device-built weighted operand bytes, residual seed = the declared bf16 embed stand-in
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/b12x_mxfp4_moe_sm121.cubin (10048 B, sha256 36a15a4f3ae12b4e7d452c1201b267d3102afa43f168bd322d6930ff3f543057)
  chain 1: device bytes admitted 64605312
MemAvailable at m17a stage 4 chain 1 rank 0 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 0 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 1 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 1 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 2 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 2 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 3 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 3 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
  chain 2: device bytes admitted 64605312
MemAvailable at m17a stage 4 chain 2 rank 0 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 0 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 1 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 1 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 2 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 2 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 3 pre-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 3 post-launch: 121135200 kB (115.5 GiB), floor 32 GiB - ok
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 40960)
  check stage 4: two full chains byte-identical: ok (40960 bf16 codes compared)
stage 4 output: 8 token rows x 5120 bf16 codes (residual + weighted routed sum over 6 slots and 4 ranks)
MemAvailable at m17a end: 121200736 kB (115.6 GiB), floor 32 GiB - ok
--- NOT claimed by m17a ---
not a chain/token/per-layer claim, not layer text (activation and residual are released embed rows), not the shared expert or the shared-add epilogue (m17b), not engine integration, not host-time saved, not a performance claim. The routed combine operand is now device-built (the weighted epilogue closes the F6 host seam); what remains declared is the activation and residual stand-in. Stage 2's 8 launches are independent; stage 4's 4 launches are serial by construction (each rank's bf16 output is the next rank's residual).
m17a: PASS (35 checks; 4 stages; tokens 8; four pinned cubins)
```

#### tokens 130 (weighted build, 35 checks; 780 routed rows, 148 distinct experts, 2,782,494,720 released expert bytes read)

```text
=== m17a: routed MoE branch at real layer-0 geometry (router, gate/up, SwiGLU+quantise, down/combine) ===
geometry: hidden 5120, intermediate 2304 = 4 rank slices of 576 (padded 640), routed experts 384, top-k 6; tokens 130
devices: four pinned cubins, each verified by size+sha256 before any allocation; each stage's comparison class is printed before the numbers it judges
what this is not: not the served forward pass, not a chain/token/per-layer claim, not a timing claim; the shared expert and the shared-add epilogue are m17b. The activation rows and the residual hidden are released embed.weight rows, a declared stand-in for layer 0's post-attention hidden.
MemAvailable at m17a start: 122076996 kB (116.4 GiB), floor 32 GiB - ok
checkpoint: /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
--- stage 1: moe_router_sm121 (released gate weight + noaux_tc bias) ---
comparison class (declared before any number): ids EXACT vs router_ref on all 130 x 6 selections; scores max abs delta <= 1e-6 and max ULP <= 8; weights max abs delta <= 1e-6 and max ULP <= 16; two launches byte-identical. The device's expf/log1pf are not libm's exp/ln_1p, which is the only known degree of freedom; ids are pinned because a tie-crossing selection would be a finding, not a tolerance (the tie rule is the kernel's strict-greater scan, lower expert id wins).
released operands: gate.weight BF16 3932160 B, gate.bias F32 1536 B (bias_vl not read)
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/moe_router_sm121.cubin (37384 B, sha256 5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b)
device 0: NVIDIA GB10 of 1 (total 124610 MiB, MemAvailable 119160 MiB)
stage 1 device bytes admitted: 5470816
MemAvailable at m17a stage 1 pre-launch: 121776596 kB (116.1 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 1 post-launch: 121776616 kB (116.1 GiB), floor 32 GiB - ok
  check stage 1: device ids exact vs twin: ok (0 mismatched selections of 780)
  check stage 1: scores within 1e-6 / 8 ULP: ok (max abs delta 1.1920928955078125e-7 at 1207, max ULP 2)
  check stage 1: weights within 1e-6 / 16 ULP: ok (max abs delta 5.960464477539063e-8 at 42, max ULP 4)
  check stage 1: two launches byte-identical: ok (205920 raw output bytes compared)
stage 1 selection: 780 routed rows; 148 distinct experts over the batch; ids/weights/scores read back
--- stage 2: routed_gate_up_sm121, 4 rank slices x 2 projections (independent launches) ---
comparison class (declared before any number): device w1 and w3 outputs BIT-IDENTICAL (ULP 0) to routed_gate_up_ref on every tokens*6*640 bf16 value of the projection, columns 576..639 exactly zero, two launches byte-identical. Twin is bit-identical to the shared b2_harness arithmetic by construction of the crate's tests; the 8 launches are independent (same activation, disjoint weights and outputs), so a later chain can stream them.
routed expert table: 148 distinct experts selected by the device router, staged as local ids 0..148; every later stage indexes this table, and the mapping is printed below
  local 0 = expert 250
  local 1 = expert 104
  local 2 = expert 64
  local 3 = expert 355
  local 4 = expert 304
  local 5 = expert 317
  local 6 = expert 16
  local 7 = expert 79
  local 8 = expert 360
  local 9 = expert 33
  local 10 = expert 117
  local 11 = expert 247
  local 12 = expert 295
  local 13 = expert 212
  local 14 = expert 157
  local 15 = expert 357
  ... 132 more entries
activation stand-in: 130 embed rows x 5120 bf16 values, D8-quantised to 665600 E4M3 bytes + 20800 ue8m0 bytes
  staged expert 250 (local 0) of 148
  staged expert 311 (local 64) of 148
  staged expert 273 (local 128) of 148
  staged expert 206 (local 147) of 148
  check stage 2: staged expert tensor names are the released global ids: ok (0 of 888 expected released expert tensor names missing from the 891 names actually read)
expert bytes: 148 experts x (w1/w3/w2 weights + scales), 2782494720 B read verbatim from released tensors
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/routed_gate_up_sm121.cubin (9752 B, sha256 115200d26c3404b7169c476e350f5575dadc3fec9e8b0ec97113cd297fa6c1af)
  rank 0: device bytes admitted 466435440 (444.8 MiB); w window rows 0..576
MemAvailable at m17a stage 2 rank 0 pre-launch: 118574992 kB (113.1 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 0 post-launch: 118576180 kB (113.1 GiB), floor 32 GiB - ok
  check stage 2 rank 0: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 0: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 0: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  rank 1: device bytes admitted 466435440 (444.8 MiB); w window rows 576..1152
MemAvailable at m17a stage 2 rank 1 pre-launch: 118999472 kB (113.5 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 1 post-launch: 118999472 kB (113.5 GiB), floor 32 GiB - ok
  check stage 2 rank 1: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 1: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 1: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  rank 2: device bytes admitted 466435440 (444.8 MiB); w window rows 1152..1728
MemAvailable at m17a stage 2 rank 2 pre-launch: 119272784 kB (113.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 2 post-launch: 119272784 kB (113.7 GiB), floor 32 GiB - ok
  check stage 2 rank 2: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 2: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 2: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
  rank 3: device bytes admitted 466435440 (444.8 MiB); w window rows 1728..2304
MemAvailable at m17a stage 2 rank 3 pre-launch: 119401764 kB (113.9 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 2 rank 3 post-launch: 119401764 kB (113.9 GiB), floor 32 GiB - ok
  check stage 2 rank 3: both projections bit-identical: ok (0 mismatched bf16 values of 998400, first at None)
  check stage 2 rank 3: columns 576..640 exactly zero: ok (0 nonzero padding values)
  check stage 2 rank 3: two launches byte-identical: ok (998400 gate + 998400 up bytes compared per projection)
--- stage 3: swiglu_mxfp8_quant_weighted_sm121 on the device gate/up bf16 values, with the device router's per-row weights as the weight operand ---
comparison class (declared before any number): E4M3 codes and ue8m0 scales BIT-EXACT against swiglu_ref::quantise_row_weighted over the same 640 bf16 values and the same per-row routing weight, for every routed row of every rank; the REAL 576-wide operand is additionally bit-exact against mxfp4_ref::routed_activation_operand over the same real values and weight (the model operand the combine consumes); two launches byte-identical. The weight operand is the router's device output buffer weights_d, read in place, so the operand is device-built and no host multiply or host quantise remains in the path (the F6 seam is closed).
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/swiglu_mxfp8_quant_weighted_sm121.cubin (13000 B, sha256 6b237e6c20301ddeda7f47ff9add5385a19e789d818388c5eb22981f1af5377e)
  rank 0: rows 780, cols 640, device bytes admitted 2511600, weights_d reused in place
MemAvailable at m17a stage 3 rank 0 pre-launch: 120282280 kB (114.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 0 post-launch: 120282280 kB (114.7 GiB), floor 32 GiB - ok
  check stage 3 rank 0: weighted E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 0: weighted ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 0: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 463320 code+scale bytes in the real width)
  check stage 3 rank 0: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  rank 1: rows 780, cols 640, device bytes admitted 2511600, weights_d reused in place
MemAvailable at m17a stage 3 rank 1 pre-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 1 post-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
  check stage 3 rank 1: weighted E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 1: weighted ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 1: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 463320 code+scale bytes in the real width)
  check stage 3 rank 1: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  rank 2: rows 780, cols 640, device bytes admitted 2511600, weights_d reused in place
MemAvailable at m17a stage 3 rank 2 pre-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 2 post-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
  check stage 3 rank 2: weighted E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 2: weighted ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 2: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 463320 code+scale bytes in the real width)
  check stage 3 rank 2: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
  rank 3: rows 780, cols 640, device bytes admitted 2511600, weights_d reused in place
MemAvailable at m17a stage 3 rank 3 pre-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 3 rank 3 post-launch: 120282296 kB (114.7 GiB), floor 32 GiB - ok
  check stage 3 rank 3: weighted E4M3 codes bit-exact: ok (0 mismatched of 499200)
  check stage 3 rank 3: weighted ue8m0 scales bit-exact: ok (0 mismatched of 15600)
  check stage 3 rank 3: real 576-wide operand bit-exact vs routed_activation_operand: ok (0 mismatched of 463320 code+scale bytes in the real width)
  check stage 3 rank 3: two launches byte-identical: ok (499200 code + 15600 scale bytes compared)
stage 3 output: device-built weighted operand per rank, 780 rows x 640 E4M3 codes + 20 ue8m0 scales (the real 576 columns equal the model's routed_activation_operand)
F6 seam closed: device router weights_d -> weighted epilogue -> device codes/scales -> combine; the host no longer multiplies the routing weight or quantises the operand
--- stage 4: b12x_mxfp4_moe_sm121 down/combine, 4 serial rank launches (residual chain) ---
comparison class (declared before any number): device bf16 BIT-EXACT against the chained mxfp4_ref::moe_combine over the SAME device-built weighted operand bytes (stage 3's codes/scales), with the same w2 slices and ids (four ranks, each rank's bf16 output is the next rank's residual hidden, so the chained rounding is part of both sides); two full chains byte-identical. Unlike b27, the activation operand is NOT host-quantised: stage 3's weighted epilogue produced it on the device from the router's weight buffer, so the F6 host seam is out of this path.
F6 seam crossing: device gate/up bf16 [rows 780][640] per rank -> device weighted epilogue (routing weight operand, no host round trip) -> device operand [rows 780][640] E4M3 + [rows 780][20] ue8m0 per rank -> device combine
w2 rank slices: 4 ranks x (242483200 packed + 15155200 scale bytes) from the released w2 tensors, each stored row sliced to 576 and padded to 640
reference: mxfp4_ref::moe_combine chained over the 4 rank slices fed by the SAME device-built weighted operand bytes, residual seed = the declared bf16 embed stand-in
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/b12x_mxfp4_moe_sm121.cubin (10048 B, sha256 36a15a4f3ae12b4e7d452c1201b267d3102afa43f168bd322d6930ff3f543057)
  chain 1: device bytes admitted 260818720
MemAvailable at m17a stage 4 chain 1 rank 0 pre-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 0 post-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 1 pre-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 1 post-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 2 pre-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 2 post-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 3 pre-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 1 rank 3 post-launch: 119116284 kB (113.6 GiB), floor 32 GiB - ok
  chain 2: device bytes admitted 260818720
MemAvailable at m17a stage 4 chain 2 rank 0 pre-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 0 post-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 1 pre-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 1 post-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 2 pre-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 2 post-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 3 pre-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
MemAvailable at m17a stage 4 chain 2 rank 3 post-launch: 119116348 kB (113.6 GiB), floor 32 GiB - ok
  check stage 4: device chain bit-exact vs moe_combine chain: ok (0 mismatched bf16 codes of 665600)
  check stage 4: two full chains byte-identical: ok (665600 bf16 codes compared)
stage 4 output: 130 token rows x 5120 bf16 codes (residual + weighted routed sum over 6 slots and 4 ranks)
MemAvailable at m17a end: 119372412 kB (113.8 GiB), floor 32 GiB - ok
--- NOT claimed by m17a ---
not a chain/token/per-layer claim, not layer text (activation and residual are released embed rows), not the shared expert or the shared-add epilogue (m17b), not engine integration, not host-time saved, not a performance claim. The routed combine operand is now device-built (the weighted epilogue closes the F6 host seam); what remains declared is the activation and residual stand-in. Stage 2's 8 launches are independent; stage 4's 4 launches are serial by construction (each rank's bf16 output is the next rank's residual).
m17a: PASS (35 checks; 4 stages; tokens 130; four pinned cubins)
```

## NOT claimed

- Not a claim that the whole routed branch is host-free: the routed epilogue itself is now device-built
  (D33, window b33 PASS) - the F6 weight multiply and the activation quantiser for the combine operand
  are not host steps in this path. What remains host in the harness is the comparison twins and the
  declared activation/residual stand-ins, not the operand path.
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
