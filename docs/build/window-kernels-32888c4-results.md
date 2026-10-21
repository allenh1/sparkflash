# Kernel occupancy window: c62fb0c base vs 32888c4 split-K, node0 2026-10-20

Window: 21:23:52-21:24:16Z (17:23:52-17:24:16 EDT), 12 invocations in 24 s, hard per-invocation
timeout 150 s, internal budget guard 780 s. Host node0, NVIDIA GB10, driver 580.173.02, kernel
Linux 6.17.0-1029-nvidia aarch64, 20 cpus. Script scripts/window-kernels.sh at commit eda1d1e
(sha256 06c4eb16...; the staged copy differs from the 32888c4 commit only by the mid-window
attn_projections bar fix, see "Script corrections"). Base stage c62fb0c binaries+cubins, new stage
32888c4 binaries+cubins; the two stages are the ones whose sha256 manifests are in provenance.txt.
Evidence: /home/user/ev-session-20261020/window/kernels-32888c4/ (summary.txt, provenance.txt,
12 logs). This page adjudicates that evidence; it does not add measurements.

## 1. Adjudication table (from summary.txt, verified against the logs)

| harness | verdict | base p50 ms | new p50 ms | speedup | roofline |
|---|---|---|---|---|---|
| routed_moe_gate_up | **BIT-IDENTICAL** | 0.9398 (w1), 0.9443 (w3) | 0.3805 (w1), 0.3845 (w3) | 2.47 / 2.46 | 12.1% (w1), 11.9% (w3) |
| routed_moe_down | **no-go (unadjudicable)** | 0.6552 | 0.4183 | 1.57 | 9.3% corrected, see section 3 |
| router | PASS | 0.2309 | 0.2279 | 1.01 | - |
| lm_reduce_batched | PASS | 2.4819, 2.4719 | 2.4897, 2.4740 | 1.00 / 1.00 | - |
| attn_projections | PASS | 11.7505, 6.3180, 12.0106 | 11.7488, 6.3363, 12.0096 | 1.00 x3 | - |
| cert_lm_head | PASS | 3.9864, 11.0936 | 3.9779, 11.1161 | 1.00 / 1.00 | 11.0% -> 11.0% |

Five of six rows are PASS or BIT-IDENTICAL. One row (routed_moe_down) is no-go for a reason that is
stated in section 3 and is a defect in this session's window invocation, not in either kernel. Under
the pre-agreed rule - any FAIL or no-go keeps the incumbent pins - the branch was not merged and the
pinned baseline stays at c62fb0c. The gate/up certification below is unaffected: it is a different
kernel, a different harness and a green bar in both stages.

## 2. Certified: the routed gate/up rewrite is bit-identical and 2.47x faster per launch

Both stages run m10_gate_up, which compares the device output against the shared CPU twin on the
released layer-0 expert bytes and prints its bar before the numbers. The decisive lines, identical in
class in both logs:

- base: "device vs twin: gate 3840/3840 bf16 values exact, up 3840/3840 exact; padding nonzero 0;
  two runs byte-identical: true"
- new: "device vs twin: gate 3840/3840 bf16 values exact, up 3840/3840 exact; padding nonzero 0;
  two runs byte-identical: true"
- base: "PASS  cubin is the pinned one: expected 9752 B / 115200d2...c1af, got 9752 B / 115200d2..."
- new: "PASS  cubin is the pinned one: expected 15672 B / 12ddc097...ac3d, got 15672 B / 12ddc097..."
- both: all 8 verdicts PASS, including "device gate is bit-identical to the twin: 3840/3840",
  "device up is bit-identical to the twin: 3840/3840", "the padding columns are exactly zero",
  "two device runs are byte-identical".

Both stages therefore produce bytes equal to the same twin on the same operands, so the two cubins
produce the same bytes: this is the byte-identity criterion the Lead set, not a tolerance bar. Timings
per launch (100 launches, same buffers): w1 0.9398 -> 0.3805 ms (2.47x), w3 0.9443 -> 0.3845 ms
(2.46x). Roofline against the 273 GB/s per-node figure: new 12,539,040 traffic bytes in 0.3805 ms =
32.95 GB/s = 12.1% of node (w3 32.61 GB/s = 11.9%); the base cubin implies 13.34 GB/s = 4.9%. At
tokens=1 this projection is still far from the bandwidth floor - the win is issue and occupancy, not
DRAM - so a further win here needs a different schedule (or fewer launches), not a bandwidth fix.

## 3. Unadjudicable row: routed_moe_down, and why (this session's invocation is invalid)

The harness loads one tensor-parallel rank's stored 576-K slice, pads it to the kernel's 640-K stride,
and prints "rank accumulator: N/30720 ... bit-exact against mxfp4_ref::moe_combine_rank" after
zeroing the f32 slot accumulator once and launching. The kernel ADDS into that accumulator; with
--repeat 200 the harness runs one warm-up launch plus 200 timed launches against the same buffer, so
the downloaded accumulator is 201 launches' worth of rank partials while the reference is one launch.
The comparison is invalid by construction at --repeat > 1, for either cubin.

That is exactly what the logs show, identically in both stages:
"rank accumulator: 0/30720 f32 partials bit-exact", "exact bf16 matches: 1 (0.0195%)",
"max abs delta: 6.382656e2" in both base and new. The identical statistics are consistent with the two
kernels being value-neutral, but they are NOT proof of byte identity: two different outputs could
share a maximum delta. The row says nothing about either kernel, and this page does not use it as a
claim. The timed numbers (0.6552 -> 0.4183 ms, 1.57x) are a per-launch kernel measurement only.

Corrected roofline for this shape: the harness's reported traffic_bytes=111,538,040 counted all 64
staged experts, but the kernel reads only the 6 routed experts (6 x 5120 x 340 B = 10,444,800 B) plus
the activation operands and the f32 slot accumulator, about 10.57 MB. On that traffic the new launch
is 10.57 MB / 0.4183 ms = 25.3 GB/s = 9.3% of node (base 16.1 GB/s = 5.9%), not the 97.7% the
instrumented line printed. The roofline helper in examples/moe_slice.rs must count routed weight
bytes, not the staged arena; that is a fix for the next session's re-run, listed below.

## 4. Ancillary rows: the unchanged kernels did not move

All four non-MoE rows are PASS with the same bars as the base stage: router 0.2309 -> 0.2279 ms
(0.0289 -> 0.0285 ms/token, 1.01x, "7 of 7 checks"); lm_reduce_batched EXACT (1.00x); attn
projections bit-identical to the in-crate reference, 3 p50s at 1.00x; cert head six rows
byte-identical to the scalar head, 1.00x, its self-reported roofline 11.0% in both stages. None of
these kernels changed, and the window shows no regression.

## 5. Script corrections made around this window

1. attn_projections bar (fixed before the slot, commit eda1d1e). The script required >=10 ^PASS
   lines; m12 prints one final verdict line and only failed checks individually. Its row is now
   final-line. Without this fix a green m12 would have failed the window for a script reason.
2. routed_moe_down row (to fix next session). Correctness and timing cannot come from one
   invocation: correctness needs --repeat 1 (one launch, so the accumulator comparison is valid),
   timing needs --repeat 200 (where the comparison is meaningless). The row must become two rows:
   a correctness row at --repeat 1 carrying the twin bar, and a timing row at --repeat 200 whose bar
   is only "exit 0 and a per-launch ms: line" - no correctness claim from it.
3. moe_slice roofline traffic (to fix next session). Count routed expert bytes
   (tokens * TOP_K * 5120 * (320 + 20)) instead of the whole staged arena.

## 6. What is proven, what is labelled, what is not claimed

Proven (device-measured, this window): the routed gate/up cubin at 32888c4 produces bytes identical
to its CPU twin on the released layer-0 operands, exactly as the c62fb0c cubin does, and is 2.47x
faster per launch at tokens=1; its pin line and its padding line both hold; two device runs are
byte-identical.

Labelled, not proven: routed_moe_down is value-neutral by the identical base/new comparison
statistics (section 3) and by the rewrite's arithmetic argument (same block terms, same ascending
accumulation order), but the green twin bar was not obtained, so it is uncertified tonight.

Not claimed: any per-position step time or end-to-end gain from these per-launch numbers; any
change to the dense, attention, router, LM-head or shared-expert kernels (none moved); any
bandwidth-limit claim from the 12% gate/up figure.

## 7. Next-session prerequisite (owner: kernel-lead), in order

1. moe_slice with --repeat 1 in both stages (no --repeat flag is the same thing): expected
   "rank accumulator: 30720/30720" and "exact bf16 matches: 30720 (100.0000%)". If green, the down
   kernel is certified and the branch can merge with the two pins (11,704 B ccb9d127... and 15,672 B
   12ddc097...). If not green, stop and investigate with an output dump before any merge.
2. Apply the two script/example fixes in section 5, re-run m10 + moe_slice (correctness and timing
   rows) both stages in one window, and adjudicate from the new summary. Nothing else needs a re-run:
   the other four rows are green and their kernels are unchanged.
3. Only then merge task/kernel-occupancy; keep the c62fb0c pins until item 1 is green.
