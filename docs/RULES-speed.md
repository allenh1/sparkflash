# Speed-track rules (in force 2026-10-21)

These rules govern the 90 tok/s track. They came from a direct user directive after a phase in which
the team produced measurement records, re-measurements and "no tok/s claim" disclaimers while the
number did not move. If a rule here conflicts with an older document's process, this file wins.

## 1. One published number

* The only speed metric is **lean decode ms/step, mean over the four ranks**, same frozen argv.
* It is **posted daily** by appending to [speed-log.csv](build/speed-log.csv)
  (`scripts/speed-log.sh append --run <window run dir>`); `scripts/speed-log.sh print` renders it.
* Every merge request states its **expected effect on ms/step before work starts**. A change with no
  expected effect on the metric is not speed work.

Current: **179.81 ms** (wsd28-host3b, 2026-10-22, integration cd233423 = Batch 4 + seam warm pass +
host-weight GEMV seam arm + attention staging slot + mHC seam host-SIMD projection + one-pass
activation quantiser + the seams' remaining encoder and thread-count tax; four-rank mean; identity
PASS; clean-row window of section 2c). Replicate 180.00 (wsd28-host3a).
Two tail costs inside the seams turned out to be the same two bugs as earlier blocks, in different
places: the FFN seam called the *plain* fp8 quantiser, which still encoded E4M3 by bisecting the 127
finite codes (62-92 ns/value; now the bit encoder, ~5x), and every host parallel helper called
`available_parallelism()` per call, which reads the cgroup CPU-quota files: 27.6 us per call against
0.15 us for the env read next to it. Cached in a `OnceLock` now; `rms_norm_bf16` alone went 38.2 ->
7.5 us on the dev box. -7.05 ms/step against the 186.86 tree. 27.8 % below the 249.25 of the night's
start.
The activation quantiser was a bisection over the 127 finite E4M3 codes with a decode per probe - 78 ns
per value, about eight times the conversion it denotes. Replaced by a bit-arithmetic encoder with the
same contract (RNE ties-to-even, saturate 0x7E, NaN -> 0x7F) plus the two-step BF16-narrow-then-quantise
fused into one pass; the *exhaustive* proof is a test over all 2^32 f32 bit patterns with zero
mismatches. -11.9 ms/step against the 198.78 tree. 25.0 % below the 249.25 of the night's start.
The mHC seams' coefficient projection (24 rows x 20480 f32, one ascending-k chain per row) was moved
from one-row-at-a-time to k-outer/rows-inner over a transposed staging copy, with the multiply and add
kept separate so the bytes are unchanged (transposed-vs-ordered is asserted bit-for-bit at every layer
before the loop starts): -15.0 ms/step against the 213.78 tree, on the 12-16 predicted. 20.2 % below
the 249.25 of the night's start.
The staging slot is -10.75 ms/step against the 224.53 of wsd22, in line with its -11.5 host-side price:
`stage_attention` rebuilt its CSA operands on every (layer, position) call - fresh window buffers, a
zero-filled 147 KB expanded store, and `quantize_main_kv` over every selected compressed row - and now
keeps them at run scope, clears only the rows the previous call wrote, and memoises the quantised rows
with a verified-bits hit test. 14.2 % below the 249.25 of the night's start.
Per-block after the seam arm, from the host-only probe on the 224.53 tree (per call, x40):
`ffn_seams` 0.524, `attention_output` 0.512 host + 0.552 wait, `moe_reduce` 0.378, `mhc_entry` 0.323,
`stage_attention` 0.321 (now ~0.031), `attention_projections` 0.234, `query_path` 0.094, `ref_seam`
0.077 - the seam arm cut `ref_seam` 0.270 -> 0.077 and `query_path` 0.120 -> 0.094, which is why its
-2.8 row result was smaller than the block deltas (the other seams absorbed the transfers).
Recorded negatives from the same windows: the posted-reduce lane 246.56 against the sync reference
228.03 (+18.5, reverted), and borrowing the attention window 227.17/227.03 against the 227.31 tree
(neutral, reverted).
Path: 1201.81 (start) ->
599.94 (WS-A2 batch, 4e4cea50) -> 469.81 (M=1 dense re-tile, 0d8797db) -> 356.39 (decode GEMV for
wq_a/wkv/wq_b/wo_b, 528b39fb) -> 350.39 (shared-expert wiring, 4f83eee5; partial - the device arm
was gated off, see 2f) -> 256.17 (shared expert's f32 reference arm on the device, d6020787,
bit-identical: 3 GEMV launches/layer at 171.1 GB/s kernel-side = 62.7 % of peak, host block
0.277 ms/layer against 2.623) -> 249.25 (wsd6-lean, 3813ecbe) -> 227.31 (Batch 4 + seam warm pass
default-on, 7ecec4c2) -> 224.53 (host-weight GEMV seam arm, 54abfd6c) -> 213.78 (attention staging
slot, 7d176935) -> 198.78 (mHC seam host-SIMD projection, 7bf7ec78) -> 186.86 (one-pass activation
quantiser, 3620ac77) -> 179.81 (seam tails: bit encoder + cached thread count, cd233423). Rows and dates live in [speed-log.csv](build/speed-log.csv); that table is the
record, not a document.

**The number is not comparable across base commits.** 3813ecbe and every 268f2ab1-based tree
differ by ~+15 ms/step on clean steps and ~+70 ms/position on prefill; the `!158` O_DIRECT loader is
the only change between them. A candidate number is publishable only against a row of the same base
in the same window, which is what section 2c makes possible.

### 1b. Where the 249.25 actually is, and why kernel wins do not move it

Three windows of measurements (2026-10-22) established an ordering constraint that now governs
the path to 150. The record with all rows is
[ws-gb10-host-device-contention-20261022](build/ws-gb10-host-device-contention-20261022.md):

- The step is **host-bound per block**, not device-bound. In the probe, `attention_output` is
  0.837 ms/layer wall against 0.202 kernel, `moe_gate_up` 1.894 against 0.637 - the host share is
  larger than the kernel everywhere except the two GEMV blocks that are already at 62-72 % of peak.
  Total device kernel time is 71.2 of the 249.25.
- **A kernel win moves the step only if the block's host share is already below its kernel time.**
  `wo_a` off the 8-block launch (`0.4879 -> 0.2022 ms/layer`, the biggest single-kernel win of the
  program so far) left the four-rank step flat at median 249 vs baseline 246.7, because that
  block's host share was 0.60 and the device time had been hidden behind it.
- **Host-side arithmetic rework makes the step worse, in every arm measured.** Interleaving the
  independent f32 chains in `mhc.rs` and/or `linalg.rs` (bit-identical, with pre-change `to_bits`
  pins) produced 378.4 (mHC only), 321.7 (linalg only) and 291.8 (both) against a 246.7 baseline,
  with a ~230 ms rank-wide stall that appears and clears. On this GB10 package the CPU and GPU share
  the memory system and the power envelope; a faster host slows the device.
- **Driver call count is not wall either.** Removing 198,720 D2D copies per step and all of the
  small-operand H2D count changes moved nothing, because with the async launch path those calls
  overlap device work and only the calls that force a drain are visible.

So the remaining levers are (a) **pipelining** the host seam of layer L+1 against the device work
of layer L, or (b) **removing** host seams or the bytes they touch - never speeding them up in
place. Both are design changes and both need a plan before rows are spent. Kernel throughput work
still has ~30 ms/step of headroom but it is currently hidden, so it is batched behind the
pipelining decision rather than rowed slice by slice.


## 2. Milestone gates are the only progress

**<= 400 ms -> <= 150 -> <= 50 -> <= 20 -> <= 11.1**. The 400 gate is cleared (179.81); next gate is 150. Anything else (a doc, a record, an unwired
kernel, a re-run that confirms the same number) is not progress.

## 2b. Batch windows; one row must answer a gate decision

A row is a fresh four-rank process: it pays the full checkpoint upload before the first decode step
(~6-7 min on four GB10s) whether it runs 9 steps or 90. Nine rows of that is an hour of cluster time
for 81 measured steps. So:

- Batch slices into one window. A window is justified when the accumulated expected movement is
  above the ~50 ms noise floor, or when a numerics-affecting slice needs its byte-identity row.
  Never run one window per slice.
- Do not file a row whose answer is already known from CI (identity gate, counter assertions) or
  from construction. A row exists to decide something.
- The probe row is for picking the next slice, not for reporting a delta. Run it when the block
  table is stale, not once per merge.
- **A row that ends at open is a result, not a number.** The first four-rank device probe row
  (pld1-probe2, 2026-10-22) had all four ranks exit 2 inside \`DeviceLayer::open\` with
  \`layer device seams: layer 0: layer device weight: no hc_attn_fn for layer 0 in the resolved set\`:
  the arm opened, the NCCL bootstrap agreed on four ranks, and the weight lookup refused before any
  stage walk, so the row wrote bind/loop/token JSON and its chain log but no \`tp4-layer-device\`
  frontier. That is the correct cheap outcome. Record it with
  \`python3 scripts/device-frontier.py --row-dir <row>\` -> verdict \`OPEN REFUSAL\` (exit 3), the
  verbatim blocker and the stage being opened (\`mhc_entry\` for layer 0). Never write it to
  speed-log.csv as a ms/step, never call it a frontier stage, and never file it as missing evidence
  (exit 2): the expensive part never ran, and no published number changes.
- Every row lands as one line in [speed-log.csv](build/speed-log.csv). No per-row documents.
- Batch merge requests the same way: one branch, one pipeline, one merge for a group of slices.

## 2c. Estimates come from the hardware floor

When an estimate misses by ~4x, the next estimate for that block must be derived from the hardware
floor - weight bytes / peak bandwidth, MACs / peak MACs - and never scaled from the previous
kernel's measured rate. Kernel work is reported as GB/s achieved against peak next to ms, and the
bar for a single-token launch is **>= 60 % of peak memory bandwidth**. Beating our own previous
kernel is not a result.

## 2d. "Host path closed" means zero host arithmetic per token

The WS-A host path is closed only when the decode step reaches **zero
crates/sparkflash-engine reference arithmetic per token** (the linalg calls and the
forward/attention.rs / forward/engram.rs reference paths reached from tp4_forward.rs). Each
remaining call gets a device kernel or a deletion; the compressor and the Engram projections are
the first two. Host arithmetic is a floor that no kernel work can remove, and it blocks capturing
the step as a CUDA graph.

## 2e. Current workstreams (2026-10-21, after the WS-D un-park)

| workstream | gate | who |
| :-- | :-- | :-- |
| WS-D decode GEMV (wq_a, wkv, wq_b, wo_b, shared expert) | >= 60 % of peak bandwidth per M=1 launch; target <= 400 ms | device-lead |
| WS-C CUDA graph capture of the device-only segments | measured launch/sync win, byte-identical; next step to <= 150 ms | driver-fixer |
| WS-B interconnect admin (MTU 9000, nvidia-peermem) | applied on 4/4 nodes and verified, not ticketed | comm-lead + root admin |
| WS-A remainder: engine calls still reached per token | list, then kernel-or-delete per call | device-lead |

## 3. Rank work by the P0 table

The biggest rows win: the zero-kernel host blocks (`ffn_shared`, `moe_reduce`, `mhc_entry`), the
per-launch drains, and the `moe_gate_up` host residual - currently
`attention_projections 5.17`, `moe_gate_up 4.77`, `ffn_shared 2.74`, `attention_output 2.39` ms/layer
with **12 D2H observation points per layer-position**. If a task does not touch a top-five row, it
waits.

## 4. CI owns the correctness check, and it is numerical (D54)

`scripts/numerical-gate.sh --run <window run dir> [...]` is the one command that gates a row: 4/4 rank
exit codes, greedy argmax tokens identical to the frozen reference, and the declared bars on the dumped
logits - top-1 agreement 1.00, top-5 set overlap >= 0.80, top-50 id overlap >= 0.96, max |delta
logprob| <= 2.0e-2 nats, mean |delta logprob| <= 2.0e-3 nats, KL(ref || run) <= 1.0e-4 nats over the
renormalised top-100, every value finite. The reference is the bit-exact CPU twin's published
instantiation, frozen in `docs/build/reference/speed-gate/<fixture>/` and keyed by sha256(positions,
prompt_len, ids): the twin stays the reference implementation and the product is allowed to differ
inside the bars.

**Byte identity is no longer the product gate.** `scripts/identity-gate.sh --run` stays as the strict
check, run when a slice claims bit identity (a pure layout or pointer change); it is not the acceptance
bar for a slice that deliberately changes summation order. The gate prints the fixture's smallest
top1-top2 margin with every row - 0.0038 nats at position 4 today - so a token flip lands in the record
beside the near tie that produced it (D39 rule 2). The rule is [D54](DECISIONS.md): it admits split
reductions, tree sums and device kernels that divide a sum, and it does not admit changed arithmetic,
non-finite values, or a bar loosened without a measured noise floor.

**A result the gate already proves gets no document, decision number or window re-run.**

## 5. Measure less, batch more

Re-run a measurement only when the expected step change is larger than the run-to-run noise
(**~50 ms** at today's 843 ms step). Otherwise batch several changes and measure once.

## 6. Cap the process

* **Decision records**: only for hard-to-reverse choices, at most about one per week; status is one
  line, not a page of caveats.
* **Investigations**: two days, then ship a change or drop the thread.
* **Inventory**: a measured but unwired kernel is inventory - wire it or delete it.
* **No host/CPU twins** of device paths. The device path is the product.

## 7. Unblock, don't work around

* The RDMA admin items (MTU, `peermem`) are escalated to someone with root this week; they are the
  only thing between WS-B and its 25 us gate.
* Each person owns one workstream (WS-A..WS-G) with an explicit exit gate.

## 8. The review question

Ask only: **what did this move** (ms/step, TTFT, or open API gaps) and **what is the next biggest bar
in the table, and who is on it**? "Nothing, but it is confirmed" is rejected.

## 9. Risk is acceptable

A short-lived regression caught by CI and reverted fast is cheaper than slow proof. Correctness is
checked by `numerical-gate.sh` in one command, so a break is cheap to detect and cheap to revert.

## 10. Focus until the gate moves

For the next two weeks the speed team works **only on WS-A**: move the host blocks onto the device
and remove the per-launch drains and the 12 D2H observation points. WS-C (graphs), WS-D (kernel
efficiency), WS-E (MTP), WS-F (prefill/serving) and WS-G (parity) stay parked until the lean step is
below 400 ms.## 2c. The clean-row protocol (in force 2026-10-22)

A row is only meaningful if the nodes start it from a defined memory state. On 2026-10-22 the same
binary measured 266 and then 500 ms/step eight minutes apart. Fifteen rows later the cause was
measured, not guessed: during a "500" row rank 0 read **1.29 GB per 1.5 s from the block device**
(`read_bytes == rchar`, every byte a cache miss) with the host at **0.27 cores**, `pgmajfault = 0`,
`pswpin = 0` and a 76 GB page-locked arena - the position loop was re-reading the checkpoint bytes
behind the host seams that the `!158` O_DIRECT loader never puts in the page cache.

Before every row:

1. `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` over the checkpoint shards **on all four nodes**
   (`/tmp/dropcache.py`; no root needed). This moved `MemFree` from 864 MB to 40 GB on .80 and
   removed the mode for the next 15 consecutive rows (every band inside 224-258 ms, no stalls).
2. Zero `sparkflash-*` processes and MemFree >= ~110 GiB per node before launch.
3. Record the **clean-step band and the median, not only the mean**; one mode step moves a 9-step
   mean by ~25 ms and three of them by ~200 ms.
4. A delta below ~20 ms needs two interleaved rows (A/B/A/B); one row is +/-5 ms on this cluster.
5. Never stage while a row is running, and never compare numbers across base commits.

The warm pass (`SPARKFLASH_SEAM_WARM`, default on since d7ffd9f7) is the fix for the mechanism: it
reads the seam tensors and the Engram rows into the process's own memory before position 0. Same
tag, same argv, cache dropped before each row, only the switch differing: **229.33 and 228.44 with
the pass, 255.11 and 256.44 without** (-25.8 / -28.0 ms/step), identity PASS on all four rows.
