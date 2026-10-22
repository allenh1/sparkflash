# GPU device-chain roadmap (the performance path after the 2026-10-20 scope direction)

**Status: roadmap, hypothesis + gate per stage.** The user direction is *use the GPU as much as
possible*: the GPU device chain is now the performance path, and the CPU host twin is the
exactness oracle plus a stopgap. This page is derived from the independent device-chain audit;
it states which parts are **verified code facts** and which are **roofline estimates**, and it
corrects the audit's collective row.

## Measurement rules

- **Never mix device and host numbers in one table.** Each row names its path (device chain vs
  host twin), and a device number only replaces the CPU stopgap when a measured device run beats
  it and the run states that path's exactness contract.
- The current best measured decode is the **CPU stopgap: 767.7 ms/step = 1.3026 tokens/s**
  (host twin, sharding stage 2, bit-identical by `plan_hash`, generated tokens and 10/10
  `step_logits_sha256`; evidence path to be supplied by tp-dev). It is not a device number.
- A stage that has not run keeps its estimate label; no estimate is promoted to a measurement
  by repetition.

## Acceptance budget (derived from the target, not measured)

The batch-19 acceptance target (D48) is **vLLM parity 50-60 tokens/s with DSpark** on this
hardware, personal target **90 tokens/s single-stream**. The budget below is the derived
arithmetic for that target - it is **not a measurement**:

| quantity | budget | note |
| --- | --- | --- |
| verify step | **<=42 ms** | at ~3.8 accepted tokens per step |
| per accepted token | **11.1 ms** | 42 ms / 3.8, i.e. ~90 tokens/s |
| weight-traffic roofline | **13.9 ms/step** | at 273 GB/s/rank, the weights read once per step |
| RDMA collectives | **~1.7 ms/step** | the one-shot collective budget |
| draft | **~2.2 ms/step** | DSpark k=5 draft work inside the step |

These are **budgets derived from the target**, not measured values; stage rows below stay
placeholders until an owner reports a measured device run. Every stage is judged against the
**42 ms/verify-step budget**, not against 'faster than the CPU stopgap'.

## Verified code facts (the basis of the audit's target list)

These are code facts, not performance claims; they say where the device work is:

1. **The tp4 host path opens no driver and runs on the CPU.** `tp4_host` states and writes
   `device_execution: false` (`crates/sparkflash-cli/src/tp4_host.rs:24`, `:2747`, `:2796`).
2. **The b12x and MoE kernels accumulate along K in a per-block scalar loop**
   (`kernels/b12x_bf16a_mxfp8w_dense_sm121.cu:99`, `kernels/b12x_mxfp4_moe_sm121.cu:183`), and
   the MoE path's per-block scales are addressed by the `spark_b12x_row_scale_*` helpers
   (`kernels/b12x_mxfp4_moe_sm121.cu:55`).
3. **The router is a low-parallelism launch:** one thread block routes 8 tokens, so the grid is
   `ceil(tokens / 8)` blocks of the router's threads over 384 experts x 5120 values
   (`crates/sparkflash-kernels/src/router_launch.rs:7-8`, `:38`).
4. **`cert_lm_head` scans twice:** one pass screens blocks/entries, a second pass scans again
   (`kernels/cert_lm_head_sm121.cu:83-111`).
5. **The device chain stages host buffers per step:** `copy_htod` calls cover the chain's
   staging inputs every step (`crates/sparkflash-device/src/chain_device.rs:566-573`), which is
   what device residency (task-34) removes.

## Stages: hypothesis and gate

The millisecond figures below are the **audit's roofline estimates**, not measurements. Each
stage's gate is a measured device run with its exactness contract stated **and its verify-step
time and accepted tokens/s compared against the 42 ms/step budget above** - a stage that is
faster than the CPU stopgap but outside the budget has not met the target.

| stage | hypothesis | estimate | gate | status |
| --- | --- | --- | --- | --- |
| Device residency (task-34) | remove the per-token H2D staging/residency copies (fact 5) | **9,100 -> ~2,500 ms** | a device run with the staging copies gone, RSS/flat-memory evidence, exactness contract stated | estimate |
| Kernel occupancy rewrite (tasks 65-68) | the b12x/MoE along-K loops (fact 2) are schedule-bound; rewrite for occupancy | **-> ~295 ms** | a device run at the same shapes, per-kernel timing, exactness contract stated | estimate |
| Router + `cert_lm_head` | the one-block router (fact 3) and the double scan (fact 4) dominate the remaining small kernels | **-> ~24 ms** | a device run isolating those kernels, exactness contract stated | estimate |
| RDMA one-shot + CUDA graphs | replace the per-layer collective launches and TCP framing with one-shot RDMA and graph replay | **-> ~15.8 ms** | a device run naming the collective budget it beat (see the correction below) | estimate |
| DSpark k=5 | speculative decode with k=5 draft tokens lifts the decode rate | **target 60+ tokens/s** | a device run with draft acceptance recorded and the exactness contract stated | target |

## Collective budget correction (audit row)

RUN 4b's **92.2 ms `collective` bucket is rank-arrival skew** - the slowest rank's arrival at
the collective - **not TCP framing cost**. The measured wire cost is **8.6 ms/step = 40 layers
x 0.216 ms**, so the all-reduce budget today is **8.6 ms**, and RDMA one-shot must beat **8.6
ms, not 92 ms**. Any future table that quotes collective time states which of the two it
measured; the correction is also recorded in `docs/build/tp4-tokens-per-second.md` section 4.

## Resident-set geometry (arithmetic from checkpoint geometry, not measured on device)

The resident-set figure is **arithmetic from the checkpoint geometry, validated by a byte
identity** - it is not a device measurement. Routed experts are FP4 at **5.895 MB per matrix**
(2304 x 5120 / 2 bytes), so 384 experts x 40 layers x 3 matrices = **275.67 GiB**. The world=1
all-resident set is **285.504 GiB** over **95,819 language-model tensors** (the earlier
95,815 note was off by four; FP4 is stored as I8 with two values per byte), and the byte identity
**509,315,486,040 = 510,286,023,000 - 970,536,960** (vision/aligner) validates the
decomposition. At world=4 the resident set is **285.504 / 4 = 71.376 GiB/rank**, matching the
world=4 **72.31 GiB/rank** design in `docs/build/tp4-design.md`; the ~0.9 GiB difference is the
replicated classes counted once per rank. Finer split (world=1, GiB): experts-column 183.779 +
experts-row 91.890 = 275.67; attention 5.15; shared 1.42; embed+head 2.47; router 0.150; mhc
0.157; mtp-heads 0.197; engram-other 0.293; norms/special ~0.001. Engram tables are the four
disk tensors (188.8 GiB) and are not resident.

| category | GiB |
| --- | ---: |
| routed experts | 275.67 |
| attention | 5.15 |
| shared | 1.42 |
| embed+head | 2.47 |
| router | 0.15 |
| mhc + norms + mtp + engram-other | 0.65 |
| **total (world=1 all-resident)** | **285.504** |

**A single GB10 node cannot hold the all-resident set** (285.504 GiB world=1 against the node's
memory), so there is no single-node full-residency assumption anywhere in this plan: the
**four-rank arm is the consistent one**, and the 71.376 GiB/rank arithmetic agrees with the
72.31 GiB/rank world=4 design. The per-category table is arithmetic, not a measurement; the
device residency stage's gate still needs its own measured evidence. The single-node chain
stays **only as a zero-collective kernel bench** (with expert traffic labelled), not as an
all-resident measurement target.

## What this roadmap does not claim

- It does not claim any device throughput number: every ms figure above is an estimate or a
  target, and the current best measured decode is the CPU stopgap 1.3026 tokens/s.
- It does not claim the CPU host twin is replaced: the twin remains the exactness oracle; a
  device stage is only credible when its comparison against the twin is stated.
- It does not sum the stages into a 'total win': attribution is per change (residency,
  occupancy, router/head, collective, DSpark), and no combined number is claimed without the
  per-change measurements.
