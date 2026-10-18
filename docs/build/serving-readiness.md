# Serving readiness: the device path to a real token

Status: **docs-only readiness map. No code changed.** Verified against code main `8b3aef9`
("!81 batch 2: !78 device M1 + !80 router delta fixes", 2026-10-18) and the device windows of
2026-10-18 (including the b5 M2-session and b6 gate/up Lead windows); the branch is rebased onto
main `79f45b3` (!85 cleanup batch 1: CLI serve fixes plus D30's routed-expert sharding decision).
!85 changes no kernel, baseline or device-crate file; its D30 consequence (routed experts
82.2 GB/rank padded to 640) is reflected in section 4. Every number below is tagged **measured**
(a named device log printed it), **derived** (arithmetic shown here over figures that are measured or
pinned), or **unverified** (no evidence this
map can cite). A wrong number in this file is a blocker: correct or remove it rather than round it.

Scope: **one** question — can the device path run a real 40-layer forward pass and produce tokens
today, and if not, what exactly is missing? This is the question that decides whether the GB10
device path can replace vLLM as a serving backend. Serving in the wider sense (4-rank EP, batching,
HTTP wiring, throughput) is out of scope except where it changes the answer; section 4 covers the
memory consequence that makes the single-node chain look different from a served deployment.

Short answer: **no end-to-end device token exists.** The per-stage kernels are pinned, most are
individually measured on 2026-10-18, but nothing wires them into a chain: the chain crate is the M1
fake-session skeleton, the m8 harness that comes closest explicitly reports
"device token NOT PRODUCED", and Phase 2 exit criterion 1 remains **OPEN**
([device-forward-chain.md](device-forward-chain.md) section 4.3). Section 3 lists the remaining work
in dependency order.

Unit note: the measured device total is **124,610 MiB** `cuDeviceTotalMem_v2`
([device-path-design.md](device-path-design.md) section 9; [roadmap-to-serving.md](roadmap-to-serving.md)
section 7.5). The project's current arithmetic framing is **124,610 MiB = 130.66 GB = 121.7 GiB**
(exact byte count 130,663,055,360 B; 130.663 GB; 121.689 GiB) (derived). This file keeps that rounding
in prose and states the exact figures where the arithmetic needs them. The brief says "124.6 GiB":
that figure is the MiB total with a unit slip, and this file uses 124,610 MiB.

---

## 1. The device stages a decoded token needs

The per-layer order is the reference forward's own order
([device-forward-chain.md](device-forward-chain.md) section 2.2; engine code
`forward/mod.rs:373` run, `:546` block): Engram read/write at its layers, mHC entry (hc_pre),
attention projections, window-KV write/rotate, attention (CSA union), mHC post-mix (hc_post), MoE
(router, routed gate/up, routed down/combine, shared expert, epilogue), residual out; then once at
the tail: collapse, final RMS norm, certified LM head, greedy argmax. Every stage is exercised in
both prefill and decode; only the attention shape/KV extent changes (the CED schedule is
`kv/src/schedule.rs:141` `ced_schedule`, `CedSchedule:82`).

Table 1 is the whole map. "Device math" is what a device chain could call today; "status" is what
has actually been executed on a GB10; "gap" is what separates the stage from a wired chain.

| # | stage | device math (kernel) | status on main | gap to a wired chain |
| - | :--- | :--- | :--- | :--- |
| 1 | embedding gather | none | host seam only; device execution never attempted | kernel missing; seam `embedding_gather` |
| 2 | attention prologue: q/kv RMSNorm, RoPE | none | host seams only | fused norm+RoPE kernel missing; seams `attn_norm`, `attn_rope` |
| 3 | mHC entry (hc_pre) and post-mix (hc_post) | `MEGA_MHC` | **measured**: stage 1 4/4 tokens, stage 4 determinism, chained stages 7/9 | chain wiring only |
| 4 | attention projections (wq_a/wq_b/wkv/wo_b) and wo_a | `MXFP8_DENSE`, `MXFP8_WO_A` | **measured**: stage-8 probes bit-exact; chained stages 7/9 | chain wiring only |
| 5 | window-KV write and ring rotation | `copy_dtod` + ring layout helpers | **measured**: corrected m8 run (11,764 copies, 198,832,128 B, 0 mismatches) | chain wiring; device KV store is Phase 3 |
| 6 | attention: CSA union format 2 (38 layers), window format 1 (layers 0-1) | `FLASH_MLA_CSA2` | **measured**: task-14 format 2, m8 dispatch, stage-2 formats 0/1 | chain wiring; builder is on main |
| 7 | MoE router | `MOE_ROUTER` | **measured**: two m9 runs, 7/7 checks each | chain wiring only |
| 8 | routed expert gate/up (w1/w3) + SwiGLU + activation quantiser | **none on main; measured on the unmerged gate/up branch** | host seam on main; the branch's gate/up projection is Lead-window bit-exact (30,720/30,720 and 499,200/499,200), with no routing weight/SwiGLU/combine/shared expert; SwiGLU quantiser kernel probed, never trusted | **kernel missing on main**; seam `moe_gate_up` |
| 9 | routed down/combine | `MXFP4_MOE` | **measured**: slice 40,960/40,960 and 163,840/163,840 exact; stage 3 PASS; chained 7/9 | chain wiring only |
| 10 | shared expert | `MXFP8_DENSE` (gate/up, down) + host SwiGLU/quantiser | **measured as GEMM**: stage-8 probe bit-exact; stage-9 K13-K15 device GEMMs with host activation | chain wiring; fused kernel optional |
| 11 | FFN epilogue: routed-weighted sum, shared add | routed sum inside `MXFP4_MOE`; shared add host | **measured** only in host references / harness combine | no device epilogue kernel; see the seam-name gap in 2.11 |
| 12 | final RMS norm | none | host seam only | **kernel missing**; class (b) seam |
| 13 | certified LM-head screening | `CERT_LM_HEAD` | **measured once per rank** (stage 5): 129,280 entries exact, 0 certified | chain wiring; exactness in the chain; block reduction is host |
| 14 | greedy argmax over the reduced maxima | none | host seam only | **kernel missing**; class (b) seam |
| 15 | dispatch / launch scheduling | none (control) | design seam `dispatch` | Phase 3 graph runner |
| 16 | Engram read/write (layers 1, 14) | none (reference code) | used as reference code in the stage-9 chain | device head is Phase 3; tables are 202.758 GB |

Two rows are **token-path class (b)** seams: the chain design forbids host math for the final norm
and the greedy argmax if a *device chain* is claimed
([device-forward-chain.md](device-forward-chain.md) section 3.2). A first run may use them on the
host only if its report says "hybrid: token produced by host math over device logits"; such a run
cannot claim Phase 3 exit criteria, and per the design it cannot claim the Phase 2 token row as
device evidence either.

---

## 2. Stage detail: pins, harnesses, measurements, APIs, gaps

Kernel pins are in `crates/sparkflash-kernels/baseline/kernel_baseline.txt:11-18` (eight entries,
nvcc release V13.0.88, arch sm_121a). The six Milestone-4 kernels are the six-entry
`m4_abi::KERNELS` array (`crates/sparkflash-kernels/src/m4_abi.rs:867`; constants at `:854, :856,
:858, :860, :862, :864`), and `m4_abi` is frozen by D26 (`docs/DECISIONS.md`), so the router and
SwiGLU kernels are pinned outside that array (router ABI: `crates/sparkflash-kernels/src/router_abi.rs`;
SwiGLU: `kernels/swiglu_mxfp8_quant_sm121.cu:1-14`). The driver enum has seven variants
(`cuda_driver.rs:156-170`), one raw loader and launch path
(`Cuda::load_kernel` `:401`, `Cuda::launch` `:436`, `launch_repeated` `:481`); only two stages
have a **typed** launcher today (attention and router; see 2.6/2.7). Everything else is launched
through parameter-word arrays built by the m7 harness.

`mega_mhc_sm121` is the baseline manifest key; the cubin entry the driver and the harness use is
`mega_mhc_fused_bf16_sm121` (`cuda_driver.rs:177`). Both names denote the same 70,312 B cubin
(`kernel_baseline.txt:11`), and a window that stages cubins needs the second name too
([phase3-indexer.md](phase3-indexer.md) section 3, "Cubins to stage"). This is a naming trap, not a
second kernel.

### 2.1 embedding gather (row stage 1)

No kernel exists. The reference lookup is `forward/mod.rs:401-432`; the table is 661,913,600
elements (`linalg.rs:81-82`). Device execution has never been attempted; the first chain's seam
`embedding_gather` is host by design (a labelled, plan-accounted seam,
[device-forward-chain.md](device-forward-chain.md) section 3.1). **Gap:** kernel missing (Phase 3);
for the first hybrid chain the gap is only the seam + ledger wiring.

### 2.2 attention prologue: q/kv RMSNorm and RoPE (row stage 2)

No kernel exists; both are class (a) host seams (`attn_norm`, `attn_rope`), reference
`attention.rs:1758` (`attention_forward`) and `attention.rs:250/266` (rope tables). **Gap:** fused
norm+RoPE kernel missing (Phase 3); host seam otherwise.

### 2.3 mHC entry and post-mix (row stage 3)

- **Pin:** `mega_mhc_sm121`, 70,312 B, sha256
  `e21293e3f6288dbe413a2b998efce6c43c8a56dd993eca5cd1bfc6138cdca618`
  (`kernel_baseline.txt:11`; 4 PTX membar lines). Source header, release semantics and the retired
  pre-B0 fold: `kernels/mega_mhc_sm121.cu:40-69`.
- **Harness/API:** `examples/m7_layer0` stage 1 (`stage1.rs`), stage 4 (`stage4.rs:81`), and the
  chained stages 7/9 load it by name and launch through `Cuda::launch` with a param-word array
  (e.g. `stage1.rs:231`). The chain has **no typed mHC launcher**: M3 must add a pinned param
  builder or reuse the stage builders.
- **Measured (measured):** stage 1 passes 4 of 4 tokens at every boundary at layers 0, 1 and 4, and
  a later recon adds layers 14 and 39 (recon paragraph `phase1-boundary-table.md:571-578`). The
  full stage-1 row (`:627`) records two exceptions: layer-0 token 0 at 5,119/5,120 codes and
  layer-4 token 0 at 5,118/5,120; every other token is 5,120/5,120, and all scales are 160/160. The
  known fixed-order vs sequential `1/rms` deltas are inside the stage bar. The dedicated
  2026-10-18T17:48Z retry staged the renamed cubin and re-ran stage 1: **4 of 4 tokens at every
  boundary** (numbers in [phase3-indexer.md](phase3-indexer.md) section 5; its logs,
  `runs/d-stage1-layer0-retry.log` and `window-stage1-20261018T174723Z.log`, are Lead-window files
  on the node, not in this repository).
  Stage 4 launched the same inputs repeatedly with byte-identical output (mHC determinism).
  The B2 chained replay (stage 7) runs the sixteen-launch chain plus four forced-routing controls
  (20 launches; `/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log`,
  23,897 B, sha256 `c2de544c1789743e73ccc5d5277dc3165d98296d0f5cc867cc819d9160754ac9`,
  UTC 04:41:22Z-04:41:26Z), and stage 9 runs it inside 763 launches per 40-row chain run
  (`phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`, 572,420 B, sha256
  `bd0f908ef2e3519512a20bd53eb86303fcfad65550f7b145a6c81c7a2c973a65`, wall 298.6 s, two runs
  bit-identical).
- **Shorthand correction:** the brief's "measured stage 2/7/9" should be read as *measured in the
  stage-1 harness and executed inside the chained stages 7 and 9*. Stage 2 is the attention stage:
  it calls the mHC **reference** helper to construct its inputs (`stage2.rs:408`) and does not load
  or launch `MEGA_MHC` (verified by kernel-load grep in the stage-2 source). So stage 2 is not a
  device mHC execution.
- **Gap:** chain wiring only; the kernel and its pins are in place.

### 2.4 attention projections and output projection (row stage 4)

- **Pins:** `b12x_mxfp8_dense_sm121`, 11,824 B, sha256
  `03e4c422f1c55141a2abde6a53038e223d0a59ffe29d2ad913b979720f150a4a`
  (`kernel_baseline.txt:12`); `b12x_mxfp8_wo_a_sm121`, 9,056 B, sha256
  `84833781a77d094945d9f8e33168ff75cae4d9bf81b2a24b531b404984160eed`
  (`kernel_baseline.txt:15`).
- **Harness/API:** stage-8 probes compare dense and wo_a against the in-crate references on
  deterministic bytes; stages 7/9 launch them for the chain. API is the raw
  `Cuda::load_kernel`/`Cuda::launch` path with stage-built parameter words; there is no typed
  dense/wo_a launcher.
- **Measured (measured):** stage 8: dense 16,384/16,384 and wo_a 8,192/8,192 bit-exact, max ULP 0
  (`phase1-boundary-table.md:628`, stage-table row 8); stages 7/9 launch both kernels per layer.
  The 40-row stage-9 column is measured end-to-end as a Phase-1 health-check class; its aggregate
  Phase-2 exactness is **not** established (see 3.2).
- **Gap:** chain wiring only.

### 2.5 window-KV write and ring rotation (row stage 5)

- **Pins/machinery:** no model kernel; `cuda_harness::WindowRingLayout:283`,
  `ring_view_position:345`, `ring_view_segments:366`, `rotate_window_ring:413` (all on main) over
  device `copy_dtod`. Row geometry: 512 B E4M3 codes + 16 B scales = 528 B per slot
  ([phase2-sequence-harness.md](phase2-sequence-harness.md) section 7).
- **Measured (measured):** corrected m8 run, 2026-10-18T18:32:40Z on node0, binary sha256
  `8645b62104324d22fec1f64f222acc2052f745fa5b8e717d49c27be3e9b0f534`
  (built from merged main `1ceeb54`), cubin `5bfee4c7...`: **11,764 copy_dtod calls, 198,832,128 B
  moved (192,806,912 B codes + 6,025,216 B scales), 0.025624 s copy-only = 7,400.000 MiB/s**; loop
  wall 2.588988 s also covers view verification and attention dispatch. 360 oldest-first views
  (24,330,240 B) read back and compared byte-for-byte: **zero mismatches**. The plan bound is
  ≤21,928 copy calls. These numbers are in-repo in
  [phase2-sequence-harness.md](phase2-sequence-harness.md) section 7 and the D25 row of
  [../DECISIONS.md](../DECISIONS.md). The run artifacts themselves
  (`/home/user/phase2-evidence/window-2026-10-18/runs2/m8-corrected-device.log`,
  `runs2/m8-sequence-report.txt` 65,518 B, sha256
  `3fc6683a5d6c6a3bb1d25ad1eb03f77f6910b84de418231b38413eea4fb1aa6f`, window
  `window-b2-20261018T183240Z.log`) are Lead-window files on the node, not in this repository or its
  evidence tree: the tree here carries `runs/m8-sequence-report.txt` at 55,039 B, the superseded
  first report, and none of its numbers are used in this map.
- **Gap:** chain wiring (the plan and helpers exist); a device KV store (matrix S1) is Phase 3.

### 2.6 attention: CSA union (format 2) and window format (format 1) (row stage 6)

- **Pin:** `flash_mla_csa2_sm121`, 60,792 B, sha256
  `5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3`
  (`kernel_baseline.txt:14`). Format-2 semantics: `kernels/flash_mla_csa2_sm121.cu:36-41`; one
  f32 softmax over window ∪ selected compressed candidates with the sink in the denominator.
- **API a chain calls (exists, typed):** `sequence_chain::Format2AttentionCall:86` with
  `params():139` and null-indexer validation `:169-180`; `cuda_harness::FlashMlaParams:42`,
  `flash_mla_params:71`, `launch_flash_mla:178` (`launch_flash_mla_repeated:193`). This is the most
  chain-ready stage: its builder is on main; the router also has a typed launcher (2.7).
- **Measured (measured):**
  - task-14 isolated union check, node0, binary sha256
    `c0a9e55001acab4a84551cc9952682116bee0772a2ea55cfeb2e135dd36dfca8`, cubin `5bfee4c7...`:
    format 2 **pass=true** at layers 21/30/39, max abs 8.111e-3 / 6.279e-3 / 6.108e-3 against the
    2e-2 bar; format 1 (window only) misses the same reference by 2.188 / 1.972 / 1.816, the
    structural-exclusion demonstration. Evidence:
    `/home/user/phase2-evidence/task14/run-task14.log`,
    `task14/union-attention-report.json`.
  - corrected m8 dispatch (same corrected run as 2.5): **360 calls (342 format 2, 18 format 1)** over
    9 compared positions × 40 layers, grid (64,1,1) block (256,1,1); 0 non-finite outputs; 40/40
    device-vs-twin comparisons at position 129 within the 2e-2 bar (worst 9.718239e-4, layer 0 head 5
    channel 48); 32,766/32,768 exact bf16 on that column; 320/320 repeated-launch determinism
    identical; dispatch wall 0.175593 s over 360 first-launch records (+320 untimed repeats, 680
    launches issued in all; 487.758 µs per record). These numbers are in-repo in
    [phase2-sequence-harness.md](phase2-sequence-harness.md) section 7; the run logs (`runs2/`,
    `window-b2-...`) are node-only (2.5).
  - stage 2 (m7 harness, formats 0/1 before format 2 landed): PASS 3/3 positions at layer 0
    (32,768/32,768 exact at positions 0 and 1; position 3 32,765/32,768 = 99.9908% against the 99.5%
    bar, max ULP 1, max abs 1.19e-7 against the 1.6e-2 bar); PASS at layer 1 (2 positions, exact
    99.9878%, max abs 7.6293945e-6) and at layer 14 (2 positions; numbers in the recon log)
    (`phase1-boundary-table.md:629`; the layer-1/14 recon paragraph is `:571-578`; recon log
    `/home/user/phase1-evidence/layer-dispatch-recon-engram-last.log`).
- **Stale-header flag:** `kernels/flash_mla_csa2_sm121.cu:53-58` still says "Format 2 has NOT been
  executed on a device". The task-14 and m8 runs above supersede that sentence; the header should be
  corrected in the Phase-3 kernel work, and nothing in this map quotes it as current.
- **Gap:** chain wiring through the existing builder; the HSI pool / top-512 candidate merge and the
  compressed-cache round trip are host seams (2.6 is attention math only), and the chain's seam
  ledger must record them.

### 2.7 MoE router (row stage 7)

- **Pin:** `moe_router_sm121`, 37,384 B, sha256
  `5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b`
  (`kernel_baseline.txt:18`; eighth entry, added in the router MR). ABI pinned in
  `router_abi.rs`, not `m4_abi` (D26 ratchet) — [phase3-router.md](phase3-router.md) sections 1-3.
- **API a chain calls (exists, typed):** `router_launch::RouterLaunchParams:48`,
  `launch:103`, `launch_repeated:112`; `KernelId::MoeRouter` (`cuda_driver.rs:170`, symbol
  `:183`).
- **Measured (measured; in-repo record):** [phase3-router.md](phase3-router.md) section 5 records the
  2026-10-18T18:32Z Lead window on binary
  `8f32174e31da3561a68029c98f0861b194310c1cf2cd1fb06bcb89ee5e78e5cb` (built from `15b86d0`),
  cubin `5992cef9...`, PASS 7/7 in both runs:
  - `--tokens 8`: 48 of 48 ids exact against the engine host path and the CPU twin; host path
    weights bit-identical to the twin; device scores max abs 1.1920929e-7, max ULP 1, 2629/3072
    bit-identical; device weights max abs 5.9604645e-8, max ULP 2, 41/48 bit-identical; two device
    runs byte-identical (12,672 B compared); p50 6.8530 ms per launch (0.8566 ms/token); the harness
    reported 4,028,288 B of device buffers.
  - `--tokens 130`: 780 of 780 ids exact; scores max ULP 2, 43126/49920 bit-identical; weights max
    ULP 4, 707/780 bit-identical; determinism 205,920 B byte-identical; p50 6.8541 ms per launch
    (0.0527 ms/token), grid (17,1,1).
- **Measured (Lead-reported, not in this repository):** a later b3 window on the node (binary
  `a3ba20e4b3f9d0cc87267bfb90dbd204f9d18ab157a88114146afc228dd1cf85`, cubin `5992cef9...`)
  repeated the same ids and ULP counts with p50 6.8528 / 6.8534 ms and 5,470,816 B of device buffers
  at 130 tokens. Those logs (`runs3/`, `window-b3-...`) live only on the node, so this map does not
  treat the b3 micro-numbers as checked-in evidence; the b2 record above is what the repository holds.
- **What the measurement is not:** the activations are the released `embed.weight` rows, a stand-in
  for the post-attention hidden (declared in the harness), so this is router arithmetic over released
  gate bytes, not layer text. The seam is still in place; no host-time saving is claimed until the
  seam-removal MR measures both sides ([phase3-router.md](phase3-router.md) section 6).
- **Gap:** chain wiring (the launch API exists); router seam removal is a separate engine MR.

### 2.8 routed expert gate/up, SwiGLU and the activation quantiser (row stage 8)

**There is no device kernel for this stage on main, and the chain cannot call one.**
`kernels/b12x_mxfp4_moe_sm121.cu:33-36` says it explicitly: the file is the down/combine
projection, and "the up/gate projections that produce the activation are the same MXFP4 GEMM and are
not duplicated here". Stage 3 says the same in its own output
(`stage3.rs:251`): the gate/up projections, SwiGLU and the activation quantiser are host arithmetic
"because no device kernel in the six-cubin set reduces K=5120 with packed FP4 weights". The roadmap
names it as the Phase-3 gap ([roadmap-to-serving.md](roadmap-to-serving.md) section 4.3 seam table,
rows "gate/up (w1/w3) GEMM" and "SwiGLU and the activation quantiser").

Verification performed for this map: on main, `ls kernels/*.cu` shows nine files — the eight
baseline kernels plus `rocenante_oneshot_sm121.cu` (a RoCE probe, not a model kernel). None is a
K=5120 routed gate/up GEMM. The driver enum on main has no such variant: `KernelId` has exactly
seven variants (`cuda_driver.rs:156-170`), and a grep finds only reference helpers
(`gate_up_rows`, `mxfp4_ref`), not a cubin entry. [phase3-router.md](phase3-router.md) section 6
says "Iteration B (the K = 5120 routed-expert gate/up FP4 GEMM) is a separate MR, not this one".

**The draft exists, unmerged, on `origin/task/phase3-gate-up` (head `fc97f64`, rebased from the
measured `bb09bf3`; Rust/.cu byte-identical):** that branch adds
`kernels/routed_gate_up_sm121.cu`, the typed `routed_gate_up_launch.rs` (with
`routed_gate_up_abi.rs` and `routed_gate_up_ref.rs`), the harness `examples/m10_gate_up.rs`, the
doc `docs/build/phase3-gate-up.md`, and a baseline entry
`routed_gate_up_sm121 cubin_bytes=9752 cubin_sha256=115200d26c3404b7169c476e350f5575dadc3fec9e8b0ec97113cd297fa6c1af`.
Its doc says the kernel computes **one** projection (w1 or w3) of one routed-expert slice per
launch — 576 real columns per rank padded to 640, D30's layout — and ends at BF16 rows, leaving
SwiGLU and the quantiser in the host chain / `swiglu_mxfp8_quant_sm121`.

**Measured (Lead window, 2026-10-18T20:22:35Z; node-only logs, not in this repository):** binary
`m10_gate_up_bb12c547_aarch64`, sha256
`bb12c547aed5f6a1a17c753db60c4330b41da4190ea62564d8e0155f1b9755b8`, 4,557,008 B, cubin 9,752 B /
`115200d26c...af`; vLLM was stopped on all four nodes for the run and restarted afterwards.
`--tokens 8`: device gate **30,720/30,720** bf16 exact against the CPU twin and up
**30,720/30,720** exact; 0 non-zero padding columns; two device runs byte-identical; twin-vs-harness
0 mismatches of 27,648 real values per projection; gate p50 **1.2734 ms** (0.1592 ms/token), up p50
**1.2741 ms**. `--tokens 130`: gate **499,200/499,200** exact and up **499,200/499,200** exact; 0
non-zero padding; two runs byte-identical; 0 mismatches of 449,280 real values per projection; gate
p50 **30.1708 ms** (0.2321 ms/token), up p50 **30.2761 ms** (0.2329 ms/token). Control gap gate
3.977737e-2 / up 3.599775e-2 against a control max 2.490959e0, inside the declared 10% bar. A first
attempt (binary `dd9921a4` from `0ff78fe`) panicked in the harness before the device path at 8
tokens (`m10_gate_up.rs:221`: 30,720 vs 27,648), recorded in window b4; it produced no device
numbers. This is a gate/up arithmetic measurement only: the **routing weight, SwiGLU, combine,
shared expert, end-to-end MoE and any host-time saving are not measured or claimed**. Nothing of the
branch is merged into main, so "no device kernel on main" stands.

The seventh baseline kernel `swiglu_mxfp8_quant_sm121` (21,776 B, sha256
`4dbef63bb07808f92fb6e61bdcb9aec53cb6afd8f28d6744aea3185759c1862e`, `kernel_baseline.txt:17`) is
the SwiGLU + MX-quantise epilogue, and it is *probed* bit-exactly against `swiglu_ref` in stage 8
(4x640: 2,560/2,560 codes + 80/80 scales; 2x2304: 4,608/4,608 + 144/144,
`phase1-boundary-table.md:628`), but **the chain does not launch it**; it stays outside
`m4_abi::KERNELS` (D26) and is not a device stage of any measured run. On main the gate/up GEMM
that produces the activation still does not exist; the unmerged branch's measured kernel above is
that GEMM, so the remaining device gap is the surrounding MoE math, not the projection itself.

**Gap:** the kernel is missing on main; the unmerged branch now measures the gate/up GEMM
bit-exactly, but a device FFN claim still needs it merged plus the routing weight, SwiGLU/quantiser,
combine and shared expert on device, and the chain wired. Until then the first chain's FFN stage is
a labelled **hybrid** host seam `moe_gate_up`.

### 2.9 routed down/combine (row stage 9)

- **Pin:** `b12x_mxfp4_moe_sm121`, 10,048 B, sha256
  `36a15a4f3ae12b4e7d452c1201b267d3102afa43f168bd322d6930ff3f543057`
  (`kernel_baseline.txt:13`). Header arithmetic and the "no gate/up duplicate" statement:
  `kernels/b12x_mxfp4_moe_sm121.cu:25-40`; the routing weight is part of the A operand, not folded
  into the GEMM.
- **Harness/API:** no typed launcher; the slice harness `examples/moe_slice.rs` and stages
  3/7/9 build parameter words and launch with `Cuda::launch`. The vertical slice design and its
  measured run are [device-path-design.md](device-path-design.md) sections 6 and 11.
- **Measured (measured):** slice: `--experts 64 --tokens 8` 40,960/40,960 exact (max abs 0e0);
  `--experts 384 --tokens 32` 163,840/163,840 exact; device-buffer tallies reconciled to the byte.
  Stage 3: combine 20,480/20,480 bf16 exact, max ULP 0, at layer 0 and PASS at layers 1 and 14
  (`phase1-boundary-table.md:630`); the printed quantisation gap (quantised vs f32 activation) is
  100% of outputs by construction because the stage's device boundary is only the combine. The
  chained stages 7/9 execute K9-K12 per row plus four forced-routing control launches per row
  (763 launches per stage-9 chain run).
- **Gap:** chain wiring only; the kernel and the A-operand arithmetic are pinned and measured. The
  FFN stage as a whole is still hybrid until 2.8 lands.

### 2.10 shared expert (row stage 10)

- **Device math:** the shared expert's three dense projections use `MXFP8_DENSE` (same pin as 2.4);
  the SwiGLU and the activation quantiser around them are **host** arithmetic (class (a) seam
  `shared_expert`, removal: fused shared-expert kernel). The stage-9 chain runs exactly this split:
  "K13-K15 shared expert: device w1/w3/w2 dense kernels, host SwiGLU + quantiser"
  (`stage7.rs:1572-1574`; stage 9 reuses the B2 helpers).
- **Measured (measured):** the dense GEMM is covered both as arithmetic (stage-8 dense probe
  bit-exact) and in the chained 40-row column (stage 9's K13-K15 device launches, part of the 763
  launches/run and of the 1,520 printed row lines). Stage 3 does **not** device-compare the shared
  expert: it computes it through `mxfp8_ref::dense_gemm` on the host and says so
  (`stage3.rs:251`).
- **Gap:** chain wiring (host seams + device GEMMs are pinned); a fused shared-expert kernel is
  Phase 3, not a first-token blocker.

### 2.11 FFN epilogue: routed-weighted sum and shared add (row stage 11)

The routed-weighted sum is folded into the down/combine kernel's arithmetic: the kernel writes
`residual + Σ_slots (routing-weighted block sums)` (`kernels/b12x_mxfp4_moe_sm121.cu:25-33`), and
stage 3 compares that against `mxfp4_ref::moe_combine`. The **shared add** is separate host
arithmetic in both the reference engine (`forward/moe.rs:662`: the shared expert always runs, on the
same input, after the routed sum) and the B2/stage-9 harness
(`crates/sparkflash-kernels/tests/support/b2_harness.rs:687`,
`combine_bf16_routed_with_shared`: elementwise `bf16 routed + f32 shared`).

**Gap:** the chain design's seam table ([device-forward-chain.md](device-forward-chain.md) section
3.1) has no dedicated name for this add; `shared_expert` covers "SwiGLU + activation quantiser
around the dense GEMM", and capture/comparison of `ffn_out` happens at `forward/boundary.rs:136`
(`residual_out` is `:140`). Since
the design's rule is that *every* host computation is a named, planned seam, the M3 implementation
must either add an explicit epilogue seam (with a `phase3_followup` removal plan) or build a device
epilogue. This is a design-level gap to settle before M3, not a kernel gap: the arithmetic is simple,
but an unledgered host add on the FFN output is exactly the kind of silent seam the ledger exists to
prevent.

### 2.12 final RMS norm (row stage 12)

Class (b) seam `final_norm`: host `linalg::rms_norm` after the tail's collapse
(`forward/mod.rs:487-490`, collapse `mhc::hc_pre`). No kernel. The design forbids host execution for
a device-chain claim, so a token produced with a host final norm is a hybrid token by definition.
**Gap:** final-norm kernel missing (Phase 3); first hybrid run may use the seam, labelled.

### 2.13 certified LM head (row stage 13)

- **Pin:** `cert_lm_head_sm121`, 14,712 B, sha256
  `a833a336f0d9fe0a1dc5febffede52de099b545cad013bf4a3c542d6e167f7ec`
  (`kernel_baseline.txt:16`). The certificate is stated at
  `kernels/cert_lm_head_sm121.cu:9-24`: per-entry upper bound over 40 hidden blocks, strict-skip
  pruning, ties preserved; grid-wide finalisation is deliberately **not** in the kernel (`:27-30`),
  so the block reduction is part of the token path.
- **API a chain calls:** no typed launcher; stage 5 loads `KernelId::CertLmHead` and launches per
  rank. The host contract is `lm_head_ref.rs`: `Screening:45`, `exact_logit:99`,
  `screen_logits:115`, `best_entry:161` (lowest-index tie-break). The kernel writes exact values
  only for uncertified entries and an upper bound for certified ones, so the contract is
  **argmax + exact-entry logits**, not "all logits comparable"
  ([device-forward-chain.md](device-forward-chain.md) section 4.2).
- **Measured (measured):** stage 5 on the released token (2026-10-17): `cert_lm_head_sm121` launched
  **once per rank** over the four 32,320-row shards of `head.weight`; device argmax **5**, value
  **13.237006187**, equal to the bf16-matched host reference; **all 129,280 entries exact, 0
  certified**; max |device − bf16 host| = **0**; certificate check not exercised (no certified
  entries); run 1.56 s including the 1.32 GB upload; a second run printed identical lines
  (`phase1-boundary-table.md:546-563`; logs `stage5-device-run-timed.log`,
  `stage5-device-final.log`). This is one hidden state from the layer-0 harness, not a chain run.
- **Stale-header flag:** `kernels/cert_lm_head_sm121.cu:31-38` still records "Verification status:
  never executed" and cites nvcc 12.9.1. That sentence is stale: four launches happened
  (one per rank); the certificate/pruning path is still unexercised, which is the part the header is
  really about.
- **Gap:** chain wiring; the finalisation (block reduction over per-block maxima) is outside the
  kernel and the design lists the whole `lm_head_argmax` as class (b) because it includes the greedy
  selection; the chain also owes the head an exactness comparison. Pruning-path validation is
  unstarted.

### 2.14 greedy argmax (row stage 14)

Class (b) seam `lm_head_argmax`: the block reduction over the four ranks' 32 per-block maxima plus
the lowest-index tie-break, `lm_head_ref::best_entry:161`, `cert_lm_head_sm121.cu:27-30`. No kernel;
head reduction/sampling kernel is Phase 3. **Gap:** kernel missing (Phase 3); a first hybrid run may
use the seam, labelled "hybrid: token produced by host math over device logits".

### 2.15 dispatch and boundary capture (row stage 15)

Dispatch is a no-arithmetic seam (S2 control; the Phase-3 CUDA-graph runner is the removal plan).
Boundary capture/comparison machinery **does exist on main**: `BoundaryStage` (`sequence_chain.rs:517`),
`reference_tensor:565`, `compare_reference_f32/bytes:582/:605`, `compare_boundary:370`, with the
D7 rule (logits only at position 129; decode tokens 130..=137 compared exactly). Stage 7/9 already
run it. **Gap:** only the chain that would call it.

The design's seam set, mapped to this map's stages (class (a) = host allowed for the first labelled
chain; class (b) = token path, host only for a declared hybrid run):

| seam | stage here | class | removal plan |
| :--- | :--- | :--- | :--- |
| `embedding_gather` | 2.1 | (a) | embedding kernel |
| `engram_rw` | 2.16 | (a) | device Engram head |
| `attn_norm`, `attn_rope` | 2.2 | (a) | fused norm+RoPE kernel |
| `window_kv_roundtrip` | 2.5 | (a) | device KV store S1 |
| `compressor_cache`, `index_key_path`, `indexer_weights` | 2.6 prologue | (a) | compressor/indexer kernels |
| `hsi_scoring`, `top512_merge`, `expanded_view` | 2.6 prologue | (a) | HSI/indexer/top-k kernels |
| `moe_router` | 2.7 | (a) | router seam-removal MR |
| `moe_gate_up` | 2.8 | (a) | K=5120 FP4 gate/up kernel |
| `shared_expert` | 2.10 | (a) | fused shared-expert kernel |
| `dispatch` (S2) | 2.15 | (a) | CUDA-graph runner |
| `final_norm` | 2.12 | **(b)** | final-norm kernel |
| `lm_head_argmax` | 2.14 | **(b)** | head reduction + sampling kernel |

The **shared add** (2.11) is not in this table because the design has no name for it; that is the gap
recorded in 2.11. `moe_router` is class (a) in design 3.2 even though the device kernel exists, because
removing it is a separate engine MR; a chain may use either the host seam or the typed device launch,
but the ledger and the report must say which one ran.

### 2.16 Engram read/write (row stage 16)

Seam `engram_rw`: reference code `forward/engram.rs:602/:797`; the stage-9 chain uses the engine's
released `engram_forward` with `ngram_hashes` over a paged `SafeTensorsSource` for layers 1 and 14
(`phase1-boundary-table.md` section 13). No device head; the store/packer exists but has never been
run against the checkpoint ([roadmap-to-serving.md](roadmap-to-serving.md) section 7.5, item 4).
**Gap:** Phase-3 device head; store residency is a serving-plan prerequisite.

---

## 3. Remaining work to the first end-to-end device token, in dependency order

### 3.1 What the chain state is today

- **M1 is merged** (`!78`, in the `!81` batch on main `8b3aef9`): `sparkflash-device` exposes
  `DeviceEngine<Session: DeviceSession>` behind `--engine reference|device`, with a
  `FakeSession` that records calls and produces no device result; it opens no driver, loads no
  cubin, allocates no device buffer, reads no checkpoint or cluster, and delegates `generate` to the
  reference engine (`crates/sparkflash-device/src/lib.rs:1-24`, `engine.rs:1-17`, `session.rs:1-9`;
  cli flag `crates/sparkflash-cli/src/serve.rs:157/:345`). Its four offline tests run without CUDA
  (`crates/sparkflash-device/tests/engine_offline.rs`), and its only dependencies are
  `sparkflash-api` and `futures` — no kernel or driver link. The capability report says
  `vision: false, fake_session: true`, and the cli default is `reference`.
- **M2 exists only on an unmerged branch, and the Lead has now executed it once.**
  `origin/task/device-engine-m2` (head `fab693c`, rebased from the measured `11ad400`) re-scopes
  the milestones and implements a **real session**: it opens the CUDA driver, loads the pinned
  `moe_router_sm121` cubin (37,384 B, `5992cef9...`, checked before the driver), runs **exactly
  one** router launch through the same `DeviceSession` seam on a worker thread, reads
  ids/weights/scores back, and derives `device_execution: true` while `chain_execution` stays
  `false`. That branch's design doc renames the old M2 chain skeleton to **M3** and the old M3 full
  run to **M4**. Lead window b5 (2026-10-18T20:09:55Z; node-only logs) executed it: binary
  `sparkflash_m2_11ad400_aarch64` sha256
  `bb06520a8e414e98314c570d48e2da07f75bfa571b2644f4746a297093d8dbb5`, 41,264,704 B; device 0
  NVIDIA GB10 (total 124,610 MiB); report line `fake_session=false, device_execution=true,
  chain=false, kernel=moe_router_sm121, cubin_sha256=5992cef9..., cubin_bytes=37384, launches=1,
  tokens=8, outputs_sha256=889671ba4b1fc5ab92fb8a1e87a0a43df40dd47e22f036914eccc93ee42191d6`,
  then listening; the M1 fake-session negative control in the same window printed
  `fake_session=true, device_execution=false, chain=false, vision=false` and launched nothing. This
  is an execution proof only: no token, no logits, no chain. None of it is merged: on main
  `chain.rs` and `window.rs` do not exist (the crate contains only `lib.rs`, `engine.rs`,
  `session.rs`), it does not depend on `sparkflash-kernels`, and there is no weight staging, window
  ring, boundary capture, seam ledger instance or token path. This map uses the branch's names for
  the remaining work (M3 chain skeleton, M4 full run) and says where main's doc still uses the older
  M2/M3 labels.

### 3.2 The honest status of the "40-layer forward pass producing tokens"

No such run exists. The closest measured artifacts are:

1. **The stage-9 40-row chained column** (2026-10-18T05:04Z): one released-checkpoint first step
   (token id 0, position 0, backbone only) as a single chained device forward of 40 rows, two runs
   bit-identical, 763 launches per run, 298.6 s. It produces **boundary tensors**, not tokens, and
   its comparison is a Phase-1 health-check class: over 40 × 18 = 720 tensor-row comparisons,
   **306 at or under the Phase-1 bar and 414 over it**
   (`phase1-boundary-table.md:817`). That is not the Phase-2 exactness run and not a token.
2. **The m8 plumbing run** (2026-10-18T18:32Z): rotation + attention dispatch over the full
   130-prefill + 8-decode schedule, 360 dispatches, all measured numbers in 2.5/2.6 — and its own
   report states, for every position 130..=137, "device token NOT PRODUCED" and "no device output:
   forward chain not wired in this iteration", with the final status "the model forward is NOT
   wired, so NO token and NO logits delta was produced" and "Phase 2 exit criterion 1 remains OPEN"
   (`runs2/m8-sequence-report.txt`, quoted verbatim).
3. **The M2 real-session execution proof** (2026-10-18T20:09:55Z, window b5, Lead-window): one
   pinned `moe_router_sm121` launch through the device session (`launches=1`, `tokens=8`,
   `device_execution=true`, `chain=false`), with the M1 fake-session negative control in the same
   window registering no launch. It is an execution proof, not a forward chain and not a token.
4. **Per-stage measurements** (router, gate/up on the unmerged branch, head, attention, down/combine,
   mHC) — each real, none of them a chain.

So the answer to the one question is: **the pieces are measured; the forward pass is not wired; no
token has been produced by the device path; criterion 1 is OPEN.**

Criterion 1's token row is exact: prefill 130 positions (>128) with next token **491**, then at least
8 decode steps producing **367, 201, 80, 3302, 201, 54, 201, 4341** exactly, compared against the
fixture that is the only truth (D25); logits are compared at position 129 only and decode has no
boundary capture (D7) ([device-forward-chain.md](device-forward-chain.md) sections 4.1 and 4.3). None
of those nine tokens has a device-produced counterpart yet.

### 3.3 Ordered work, M1/M2 and the chain (M3/M4 in the M2 branch's re-scope)

The design's milestone split ([device-forward-chain.md](device-forward-chain.md) section 7) is the
right frame. Main's doc still labels the chain skeleton M2 and the full run M3; the unmerged
`origin/task/device-engine-m2` branch re-scopes M2 to the session/launch proof just described and
renames the chain work **M3** and the full run **M4**. The ordering below uses the branch's names,
respects data dependencies, and separates what is already on main.

**M3-a — chain skeleton plumbing (no token yet).**
1. Add the kernel/fixture dependencies to `sparkflash-device` and the chain files `chain.rs`
   (per-layer graph and seam ledger) and `window.rs` (ring layout/rotation) per design 7 (main's doc
   calls them the M2 files; in the branch's re-scope this is M3).
2. Load the driver and the six chain cubins (plus the router cubin if the device router is used),
   verifying cubin sha256 and size against
   `kernel_baseline.txt` before any launch (the m7 harness already does this; the device crate
   currently does none of it).
3. Add typed param builders for the five stages that lack them (mHC, dense, wo_a, MXFP4 down/combine,
   head); only attention (`FlashMlaParams`/`launch_flash_mla`) and router
   (`RouterLaunchParams`/launch) are typed today. Each builder must be ABI-pinned like the router's.

**M3-b — weight staging and budget admission.**
4. Stage the dense, mHC, shared-expert, head and embedding weights through `WeightSource`
   (`embed.weight` is 1.324 GB bf16, section 4); stream routed experts per layer (no data path
   exists; section 4 shows why all-resident is impossible).
5. Admit the device budget before allocating (the m8 plumbing run's 9,926,144 B across 176 buffers
   is a lower bound with no weights and no head; design 5: `device-forward-chain.md:369-370`;
   the slice's per-run tallies are in `device-path-design.md` section 11).
6. Handle the paged Engram source for layers 1 and 14 (the engine's own `engram_forward` is the
   measured reference, but the packer has never run against the checkpoint).

**M3-c — first hybrid token (the first real end-to-end run).**
7. Implement the per-layer order of design 2.2 with the seams of design 3.1, using the existing
   builders (`Format2AttentionCall`, `WindowRingLayout`/`rotate_window_ring`) and the new typed
   builders. The FFN stage is **hybrid** (host gate/up + host SwiGLU/quantiser over device
   down/combine) unless the unmerged gate/up branch merges and is wired first; in either case the
   routing weight, SwiGLU/quantiser and shared expert stay host for the first token. The seam ledger
   records every seam, prints before comparisons and fails its self-check on a missing/unplanned
   seam.
8. Run the token path as the design allows for a hybrid first run: host `final_norm`, head screening,
   host block reduction + `best_entry` argmax. The report must say "hybrid: token produced by host
   math over device logits"; per design 3.2 that run cannot claim Phase 3 exit criteria and cannot
   claim the Phase 2 token row as device evidence.
9. Settle the epilogue seam (2.11) before this run: either a named host seam with a removal plan or a
   device epilogue. Do not let the shared add be unledgered.

**M3-d — boundary capture and the Phase-2 comparison harness.**
10. Capture the 18 boundary tensors per row at position 129 only, and compare tokens 130..=137 exactly
    against the fixture (D7; the fixture is the only truth, D25). Reuse `BoundaryStage` and the
    pre-registered bars; the m7 stage 7/9 harness already exercises the comparison path.

**M4 — the full 130 + 8 run and criterion 1.**
11. Run the full prefill+decode schedule and write the report artifacts; criterion 1's PASS definition
    is design 4.3. The run may only be called passing as the hybrid claim described there (host FFN
    seams labelled, token path labelled hybrid, head labelled NOT YET DEVICE-EXECUTED AS A PRUNING
    PATH), and Phase 3 claims require removing the class-(b) host seams (final norm, argmax) — which
    need kernels (2.12/2.14) — plus the gate/up kernel for a device FFN claim (2.8; the draft is
    unmerged on `origin/task/phase3-gate-up`, now with Lead-window gate/up numbers, but the routing
    weight, SwiGLU, combine and shared expert are still unmeasured on device and unwired).

**Beyond criterion 1 (serving replacement of vLLM).** Even M4 (the full 130 + 8 run) runs one sequence
on one node. Serving
needs the 4-rank EP/TP plan and memory plan (section 4), the engine-format KV store (S1), batching,
the HTTP engine seam wired to a real session (M1 only exposes the flag), and the Phase-3 performance
exit criteria. Nothing in this map claims any of that; it is the dependency chain after the first
token, not part of it.

### 3.4 Missing kernels, in dependency order for a *device* (non-hybrid) claim

1. Routed gate/up K = 5120 FP4 GEMM (blocks the device FFN stage; the largest gap, 2.8).
2. Final-norm kernel (blocks a device token path, class (b)).
3. Head block-reduction / greedy-argmax kernel (blocks a device token path, class (b)).
4. Embedding gather kernel (input side; host seam is allowed for a hybrid run).
5. Fused norm+RoPE kernel (attention prologue; host seam allowed).
6. Optional later: fused shared-expert kernel, HSI/indexer kernels, device KV store (S1), compressor
   kernel, CUDA-graph runner.

For the first hybrid token the missing kernels above can remain host, each as a labelled seam (the
FFN stage stays labelled hybrid); the class-(b) seams can stay host only in a labelled hybrid run.

---

## 4. Memory budget for the 40-layer chain

The brief asks for the arithmetic against "124.6 GiB"; the measured device total is 124,610 MiB
(section 0 note), and this section shows the numbers rather than restating a label.

### 4.1 Measured and derived components

- Device total: **124,610 MiB measured** = 130,663,055,360 B = 130.66 GB = 121.7 GiB (exact
  130.663 GB / 121.689 GiB) (derived; `cuDeviceTotalMem_v2`,
  [device-path-design.md](device-path-design.md) section 9,
  [roadmap-to-serving.md](roadmap-to-serving.md) section 7.5).
- BF16 LM head: 129,280 vocabulary × 5,120 hidden × 2 B = **1,323,827,200 B = 1.324 GB = 1.233 GiB**
  (derived). Per rank of 4 (32,320 rows): 330,956,800 B = 0.331 GB. This is the design's "~1.3 GB
  bf16 head" (`device-forward-chain.md:369`); the arithmetic matches it.
- BF16 embedding table: 661,913,600 elements × 2 B = **1,324 MB** (derived; `linalg.rs:81-82`).
- Dense attention weights: "about 127 MB per layer, 5.07 GB for forty layers"
  ([roadmap-to-serving.md](roadmap-to-serving.md) section 4.6, derived from the byte table).
- Replicated group (shared experts, gates, norms, mHC, Engram projections, compressors, MTP extras,
  image): **3.43 GB per rank**, design arithmetic in the roadmap's table
  ([roadmap-to-serving.md](roadmap-to-serving.md) section 7.5).
- Routed experts: "18.8 MB each" of released bytes (roadmap section 4.6; derived from the released
  byte table). The only device-verified layout shards each expert's 2,304-value intermediate width
  across four ranks as 576 real columns padded to 640, +11.1% (D30), so a stored expert costs
  18.8 × 640/576 = **20.9 MB**; 74.0 GB/rank unpadded becomes 82.2 GB/rank padded.
- Whole routed-expert set, forty layers, all 384 experts resident: 384 × 18.8 MB × 40 =
  **288,768 MB = 288.8 GB** of released bytes (derived), or **288.8 × 640/576 = 320.9 GB** in the
  padded D30 layout — more than twice the device either way.

### 4.2 Why the whole model cannot be resident (and what the reference plan assumes)

The plan's sharding (world 4, with D30's TP=4 routed-expert slice padded to 640) gives: routed
experts 82.2 GB/rank (74.0 unpadded + 8.22 padding), Engram row shard 50.7 GB/rank,
dense+embed/head/sink/weights_proj about 2.0 GB/rank, replicated 3.43 GB/rank — **about 138.3 GB
resident per rank** against a 130.66 GB device, i.e. **about 7.7 GB over** (derived:
130.66 − 138.33 = −7.67). The weights do not fit even before KV and activations, which is why the
plan moves the Engram tables to local NVMe (50.7 GB/rank on disk, 1 GiB cache): resident drops to
**about 87.6 GB** (82.2 + 2.0 + 3.43 = 87.63), leaving **about 43 GB** for KV, activations and the
Engram cache (derived: 130.66 − 87.63 = 43.03; roadmap section 7.5, D30). Caveats from the roadmap
stand: usable device memory is below `cuDeviceTotalMem_v2` and the driver/context overhead is
unmeasured; the Engram packer has never been run against the checkpoint.

For **one-node chain work** (M3/M4 in the M2 branch's re-scope) the reference 4-rank sharding is not
available (the chain is single-node and does not shard experts), so the working set has to be built
from whole-model components: dense
attention 5.07 GB + replicated 3.43 GB + head 1.324 GB + embedding 1.324 GB = **≈11.15 GB** (derived,
using the roadmap figures; shared experts/gates/norms are inside the 3.43 GB replicated group, so
they are not counted twice), plus routed experts actually needed.

### 4.3 First-chain expert working set (the part that decides feasibility)

- All 384 experts of one layer: 384 × 18.8 MB = **7.22 GB** of released bytes, or 384 × 20.9 MB =
  **8.02 GB** in the padded D30 layout (the layout the chain must use) (derived). A layer-at-a-time
  staging therefore has a worst case of about 8.0 GB of experts.
- 64 experts per layer (the roadmap's example): 64 × 20.9 MB = **1.34 GB per layer** (1.20 GB
  unpadded), and 64 × 20.9 MB × 40 = 53.48 GB for forty layers if every layer's 64 were resident at
  once — the roadmap's "about 53 GB for forty layers" is the unpadded 48.13 + 5.07 dense = 53.2 GB;
  in the padded layout it is 53.48 + 5.07 = **58.55 GB** (derived; excludes head, embedding and
  replicated, so do not quote it as a full working set).
- 8 experts per layer: 8 × 20.9 MB × 40 = **6.68 GB** for forty layers (6.02 GB unpadded); with dense
  attention that is 11.75 GB padded (11.09 unpadded), so the roadmap's "about 12 GB" matches the
  padded layout and reads high against released bytes (derived).

Three honest working-set scenarios for one node (derived; excludes KV, activations, CUDA context and
the Engram tables, which the plan puts on NVMe: on one node the whole 202.758 GB table set goes to
local NVMe — the 50.7 GB figure in the world-4 plan is a per-rank row shard, not a one-node
requirement):

| scenario | dense + replicated + head + embed | experts | total | free of 130.66 GB |
| :--- | ---: | ---: | ---: | ---: |
| all 384 experts of one layer staged, padded D30 layout | 11.15 GB | 8.02 GB | 19.17 GB | 111.49 GB |
| 64 experts/layer cache, padded D30 layout | 11.15 GB | 1.34 GB | 12.49 GB | 118.17 GB |
| 8 experts/layer cache, padded D30 layout | 11.15 GB | 0.17 GB | 11.32 GB | 119.34 GB |

The first row is the correct conservative bound for a chain that does not know the routing in
advance; the other two require a streaming/caching expert path that **does not exist yet**. With
unpadded release bytes the rows are 18.37 / 12.35 / 11.30 GB, but the chain has to store D30's padded
slices, so the padded rows are the ones the budget must admit. What the
M3 implementation must guarantee is that the expert set for a layer is staged before its MoE block
runs and evicted after, and that the budget admission accounts for it. No measured peak exists for a
wired chain: the only measured device allocation by the device path is the m8 plumbing run's
9,926,144 B across 176 buffers, with no weights and no head (design's lower bound,
`device-forward-chain.md:369-370`).

**What is unverified here:** the exact per-layer dense weight splitting (127 MB/layer is an average
over forty layers; the released per-layer tensor sizes differ), the real routing distribution for the
130-token prompt (so the actual number of distinct experts per layer is unknown), and the device's
usable memory below `cuDeviceTotalMem_v2`. The first two are measurable with the host reference or a
routed-expert read audit once M3 stages weights; the third needs a node census. Until then 19.17 GB
(111.49 GB free) is the conservative M3 plan and the headroom claim, not a measured figure; the same
rows with unpadded release bytes are 18.37 / 12.35 / 11.30 GB.

---

## 5. Evidence index and corrections

### 5.1 Commits and docs of record

- main `8b3aef9` = "!81 batch 2: !78 device M1 + !80 router delta fixes" (the code tree this map
  was verified against). Main has since advanced through docs-only facts (`e19c933`, !79: D29, D25's
  18:32Z numbers) and `79f45b3` (!85 cleanup batch 1: CLI `serve` checkpoint fixes plus D30's
  routed-expert sharding); neither changes a kernel, baseline or device-crate file. Chain APIs on
  main from `1ceeb54` (!63): `Format2AttentionCall`,
  `SeamKind/Record/Ledger`, `BoundaryStage`.
- Design and contracts: [device-forward-chain.md](device-forward-chain.md) (M1/M2/M3 at section 7 on
  main; the unmerged M2 branch re-scopes to M1/M2/M3/M4; head contract 4.2; criterion-1 definition
  4.3; seam set 3.1), [device-path-design.md](device-path-design.md),
  [roadmap-to-serving.md](roadmap-to-serving.md) (sections 4.3, 4.6, 7.5), [phase1-boundary-table.md](phase1-boundary-table.md),
  [phase3-router.md](phase3-router.md), [phase3-indexer.md](phase3-indexer.md),
  [task14-union-attention-case.md](task14-union-attention-case.md), [../DECISIONS.md](../DECISIONS.md)
  (D7, D24, D25, D26, D30).
- The corrected m8 record is on main in [phase2-sequence-harness.md](phase2-sequence-harness.md)
  section 7 (landed with !76 via the !77 batch), and the D25 row of [../DECISIONS.md](../DECISIONS.md)
  now records the 2026-10-18T18:32Z numbers. This map cites that in-repo record; the node-only run
  artifacts are named in 5.2 where they are the only source.

### 5.2 Windows and logs

Node-only paths in this section are relative to
`/home/user/phase2-evidence/window-2026-10-18/` on the node; they are not present in this
repository or the evidence tree on this host.

| evidence | where it can be read | what it establishes |
| :--- | :--- | :--- |
| corrected m8 plumbing window | **in-repo:** [phase2-sequence-harness.md](phase2-sequence-harness.md) section 7 and the D25 row; **node-only:** `window-b2-20261018T183240Z.log`; `runs2/m8-corrected-device.log`; `runs2/m8-sequence-report.txt` (65,518 B, sha256 `3fc6683a5d6c6a3bb1d25ad1eb03f77f6910b84de418231b38413eea4fb1aa6f`) | 11,764 copies / 198,832,128 B / 0.025624 s / 7,400.000 MiB/s; 360 views zero mismatches; 360 dispatches; no token produced |
| router b2 window | **in-repo:** [phase3-router.md](phase3-router.md) section 5; **node-only:** `window-b2-20261018T183240Z.log`; `runs2/m9-router-tokens8.log`, `runs2/m9-router-tokens130.log` | binary `8f32174e...`, cubin `5992cef9...`; 48/48 and 780/780 ids exact; ULP/bit counts; p50 6.8530/6.8541 ms; 4,028,288 B buffers |
| router b3 window | **Lead-reported only** (node: `window-b3-...`, `runs3/m9-d14306c-tokens{8,130}.log`; not in this repo or evidence tree) | same ids/ULP counts; p50 6.8528/6.8534 ms; 5,470,816 B buffers at 130 tokens |
| stage-1 retry | **in-repo:** [phase3-indexer.md](phase3-indexer.md) section 5; **node-only:** `window-stage1-20261018T174723Z.log`; `runs/d-stage1-layer0-retry.log` | stage 1 PASS 4/4 tokens every boundary with the renamed pinned cubin `e21293e3...` |
| gate/up b6 window | **Lead-reported only** (node: `window-b6-20261018T202235Z.log`; `runs6/m10-tokens8.log`, `runs6/m10-tokens130.log`; not in this repo or evidence tree) | binary `bb12c547...`, 4,557,008 B; gate/up 30,720/30,720 and 499,200/499,200 exact; padding 0; determinism byte-identical; p50 1.2734/1.2741 ms (tokens 8) and 30.1708/30.2761 ms (tokens 130); control gap 3.977737e-2/3.599775e-2 |
| M2 session b5 window | **Lead-reported only** (node: `window-b5-20261018T200757Z.log`; `runs5/m2-real.log`, `runs5/m2-fake.log`; not in this repo or evidence tree) | binary `bb06520a...`, 41,264,704 B; one router launch, `fake_session=false`, `device_execution=true`, `chain=false`, `launches=1`, `tokens=8`, outputs sha256 `889671ba...`; M1 fake-session negative control in the same window |
| task-14 union check | this evidence tree: `/home/user/phase2-evidence/task14/run-task14.log`; `union-attention-report.json` | format 2 pass=true at layers 21/30/39; format 1 misses by ~1.8-2.2 |
| chained 40-row column | this host: `/home/user/phase1-evidence/phase1-c-stage9-40rows-harness5b-20261018T050416Z.log` (572,420 B, sha256 `bd0f908e...`); device pair `stage9-out-r5b/device.{manifest,blob}` (shas `6093c74b...`/`fecd813b...`) | 40 rows × 2 runs, 763 launches/run, 298.6 s, bit-identical; Phase-1 aggregate 306 under / 414 over bar |
| B2 chained replay | this host: `/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log` (23,897 B, sha256 `c2de544c...`) | 20 launches (16 chain + 4 forced-routing control), UTC 04:41:22Z |
| head | this host: `/home/user/phase1-evidence/stage5-device-run-timed.log`, `stage5-device-final.log` | 4 launches (one per rank), 129,280 exact, 0 certified, argmax 5, 1.56 s incl. 1.32 GB upload |
| window-cubin set | **node-only:** nine files = eight baseline cubins + the renamed mHC copy, all matching `kernel_baseline.txt:11-18` when checked; **this evidence tree has seven** (`cubins/`: no `moe_router`, no renamed `mega_mhc_fused` copy) | pin check |

The b4 first gate/up attempt (`dd9921a4` from `0ff78fe`) panicked at `m10_gate_up.rs:221`
(30,720 vs 27,648) and produced no numbers; b6 is the completed gate/up window with both shapes.
The corrected-run report's sha256 is recorded in [phase2-sequence-harness.md](phase2-sequence-harness.md)
section 7; the copy in this evidence tree (`runs/m8-sequence-report.txt`, 55,039 B, sha256
`bf230905b8078d3add3baf278420c952a38a5ef52f07e3efcb23647c6f74b92b`) is the superseded first report
and is not cited.

### 5.3 Corrections to earlier text (read this before citing the older docs)

- `device-forward-chain.md` section 2.1 says "the corrected device path has not run". That is **still
  true for the 40-layer chain**; only its parenthetical claim that the corrected m8 binary itself has
  not run is stale — the m8 plumbing binary ran at 2026-10-18T18:32Z (2.5/2.6), while the chain is
  still not wired. Section 2.2 says the baseline has seven entries and the chain launches six
  kernels: the six-kernel statement is **still true** (the chain launches the same six; the router is
  a seam, not one of them), and only "seven entries" is stale — the baseline has **eight** since the
  router was added. Fix the parenthetical and the entry count, not the two true statements.
- `kernels/flash_mla_csa2_sm121.cu:53-58` says format 2 was never executed on a device; task-14 and
  the m8 dispatch supersede it (2.6).
- `kernels/cert_lm_head_sm121.cu:31-38` says "never executed"; four launches are recorded (2.13).
  The pruning path is still unexercised; that part of the header remains true and is not claimed.
- [phase3-router.md](phase3-router.md) section 5 is the repository's router record (binary
  `8f32174e` from `15b86d0`, p50 6.8530/6.8541 ms). A later b3 window (binary `a3ba20e4`) repeated
  the ids and ULP counts with p50 6.8528/6.8534 ms, but its logs are Lead-reported node files, not
  checked-in evidence, so no b3 number is treated as in-repo here.
- The `origin/task/phase3-gate-up` doc on its head `fc97f64` still says the harness has "no device
  numbers yet"; the 2026-10-18T20:22:35Z Lead window (5.2) supersedes that for the gate/up projection
  only (and the branch's device run has now been executed).
- The brief's "measured stage 2/7/9" for mHC is corrected in 2.3: stage 2 calls the mHC reference
  helper and does not launch the kernel; stages 7 and 9 execute it chained.
- The brief's "124.6 GiB" is the 124,610 MiB total with a unit slip (section 0 note).

### 5.4 What this map does not claim

No device token, no logits comparison against the fixture from a chain, no Phase 2 or Phase 3 exit
criterion, no serving throughput, no memory measurement for a wired chain, and no claim about the
unmerged `origin/task/phase3-gate-up` draft beyond its Lead-window gate/up arithmetic: the routing
weight, SwiGLU, combine, shared expert, end-to-end MoE, host-time savings and chain correctness are
all unclaimed. The one claim is the map itself: the stages,
their pins, their measured status, the gaps, and the ordering of the remaining work.
