# Kernel occupancy inventory: one decode position, launches and bytes

**Status: code-derived inventory + recorded-measurement anchors (2026-10-20, kernel-lead, task-2).**
**Update, same session:** the two routed-MoE kernels were then rewritten with the split-K tiling
described in section 4. Launch counts and bytes are unchanged (the same launches address the same
weights); only the per-launch geometry of rows 2, 3 and 5 below changed, and the table reflects the
current source. Section 4 states the reduction contract that keeps the rewrite bit-identical.
This page is the WS-C item 1 deliverable: every kernel launch in one decode position of the
current single-node device chain, the bytes each launch reads and writes, the arithmetic class, and
therefore the ordered fusion/optimization target list. It is written from the chain code and the
kernel sources on branch `task/kernel-occupancy` (base `c62fb0c`); where a recorded measurement
exists it is cited with its path and kept in a separate column. **No number here is a new device
measurement**: the byte counts are arithmetic from the shapes the chain launches, and the only
measured rows are the b53/b68 chain-log rows and the batch-19 harness table, both quoted below.

Method, so the numbers are checkable rather than asserted:

- The launch census is the chain's own ledger, recorded on hardware, not re-counted by hand:
  `device_launches=1168` with `entry_launches=42 post_launches=40
  attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680
  shared_moe_launches=160 head_launches=5 tail_norm_launches=1`
  (`docs/build/device-forward-chain.md:1438`, b53 split-counter accounting record; the same
  ledger is stated in code at `crates/sparkflash-cli/src/device_chain.rs:1304` and
  `docs/STATUS-2026-10-19.md:41`). `head_launches=5` = 4 rank shards of `cert_lm_head_sm121`
  plus the `lm_head_reduce_sm121` cut-over; b53's own line says the reduce adds the 1,168th launch.
- The shapes are the released geometry: hidden 5120, 40 layers, 384 routed experts top-6,
  intermediate width 2304 as four rank slices of 576 padded to 640 (D30), 64 heads x head_dim 512,
  q_lora_rank 1280, o_groups 8 x o_lora_rank 1024, vocab 129,280 as four 32,320 shards
  (`kernels/m4_common_sm121.cuh:56-105`, `crates/sparkflash-kernels/src/m7_layer0_shapes.rs:56-93`).
- The device chain this page describes runs the **attention and shared-expert weights at their
  whole (TP=1) widths** and the **routed experts as four rank-slice launches**; that is the b68
  single-node chain, not the future four-rank arm. Per-position bytes are therefore one node's
  traffic, and the roadmap's per-rank resident set is a different accounting.
- Byte counts count what the kernel arguments address: weights + scales + activations read, and
  outputs written. Read-modify-write accumulators count as both. `slot_acc` readback bytes are
  included in the combine row. Cubin pin names are in `crates/sparkflash-kernels/baseline/kernel_baseline.txt`.

## 1. Launch inventory for one decode position (40 layers, tokens = 1)

| # | class | kernel | launches/position | geometry per launch | bytes read | bytes written | arithmetic class |
|---|---|---|---:|---|---:|---:|---|
| 1 | routed MoE | `moe_router_sm121` | 40 | 1 block x 256 thr (1 token; 8 tokens/block), 384 experts x 5120 bf16 gate | 3,933,696 | 48 | BF16 dot per expert, serial ascending K; softplus/sqrt/logistic; top-6 scan |
| 2 | routed MoE | `routed_gate_up_sm121` (w1) | 40 x 4 ranks = 160 | grid (6 rows, 20 col tiles) x 256 thr = 32 output cols x 8 parallel K groups, 576 real rows/640 padded, K = 5120 | 9,405,600 | 7,680 | W4A8 GEMV: independent 32-value block partials, one ascending-order reducer per column, one f32 rounding per block |
| 3 | routed MoE | `routed_gate_up_sm121` (w3) | 160 | same | 9,405,600 | 7,680 | same |
| 4 | routed MoE | `swiglu_mxfp8_quant_weighted_sm121` | 40 x 4 ranks = 160 | 6 rows x 640, weighted SwiGLU + D8 quantise | 15,384 | 3,564 | elementwise SwiGLU (exact), one bf16 round, FP8 ceil-rule quantise |
| 5 | routed MoE | `b12x_mxfp4_moe_sm121` (combine) | 40 x 4 ranks = 160 | grid (1 token, 160 hidden tiles) x 256 thr = 32 output cols x 8 parallel K groups, K = 640 (20 blocks), 5120 outputs, 6 slots | 10,571,320 | 122,880 (accum) | W4A8 GEMV: independent block partials, one ascending-order reducer per (slot, column), f32 slot add |
| 6 | attention projections | `b12x_mxfp8_dense_sm121` (wq_a) | 40 | 64x64 tile, 256 thr, 1 row x 1280 cols, K = 5120 | 6,565,280 | 2,560 | MXFP8 block-scaled GEMM, scalar reference path, per-element serial K |
| 7 | attention projections | `b12x_mxfp8_dense_sm121` (wq_b) | 40 | 1 x 32768 cols, K = 1280 | 41,985,320 | 65,536 | same |
| 8 | attention projections | `b12x_mxfp8_dense_sm121` (wkv) | 40 | 1 x 512 cols, K = 5120 | 2,629,280 | 1,024 | same |
| 9 | attention projections | `b12x_mxfp8_wo_a_sm121` (wo_a) | 40 | grouped 8 x 1024, 1 x 8192, K = 4096 | 33,591,424 | 16,384 | MXFP8 grouped low-rank GEMM, scalar path |
| 10 | attention project. | `b12x_mxfp8_dense_sm121` (wo_b) | 40 | 1 x 5120, K = 8192 | 41,992,448 | 10,240 | MXFP8 block-scaled GEMM, scalar path |
| 11 | attention body | `flash_mla_csa2_seq_sm121` | 40 | 64 head blocks x 256 thr, window 128 + <=512 selected latents, latent 512 | 292,736 | 196,608 | MLA attention: per-head online softmax over window+selected, FP4 latent decode |
| 12 | mHC | `mega_mhc_sm121` | 42 entry + 40 post = 82 | 4 tiles x 128 thr, predictor [24 x 20480] f32, 4 x 5120 streams | 2,089,260 | 90,132 | matrix-free projection + 20 Sinkhorn iterations + RMS + D8 quantise |
| 13 | shared expert | `b12x_mxfp8_dense_sm121` (w1, w3, w2) | 40 x 3 = 120 | 2304-wide FP8 GEMMs (w1/w3 K = 5120; w2 [5120,2304]) | 35,430,000 | 14,400 | MXFP8 block-scaled GEMM, scalar path (w2 down: K = 2304) |
| 14 | shared expert | `swiglu_mxfp8_quant_sm121` | 40 | 2304-wide SwiGLU + D8 quantise | 9,216 | 2,376 | elementwise + quantise |
| 15 | certified head | `cert_lm_head_sm121` | 4 ranks | grid 32 chunks x 128 thr, 32,320 entries x 5120 bf16 per shard | 333,350,400 | 161,700 | bound screen (40 blocks) then exact BF16 dot for survivors; **scans twice** |
| 16 | head reduce | `lm_head_reduce_sm121` | 1 | 4 shards x 32,320 logits f32 | 517,125 | 20 | f32 max/argmax over 4 shards |
| 17 | tail | `rmsnorm_sm121` | 1 | 1 row x 5120 bf16 | 20,480 | 10,240 | RMSNorm, f32 statistics, bf16 out |

Counts: 40 + 160 + 160 + 160 + 160 + 40x5 + 40 + 82 + 120 + 40 + 4 + 1 + 1 = **1,168 launches**,
matching the recorded ledger. The 42 mHC entry launches are 40 layers plus the 2 extra identity
entries at the Engram layers (1 and 14); the Engram seam itself is still on the host, so
`engram_embed_sm121`, `engram_wkv_sm121` and `engram_gate_write_sm121` are pinned but **not
launched in this chain** (device_chain.rs:847).

## 2. Ordered by bytes (the fusion/optimization target list)

Per decode position, code-derived from section 1. A "class" is a group of launches that share a
kernel and an operand layout and would be fused or rewritten together.

| rank | class | kernels | launches | bytes read/position | bytes written/position | share of read | recorded measurement anchor |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | attention projections (all five) | 5 x `b12x_mxfp8_*` | 200 | **5,070,550,272 B = 5.071 GB** | 75,744 B | 39.4% | staged_matrix_bytes = 5,069,568,000 B (attention; b68 `docs/build/device-forward-chain.md:1004`); attention_device_us = 546,175 (b68 pos 2, `:1483`) |
| 1a | of which wo_a + wo_b | wo_a, wo_b | 80 | 3,023,354,880 B = 3.023 GB | 26,624 B | 23.5% | included above |
| 1b | of which wq_a + wq_b + wkv | 3 x dense | 120 | 2,047,195,392 B = 2.047 GB | 49,120 B | 15.9% | included above |
| 2 | routed MoE (all four sub-kernels) | router, gate/up, weighted SwiGLU, combine | 680 | **4,868,857,600 B = 4.869 GB** | 22,691,328 B | 37.8% | routed totals: staging_ms=3445 device_us=388431 host_bytes=4,515,856,824 (b53, `:1435`); routed_device_us=387,140 (b68 pos 2) |
| 2a | of which gate/up GEMV | routed_gate_up (w1+w3) | 320 | 3,009,792,000 B = 3.010 GB | 2,457,600 B | 23.4% | m10 PASS 8/8 verdicts (batch19-plan §7) |
| 2b | of which routed down/combine | b12x_mxfp4_moe | 160 | 1,691,411,200 B = 1.691 GB | 19,660,800 B | 13.1% | no current-ABI device row; m7 stage 3's old-ABI row is explicitly retired in the kernel header |
| 2c | of which router | moe_router | 40 | 157,347,840 B = 0.157 GB | 1,920 B | 1.2% | m9 PASS 7/7, 0.0285 ms/token (batch19-plan §7) |
| 3 | shared expert (dense) | 3 x dense GEMM + 1 swiglu | 160 | **1,417,300,000 B = 1.417 GB** | 576,000 B | 11.0% | staged_shared_bytes=1,417,171,200, shared staging_ms=1758, shared_device_us=306,274 (b53, `:1436`) |
| 4 | certified LM head | cert_lm_head (4 shards) + lm_head_reduce | 5 | **1,333,401,600 B = 1.333 GB** | 161,700 B | 10.4% | cert batched 2.853x PASS 6/6 (m22, batch19-plan §7); cert_lm_head at 11.1% of roofline (brief; m22 prints its own number) |
| 5 | mHC (entry + post) | mega_mhc | 82 | **171,319,320 B = 0.171 GB** | 7,390,824 B | 1.3% | mHC batched 4.32x (m23, batch19-plan §7); M7 stage 1 4/4 tokens |
| 6 | attention body | flash_mla_csa2_seq | 40 | **11,709,440 B = 0.0117 GB** | 7,864,320 B | 0.09% | sequence batched 3.567x PASS 6/6 (m20, batch19-plan §7) |
| 7 | head reduce / tail | lm_head_reduce + rmsnorm | 2 | 537,605 B | 10,260 B | 0.004% | lm-reduce 1.01x (m24, no win) |
| | **total** | 17 kernel classes | **1,168** | **12,873,976,109 B = 12.874 GB** | 89,742,188 B | 100% | implied roofline floor at 273 GB/s = **47.2 ms/position** |

**Reading the order honestly.** The program brief's ordering is "routed MoE first (most bytes), then
wo_a/wo_b, lm_head, attention projections, router/head, mHC". The code-derived bytes confirm the
*composition* but not the exact order: the five attention projections (5.071 GB) are **4.2% larger
than the whole routed MoE family** (4.869 GB), and the wo_a/wo_b pair alone (3.023 GB) is the
single largest *pair-of-kernels* target. The brief's order is exactly right if "attention
projections" means only wq_a/wq_b/wkv (2.047 GB), which is then the 5th target behind wo_a/wo_b
(3.023 GB) and lm_head (1.333 GB). I state the code-derived order in the table and flag the
difference rather than repeat the brief; the *priority* still starts at routed MoE, and the reason
is measured time, not bytes: in the b68 position-2 row the MoE bucket's device time
(`moe_device_us=4,106,543`) is 78% of the position's `total_device_us=4,652,718`, against
`attention_device_us=546,175` for 4.2% more bytes (`:1483`). Byte traffic is not what makes the
MoE kernels slow; their scalar serial-K schedule does (see section 4).

## 3. Recorded-measurement anchors (verbatim chain-log rows, not re-measured here)

b53 split-counter accounting record (`docs/build/device-forward-chain.md:1435-1438`):

```text
routed MoE totals: layers=40 router_launches=40 gate_up_launches=320 weighted_launches=160 combine_launches=160 moe_launches=680 staging_ms=3445 device_us=388431 staged_expert_bytes=9349493600 host_bytes=4515856824 ...
shared expert totals: layers=40 shared_launches=160 staging_ms=1758 device_us=306274 total_us=2064542 staged_shared_bytes=1417171200 host_bytes=620800 ...
chain totals: layers=40 ... attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680 shared_moe_launches=160 head_launches=5 tail_norm_launches=1 moe_launches=840 device_launches=1168 ... device_us=5506333 ... staging_ms=12793 host_ms=14368 total_chain_ms=17452 host_bytes=4591049624 staged_matrix_bytes=16073744000 ...
```

b68 steady-state position 2 (`:1483`), the only per-position timing row used in this page:

```text
attention_staging_us=92278 attention_device_us=546175 moe_staging_us=4964229 moe_device_us=4106543 routed_staging_us=3682850 routed_device_us=387140 shared_staging_us=1281379 shared_device_us=312517 total_staging_us=5056507 total_device_us=4652718
```

Labels: b53's `staged_expert_bytes` = 9,349,493,600 B is the chain's *cumulative uploaded routed
expert bytes* counter (`MoeDeviceStats::staged_bytes`, `crates/sparkflash-cli/src/moe_device.rs:120`),
which counts the full 2304-wide expert tensors read from the checkpoint **and** the per-rank
padded slices staged for the kernels, so it is larger than the kernels' own DRAM traffic and is
not used as a byte count here - it is evidence of the per-position re-staging defect that device
residency removes. `host_bytes=4,515,856,824` for the routed family is the host-side read of the
same expert tensors and matches the code-derived full-expert read per position
(6 experts x 3 matrices x 40 layers = 4.512 GB), which is a useful cross-check on section 2's
device-read column.

The batch-19 harness rows quoted in the last column of section 2 (m9, m10, m20, m22, m23, m24, m12)
are from `docs/build/batch19-plan.md` section 7 and are the only rows in this page that were
produced by the named example harnesses; every other number is arithmetic from code.

## 4. Why routed MoE first: the schedule, stated as a code fact

Before the rewrite (the state the b68 row measured), both kernels ran one serial chain per output
element: `routed_gate_up_sm121` walked the whole K = 5120 in ascending 32-value blocks in one
thread (160 blocks, grid (tokens x top-6, 3) = 18 blocks for one decode token), and
`b12x_mxfp4_moe_sm121` walked K = 640 in 20 blocks and six slots in one thread (grid (tokens, 20) =
20 blocks). That is why 4.9 GB of traffic took 4.1 s: the kernels were latency-bound at 8 warps per
SM, not bandwidth-bound at 273 GB/s.

Both kernels now compute the same per-block terms with 32 output columns x 8 parallel K groups per
block and one reducer thread per column: the gate/up grid becomes (rows, 20) and the combine grid
becomes (tokens, 160), so a decode token occupies 120 and 160 blocks instead of 18 and 20. The
arithmetic contract is unchanged and is what makes the rewrite shippable under the byte-exactness
rule:

- For each output element the reference has exactly one f32 term per 32-value K block,
  `t_b = (partial_b * act_scale_b) * w_scale_b`, in ascending `b`, and the element is
  `fold(+0.0f, t_0 .. t_{K_BLOCKS-1})`. The new kernel computes each `t_b` in a different thread
  and stages it in shared memory; one reducer then executes the same ascending fold from `+0.0f`
  and the same unrounded add into the rank f32 slot accumulator. No add is reassociated, reordered
  or widened, so the device output is bit-identical for the same inputs - the split is a schedule
  change, not a numeric one.
- The gate/up kernel's scalar loop rounds once per 32-value block by construction
  (`__fadd_rn`/`__fmul_rn`), and the combine kernel's block term is now written with the same
  explicit intrinsics, so the ordering does not depend on nvcc's `-fmad` default. The old combine
  source let nvcc contract the last multiply into the add (the default cubin had 8 FFMA,
  `-fmad=false` had none); that is value-neutral wherever the old kernel could pass its own
  bit-exact harness, because an E4M3 x E4M3 product and a power-of-two ue8m0 scale are exact in
  f32, and where it could differ the new source is the twin's sequence.
- The dense kernel cannot be split this way: its scalar reference path accumulates element by
  element (`acc += a_value * w_value` over 5120 values), so every reordering changes the roundings.
  Its existing K-parallel variants declare themselves tolerance paths and are not shippable to the
  bytes-exact dense harnesses (m12/m13), which is why the dense family is a fusion target for a later
  session, not an occupancy target for this one.

## 5. Launch-count reductions available WITHOUT a kernel rewrite
(one-page note requested by the Lead; all figures are launch-count arithmetic, no timing claimed)

The 1,168 launches are 1,168 host launches per position; the b53 record shows
`total_chain_ms=17452` against `device_us=5506333` (5.5 s device), i.e. ~12 s/position is host
launch + staging + readback serialisation, not kernel time. In order of launches removed:

1. **CUDA graph replay of the whole position** (no kernel or arithmetic change): 1,168 launches
   become one graph launch per position (plus graph-internal node dispatch). This is the roadmap's
   "RDMA one-shot + CUDA graphs" stage and the largest single launch-count reduction available. It
   is orthogonal to the kernel work and needs graph-capture-safe buffers (no per-launch host
   readbacks inside the graph: today the router ids, the cert shard results and the mHC
   coefficients are read back per layer, which must become device-resident or graph nodes).
2. **Remove the per-position H2D staging** (device residency, task-34): not a launch-count
   reduction, but the b53 numbers put staged_matrix_bytes at 16.07 GB/position and staging at
   12.8 s; every staged weight is one host-side copy per launch. Residency removes the copies, not
   the launches.
3. **Multi-stream the independent launches** (no kernel change): the eight gate/up launches per
   layer are independent (same activation, disjoint weights and outputs; m17a's header states
   this), as are the five attention projections and the 4 rank-slice attention passes. Issuing them
   on 2-4 streams removes serialisation, not launches; graph capture plus streams is the same win.
4. **Use the already-measured batched entry points where a batch axis exists** (kernel code exists
   and is measured, so this is a harness/launch change, not a kernel rewrite): cert_lm_head_batched
   (m22, 6 rows, 2.853x), mega_mhc batched (m23, 4.32x), flash_mla_csa2_seq batched (m20, 3.567x),
   lm_head_reduce batched (m24, 1.01x no win). At decode tokens=1 there is no row batch today, but
   the DSpark verify step brings 6 rows and the four ranks can share a launch where the ABI takes
   row offsets. This is the one launch-count reduction that does not need graphs.
5. **Drop the 2 extra identity mHC entries** at the Engram layers (1 and 14): the Engram seam is
   host-side, so those layers run the mHC entry twice. Device Engram removes 2 launches/position;
   that is a kernel+wiring change, listed here only as the one launch-count saving visible inside
   the 322-chain launches.

What cannot be removed without a kernel/ABI change: the 8 gate/up -> 12 per-layer launches (fuse
w1+w3 into one launch, then fuse the weighted SwiGLU + quantise into the epilogue: 12 -> 4
launches/layer, 480 -> 160 per position), the 4 serial combine launches (4 -> 1 launch looping ranks
inside, no numeric change), and the per-layer router-id readback (device-side routing, which also
removes 40 D2H syncs/position).

## 6. Non-claims

- No new device measurement is reported here; every microsecond is quoted from a recorded log row
  with its path, and every byte is arithmetic from the shapes the chain launches.
- The 47.2 ms roofline floor is the **single-node chain's** weight traffic at the documented
  273 GB/s/node figure; it is not the roadmap's 13.9 ms/step budget (which is per rank on the
  four-rank arm) and it is not a prediction of achieved throughput.
- Launch counts are structure, not cost; the b53 host/device split is the only evidence here that
  host-side serialisation, not kernel time, dominates the position.
- No ordering in section 2 is a decision to fuse; it is the byte-ordered evidence the next session
  chooses from.
