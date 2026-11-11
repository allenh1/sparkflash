# Vendor survey: spark-ds41f (2026-11-09)

What [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) does that we do
not, measured on the same four GB10s, and which parts of it transfer to a native Rust + CUDA engine.
Read on 2026-11-09 at their `8a0f5630` (PR #1035, release `spark-ds41f-1107`). Their stack is
vLLM ("karmic-kraken-beta") + DeepSeek TileKernels + TileLang + sparknet RoCE + patched NCCL, so the
patches themselves (`patches/{vllm,tilelang,nccl,b12x}`) are 44 + 4 + 4 + 12 files we cannot apply;
the *mechanisms* are what this page is for.

Provenance of their numbers: `manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched.json`
(decode suite) and `manifests/suites/2026-11-07-spark-ds41f-1107-tp4/`. Ours: [performance.md](performance.md).

## 1. Head-to-head

Per-stream tok/s over the decode window, 256-token completions, temperature 0. Their prompt set is
their own 256-template set; ours replicates sparkDash's prompts request for request, so read the
ratio as +/-10% and the *shape* as exact.

| code | c1 | c2 | c4 | c8 | c16 |
| :-- | --: | --: | --: | --: | --: |
| SparkFlash | **131.8** | **92.2** | 60.8 | 38.6 | 23.1 |
| spark-ds41f-1107 | 101.7 | 79.1 | **72.9** | **67.7** | **46.3** |
| aggregate ratio | 1.30x | 1.11x | 0.77x | 0.50x | 0.41x |

Prose is the same shape with a smaller c1 lead (86.2 against 77.1) and a worse tail (18.3 against
38.9 at sixteen). Prefill: 6,400 tok/s at 32K against their 8,508, and 4,700 at 209K against their
8,261 at 262K; theirs is flat to 262K and 6,012 at 1M, ours decays with depth.

## 2. Why the curve bends

Measured on 2026-11-09 on the four Sparks, shipped `main` (release build, 16 lanes, 256-token
completions, temperature 0, thinking off, ~800-token prompt). A verify step serves the whole batch,
so the counter's per-sequence steps are summed: forwards = verify_steps / streams.

| streams | rows a forward | forward ms | tokens a forward | aggregate tok/s |
| --: | --: | --: | --: | --: |
| 1 | 5.3 | 40 | 4.4 | 110 |
| 2 | 10.0 | 45 | 7.4 | 193 |
| 4 | 18.4 | 66 | 8.4 | 253 |
| 8 | 33.0 | 114 | 15.6 | 274 |
| 16 | 59.9 | 140 | 57 | 408 |

The base is level with theirs - 30 ms for the fixed part of a 40-layer forward, against their
30.55 ms c1 step - and the whole difference is the marginal verify row: **~2.0 ms here against
their 0.8 ms**. Their release record puts the same number at 2.9 ms a live row, and their step
grows 2.0x from c1 to c16 where ours grows 3.5x.

Two facts fall out of the same sweep. Depth already adapts: 4.26 drafts a step at one stream
falling to 2.74 at sixteen, with acceptance rising 0.81 -> 0.95, so our drafter is winning the
argument and `STEP_COST` is not obviously mis-set. And the row is the whole target: at 0.8 ms a
row, 60 rows would cost 78 ms instead of 140, which is 730 tok/s at sixteen streams - their number.

Their c1 budget says where those milliseconds are: 5,708 MB a step, 70% of what the part can carry,
of which 2,520 MB is routed experts (13.8 distinct x 4.57 MB x 40 layers) and 1,543 MB attention
projections. Our own capture ([performance.md](performance.md#L119-L123), PDL off) splits the
growth from 6 to 39 rows as routed experts +60.8 ms (51%), collectives +33.1 (28%), dense +12.0
(10%), attention +5.7 (5%). A row that names 1.4 new experts is 263 MB of weight, which is 1.1 ms
at 240 GB/s: we spend ~2.0, they spend 0.8 because a 64-row tile amortises the read and a dead row
reads nothing.

## 3. Their catalogue, and what transfers

**Adopt.** Measured deltas are theirs unless marked.

| Idea | Their evidence | What it maps to here |
| :-- | :-- | :-- |
| Per-row dead/dummy masking through the router (`0010`, `0035`, `0042`) | 0.37 ms dead row against 2.9 live; 6-12%/step single stream without it | Already ours by truncation, see 4 |
| Fused split-K reduce + router epilogue (`0034`) | one launch fewer per MoE layer, bit-stable | `moe_router_sm121`, 25 us a layer |
| Routed experts as 64-row grouped tiles over a dispatch (`0029`, `0043`) | 0.4 us per missing row on a 64-row tile | the marginal-row fix, a kernel project |
| mHC 2-launch PDL + register-folded streams (`0739`/`0755`, `0040`) | -344..-412 us per 40-layer pass | `mega_mhc_sm121` |
| Direct producer writes into the registered send buffer | 1.3-1.6% decode | removes a staging copy a step |
| Greedy argmax epilogue in the head (`0605`) | -134..-193 us a call at 65-96 rows | only on the sampled path; our certified head already prunes at greedy |
| 12-bit packed BF16 head (`0012`, `0028`) | lossless, 75% of bytes, 1.03 ms against 1.36 | optional; see above |
| Bulk collectives on 4 NCCL channels, one-shot RoCE below 2 MiB | +22.5% prefill at 4K-64K | check our channel count |
| Sequence-parallel indexer split (`0025`) | 25% of a 4,096-token chunk | our SP seams |
| Context-aware chunk budget (`0584`) | -40.8% max step at 128K | our chunk planner |
| Prefill yield / late admission (`0222`, `0545`) | short-request TTFT 3.30 s -> 1.02 s beside a 131K prefill | UX, not throughput |

**Reject, with their own numbers.** BF16 sparse attention (~3% relative error against reference,
off), latent sparse attention (`0741`, off; top-512 recall 65-98%), verify-every-draft (-19% at c8),
tree drafting (-38%), a lower row-cost constant (regressed), lossless weight compression (1.4-4.7x
decode cost), an aux stream for indexer/KV overlap (reverted), sliced SP collectives over one-shot
RDMA (reverted after a thermal halt), 8 or 12 NCCL channels on TP3.

**Not applicable.** TMA `fence.proxy.async` stage-release bugs (`b12x 0004`/`0005`, which shipped
wrong outputs in 609 of 12,000 calls): our kernels use `cp.async` and `__syncthreads`, with no
mbarrier/TMA stage release anywhere in `kernels/`. Display carve-out needs root and a firmware
reserve. Everything PyTorch/vLLM-shaped.

## 4. Checked against our tree: dead rows are already solved

`tp4_forward/batch.rs:2249` forwards `n = depth + 1` rows - the anchor plus the depth the policy
chose - so a row we do not intend to commit is never computed, which is strictly cheaper than their
"compute the row, mask its experts". Their per-row mask exists here as
`scheduler/src/dspark.rs`'s `RouterMask`/`ExpertSource::Anchor` and is exercised by
`scheduler/tests/decisions.rs`, but nothing in the serving path calls `router_mask()`; it is the
model of a policy we satisfy by truncation instead.

## 5. The open item: a verify row costs 2 ms, and it should cost 0.8

`tp4_forward/spec.rs:1579` prices a step at `27 + 4 x depth` with no term for the batch. That is
close enough to the measured base (30 ms) and to the measured row (~2 ms), and the policy already
drops from 4.26 drafts at one stream to 2.74 at sixteen, so the depth model is not the lever the
sweep in section 2 says it is. It would matter only if rows were cheap.

Two things would make it wrong in the other direction. Pricing the *shared* step rather than the
sequence's own rows, with the c1 constant, extrapolates to 283 ms at 64 rows against a measured
140 - it over-prices and collapses depth to zero, the documented failure mode in
[optimization-strategy.md](optimization-strategy.md#L483). And their `0587` control shows a global
depth cap costs high-confidence requests 43.8% of their tokens a step, so any cap has to be per
request against a measured row cost.

What the sweep does support is the row itself. At 0.8 ms a row our sixteen-stream step would be
78 ms instead of 140, which is 730 tok/s against their 741: the entire remaining decode gap is the
marginal row, not the policy and not the base.

## 6. Where the decode step actually goes

The per-kernel ledger, read at drain with `SPARKFLASH_STAGE_TIMING=1`, 256 tokens a stream,
per forward (one forward serves the batch). The ledger forces the drained launch path, so its wall
is ~1.7x the clean step; the *share* is what it is for.

| | c1, 58 ms | c16, 164 ms | per-row delta |
| :-- | --: | --: | --: |
| routed MoE (gate/up, down, router, combine) | 29.8 ms (51%) | 101 ms (62%) | **+69 ms (65%)** |
| dense projections (the GEMV/MMA family) | 9.3 ms (16%) | 13 ms (8%) | +12 (11%) |
| attention (scores, values, softmax) | 1.7 ms (3%) | 8.7 ms (5%) | +7.4 (7%) |
| mHC | 1.9 ms | 5.5 ms | +3.6 (3%) |
| lm head | 2.7 ms | 3.5 ms | +1.7 |
| collectives (`opt_sum_ranks_bf16`) | **0.55 ms (1%)** | **0.84 ms (0.5%)** | **+0.3 (0.3%)** |
| all kernels | 58.3 ms | 164 ms | +106 ms |

Two entries here contradict what this page said before it was measured, and the contradiction is
the finding:

- **Collectives are ~1% of a decode step, not 28%.** The 2026-11-01 PDL-off capture attributed
  +33.1 ms of the c1-to-c8 growth to them; on this build, across the whole c1-to-c16 range, they
  move by 0.3 ms. The transport is not where the step goes, and "fix the collectives" is a dead
  branch.
- **The marginal row is ~1.94 ms of kernel time and 65% of it is routed MoE.** A row that names
  1.4 new experts is 263 MB of weight, which is 1.26 ms at 208 GB/s: the expert read is at its byte
  floor already. The row cannot be made cheaper by reshaping the kernel; only fewer bytes can do
  that.

What is *not* at the floor is the aggregate rate. At c1 the step reads ~7.7 GB of weights in
58.3 ms (133 GB/s) where their c1 budget is 5.7 GB in 30.55 ms (187 GB/s), so there are two
separable deficits: we read 1.35x the bytes because we verify 5.3 rows against their 3.4 (and that
pays: 4.4 tokens a step against 3.0), and we carry those bytes at 0.7x their rate.

### 6.0 Is the row latency-bound or bytes-bound?

Bytes. At c16 the two dedup expert kernels move the expert weights at **177 GB/s**: gate/up carries
two thirds of an expert's bytes (10.5 GB in 59.3 ms) and down one third (5.3 GB in 30.1 ms), against
this part's 240-267 GB/s. Calls per forward grow 49 -> 63 while rows grow 5.3 -> 60, so each call
absorbs 8.6x the rows in 3.4x the time: fixed per-call overhead is amortising, not accumulating.
There is ~1.4x of headroom in the kernel and no more; a reshape cannot make a row cheaper than the
263 MB it names.

What that leaves, in order: the expert read at 177 of ~240 GB/s (worth ~35% of the MoE time), and
the **non-MoE 33 ms of a 164 ms forward at c16** (20%), which moves far fewer bytes per
millisecond - dense projections 13 ms, attention 8.7, mHC 5.5, head 3.5, engram 1.8, collectives
0.8, and a 5.4 ms tail of 72 kernels under 250 us each. At c1 the non-MoE share is 28.5 ms of 58.3.
That tail is what a fused stack like theirs does not pay: every layer runs several small launches
(`prologue_bf16`, `rmsnorm_bf16`, `mhc_post`, `mhc_tail`, `kv`, `attn_scores/values/softmax`,
`sum_ranks_bf16`), and the ledger counts 102 distinct kernels where their chain runs fused epilogues.

### 6.0.1 Correction: §6's milliseconds were normalised by a guessed forward count

The ledger carried calls, time and a phase, but no row count, so "per forward" was inferred from a
kernel's launch count. The inference was wrong: `opt_router_sm121` launches once a layer, which
puts this run at 142 forwards, not the 72 §6 assumed. Every absolute millisecond in §6 is therefore
about twice too large, and the reading-rate numbers in §6.0 inherit the error. The shares survive,
because they are ratios taken inside one forward.

The ledger now carries rows (one step is one forward, and a kernel's rows are counted once a step,
not once a launch), so the table can be stated rather than inferred. Same binary, same runs, ledger
on, prompt 829 tokens, 256 greedy tokens a stream:

| | c1 | c16 |
| :-- | --: | --: |
| forwards | 97 | 142 |
| rows a forward | 4.08 | 29.5 |
| tokens a forward | 2.63 | 26.0 |
| kernel time a forward | 32.2 ms | 85.9 ms |
| kernel time a row | 7.89 ms | 2.91 ms |

| the forward, by family | c1 | c16 |
| :-- | --: | --: |
| routed MoE (gate/up + down) | 15.0 ms (47%) | 55.1 ms (64%) |
| dense (the GEMV/MMA families) | 8.4 ms (26%) | 10.0 ms (12%) |
| attention (scores, values, softmax, kv) | 1.5 ms (5%) | 6.9 ms (8%) |
| mHC | 2.0 ms (6%) | 3.2 ms (4%) |
| lm head | 1.4 ms (4%) | 1.8 ms (2%) |
| router and combine | 0.7 ms (2%) | 2.5 ms (3%) |
| collectives (`opt_sum_ranks_bf16`) | 0.38 ms (1%) | 0.60 ms (1%) |

The expert read, with bytes from the row count instead of a call count: at c1 four rows name about
5.8 of this rank's 96 experts a layer (684 MB a forward) in 9.6 ms of gate/up, which is **71 GB/s**;
at c16 thirty rows name about 35 (4.2 GB) in 31.7 ms, which is **132 GB/s**, and the down projection
89 GB/s. Against this part's 273 GB/s that is 26% and 48%, not the 74% §6.0 claimed.

Both are lower bounds, and the bracket is why: it synchronises after every launch, so kernels never
overlap and each one starts cold. At c1 the armed kernel sum is 32.2 ms a forward against a 23.9 ms
unarmed wall, so the bracket inflates a kernel's own time by at least a third there. That leaves a
band of 132-180 GB/s at c16, and the band straddles the decision: at 132, reading the experts at 250
takes 25 ms out of an 86 ms kernel budget, a fifth of the wall; at 180 it takes 11 ms out.

So the next step was not another kernel change but a bracket that does not drain: pool the event pairs
and read them at drain, then re-run c1 and c16. That is what the probe does now (one pair is recorded
again rather than recreated, and a pool of them is resolved behind one synchronise), and it took the
armed c1 wall from 97 ms a forward to 75. It is still 3x the clean step, because two
`cuEventRecord`s a launch at 13 us each is 34% of that wall - the events, not the drain, are what is
left. The absolute numbers in the next section therefore come from nsys, which owes the ledger
nothing.

### 6.0.2 The authoritative split: nsys, with no probe and no drain

The ledger cannot be the source for absolute per-kernel numbers, and now for two reasons. Its own
bracket drains (13 us a `cuEventRecord` on the legacy stream, 34% of an armed c1 wall), and - the one
that matters more - **it only sees launches that go through our driver wrapper**, so every NCCL
kernel is invisible to it. The split below is nsys on rank 0 with no probe armed at all: the same
binary, the same prompt, 16 lanes, 256 tokens a lane, 2720 tokens off 111 forwards, 13.28 s of summed
kernel time in an 11.6 s window (13% of it overlaps another stream).

| the run, by family | kernel time | share |
| :-- | --: | --: |
| routed MoE kernels (gate/up, down, router, combine, shared) | 5.52 s | 41.6% |
| **NCCL (all-to-all, AllGather, ReduceScatter)** | **1.85 s** | **13.9%** |
| dense projections (the GEMV/MMA families) | 2.13 s | 16.1% |
| `opt_rows_prefetch_l2` | 0.85 s | 6.4% |
| attention (scores, values, softmax, kv, rope) | 0.70 s | 5.3% |
| mHC | 0.48 s | 3.6% |
| norms and prologues | 0.38 s | 2.9% |
| LM head | 0.27 s | 2.0% |
| Engram | 0.09 s | 0.7% |

The single kernels, per launch: gate/up 779 us (3860 launches), down 398 us (4148), `prefetch_l2`
96 us (8809), router 42 us (4460), attention scores 53 us (4460), down-combine 150 us (600).

**The expert read is at 134 GB/s, and the down projection at 131.** Per launch, gate/up moves about
105 MB (35 distinct experts of this rank's 96, 2.95 MB each) and down about 52 MB, so against this
part's 273 GB/s both are at **~49% of peak**, not the 74% the ledger-based figure claimed. That is
the headroom: reading the experts at 250 GB/s takes 13 ms out of a 55 ms MoE budget at c16. (§6.0.5
corrects both numbers: a call cannot name more experts than it has rows, so the ceiling is 112 GB/s
at c16 and 47 at c1, and the block count is not what limits it.)

Two things in this table contradict earlier sections and are the point of taking the profile:

- **The collectives are 14% of the kernel time, not 1%.** Section 6 killed "fix the collectives" on
  `opt_sum_ranks_bf16`'s 0.6 ms a forward - but that kernel is a device-side rank sum, not the
  collective. The collectives are 166 NCCL SendRecv launches a forward at 64 us each: the MoE
  all-to-all, 10.6 ms a forward, plus the AllGather ring. So MoE is 41.6% in its own kernels and
  about 51% with the all-to-all that feeds it, and the transport is neither free nor the whole story.
- **`opt_rows_prefetch_l2` is 6.4%** - 7.6 ms a forward of prefetching, 79 calls of 96 us. Whether
  it is prefetching what the next layer reads, or evicting what this one needed, is a question with
  an easy experiment and no measurement yet.

### 6.0.3 The all-to-all is at the transport's floor, and no switch moves it

The profile put NCCL at 13.9% of the kernel time, so the first thing to try was the transport. It is
already as fast as this fabric goes at these sizes, and here is the evidence, all of it measured on
the four nodes.

**In isolation.** `gpu-reduce-bench --device-only` runs the same NCCL wrapper the layer uses, one
rank a node, forty enqueues behind one synchronise - the shape a layer port has. AllReduce(Sum),
p50 a call:

| payload | rows it is | p50 | enqueue p50 |
| --: | --: | --: | --: |
| 10,240 B | 1 | 43.2 us | 4.1 us |
| 40,960 B | 4 | 48.5 us | 4.3 us |
| 151,040 B | 15 | 78.6 us | 5.0 us |
| 302,080 B | 29.5 | 116.8 us | 6.5 us |

**In situ** the step's exchange is two `ncclDevKernel_SendRecv` launches of 63.8 us, so 128 us for
the same 302 KB - within 10% of the isolated number. There is no overhead to remove: the collective
is the transport.

**The channel count wins in the microbench and does nothing in situ.** Interleaving the two configs
in a deliberately non-alternating order - the alternating order I tried first aliases with a
period of drift, and produced a false 30% win twice - gives a clean split at 302 KB: baseline 120.5,
123.9, 124.7, 132.3 us against `NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16` at 85.3, 88.3, 90.7,
95.8 us, no overlap. That is 28% off a four-rank `AllReduce`. **In the step it is worth nothing:**
profiling the serving run with that environment gives `ncclDevKernel_SendRecv` at 62.8 us against
the baseline's 63.8, 10.91 ms a forward against 10.57, and a share of 9.7% against 8.9%. The bench
measures NCCL's ring `AllReduce`; the layer issues six point-to-point enqueues, and the channel
count that spreads a ring across more SMs does not touch them.

The other switches do not move it either: at 302 KB `NCCL_ALGO=Tree` is 233 us (2x worse),
`NCCL_NET_GDR_LEVEL=SYS` 118 us and `NCCL_DMABUF_ENABLE=1` 127 us, both inside the run-to-run
spread. `NCCL_PROTO=Simple` and `LL128` remain unmeasured; neither can help the p2p exchange for the
same reason the channel count does not. The marginal rate is ~4 GB/s a rank, which is what a
four-rank ring of 300 KB messages gets here.

The general lesson, and it has cost this page two wrong claims: **benchmark the call the layer
issues.** A transport microbenchmark is only a floor for the serving path if it enqueues the same
operation, and NCCL's AllReduce and a group of SendRecv are not the same operation.

**Our own one-shot is not the answer either, and the instrument that says so is broken.**
`rocenante-probe` reports 2097.1 us at 64 KB and 2096.9 us at 296 KB - identical within a 13 us
spread over fifty samples - and 3146 us at 1 MiB. A flat 2.1 ms across a 4.5x size range is a poll
interval, not a transfer, and a probe that cannot resolve below a millisecond has not been measuring
the one-shot path at all: the doctor's 25 us gate for that path has been reading a floor. The
b12x-style one-shot is therefore unmeasured rather than refuted, and nothing on this page justifies
porting it to the device arm.

What that leaves for the all-to-all: the layer needs two of these a layer and cannot fuse them,
because the attention reduce feeds the residual the MoE then reads. Halving the launch count with a
single AllReduce instead of six point-to-point enqueues would save ~9% of 128 us an exchange, which
is 0.9 ms of a 140 ms forward. The transport is a dead end at 16 lanes; at c1, where the payload is
one row and the 43 us floor is the whole cost, it is 84 calls of 43 us = 3.6 ms of a 24 ms step, and
the floor moves only with the number of collectives.

### 6.0.4 The L2 prefetch warms exactly its spans, and the wall does not move

§6.0.2 left one question open: whether `opt_rows_prefetch_l2` prefetches what the next layer reads,
or evicts what this one needed. Both halves now have answers - the first from the code, the second
from an A/B with the ledger on and off.

**It prefetches the next consumer's first read, in consumption order.** `rows.rs` issues it twice a
layer on a side stream that waits for everything queued so far, so it starts as the run's stream
enters the collective. `prefetch_ffn` rides the attention all-reduce and covers the *current*
layer's FFN entry: the mHC seam's mix rows, the router's gate matrix and the shared expert's w1 and
w3 slice. `prefetch_attention(layer + 1)` rides the MoE all-reduce and covers the *next* layer's
entry: that seam's mix rows, wq_a, wkv and wq_b (`SPARKFLASH_PREFETCH_WQB=0` drops the last). Every
span is the first thing the following kernels read, and the spans are queued in that order, so a
ring that overruns its window truncates the tail rather than the head.

**At the wall it is worth nothing.** Pinning the prompt (`BENCH_NONCE`, see below) makes an arm
repeat to 0.1%, which is what it takes to see a 1% effect; c16 is 2941 tokens and c1 179, identical
in every unarmed arm, 16 lanes of 256 greedy tokens (the "on" rows below are the eight-block
geometry of the next paragraph but one; the one-block build measured 250.2 at c16, the same):

| | prefetch on | prefetch off |
| :-- | --: | --: |
| c16 tok/s, unarmed | 250.2, 249.1 | 250.1, 249.4 |
| c16 tok/s, ledger on | 233.7, 231.4 | 231.8, 231.9 |
| c1 tok/s, unarmed | 74.4, 73.7 | 73.0 |

**And the parts of that nothing are visible in the ledger.** Two runs of the same arm agree to
within 5% on every kernel and to 0.5% on most, so a 25% or 39% difference between arms is real:

| kernel (c16, 132 forwards) | prefetch on | off | |
| :-- | --: | --: | :-- |
| `opt_router_sm121` | 178.4 ms | 223.5 ms | its own span: 25% saved |
| `opt_rows_mhc_partials_rb4` | 231.4 | 283.3 | the seam mix rows: 18% saved |
| `opt_gemv_rows_f32a_f32o_r4c1` | 273.2 | 295.3 | 8% saved |
| `opt_rows_mhc_post` | 183.9 | 132.5 | **not prefetched, 39% slower** |
| decode, all kernels | 75.72 ms a forward | 77.57 | 2.4% saved |

So the prefetch does work on what it covers, and it costs at the one kernel it does not cover: the
FFN post runs immediately after the collective, i.e. while the prefetch is still streaming, and it
loses a third of its time to that traffic. The two effects cancel almost exactly.

**Geometry is not the lever.** The prefetch was one block of one warp; the spans are ~13 MB against
a 64 us collective window, and nsys puts the kernel at 96 us a launch, so a third of it spills
into the step it was meant to get ahead of. Striding the same chunks over eight blocks of one warp
(one per SM) changed the balance above by under 2% and the wall not at all. The window, not the
issue rate, is what binds: at that size the prefetch is limited by the bandwidth the collective
leaves over, so only *fewer or better-chosen bytes* can help - the obvious candidate being the FFN
post mix it currently competes with. Nothing here justifies a third attempt at the launch shape, and
the feature as it stands is a wash rather than a mistake.

**What it cost to measure it, and a correction to §6.0.1.** §6.0.1 blamed the leftover on the two
`cuEventRecord`s a launch, on a binary that predated 286c80ff - the commit that made the bracket
stop draining. Once that dispatch is actually deployed the ledger costs **7% at c16** (12.59 s of
wall against 11.76) instead of 52%, and **94% at c1** (4.68 s against 2.41), which is the same story
as §6.0.1 with a smaller constant: the events per launch are what is left, and c1 has ~1400 launches
in a 24 ms step. Anything below c16 still needs nsys or an unarmed run.

The other thing this measurement needed: a fixed prompt. `bench/ledger.py` draws a random nonce, so
each arm was a different workload - the same binary produced between 2786 and 4059 tokens across
eight c16 arms - and that spread is larger than every effect measured here. `BENCH_NONCE` pins it.

### 6.0.5 The routed expert read is capped at 41% of this part, and the block count is not why

§6.0.2 put the expert read at 134 GB/s, 49% of this part's 273, and derived that from an estimate of
how many experts a call names. The estimate was too high, and the launch geometry says so: the dedup
kernels run `grid (576 / 32, rows)` and `grid (5120 / 64, rows)`, and every block past the plan's
distinct-expert count exits, so a call reads `count <= rows` experts. At c16 a forward carries 25.5
rows and at c1 3.6, which caps the c16 call at 25 experts, not the 35 §6.0.2 assumed. The rate is
therefore **at most 112 GB/s at c16 - 41% of this part - and at most 47 GB/s at c1**.

| per call | c16 | c1 |
| :-- | --: | --: |
| `opt_moe_gate_up_dedup_sm121` | 674 us, 40 a forward | 225 us, 40 a forward |
| `opt_moe_down_dedup_sm121` | 328 us, 42.7 | 110.5 us, 41.2 |
| together | 41.0 ms of 75.7 (54%) | 13.6 ms of 34.3 (40%) |
| bytes a call (2.95 MB / 1.6 MB an expert) | <= 74 MB / 40 MB | <= 10.6 MB / 4.8 MB |
| blocks a call (18 / 80 tiles an expert) | 459 / 2040 | 65 / 288 |

**Four times the blocks buys 11%, so occupancy is not the lever.** The down projection splits an
expert into 80 tiles where gate/up splits it into 18, and per byte read it is 11% faster at both
levels: 124 GB/s against 112 at c16, 52 against 47 at c1. If the deficit were resident blocks, the
kernel
with 4.4x as many of them would not be within 11% of the one with the fewest. What is left is inside
the block: 128 threads, a warp owning 8 columns, and the weight row streamed **one byte a load**
(`w_row[i0 / 2u]`, decoded a nibble at a time) over the 160 blocks of K - so the instruction stream
is 320 loads and 320 dependent multiply-add pairs per thread, and the memory system sees one-byte
requests with a 2560-byte stride between neighbouring lanes.

**The instrument that would separate the two is not available here.** `ncu` needs counters and this
host refuses them to a normal user: `ERR_NVGPUCTRPERM` on a locally compiled 64-thread kernel,
before the model is involved. So the bound above is measured and the mechanism is inferred, and the
page should be read that way.

**What that leaves.** The contained change is to vectorise the weight stream - 16-byte loads of 32
values, the same values in the same ascending 32-value fold, only the load instruction differing -
and re-measure: at 41% of peak with one-byte loads, the instruction stream is the first suspect. If
the rate does not move, the next reading to take is the plan's own count (`OptMoePlan::count`, the
first word of `plan_out`), which would turn the ceiling into an exact rate and say whether the c16
call reads 25 experts or 13.

## 6.0.6 The direct-RDMA exchange: the GB10 gate is open, measured

The vendor note proposes replacing the NCCL exchange with one RDMA write per peer plus an arrival
flag the sum kernel spins on. Its open question was the first one worth answering, because
`nvidia-peermem` is still absent on these nodes, and it is now answered on the hardware:

| question | measurement, .80 and a worker (GB10, ConnectX-7, `rocep1s0f1`, GID 3) |
| :-- | :-- |
| does `ibv_reg_mr` take GPU-visible pinned memory? | **yes**: `cudaHostAlloc(MAPPED, PORTABLE)`, 4 MiB, registered on both nodes, key returned |
| does it take `cudaMalloc` device memory? | **no**: `EBADF` ("Bad address"), so the peermem limitation is real and unchanged |
| does the GPU read exactly what the NIC wrote? | **yes**: 2000 round trips of 10 KB, payload verified word for word, **0 bad words** |
| does the host see the GPU's writes, and the GPU its own? | **yes**: the mapped alias is the host pointer itself (`host_alias_equal=1`) |
| one 10 KB RDMA write | **5.73 us** average (4.91 min, p99 10.8) on one rail, `ib_write_lat` |
| the collective it would replace | **43 us** (the other effort's microbench), 84-95 of them in a c1 step |

**The doorbell is cheap only if nothing fences it.** A resident kernel publishing a word to the host
measures 1.07 us (plain store), 1.39 us (`st.global.wt`), and **7.54 us with a
`__threadfence_system()`**, including a 3.26 ms outlier. The store is not the cost, the fence is:
what the doorbell has to say is "the bytes are in memory", and a write-through store says it.

**A fresh kernel launched at an idle GPU starts 2.2 ms late** (2.9 us to enqueue, 2.2 ms to
completion, consistently), so a kernel that exists only to wait for a peer is 2.2 ms worse than a
resident one; a resident spin kernel sees an arriving flag in about 2222 cycles, 1.2 us. Any
per-collective kernel here has to be launched into a busy stream, which is a constraint this
platform imposes and not one the vendor note mentions.

**Projected per exchange: about 11 us** against 43 - 1 doorbell, ~1.5 post, 5.7 wire, ~1.2 for the
peer's kernel to see the flag, ~2 fold. Across 84-95 exchanges that is 2.5-3.5 ms of a step: the
10% at c1 the note predicts, ~6% at c16.

**The shape that keeps the bits identical**: the peer writes into the same `gather + peer * count *
2` slot the NCCL path writes into, so `opt_sum_ranks_bf16` is untouched and the rank-order fold
stays the same fold, with the 8-byte flag appended after the row. The send side needs one new kernel
- copy the row into the mapped slot, one fence, publish `seq` - because the NIC cannot read
`cudaMalloc` memory. The proxy posts three writes and spins on three arrival flags;
`sparkflash-comm`'s `rdma` feature is already forwarded from the CLI's `cuda` feature and already
dlopens libibverbs, so nothing new links.

What is left to build: that staging kernel, the QP mesh over the existing TCP control channel, the
proxy (a host spin would serialise the stream by ~6 us a collective, so it belongs on its own
thread), and the A/B. The two probes behind the table are in `scripts/probes/`: `gb10-rdma-gate.cu`
(registration, coherence, the 2000-round-trip verify) and `gb10-doorbell.cu` (the three doorbell
variants). The gate is what says the build is worth starting.

**Measured on four ranks, both collective sizes.** The projection above is now a measurement:
`scripts/probes/gb10-exchange-4rank.cu` runs the real exchange - a registered slot per peer, one
write per peer, an arrival flag per rank - on all four nodes, and takes the time from posting to
every peer's flag. 200 rounds, the first 16 discarded, on a box that was serving requests during
the measurement:

| row | exchange (first arrival) | exchange (all three peers) | NCCL group |
| --: | --: | --: | --: |
| 10 KiB (c1) | 5.5-6.0 us | **7.3 us** p50 (p99 9-13, max 17) | 43 us |
| 160 KiB (c16) | 42.5 us | **46.5 us** p50 (p99 50, max 53) | 43 us |

Re-measured on an idle box afterwards (the group up, nothing being served, load 0.25) the two numbers
came back 7.30 us and 46.5 us: the same within noise, so at these sizes the exchange is latency-bound
and does not care about the collective traffic sharing the NICs. That also means the first run, taken
while the box was serving, was not distorted by it.

So the fabric half of the exchange is **7.3 us, not the 11 us projected**, and the decode row is a
5.9x win on the collective. The 160 KiB row is not: 46.9 us against 43, a wash inside the noise,
which is the same "the all-to-all is at the transport's floor" result §6.0.3 measured from the
other direction. What that means for a step: 43 - 7.3 = 35.7 us saved per collective, times
84-95 collectives, is **3.0-3.4 ms of a 15.5 ms decode step**, about a fifth of it - which is the
direction the vendor note predicts and a little better than it. At c16 there is nothing to chase.

That projection is the exchange alone. It does not include the staging kernel's
`__threadfence_system()`, measured above at 7.5 us in isolation: if the fence costs that in
the loop too, the exchange is ~15 us rather than 7.3 and the saving is ~2.7 ms instead. Which of the
two it is can only be settled by the A/B.

The 46.9 us at 160 KiB is 10.2 GB/s of aggregate write, 41% of a 200 Gb/s link, and it is the number
to look at twice if the large-row case ever matters: 41 packets per write at MTU 4096 is where a
completion-queue or PCIe write ceiling would show up, and this measurement cannot tell them apart
from NCCL traffic on the same NIC.

**Two bugs, both mine, both about the wait.** The first version compared the arrival flag with `==`
and counted transitions; it deadlocked three ranks out of four on every run. `--trace` showed why:
one rank would read `flag 2` while waiting for `1`, because a peer had already published the next
round, and a transition counter that skips "already satisfied" peers never reaches a full set. The
wait has to be a predicate over every peer, re-evaluated, compared `>=` and never `==`. The second
version trusted GID index 3; a wrong index builds a QP that connects, accepts posts and never
delivers, which reads exactly like a lost packet. Selecting the GID by the rank's own rail address
is what fixed it, and both are in the probe's header so the next person does not repeat them.

**The module exists and runs on four ranks.** `crates/sparkflash-comm/src/rdma/write_exchange.rs`
carries `WriteExchange` and `SlabLayout`: the QP mesh over the existing endpoint-exchange control
plane (`qpinfo::exchange_endpoints`, so nothing new listens), the payload-plus-flag post, the
`>=` arrival wait, and the completion reap that has to happen before the send row is rewritten.
`write-exchange-smoke` drives it on all four nodes with the row staged on the host, checks every
peer's row and flag word for word every round, and reports **p50 9.0 us at 10 KiB and 46.7 us at
160 KiB** - the probe's numbers plus about 1.7 us for the reap.

Three things only the hardware run could say, and all three are in the code:

- **The flag has to be published from inside the registered region.** A first version posted an
  8-byte SGE pointing at a stack buffer for the flag; every payload write completed SUCCESS and the
  QP then died with `LOC_PROT_ERR` (status 4), which downstream looks exactly like peers that
  never publish. Completion status on timeout is what named it, and the statuses are in the
  timeout's error text for that reason.
- **One row set is not enough.** With one, a peer that runs a round ahead overwrites the rows a
  slower rank is still reading - the smoke test caught it as "peer 3's row reads ...2, not ...1".
  Two sets, selected by `seq & 1`, are sufficient: a rank reaches round n + 1 only after every peer
  published n, and it stages n + 1 only after its reduction of n is enqueued on the same stream, so
  the set a peer writes for n + 2 is free by then.
- **The first collective of a run is not the one to measure.** Round 0 costs 10-22 ms on every
  rank (first-packet neighbour resolution and QP ramp); every round after it is the 9 us above. A
  serving path has to warm the mesh before the first timed step, or the first token pays it.

At the point in the work this section describes, nothing called the exchange from a serving step: an
unwired module is not a speedup, and the A/B below did not exist yet. The step that was missing was
the run-scope open - the exchange has to be opened once per process rather than per step, because
`DeviceLayer::open` runs once per step and a 5 MiB pinned registration plus a QP mesh per step would
cost more than it saves. It is now opened on the first collective and is the default arm, so the
paragraphs below are a record of how the A/B was obtained rather than a description of the tip.

**The A/B, and why it is a 2% win and not a 20% one.** The exchange is now wired behind
`SPARKFLASH_EXCHANGE=rdma`: it opens once per process on the first collective
(`crates/sparkflash-deepseek-forward/src/rdma_exchange.rs`, a `OnceLock` and a mutex, so the pinned
registration and the QP mesh are paid once and every step borrows them) and replaces the per-layer
group with a staging kernel, one write per peer, and the same `opt_sum_ranks_bf16` fold over the
slab. Both arms ran the same binary, the switch being the only difference:

| shape | NCCL group | direct-RDMA exchange | change |
| :-- | --: | --: | --: |
| c1 decode | 64.5 / 64.6 / 64.8 tok/s | 65.9 / 66.0 / 66.2 tok/s | **+2.2%** |
| c16 batch | 247.9 / 247.9 / 248.9 tok/s | 249.4 / 250.3 tok/s | inside the noise |

The tokens are the group's: the same greedy 48-token generation came back byte for byte on both arms.
All four ranks announce the open, the control plane rendezvoused on port 29702, and the slab is the
engine's own (4 x 655,360-byte rows; the total is `SlabLayout::total_bytes`, which grows when a
region is added - it has since). This first A/B predates the two-rail build, so its rail counters
read one rail's traffic and the other rail's zero; on the default build both rails carry the row.

So the exchange is on the path. It is also not worth 3 ms a step, and the arithmetic says why. **In
the one-rail build this paragraph measures, the wire is 7.3 us and the exchange as wired costs about
35 us a collective**: the staging kernel and its `__threadfence_system()` (7.5 us in isolation, with
the outliers that measurement showed), the host's spin on the doorbell - which blocks the launch
stream until the GPU reaches the staging kernel - and the completion reap. Two of those three are the
handshake around the wire, not the wire.

Both of those readings were superseded. The count is 86 collectives a step, the mean row is 56-60 KiB
so the wire is 11-17 us rather than 7.3, and taking the host off the path removed the doorbell spin
from the critical path altogether; [rdma-exchange.md](rdma-exchange.md) records the reworked numbers
and the step deltas they produce. The one-rail 119 MiB traffic figure quoted above is likewise a
retracted term: 86 collectives of 56-60 KiB sent to three peers is about 14 MB per rank per step,
half of it on each rail, and a run's total is that times its steps.

[rdma-exchange.md](rdma-exchange.md) is the full experiment record: the gate, both probes, the module
and its three hardware findings, and the numbers above with the reasoning behind them.

That is the next thing to attack, and it is a different problem from the one this section set out to
solve: keep the row staging resident (or post the write from a device-side doorbell) so the host
never blocks, and the ~35 us floor should fall toward the wire's 7.3. Until then the exchange stays
an experiment that is now the default arm rather than the group; `SPARKFLASH_EXCHANGE=nccl` puts the
group back, and [rdma-exchange.md](rdma-exchange.md) records why the default is not a claim of speed.

## 6.1 The probe was broken, and is now fixed

`SPARKFLASH_STAGE_TIMING=1` armed the bracket and recorded nothing: nothing outside the test module
ever installed `StageRecorder`, and `stage_timing_report()` returns `None` until it exists. A run
paid the bracket's per-launch drain (2.2x at c1) for an empty ledger, and
`device-chain --timing-json` has been writing a timing file with no ledger section in it. The fix
is four lines in `cuda_driver.rs`: install the recorder on the first bracketed launch. The numbers
above are the first readings this probe has ever produced from a serving run; `serve` also gained
a drain-time print (`serve.rs`), because the ledger had no reader on that path at all.

## 7. Accuracy

Same numerical posture: FP32 logits, mHC stream projection in FP32 (theirs as three exact TF32
parts), FP32 shared-expert SwiGLU with one rounding, MoE partials rounded once per rank, rows
batch-invariant at temperature 0. They refuse FP8-attention arithmetic in the default path (measured
~3% relative error) and keep LSA off. Their determinism gate is 20/20 shapes bit-identical; ours
gates greedy bit-identity at any batch and byte-identity across eight and sixteen streams with
`SPARKFLASH_PREFILL_EXACT=1`.

On the one instrument both ran, we are ahead: tool-eval-bench 94/100 with no safety warnings against
their 91/100, which fails its safety gate on two scenarios. They publish more breadth (RULER 26/26,
LongBench 8/12 pilot, HumanEval+ 156/164, vision 83.25%) and a much larger test corpus (6,686).

Worth stealing from their incident record, none of which we gate today: sink-token rejection (their
`Say hi.` returned an empty answer until fixed), loop containment, and a logprob-saturation test
(theirs served exactly 0 nats past 16.6, up to 100% wrong at margins of 12-16).