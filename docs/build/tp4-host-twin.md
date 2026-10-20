# tp4-host: the window-free host TWIN and the four-node output run

Task-53 (S5.2a) + task-57 (residency). Owner: tp-dev. Status 2026-10-19: host TWIN landed and
measured on the dev host; three four-node runs produced the deliverable token sequence; task-57 made
every non-Engram weight resident in-process per rank and removed the per-step re-reads.

## 1. Why this exists

The single-node online baseline (node0:18000, released checkpoint, reference engine) answered
`/v1/completions` in **2m54s for 7 prompt + 4 completion tokens**, about 16 s/token, because
every token re-reads the weights. The four-node runs show the same problem moved into the TP path:
each rank has only its quarter slice on disk, but the whole replicated set is re-read every token.
Task-57 then made the plan slices and the replicated set resident and re-measured.

The stage has three jobs:

1. prove, without a driver and without a GPU window, that the sharded four-rank forward produces the
   same token sequence and near-identical logits as the single-rank reference engine
   (`InferenceEngine::prefill`), over the task-33 TCP transport;
2. run that same forward on four GB10 nodes and record wall time and bytes read per token; and
3. with `--resident-cache`, hold the rank's whole non-Engram working set in one in-process cache
   and prove that the per-token re-reads collapse, with the token sequence unchanged.

Per the Lead's 2026-10-19 scope the deliverable is **output**: a token sequence, and the baseline
beaten. The per-seam L2 numbers are **recorded**, not used as a gate.

## 2. What actually runs: Phase 1 (an intermediate, NOT vLLM's TP=4)

Only one class is sharded: **the routed experts of the 40 language layers**, with the design's
section-3 rule.

| class | rule |
| --- | --- |
| routed experts `w1`/`w3` (+ their scales) | Column: stored rows / world; local logical shape `[inter/world, dim]` |
| routed experts `w2` (+ its scale) | Row: stored input columns / world; local logical shape `[dim, inter/world]` |
| attention (wq_a, wq_b, wkv, wo_a, wo_b), indexer, compressor | replicated (full tensor) |
| embed, LM head | replicated (full; the embedding row is read by byte range, not materialised) |
| shared expert, router, mHC, norms, MTP heads | replicated (full) |
| Engram non-table tensors | replicated (full) |
| Engram tables | **disk-resident**, read per token by row range through the paged reader |
| **MTP expert blocks** (`mtp.0/1/2.ffn.experts.*`) | **replicated full** - outside the `layers.` plan |

Each layer runs its local routed block on the local expert slices and then **one canonical bf16
all-reduce** sums the four partials (bf16 payloads, f32 ascending-rank accumulation from 0.0, one
bf16 RNE round). The shared expert is added afterwards in host f32 by the replicated code. That is
40 all-reduces per token, one per layer, and no all-gather.

**Naming.** This is TP-style sharding of the expert intermediate width plus a replicated remainder.
It computes the same arithmetic as the reference (every replicated class is a numerical no-op by
construction), but it is *not* vLLM's TP=4 layout (which also splits attention and the head). Gate B
still needs the real TP layout and the device collectives (tasks 34/35/36). The engine's
`new_tp` / `ResidentSource` path (task-54/59) is proven on fixtures but is **not** the node path
of this document; tonight's served path is the task-53 loop. Neither is vLLM TP=4.

### Per-rank bytes: forecast vs the plan that actually runs

The released checkpoint index has **96,085 tensors**. The task-53 plan (`plan_hash
0xcd8adc09d2c71d55`, same on all four ranks) shards **92,160** of them: the language-layer routed
experts `w1/w2/w3` weights and scales (40 layers x 384 experts x 6 tensors = 92,160). Everything
else that is not an Engram table is replicated in full.

| class | names | GiB / rank | source |
| --- | --- | --- | --- |
| plan slices (language routed experts) | 92,160 | **67.236** | measured `resident_cache.plan_slice_bytes` = 72,194,457,600 B |
| replicated non-expert classes | 1,617 | **10.738** | measured, matches the design categories |
| replicated **MTP expert** blocks (full) | 2,304 | **6.724** | 3 MTP layers x 128 experts x 6 tensors at full I8/E8M0 size |
| **total non-Engram working set** | 96,081 | **84.699** | measured `resident_cache.bytes` = 90,944,617,800 B |
| Engram tables (NOT resident; paged by row) | 2 | 188.833 | disk-resident, read per token |

The design forecast in the first revision of this doc was **78.752 GiB/rank** for "expert slice +
replicated rest". It is wrong by **+5.947 GiB** and the discrepancy is now fully accounted:

| item | GiB |
| --- | --- |
| forecast (68.917 expert slice + 9.835 replicated) | 78.752 |
| plan slices are smaller: the forecast's 68.917 included the MTP experts at quarter size (6.724/4 = 1.681) | -1.681 |
| MTP experts are replicated at FULL size, not sharded | +6.724 |
| replicated classes the forecast did not count (MTP non-expert, engram-other, vision/aligner extras) | +0.903 |
| **measured** (67.236 + 10.738 + 6.724) | **84.699** |

The earlier header-walk estimate of the same set (`resident_bytes_estimate`, 90,944,617,800 B) was
therefore **right**, not an over-count: the resident cache fill reproduces it byte for byte on every
rank. The 78.752 forecast was a category error in the expert accounting, not a layout change.

## 3. The host TWIN fixture

`--write-fixture DIR` writes a deterministic safetensors checkpoint for a 40-layer tiny model:
`ModelConfig::tiny()` with `n_layers = 40`, all compression ratios 0, Engram at layers 1 and 14
with 1024 rows each, and `moe_inter_dim = 128` (tiny's 32 would leave a one-column expert-scale
grid that cannot be divided at world 4; 128 gives 4 scale columns and divides by 2 and by 4).

Every tensor is finite f32 from the seed `0x5eed`, except the Engram tensors whose plan storage is
FP8: the table weight and the `wkv` weight are E4M3 codes and their scales are E8M0 127. That is
deliberate: the engine's paged Engram reader is a byte-range reader whose row arithmetic assumes the
header's stored dtype, so those tensors must be FP8 in the file. The fixture is 2193 tensors,
21,878,732 bytes, hash `0xb656f193fb1b27f2` for the default seed (the writer is deterministic).

## 4. Exact host commands (dev host, measured)

```
# 1. write the fixture (one process, no ranks)
./target/debug/sparkflash device-chain --run-stage tp4-host --write-fixture /tmp/t53host/fixture

# 2. two ranks, loopback TCP, reference check and seam recording on rank 0
./target/debug/sparkflash device-chain --run-stage tp4-host \
  --tp-rank 0 --tp-world-size 2 \
  --comm-peers 127.0.0.1,127.0.0.1 --comm-base-port 24111 \
  --fixture /tmp/t53host/fixture --prompt-tokens 3,7,11 --max-tokens 3 \
  --record-seams --reference-check --evidence-dir /tmp/t53host/evidence &
./target/debug/sparkflash device-chain --run-stage tp4-host \
  --tp-rank 1 --tp-world-size 2 \
  --comm-peers 127.0.0.1,127.0.0.1 --comm-base-port 24111 \
  --fixture /tmp/t53host/fixture --prompt-tokens 3,7,11 --max-tokens 3 \
  --evidence-dir /tmp/t53host/evidence &
wait
```

Measured (debug build, dev host, 5 forward steps = 3 prompt + 2 decode):

| quantity | value |
| --- | --- |
| generated tokens, both ranks | `[6, 6, 6]` (all 5 step tokens 6) |
| reference tokens (single-rank `prefill` of the same prefixes) | `[6, 6, 6, 6, 6]` - **equal** |
| max logits relative L2 vs reference | **1.28e-7** (bar 2.0e-2) |
| max routed-seam relative L2 vs unsharded f32 routed block | **2.75e-3** (engine-dev bar 5.0e-3) |
| seam comparisons | 200 (40 layers x 5 steps) |
| all-reduces | 200, 12,800 elements (40 x 5 x dim 64) |
| shard plan | 960 expert tensors, hash `0x6831bd20a2ecba3d` |
| resident estimate | 13.75 MB (tiny fixture) |
| disk bytes (`read_bytes`) | 0 (warm page cache; `rchar` 88 MB of reads) |
| wall | 6.4 s (load 56 ms, 3.75 s run) |

The tokens are all `6` because the tiny finite fixture's argmax saturates; that does not weaken the
equality proof (the per-step logits match to 1.3e-7) and it is the same fixture the engine's own
hybrid tests use. In-tree tests run the same comparison at world 2 and world 4
(`cargo test -p sparkflash-cli tp4_host::`, **8 tests**, about 7 s), including the resident-cache
accounting test.

## 5. The four-node runs, cold (runs 1 and 2)

The same binary, `--checkpoint` instead of `--fixture`:

```
sparkflash device-chain --run-stage tp4-host \
  --tp-rank R --tp-world-size 4 \
  --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --comm-base-port 29601 \
  --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --prompt "The capital of France is" --tokenizer PATH/tokenizer.json \
  --max-tokens 8 --evidence-dir DIR
```

Rank R binds `29601 + R`; every rank reads its own node's local checkpoint. One ssh per rank, all
four stopped at the end. Evidence per rank is `tp4-host-rR.json` plus `tp4-host-seams-r0.json`
when seams were recorded, and `tp4-host-summary.json` from rank 0; the launcher collects the
per-rank files from every node.

| run | prompt | tokens | text | wall | mean step | rchar/token | disk/token (rank 0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `The capital of France is` (8 tokens) | `[11111,16,455,5185,294,11111,344,223]` | " Paris. The population of Paris is " | 202.6 s | 16.85 s | 9,649,668,893 B | 361,638,229 B |
| 2 | `Question: What is the capital of France?` + newline + `Answer:` (11 tokens) | `[11111,201,8197,14,270,3417,344,28]` | " Paris\nNow, the question is:" | 285.0 s | 15.81 s | 9,650,133,557 B | 292,110,791 B |

Both runs are **cold**: `load_ms` 216 / 233 ms (plan and headers only, no weights read), and each
token re-reads the replicated set (about 9.65 GB/token/rank, mostly page-cache hits but still
`read()` copies), plus 13.5-20.3 GB of plan-slice reads across the run. That is the problem
task-57 fixes.

## 6. The resident run (task-57, run 3)

Same command and prompt as run 2, plus `--resident-cache`. At startup each rank reads its
non-Engram working set **once** (plan slices via `ShardPlan::read`, replicated names via the index,
Engram tables excluded) into one `ResidentCache`, an `Arc` shared by both weight views (the
engine's `WeightSource` and the MoE backend's local view) so the set is held once, not twice.
Filling is 84.699 GiB per rank; Engram tables stay on NVMe and are paged by row range through the
same source's `read_ranges`.

Command (binary sha256 `32ffec4492d19630f858d42bf1efeae209d3e760860c8e1b401712d8f3914b04`):

```
sparkflash.resident device-chain --run-stage tp4-host \
  --tp-rank R --tp-world-size 4 \
  --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --comm-base-port 29601 \
  --checkpoint <snapshot> --prompt $'Question: What is the capital of France?\nAnswer:' \
  --tokenizer <snapshot>/tokenizer.json --max-tokens 8 --resident-cache \
  --evidence-dir <node>/evidence-run3
```

| acceptance item | before (run 2) | after (run 3, rank 0) | verdict |
| --- | --- | --- | --- |
| token sequence | `[11111,201,8197,14,270,3417,344,28]` | **identical** | PASS |
| completion text | " Paris\nNow, the question is:" | **identical** | PASS |
| `rchar_per_token` | 9,650,133,557 B | **361,700 B** | PASS, 26,680x lower (bar was 100x) |
| `read_bytes_disk_per_token` | 292,110,791 B | **418,474 B** | PASS, 698x lower, Engram-row scale |
| `resident_cache.bytes` | - (resident estimate only) | 90,944,617,800 B = **84.699 GiB** | matches the header-walk estimate exactly |
| cache fill | - | 233-258 s (`fill_ms` 233,897-257,552) | one cold pass |
| RSS at Ready / VmHWM | about 1.0-1.1 GiB | **85.835 GiB / 90.302 GiB** | within the 117-118 GB nodes, below the 95-105 GiB forecast |
| mean step | 15,808 ms | **13,948 ms** | 11.8% faster wall; see compute-bound conclusion |
| decode all-reduces | 720 | 720, same 3,686,400 elements | unchanged |
| `shard_read_calls` during forward | 25,920 | **0** (all 27,360 plan reads are cache hits) | PASS |

Per rank: mean step 13,948 / 14,017 / 15,276 / 14,411 ms; measured collective 223 / 409 / 1,694 / 719
ms/step (rank 2's mean is pulled up by one 37.5 s step and a transport tail; the other three ranks
stay within 14.0-14.4 s). Per-position step microseconds are in `timing.step_ms`; `breakdown`
splits a step into collective (measured around the canonical reduce) and compute (the remainder).

### Cache state, stated explicitly

Run 3 is **page-cache-cold for the resident fill** (first touch after the runs 1/2 re-read sweeps and
with 85 GiB of anonymous memory allocated), but **weights are in-process** after the fill: the plan
and replicated bytes are owned by the process, not by the page cache. The Engram tables are not
resident and are still read per token, which is why `read_bytes_disk_per_token` is 0.42 MB and not
literally zero. The 9.65 GB/token of runs 1/2 was the replicated set being re-read and re-decoded
from the page cache; run 3 removes both the re-read and the re-decode of the resident classes
(`cache_hits` 27,360 = 18 steps x 1,520 plan reads).

### The conclusion Lead asked for, plainly

**Residency removed the I/O; the step is now COMPUTE-bound on the host reference engine.** After the
cache, rchar/token is 0.36 MB (Engram rows only) and compute is 13,724 ms of the 13,948 ms step
(collective 223 ms = 1.6%). The remaining time is host matmul plus, for the non-cached classes,
weight decode: about 13.7 s of CPU arithmetic per token across 40 layers, single-threaded per layer.
The TP=4 host path is correct and coherent but it is **not fast**, and the next speed lever is the
device path (M4/M5), not more residency — with one host-side exception recorded in the next
subsection, which halves the step before any device work. Run 3 (14.0 s/step, weights in-process) versus run 2
(15.8 s/step, re-read from page cache) shows residency buys only about 12% wall: the cost was never
just the `read()` bytes, it was also CPU decode of every class each token, and only residency can
remove the reads while the compute stays. No overclaim: this is not vLLM TP=4 and not the device
path.

### The allocator correction (batch 15, kernel-dev's measurement)

The "compute-bound" sentence above needs one measured correction, not a retraction. Re-raising the
glibc large-allocation thresholds at process startup (M_MMAP_THRESHOLD and M_TRIM_THRESHOLD, both
asked for 1 GiB) moves the per-layer attention output-projection buffers - wo_a 8192x4096 f32 (134 MB)
and wo_b 5120x8192 f32 (168 MB), allocated and then freed once per layer - from mmap/munmap (page
map plus zero fill per layer) onto retained heap. On the committed-binary four-rank resident run
(same checkpoint, tokens, ports and evidence schema) the clean step goes 13.827 -> 7.929 s, the
attention block 7.942 -> 3.251 s, the shared expert 2.017 -> 1.136 s and the reduce 0.340 -> 0.164 s;
tokens stay [11111, 201] (" Paris"), rchar/token is unchanged (25,114 -> 25,118 B, so no re-read was
introduced), and rank 0 VmHWM rises 90.265 -> 90.462 GiB (the retained heap).

The pair is the mechanism, measured rather than inferred from the option labels: attention per layer
is 214-219 ms untuned, 223-248 ms with the mmap raise only, 221-245 ms with the trim raise only, and
82-94 ms with both; a 200 MB allocate-touch-free probe keeps ~205 MB of RSS after free only with
both. Asking for both at 32 MiB behaves like untuned (211-224 ms), which falsifies the idea that
glibc silently clamps the 1 GiB request to 32 MiB on this host: the effective values really are the
raised pair, and the 32 MiB row is the falsification, not a caveat. Attribution: kernel-dev's branch
task/allocator-tuning; source and the four-rank A/B live in docs/build/allocator-tuning.md, and the
allocator_tuning header these evidence files now carry is written from
sparkflash_cli::allocator::tune so a run is self-describing. What remains after the pair is CPU
arithmetic, so the device path is still the next lever; the corrected conclusion is "residency
removed the I/O, then the allocator pair removed a large share of the per-layer allocation churn",
not "the host step was irreducible compute".

### Why step time was 14 s, not the 8-9 s a microbench suggests

The `breakdown` evidence (per-step `rchar`, `step_collective_ms`) plus kernel-dev's planned
per-layer/per-class profile (task-57 follow-up: the `--profile` hook lands in the same commit as
the Tp4Rank extraction) answer this. The candidate accounts: (a) the routed FFN runs over the 96
selected experts per layer in host f32/int8 decode, (b) every replicated class (10.738 GiB) is
decoded once per token, (c) 40 collectives per token with a millisecond-scale TCP tail, (d)
single-threaded layer execution (300-325% process CPU observed = about 3 cores, so some intra-layer
overlap exists but the critical path is serial). The microbench is per-class arithmetic; the
engine's per-token decode of every class plus collective serialization is the gap.

## 7. Library surface: Tp4Rank (task-55 hand-off)

The CLI stage is a thin driver over a public per-rank type in the same module, so serve (task-55)
does not fork the lockstep loop. The surface (extraction commit 47f8ccb, all in
crates/sparkflash-cli/src/tp4_host.rs):

- Tp4RankConfig { rank, world_size, peers, base_port, checkpoint, element_limit, resident_cache, timeout }
- Tp4Rank::open(&Tp4RankConfig) -> Result<Tp4Rank, Tp4Error>: opens the checkpoint, builds the shard
  plan and the one resident cache, connects the collective.
- begin_request(Tp4Sampling { temperature, seed }) -> Result<(), Tp4Error>: greedy only (a non-zero
  temperature is refused with Tp4Error), resets SequenceState and the request counters while keeping
  the engine and the resident cache.
- set_sampling(Tp4Sampling), sampling(), rank(), world_size()
- prefill(&[u32]) -> Result<StepOutput, Tp4Error> (one step per prompt token, returns the LAST
  position only), decode(u32) -> Result<StepOutput, Tp4Error>, metrics() -> &Tp4Metrics
- StepOutput { token: Option<u32>, logits: Vec<f32> }; Tp4Metrics { load_ms, cache_fill_ms,
  resident_cache_bytes, resident_cache_tensors, rss_load_kib, rss_peak_kib, reduce_calls,
  reduce_elements, reduce_ms_total, shard_read_calls, shard_read_bytes, steps, last_step_ms,
  plan_hash }

No HTTP, tokenizer or control protocol lives in Tp4Rank: token ids are the contract and serve owns
the rank0 control channel (base_port + 100). A request of max_tokens N is prefill(prompt) plus
N - 1 decode calls, which is the same 18-step shape as the run-3 evidence. The engine-facing names
(prefill/decode) are kept so the task-59 swap to InferenceEngine::new_tp plus the engine's
ResidentSource is mechanical.

## 8. Profiling hook: --profile / SPARKFLASH_TP4_PROFILE=1

Opt-in, timing only; the tokens and the rest of the evidence are unchanged when it is off (fixture
run with and without: [6, 6] both times). It writes a profile object into tp4-host-rR.json:

- header: reference_threads (the SPARKFLASH_REFERENCE_THREADS value as seen by the process).
- steps[]: position, wall_us, user_us and sys_us (process CPU from /proc/self/stat deltas), rchar,
  read_bytes, collective_us, layers[].
- layers[] (40 per step): provider_us (host entry, includes the Engram forward when the layer has
  Engram rows), deferred_us (the part taken with a MidLayer handoff), post_us, ffn_local_us
  (moe::routed_forward over the local shard), ffn_reduce_us (canonical all-reduce), shared_expert_us,
  engine_host_ms (the engine's own per-layer progress timer) and unattributed_us = step wall minus
  the measured layer components.

Fixture smoke with the flag on (world 2): step 0 wall 543,894 us, collective 20,000 us, layer 0
ffn_local 6,894 us, ffn_reduce 14,086 us, shared 4,885 us, engine_host_ms 28, unattributed 517,155 us.
On the tiny fixture most of the step is engine-internal (attention/head/decode), which is the
boundary this hook exposes rather than hides. The released-checkpoint per-layer numbers are
kernel-dev's run; this document records the tool and the schema.

## 9. Boundaries and what this does not prove

- No driver is opened, no CUDA context is created, no GPU kernel runs, and vLLM is not used on the
  nodes; the stage is a CPU forward over the checkpoint's bytes. `device_execution: false` is
  written unconditionally.
- The layout is the Phase-1 intermediate above, **not** the design's full TP map and not vLLM's
  TP=4. Gate B (serving parity against vLLM TP=4) still needs the real layout and device
  collectives (tasks 34/35/36).
- The per-seam L2 numbers are recorded, not gated. The declared bars are engine-dev's: routed seam
  relative L2 <= 5.0e-3, end-to-end logits <= 2.0e-2 at world 2/4; the measured 2.75e-3 / 1.28e-7 are
  reported next to them.
- The resident cache is **CLI-side** tonight: one `Arc<ResidentCache>` shared by the engine's
  `WeightSource` and the MoE backend's local view, with Engram tables excluded and paged. The
  follow-up (task-59) moves the same cache into `sparkflash-engine`'s
  `parallel_tp::ResidentSource` and deletes the CLI cache; it is a mechanical swap because
  `new_tp` takes the rank's plain per-rank source and builds the resident set itself.
- The device-buffer collective path (task-35), the ring path for prefill-sized collectives
  (task-36) and serving orchestration (task-55) are out of scope here.

## 10. Parked tasks

task-23 (full TP map), task-27 (D30 padding + m4_abi quarantine), task-35 (RDMA one-shot) and
task-36 (prefill collectives) stay claimed by tp-dev but are **parked**: not worked on this
milestone.
