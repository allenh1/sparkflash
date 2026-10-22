
# DeepInfer as a reference: serving-level features and measurable speedups

Slice owner: serving-level features / measurable speedups (DSpark + confidence-scheduled MTP
verification, SWA bounded replay & YOCO prefill truncation, asynchronous scheduling,
batch-invariant inference, deterministic sampling, structured outputs, benchmark harness).
Reference: `/home/user/deepinfer` (public release; submodules populated, 761 files under `vendor/`).
Ours: `/home/user/deepseek-v4.1-flash-spark`, TP=4, 1 GPU/node (sm_121a), 40 backbone layers +
3 DSpark layers, `compress_ratios=[0,0, 2x18, 1x20, 0,0,0]`, `window_size=128`,
`kv_source_layer_ids=[2,8,14,20]`, `index_topk=512`.

Everything below is from reading code, not the README. Where something is unconfirmed I say "guess"
explicitly. No cluster/GPU jobs were run; no other files were touched.

---

## 0. Verdict in one table (ranked by expected value to us)

| # | Candidate | Where | Runs on sm_121a? | Effort (eng-days, our codebase) | Payoff decode | Payoff TTFT | Output-changing? |
|---|---|---|---|---|---|---|---|
| 1 | **DSpark draft head + exact greedy verification, fixed width** | `models/mtp/dflash.py`, `layers/tilelang_ops/fused_sample.py` | model portable, their kernels not (we write our own) | 20-30 | **+2 to +3x tok/s** (production: 3.29 accepted of 5) | none (drafts only run in decode) | **No, if** row-count invariance holds; else yes |
| 2 | **Row-count-invariance gate** (1-row vs 6-row logits) | new harness; shape of `tests/models/test_batch_invariant.py` | yes (pure test) | 2-4 | prerequisite for #1 (unlocks it) | none | n/a (it is the measurement) |
| 3 | **Decode perf curve profiler** (step-ms vs total_q, trimmed mean, cached) | `worker/model_runner.py:561-631`, `worker/decode_perf_cache.py` | yes | 2-3 | indirect: picks optimal verify width k, bounds the MTP ceiling | none | no |
| 4 | **YOCO truncation *planning rules*** (134/140, hit cap L-134, 128-granularity replay, drop consumer SWA) | `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-360`, `worker/cache_engine.py:93-133`, `doc/inference_deployment_qa.md` | yes (design rule) | 3-5 | none | 0 now; up to **~48% of prefill row-work** at 4K prompts *once prefill is row-bound* | no (structural argument, see 2.3) |
| 5 | **Counter-based deterministic sampling** (SplitMix64 over `(seed, position)`) | `layers/tilelang_ops/random_sampling.py:1-100` | yes, ~20 lines of Rust | 1-2 | none directly; makes MTP verification testable/replayable | none | no (new RNG baseline) |
| 6 | Structured outputs (xgrammar + per-step bitmask) | `deepinfer/structure_output/*` | logic yes, library no (xgrammar is C++/py) | 15-25 | none | none | **yes** (constrained decoding) |
| 7 | Async scheduling / ZOS | `csrc/engine/zos.cpp`, `engine/llm_engine.py:860-1014` | concept yes | 8-12 | <1% at 930 ms/step | <1% | no |
| 8 | Batch-invariant *flag* (`DEEPINFER_BATCH_INVARIANT`) | `worker/runtime_config.py:57-70` | **no** (deep_gemm / tile_kernels / FlashMLA) | n/a | negative (disables split-KV and fast reductions) | negative | no |

**Single highest-leverage item: #1.** It is the only thing in the repo that changes our 1.06 tok/s
by a factor rather than a percentage, and the repo hands us the full mechanical recipe (a 5-token
draft in *one* forward over noise-token slots, a Markov correction at sampling time, and an exact
accept rule). Everything else is second-order for us.

---

## 1. DSpark / MTP draft + verification

### 1.1 What the draft model actually is (not "5 sequential MTP steps")

`deepinfer/model_executor/models/mtp/dflash.py`:

- 3 extra layers (`num_nextn_predict_layers=3`, ids 40/41/42: `dflash.py:99`,
  `config.py::_build_attn_module_list`), 128 routed experts / top-3
  (`dspark_n_routed_experts` / `dspark_num_experts_per_tok` in `reference/hf_config.json`).
- **The draft pass is one forward over a fixed block of 5 slots**, not five passes:
  `_embed_draft_init_tokens` (`dflash.py:230-239`) fills slot 0 with the last golden token and slots
  1..4 with `dspark_noise_token_id` (128799); `forward_drafts` (`:241-269`) runs the 3 layers once
  over those `req_cnt * 5` rows.
- **The dependency between draft tokens is injected at sampling time by a Markov head**:
  `sample_drafts` (`:279-345`) loops `step_i` 0..4 and at each step adds
  `markov_w2(markov_w1(draft_token_ids))` to the logits, where `draft_token_ids` was updated with the
  token sampled at the previous step. So 5 draft tokens cost one draft forward + 5 cheap sampling
  steps, not 5 forwards.
- A confidence head (`ConfidenceHead`, `:46-79`) emits a per-position acceptance probability from
  `[hidden; markov_emb]`.
- **The draft attention KV is written by the main model, not by the draft tokens.**
  `DFlash.update_main_cache` (`:198-228`) runs inside the main forward (`dflash_input` captured at
  layers `dspark_target_layer_ids=[37,38,39]`, projected by a 3x5120 -> 5120 `ReplicatedLinear` plus
  RMSNorm into per-layer `Wkv` K) and stores into the same SWA ring. There is no draft prefill and no
  draft context pass. This is the main reason DSpark is affordable: drafting is 3 small MoE layers
  over 5 rows with no context-length term.
- Draft KV shares the **SWA ring**: window 128 + 6 spare = 134 (`cache_engine.py:93`,
  `get_inference_swa_block_size = window + MAX_BACKTRACKING_TOKENS`; the deployment doc calls the 6
  spare "absorbed by the 6 extra positions the slot's ring buffer has beyond the window"). Rejected
  draft KV is not rolled back: the next draft write overwrites those slots before any read
  (`dflash.py:204-209`). That maps directly onto our ring-buffer SWA cache.

### 1.2 The verification rule is exact, and collapses to argmax-equality for greedy

`layers/tilelang_ops/fused_sample.py` (`fused_sample_ref`, lines 219-250) is the readable spec:

```
residual   = relu(sample_probs - exp(draft_logprob - DRAFT_PROB_PENALTY))   # q - alpha*p
accept_lp  = clamp(log q[draft] - log p[draft], max=0)
U          = splitmix64(seed ^ hash(3*position + 2)) in [0,1)
accepted   = (exp(accept_lp) > U) and draft_id >= 0
out        = accepted ? draft_id : sample_from(residual)
```

with `DRAFT_PROB_PENALTY = 5e-5` (`tilelang_ops/common.py`), present only so the residual is not
degenerate; `common.py` carries the proof that the emitted distribution stays within `(1+eps)` of the
target, `eps = 1 - exp(-5e-5)`.

For **greedy** (our case) `q` is one-hot at the target argmax and `p[draft]=1`, so `accept_lp = 0` iff
`draft == argmax(target)` and `-inf` otherwise. **We should not port the probabilistic machinery at
all**; the exact greedy rule is:

> accept draft token j iff `draft_j == argmax(logits_row_j)`; stop at the first mismatch; emit that
> argmax as the next token. Emitted tokens per step = accepted + 1.

Output-identical to plain greedy by construction, *provided* `logits_row_j` in a (1+k)-row forward
equals the 1-row forward's logits for the same position — see 1.6.

### 1.3 Variable-length scheduling (`DEEPINFER_ENABLE_MTP_VARLEN`) — real, but not for us

`csrc/engine/mtp_varlen.hpp:133-267` (`MTPVarlen::schedule`) is the README's "confidence-scheduled
verification". Mechanically:

1. Each request carries `accept_rates[j] = prod_{k<=j} confidence[k]` (cumulative product, computed in
   `process_MTPInfo`, `scheduler.hpp:1196-1262`) — the estimated probability that draft j is reached
   *and* accepted.
2. All (request, position) pairs are sorted by rate descending; a running
   `cur_expected = num_seqs + sum(rates)` and running `total_q` are walked, and at each step
   `throughput = cur_expected / (baseline_ms(graph_size) + slope_ms*graph_size + sampler_ms)` is
   evaluated against a **measured decode perf curve**; the `total_q` with the best throughput wins.
3. The budget is distributed to requests by a second sorted queue giving per-request
   `seqlens_q in [1, 1+nextn]`. With no perf curve it degenerates to fixed width
   (`:153-158`: `1 + draft_tokens.size()`).
4. Under asynchronous scheduling the CPU cannot see step N's confidence, so it fixes only `total_q`;
   the per-request split is recomputed **on the GPU** (`tilelang_ops/schedule_mtp_varlen.py:12-30`,
   `compute_seqlens_q_by_confidence_rates`: cumprod -> global top-k by rate -> fused remap scatter).

**Why we must not build this.** Their own production table (`doc/inference_deployment_qa.md`,
"Speculative decoding"):

| conc (seqs/rank) | 16 | 32 | 48 | 60 | 72 | 96 |
|---|---|---|---|---|---|---|
| dynamic vs fixed-5 gain | **-1.5%** | +1.6% | +3.6% | +5.4% | +6.7% | +11.0% |

At concurrency 1 — our entire deployment — dynamic-length verification is pure overhead. It exists
to stop one request's low-probability drafts stealing verification slots from another request; with
one request there is nobody to steal from. **Use fixed width k and pick k offline from our own step
curve (#3).** The same table gives the number that matters: **accepted length 3.29 per step from 5
drafts** (3.20 with dynamic). That is the whole business case for #1.

### 1.4 KV / block bookkeeping we must copy (easy to get wrong)

- `Sequence::get_num_blocks()` includes `get_nextn_len()` (`csrc/engine/sequence.hpp:292`): the block
  table must be **reserved for unverified drafts**; the GPU may grow `seqlen_q` to `nextn+1`
  (`zos.cpp:220-226`).
- Decode advances the KV length optimistically by `seqlen_q` and then **commits only the accepted
  count**: `advance_optimistically` (`scheduler.hpp:927`) then `commit_processed_tokens(token_cnt)`
  (`scheduler.hpp:997-1000`). Rejected KV is logically truncated by the committed length and
  physically overwritten next step.
- The MTP info pool is cross-step write-then-read: step N's draft sampler writes
  `seq.mtp.info_slot`; step N+1's verification reads it via `mtp_info_read_rows`
  (`prepare_sample.cpp:209-226`). The **final row reads slot 0** (padding) so the last position is a
  fresh sample, never a rejection test; `info_slot == 0` is reserved for padding
  (`scheduler.hpp:792-796`).
- `position_end = seqlen_k - seqlen_q + max_seqlen_q + nextn` (`scheduler.hpp:700`): RoPE /
  position-indexed buffers must be sized for the *optimistic* position, not the committed one. We
  will hit this — we write the last accepted+1 position speculatively.

### 1.5 Mapping onto our engine, and the honest payoff

Change shape in `sparkflash-engine`:

1. New stage `draft`: 3 layers, 128 experts top-3, hidden 5120, inter 2304, input = captured hidden
   of layers 37/38/39 (we already materialise per-layer activations), 5 rows/request. Draft attention
   reads the main SWA ring (window 128) with the same slot mapping plus intra-block rows.
2. Verify: run the 40-layer chain on `1+k` rows (k=5 default), each row attending the prefix plus the
   preceding draft rows (causal inside the block); argmax per row; accept the longest matching
   prefix.
3. Commit: append accepted + 1, truncate ring slots, free blocks for rejected rows.

Cost side, back-of-envelope: main chain is 40 layers of ~6.8 GB FP4 experts per node shard; the draft
is 3 layers x 128 experts ~ 7 GB total (about 2.7% of the backbone). At 930 ms/step we are
bandwidth/latency-bound, not row-bound, so `1+k` rows should cost a few percent rather than k times.
Expected `(1+a) / (1 + draft_overhead)` tokens per step time. With their measured a = 2.29 and 8-10%
overhead: **~3.0-3.2 tok/s**. With a pessimistic a = 1.5: **~2.3 tok/s**. Plan on **2.5x, range
2.2-3.1x**; do not plan on 4x.

Effort: 20-30 eng-days for the draft head + verification + KV/commit path, with two unknowns:
(a) tapping layers 37-39 + the 3x5120 projection inside our 40-layer TP chain without perturbing
backbone bit-identity (read-only tap, so it should be free), and (b) sharding 3 DSpark MoE layers
across our 4 nodes.

### 1.6 Risk: exactness hinges on **row-count invariance**, which the repo cannot give us

Their `DEEPINFER_BATCH_INVARIANT=1` (`worker/runtime_config.py:57-70`) is exactly a row-count
invariance switch, implemented by turning *off* fast paths: `deep_gemm.use_deterministic_algorithms(True)`,
`tile_kernels.use_deterministic_algorithms(True)`, FlashMLA `BATCH_INVARIANT` = split-KV disabled
(`vendor/FlashMLA/csrc/api/sparse_decode.cpp:223`), and one fused norm+rope path disabled
(`v4_modules/attn.py:86-91`). Both libraries are NVIDIA-only, which is why the README says
"NVIDIA only".

Two observations that matter:

- Their decode attention is *already* partition-independent at our topk:
  `enable_split_kv = !batch_invariant && !(topk + extra_topk <= 640)` (`sparse_decode.cpp:223`) and
  our `index_topk = 512`, so split-KV is off by default there.
- We have no split-KV and no NVLink-tree reductions, but we do have a per-layer bf16 all-reduce and
  hand-written MXFP8 GEMMs whose tile/grid shapes change with row count. Our own batch-17 fix is
  evidence that row count has bitten us before.

So the gate must be empirical and *required*:

1. **Offline row-count test**: fixed prefix of length L; run the chain with 1 row at position L and
   with 6 rows at L..L+5 fed the *golden* tokens; compare row 0's logits bit-for-bit, then row 1's,
   etc. Repeat for L in {37, 129, 517, 4097} and >= 3 prompts. Any non-zero diff makes MTP an
   explicitly-labelled behaviour change (and means the batch-17 class of bug is still live).
2. **End-to-end token equality**: greedy, MTP off vs on, >= 2000 tokens over >= 8 prompts, zero
   divergence. Report `avg_verified_len`, per-position accept rates, and the divergence position
   (`stats_logger.py:300-312` is the metric set to copy: `avg_nextn_accept_rate`,
   `nextn_accept_rate [r0..r4]`, `avg_verified_len`, `avg_confidence_error`).
3. If (1) fails, fix the offending kernel — or pad every forward to a fixed row count (a 6-row
   floor is cheap for us and is the pragmatic fallback).

---

## 2. YOCO / SWA bounded replay and "keep 140 tokens"

### 2.1 What our checkpoint actually is

`_n_prefill_layer(cfg)` = **the first layer index whose `compress_ratio == 1`**
(`hugging_face/config.py:158-161`); a layer is a "consumer" iff
`n_prefill_layer <= i < num_hidden_layers` (`v4_modules/yoco.py:19-21`). For our `compress_ratios`
that is **n_prefill_layer = 20**: layers 0-19 producers, layers 20-39 consumers. Two independent
confirmations: the flops estimator splits `producer = n_prefill_layer * per_row` vs
`consumer = (n_layer - n_prefill_layer) * per_row` (`flops_estimator.hpp:83-121`), and the deployment
doc says *"For V4.1, the stored layers are the first 20."*

**So exactly half of our model can run its prefill rows on only the last keep_tokens positions.**
That is the entire content of `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`.

### 2.2 Mechanism, exactly

`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:203-346`:

- `clip_prefix = clamp(data_len - keep_tokens - q_start, 0, q_len - 1)` — rows are dropped from the
  **front** of the chunk only; a fully clipped chunk still keeps its last row (`:225`).
- Compact layout: `yoco_consumer_query_indices` (full -> compact gather), `cu_seqlens_q`, `positions`,
  plus compact copies of `ks`/`ke` per sparse ratio and `kv_ca_index_offset_len` per dense ratio
  (`:291-332`).
- **Compressed-row key ranges are recomputed from absolute positions, not compact indices**:
  `ke = base + (q_start + j + 1) / ratio` with `base = seqlen_k / ratio` per sequence (`:300`,
  `:313-316`). This is the single most important line for "correct handling of topk / compressed
  rows": get it wrong and compact rows attend the wrong compressed keys, which looks like small
  numeric drift rather than a bug.
- SWA visibility is the **intersection** of window and keep rectangle:
  `abs_lo = clamp(max(pos - window + 1, data_len - keep, valid_start), 0, pos + 1)` (`:155`).
- Ring writes only for rows (a) inside the rectangle and (b) in the last `swa_block_size` rows of the
  chunk; everything else gets `slot_mapping_store_swa = -1` (`:266-271`).
- `mask` mode computes identical visibility but keeps all rows: a numerics-alignment mode with
  **no** speedup (`prepare_sample.cpp:51-53`).

### 2.3 Why it can be exact (and the precondition we must satisfy)

The argument is *not* "140 > 128 so nothing is lost"; it is structural, in `v4_modules/attn.py` +
`decoder_layer.py`:

1. At the compaction boundary (first consumer) `compact_yoco_boundary` produces the compact copies
   **and keeps the full hidden** (`yoco.py:93-123`, `decoder_layer.py:134-142`).
2. `_store_yoco_full_kv` (`attn.py:672-720`) writes **main KV + indexer KV for all positions, in the
   full layout**, from that full hidden, *before* any compaction.
3. Consumer layers then run on compact rows with `store_kv = not yoco_full_kv_stored`
   (`attn.py:537-587`): they **skip all store/compression** and only compute the compact query's
   top-k. Derived per-query caches are invalidated rather than gathered (`candidate_mqa_lens = None`,
   schedule metadata cleared, `yoco.py:116-122`).
4. Consumer `kv_swa` is **projected from one shared hidden** captured at the boundary
   (`yoco_swa_share_hidden`, default True: `attn.py:249`, `:289-292`, `:389-397`;
   `shared_hidden_kv_proj` for kv-source *and* index-source layers, `:319-380`). So a consumer
   layer's KV at position p depends on the *producer* output at p, never on a truncated query's
   attention result.

Consequence: only the consumer layers' own output hidden states in the first `keep - window` rows of
the rectangle are computed with a narrowed window, and those hidden states are used only for their
own (discarded) rows. Every byte of stored KV, and the last row's logits, match full prefill.

**Precondition for us**: our consumer layers must derive CSA/SWA KV from a single boundary (owner)
hidden, not from each layer's own evolving hidden. Our pointer-to-owner-cache emulation is the right
shape, but the owner must be the **boundary layer's attn-norm hidden** (layer 20), published once in
full layout, with consumer layer *stores* disabled. If any consumer layer in our engine stores KV
computed from its own hidden, compact truncation is **not** exact. Testable: compare stored KV bytes
at positions < L-140 after truncated vs full prefill. That test should exist before the feature.

### 2.4 The constants, and the trap inside `KEEP_TOKENS=140`

- ring rows = `window + MAX_BACKTRACKING_TOKENS` = 128 + 6 = **134** (`cache_engine.py:93`).
- `DEEPINFER_MAX_BACKTRACKING_TOKENS = max(value, USE_NEXTN_DECODE + 1)` (`environ.py:132`) — so
  `examples/benchmark.py:161` setting it to `"1"` is a **no-op whenever nextn > 0**, because the
  parser floors it at nextn+1. Copy that constant and you copy a value their own code overrides.
- hard check: `keep_tokens >= window_size + max_backtracking` (`prepare_tensor_input_v4.cpp:355`),
  i.e. 134; they use 140, leaving 6 rows of slack.
- prefix-cache rules that fall out (`doc/inference_deployment_qa.md`, "KV cache hits and
  recomputation"): max hit length for a prompt of length L is `L - 134`; hit length is determined by
  the block KV cache and then backed off by one 128-block if the SWA state is missing; SWA state is
  stored only at 128-token multiples; the compressor accumulators (`ca_state`/`ica_state`, sized
  `ratio + backtracking - 1`, `cache_engine.py:133`) are in the **dropped** region because
  compression windows align to the 128-token store granularity. **Direct implication: if we only ever
  replay from a 128-aligned boundary we do not need to persist `ca_state`/`ica_state` at all.** That
  removes a whole class of "partial compression window" state from our KV planner.

### 2.5 Payoff for us: zero today, large later — and not a TTFT lever yet

Row-work reduction from compact truncation, using their own cost model (`flops_estimator.hpp`:
producer rows = N, consumer rows = keep): `saved = 0.5 * (1 - 140/N)` of non-attention row-work, plus
a similar order for attention/indexer work:

| prompt N | 37 | 140 | 256 | 512 | 1024 | 2048 | 4096 | 32768 |
|---|---|---|---|---|---|---|---|---|
| row-work saved | 0% | 0% | 22.7% | 36.3% | 43.2% | 46.6% | 48.3% | 49.6% |

Our TTFT is 39-44 s for a **37-token** prompt at ~0.9 s/token: N < keep_tokens, `clip_prefix = 0`,
nothing is dropped, contribution exactly zero. Even at 4K tokens, "48% of rows" is 48% of *compute*,
and our per-token cost (0.9 s) is within 5% of our decode step (0.93 s) — the signature of an engine
bound by *passes over weights*, not by rows per pass. The reference says the same from the other
side: "nearly halving total prefill computation" — halving *compute*, in an engine whose prefill is
compute-bound at 4096-65536 tokens/rank.

Honest ordering: **chunked parallel prefill first (make prefill row-bound), then YOCO.** YOCO is a
multiplier on a fix we have not landed. What is worth doing now is cheap and non-speculative: adopt
134/140 and the `L-134` hit cap in the `sparkflash-kv` planner, mark layers >= 20 as droppable SWA
state (halves the state we persist and transfer), and record the "consumer stores disabled; KV
written once at the owner boundary in full layout" invariant in the CSA design.

### 2.6 Where the reference is weaker than it looks

`examples/yoco/yoco_equiv.py` + `run_yoco_equiv_l2.sh` compare **mask vs compact only**
(`_collect_mask_payload(args, False/True)`): they prove the two modes agree with each other, not that
either agrees with untruncated prefill. L1 compares SWA pivots / visible keys / store slots /
compressed key ranges; L2 compares logprobs element-wise on a teacher-forced trajectory (8 devices,
real checkpoint) and, with `--mtp N`, the confidence score bitwise. That is a good harness shape —
**copy the harness and add the missing leg**: `off` vs `compact` (and `off` vs `mask`) on the same
trajectory. Without it, equivalence rests on the structural argument in 2.3 alone.

---

## 3. Asynchronous scheduling (zero-overhead batch scheduler)

`csrc/engine/zos.cpp` + `engine/llm_engine.py:860-1014`. What it is: schedule(N+1) on the CPU while the
GPU runs forward(N)/sample(N); the CPU optimistically guesses `seqlens_q`, the GPU corrects the real
lengths, and sampler metadata is patched from pinned buffers (`update_metadata_for_sampling_zos`,
`zos.cpp:279-305`). `prepare_zos_inputs` also reuses the previous step's device tensors by `seq_id`
remap, re-uploading only "fresh" rows (`:142-196`). Its purpose, per the deployment doc, is that
without it the GPU idles during `post-process(N)` and `schedule(N+1)` — GPU idle, not CPU cost.

For us at 930 ms/step host work is ~1%. **Skip ZOS.** Two pieces still matter:
- the `advance_optimistically` / `commit_processed_tokens` bookkeeping (1.4) is the correct KV
  accounting for MTP regardless of asynchrony;
- `update_mtp_runtime(num_seqs, graph_size, forward_ms, sampler_ms)` (`scheduler.hpp:663-665`) names
  the measurement points: forward span and sampler span per step, from CUDA events, tagged with graph
  (row) size. Emit those three numbers in `sparkflash-trace` now; they are the input to every
  decision in this report.

Verdict: not worth porting (<1%); steal the instrumentation.

---

## 4. Batch-invariant inference and deterministic sampling

### 4.1 Batch invariance

"NVIDIA only" in the README, and the code says why: `configure_worker_kernels` sets
`deep_gemm.use_deterministic_algorithms(True)` and `tile_kernels.use_deterministic_algorithms(True)`
under `DEEPINFER_BATCH_INVARIANT` (`worker/runtime_config.py:57-70`), FlashMLA drops split-KV, and
the fused norm+rope+cast path is disabled. So in their world "batch invariant" means "give up the
fast nondeterministic reductions". The test (`tests/models/test_batch_invariant.py`) is the useful
artifact: generate 64 sequences of random length batched, then re-run each one *singly* with
`golden_token_ids` forced to the batched output and require tokens **and** logprobs **and** top-5
ids/logprobs to match. That is the harness we need for MTP (1.6), pointed at row count instead of
batch size.

The flag is not applicable to us, but **the property is a hard requirement** for MTP exactness and is
cheap to test. Note that "batch invariance" in their sense (batch composition) is weaker than what
MTP needs (row count with *different* causal prefixes); the MTP test must force draft tokens into the
extra rows, not merely vary the batch.

### 4.2 Deterministic sampling — a small, portable, high-value idea

`layers/tilelang_ops/random_sampling.py:1-100`: sampling is **counter-based**, not stream-based. The
uniform is `splitmix64(seq_seed ^ hash(3*sample_position + case))` with `case=0` for target sampling
and `case=1` for the residual draw, then a fp64 CDF inversion. Documented property: *"Deterministic in
(seq_seeds, sample_positions, is_nextn_predict): independent of batch size and row order;
zero-probability columns are never sampled."*

Why it matters to us even without speculative decoding: sampling becomes a **pure function of
(seed, position)**, so replay/verification can re-derive any token without replaying an RNG stream;
our bit-identical claims stop depending on batch composition (exactly the batch-17 class of bug); and
MTP tests become reproducible by construction (the `case` split keeps draft and residual draws on
independent streams). Cost in Rust: ~20 lines plus the test vectors from
`test_random_sampling_batch_invariant`. Caveat: switching RNG changes our output stream, so it is a
new baseline and must be labelled a behaviour change with re-frozen fixtures.

---

## 5. Structured outputs

`deepinfer/structure_output/{compiler,matcher,so_matcher_builder,mask}.py`: xgrammar compiles
regex / JSON schema / EBNF to an automaton asynchronously on a `ThreadPoolExecutor` before the request
enters the engine (`so_matcher_builder.py:15-60`); each decode step advances the matcher, fills a
`[bsz, vocab]` bitmask (`allocate_vocab_mask`, pinned, double-buffered on a side stream,
`sampler.py:301-408`) and applies it per verification step index. Details worth recording: the matcher
is advanced for **accepted draft tokens** and rolled back (`matcher.rollback(num_accepted_steps)`,
`sampler.py:366-376`); masks are prepared per step so verification of token j uses the mask valid
after j-1 accepts; and a request that cannot be recomputed after preemption is killed rather than
silently freed (`scheduler.hpp:769-780`).

Verdict: the architecture (async compile, per-step masks, per-draft-step masks, rollback on reject)
is worth writing down for later; the library is not portable (xgrammar is C++/Python). A Rust-native
equivalent exists (llguidance) but is outside this repo — flag as a guess on effort. Priority: after
MTP and prefill. Output-changing by definition, so it can never be part of a bit-identical claim.

---

## 6. Benchmark harness and the numbers it produces

- `examples/benchmark.py`: prefills a synthetic random-token corpus (`--context_length`,
  `--non_cached_tokens`), forks one cached prefix to build the decode batch
  (`fork_request_batch`, so all decode requests share an identical KV length), then measures
  **steady-state windows**: prefill 5 warmup + 10 measure; decode 50 warmup + 60 measure with two
  boundary-aligned windows and outlier trimming; `PROFILE=0` for sweep points because the profiler
  perturbs the reading. `--result_jsonl` appends one raw record per point.
- **Decode throughput is `avg_SPS * bs * (1+nextn) / cp`** with the comment *"VARLEN=0 下每 step 恒提交
  1+nextn 个 token，故与接受率无关"* (`benchmark.py:405-408`), and `plot_pareto.py:58-65` rescales both
  axes by `tau = 1 + nextn * draft_acceptance` where **`draft_acceptance` is a CLI argument, not a
  measurement** (`--draft_acceptance`, default 1.0). So the published decode Pareto is a fixed-5,
  full-acceptance *step-time* measurement re-scaled to a hypothetical acceptance rate: read it as a
  step-time curve, not as a speculative-decoding measurement. (This does not weaken the production
  table in the deployment doc, which reports measured accepted lengths.)
- `examples/sweep/run_sweep.sh` sweeps one dimension (decode: batch 8..256 at fixed context and nextn;
  prefill: 4096..65536 tokens/rank with `--yoco_clip` fixed at 140), one port per point to dodge
  TCPStore TIME_WAIT. Copy that shape: one knob at a time, steady-state windows, machine-readable
  rows, plot last.
- **`profile_decode_perf` (`worker/model_runner.py:561-631`) is the most directly useful artifact for
  us**: at startup it replays every fixed batch size, 1 warmup + 5 timed with CUDA events + trimmed
  mean, all-reduces the times across ranks, and caches `(total_q, ms)` to disk keyed by model+parallel
  config (`decode_perf_cache.py`). That curve is what `MTPVarlen` optimizes against and what tells us
  (a) whether we are row-bound or weight-bound, (b) the best fixed k, (c) the ceiling for both MTP
  and YOCO. Build it as `sparkflash-cli bench decode-perf-curve` *before* implementing MTP; without
  it we cannot honestly choose k or predict the gain.

---

## 7. Things that CONTRADICT / constrain our current design

1. **The confidence scheduler is a wide-EP feature, not part of "speculative decoding" for us.** Their
   own table: **-1.5% at 16 seqs/rank**, only positive at >= 48. If the plan was "implement DSpark
   including confidence scheduling", the plan is wrong at TP=4 / concurrency 1: implement the draft
   head + fixed-width verification, and skip `MTPVarlen`, `SamplingRuntimeEstimator`,
   `ForwardRuntimeEstimator`, the perf-curve plumbing and the GPU-side
   `compute_seqlens_q_by_confidence_rates` entirely.
2. **MTP cannot improve TTFT.** Drafts are produced only in decode steps
   (`prepare_sample.cpp:49` `nextn_token_cnt = is_prompt ? 0 : ...`; `sampler.py:155`
   `max_MTP_value > 0 and not is_prompt`; `process_MTPInfo` publishes draft info only for non-prompt
   runs). The first decode step emits exactly 1 token. Any plan expecting speculation to cut
   first-token latency is wrong.
3. **YOCO truncation is not a TTFT lever for a 37-token prompt** (2.5): zero rows dropped, and our
   prefill is weight-pass-bound. If TTFT is the goal the levers are fewer passes over weights (real
   chunked prefill), fewer bytes, more nodes — not row truncation.
4. **Consumer-layer KV exactness rests on a shared boundary hidden, not on the keep window.** If our
   CSA pointer-to-owner-cache publishes per-layer KV computed from each layer's own hidden, YOCO
   truncation is inexact and "keep 140" does not transfer. The owner must be the boundary layer (20),
   writing in full layout, with consumer stores disabled. This is the same shape as the hole we just
   fixed: a publication that must be complete *before* the consumer layout changes.
5. **Compressed/topk rows must be re-derived from absolute positions after a layout change**
   (`ke = base + (q_start + j + 1)/ratio`, `prepare_tensor_input_v4.cpp:300`), and cached per-query
   derived state must be *invalidated*, not gathered (`yoco.py:116-122`). Any topk/candidate cache of
   ours that survives a row-layout change is suspect.
6. **`DEEPINFER_MAX_BACKTRACKING_TOKENS=1` in `examples/benchmark.py:161` is a no-op when nextn>0**
   (`environ.py:132` floors it at nextn+1). Do not copy "1" as a validated constant.
7. **Consumer SWA is explicitly droppable state** ("the dropped region holds the YOCO consumer SWA
   layers, whose recovery relies on prefill recomputation"). Persisting all 40 layers' SWA stores 2x
   what is useful and implies a reuse guarantee we cannot honour.
8. **Their "batch invariant" claim is a performance-regression switch on NVIDIA-only libraries**, not
   evidence that the model is batch-invariant, and not the property MTP needs. Treat any
   batch-invariance claim about our engine as unmeasured until the 1-row vs 6-row test passes.

---

## 8. Portable-code inventory (re-read when implementing)

| Purpose | File:lines |
|---|---|
| Draft block forward + Markov head + confidence | `deepinfer/model_executor/models/mtp/dflash.py:46-79, 230-345` |
| Draft KV write from target-layer features | `.../mtp/dflash.py:198-228`; `.../models/decoder_layer.py:208-238` |
| Draft metadata (fixed stride, ring slots, intra-block indices) | `.../input_metadata/dflash.py:36-79` |
| Exact accept / residual rule (readable reference) | `.../tilelang_ops/fused_sample.py:200-250` + `.../tilelang_ops/common.py` (proof) |
| MTP KV commit + accepted-prefix extraction | `csrc/engine/scheduler.hpp:872-1033`; `csrc/engine/sequence.hpp:118-166, 292-297` |
| Sampler input packing for verification | `csrc/engine/prepare_sample.cpp:40-284` |
| Confidence -> cumulative accept rates | `csrc/engine/scheduler.hpp:1196-1262` |
| YOCO truncation math (visibility, ks/ke, slots) | `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-360` |
| YOCO boundary compaction + cache invalidation | `.../v4_modules/yoco.py:55-123`; `.../models/decoder_layer.py:134-150` |
| Full-KV store once at the owner | `.../v4_modules/attn.py:525-587, 672-720` |
| Shared-hidden projections | `.../v4_modules/attn.py:249-292, 319-397` |
| Ring sizes / dropped layers / accumulator sizing | `deepinfer/worker/cache_engine.py:93-137` |
| Equivalence harness shape (L1 metadata / L2 logprob) | `examples/yoco/yoco_equiv.py`, `examples/yoco/run_yoco_equiv_l2.sh` |
| Batch-invariance harness shape | `tests/models/test_batch_invariant.py` |
| Counter-based sampling | `.../tilelang_ops/random_sampling.py:1-100` |
| Perf-curve profiler + cache format | `worker/model_runner.py:561-631`, `worker/decode_perf_cache.py` |
| Accept-rate metrics to emit | `deepinfer/engine/stats_logger.py:300-312` |
| Ops facts (ring 134, store at 128, hit cap L-134) | `doc/inference_deployment_qa.md` "KV cache hits and recomputation", "Speculative decoding" |

---

## 9. Recommended order of work (my opinion, given the above)

1. **Decode perf curve + step-span instrumentation** (2-3 d). Tells us row-bound vs weight-bound and
   picks k; everything else is guesswork without it.
2. **Row-count / batch-invariance gate** (2-4 d). Required for any claim that MTP is exact; also a
   regression test for the batch-17 class of bug.
3. **DSpark draft head + fixed-width greedy verification** (20-30 d). The only factor-level win. Do
   not port `MTPVarlen`.
4. **KV planner rules** (3-5 d): 134/140, hit cap `L-134`, 128-granularity replay, drop consumer SWA
   (layers >= 20), drop `ca_state` at aligned boundaries. No output change.
5. **Real chunked parallel prefill** (already in flight) — this is what makes #6 worth anything.
6. **YOCO compact truncation** (10-15 d *after* #5, gated on the boundary-shared-hidden invariant in
   2.3 and on an `off`-vs-`compact` equivalence run).
7. Counter-based sampling (1-2 d) when we next touch the sampler and can afford a new baseline.
8. Structured outputs, ZOS: defer; not justified at our concurrency and step time.

---

## 10. Confidence of these claims

- Verified by reading code: 1.1-1.4, 2.1-2.4, 3, 4, 5, 6.
- Verified by their documents (not their code): production accepted length 3.29/5, the
  dynamic-vs-fixed table, "stored layers are the first 20", "nearly halving total prefill
  computation", ring 134, 128-token store granularity.
- **Guess / unverified**: our step-time split (bandwidth vs row-bound) — needs #1; the achievable
  acceptance rate on our prompts (2.29 accepted is their traffic); the DSpark MoE effort estimate;
  llguidance as a Rust SO alternative (outside this repo); whether any current kernel of ours is
  already row-count invariant (batch-17 history suggests not, but it must be measured).
