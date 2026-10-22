# DeepInfer as a reference implementation: architecture & runtime survey

**Slice:** architecture and runtime survey (model definition, executor/worker layout, parallelism,
request flow prefill -> decode -> sampling) and what transfers to our TP=4 host twin.
**Repo:** `/home/user/deepinfer`, shallow clone at commit `774ce24` ("update Q&A EN doc").
**Vendor submodules are present** (not absent as the brief feared): `vendor/DeepEP` @ `93eb6eb`,
`vendor/DeepEP-Ascend` @ `3b25377`, `vendor/DeepGEMM` @ `1fc9c66`, `vendor/DeepGEMM-Ascend` @ `3a353262`,
`vendor/DeepSelect` @ `bfa4507`, `vendor/FlashMLA` @ `2e5429f`. 342 files / 17 MB; ~17.6 kLOC in the
python+c++ runtime, plus the kernels.
**Method:** read the runtime end to end (model modules, worker, engine, C++ scheduler, cache engine,
input-metadata builders), the HF->runtime config adapter, the sanitised production Q&A
(`doc/inference_deployment_qa.md`), and the kernel entry points. Everything below is cited to a file
and line. Where I infer rather than read, it is marked **[guess]** or **[inferred]**.

Nothing here was executed: no cluster or GPU job was run, and no file outside
`docs/investigations/deepinfer/` was touched.

---

## 0. TL;DR for the Lead

1. **DeepInfer has no tensor parallelism and no context parallelism in production.** "TP and CP are not
   used for now" (`doc/inference_deployment_qa.md:53`). Attention is **DP**, MoE is **EP** (DP==EP),
   decode is Wide-EP (EP64-128), prefill EP2-8. The CP code that exists (`parallel_utils/cp.py`,
   `prepare_cp_mirror.py`) is a mirror/zigzag context split that production does not enable.
   **Consequence: DeepInfer cannot validate our TP=4 attention weight sharding.** There is no
   reference implementation of "shard wq_b by head, wo_a by o_group" anywhere in it. Our sharding must
   keep being validated against the HF reference `model.py` and our own fixtures.
2. **The checkpoint's DAG is exactly what our `layer_plan.rs` already encodes** - 4 KV owners
   (`[2,8,14,20]`), 8 index sources (`[2,8,14,20,24,28,32,36]`), candidate pool at 20, 20 SWA rings,
   decoder half 20..39 - and I found the authoritative derivation code for all of it:
   `deepinfer/hugging_face/config.py:62-85` + `:158-161`. Two details we should adopt or re-check:
   `n_prefill_layer` is **derived** as "first `compress_ratios[i] == 1`" (not a config key, not a
   constant), and there is a **third, independent map** `layer_index_k_map` (the indexer *K* cache),
   validated as a refinement chain `index-topk subset index-K subset KV`
   (`v4_modules/cross_layer.py:40-48`).
3. **The shared-hidden / YOCO machinery is inert for this checkpoint.** `yoco_swa_share_hidden`
   defaults to **False** for V4.1-Flash (`hugging_face/config.py:47`), and no consumer layer is a KV
   source, so `yoco_shared_hidden_required()` returns False (`v4_modules/yoco.py:23-45`). Decoder-layer
   SWA KV is projected from **that layer's own hidden state**, not from the encoder hidden. Only the
   *compressed block KV* is shared, and only by **pointer alias** of the owner's cache
   (`worker/cache_engine.py:429-431`, `:446-461`). This confirms our D38/D39 fix direction and says
   we need no cross-layer hidden-state plumbing at all.
4. **DSpark/MTP is a parallel 5-token block drafted by 3 stages, not 5 sequential drafts**
   (`models/mtp/dflash.py:241-345`, config `dspark_block_size=5`, `num_nextn_predict_layers=3`).
   Production numbers: **accepted length 3.2-3.3 of 5 verified** (`doc/inference_deployment_qa.md:247`).
   That is the single largest transferable tok/s lever for us (~2.2-2.5x on paper), and it is also the
   most expensive thing on the list.
5. **Decoder-half prefill truncation is the only big TTFT lever in DeepInfer, and it is worth nothing
   at 37 tokens.** Cold-prompt truncation is exact for the kept suffix (window + backtracking <= keep),
   but our 37-token prompt is shorter than `KEEP_TOKENS=140`, so the optimum there is just "one
   varlen forward". Our 0.66 s/token chunked prefill therefore points at kernel/chunk-shape
   inefficiency, not at missing scheduler features (see section 7, item T5 and the diagnostic).

---

## 1. Repo map

```
deepinfer/
  hugging_face/config.py        HF config.json -> runtime AttrDict: topology maps, n_prefill_layer,
                                dflash_cfg, engram cfg, quant layout checks. THE authority on config.
  model_executor/
    models/deepseek.py          DeepseekForCausalLM: builds 40 layers + 3 DFlash stages, allocates
                                CrossLayerRuntime, forward = 1 loop over layers + MTP driver.
    models/decoder_layer.py     DeepseekDecoderLayer.forward: hc_pre -> attn -> hc_post -> engram ->
                                ffn -> yoco capture / compact at the CED boundary.
    models/v4_modules/
      attn.py                   V4AttentionBase/V4Attention (projections, KV store, sparse/dense/swa
                                paths), DFlashAttention. 1008 lines, the densest file in the repo.
      index_attn.py             MQA indexer: p_sum, candidate-block topk, index-K cache, topk publish.
      compress.py               Compressor (C1/C2 gated low-rank) + KProject (index-K from pre-RoPE KV).
      cross_layer.py            SharedLayerTopology (3 maps + candidate map + validation) and
                                CrossLayerRuntime (per-forward publish/consume slot).
      yoco.py                   consumer predicates, shared-hidden requirement, boundary compaction.
      mlp.py                    DeepseekMoE: gate, DeepEP dispatch/combine, mega-MoE, EPLB status.
      engram.py                 Engram: hash n-gram lookup over an engram-parallel group, fused KV
                                projection, gate into hidden.
      hc*.py / mhc_ops.py       mHC (hc_mult=4) residual-stream machinery.
      mla_attn.py               container-level MLA/indexer parameter trees + load_paged_kv_cache use.
    layers/sampler.py           sampling + speculative verify (accept prefix, residual resample) +
                                confidence stats. sampler_utils.py: logits/temperature/top-k.
    layers/tilelang_ops/        ~40 TileLang kernels (cuda/ and asc/ variants): fused norm+rope+cast+
                                store, compress, kv_index_cat, gather, candidate_blocks, sampler,
                                prepare_cp_mirror, schedule_mtp_varlen, structure_output, ...
  worker/
    model_runner.py             prepare_inputs/prepare_sample wrappers, graph capture, PPL profiling.
    worker.py                   execute_model_step: forward -> post_process(prev) -> sample -> defer.
    cache_engine.py             KV cache carving: per-layer byte offsets, ca/ica/swa/state caches,
                                share-kv alias by tensor identity, SWA rings, kvstore layout.
    cuda_graph_runner.py        captured batch sizes {1,2,4,..,interval,..,max}, padding seq metadata.
    dflash_cuda_graph_runner.py separate graph for the 3 DFlash stages.
    decode_perf_cache.py        (bs, ms) decode curve used by the confidence scheduler.
  engine/llm_engine.py          add_request/step loop, DeferredRelease, engine types P/C/PD.
  csrc/engine/
    scheduler.hpp               the real scheduler (1329 lines): DP vote prompt/decode, chunked
                                prefill with cp_align + sum-K budget + flops estimator, decode LCFS,
                                preemption, MTP varlen budget, prefix-cache branch ids.
    mtp_varlen.hpp              confidence -> (graph size, per-request seqlen_q) solver.
    flops_estimator.hpp         FP8-equivalent FLOP cost model per layer/row (mlp/router/hc/engram/attn).
    input_metadata/prepare_tensor_input_v4.cpp
                                all per-step index algebra: ks/ke, klen, kv_ca_index, SWA pivots,
                                YoCo truncation & compaction, block load plan. 655 lines.
    input_metadata/block_load_plan.cpp  coalesced (src_block, begin, end, dst_offset) segments.
    zos.cpp                     zero-overhead scheduling: step N input_ids from step N-1 drafts.
  csrc/engine/block_manager.hpp, sequence.hpp, prepare_sample.cpp, process_output.hpp
  deep_cluster/{hardware,topology}/   platform detection, bf16 tflops limits (used by the cost model).
  examples/offline_inference.py, benchmark.py, yoco/yoco_equiv.py, _config.py, _launch.py
  tests/models/test_batch_invariant.py, test_kl.py, test_eplb_policy.py, offline_decode_ppl.json
  vendor/{FlashMLA,DeepGEMM,DeepEP,DeepSelect}(+ -Ascend twins)
```

---

## 2. The model definition, as DeepInfer actually builds it

### 2.1 The checkpoint's real topology (from `reference/hf_config.json`, read by the adapter)

```
num_hidden_layers 40, hidden 5120, heads 64, head_dim 512, qk_rope 64, kv_lora 448,
q_lora_rank 1280, o_lora_rank 1024, o_groups 8, hc_mult 4 (hc_sinkhorn_iters 20),
vocab 129280, 384 routed experts + 1 shared, top-6, sqrtsoftplus, noaux_tc, scaling 1.5,
swiglu_limit 10.0, sliding_window 128, compress_rope_theta 160000, rope yarn 16x from 65536,
compress_ratios[43] = [0,0, 2 x18, 1 x20, 0,0,0]        (40 backbone + 3 MTP rows)
kv_source_layer_ids    = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20, candidate_topk_blocks 2048, candidate_block_size 8,
                                topk_block_score_reduce = "max"
index_n_heads 32, index_head_dim 128, index_topk 512
engram_layer_ids [1, 14], engram_max_ngram_size 4, n_heads 8, head_dim 256,
                                engram_num_embeddings ~384e6 each, compressed_vocab 99092
num_nextn_predict_layers 3, dspark_block_size 5, dspark_noise_token_id 128799,
dspark_target_layer_ids [37,38,39], dspark_markov_rank 256,
dspark MoE: 128 experts, top-3
```

### 2.2 How DeepInfer derives the runtime graph (the part worth copying)

`deepinfer/hugging_face/config.py:62-85` is the whole sharing topology in 20 lines:

```python
def _source_map(source_ids, n_layer):        # config.py:62
    ids = sorted(source_ids)
    return {layer: max(s for s in ids if s <= layer) for layer in range(ids[0], n_layer)}

layer_index_module_map = _source_map(index_source_layer_ids, n_layer)   # topk publish/consume
layer_kv_map           = _source_map(kv_source_layer_ids, n_layer)      # ca_cache alias
layer_index_k_map      = _source_map(text.get("index_k_source_layer_ids", kv_source_layer_ids), n_layer)
layer_candidate_source_map = {layer: 20 for layer in range(20, n_layer)}  # not sparse: no segment cut
```

`n_prefill_layer` (`config.py:158-161`):

```python
def _n_prefill_layer(text):
    """The first layer with ratio==1: it and everything before it form the prefill side,
       the layers after it are yoco consumer layers."""
    return next(i for i, ratio in enumerate(text["compress_ratios"]) if ratio == 1)
```

That is 20 for this checkpoint, which is why the production Q&A says "encoder layers 0..=20, decoder
layers 21..=39" (`doc/inference_deployment_qa.md:187`). **Our `sparkflash-kv/src/config.rs:49` hardcodes
`ENCODER_LAST_LAYER = 20`; the right rule is the one above (first ratio==1).** Same value today,
different behaviour for the next checkpoint - a 5-line change.

The validation in `v4_modules/cross_layer.py:36-48` is the part I would port as assertions, because
each clause is a silent-corruption guard:

- the three maps must have **nested segment granularity**: `index-topk subset index-K subset KV`
  (every index-topk segment lies inside one index-K segment, which lies inside one KV segment);
- every consumer must itself carry the shared config (else it reads an independent cache that is never
  written - exactly the class of bug we just fixed);
- sharing is **contiguous**: a non-participating layer sandwiched inside a segment is rejected because
  it gets padded with a self-reference and cuts the segment;
- sharing must not cross the CED boundary: `assert n_prefill_layer == n_layer or
  layer_kv_map[n_prefill_layer] == n_prefill_layer` (`cross_layer.py:116`);
- the candidate map must be monotonically non-decreasing in layer order and its source must be an
  index-module source.

### 2.3 Attention: three owners, and what a "Reuse" layer actually does

`v4_modules/attn.py:752-800` (`V4Attention.forward`) branches on four booleans that DeepInfer keeps
strictly independent (`attn.py:245-258`): `is_kv_source`, `is_index_source`, `is_index_k_source`,
`is_yoco_hidden_consumer_layer`. The interesting combinations for us:

- **KV source without indexer**: reuses `shared_attn.topk_idxs` for attention but still publishes its own
  compressed KV (that is the "dense ratio" path, `_compute_kv_ca_index` else-branch,
  `attn.py:568-581`).
- **Index-K source that is not a KV source**: consumes the KV source's *pre-RoPE* compressed latent and
  runs only its own `KProject`+norm to build `ica_cache` (`attn.py:549-551`, `compress.py:32-53`).
- **Index source (Reindex)**: runs the indexer, publishes topk, but *does not own KV*; in *decode* it
  reads the KV owner's `ica_cache` tensor directly (`cache_engine.py:429-431`), in *prefill* it reuses the
  owner's loaded KV (`index_attn.py:556-557`).
- **Plain Reuse layer**: `kv_ca_index = cross_layer_runtime.topk_indices` and nothing else
  (`attn.py:568-570`), plus the KV owner's `ca_cache` by alias.

`CrossLayerRuntime` (`cross_layer.py:203-260`) is the per-forward slot: `shared_hidden` /
`pre_rope_kv` / `topk_indices` / `block_candidate_idxs` / `candidate_topk_block` /
`candidate_mqa_schedule_metadata` / `loaded_prefill_kv` / `loaded_prefill_index_kv`. **It is reset per
forward step and keyed by nothing** - there is exactly one instance per rank per batch, so the
producer must be scheduled before the consumer within the batch. Ours is the `with_shared(...)`
singleton in `crates/sparkflash-engine/src/forward/attention.rs:123-141`, same shape, same hazard.

### 2.4 Compressed KV (`compress.py`)

`Compressor` supports **C1 and C2 only**: `kv_compress_ratio in (1, 2)`, and it hard-rejects
`compress_overlap` and `use_ape` (`compress.py:73-78`). C1 returns the projected latent unchanged
(`compress.py:89-93`); C2 runs `compress_and_update_state_{prefill,decode}` against an fp32 **state
ring** of `ratio + max_backtracking - 1` rows, because pooling two tokens causally needs the half-open
group state to survive across steps (`worker/cache_engine.py:_state_bytes`, `_needs_compress_state`).
The compressor returns the **pre-RoPE** latent that feeds both the indexer's KProject and, after norm +
rope, the main KV; `compress_rope_theta=160000` is used for the compressed KV, not `rope_theta=10000`.

### 2.5 MoE (`mlp.py`)

Gate -> `tile_kernels.moe.moe_topk_gate` (sqrtsoftplus scores, `e_score_correction_bias` used for
selection only, top-6, `norm_topk_prob`, `routed_scaling_factor=1.5`) -> DeepEP dispatch ->
per-expert `swiglu_forward_and_per_token_cast(..., topk_weights=...)` -> grouped GEMM -> combine, with
the shared expert added **inside the combine** (`combine(..., shared_out, ...)`, `mlp.py:604-610`).
Two arithmetic facts worth flagging against our M2/M3 audit items:

- the router weight is applied to the **SwiGLU output, before w2** (`mlp.py:19` import of
  `swiglu_forward_and_per_token_cast` and the `topk_weights=dispatched_topk_weights` call site), so the
  combine is a plain sum - the released `model.py` instead scales at the combine. Mathematically equal,
  **not** bit-equal (one extra rounding on the activation vs on the output). **[inferred from the call
  signature; I did not read the kernel]** - worth one experiment before we "fix" M2 in the other
  direction.
- shared-expert output is added at combine time, not folded into the routed sum first.

### 2.6 Engram (`engram.py`)

Not disk-centric: the tables are **rank-sharded over an engram-parallel group** and fetched per token
over DeepEP (`EngramBuffer.fetch(hashes)` -> hooks), with `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` to keep
them resident. `engram_layer_ids=[1,14]` produce a fused KV projection whose output width is
`n_embed*(hc_mult+1)` - i.e. Engram writes a KV-side contribution *and* a gated hidden contribution,
applied after `hc_post` (`decoder_layer.py` Engram call site, `engram.py:170-219`). It is asserted to
live only in the encoder half (`deepseek.py:136`). For us the mapping is `sparkflash-engram` (hash +
tiered cache + io_uring store) - the *sources* differ (they fetch from ranks, we read disk), the
arithmetic after the gather is the same.

### 2.7 mHC

`hc_mult=4`, 20 sinkhorn iterations, `hc_pre_init 3.0`, `hc_post_mult 2.0`, `hc_dynamic_scale 0.01`;
`hc_pre` is a `[q, C*H] @ [C*H, C*(C+2)]` TF32/FP32 contraction, `hc_post` an FP32 `C*(C+1)*H`
contraction (`csrc/engine/flops_estimator.hpp:101-121`). This file is the cheapest way to see exactly
which GEMMs DeepInfer thinks mHC costs.

---

## 3. Executor / worker architecture

Three layers, cleanly separated, and the split is the interesting part:

```
examples/offline_inference.py:95            LLM.generate
  engine/llm_engine.py:860   step()        Scheduler.schedule -> SchedulerOutputs(seqlens_q, ...)
  engine/llm_engine.py:1017  _run_workers  -> driver worker
  worker/worker.py:89        execute_model_step(is_prompt, ...)
       1. model_runner.execute_model -> CudaGraphRunner.replay / eager   (GPU)
       2. self._post_process_step(prev_step_state)                       (CPU, overlaps 1)
       3. sampler -> verify + sample                                     (GPU)
       4. post_process(current) if prompt or !ZOS, else defer            (CPU)
  engine/llm_engine.py       process_output -> RequestOutput
```

Three design choices worth naming:

**(a) CUDA-graph batch-size quantisation with padding sequences.**
`worker/cuda_graph_runner.py:41-51` captures `{1,2,4,...,interval} union {interval,2*interval,...,max}`
and `select(total_q)` takes the smallest captured size >= `total_q` (`:143-148`), padding with
`padding_seq_metadata` (`:93`). Decode therefore never runs off-graph, and step cost is a staircase in
`total_q`. With `max_num_seqs*(1+nextn)` as the ceiling (`worker/model_runner.py:77`), one sequence with
5 drafts needs a 6-row graph. **This is the mechanism that makes MTP affordable there**: the verify
batch is already in the graph, so a 5-token verify costs one replay.

**(b) Zero-overhead scheduling (ZOS), `csrc/engine/zos.cpp` + `worker/worker.py`.**
Step N's forward is launched with `input_ids[0] = ` the committed token and
`input_ids[1..] = seq->mtp.draft_tokens` (`zos.cpp:66-68`) - i.e. the *predicted* drafts are used as the
next step's real input before verification has returned. Lengths are then corrected on device
(`compute_seqlens_k_zos`, `compute_seqlens_q_by_confidence_rates`, `prepare_zos_inputs`), and the CPU
learns the true `seqlens_q` from a pinned D2H (`worker/worker.py:_SeqlensCorrection`,
`cpu_timer.record("worker.post_process...")`). The scheduler is therefore **optimistic about
acceptance** and the GPU fixes it up. This is only worth porting if CPU scheduling is on the critical
path; at 939 ms/step it is not. Do not port it.

**(c) The C++ scheduler is the policy engine, python is the plumbing.**
`csrc/engine/scheduler.hpp` holds: the DP-group vote for prompt-vs-decode (all-reduce MIN over
remaining budget), chunked prefill with `cp_align` for CP padding and a `prefill_max_sum_seqlen_k`
budget (`:444-471`), decode scheduling in LCFS order with preemption/recompute, and the MTP varlen
budget. Its cost model is a FLOPs estimator, not wall-time
(`csrc/engine/flops_estimator.hpp:39-45`, per-layer MLP/router/hc/engram/attn terms), with
`(bs, ms)` curves layered on top for graph-size choice (`worker/decode_perf_cache.py`,
`mtp_varlen.hpp:269-274`).

**Priority inversion worth noting:** the scheduler decides prompt-vs-decode *per step*, and a decode
step is never allowed to grow the graph beyond the captured ladder - so a request that just finished
prefill lands in the next decode step at whatever graph size is >= its total_q, not at its exact size.
That is why `seqlens_q` are "optimistic estimation in async-scheduling varlen case"
(`scheduler.hpp:62-63`).

---

## 4. Parallelism: what it actually is

| axis | group | size in prod | what it splits | file |
|---|---|---|---|---|
| attention DP | `PG.dp` | 64-128 (decode), 2-8 (prefill) | requests | `parallel_utils/parallel_state.py:92-94` |
| MoE EP | `PG.ep` | == DP | experts (contiguous rank blocks) | `:101-106` |
| engram parallel | `PG.enp` | configurable | Engram hash tables | `:108-110` |
| CP | `PG.cp` | **1 (unused)** | tokens (mirror/zigzag) | `:96-99`, `cp.py` |
| TP | - | **does not exist** | - | - |

`parallel_state.py:92` makes the layout explicit: ranks are `(replica dp cp)` for the DP group and
`(dp cp)` for the CP group, so DP groups are strided by `cp@@ and EP groups are contiguous - EP lives
*inside* the scale-up domain, DP crosses nodes.

**Mirror CP** (`layers/tilelang_ops/cuda/prepare_cp_mirror.py:73-113`), if we ever want it without TP:
a sequence of length `sq` is cut into `2*cp` pieces of `ps = ceil(sq/(2*cp))`; rank `r` owns pieces
`r` and `2*cp-1-r` (`half = blk//cp_size; rank = blk + half*(2*cp-1-2*blk)`, `:77-79`). Rank 0 therefore
gets the head and the tail, which balances causal attention work. Per-rank key ranges come out of the
same kernel as `ks/ke` per compress ratio (`:52-53`), and invalid rows are `-1` (`:111`). CP
communication is a bucketed all-gather of per-token tensors (`cp.py:42-51`): the projected KV, the SWA
KV, and the indexer's K. This is portable *in principle* to our TCP/RDMA fabric, but note the group's
requirement: DeepEP's `BucketBuffer` requires NVLink **and** RDMA bandwidth to be detectable even for a
single-transport group (`vendor/DeepEP/README.md:57`), so we would write our own all-gather.

**Why this matters for us:** DeepSeek's answer to "one GPU per node, no NVLink" would be *EP only*
(their decode shape is exactly that: many nodes, one EP domain, DP on top). We chose TP=4. Both are
defensible, but only one of them is validated by this repo, and it is not ours.

---

## 5. Single request, prefill -> decode -> sampling (exact chain)

1. **Admit.** `llm_engine.py:add_request` builds a `Sequence` with *both* `prompt_token_ids` and
   `compressed_prompt_token_ids` (Engram's compressed tokenizer, `:564-567`), a per-sequence `state_id`
   (SWA/compressor rings), sampling params, and optionally `so_args` (structured output) and
   `golden_token_ids` (teacher forcing).
2. **Schedule.** `Scheduler::schedule` (`scheduler.hpp`) produces `SchedulerOutputs`:
   `scheduled_seqs`, `prompt_run`, `num_batched_tokens`, `seqlens_q` (per request, may exceed 1),
   `sum_seqlen_qxk`, `max_seqlen_k`, `attn_tflops`, `total_tflops`, and the block/state
   `clone_states`/`zero_states` lists for prefix-cache forking.
3. **Metadata.** `prepare_prompt_v4` / `prepare_decode_v4`
   (`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`) compute, per query row: `positions`,
   `slot_mapping_store_swa`, `kv_swa_full_index` (as pivots), `ks`/`ke` **per compress ratio**,
   `klen`, `kv_ca_index` offset/len, `block_tables`, and the YoCo compaction fields. The integer
   algebra is documented in comments at `:478-490`, `:534-539`, `:588-640` and is worth reading as the
   spec of our own prefill metadata.
4. **Prefill forward.** `DeepseekForCausalLM.forward` (`models/deepseek.py:374`) loops the 40 rows with
   one `CrossLayerRuntime`; each `DeepseekDecoderLayer.forward` (`decoder_layer.py:97`) runs
   `hc_pre -> attention -> hc_post -> (engram) -> ffn -> hc_post`, capturing `shared_hidden` at the
   boundary row. Prefill KV is *gathered* into a per-sequence contiguous buffer by the load plan
   (`block_load_plan.cpp:48-59`, `load_paged_kv_cache`), and attention reads it through `kv_ca_index`.
5. **Sample.** `Sampler.forward` (`layers/sampler.py:114`) -> `process_sampling`/`get_logits` ->
   `deep_select.interface.topk` over the (optionally force-top-k'd) vocab -> sampled token, logprobs,
   `sum_smaller_probs`.
6. **Draft.** With MTP on, `DFlash.run_drafts` (`mtp/dflash.py:347`) runs **one** forward over the
   `dspark_block_size=5` positions of the block (position 0 = the golden token, 1..4 =
   `dspark_noise_token_id`), through the **3** DFlash stages, with the markov head, and returns per-step
   logits + a **confidence** per position (`ConfidenceHead`, `dflash.py:46-79`).
7. **Verify (next step).** The 5 drafts become the next decode step's `seqlen_q = 1..6` rows. The
   sampler compares draft logprob under old vs new distribution, accepts a **prefix**, resamples the
   first rejection from the residual (`layers/sampler.py:74-83` doc, `tilelang_ops/sampler.py`:
   `raw_accept` = "rejection-sampling decisions", `get_accept_result` = prefix + golden override), and
   emits `accepted prefix + final row`. SWA ring slots for rejected rows are simply overwritten on the
   next step, which is why the ring is `window + MAX_BACKTRACKING_TOKENS` with
   `MAX_BACKTRACKING >= nextn+1 = 6` (`environ.py:132`, `cache_engine.py:88-93`).
8. **Scheduling of verification budget.** `mtp_varlen.hpp` solves two nested problems from the
   confidence vector: choose `total_q` (graph size) maximising
   `expected_tokens / (baseline_ms + slope*graph + sampler_ms)`, then allocate `seqlen_q` per request
   greedily by descending expected acceptance (`:251-267`). Production effect: verified length 5.00 ->
   3.2-4.4, accepted length unchanged 3.29 -> 3.20, +4-7% throughput at 60 seqs/rank
   (`doc/inference_deployment_qa.md:247-256`).

Note the *separation of concerns*: confidence is **only** a scheduler prior (and a stats signal,
`batch_confidence_error = sum |accept_prob - confidence|`, `tilelang_ops/sampler.py:585-589`).
Acceptance itself is textbook speculative rejection sampling. If we ever implement "confidence-scheduled
verification", we must not let the confidence head touch the accept decision, or we change the model.

---

## 6. KV cache layout and addressing (the parts that constrain our Rust KV)

`worker/cache_engine.py:400-477` carves one byte-buffer per block into per-layer slices, advancing a
single `kv_offset` cursor per layer:

- **Main (compressed) KV**: `token_per_block = block_size // ratio` tokens per block
  (`block_size=128`), each token a fixed byte record - **528 B** for F8_E4M3-per-32
  (`512 B e4m3 + 16 B ue8m0`) or **288 B** for the FP4 path (`256 B e2m1 + 32 B e4m3`, i.e. **per-16
  e4m3**, *not* per-32 ue8m0), scale bytes stored *inside* the token record after the data
  (`vendor/FlashMLA/csrc/cuda_kernels/kv_cache_format.h:5-20`). Aligned up to 32 B per layer.
- **Indexer KV**: `token_per_block` = `block_size // ratio`, bytes/token = `index_head_dim/2 +
  index_head_dim/32` = 64 + 4 = **68 B** (FP4 e2m1 + ue8m0 per 32) (`cache_engine.py:73-80`).
- **Share-KV / share-index-K**: **no bytes allocated** for a consumer; the consumer's `KVCache` holds
  the *same tensor object* as the owner's (`:429-431`, `:446-461`). Pointer semantics, exactly as we
  concluded. The `kv_offset` does not advance for aliased layers.
- **SWA**: a *per-sequence* ring, `swa_cache[state_id, window + MAX_BACKTRACKING, 1, 528]`
  (`:407-423`), not paged; `slot_mapping_store_swa` is `state_id*swa_block_size + pos % swa_block_size`
  (`prepare_tensor_input_v4.cpp:268-271`), with `-1` for rows that must not write.
- **Compressor state**: fp32 ring `[ratio + max_backtracking - 1, 2, head_dim]` per sequence
  (`cache_engine.py:441-443`, `:463-465`), plus a separate ring when the indexer has its own compressor.
- **Decode addressing** (the formula our paged path must match):
  `kv_ca_index[pos] = block_tables[pos // tpb] * tpb + pos % tpb`
  (`prepare_tensor_input_v4.cpp:606-611`); prefill instead gathers contiguous segments
  `[src_block_id, src_begin, src_end, dst_offset]` (`block_load_plan.cpp:48-59`) and uses
  `kv_ca_index[i_q, :len] = offset[i_q] + arange(len)`.

So the whole KV footprint of this model is **4 compressed-KV owners + 8 indexer-K owners + 20 SWA rings
+ up to 2 compressor-state rings**, not 40 of each. Any per-layer allocation in our engine is 10x waste.
(The Q&A works the SWA side out loud: "20 stored layers x 134 tokens x 528 B = 1,415,040 B per
sequence", `doc/inference_deployment_qa.md`, SWA KV cache section.)

---

## 7. Ranked: what transfers, what does not

Effort: S < 1 day, M 1-3 days, L 1-2 weeks, XL > 2 weeks. Payoff is against our current 939 ms/step,
39-44 s TTFT, 1.0642 tok/s.

### T1. DSpark parallel-block draft + prefix-verified varlen decode - **highest leverage for tok/s**
- **Path.** `deepinfer/model_executor/models/mtp/dflash.py` (`run_drafts:347`, `forward_drafts:241`,
  `sample_drafts:279`, `ConfidenceHead:46`), `csrc/engine/mtp_varlen.hpp`,
  `layers/tilelang_ops/schedule_mtp_varlen.py`, `layers/sampler.py:430-540`, `worker/model_runner.py:67-90`.
- **What.** One forward over a 5-token block (golden + 4 noise tokens) through 3 DFlash stages, each
  stage a full decoder block with a 128-expert/top-3 MoE and its own SWA ring; per-position logits via
  the markov head; per-position confidence; next step verifies the accepted prefix in the *same* graph
  as a normal decode step (`seqlen_q in 1..6`).
- **Hardware.** Portable *conceptually*; the kernels used are SM100-gated (see section 8), so we would
  implement it on sm_121a with our own kernels. Nothing about the algorithm needs SM100.
- **Mapping.** `sparkflash-scheduler` (already advertises a "DSpark confidence scheduler"),
  `sparkflash-engine/src/forward/mtp.rs` (already models the 3 stages with boundary capture),
  `sparkflash-engine/src/forward/attention.rs:963-1008` equivalent (draft attention against
  `swa_cache` + `dflash_kv_index`), and a `n`-row decode step in `sparkflash-device`.
- **Payoff.** Production accepted length is **3.2-3.3 of 5** (Q&A:247). If our step with 5 draft rows
  costs ~1.25-1.4x a 1-row step (the block's draft forward is 3 stages * 5 rows; the verify is the
  normal 40 layers * 6 rows), the ceiling is **~2.2-2.5x decode tok/s** (1.06 -> ~2.4-2.6 tok/s). At
  our batch of <=17 the draft forward is a *small* extra GEMM (3 layers, 128 experts) - the win is
  larger for us than for a 60-seq batch. **This is the item I would spend the next big block on.**
- **Risk / verification.** (i) The MTP heads must be loaded from the checkpoint's `mtp.{0,1,2}`
  tensors - bit-identical drafting is not required for correctness of the *output distribution*, but it
  is required for reproducing a given token stream; (ii) the verify path must reproduce the reference's
  accept rule (prefix accept, residual resample at first rejection) and its logprob bookkeeping;
  (iii) SWA ring backtrack margin must be >= nextn+1 or accepted tokens overwrite live slots;
  (iv) `dspark_markov_rank=256` bigram head is part of the *model*, not a trick - omission changes
  output; (v) label it explicitly: with MTP on, the token stream differs from greedy non-MTP decode
  even at temperature 0. Our acceptance language must say so (D39 already retired token equality, which
  helps).
- **Guess.** Whether our 939 ms/step is weight-streaming-bound (MTP then pays ~linearly) or
  latency/launch-bound (MTP pays slightly less). **Measure first**: time a 1-row vs 6-row decode step.
  If 6 rows cost < 1.5x 1 row, T1 is the play.

### T2. CED decoder-half prefill truncation + SWA bounded replay - **the only big TTFT lever**
- **Path.** `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:145-168` (`yoco_swa_visibility`),
  `:203-344` (`fill_yoco_truncation_metadata`), `models/v4_modules/yoco.py:47-123`
  (`compact_yoco_boundary`), `layers/tilelang_ops/prepare_cp_mirror.py:79-120`
  (`yoco_cp_boundary_fused`), `doc/inference_deployment_qa.md:181-191`.
- **What.** Encoder rows 0..=20 run over the whole prompt; decoder rows 21..=39 run **only over the last
  `keep_tokens` positions** (`KEEP_TOKENS=140 = window 128 + backtrack 6 + slack`). Per kept row the SWA
  visible domain is `max(pos-window+1, data_len-keep, valid_start) .. pos`, decomposed into two pivot
  segments (cached ring + current chunk). Compressed rows use `ke = (pos+1)/ratio` with a per-sequence
  base, so compressed key ranges are derived from the same geometry. Cold-prompt truncation is **exact**
  for the kept suffix; the *replay* on a cache hit is documented as **approximate** ("yields an
  approximate SWA KV cache", Q&A:183) and is never written to the prefix cache (Q&A:187).
- **Hardware.** Pure index arithmetic - fully portable, no kernels required.
- **Mapping.** `sparkflash-kv/src/schedule.rs` already has the CED plan; what is missing is the
  visibility algebra (pivots) and the compressed-range derivation. `sparkflash-kv/src/swa.rs` already
  splits encoder/decoder.
- **Payoff.** "Nearly halving total prefill computation" (Q&A:187) for prompts > 140 tokens; **zero at
  37 tokens**. Our host reference currently runs all 40 layers over the whole prompt
  (`sequence_fixture.rs:12`, `:646`), i.e. we already accept a D7 divergence; this item is about making
  the *device* path the labelled one and pinning the boundary algebra.
- **Risk / verification.** Bit-identical for cold prompts if `keep >= window + max_backtracking` and the
  kept range is a suffix; the harness to copy is `examples/yoco/yoco_equiv.py`, which compares
  `mask` mode (compute all rows, mask visibility) against `compact` mode (truncate rows) on both
  metadata and end-to-end logprobs. That is exactly the "explicitly-labelled behaviour change" pattern
  we require: implement mask mode first (mechanically equivalent to full prefill), then compact.

### T3. KV byte layout, owner aliasing and the decode index formula
- **Path.** `worker/cache_engine.py:400-500`, `vendor/FlashMLA/csrc/cuda_kernels/kv_cache_format.h:5-20`,
  `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:534-640`,
  `csrc/engine/input_metadata/block_load_plan.cpp`.
- **What.** See section 6. The FP4 main-KV scale granularity is **16 dims, e4m3**; the indexer KV is
  **32 dims, ue8m0**. Per-block layout is token-major with scale bytes appended to each token record;
  blocks are 128 tokens (compressed blocks hold `128/ratio`).
- **Hardware.** Fully portable (it is a byte layout).
- **Mapping.** `sparkflash-kv/src/{layout,fp4,paged}.rs`, `sparkflash-kv/src/plan.rs`.
- **Payoff.** Not a speedup by itself, but it is the difference between 288 B and 528 B per token if we
  still carry F8 KV (1.8x attention-cache traffic and 1.8x cache capacity), and it prevents a
  scale-granularity mismatch that would be silently wrong-but-plausible.
- **Risk.** Every byte must match or dequantised values differ in the last bits; verify with a
  round-trip fixture against the HF `model.py` quantiser, not against DeepInfer.

### T4. Topology audit against `_source_map` + the three-map validation chain
- **Path.** `deepinfer/hugging_face/config.py:62-85`, `:158-161`; `v4_modules/cross_layer.py:22-140`.
- **What.** Derivation + validation of `layer_kv_map`, `layer_index_module_map`, `layer_index_k_map`,
  `layer_candidate_source_map`, and the CED-boundary assertion.
- **Hardware.** Portable.
- **Mapping.** `sparkflash-engine/src/layer_plan.rs` (already models Full/Reindex/Reuse/SwaOnly/Mtp),
  `sparkflash-kv/src/config.rs:49` (replace `ENCODER_LAST_LAYER = 20` with "first ratio==1").
- **Payoff.** None directly; it is insurance. The one thing I would *add* is `layer_index_k_map` as a
  distinct map plus the nesting assertion, because for the next checkpoint the indexer-K source need not
  be the KV source, and a wrong assumption there produces a cache that is never written - our D38 bug
  class exactly.
- **Risk.** Low. Add as assertions in `layer_plan.rs` tests.

### T5. Prefill as one varlen forward with a FLOPs budget - **diagnostic first**
- **Path.** `csrc/engine/scheduler.hpp:444-520` (chunk budget, `cp_align`, sum-K cap),
  `csrc/engine/flops_estimator.hpp` (cost model), `doc/inference_deployment_qa.md:115` (quadratic
  `cost(q,c)` solve in production).
- **What.** Prefill chunk size is *solved*, not fixed: take the largest `q` that fits both the token
  budget (32768) and the time budget. Everything else about prefill is "one varlen forward".
- **Payoff / honest read.** DeepInfer has **no special prefill trick** that would help a 37-token
  prompt. Its prefill is one forward. If our 37-token prefill costs ~26-44 steps' worth of time, the
  problem is our kernel/chunk efficiency, not scheduling. **Concrete diagnostic before any work:** time
  prefill of 1/8/37 tokens and check whether step time is flat (weight-streaming/launch bound: then a
  37-token prefill should approach ~1 step and TTFT should fall to low seconds) or linear in tokens.
  Our own numbers (0.9 s/token; chunked 0.66 s/token) look linear, which suggests the chunked path is
  still issuing per-token work.
- **Risk.** None to numerics; it is a measurement.

### T6. EPLB as *replication*, not shuffling - **portable idea, real decode payoff if we are
  expert-bandwidth bound**
- **Path.** `model_executor/eplb/policy.py` (`_batch_greedy_indices:14-88`, `rebalance_experts_global`),
  `eplb/__init__.py` (`physical_to_logical_map`, `logical_to_physical_map`, shuffle over symm_mem),
  `tests/models/test_eplb_policy.py`.
- **What.** Add `num_extra_experts` redundant copies of hot logical experts; route tokens to a *local
  replica*; recompute placement on a window of per-step counts by greedily minimising the
  max-over-steps rank load; move weights only every 1000-10000 steps (and only over NVLink).
- **Hardware.** The *placement policy* is pure tensor math (portable). The *shuffle* uses
  `torch.distributed._symmetric_memory` / IPC and is NVLink-class only - on our TCP fabric we would
  **replicate at load time instead** (choose K hot experts per rank, never move them), which costs RAM
  and no fabric. We have 121 GB/node; 384 experts * ~20 MB FP4 = 7.7 GB per full copy, so a handful of
  replicated hot experts is cheap.
- **Mapping.** `sparkflash-engine/src/forward/moe.rs` routing indirection + `sparkflash-comm`.
- **Payoff.** Q&A:57 says Wide-EP's whole benefit is *reducing per-step memory traffic in the MoE part*,
  and EPLB is what makes Wide-EP effective. At EP4 we are the opposite extreme: if our 939 ms step is
  dominated by streaming routed experts, routing hot experts to a local replica is the only lever that
  does not need more nodes. **[guess]** - depends on the same measurement as T1.
- **Risk.** Routing indirection must be bit-identical: the replica choice may not change *which* experts
  a token sees nor the summation order of its top-k contributions. Their map is
  `[layer, logical, redundant+1]`, invalid = -1 (`eplb/__init__.py:52-56`) - the ordering discipline is
  the thing to copy.

### T7. DeepSelect exact top-k (threshold + radix select)
- **Path.** `vendor/DeepSelect/docs/DeepSelect-deep-dive.md:5-49`,
  `deepinfer/model_executor/layers/tilelang_ops/sampler.py` (usage), `index_attn.py:494-503`.
- **What.** Keep a running k-th-largest threshold `T`; scan blocks in random order, keep only `> T`,
  compact with radix-select when the candidate buffer exceeds `k+B2`; exact, O(N) expected, ties broken
  by index. Interface we must match: `sorted_index=True`, `idx_oob_fill_value=-1`, `end=<valid klen>`,
  `return_value=False`.
- **Hardware.** Portable (algorithm), though the fast versions use PTX `set.s32.bf16x2` etc.
- **Mapping.** `sparkflash-kernels` top-k, `sparkflash-engine/src/forward/attention.rs:1650+`
  (`choose_topk`-equivalent).
- **Payoff.** Small for us today (klen is tiny at 37 tokens; 512-of-N only matters for long context).
  Worth having *before* long-context work, not now.
- **Risk.** "Random block order" is nondeterministic by construction; to stay reproducible we must fix
  the permutation seed and the tie-break. Our reference sorts selected indices ascending, which they do
  too (`sorted_index=True`) - that part matches.

### T8. Batch-invariant decode (only if we need reproducibility)
- **Path.** `vendor/FlashMLA/csrc/api/sparse_decode.cpp:24,57,93,136,330`
  (`SparseDecodeFeatures::BATCH_INVARIANT`), `environ.py:197`, `worker/runtime_config.py:63-67`,
  `tests/models/test_batch_invariant.py`.
- **What.** A kernel feature flag that fixes the split-KV reduction order so a request's logits do not
  depend on batch composition. It is explicitly "slower" (`eval/run_eval.py:74`).
- **Payoff.** Zero tok/s; it *buys* determinism. Only relevant if we adopt a token-equality acceptance
  bar again. D39 retired it, so: read for the reduction-order idea, do not port.
- **Risk.** Note the interaction: batch-invariance is disabled when EPLB rearrangement is active
  (`parallel_utils/userbuffer.py:79`) - i.e. replication itself breaks invariance.

### T9. Structured outputs / xgrammar bitmask - not now
- `deepinfer/structure_output/`, `layers/tilelang_ops/structure_output.py` (per-step bitmask applied by
  step index), `examples/_structured_output.py`. Interface detail worth remembering: the bitmask is
  applied *per verification step* (`apply_token_bitmask_by_step_inplace`), which is the only correct way
  to constrain a speculative decode. Out of scope for performance.

### T10. What does **not** transfer at all
- **FlashMLA sparse decode/prefill**: hard gate `TORCH_CHECK(arch.is_sm100f(), ...)`
  (`vendor/FlashMLA/csrc/api/sparse_decode.cpp:190`); kernels live under
  `cuda_kernels/sm100/...` and use tcgen05-family UMMA + TMA. sm_121a is not SM100f.
- **DeepGEMM**: FP4/packed-SF paths assert `arch_major == 10` (`csrc/utils/layout.hpp:69,78,91`);
  the JIT compiles `sm100_*` kernels only (`csrc/jit_kernels/impls/sm100_*`). DeepSelect's build
  hardcodes `-gencode arch=compute_100a,code=sm_100a` and requires NVCC >= 12.9
  (`vendor/DeepSelect/setup.py:117-125`).
- **DeepEP**: NVLink for intranode (`README.md:54`), NCCL multimem for NVLink reductions
  (`:328`), `BucketBuffer` wants NVLink *and* RDMA bandwidth detectable (`:57`). Our fabric is plain
  TCP/RDMA with no NVLink; the dispatch/combine *semantics* transfer, the transport does not.
- **CUDA 13 / toolkit assumptions**: `build.sh` pins `nvidia-nccl-cu13==2.32.3`, `README.md:50`
  requires CUDA 13 and GCC >= 13; the C++ engine extension is compiled against torch + NCCL.
  We are on a different toolchain and hand-written kernels; nothing here is a drop-in.
- **The C++ engine extension itself** (`csrc/engine/pybind*.cpp`): the *ideas* (scheduler, metadata
  algebra, ZOS) transfer; the code is a torch extension.

---

## 8. Where DeepInfer **contradicts or corrects** our current design

**C1. There is no TP reference to validate against.** `doc/inference_deployment_qa.md:53`: "TP and CP
are not used for now." Attention is DP, MoE is EP, decode is Wide-EP. Consequence for the work we are
about to ship (bit-identical TP=4 attention weight sharding): DeepInfer cannot serve as its oracle.
The oracle must remain the HF reference `model.py` plus our own device-vs-host twins. Stated
positively: our TP=4 is a memory-driven choice this repo does not constrain. Do not spend time looking
for "how DeepInfer shards wq_b" - it does not.

**C2. Decoder SWA KV is per-layer from the layer's own hidden - not from the shared encoder hidden.**
`hugging_face/config.py:47` sets `yoco_swa_share_hidden=False`; `attn.py:249` therefore computes
`yoco_swa_share_hidden = False` and `attn.py:395-397` (the "project kv_swa from shared hidden" path) is
dead for this checkpoint; `fuse_kv_swa=True` (`attn.py:280`). Q&A:187 agrees: "the decoder layers'
block KV cache is projected from the last encoder layer's hidden state, **so only the decoder SWA KV
cache has to be computed per layer**". If anyone proposed deriving decoder SWA KV from the shared
hidden, that would be wrong.

**C3. The shared-hidden plumbing is inert; only the compressed-KV alias is real.**
`yoco_shared_hidden_required()` (`v4_modules/yoco.py:23-45`) returns False for this config: the SWA
consumer condition is off (C2) and the "packed KV source after the boundary" condition is off because
`kv_source_layer_ids=[2,8,14,20]` has no layer > 20. `shared_hidden_kv_proj` is built only for a
consumer that is itself a KV source (`attn.py:323-373`), i.e. layer 20 only, where
`_resolve_shared_hidden` (`attn.py:465-479`) hands it that layer's own attn-norm hidden. So the only
cross-layer state that matters is **the topk publish/consume slot and the KV cache alias**. Our engine
should have zero hidden-state plumbing between layers - and if it has any, it is dead weight that can
only introduce wrong-row bugs.

**C4. Topk/compressed-row semantics - four details that change results if we get them wrong.**
1. `ke = (q_start + j + 1) / ratio` with a **per-sequence base** `base += max(1, seqlen_k / ratio)`
   (`prepare_tensor_input_v4.cpp:292-320`). Compressed rows are addressed per sequence, not globally
   paged, in prefill. A global paged coordinate would be off by the base.
2. A **dense (no-indexer) compressed layer still runs sparse attention** with a *precomputed*
   `kv_ca_index` covering the full compressed range (`:606-611`, `attn.py:579-580`). "No indexer" does
   not mean "dense attention over all keys in the kernel".
3. **Reindex layers do not run a global top-k.** The candidate source (layer 20) reduces `p_sum` over
   *blocks* (`candidate_topk_blocks=2048`, `block_size=8`, reduce=`max`, `index_attn.py:555-569`) and
   the later index sources select their top-512 **inside that candidate set only**, then map back with
   `restore_candidate_indices_inplace` (`index_attn.py:504-505`). A full top-k at layers 24/28/32/36 is
   a different model. Our `layer_plan.rs:19-23` already models the candidate pool - make sure the
   *indexer* actually restricts to it, not just the plan.
4. Top-k is **per query row**, `end=topk_lens=min(index_topk, klen)`, `sorted_index=True`,
   `idx_oob_fill_value=-1`, and the result is converted to a *physical* slot through the block table in
   place (`index_attn.py:494-513`). Also: a KV source without its own indexer **consumes** the previous
   index source's topk for attention while publishing its own KV (`attn.py:568-570`, `:585-587`) - so
   "who publishes" and "who owns KV" must stay independent flags.

**C5. MTP is a parallel block, not a loop.** `dspark_block_size=5` positions drafted by
`num_nextn_predict_layers=3` stages in **one** forward (`mtp/dflash.py:241-345`; `config.py:177-192`).
A 5-sequential-drafts design would cost ~3-5x more compute for the same drafts. Related: the draft
input at position 0 is the golden token and positions 1..4 are `dspark_noise_token_id=128799`
(`_embed_draft_init_tokens`) - the block is a diffusion-style draft, not a chain.

**C6. Verification length is a scheduling decision, acceptance is not.** Confidence only selects *how
many* drafts to verify (`mtp_varlen.hpp:199-267`); acceptance is rejection sampling with prefix
semantics (`sampler.py:74-83`, `tilelang_ops/sampler.py:251-299`). Anything else is a different model.
Production: verified 3.2-4.4 of 5, accepted 3.2-3.3 (Q&A:247).

**C7. Prefill scheduling: what is actually special.** Not the chunk size - the *solve*. And the only
big structural win is CED truncation (T2), which is worth nothing below 140 tokens. If our TTFT story
is "we need a better prefill scheduler", DeepInfer's evidence says otherwise: the scheduler's job is to
keep one big varlen forward fed. Our 0.66-0.9 s/token must be explained before any scheduler work.

**C8. `n_prefill_layer` is derived, not configured.** `hugging_face/config.py:158-161`. We hardcode 20
(`sparkflash-kv/src/config.rs:49`). Same today; derive it.

**C9. `MAX_BACKTRACKING_TOKENS` is a KV-layout parameter, not a tuning knob.**
`environ.py:132` forces `>= nextn+1`; `cache_engine.py:88-93` uses it to size the SWA ring. If we add
MTP we must widen every SWA ring and the compressor state ring by the same amount *before* the first
accept, or verification will corrupt state.

---

## 9. Suggested order of work (my recommendation)

1. **Measurement, 1 hour, no code:** decode step time vs rows (1/2/6/17) and prefill step time vs chunk
   size (1/8/37). This decides between T1 (rows are cheap -> MTP pays) and T6 (rows are expensive ->
   expert traffic is the wall), and it explains TTFT.
2. **T2-T4 as audits:** port the visibility algebra + the three-map validation + the FP4-16/e4m3 KV byte
   layout into assertions. Cheap, kills a class of silent bugs, and unblocks long-prompt TTFT.
3. **T1 if (and only if) the measurement says a 6-row step costs < ~1.5x a 1-row step.** It is the only
   item that changes tok/s by more than 30%.
4. Do **not** port ZOS, batch-invariance, CUDA-graph batch ladders, EPLB shuffles, or any vendor kernel.

---

## 10. Confidence ledger

- **Read directly (high confidence):** all file/line citations above; the config topology and its
  derivation; the KV byte layout; the topk/candidate/pivot algebra; the MTP block structure and the
  acceptance/confidence split; production claims quoted from `doc/inference_deployment_qa.md`
  (accepted length 3.29/3.20, dynamic verification 3.2-4.4, "TP and CP are not used", "nearly halving
  total prefill computation", EP64/DP64 shape).
- **Inferred, not read:** that `swiglu_forward_and_per_token_cast(topk_weights=...)` applies the router
  weight *before* w2 (call signature only; kernel is in the `tile_kernels` wheel, not in `vendor/`);
  that the 288 B FP4 KV path is what our engine should target; the exact per-step cost split of our own
  939 ms baseline (I did not measure anything).
- **Guesses, labelled:** the 2.2-2.5x MTP payoff for our batch size; that our decode is expert-traffic
  bound; that our chunked prefill is still issuing per-token work.
- **Not examined:** TileLang kernel bodies (`layers/tilelang_ops/*.py` beyond the metadata builders),
  DeepEP internals, vision encoder, structured-output matcher, eval harness.
