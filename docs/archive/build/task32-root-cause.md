# Task-32 root cause: the position-129 disagreement is an oracle measurement defect, not an engine window or arithmetic error

Status: 2026-10-19 17:58Z. Owner: audit-dev. Branch: task32-f32-vs-vllm (not merged).
Evidence recorded in this file, in scripts/compare_fixture_vs_oracle.py, and in the JSON files
listed under "Evidence" at the end (all with sha256).

## 1. The three controls the Lead asked for

### 1.1 Short context (8 tokens, same ids, f32 against the served oracle at position 7)

    python3 scripts/compare_fixture_vs_oracle.py \
        --fixture f32-prompt8.json --oracle native-oracle-suite.json --suite suite.json \
        --prompt-id b69-fixture-130 --continuation '[491,367,201,80,3302,201,54,201,4341,0]' \
        --position 7

    top1: f32 223 / oracle 565  DIFFER
      top5_overlap 0.6000   top20_overlap 0.4500   top100_overlap 0.3700
      top1000_overlap 0.3470
      max_abs_delta 9.28836   mean_abs_delta 2.63837
      kl_oracle_to_device 1.42693   js_divergence 0.203897

The two top-1 tokens are a near-tie in both implementations (f32 223 at -1.43 and 565 at -1.78;
oracle 565 at -1.50 and 223 at -2.94) and 8 of the top-10 tokens are shared, so this is a
flat-distribution tail reordering, not the position-129 signature. It does show that f32 and the
served model never agree token-for-token on this all-random prompt, which is why rank-1 agreement
alone cannot be the gate.

Engine reproducibility: the current engine (main 67412cc, rebuilt with --prompt-ids support)
reproduces the old golden fixture bit-exactly at both 8 and 130 tokens - identical f32 logits,
maxdiff 0.0 - so the fixture and these controls are not stale.

### 1.2 BOS / special tokens

- The fixture prompt ids (130 random ids, seed 7) contain no 0 (BOS), 1 (EOS), 2 (engram pad),
  129264 (image sentinel) or 128799 (DSpark noise). IMAGE_SENTINEL_BASE_ID in the image is 129264
  and image_sentinel_mask() tests 129264 or IMAGE_PAD_ID, so no multimodal branch can fire.
- The engine's prefill consumes the prompt ids as given: the forward path has no BOS insertion
  (bos_token_id is parsed into the config but not used by prefill).
- Alignment sanity check on the served model: teacher-forced top-1 accuracy on the seven natural
  prompts is 40-60% (code 63/159, math 72/129, prose 125/235, tech 169/276, multilingual 76/149,
  json 95/139, long 237/431) - a shifted or BOS-prefixed stream would not score like this.
So neither side prepends or shifts a token.

### 1.3 Window and per-layer RoPE

- Engine prefill mask (crates/sparkflash-engine/src/forward/attention.rs, window_indices):
  first = max(0, q + 1 - window_size), so at query 128 the keys are 1..128 and at 129 they are
  2..129.
- Served reference (vllm/models/deepseek_v4_1/common/ops/cache_utils.py, the SWA index kernel):
  left_add = 0 without an image, swa_start = max(pos - (WINDOW_SIZE - 1) - left_add, 0), so at pos
  128 the tokens are 1..128 and at 129 they are 2..129. Identical.
- Per-layer RoPE (rope_tables_for_layer): compress_ratio > 0 layers use compress_rope_theta
  (160000) with the YaRN tables rooted at original_seq_len 65536; compress_ratio == 0 layers use
  plain rope_theta (10000). This matches the vLLM reference comment in rope.py ("pure
  sliding-window layers use plain RoPE ... no YaRN").
- Config diff: the fixture's recorded config agrees with the checkpoint's text_config on
  compress_ratios, kv_source_layer_ids, index_source_layer_ids, index_topk (512),
  candidate_source_layer_id, candidate_topk_blocks, candidate_block_size, engram_layer_ids,
  window_size, rope_theta/rope_factor/beta_fast/beta_slow and the expert/top-k fields. No config
  mismatch found.
So the engine is not using a different window, RoPE or config. The disagreement has to come from
the computation itself; the next two measurements localise it.

## 2. The disagreement is position-local

### 2.1 Position 128 (129-token prompt): f32 and the served model agree

    python3 scripts/compare_fixture_vs_oracle.py \
        --fixture f32-prompt128.json --oracle native-oracle-suite.json --suite suite.json \
        --prompt-id b69-fixture-130 --continuation '[491,367,201,80,3302,201,54,201,4341,0]' \
        --position 128

    top1: f32 201 / oracle 223  DIFFER (near-tie: f32 -3.04, oracle -3.65)
      top5_overlap 0.8000   top20_overlap 0.8000   top100_overlap 0.8000
      max_abs_delta 2.62262   mean_abs_delta 0.369554
      kl_oracle_to_device 0.125926   js_divergence 0.0291567

That oracle row came from a 140-token request. Re-measured with a matched 130-token prompt
(section 2.3), the same position gives f32 top-1 201 = oracle top-1 201, KL 0.082 - agreement to
well inside every metric floor.

### 2.2 Position 129: the oracle entry is degenerate

    top1: f32 491 / oracle 367  DIFFER
      top5_overlap 0.4000   top20_overlap 0.5500   top100_overlap 0.4800
      max_abs_delta 6.72404   mean_abs_delta 2.33896
      kl_oracle_to_device 3.18271   js_divergence 0.529227

The distributions either side of the disagreement are qualitatively different:

    position  entropy of the oracle's returned band   f32 entropy over the same band
    7         4.40                                     2.89
    128       5.43                                     4.50
    129       0.53                                     4.30

and the served model's own top-1 probability is 0.026 at p128, 0.921 at p129 and 0.218 at p130
(released-oracle run of 17:28Z: 0.82 at p129). A model does not go diffuse -> near-one-hot ->
diffuse one token apart while the f32 reference stays continuous across 128/129; the near-one-hot
spike is the signature of a corrupted query, not of an arithmetic gap. Two independently served
checkpoints (released and heretic) produce 367 first, which points at the serving stack rather
than the weights.

The spike is not a shape artefact either. With a matched 131-token prompt on an idle server
(section 2.3) the same row still returns 367 first at p = 0.448 (entropy 2.73 against 5.31 at
p128), while f32 stays diffuse. Shape changes how sharp the anomaly looks (0.92 at 140 tokens,
0.45 at 131 tokens idle, 0.69-0.91 in a batch of four) but not that it exists. Section 2.4 adds a
second capture-condition dependence: concurrent traffic moves the same 131-token row from p = 0.448
to p = 0.882. The invariant across every capture is the argmax 367, not the sharpness.

### 2.3 Matched-shape oracle rows: the shape effect, and what it changes

A vLLM prompt_logprobs row p is computed in the prefill of the whole request prompt, so its value
depends on the request's total prompt length and batch composition (section 4.1 quantifies this).
The engine comparison in 2.1/2.2 used oracle rows from 139/140-token requests, not from the
engine's own prompt length. Re-measuring with the minimal prompt that still contains row p
(row p = prompt_logprobs[p+1], so the minimal prompt is p+2 tokens of the fixture stream):

    row  prompt len  f32 top1  oracle top1   top100 overlap  mean|d|  KL(o||d)  oracle entropy
    7    9           223       565           0.45            2.281    1.712     4.144
    128  130         201       201           0.80            0.354    0.082     5.149
    129  131         491       367           0.48            1.125    1.726     2.732

So:

- p128 is agreement once the shapes are matched (identical top-1, KL 0.082), one token before the
  incident. This is the strongest evidence that the engine's window/RoPE/config are right.
- p129 is still a disagreement and still an anomalous oracle row (idle capture: entropy 2.73 vs
  5.15 at p128, top-1 p 0.448 vs 0.037) - less sharp than the 140-token row (0.92) but the same
  spike, with 367 first and the f32-preferred 491 second. It is not stable even as a shape effect:
  with one concurrent generation in flight the same 131-token row gives p = 0.882 / entropy 0.727
  (section 2.4). Only the argmax 367 is invariant across captures.
- p7 still disagrees under matched shape (KL 1.71, top-100 0.45): that is the genuine
  simulation-vs-native gap on a flat distribution, not a shape artefact.
- Protocol consequence: every engine-vs-oracle comparison must record the oracle prompt length and
  request shape, and should request the minimal shape for the compared row. A row taken from a
  140-token request is not the same quantity as a row taken from a 130-token request at the same
  position; the difference is up to ~2 nats in the tails and up to a factor of two in top-1
  probability at a degenerate position.

### 2.4 Capture-condition sensitivity: concurrent traffic is a third variable

The vLLM here runs with enable_prefix_caching=True, chunked_prefill=True, TP=4, DP=1 and dspark
speculative decoding (num_speculative_tokens=5), and it is shared: while probing I saw an external
client (192.0.2.71) sending /v1/chat/completions. A controlled test isolates the effect: the same
four request bodies, sent idle, then while one long generation runs in a background thread, then
idle again:

    row/len        idle (x2, identical)          concurrent generation        idle again
    129 / 139      p 0.820  entropy 1.145        p 0.928  entropy 0.503        p 0.820 1.145
    129 / 131      p 0.448  entropy 2.732        p 0.882  entropy 0.727        p 0.448 2.732
    128 / 130      p 0.037  entropy 5.149        p 0.028  entropy 5.310        p 0.037 5.149
    7   / 9        p 0.127  entropy 4.144        p 0.263  entropy 3.665        p 0.127 4.144

Each block is internally bit-identical; the concurrent-generation block differs from both idle
blocks and the switch is reversible. So prompt_logprobs are deterministic per (request shape,
batch composition), and the batch composition changes when another request shares the forward
pass. This is a third variable after prompt length and batch-of-N composition, and it was active
during part of the capture window: the capture that produced p = 0.448 / entropy 2.73 was an idle
capture (it reproduces exactly in idle blocks), while the length-probe capture of the same row at
p = 0.882 was taken while traffic was in flight. The consequence for D40/D41 is that an oracle
capture is only defined together with its batch state: capture on an idle server (no other
clients), record that it was idle, and do not mix captures from different load states. The switch
is reversible - the idle value returns exactly when the concurrent generation finishes - so a busy
capture cannot be corrected post hoc; it must be re-taken.

## 3. Is the served variant a known-buggy one? (Lead item a)

The launch recipe is recorded in phase2-evidence/window-2026-10-18/window-vllm.sh line 39: it
launches vllm/vllm-openai:deepseekv41-flash-0909 with exactly five mods - engram-disk-reference,
sm120-dsv4-topk, fix-dsv41-sparse-topk-oob, diag-dsv41-decode-topk, install-ray - and the applied
list and each mod's output are in start-vllm.log. Two candidates the Lead flagged:

- fix-dsv4-c128a-prefill-short-extend (not applied): its patch.py edits
  vllm/models/deepseek_v4/sparse_mla.py - the V4 path, not the V4.1 path our model uses - and it
  fixes a crash ("Compressed sparse MLA prefill requires compressed sparse indices") that only
  occurs on a mixed short-extend batch, i.e. a chunked-prefill continuation with
  query_len <= decode_threshold. Our oracle prompts are single-chunk prefills of 139/140 tokens,
  and the server never crashed. Absent evidence of a silent-logits effect, this mod is not the
  explanation.
- fix-dsv4-sparse-swa-spec-config (not applied): its patch restores
  "spec_config = self.vllm_config.speculative_config" in
  vllm/v1/attention/backends/mla/sparse_swa.py. The shipped image already contains that line
  (count 1 in the image source, inspected with docker run on the image), so the mod is a no-op
  backport for older images and the image we serve is not missing it.

What the applied mods do touch: sm120-dsv4-topk only adds template specialisations for topk=1152
on the sliding-window-only layers plus one sparse_attn_indexer gate (it is a dispatch width, not a
semantic change); fix-dsv41-sparse-topk-oob masks out-of-range block-table reads in
_compute_global_topk_indices_and_lens_kernel (bounds-only by design, all in-range values
unchanged). Neither documents a window-index semantic change.

Conclusion for (a): the two unapplied patch mods do not make this variant "known-buggy at window
edges" - one fixes a crash in a code path we do not take, the other is already in the image. The
p129 defect therefore has to be quarantined or re-measured on the evidence, not on a known patch.
The honest sentence is: a documented-recipe variant with an unexplained p129 defect.

### 3.1 The oracle measurement itself (shape, root, artefact check)

- Served root: 1bae0e1088993b9b82de08f59de5be5a34e42170 (recorded in native-oracle-released.json and
  re-read from /v1/models by the scripts), image vllm/vllm-openai:deepseekv41-flash-0909, launched
  from window-vllm.sh, health at 17:28:14Z.
- Request shape (oracle_released.py): POST /v1/completions with
  {model: deepseek-v4.1-flash, prompt: <139 ids>, max_tokens: 1, temperature: 0.0, top_p: 1.0,
  ignore_eos: true, prompt_logprobs: 1000}. The 139 ids are the 130 fixture token ids followed by
  the fixture's first 8 next_token values and the sentinel 0; row p is prompt_logprobs[p + 1], i.e.
  P(token | ids[0..p]). The suite run used the same request with a 140-token prompt (130 suite ids +
  9 continuation + sentinel).
- The anomaly is reproducible in rank-1 (9/9 across the two runs) but NOT in row values: p129
  top-1 is 367 with p = 0.82 in the 139-token run and 0.92 in the 140-token suite run, and its
  rank-2 differs (491 at -3.83 vs 67 at -3.95). That value difference is prompt-shape and
  batch-state sensitivity, not sampling noise: warm same-body repeats are bit-identical in a fixed
  batch state (sections 2.4/4.1), so the oracle is deterministic per (shape, batch state). "Rank-1
  9/9" must therefore never be read as "the oracle is deterministic at this row's value"; it is a
  stability statement about the argmax only.
- It is NOT the observed-token one-hot gather artefact reported by serve-dev: that signature is
  top-1 == the prompt's target token with logprob 1e-5..1e-4 and top-2 around -10. Our p129 row has
  top-1 367 while the observed target token is 491 at rank 2 (-3.83) in one run and rank 7 (-5.45)
  in the other, and top-2 is -3.95/-3.83. It is a genuine, anomalous predictive distribution, not a
  gather fallback.
- serve-dev's independent patch test corroborates section 3: the image already contains the
  sparse_swa spec_config fix (the mod's patch.py fails its pattern on this image and would abort a
  launch), c128a-prefill-short-extend applies but is V4.0-only (the served arch is
  DeepseekV41ForCausalLM and never calls _build_c128a_metadata), and fix-dsv41-sparse-topk-oob was
  live in this window (start-vllm.log shows its cache_utils.py patch on all four nodes at 17:18Z).

## 4. D41 floor: two oracle runs on the identical 130-token prefix, at two shapes (Lead item b)

Both runs are the same server (started 17:18:44Z); run A is the first request 16 s after health
(17:28:30Z, prompt length 139: 130 ids + 8 continuation + sentinel), run B is the suite request
(17:30:22Z, prompt length 140: 130 ids + 9 continuation + sentinel). The 130-token prefix is
identical, so the difference between the two runs is **prompt-shape sensitivity, not sampling
noise**: the same-body warm repeats in section 4.1 are bit-identical, so a floor frozen from this
pair is a shape envelope (worst case KL 0.47, top-100 0.60, 1.41 nats mean), not a noise floor.
D41 must freeze the shape (prompt length and batch) with the floor; the same-shape floor is zero.

    position  rank1  top5   top20  top100 top1000  |d| p50  p90   p99   max    KL(shared band)
    129       same   0.60   0.65   0.66   0.672    1.38     2.35  3.29  4.20   0.136
    130       same   0.20   0.55   0.60   0.715    0.59     1.53  2.41  3.16   0.466
    131       same   1.00   0.85   0.88   0.840    0.13     0.33  0.61  0.78   0.025
    132       same   0.80   0.90   0.82   0.835    0.18     0.53  0.90  1.18   0.037
    133       same   0.60   0.75   0.85   0.843    0.31     0.78  1.31  1.78   0.162
    134       same   0.60   0.75   0.79   0.689    0.31     0.87  1.46  2.19   0.211
    135       same   0.60   0.80   0.87   0.832    0.20     0.52  0.86  1.18   0.055
    136       same   0.80   0.65   0.81   0.797    0.62     1.43  2.12  2.56   0.259
    137       same   0.80   0.80   0.92   0.887    0.15     0.38  0.72  0.88   0.023
    mean             0.667  0.744  0.80   0.79

Rank-1 is stable 9/9. Everything downstream of rank-1 is not: only 672 of the 1000 returned ids
are common at p129, and shared-token logprobs move by 1.4 nats at the median, up to 4.2.

Plain consequence for D40: a 3x floor on max_abs_delta is 12.6 nats. The returned band spans about
10 nats, so that bar admits everything and the provisional max_abs_delta bar is useless if it is
built from this floor. The stable numbers to build a gate on are rank-1 agreement, top-k set
overlap, and the KL over the shared returned band:

- rank-1: floor 9/9 (bar: must stay 9/9 until a repeat shows otherwise)
- top-100 set overlap: floor 0.60 (mean 0.80); a bar of 0.60 keeps the measured floor and is
  still discriminating
- KL over the shared returned band: worst floor 0.47, so 3x floor = 1.4
- per-token |d| (p99 3.29, max 4.20) is a diagnostic, not a bar

Cross-check against the engine numbers of section 2, same metric family: f32-vs-oracle
KL is 0.126 at p128 (passes a 1.4 bar), 1.43 at p7 (borderline), 3.18 at p129 (fails). The
KL/top-k gate therefore separates the p129 defect from ordinary agreement, while max_abs_delta
does not.

Caveat to record with the floor: the two runs differ in prompt length (139 vs 140), so part of
this floor is shape sensitivity, not pure nondeterminism. The floor should be re-measured as a
repeat pair with identical prompt length before it is frozen; the numbers above are an upper
bound.

What the floor means for the bars:

- top-5 set overlap: the measured floor is 0.667 mean (min 0.20), which is below D40's provisional
  top-5 bar of 0.90. That bar is unmeasurable as written - run-to-run noise alone fails it. D41
  must either lower the top-5 bar below the floor (which makes it loose) or drop top-5 in favour of
  top-20 (floor 0.744, min 0.55) and top-100 (floor 0.80, min 0.60).
- top-100 set overlap: floor 0.60 mean 0.80 is the most useful single set metric; a 0.60 bar keeps
  the measured worst case and still rejects the p129 anomaly (f32-vs-oracle top-100 overlap 0.48).
- KL over the shared returned band: floor 0.47, 3x = 1.4; f32-vs-oracle 0.126 (p128, pass), 1.43
  (p7, borderline), 3.18 (p129, fail). Best distributional discriminator of the three.
- rank-1: floor 9/9; keep as a hard "no flips" criterion only for positions where the model is not
  degenerate (p129 of the random prompt is quarantined, section 8).
- max_abs_delta: vacuous (3x floor 12.6 nats vs a 10-nat band); must not be a D41 bar.

### 4.1 Warm repeats: identical shape is deterministic, shape changes are the floor

Run after the numbers above, on the released server restarted at 18:03:25Z (served root
1bae0e1089..., warm, no cold-start effect).

Three identical sequential requests (same 139-token prompt, same JSON body, /v1/completions,
max_tokens 1, temperature 0, top_p 1, prompt_logprobs 1000) produce **bit-identical**
prompt_logprobs at all nine compared positions: rank-1 9/9, top-k overlap 1.000 for k=5/20/100/1000,
|difference| 0.000 (p50/p90/p99/max), KL 0.000. p129 top-1 is 367 at logprob -0.19884082674980164
in all three runs, which is exactly the frozen 17:28 released file's value. So identical-shape
repeats in a fixed batch state have a floor of zero: the oracle is deterministic, warm or cold.
The 1.41-nat table in section 4 is therefore **shape sensitivity** (prompt 139 vs 140 tokens), not
run-to-run noise. Name that drift explicitly wherever D41 is recorded: the 0.82-vs-0.92 top-1
probability at p129 and the rank-2 491-vs-67 difference in that table are length/shape drift, and
"rank-1 9/9" does not mean the row values are reproducible.

Batch state is a second condition (section 2.4): the deterministic zero floor only holds while no
other request shares the forward pass. With one long generation running concurrently the same
139-token body gives -0.0749 instead of -0.1988 at p129, and the value returns exactly when the
concurrent generation finishes. Captures taken at different load states are not comparable, so a
D41 floor must state the load state as well as the shape.

A labelled shape pair - one request with four identical copies of the same prompt (batch 4) versus
the single-prompt runs - shows what shape does:

- p129 top-1 stays 367, but its probability moves to 0.690 / 0.806 / 0.839 / 0.908 (single: 0.820).
- worst |difference| 4.059 nats (p90 2.121), worst KL over the shared band 0.932, minimum top-100
  overlap 0.610, shared ids 663-890 of 1000.
- one rank-1 flip, at p135 for choice 2, where the single-run top-1 probability is 0.019 (a
  genuinely flat position).

So the reproducible floor is a *shape* envelope, not a noise floor: zero for a fixed request shape,
up to ~2.1 nats p90 / 4.1 nats max / KL 0.93 / top-100 0.61 when batch or prompt length changes,
with rank-1 flips confined to flat positions. Consequences for D41:

- Define the floor per protocol: floor_same_shape = 0, floor_shape_envelope as measured above. A
  3x rule on a zero floor is degenerate, so bars must be stated as "at a fixed, recorded request
  shape"; the 3x multiplier then applies to the cross-shape envelope when shape cannot be fixed.
- Rank-1 agreement is not shape-invariant at flat positions (p135 here, where oracle top-1 p is
  0.019). Either require rank-1 only where the oracle's own top-1 probability is above ~0.05, or
  state the criterion as "rank-1 flips allowed only where top1 probability < 0.05".
- top-5 at 0.667 mean (single-shape repeats are 1.000; cross-shape floor 0.61) is still below D40's
  provisional 0.90 bar, so that bar remains unmeasurable as written; top-20/top-100 and the KL
  over the shared band are the stable metrics.
- The p129 anomaly survives all of this: it is bit-identical across identical repeats, moves only
  its sharpness with shape (0.92 / 0.45 / 0.69-0.91), and keeps 367 first. It is a stable property
  of that query in this stack, not noise. The Lead records the D41 decision itself; section 7
  lists the repeat artefacts.

### 4.2 The irreducible shape floor of a matched-shape comparison

An engine comparison cannot use an oracle point record at exactly the engine's prompt length: row p
= prompt_logprobs[p+1] needs a prompt of at least p+2 tokens, while the engine prefill has p+1.
So the matched comparison always carries a 1-token shape difference. Measuring the same row at
lengths p+2, p+3, p+4 bounds what that costs. IMPORTANT: the capture below was taken while
concurrent traffic was in flight (busy batch state, section 2.4 - the row-129 numbers are the busy
values p 0.882/0.822/0.598). It measures the *length* effect within one batch state, which is what
a length envelope needs, but its absolute numbers are busy-mode and therefore NOT the D41 floor.

STATUS: the idle-mode re-capture (same ids, same six shapes, idle_A/busy/idle_B blocks with
per-block external-client proof) is PENDING - blocked by window policy because the vLLM instance is
in use by the user and must not be disturbed. D41 must not freeze the busy-mode table above in the
meantime; the staged script is phase2-evidence/window-2026-10-18/oracle_length_envelope_idle_task32.py.

    row  len  top1 (prob)          pairs     p90 |d|  max |d|  KL     top-100 shared
    128  130  201 (0.0285)         130/131   0.34     1.01     0.041  0.832
    128  131  223 (0.0210)         130/132   0.37     0.81     0.025  0.809
    128  132  201 (0.0294)         131/132   0.38     0.90     0.051  0.801
    129  131  367 (0.8823)         131/132   1.68     3.02     0.053  0.700
    129  132  367 (0.8219)         131/133   1.89     3.74     0.235  0.683
    129  133  367 (0.5980)         132/133   1.10     2.31     0.208  0.744

Readings:

- Even a 1-token length change can flip rank-1 at a flat position (row 128: 201 at length 130,
  223 at 131, 201 at 132 - the two tokens are a near-tie) and can move the p129 tails by up to
  3.7 nats. The p129 rank-1 stays 367 at all three lengths but its probability moves 0.88 -> 0.60.
- The honest floor for a shape-matched comparison is therefore the length envelope, not zero:
  KL up to 0.235, p90 up to 1.89, max up to 3.74, top-100 overlap down to 0.68, with rank-1 flips
  allowed only where the two tokens are near-ties.
- Against that envelope, the measured f32-vs-oracle numbers are informative: KL 0.082 at p128 is
  inside the floor (agreement), KL 1.726 at p129 and 1.712 at p7 are far outside it (real
  disagreement). A 3x-floor KL bar would be 0.71 - p128 passes, p7/p129 fail - so the KL over the
  shared band remains the discriminating metric, provided the floor is the length envelope.
- These probes should be repeated for the exact rows a future D40 gate uses; the numbers above are
  a measured envelope for rows 128/129 on this prompt, not a universal constant.

## 5. Which side the device agrees with (Lead item c): neither - three-way divergence

Source: window-2026-10-18/runs70/logits-130x8.jsonl + .bin (9 positions 129..137, device chain
sparkflash_m3c_c4c_seqreuse_4f599b7_aarch64, sha256 7e0ce4bf...). The device dump also carries a
host-side reference and head A/B blocks; "reference vs fixture ... equal=true" for all nine tokens
means the host side of the chain reproduces the golden fixture (491, 367, 201, ...). The dumped
device logits are the hybrid/device path.

At position 129, with the same 130-token prefix, the three implementations give three answers:

    implementation  top-1 (value)                  top-5
    f32 reference   491 (logit 13.3877)            491, 2463, 223, 63438, 67
    device native   8060 (logit 19.4155)           8060, 67, 16187, 3108, 71484
    served oracle   367 (logprob -0.0819, p 0.92)  367, 491, 813, 37852, 67

Pairwise agreement at p129:

    pair            top5   top20  top100 top1000   corr over oracle band
    f32 / oracle    0.40   0.55   0.48   0.529      0.600
    device / oracle 0.20   0.10   0.22   0.197      0.335
    device / f32    0.20   0.20   0.23   -          0.533 (full vocab)

Device against f32 on the full vocabulary at p129: mean |difference| 2.4861, max 15.009581 at
index 5986, 128,528 of 129,280 logits outside the 0.02 bar, correlation 0.533. Those are exactly
the b68 numbers in section 5 of native-arithmetic-audit.md (128,528 / 15.009581 / argmax 8060), so
the class-A criterion-1 failure at p129 is deterministic and reproduces bit-for-bit in the newer
seqreuse chain; it is not caused by anything the oracle does.

The device-vs-f32 comparison at p128 is NOT measured. The log line "head A/B: position=128
host_top1=201 device_top1=201 block_max_equal=true ... 14.054767609" applies the host head to the
device hidden and compares it with the device head on the same hidden: both sides read the same
hidden, so it validates the head step only. The fixture stores logits only for the prefill step
129, so the full-vocabulary comparison exists only at 129..137. Consequence: p128 is not evidence
that the device was faithful one position before the incident; the device's first divergence point
is unmeasured and may be p128 or earlier. (The f32-vs-oracle p128 agreement in section 2.3 is a
separate, valid statement.)

### 5.1 The whole region p129..p137 (device defect, owned by task-38 / kernel-dev)

The device dump's jsonl also carries reference_id (host f32 greedy token) and hybrid_id (device
greedy token) for every dumped position. They differ at every dumped position:

    position   129   130   131   132   133   134   135   136   137
    device     8060  8060  1168  50673 303   531   6525  5719  14171
    f32        491   367   201   80    3302  201   54    201   4341

Caveat that must travel with this table: p129 is the only teacher-forced comparable row. From
p130 on the device consumes its own previous token, so those rows compare two different decode
streams; they show the divergence does not heal but they cannot be used as independent evidence
that each later position is individually wrong. The p129 row is the one that proves the split.

The Lead has assigned this region to task-38 (kernel-dev): the divergence starts where the 128-token
window first evicts and is a single deterministic device defect in the eviction transition
(window/sink bounds, mask, KV ring slot, or a stale evicted-slot read), not an arithmetic or
oracle issue - no quantiser moves a logit by 15 nats. This task does not chase it; the evidence
paths are runs70/logits-130x8.{jsonl,bin,log} with the sha256s in section 7.

Consequence for this task: p129 of this all-random prompt is a three-way split, and the region
p129..p137 belongs to the device defect, not to the oracle. The device's disagreement with f32 says
the f32 reference is not native there (as the audit's class-A reading already implied); its
disagreement with the oracle says the oracle is not a shared ground truth either. No
implementation's p129 token can be a gate, which is the basis for the quarantine in section 8.

## 6. Reproducer

    # engine f32 fixture for a real prompt (the generator now takes --prompt-ids)
    sequence_fixture_task32 --config released --checkpoint <snapshot> \
        --prompt-ids b69-130.json --decode-steps 1 --logits prefill --provenance node \
        --out f32-prompt130.json
    # offline comparison, no server
    python3 scripts/compare_fixture_vs_oracle.py --fixture f32-prompt130.json \
        --oracle native-oracle-suite.json --suite oracle_prompt_suite.json \
        --prompt-id b69-fixture-130 \
        --continuation '[491,367,201,80,3302,201,54,201,4341,0]' --position 129

## 7. Evidence (sha256 of every file used)

Engine side (all produced offline, no server):
- sequence-130-released.json (golden fixture, old f72738b binary): e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308
- b69-130-fixture.json (current main + --prompt-ids, logits bit-identical to the fixture prefill): 5f35be3e2c56abf96b743131e45570f2d6bc554d4a9ae49195945a229f7fd89f
- f32-prompt8.json (8-token prefix): 321c2aaffee5b14b16c3bc1161c6eeb5556ecf03bd3a49bf78b1dfb17291a346
- f32-prompt128.json (129-token prefix): 466d36b5a3627c8850adbb0f51ca1d799f9ceb5912321a9972504c39fa5a859f
- b69-8.json current binary vs f32-prompt8.json: logits identical, maxdiff 0.0 (engine reproducible)

Oracle side:
- native-oracle-suite.json (suite run, 17:30:22Z, prompt_len 140): efce16c2637aeab70475a43b8dc0c5cbe1f066b6bb9cd671480c7d7e03c4572a
- native-oracle-released.json (first request after health, 17:28:30Z, prompt_len 138): 8ecc423d0e6240811421d95823caaeab396209a8cef8d7ef9b6fe54d3521f475
- oracle_prompt_suite.json (8 prompts, sha256 recorded inside both oracles): 42f66ad2...

Device side (Lead window b70-device):
- logits-130x8.jsonl: f548d9c3115c0557cc277ed1b32ce2c82985ec38d299869163759f31fc30aea6
- logits-130x8.jsonl.bin: 43b31438c01955c8359457e077322fc39e6c6a76f330bdf958e87d9ddcad07b5
- logits-130x8.log: carries the "reference vs fixture equal=true" 9/9 check and the
  "logits 129: 128528 ... max 15.009581 ... argmax 8060" class-A line (sha256 recorded in the
  task-38 handoff copy)
- device binary sparkflash_m3c_c4c_seqreuse_4f599b7_aarch64: 7e0ce4bf2390ec078f70098eb5e77881b518e5d0ed58f9155b669a355a8d48cc

Warm repeats and shape probes (same released server, restarted 18:03:25Z, root 1bae0e1089...):
- native-oracle-repeats.json (3 identical sequential requests, 139 tokens):
  86731124ec814f1b103a42c04e682e2a38a371029185b61f07a24df1a53368ca
- d41-floor-warm.json (bit-identical repeat comparison): 2434ae707eb97abaf6a12147328abb0c1ea9f034d84bd67186def96498726920
- native-oracle-batch4.json (one request, 4 identical prompts): d6e2e1e840972b92cde19609c33a8d98a1ae133e90aa24a0a368d49b56fa37ae
- d41-shape.json (batch4 vs single shape probe): cc02a30383f60ceb6806f7dfb814c9852d528fdd66b1699b4a9fbff01240e874
- native-oracle-matched-shape.json (minimal-shape rows 7/128/129, idle capture): 5e89c31d60fa220d9095401139d6efba4a8651734df64bab5d0c346c3474c9a7
- native-oracle-batch-mode.json (idle / concurrent-generation / idle blocks, section 2.4):
  8fbd4f76e63909ee37948d26d1289aa91501b5d278e8fa8646e883fdc364260e
- native-oracle-length-probe.json (length envelope, busy batch state): 5fae4ee9933a551e9a1da486971ceb6e1c37fd9b8c78402e7fc90d67607f6400
- response-00.json (raw first repeat): 8e09d416c76377f047d906adceb29b679825e655b2ebdd4f66df958343d8ebc8
  (repeats differ only in response id/created fields; prompt_logprobs are identical)

Analysis records (reproducible from the files above):
- control-p7.json: ad1c209b6d12a751a4cac1308de0a079818edf03cb93edbadaf6ff07a8eb3a42
- control-p128.json: 415f8983a195a5e1d2d3acb4819dbb0fb5cac8e4c09cdf68c62149768e230358
- control-p129.json: 04f9b8e234ff6bf782ea09f1b041abbe5297754847b339e8b06fa0e35095e1b9
- control-natural-p7.json (code-rust-borrow, in-distribution short control): f32 786 vs oracle 5238,
  mean |delta| 1.228, KL 1.013, top-100 overlap 0.48; same flat-tail signature as p7 random
- three-way-p129.json: d51b97103160da04850bbdfbcf6724f3c74517d92c5e46c191df3d121f7fb0cb
- d41-floor.json: per-position rank-1/top-k/percentile/KL table of section 4
- oracle-repeat.json: per-position repeat metrics for positions 129..137

Fixture-engine control (in-distribution, natural prompt, short context) also disagrees: 8 tokens of
"code-rust-borrow": f32 top-1 786, oracle 5238, mean |delta| 1.228, KL 1.013, top-100 overlap 0.48.
That is the sim-vs-native flat-tail gap, present in-distribution too, so no bar may be built on
token identity even for natural prompts.

## 8. Decision (proposed; no engine change)

1. Quarantine position 129 of the random-token prompt from every token-level D40 bar, for both the
   oracle and the f32 column. The reason is the three-way divergence of section 5: f32 491, device
   8060, oracle 367, with f32 and oracle agreeing at p128 (top-1 identical, KL 0.082 at matched
   shape). This is not a serving-stack patch issue (the two candidate patch mods are a no-op
   backport and a crash fix on a path we do not take); it is a query-specific anomaly on this OOD
   prompt whose argmax is stable but whose sharpness depends on shape and batch state (sections
   2.3/2.4: top-1 p 0.448 idle at 131 tokens, 0.882 with concurrent traffic, 0.820 idle at 139
   tokens). Do not describe it as "near-one-hot" without the capture conditions. The region
   p129..p137 is task-38 (kernel-dev, device eviction defect), not this task.
2. Do not treat the fixture's f32 tokens as superseded by the oracle. Keep f32 as the dequantised
   simulation reference (D39) and record the oracle as the behavioural reference only where its
   shape is not degenerate. At p129 neither column is truth.
3. D40 floor/bars: use rank-1 agreement, top-100 set overlap, and KL over the shared returned band,
   measured at a fixed request shape. Do not use max_abs_delta: single-shape floor is 0 and the
   cross-shape floor is 4.06 nats, so a 3x rule gives either 0 or 12.2 nats against a ~10-nat band
   - vacuous either way. Keep per-token |delta| as a diagnostic only.
4. Record the request shape AND the batch/load state with every oracle row, and request the
   minimal matched shape for the compared position (section 2.3: the same row moves by up to ~2
   nats in the tails and its top-1 probability nearly doubles between a 140-token and a 130-token
   request). Capture with no other client on the server (section 2.4: one concurrent generation
   moves the same row by 0.12-0.7 nats deterministically), or record that the capture was busy and
   do not mix it with idle captures. The load-state switch is reversible, so a busy capture cannot
   be corrected post hoc - it must be re-taken. Treat the floor as: floor_same_shape = 0 (identical-shape
   repeats are bit-identical in a fixed batch state); floor_length (section 4.2, busy-mode capture,
   PENDING idle re-capture - window policy blocks server contact, so this number must not be frozen
   as the D41 floor until the staged idle capture runs): KL up to 0.235, p90 1.89, max 3.74,
   top-100 overlap down to 0.68; floor_batch (4x batch): KL 0.93, p90 2.12, max 4.06, top-100 0.61; floor_concurrent
   (one long generation in flight, section 2.4): row-dependent, up to 0.7 nats and 0.43 in top-1
   probability; floor_length_pair (139 vs 140 token prompts): 1.41 nats mean, KL 0.47, top-100 0.60.
   Apply the 3x rule to the applicable floor, not to a mixed number; for a matched-shape idle gate
   the 3x KL bar is 0.71, and the measured f32-vs-oracle KL is 0.082 at p128 (pass), 1.71 at p7 and
   1.73 at p129 (fail).
5. Rank-1 flips occur only at flat positions (p135: oracle top-1 p = 0.019 in the batch-4 pair), so
   state rank-1 bars as "flips allowed only where the oracle's own top-1 probability < 0.05" or
   require rank-1 only at non-flat positions.
6. The device path's p129 class-A divergence (b68, reproduced bit-for-bit by b70) belongs to the
   device-arithmetic work and is now task-38; it is not explained by, and does not depend on, the
   oracle. Any stage bar that used p129 device-vs-f32 must quote the class-A status (audit doc
   section 5) rather than a bare mismatch count.
7. No engine fix in this task: the engine reproduces its golden fixture bit-exactly on current main
   (8- and 130-token runs, maxdiff 0.0) and agrees with the served model at p128 at matched shape.
