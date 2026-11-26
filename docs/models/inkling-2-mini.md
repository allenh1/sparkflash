<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Inkling-2-Mini

Inkling-2-Mini is Thinking Machines' first released looped transformer. It is a 24-layer
multimodal MoE decoder whose middle twelve layers run three times with shared weights (48 layer
applications per token), with a message path between passes. This page is the model's working
record: intake, scan, the wiring the layout states, and what is still open.

## Intake

| Field | Value |
| --- | --- |
| Hub id | `thinkingmachines/Inkling-2-Mini` |
| Revision | `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` |
| License | Apache-2.0 (model card, `README.md`) |
| Producer reference | `huggingface/transformers`, `src/transformers/models/inkling2_loop/`, commit `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3` ("Add Inkling2 Mini (#51050)", 2026-11-21) |
| Stored formats | NVFP4 routed experts (E4M3 block scales per 16 values, F32 amax per expert); BF16 dense projections, embedding, head, norms, F32 controller gates and query banks; the drafter is BF16 |
| Target | sm_121 (GB10, the .100 development host and the DGX Spark reference), tensor-parallel widths 1, 2 and 4 |
| Checkpoint | `model-00001-of-00001.safetensors` (1,244 tensors, one shard) plus `mtp.safetensors` (54 tensors, no index) |

The release is one checkpoint of about 20 GB. `config.json` names
`Inkling2LoopForConditionalGeneration` / `inkling2_loop`; the text block is
`inkling2_loop_text`, the tower `inkling2_vision`, and the drafter is described by
`mtp_config`. The vision and drafter tensor names are the same ones Inkling-2-Small uses; the
language stack adds the loop and its message path.

## Geometry

| Fact | Value | Source |
| --- | --- | --- |
| Hidden width | 2,304 | `text_config.hidden_size` |
| Decoder layers | 24 | `text_config.num_hidden_layers` |
| Query / KV heads | 24 / 8, head width 128 | `text_config.num_attention_heads`, `num_key_value_heads`, `head_dim` |
| Layers per attention-residual block | 6 | `text_config.n_layers_per_attn_res_block` |
| Sliding layers | 0-4, 6-10, 12-16, 18-22 (window 1024) | `text_config.local_layer_ids` |
| Full layers | 5, 11, 17, 23 | complement of `local_layer_ids` |
| Loop core | layers 6..18, 3 passes, gate rank 64, not variable | `text_config.loop_config` |
| Routed experts | 128 experts, 8 active, intermediate 1,152, route scale 8 | `text_config.n_routed_experts`, `num_experts_per_tok`, `intermediate_size`, `route_scale` |
| Vision tower | width 1,024, 23 blocks, 16 heads, patch 20, merge 2, projection 4,096 | `vision_config` |
| Drafter | 3 stages, width 2,304, 24/8 heads, Markov rank 256, block 8 | `mtp_config` |

## What the producer computes

`modeling_inkling2_loop.py` is the reference. The parts that differ from Inkling-2-Small:

1. **The loop** (`Inkling2LoopTextModel.forward`, :1390-1470). Layers `[0, 6)` run once, layers
   `[6, 18)` run `num_passes` times from the pre-loop hidden state, layers `[18, 24)` run once.
   Every pass after the first starts from `pre_loop_hidden_states` and the pre-loop block-state
   stack, not from the previous pass's output.
2. **The cross-pass message** (`Inkling2LoopMessageAttentionResidual`, :147-176). Four learned
   queries per loop layer (query, key, value, MLP) attend over the previous pass's block
   summaries plus its final hidden state; the result is that loop layer's message for the four
   streams. A shared unweighted RMSNorm precedes the attention; no causal mask.
3. **The message gates** (`Inkling2LoopMessageGate`, :81-99).
   `(1 - tanh(forget)) * hidden + tanh(gain) * message`, with both gates projected from an
   unweighted-RMSNorm of the concatenated hidden state and message through a rank-64 down
   projection. One gate per stream (`q`, `k`, `v`, `mlp`) per loop layer.
4. **Full-attention input norms** (:1108-1195): separate `q_input_layernorm`,
   `k_input_layernorm`, `v_input_layernorm` before the projections; the sliding layers share one
   `input_layernorm`.
5. **Dynamic short convolutions on the four full layers** (`Inkling2LoopDynamicShortConvolution`,
   :440-500): taps are `static_weight` plus a per-token projection of the layer input, and the
   residual is divided by `sqrt(2k + 1)` rather than `sqrt(k + 1)`.
6. **The drafter** is a DFlash-style stage with a Markov head and separate context projections
   (`*_ctx`), not the DSpark stage of Inkling-2-Small.

## Scan

`sparkflash onboard thinkingmachines/Inkling-2-Mini` reads the config, the tokenizer files and
the safetensors headers and prints one row a part. At the tip of this work it reports, for sm121
at TP 1, 2 and 4 (the rows are identical; only the rank shapes move):

| Status | Rows | What they are |
| --- | --- | --- |
| `resolved` | 9 | the embedding, the two global norms, the head, the drafter's `mtp` row, and the five vision parts whose bodies already exist |
| `not built` | 97 | the four per-layer parts at this release's geometry (2,304-hidden attention, 2,304-wide static convolution, 128x2,304 experts, the 2,304 f32 block residual) over 24 layers, plus the 4,096-wide vision merger |
| `no part kind` | 28 | `cross_pass_attention` and `loop_message_gate` on the twelve core layers, and `dynamic_short_conv` on the four full layers |
| not checked | 54 | the drafter's names in `mtp.safetensors`. It is a separate file with no index, and the Hub reader indexes `model-*.safetensors` shards only; a `--config` scan beside a local checkpoint reads it |

The two coverage checks are clean of model facts: **every one of the 1,244 indexed tensors is
claimed** by the layout, and every config key is read or ignored with a reason. The fit, from the
scan's own arithmetic at the checkpoint's 1,048,576-token window:

```text
TP1: weights 18.2 GiB, KV 16.1 GiB, total 34.3 GiB of 113.7 GiB  - fits
TP2: weights  9.4 GiB, KV  8.0 GiB, total 17.4 GiB of 113.7 GiB  - fits
TP4: weights  5.0 GiB, KV  4.0 GiB, total  9.0 GiB of 113.7 GiB  - fits
```

That is 188 rows in all. The scan is therefore at stage 1's exit for everything the Hub index
can see: no `unknown key` and no `unclaimed tensor` row, and the only `unknown layout` rows
in the whole report are those 54 drafter names. What remains is kernel instances (stage 3) and
three part kinds (stage 4).

The parse and the scan were run on the `.100` development host (aarch64 GB10, sm_121) as well
as on the x86_64 workstation, and produce the same rows; the configuration parse and the
`inkling` test pass on both.

## The device run on .100

`examples/device-prefill` uploads the checkpoint's tensors directly (no converted rank file),
loads the vision tower on rank 0, and prefills `[1, 100, 200]`:

```text
[example] vision tower on rank 0: 634443776 bytes
next Some(1)
1      26.375000
198    25.625000
68014  25.625000
220    24.500000
real   0m3.3s
```

The host reference's top scores for the same prompt are 26.384371, 25.655384, 25.572002 and
24.427736: the device ranks the vocabulary the same way and stays inside the arm's BF16
tolerance. The tiny looped fixture's device walk is held to the host reference exactly by
`the_looped_fixture_walks_every_application_without_a_refusal`.

## Serving on .100

`sparkflash serve --checkpoint ~/models/inkling-2-mini` now detects this release through its
descriptor (`thinkingmachines/Inkling-2-Mini`, `inkling2-mini-nvfp4`), opens the arm's device
backend, loads the vision tower and (with `SPARKFLASH_INKLING_DRAFT=dspark`) the drafter, and
serves the OpenAI HTTP API:

```text
sparkflash serve: thinkingmachines/Inkling-2-Mini (inkling2-mini-nvfp4) at TP=1 on
  /home/allenh1/models/inkling-2-mini, context 1048576, Inkling device
sparkflash serve: listening on http://127.0.0.1:8012
```

* a chat completion returns model text (19 prompt tokens, 24 generated, greedy);
* a chat request with an `image_url` gains the tower's span (the same prompt goes from 21 to 25
  tokens) and answers, with the tower resident on rank 0;
* the drafter's 738 MB load and its packed projections print at startup, and its rounds verify
  drafts (the speculative block above).

The drafter is live: the served response's `x_deepseek_spark_stats.dspark_speculative` block
reports its rounds. A 16-token answer proposed 12 drafts across 4 verify steps and accepted 11
(3.75 tokens a step, 91.7%); an image request's 16 tokens proposed 15 across 6 steps and
accepted 9 (60%). `StepOutput::draft` stays empty by the API's convention — the driver consumes
the accepted rows inside the round — so the speculative block, not that field, is the evidence.

Measured on .100 from the same requests: TTFT 1.06 s for the 25-token image prompt, decode
19.8 tok/s with speculation on. The image-request `usage.prompt_tokens_details.image_tokens`
reads zero because the generic host driver hard-codes that field; the prompt itself grows by the
tower's four placeholders (21 → 25), and the arm's own image prefill changes the answer (the
same tokens with the tower reach top logit 220 against 1 without it).

The DSpark drafter was the earlier refusal: this release's `mtp_config.target_layer_ids`
has four taps and its `fuse_weight` is `[4, 4]`, while the arm's drafter is built for
Inkling-2-Small's eight taps and six fusion rows. The kernels already take `taps_per_row` as
a runtime argument, so the work is the arm's geometry, not a new body.

## Rule-2 table

Every part the description resolves, the producer function it is checked against, and the types
the producer computes it in. A row that is `not built` still states its function; the scan has
only proved a kernel is missing.

| Part | Producer function (file:line) | Types | Layout fact |
| --- | --- | --- | --- |
| `relative_attention` | `Inkling2LoopSlidingAttention` :590-682, `FullAttention` :683-773: GQA, learned relative bias, sink, output gate, q/rel L2 clip; the full layers also log-scale queries and relative states | BF16 projections and clip scales; `float()` around clip, gate, sinks and softmax; f32 softmax | relative bias, sink, gate, clips, log-scaling bounds |
| `block_attention_residual` | `Inkling2LoopBlockAttentionResidual` :871-898: unweighted RMSNorm over the stacked summaries plus the partial state, then a `hidden**-0.5` query attention (1 or 3 queries) | f32 query bank; `float()` states; f32 attention | 6 layers a block; 1 sliding / 3 full / 1 MLP query |
| `short_conv` | `Inkling2LoopShortConvolution` :384-438: static taps; `(x + res) / sqrt(k + 1)` | BF16 tap table; f32 accumulation | `sconv_kernel = 4` |
| `dynamic_short_conv` | `Inkling2LoopDynamicShortConvolution` :440-500: taps are `static_weight + weight_proj(input)`; `(x + res) / sqrt(2k + 1)` | BF16 static and projected taps; `float()` weights and accumulation | `sconv_kernel = 4`; only the four full layers |
| `moe` | `Inkling2LoopSparseMoeBlock` :857-870 and `TopKRouter` :1195-1218: sigmoid score, top-8, correction bias, `route_scale` 8 | experts NVFP4 (layers 0-19) and MXFP8 (20-23); BF16 router weight, f32 bias, f32 routing | 128 routed, 8 active, 1,152 intermediate |
| `loop_message_gate` | `Inkling2LoopMessageGate` :81-99: unweighted RMSNorm of the concatenated hidden and message, down to rank 64, two `tanh` gates | the checkpoint stores BF16 where the producer constructs f32 parameters; the forward casts to f32 and returns f32 | rank 64, four streams a layer |
| `cross_pass_attention` | `Inkling2LoopMessageAttentionResidual` :147-176: unweighted RMSNorm over the previous pass's summaries and final state, learned queries, scale `hidden**-0.5`, not causal | f32 query bank, f32 states | 24 queries (two per core layer per bank) |
| `moe` (MXFP8) | as above | E4M3 weight, ue8m0 scale every 32 columns | layers 20-23 |
| `vision_*` | the Inkling-2-Small tower (`src/vision.rs`), unchanged in this release | BF16 | width 1,024, 23 blocks, merge 2 |
| `mtp` | the DFlash stage: `*_ctx` projections and the Markov head | BF16 | 3 stages, Markov rank 256 |

Two open type questions are recorded here rather than guessed into the layout. The message gates'
checkpoint tensors are BF16 while the producer builds those parameters in f32, so the body must
cast exactly where the producer's `forward` does rather than keeping the parameters f32. And the
checkpoint splits the producer's `(12, 4, hidden)` cross-pass query bank into two 24-row tensors;
which two streams each bank holds is a conversion fact, not a config fact.

## The loop in the arm

The arm's typed configuration parses this release unchanged (geometry is config-driven): the
`mini` test in `crates/sparkflash-model-inkling2/tests/mini.rs` pins 24 layers, 24/8 heads,
128 experts, three drafter stages and the 23-block tower. The shared `Architecture` now carries
`loop_plan`, and the arm reads it into `Inkling2Config::loop_plan`.

The f32 host reference implements the schedule (`forward.rs`):

* `cache_slots()` is `num_hidden_layers + (passes - 1) * core_len`, and `cache_slot(layer,
  pass)` maps a core layer's later pass onto the appended copy, matching the producer's
  `cache.layers += copy.deepcopy(loop_layers)`;
* the token schedule is prelude, core `passes` times, coda; a pass after the first restarts from
  the prelude's hidden state and block stack and reads the previous pass's summary the way
  `:1419` does (`block_states[num_pre_loop_blocks + 1:]` plus the pass's final hidden);
* `cross_pass` runs the four learned queries over those states (unweighted RMSNorm, no mask,
  `hidden**-0.5` scale), and `message_gate` applies
  `(1 - tanh(forget)) * source + tanh(gain) * message` per stream;
* the tensor plan names the 12 controller tensors a core layer and the two cross-pass banks, so
  the converter writes them.

On the real checkpoint (revision `c9d0283`) `sparkflash convert --checkpoint DIR
--tp-world-size 1` writes one 18.9 GB rank file with 828 weights, verified slot by slot against
the checkpoint, and `examples/prefill` in the arm runs the three-token prompt `[1, 100, 200]`
through the f32 reference end to end (finite logits; top scores 26.38, 25.66, 25.57). The
conversion path therefore covers the loop tensors, the drafter and the tower at this geometry.

The split of the producer's `(12, 4, hidden)` query bank into the checkpoint's two 24-row
tensors is the one mapping this work states without tensor evidence: the layout and the host
reference place the query and MLP streams in `middle_cycle_cross_pass_queries` and the key and
value streams in `middle_cycle_cross_pass_kv_queries`. A producer capture disambiguates it; a
wrong guess changes the messages, not the shapes, so it must be checked before numerics v1.

## Work list

| Row | Stage | State |
| --- | --- | --- |
| `relative_attention` 3,072x2,304 | 3 | resolved: the Inkling bodies take the row width as a launch word, so the catalogue rule now states both released widths (2,304 and 4,608) instead of one constant |
| `short_conv` 2,304x1x4 | 3 | resolved the same way |
| `block_attention_residual` 2,304 f32 | 3 | resolved the same way |
| `moe` 128x2,304 NVFP4 and MXFP8 | 3 | resolved: the expert bodies take the rank's run as a runtime argument; the rule states both widths |
| `vision_merger` 4,096x4,096 | 3 | **open**: the merger resolves at the tensor's 4,096 output, but the plan's `out_hidden` for this release is the 2,304 text width (the config states no `out_hidden_size`), so the row refuses as "output 2304 is not one of [4096, 4608]". The fix is the vision geometry's merger output, not a catalogue line |
| `loop_message_gate` | 4 | done: `inkling_message_gate` (`kernels/inkling_message_sm121.cu`) runs the two tanh gates per row; the checkpoint's BF16 controllers are widened once at load into device-resident copies |
| `cross_pass_attention` | 4 | done: `inkling_cross_pass` runs the four queries a core layer over the previous pass's summary rows and final hidden, softmaxed and scaled as the producer writes them |
| `dynamic_short_conv` | 4 | done: the released body already implements the projected taps and `sqrt(2k+1)`; the part is now catalogued as its own kind |
| the loop schedule | 4 | done: the device plan builds the 48 applications with per-pass caches and the prelude rewind, and `MessageRead`/`MessageGate` launch the two new bodies instead of copying rows to the host |
| the drafter | 4/7 | loads at this release's geometry (4 taps, 4 fusion rows) and its rounds run (12 proposed / 11 accepted, then 15 / 9); the checkpoint's `mtp.safetensors` names 54 tensors the shared plan spells differently, which the scan lists and the drafter's own loader maps |
| stages 6-8 | 6-8 | prove, capture, convert, serve, register numerics |

## Checklist

```text
[x] 0  revision, license, producer reference, formats, targets, widths recorded
[x] 1  scan report at every target and width in the model doc; every row has a stage
[x] 1  every resolved part computes the producer's function (rule 2 check)
[x] 2  layout: every key read or ignored with a reason; every tensor claimed or unloaded
[x] 2  every fact cited to a producer line, config path or tensor
[x] 2  rule-2 table (function, types) in the model doc
[ ] 2  layout equality at TP 1, 2, 4 (the helper's oracle is the arm crate, which does not run this release)
[x] 3  the five runtime-shaped rows resolve at both released widths (the catalogue rules state 2,304 and 4,608); the vision merger's output width is the one open row (work list)
[x] 4  new part kinds named by operation: host body in the f32 reference, device body launched by the walk (MessageRead/MessageGate)
[x] 4  the f32 host reference runs the loop, its gates and the cross-pass read (tiny looped fixture test)
[x] 4  kernels by operation with their ABI (inkling_message_gate and inkling_cross_pass), exported by the Inkling image and checked against the catalogue
[x] 4  the looped device fixture walks every application; the real checkpoint matches the f32 reference's top logits
[ ] 5  target port complete for the whole work list
[ ] 6  synthetic variant exercises every split at worlds 2 and 4
[ ] 6  device arm exact ids at worlds 1, 2, 4 (or an owner-approved gate, measured)
[ ] 6  conformance suite passes
[ ] 7  tolerances written first; producer capture on synthetic and real; towers
[ ] 8  converted from the description; numerics v1 registered; check and ci pass
[x] 8  served gate: the HTTP API answers a chat completion and an image request on .100
[x] 8  the drafter's rounds accept (12 proposed / 11 accepted and 15 / 9 on two requests)
[ ] 8  served gate: speed table; long-window probe
[x] 8  usage.prompt_tokens_details.image_tokens reports the prepared spans (the host driver no longer hard-codes zero)
[ ] 9  template, parsers, sampling, preprocessing, drafters from the checkpoint
[ ] 10 open-manifest-ab: DeepSeek identical and no slower; production restored
[x] 11 the onboarding branch carries the work signed off; DECISIONS/NOTICE and the end-of-run battery stay with the merge
```