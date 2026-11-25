# Milestone 4: GLM runs on the base, and its crate is deleted

Status: work breakdown, 2026-11-25. It details milestone 4 of
[05-runtime-models.md](05-runtime-models.md) (section 5.2, "Then the models move"). After
[milestone 3](07-base-ranks-formats-seam.md) the base runs at one, two and four ranks, takes FP8
tiles, BF16, NVFP4 and row-scaled MXFP8, and runs a foreign part through the seam, all proved on
synthetic checkpoints. This milestone moves the first real model onto it. GLM-5.3-Flash runs on
DeepSeek's forward from its description. The base supplies the engine: the tick, the lanes, the
plan walk, the seam and the kernel templates. It does not supply GLM's arithmetic. Every GLM part
computes what the producer's reference computes, in the types the producer computes it in, either
as an instance of a shared body or as a shared part kind of its own. The move reorders sums and
puts each value in the producer's type, and that is GLM's one declared bit change. The vision
tower runs as a stack. Then `crates/sparkflash-model-glm5-next` is deleted.

Line numbers are the tree's at `m3/finish` `bdeb563a`.

## 1. What the tree says today

### 1.1 The model against the base

| | GLM-5.3-Flash | the base (DeepSeek V4.1 Flash) |
| --- | --- | --- |
| Layers | 45: 34 KDA, 11 sparse MLA (every fourth, from 3) | 40, all MLA |
| Hidden, vocabulary | 4,096; 154,880 | 5,120; 129,280 |
| Attention | V3-form MLA: `q_a`/`q_b`, `kv_a_proj_with_mqa`, `kv_b`; NoPE (`qk_rope_head_dim` 0); 64 heads of 256; q-LoRA 1,536, kv-LoRA 512; plain `o_proj`; no sink, no window | V4 MLA: one 512-wide latent (`wkv`), 64 rotated dimensions, a sink per head, a 128-position window, CSA2 compression, grouped `wo_a`/`wo_b` |
| Selection | indexer of 32 heads of 128 over raw positions, top-2,048, keys pooled by 4 with a gate and a position table, LayerNorm key norm with a bias, tail always selected | indexer over the compressed keys, top-512 |
| Recurrence | KDA in 34 layers: 64 heads of 128, short convolution 4, per-channel forget gate | none in the release (GDN on the synthetic only) |
| Residual | mHC, `hc_mult` 4, 20 Sinkhorn steps. The mix weight is BF16 and reads an unquantised input. The norm epsilon is `rms_norm_eps` (1e-5). The head folds the streams by their mean | mHC on the same schedule. The mix weight is F32 and reads a quantised input. The norm epsilon is 1e-20. The head folds with `hc_pre` and the last pre-mix |
| Types | The producer's dtype is BF16: streams and module outputs are BF16. It casts to f32 inside the norms, the mix coefficients and Sinkhorn, the router, the KDA decay, recurrence and state, the indexer's scores and pool softmax, and the attention softmax. The crate serves f32 throughout, which is wider than the producer | BF16 module outputs as its reference casts them, and a BF16 wire |
| Input to an FP8 linear | `activation_scheme: dynamic`: E4M3 codes on 128-column tiles, the scale `amax / 448` in f32 with no power-of-two rounding. A BF16 linear reads its input unquantised | the release's W8A8 quantiser on the BF16 module output, with power-of-two scales |
| FFN | 3 dense (12,288), then 42 MoE: 288 experts, top-8, one shared, sigmoid `noaux_tc` with a correction bias, scale 2.5, `swiglu_limit` 10 | MoE: 384 experts, top-6, `sqrtsoftplus` with two biases |
| Stored format | FP8 E4M3 on 128 x 128 tiles with f32 inverse scales (dense, shared and routed); BF16 KDA, `kv_b`, indexer, embedding and head | MXFP8 dense on 32-tiles, MXFP4 experts |
| Draft | one V3-form MTP layer (`enorm`, `hnorm`, `eh_proj`, a sparse-MLA and MoE block, shared head); DFlash 2 block drafter when the profile names one | the DSpark stage (`main_proj`, Markov head, confidence head) |
| Vision | 24 blocks of 1,024 (16 heads of 64), qkv bias, q/k RMSNorm, axial rotary, 3-D patch embed (14, two frames), 2 x 2 downsample convolution to 4,096, merger proj, LayerNorm, GELU, gated SwiGLU at 10,240 | 32 blocks of 1,024, intermediate 2,816, 3 x 3 unfold, aligner to 5,120 |
| Release | 328.3 GB FP8, 82 GB a rank at four ranks | 72 GiB a rank at four ranks |

The release fits at four ranks only (164 GB a rank at two). Worlds 1 and 2 are proved on the
synthetic checkpoint, as in milestone 3.

### 1.2 What the crate is

* **Size.** About 21.5k lines: `device.rs` 7,847, `walk.rs` 1,365, `forward.rs` (the host
  reference) 1,478, `config.rs` 1,335, `tensor.rs` 1,014, `rank.rs` 833, `codec.rs` 787,
  `serve.rs` 601, `convert.rs` 314, the tests 2,324, and `config/` (the profile and an HF config).
  It does not depend on the base.
* **Kernels.** Five GLM-only sources, 2,644 lines: `glm_decode_sm121.cu` 1,563,
  `glm_attention_sm121.cu` 526, `glm_prefill_sm121.cu` 368, `glm_moe_sm121.cu` 132,
  `glm_common_sm121.cuh` 55, built as `Image::GlmDecode`, `GlmPrefill`, `GlmAttention` and
  `GlmMoe` (`sparkflash-kernels/build.rs:762-781`). Images shared with Qwen: `qwen4_exp`,
  `qwen4_scan`, `qwen4_moe`, `moe_plan`, `dflash2`. None of them is on `kernels/instances.toml`.
  Inkling loads `glm_prefill_bf16_wide` from `GlmPrefill`
  (`sparkflash-model-inkling2/src/device.rs:923-958`).
* **Bodies GLM already shares.** `shared_prefill_mma_body`, `shared_prefill_q8_body` and
  `shared_prefill_mma_rows_body` (`shared_prefill_mma_sm121.cuh`). Milestone 3's F3 made these the
  base's FP8-tile bodies (`operand.rs:619` claims `glm_q8_body` for `FP8_E4M3_TILE_128`). Also
  `shared_moe_router_body`, the indexer pool and score bodies (`q4e_indexer_pool_body`,
  `q4e_indexer_score_body`, also in `qwen4_attn`), and `GpuDflash2`
  (`sparkflash-ops/src/dflash.rs`).
* **Projections.** Decode runs the tiled row GEMV (`gemv_small`, `device.rs:4152`). The FP8
  few-row MMA body was measured slower and retired: numerics version 3,
  [v3-gemv.md](../numerics/glm53/v3-gemv.md). Past eight rows the bulk prefill runs the wide BF16
  tile and the FP8 128-tile MMA (`bulk_gemm`, `device.rs:3882`, `WidePolicy::GLM`).
* **Attention.** Absorbed. The query is projected into the latent once a row. Decode runs four
  launches (scores, softmax, mix, project, `glm_decode_sm121.cu:406-624`), and the bulk prefill
  runs `glm_prefill_mla_attend` and the per-head MMA. `kv_b` is dequantised to f32 at load. A
  BF16 `kv_b` gave the same bits but was slower in serve, and was reverted. The served latent
  cache is E4M3: 512 codes and four f32 scales a position, 528 B. With the indexer's keys and
  pool, that is 18.2 GiB a rank at the 1,048,576-position window.
* **KDA.** Fused input projections, short convolution, decay, then a chunked scan for prompts
  (`q4p_kda_prep`, `q4p_kda_scan`) and a column recurrence for tokens (`q4e_gdn_recur_columns`,
  with the output norm and gate fused). Split by heads: q/k/v, `f_b`, `g_b` and `b` by rows, and
  `o_proj` by columns. `f_a` and `g_a` are whole.
* **State.** Each sequence holds positional regions on lane memory (latent, scales, index keys,
  gates, pool) and plain allocations for each KDA layer (conv history, f32 state, a spare for
  verify). Rewind is a `Checkpoint` over the KDA regions, the MTP hidden and the drafter rings
  (`device.rs:2551`, `3510`, `3539`). Prefix reuse is all-or-nothing (`rank.rs:475`).
* **Collective.** Partials on an f32 wire, summed in rank order. One-shot below 16 rows,
  two-shot from 16. NCCL by default, TCP on request. The served group adds the doorbell RDMA
  exchange (`sparkflash-cli/src/serve.rs:4178-4243`).
* **Serving.** `GlmWalk = PlanWalk<Shared>` on `sparkflash-runtime`, four fused lanes by default
  on `LanePool`. It drafts with MTP, or with DFlash 2 when the profile names a drafter, at an
  adaptive depth. The serve route is `Route::Pack`: `struct Glm` in
  `sparkflash-model/src/arms.rs:546-615`,
  `GlmArm` and `TYPED_PACKS` in `sparkflash-cli/src/serve/pack.rs:406-608`. The prompt codec is
  `GlmCodec`, and the output dialect is `GlmResponseEngine` (`sparkflash-api/src/glm.rs`, outside
  the crate).
* **Weights.** A converted rank file in the walk's upload order (`SlotPack`,
  `sparkflash_weights::rank_file`). It loads in 13-14 s a rank. It is not the base's
  `sparkflash_device::converted` format.
* **Records.** The conformance suite runs on `GlmWalk` (`walk.rs:1364`). The device tests compare
  against the host twin at worlds 1, 2 and 4 (`tests/device_forward.rs`). Numerics: registry
  config world 4 at version 3, reference `numerics/references/glm53-tp4-sm121a.json`, anchor
  `glm53-anchor.json`, plus coverage and monitor entries.
* **Speed on record** (the milestone-3 A/B, four nodes, `scripts/served-gate.py`):

  | | alone | four at once |
  | --- | --- | --- |
  | greedy lanes | 44.5-45.2 tok/s | 55.5-56.5 tok/s each |
  | sampled lanes | 38.6-38.8 tok/s | 46.2-47.4 tok/s each |

  Two earlier records: bulk prefill at 979-993 tok/s on a 6.6k-token prompt (D101), and DFlash 2
  at 80.7 tok/s on one stream at the 1M window, with 90.3 % of drafts accepted (2026-11-05).
* **Served gates.** Lanes and interleave pass. Turns fail, as on the reference build (`cached 0`,
  section 3.2 of doc 07).

### 1.3 What the base lacks for GLM

* **The value table is V4's.** `values_from_plan` (`plan.rs:102-226`) requires `o_groups`,
  `o_lora_rank`, `rope_head_dim`, `rope_theta`, the YaRN factors, `compress_ratios`,
  `compress_rope_theta`, the
  candidate pool's three values, `dspark_markov_rank`, `gate_temp` and the Engram facts. GLM
  states none of them. It has facts the table cannot hold: `kv_b`'s widths, the KDA geometry,
  the indexer's pool and tail rule, and the mix weight's format.
* **No KDA body.** `forward_has_body` (`plan.rs:1583`) lists `GdnScan` and not `KdaScan`. The
  layout plans `kda_scan` (`engine_plan.rs:1820`), and its catalogue entries live in the Qwen
  images (`parts.rs:270`, `q4p_kda_*`).
* **The recurrent split is a rule, not a fact.** `whole_recurrent_block`
  (`engine_plan.rs:1712`) makes any GDN block in an MLA stack whole on every rank. That suits the
  synthetic `gdn-*` variants. GLM's 34 KDA layers split by heads, and a whole KDA on every rank
  would run four times the work in three quarters of the layers.
* **The MLA body is V4's.** It has no `kv_b` absorb or project, no NoPE instance, no mode
  without a sink, window and compression, and no plain `o_proj`. The attention core itself is the
  same operation: a 512-wide MQA latent over selected positions.
* **The indexer is V4's.** GLM's keys come from the hidden state through a LayerNorm with a
  bias, are pooled by four with a gate and a position table, and always include the tail. The
  shared role still says `not_moved("pooled indexer", ...)` (`roles.rs:2048`, `:2062`).
* **The mix.** The shared role still says `not_moved("manifold mix", ...)` for GLM's `mix_in`,
  `mix_out` and head fold (`roles.rs:1901-1916`). The base's mHC body is DeepSeek's instance:
  an F32 mix weight on a quantised input, a norm epsilon of 1e-20, BF16-rounded streams
  (`hc_pre_bf16`, `hc_post_bf16`), and a head fold through the last pre-mix. GLM's differs in each
  (`sparkflash-weights/src/mhc.rs`, `forward.rs:1105`).
* **Types and recipe.** The base rounds module outputs to BF16 and reduces on a BF16 wire. That
  is GLM's producer dtype too, so GLM takes it (rule 2). What the base lacks is GLM's own f32
  casts where they differ from DeepSeek's, and GLM's activation recipe: the base quantises an FP8
  linear's input with DeepSeek's W8A8 quantiser, and GLM's release uses its own
  (`quant::quantize_activation_fp8_tiles`).
* **Fusion across foreign blocks.** S2 runs a foreign block by landing the residual and calling
  parts. Milestone 3 left the frame that carries the pending result across the boundary for
  later, as
  a speed step to measure when a model moves (07 section 4, S). With 34 of 45 blocks foreign,
  that measurement is now GLM's.
* **Chunked prefill through a foreign part.** S2 runs a stack with a foreign part one prompt
  chunk at a time. GLM's bulk prefill (979-993 tok/s) is its prefill lever and has to survive the
  move.
* **The draft head.** The base's `Mtp` is the DSpark stage. GLM's MTP layer is a full sparse-MLA
  and MoE block with its own norms and projection. The base has no DFlash 2 attachment.
* **The tower is DeepSeek's.** `sparkflash-kernels/src/vision_tower.rs:25-66` fixes the width,
  32 layers, the intermediate width, the 3 x 3 unfold and the aligner to 5,120.
  `sparkflash-vision` is DeepSeek's image front end. GLM's tower has never run: it is planned in
  `tensor.rs:652` (1.13 GB, "planned but not served"), and the walk's `Vision` is `NoVision`
  (`walk.rs:495`).
* **Weights.** The base loads `sparkflash_device::converted`, and GLM writes `SlotPack` files.
* **Names outside the crate.** Twenty-odd files `include_str!` the crate's `config/` files
  (`sparkflash-model`: `arm.rs`, `compose.rs`, `tp_plan.rs`, `coverage.rs`, `architecture.rs`;
  `sparkflash-ops`: `widen.rs`, `tensor_plan.rs`, `parts.rs` and four tests; `sparkflash-web`:
  `fit.rs`, `hf.rs`; `sparkflash-cli`: `convert.rs`, `serve/pack.rs`, the serve tests). GLM policy
  is named in `sparkflash-ops` (`images.rs`, `plan.rs:792-941`, `projection.rs`, `roles.rs`,
  `attention.rs`, `epilogue.rs`, `resolver.rs`, `tower.rs`, `engine_plan.rs`). The profile has
  `SPARKFLASH_GLM_*` aliases (`profile.rs:643-694`). The CLI has `glm_tp4.rs` (782 lines),
  `open_glm_tp4_rank` and `open_glm_lanes`. Scripts: `serve-four-node.sh`, `one-host-group.sh`,
  `owed-cluster-window.sh`, `onboard-gate.sh`, `numerics.py`. The `qwen4-exp` tests exercise GLM
  kernels.

## 2. The rules

1. **Nothing changes for the released DeepSeek** unless the change is declared under rule 3.
   Every step is held to `scripts/open-manifest-ab.py`: manifest, launch trace, tick trace, speed.
2. **GLM computes what its producer computes, in the producer's types.** The base is the engine
   and the template library, not the arithmetic. What a GLM part computes is fixed by the
   checkpoint's config, its tensor index and the producer reference the layout cites
   (`modeling_glm5_next.py`).
   * **The function is GLM's.** The pool rule and the always-selected tail, the router rule, the
     head fold, the clamps and every epsilon come from the description. A part the model was not
     trained with is never substituted for one of its own: no CSA2, no sink, no window, no
     rotation, no grouped output. Where the base's body computes a different function, GLM's is
     its own instance or its own part kind. A GLM fact is never changed to fit a body.
   * **Each value has the smallest type that is faithful to the producer's calculus** (the
     owner's decision, 2026-11-25). That is the type the producer computes it in:
     * BF16, the config's `dtype`, for the streams and every module output. The stream update
       itself is BF16: the producer casts `post` and `comb` to BF16 before it applies them.
     * f32 where the producer casts to it, and nowhere else: the norms' mean square, the mix
       coefficients and the Sinkhorn, the router's logits (`moe_router_dtype`), the KDA decay,
       recurrence and stored state, the gated output norm, the indexer's scores and its pool
       softmax, and the attention softmax.
     * the release's activation recipe into an FP8 linear, on the BF16 module output. A BF16
       linear reads the BF16 output as it is, so a tensor-core body on BF16 activations computes
       the model's own operands.

     The crate serves f32 throughout, wider than the model's calculus. Version 4 puts each value
     in the producer's type. Nothing goes narrower than the producer's type. The E4M3 latent
     cache is the one exception, declared at version 1 and kept as served (A2).
   * **The wire.** A rank's partial is a piece of a sum the producer rounds once, at the module
     output. BF16 is the expected wire, as on the base. It is confirmed by measurement: world 4
     on a BF16 wire against the same sum taken in f32 and rounded once, on the synthetic
     checkpoint and at the real capture. If the two are apart by more than a reorder's bar, the
     wire is f32 and the sum is rounded once.
   * **The order of f32 sums may change:** a different split-K, a tile, a fused launch, the rank
     fold.
   * **Version 4 is the reorder and the types together,** declared in the numerics registry with
     the owner's decision recorded in [DECISIONS.md](../DECISIONS.md). It is checked by:
     * coherent output on the served gate's prompts, greedy and sampled;
     * the producer comparison of section 3. This is the check that decides faithfulness:
       version 4 is no further from the producer than version 3 is, and is expected closer;
     * teacher-forced drift against the version-3 reference and the anchor, inside a bar written
       here before the move. The ceiling is top-1 agreement of at least 98 % and a mean absolute
       NLL delta of at most 0.02 nat: half of the E4M3 cache's distance, the smallest trade GLM
       already serves (96.8 %, 4.3e-02 nat). Tighter numbers come from the synthetic first, where
       the host reference runs both type maps and the types' share is measured alone.

     For scale: Qwen's TP floor is one flip and a mean KL of 2.3e-04, and Qwen's tensor-core
     decode moved 7.0e-04 nat at 99.39 %. GLM's version 2 measured 93.4 % and 0.137 nat, and its
     cause was never separated. One ulp can move an activation code at a quantiser boundary, so a
     reorder may cost more on GLM than on Qwen. If the synthetic shows the ceiling cannot be met,
     the number and its cause go to the owner before the move. The ceiling is not raised
     afterwards. The NUMERICS.md card bars (0.90, 1.0 nat) would pass every number above, so they
     are not this move's bar.

   A change that claims the same arithmetic stays bit-equal on the same inputs: the KDA body, the
   DFlash 2 drafter, the converted file.
3. **The faster body is the one copy.** Before each operation moves, GLM's body and the base's
   are measured at every row count GLM serves (1, 4, 8, prompt chunks), and at the positions that
   matter for attention (2k, 13k, 100k, 1M). The faster one is kept. Only bodies that compute
   GLM's function on the producer's operand types compete (rule 2). A faster body that narrows
   an operand below the producer's type or computes a neighbouring function is not a candidate.
   * If GLM's body is faster for an operation DeepSeek runs, it becomes the shared body. DeepSeek
     then takes its own one-time change under the same check: a numerics version, an A/B, and
     never slower.
   * A body that is faster for one model and slower for the other stays two instances of one
     template, chosen by the speed table. It is never two sources.
4. **GLM is no slower on the base.** Alone, four at once, greedy and sampled, bulk prefill at
   6.6k and at 100k, the first token at the 1M window, and DFlash 2 acceptance, against the crate
   build on alternated fresh boots. A row that loses and cannot be recovered stops the move, and
   the measurement is brought to the owner.
5. **No model fact in Rust.** GLM's facts live in `models/layouts/glm5-next.toml` and the
   checkpoint. The base's value table reads a fact only when a part that reads it is planned. A
   rule that decides a split from what a stack contains (`whole_recurrent_block`) becomes a split
   the layout states.
6. **No model name in a kernel file, symbol or image.** A GLM body that survives is renamed by
   its operation and goes on the instance list. A GLM body that loses is deleted.
7. **Refuse by name.** A GLM option the base does not apply is refused at open and names its
   replacement. That covers every `SPARKFLASH_GLM_*` switch, `latent_cache`, the drafter choice,
   `--concurrency` and video.

## 3. Proof

* **The GLM-shaped synthetic checkpoint.** The second-shape writer (`scripts/second-shape.py`)
  gains a `glm` variant: KDA blocks split by heads, absorbed NoPE MLA every fourth block, the
  pooled indexer, a dense prefix, the `noaux_tc` router, FP8 tiles with f32 scales, BF16 `kv_b`
  and KDA, the V3 MTP layer, and a tower at small widths.
  * Before the crate goes, its host reference (`forward.rs`) moves into the shared host
    references (`sparkflash-weights`, beside `gqa.rs`). The variant is compared with it at each
    world's own split (07 section 3), at worlds 1, 2 and 4, on the device arm.
  * Exact ids are the bar. The tie band applies only where 07 section 3 already admits it (a
    device kernel the host cannot reproduce), with the band measured or stated as there.
* **The producer reference, for the text model.** The crate, its host reference and the base
  all read the producer through one reading, ours. Two captures check that reading from outside
  it (plan 05 section 5.6):
  * On the `glm` synthetic variant, the producer's own classes (`transformers` at the revision
    the layout cites, at the config's BF16) run the tiny checkpoint. The shared host reference,
    on the producer's type map, matches their logits inside a tolerance written before the run.
    With the types aligned, that tolerance is a reorder's. This checks the wiring part by part:
    the mix and its epsilons, the pool and the tail, the router, the head fold, the activation
    recipe, and where each f32 cast sits.
  * On the real checkpoint, a producer-side stack that serves it (vLLM has, on these four nodes)
    is teacher-forced on the anchor's tokens once, before the move. That distance is recorded
    for the crate build. The base build must be no further from the producer than the crate
    build is. If no producer-side stack can be stood up, that is said here, and the synthetic
    capture stands alone.
* **The crate against the base on the same checkpoint.** On the synthetic checkpoint the crate's
  device arm and the base's are run teacher-forced. This is the drift check's small form: it
  measures each operation's change in isolation, and the version-4 bars are set from it. A part
  whose drift is larger than a reorder of its sums and the producer's types can explain is a
  changed function or a wrong type, and it is fixed, not declared.
* **The suite.** `conformance!(DeepSeekBase)` passes on the `glm` variant: whole = chunked =
  token with a long prompt, continued = cold through a KDA save point, rewound = cold, drafted =
  plain with MTP and with DFlash 2, alone = four, and TP 1 = 2 = 4.
* **The real checkpoint, four nodes.**
  * The version-4 numerics capture against the version-3 reference and the anchor, inside rule
    2's bars.
  * The served gate on the base build: lanes, sampled lanes, interleave and turns. Turns fail
    today on every arm. The bar here is no new failure, and the same failure as the crate build.
  * Rule 4's speed table, the base build against the crate build on alternated fresh boots.
  * The 1M window with the E4M3 latent: the window allocates beside the weights, and the
    5,976-token secret-number probe (the indexer selecting outside the last 2,048) answers as it
    does today.
* **The tower.** Its output is compared against the producer reference
  (`transformers` `modeling_glm5_next.py`, the revision the layout cites) on a fixed set of
  images, through the reference-capture script (plan 05 section 5.6), inside a tolerance written
  before the run. Then image chat on the served model, greedy and sampled.
* **The released DeepSeek.** One rule-1 window at the end of each line that touches a shared
  body (A, M, D), and the end battery.

## 4. The work

### V. The values and the description (first, one owner)

* **V1. The value table reads by part.** A fact is required only when a part that reads it is in
  the plan. Today's V4 facts become those parts' facts: the rope and YaRN facts for a rotated
  MLA, the sink for an MLA with sinks, `o_groups` and `o_lora` for the grouped output, the
  compression and candidate pool for CSA2, `dspark_*` for the DSpark stage, Engram's for Engram.
  GLM's facts come from the layout: the MLA form (absorbed V3 or V4 latent), `kv_b`'s widths,
  the KDA geometry, the indexer's pool, gate, position table, key norm and tail rule, the mix
  weight's format and its norm epsilon, the head fold, the router rule and its accumulator
  (`moe_router_dtype`), and the activation recipe (`quantization_config.activation_scheme`).
  The stream type is the config's `dtype`. The f32 casts are the layout's, each cited to the
  producer reference's line, since no config key states them.
* **V2. The split is data.** The layout states each recurrent block's split. GLM's KDA splits by
  heads, and the `gdn-*` synthetic variants state whole. `whole_recurrent_block` is deleted, and
  the test that holds every head on every rank reads the stated split.
* **V3. The facts leave the crate.** `config.rs`'s typed parse (1,335 lines) is replaced by the
  layout reader. Every `include_str!` of the crate's `config/` moves to a fixture under
  `models/` or the test tree.
* Done when the base opens the real GLM checkpoint's description at worlds 1, 2 and 4, and the
  scan lists exactly the rows the lines below close. Rule 1 holds.

### A. The sparse layers

* **A1. Absorbed MLA on the base's attention.** The base's MLA gains the V3 form: the `q_a`/`q_b`
  projections, the `kv_a` latent with its norm, the absorb (the query through `kv_b`'s key rows),
  the project (the mixed latent through `kv_b`'s value rows) and a plain `o_proj`.
  * The core (scores, softmax and mix over the 512-wide latent at the selected positions) is one
    operation. The base's body at GLM's instance arguments (no rotated dimensions, no sink, no
    window, no compression, top-2,048, 16 heads a rank, scale 1/16) is measured against GLM's
    four-launch split at 1, 4 and 8 rows and at 2k, 13k, 100k and 1M positions. The faster one is
    the shared body (rule 3).
  * `kv_b`'s operand form (f32 at load, or BF16 widened in the kernel) is measured again on the
    winning body. BF16 was slower in serve on GLM's body for reasons outside its own code (the
    model doc, *Context*).
* **A2. The latent cache.** The E4M3 latent with four f32 block scales a position is a
  positional lane region, beside the base's own cache form. It is a serving trade the release
  does not make (96.8 % top-1 against the anchor, [v1](../numerics/glm53/v1-e4m3-window.md)), so
  it is chosen by the profile, not stated by the layout as a model fact. It stays the default,
  as served today. The f32 latent stays selectable through the shared switch: it is the anchor's
  arithmetic and the parity form for tests. Prefix adoption, the slide and the 1M window work on
  it as on the base's.
* **A3. The pooled indexer as a part kind.** GLM's keys (`wk` from the hidden, LayerNorm with a
  bias), the gated pool of four with its position table, the always-selected tail, top-2,048,
  and the MTP's shared selection (`index_share_for_mtp_iteration`).
  * The arithmetic is the producer's, as the crate runs it today on the shared pool and score
    bodies (`q4e_indexer_pool_body`, `q4e_indexer_score_body`): a pooled key is a per-channel
    softmax over the pool's four tokens of gate plus position, the pool covers the index keys
    only, and attention still reads the raw positions of the chosen pools. Those bodies are the
    part kind. CSA2 is not used: it pools the latents themselves, only DeepSeek V4.1 was trained
    with it, and GLM's selection is never reshaped to it.
  * The `not_moved("pooled indexer")` stubs go.
* **A4. The bulk prefill attention.** `glm_prefill_mla_attend` and the per-head MMA against the
  base's prompt attention, under the same rule.
* Done when the `glm` synthetic variant's sparse layers agree with the host reference at worlds
  1, 2 and 4, and the measurements are written here.

### K. KDA through the seam

* **K1. One KDA body.** `ops::roles::KdaLayer` sits beside `GdnLayer`, built from GLM's `kda`,
  `scan` and `recur` (`device.rs:4579-4812`). Its kernels already live in the shared Qwen images
  (`q4p_kda_prep`, `q4p_kda_scan`, `q4e_kda_decay`), and the convolution and the column
  recurrence (`q4e_gdn_recur_columns`) are the ones GDN runs. The host reference is the shared
  one in `sparkflash-weights/src/linear_attention.rs` (`kda_channel_decay`), and the token
  convolution and gated norm launchers in `sparkflash-ops/src/gdn.rs` (`TokenConv`, `GatedNorm`)
  already serve both. `forward_has_body` gains `KdaScan`.
* **K2. The split.** KDA runs by heads at world W (V2), its `o_proj` partial goes through the
  base's reduce, and `f_a`/`g_a` stay whole.
* **K3. State.** S3's recurrent region holds each KDA layer's conv history and f32 state. The
  save point, the verify snapshots, the commit and adoption only at a saved point work as for
  GDN. The decoder skip is off for a KDA stack, and the ledger says so. The lane stride for 34
  recurrent layers is measured against the lane memory budget (GLM's concurrency budget is
  24 GiB today).
* **K4. Chunks.** A KDA block takes interleaved prompt chunks: the chunked scan resumes from the
  region, so the S2 refusal of split chunks lifts for KDA. Proved as whole = chunked on the `glm`
  variant.
* **K5. Fusion across the boundary.** The frame that carries the pending result across a foreign
  block, left open in milestone 3, is built if rule 4's prefill or decode rows need it. Its
  measurement is recorded here either way.
* Done when the `glm` variant's KDA blocks agree at worlds 1, 2 and 4. The GDN users of the
  shared kernels keep their bit records.

### M. The residual, the FFN, the head and the sum

* **M1. mHC.** One manifold template, with what differs between the two models as arguments:
  the mix weight's format, whether its input is quantised, the norm epsilon, and the head fold.
  GLM's instance follows the producer: the flattened streams widened to f32, the weightless norm
  at `rms_norm_eps`, the BF16 weight widened exactly to f32 on an unquantised input, the
  coefficients and Sinkhorn in f32, `post` and `comb` cast to BF16, the BF16 stream update, and
  the mean of the streams at the head. DeepSeek's instance is unchanged. The base's fold
  (`hc_pre` with the last pre-mix) is not applied to GLM. The `not_moved("manifold mix")` stubs
  go.
* **M2. Dense MLP and shared expert.** F3's FP8-tile bodies at GLM's widths, with GLM's
  activation recipe: the input is quantised on 128-column tiles at `amax / 448` in f32, as the
  release's dynamic scheme does. The base's own activation quantiser is not applied. For decode,
  GLM's tiled row GEMV (version 3) is measured against a base token body on the same operands,
  and the faster is kept. A body that does not read the release's quantised activation is not a
  candidate (rule 3). For a BF16 linear the BF16 module output is the model's operand, so the
  tensor-core bodies compete there on equal terms.
* **M3. The router.** The base's router takes the rule: sigmoid `noaux_tc` with the correction
  bias, renormalised, scaled by 2.5. The 288-expert, top-8 instance.
* **M4. Routed experts.** The FP8-tile routed route (`routed_format_sm121`) at GLM's widths. TP
  slices the expert intermediate. The bulk `q8` path is kept or replaced by measurement.
* **M5. Embedding and head.** GLM's embedding is whole on every rank and the base's is split by
  rows. The split is the layout's statement, so GLM's file states it. The head is BF16 rows at a
  vocabulary of 154,880 on the base's head and its reduction, with the argmax slices.
* **M6. The sum.** GLM's partials go on the base's BF16 wire and RDMA exchange, which replace
  its f32 wire and its one-shot/two-shot switch. Rule 2's wire check confirms it: against the f32
  sum rounded once, on the synthetic checkpoint first. If it fails, the base's reduce takes the
  wire format as an argument (`ops::tp::Wire` already names `F32` and `Bf16`), and GLM sums in
  f32 and rounds once.
* Done when the `glm` variant's dense, MoE and head parts agree at worlds 1, 2 and 4, and rule 1
  holds for the shared bodies.

### D. Drafting

* **D1. GLM's MTP layer as a draft stack.** Plan 05 section 5.3 says draft heads are stacks the
  scan finds. GLM's is the base's own blocks (sparse MLA with the shared selection, MoE) between
  `enorm`/`hnorm`/`eh_proj` and the shared head. The base's `Mtp` part stays the DSpark stage's
  name, and the stack is described by blocks, not by one opaque part.
* **D2. DFlash 2.** `GpuDflash2` (`sparkflash-ops/src/dflash.rs`) attaches to the base when the
  profile names a drafter. Every rank loads its shard, and the depth is the shared speculator's,
  adaptive. It runs in joint rounds across lanes on the base's verify. Its acceptance and tokens
  a step are at least the crate's.
* **D3. The draft head's E4M3 copy** (`draft_head`, `device.rs:3003`) is kept if measured faster.
* Done when drafted = plain on the `glm` variant with each drafter, and rule 4's DFlash 2 row
  holds on the real checkpoint.

### I. The vision tower

* **I1. The tower as a stack on the base's vision part kinds.** `vision_tower.rs`'s constants
  become instance arguments from the plan: layers, intermediate, out width, patch, frames, merge
  size. GLM's differences each become an argument of the existing body or a part-kind variant,
  never a third copy:
  * the qkv and projection biases;
  * q/k RMSNorm;
  * the 3-D patch convolution over two frames;
  * the 2 x 2 downsample convolution;
  * the merger: projection, LayerNorm, GELU, gated SwiGLU.
* **I2. The front end.** Preprocessing comes from `preprocessor_config.json`: the image side,
  merge and temporal patch. The image and video tokens and their start and end tokens are
  spliced into the prompt, and the codec stops refusing image parts. Video runs through the same
  patch embed and is in scope. If any of it cannot run, it is refused by name, never dropped.
* **I3. Where it runs.** On the rank the base's tower already uses, with the merged rows handed
  to the language stack as the base hands DeepSeek's.
* Done when the tower matches the producer reference inside its stated tolerance, image and
  video chats answer on the served model, and the scan's vision rows for GLM are resolved. This
  is also half of what holds the gate of plan 05 step 1c.

### W. Weights

* **W1. GLM's ranks in the base's converted format**, computed from the description (plan 05
  section 5.4 for this model). The 13-14 s load must not regress. The files live under
  `~/.cache/sparkflash/converted/<model--hash>/`.
* **W2. The synthetic writer.** The crate's `write_synthetic_checkpoint` is replaced by the
  shared writer's `glm` variant.

### S. Serving, and the deletion

* **S1. Served by the base.** GLM serves through the base's path: the tick, lanes to 32, chunked
  prefill, prefix reuse, the RDMA exchange. It does not go through `Route::Pack`, `GlmArm` or
  `TYPED_PACKS`. The `struct Glm` source in `arms.rs` goes, and the descriptor comes from the
  scan. The `SPARKFLASH_GLM_*` aliases are refused by name with the shared switch that replaces
  each one.
* **S2. Prompt and dialect.** `GlmCodec` moves to the shared codec home, named by what the
  template emits. It is the one hand-written codec this milestone carries, and plan 05 section
  5.3 (milestone 6) replaces it with the template engine. `GlmResponseEngine` stays in
  `sparkflash-api` as a parser chosen by the template's tags.
* **S3. Kernels.** Each GLM-only source either supplies the surviving shared body, renamed by
  operation and on the instance list, or is deleted. Inkling's `glm_prefill_bf16_wide` follows
  the renamed body, and the `qwen4-exp` tests of GLM kernels move with them.
* **S4. Numerics.** Version 4 is registered with its report under `docs/numerics/glm53/`. The
  coverage and monitor recorders (`device/numerics.rs`) move to the base's shared recorder. The
  registry's GLM entries name the description, not the crate.
* **S5. The deletion.** The crate leaves the workspace, `sparkflash-cli`'s dependencies, the
  `include_str!` users (V3), `glm_tp4.rs`, the scripts and the docs. The `one_stack` allow-list
  shrinks by it, and the CLI line limit falls.
* Done when nothing in the tree names the crate, and the end battery passes.

## 5. Order

1. V first, with one owner. Everything else opens through it.
2. After V, in parallel: K, A, M and I. W can run beside
   them.
3. D after A and M, because GLM's MTP layer is a sparse-MLA and MoE block.
4. The `glm` synthetic proof at worlds 1, 2 and 4 as each line lands. Then the real checkpoint:
   the version-4 capture and the speed table, batched in one four-node window.
5. S last: serving on the base, then the deletion, then the end battery.

The forward crate has one owner at a time for each of A, K's seam work and D. They touch the
same open path and merge often.

## 6. Out of scope

* Moving Qwen, Inkling, dense or Dory (milestone 5). A shared kernel or body that GLM's move
  touches keeps their bits unchanged.
* The serving and weight paths without model dispatch for the other models (milestone 6).
* The chat-template engine (milestone 6).
* The failures every arm shares: turns reporting `cached 0`, the Qwen interleave crash, the
  Dory d2 flake. GLM's served gate must not add one.

## 7. Stopping

As in milestones 2 and 3:

* A step that cannot be made free for the released DeepSeek is reverted, measured and brought to
  the owner.
* The same holds for a GLM row that rule 4 cannot recover.
* A step that can be made only by changing what GLM computes, or by taking a value below the
  producer's type, is not taken. The measurement is brought to the owner.
* Aborting a step is acceptable. A slower DeepSeek is not, and neither is a slower GLM shipped
  under the name of the move.
