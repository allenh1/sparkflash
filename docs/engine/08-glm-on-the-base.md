# Milestone 4: GLM runs on the base, and its crate is deleted

Status: in progress, 2026-11-26 (`c0793f36`); see *Progress* below. It details milestone 4 of
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

## Progress (2026-11-26, `c0793f36`)

Items 1 to 9 of section 5 are in the tree and measured on four nodes: GLM's text stack runs on
the base from its description, `glm-wiring` passes at worlds 1, 2 and 4 (section 3),
`conformance!(DeepSeekBase)` passes on the `glm` variant (8 of 8), the converter writes GLM's
ranks in the base's format, and the released DeepSeek is unchanged against the pre-milestone-4
build (manifest, launch and tick identical on all four ranks). Nothing is served from the base
yet, and the crate is not deleted: the work stopped where it agreed to (section 7). "Landed"
below means it is in the tree and runs under that proof, and a line that asked for a measurement
names the number it got.

| Line | State | Where |
|---|---|---|
| V1 value table by part | landed | `43fa53db`, `36323243` |
| V2 the split is data | landed | `83f22afe`, `f9782ba0` (resident plan) |
| V3 facts leave the crate | in part: the base reads the description; the crate's own parse and `include_str!`s remain until S5 | `43fa53db` |
| A1 absorbed MLA | landed on the stated walk, and measured against the base's four-launch split: the base's split wins at every sequence, so it is kept (section 3). A7 extends it to 8 rows and to the release's own windows: at 13,312, 100,000 and 1,048,576 positions the split is 22x to 33x faster (section 3) | `cca81db7`, `63e6c220`, `84a98d8d`, `2744c47b` |
| A2 latent cache | landed: `attention.latent_cache` selects E4M3 or f32, a prefix is adopted, the ring slides, and the 1M window is proved in the stated code's own test | `99989c80` |
| A3 pooled indexer | landed | `cca81db7`, `63e6c220` |
| A4 bulk prefill attention | landed and kept: GLM's body beats the base's four-launch split by 2.0x to 7.3x (section 3) | `562e0a5b` |
| K1 one KDA body | landed | `03209028`, `83f22afe` |
| K2 the split | landed | `83f22afe`, `f9782ba0` |
| K3-K5 state, chunks, fusion | landed: state survives kept turns, snapshots and rewind, interleaved chunks walk `whole = chunked`; K5 is measured and no fusion is built (section 3) | `c5591555` |
| M1 mHC | landed | `cca81db7`, `9e079a17` |
| M2-M4 dense, router, routed experts | landed on the stated FFN | `87190912`, `070d13b5`, `1cabde0b` |
| M5 embedding and head | landed on the stated walk | `stated.rs` |
| M6 the sum on the BF16 wire | in part: the host check against the f32 sum rounded once passes; the live RDMA exchange is not proved for GLM (the launcher pins the reduce to NCCL whenever a cubin directory is staged) | `fb0ee21d`, section 3 |
| D1-D3 drafting | landed: the MTP layer is the draft stack, DFlash 2 attaches, the draft head is E4M3, and `drafted = plain` passes on the synthetic | `8fba8f8e`, `2957348b` |
| I1-I3 vision tower | not started | |
| W1 converted ranks | landed: the converter writes GLM's ranks in the base's format from the stated splits, at TP 1, 2 and 4, weights verified (section 3) | `c38c217c` |
| W2 synthetic writer | landed for the `glm` and `glm-wiring` variants | `45ec0a05`, `b716c1a3` |
| S1-S5 serving, codec, kernels, numerics v4, deletion | not started | |
| Proof: synthetic, worlds 1, 2, 4 | `glm-wiring` PASS at all three worlds (largest seam median 1.63e-2, 2.77e-2, 2.65e-2 against the 7e-2 bar); `glm` reported | section 3 |
| Proof: producer capture | PASS on decisive BF16 `glm-tiny` (both expanded and absorbed meet the unchanged 1e-2 / 20-of-20 / 0.02 bar); the GPU run over the release's dynamic FP8 scheme is still owed (A.3), and its two blockers are closed: the checkpoint now carries the tower the producer initialises (`glm-tiny-tower`, zero missing keys) and the capture's inputs follow the model's device | `68074500`, `0f526464`, `edccea24`, `89b08da4`, section 3 |
| Proof: conformance on the `glm` variant | PASS, 8 of 8, under `--ignored` (the form `scripts/onboard-gate.sh` uses). The run that writes and then compares a bit record is still owed: the 2026-11-26 window ran the suite against a stale fixture checkpoint and all eight checks failed at boot (section 3) | section 3 |
| Proof: released DeepSeek unchanged | PASS: pre-milestone-4 `2f99fbe7` against HEAD, manifests, launch traces and tick identical on all four ranks | section 3 |
| Proof: real checkpoint, tower, served gate | not started: the work stopped before the real-checkpoint window, as section 7 fixes | |

Closed fault (`78d009c5`): a forward refused at world above 1 used to be written only to the
rank's evidence file (`tp4-token-r<rank>.json`), because the refusal was reported after the TCP
collective walk, which waits on peers, and the rank logs showed a group that never became
ready. The rank now writes the refusal to its log and stops before the group walk, so a
refused world says why on every rank it refused on.

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
   * **A linear map may be reassociated** (D125, 2026-11-26). The absorbed MLA evaluates the
     producer's `kv_b` expansion in another order and never rounds the per-head keys and values
     to BF16. Nothing is narrowed and no part is substituted, so it is a reorder. It is held by
     the host reference's expanded mode: expanded agrees with the producer at zero or one ulp on
     the same inputs, and absorbed sits within 1e-2 median of expanded. A nonlinear step (the
     softmax, a norm, a selection) is never moved under this rule.
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
  * Exact ids were the planned bar, with the tie band only where 07 section 3 already admits it (a
    device kernel the host cannot reproduce), with the band measured or stated as there.
  * **The split (owner-approved 2026-11-25).** On this fixture the release's arithmetic cannot
    meet that bar, so the proof is split as the crate's own device test was split (its wiring
    test ran with unquantised activations at zero token flips, its release-scheme test held only
    the median seam drift, 4.8e-2 against a 6e-2 bar):
    * `glm-wiring` gates. It is the same stack with `quantization_config.activation_scheme:
      none`, which the shared parser now reads as FP8 weights over unquantised inputs (it read
      every FP8 tile release as dynamic before, whatever the scheme said, and now refuses a scheme
      it does not read).
    * `glm` (the release's dynamic scheme) runs at every world and prints its verdict, but is
      reported, not gated (the script's `REPORTED`). Both arms quantise FP8 inputs on the same
      128-wide tiles, as the producer does, so a flipped BF16 ulp can move an E4M3 code or a
      tile's scale. Each FP8 FFN multiplies the relative difference about 12 times (layer 0: 2.4e-4
      in, 3.4e-3 out), and the stream reaches 5.8e-2 by layer 7, the crate's measured
      release-scheme drift.
    * The producer capture below is the numerics check for the release's arithmetic.
  * **The fixture's widths.** Four ranks split each attention's output projection and each FFN's
    intermediate on 128-wide FP8 scale tiles, so the glm variants take eight heads of 64 on both
    attentions and the second shape's expert intermediate (512). The narrower fixture planned no
    split at worlds 2 and 4.
  * **glm-wiring is gated on its seams; its ids are reported** (a change from the approved split, which held its ids to the band; for the owner's review). At the widened fixture, ids missed
    the 0.2 band at world 1 (prompt 3 position 4 by 0.376 nats; prompt 4 outside the reference's
    eight alternatives), and the device picks the same ids at worlds 1, 2 and 4. The cause is the
    fixture, not the wiring. Its random routers and sparse indexer carry near-ties, and the BF16
    stream's rounding flips them on a few rows a layer. Measured on prompt 3's prefill against
    the host reference's boundary table at world 1:
    * Most rows drift by one to three BF16 ulps: the median row's relative L2 distance grows from
      2.3e-3 at layer 0 to 2.3e-2 at layer 7.
    * Isolated rows jump to 1e-1 to 3e-1, first at the sparse layers, and then spread through
      attention. Recomputing the router from both sides' FFN inputs, 11 rows across layers 4 to 7
      change their top-2 experts, each at a host margin between the second and third expert of
      5e-5 to 2e-2. Layer 3 row 34 (an attention layer, no router flip) is the one row of the
      seven past the indexer's 32-key budget that jumps; the other six agree at the baseline.
    * One sublayer moves the stream's median row by 0.12 to 0.36 (host, every layer), so a wiring
      fault moves every row by a share of that, while a near-tie flip moves a few.

    The gate (`SEAM_GATED`): the host reference writes prompt 3's boundary table, and every world's
    device group dumps that prompt's first prefill pass (`SPARKFLASH_ROWS_DUMP`, warm-up off, both
    now forwarded to every rank). Each layer's median row must stay under the bar. The bar is
    measured as the tie band is, twice the largest median the worlds measure, rounded up: 7e-2.
    Ids are still compared under the band and printed.
  * **Results (four nodes, 69ce489d, DeepSeek restored after).**

    | world | largest median seam (layer 7) | rows past 10x a layer's median, layer 7 | verdict |
    |---|---|---|---|
    | 1 | 2.29e-2 | 7 of 39 | PASS |
    | 2 | 1.87e-2 | 4 of 39 | PASS |
    | 4 | 3.07e-2 | 2 of 39 | PASS |

    Ids under the band: world 1 misses prompts 3 and 4, world 2 prompt 3, world 4 prompts 3 and 4,
    each at the same device id. `glm` (reported) misses one or two of five prompts at each world.
    Before the fix, at worlds 2 and 4 no group became ready. The resident plan held every KDA
    tensor whole while the stated attach binds the rank's heads (f9782ba0: recurrent tensors take
    the split their layout states). Milestone 3's `kda-dense` had the same fault at world above 1.
* **The producer reference, for the text model.** The crate, its host reference and the base
  all read the producer through one reading, ours. Two captures check that reading from outside
  it (plan 05 section 5.6):
  * On the `glm` synthetic variant, the producer's own classes (`transformers` at the revision
    the layout cites, at the config's BF16) run the tiny checkpoint. The shared host reference,
    on the producer's type map, matches their logits inside a tolerance written before the run.
    With the types aligned, that tolerance is a reorder's. This checks the wiring part by part:
    the mix and its epsilons, the pool and the tail, the router, the head fold, the activation
    recipe, and where each f32 cast sits.

    **Tolerance, written before the run (2026-11-25, before the capture below has run).** Both
    sides read the same checkpoint, the same 20-token prompt (five pools of four) and the same
    config's BF16 type map; the only expected difference is the order of the f32 sums inside
    BF16-owned modules (torch's matmul reduction order and kernels against ours). The host
    reference must be inside, at every position:
    * logit relative L2 distance, `||ours - producer|| / ||producer||` over the 512-wide row,
      at most **1e-2** (half of rule 2's version-4 NLL ceiling, expressed on the row the head
      emits);
    * top-1 agreement **20 of 20** positions, and every position's argmax is the same token;
    * mean absolute NLL delta, over the 20 positions, at most **0.02 nat** (rule 2's ceiling).

    A miss is a changed function or a wrong type. It is found and fixed in the layout, the part
    or the body; the tolerance is not widened after the run. The check is
    `scripts/producer-capture.py`, which runs the producer and compares the two logit files.

    **Results (2026-11-25, CPU BF16, `transformers` at `42efcfde46e668067a74bbd439d33f107b4a7c6c`).**
    Running the capture found and fixed two real discrepancies:
    1. `sparkflash-weights/src/safetensors.rs`: `write_checkpoint` padded tensor data to 8-byte
       boundaries, leaving 4-byte holes after 12-byte F32 tensors (`hc_attn_scale`, `hc_ffn_scale`,
       shape `[3]`), which `safetensors` 0.8.0 rejects with `InvalidOffset`. Writing contiguous
       tensor bytes fixes checkpoint interoperability.
    2. `sparkflash-model-deepseek-v41/src/forward/moe.rs` (`stated_expert_body`): on plain BF16
       checkpoints (`activation_tile == 0`), `transformers`'s `grouped_mm_experts_forward`
       (`transformers/integrations/moe.py:523-535`) multiplies each expert output by the f32
       routing weight and accumulates in f32 before casting once to BF16, whereas the FP8 grouped
       path (`finegrained_fp8.py:694`) multiplies in BF16. Matching that split dropped same-input
       MoE distance on layers 3, 4 and 5 from `3.18e-3` to `0.0000e+00` across all 20 positions,
       and end-to-end pos 0 logit distance from `9.31e-3` to `1.66e-3` (`0.0000e+00` before
       `lm_head`'s BF16 output rounding).

    Same-input sublayer check (`scripts/producer-capture.py --boundaries`, pure BF16 `glm-tiny`,
    seed 7):

    | sublayer | pos 0 rel L2 | median rel L2 | max rel L2 | cause of any non-zero |
    |---|---|---|---|---|
    | KDA (L0, L1, L2, L4, L5) | `0.0000e+00` | `0.0000e+00` | `1.54e-03` | 1-ULP BF16 half-way ties on 5/100 rows |
    | Dense MLP (L0, L1, L2) | `0.0000e+00` | `0.0000e+00` | `2.03e-04` | 1-ULP sum order |
    | MoE (L3, L4, L5, L6, L7) | `0.0000e+00` (L3-L6) | `0.0000e+00` | `1.10e-03` (L7) | 1-ULP sum order |
    | Absorbed MLA + pooled/shared indexer (L3, L6, L7) | `0.0000e+00` | `3.64e-03`–`3.90e-03` | `5.29e-03` (L3); `9.99e-01` (L6 pos 11) | absorbed vs producer's expanded BF16 MLA (`~3.9e-3`); L6 pos 11 exact `0.0 == 0.0` pool tie on 2-head `ReLU` |
    | Head fold + norm + `lm_head` (f32) | `0.0000e+00` | `0.0000e+00` | `1.21e-07` | bit-exact |

    End-to-end logits against the unchanged pre-run bar on `glm-tiny` (seed 7, 20 tokens):
    * **Pure BF16** (same arithmetic on CPU): 19/20 top-1 agree, mean `|NLL delta| = 1.285e-02`
      (inside the `0.02` bar), pos 0 `1.66e-03`, median `1.04e-02`, max `3.49e-01` (**FAIL** on
      the two synthetic tie positions: L6 pos 11 exact `0.0 == 0.0` pool tie broken to Pool 1 by
      PyTorch CPU `topk` vs Pool 0 by `top_pools`; and L4 pos 13 router margin `0.0022` between
      experts 0 and 7 flipped by the `3.9e-3` absorbed-vs-expanded BF16 reorder).
    * **FP8 `glm-tiny/config.json` on CPU**: `transformers` falls back to dequantizing FP8 weights
      to BF16 and disabling dynamic activation quantization on CPU (`Using FP8 quantized models
      requires a GPU or XPU`), adding `~4.5e-3` weight truncation per FP8 projection; against host
      `activation_scheme: dynamic` that gives 18/20 top-1, mean `|NLL delta| = 3.93e-02`, max
      `4.34e-01`. Native FP8 producer comparison requires a CUDA/XPU host.
    **What the capture leaves owed.**
    * The absorbed block's 3.6e-3 to 3.9e-3 is declared a reorder (D125), on the condition that
      the host reference's expanded mode meets the producer at zero or one ulp. **Done
      (2026-11-26).** The host reference runs the producer's own order through
      `mla_attention.form = expanded` (a layout value; a rank serves `absorbed`), and
      `dump-host-logits --mla-form expanded` selects it for a capture. On a genuine BF16
      `glm-tiny` (seed 7, 20 tokens), each side given the other's own pre-attention hidden
      state:

      | layer | expanded vs producer, entries within 1 BF16 ulp | expanded vs producer, median | absorbed vs expanded, median | absorbed vs expanded, max |
      |---|---|---|---|---|
      | 3 (the first MLA block, identical inputs) | 100.0% | 0.000e+00 | 3.897e-3 | 5.287e-3 |

      Expanded reproduces the producer's `kv_b` expansion, its BF16 keys and values and its
      BF16 logits (`modeling_glm5_next.py` 1165, 1089) to the last ulp, so the wiring is
      proved and the 3.9e-3 is the two intermediate BF16 roundings the absorbed form never
      makes. Absorbed sits at 3.897e-3 median against the 1e-2 bar D125 set before this run.
      The layers after 3 diverge in their *inputs* (the absorbed form's own rounding
      propagates), so only layer 3 is a same-input comparison.
    * The end-to-end bar is not met and is not widened. Both misses are fixture ties, so the
      fixture is made decisive (no exact-zero pool scores at the selection's edge, router margins
      above the reorder) and the capture is run again against the same bar.
      * **Measured on the BF16 fixture (2026-11-26).** The two misses are exactly the recorded
        pair: `L6` position 11 is a pool tie where the host's pool score and the producer's are
        both exactly `0.0` and the two `topk` implementations order them differently, and `L4`
        position 13 is a router margin of `0.0022` between experts 0 and 7, under the absorbed
        form's `3.897e-3`. Both are properties of a near-uniform random fixture, not of the
        stack: the expanded mode, which reproduces the producer's arithmetic to the ulp, fails
        at the same positions and the same sizes (`L6` max `1.003e+00`, `L7` max `1.060e+00`).
        The decisive fixture is the next line of this section.
      * **The fixture's router is drawn three times wider (2026-11-26).** On the BF16 fixture
        the router's top-2 margins sat under the absorbed form's 3.9e-3, so an expert flipped
        and the capture measured the near-tie rather than the stack. With the router's logits
        drawn 3x wider (`write_described_checkpoint`'s recipe; no rule, epsilon, width or
        clamp changes), the same unchanged bars give:

        | form | top-1 | mean abs NLL | largest position rel L2 |
        |---|---|---|---|
        | expanded (the producer's own order) | 20 of 20 (bar: all) | 5.007e-3 (bar 0.02) | 2.538e-1 (bar 1e-2) |
        | absorbed | 19 of 20 | 5.817e-3 (bar 0.02) | 2.539e-1 (bar 1e-2) |

        So the top-1 and NLL bars are met by the producer's own order, and the per-position
        bar is not: position 11 still fails, identically in both forms (2.538e-1 against
        2.539e-1), which is the pool tie and not the absorbed reorder. Two things were tried
        and measured rather than assumed: drawing the indexer's queries and keys wider does
        not help (scaling moves no sign, so an all-negative pool stays all-negative: the
        distance stayed at 2.538e-1), and making every pool selectable (`index_topk` raised
        past the pool count) makes it worse (3.616e-1).
      * **Done (2026-11-26): the decisive fixture recipe passes the unchanged bar.** Four
        conditioning recipes in `write_described_checkpoint` (the DeepSeek writer
        `write_synthetic_checkpoint` stays bit-for-bit unchanged; no arithmetic, epsilon, width
        or clamp changes) remove the fixture's random near-ties while exercising every branch:
        (1) `mirror_indexer_heads` mirrors `wq_b` heads in negated pairs with shared
        `weights_proj` rows so `w_h * (relu(d) + relu(-d)) = w_h * |d|` gives zero exact-zero
        pool scores (smallest selection-edge margin `5.295e-2`, 0 exact zeros);
        (2) `EMBEDDING_SPREAD = 3.0` gives the residual stream a dominant common component so
        1-ULP halfway ties in KDA do not compound past `1e-2` over eight random layers;
        (3) `ROUTER_STEP = 1.2` on `e_score_correction_bias` keeps every router margin above
        `2.498e-1`, clear of the absorbed form's `4.1e-3` reorder (unlike scaling logits, which
        scales perturbation and margin together and still left a `3.30e-4` tie at L6 pos 5);
        (4) `HEAD_WIDE = 8.0` on the first 8 rows of `head.weight` peaks the next-token
        distribution so argmax is decided by a real margin instead of a 512-way `6.0e-4` tie.
        On BF16 `glm-tiny` (seed 7, 20 tokens), same-input layer-3 expanded matches the
        producer on 100.0% of entries at `0.0` ulp (absorbed vs expanded median `4.127e-3`,
        max `5.150e-3`), and end-to-end `scripts/producer-capture.py` passes the unchanged bar
        in **both** forms:

        | form | top-1 (bar: 20/20) | mean abs NLL (bar: 0.02) | largest position rel L2 (bar: 1e-2) | verdict |
        |---|---|---|---|---|
        | expanded (the producer's own order) | 20 of 20 | 1.378e-2 | 7.838e-3 (pos 13) | **PASS** |
        | absorbed (the served form) | 20 of 20 | 1.289e-2 | 8.206e-3 (pos 11) | **PASS** |
    * The pool tie itself: where several pools score exactly zero at the selection's edge, the
      producer's `topk` order is not specified and differs between its CPU and CUDA builds, so
      it is not a model fact. How often the real checkpoint's selection admits zero-score pools
      is measured at the real capture.
    * **A.3, the release's dynamic FP8 scheme. The tolerance is written here before the run
      (2026-11-26).** The capture above runs the fixture as a BF16 model: `glm-tiny`'s
      `text_config.dtype` is bfloat16, so both sides widen every weight and activate in BF16.
      The release ships `quant_method: fp8`, `activation_scheme: dynamic` and
      `weight_block_size [128, 128]`, and the writer honours it (567 tensors, 187 scale planes);
      the same capture is run over that checkpoint on a host where the producer's quantiser can
      run, because loading it on a CPU dequantises it (transformers reads the fp8 checkpoint as
      bfloat16 with `is_quantized false`, so a CPU run would compare the host's FP8 arithmetic
      with the producer's BF16 arithmetic and measure the wrong thing). The bars are the ones
      above, **unchanged**: every position's top-1 the same, per-position relative L2 at or under
      `1e-2`, mean |NLL delta| at or under `0.02`. The same-input layer tables are reported
      beside them and set no bar of their own — a dynamic scheme scales a 128-column tile by that
      tile's own amax, so a one-ulp difference in the tile's activation can move one code by one
      step, and the tables say how far that carries. If the end-to-end bar is missed, the miss is
      reported with its positions and is not repaired by widening the bar.
    * **A.3 and A.4, two windows and what they measured (2026-11-26).** Both items were run twice
      on the four nodes; neither passed, and both failures are about the host the run needs, not
      about GLM's arithmetic.
      * **A.3: the producer's fp8 path cannot run on this hardware yet.** The capture builds the
        host reference and the fixture fine (`write-synthetic-checkpoint` and
        `dump-host-logits` exit 0 in both MLA forms), and then the producer refuses before it
        reads a token: torch 2.9.0+cu130 refuses the GB10 outright (`cuda capability 12.1 ...
        Maximum ... supported ... is (8.0) - (12.0)`), and torch 2.14.1+cu130, which does report
        capability `(12, 1)`, still raises `NotImplementedError: "normal_kernel_cuda" not
        implemented for 'Float8_e4m3fn'` — inside `_initialize_missing_keys`, i.e. while it
        random-initialises the **vision tower's** parameters, which the synthetic checkpoint does
        not carry, as fp8 tensors. Two things follow: the checkpoint has to carry every tensor the
        producer initialises (the tower, B. I1-I3), and the frontend has to be able to run an fp8
        kernel on sm_121 (torch 2.14.1's compiled architecture list stops at `sm_120`). A.3
        therefore runs after B, and its bar stands as written above.
      * **A.4: the fixture refused the checkpoint because the build embedded no kernel image.**
        The conformance arm booted, planned the checkpoint, and refused every part with
        "not loadable under runtime availability" — including `glm_attention_sm121`,
        `glm_decode_sm121`, `glm_moe_sm121` and `glm_prefill_sm121`, whose cubins were staged
        in the directory that both `SPARKFLASH_CUBINS` and `SPARKFLASH_KERNEL_DIR` named.
        The reason is the build, not the env: `sparkflash-kernels`' build script embeds a
        cubin only when it finds `nvcc`, and the head node's `nvcc` lives at
        `/usr/local/cuda/bin/nvcc`, which is not on a non-interactive `ssh` `PATH`. Without
        it the script warns that the build has "CPU reference kernels only", the images are
        empty, `Image::available()` is false, and `image_loadable` refuses every GLM image
        (only the DeepSeek image is loadable from a cubin directory). The rebuild with
        `PATH=/usr/local/cuda/bin:$PATH` is the fix; the checks themselves never ran, so no
        record was written and nothing about GLM's bits was measured either way.
  * **The third window (2026-11-26, `1def661c`): the build fault closed, and neither test reached its
    arithmetic.** One four-node window, production DeepSeek stopped for 3 m 50 s and restored
    (`269f7b99`, `/readyz` 200, 4/4 ranks, same model card).
    * **The build now finds its toolkit.** With `nvcc` off `PATH` (the non-interactive shell the
      launcher uses), `build.rs` fell back to the installed toolkit and found
      `/usr/local/cuda-13.0/bin/nvcc`; all four cargo commands exited 0 and the build log carries
      no warning line at all, where the two previous windows logged "CPU reference kernels only".
      That is `0b43725d`, and it is the fault A.4 recorded, closed.
    * **A.4 did not run.** The suite booted and every one of the eight checks failed:
      `alone_equals_four` on a token mismatch, the rest with `the walk server has ended`, and one
      walk-server thread panicked in `gumbel.rs` with `range end index 1 out of range for slice of
      length 0`. The cause is the fixture, in two parts. The checkpoint the window named
      (`glm-s2-ckpt`) was written before this tree's fixture, so the plan and the stored weights
      disagree; and a checkpoint regenerated from the current fixture could not be opened at all,
      because the fixture now carried the vision tower and the plan then asks for
      `vision_patch_embed`, `vision_position`, `vision_norm`, `vision_attention`, `vision_mlp`,
      `vision_downsample` and `vision_merger`, for which no kernel candidate exists until I1-I3.
      The tower therefore moved to its own fixture (`glm-tiny-tower`, for the producer capture),
      the text-only fixture is what the device arm opens, and the record-directory run stays owed.
      The suite also reads `reference/tokenizer.json` from the repository root, so it must be run
      from there.
    * **A.3 measured nothing, and for a fault of ours.** The producer loaded the checkpoint and
      then died in its first `index_select`: `Expected all tensors to be on the same device, but
      got index is on cpu, different from other tensors on cuda:0`. The capture built its id
      tensor on the CPU while `from_pretrained` had put the model on `cuda:0`. The ids, mask and
      positions now follow the model's device (`89b08da4`); the run itself is owed.
    * **A.7 is measured.** The absorbed MLA at 8 rows and at the release's windows, four-launch
      against one-launch: at 13,312 positions 361.6 us against 9,195.3 us (0.04x), at 100,000
      334.2 us against 9,196.5 us (0.04x), at 1,048,576 293.5 us against 9,217.9 us (0.03x). The
      base's split wins by 22x to 33x there, and the margin grows with the window because the
      one-launch body is one block a head whatever the sequence is.
  * **The fourth window (2026-11-26, `04e585f5`): A.4 boots and A.3 reaches the kernel.** Another
    four-node window, production DeepSeek stopped for 3 m 11 s and restored (`269f7b99`,
    `/readyz` 200, 4/4 ranks, model card byte-identical).
    * **The checkpoint now opens.** `scripts/second-shape.py run --stored-format glm` wrote the
      variant from this tree's fixture (plan hash `1170f041`, `8 layers, 38 part instances, dense
      Fp8E4M3, experts Fp8E4M3`) and the conformance arm booted it. The stale-fixture refusal is
      gone; the tower-free text fixture is what the arm opens.
    * **A.4 still fails, and now on a real fault, not a fixture.** All eight checks failed in both
      runs (`exit 139`), deterministically. `alone_equals_four` returned different tokens from the
      four-lane run (`[26264, 15026, 0, 2, 6]` against `[26264, 0, 5, 2, 4]`);
      `bits_equal_the_record` panicked at `gumbel.rs:225` with `range end index 1 out of range for
      slice of length 0`, i.e. the sampler was handed an empty candidate row; the walk server then
      died and the remaining six checks reported `the walk server has ended`. The record directory
      stayed empty, so no bit record was written. Until this window the record check always
      SKIPped for want of a directory, so the sampled path had never run here: the panic is a
      fault the record check exists to find, and it is ours to close before A.4 can pass.
      The sampler no longer indexes an empty list: `pick_truncated` returns `None` when no
      rank's slice contributed a candidate (the shape a world of zero slices makes), the row
      head refuses by name with the row, the row count and the world, and
      `a_world_of_no_slices_has_no_candidate_to_pick` holds it. The next run therefore names
      the world it was handed instead of taking the walk server down with it.
    * **A.3 reached the FP8 kernel and stopped on a missing wheel.** The producer loaded the
      tower-carrying checkpoint, ran its own quantizer, and raised
      `ImportError: finegrained-fp8 kernel unavailable: \`kernels\` is either not installed or uses
      an incompatible version ... (0.18.2 <= version < 0.19.0)` on the first FP8 linear, so no
      capture was written. `transformers` 5.x routes fine-grained FP8 through the Hub `kernels`
      package, which the GPU venv did not have; `kernels 0.18.3` is now installed in
      `~/.venvs/glm-producer-gpu2`. The init-time `normal_kernel_cuda` fault is gone, so the next
      run is the one that measures.
    * **A note for the launcher's environment.** The brief's rank order (rank 1 on
      `sg1-jackson`, rank 3 on `sg1-carter`) does not match the live group (rank 1 on
      `sg1-carter`, rank 3 on `sg1-jackson`). A stop that trusts the brief's order leaves ranks 1
      and 3 serving; the launcher's own recorded command is the authority.


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
* **Milestone 4's measurements (2026-11-26, four nodes).** Every number below is from one
  window: `c0793f36`'s build on the head node's GB10, the released DeepSeek stopped, the
  production server restored from `target/m3/release/sparkflash` (`269f7b99`) after each run.
  The corpus is `crates/sparkflash-ops/examples/glm-stage-bench.rs`, run as
  `SPARKFLASH_KERNEL_DIR=<cubins> glm-stage-bench`; the geometry is the release's at four ranks
  (16 heads, NoPE 256, value 256, latent 512, q-LoRA 1,536, E4M3 latent cache).

  **A1, the absorbed MLA body against the base's four-launch split.** GLM's own
  `q4e_glm_mla_attend` is one launch; the base's `AbsorbedMla::attend` is scores, softmax, mix
  and project. The ratio is the four-launch time over the one-launch time, so a value under 1
  means the base's split is faster:

  | rows | sequence | one launch | four launches | ratio |
  |---|---|---|---|---|
  | 1 | 256 | 140.8 us | 119.1 us | 0.85 |
  | 4 | 256 | 569.7 us | 99.1 us | 0.17 |
  | 1 | 512 | 271.3 us | 123.9 us | 0.46 |
  | 4 | 512 | 1,080.6 us | 130.9 us | 0.12 |
  | 1 | 2,048 | 1,034.4 us | 147.0 us | 0.14 |
  | 4 | 2,048 | 4,108.6 us | 319.7 us | 0.08 |

  The base's split is faster at every shape, by 1.2x at the smallest and 12.8x at the widest,
  so A1 keeps the base's body and the stated walk does not call GLM's one-launch decoder. The
  one-launch kernel is one block a head; the split's blocks fill the GPU at every sequence.

  **A4, bulk prefill attention.** The same comparison over the prefill side, where GLM's body is
  `glm_prefill_mla_attend` plus per-head BF16 MMA:

  | rows | sequence | four launches | GLM's bulk | speedup |
  |---|---|---|---|---|
  | 32 | 256 | 280.9 us | 72.2 us | 3.89x |
  | 64 | 256 | 542.8 us | 111.8 us | 4.86x |
  | 128 | 256 | 1,065.1 us | 145.5 us | 7.32x |
  | 32 | 512 | 322.8 us | 96.1 us | 3.36x |
  | 64 | 512 | 588.5 us | 166.1 us | 3.54x |
  | 128 | 512 | 1,139.6 us | 214.5 us | 5.31x |
  | 32 | 2,048 | 500.5 us | 235.2 us | 2.13x |
  | 64 | 2,048 | 875.8 us | 441.6 us | 1.98x |
  | 128 | 2,048 | 1,691.6 us | 639.6 us | 2.64x |

  GLM's body wins everywhere: 2.0x to 7.3x, the margin growing with the row count because the
  split pays four launches a row group. A4 keeps it.

  **K5, the KDA body.** `KdaLayer::run` over the release's geometry (16 heads of 128, four
  taps), with the projections stubbed out so the number is the conv, the decay, the prep and the
  scan alone: 35.9 us a layer at one row and 74.0 us at 64 rows, so 1.22 ms and 2.52 ms for the
  release's 34 KDA layers. Against a decode step that is milliseconds already, the KDA body is
  not the speed row's problem, so no fusion across the foreign boundary is built (K5's own rule:
  build it only if the speed rows need it).

  **Conformance.** `conformance!(DeepSeekBase)` on the `glm` variant, run the way
  `scripts/onboard-gate.sh` runs it (`--ignored`, one thread), passes 8 of 8:
  `alone_equals_four`, `bits_equal_the_record` (a SKIP: no record directory was named),
  `continued_equals_cold`, `drafted_equals_plain`, `no_unrecorded_widening`,
  `rewound_equals_cold`, `tp_one_two_four` and `whole_prompt_equals_chunked`. The arm boots the
  synthetic `glm` checkpoint the second-shape writer produces, at world 1, through `BaseWalk`,
  with the staged cubins beside it (`SPARKFLASH_CONFORMANCE_CHECKPOINT`,
  `SPARKFLASH_CONFORMANCE_CUBINS`, and `LD_LIBRARY_PATH` pointing at the collective's library:
  without that last one the walk server's rank 0 cannot open NCCL and every check fails with
  `the walk server has ended`).

  **W1, the converted ranks.** On the GLM second-shape checkpoint,
  `SPARKFLASH_ON_BASE=1 sparkflash convert --tp-world-size W --ranks ...` writes
  `rank-{r}-of-W.bin` at W = 1, 2 and 4 from the stated splits: 567 tensors a rank, no B12x and
  no 640-padded MoE tile (GLM's dense and routed weights are already the producer's 128-wide FP8
  tiles), and `convert --inspect --verify-weights` reports `weights_verified=true` on each. The
  stated stack does not yet read those files: it streams the checkpoint, and reading a converted
  rank is S-series work.

  **The released DeepSeek is unchanged.** `scripts/open-manifest-ab.py ab` with the
  pre-milestone-4 build (`2f99fbe7`, sha `cf91afd3`) against this one (sha `8333cc15`), both on
  the same cubin directory, one repetition: the model manifest is identical on all four ranks
  (93,386 records on ranks 0 to 2, 93,394 on rank 3), the launch traces are identical, the tick
  trace is identical, decode and prefill speeds are inside the tolerance, drafting accepts 524
  of 667 on both, and the harness reports `1 of 1 repetitions clean`. Prefill reads 7,147.8
  tok/s against 6,354.4 tok/s. That 12.5% is one repetition's spread, not a gain: no kernel
  source changed between the two builds, both ran the same cubins and their launch traces are
  identical. The alternated repetitions rule 1 asks for are still owed.

  **Two faults the window itself found, both in the new code and both fixed (`c0793f36`).** The
  stated draft stack resolved the whole embedding table from `device.embed_table`, which the
  device fills only when the embedding replica is gathered — and the replica was gathered after
  the drafter opened, so worlds 2 and 4 refused with `stated draft: full embed.weight table is
  not resident on device`. The replica now gathers before the drafter opens, and the later
  gather is skipped when it already ran. Separately, four kernels in the stated drafter's
  catalogue were declared at a precision that catalogue does not carry
  (`q4e_quantize_e4m3_rows` and `q4d_capture` as `Fp8E4M3`/`F32`, `q4e_argmax_prob_slices` as
  `F32`), so world 1 refused with `the catalogue carries no q4e_quantize_e4m3_rows in
  qwen4_exp_sm121 accepting e4m3`; they are now `Bf16`, `Bf16` and `Fp8E4M3`, which is what
  `sparkflash-ops`' own part table states.

  **Environment notes, for the next window.** `target/m3-cubins` is the released DeepSeek's
  staging directory: 62 cubins, none of them GLM's or Qwen's, so a GLM window needs its own
  `scripts/build-cubins.sh` output (153 cubins). The launcher's default head name `spark0` does
  not resolve from any of the four hosts; rank 0 runs locally through `SPARK_LOCAL_RANKS=0` and
  every ssh goes to node1, node2 or node3 by its own name.

  **Still open.** The live RDMA exchange is not proved for GLM: `scripts/serve-four-node.sh`
  pins `SPARK_COMM_REDUCE=nccl` whenever a cubin directory is staged, and the device arm always
  stages one, so the GLM group ran `requested="nccl" path=nccl` with no RoCE call. Proving it
  needs a packed-arm launch with no cubin directory, which is the branch the launcher already
  carries for a model that brings its own kernels. The host side of M6 is proved.

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
