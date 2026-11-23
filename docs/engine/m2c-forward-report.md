# Milestone 2 part C — the forward opens from the plan

Worktree `forward/part-c`, branch `forward/part-c`, based on `onboard/sprint2` at
`14f9bcf`. Each commit below records its exact commands and the comparison evidence it could
produce. The four-rank cluster was held by the running production service throughout this work
(checked with cluster-test-7 on 2026-11-23T00:24:53Z: /readyz 200, ranks 4/4 running, no
window), so the manifest/launch/tick comparisons and the speed A/B could not be measured in this
session. That is the blocking condition recorded per commit; nothing below claims equality that
was not measured. See "Blockers" at the end for the precise unproven set.

## PR 1 — tensor names come from the plan's part instances

Commit: `forward: open from the plan and name tensors from its claims`

What changed:

* `sparkflash-deepseek-forward/Cargo.toml` no longer depends on
  `sparkflash-model-deepseek-v41`; it depends on `sparkflash-ops` instead.
* new `src/plan.rs`: builds the `EnginePlan` from the checkpoint's `config.json` and the
  layout at the chain's world/rank, installs it once, and exposes the plan's facts, tensor
  claims and part symbols by name.
* `tp4_resident::plan_for_chain` builds the plan from the same `--checkpoint` the resident
  setup reads; `tp4_chain` installs it before the forward preflight.
* every layer/global tensor name the device bind, attention body, MoE path, Engram seam, MTP
  draft and operand census construct now comes from a part instance's claim
  (`layer_claim`, `layer_prefix`, `layer_root`, `mtp_claim`, `named`). A name the plan does
  not claim panics at the call site with the block and tail, rather than launching on a guessed
  string.

Commands run:

```
cargo fmt -p sparkflash-deepseek-forward
cargo check -p sparkflash-deepseek-forward --offline                  # pass
cargo check -p sparkflash-cli --offline                              # pass
cargo test -p sparkflash-deepseek-forward --lib plan::tests --offline -- --nocapture
# 2 passed: every_name_the_open_uses_is_a_claim, the_language_parts_all_have_a_body
```

Comparison evidence: **not measured**. The cluster-held production service is the only
four-rank DeepSeek; taking it down needs a lead-scheduled window and the saved teardown/restore
(`/home/allenh1/cluster-tests/20261122T202505Z/launch-command.sh`). The CPU plan-vs-baseline test
`cargo test -p sparkflash-ops --test deepseek_plan_manifest` passes on this host (2 passed),
which is the input side of the comparison, not the forward trace.


## PR 2 — kernel handles come from the resolved catalogue entries

Commit: 'forward: take kernel handles from the plan's catalogue entries'

What changed:

* plan::KernelHandles reads every handle the forward opens from the plan's PartSymbols
  entries, in catalogue order, at install time. The launch shape is unchanged: the same
  symbols, the same cubins, the same four-rank grids.
* the MoE router, routed gate/up, weighted SwiGLU, stored combine, SwiGLU, certified head,
  RMS norm, BF16-activation wo_a, CSA2 sequence attention, Engram embed/gate/wkv, reference
  candidate select, the opt router, the decode GEMV and every MTP symbol now come from
  plan::handles() instead of a string constant in the forward.
* a test pins every handle to an entry in the plan's catalogue symbol set.

Commands run:

    cargo fmt -p sparkflash-deepseek-forward
    cargo check -p sparkflash-deepseek-forward --offline                  # pass
    cargo check -p sparkflash-cli --offline                              # pass
    cargo test -p sparkflash-deepseek-forward --lib plan::tests --offline -- --nocapture
    # the_kernel_handles_are_catalogue_entries, every_name_the_open_uses_is_a_claim,
    # the_language_parts_all_have_a_body all pass

Comparison evidence: not measured, same cluster blocker as PR 1.

Remaining symbol strings (catalogue gap, not forward choice): the plan's DeepSeek symbol set
does not contain b12x_mxfp8_dense_sm121, b12x_mxfp8_wo_a_sm121, flash_mla_csa2_sm121,
b12x_mxfp4_moe_sm121 (the pre-added combine), lm_head_reduce_sm121, pool_group_sm121,
engram_stream_convert_sm121, engram_write_sm121 or moe_routed_device_sm121. They stay literal
until part B's catalogue claims them; the plan test's exact symbol set is the authority and it
does not.


## PR 3 — model values from the plan, buffer sizes computed at open

Commit: 'forward: read model values and sizes from the open plan'

What changed:

* values_from_plan fills the loop's value object (vocabulary, width, layers, heads, latent,
  expert counts, mHC streams, Engram rows, vision fields and every other field the forward
  reads) from EnginePlan::model_facts. No model value is parsed inside the forward crate.
* HIDDEN, VOCAB/VOCAB_PER_RANK, HC_MULT, PREDICTOR_ROWS, MLA_LATENT, DUMP_BLOCK_BYTES,
  REDUCE_CALLS_PER_STEP, CHUNKS_PER_RANK, LOGITS_BYTES_PER_RANK, SIDE_BYTES,
  W_BLOCK_MAX_BYTES, WINDOW_CODE_BYTES, WINDOW_SCALE_BYTES and DEFAULT_COMBINE_ROWS are no
  longer literal consts: each is computed at open from the installed plan's facts and read
  through plan::hidden()/vocab()/vocab_per_rank()/hc_mult()/predictor_rows()/mla_latent().
  The per-token accessors read atomics published once at install, not a lock.
* the device row-geometry refusal compares against the plan's dim and hc_mult rather than
  5120 and 4.

Commands run:

    cargo fmt -p sparkflash-deepseek-forward
    cargo check -p sparkflash-deepseek-forward --offline                  # pass
    cargo check -p sparkflash-cli --offline                              # pass
    cargo test -p sparkflash-deepseek-forward --lib --offline
    # 302 passed; 0 failed; 1 ignored

Comparison evidence: not measured, same cluster blocker as PR 1. Speed A/B is required for
this PR because it touches every-token code; it is blocked on the same window and is not
claimed.

Remaining model literals in non-test sources: the MTP draft's compile-time array geometry
(BLOCK 5, STAGES 3, EXPERTS 128, TOP_K 3, MARKOV_RANK 256) and the kernel-width constants
(KV_ROTATED_BYTES/KV_LATENT_BYTES, MLA latent store bytes) are still consts used in fixed-size
arrays. They are the next slice of "sizes computed at open" and are not claimed done here.


## PR 4 — the layer body runs the block's parts

Commit: 'forward: run the layer body from the block parts'

What changed:

* plan::block_has_part / layer_fact / layer_fact_bool / layer_compress_ratio /
  layers_with_fact / candidate_source_layer / tap_layers read the per-layer branches from the
  block's part instances and their facts instead of layer numbers.
* the device layer walk runs each block's parts: a block without the Engram part skips the
  Engram write, a block without MlaAttention skips the attention sublayers, a block without the
  MoE part skips the FFN. On the released checkpoint every language block carries all three, so
  the executed stage sequence is unchanged.
* install refuses an unknown part by name, and refuses the fused attention/FFN seam when a
  block carries the MoE part without the MlaAttention part it covers (and a compression or
  indexer sublayer without the attention part it belongs to). The fused seam is chosen only for
  the adjacency it covers.
* the kv/index source lists used by the bind and the candidate source layer now come from the
  blocks' kv_source / index_source / candidate_source facts.

Commands run:

    cargo fmt -p sparkflash-deepseek-forward
    cargo check -p sparkflash-deepseek-forward --offline                  # pass
    cargo check -p sparkflash-cli --offline                              # pass
    cargo test -p sparkflash-deepseek-forward --lib plan::tests --offline -- --nocapture
    # 4 passed (names, kernel handles, per-layer facts, every language part has a body)
    cargo test -p sparkflash-deepseek-forward --lib --offline
    # 302 passed; 0 failed; 1 ignored

Comparison evidence: not measured, same cluster blocker. Speed A/B for PR 4 is blocked with
it and is not claimed.

## Remaining pinned-list entries after PR 4

The part A branch recorded only the baseline and the plan-vs-manifest test; it did not land a
count-pinning rule test. The counts below are this branch's own inventory, measured with
grep over crates/sparkflash-deepseek-forward/src (the four-rank assumptions are left in place
as milestone 3):

| list | count | where |
|---|---:|---|
| direct sparkflash-model-deepseek-v41 dependency/import | 0 | Cargo.toml and all sources |
| ModelConfig::v41_flash / v41_flash_cached constructors | 0 | all sources |
| format!("layers...") tensor construction | 4 production-adjacent test fixtures (tp4_resident:1736, tp4_smoke:616/631/632/2513, layer_device:2075, tp4_forward:20167) | test-only |
| format!("mtp...") tensor construction | 0 | |
| kernel-symbol strings | 41 occurrences, mostly evidence/test strings; the production loaders for b12x_mxfp8_dense_sm121, b12x_mxfp8_wo_a_sm121, flash_mla_csa2_sm121, b12x_mxfp4_moe_sm121, lm_head_reduce_sm121, pool_group_sm121, engram_stream_convert_sm121, engram_write_sm121 and moe_routed_device_sm121 stay literal because the catalogue does not claim them | catalogue gap from part B |
| literal 5120 | 91 occurrences, almost all in unit-test expected values and kernel-width comments | tests, not the open path |

The dependency/import criterion is met. The symbol list is not empty because the catalogue
does not carry nine symbols the forward still loads; that is the part B gap recorded under PR 2.

## Blockers

1. **Four-rank manifest/launch/tick comparison.** The baseline is on the head node at
   `/home/allenh1/cluster-tests/20261122T202505Z`; the harness must run from
   `/home/allenh1/m2a-harness-prod` with its own `serve-four-node.sh`. No window was held at
   the time of writing, and the primary agent is serving on the same ports/GPUs. Needs a scheduled
   window (<55 min) and the saved restore command.
2. **Speed A/B on the real four-rank DeepSeek group.** Same window; `scripts/speed-log.sh`
   (baseline vs change, repeated) has not run.
