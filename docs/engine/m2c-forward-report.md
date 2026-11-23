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

## Blockers

1. **Four-rank manifest/launch/tick comparison.** The baseline is on the head node at
   `/home/allenh1/cluster-tests/20261122T202505Z`; the harness must run from
   `/home/allenh1/m2a-harness-prod` with its own `serve-four-node.sh`. No window was held at
   the time of writing, and the primary agent is serving on the same ports/GPUs. Needs a scheduled
   window (<55 min) and the saved restore command.
2. **Speed A/B on the real four-rank DeepSeek group.** Same window; `scripts/speed-log.sh`
   (baseline vs change, repeated) has not run.
