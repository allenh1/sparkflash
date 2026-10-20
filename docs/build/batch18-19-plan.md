# Batch 18/19 plan - skeleton (numbers only from owners)

**Status: skeleton, scope updated 2026-10-20.** The user direction is *use the GPU as much as
possible*: the **GPU device chain is the performance path**, and the CPU host twin is the
exactness oracle plus a stopgap. This page therefore labels every CPU measurement as a
**stopgap data point** and keeps the GPU stages on `docs/build/gpu-device-roadmap.md` as
hypothesis + gate. **Do not invent numbers, field names, frame layouts or acceptance bars**:
every placeholder below names its owner and must be filled from that owner's measured
artefacts. A number quoted without an evidence path is incomplete, and device and host numbers
never share one table.

## 0. Binding decisions and the current best number

- **D44 oracle split** (`docs/DECISIONS.md` D44): prompt rendering is pinned to vLLM
  `encoding.py` (the five published fixture id arrays stay exact); API contracts follow
  `deepseek-recipe` 0.2 (the three `max_tokens` tiers 8192/65536/131072, `response_format`,
  prefix continuation, deferred tools). The request field's surface picks the oracle.
- **D45 resident serving, downgraded** (task-62): recorded as **oracle/stopgap evidence**, not
  a roadmap, after the GPU scope direction; the measured counters are kept below.
- **D46/D47** (`docs/DECISIONS.md` D46, D47): task-63 cheap API-surface parity and the
  protocol-layer adoption scope (protocol layer only; the recipe renderer/tokenizer stays out).
- **Acceptance target (D48)**: vLLM parity **50-60 tokens/s with DSpark** on this hardware,
  personal target **90 tokens/s single-stream**, with the derived per-step budget (<=42 ms per
  verify step at ~3.8 accepted tokens, 11.1 ms/accepted token, 13.9 ms weight-traffic roofline
  at 273 GB/s/rank, ~1.7 ms RDMA collectives, ~2.2 ms draft). The budgets are derived, not
  measured; the GPU roadmap is written against this budget, not against 'faster than the CPU
  stopgap'.
- **Current best measured decode: 767.7 ms/step = 1.3026 tokens/s** (CPU host twin, sharding
  stage 2, bit-identical by `plan_hash`, generated tokens and 10/10 `step_logits_sha256`). It is
  the **stopgap** number until a device run beats it and states that path's exactness contract.

## 1. CPU-path attention sharding, stages 1-3 - STOPGAP, not the long-term path

Owner: **tp-dev**. This is the CPU-path attention sharding: a **stopgap data point and the
exactness oracle, explicitly not the long-term performance path** (the GPU device chain is).

**Reported headline (Lead 2026-10-20; evidence path pending tp-dev, expected tree 8ffaae6 /
binary bbb20cf5):** default sharded **767.7 ms/step = 1.3026 tokens/s** vs replicated
**879.0 ms/step**, bit-identical (`plan_hash`, generated tokens, 10/10 `step_logits_sha256`).

| stage | what moves | measured effect (owner to fill) | evidence path |
| --- | --- | --- | --- |
| 0 baseline (replicated) | - | `TODO(tp-dev)` | `TODO(tp-dev)` |
| 1 head rows | attention weight head rows | `TODO(tp-dev)` | `TODO(tp-dev)` |
| 2 `wq_b` by head, `wo_a` by o_group, exact f32-byte `o_lora` gather | attention projections | reported: 767.7 ms/step, 1.3026 tokens/s vs 879.0 replicated, bit-identical | `TODO(tp-dev)` (tree 8ffaae6 / binary bbb20cf5 expected) |
| 3 `wo_b` output rows by hidden + exact residual gather | output projection | `TODO(tp-dev)` | `TODO(tp-dev)` |

Rules: the exactness rule per stage (bit-identical `plan_hash`/tokens/logits), the per-rank
number for each line (not just rank 0), and the window or run-context id of every measurement.
Stage 3 is only claimed once it lands. The section states plainly that these are CPU-path
numbers and do not transfer to the device path.

## 2. Resident serving (task-62) - ORACLE/STOPGAP evidence, not a roadmap

Owner: task-62. After the GPU scope direction these CPU items are **downgraded to
oracle/stopgap evidence**: keep the measured counters, do not present them as the plan.

Measured counters already reported (Lead 2026-10-20; evidence path to be supplied by the
owner): **1.286 GB/step owned copies**, **7.89 GB/step borrowed**, **flat RSS 86.00 GiB** over
the warm rounds. These stay as the evidence that the host path can be measured, not as a target.

Still to fill if the owner lands the rest (placeholders, no numbers until then):

- [ ] resident-set arithmetic (world=1 285.504 GiB / 71.4 GiB per rank at world=4, matching
  the 72.31 GiB/rank design): recorded in `docs/build/gpu-device-roadmap.md` as
  arithmetic-from-geometry, not a device measurement; a single GB10 node cannot hold the
  all-resident set, so the four-rank arm is the consistent one.
- [ ] decode-once-at-open: `TODO(owner)` - what is decoded once and the per-request saving;
- [ ] CLI `ResidentCache` folded into the engine: `TODO(owner)` - the surviving surface and the
  removed CLI path;
- [ ] one long-lived four-rank instance with warm request rounds: `TODO(owner)` - N rounds and
  the RSS series shape (the 86.00 GiB plateau is the reported headline, not a bar);
- [ ] same-run comparison that the stage-2 stopgap number does not regress: `TODO(owner)`.

## 3. API-contract P0s (task-63 cheap fixes)

Owner: **serve-dev**. These are a labelled sampling/API-surface change, **not** prompt
renderer changes (D44):

- [x] `max_tokens` contract: defaults 8192 outside thinking / 65536 in thinking /
  131072 in thinking with effort 100; explicit values are context-lowered to
  `max_seq_len - prompt_tokens` (floor 1); upper bound `MODEL_MAX_OUTPUT_TOKENS = 393216`,
  advertised by `/v1/models.max_output_tokens` and used as the CLI `serve --max-tokens`
  default. The request field names are `max_tokens` / `max_completion_tokens`; Chat Completions
  and Responses reject a value above the bound, Anthropic Messages lower it to 393216 first.
- [x] Anthropic cache accounting: `input_tokens = prompt_tokens.saturating_sub(cached_tokens)`,
  `cache_read_input_tokens = cached_tokens`, `cache_creation_input_tokens = 0`; the two
  partitions sum to `prompt_tokens`. Covered by
  `router::anthropic_input_tokens_exclude_cache_reads_and_partition_the_prompt`.
- [x] effort-table collapse: `low`/`minimal` = 25, `high`/`medium` = 50, `xhigh` = 75,
  `max` = 100, absent/null/default = 50; the seam resolver and `reasoning.rs` share the
  constants and `encoding::recipe_effort_score` reverses to the same scores.
- [x] cache usage fields: `usage.prompt_cache_hit_tokens = usage.prompt_tokens_details.cached_tokens`
  and `usage.prompt_cache_miss_tokens = usage.prompt_tokens - prompt_cache_hit_tokens`,
  recomputed by the batch aggregator and carried in the streaming final usage chunk.
- [x] temperature/top_p policy: in thinking mode `temperature = 1.0` and
  `top_p = max(requested, 0.95)` (default 1.0); non-thinking requests keep the caller's
  values. Labelled a sampling behaviour change, not a prompt change.
- [x] decision entries: `docs/DECISIONS.md` D46 (cheap fixes) and D47 (protocol-layer
  adoption scope), appended after D45 in the append-only region.

The larger protocol-layer adoption (7-10 days) is scheduled separately, not in this batch.

## 4. DeepInfer-derived design docs (docs-only pages)

Owner: **engine-dev**.

- DSpark/MTP design: **deferred out of batch 19** - `docs/engine/mtp-dspark-design.md`
  (`task/mtp-dspark-design`) was not merged; batch 19 carries the D50 decision entry and the
  device-wiring doc instead, and the design page lands with the batch that wires the drafter;
- Dispatch diagnostic results: **deferred** - the diagnostic's fields and their measured scope
  land with the serving-diagnostic batch, not here.

No throughput or acceptance number is quoted here until engine-dev supplies the field names and
the measured artefact.

## 5. GPU device-chain roadmap - the performance path

`docs/build/gpu-device-roadmap.md` carries the device stages derived from the independent
audit, each as **hypothesis + gate** against the D48 budget (<=42 ms/verify step at ~3.8
accepted tokens): device residency (task-34), the kernel occupancy rewrite (tasks 65-68),
router + `cert_lm_head`, RDMA one-shot + CUDA graphs, and DSpark k=5. It states which parts
are verified code facts and which are roofline estimates, and it corrects the collective row
(8.6 ms/step of wire cost, not the 92.2 ms rank-arrival bucket).

**Attribution rule:** the docs show each change separately - the sharding A/B, the zero-copy
before/after, and the stage-3 third arm - and no summed 'total win' number is claimed without
the per-change measurements behind it.

## 6. Evidence protocol for this page

Each section is filled from owner-provided artefacts, not from memory: numbers and field names
arrive with the evidence file path and are quoted with their source. A section without measured
numbers keeps its checkboxes and says so; no number is invented, and a changed bar after a gate
run is a new decision.
