# OpenAI surface: remaining gaps and issue list (S7/S8)

Status: working list, updated 2026-10-21 on branch `task/serve-n1-ops`. It replaces the
prose residual section of [openai-surface-gaps.md](openai-surface-gaps.md) §5 as the actionable
tracker; that note remains the record of what task-20 landed, and this one lists what is still
open, who owns it, and whether it depends on device work.

Paths and labels: everything marked **serving** is host-fixable in
`crates/sparkflash-api` / `crates/sparkflash-cli` and testable on this host. **device**
means it needs a device window (nodes/GPU, device writer, TP=4 run) and cannot be closed here.
**QA** means the artifact exists but the evidence has to be recorded in a window. Every item is
**UNMEASURED** on device unless a device window says otherwise; no tok/s, TTFT or token-acceptance
claim is attached to any item.

## 1. Closed in this block (for context)

| ID | gap | disposition |
| :--- | :--- | :--- |
| N1 | unseeded `n > 1` reached every choice with seed `None`, which the host driver maps to seed 0, so unseeded choices were copies of one sample | **closed host-side**: the router now draws a fresh seed per choice for unseeded `n > 1` (`AppState::choice_seed`), `n == 1` keeps the byte-compatible `None` path. Router tests pin distinct seeds, seeded `seed + i` reproducibility and end-to-end differing choice text. The device path still cannot sample (`DeviceEngine` refuses non-greedy), see D2. |

## 2. Open gaps

| ID | gap | where it stands today | owner | device dependency | next action / acceptance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | constrained decoding (`response_format` `json_object` / `json_schema`) absent | typed refusal in [server.rs](../../crates/sparkflash-api/src/server.rs) (`refuse_unenforceable_response_format`); [openai-surface-gaps.md](openai-surface-gaps.md) §5 | serving (engine seam) | none to build; a device claim would need a device logits processor | implement a decoder logits processor in the engine seam, then accept the two modes instead of refusing; wire census in [scripts/openai-surface-census.py](../../scripts/openai-surface-census.py) |
| S2 | `response_format` not in the completions request schema; unknown JSON is silently ignored | [serve-http schema tests](../../crates/sparkflash-cli/tests/serve_http.rs); openai-surface-gaps §5 | serving | none | add the field to the completions schema with the same refusal rule as chat |
| S3 | TP=4 `n > 1` refused at the router (`independent_choices() = false`) | refusal in [server.rs](../../crates/sparkflash-api/src/server.rs) (`resolve_choice_count`); TP=4 driver override in [serve.rs](../../crates/sparkflash-cli/src/serve.rs) | device | **yes**: the four-rank group samples one shared window; independent per-rank/per-choice sampling has to be designed with the device writer | after the device sampler lands, remove the override and prove distinct choices across a TP=4 window; until then the refusal is the honest answer |
| S4 | device-engine `n > 1` unproven: `DeviceEngine` keeps the trait default `independent_choices() = true` but refuses non-greedy sampling, so temperature > 0 choices end in a 503 and greedy choices are deterministic copies | [crates/sparkflash-device/src/engine.rs](../../crates/sparkflash-device/src/engine.rs); device path noted in openai-surface-gaps §5 | device | **yes** | make the device engine sample per choice (or override `independent_choices` to false until it can); then a device window with `n = 2`, temperature > 0, distinct outputs |
| S5 | no `best_of` | rejected as unknown/ignored per endpoint today (openai-surface-gaps §5) | serving | none | either accept with vLLM's semantics (sample best_of, return n) or refuse typed with `param: "best_of"`; do not silently ignore |
| S6 | streaming completions report `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` as 0 | pre-existing streaming shape, openai-surface-gaps §5 | serving | none | derive the fields from the same plan the buffered response uses; SSE byte tests already pin the frames |
| S7 | bearer auth is a static local key only (no OIDC, multi-key, rotation, scopes) | [server.rs](../../crates/sparkflash-api/src/server.rs) `require_bearer_auth`; openai-surface-gaps §5 | serving | none | document multi-key/rotation as out of scope for this deployment skeleton, or add a keyring; do not claim production auth |
| S8 | no `http_*` instrumentator-style router metrics and no `GET /load` | [serve-ops-metrics.md](serve-ops-metrics.md) §3 | serving | none | add request/latency counters (axum middleware using the existing registry) and a `/load` shape, or keep documented as absent |
| S9 | prefix-cache hit/query counters, GPU KV-cache usage, queue-time histograms, swapped/preempted gauges absent | serve-ops-metrics.md §3 | device | **yes**: there is no host prefix cache or device KV pool to report | export them when the paged-KV and prefix-cache device paths exist (serving plan §12 item S3); values must come from the device accounting, never a fabricated zero |
| S10 | unseeded `n == 1` requests are still fully deterministic: the host driver maps seed `None` to 0, so two identical unseeded requests produce the same sample | deliberate byte-compatibility decision in this block: only unseeded `n > 1` draws new seeds; vLLM samples unseeded requests independently | serving | none | decide whether to randomize unseeded `n == 1` too (it would change goldens/byte tests and is not required for OpenAI compatibility); if yes, draw fresh seeds for every unseeded request and re-record the affected wire tests |
| D1 | device path: paged KV + prefix cache (serving plan §12 item S3) not wired through serving | [serving-s7-s8-plan.md](serving-s7-s8-plan.md) §12; no device window records it | device | **yes** | device writer lands paged KV/prefix cache, then serving exposes the counters and a window measures hit/miss |
| D2 | device path: continuous batching + chunked prefill (serving plan §12 item S4) not on device; `--continuous-batching` is refused with `--engine device` | validation in [serve.rs](../../crates/sparkflash-cli/src/serve.rs) (batching + device refusal) | device | **yes** | device scheduler consumes the same `Job` shape; then remove the refusal and run a device batching probe |
| D3 | S7 record residuals: 9/9 reference matches (m1y34 2/9, m1y14 2/9) and `token_claimed = true` on every rank | [serving-s7-record-draft.md](serving-s7-record-draft.md) §7 | device | **yes** | device writer fixes the Engram-write cause; then one window with 9/9 and the asserted claim; serving needs no change |
| D4 | final S7/S8 gate: D38 vLLM A/B at concurrency 1/8/32 | [serving-s7-s8-plan.md](serving-s7-s8-plan.md) §12 | QA + device | **yes** | run in the device/vLLM window; no host surrogate is accepted as the gate |
| O1 | 72 h soak not run | ops skeleton only ([serve-deployment.md](serve-deployment.md)) | QA | **yes** (needs the served deployment) | run the soak against the deployed four-node/device server; record drain/restart/backpressure behavior |
| O2 | Prometheus dashboards/alert rules not checked in | metric names and labels documented in [serve-deployment.md](serve-deployment.md) §3.4 and [serve-ops-metrics.md](serve-ops-metrics.md) | serving | none | add a dashboard JSON/alert-rule file in a follow-up if the operator wants them versioned here |
| O3 | container image not built; four-node launch only dry-run | [Dockerfile](../../Dockerfile), [scripts/serve-four-node.sh](../../scripts/serve-four-node.sh), [serve-deployment.md](serve-deployment.md) | serving + QA | container build is host work; the launch is only meaningful on the four nodes (**device**) | build the image in CI/a builder host; dry-run is the current contract; first real launch belongs to the device window |
| O4 | all evidence is host path / synthetic tiny plan or scripted stub, UNBARED | openai-surface-gaps §4; serve-ops-metrics.md | QA | **yes** | the device window records the bared runs; no host number is promoted |

## 3. Ordering

1. Device window unblocks D3 (9/9 + claim), then D1/D2 (paged KV/prefix cache, batching on
   device), then D4 (the D38 vLLM A/B) — those are the S7/S8 gate, and no serving change
   substitutes for them.
2. Serving-only items (S1, S2, S5, S6, S7, S8, O2) can be picked up in parallel; they change the
   API surface, not the device story.
3. S3/S4 and S9 wait on the device sampler/KV work; the honest state until then is the typed
   refusal / unproven label documented here.

## 4. Non-claims

This list is a gap tracker, not a result. It makes no throughput, TTFT, token-acceptance, GPU,
device or vLLM claim, and it does not promote any host census number. The n > 1 unseeded fix is
proven at the router level on this host only; the device and TP=4 rows remain unproven.
