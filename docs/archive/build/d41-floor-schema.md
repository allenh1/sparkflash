# D41 capture schema, version 1

**Status: task-41 / task-42 interface, frozen 2026-10-19.** This file is the one schema between
`scripts/d41_recapture.py` (task-42, audit-dev: produces a capture from a running vLLM) and
`scripts/d41_floors.py` (task-41, serve-dev: consumes captures and computes the D41 floors and
bars). Neither script may invent its own field names. Any change to this file is a new schema
string and both scripts change together.

The capture format deliberately keeps `scripts/oracle_native.py`'s envelope and position records,
so a capture record can be built from `parse_prompt_logprobs` output without another mapping.
Everything is offline JSON: the floor computation never touches a server or a GPU. Unknown/extra
fields (for example `plan`, `warnings`, `status`, block `load_generator`) are tolerated and ignored;
only the fields below are consumed and validated.

## Top level

| field | type | required | meaning |
| :-- | :-- | :-- | :-- |
| `schema` | string | yes | exactly `"sparkflash-d41-capture/1"` |
| `capture_id` | string | yes | stable id of the capture, e.g. `task32-idle-2026-10-19T17:30Z` |
| `producer` | object | yes | `{"script": "scripts/d41_recapture.py", "script_sha256": <hex>}` |
| `started_utc`, `finished_utc` | string | yes | RFC 3339 UTC; `finished >= started` |
| `served` | object | yes | `{"id": <served model id>, "root": <served snapshot root>}` from `/v1/models`; both fields are required |
| `expect_root` | string | yes | snapshot root the capture required (D38). `served.root` (trailing slash stripped) must equal it, end with `/<expect_root>`, or end with `<expect_root>`; a mismatch is refused even when `root_verified` is true |
| `root_verified` | bool | yes | must be `true`; a false value is refused |
| `recipe` | object | yes | `{"endpoint", "max_tokens", "temperature", "top_p", "prompt_logprobs", "ignore_eos"}`; `prompt_logprobs >= 1000` so top-1000 is measurable |
| `suite` | object | yes | `{"path", "sha256", "prompt_count"}`; the prompt-suite sha256 is the provenance hash of the prompts |
| `blocks` | list | yes | non-empty list of block objects |

## Block

| field | type | required | meaning |
| :-- | :-- | :-- | :-- |
| `block_id` | string | yes | unique within the capture |
| `class` | string | yes | `repeat`, `idle_distractor` or `busy_distractor` |
| `started_utc`, `finished_utc` | string | yes | UTC |
| `prompt_len` | int or null | yes | `int` only when every record in the block has that same `prompt_len` (repeat blocks); `null` when the records differ (distractor blocks). It is metadata, never used to infer a pair's shape - each record carries its own `prompt_len` |
| `batch_size` | int | yes | the block's metadata batch size, `1` for every single-request capture. The validator only requires `int >= 1` (a deliberate batch-shape distractor may use another value); server concurrency (the busy block's co-tenancy) is not request batch size and never changes this field |
| `idleness` | object | yes | the load-state proof, below |
| `pairs` | list of `[record_id, record_id]` | distractors yes, repeat optional | the comparisons this block declares. Required for both distractor classes; for `repeat` the pairs may be omitted because the validator pairs adjacent identical request bodies itself |
| `requests` | list | yes | at least one record (a repeat block needs the same body sent twice) |

### Idleness object

```json
{
  "metrics_source": "/metrics",
  "metrics_before": {"num_requests_running": 0, "num_requests_waiting": 0, "raw_sha256": "..."},
  "metrics_after":  {"num_requests_running": 0, "num_requests_waiting": 0, "raw_sha256": "..."},
  "metrics_during": [{"num_requests_running": 1, "num_requests_waiting": 0, "raw_sha256": "...", "at_utc": "..."}],
  "observed_other_traffic": false,
  "external_peers": [],
  "idle_verdict": "idle"
}
```

- `metrics_source` must be exactly `/metrics`.
- `raw_sha256` is the sha256 of the raw Prometheus text the parsed counters came from. It is the
  provenance hash for the proof; the raw text need not be embedded.
- `metrics_during` is optional. A capture may sample while its own request is in flight
  (`num_requests_running == 1`, `num_requests_waiting == 0`); that is still idle. Any during
  snapshot with `num_requests_running > 1` or `num_requests_waiting > 0` is co-tenant traffic and
  must have been tagged busy by the capture; it is refused inside an idle block.
- `external_peers` names other clients/peers observed; empty means none.
- Class rules enforced by `d41_floors.py`:
  - the class tag and the load-state verdict must agree: `repeat` and `idle_distractor` require
    `idle_verdict == "idle"`; `busy_distractor` requires `idle_verdict == "busy"`. A mismatched
    tag is refused rather than silently mixed into the other class's floor.
  - `repeat` and `idle_distractor`: `idle_verdict == "idle"`,
    `observed_other_traffic == false`, `external_peers == []`, **both** `metrics_before` and
    `metrics_after` show `num_requests_running == 0` and `num_requests_waiting == 0`, and every
    `metrics_during` snapshot has `num_requests_running <= 1` and `num_requests_waiting == 0`.
    A non-zero baseline is not silently mixed in; the block must have been tagged `busy` by the
    capture.
  - `busy_distractor`: `idle_verdict == "busy"`, `observed_other_traffic == true`, and evidence
    of co-tenancy: a non-empty `external_peers` **or** a `metrics_during` snapshot with
    `num_requests_running > 1` or `num_requests_waiting > 0`.
  - `idle_verdict == "unproven"` (or any other value) is always refused: a capture cannot be
    classified after the fact.

## Record

```json
{
  "id": "idle-A-repeat-0",
  "prompt_len": 139,
  "batch_size": 1,
  "prompt_sha256": "<sha256 of json.dumps(prompt_ids)>",
  "request_sha256": "<sha256 of the exact request body bytes>",
  "response_sha256": "<sha256 of the raw response body>",
  "wall_s": 0.041,
  "repeat_of": null,
  "positions": [
    {"position": 129, "logprob_count": 1000, "top": [[367, -0.19884082674980164], [...]]}
  ]
}
```

- `positions` follows `scripts/oracle_native.py`: position `p` is `prompt_logprobs[p + 1]`
  (the next-token distribution after consuming position `p`), and `top` is the ordered
  `[token_id, logprob]` list, sorted by logprob descending then token id. Token ids are unique;
  logprobs are finite. A token-keyed dict is not accepted.
- `logprob_count` may be omitted; when present it must equal `len(top)`.
- `position + 2 <= prompt_len` for every position (the position must exist in the prompt).
- `repeat_of` is optional provenance. The validator pairs a repeat block's records by identical
  `request_sha256` + `prompt_sha256` + `prompt_len` + `batch_size`, not by this field.
- No generated token is compared anywhere (D39): only the two distributions.

## Pair semantics

- `repeat`: the same request body sent twice consecutively; the pair must be bit-identical
  (KL 0, `|delta| 0`, top-k overlap 1.0, rank-1 same) or `d41_floors.py` refuses the run. This
  class is the zero-noise control and never contributes to a bar.
- `idle_distractor`: close-margin prompt pairs at different shapes, captured while the instance
  is idle. This is the class the D41 gate bars are derived from. Records in one pair must share at
  least one position and differ in `prompt_len` or `batch_size`.
- `busy_distractor`: the same kind of pair captured while other work is in flight. It records the
  load-state envelope; it is **not** the D41 gate floor (docs/build/task32-evidence/README.md,
  task32-root-cause.md section 4.3), and the floor file records it as a separate class.

## Floor and bar output (what the validator emits)

`scripts/d41_floors.py` writes one JSON document (`schema` `"sparkflash-d41-floors/1"`) and one
markdown table. The JSON carries:

- `provenance`: each input capture's path and sha256, the floor script's own sha256, the
  prompt-suite sha256, served root, and the tool/argv used;
- `capture_validation`: per class, block and pair counts and the idleness evidence summary;
- `classes`: per class, per-metric values (every pair and position comparison), the median/p90/max
  used as the floor, and outlier records;
- `bars`: for each D41 bar name, the provisional D40 value, the measured floor, the rule, the
  derived value and whether it requires review; plus `gate_bars` with the plain
  `{name: value}` mapping `scripts/compare_native_vs_vllm.py` consumes;
- `dropped_bars`: `top5_overlap_mean_min` is dropped by D41 (the measured floor is below the
  provisional bar; k >= 20 only).

Exit codes: `0` floor computed; `2` capture refused (schema, provenance, idleness, repeat
non-zero, sample too small, a required class missing) and nothing written; `3` the floor requires
a bar looser than the D40 provisional (written, labelled `requires_review` - the D41 record has to
say so, it is not absorbed).

## Sample minima

The floor script enforces its own counting (a capture cannot declare its way past it):
`--min-pairs` (default 3) per required class and `--min-position-comparisons` (default 12) per
required distractor class. A capture below either is refused by name, not padded.
