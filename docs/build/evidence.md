# Evidence and fixtures

What a run leaves behind, what each artifact means, and which script consumes it. Process rules for
which runs to do are in [`../RULES.md`](../RULES.md).

## Run directory (every four-rank row)

```text
<window>/<row>/
  r0/ .. r3/               per-rank evidence
  argv.txt                 frozen argv, one line per rank
  summary.txt              the run summary
  row.txt                  row tag, commit, binary sha256, clean-row notes
```

Each rank directory contains at minimum:

| File | Contents | Consumer |
| :-- | :-- | :-- |
| `chain-rN.rc` | process exit code | `identity-gate.sh`, `device-frontier.py` |
| `tokens.ids` | generated token ids | both gates |
| `logits_dump.bin` + `.jsonl` | f32 logits rows + index (position, argmax, ids, offset) | `numerical-gate.sh` |
| `logits_dump.bin_sha256` | sha256 of the logits blob | `identity-gate.sh` |
| `embed-rN.json` | `evidence_mode`, `evidence_digest_calls`, `layer_chain` | `identity-gate.sh` |
| `tp4-host-rN.json` | host-path counters, step times, load/RSS | evidence, docs |
| `chain-rN.log` | stage/launch log, refusal text | `device-frontier.py` |
| `collective_check` (device row) | per-rank collective count/sequence | `check-collective-count.py` |

Common fields in the JSON records: `schema`, `commit`, `binary_sha256`, `device_execution` (false on
the host path), `evidence_mode`, `world`, `rank`, and the frozen argv. Evidence is immutable: never
edit a row to fit a gate; re-run or freeze a new reference.

## Speed rows

A speed row is the nine-token frozen fixture at a fixed step count. Its identity is
`fixture_key = sha256(positions, prompt_len, token_ids)`. The published row is one line in
[`speed-log.csv`](speed-log.csv): date, commit, run name, mean step, token/logit PASS, note.

The reference artifact for `numerical-gate.sh` lives in
[`reference/speed-gate/<fixture>/`](reference/speed-gate/): the frozen CPU-twin logits plus the
fixture metadata and the reference tokens. The gate reports the reference top1-top2 margin with each
row; a new prompt has no reference until one is frozen.

## Device-arm evidence

`scripts/device-frontier.py --row-dir <row>` writes/validates the stage frontier record:

| Field | Meaning |
| :-- | :-- |
| `schema` | frontier schema version |
| `commit`, `binary_sha256` | identity of the run |
| `layers[].stage` | last stage reached per layer |
| `layers[].verdict` | `PASS`, `REFUSED` (open refusal, exit 3), `FAIL` |
| `layers[].kernel`, `layers[].host_step` | blocker named by the contract |
| `collective_check` | per-rank count/sequence for the two per-layer reduces |

A refusal is a valid frontier result, not a failure. An exit 2 (record with no `collective_check`)
is a stop: the row cannot testify, so it is not a pass.

## Host-gate artifact

`scripts/gate-artifact.py` (invoked from the qualification path) emits one machine-readable JSON
per gate:

```json
{
  "schema": "sparkflash-host-gate/1",
  "gate": "qeval",
  "state": "pass|fail|not_run|stub",
  "value": 72,
  "bar": 75,
  "unit": "of 75",
  "command": "...",
  "commit": "...",
  "binary_sha256": "...",
  "artifacts": ["..."]
}
```

The artifact is written next to the run; a gate without an artifact is a plan, not evidence. The
emitter is self-tested by the host-gate CI job.

## Phase-1 boundary table

The per-layer boundary capture is a manifest plus a binary blob (18 tensors per row: boundary
tensors, residual, selected ids, norms and the routing record). Layout rules, tensor order and the
comparison classes are fixed by `phase1-boundary-table.md` (archived). The perturbation test proves
the table is sensitive to a middle-layer change; the replay protocol substitutes a captured input
and checks the downstream boundary. Use the emitted manifest, never a hand-built row.

## D41 capture and floors

`d41-window.py` captures oracle rows; `d41-floors.py` derives floors. The schema is versioned
(`sparkflash-d41-capture/1`):

| Level | Fields |
| :-- | :-- |
| top | `schema`, `world`, `model_root`, `fixture`, `idleness`, `blocks` |
| block | `id`, `state` (`idle`/`busy`/`idle_b`), `records` |
| record | `position`, `token`, `topk` ids/logprobs, `prompt_len`, `batch_size`, timing |

The floor output is a JSON bars file (`d41-floor.json`, `d41-floor-warm.json`) with per-metric
minima and the shared support used for KL. The programme is parked (D52); do not freeze a new bar
from a busy capture. `d41-bars-dryrun.py` checks the bars against a capture without freezing.

## Oracle (D38)

`native-oracle-suite.json` is the vLLM oracle shape consumed by `scripts/compare_fixture_vs_oracle.py`
and the D38 A/B runner: suite sha256, model root, per-prompt ids, per-position top-k logprobs
(`top_logprobs`), and the continuation sentinel 0. The device side is a fixture JSON with
`provenance.node`, prompt ids, positions and f32 logits. The runner verifies the suite hash, the
prompt-prefix relation and the next-token target before comparing.

## Frozen bundles in this directory

| Path | Contents |
| :-- | :-- |
| `task32-evidence/` | Offline D41/three-way records: control, floor, shape, repeat, batch-mode. Inputs are identified by sha256 in its README; large dumps stay on the node. |
| `task53-evidence/` | Per-rank `tp4-host-rR.json` for the resident four-node run: tokens, rchar/disk per token, mean step, RSS. |
| `d38-vllm-baseline-20261021.json` | CI perf-regression baseline (`scripts/perf_regression_gate.py`). |
| `d41-*.json` | Frozen D41 floors/bars and their metadata. |
| `reference/speed-gate/` | Frozen CPU-twin reference for the numerical gate. |

## Rules

- One row, one directory, one identity. A row copied to another directory is still the same row;
  relabeling it does not make it a new measurement.
- The binary sha256 is part of the identity. A rebuilt binary is a different run.
- Store the raw artifact, the gate output and the command that produced them. A summary alone is
  not evidence.
- Do not commit large blobs; record their node path and sha256 in the bundle README.

## Archived sources

`phase1-boundary-table.md`, `logits-dump.md`, `gate-artifact.md`, `d38-vllm-ab-protocol.md`,
`d41-capture-protocol-v2.md`, `d41-floor-schema.md`, `d41-floor-computation.md`,
`task32-evidence/README.md`, `task53-evidence/README.md`, `task32-root-cause.md`,
`d48-window-protocol.md`.
