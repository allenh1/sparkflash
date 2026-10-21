# D38 A/B protocol: device-arm token/logits dump vs the vLLM oracle (2026-10-20)

**Status: prepared for the next token window; not yet executed.** No device forward token exists as
of the m1-bind window (forward not run; the routed down-projection launch compatibility is the named
blocker). This page fixes the steps and artefact layout so the first token run produces a
gate-grade comparison in one pass. It makes no token-equality claim (D39) and no tok/s claim.

## What is compared

| side | producer | evidence |
| :-- | :-- | :-- |
| device arm | TP=4 device chain forward stage, frozen tag `b69ebb2` | `sparkflash-tp4-token/1` envelope per rank + the frozen full-vocabulary logits dump (rank0 primary) |
| vLLM oracle | released-snapshot serving instance, same prompt suite | `sparkflash-native-oracle/1` JSON with per-position top-k `logprobs` and `target_token` |

The sides are joined on the integer position; the comparison is distributional (D39): top-1 hit,
top-k overlap, KL/JS, absolute logprob deltas, and a prefix-match check. Token ids are a diagnostic,
never an equality gate.

## The device-side contract (published to device-lead 2026-10-20)

One `tp4-token-r<R>.json` per rank in the same rank directory as `tp4-bind-rR.json` and
`tp4-loop-rR.json`, schema `sparkflash-tp4-token/1`:

```json
{
  "schema": "sparkflash-tp4-token/1",
  "stage": "tp4-chain", "loop_step": "C4-token",
  "rank": 0, "world": 4, "forward": "run",
  "tokens": {"ids": [12345], "positions": [129], "start_position": 129, "prompt_len": 130,
             "argmax_logits": [18.75], "top1_margins": [0.42], "logits_sha256": "<hex>"},
  "positions_run": 1,
  "step_ms": {"samples": [31.2], "mean_ms": 31.2, "worst_ms": 31.2},
  "wall_ms": [47.8], "reduce_calls_per_step": 80,
  "head": {"rule": "rank-sharded head rows all-gathered over tcp; every rank argmaxes the full vector; rank0 dumps full logits", "vocab": 129280, "tie_break": "lowest token id"},
  "logits_dump": {"path": "tp4-logits-r0.jsonl", "bin": "tp4-logits-r0.jsonl.bin",
                  "sha256": "<hex>", "bin_sha256": "<hex>", "dtype": "f32-le",
                  "vocab": 129280, "positions": [129], "rows": 1},
  "error": null
}
```

The dump is exactly the frozen logits-dump schema (`docs/build/logits-dump.md`): JSONL rows with
`position, device_argmax, logits_argmax, reference_id, hybrid_id, fixture_id, exact_token_equal,
vocab, logits_offset, logits_bin, top20`, plus a sidecar `<dump>.bin` of consecutive little-endian
f32 full-vocabulary blocks with no padding. Rank 0 is the primary dump owner; replicas are allowed
but must have identical sidecar bytes. All ranks must emit the same token id (rank-consensus rule).

Refusal is a labelled outcome, not a bind failure: `forward: "refused"`, empty `tokens.ids`,
`positions_run: 0`, `logits_dump: null`, a named `error`, nonzero process exit. No token JSON at
all means the forward stage was not run.

### Exact device feed for the `b69-fixture-130` sequence (130 prompt + 9 steps)

The fixture `/home/user/phase2-evidence/sequence-130-released.json` has `prompt_tokens` of length
130 and `steps` at positions 129..137 with `next_token`
`[491, 367, 201, 80, 3302, 201, 54, 201, 4341]`. The device run is **teacher-forced**: it feeds the
130 prompt ids plus the first **8** `next_token`s (the 9th is only the final prediction, never a
device input), i.e. **138 ids total**, and emits **9 full-vocabulary logits rows at positions
129..137**.

| item | value |
| :-- | :-- |
| sequence to feed | `prompt_tokens` (130) + `steps[0..7].next_token` = 138 ids |
| positions to read/dump | 129,130,131,132,133,134,135,136,137 (9 rows) |
| `tokens.ids` in the envelope | the 9 argmax ids of those rows (rank-consensus) |
| dump `reference_id` per row | `[491, 367, 201, 80, 3302, 201, 54, 201, 4341]` |
| oracle positions to join | the same 9 positions; canonical oracle prompt id `b69-fixture-130` |

The oracle side is longer by construction (suite 130 + 9 continuation + one sentinel = 140 prompt
tokens), because it needs `prompt_logprobs` at index `p+1` for every device position. The mapping is
`p = vLLM prompt_logprobs index - 1`. A dump row with a null `reference_id` cannot prove the prefix
and is counted as a prefix mismatch, so the forward loop must populate `reference_id` (or
`hybrid_id`) from the token table for a gate-grade comparison.

## The vLLM oracle

### The canonical oracle is already on disk (`native-oracle-suite.json`)

`/home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json` (153 MB) is a
`sparkflash-native-oracle/1` document captured against the released root (`root_verified: true`,
served root `.../snapshots/1bae0e10...`). It contains prompt **`b69-fixture-130`**: prompt_len 140
(130 suite + 9 fixture continuation + sentinel 0), 139 positions (0..138), 1000 logprobs per
position, `target_token` per position, `continuation.source` pointing at
`sequence-130-released.json`. The runner consumes it today and it makes the next device token run
**measurable offline, with no redeploy of the released-snapshot vLLM**: the oracle is a file, not a
live service. Freshness caveat: it was generated at `2026-10-19T17:30:37Z` against the released
root; that is acceptable for D38 (root verified), and a refresh is only needed if the Lead wants a
newer oracle capture slot.

### The simplified `native-oracle-released.json` is NOT consumable by the runner

`/home/user/ev-session-20261020/qa/artifacts/native-oracle-released.json` is a reduced ad-hoc
record: keys `source`, `served_root`, `fixture`, `prompt_len`, `positions`, `meaning`, `wall_s`,
`response_sha256`, `vocab_size`, `per_position` (9 entries keyed by device position, each with
`dist_size`, `top1`, `top`, `logprobs`). It has **no `schema` key and no `prompts` list**, so
`accept_device_token.py --oracle` rejects it with exit 2 (`unknown oracle schema None`). Do not pass
it to the runner. Minimal alternative: use the canonical suite above (no re-capture needed); only
if the 16:11 numbers are specifically required would a host-only converter be written, and that
conversion still needs the exact request prompt/recipe pinned first.

### Canonical oracle shape consumed by the runner

Required: schema `sparkflash-native-oracle/1`; `status: ok`; `served.root` matching the released
snapshot and `root_verified: true`; `prompts[]` entries with `id`, `status: ok`, `prompt_len`, and
`positions[]` of `{position, target_token, logprobs: {token_id: logprob, ...}}`. The comparison uses
the oracle's logprobs support (recipe cap 1000): top-1 agreement, top-k overlap at 5/20/100/1000,
max/mean absolute logprob deltas, KL/JS divergence, and a prefix match of `target_token` against the
dump row's `reference_id`/`hybrid_id`. Only a `--oracle-bars` file with schema
`sparkflash-d41-bars/1` can turn those numbers into a gate; without one the leg is `UNBARED`, and
the published D40 values are a diagnostic labelled `D40-PROVISIONAL-NOT-A-GATE`.

### The twin leg: what `--twin` expects and how to produce it

`--twin` takes one or more files/directories of **host-side token evidence**, scanned recursively
(JSON only, dumps never auto-promoted). Accepted layouts:

| layout | token field | maker |
| :-- | :-- | :-- |
| `sparkflash-device-token-evidence/1` (canonical) | `token_ids` (+ optional `positions`) | hand-writable sidecar, or `scripts/twin_evidence.py` |
| `sparkflash-tp4-token/1` envelope | `tokens.ids` / `tokens.positions` | a host-side or second arm run |
| `tp4-host` evidence | `generated_tokens` (`step_tokens` are not token ids) | `--run-stage tp4-host` |

The comparison joins on explicit positions when **both** sides name them, otherwise on sequence
index; all ranks carrying tokens must agree; PASS only when every compared position is equal
(`--twin-positions-min`, default 1). A twin PASS plus an `UNBARED`/`UNMEASURED` oracle is the M1
token gate: overall PASS, exit 0, with the oracle recorded and not numerically gated.

> **Never pass the device run tree itself as `--twin`.** A tree compared with itself trivially
> PASSes; the runner cannot detect provenance, so this is a reviewer rule. The twin must be an
> independent host-side run (or the declared host-reference fixture below).

**Option A - independent host twin (needs the four nodes, about 7-8 min, no GPU).** After the
device token run and before vLLM restore, run the Phase-1 host path with the same released
checkpoint, the 130 fixture prompt ids and `--max-tokens 9`, one rank per node:

```sh
# from the dev host; BIN/CKPT/PEERS/EV_TWIN from the window wrapper, PORT free
for R in 0 1 2 3; do
  ssh <node-R> "mkdir -p $EV_TWIN && cd $EV_TWIN && nohup $BIN device-chain --run-stage tp4-host \
    --tp-rank $R --tp-world-size 4 --comm-peers $PEERS --comm-base-port $PORT \
    --checkpoint $CKPT --prompt-tokens <130 comma-separated fixture ids> --max-tokens 9 \
    --evidence-dir $EV_TWIN > twin-r$R.log 2>&1; echo \$? > twin-r$R.rc" &
done; wait
# copy back to /home/user/ev-session-<date>/window/<twin-run>/rankR/ then:
#   --twin /home/user/ev-session-<date>/window/<twin-run>
```

It writes `tp4-host-r0..r3.json` + `tp4-host-summary.json`; pass the directory directly. It is the
host sharded Phase-1 path (not vLLM TP=4), used as the independent twin. This cannot run on the dev
host for the real checkpoint: world 4 needs about 4 x 85 GiB resident (world 1 needs about 286 GiB);
the dev host has ~131 GiB available.

**Option B - dev-host-only host-reference twin (no nodes, seconds).** The pinned fixture already
contains the host reference-engine token sequence (`steps[].next_token` on the released
checkpoint). Convert it to the canonical twin document:

```sh
python3 scripts/twin_evidence.py \
  --fixture /home/user/phase2-evidence/sequence-130-released.json \
  --prompt-id b69-fixture-130 --out-dir /home/user/ev-session-<date>/qa/window/<run>-twin
# -> twin-r0.json, kind host-twin, execution host-reference-fixture, positions 129..137,
#    token_ids [491, 367, 201, 80, 3302, 201, 54, 201, 4341]
```

Then add `--twin /home/user/ev-session-<date>/qa/window/<run>-twin`. This is labelled
`host-reference-fixture`, not a second independent four-rank run; the Lead decides whether it
satisfies the D40 twin gate or whether the twin leg stays UNMEASURED pending Option A.

**Negative window (forward refused).** A refused forward produces no token ids: the token row is
`NOT-ATTEMPTED` with the named blocker, the oracle leg is `UNMEASURED` (there are no explicit
positions to join; nothing is guessed), and the overall row is `PRE-TOKEN-PASS` with exit 3. Do not
supply `--twin` for that run: twin absence is NOT_ATTEMPTED, never FAIL.

## Step-by-step window order

1. **Verify the serving instance** is the released snapshot (D38 root) and idle. If the window order
   is vLLM -> device, restore vLLM only after the device arm's GPU work; record the times.
2. **Run the device forward** with the frozen binary/cubins and dump flag, writing evidence to
   `/home/user/ev-session-<date>/window/<run-name>/rank<R>/`; the additive files are
   `tp4-token-rR.json`, `tp4-logits-r0.jsonl`, `tp4-logits-r0.jsonl.bin`; the pre-token
   bind/walk/names files must stay byte-identical.
3. **Capture/verify the oracle** for the same prompt; record sha256 and root.
4. **Run the one-command acceptance** with the oracle, pinned positions and the D41 bars file if one
   exists; without a bars file the oracle leg is explicitly `UNBARED`, which is not a pass.
5. **If a D41 bars file exists**, record the direct distributional comparison:

```sh
python3 scripts/compare_native_vs_vllm.py \
  --dump <evidence>/rank0/tp4-logits-r0.jsonl \
  --oracle /home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json \
  --oracle-prompt b69-fixture-130 --positions 129,130,131,132,133,134,135,136,137 \
  --expect-vocab 129280 --bars <d41-bars.json> \
  --out-json <qa>/o4-token.json --out-md <qa>/o4-token.md
```

   Without `--bars` the same command records the metrics labelled `UNBARED`
   (D40-PROVISIONAL-NOT-A-GATE); it never becomes acceptance evidence.
6. **Twin leg (separate row, never mixed with the oracle)**: if the host twin emits its own
   `sparkflash-tp4-token/1` evidence, pass it with `--twin`; the twin comparison is device vs host,
   the oracle comparison device vs serving. Device and host numbers never share a table.
7. **Record**: copy all artefacts, write sha256 manifests, commit the dated record with exact
   commands and sha256s, label every row with its explicit verdict.

## The one command

```sh
python3 scripts/window_acceptance.py \
  --device-dir /home/user/ev-session-<date>/window/<run-name> \
  --out-dir /home/user/ev-session-<date>/qa/window/<run-name> \
  --positions 129,130,131,132,133,134,135,136,137 \
  --expected-steps 3 --expected-root 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --expect-attention-bound 208 \
  --expect-moe-bounds routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120 \
  --expect-manifest-lines 185481 \
  --oracle /home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json \
  --oracle-prompt b69-fixture-130 [--provisional-d40] \
  [--twin <host-twin evidence>] [--require-token]
```

It writes `window-acceptance.json/.md`, both component reports and logs, `evidence-sha256.txt` and
`artefacts-sha256.txt`. Rows are explicit: bind PASS is a real measurement; token NOT-ATTEMPTED means
the forward stage did not run; token MEASURED with oracle UNBARED is a recorded measurement with no
declared gate; step time is a step-budget row, with accepted tokens/step and tok/s labelled
NOT-ATTEMPTED until a real DSpark/MTP verify step measures them.

Exit codes for a token-measured run: with the canonical oracle and **no** D41 bars file (the only
kind of bars that exists), the oracle row is `UNBARED` and the overall row is
`TOKEN-MEASURED-UNBARED`, **exit 3** - even when `--provisional-d40` is passed, which only records a
diagnostic. If a host-twin evidence tree is supplied with `--twin` and the twin comparison PASSes
the M1 token gate, the acceptance runner's overall verdict is PASS (oracle recorded, not numerically
gated) and the window command mirrors it as **exit 0**; without a twin, exit 0 is impossible. Exit 1
is a failure, exit 2 is bad input (including a non-canonical oracle file or a root mismatch).

## Artefact layout

```text
/home/user/ev-session-<date>/
  window/<run-name>/                 # node evidence, Lead-owned, read-only after the run
    rank0..rank3/tp4-bind-rR.json    # pre-token evidence (m1-bind shape, byte-identical)
    rank0..rank3/tp4-loop-rR.json
    rank0..rank3/tp4-token-rR.json   # additive forward evidence (schema above)
    rank0/tp4-logits-r0.jsonl(.bin)  # rank0 primary dump
    argv.txt, run.meta, chain-rR.log, chain-rR.rc
  qa/window/<run-name>/              # outputs of this workspace
    d48-step-gate.json/.md, accept-device-token.json/.md
    window-acceptance.json/.md, evidence-sha256.txt, artefacts-sha256.txt
  qa/o4/<run-name>/                  # optional: direct compare_native_vs_vllm record
    native-vs-vllm.json/.md
```

## What this protocol does not claim

- token equality: D39 forbids a token-equality gate; token ids are a diagnostic only;
- a pass without a D41 bars file: unbared metrics are recorded, not accepted;
- twin-match as an oracle: device-vs-host and device-vs-serving are separate rows;
- tok/s: accepted tokens/s requires the real verify step (D48/D50); ms/step alone is target-only;
- anything measured against a different checkpoint root: D38 root verification is a precondition,
  never bypassed.
