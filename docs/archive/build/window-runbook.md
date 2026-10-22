# vLLM window runbook (Lead-owned)

Status: recorded 2026-10-18 from the Lead's window protocol. The window itself is **Lead-owned**: the
Lead stops and restarts the reference vLLM instance; teammates ask for a window and get one, they do
not stop the server. This file exists so the protocol survives the window: a restart that is
improvised from memory is how a live client loses its server.

## Why the window has to be short and why it always restarts

A live client at `192.0.2.71` uses the vLLM server at `http://192.0.2.80:8000`. While vLLM is
stopped, that client has no model. So:

* the window is as short as the evidence run needs (device work is prepared and staged on the nodes
  before the stop);
* **the server is always restarted at the end of the window**, from the recorded launch command, and
  the health check below is run before the window is called finished - not left to the next person.

## The canonical launch recipe

**Update 2026-10-19 (D38 actioned, D39):** the canonical recipe now serves the **released** checkpoint. /home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe-released.sh (sha256 a3ac461fa02a945181a17b509f7695ca39b38930eee529a0f9fd9151061072c4) is the previous served recipe (vllm-launch-recipe-heretic-12d58ad0.sh, sha256 2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574, retained) with the model path switched to deepseek-ai/...-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 and --max-logprobs 1000 added for the numeric comparison; window-vllm.sh (sha256 a3ba6763cc8f5087ab65dc7771e28e37e9b3a18842d2e72542c3f22f7344ef70) starts it. The served root is recorded from /v1/models in every evidence log. The old paragraph below is kept as history.

Preserved at
`/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh` (1,809 B, sha256
`2a3e2c7719532149d3291d11964ace46bf27d9a0b656c39258d5a6295c552574`), generated from the recipe
`deepseek-v4.1-flash-1017-heretic`. It exports the model/NCCL environment and ends in the
`vllm serve` command; the NCCL part is single-rail (`NCCL_IB_HCA=rocep1s0f1`, see
`rdma-backend.md`, section 3, for why the launcher's twin-pair value does not win).

## Start (the Lead's command)

On the head node (`node0`), from `/home/user/spark-vllm-docker/` - the launcher directory
lives on the node, not on the dev host - the recorded start is:

```sh
./launch-cluster.sh --apply-mod deepseek-v4.1-flash-1017-heretic \
  -n 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --eth-if enp1s0f1np1 --ib-if rocep1s0f1,roceP2p1s0f1
```

The flag spelling and the `--apply-mod` recipe name are as reported by the Lead from the 2026-10-18
window; the launcher script is not on this dev host, so this file cannot re-check its usage text. The
preserved recipe is the canonical environment (`--apply-mod` is what regenerates it); if the launcher
and the recipe disagree on `NCCL_IB_HCA`, the recipe's export is what the running container shows.

## Stop (the Lead's action)

The `launch-cluster.sh` `stop` action, which runs `docker stop vllm_node` on all four nodes. Two
practical points:

* drain/quiet the client first if the window is long; the client at `192.0.2.71` fails from the
  moment the server stops;
* the device evidence run must be staged before the stop, because staging over the network while the
  server is down is window time wasted.

## Launching evidence processes on the nodes: the ssh trap

Every rank is one process on its own node, started over ssh. Two S5.1 attempts failed on the
launch, not in the code: `ssh host 'nohup binary ... &'` **without `-n`** keeps the session open
until the backgrounded child exits, because the child inherits the channel. The second rank then
starts after the first rank's connect deadline, and the run fails with `Connection refused` while
the first rank reports a truncated hello handshake - even though the transport is correct. Use the
launch the S5.1 gate ran with (both ranks as local ssh children of one shell, so both sessions are
backgrounded before either wait):

```sh
ssh -n node0 "binary --tp-rank 0 ... > /home/user/phase2-evidence/.../r0.log 2>&1" &
ssh -n node1 "binary --tp-rank 1 ... > /home/user/phase2-evidence/.../r1.log 2>&1" &
wait
```

`ssh -n` with a detached child (`nohup setsid binary ... < /dev/null &`) is the other correct
form. Two further points, both measured:

* a rank that loads a checkpoint before its first collective needs more than the 5 s transport
  defaults; pass `--comm-timeout 120 --comm-connect-timeout 120` (the defaults stay 5 s);
* never probe a listening rank with `/dev/tcp` just to check: a connection that is not the
  handshake is a protocol violation, the peer logs it and the run is lost.

## Restart and health check

Restart with exactly the recorded start command above (not a hand-typed variant: `vllm serve`
parameter drift is a silent behaviour change). Then:

```sh
curl -fsS -o /dev/null -w '%{http_code}\n' http://192.0.2.80:8000/v1/models
```

Expect `200`. The window is finished only after this returns 200 and the client's next request
succeeds; if the health check fails, the window is not over and the server is restarted again.

## What this file does not do

* It does not automate the stop/start from the dev host; the dev host has no access to the nodes.
* It records no timings or throughput of the vLLM instance (`rdma-backend.md` and the qualification
  gates own those questions).
* The exact `launch-cluster.sh` usage text and any other `--apply-mod` recipes are on
  `node0`; this runbook records the one window command the Lead reported, not a catalogue.

## Native oracle and distributional comparison (O2/O3/O5, 2026-10-19)

Harness: `scripts/oracle_native.py`, `scripts/compare_native_vs_vllm.py`, suite
`scripts/oracle_prompt_suite.json`. This adds a step after "Restart and health check": the window is
still Lead-owned, and the stop/start protocol above is unchanged.

### 1. Device dump (device window, before the released server starts)

The device side already writes the comparison input with `--dump-logits` (O6, in tree:
`crates/sparkflash-cli/src/logits_dump.rs`; schema [logits-dump.md](logits-dump.md)): one JSON object per
compared position (position, device argmax, vocab, logits_offset, top20, ...) plus `<dump>.bin`
with the full-vocabulary little-endian f32 logits in the same position order. The dump is written before the token-parity gate can fail, so a failing run still
leaves its logits. Stage the run and the dump before the stop; the comparison is offline afterwards.

Check the pair before the restart (the comparison refuses a mismatch, but catching it here is cheaper
than discovering it after the window):

```sh
python3 - <<'EOF'
import json, os
dump = "<run>/logits-130x8.jsonl"
rows = [json.loads(line) for line in open(dump) if line.strip()]
assert [row["position"] for row in rows] == list(range(129, 138)), [row["position"] for row in rows]
assert all(int(row["vocab"]) == 129280 for row in rows), {row["position"]: row["vocab"] for row in rows}
assert os.path.basename(rows[0]["logits_bin"]) == os.path.basename(dump) + ".bin"
assert all(int(row["logits_offset"]) == sum(int(prev["vocab"]) * 4 for prev in rows[:i])
           for i, row in enumerate(rows))
assert os.path.getsize(dump + ".bin") == sum(int(row["vocab"]) * 4 for row in rows)
print("dump/sidecar pair consistent:", len(rows), "positions")
EOF
```

A dump whose rows name another sidecar, whose sidecar is the wrong length, whose offsets do not
tile, or whose vocab is not 129,280 is refused by the comparison (exit 2, nothing written) instead of
being compared silently. The comparison command passes `--expect-vocab 129280` so the vocab check is
enforced in the gate run, not only in this pre-flight.

### 2. Oracle (after the released recipe answers /v1/models)

```sh
python3 scripts/oracle_native.py \
  --suite scripts/oracle_prompt_suite.json \
  --fixture /home/user/phase2-evidence/sequence-130-released.json \
  --out /home/user/phase2-evidence/window-2026-10-18/native-oracle-130x8.json \
  --wait-seconds 2700
```

* The served root is read from `/v1/models` and must end with the released snapshot root
  `1bae0e1088993b9b82de08f59de5be5a34e42170`; a mismatch exits 2 **before** any completion request
  (D38: an oracle run against another snapshot proves nothing about this checkpoint).
* Request: `prompt = suite ids + continuation + one sentinel token`, `prompt_logprobs=1000`
  (the released recipe passes `--max-logprobs 1000`), temperature 0, max_tokens 1. The continuation is
  the fixture's `steps[].next_token` list, so the server is teacher-forced on the same prefix the device
  compared and returns the decode-position distributions too. Without `--fixture`/`--continuation`
  a single sentinel covers positions `0..n-1`.
* Position convention: vLLM's `prompt_logprobs[i]` scores prompt token `i` given `0..i-1`; entry 0 is
  null (no context). That is the distribution a logits row at position `p=i-1` produces, so the oracle
  JSON stores positions keyed by **device position** `p`; the comparison joins the two files on it.
* Evidence: served root, `/v1/models` body sha256, suite sha256, tokenizer sha256, per-prompt request and
  response sha256, wall time and the prompt_logprobs integer are all written into the JSON and a
  timestamped `.log` beside it.
* The script waits for the server (bounded by `--wait-seconds`) instead of being polled by hand; a model
  load of more than 25 minutes is expected, so a retry loop is normal. Re-running is idempotent for the
  same suite and continuation; the file is evidence, not state.
* **The 138-token oracle cannot satisfy `positions_min = 9`.** `native-oracle-released.json`
  (prompt_len 138, sha256 `8ecc423d…`) covers p129..p136 only - entry 0 is null, so 138 prompt tokens
  give positions 0..136 - which is 8 compared positions. The 130 + 8 comparison needs p129..p137: use a
  140-token oracle (the suite run, `native-oracle-suite.json`, sha256 `efce16c2…`, prompt_len 140,
  positions 0..138) or a fresh `--only b69-fixture-130 --fixture …` request. Reusing a recorded oracle
  is allowed only after its sha256 and served root are re-verified in this window and the reuse is
  recorded verbatim in the run record.

### 3. Comparison (offline, on this host or anywhere)

```sh
python3 scripts/compare_native_vs_vllm.py \
  --dump /home/user/phase2-evidence/window-2026-10-18/runs70-accept/logits-130x8.jsonl \
  --oracle /home/user/phase2-evidence/window-2026-10-18/native-oracle-130x8.json \
  --oracle-prompt b69-fixture-130 \
  --positions 129,130,131,132,133,134,135,136,137 \
  --expect-vocab 129280 \
  --out-json /home/user/phase2-evidence/window-2026-10-18/runs70-accept/compare-native-vs-vllm.json \
  --out-md /home/user/phase2-evidence/window-2026-10-18/runs70-accept/compare-native-vs-vllm.md
```

It joins on position (without `--positions`: the intersection of dump and oracle positions; the gate
command pins the list below) and reports per position and in aggregate: top-1 agreement,
top-5/20/100/1000 overlap, max/mean |logprob delta| over the oracle's ids, KL(oracle || device) and JS
of the renormalised top-k, the oracle top-k retained mass, and a `prefix_match` flag (device reference
token vs oracle continuation token - a mismatch means the two distributions are conditioned on
different prefixes and the row is not like-for-like). It never compares token identity (D39 rule 2).
Add `--bars <D41 gate file>` to gate; without it the verdict is `UNBARED` (measured evidence, no
verdict). Exit codes: 0 `PASS` or `UNBARED`, 1 `FAIL` or `NOT_LIKE_FOR_LIKE`, 2 bad input (including a
bars file that is not a D41 gate file).

Shape rules that matter more than the flags:

* `--positions 129,130,131,132,133,134,135,136,137` is pinned, not optional, and every requested
  position must exist in BOTH the dump and the oracle - a missing or duplicate position is refused
  (exit 2, nothing written), never silently dropped by intersection. A prefill-only dump has 130
  positions (0..129); compared against the continuation oracle it would give 130 positions, pass
  `positions_min` trivially, and score mostly rows whose reference token is not the oracle's
  teacher-forced target (the sanity-prefix path, see logits-dump.md). The D40 shape is the 9-position
  full 130 + 8 claim.
* `--expect-vocab 129280` is passed by the gate command. Without it a self-consistent dump with the
  wrong vocab would be compared against the 129,280-vocab oracle and only the oracle id range would
  notice; any row disagreeing is refused (exit 2).
* `prefix_match` is tri-state. `UNKNOWN` means the dump could not prove like-for-like (no usable
  `reference_id`, or the oracle has no `target_token`); it is counted in `prefix_mismatches`, so a
  dump that cannot prove its prefix is not silently scored. Any UNKNOWN/MISMATCH row forces verdict
  `NOT_LIKE_FOR_LIKE` and exit 1 even when a gate file's bars pass and even when there is no gate file;
  the evidence file is still written with the `prefix_mismatches`/`prefix_unknown` counts.
* **Footgun:** `--reference-check` returns immediately after the reference pass
  (`device_chain.rs`, the `if sequence && args.reference_check` early return) and writes **no dump**.
  It is a reference-only diagnostic, never the dump run; using it in a window wastes the window.

### O5: the pinned prompt suite

`scripts/oracle_prompt_suite.json` (sha256 `42f66ad2d8d81ff555eeee2ef837e0cd6ca3cb59bde57320223d9c9b9ef19cd7`)
pins 8 prompts, 130-432 ids, with the b69 fixture prompt first and the rest spanning code, mathematics,
prose, a GPU-memory technical note, a mixed English/Chinese/French paragraph, a JSON tool-call schema
and a long quantisation essay. The **ids are the canonical input** (the text is provenance); the
released tokenizer's sha256 is recorded in the file, and every later comparison must pin this file's
sha256 rather than re-tokenising free text. The file changes only when the comparison set is
deliberately changed, and that change is recorded with its new sha256.

### Bars (Lead-owned, D40)

The comparison evaluates these names and refuses any other, so a typo cannot silently pass:
`positions_min`, `top1_agreement_min`, `top5_overlap_mean_min`, `top20_overlap_mean_min`,
`top100_overlap_mean_min`, `top1000_overlap_mean_min`, `max_abs_delta_max`, `mean_abs_delta_max`,
`kl_mean_max`, `kl_p95_max`, `js_mean_max`. Values are floors (`_min`) or ceilings (`_max`) over the
aggregate object.

A `--bars` file must be a D41 gate file: `{"schema": "sparkflash-d41-bars/1", "status": "ok",
"bar_class": "idle_distractor", "bars": {"top1_agreement_min": 1.0, ...}}`. Strictness is the
default, there is no permissive flag: a plain `{"bars": {...}}` file (demo-scale synthetic bars,
hand-copied provisional numbers), a `status: "requires_review"` floor file, or a busy-derived bar file
is refused (exit 2) with no evidence file, so none of them can gate silently.

### 4. B70 acceptance window (PRE-FIX DIAGNOSTIC, tip-agnostic)

The B70 window is the first machine-readable 130 + 8 dump from a main-built binary plus the released
oracle comparison. It is a **pre-fix diagnostic**: the native-arithmetic fixes (task-4..task-9) are not
in it and the D40 bars do not exist yet, so the run is labelled PRE-FIX, records numbers, and claims no
pass/fail. It is not acceptance evidence; a later window after the fixes reuses this procedure with a new
tip and a new run directory.

Never write into `runs70/`: it holds the 2026-10-18 b70-device dump (`f548d9c3…`/`43b31438…`) from the
diagnostic-branch binary `4f599b7` (merge-base `acc0189`, before the `!105`..`!128` main line), which is
named by sha in `docs/build/task32-root-cause.md` and is evidence, not scratch. Use `runs70-accept/`.

```sh
EVID=/home/user/phase2-evidence/window-2026-10-18
RUN=$EVID/runs70-accept
CKPT=/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
T=$(git -C /home/user/deepseek-v4.1-flash-spark rev-parse origin/main)   # main sha at staging time; re-tag the manifest here, never reuse 0639413's binary
MANIFEST=<worktree>/window-artifacts-${T:0:7}.txt
BIN=<node path of the binary built at T>          # sha256 in $MANIFEST
CUBINS=<node path of cubins-main-${T:0:7}>       # 19 files, sha256 in $MANIFEST
```

Staging: build at T (`scripts/build-aarch64.sh --features=cuda`), write `$MANIFEST` with
the binary sha256, the 19 cubin sha256s and the baseline success line, and run the pre-flight checks from
the staging section (fixture `e6df8290…` 2,536,194 B; suite `42f66ad2…`; flags present in the binary;
checkpoint directory basename `1bae0e10…`). If `kernels/` is unchanged from a previous tip the cubins
should be byte-identical, but the binary sha does not carry over.

Device run (vLLM stopped, no compute apps):

```sh
mkdir -p "$RUN"
date -u
timeout 6000 "$BIN" device-chain \
  --run-stage hybrid_sequence --checkpoint "$CKPT" \
  --cubin-dir "$CUBINS" \
  --fixture /home/user/phase2-evidence/sequence-130-released.json \
  --prompt-tokens 130 --decode-steps 8 --time-budget-s 4800 \
  --dump-logits "$RUN/logits-130x8.jsonl" \
  --attention-twin-positions 126,127,128,129,130 --operand-audit \
  > "$RUN/seq130x8.log" 2>&1
rc=$?; date -u; echo "RUN seq130x8 exit=$rc"
sha256sum "$RUN/logits-130x8.jsonl" "$RUN/logits-130x8.jsonl.bin" "$RUN/seq130x8.log"
```

The run must print `reference vs fixture: positions=9 all_equal=true`; if it does not, there is no dump.
Then run the dump/sidecar assertion from section 1, the oracle step with a fresh
`--only b69-fixture-130 --fixture …` request (or the `efce16c2…` suite oracle reused only after its
sha256 **and** served root are re-verified in this window and the reuse is recorded verbatim), and
the comparison with `--oracle-prompt b69-fixture-130`, the pinned
`--positions 129,130,131,132,133,134,135,136,137` and `--expect-vocab 129280`. A pre-fix token table
showing 0/9 `reference_id == hybrid_id` is expected here and is not a comparison failure. If any row's
prefix cannot be proven (UNKNOWN/MISMATCH), the comparison still writes the evidence and exits 1 with
verdict `NOT_LIKE_FOR_LIKE`; that is a prefix problem to investigate, not a bars failure.

Artefact slots (fill at run time; never invent a sha):

| artefact | path | sha256 |
| :-- | :-- | :-- |
| staging manifest | `$MANIFEST` | TBD |
| binary | `$BIN` | TBD |
| cubins | `$CUBINS/` (19 files) | TBD (`sha256sum *.cubin`) |
| dump | `$RUN/logits-130x8.jsonl` | TBD; 9 rows, positions 129..137, 11 fields |
| sidecar | `$RUN/logits-130x8.jsonl.bin` | TBD; 4,654,080 B |
| oracle | `$EVID/native-oracle-130x8.json` (+`.log`) | TBD; prompt_len 140, positions 0..138 |
| comparison | `$RUN/compare-native-vs-vllm.json` (+`.md`) | TBD; UNBARED until the bars file exists |

### 5. D41 floors: capture to bars (draft, not run)

The gate file is produced offline from captures; the comparison never invents it. **This sequence is a
draft for when the Lead opens the D41 capture window; preparing it must not start anything.** With the
released server up and genuinely idle, on the node (the plan pins the golden-fixture prompt shapes; plan
sha256 `22746d98123a4ee67b33edf18c432e5fe067728076a1494b1ebd2514727353a6`):

```sh
EVID=/home/user/phase2-evidence/window-2026-10-18
T=$(git -C /home/user/deepseek-v4.1-flash-spark rev-parse origin/main)   # sha recorded with the bars
python3 scripts/d41_recapture.py \
  --base-url http://127.0.0.1:8000 \
  --plan scripts/d41_capture_plan.json \
  --out "$EVID/d41-capture-$(date -u +%Y%m%dT%H%M%SZ).json" \
  --capture-id d41-b70 --tag d41 --expect-root 1bae0e1088993b9b82de08f59de5be5a34e42170
# offline, no server (repeat --capture FILE, one per capture, if classes are split):
CAPTURE=$(ls -1t "$EVID"/d41-capture-*.json | head -1)
python3 scripts/d41_floors.py --capture "$CAPTURE" \
  --out-json "$EVID/d41-floors.json" --out-md "$EVID/d41-floors.md" \
  --out-bars "$EVID/d41-bars.json" --source-revision "$T"
```

The plan carries the three required classes (`repeat`, `idle_distractor`, `busy_distractor`); the floor
run refuses without them and exits 3 on `requires_review`, in which case the bars file is written but
must be reviewed and re-recorded before it gates. `d41-bars.json` is a `sparkflash-d41-bars/1` file
whose plain `bars` object `compare_native_vs_vllm.py --bars` consumes directly - and the consumer now
enforces that shape: `demo-scale/bars.json`, hand-copied provisional numbers and a `requires_review`
floor file are refused with exit 2, not accepted as gates. Until a `status: "ok"` D41 bars file exists
the comparison is deliberately UNBARED; a `requires_review` floor must be investigated and re-recorded
with status `ok` before it can gate.

### Unit tests

```sh
python3 -m unittest discover -s scripts/tests -v
```

The host-only suite (no server, no GPU) covers log-softmax stability, top-k/overlap, KL/JS, dump and
oracle JSON parsing (including the dump/sidecar self-consistency checks), position/prefix mapping - a
missing `reference_id` is UNKNOWN, not a match - bar evaluation, the D41 gate-file shape,
`--expect-vocab`, pinned positions, the NOT_LIKE_FOR_LIKE verdict, and an end-to-end synthetic comparison.

## Host-side staging: the window is now copy + run (2026-10-19)

Since the dev host has the pinned CUDA image and `nvcc` compiles `sm_121a` with **no GPU present**,
every compiled artifact a window needs is produced here and checked here. The window itself should
contain no compilation: copy, stop, run, restart.

The pinned-image check - the same compiled-form check the `cuda-abi` CI job runs - is now the standing
check for every batch that touches `kernels/` or the baseline:

```sh
docker run --rm -v <worktree-under-/home>:/w -w /w nvidia/cuda:13.0.3-devel-ubuntu22.04 \
  sh -c 'SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc sh crates/sparkflash-kernels/scripts/check-kernel-baseline.sh'
```

* mount the worktree under `/home`: the docker daemon here cannot see an agent's private `/tmp`, and a
  `-v /tmp/...` mount silently produces an empty root-owned directory (seen 2026-10-19);
* image `nvidia/cuda:13.0.3-devel-ubuntu22.04` (id `3869b846a8cc`), no `--gpus` needed;
* success line: `all kernels match the recorded baseline (V13.0.88, cuda_13.0.r13.0/compiler.36424714_0)`;
* the Rust test that wraps this script skips without `nvcc`, so on a plain host it is not evidence.

What that buys, measured: F2's pin row was produced this way (the sequence kernel moved 38080 -> 38232 B
with a new digest and `ptx_membar_lines` still 0), and the three Engram kernels were pinned the same way;
`check-kernel-baseline.sh` then verified all 18 rows on the batch tree and again on the landed `main`.

### Artifact set for a window on main 0639413

`/home/user/wt/win-main/window-artifacts-0639413.txt` records, for `main` 0639413 (tree
`d0ee03e7d65ab5e8cebd7b06bcd381a45e6f15d4`), built with `scripts/build-aarch64.sh --features=cuda`:

| artifact | value |
| :-- | :-- |
| binary | `target/aarch64-unknown-linux-gnu/release/sparkflash`, 61,758,952 B, sha256 `1e0f0a1d45b5f754fedd87697a43b743eb7747173771478acbfe53b1dfd138af` |
| cubins | `cubins-main-0639413/`, 19 files (16 kernels + 3 Engram), per-file sha256 in the manifest |
| fixture | `/home/user/phase2-evidence/sequence-130-released.json`, 2,536,194 B, sha256 `e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308` |
| baseline | 18 rows, toolkit `V13.0.88` |

**Fixture sha correction (2026-10-19T18:55Z, B70 host prep).** The earlier row named
`c9aa2cd5bb…`; the label was wrong, not the file. `c9aa2cd5…` is the host/CI fixture sha recorded for
the task-10 branch in [build/native-arithmetic-audit.md](native-arithmetic-audit.md) section 4, and
task-9's native Engram seam superseded it: the file in this tree,
`crates/sparkflash-engine/tests/fixtures/sequence-130-tiny.json`, hashes
`17030896f4cb5b35266bf97003918489e0551010835b817438a143360d678780`. The golden sequence fixture the
window passes with `--fixture` is a different file: `/home/user/phase2-evidence/sequence-130-released.json`,
2,536,194 B, sha256 `e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308` (D25; the
dev-host copy of record is `/home/user/phase2-evidence/devhost-copy-f72738b/sequence-130-released.json`,
same sha). Pre-flight hashes the node path and must get `e6df8290…`; `c9aa2cd5…` is never a hash of the
golden fixture and must not be accepted for it.

**Per-tip rule.** This manifest is valid only for the `main` sha it names; do not carry its binary or
cubin shas to another tip. If `kernels/` is unchanged the cubins can be byte-identical, but the binary
is not, and a manifest pinned to a sha that is no longer `main` is a trap for the next reader. Rebuild
the manifest at the tip the window runs. If any `kernels/*.cu` changed, the cubin shas and possibly the
baseline rows change with it, so re-run the baseline check rather than comparing to the 0639413 table.

The binary was checked to carry the audit flags (`--operand-audit`, `--host-attention-from`,
`--attention-twin-positions`, `--dump-logits`, `tp4-resident`) before being recorded, so a flag dropped
by a refactor becomes a staging failure here instead of a window surprise.

### The two commands the window runs

```sh
# 1. the audit: twin + operand audit over the 130-token fixture, 8 decode steps
<binary> device-chain --run-stage hybrid_sequence --checkpoint <ckpt> --cubin-dir cubins-main-0639413 \
  --fixture /home/user/phase2-evidence/sequence-130-released.json \
  --prompt-tokens 130 --decode-steps 8 --time-budget-s 4800 \
  --dump-logits <run>/logits-130x8.jsonl \
  --attention-twin-positions 126,127,128,129,130 --operand-audit

# 2. the Engram harness (task-9 / F4): device-vs-twin bit-exactness, exits 1 on a real difference
cargo run -q -p sparkflash-kernels --release --example engram_native_slice -- --cubins cubins-main-0639413
```

The device-chain run needs only the binary and the cubins; the harness needs a CUDA-capable build of the
kernels crate, so it runs from a node checkout. Rule: no `nvcc` on the nodes inside a window - if a window
needs a compiler, staging was wrong.

## Build-cache hygiene across worktrees

Worktrees that share one `CARGO_TARGET_DIR` can serve kernels and test binaries built from a
DIFFERENT tree. The kernels build script's clang probe and the crate fingerprints key on the package
and feature set, not on the worktree path, so a tree whose `kernels/` set differs can fail to build,
or - worse - a test run can be a cached binary from the other tree and its counts are not this tree's.

The concrete symptom seen 2026-10-19: `sparkflash-kernels`' build script stops with

```
fatal error: 'b12x_bf16a_mxfp8w_dense_sm121.cu' file not found
```

while that `.cu` is absent from this tree's `kernels/` (it is in the other branch): the probe output
came from the shared target dir, not from this tree. Same class as engine-dev's `CARGO_MANIFEST_DIR`
catch. Rules before any window run:

- build each worktree with its own target dir (`CARGO_TARGET_DIR=<worktree>/target`) or clear the
  shared dir before a gate run;
- after switching worktrees, verify the binary's `sha256` against the expected cubin pin instead of
  trusting the path;
- in a window log, a launch/support count that matches the OTHER branch's kernel set is this bug.

## Scripted edits and checkout-relative defaults

Same root cause as the two hazards above, one level up: tooling defaults are CHECKOUT-relative, not
worktree-relative. A patch script run from a worktree but writing to a relative path edited the main
checkout instead; the diff was exactly the intended hunks, so nothing was corrupted, but the worktree
run and the checkout silently diverged. Seen 2026-10-19 during the task-7 host work (caught and
reverted with `git checkout --` in the same turn). Rules:

- every scripted edit takes ABSOLUTE paths (`git -C <worktree>` for git commands, absolute file
  arguments otherwise);
- after any scripted edit, run `git -C <worktree> status --short` and `git -C <main-checkout> status
  --short` in the same step - a clean main checkout is part of the gate, not an assumption;
- a green gate on a tree you did not intend to edit proves nothing about the tree you did edit.

## rerere: a silent replay hazard

`.git/rr-cache` (this repo sets `rerere.enabled=true`) is shared through `.git` by every worktree, and
a resolution recorded from a conflict-marked file is reused SILENTLY by later replays of the same
conflict. The dangerous case is the one nothing catches: a docs-only or comment-only commit that
"replays clean" while the resolution it reused was the raw conflict markers, because no compiler runs
on it. Seen 2026-10-19: a rehearsal staged a conflicted file without resolving it, rerere recorded
that post-image, and the next replay committed `<<<<<<< HEAD` markers and reported success. Rules:

- after every replay, `grep -c '^<<<<<<<'` on the touched files - code gates catch code, nothing
  catches docs;
- if rerere auto-resolves a conflict, read the result instead of trusting it;
- if a poisoned entry is suspected, clear `.git/rr-cache` (it is only a convenience cache).

