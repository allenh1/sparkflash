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

The device side already writes the comparison input with `--dump-logits` (O6, branch
`task/logits-dump`): one JSON object per compared position (position, device argmax, vocab,
logits_offset, top20, ...) plus `<dump>.bin` with the full-vocabulary little-endian f32 logits in the
same position order. The dump is written before the token-parity gate can fail, so a failing run still
leaves its logits. Stage the run and the dump before the stop; the comparison is offline afterwards.

### 2. Oracle (after the released recipe answers /v1/models)

```sh
python3 scripts/oracle_native.py \
  --suite scripts/oracle_prompt_suite.json \
  --fixture /home/user/phase2-evidence/sequence-130-released.json \
  --out /home/user/phase2-evidence/window-2026-10-18/native-oracle-130.json \
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

### 3. Comparison (offline, on this host or anywhere)

```sh
python3 scripts/compare_native_vs_vllm.py \
  --dump /home/user/phase2-evidence/window-2026-10-18/runs69/logits-130x0.jsonl \
  --oracle /home/user/phase2-evidence/window-2026-10-18/native-oracle-130.json \
  --oracle-prompt b69-fixture-130 \
  --out-json /home/user/phase2-evidence/window-2026-10-18/compare-native-vs-vllm.json \
  --out-md /home/user/phase2-evidence/window-2026-10-18/compare-native-vs-vllm.md
```

It joins on position (default: the intersection of dump and oracle positions) and reports per position
and in aggregate: top-1 agreement, top-5/20/100/1000 overlap, max/mean |logprob delta| over the oracle's
ids, KL(oracle || device) and JS of the renormalised top-k, the oracle top-k retained mass, and a
`prefix_match` flag (device reference token vs oracle continuation token - a mismatch means the two
distributions are conditioned on different prefixes and the row is not like-for-like). It never compares
token identity (D39 rule 2). Add `--bars <D40 file>` to gate; without it the verdict is `UNBARED`
(measured evidence, no verdict). Exit codes: 0 pass/unbared, 1 a declared bar failed, 2 bad input.

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
aggregate object; a bar file is `{"bars": {"top1_agreement_min": 1.0, "max_abs_delta_max": 0.5, ...}}`.

### Unit tests

```sh
python3 -m unittest discover -s scripts/tests -v
```

47 host-only tests (no server, no GPU): log-softmax stability, top-k/overlap, KL/JS, dump and oracle
JSON parsing, position/prefix mapping, bar evaluation, and an end-to-end synthetic comparison.

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

