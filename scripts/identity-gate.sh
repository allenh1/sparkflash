#!/usr/bin/env bash
# One command that proves byte identity for a four-rank lean window run.
#
#   scripts/identity-gate.sh --run <window run dir>   # rank0..rank3 evidence, exits 0 on PASS
#   scripts/identity-gate.sh --anchor                 # CI: the anchor constants are well-formed
#   scripts/identity-gate.sh --self-test              # CI: the checker accepts a good row and
#                                                     #     rejects a corrupted one, no device
#
# --run checks, in one pass and with one summary line:
#   * each rank's chain-rN.rc says 0 (the row really finished);
#   * tokens.ids is exactly the frozen 9-token sequence on all four ranks;
#   * logits_dump.bin_sha256 is exactly the frozen sha on all four ranks;
#   * the embed artifact states evidence_mode, and evidence_digest_calls is 0 with an empty
#     layer_chain when the mode is off (a chain-free run can never read as chain-present).
#
# This is the automation the speed-track rules require: no hand re-run and no document for a result
# this command already proves.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: identity-gate.sh --run DIR | --anchor | --self-test" >&2
  exit 2
fi
exec python3 - "$@" <<'PY'
import json
import os
import subprocess
import sys
import tempfile

TOKENS = [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86]
SHA = "aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2"
MODE_KEYS = ("evidence_mode", "evidence_digest_calls")


def load(path):
    with open(path) as handle:
        return json.load(handle)


def check_run(run, quiet=False):
    problems = []
    if not os.path.isdir(run):
        return [f"run directory does not exist: {run}"]
    token_sets, sha_sets = set(), set()
    for rank in range(4):
        base = os.path.join(run, f"rank{rank}")
        token_path = os.path.join(base, f"tp4-token-r{rank}.json")
        rc_path = os.path.join(base, f"chain-r{rank}.rc")
        embed_path = os.path.join(base, f"tp4-forward-embed-r{rank}.json")
        for path in (token_path, rc_path):
            if not os.path.exists(path):
                problems.append(f"rank{rank}: missing {os.path.basename(path)}")
        if problems:
            continue
        rc = open(rc_path).read().strip()
        if rc != "0":
            problems.append(f"rank{rank}: exit code {rc!r}")
        data = load(token_path)
        tokens = data.get("tokens", {}).get("ids")
        token_sets.add(json.dumps(tokens))
        if tokens != TOKENS:
            problems.append(f"rank{rank}: tokens differ from the frozen sequence: {tokens}")
        sha = data.get("logits_dump", {}).get("bin_sha256")
        sha_sets.add(sha)
        if sha != SHA:
            problems.append(f"rank{rank}: logits bin sha {str(sha)[:16]} != {SHA[:16]}")
        if not os.path.exists(embed_path):
            problems.append(f"rank{rank}: missing tp4-forward-embed-r{rank}.json")
            continue
        embed = load(embed_path)
        missing = [key for key in MODE_KEYS if key not in embed]
        if missing:
            problems.append(f"rank{rank}: embed artifact lacks {missing}")
            continue
        mode, calls = embed["evidence_mode"], embed["evidence_digest_calls"]
        chain = embed.get("layer_chain") or []
        if mode not in ("recording", "off"):
            problems.append(f"rank{rank}: unknown evidence_mode {mode!r}")
        if mode == "off" and (calls != 0 or chain):
            problems.append(f"rank{rank}: evidence_mode off but digest_calls={calls}, chain={len(chain)}")
        if mode == "recording" and not chain:
            problems.append(f"rank{rank}: evidence_mode recording but layer_chain is empty")
    if len(token_sets) != 1:
        problems.append(f"ranks disagree on tokens: {sorted(token_sets)}")
    if len(sha_sets) != 1:
        problems.append(f"ranks disagree on logits sha: {sorted(str(s)[:16] for s in sha_sets)}")
    if not quiet and not problems:
        print(f"identity-gate PASS: {run} - 4/4 ranks rc=0, tokens exact, logits bin sha {SHA[:16]}..., mode fields consistent")
    return problems


def _write_fixture(root, *, bad_rank=None, bad_kind=None):
    for rank in range(4):
        base = os.path.join(root, f"rank{rank}")
        os.makedirs(base)
        open(os.path.join(base, f"chain-r{rank}.rc"), "w").write("0\n")
        tokens = list(TOKENS)
        if rank == bad_rank and bad_kind == "tokens":
            tokens[-1] = 1
        json.dump({"tokens": {"ids": tokens}, "logits_dump": {"bin_sha256": SHA}},
                  open(os.path.join(base, f"tp4-token-r{rank}.json"), "w"))
        calls = 7 if (rank == bad_rank and bad_kind == "digests") else 0
        json.dump({"evidence_mode": "off", "evidence_digest_calls": calls, "layer_chain": []},
                  open(os.path.join(base, f"tp4-forward-embed-r{rank}.json"), "w"))


def self_test():
    root = tempfile.mkdtemp(prefix="identity-gate-")
    good = os.path.join(root, "good")
    _write_fixture(good)
    problems = check_run(good, quiet=True)
    assert not problems, f"self-test: good row rejected: {problems}"
    bad_tokens = os.path.join(root, "bad-tokens")
    _write_fixture(bad_tokens, bad_rank=0, bad_kind="tokens")
    problems = check_run(bad_tokens, quiet=True)
    assert any("tokens differ" in p for p in problems), f"self-test: corrupted tokens accepted: {problems}"
    bad_digests = os.path.join(root, "bad-digests")
    _write_fixture(bad_digests, bad_rank=1, bad_kind="digests")
    problems = check_run(bad_digests, quiet=True)
    assert any("digest_calls" in p for p in problems), f"self-test: chain-free row with digests accepted: {problems}"
    print("identity-gate self-test PASS: a good four-rank row passes, corrupted tokens and a chain-free row with digest calls are rejected")


def anchor():
    assert len(TOKENS) == 9 and all(isinstance(t, int) for t in TOKENS), "anchor tokens malformed"
    assert len(SHA) == 64 and all(c in "0123456789abcdef" for c in SHA), "anchor sha malformed"
    print(f"identity-gate anchor OK: {len(TOKENS)} tokens, sha {SHA[:16]}...")


args = sys.argv[1:]
if args[:1] == ["--run"] and len(args) >= 2:
    problems = check_run(args[1])
    for problem in problems:
        print("identity-gate FAIL:", problem, file=sys.stderr)
    sys.exit(1 if problems else 0)
if args[:1] == ["--anchor"]:
    anchor()
    sys.exit(0)
if args[:1] == ["--self-test"]:
    self_test()
    sys.exit(0)
sys.exit("usage: identity-gate.sh --run DIR | --anchor | --self-test")
PY
