#!/usr/bin/env bash
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The S1 device arm's one-command D54 gate: 32 frozen prompts x 128 greedy tokens against the
# frozen CPU-twin reference.
#
#   scripts/qa-d54.sh --run <run-dir> [--run <run-dir> ...] [--reference DIR] [--manifest FILE]
#   scripts/qa-d54.sh --self-test
#
# What it does: verifies the frozen 32-prompt fixture is really frozen (manifest shape, the
# 0.0038-nat-margin prompt at position 4, 32 complete per-fixture reference directories), then runs
# scripts/numerical-gate.sh --run against that reference, then prints the position with the largest
# |delta logprob| per rank. The bars and the PASS/FAIL verdict belong to numerical-gate.sh alone:
# this script declares no tolerance and re-derives none. The worst-position line is diagnostic (it
# names the position a failing row is decided by); it never turns a gate PASS into a FAIL or back.
#
# Exit codes (fail-closed contract):
#   0  the numerical gate passed every rank of every run; the worst position is printed
#   1  the numerical gate failed (the bars are numerical-gate.sh's, not this script's)
#   2  no frozen reference / invalid fixture / incomplete run evidence / usage: NOTHING was compared
#      and the reason (with the freeze procedure) is printed. The script never invents a reference.
#
# Freeze state at writing time (2026-10-22, host without GPU/node): there is no 32 x 128 fixture in
# this tree. The only frozen reference in-tree is the 9-position speed-gate row
# docs/build/reference/speed-gate/fab52c00a0dc6f6c/ (one 9-token prompt per rank, whose tightest
# top1-top2 margin happens to be 0.0038 nats at position 4). It is not the S1 fixture, so this
# script exits 2 here until the freeze procedure in docs/build/s1-qa-harness.md has been run.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GATE="$ROOT/scripts/numerical-gate.sh"
REFERENCE="${SPARKFLASH_S1_D54_REFERENCE:-$ROOT/docs/build/reference/s1-d54}"
MANIFEST="${SPARKFLASH_S1_D54_MANIFEST:-$REFERENCE/manifest.json}"
SELF_TEST=0
RUNS=()

usage() {
  printf '%s\n' \
    "usage: scripts/qa-d54.sh --run RUN [--run RUN ...] [--reference DIR] [--manifest FILE]" \
    "       scripts/qa-d54.sh --self-test" >&2
  exit 2
}

# The fixture schema and freeze procedure, printed whenever the script fails closed. It is the same
# text as docs/build/s1-qa-harness.md's "Fixture freeze" section, kept beside the check so a reader
# who hits the refusal does not have to find the document first.
freeze_state() {
  printf '%s\n' \
    "qa-d54: no frozen reference" \
    "  expected manifest:  ${MANIFEST}" \
    "  expected reference: ${REFERENCE}/<fixture>/{rank0..3}.json" \
    "  declared shape:     32 prompts x 128 greedy tokens, 0.0038-nat margin prompt at position 4" \
    "  fixture state:      $(if [ -f "$MANIFEST" ]; then echo "manifest present"; else echo "manifest ABSENT"; fi), $(if [ -d "$REFERENCE" ]; then echo "reference dir present"; else echo "reference dir ABSENT"; fi)" \
    "  in-tree references: docs/build/reference/speed-gate/ is the 9-position speed row (one prompt);" \
    "                      it is NOT the 32 x 128 S1 fixture and must not be used as one." \
    "  freeze procedure (on a node, four ranks, host arm - the CPU twin is the reference):" \
    "    1. tokenize the 32 frozen prompts; write docs/build/reference/s1-d54/manifest.json with" \
    "       schema s1-d54-fixture-v1, 32 prompt entries, greedy_tokens 128, and" \
    "       margin={position: 4, nats: 0.0038, tolerance: 5e-5} (schema in docs/build/s1-qa-harness.md)" \
    "    2. run each prompt through the host arm with the logits dump enabled, one evidence dir per prompt" \
    "    3. for each evidence dir: scripts/numerical-gate.sh --freeze <run-dir> docs/build/reference/s1-d54" \
    "       (it keys each frozen reference by sha256(positions, prompt_len, ids), so a new prompt needs" \
    "       one more freeze, never a gate change - D54 item 2)" \
    "    4. re-run scripts/qa-d54.sh --run <run-dir> and keep the output on the record" \
    "  Exiting 2: nothing was compared."
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run)
      [ $# -ge 2 ] || usage
      RUNS+=("$2")
      shift 2
      ;;
    --reference)
      [ $# -ge 2 ] || usage
      REFERENCE=$2
      shift 2
      ;;
    --manifest)
      [ $# -ge 2 ] || usage
      MANIFEST=$2
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    -h | --help) usage ;;
    *) usage ;;
  esac
done

python_helper() {
  python3 - "$@" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

PROMPTS = 32
GREEDY_TOKENS = 128
MARGIN_POSITION = 4
MARGIN_NATS = 0.0038
MARGIN_TOLERANCE = 5.0e-5
SCHEMA = "s1-d54-fixture-v1"
# The fallback comparison support; the real value is read from numerical-gate.sh's COMPARE_K so the
# diagnostic cannot drift from the gate it accompanies (the bars stay the gate's, unmodified).
COMPARE_K = 100


def die(message):
    print(f"qa-d54: {message}", file=sys.stderr)
    sys.exit(2)


def load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        die(f"cannot read {path}: {error}")


def fixture_key(token_json):
    """The numerical gate's key: sha256(positions, prompt_len, ids), first 16 hex characters."""
    key = {
        "positions": token_json.get("tokens", {}).get("positions"),
        "prompt_len": token_json.get("tokens", {}).get("prompt_len"),
        "ids": token_json.get("tokens", {}).get("ids"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def complete_fixtures(reference_dir):
    root = pathlib.Path(reference_dir)
    if not root.is_dir():
        return []
    complete = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and all(
            (path / f"rank{rank}.json").is_file() for rank in range(4)
        ):
            complete.append(path)
    return complete


def validate_fixture(manifest_path, reference_dir):
    """The freeze check: refuse (exit 2) unless the declared 32 x 128 fixture is really frozen."""
    if not os.path.isfile(manifest_path):
        die(f"the fixture manifest is absent ({manifest_path}); no frozen reference")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        die(f"fixture manifest schema is {manifest.get('schema')!r}, expected {SCHEMA!r}")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != PROMPTS:
        die(
            "fixture manifest declares "
            f"{len(prompts) if isinstance(prompts, list) else 'no'} prompt(s); the S1 D54 fixture "
            f"is {PROMPTS}"
        )
    greedy = manifest.get("greedy_tokens")
    if greedy != GREEDY_TOKENS:
        die(f"fixture manifest declares {greedy} greedy tokens; the S1 D54 fixture is {GREEDY_TOKENS}")
    margin = manifest.get("margin") or {}
    if margin.get("position") != MARGIN_POSITION:
        die(
            f"fixture manifest margin position is {margin.get('position')!r}; the declared margin "
            f"prompt is position {MARGIN_POSITION}"
        )
    nats = float(margin.get("nats", 0.0))
    tolerance = float(margin.get("tolerance", MARGIN_TOLERANCE))
    if abs(nats - MARGIN_NATS) > tolerance:
        die(
            f"fixture manifest margin is {nats} nats, not the declared {MARGIN_NATS} (tolerance "
            f"{tolerance})"
        )
    if not os.path.isdir(reference_dir):
        die(f"the frozen reference directory is absent ({reference_dir}); no frozen reference")

    fixtures = complete_fixtures(reference_dir)
    if len(fixtures) != PROMPTS:
        die(
            f"the frozen reference holds {len(fixtures)} complete fixture(s) (rank0..3); the S1 D54 "
            f"fixture is {PROMPTS} prompts x {GREEDY_TOKENS} tokens"
        )
    # The 0.0038-nat margin prompt must actually be in the reference: the gate prints the reference's
    # tightest margin, so a reference frozen without it would silently move the decisive position.
    found = None
    for path in fixtures:
        summary = load_json(path / "rank0.json")
        positions = summary.get("positions") or []
        if not positions:
            continue
        index = min(range(len(positions)), key=lambda at: positions[at].get("margin", 0.0))
        value = float(positions[index].get("margin", 0.0))
        if index == MARGIN_POSITION and abs(value - nats) <= tolerance:
            found = (path.name, value)
            break
    if found is None:
        die(
            "the frozen reference has no fixture whose tightest top1-top2 margin is "
            f"{nats} nats at position {MARGIN_POSITION}: the 0.0038-nat margin prompt was not "
            "frozen (D54 item 2)"
        )
    print(
        f"qa-d54: fixture OK: {PROMPTS} prompts x {GREEDY_TOKENS} greedy tokens, margin prompt "
        f"{found[1]:.6f} nats at position {MARGIN_POSITION} (fixture {found[0]}), "
        f"{len(fixtures)} frozen fixtures x 4 ranks"
    )


def read_logits(path, rows, vocab):
    with open(path, "rb") as handle:
        data = handle.read()
    need = rows * vocab * 4
    if len(data) < need:
        die(f"logits dump {path} is {len(data)} bytes, needs {need}")
    out = []
    for row in range(rows):
        out.append(struct.unpack_from(f"<{vocab}f", data, row * vocab * 4))
    return out


def log_softmax_top(values, k):
    best = max(values)
    exps = [math.exp(value - best) for value in values]
    log_total = math.log(sum(exps)) + best
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:k]
    return order, [values[index] - log_total for index in order]


def gate_compare_k(gate_path):
    """Reads numerical-gate.sh's COMPARE_K, so the diagnostic support follows the gate's own constant.

    The gate stays the only place the bars and the comparison support are declared; this script only
    reports a per-position maximum over the same support. A gate whose constant cannot be read falls
    back to 100 and says so on stderr, because a silent default is how the two would drift apart.
    """
    try:
        with open(gate_path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError as error:
        print(
            f"qa-d54: cannot read {gate_path} for COMPARE_K ({error}); using {COMPARE_K}",
            file=sys.stderr,
        )
        return COMPARE_K
    match = re.search(r"^COMPARE_K\s*=\s*(\d+)", source, re.MULTILINE)
    if match is None:
        print(
            f"qa-d54: {gate_path} declares no COMPARE_K; using {COMPARE_K}",
            file=sys.stderr,
        )
        return COMPARE_K
    return int(match.group(1))


def run_positions(run, rank, compare_k=COMPARE_K):
    base = os.path.join(run, f"rank{rank}")
    token_path = os.path.join(base, f"tp4-token-r{rank}.json")
    token = load_json(token_path)
    dump = token.get("logits_dump", {})
    path, rows, vocab = dump.get("bin"), dump.get("rows"), dump.get("vocab")
    if not path or not os.path.exists(path):
        # The evidence copy beside the token file wins, as in numerical-gate.sh.
        beside = os.path.join(base, f"tp4-logits-r{rank}.jsonl.bin")
        path = beside if os.path.exists(beside) else path
    if not path or not os.path.exists(path):
        die(f"{run} rank{rank}: logits dump missing ({path}); cannot print the worst position")
    if not isinstance(rows, int) or not isinstance(vocab, int):
        die(f"{run} rank{rank}: logits dump metadata missing rows/vocab")
    return token, [
        log_softmax_top(values, compare_k) for values in read_logits(path, rows, vocab)
    ]


def worst_positions(reference_dir, runs, compare_k=COMPARE_K):
    """Per rank, the position with the largest |delta logprob| over the reference top-COMPARE_K.

    The transform and the support are numerical-gate.sh's (its COMPARE_K of the reference, shared
    ids); only the per-position maximum is this script's, because the gate reports one aggregate
    line. Returned for the self-test, printed by the caller.
    """
    report = []
    overall = None
    for run in runs:
        for rank in range(4):
            token, positions = run_positions(run, rank, compare_k)
            key = fixture_key(token)
            reference_path = os.path.join(reference_dir, key, f"rank{rank}.json")
            if not os.path.isfile(reference_path):
                die(
                    f"{run} rank{rank}: no frozen reference for fixture {key} "
                    f"({reference_path}); no frozen reference"
                )
            reference = load_json(reference_path)
            if len(reference.get("positions") or []) != len(positions):
                die(
                    f"{run} rank{rank}: reference has "
                    f"{len(reference.get('positions') or [])} position(s), run dumped "
                    f"{len(positions)}"
                )
            worst = None
            for index, (ref_position, got) in enumerate(
                zip(reference["positions"], positions)
            ):
                got_ids, got_logprobs = got
                got_at = dict(zip(got_ids, got_logprobs))
                deltas = [
                    abs(ref_logprob - got_at[token_id])
                    for token_id, ref_logprob in zip(
                        ref_position["ids"][:compare_k],
                        ref_position["logprobs"][:compare_k],
                    )
                    if token_id in got_at
                ]
                if not deltas:
                    die(
                        f"{run} rank{rank}: position {index} shares no top-{compare_k} ids with "
                        "the reference"
                    )
                delta = max(deltas)
                if worst is None or delta > worst["max_abs"]:
                    worst = {
                        "run": os.path.basename(run.rstrip("/")),
                        "rank": rank,
                        "position": index,
                        "max_abs": delta,
                        "margin": float(ref_position.get("margin", 0.0)),
                    }
            report.append(worst)
            if overall is None or worst["max_abs"] > overall["max_abs"]:
                overall = worst
    for item in report:
        print(
            f"worst position: {item['run']} rank{item['rank']} position {item['position']} "
            f"max|dlogprob| {item['max_abs']:.3e} (ref top1-top2 margin {item['margin']:.4f} nats)"
        )
    print(
        f"WORST: {overall['run']} rank{overall['rank']} position {overall['position']} "
        f"max|dlogprob| {overall['max_abs']:.3e} (ref margin {overall['margin']:.4f} nats)"
    )
    return report, overall


def synthetic_fixture(root, prompts=PROMPTS, greedy=GREEDY_TOKENS, margin_position=MARGIN_POSITION,
                      margin_nats=MARGIN_NATS):
    manifest = {
        "schema": SCHEMA,
        "prompts": [{"id": f"p{index:02d}", "text": f"prompt {index}"} for index in range(prompts)],
        "greedy_tokens": greedy,
        "margin": {
            "position": margin_position,
            "nats": margin_nats,
            "tolerance": MARGIN_TOLERANCE,
        },
        "reference": "docs/build/reference/s1-d54",
    }
    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle)
    reference_dir = os.path.join(root, "reference")
    for index in range(prompts):
        fixture = os.path.join(reference_dir, f"fixture{index:02d}")
        os.makedirs(fixture)
        positions = []
        for position in range(greedy):
            margin = 0.05
            if position == margin_position:
                margin = margin_nats
            positions.append(
                {
                    "ids": [position % 7, (position + 1) % 7, (position + 2) % 7],
                    "logprobs": [0.0, -margin, -margin - 0.01],
                    "margin": margin,
                }
            )
        summary = {"fixture": f"fixture{index:02d}", "tokens": [1], "positions_run": greedy,
                   "rows": greedy, "vocab": 8, "positions": positions}
        for rank in range(4):
            with open(os.path.join(fixture, f"rank{rank}.json"), "w") as handle:
                json.dump(summary, handle)
    return manifest_path, reference_dir


def expect_refusal(root, label, **overrides):
    work = tempfile.mkdtemp(prefix="qa-d54-invalid-")
    try:
        manifest_path, reference_dir = synthetic_fixture(
            work,
            prompts=overrides.get("prompts", PROMPTS),
            greedy=overrides.get("greedy", GREEDY_TOKENS),
            margin_position=overrides.get("margin_position", MARGIN_POSITION),
            margin_nats=overrides.get("margin_nats", MARGIN_NATS),
        )
        if overrides.get("drop_fixture"):
            shutil.rmtree(os.path.join(reference_dir, "fixture00"))
        try:
            validate_fixture(manifest_path, reference_dir)
        except SystemExit as attack:
            if attack.code == 2:
                print(f"  self-test: refused {label} (exit 2)")
                return
            raise
        die(f"self-test: the freeze check accepted {label}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def self_test(root, gate):
    # 1. The declared fixture and its freeze shape are accepted.
    work = tempfile.mkdtemp(prefix="qa-d54-selftest-")
    try:
        manifest_path, reference_dir = synthetic_fixture(work)
        validate_fixture(manifest_path, reference_dir)
        for label, overrides in [
            ("31 prompts", {"prompts": 31}),
            ("127 greedy tokens", {"greedy": 127}),
            ("margin at position 3", {"margin_position": 3}),
            ("margin 0.05 nats", {"margin_nats": 0.05}),
            ("31 frozen fixtures", {"drop_fixture": True}),
        ]:
            expect_refusal(root, label, **overrides)

        # 2. The worst-position finder picks the perturbed position, not the fixture's margin position.
        run = os.path.join(work, "run")
        vocab, rows = 16, 6
        base = [[float((token * 3 + row) % 11) for token in range(vocab)] for row in range(rows)]
        run_dir = os.path.join(run, "rank0")
        os.makedirs(run_dir)
        token = {
            "tokens": {"ids": [1, 2, 3], "positions": [0, 1, 2], "prompt_len": 3},
            "logits_dump": {
                "bin": os.path.join(run_dir, "logits.bin"),
                "rows": rows,
                "vocab": vocab,
            },
        }
        # The frozen reference is the CPU twin's log-softmax of the unperturbed rows.
        reference_positions = []
        for values in base:
            order, logprobs = log_softmax_top(values, vocab)
            reference_positions.append(
                {"ids": order, "logprobs": logprobs, "margin": logprobs[0] - logprobs[1]}
            )
        frozen = os.path.join(work, "reference2", fixture_key(token))
        os.makedirs(frozen)
        for rank in range(4):
            with open(os.path.join(frozen, f"rank{rank}.json"), "w") as handle:
                json.dump(
                    {"fixture": fixture_key(token), "tokens": [1, 2, 3], "rows": rows,
                     "vocab": vocab, "positions": reference_positions},
                    handle,
                )
        perturbed = [row[:] for row in base]
        perturbed[3][0] += 0.5
        for rank in range(4):
            rank_dir = os.path.join(run, f"rank{rank}")
            os.makedirs(rank_dir, exist_ok=True)
            logits_path = os.path.join(rank_dir, "logits.bin")
            with open(logits_path, "wb") as handle:
                for values in (perturbed if rank == 0 else base):
                    handle.write(struct.pack(f"<{vocab}f", *values))
            rank_token = json.loads(json.dumps(token))
            rank_token["logits_dump"]["bin"] = logits_path
            with open(os.path.join(rank_dir, f"tp4-token-r{rank}.json"), "w") as handle:
                json.dump(rank_token, handle)
        report, overall = worst_positions(os.path.join(work, "reference2"), [run])
        if overall["position"] != 3:
            die(f"self-test: worst position expected 3, got {overall['position']}")
        if abs(overall["max_abs"]) <= 1.0e-6:
            die(f"self-test: worst |dlogprob| expected non-zero, got {overall['max_abs']}")
        print("  self-test: worst position 3 found in 4 ranks")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # 3. The fail-closed path: no manifest means exit 2 with "no frozen reference" and no comparison.
    probe = tempfile.mkdtemp(prefix="qa-d54-missing-")
    try:
        environment = dict(os.environ)
        environment["SPARKFLASH_S1_D54_MANIFEST"] = os.path.join(probe, "absent-manifest.json")
        environment["SPARKFLASH_S1_D54_REFERENCE"] = os.path.join(probe, "absent-reference")
        completed = subprocess.run(
            ["bash", os.path.join(root, "scripts", "qa-d54.sh"), "--run",
             os.path.join(probe, "absent-run")],
            capture_output=True, text=True, env=environment, check=False,
        )
        combined = completed.stdout + completed.stderr
        if completed.returncode != 2:
            die(f"self-test: no-freeze path exited {completed.returncode}, expected 2\n{combined}")
        if "no frozen reference" not in combined:
            die(f"self-test: no-freeze path did not say 'no frozen reference'\n{combined}")
        print("  self-test: absent fixture exits 2 with 'no frozen reference'")
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    print("QA-D54 SELF-TEST PASS")


def main():
    mode = sys.argv[1]
    if mode == "validate":
        validate_fixture(sys.argv[2], sys.argv[3])
        return 0
    if mode == "worst":
        reference_dir, gate_path = sys.argv[2], sys.argv[3]
        compare_k = gate_compare_k(gate_path)
        worst_positions(reference_dir, sys.argv[4:], compare_k)
        return 0
    if mode == "self-test":
        self_test(sys.argv[2], sys.argv[3])
        return 0
    die(f"unknown helper mode {mode!r}")


sys.exit(main())
PY
}

if [ "$SELF_TEST" = "1" ]; then
  python_helper self-test "$ROOT" "$GATE"
  exit 0
fi

if [ "${#RUNS[@]}" -eq 0 ]; then
  usage
fi

# The freeze check comes first: a missing fixture must be reported as exit 2 ("no frozen reference"),
# never as a gate failure, and the run evidence is secondary to that.
if [ ! -f "$MANIFEST" ] || [ ! -d "$REFERENCE" ]; then
  freeze_state
fi
for run in "${RUNS[@]}"; do
  if [ ! -d "$run" ]; then
    freeze_state
  fi
  for rank in 0 1 2 3; do
    if [ ! -f "$run/rank${rank}/tp4-token-r${rank}.json" ]; then
      printf 'qa-d54: run evidence incomplete: %s/rank%s/tp4-token-r%s.json is missing\n' \
        "$run" "$rank" "$rank" >&2
      printf 'qa-d54: the four-rank run directory is the layout run-fourrank.sh writes; refusing (exit 2)\n' >&2
      exit 2
    fi
  done
done

python_helper validate "$MANIFEST" "$REFERENCE"

# The gate decides PASS/FAIL; this script only forwards that verdict and adds the worst position.
set +e
GATE_OUTPUT=$(SPARKFLASH_GATE_REFERENCE="$REFERENCE" bash "$GATE" --run "${RUNS[@]}" 2>&1)
GATE_STATUS=$?
set -e
printf '%s\n' "$GATE_OUTPUT"

if [ "$GATE_STATUS" -ne 0 ] && printf '%s\n' "$GATE_OUTPUT" | grep -q "no frozen reference"; then
  freeze_state
fi
if [ "$GATE_STATUS" -ne 0 ] && [ "$GATE_STATUS" -ne 1 ]; then
  printf 'qa-d54: numerical-gate.sh exited %s; treating it as a harness failure (exit 2)\n' \
    "$GATE_STATUS" >&2
  exit 2
fi

python_helper worst "$REFERENCE" "$GATE" "${RUNS[@]}"

exit "$GATE_STATUS"
