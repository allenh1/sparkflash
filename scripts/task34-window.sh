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

# Turnkey single-node window for task-34 S5.2. engine-dev never starts windows; tp-dev runs this.
# Usage: scripts/task34-window.sh <worktree> <checkpoint> <cubin-dir> <evidence-dir> [summary.json]
set -euo pipefail
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
  echo "usage: task34-window.sh <worktree> <checkpoint> <cubin-dir> <evidence-dir> [summary.json]" >&2
  exit 2
fi
ROOT=$1
CKPT=$2
CUBINS=$3
OUT=$4
SUMMARY=$5
if [ -z "$SUMMARY" ]; then SUMMARY=$OUT/summary.json; fi
BIN=$ROOT/target/aarch64-unknown-linux-gnu/release/sparkflash
MARGIN_GIB=88
mkdir -p "$OUT"
fail() { echo "task34-window: $*" >&2; exit 1; }

# Pre-flight: one shot per window, so refuse to allocate if the box cannot hold the 77 GB set.
avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
avail_gib=$((avail_kib / 1024 / 1024))
if [ "$avail_gib" -lt "$MARGIN_GIB" ]; then
  fail "MemAvailable $avail_gib GiB < $MARGIN_GIB GiB margin: aborting before allocating the world=1 arena"
fi
if [ ! -x "$BIN" ]; then fail "binary not found: $BIN (build with scripts/build-aarch64.sh --features=cuda)"; fi
echo "pre-flight: MemAvailable=$avail_gib GiB margin=$MARGIN_GIB GiB"

echo "== 0. provenance"
( cd "$ROOT" && git rev-parse HEAD && sha256sum "$BIN" ) | tee "$OUT/provenance.txt"

echo "== 1a. plan-only dry run: per-category stored/payload bytes, no read and no driver"
"$BIN" device-chain --run-stage tp4-resident --checkpoint "$CKPT" \
  --tp-rank 0 --tp-world-size 1 --plan-only --evidence-dir "$OUT/plan-only" \
  | tee "$OUT/plan-only.txt"

# Fail early if the world=1 plan does not fit THIS host: the streaming arena still needs the whole
# resident set, so compare MemAvailable against the plan-only resident total plus an 8 GiB margin.
plan_gib=$(python3 - "$OUT/plan-only/tp4-resident-r0.json" <<'PY2'
import json, sys
ev = json.load(open(sys.argv[1]))
a = ev.get("arena") or {}
print("%d" % int(a.get("gib") or 0))
PY2
)
avail_kib2=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
avail_gib2=$((avail_kib2 / 1024 / 1024))
need_gib=$((plan_gib + 8))
if [ "$avail_gib2" -lt "$need_gib" ]; then
  fail "plan-only resident $plan_gib GiB + 8 GiB margin needs $need_gib GiB but MemAvailable is $avail_gib2 GiB: aborting before the streaming arena (this host cannot hold the world=1 set)"
fi
echo "plan-only check: resident=$plan_gib GiB MemAvailable=$avail_gib2 GiB"


echo "== 1b. streaming world=1 arena (one tensor at a time; MemAvailable floor enforced in-binary)"
"$BIN" device-chain --run-stage tp4-resident --checkpoint "$CKPT" \
  --tp-rank 0 --tp-world-size 1 --upload --evidence-dir "$OUT" \
  | tee "$OUT/tp4-resident-rank0.txt"


echo "== 2. single-stream chain: resident arena + per-kernel ledger"
SPARKFLASH_STAGE_TIMING=1 SPARKFLASH_RESIDENT_ATTENTION=1 \
  "$BIN" device-chain --run-stage hybrid_sequence --checkpoint "$CKPT" --cubin-dir "$CUBINS" \
  --prompt-tokens 8 --decode-steps 1 --timing-json "$OUT/timing.json" \
  2>&1 | tee "$OUT/chain.txt"

# Fail the run, do not just print: a resident category that moves weight bytes invalidates the number.
if ! grep -qE 'weight_bytes_per_token=0(\.0+)?([[:space:]]|$)' "$OUT/chain.txt"; then
  fail "weight_bytes_per_token is not 0: refusing to report a kernel number from a leaking run"
fi
echo "H2D check: weight_bytes_per_token=0 for all resident categories"

echo "== 3. machine-readable summary"
python3 - "$OUT" "$SUMMARY" "$BIN" <<'PY'
import json, os, re, subprocess, sys
out, summary_path, bin_path = sys.argv[1:4]
ev = json.load(open(os.path.join(out, "tp4-resident-r0.json")))
timing = json.load(open(os.path.join(out, "timing.json")))
chain = open(os.path.join(out, "chain.txt")).read()
prov = open(os.path.join(out, "provenance.txt")).read().split()
def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None
arena = ev.get("arena") or {}
groups = [{k: g.get(k) for k in ("group", "tensors", "tiled", "stored_bytes", "bytes", "gib", "allocate_ms", "upload_ms", "hash", "verified_tensors")} for g in arena.get("groups", [])]
steps = timing.get("steps") or timing.get("step_samples") or []
step_ms = None
if steps:
    total = steps[-1].get("total_ns") or steps[-1].get("step_total_ns") or 0
    if total: step_ms = total / 1e6
summary = {
    "schema": "sparkflash-task34-window/1",
    "tree": prov[0],
    "binary_sha256": subprocess.check_output(["sha256sum", bin_path]).decode().split()[0],
    "plan_label": ev.get("plan_label"),
    "mode": arena.get("mode"),
    "groups": groups,
    "mem_available_before_bytes": ev.get("mem_available_before_bytes"),
    "mem_available_after_bytes": ev.get("mem_available_after_bytes"),
    "step_ms": step_ms,
    "staging_ms": num(r"staging_ms=([0-9.]+)", chain),
    "weight_bytes_per_token": num(r"weight_bytes_per_token=([0-9.]+)", chain),
    "activation_bytes_per_token": num(r"activation_bytes_per_token=([0-9.]+)", chain),
    "kernels": timing.get("kernels", []),
    "per_step_kernels": timing.get("per_step_kernels", []),
    "tokens": {"prompt": 8, "decode": 1, "total": 9},
}
json.dump(summary, open(summary_path, "w"), indent=2, sort_keys=True)
print("summary: " + summary_path)
PY

echo "== done: quote summary.json, do not hand-transcribe"
