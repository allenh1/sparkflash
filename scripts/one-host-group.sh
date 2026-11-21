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

# A serving group's ranks as processes on one host (route b of docs/engine/04-onboarding-review.md,
# "A group on one host"): boot every rank of `<arm>` at `[world]` ranks on loopback, wait for the
# group, send the served checks (scripts/served-gate.py), and stop every rank by its PID.
#
#   scripts/one-host-group.sh <arm> [world] --model ID --checkpoint DIR [--converted DIR]
#                             [--long FILE] [--port PORT] [--apply]
#
# <arm> names the group the ranks form, which is the serve flag set they take:
#   dense, glm, qwen   --tp4-rank: a tensor-parallel group, partials summed on the host seam
#                      (SPARK_COMM_REDUCE=tcp), since one GPU has no NCCL peer of its own
#   inkling            --rank: the device arm's group, over --converted rank files
#   dory               --recurrence-rank: the latent-recurrence ring (NCCL, --recurrence-net any)
# world is 2 or 4 (default 2). Dory's ring takes world == its loop count and needs an NCCL
# library the serve binary can load; NCCL also refuses two ranks on one GPU, so a one-GPU host
# cannot form the ring and Dory's group is owed to scripts/owed-cluster-window.sh.
#
# The default is a dry run: the script prints the commands and touches nothing. --apply runs them.
# It refuses to start when the host's available memory is below the ranks' weights plus a
# per-rank allowance, rather than let a rank be killed part way through loading.
set -euo pipefail

usage() {
  sed -n '/^#   scripts\/one-host-group.sh/,/^#                             \[--long/p' "$0" | sed 's/^#   //'
  exit 2
}

ARM="${1:-}"
[ -n "$ARM" ] || usage
shift
WORLD=2
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  WORLD="$1"
  shift
fi
MODEL="" CHECKPOINT="" CONVERTED="" LONG="" PORT=8126 APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --converted) CONVERTED="$2"; shift 2 ;;
    --long) LONG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --dry-run) APPLY=0; shift ;;
    *) echo "one-host-group: unknown argument $1" >&2; usage ;;
  esac
done
[ "$WORLD" -ge 2 ] || { echo "one-host-group: a group has two ranks or more, not $WORLD" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "one-host-group: --model ID is required (the id the ranks serve)" >&2; exit 2; }
[ -n "$CHECKPOINT" ] && [ -d "$CHECKPOINT" ] || { echo "one-host-group: --checkpoint DIR must be a directory" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SPARKFLASH_BIN:-$ROOT/target/release/sparkflash}"
LOGS="${ONE_HOST_LOGS:-${TMPDIR:-/tmp}/one-host-group-$ARM-$WORLD}"
PEERS="$(printf '127.0.0.1%.0s,' $(seq 1 "$WORLD"))"
PEERS="${PEERS%,}"

# Each rank's group flags, and the weights one rank holds as a share of the checkpoint's.
ENVIRONMENT=()
case "$ARM" in
  dense|glm|qwen)
    ENVIRONMENT=(SPARK_COMM_REDUCE=tcp)
    group() { echo "--tp4-rank $1 --tp4-world-size $WORLD --tp4-peers $PEERS --tp4-base-port 29610 --tp4-control-port 29620"; }
    SHARE="$WORLD" ;;
  inkling)
    [ -n "$CONVERTED" ] || { echo "one-host-group: inkling's group reads --converted DIR (its rank files)" >&2; exit 2; }
    ENVIRONMENT=(SPARK_COMM_REDUCE=tcp)
    group() { echo "--rank $1 --world $WORLD --peer $PEERS --control-port 29620 --converted $CONVERTED"; }
    SHARE="$WORLD" ;;
  dory)
    group() { echo "--recurrence-rank $1 --recurrence-peers $PEERS --recurrence-port 29801 --recurrence-net any"; }
    # Every rank of the ring holds the loop's weights; the loops, not the weights, are divided.
    SHARE=1 ;;
  *) echo "one-host-group: no group for arm $ARM (dense, glm, qwen, inkling, dory)" >&2; exit 2 ;;
esac

# The memory refusal: the ranks' weights, plus 6 GiB a rank for its caches, context and runtime.
WEIGHTS_DIR="${CONVERTED:-$CHECKPOINT}"
weight_bytes=$( (find -L "$WEIGHTS_DIR" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -exec du -kL {} + 2>/dev/null || true) | awk '{s += $1} END {print s * 1024}')
need=$(( weight_bytes / SHARE * WORLD + WORLD * 6 * 1024 * 1024 * 1024 ))
available=$( (awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo 2>/dev/null || true) | awk '{print $1 + 0} END {if (NR == 0) print 0}')
echo "one-host-group: $ARM at $WORLD ranks needs $((need >> 30)) GiB; $((available >> 30)) GiB available"

serve_command() {
  local rank=$1
  echo "${ENVIRONMENT[*]:-} $BIN serve --model $MODEL --checkpoint $CHECKPOINT --bind 127.0.0.1:$((PORT + rank * 10)) $(group "$rank")"
}
GATE=(python3 "$ROOT/scripts/served-gate.py")
if [ -n "$LONG" ]; then
  CHECKS=("all $PORT --long $LONG")
else
  CHECKS=("lanes $PORT" "lanes $PORT --sampled" "turns $PORT")
fi

if [ "$APPLY" -eq 0 ]; then
  echo "# dry run: pass --apply to execute"
  for rank in $(seq $((WORLD - 1)) -1 0); do
    echo "$(serve_command "$rank") > $LOGS/rank$rank.log 2>&1 &"
  done
  for check in "${CHECKS[@]}"; do echo "${GATE[*]} $check"; done
  echo "# then every rank is stopped by its PID"
  exit 0
fi

[ -x "$BIN" ] || { echo "one-host-group: no binary at $BIN; build sparkflash-cli or set SPARKFLASH_BIN" >&2; exit 2; }
if [ "$available" -lt "$need" ]; then
  echo "one-host-group: refusing: $((need >> 30)) GiB needed, $((available >> 30)) GiB available" >&2
  exit 3
fi
mkdir -p "$LOGS"
PIDS=()
stop() {
  # Rank 0 first, so the group ends as a served group does; then whatever rank is left.
  for index in $(seq $((${#PIDS[@]} - 1)) -1 0); do
    kill "${PIDS[$index]}" 2>/dev/null || true
    sleep 2
  done
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && { echo "one-host-group: pid $pid did not stop; killing it"; kill -9 "$pid" 2>/dev/null || true; }
  done
  return 0
}
trap stop EXIT
for rank in $(seq $((WORLD - 1)) -1 0); do
  # shellcheck disable=SC2046
  env ${ENVIRONMENT[@]+"${ENVIRONMENT[@]}"} "$BIN" serve --model "$MODEL" --checkpoint "$CHECKPOINT" \
    --bind "127.0.0.1:$((PORT + rank * 10))" $(group "$rank") > "$LOGS/rank$rank.log" 2>&1 < /dev/null &
  PIDS+=($!)
done
# PIDS holds rank WORLD-1 first and rank 0 last.
ready=0
for _ in $(seq 1 450); do
  if grep -qiE "listening|serving on" "$LOGS/rank0.log" 2>/dev/null; then ready=1; break; fi
  for index in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$index]}" 2>/dev/null; then
      rank=$((WORLD - 1 - index))
      echo "one-host-group: rank $rank exited before the group was ready" >&2
      tail -20 "$LOGS/rank$rank.log" >&2
      exit 1
    fi
  done
  sleep 2
done
[ "$ready" -eq 1 ] || { echo "one-host-group: rank 0 never listened; logs in $LOGS" >&2; exit 1; }
status=0
for check in "${CHECKS[@]}"; do
  # shellcheck disable=SC2086
  "${GATE[@]}" $check > "$LOGS/check.log" 2>&1 || status=1
  cat "$LOGS/check.log" >> "$LOGS/gate.log"
  grep -E "VERDICT|FAIL" "$LOGS/check.log" || true
done
echo "one-host-group: $ARM at $WORLD ranks: served checks $([ $status -eq 0 ] && echo PASS || echo FAIL); logs in $LOGS"
exit $status
