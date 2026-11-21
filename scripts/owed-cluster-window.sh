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

# Everything that still needs four real nodes once scripts/one-host-group.sh has run what one
# host can (docs/engine/04-onboarding-review.md, "The 2026-11-21 sprint: a group on one host"),
# as one ordered list, so a single granted cluster window pays all of it.
#
#   scripts/owed-cluster-window.sh            print every step's commands (the default; no network)
#   scripts/owed-cluster-window.sh --apply    run them in order on the cluster's rank-0 node
#
# Each step launches a group through the site's own launcher, runs its checks against rank 0 and
# stops the group, whatever the checks said; --apply runs every step and prints one line a step.
#
# The site supplies (placeholders here, so nothing names a real host):
#   SPARK_NODES              four node addresses in rank order, e.g. "192.0.2.10 192.0.2.11 ..."
#   SPARKFLASH_BIN           the build under test, the same path on every node
#   SPARKFLASH_OLD_BIN       the frozen pre-move build with the tick-trace emission cherry-picked
#   DEEPSEEK_CHECKPOINT      the DeepSeek-V4.1 checkpoint directory on every node
#   GLM_CHECKPOINT           GLM-5.3-Flash FP8, QWEN_CHECKPOINT Qwen3.8-Flash-Next NVFP4
#   DENSE_CHECKPOINT         a dense checkpoint (the synthetic one is enough), DENSE_MODEL its id
#   LONG_PROMPT              the served checks' long prompt (a text file of 60,000 characters)
#   INKLING_SITE             the Inkling site file scripts/inkling-env.sh reads
set -uo pipefail

APPLY=0
case "${1:-}" in
  --apply) APPLY=1 ;;
  ""|--dry-run) ;;
  *) echo "usage: scripts/owed-cluster-window.sh [--dry-run|--apply]" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SPARKFLASH_BIN:=$ROOT/target/release/sparkflash}"
: "${SPARKFLASH_OLD_BIN:=<the frozen pre-move build>}"
: "${DEEPSEEK_CHECKPOINT:=<DeepSeek-V4.1 checkpoint>}"
: "${GLM_CHECKPOINT:=<GLM-5.3-Flash FP8 checkpoint>}"
: "${QWEN_CHECKPOINT:=<Qwen3.8-Flash-Next NVFP4 checkpoint>}"
: "${DENSE_CHECKPOINT:=<dense checkpoint>}"
: "${DENSE_MODEL:=<dense model id>}"
: "${LONG_PROMPT:=<long prompt file>}"
GATE="python3 $ROOT/scripts/served-gate.py all 8000 --long $LONG_PROMPT"
FOUR="$ROOT/scripts/serve-four-node.sh"

# One step: a name, then the commands, each run in order; the last command of a step always runs.
STEPS=(
  "dense TP 4 on the NCCL leg|SPARKFLASH_CHECKPOINT=$DENSE_CHECKPOINT SPARKFLASH_MODEL=$DENSE_MODEL SPARK_COMM_REDUCE=nccl $FOUR launch|$GATE|$FOUR stop"
  "GLM-5.3 TP 4, released checkpoint, NCCL leg|SPARKFLASH_CHECKPOINT=$GLM_CHECKPOINT SPARKFLASH_MODEL=glm5.3-flash-fp8 SPARK_COMM_REDUCE=nccl $FOUR launch|$GATE|python3 $ROOT/scripts/numerics.py check --url http://127.0.0.1:8000|$FOUR stop"
  "Qwen3.8 TP 4, released checkpoint, NCCL leg|SPARKFLASH_CHECKPOINT=$QWEN_CHECKPOINT SPARKFLASH_MODEL=qwen3.8-flash-next-nvfp4 SPARK_COMM_REDUCE=nccl $FOUR launch|$GATE|python3 $ROOT/scripts/numerics.py check --url http://127.0.0.1:8000|$FOUR stop"
  "Inkling's image group (the request frame before the start)|$ROOT/scripts/inkling-serve-group.sh launch|python3 $ROOT/scripts/inkling-image-probe.py|$GATE|$ROOT/scripts/inkling-serve-group.sh stop"
  "DeepSeek TP 4 on the two-rail RDMA exchange (the default)|SPARKFLASH_CHECKPOINT=$DEEPSEEK_CHECKPOINT $FOUR launch|$GATE|$FOUR stop"
  "DeepSeek's Batch device gate (the tick-trace A/B)|python3 $ROOT/scripts/tick-trace-ab.py --old $SPARKFLASH_OLD_BIN --new $SPARKFLASH_BIN --checkpoint $DEEPSEEK_CHECKPOINT --site-script $FOUR"
)

failed=0
for step in "${STEPS[@]}"; do
  IFS='|' read -r -a parts <<< "$step"
  name="${parts[0]}"
  echo "== $name"
  status=0
  for index in $(seq 1 $((${#parts[@]} - 1))); do
    command="${parts[$index]}"
    if [ "$APPLY" -eq 0 ]; then
      echo "   $command"
      continue
    fi
    # A failed check does not skip the step's stop: the last command always runs.
    if [ "$status" -ne 0 ] && [ "$index" -lt $((${#parts[@]} - 1)) ]; then
      continue
    fi
    bash -c "$command" || status=1
  done
  if [ "$APPLY" -eq 1 ]; then
    echo "   $name: $([ "$status" -eq 0 ] && echo PASS || echo FAIL)"
    [ "$status" -eq 0 ] || failed=1
  fi
done
[ "$APPLY" -eq 1 ] || echo "# dry run: pass --apply on the cluster's rank-0 node to execute"
exit "$failed"
