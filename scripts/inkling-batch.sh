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
#
# One four-host run of the batch arm (sparkflash inkling-tp4), for the prefill gate.
#
#   ./inkling-batch.sh --prompt 371,151451,149879,151194,149884 --decode 9
#
# The arguments after the script name go to every rank. The prompt is token ids rather than text
# so the run has no tokenizer in the way: the chunked prefill and the row-at-a-time walk are
# compared on the ids they generate, and the arm's own tokenizer is not one of the variables.
#
# PREFILL=tensor (the default), chunk (the decode step's bits), rows or gemm selects the prefill
# walk (SPARKFLASH_INKLING_PREFILL; an unknown one is refused at open), and an exported
# SPARKFLASH_INKLING_ATTENTION (fast, split, serial or check) the attention walk, as every exported
# SPARKFLASH_INKLING_* variable reaches every rank. --text-file reads a
# prompt too long for one argument from a file at the same path on every rank. The hosts, ports
# and NCCL environment come from the site file scripts/inkling-env.sh reads. NSYS=/path/report
# profiles rank 0 under Nsight Systems (CUDA and OS runtime traces) into that report; NSYS_ALL=1
# also profiles ranks 1-3, each into ${NSYS}-rank<r> on its own host.
set -uo pipefail
source "$(dirname "$0")/inkling-env.sh"
MODEL=${MODEL:-$MODEL_DIR/ckpt}
CONVERTED=${CONVERTED:-$MODEL_DIR/converted}
CONTEXT=${CONTEXT:-4096}
DECODE=${DECODE:-9}

common=(
  inkling-tp4
  --model "$MODEL"
  --converted "$CONVERTED"
  --peer "$PEERS"
  --world 4
  --context "$CONTEXT"
  --decode "$DECODE"
  "$@"
)

for rank in 1 2 3; do
  remote_profile=""
  # NSYS_ALL=1 profiles every rank: rank r writes ${NSYS}-rank$r on its own host.
  [ -n "${NSYS:-}" ] && [ "${NSYS_ALL:-0}" = 1 ] \
    && remote_profile="nsys profile -t cuda -f true -o ${NSYS}-rank$rank"
  $SSH "$(rank_host $rank)" "cd $LOG && ( setsid nohup env ${RANK_ENV[*]} $remote_profile $BIN ${common[*]} --rank $rank > $LOG/batch-rank$rank.log 2>&1 < /dev/null & ) ; echo started-rank$rank"
  sleep 1
done
echo "batch: rank 0 here, prefill $PREFILL"
profile=()
[ -n "${NSYS:-}" ] && profile=(nsys profile -t cuda,osrt -f true -o "$NSYS")
env "${RANK_ENV[@]}" "${profile[@]}" "$BIN" "${common[@]}" --rank 0 2>&1 | tee "$LOG/batch-rank0.log" \
  | grep -E 'prompt|generated|prefill|decode step|best|error|refus' || true
echo "batch: done"
