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
# SPARKFLASH_INKLING_PREFILL=chunk (the default here) or rows selects the prefill walk.
set -uo pipefail
LOG=${LOG:-/home/allenh1/inkling-bin}
BIN=${BIN:-/home/allenh1/inkling-bin/sparkflash}
MODEL=${MODEL:-/home/allenh1/models/inkling2-small/ckpt}
CONVERTED=${CONVERTED:-/home/allenh1/models/inkling2-small/converted}
PEERS=${PEERS:-192.168.1.80:29600,192.168.1.68:29600,192.168.1.63:29600,192.168.1.88:29600}
CONTEXT=${CONTEXT:-4096}
TRACE=${TRACE:-0}
PREFILL=${PREFILL:-chunk}
DECODE=${DECODE:-9}
declare -A HOSTS=( [1]=192.168.1.68 [2]=192.168.1.63 [3]=192.168.1.88 )
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"

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
  ip=${HOSTS[$rank]}
  $SSH allenh1@$ip "cd $LOG && ( SPARKFLASH_INKLING_TRACE=$TRACE SPARKFLASH_INKLING_PREFILL=$PREFILL setsid nohup $BIN ${common[*]} --rank $rank > $LOG/batch-rank$rank.log 2>&1 < /dev/null & ) ; echo started-rank$rank"
  sleep 1
done
echo "batch: rank 0 here, prefill $PREFILL"
SPARKFLASH_INKLING_TRACE=$TRACE SPARKFLASH_INKLING_PREFILL=$PREFILL $BIN ${common[*]} --rank 0 2>&1 | tee $LOG/batch-rank0.log | grep -E 'prompt|generated|prefill|decode step|best' || true
for rank in 1 2 3; do
  $SSH allenh1@$ip true >/dev/null 2>&1 || true
done
echo "batch: done"
