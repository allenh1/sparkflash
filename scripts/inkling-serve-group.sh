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
# Four-rank Inkling-2-Small device-arm server. Rank 0 binds HTTP and the control channel; ranks
# 1-3 mirror its prefill and decode on their own converted rank files.
#
#   TRACE=1 ./inkling-serve-group.sh launch    # rank 0 blocks; the others are detached over ssh
#   ./inkling-serve-group.sh stop
#
# The defaults name the four-node window (sg1-oneill/jackson/carter/teilc) and the converted ranks
# under /home/allenh1/models/inkling2-small; BIN, CONTEXT, BIND and TRACE override what a run needs.
set -uo pipefail
LOG=/home/allenh1/inkling-bin
BIN=${BIN:-/home/allenh1/inkling-bin/sparkflash}
CKPT=/home/allenh1/models/inkling2-small/ckpt
CONVERTED=/home/allenh1/models/inkling2-small/converted
PEERS=192.168.1.80:29600,192.168.1.68:29600,192.168.1.63:29600,192.168.1.88:29600
CONTEXT=${CONTEXT:-4096}
BIND=${BIND:-0.0.0.0:8010}
TRACE=${TRACE:-0}
declare -A HOSTS=( [1]=192.168.1.68 [2]=192.168.1.63 [3]=192.168.1.88 )
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes"

common=(
  --model inkling2
  --checkpoint "$CKPT"
  --converted "$CONVERTED"
  --peer "$PEERS"
  --world 4
  --context "$CONTEXT"
  --tokenizer "$CKPT/tokenizer.json"
)

case "${1:-launch}" in
stop)
  for rank in 1 2 3; do
    $SSH allenh1@${HOSTS[$rank]} "pkill -f 'sparkflash serve --model inkling2'" >/dev/null 2>&1 || true
  done
  pkill -f 'sparkflash serve --model inkling2' >/dev/null 2>&1 || true
  echo "serve-group: stopped"
  ;;
launch)
  for rank in 1 2 3; do
    ip=${HOSTS[$rank]}
    # setsid + </dev/null detaches the rank from the ssh channel: without it sshd waits for the
    # inherited stdin/stdout and the launcher never reaches rank 0.
    $SSH allenh1@$ip "cd /home/allenh1/inkling-bin && ( SPARKFLASH_INKLING_TRACE=$TRACE setsid nohup $BIN serve ${common[*]} --rank $rank > $LOG/group-rank$rank.log 2>&1 < /dev/null & ) ; echo started-rank$rank"
    sleep 1
  done
  echo "serve-group: rank 0 here"
  cd /home/allenh1/inkling-bin
  SPARKFLASH_INKLING_TRACE=$TRACE $BIN serve ${common[*]} --rank 0 --bind "$BIND" > $LOG/group-rank0.log 2>&1
  echo "serve-group: rank 0 exited $?"
  ;;
*)
  echo "usage: $0 [launch|stop]" >&2
  exit 2
  ;;
esac
