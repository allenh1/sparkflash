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
# The hosts, ports, paths and NCCL environment come from the site file scripts/inkling-env.sh
# reads; BIN, CONTEXT, BIND, TRACE and PREFILL override what a run needs.
#
# CONTEXT defaults to the released window, 1,048,576 positions: eight full layers of KV is 8 KiB
# a token a rank, so a rank's cache is 8.04 GiB and the four rank files plus it are 99.4 GiB of
# the engine's 113.7 GiB budget. A smaller CONTEXT is the way to run when something else is using
# the memory. The tensor-core chunked prefill is the default; PREFILL=chunk keeps a prompt's rows
# on the decode step's bits and PREFILL=rows forces the row-at-a-time walk.
#
# LANES=N (default 4) serves N requests at once as fused lanes on every rank: each keeps its own
# conversation, so a client's next turn continues the lane that holds it, and a lane drafts with
# DSpark as a request alone does. The lanes' caches are backed as they grow, from one pool a
# rank's lanes share: one whole CONTEXT's worth unless SPARKFLASH_LANE_MEMORY_MB says otherwise.
# A lane that needs more takes it from the conversation that has idled longest. LANES=1 serves
# one request at a time.
set -uo pipefail
source "$(dirname "$0")/inkling-env.sh"
CKPT=${CKPT:-$MODEL_DIR/ckpt}
CONVERTED=${CONVERTED:-$MODEL_DIR/converted}
CONTEXT=${CONTEXT:-1048576}
BIND=${BIND:-0.0.0.0:8000}
LANES=${LANES:-4}

common=(
  --model inkling2
  --checkpoint "$CKPT"
  --converted "$CONVERTED"
  --peer "$PEERS"
  --world 4
  --context "$CONTEXT"
  --tokenizer "$CKPT/tokenizer.json"
)
if [ "$LANES" -gt 1 ]; then
  common+=(--concurrency "$LANES" --fuse-batch)
fi

case "${1:-launch}" in
stop)
  for rank in 1 2 3; do
    $SSH "$(rank_host $rank)" "pkill -f '[s]erve --model inkling2'" >/dev/null 2>&1 || true
  done
  pkill -f '[s]erve --model inkling2' >/dev/null 2>&1 || true
  echo "serve-group: stopped"
  ;;
launch)
  for rank in 1 2 3; do
    # setsid + </dev/null detaches the rank from the ssh channel: without it sshd waits for the
    # inherited stdin/stdout and the launcher never reaches rank 0.
    $SSH "$(rank_host $rank)" "cd $LOG && ( setsid nohup env ${RANK_ENV[*]} $BIN serve ${common[*]} --rank $rank > $LOG/group-rank$rank.log 2>&1 < /dev/null & ) ; echo started-rank$rank"
    sleep 1
  done
  echo "serve-group: rank 0 here"
  cd "$LOG"
  env "${RANK_ENV[@]}" "$BIN" serve "${common[@]}" --rank 0 --bind "$BIND" > "$LOG/group-rank0.log" 2>&1
  echo "serve-group: rank 0 exited $?"
  ;;
*)
  echo "usage: $0 [launch|stop]" >&2
  exit 2
  ;;
esac
