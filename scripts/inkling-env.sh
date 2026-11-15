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
# The site settings the Inkling launch scripts share; sourced, not run.
#
# A site file (INKLING_SITE, default ~/inkling-bin/site.env) sets what differs between clusters;
# the defaults below are placeholders (TEST-NET addresses) so nothing here names a real host.
#
#   INKLING_NODES       the four hosts in rank order, space-separated (rank 0 is this host)
#   INKLING_USER        the ssh user on the other ranks (empty: ssh's own default)
#   INKLING_PORT        the model collective's base port (each rank listens on it)
#   INKLING_HOME        where the binary, logs and model directories live on every rank
#   SPARKFLASH_NCCL_LIB a directory holding libnccl.so.2 on every rank, prepended to
#                       LD_LIBRARY_PATH; the NCCL device leg needs it
#   SPARK_COMM_REDUCE   nccl (the default) or tcp: the collective leg, forwarded to every rank
#   NCCL_*              every NCCL_ variable set is forwarded to every rank (NCCL_IB_HCA,
#                       NCCL_IB_GID_INDEX, NCCL_SOCKET_IFNAME, ...)
#   SPARKFLASH_EXCHANGE the direct-RDMA exchange (the GB10 doorbell) in front of the NCCL leg:
#                       unset is the default (on), rdma requires it, nccl leaves NCCL alone
#   SPARK_COMM_PEERS    the doorbell's control-plane table, one address per rank in rank order
#                       (default: INKLING_NODES); its rails and GIDs come from each rank's fabric
#   SPARK_COMM_CONTROL_PORT the doorbell's control port (default: INKLING_PORT + 105)
#   SPARK_COMM_RAIL_COUNT   the doorbell's rails (default: the backend's, two)
#   SPARKFLASH_RAIL_DEVICES the doorbell's two RDMA devices as rail0,rail1 (default: discovered);
#                       with SPARK_COMM_RAIL_COUNT=1 only rail0 carries the rows
#   SPARKFLASH_REUSE_BY_TEXT 0 stops a held reply from continuing a prompt that spells it with
#                       other ids (default: on); forwarded to every rank
SITE=${INKLING_SITE:-$HOME/inkling-bin/site.env}
# shellcheck disable=SC1090
[ -f "$SITE" ] && source "$SITE"
INKLING_HOME=${INKLING_HOME:-$HOME}
read -r -a NODES <<< "${INKLING_NODES:-192.0.2.10 192.0.2.11 192.0.2.12 192.0.2.13}"
INKLING_PORT=${INKLING_PORT:-29600}
LOG=${LOG:-$INKLING_HOME/inkling-bin}
BIN=${BIN:-$INKLING_HOME/inkling-bin/sparkflash}
MODEL_DIR=${MODEL_DIR:-$INKLING_HOME/models/inkling2-small}
TRACE=${TRACE:-0}
PREFILL=${PREFILL:-tensor}
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR"

PEERS=""
for node in "${NODES[@]}"; do
  PEERS+="${PEERS:+,}$node:$INKLING_PORT"
done
# The doorbell's table defaults to the ranks' own hosts, so a launch that names INKLING_NODES and
# nothing else still configures it: an unset table makes every rank stand down to NCCL together.
SPARK_COMM_PEERS=${SPARK_COMM_PEERS:-$(IFS=,; printf '%s' "${NODES[*]}")}
SPARK_COMM_CONTROL_PORT=${SPARK_COMM_CONTROL_PORT:-$((INKLING_PORT + 105))}

# The environment every rank runs with, as KEY=value words for `env`.
RANK_ENV=(SPARKFLASH_INKLING_TRACE="$TRACE" SPARKFLASH_INKLING_PREFILL="$PREFILL")
[ -n "${SPARK_COMM_REDUCE:-}" ] && RANK_ENV+=(SPARK_COMM_REDUCE="$SPARK_COMM_REDUCE")
RANK_ENV+=(SPARK_COMM_PEERS="$SPARK_COMM_PEERS" SPARK_COMM_CONTROL_PORT="$SPARK_COMM_CONTROL_PORT")
[ -n "${SPARKFLASH_EXCHANGE:-}" ] && RANK_ENV+=(SPARKFLASH_EXCHANGE="$SPARKFLASH_EXCHANGE")
[ -n "${SPARK_COMM_RAIL_COUNT:-}" ] && RANK_ENV+=(SPARK_COMM_RAIL_COUNT="$SPARK_COMM_RAIL_COUNT")
[ -n "${SPARKFLASH_RAIL_DEVICES:-}" ] && RANK_ENV+=(SPARKFLASH_RAIL_DEVICES="$SPARKFLASH_RAIL_DEVICES")
[ -n "${SPARKFLASH_REUSE_BY_TEXT:-}" ] && RANK_ENV+=(SPARKFLASH_REUSE_BY_TEXT="$SPARKFLASH_REUSE_BY_TEXT")
[ -n "${SPARKFLASH_NCCL_LIB:-}" ] && RANK_ENV+=(LD_LIBRARY_PATH="$SPARKFLASH_NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
for name in $(compgen -v | grep '^NCCL_'); do
  RANK_ENV+=("$name=${!name}")
done
# Every other arm switch the caller exported (the chunk width, the attention walk, ...): a rank
# that walked a different schedule from its peers would meet them at the wrong collective.
for name in $(compgen -e | grep '^SPARKFLASH_INKLING_' | grep -v -e '^SPARKFLASH_INKLING_TRACE$' -e '^SPARKFLASH_INKLING_PREFILL$'); do
  RANK_ENV+=("$name=${!name}")
done

# ssh target for a rank's host.
rank_host() {
  echo "${INKLING_USER:+$INKLING_USER@}${NODES[$1]}"
}
