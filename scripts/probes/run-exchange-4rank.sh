#!/usr/bin/env bash
# Runs the four-rank exchange probe on a live cluster and prints one line per rank.
#
#   scripts/probes/run-exchange-4rank.sh <bytes> [iters] [port]
#
# The probe takes a rank-order address table and its own address; this script is what supplies
# them. Nothing is baked in - the addresses, the node names and the rail device all come from the
# environment, because which addresses a rank is on is a deployment fact and a default here would
# be one fleet's addresses in the repository.
#
# Environment:
#   SPARK_NODES       four rail addresses in rank order (overrides the table below)
#   SPARK_HOSTS       four ssh targets in rank order (overrides the names below)
#   SPARK_COMM_PEERS  peer table handed to every rank (default: SPARK_NODES joined by commas)
#   SPARK_EXCH_DEV    the RDMA device those addresses live on (default rocep1s0f1)
#   SSH_USER          ssh user (default $USER)
#
# Rank 0 is the head: this script reaches it directly and every other rank through it, and only
# the head builds, because the workers in this fleet are run-only. That is also why the binary is
# copied from the head to the workers rather than from wherever this is run.
#
# Measured 2026-11-09 on the four-node GB10 cluster, with the serving group up and serving:
#   10240  B: first arrival 5.7 us, all three peers 7.3 us (p50 7.31, p99 9.0, max 10.6)
#   163840 B: first arrival 43.4 us, all three peers 46.9 us (p50 46.5, p99 57.1, max 60.7)
# Re-run on an idle box both came back within noise of these, so at these sizes the exchange is
# latency-bound and does not care about the collective traffic beside it. The NCCL group it
# replaces costs 43 us, so it wins on a decode row and is a wash on a 160 KiB one. What that bought
# in a serving step is in docs/rdma-exchange.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BYTES="${1:-10240}"
ITERS="${2:-200}"
PORT="${3:-18860}"
SSH_USER="${SSH_USER:-$USER}"
DEV="${SPARK_EXCH_DEV:-rocep1s0f1}"
# Required, not defaulted: this file used to carry TEST-NET addresses and placeholder host names,
# which meant a run with neither variable set quietly measured documentation addresses instead of
# failing. Nothing about the fabric is baked in here.
: "${SPARK_NODES:?set SPARK_NODES to the four rail addresses in rank order}"
: "${SPARK_HOSTS:?set SPARK_HOSTS to the four ssh targets in rank order}"
NODES=(${SPARK_NODES})
HOSTS=(${SPARK_HOSTS})
# SPARK_COMM_PEERS follows the crate's comma-separated convention; the probe's --peer takes four
# separate arguments, so the table is split here rather than there.
PEERS="${SPARK_COMM_PEERS:-$(IFS=,; echo "${NODES[*]}")}"
PEERS_SPACE="${PEERS//,/ }"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
BIN=/tmp/gb10-exchange-4rank

die() { printf 'run-exchange-4rank: %s\n' "$*" >&2; exit 1; }

[ "${#NODES[@]}" -eq 4 ] || die "SPARK_NODES needs four addresses, got ${#NODES[@]}"
[ "${#HOSTS[@]}" -eq 4 ] || die "SPARK_HOSTS needs four targets, got ${#HOSTS[@]}"

# Rank 0 directly, every other rank through rank 0.
remote() {
  local rank=$1; shift
  if [ "$rank" -eq 0 ]; then
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${HOSTS[0]}" "$*"
  else
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${HOSTS[0]}" "ssh -n ${SSH_OPTS[*]} $SSH_USER@${HOSTS[$rank]} '$*'"
  fi
}

# A probe left running from an earlier attempt holds both the binary and the port, and scp then
# fails with "text file busy" on a worker that looks perfectly healthy. The bracket in the pattern
# keeps pkill from matching the shell that is running it.
for rank in 0 1 2 3; do
  remote "$rank" "pkill -f 'gb10-exchange-4ran[k]' || true" &
done
wait

# Only the head builds; the workers are run-only.
scp -q "${SSH_OPTS[@]}" "$HERE/gb10-exchange-4rank.cu" "$SSH_USER@${HOSTS[0]}:/tmp/gb10-exchange-4rank.cu"
remote 0 "cc -O2 -x c -o $BIN /tmp/gb10-exchange-4rank.cu -libverbs"
for rank in 1 2 3; do
  remote 0 "scp -q $BIN $SSH_USER@${HOSTS[$rank]}:$BIN"
done

for rank in 0 1 2 3; do
  CMD="$BIN --rank $rank --peer $PEERS_SPACE --myip ${NODES[$rank]} --dev $DEV --port $PORT --bytes $BYTES --iters $ITERS"
  remote "$rank" "timeout 180 $CMD > /tmp/exch-$rank.txt 2>&1" &
done
wait

for rank in 0 1 2 3; do
  printf '%s\n' "--- rank $rank ---"
  remote "$rank" "cat /tmp/exch-$rank.txt"
done