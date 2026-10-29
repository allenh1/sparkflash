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

# Single-command four-node TP=4 serving launcher.
#
# This is the host-side command builder and process launcher for the four-rank serving topology:
# one sparkflash serve process per node, rank 0 owning HTTP, the tokenizer and the control
# channel, ranks 1-3 running the resident forward. It does not make a measurement and does not
# claim one: the print mode is offline, and a launch only starts what the operator staged.
#
# The binary must be built with the tp4-serve feature (and cuda for the device path):
#   cargo build --release -p sparkflash-cli --features cuda,tp4-serve
#
# Usage:
#   scripts/serve-four-node.sh print     print the four commands (default; no network access)
#   scripts/serve-four-node.sh launch    start four ranks over ssh, one per node
#   scripts/serve-four-node.sh stop      send SIGTERM to every rank (the graceful-drain signal)
#   scripts/serve-four-node.sh status    report whether each rank process is running
#
# Environment (required for launch; print only needs the paths):
#   SPARK_NODES            4 node addresses in rank order (space separated)
#   SPARKFLASH_CHECKPOINT  sharded checkpoint directory, same path on every node
#   SPARKFLASH_TOKENIZER   tokenizer.json path, same path on every node
# Device path (what makes it fast; leave SPARKFLASH_CUBINS unset for the CPU reference engine):
#   SPARKFLASH_CUBINS      cubin directory on every node (scripts/build-cubins.sh): --tp4-device
#   SPARK_PEERS            the four RoCE addresses the collectives use, in rank order
#                          (default: SPARK_NODES)
#   SPARKFLASH_NCCL_LIB    directory holding libnccl.so.2 on every node (prepended to LD_LIBRARY_PATH)
#   SPARKFLASH_CONVERTED_DIR  directory of `sparkflash convert` rank files on every node (fast boot)
#   NCCL_*                 every NCCL_ variable set here is passed to every rank (NCCL_IB_HCA,
#                          NCCL_SOCKET_IFNAME, NCCL_IB_GID_INDEX, ...)
# Options:
#   SPARKFLASH_BIN         binary on every node (default <repo>/target/release/sparkflash)
#   SPARKFLASH_HTTP_BIND   rank-0 HTTP bind (default 0.0.0.0:8000)
#   SPARKFLASH_TP4_BASE_PORT     collective base port (default 29601; rank R gets base+R)
#   SPARKFLASH_TP4_CONTROL_PORT  rank-0 control port (default base + 100)
#   SPARKFLASH_LOG_ROOT    remote log directory (default <repo>/target/serve-tp4)
#   SPARKFLASH_DRAIN_DELAY_MS    rank-0 pre-stop delay (default 5000)
#   SPARKFLASH_DRAIN_TIMEOUT_MS  rank-0 drain deadline (default 30000)
#   SPARK_MAX_SEQ_LEN      served sequence window (forwarded to every rank; the web UI sets it)
#   SPARKFLASH_LANES       concurrent sequence slots (forwarded to every rank; the web UI sets it)
#   SPARKFLASH_PREFILL_*, SPARKFLASH_BATCH_*, SPARKFLASH_DECODE_*  scheduler knobs (forwarded to
#                          every rank: the ranks must agree on every tick)
#   SPARKFLASH_DEVICE_TOKEN_DIR  token window (or live window parent) served by rank 0
#   SPARKFLASH_DEVICE_TOKEN_LIVE=1        follow that directory as a live window directory (W5)
#   SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM=1  accept a window without the device's claim assertion
#   SSH_USER, SSH_OPTS     ssh user and options (defaults: current user, BatchMode)
#   SPARK_LOCAL_RANKS      ranks that run on this node without ssh (space separated, e.g. "0";
#                          the web UI sets it from `local = true`), for a lead that cannot ssh to
#                          itself
#   SPARKFLASH_DRY_RUN=1   same as print, for scripts that want one code path
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-print}"
[ "${SPARKFLASH_DRY_RUN:-0}" = "1" ] && MODE=print
SSH_USER="${SSH_USER:-${USER:-}}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8}"
NODES_RAW="${SPARK_NODES:-}"
CHECKPOINT="${SPARKFLASH_CHECKPOINT:-}"
TOKENIZER="${SPARKFLASH_TOKENIZER:-}"
BIN="${SPARKFLASH_BIN:-$REPO_ROOT/target/release/sparkflash}"
HTTP_BIND="${SPARKFLASH_HTTP_BIND:-0.0.0.0:8000}"
BASE_PORT="${SPARKFLASH_TP4_BASE_PORT:-29601}"
CONTROL_PORT="${SPARKFLASH_TP4_CONTROL_PORT:-$((BASE_PORT + 100))}"
LOG_ROOT="${SPARKFLASH_LOG_ROOT:-$REPO_ROOT/target/serve-tp4}"
RUN_ID="${SPARKFLASH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRAIN_DELAY_MS="${SPARKFLASH_DRAIN_DELAY_MS:-5000}"
DRAIN_TIMEOUT_MS="${SPARKFLASH_DRAIN_TIMEOUT_MS:-30000}"
DEVICE_TOKEN_DIR="${SPARKFLASH_DEVICE_TOKEN_DIR:-}"
DEVICE_TOKEN_LIVE="${SPARKFLASH_DEVICE_TOKEN_LIVE:-0}"
DEVICE_TOKEN_DERIVED_CLAIM="${SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM:-0}"
CUBINS="${SPARKFLASH_CUBINS:-}"
NCCL_LIB="${SPARKFLASH_NCCL_LIB:-}"
CONVERTED_DIR="${SPARKFLASH_CONVERTED_DIR:-}"

read -r -a NODES <<< "$NODES_RAW"
read -r -a LOCAL_RANKS <<< "${SPARK_LOCAL_RANKS:-}"
die() { printf 'serve-four-node: %s\n' "$*" >&2; exit 1; }

[ "${#NODES[@]}" -eq 4 ] || die "SPARK_NODES needs exactly 4 addresses in rank order, got ${#NODES[@]}"
# Only a start reads the weights; stop and status work without them.
if [ "$MODE" = print ] || [ "$MODE" = launch ]; then
    [ -n "$CHECKPOINT" ] || die "SPARKFLASH_CHECKPOINT is required"
    [ -n "$TOKENIZER" ] || die "SPARKFLASH_TOKENIZER is required"
fi
for rank in ${LOCAL_RANKS[@]+"${LOCAL_RANKS[@]}"}; do
    case "$rank" in 0|1|2|3) ;; *) die "SPARK_LOCAL_RANKS entries must be ranks 0-3, got '$rank'" ;; esac
done
PEERS="$(IFS=,; printf '%s' "${NODES[*]}")"
if [ -n "${SPARK_PEERS:-}" ]; then
    read -r -a PEER_LIST <<< "$SPARK_PEERS"
    [ "${#PEER_LIST[@]}" -eq 4 ] || die "SPARK_PEERS needs exactly 4 addresses in rank order, got ${#PEER_LIST[@]}"
    PEERS="$(IFS=,; printf '%s' "${PEER_LIST[*]}")"
fi

# The environment every rank starts with. The drain knobs only matter on rank 0, which owns the
# HTTP surface, but exporting them uniformly keeps the four command lines identical apart from the
# rank-specific flags.
rank_env() {
    printf 'SPARKFLASH_SERVE_DRAIN_DELAY_MS=%s SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS=%s' \
        "$DRAIN_DELAY_MS" "$DRAIN_TIMEOUT_MS"
    # The two settings the control plane owns. They are forwarded from the launcher environment
    # rather than hard-coded so a web-driven launch and a hand-written one run the same flags.
    [ -n "${SPARK_MAX_SEQ_LEN:-}" ] && printf ' SPARK_MAX_SEQ_LEN=%s' "$SPARK_MAX_SEQ_LEN"
    [ -n "${SPARKFLASH_LANES:-}" ] && printf ' SPARKFLASH_LANES=%s' "$SPARKFLASH_LANES"
    [ -n "${SPARKFLASH_SERVE_API_KEY:-}" ] && printf ' SPARKFLASH_SERVE_API_KEY=%q' "$SPARKFLASH_SERVE_API_KEY"
    if [ -n "$CUBINS" ]; then
        printf ' SPARK_COMM_REDUCE=nccl SPARKFLASH_LEAN_HEAD=1'
        [ -n "$NCCL_LIB" ] && printf ' LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' "$NCCL_LIB"
        local var
        for var in $(compgen -e | grep '^NCCL_' || true); do
            printf ' %s=%s' "$var" "${!var}"
        done
    fi
    [ -n "$CONVERTED_DIR" ] && printf ' SPARKFLASH_CONVERTED_DIR=%s' "$CONVERTED_DIR"
    # The scheduler's knobs: every rank runs the same ticks, so a prefill or batching setting
    # that only a local rank 0 inherited would split the group's collectives.
    local knob
    for knob in $(compgen -e | grep -E '^SPARKFLASH_(PREFILL|BATCH|DECODE)_' || true); do
        printf ' %s=%q' "$knob" "${!knob}"
    done
    return 0
}

# The one command line per rank. Rank 0 binds the public HTTP address; ranks 1-3 keep a loopback
# bind because they never serve HTTP. The device token window flags are rank-0 only: workers never
# detokenize, so a window on a worker would be an unused second copy.
rank_command() {
    local rank="$1"
    printf '%s serve --checkpoint %s --tokenizer %s' "$BIN" "$CHECKPOINT" "$TOKENIZER"
    printf ' --tp4-rank %s --tp4-world-size 4 --tp4-peers %s' "$rank" "$PEERS"
    printf ' --tp4-base-port %s --tp4-control-port %s' "$BASE_PORT" "$CONTROL_PORT"
    [ -n "$CUBINS" ] && printf ' --tp4-device %s' "$CUBINS"
    if [ "$rank" -eq 0 ]; then
        printf ' --bind %s' "$HTTP_BIND"
        if [ -n "$DEVICE_TOKEN_DIR" ]; then
            printf ' --device-token-dir %s' "$DEVICE_TOKEN_DIR"
            [ "$DEVICE_TOKEN_LIVE" = "1" ] && printf ' --device-token-live'
            [ "$DEVICE_TOKEN_DERIVED_CLAIM" = "1" ] && printf ' --device-token-derived-claim'
        fi
    else
        printf ' --bind 127.0.0.1:0'
    fi
}

is_local() {
    local rank="$1" entry
    for entry in ${LOCAL_RANKS[@]+"${LOCAL_RANKS[@]}"}; do
        [ "$entry" = "$rank" ] && return 0
    done
    return 1
}

# Runs a command for one rank: on this node for a local rank, over ssh otherwise.
remote() {
    local rank="$1"; shift
    if is_local "$rank"; then
        bash -c "$*" < /dev/null
    else
        ssh -n ${SSH_OPTS} "${SSH_USER:+$SSH_USER@}${NODES[$rank]}" "$*"
    fi
}

# The detached start for one rank. `exec` replaces the backgrounded shell with the rank, so
# nothing keeps the caller's stdout open: without it ssh (and a web job) waits until the rank
# exits, and `$!` is the rank's own pid.
start_line() {
    local rank="$1"
    printf "mkdir -p '%s/%s' && cd '%s' && exec nohup env %s %s > '%s/%s/rank%s.log' 2>&1 &" \
        "$LOG_ROOT" "$RUN_ID" "$REPO_ROOT" "$(rank_env)" "$(rank_command "$rank")" \
        "$LOG_ROOT" "$RUN_ID" "$rank"
}

print_plan() {
    printf 'serve-four-node: run %s\n' "$RUN_ID"
    printf 'serve-four-node: nodes %s\n' "$NODES_RAW"
    printf 'serve-four-node: peers %s\n' "$PEERS"
    local rank
    for rank in 0 1 2 3; do
        if is_local "$rank"; then
            printf 'serve-four-node: rank %s -> %s (this node)\n' "$rank" "${NODES[$rank]}"
            printf '  bash -c "%s"\n' "$(start_line "$rank")"
        else
            printf 'serve-four-node: rank %s -> %s\n' "$rank" "${NODES[$rank]}"
            printf '  ssh -n %s@%s "%s"\n' "$SSH_USER" "${NODES[$rank]}" "$(start_line "$rank")"
        fi
    done
    printf 'serve-four-node: rank 0 serves http://%s/health, /readyz and /metrics\n' "$HTTP_BIND"
}

case "$MODE" in
    print)
        print_plan
        ;;
    launch)
        [ -x "$BIN" ] || printf 'serve-four-node: local binary %s is not executable; the remote nodes must have it staged\n' "$BIN" >&2
        local_plan="$LOG_ROOT/$RUN_ID/local-commands.txt"
        mkdir -p "$LOG_ROOT/$RUN_ID"
        print_plan > "$local_plan"
        for rank in 0 1 2 3; do
            remote "$rank" "$(start_line "$rank") echo \$!" \
                | sed "s/^/rank $rank pid /"
        done
        printf 'serve-four-node: launched; commands in %s\n' "$local_plan"
        printf 'serve-four-node: rank 0 health endpoint http://%s/health\n' "$HTTP_BIND"
        ;;
    stop)
        for rank in 0 1 2 3; do
            # SIGTERM is the drain signal: rank 0 turns /readyz 503, finishes in-flight requests and
            # exits 0. The pattern is the exact rank flag, so no other serve process is matched.
            # The bracket keeps the pattern from matching the shell that carries it.
            remote "$rank" "pkill -TERM -f '[s]parkflash serve --checkpoint .* --tp4-rank $rank ' >/dev/null 2>&1 || true" || true
            printf 'serve-four-node: rank %s SIGTERM sent\n' "$rank"
        done
        printf 'serve-four-node: in-flight requests drain for up to %s ms\n' "$DRAIN_TIMEOUT_MS"
        ;;
    status)
        for rank in 0 1 2 3; do
            if remote "$rank" "pgrep -f '[t]p4-rank $rank ' >/dev/null" >/dev/null 2>&1; then
                printf 'serve-four-node: rank %s running on %s\n' "$rank" "${NODES[$rank]}"
            else
                printf 'serve-four-node: rank %s not running on %s\n' "$rank" "${NODES[$rank]}"
            fi
        done
        ;;
    *)
        die "unknown mode '$MODE' (expected print, launch, stop or status)"
        ;;
esac
