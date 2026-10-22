#!/usr/bin/env bash
# Single-command four-node TP=4 serving launcher (skeleton).
#
# This is the host-side command builder and process launcher for the four-rank serving topology:
# one sparkflash serve process per node, rank 0 owning HTTP, the tokenizer and the control
# channel, ranks 1-3 running the resident forward. It does not make a measurement and does not
# claim one: the print mode is offline, and a launch only starts what the operator staged.
#
# The binary must be built with the tp4-serve feature:
#   cargo build --release --features tp4-serve -p sparkflash-cli
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
# Options:
#   SPARKFLASH_BIN         binary on every node (default <repo>/target/release/sparkflash)
#   SPARKFLASH_HTTP_BIND   rank-0 HTTP bind (default 0.0.0.0:8000)
#   SPARKFLASH_TP4_BASE_PORT     collective base port (default 29601; rank R gets base+R)
#   SPARKFLASH_TP4_CONTROL_PORT  rank-0 control port (default base + 100)
#   SPARKFLASH_LOG_ROOT    remote log directory (default <repo>/target/serve-tp4)
#   SPARKFLASH_DRAIN_DELAY_MS    rank-0 pre-stop delay (default 5000)
#   SPARKFLASH_DRAIN_TIMEOUT_MS  rank-0 drain deadline (default 30000)
#   SPARKFLASH_DEVICE_TOKEN_DIR  token window (or live window parent) served by rank 0
#   SPARKFLASH_DEVICE_TOKEN_LIVE=1        follow that directory as a live window directory (W5)
#   SPARKFLASH_DEVICE_TOKEN_DERIVED_CLAIM=1  accept a window without the device's claim assertion
#   SSH_USER, SSH_OPTS     ssh user and options (defaults: current user, BatchMode)
#   SPARKFLASH_DRY_RUN=1   same as print, for scripts that want one code path
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-print}"
[ "${SPARKFLASH_DRY_RUN:-0}" = "1" ] && MODE=print
SSH_USER="${SSH_USER:-${USER:-allenh1}}"
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

read -r -a NODES <<< "$NODES_RAW"
die() { printf 'serve-four-node: %s\n' "$*" >&2; exit 1; }

[ "${#NODES[@]}" -eq 4 ] || die "SPARK_NODES needs exactly 4 addresses in rank order, got ${#NODES[@]}"
[ -n "$CHECKPOINT" ] || die "SPARKFLASH_CHECKPOINT is required"
[ -n "$TOKENIZER" ] || die "SPARKFLASH_TOKENIZER is required"
PEERS="$(IFS=,; printf '%s' "${NODES[*]}")"

# The environment every rank starts with. The drain knobs only matter on rank 0, which owns the
# HTTP surface, but exporting them uniformly keeps the four command lines identical apart from the
# rank-specific flags.
rank_env() {
    printf 'SPARKFLASH_SERVE_DRAIN_DELAY_MS=%s SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS=%s' \
        "$DRAIN_DELAY_MS" "$DRAIN_TIMEOUT_MS"
}

# The one command line per rank. Rank 0 binds the public HTTP address; ranks 1-3 keep a loopback
# bind because they never serve HTTP. The device token window flags are rank-0 only: workers never
# detokenize, so a window on a worker would be an unused second copy.
rank_command() {
    local rank="$1"
    printf '%s serve --checkpoint %s --tokenizer %s' "$BIN" "$CHECKPOINT" "$TOKENIZER"
    printf ' --tp4-rank %s --tp4-world-size 4 --tp4-peers %s' "$rank" "$PEERS"
    printf ' --tp4-base-port %s --tp4-control-port %s' "$BASE_PORT" "$CONTROL_PORT"
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

remote() {
    local rank="$1"; shift
    ssh -n ${SSH_OPTS} "$SSH_USER${NODES[$rank]}" "$*"
}

print_plan() {
    printf 'serve-four-node: run %s\n' "$RUN_ID"
    printf 'serve-four-node: nodes %s\n' "$NODES_RAW"
    printf 'serve-four-node: peers %s\n' "$PEERS"
    local rank
    for rank in 0 1 2 3; do
        printf 'serve-four-node: rank %s -> %s\n' "$rank" "${NODES[$rank]}"
        printf '  ssh -n %s@%s "mkdir -p %s/%s && cd %s && nohup env %s %s > %s/%s/rank%s.log 2>&1 &"\n' \
            "$SSH_USER" "${NODES[$rank]}" "$LOG_ROOT" "$RUN_ID" "$REPO_ROOT" \
            "$(rank_env)" "$(rank_command "$rank")" "$LOG_ROOT" "$RUN_ID" "$rank"
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
            remote "$rank" "mkdir -p '$LOG_ROOT/$RUN_ID' && cd '$REPO_ROOT' && nohup env $(rank_env) $(rank_command "$rank") > '$LOG_ROOT/$RUN_ID/rank$rank.log' 2>&1 & echo $!" \
                | sed "s/^/rank $rank pid /"
        done
        printf 'serve-four-node: launched; commands in %s\n' "$local_plan"
        printf 'serve-four-node: rank 0 health endpoint http://%s/health\n' "$HTTP_BIND"
        ;;
    stop)
        for rank in 0 1 2 3; do
            # SIGTERM is the drain signal: rank 0 turns /readyz 503, finishes in-flight requests and
            # exits 0. The pattern is the exact rank flag, so no other serve process is matched.
            remote "$rank" "pkill -TERM -f 'sparkflash serve --checkpoint .* --tp4-rank $rank ' >/dev/null 2>&1 || true" || true
            printf 'serve-four-node: rank %s SIGTERM sent\n' "$rank"
        done
        printf 'serve-four-node: in-flight requests drain for up to %s ms\n' "$DRAIN_TIMEOUT_MS"
        ;;
    status)
        for rank in 0 1 2 3; do
            if remote "$rank" "pgrep -f 'tp4-rank $rank ' >/dev/null" >/dev/null 2>&1; then
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
