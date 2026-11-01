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

# Single-node TP=1 serving launcher for model packs that bring their own arm.
#
# Qwen3.8-Flash-Next is served directly from its Hugging Face checkpoint: there is no per-rank
# conversion, no NCCL, and no SSH. This is the local counterpart of serve-four-node.sh.
#
# Usage:
#   scripts/serve-single-node.sh print     print the command (default; no network access)
#   scripts/serve-single-node.sh launch    start the server detached
#   scripts/serve-single-node.sh stop      send SIGTERM to the server
#   scripts/serve-single-node.sh status    report whether the server process is running
#
# Environment:
#   SPARKFLASH_MODEL        pack id, e.g. qwen3.8-flash-next-nvfp4 (required)
#   SPARKFLASH_CHECKPOINT   checkpoint directory (required to launch)
#   SPARKFLASH_TOKENIZER    tokenizer.json path (required to launch)
#   SPARKFLASH_CONTEXT      context window in tokens (default: the pack's default)
#   SPARKFLASH_HTTP_BIND    listen address (default 127.0.0.1:8000)
#   SPARKFLASH_BIN          binary (default <repo>/target/release/sparkflash)
#   SPARKFLASH_LOG_ROOT     log and pid directory (default <repo>/target/serve-single)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-print}"
BIN="${SPARKFLASH_BIN:-$REPO_ROOT/target/release/sparkflash}"
MODEL="${SPARKFLASH_MODEL:-}"
CHECKPOINT="${SPARKFLASH_CHECKPOINT:-}"
TOKENIZER="${SPARKFLASH_TOKENIZER:-}"
CONTEXT="${SPARKFLASH_CONTEXT:-}"
HTTP_BIND="${SPARKFLASH_HTTP_BIND:-127.0.0.1:8000}"
LOG_ROOT="${SPARKFLASH_LOG_ROOT:-$REPO_ROOT/target/serve-single}"
STATE_DIR="$LOG_ROOT/single"
PIDFILE="$STATE_DIR/serve.pid"
LOG="$STATE_DIR/rank0.log"

die() { printf 'serve-single-node: %s\n' "$*" >&2; exit 1; }

running_pid() {
    [ -s "$PIDFILE" ] || return 1
    local pid
    pid="$(cat "$PIDFILE")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
}

command_line() {
    printf '%s serve --model %s --checkpoint %s --tokenizer %s --bind %s' \
        "$BIN" "$MODEL" "$CHECKPOINT" "$TOKENIZER" "$HTTP_BIND"
    [ -z "$CONTEXT" ] || printf ' --context %s' "$CONTEXT"
}

case "$MODE" in
    print)
        [ -n "$MODEL" ] || die "SPARKFLASH_MODEL is required"
        [ -n "$CHECKPOINT" ] || die "SPARKFLASH_CHECKPOINT is required"
        [ -n "$TOKENIZER" ] || die "SPARKFLASH_TOKENIZER is required"
        printf 'serve-single-node: %s\n' "$(command_line)"
        ;;
    launch)
        [ -n "$MODEL" ] || die "SPARKFLASH_MODEL is required"
        [ -n "$CHECKPOINT" ] || die "SPARKFLASH_CHECKPOINT is required"
        [ -n "$TOKENIZER" ] || die "SPARKFLASH_TOKENIZER is required"
        [ -x "$BIN" ] || die "local binary $BIN is not executable; run scripts/setup.sh first"
        if pid="$(running_pid)"; then
            printf 'serve-single-node: already running (pid %s)\n' "$pid"
            exit 0
        fi
        mkdir -p "$STATE_DIR"
        cd "$REPO_ROOT"
        args=(serve --model "$MODEL" --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" --bind "$HTTP_BIND")
        [ -z "$CONTEXT" ] || args+=(--context "$CONTEXT")
        # Detached from the web job, so the job can finish and the server keeps serving.
        nohup "$BIN" "${args[@]}" >"$LOG" 2>&1 </dev/null &
        pid=$!
        printf '%s\n' "$pid" >"$PIDFILE"
        sleep 0.5
        if ! kill -0 "$pid" 2>/dev/null; then
            printf 'serve-single-node: the server exited during start-up; log %s:\n' "$LOG" >&2
            tail -n 40 "$LOG" >&2 || true
            rm -f "$PIDFILE"
            exit 1
        fi
        printf 'serve-single-node: started pid %s on %s\n' "$pid" "$HTTP_BIND"
        printf 'serve-single-node: log %s\n' "$LOG"
        ;;
    stop)
        if pid="$(running_pid)"; then
            kill -TERM "$pid" 2>/dev/null || true
            for _ in $(seq 1 100); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
            rm -f "$PIDFILE"
            printf 'serve-single-node: stopped pid %s\n' "$pid"
        else
            rm -f "$PIDFILE"
            printf 'serve-single-node: not running\n'
        fi
        ;;
    status)
        if pid="$(running_pid)"; then
            printf 'serve-single-node: running pid %s on %s\n' "$pid" "$HTTP_BIND"
        else
            printf 'serve-single-node: not running\n'
        fi
        ;;
    *)
        die "unknown mode '$MODE' (expected print, launch, stop or status)"
        ;;
esac
