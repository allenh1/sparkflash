#!/usr/bin/env bash
# Fabric measurement of the layer's all-reduce arms: host spawn/join reduce vs a dlopen'd vendored
# NCCL, same nodes, same payloads, same exchange order.
#
#   scripts/gpu-reduce-bench.sh fetch|build|stage|preflight|run|collect|check
#
# This runner does not touch any node configuration and never runs while the TP4 vLLM deployment is
# up: preflight refuses on a GPU process or the :8000 listener unless SPARK_WSB_ALLOW_BUSY=1 is set
# inside a coordinated window, and a run beside a serving deployment is labelled, never quoted.
#
# The report contract is per-call microseconds on the fabric at the layer's payload sizes:
#   10,240 B per rank is the served decode payload of both cross-rank collectives
#   (attention-output delta and MoE routed-row delta, 5,120 bf16 elements each);
#   20,480 B is the two-token shape, reported as a scaling row.
set -euo pipefail
ROOT=$(cd "${0%/*}/.." && pwd)
TARGET=aarch64-unknown-linux-gnu
BIN=${CARGO_TARGET_DIR:-$ROOT/target}/$TARGET/release/gpu-reduce-bench
DEPS=${SPARK_NCCL_DEPS:-$ROOT/target/deps/nccl}
LIB=$DEPS/libnccl.so.2
SSH_USER=${SSH_USER:-allenh1}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
NODES=(${SPARK_NODES:-192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68})
NAMES=(node0 node1 node3 node2)
BASE_PORT=${SPARK_REDUCE_PORT:-19701}
CONTROL_PORT=${SPARK_REDUCE_CONTROL_PORT:-19755}
ADDRS=""
for i in 0 1 2 3; do ADDRS="$ADDRS${ADDRS:+,}${NODES[$i]}:$BASE_PORT"; done
CONTROL="${NODES[0]}:$CONTROL_PORT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REMOTE_DIR=${SPARK_REDUCE_DIR:-/home/$SSH_USER/gpu-reduce/$STAMP}
LOCAL_DIR=$ROOT/target/gpu-reduce/$STAMP
SAMPLES=${SPARK_REDUCE_SAMPLES:-200}
WARMUP=${SPARK_REDUCE_WARMUP:-20}
MODE=${SPARK_REDUCE_MODE:-batch}
BATCH=${SPARK_REDUCE_BATCH:-40}
BYTES=${SPARK_REDUCE_BYTES:-10240,20480}
RUN_TIMEOUT_S=${SPARK_REDUCE_TIMEOUT:-600}
ALLOW_BUSY=${SPARK_WSB_ALLOW_BUSY:-0}

say() { printf '%s\n' "$*" >&2; }
die() { say "gpu-reduce-bench: $*"; exit 1; }
remote() { local rank=$1; shift; ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"; }
remote_bg() { local rank=$1; shift; ssh -f -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"; }

fetch() {
    bash "$ROOT/scripts/fetch-nccl.sh" "$DEPS"
}

build() {
    local env_prefix=(
        CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
        "CC_${TARGET//-/_}=${CC_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cc}"
        "CXX_${TARGET//-/_}=${CXX_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cxx}"
        "AR_${TARGET//-/_}=${AR_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-ar}"
        "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-/home/user/opt/bin/sparkflash-aarch64-cc}"
    )
    say "cross-building gpu-reduce-bench for $TARGET"
    ( cd "$ROOT" && env "${env_prefix[@]}" "${CARGO:-$HOME/.cargo/bin/cargo}" +1.98.1 build --release \
        -p sparkflash-comm --target "$TARGET" --bin gpu-reduce-bench )
    [ -x "$BIN" ] || die "build finished without $BIN"
    say "bench: $BIN sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
}

stage() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    [ -s "$LIB" ] || die "no vendored libnccl at $LIB (run: $0 fetch)"
    local r ip
    for r in 0 1 2 3; do
        ip=${NODES[$r]}
        remote "$r" "mkdir -p '$REMOTE_DIR/bin' '$REMOTE_DIR/lib' && chmod 700 '$REMOTE_DIR'"
        scp -q "$BIN" "$SSH_USER@${ip}:$REMOTE_DIR/bin/"
        scp -q "$LIB" "$SSH_USER@${ip}:$REMOTE_DIR/lib/libnccl.so.2"
        say "rank $r ${NAMES[$r]}: $(remote "$r" "sha256sum '$REMOTE_DIR/bin/gpu-reduce-bench' '$REMOTE_DIR/lib/libnccl.so.2' | tr '\n' ' '")"
    done
    say "staged into $REMOTE_DIR (compare the two shas with: sha256sum $BIN $LIB)"
}

preflight() {
    local r out busy=0
    say "read-only preflight: no node configuration is changed"
    for r in 0 1 2 3; do
        out=$(remote "$r" 'printf "host=%s gpu_apps=%s port8000=%s mtu=%s peermem=%s\n" "$(hostname)" "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed "/^[[:space:]]*$/d" | wc -l)" "$(ss -ltnH "sport = :8000" 2>/dev/null | wc -l)" "$(cat /sys/class/net/enp1s0f1np1/mtu 2>/dev/null || echo absent)/$(cat /sys/class/net/enP2p1s0f1np1/mtu 2>/dev/null || echo absent)" "$([ -d /sys/module/nvidia_peermem ] && echo loaded || echo absent)"' 2>&1) || { say "rank $r unreachable: $out"; busy=1; continue; }
        say "rank $r ${NAMES[$r]}: $out"
        printf '%s\n' "$out" | grep -q 'gpu_apps=0' || busy=1
        printf '%s\n' "$out" | grep -q 'port8000=0' || busy=1
    done
    if [ "$busy" != "0" ] && [ "$ALLOW_BUSY" != "1" ]; then
        die "refusing to benchmark: a GPU process or the :8000 listener is present. A latency measured next to the serving deployment is not a fabric number; wait for a free slot or set SPARK_WSB_ALLOW_BUSY=1 inside a coordinated window."
    fi
    [ "$busy" = "0" ] || say "WARNING: SPARK_WSB_ALLOW_BUSY=1: labelled busy; not a clean fabric number"
    say "preflight complete"
}

run() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    mkdir -p "$LOCAL_DIR"
    preflight
    say "launching four ranks: payloads $BYTES, mode $MODE batch $BATCH, samples $SAMPLES warmup $WARMUP"
    local r
    for r in 0 1 2 3; do
        remote_bg "$r" "cd '$REMOTE_DIR' && nohup env LD_LIBRARY_PATH='$REMOTE_DIR/lib' './bin/gpu-reduce-bench' --rank $r --world 4 --addrs '$ADDRS' --control '$CONTROL' --bytes $BYTES --samples $SAMPLES --warmup $WARMUP --mode $MODE --batch $BATCH --json > 'bench-rank$r.json' 2> 'bench-rank$r.err' < /dev/null & echo \$! > 'bench-rank$r.pid'"
    done
    local waited=0 ready=0
    while [ "$waited" -lt "$RUN_TIMEOUT_S" ]; do
        ready=0
        for r in 0 1 2 3; do
            remote "$r" "test -s '$REMOTE_DIR/bench-rank$r.json'" 2>/dev/null && ready=$((ready + 1))
        done
        [ "$ready" -eq 4 ] && break
        sleep 5
        waited=$((waited + 5))
    done
    [ "$ready" -eq 4 ] || say "TIMEOUT: only $ready of 4 ranks wrote a JSON file in ${RUN_TIMEOUT_S}s"
    for r in 0 1 2 3; do
        say "--- rank $r stderr (tail)"
        remote "$r" "tail -n 4 '$REMOTE_DIR/bench-rank$r.err' 2>/dev/null || true"
    done
    collect
}

collect() {
    mkdir -p "$LOCAL_DIR"
    local r ip
    for r in 0 1 2 3; do
        ip=${NODES[$r]}
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.json" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no JSON at $REMOTE_DIR"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.err" "$LOCAL_DIR/" 2>/dev/null || true
    done
    say "collected per-rank JSON into $LOCAL_DIR"
    for r in 0 1 2 3; do
        [ -s "$LOCAL_DIR/bench-rank$r.json" ] && say "--- rank $r" && cat "$LOCAL_DIR/bench-rank$r.json"
    done
}

case "${1:-run}" in
    fetch) fetch ;;
    build) build ;;
    stage) stage ;;
    preflight) preflight ;;
    run) run ;;
    collect) collect ;;
    check)
        say "check: loading libnccl on rank 0 (no GPU work)"
        remote 0 "LD_LIBRARY_PATH='$REMOTE_DIR/lib' '$REMOTE_DIR/bin/gpu-reduce-bench' --check-nccl" ;;
    *) die "usage: $0 fetch|build|stage|preflight|run|collect|check" ;;
esac
