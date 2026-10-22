#!/usr/bin/env bash
# RDMA bring-up runner for the four-node DGX Spark fabric (task-35).
#
# This is the executable form of docs/archive/build/rdma-bringup.md. It does not invent a measurement:
# every number a run records comes from the existing hardware gates (crates/sparkflash-comm/tests/
# hardware_gates.rs) and from rocenante-probe (crates/sparkflash-comm/src/bin/rocenante-probe.rs).
# The runner only stages, launches, collects, and stops those binaries.
#
# Rank table used by this runner (the table the 2026-10-20 run used, and the one task-53/55 should
# use for this cluster; see the doc for why it differs from the design doc's rail-1 default):
#   rank 0 node0 192.0.2.80
#   rank 1 node1 192.0.2.88
#   rank 2 node3 192.0.2.59
#   rank 3 node2 192.0.2.68
#
# Usage:
#   scripts/rdma-bringup.sh build                 cross-build the gate and probe binaries
#   scripts/rdma-bringup.sh linkstate             record per-node rail state and rate
#   scripts/rdma-bringup.sh stage                 copy the two binaries to every node
#   scripts/rdma-bringup.sh gates [TEST]          run all three gates, or one exact test
#   scripts/rdma-bringup.sh probe [BYTES]         run the probe on all four ranks
#   scripts/rdma-bringup.sh collect               copy logs and JSON back here
#   scripts/rdma-bringup.sh cleanup               stop any rank process, confirm port 18515 is free
#   scripts/rdma-bringup.sh all                   linkstate, stage, gates, probe, collect, cleanup
#
# Environment:
#   EVIDENCE_ROOT  remote evidence root (default /home/user/phase2-evidence/window-2026-10-18)
#   RUN_ID         run directory name (default rdma-t35-<UTC timestamp>)
#   LOCAL_DIR      local copy destination (default <repo>/target/rdma-bringup/<RUN_ID>)
#   SPARK_NODES    four control-plane addresses in rank order (overrides the table above)
#   SPARK_COMM_PEERS  peer table given to every rank (default: the table above)
#   GATE_BIN, PROBE_BIN  local binaries (defaults: newest aarch64 release artifacts)
#   SPARK_COMM_RAIL_COUNT  pass 1 for the deployed single-rail mode (default: backend default, 2)
#   SAMPLES, WARMUP, TIMEOUT_MS  probe and gate timing knobs
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/home/user/phase2-evidence/window-2026-10-18}"
RUN_ID="${RUN_ID:-rdma-t35-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_DIR="$EVIDENCE_ROOT/$RUN_ID"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/target/rdma-bringup/$RUN_ID}"
SSH_USER="${SSH_USER:-allenh1}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
NODES=(${SPARK_NODES:-192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68})
NAMES=(node0 node1 node3 node2)
PEERS="${SPARK_COMM_PEERS:-192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68}"
CONTROL_PORT="${SPARK_COMM_CONTROL_PORT:-18515}"
TIMEOUT_MS="${SPARK_COMM_TIMEOUT_MS:-60000}"
SAMPLES="${SAMPLES:-200}"
WARMUP="${WARMUP:-20}"
PROBE_BYTES="${PROBE_BYTES:-10240}"
BIN_DIR="$REMOTE_DIR/bin"

default_gate() {
    ls -t "$REPO_ROOT"/target/aarch64-unknown-linux-gnu/release/deps/hardware_gates-* 2>/dev/null \
        | grep -v '\.d$' | head -1
}
GATE_BIN="${GATE_BIN:-$(default_gate || true)}"
PROBE_BIN="${PROBE_BIN:-$REPO_ROOT/target/aarch64-unknown-linux-gnu/release/rocenante-probe}"

say() { printf '%s\n' "$*"; }
die() { printf 'rdma-bringup: %s\n' "$*" >&2; exit 1; }

need_binaries() {
    [ -n "$GATE_BIN" ] && [ -x "$GATE_BIN" ] || die "no executable gate binary (set GATE_BIN; run: $0 build)"
    [ -x "$PROBE_BIN" ] || die "no executable probe binary (set PROBE_BIN; run: $0 build)"
}

remote() {
    local rank="$1"; shift
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

remote_out() {
    local rank="$1"; shift
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

# Detached launch: ssh -f returns as soon as the remote shell has started the job,
# so four ranks are launched within the rendezvous timeout instead of serially.
remote_bg() {
    local rank="$1"; shift
    ssh -f -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

build() {
    local target=aarch64-unknown-linux-gnu
    local env_prefix=(
        CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
        SPARKFLASH_RDMA_INCLUDE="${SPARKFLASH_RDMA_INCLUDE:-/home/user/opt/aarch64-rdma/include}"
        "CC_${target//-/_}=${CC_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cc}"
        "CXX_${target//-/_}=${CXX_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cxx}"
        "AR_${target//-/_}=${AR_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-ar}"
        "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-/home/user/opt/bin/sparkflash-aarch64-cc}"
    )
    local cargo="${CARGO:-$HOME/.cargo/bin/cargo}"
    say "building $target gate and probe (rdma feature, system libibverbs shim)"
    ( cd "$REPO_ROOT" && env "${env_prefix[@]}" "$cargo" +1.98.1 build -p sparkflash-comm \
        --features rdma --release --target "$target" --bin rocenante-probe )
    ( cd "$REPO_ROOT" && env "${env_prefix[@]}" "$cargo" +1.98.1 test -p sparkflash-comm \
        --features rdma --release --target "$target" --test hardware_gates --no-run )
    local tdir="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
    GATE_BIN="$(ls -t "$tdir"/"$target"/release/deps/hardware_gates-* | grep -v '\.d$' | head -1)"
    PROBE_BIN="$tdir/$target/release/rocenante-probe"
    say "gate:  $GATE_BIN ($(sha256sum "$GATE_BIN" | cut -d' ' -f1))"
    say "probe: $PROBE_BIN ($(sha256sum "$PROBE_BIN" | cut -d' ' -f1))"
    say "the build log must NOT contain 'rdma feature built without <infiniband/verbs.h>'"
}

linkstate() {
    mkdir -p "$LOCAL_DIR"
    local r ip out
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        out="$LOCAL_DIR/linkstate-rank$r.txt"
        say "link state rank $r ($ip, ${NAMES[$r]})"
        remote_out "$r" '
            echo "node=$(hostname) address='"$ip"'"
            echo "=== devices"; ibv_devinfo -l 2>&1
            for d in rocep1s0f1 roceP2p1s0f1 rocep1s0f0 roceP2p1s0f0; do
                printf "%s rate=" "$d"; cat /sys/class/infiniband/$d/ports/1/rate 2>&1
                printf "%s state=" "$d"; cat /sys/class/infiniband/$d/ports/1/state 2>&1
            done
            echo "=== rdma link"; rdma link show 2>&1
            echo "=== addresses"; ip -br addr show 2>&1
            echo "=== libibverbs"; ldconfig -p 2>/dev/null | grep -m2 libibverbs
            echo "=== uverbs"; ls -l /dev/infiniband/uverbs* 2>&1
            echo "=== port 18515"; ss -ltn 2>/dev/null | grep -E ":'"$CONTROL_PORT"'\b" || echo free
        ' > "$out" 2>&1 || true
    done
    say "wrote $LOCAL_DIR/linkstate-rank*.txt"
}

stage() {
    need_binaries
    local gate_name probe_name r ip
    gate_name="$(basename "$GATE_BIN")"
    probe_name="$(basename "$PROBE_BIN")"
    say "staging into $REMOTE_DIR/bin"
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        remote "$r" "mkdir -p '$BIN_DIR' && chmod 700 '$REMOTE_DIR'"
        scp -q "$GATE_BIN" "$PROBE_BIN" "$SSH_USER@${ip}:$BIN_DIR/"
        remote "$r" "chmod +x '$BIN_DIR/$gate_name' '$BIN_DIR/$probe_name'; sha256sum '$BIN_DIR/$gate_name' '$BIN_DIR/$probe_name'"
    done
    say "staged; compare the printed sha256 with: sha256sum $GATE_BIN $PROBE_BIN"
}

launch_gate() {
    local r="$1" test_name="${2:-}"
    local ip gate_name log pidfile rail_env=""
    ip="${NODES[$r]}"
    gate_name="$(basename "$GATE_BIN")"
    log="gates-rank$r.log"
    pidfile="gates-rank$r.pid"
    if [ -n "${SPARK_COMM_RAIL_COUNT:-}" ]; then rail_env="SPARK_COMM_RAIL_COUNT=$SPARK_COMM_RAIL_COUNT"; fi
    local filter=""
    if [ -n "$test_name" ]; then filter="--exact $test_name"; fi
    remote_bg "$r" "cd '$REMOTE_DIR' && nohup env SPARK_COMM_HW_PERMISSION=1 SPARK_COMM_RANK=$r SPARK_COMM_WORLD=4 SPARK_COMM_PEERS='$PEERS' SPARK_COMM_CONTROL_PORT=$CONTROL_PORT SPARK_COMM_TIMEOUT_MS=$TIMEOUT_MS $rail_env './bin/$gate_name' --ignored --test-threads=1 --nocapture $filter > '$log' 2>&1 < /dev/null & echo \$! > '$pidfile'"
}

launch_probe() {
    local r="$1" bytes="${2:-$PROBE_BYTES}"
    local ip probe_name log pidfile rail_env=""
    ip="${NODES[$r]}"
    probe_name="$(basename "$PROBE_BIN")"
    log="probe-rank$r.json"
    pidfile="probe-rank$r.pid"
    if [ -n "${SPARK_COMM_RAIL_COUNT:-}" ]; then rail_env="SPARK_COMM_RAIL_COUNT=$SPARK_COMM_RAIL_COUNT"; fi
    remote_bg "$r" "cd '$REMOTE_DIR' && nohup env SPARK_COMM_HW_PERMISSION=1 SPARK_COMM_CONTROL_PORT=$CONTROL_PORT SPARK_COMM_TIMEOUT_MS=$TIMEOUT_MS $rail_env './bin/$probe_name' --rank $r --world 4 --peer '$PEERS' --bytes $bytes --samples $SAMPLES --warmup $WARMUP --timeout-ms $TIMEOUT_MS --json > '$log' 2> 'probe-rank$r.err' < /dev/null & echo \$! > '$pidfile'"
}

wait_for() {
    local pattern="$1" timeout="$2" what="$3" waited=0 r done
    while [ "$waited" -lt "$timeout" ]; do
        done=1
        for r in 0 1 2 3; do
            if ! remote_out "$r" "test -n \"\$(grep -l -m1 '$pattern' '$REMOTE_DIR/$what-rank$r.'* 2>/dev/null)\"" >/dev/null 2>&1; then
                done=0
            fi
        done
        [ "$done" -eq 1 ] && return 0
        sleep 2; waited=$((waited + 2))
    done
    return 1
}

gates() {
    need_binaries
    local test_name="${1:-}" waited
    say "launching the four gate processes (control port $CONTROL_PORT, peers $PEERS)"
    for r in 0 1 2 3; do launch_gate "$r" "$test_name"; done
    say "waiting for four 'test result:' lines (timeout 300 s)"
    waited=0
    while [ "$waited" -lt 300 ]; do
        local n=0
        for r in 0 1 2 3; do
            if remote_out "$r" "grep -q 'test result:' '$REMOTE_DIR/gates-rank$r.log' 2>/dev/null"; then n=$((n + 1)); fi
        done
        [ "$n" -eq 4 ] && break
        [ "$n" -gt 0 ] && [ "$waited" -ge 60 ] && [ "$n" -lt 4 ] && say "only $n/4 ranks finished after ${waited}s"
        sleep 2; waited=$((waited + 2))
    done
    for r in 0 1 2 3; do
        say "--- rank $r (${NAMES[$r]})"
        remote_out "$r" "grep -E 'test result:|replay .* failed|one-shot all-reduce|dual-rail run' '$REMOTE_DIR/gates-rank$r.log' 2>/dev/null || echo 'no result: see the log'"
    done
    if [ "$waited" -ge 300 ]; then
        say "timed out; the ranks were stopped by cleanup"
    fi
}

probe() {
    need_binaries
    local bytes="${1:-$PROBE_BYTES}" waited=0 n
    say "launching the four probe processes (bytes $bytes, samples $SAMPLES, warmup $WARMUP)"
    for r in 0 1 2 3; do launch_probe "$r" "$bytes"; done
    while [ "$waited" -lt 300 ]; do
        n=0
        for r in 0 1 2 3; do
            if remote_out "$r" "test -s '$REMOTE_DIR/probe-rank$r.json' 2>/dev/null" >/dev/null 2>&1; then n=$((n + 1)); fi
        done
        [ "$n" -eq 4 ] && break
        sleep 2; waited=$((waited + 2))
    done
    for r in 0 1 2 3; do
        say "--- rank $r (${NAMES[$r]})"
        remote_out "$r" "cat '$REMOTE_DIR/probe-rank$r.json' 2>/dev/null; test -s '$REMOTE_DIR/probe-rank$r.err' && { echo; tail -2 '$REMOTE_DIR/probe-rank$r.err'; }" || true
    done
}

collect() {
    mkdir -p "$LOCAL_DIR"
    local r ip
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        mkdir -p "$LOCAL_DIR/rank$r"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/*" "$LOCAL_DIR/rank$r/" 2>/dev/null || true
    done
    say "copied per-rank logs and JSON into $LOCAL_DIR/rank0..3"
}

cleanup() {
    local r ip
    say "stopping every rank process and checking port $CONTROL_PORT"
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        remote "$r" "for f in '$REMOTE_DIR'/*.pid; do [ -f \"\$f\" ] || continue; p=\$(cat \"\$f\"); kill \"\$p\" 2>/dev/null || true; done; pkill -f '[b]in/hardware_gates-' 2>/dev/null || true; pkill -f '[b]in/rocenante-probe' 2>/dev/null || true; sleep 1; ss -ltn 2>/dev/null | grep -E ':$CONTROL_PORT\b' >/dev/null && echo 'PORT_STILL_LISTENING' || echo 'port free'"
    done
}

usage() { sed -n '1,35p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"
[ -n "$cmd" ] || { usage; exit 2; }
shift || true
case "$cmd" in
    build) build ;;
    linkstate) linkstate ;;
    stage) stage ;;
    gates) gates "${1:-}" ;;
    probe) probe "${1:-}" ;;
    collect) collect ;;
    cleanup) cleanup ;;
    all) linkstate; stage; gates; probe; collect; cleanup ;;
    *) usage; exit 2 ;;
esac
