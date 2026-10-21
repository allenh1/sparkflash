#!/usr/bin/env bash
# WS-B four-rank all-reduce microbenchmark runner (task-02 / WS-B, 2026-10-21).
#
# This is the executable form of docs/build/ws-b-collectives-20261021.md section 6. It stages and
# launches the existing bench binary (crates/sparkflash-comm/src/bin/ws-b-allreduce-bench.rs) on
# the four GB10 nodes; the binary owns every number, the runner owns nothing but files and
# processes. No number is available until 'run' completes on the nodes, and no runner path may
# invent one.
#
# The live TP4 vLLM deployment (http://192.0.2.80:8000, ~111/121 GB per node) is sharing this
# fabric and also serves this session's agent harness. This runner NEVER stops, restarts or
# reconfigures it, and it refuses to launch a benchmark while a GPU process or the :8000 listener
# is present (override only for a deliberately supervised, coordinated run with
# SPARK_WSB_ALLOW_BUSY=1). A latency measured next to a serving deployment is not a fabric number;
# it is a warning that the slot was not free.
#
# Operational blocker (Lead's report, not measured here): a "free window" is not "vLLM is stopped",
# because stopping it cuts the model backend of the sessions coordinating the run. It needs explicit
# human authorisation plus a restore path: docker stop vllm_node on all four nodes, restore with
# "python3 run-recipe.py deepseek-v4.1-flash-1017-heretic" on .80, ~13 min to health. Confirm the
# container name, recipe path and health check against the serving owner's runbook before use.
# This refusal is why no fabric number exists today - see docs/build/ws-b-infra-tickets-20261021.md 3.1.
#
# Usage:
#   scripts/ws-b-allreduce-bench.sh preflight    read-only busy/MTU/peermem/port check on all four nodes
#   scripts/ws-b-allreduce-bench.sh build        cross-build the aarch64 bench binary (rdma feature)
#   scripts/ws-b-allreduce-bench.sh selftest     run the loopback harness on this host (fabric=false)
#   scripts/ws-b-allreduce-bench.sh stage        copy the binary to every node under the evidence root
#   scripts/ws-b-allreduce-bench.sh run          preflight, then launch all four ranks and wait
#   scripts/ws-b-allreduce-bench.sh collect      copy per-rank JSON and stderr back here
#   scripts/ws-b-allreduce-bench.sh cleanup      stop only ws-b-allreduce-bench processes, check the port
#   scripts/ws-b-allreduce-bench.sh all          preflight, build, stage, run, collect, cleanup
#
# Environment:
#   EVIDENCE_ROOT   remote evidence root (default /home/user/phase2-evidence/ws-b-20261021)
#   RUN_ID          run directory name (default ws-b-allreduce-<UTC timestamp>)
#   LOCAL_DIR       local destination (default <repo>/target/ws-b-allreduce/<RUN_ID>)
#   SPARK_NODES     four control-plane addresses in rank order (overrides the table below)
#   SPARK_COMM_PEERS  peer table given to every rank (default: the table below)
#   SPARK_COMM_CONTROL_PORT  default 18516 (deliberately not the probe/gate 18515)
#   SAMPLES, WARMUP  timed/decode samples and warmup exchanges (default 200 / 20)
#   GATE_US         p50 gate in microseconds (default 25)
#   RUN_TIMEOUT_S   collection deadline (default 300)
#   SPARK_WSB_ALLOW_BUSY  1 permits a run while the nodes are busy (only with a coordinated window)
#
# Rank table: the table the 2026-10-20 bring-up used and the one the probe runner records:
#   rank 0 node0 192.0.2.80
#   rank 1 node1 192.0.2.88
#   rank 2 node3 192.0.2.59
#   rank 3 node2 192.0.2.68
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/home/user/phase2-evidence/ws-b-20261021}"
RUN_ID="${RUN_ID:-ws-b-allreduce-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_DIR="$EVIDENCE_ROOT/$RUN_ID"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/target/ws-b-allreduce/$RUN_ID}"
SSH_USER="${SSH_USER:-allenh1}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
NODES=(${SPARK_NODES:-192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68})
NAMES=(node0 node1 node3 node2)
PEERS="${SPARK_COMM_PEERS:-192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68}"
CONTROL_PORT="${SPARK_COMM_CONTROL_PORT:-18516}"
TIMEOUT_MS="${SPARK_COMM_TIMEOUT_MS:-5000}"
SAMPLES="${SAMPLES:-200}"
WARMUP="${WARMUP:-20}"
GATE_US="${GATE_US:-25}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-300}"
ALLOW_BUSY="${SPARK_WSB_ALLOW_BUSY:-0}"
TARGET=aarch64-unknown-linux-gnu
BIN_DIR="$REMOTE_DIR/bin"
BIN="$REPO_ROOT/target/$TARGET/release/ws-b-allreduce-bench"
REMOTE_BIN="$BIN_DIR/ws-b-allreduce-bench"

say() { printf '%s\n' "$*"; }
die() { printf 'ws-b-allreduce-bench: %s\n' "$*" >&2; exit 1; }

remote() {
    local rank="$1"; shift
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

# Detached launch: ssh -f returns once the remote shell has started the job, so all four ranks
# reach the rendezvous inside the bench deadline instead of being launched serially.
remote_bg() {
    local rank="$1"; shift
    ssh -f -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

# Read-only node survey. The body is a quoted heredoc, so it runs on the node unchanged and this
# repo never writes to a node outside the evidence directory.
remote_survey() {
    local rank="$1"
    # No -n here: the heredoc IS stdin for the remote bash.
    ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" 'bash -s -- '"$CONTROL_PORT" <<'REMOTE'
port="$1"
printf 'host=%s\n' "$(hostname)"
printf 'mtu_rail1=%s\n' "$(cat /sys/class/net/enp1s0f1np1/mtu 2>/dev/null || echo absent)"
printf 'mtu_rail2=%s\n' "$(cat /sys/class/net/enP2p1s0f1np1/mtu 2>/dev/null || echo absent)"
if [ -d /sys/module/nvidia_peermem ]; then printf 'peermem=loaded\n'; else printf 'peermem=absent\n'; fi
gpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
fi
printf 'gpu_compute_apps=%s\n' "$gpu"
printf 'port8000_listeners=%s\n' "$(ss -ltnH 'sport = :8000' 2>/dev/null | wc -l)"
printf 'control_port_listeners=%s\n' "$(ss -ltnH "sport = :$port" 2>/dev/null | wc -l)"
states=""
for d in rocep1s0f1 roceP2p1s0f1; do
    s="$(cat "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null || echo absent)"
    states="$states$d=$s "
done
printf 'rdma_states=%s\n' "$states"
REMOTE
}

preflight() {
    mkdir -p "$LOCAL_DIR"
    local report="$LOCAL_DIR/preflight.txt"
    : > "$report"
    say "read-only preflight: no node configuration is changed and vLLM is never touched"
    local r out busy=0 mtu_bad=0 peermem_absent=0
    for r in 0 1 2 3; do
        if ! out="$(remote_survey "$r" 2>&1)"; then
            printf '=== rank %s %s %s\n%s\n' "$r" "${NODES[$r]}" "${NAMES[$r]}" "$out" >> "$report"
            say "rank $r ${NODES[$r]} unreachable"
            busy=1
            continue
        fi
        printf '=== rank %s %s %s\n%s\n' "$r" "${NODES[$r]}" "${NAMES[$r]}" "$out" >> "$report"
        local gpu port8000 portctl mtu1 mtu2 peermem
        gpu="$(printf '%s\n' "$out" | sed -n 's/^gpu_compute_apps=//p')"
        port8000="$(printf '%s\n' "$out" | sed -n 's/^port8000_listeners=//p')"
        portctl="$(printf '%s\n' "$out" | sed -n 's/^control_port_listeners=//p')"
        mtu1="$(printf '%s\n' "$out" | sed -n 's/^mtu_rail1=//p')"
        mtu2="$(printf '%s\n' "$out" | sed -n 's/^mtu_rail2=//p')"
        peermem="$(printf '%s\n' "$out" | sed -n 's/^peermem=//p')"
        if ! printf '%s\n' "$out" | grep -q '^host='; then
            say "rank $r ${NAMES[$r]}: survey incomplete (no host= line); treating the node as busy"
            busy=1
            continue
        fi
        say "rank $r ${NAMES[$r]}: gpu_apps=$gpu port8000=$port8000 control_port=$portctl mtu=${mtu1}/${mtu2} peermem=$peermem"
        [ "$gpu" != "0" ] && busy=1
        [ "$port8000" != "0" ] && busy=1
        [ "$portctl" != "0" ] && busy=1
        [ "${mtu1:-0}" != "9000" ] && mtu_bad=1
        [ "${mtu2:-0}" != "9000" ] && mtu_bad=1
        [ "$peermem" = "absent" ] && peermem_absent=1
    done
    say "wrote $report"
    if [ "$busy" != "0" ]; then
        if [ "$ALLOW_BUSY" != "1" ]; then
            die "refusing to benchmark: a GPU process, the :8000 listener or another rank is present on the nodes. The live TP4 vLLM deployment holds ~111/121 GB per node and a concurrent run would both bias the latency and perturb serving. Wait for a free slot, or set SPARK_WSB_ALLOW_BUSY=1 only inside a coordinated window."
        fi
        say "WARNING: SPARK_WSB_ALLOW_BUSY=1: running beside a busy node; the resulting latency is NOT a clean fabric number and must be labelled as such"
    fi
    if [ "$mtu_bad" != "0" ]; then
        say "WARNING: fabric MTU is not 9000 on every port (infra ticket B1 still open). The one-shot host path can still run at MTU 1500 (2026-10-20 measured 10 KiB medians 36.0-37.1 us dual rail), but a 25 us p50 pass is not expected before B1."
    fi
    if [ "$peermem_absent" != "0" ]; then
        say "WARNING: nvidia_peermem is not loaded on every node (infra ticket B2 still open). The host-buffer one-shot path does not need GPUDirect, but the device-buffer path and any GPU-buffer RDMA measurement do."
    fi
    say "preflight complete: the live vLLM deployment is untouched"
}

build() {
    local env_prefix=(
        CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
        SPARKFLASH_RDMA_INCLUDE="${SPARKFLASH_RDMA_INCLUDE:-/home/user/opt/aarch64-rdma/include}"
        "CC_${TARGET//-/_}=${CC_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cc}"
        "CXX_${TARGET//-/_}=${CXX_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-cxx}"
        "AR_${TARGET//-/_}=${AR_aarch64_unknown_linux_gnu:-/home/user/opt/bin/sparkflash-aarch64-ar}"
        "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-/home/user/opt/bin/sparkflash-aarch64-cc}"
    )
    local cargo="${CARGO:-$HOME/.cargo/bin/cargo}"
    say "cross-building $BIN (rdma feature, system libibverbs shim)"
    ( cd "$REPO_ROOT" && env "${env_prefix[@]}" "$cargo" +1.98.1 build --release -p sparkflash-comm \
        --features rdma --target "$TARGET" --bin ws-b-allreduce-bench )
    [ -x "$BIN" ] || die "build finished without $BIN"
    say "bench: $BIN ($(sha256sum "$BIN" | cut -d' ' -f1))"
    say "the build log must NOT contain 'rdma feature built without <infiniband/verbs.h>'"
}

selftest() {
    mkdir -p "$LOCAL_DIR"
    say "loopback self-test on this host: real rank threads, real canonical reduction, fabric=false"
    ( cd "$REPO_ROOT" && cargo build -p sparkflash-comm --bin ws-b-allreduce-bench )
    "$REPO_ROOT/target/debug/ws-b-allreduce-bench" --transport loopback \
        --bytes 10240,61440,368640 --samples "${SELFTEST_SAMPLES:-50}" --warmup 5 --json \
        > "$LOCAL_DIR/selftest-loopback.json"
    say "wrote $LOCAL_DIR/selftest-loopback.json (fabric=false, gate_applied=false: this is NOT the WS-B gate)"
}

stage() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    local r ip
    say "staging into $REMOTE_DIR/bin"
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        remote "$r" "mkdir -p '$BIN_DIR' && chmod 700 '$REMOTE_DIR'"
        scp -q "$BIN" "$SSH_USER@${ip}:$BIN_DIR/"
        remote "$r" "chmod +x '$REMOTE_BIN'; sha256sum '$REMOTE_BIN'"
    done
    say "staged; compare the printed sha256 with: sha256sum $BIN"
}

run() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    mkdir -p "$LOCAL_DIR"
    preflight
    say "launching four ranks: control port $CONTROL_PORT, samples $SAMPLES, warmup $WARMUP, gate p50 <= $GATE_US us on decode rows"
    local r ip
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        remote_bg "$r" "cd '$REMOTE_DIR' && nohup env SPARK_COMM_PEERS='$PEERS' SPARK_COMM_CONTROL_PORT=$CONTROL_PORT SPARK_COMM_TIMEOUT_MS=$TIMEOUT_MS './bin/ws-b-allreduce-bench' --transport rdma --world 4 --rank $r --peer '$PEERS' --samples $SAMPLES --warmup $WARMUP --gate-us $GATE_US --timeout-ms $TIMEOUT_MS --json > 'bench-rank$r.json' 2> 'bench-rank$r.err' < /dev/null & echo \$! > 'bench-rank$r.pid'"
    done
    local waited=0 ready r
    while [ "$waited" -lt "$RUN_TIMEOUT_S" ]; do
        ready=0
        for r in 0 1 2 3; do
            if remote "$r" "test -s '$REMOTE_DIR/bench-rank$r.json'" 2>/dev/null; then
                ready=$((ready + 1))
            fi
        done
        [ "$ready" -eq 4 ] && break
        sleep 5
        waited=$((waited + 5))
    done
    if [ "$ready" -ne 4 ]; then
        say "TIMEOUT: only $ready of 4 ranks wrote a JSON file inside ${RUN_TIMEOUT_S}s; collecting what exists"
    fi
    for r in 0 1 2 3; do
        say "--- rank $r JSON"
        remote "$r" "cat '$REMOTE_DIR/bench-rank$r.json' 2>/dev/null || echo 'no JSON: see bench-rank$r.err'"
        say "--- rank $r stderr (tail)"
        remote "$r" "tail -n 5 '$REMOTE_DIR/bench-rank$r.err' 2>/dev/null || true"
    done
    say "run finished; no number is quoted here: read the collected JSON and the gate verdict"
}

collect() {
    mkdir -p "$LOCAL_DIR"
    local r ip
    for r in 0 1 2 3; do
        ip="${NODES[$r]}"
        say "collecting rank $r ($ip ${NAMES[$r]})"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.json" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no JSON"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.err" "$LOCAL_DIR/" 2>/dev/null || true
    done
    say "copied per-rank JSON and stderr into $LOCAL_DIR (preflight.txt is local)"
}

remote_cleanup() {
    local rank="$1"
    # No -n here: the heredoc IS stdin for the remote bash.
    ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" 'bash -s -- '"$REMOTE_DIR"' '"$CONTROL_PORT" <<'REMOTE'
dir="$1"
port="$2"
for f in "$dir"/*.pid; do
    [ -f "$f" ] || continue
    kill "$(cat "$f")" 2>/dev/null || true
done
pkill -f '[w]s-b-allreduce-bench' 2>/dev/null || true
sleep 1
if [ "$(ss -ltnH "sport = :$port" 2>/dev/null | wc -l)" = "0" ]; then
    echo 'port free'
else
    echo 'PORT_STILL_LISTENING'
fi
REMOTE
}

cleanup() {
    local r
    say "stopping only ws-b-allreduce-bench processes and checking control port $CONTROL_PORT"
    for r in 0 1 2 3; do
        remote_cleanup "$r"
    done
}

usage() { sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"
[ -n "$cmd" ] || { usage; exit 2; }
shift || true
case "$cmd" in
    preflight) preflight ;;
    build) build ;;
    selftest) selftest ;;
    stage) stage ;;
    run) run ;;
    collect) collect ;;
    cleanup) cleanup ;;
    all) build; stage; run; collect; cleanup ;;
    *) usage; exit 2 ;;
esac
