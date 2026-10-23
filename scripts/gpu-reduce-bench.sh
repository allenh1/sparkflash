#!/usr/bin/env bash
# Fabric measurement of the layer's all-reduce arms: host spawn/join reduce vs a dlopen'd vendored
# NCCL, same nodes, same payloads, same exchange order.
#
#   scripts/gpu-reduce-bench.sh fetch|build|stage|preflight|run|collect|check|gate
#
# This runner does not touch any node configuration and never runs while the TP4 vLLM deployment is
# up: preflight refuses on a GPU process or the :8000 listener unless SPARK_WSB_ALLOW_BUSY=1 is set
# inside a coordinated window, and a run beside a serving deployment is labelled, never quoted.
#
# The one exception is `gate`: the S1 collectives gate. It LOCKS THE GPU CLOCKS to their maximum,
# records gpu_mhz/power_w for the measurement window, and RESTORES the clocks in an EXIT trap. A gate
# run that cannot lock clocks is invalid and is refused before any collective is timed (override only
# for a labelled-invalid run with SPARK_REDUCE_ALLOW_UNLOCKED=1). It measures the DEVICE arm only:
#
#   10,240 B per rank (5,120 bf16 = the served decode payload of both per-layer collectives),
#   10,000 timed iterations, per-call total (enqueue + the stream sync that completes it),
#   gate: p50 <= 30 us AND p99 <= 60 us,
#   valid only if every sampled gpu clock is >= 90% of that rank's max graphics clock,
#   per-rank histogram posted to $LOCAL_DIR/histogram-rank<r>.json and summarised in
#   $LOCAL_DIR/gate-summary.json.
#
#   GATE : scripts/gpu-reduce-bench.sh fetch && scripts/gpu-reduce-bench.sh build \
#          && scripts/gpu-reduce-bench.sh stage && scripts/gpu-reduce-bench.sh gate
#   (root or passwordless sudo on the four nodes is required for `nvidia-smi -lgc`;
#    the gate exits 0 only for PASS, 1 for FAIL, 2 for an INVALID measurement)
#
# The report contract for `run` is per-call microseconds on the fabric at the layer's payload sizes:
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
# The S1 collectives gate (see the header). 10,240 B is 5,120 bf16, the served per-layer payload.
GATE_BYTES=${SPARK_GATE_BYTES:-10240}
GATE_SAMPLES=${SPARK_GATE_SAMPLES:-10000}
GATE_WARMUP=${SPARK_GATE_WARMUP:-200}
GATE_P50_US=${SPARK_GATE_P50_US:-30}
GATE_P99_US=${SPARK_GATE_P99_US:-60}
GATE_MIN_CLOCK_PCT=${SPARK_GATE_MIN_CLOCK_PCT:-90}
GATE_ALLOW_UNLOCKED=${SPARK_REDUCE_ALLOW_UNLOCKED:-0}
GATE_LOCKED=0      # 1 only when ALL FOUR ranks locked (the gate's validity rule)
GATE_LOCKED_ANY=0  # 1 when at least one rank locked: the EXIT trap must still restore
GATE_MAXCLK=()
GATE_FLAGS_STARTED=0

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

# ---------------------------------------------------------------------------------------------------
# The S1 collectives gate: clock-locked, 10,000-iteration 10 KiB GPU->GPU all-reduce.
# ---------------------------------------------------------------------------------------------------

# Restores the clocks and stops the samplers, whatever way the gate exits. Registered as the EXIT
# trap only by gate(), so no other mode touches node state.
gate_cleanup() {
    local r
    if [ "$GATE_FLAGS_STARTED" = "1" ]; then
        for r in 0 1 2 3; do
            remote "$r" "rm -f '$REMOTE_DIR/sample.flag'" 2>/dev/null || true
        done
    fi
    if [ "$GATE_LOCKED_ANY" = "1" ]; then
        for r in 0 1 2 3; do
            remote "$r" "sudo -n nvidia-smi -rgc >/dev/null 2>&1 || nvidia-smi -rgc >/dev/null 2>&1 || true" 2>/dev/null || true
        done
        say "gate: clocks restored (nvidia-smi -rgc) on all four ranks"
    fi
}

# Locks every rank's GPU clock to its maximum graphics clock. A gate that cannot lock is refused:
# an unlocked measurement cannot be compared with the 30/60 us ceilings.
gate_lock_clocks() {
    local r max ok locked=0
    for r in 0 1 2 3; do
        max=$(remote "$r" "nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -n1" 2>/dev/null | tr -d '[:space:]') || max=""
        case "$max" in
            ''|*[!0-9]*)
                say "rank $r ${NAMES[$r]}: no numeric max graphics clock: '${max:-<empty>}'"
                GATE_MAXCLK[$r]=""
                ok=0
                ;;
            *)
                GATE_MAXCLK[$r]=$max
                if remote "$r" "sudo -n nvidia-smi -lgc '$max' >/dev/null 2>&1"; then
                    ok=1
                elif remote "$r" "nvidia-smi -lgc '$max' >/dev/null 2>&1"; then
                    ok=1
                else
                    ok=0
                fi
                ;;
        esac
        if [ "$ok" = "1" ]; then
            GATE_LOCKED_ANY=1
            locked=$((locked + 1))
            say "rank $r ${NAMES[$r]}: gpu clock locked at max ${GATE_MAXCLK[$r]} MHz"
        else
            say "rank $r ${NAMES[$r]}: could not lock the gpu clock"
        fi
    done
    # Validity needs EVERY rank locked: one unlocked rank is enough to make the four-rank
    # collective's tail incomparable with the ceilings, even if that rank's clock looks high.
    if [ "$locked" -eq 4 ]; then
        GATE_LOCKED=1
    fi
    if [ "$GATE_LOCKED" != "1" ]; then
        if [ "$GATE_ALLOW_UNLOCKED" = "1" ]; then
            say "WARNING: SPARK_REDUCE_ALLOW_UNLOCKED=1: continuing UNLOCKED; this gate run is INVALID and must not be quoted"
        else
            die "the gate needs ALL FOUR GPU clocks locked (root or passwordless sudo for 'nvidia-smi -lgc'); ${locked} of 4 locked. Refusing to time a partially locked run; a fixable window may be labelled invalid with SPARK_REDUCE_ALLOW_UNLOCKED=1, but its numbers are not the gate."
        fi
    fi
}

# Starts one local sampler per node: nvidia-smi reads clocks.sm and power.draw at ~10 Hz into a
# per-rank CSV on that node. The flag file is what stops it; gate_cleanup removes the flag on a
# failure path too.
gate_start_samplers() {
    local r
    for r in 0 1 2 3; do
        remote "$r" "mkdir -p '$REMOTE_DIR' && : > '$REMOTE_DIR/clocks-rank$r.csv' && touch '$REMOTE_DIR/sample.flag'"
        remote_bg "$r" "while [ -e '$REMOTE_DIR/sample.flag' ]; do nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1; sleep 0.1; done > '$REMOTE_DIR/clocks-rank$r.csv'"
    done
    GATE_FLAGS_STARTED=1
    sleep 0.5
}

gate_stop_samplers() {
    local r
    for r in 0 1 2 3; do
        remote "$r" "rm -f '$REMOTE_DIR/sample.flag'" 2>/dev/null || true
    done
    GATE_FLAGS_STARTED=0
    sleep 1
}

# Launches the four device-only ranks. ssh -f already backgrounds the client, so the remote shell can
# run the bench in the foreground and write its real exit status: 0 pass, 6 gate fail, 3 error.
gate_launch() {
    local r
    for r in 0 1 2 3; do
        remote_bg "$r" "cd '$REMOTE_DIR' && env LD_LIBRARY_PATH='$REMOTE_DIR/lib' './bin/gpu-reduce-bench' --rank $r --world 4 --control '$CONTROL' --device-only --bytes $GATE_BYTES --samples $GATE_SAMPLES --warmup $GATE_WARMUP --mode sync --p50-max-us $GATE_P50_US --p99-max-us $GATE_P99_US --histogram '$REMOTE_DIR/histogram-rank$r.json' --json > 'bench-rank$r.json' 2> 'bench-rank$r.err'; echo \$? > 'bench-rank$r.status'"
    done
    local waited=0 ready=0
    while [ "$waited" -lt "$RUN_TIMEOUT_S" ]; do
        ready=0
        for r in 0 1 2 3; do
            remote "$r" "test -s '$REMOTE_DIR/bench-rank$r.status'" 2>/dev/null && ready=$((ready + 1))
        done
        [ "$ready" -eq 4 ] && break
        sleep 2
        waited=$((waited + 2))
    done
    [ "$ready" -eq 4 ] || die "TIMEOUT: only $ready of 4 ranks finished in ${RUN_TIMEOUT_S}s (a stuck rank is a failed gate, never a partial number)"
    for r in 0 1 2 3; do
        say "rank $r ${NAMES[$r]} status $(remote "$r" "cat '$REMOTE_DIR/bench-rank$r.status'")"
        remote "$r" "tail -n 6 '$REMOTE_DIR/bench-rank$r.err' 2>/dev/null || true"
    done
}

gate_collect() {
    mkdir -p "$LOCAL_DIR"
    local r ip
    for r in 0 1 2 3; do
        ip=${NODES[$r]}
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.json" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no bench JSON"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.status" "$LOCAL_DIR/" 2>/dev/null || true
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/bench-rank$r.err" "$LOCAL_DIR/" 2>/dev/null || true
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/histogram-rank$r.json" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no histogram JSON"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/clocks-rank$r.csv" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no clock sample CSV"
    done
    : > "$LOCAL_DIR/clocks-max.txt"
    for r in 0 1 2 3; do printf '%s,%s\n' "$r" "${GATE_MAXCLK[$r]:-}" >> "$LOCAL_DIR/clocks-max.txt"; done
}

# The gate verdict: the bench computed each rank's p50/p99 and its own pass flag; this step checks
# those, the clocks' 90%-of-max validity rule and the histogram's presence, and writes one summary.
# Exit 0 PASS, 1 FAIL, 2 INVALID.
gate_verdict() {
    command -v python3 >/dev/null 2>&1 || die "the gate verdict needs python3"
    local head
    head=$(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)
    python3 - "$LOCAL_DIR" "$GATE_BYTES" "$GATE_SAMPLES" "$GATE_WARMUP" "$GATE_P50_US" "$GATE_P99_US" "$GATE_MIN_CLOCK_PCT" "$GATE_LOCKED" "$GATE_ALLOW_UNLOCKED" "$STAMP" "$head" <<'PY'
import json
import os
import sys

(local_dir, gate_bytes, samples, warmup, p50_max, p99_max, min_clock_pct,
 locked, allow_unlocked, stamp, head) = sys.argv[1:12]
gate_bytes, samples, warmup = int(gate_bytes), int(samples), int(warmup)
p50_max, p99_max, min_clock_pct = float(p50_max), float(p99_max), float(min_clock_pct)
locked = locked == "1"
allow_unlocked = allow_unlocked == "1"

clock_problems = []
latency_problems = []
ranks = []
max_clock = {}
with open(os.path.join(local_dir, "clocks-max.txt")) as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        rank, _, value = line.partition(",")
        if value.strip():
            max_clock[int(rank)] = float(value)

for rank in range(4):
    entry = {"rank": rank, "p50_us": None, "p99_us": None, "gate_pass": False,
             "min_clock_mhz": None, "mean_clock_mhz": None, "power_w_mean": None,
             "histogram_present": False, "status": None}
    status_path = os.path.join(local_dir, "bench-rank%d.status" % rank)
    if os.path.exists(status_path):
        with open(status_path) as handle:
            text = handle.read().strip()
        entry["status"] = int(text) if text.isdigit() else None
    if entry["status"] not in (0, 6):
        latency_problems.append("rank %d bench status %r" % (rank, entry["status"]))
    bench_path = os.path.join(local_dir, "bench-rank%d.json" % rank)
    if os.path.exists(bench_path):
        with open(bench_path) as handle:
            document = json.load(handle)
        arms = [arm for arm in document.get("arms", [])
                if arm.get("arm") == "nccl" and int(arm.get("bytes", -1)) == gate_bytes]
        if not arms:
            latency_problems.append("rank %d JSON carries no nccl arm at %d B" % (rank, gate_bytes))
        else:
            arm = arms[0]
            gate = arm.get("gate", {})
            entry["p50_us"] = gate.get("p50_us")
            entry["p99_us"] = gate.get("p99_us")
            entry["gate_pass"] = bool(gate.get("pass", False))
            if entry["p50_us"] is None or entry["p99_us"] is None:
                latency_problems.append("rank %d JSON has no gate block (was the bench run with both ceilings?)" % rank)
            if not entry["gate_pass"]:
                latency_problems.append("rank %d latency gate failed (p50 %r us <= %s, p99 %r us <= %s)" % (
                    rank, entry["p50_us"], p50_max, entry["p99_us"], p99_max))
            if gate.get("p50_max_us") != p50_max or gate.get("p99_max_us") != p99_max:
                latency_problems.append("rank %d gate block thresholds do not match this run" % rank)
            if arm.get("mode") != "sync":
                latency_problems.append("rank %d gate arm is not the sync per-call total (mode %r)" % (
                    rank, arm.get("mode")))
    else:
        latency_problems.append("rank %d wrote no bench JSON" % rank)
    histogram = os.path.join(local_dir, "histogram-rank%d.json" % rank)
    entry["histogram_present"] = os.path.exists(histogram)
    if not entry["histogram_present"]:
        latency_problems.append("rank %d wrote no histogram" % rank)
    clock_path = os.path.join(local_dir, "clocks-rank%d.csv" % rank)
    clocks, powers = [], []
    if os.path.exists(clock_path):
        with open(clock_path) as handle:
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) < 2:
                    continue
                try:
                    clocks.append(float(fields[0]))
                except ValueError:
                    pass
                try:
                    powers.append(float(fields[1]))
                except ValueError:
                    pass
    if clocks:
        entry["min_clock_mhz"] = min(clocks)
        entry["mean_clock_mhz"] = sum(clocks) / len(clocks)
        floor = max_clock.get(rank)
        if floor:
            if entry["min_clock_mhz"] < min_clock_pct / 100.0 * floor:
                clock_problems.append("rank %d clock dropped to %.0f MHz, below %.0f%% of its %.0f MHz max"
                                % (rank, entry["min_clock_mhz"], min_clock_pct, floor))
        else:
            clock_problems.append("rank %d has no max-clock reading to validate against" % rank)
    else:
        clock_problems.append("rank %d produced no clock samples during the window" % rank)
    if powers:
        entry["power_w_mean"] = sum(powers) / len(powers)
    ranks.append(entry)

if not locked:
    clock_problems.append("clocks were not locked (an unlocked gate is invalid)")
if allow_unlocked:
    clock_problems.append("SPARK_REDUCE_ALLOW_UNLOCKED=1 labelled this run invalid")
valid = not clock_problems
passed = valid and not latency_problems
summary = {
    "schema": "sparkflash-reduce-gate/1",
    "stamp": stamp,
    "git_head": head,
    "payload_bytes": gate_bytes,
    "samples": samples,
    "warmup": warmup,
    "mode": "sync",
    "thresholds": {"p50_max_us": p50_max, "p99_max_us": p99_max},
    "clock": {
        "locked": locked,
        "min_clock_pct": min_clock_pct,
        "max_mhz": {str(rank): value for rank, value in sorted(max_clock.items())},
        "valid": valid,
    },
    "histogram_paths": [os.path.join(local_dir, "histogram-rank%d.json" % rank) for rank in range(4)],
    "ranks": ranks,
    "valid": valid,
    "pass": passed,
    "clock_problems": clock_problems,
    "latency_problems": latency_problems,
}
with open(os.path.join(local_dir, "gate-summary.json"), "w") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
for entry in ranks:
    print("rank %d: p50 %s us p99 %s us clocks %s/%s MHz power %s W gate %s"
          % (entry["rank"], entry["p50_us"], entry["p99_us"], entry["min_clock_mhz"],
             entry["mean_clock_mhz"], entry["power_w_mean"], "PASS" if entry["gate_pass"] else "FAIL"))
print("gate: %s" % ("PASS" if passed else ("FAIL" if valid else "INVALID")))
print("gate summary: %s" % os.path.join(local_dir, "gate-summary.json"))
if passed:
    sys.exit(0)
for problem in clock_problems + latency_problems:
    print("gate problem: %s" % problem)
sys.exit(2 if not valid else 1)
PY
}

gate() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    [ -s "$LIB" ] || die "no vendored libnccl at $LIB (run: $0 fetch)"
    mkdir -p "$LOCAL_DIR"
    trap gate_cleanup EXIT
    preflight
    say "gate: locking gpu clocks (the only node-state change this runner makes; restored on exit)"
    gate_lock_clocks
    gate_start_samplers
    say "gate: device-only 10,240 B all-reduce, $GATE_SAMPLES iterations, p50 <= $GATE_P50_US us, p99 <= $GATE_P99_US us"
    gate_launch
    gate_stop_samplers
    gate_collect
    for r in 0 1 2 3; do
        [ -s "$LOCAL_DIR/bench-rank$r.json" ] && say "--- rank $r JSON (gate fields at the end)" && cat "$LOCAL_DIR/bench-rank$r.json"
    done
    local status=0
    gate_verdict || status=$?
    say "gate: histogram files: $LOCAL_DIR/histogram-rank{0,1,2,3}.json"
    exit "$status"
}

case "${1:-run}" in
    fetch) fetch ;;
    build) build ;;
    stage) stage ;;
    preflight) preflight ;;
    run) run ;;
    collect) collect ;;
    gate) gate ;;
    check)
        say "check: loading libnccl on rank 0 (no GPU work)"
        remote 0 "LD_LIBRARY_PATH='$REMOTE_DIR/lib' '$REMOTE_DIR/bin/gpu-reduce-bench' --check-nccl" ;;
    *) die "usage: $0 fetch|build|stage|preflight|run|collect|check|gate" ;;
esac
