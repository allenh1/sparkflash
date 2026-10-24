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
# The one exception is `gate`: the S2-a collectives gate. It measures at the nodes' NATIVE BOOST
# CLOCKS (2405-2411 MHz on these GB10 parts, which hold them without a lock), records gpu_mhz and
# power_w for the window, and clears any lingering artificial lock first: the operator decision of
# 2026-10-23 retired the 1500 MHz clock lock as a measurement protocol. A window whose *active* mean
# clock falls below 90% of the parts' maximum is still refused as invalid (idle samples below 500 MHz
# are excluded: min() over a boost window is an idle artefact, not a throttle). It measures the
# DEVICE arm only:
#
#   10,240 B per rank (5,120 bf16 = the served decode payload of both per-layer collectives),
#   send/recv registered with the communicator before the first enqueue (ncclCommRegister, the
#   run's S2-a shape),
#   10,000 timed iterations, per-call total (enqueue + the stream sync that completes it),
#   gate: p50 <= 75 us AND p90 <= 104 us AND p99 <= 251 us, every ceiling 1.5x the measured worst
#   of 40 rank-samples over 10 post-fix runs (p50 47.89-49.76, p90 58.51-69.39, p99 78.72-167.20;
#   D57). Three bars because the p99 on this fabric is BIMODAL and therefore is not a pass/fail
#   instrument at that magnitude: six runs measured p99 78.7-87.5 us and four measured 149.6-167.2 us
#   with the same p50 (48.2-48.5) and the same 2411 MHz clocks, so the p99 mode is not the collective.
#   p50 detects a central shift (a transport or clock fallback), p90 detects a tail regression and is
#   stable (58.5-69.4 us), and p99 is a catastrophic-tail alarm only: the ARP-broken fabric of this
#   window (p50 56.7-58.9, p90 402-448, p99 583-668 us) fails all three. The floor is a protocol set
#   (provenance.git_dirty = false), not an outcome selection; docs/build/fabric.md 5.2 lists every
#   row with the protocol it ran under, including the ones no longer on the floor,
#   AND both buffers registered on every rank AND one NCCL build and one transport environment on all
#   four ranks AND GPUDirect RDMA active ON EVERY RANK THAT REPORTS SUPPORT for it: the driver's own
#   CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED / _DMA_BUF_SUPPORTED decide whether the clause
#   binds (DGX Spark GB10 reports both 0, so GDR is recorded as undrivable there, not waived),
#   valid only if every rank's active mean gpu clock is >= 90% of its maximum graphics clock,
#   per-rank histogram posted to $LOCAL_DIR/histogram-rank<r>.json, NCCL's INIT/NET log to
#   $LOCAL_DIR/nccl-rank<r>.log (the GDR evidence), the capability probe to
#   $LOCAL_DIR/platform-rank<r>.json, all summarised in $LOCAL_DIR/gate-summary.json.
#
# The operator's NCCL transport variables (NCCL_FORWARD below, the same list the run header
# records) are forwarded to every rank verbatim, and each rank's JSON states the values it ran with.
#
#   GATE : scripts/gpu-reduce-bench.sh fetch && scripts/gpu-reduce-bench.sh build \
#          && scripts/gpu-reduce-bench.sh stage && scripts/gpu-reduce-bench.sh gate
#   (no root or clock lock is needed; the gate exits 0 only for PASS, 1 for FAIL, 2 for INVALID)
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
# The second RoCEv2 rail. The bench talks on both, so the gate's warm-up has to cover both.
RAIL2=(${SPARK_RAIL2_NODES:-192.0.2.85 192.0.2.84 192.0.2.63 192.0.2.64})
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
# The S2-a collectives gate (see the header). 10,240 B is 5,120 bf16, the served per-layer payload.
GATE_BYTES=${SPARK_GATE_BYTES:-10240}
GATE_SAMPLES=${SPARK_GATE_SAMPLES:-10000}
GATE_WARMUP=${SPARK_GATE_WARMUP:-200}
GATE_P50_US=${SPARK_GATE_P50_US:-75}
GATE_P90_US=${SPARK_GATE_P90_US:-104}
GATE_P99_US=${SPARK_GATE_P99_US:-251}
GATE_MIN_CLOCK_PCT=${SPARK_GATE_MIN_CLOCK_PCT:-90}
# The boost protocol's clock target is the measured steady-state sm clock of these GB10 parts
# (2405-2411 MHz), NOT the 3003 MHz `clocks.max.graphics` spec ceiling, which is never sustained.
GATE_BOOST_MHZ=${SPARK_GATE_BOOST_MHZ:-2411}
# Samples below this are the part idling between phases (201-1150 MHz observed), not a working clock:
# the validity statistic must not average them in, or a long NCCL init reads as a throttle.
GATE_IDLE_MHZ=${SPARK_GATE_IDLE_MHZ:-1500}
GATE_LOCKED=1      # Native boost clocks (~2405-2411 MHz) on all four GB10 nodes; no 1500 MHz lock
GATE_LOCKED_ANY=0
GATE_MAXCLK=()
GATE_FLAGS_STARTED=0

say() { printf '%s\n' "$*" >&2; }
die() { say "gpu-reduce-bench: $*"; exit 1; }

# The NCCL transport variables the run header records (sparkflash_comm::nccl::TRANSPORT_ENV; a
# sparkflash-comm test keeps the two lists equal). Set ones are forwarded to every rank verbatim;
# unset ones stay unset there, so NCCL's own defaults apply and the JSON says null.
NCCL_FORWARD=(NCCL_IB_HCA NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_ALGO NCCL_PROTO NCCL_IB_GID_INDEX
    NCCL_SOCKET_IFNAME NCCL_DMABUF_ENABLE NCCL_LOCAL_REGISTER)
nccl_env() {
    local name value out=""
    for name in "${NCCL_FORWARD[@]}"; do
        [ -n "${!name+set}" ] || continue
        value=${!name}
        case "$value" in *"'"*) die "$name carries a single quote; refusing to forward it to the ranks" ;; esac
        out="$out $name='$value'"
    done
    printf '%s' "$out"
}
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
        remote_bg "$r" "cd '$REMOTE_DIR' && nohup env LD_LIBRARY_PATH='$REMOTE_DIR/lib'$(nccl_env) './bin/gpu-reduce-bench' --rank $r --world 4 --addrs '$ADDRS' --control '$CONTROL' --bytes $BYTES --samples $SAMPLES --warmup $WARMUP --mode $MODE --batch $BATCH --json > 'bench-rank$r.json' 2> 'bench-rank$r.err' < /dev/null & echo \$! > 'bench-rank$r.pid'"
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
# The S2-a collectives gate: 10,000-iteration 10 KiB GPU->GPU all-reduce at native boost clocks.
# ---------------------------------------------------------------------------------------------------

# Restores the clocks and stops the samplers, whatever way the gate exits. Registered as the EXIT
# trap only by gate(), so no other mode touches node state.
gate_cleanup() {
    local r
    # A rank whose peers never arrived keeps spinning in its collective wait, holding the device and
    # the page cache for the next run: an orphan left behind is exactly what makes the NEXT preflight
    # report gpu_apps=1 and refuse the window. However this gate exits, no bench outlives it.
    for r in 0 1 2 3; do
        remote "$r" "pkill -9 -f 'gpu-reduce[-]bench' 2>/dev/null; true" 2>/dev/null || true
    done
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

# Queries every rank's maximum graphics clock (2411 MHz on GB10) and clears any lingering
# artificial clock lock so all four nodes run at native boost clocks (~2405-2411 MHz).
gate_lock_clocks() {
    local r max ok=0
    for r in 0 1 2 3; do
        remote "$r" "sudo -n nvidia-smi -rgc >/dev/null 2>&1 || nvidia-smi -rgc >/dev/null 2>&1 || true" 2>/dev/null || true
        max=$(remote "$r" "nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -n1" 2>/dev/null | tr -d '[:space:]') || max=""
        case "$max" in
            ''|*[!0-9]*)
                say "rank $r ${NAMES[$r]}: no numeric max graphics clock: '${max:-<empty>}'"
                GATE_MAXCLK[$r]=""
                ;;
            *)
                GATE_MAXCLK[$r]=$max
                ok=$((ok + 1))
                say "rank $r ${NAMES[$r]}: native boost mode (max graphics clock ${GATE_MAXCLK[$r]} MHz)"
                ;;
        esac
    done
    if [ "$ok" -eq 4 ]; then
        GATE_LOCKED=1
    else
        GATE_LOCKED=0
    fi
}

# Starts one local sampler per node: nvidia-smi reads clocks.sm and power.draw at ~10 Hz into a
# per-rank CSV on that node. The flag file is what stops it; gate_cleanup removes the flag on a
# failure path too.
gate_start_samplers() {
    local r
    for r in 0 1 2 3; do
        remote "$r" "mkdir -p '$REMOTE_DIR' && : > '$REMOTE_DIR/clocks-rank$r.csv' && touch '$REMOTE_DIR/sample.flag'"
        remote_bg "$r" "while [ -e '$REMOTE_DIR/sample.flag' ]; do echo \"\$(date +%s.%N),\$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1)\"; sleep 0.1; done > '$REMOTE_DIR/clocks-rank$r.csv'"
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
# NCCL's INIT/NET log goes to its own file (NCCL_DEBUG_FILE), never to stdout, which is the JSON: the
# channel lines in it ("via NET/IB/<n>/GDRDMA") are the GDR evidence the verdict reads. INFO logging
# is init and connect time only; the timed collectives log nothing.
# Records whether each rank's platform can do GPUDirect RDMA at all, from the driver's own device
# attributes (cuDeviceGetAttribute). The GDR clause is only judgeable against this: a node that
# reports CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 0 cannot activate GDR, so "GDR is not
# active" there is a platform fact, not a regression in the collective path. The spare attribute
# (SM count) is read in the same call as a sanity check on the probe itself.
gate_probe_platform() {
    local r
    for r in 0 1 2 3; do
        remote "$r" "mkdir -p '$REMOTE_DIR' && python3 - '$REMOTE_DIR/platform-rank$r.json' <<'PY'
import ctypes, json, sys
out = {\"gdr_supported\": None, \"dmabuf_supported\": None, \"sm_count\": None, \"probe\": \"unavailable\"}
try:
    cu = ctypes.CDLL(\"libcuda.so.1\")
    if cu.cuInit(0) == 0:
        dev = ctypes.c_int()
        if cu.cuDeviceGet(ctypes.byref(dev), 0) == 0:
            val = ctypes.c_int()
            def attr(number):
                return val.value if cu.cuDeviceGetAttribute(ctypes.byref(val), number, dev) == 0 else None
            out = {\"gdr_supported\": attr(116), \"dmabuf_supported\": attr(124),
                   \"sm_count\": attr(16), \"probe\": \"ok\"}
except OSError as error:
    out[\"error\"] = str(error)
with open(sys.argv[1], \"w\") as handle:
    json.dump(out, handle)
PY" 2>/dev/null || say "rank $r: platform capability probe failed"
    done
}

# Starts one rank's bench in the background. Every rank needs the same control address and the same
# forwarded NCCL transport env; only --rank differs.
gate_launch_rank() {
    local r=$1 forwarded=$2
    remote_bg "$r" "cd '$REMOTE_DIR' && env LD_LIBRARY_PATH='$REMOTE_DIR/lib' NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE='$REMOTE_DIR/nccl-rank$r.log'$forwarded './bin/gpu-reduce-bench' --rank $r --world 4 --control '$CONTROL' --device-only --bytes $GATE_BYTES --samples $GATE_SAMPLES --warmup $GATE_WARMUP --mode sync --p50-max-us $GATE_P50_US --p90-max-us $GATE_P90_US --p99-max-us $GATE_P99_US --histogram '$REMOTE_DIR/histogram-rank$r.json' --json > 'bench-rank$r.json' 2> 'bench-rank$r.err'; echo \$? > 'bench-rank$r.status'"
}

gate_launch() {
    local r forwarded
    forwarded=$(nccl_env)
    say "gate: forwarded NCCL env:${forwarded:- (none: NCCL defaults)}"
    for r in 0 1 2 3; do
        remote "$r" "cd '$REMOTE_DIR' && rm -f 'bench-rank$r.status' 'bench-rank$r.json' 'bench-rank$r.err' 'nccl-rank$r.log' 'histogram-rank$r.json'"
    done
    gate_launch_rank 0 "$forwarded"
    # Rank 0 owns the control plane and every peer DIALS it, so a peer that starts before the listener
    # is up gets ECONNREFUSED and exits while the other three wait for it (observed 2026-10-23: rank 1
    # failed in 0.2 s and ranks 0/2/3 hung until the run timeout). Wait for the port, then start the
    # peers. The listener is init-time only: no timed collective is affected by the stagger.
    local plane_port=${CONTROL##*:} waited_plane=0 died_early=0
    while [ "$waited_plane" -lt 120 ]; do
        if remote 0 "ss -ltnH 'sport = :$plane_port' 2>/dev/null | grep -q ." 2>/dev/null; then
            break
        fi
        # Rank 0 can also fail before it ever listens (a bad flag, a missing library, a busy GPU).
        # Without this the gate waits two minutes for a port that is never coming and then blames
        # the control plane: 2026-10-24 lost 120 s to a binary that had already exited with status 2.
        if remote 0 "test -s '$REMOTE_DIR/bench-rank0.status'" 2>/dev/null; then
            died_early=1
            break
        fi
        sleep 1
        waited_plane=$((waited_plane + 1))
    done
    if [ "$died_early" -eq 1 ]; then
        say "rank 0 ${NAMES[0]}: exited before opening :$plane_port (status $(remote 0 "cat '$REMOTE_DIR/bench-rank0.status'" 2>/dev/null))"
        remote 0 "tail -n 5 '$REMOTE_DIR/bench-rank0.err' 2>/dev/null" 2>/dev/null | sed 's/^/    rank 0: /' || true
    elif [ "$waited_plane" -ge 120 ]; then
        say "rank 0 ${NAMES[0]}: control plane :$plane_port never came up in 120 s"
    else
        say "gate: rank 0 control plane :$plane_port is listening (${waited_plane}s)"
    fi
    gate_launch_rank 1 "$forwarded"
    gate_launch_rank 2 "$forwarded"
    gate_launch_rank 3 "$forwarded"
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
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/nccl-rank$r.log" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no NCCL log (no GDR evidence)"
        scp -q "$SSH_USER@${ip}:$REMOTE_DIR/platform-rank$r.json" "$LOCAL_DIR/" 2>/dev/null || say "rank $r: no platform capability record"
    done
    : > "$LOCAL_DIR/clocks-max.txt"
    for r in 0 1 2 3; do printf '%s,%s\n' "$r" "${GATE_MAXCLK[$r]:-}" >> "$LOCAL_DIR/clocks-max.txt"; done
}

# The gate verdict: the bench computed each rank's p50/p99 and its own pass flag; this step checks
# those, the clocks' 90%-of-lock validity rule, the histogram's presence, GPUDirect RDMA from NCCL's
# own channel lines, the buffer registration from the bench's stats, and one NCCL build and one
# transport environment across the four ranks, and writes one summary. Exit 0 PASS, 1 FAIL,
# 2 INVALID.
gate_verdict() {
    command -v python3 >/dev/null 2>&1 || die "the gate verdict needs python3"
    local head dirty script_sha
    head=$(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)
    # Provenance, because a run recorded against a commit whose tree cannot produce its own result is
    # not reproducible: one earlier floor row in this window claimed a commit that cannot emit its
    # schema. A floor row is only admissible with git_dirty = false.
    if ( cd "$ROOT" && git rev-parse --git-dir >/dev/null 2>&1 ); then
        if [ -n "$(cd "$ROOT" && git status --porcelain -- crates scripts 2>/dev/null)" ]; then
            dirty=true
        else
            dirty=false
        fi
    else
        dirty=unknown
    fi
    script_sha=$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1 || echo unknown)
    python3 - "$LOCAL_DIR" "$GATE_BYTES" "$GATE_SAMPLES" "$GATE_WARMUP" "$GATE_P50_US" "$GATE_P90_US" "$GATE_P99_US" "$GATE_MIN_CLOCK_PCT" "$GATE_BOOST_MHZ" "$GATE_IDLE_MHZ" "$STAMP" "$head" "$dirty" "$script_sha" <<'PY'
import json
import os
import re
import sys

(local_dir, gate_bytes, samples, warmup, p50_max, p90_max, p99_max, min_clock_pct,
 boost_mhz, idle_mhz, stamp, head, git_dirty, script_sha) = sys.argv[1:15]
gate_bytes, samples, warmup = int(gate_bytes), int(samples), int(warmup)
p50_max, p90_max, p99_max, min_clock_pct = float(p50_max), float(p90_max), float(p99_max), float(min_clock_pct)
boost_mhz, idle_mhz = float(boost_mhz), float(idle_mhz)

# NCCL's connection line for one channel and direction, e.g.
#   NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA
# A NET channel without the /GDRDMA suffix stages through host memory; NET/Socket never has it.
CHANNEL = re.compile(r"Channel \d+/\d+ : .* via (NET/\S+)")


def gdr_evidence(path):
    """The rank's NET channels and how many of them run GPUDirect RDMA, from NCCL's own log."""
    evidence = {"log_present": os.path.exists(path), "net_channels": 0, "gdr_channels": 0,
                "transports": [], "active": False}
    if not evidence["log_present"]:
        return evidence
    transports = set()
    with open(path, errors="replace") as handle:
        for line in handle:
            match = CHANNEL.search(line)
            if not match:
                continue
            evidence["net_channels"] += 1
            transport = match.group(1)
            transports.add(transport)
            if transport.endswith("/GDRDMA"):
                evidence["gdr_channels"] += 1
    evidence["transports"] = sorted(transports)
    evidence["active"] = evidence["net_channels"] > 0 and evidence["gdr_channels"] == evidence["net_channels"]
    return evidence


clock_problems = []
latency_problems = []
transport_problems = []
# Facts the gate records without failing on: a clause this platform cannot satisfy at all (GDR), so
# the PASS/FAIL stays about the collective path and the platform limit is stated in the summary.
transport_problems_note = []
ranks = []
builds = {}
envs = {}
max_clock = {}


def platform_record(rank):
    """The rank's driver-reported device capabilities, or an empty record if the probe did not run."""
    path = os.path.join(local_dir, "platform-rank%d.json" % rank)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}
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
             "histogram_present": False, "status": None, "registered_buffers": None,
             "registered_bytes": None, "gdr": None}
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
        nccl = document.get("nccl", {})
        builds[rank] = (nccl.get("version"), nccl.get("library"))
        envs[rank] = nccl.get("env")
        if envs[rank] is None:
            transport_problems.append("rank %d JSON records no NCCL transport env (a pre-S2 bench binary?)" % rank)
        arms = [arm for arm in document.get("arms", [])
                if arm.get("arm") == "nccl" and int(arm.get("bytes", -1)) == gate_bytes]
        if not arms:
            latency_problems.append("rank %d JSON carries no nccl arm at %d B" % (rank, gate_bytes))
        else:
            arm = arms[0]
            gate = arm.get("gate", {})
            # The verdict RE-DERIVES the latency comparison from the reported numbers. The bench's own
            # `gate.pass` boolean is never trusted: it is computed inside the binary under test, so a
            # stale binary, a bench bug or an edited JSON would otherwise be able to report PASS.
            entry["p50_us"] = gate.get("p50_us", arm.get("p50_us"))
            entry["p90_us"] = arm.get("p90_us")
            entry["p99_us"] = gate.get("p99_us", arm.get("p99_us"))
            entry["bench_reported_pass"] = bool(gate.get("pass", False))
            if entry["p50_us"] is None or entry["p99_us"] is None or entry["p90_us"] is None:
                latency_problems.append("rank %d JSON has no p50/p90/p99 triple (was the bench run with both ceilings?)" % rank)
                entry["gate_pass"] = False
            else:
                for label, value, ceiling in (("p50", entry["p50_us"], p50_max),
                                             ("p90", entry["p90_us"], p90_max),
                                             ("p99", entry["p99_us"], p99_max)):
                    if value > ceiling:
                        latency_problems.append("rank %d latency gate failed: %s %.2f us > %s" % (
                            rank, label, value, ceiling))
                entry["gate_pass"] = (entry["p50_us"] <= p50_max and entry["p90_us"] <= p90_max
                                      and entry["p99_us"] <= p99_max)
            if gate.get("p50_max_us") != p50_max or gate.get("p99_max_us") != p99_max:
                latency_problems.append("rank %d gate block thresholds (%r/%r) do not match this run's (%r/%r)" % (
                    rank, gate.get("p50_max_us"), gate.get("p99_max_us"), p50_max, p99_max))
            if gate.get("p90_max_us") != p90_max:
                latency_problems.append("rank %d gate block has no p90 ceiling (the binary predates it: rebuild and restage)" % rank)
            if entry["bench_reported_pass"] != entry["gate_pass"]:
                latency_problems.append("rank %d bench reported pass=%r but its own numbers say %r: the binary and the verdict disagree" % (
                    rank, entry["bench_reported_pass"], entry["gate_pass"]))
            if arm.get("mode") != "sync":
                latency_problems.append("rank %d gate arm is not the sync per-call total (mode %r)" % (
                    rank, arm.get("mode")))
            # Why the collective is correct, not just fast: the bench compares the reduced buffer
            # against the expected sum and records the verdict. Nothing in this script used to read it.
            check = arm.get("check", {})
            entry["check"] = check
            if check.get("differing") != 0:
                latency_problems.append("rank %d reduced buffer is WRONG: %r elements differ from the expected sum" % (
                    rank, check.get("differing")))
            if check.get("max_ulp") != 0:
                latency_problems.append("rank %d reduced buffer is not bit-exact (max_ulp %r)" % (
                    rank, check.get("max_ulp")))
            if check.get("finite") is not True:
                latency_problems.append("rank %d reduced buffer has non-finite values (finite %r)" % (
                    rank, check.get("finite")))
            stats = arm.get("stats", {})
            # And that it actually ran the requested number of collectives: a bench that stopped early
            # would report a beautiful percentile over a truncated sample.
            entry["collectives"] = stats.get("collectives")
            if (entry["collectives"] or 0) < samples:
                latency_problems.append("rank %d timed only %r of %d requested collectives: the percentile would describe a truncated sample" % (
                    rank, entry["collectives"], samples))
            entry["registered_buffers"] = stats.get("registered_buffers")
            entry["registered_bytes"] = stats.get("registered_bytes")
            if (entry["registered_buffers"] or 0) < 2 or (entry["registered_bytes"] or 0) < 2 * gate_bytes:
                transport_problems.append("rank %d did not register its send/recv buffers (registered %r buffers, %r B)" % (
                    rank, entry["registered_buffers"], entry["registered_bytes"]))
    else:
        latency_problems.append("rank %d wrote no bench JSON" % rank)
    entry["gdr"] = gdr_evidence(os.path.join(local_dir, "nccl-rank%d.log" % rank))
    # The GDR clause is "active wherever the platform supports it", read from the driver's own device
    # attributes rather than assumed. A node whose CUDA driver reports
    # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 0 cannot activate GDR at all (DGX Spark GB10
    # reports 0, and DMA-BUF 0 with it), so an inactive channel there is a platform fact to record,
    # not a failure of this collective path. Where support IS reported, inactive GDR still fails.
    platform = platform_record(rank)
    entry["platform"] = platform
    # Unprobed is NOT unsupported: if the probe did not run, the clause stays required (conservative),
    # so a missing probe can never be used to escape the GDR requirement.
    probe_ok = (platform.get("probe") == "ok"
                and platform.get("gdr_supported") in (0, 1)
                and platform.get("dmabuf_supported") in (0, 1)
                and isinstance(platform.get("sm_count"), int) and platform.get("sm_count") > 0)
    entry["gdr_required"] = (not probe_ok) or platform.get("gdr_supported") == 1
    if not probe_ok:
        transport_problems.append("rank %d driver-reported device capabilities are unconfirmed (no usable capability probe): GDR stays required" % rank)
    if not entry["gdr"]["log_present"]:
        transport_problems.append("rank %d wrote no NCCL log: GPUDirect RDMA cannot be confirmed" % rank)
    elif entry["gdr"]["net_channels"] == 0:
        transport_problems.append("rank %d NCCL log has no NET channel line at all: the transport that actually carried the collective is unconfirmed" % rank)
    elif entry["gdr_required"] and not entry["gdr"]["active"]:
        transport_problems.append("rank %d GPUDirect RDMA not active although the driver reports support: %d of %d NET channels via GDRDMA (%s)" % (
            rank, entry["gdr"]["gdr_channels"], entry["gdr"]["net_channels"],
            ", ".join(entry["gdr"]["transports"]) or "no NET channel lines"))
    elif not entry["gdr_required"] and not entry["gdr"]["active"]:
        transport_problems_note.append("rank %d GPUDirect RDMA inactive and the driver reports GPU_DIRECT_RDMA_SUPPORTED=%r, DMA_BUF_SUPPORTED=%r: not drivable on this platform, recorded not waived" % (
            rank, platform.get("gdr_supported"), platform.get("dmabuf_supported")))
    histogram = os.path.join(local_dir, "histogram-rank%d.json" % rank)
    entry["histogram_present"] = os.path.exists(histogram)
    if not entry["histogram_present"]:
        latency_problems.append("rank %d wrote no histogram" % rank)
    clock_path = os.path.join(local_dir, "clocks-rank%d.csv" % rank)
    clocks, powers, stamps = [], [], []
    if os.path.exists(clock_path):
        with open(clock_path) as handle:
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) >= 3:
                    # ts,clocks.sm,power.draw (the current sampler)
                    field_clock, field_power = fields[1], fields[2]
                    try:
                        stamps.append(float(fields[0]))
                    except ValueError:
                        pass
                elif len(fields) == 2:
                    # clocks.sm,power.draw (rows taken before the sampler timestamped its samples)
                    field_clock, field_power = fields[0], fields[1]
                else:
                    continue
                try:
                    clocks.append(float(field_clock))
                except ValueError:
                    pass
                try:
                    powers.append(float(field_power))
                except ValueError:
                    pass
    if clocks:
        # SCOPE, stated exactly because it is easy to overclaim: this is a WINDOW-LEVEL check, not a
        # measurement of the clock during the timed collectives. The samples carry no attribution to
        # the timed phase (10,000 iterations at ~48 us is ~0.48 s of the multi-second window, and the
        # sampler runs at ~4 Hz), so no statistic over them can prove what clock the collectives ran
        # at. What it does prove is that the window was not idle and that no artificial lock pinned
        # the part low: the MEDIAN OF THE NON-IDLE SAMPLES excludes idle states (201-1150 MHz between
        # phases) and a part sitting at a throttled working clock for most of the window moves it down.
        # Measured on four GB10 nodes: 2398-2554 MHz across the rank-windows with samples. A run whose
        # clocks matter per-collective needs an instrumented step row (a follow-up, not this gate).
        active = sorted(c for c in clocks if c >= idle_mhz)
        entry["non_idle_samples"] = len(active)
        entry["clock_window_s"] = (max(stamps) - min(stamps)) if len(stamps) > 1 else None
        entry["max_clock_mhz"] = max(clocks)
        entry["min_idle_clock_mhz"] = min(clocks)
        if not active:
            clock_problems.append("rank %d never left its idle clock during the window (no sample >= %.0f MHz): the timed collectives cannot have run at boost" % (
                rank, idle_mhz))
        else:
            entry["min_clock_mhz"] = active[0]
            entry["active_median_mhz"] = active[len(active) // 2]
            entry["mean_clock_mhz"] = sum(active) / len(active)
            if entry["active_median_mhz"] < min_clock_pct / 100.0 * boost_mhz:
                clock_problems.append("rank %d active median clock %.0f MHz dropped below %.0f%% of the %.0f MHz boost target (%d non-idle samples)" % (
                    rank, entry["active_median_mhz"], min_clock_pct, boost_mhz, len(active)))
    else:
        clock_problems.append("rank %d produced no clock samples during the window" % rank)
    if powers:
        entry["power_w_mean"] = sum(powers) / len(powers)
    ranks.append(entry)

if len(set(builds.values())) > 1:
    transport_problems.append("the ranks loaded different NCCL builds: %r" % builds)
if len({json.dumps(env, sort_keys=True) for env in envs.values()}) > 1:
    transport_problems.append("the ranks ran with different NCCL transport env: %r" % envs)
valid = not clock_problems
passed = valid and not latency_problems and not transport_problems
build = next(iter(builds.values()), (None, None))
summary = {
    "schema": "sparkflash-reduce-gate/3",
    "stamp": stamp,
    "git_head": head,
    "provenance": {
        "git_head": head,
        "git_dirty": git_dirty,
        "script_sha256": script_sha,
        "admissible_as_floor_row": git_dirty == "false",
    },
    "payload_bytes": gate_bytes,
    "samples": samples,
    "warmup": warmup,
    "mode": "sync",
    "fabric": {"warmed": True, "method": "ping every peer on both rails, then 3 s settle"},
    "thresholds": {"p50_max_us": p50_max, "p90_max_us": p90_max, "p99_max_us": p99_max},
    "clock": {
        "protocol": "native-boost",
        "lock": None,
        "boost_target_mhz": boost_mhz,
        "idle_floor_mhz": idle_mhz,
        "statistic": "median of non-idle sm samples >= min_clock_pct of the boost target",
        "scope": "window-level sanity check: NOT a measurement of the clock during the timed collectives (the samples carry no phase attribution)",
        "min_clock_pct": min_clock_pct,
        "max_mhz": {str(rank): value for rank, value in sorted(max_clock.items())},
        "valid": valid,
    },
    "histogram_paths": [os.path.join(local_dir, "histogram-rank%d.json" % rank) for rank in range(4)],
    "ranks": ranks,
    "valid": valid,
    "pass": passed,
    "nccl": {"version": build[0], "library": build[1], "env": next(iter(envs.values()), None)},
    "gdr_active": bool(ranks) and all(entry["gdr"] and entry["gdr"]["active"] for entry in ranks),
    "gdr_required": any(entry.get("gdr_required") for entry in ranks),
    "gdr_platform": {str(entry["rank"]): (entry.get("platform") or {}) for entry in ranks},
    "buffers_registered": bool(ranks) and all((entry["registered_buffers"] or 0) >= 2 for entry in ranks),
    "clock_problems": clock_problems,
    "latency_problems": latency_problems,
    "transport_problems": transport_problems,
    "transport_notes": transport_problems_note,
}
with open(os.path.join(local_dir, "gate-summary.json"), "w") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
for entry in ranks:
    gdr = entry["gdr"] or {}
    print("rank %d: p50 %s us p90 %s us p99 %s us clock median %s MHz (%s non-idle, max %s) power %s W gdr %s/%s registered %s gate %s"
          % (entry["rank"], entry["p50_us"], entry["p90_us"], entry["p99_us"], entry.get("active_median_mhz"),
             entry.get("non_idle_samples"), entry.get("max_clock_mhz"), entry["power_w_mean"],
             gdr.get("gdr_channels"), gdr.get("net_channels"), entry["registered_buffers"],
             "PASS" if entry["gate_pass"] else "FAIL"))
print("nccl: %s (%s) env %s" % (summary["nccl"]["version"], summary["nccl"]["library"],
                                json.dumps(summary["nccl"]["env"], sort_keys=True)))
print("gate: %s" % ("PASS" if passed else ("FAIL" if valid else "INVALID")))
print("gate summary: %s" % os.path.join(local_dir, "gate-summary.json"))
# Notes are printed whether the gate passes or not: a note says what the gate could NOT establish
# (for example that the platform cannot drive GDR), and hiding it on a PASS is how a reader comes to
# believe the gate proved more than it did.
for note in transport_problems_note:
    print("gate note: %s" % note)
if passed:
    sys.exit(0)
for problem in clock_problems + latency_problems + transport_problems:
    print("gate problem: %s" % problem)
sys.exit(2 if not valid else 1)
PY
}

# A gate must not measure a cold fabric. Both RoCEv2 rails share 192.0.2.0/24, so a peer's
# neighbour entry can resolve on the wrong interface and the first use after any change pays for the
# resolution. Measured in this window: two runs taken immediately after an ARP flush had p99
# 161-167 us while the same script on a warm fabric measured 81-85 us -- a 2x tail difference caused
# by neighbour state, not by the collective. So every rank exchanges traffic with every peer on both
# rails first, then the gate waits for the entries to settle before it samples or times anything.
gate_warm_fabric() {
    local r i rest=""
    say "gate: warming neighbour state between all rank pairs on both rails"
    for i in 0 1 2 3; do
        rest="$rest ${NODES[$i]} ${RAIL2[$i]}"
    done
    for r in 0 1 2 3; do
        remote "$r" "for ip in$rest; do ping -c 3 -i 0.1 -W 1 \"\$ip\" >/dev/null 2>&1 || echo \"warm: \$ip unreachable\"; done; true" \
            || die "rank $r: neighbour warm-up failed"
    done
    sleep 3
}

gate() {
    [ -x "$BIN" ] || die "no cross-built bench at $BIN (run: $0 build)"
    [ -s "$LIB" ] || die "no vendored libnccl at $LIB (run: $0 fetch)"
    mkdir -p "$LOCAL_DIR"
    trap gate_cleanup EXIT
    preflight
    say "gate: clearing any clock lock and reading each rank's native boost clock"
    gate_lock_clocks
    gate_start_samplers
    gate_probe_platform
    gate_warm_fabric
    say "gate: device-only 10,240 B all-reduce at native boost clocks, $GATE_SAMPLES iterations, p50 <= $GATE_P50_US us, p90 <= $GATE_P90_US us, p99 <= $GATE_P99_US us, buffers registered, GDR required where the platform supports it"
    gate_launch
    gate_stop_samplers
    gate_collect
    for r in 0 1 2 3; do
        [ -s "$LOCAL_DIR/bench-rank$r.json" ] && say "--- rank $r JSON (gate fields at the end)" && cat "$LOCAL_DIR/bench-rank$r.json"
    done
    local status=0
    gate_verdict || status=$?
    say "gate: histogram files: $LOCAL_DIR/histogram-rank{0,1,2,3}.json; NCCL logs: $LOCAL_DIR/nccl-rank{0,1,2,3}.log"
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
