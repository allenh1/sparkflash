#!/usr/bin/env bash
# WS-B fabric before/after probe: re-runnable evidence for the MTU 9000 + peermem change.
#
# The Lead applies the host side; this script produces the one table that decides keep/revert.
# It is read-only except for staging the existing probe binary into an evidence directory and
# running the probe itself. It NEVER stops or restarts vLLM and refuses a busy node.
#
# Subcommands:
#   state LABEL                 capture MTU / active_mtu / peermem / DF-8972 / busy state (read-only)
#   stage                       copy the cross-built rocenante-probe to the active ranks
#   probe PHASE                 run dual 10 KiB, single 10 KiB and dual 2 MiB four-rank groups
#   prepare-before              copy the archived 2026-10-19 MTU-1500 numbers as the before column
#   report                      one before/after markdown table with the keep/revert verdict
#   all PHASE                   state + stage + probe + report (PHASE=after for the change)
#   selftest                    loopback probe on this host (fabric=false, not the table)
#
# Environment: EVIDENCE_ROOT, RUN_ID, WORLDS (4, or 2 for the two-node hop), SAMPLES_LAT (200),
# SAMPLES_BW (50), WARMUP_LAT (20), WARMUP_BW (5), SPARK_WSB_ALLOW_BUSY (must stay 0).
#
# What "keep" means numerically (checked by report):
#   - DF-8972: 100% of directed pairs pass on both rails (24/24 for world 4; 4/4 for world 2)
#   - netdev MTU: 9000 on all fabric netdevs (8 for world 4)
#   - RDMA active_mtu: 4096 on all active f1 devices. NOTE: 9000 is NOT a valid RoCE MTU;
#     IB encodes 256/512/1024/2048/4096, so 4096 is the correct ceiling (max_mtu reads 4096).
#   - latency/bandwidth: informational; keep only if no regression >10%. A p50 <= 25 us on the
#     dual-rail 10 KiB row would additionally meet the WS-B gate and is called out separately.
# If DF-8972, netdev MTU or active_mtu fails after the change: REVERT, no partial state.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/home/user/phase2-evidence/ws-b-20261021}"
RUN_ID="${RUN_ID:-fabric-verify-$(date -u +%Y%m%dT%H%M%SZ)}"
DIR="$EVIDENCE_ROOT/$RUN_ID"
SSH_USER="${SSH_USER:-user}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
NODES=(198.51.100.80 198.51.100.88 198.51.100.59 198.51.100.68)
NAMES=(spark-0 spark-1 spark-2 spark-3)
PEERS="${SPARK_COMM_PEERS:-198.51.100.80,198.51.100.88,198.51.100.59,198.51.100.68}"
WORLD="${WORLDS:-4}"
CONTROL_PORT_BASE="${SPARK_COMM_CONTROL_PORT:-18517}"
SIZE_LAT=10240
SIZE_BW=2097152
SAMPLES_LAT="${SAMPLES_LAT:-200}"
SAMPLES_BW="${SAMPLES_BW:-50}"
WARMUP_LAT="${WARMUP_LAT:-20}"
WARMUP_BW="${WARMUP_BW:-5}"
ALLOW_BUSY="${SPARK_WSB_ALLOW_BUSY:-0}"
ARCHIVE="${BEFORE_ARCHIVE:-/home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19}"
TARGET=aarch64-unknown-linux-gnu
PROBE="$REPO_ROOT/target/$TARGET/release/rocenante-probe"
REMOTE_BIN="$DIR/bin/rocenante-probe"
declare -A RAIL1=()
declare -A RAIL2=()

say() { printf '%s\n' "$*"; }
die() { printf 'ws-b-fabric-verify: %s\n' "$*" >&2; exit 1; }

remote() {
    local rank="$1"; shift
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

remote_bg() {
    local rank="$1"; shift
    ssh -f -n "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" "$*"
}

ranks() { seq 0 "$((WORLD - 1))"; }

read_addrs() {
    local rank out
    for rank in $(ranks); do
        out="$(remote "$rank" "ip -4 -br addr show dev enp1s0f1np1; ip -4 -br addr show dev enP2p1s0f1np1" 2>/dev/null || true)"
        RAIL1["$rank"]="$(printf '%s\n' "$out" | awk '$1=="enp1s0f1np1"{split($3,a,"/"); print a[1]}')"
        RAIL2["$rank"]="$(printf '%s\n' "$out" | awk '$1=="enP2p1s0f1np1"{split($3,a,"/"); print a[1]}')"
        [ -n "${RAIL1[$rank]:-}" ] || die "rank $rank has no rail-1 address on enp1s0f1np1"
        [ -n "${RAIL2[$rank]:-}" ] || die "rank $rank has no rail-2 address on enP2p1s0f1np1"
    done
}

peer_table() {
    IFS=, read -r -a _arr <<< "$PEERS"
    printf '%s' "$(IFS=,; echo "${_arr[*]:0:WORLD}")"
}

check_idle() {
    local rank gpu port
    for rank in $(ranks); do
        gpu="$(remote "$rank" "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true")"
        port="$(remote "$rank" "ss -ltnH 'sport = :8000' 2>/dev/null | wc -l")"
        if [ "$gpu" != "0" ] || [ "$port" != "0" ]; then
            if [ "$ALLOW_BUSY" != "1" ]; then
                die "rank $rank is busy (gpu_apps=$gpu port8000=$port); refusing to measure beside vLLM. Wait for the quiet window."
            fi
            say "WARNING: busy slot accepted via SPARK_WSB_ALLOW_BUSY=1; numbers are not a clean fabric baseline"
        fi
    done
}

need_probe() {
    [ -x "$PROBE" ] || die "no cross-built probe at $PROBE; build it with cargo +1.98.1 build --release -p sparkflash-comm --features rdma --target $TARGET --bin rocenante-probe"
    say "probe: $PROBE ($(sha256sum "$PROBE" | cut -d' ' -f1))"
}

stage() {
    need_probe
    mkdir -p "$DIR/results"
    local rank
    for rank in $(ranks); do
        remote "$rank" "mkdir -p '$DIR/bin' '$DIR/results'"
        scp -q "$PROBE" "$SSH_USER@${NODES[$rank]}:$REMOTE_BIN"
        remote "$rank" "chmod +x '$REMOTE_BIN'"
    done
    say "staged probe on $WORLD ranks under $DIR"
}

state() {
    local label="${1:-before}"
    mkdir -p "$DIR"
    read_addrs
    local out="$DIR/state-$label.txt"
    : > "$out"
    echo "# DF matrix: same-rail directed pairs only. rail1 netdev enp1s0f1np1 (one address each), rail2 enP2p1s0f1np1 (one address each). Cross-rail pairs cross separate L2 islands sharing 198.51.100.0/24 and are not an MTU test; they are not counted." >> "$out"
    local rank peer ri addr r1 r2
    for rank in $(ranks); do
        {
            echo "## rank $rank ${NAMES[$rank]} ${NODES[$rank]}"
            ssh "${SSH_OPTS[@]}" "$SSH_USER@${NODES[$rank]}" 'bash -s' <<'REMOTE'
for n in enp1s0f1np1 enP2p1s0f1np1 enp1s0f0np0 enP2p1s0f0np0; do
    if [ -e "/sys/class/net/$n" ]; then echo "NETDEV $n mtu=$(cat /sys/class/net/$n/mtu)"; else echo "NETDEV $n mtu=absent"; fi
done
for d in rocep1s0f1 roceP2p1s0f1 rocep1s0f0 roceP2p1s0f0; do
    if [ -d "/sys/class/infiniband/$d" ]; then
        state="$(cat /sys/class/infiniband/$d/ports/1/state | cut -d: -f1)"
        active="?"
        max="?"
        line="$(ibv_devinfo -v -d $d 2>/dev/null | grep -i active_mtu | head -1)"
        [ -n "$line" ] && active="$(printf '%s' "$line" | awk '{print $2}')"
        line="$(ibv_devinfo -v -d $d 2>/dev/null | grep -i max_mtu | head -1)"
        [ -n "$line" ] && max="$(printf '%s' "$line" | awk '{print $2}')"
        echo "RDMA $d state=$state active_mtu=$active max_mtu=$max"
    fi
done
echo "PEERMEM count=$(lsmod | grep -cE '^(nvidia_peermem|nv_peer_mem)' || true)"
echo "GPUAPPS $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
echo "PORT8000 $(ss -ltnH 'sport = :8000' 2>/dev/null | wc -l)"
REMOTE
        } >> "$out"
        for ri in 1 2; do
            for peer in $(ranks); do
                [ "$peer" = "$rank" ] && continue
                if [ "$ri" = "1" ]; then addr="${RAIL1[$peer]}"; iface=enp1s0f1np1; else addr="${RAIL2[$peer]}"; iface=enP2p1s0f1np1; fi
                if remote "$rank" "ping -M do -s 8972 -I $iface -c 1 -W 2 $addr >/dev/null 2>&1"; then r1=pass; else r1=fail; fi
                if remote "$rank" "ping -M do -s 1472 -I $iface -c 1 -W 2 $addr >/dev/null 2>&1"; then r2=pass; else r2=fail; fi
                echo "DF rail$ri $rank->$peer mtu8972=$r1 mtu1472=$r2" >> "$out"
            done
        done
    done
    say "wrote $out"
}

probe_phase() {
    local phase="${1:-after}"
    check_idle
    stage
    local peers; peers="$(peer_table)"
    local idx=0 mode size port rank samples warmup rail_env waited ready
    for mode in dual single; do
        for size in "$SIZE_LAT" "$SIZE_BW"; do
            [ "$mode" = "single" ] && [ "$size" = "$SIZE_BW" ] && continue
            port="$((CONTROL_PORT_BASE + idx))"
            idx=$((idx + 1))
            if [ "$size" = "$SIZE_LAT" ]; then samples="$SAMPLES_LAT"; warmup="$WARMUP_LAT"; else samples="$SAMPLES_BW"; warmup="$WARMUP_BW"; fi
            say "run $phase $mode $size B: $WORLD ranks, samples=$samples warmup=$warmup port=$port"
            rail_env="SPARK_COMM_RAIL_COUNT=2"
            [ "$mode" = "single" ] && rail_env="SPARK_COMM_RAIL_COUNT=1"
            for rank in $(ranks); do
                remote_bg "$rank" "cd '$DIR' && nohup env $rail_env SPARK_COMM_PEERS='$peers' SPARK_COMM_CONTROL_PORT=$port SPARK_COMM_TIMEOUT_MS=5000 ./bin/rocenante-probe --rank $rank --world $WORLD --peer '$peers' --bytes $size --samples $samples --warmup $warmup --json > 'results/lat-$phase-$mode-$size-rank$rank.json' 2> 'results/lat-$phase-$mode-$size-rank$rank.err' < /dev/null &"
            done
            waited=0
            while [ "$waited" -lt 180 ]; do
                ready=0
                for rank in $(ranks); do
                    if remote "$rank" "test -s '$DIR/results/lat-$phase-$mode-$size-rank$rank.json'" 2>/dev/null; then ready=$((ready + 1)); fi
                done
                [ "$ready" -eq "$WORLD" ] && break
                sleep 3
                waited=$((waited + 3))
            done
            if [ "$ready" -ne "$WORLD" ]; then
                for rank in $(ranks); do remote "$rank" "tail -n 3 'results/lat-$phase-$mode-$size-rank$rank.err' 2>/dev/null || true"; done
                die "$phase $mode $size: only $ready/$WORLD ranks produced a JSON; no number reported"
            fi
            for rank in $(ranks); do
                scp -q "$SSH_USER@${NODES[$rank]}:$DIR/results/lat-$phase-$mode-$size-rank$rank.json" "$DIR/results/" 2>/dev/null || true
                scp -q "$SSH_USER@${NODES[$rank]}:$DIR/results/lat-$phase-$mode-$size-rank$rank.err" "$DIR/results/" 2>/dev/null || true
            done
            say "collected $phase $mode $size"
            sleep 1
        done
    done
}

prepare_before() {
    mkdir -p "$DIR/results-before"
    [ -d "$ARCHIVE" ] || die "before archive $ARCHIVE not found; cannot build the before column"
    local rank src
    for rank in 0 1 2 3; do
        src="$ARCHIVE/rank$rank/probe-rank$rank.json"
        [ -f "$src" ] && cp "$src" "$DIR/results-before/lat-before-dual-$SIZE_LAT-rank$rank.json"
        src="$ARCHIVE/rank$rank/probe-1rail-rank$rank.json"
        [ -f "$src" ] && cp "$src" "$DIR/results-before/lat-before-single-$SIZE_LAT-rank$rank.json"
        src="$ARCHIVE/rank$rank/probe-2mib-rank$rank.json"
        [ -f "$src" ] && cp "$src" "$DIR/results-before/lat-before-dual-$SIZE_BW-rank$rank.json"
    done
    say "prepared before column from $ARCHIVE (2026-10-19, MTU 1500, GPUDirect off)"
}

report() {
    [ -d "$DIR/results" ] || die "no after results in $DIR; run: $0 probe after"
    [ -d "$DIR/results-before" ] || prepare_before
    local out="$DIR/before-after.md"
    python3 - "$DIR" "$WORLD" "$out" "$ARCHIVE" <<'PY'
import glob, json, os, statistics, sys

dir_path, world, out_path, archive = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
SIZE_LAT, SIZE_BW = 10240, 2097152

def load(directory, mode, size, phase):
    files = sorted(glob.glob(os.path.join(directory, f"lat-{phase}-{mode}-{size}-rank*.json")))
    rows = []
    for path in files:
        try:
            with open(path) as fh:
                rows.append(json.load(fh))
        except Exception as exc:
            print(f"warning: unreadable {path}: {exc}", file=sys.stderr)
    if not rows:
        return None
    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.median(vals) if vals else None
    return {
        "n": len(rows),
        "p50": med("oneshot_allreduce_us"),
        "p90": med("p90_us"),
        "p99": med("p99_us"),
        "min": min((r["min_us"] for r in rows if isinstance(r.get("min_us"), (int, float))), default=None),
        "samples": min((r.get("samples", 0) for r in rows), default=0),
        "p50_values": [r["oneshot_allreduce_us"] for r in rows if isinstance(r.get("oneshot_allreduce_us"), (int, float))],
    }

def read_state(directory):
    path = os.path.join(directory, "state-after.txt")
    if not os.path.exists(path):
        cands = sorted(glob.glob(os.path.join(directory, "state-*.txt")))
        path = cands[-1] if cands else None
    if not path:
        return None
    net, rdma, df, busy, peermem = [], [], [], {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("NETDEV "):
                parts = line.split()
                net.append(parts[1:])
            elif line.startswith("RDMA "):
                parts = line.split()
                rdma.append(parts[1:])
            elif line.startswith("DF "):
                parts = line.split()
                df.append(parts[1:])
            elif line.startswith("PEERMEM "):
                peermem.append(int(line.split("=", 1)[1]))
            elif line.startswith("GPUAPPS "):
                busy["gpu"] = max(int(busy.get("gpu", 0)), int(line.split()[1]))
            elif line.startswith("PORT8000 "):
                busy["port"] = max(int(busy.get("port", 0)), int(line.split()[1]))
    return {"net": net, "rdma": rdma, "df": df, "busy": busy, "peermem": peermem}

def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def fmt(stat, key, digits=2):
    if not stat or stat.get(key) is None:
        return "not run"
    return f"{stat[key]:.{digits}f}"

def spread(stat, key="p50_values", digits=2):
    if not stat or not stat.get(key):
        return ""
    vals = stat[key]
    if len(vals) < 2:
        return ""
    return f" ({min(vals):.{digits}f}-{max(vals):.{digits}f})"

def delta(a, b):
    if not a or not b or not a.get("p50") or not b.get("p50"):
        return "n/a"
    return f"{100.0 * (a['p50'] / b['p50'] - 1):+.1f}%"

def delta_key(a, b, key):
    if not a or not b or not a.get(key) or not b.get(key):
        return "n/a"
    return f"{100.0 * (a[key] / b[key] - 1):+.1f}%"

def bw(stat):
    if not stat or not stat.get("p50"):
        return None, None
    posted = SIZE_BW * (world - 1)
    gbps = posted * 8 / (stat["p50"] * 1e-6) / 1e9
    return gbps, 100.0 * gbps / 400.0

after = read_state(dir_path)
d_after = load(os.path.join(dir_path, "results"), "dual", SIZE_LAT, "after")
s_after = load(os.path.join(dir_path, "results"), "single", SIZE_LAT, "after")
b_after = load(os.path.join(dir_path, "results"), "dual", SIZE_BW, "after")
d_before = load(os.path.join(dir_path, "results-before"), "dual", SIZE_LAT, "before")
s_before = load(os.path.join(dir_path, "results-before"), "single", SIZE_LAT, "before")
b_before = load(os.path.join(dir_path, "results-before"), "dual", SIZE_BW, "before")
two_node = world != 4
if two_node:
    # The only archived baseline is four-rank; a two-rank all-reduce has a different exchange, so
    # do not print a misleading delta. The state gate below is the decision for the hop test.
    d_before = s_before = b_before = None

bw_after, bw_after_pct = bw(b_after)
bw_before, _ = bw(b_before)

net_total = len(after["net"]) if after else 0
net_ok = sum(1 for row in (after["net"] if after else []) if row[1] == "mtu=9000")
active = [row for row in (after["rdma"] if after else []) if row[0] in ("rocep1s0f1", "roceP2p1s0f1")]
active_ok = 0
max_ok = 0
for row in active:
    fields = dict(item.split("=", 1) for item in row[1:])
    active_ok += 1 if fields.get("active_mtu") == "4096" else 0
    max_ok += 1 if fields.get("max_mtu") == "4096" else 0
df_rows = after["df"] if after else []
df_total = len(df_rows)
df_ok = sum(1 for row in df_rows if "mtu8972=pass" in row)
pm_counts = after["peermem"] if after else []
pm_ok = sum(1 for count in pm_counts if count >= 1)

lines = []
lines.append(f"# WS-B fabric before/after: {os.path.basename(dir_path)}")
lines.append("")
if two_node:
    lines.append("Two-node hop run (WORLDS=2, ranks .80 and .88): the archived baseline is four-rank, so the before column is withheld rather than compared to a different exchange. The state gate below is the decision.")
else:
    lines.append("Before = archived 2026-10-19 run at MTU 1500 / GPUDirect off (" + archive + "), same probe and sizes; it is not a fresh measurement. After = this run. Latency is p50 across ranks with per-rank min-max in parentheses; n is samples per rank.")

lines.append("")
lines.append("| Metric | Before | After | Delta | Keep rule |")
lines.append("| --- | --- | --- | --- | --- |")
lines.append(f"| 10 KiB p50 dual rail (us) | {fmt(d_before, 'p50')}{spread(d_before)} | {fmt(d_after, 'p50')}{spread(d_after)} | {delta(d_after, d_before)} | no regression >10% |")
lines.append(f"| 10 KiB p90 dual rail (us) | {fmt(d_before, 'p90')} | {fmt(d_after, 'p90')} | {delta_key(d_after, d_before, 'p90')} | report only |")
lines.append(f"| 10 KiB p99 dual rail (us) | {fmt(d_before, 'p99')} | {fmt(d_after, 'p99')} | n/a | tail watch (archived schema has no dual p99) |")
lines.append(f"| 10 KiB p50 single rail (us) | {fmt(s_before, 'p50')}{spread(s_before)} | {fmt(s_after, 'p50')}{spread(s_after)} | {delta(s_after, s_before)} | no regression >10% |")
lines.append(f"| 10 KiB p99 single rail (us) | {fmt(s_before, 'p99')} | {fmt(s_after, 'p99')} | {delta_key(s_after, s_before, 'p99')} | tail watch |")
lines.append(f"| 2 MiB p50 dual rail (us) | {fmt(b_before, 'p50')} | {fmt(b_after, 'p50')} | {delta(b_after, b_before)} | no regression >10% |")
lines.append(f"| 2 MiB p90 / p99 dual rail (us) | {fmt(b_before, 'p90')} / {fmt(b_before, 'p99')} | {fmt(b_after, 'p90')} / {fmt(b_after, 'p99')} | {delta_key(b_after, b_before, 'p90')} / n/a | tail watch |")
before_bw_s = f"{bw_before:.2f}" if bw_before is not None else "not run"
after_bw_s = (
    f"{bw_after:.2f} ({bw_after_pct:.1f}% of 400G)" if bw_after is not None else "not run"
)
bw_delta = (
    f"{100.0 * (bw_after / bw_before - 1):+.1f}%" if bw_after is not None and bw_before else "n/a"
)
lines.append(f"| 2 MiB posted bandwidth (Gb/s) | {before_bw_s} | {after_bw_s} | {bw_delta} | no regression >10% |")
lines.append(f"| samples per rank (10 KiB / 2 MiB) | {d_before['samples'] if d_before else 'n/a'} / {b_before['samples'] if b_before else 'n/a'} | {d_after['samples'] if d_after else 'n/a'} / {b_after['samples'] if b_after else 'n/a'} | n/a | stated sample counts |")
lines.append("")
lines.append("## State gate (the keep/revert decision)")
lines.append("")
lines.append("| Check | After readings | Pass |")
lines.append("| --- | --- | --- |")
lines.append(f"| fabric netdev MTU = 9000 | {net_ok}/{net_total} | {'yes' if net_total and net_ok == net_total else 'NO'} |")
lines.append(f"| RDMA active_mtu = 4096 (9000 is not a valid RoCE MTU) | {active_ok}/{len(active)} | {'yes' if active and active_ok == len(active) else 'NO'} |")
lines.append(f"| RDMA max_mtu = 4096 (ceiling, informational) | {max_ok}/{len(active)} | info |")
lines.append(f"| DF-8972 same-rail directed pairs pass (both rails) | {df_ok}/{df_total} | {'yes' if df_total and df_ok == df_total else 'NO'} |")
lines.append(f"| nvidia_peermem loaded (not supported on GB10/580: modprobe EINVAL) | {pm_ok}/{len(pm_counts)} | info - not a gate |")
busy = after["busy"] if after else {}
lines.append(f"| busy slot (max gpu_compute_apps / :8000) | {busy.get('gpu', '?')} / {busy.get('port', '?')} | must be 0 / 0 |")
lines.append("")
keep = bool(
    net_total
    and net_ok == net_total
    and active
    and active_ok == len(active)
    and df_total
    and df_ok == df_total
)
lines.append(f"**Verdict: {'KEEP' if keep else 'REVERT / NOT READY'}** - keep only if the three state gates pass (netdev 9000, active_mtu 4096, same-rail DF-8972 100%); latency rows are secondary. nvidia-peermem is not a gate: it does not load on GB10/580 (modprobe EINVAL), so GPUDirect has to come from the dma-buf path. If a gate fails, revert (runtime ip link, restore netplan backup).")
if d_after and d_after.get("p50") is not None and d_after["p50"] <= 25.0:
    lines.append("")
    lines.append("**WS-B gate: MET on this run** - dual-rail 10 KiB p50 <= 25 us; still report the sample count and conditions.")
lines.append("")
table = "\n".join(lines)
with open(out_path, "w") as fh:
    fh.write(table + "\n")
print(table)
PY
    say "wrote $out"
}

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"
[ -n "$cmd" ] || { usage; exit 2; }
shift || true
case "$cmd" in
    state) state "${1:-before}" ;;
    stage) stage ;;
    probe) probe_phase "${1:-after}" ;;
    prepare-before) prepare_before ;;
    report) report ;;
    all) state after; probe_phase after; report ;;
    selftest) "$REPO_ROOT/target/debug/rocenante-probe" --transport loopback --bytes 10240 --samples 20 --warmup 2 --json ;;
    *) usage; exit 2 ;;
esac
