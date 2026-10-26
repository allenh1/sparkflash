#!/usr/bin/env bash
# Four-rank bind-check window run for the device arm (task-1 / M1, "no token yet" variant).
#
# WHAT THIS RUNS AND WHY
#   One process per node runs: sparkflash device-chain --run-stage tp4-chain --upload --loop.
#   That stage plans the rank's 72.308 GiB resident slice from checkpoint headers, streams the arena
#   into managed device memory, binds every device-served tensor set to it with set equality per
#   class and names in both directions (attention 208, routed gate/up 61,440 + down 30,720, shared
#   240, router 120, tail 3, plus a census over every resident shard), then opens the real four-rank
#   TCP collective and walks the 80 reduces of one 40-layer step, verifying the canonical bf16
#   ascending-rank sum byte for byte.
#
#   tp4-bind-r<R>.json is written progressively INSIDE the arena epoch (after the attention bind,
#   then after the MoE bind, then complete) so a later bind or walk failure cannot erase an earlier
#   result. NO token is produced and NO forward pass runs: forward=not-run, tokens=0,
#   hybrid_sequence.executed=false. Do not quote a tok/s number from this run.
#
#   --cubin-dir is deliberately NOT passed: with it, --loop would invoke the world=1 hybrid
#   diagnostic, which must not run in four-rank mode. The bind path needs no cubins.
#
# THE ONE COMMAND THAT MATTERS
#   scripts/tp4-window-bind.sh --print-argv <R>        # bind/walk argv, R in 0..3
#   scripts/tp4-window-bind.sh --print-argv-token <R>  # token argv (needs TOKEN_IDS, KERNEL_CUBIN_DIR)
#   scripts/tp4-window-bind.sh --rank-local-token <R>  # one rank with --forward, local binary
#   scripts/tp4-window-bind.sh all                # preflight+stage+launch+wait+collect+verdict
#   scripts/tp4-window-bind.sh --rank-local <R>   # run exactly one rank (Lead's runner reuse)
#
# TIMING (2026-10-20 window, same stage): arena boot 332.4-350.9 s/rank, header load ~10 s, walk is
# seconds. Plan ~6-7 min wall launch-to-evidence with a ~20 s stagger. WALL_LIMIT is a timeout.
#
# EVIDENCE CONTRACT (per rank dir)
#   tp4-bind-r<R>.json      schema sparkflash-tp4-bind/1; complete bind object
#   tp4-bind-names-r<R>.txt one line per checked name: "class<TAB>set<TAB>name"; 185,481 lines
#                           (185,462 class-set lines + 19 attention host-seam lines; no trailing newline)
#   tp4-loop-r<R>.json      schema sparkflash-tp4-loop/1; forward=not-run, tokens=0, walk
#   run.log / run.exit      stdout+stderr / exit code
#
# PASS/FAIL lines the verdict prints (grep these downstream)
#   PASS rank R: bind complete, 208/208 attention, 92420 moe names, census 95818 resident, 80 reduces/step
#   VERDICT: PASS - four-rank arena boot + full-rank bind coverage + 80-reduce walk, no token claimed
#   FAIL rank R: <named reason>
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO_ROOT/target/aarch64-unknown-linux-gnu/release/sparkflash}"
CHECKPOINT="${CHECKPOINT:-/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170}"
REMOTE_BIN="${REMOTE_BIN:-/home/user/sparkflash-bin/sparkflash}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/home/user/ev-session-20261020/device-lead/window-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-m1-bind}"
BASE_PORT="${BASE_PORT:-29601}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
DECODE_STEPS="${DECODE_STEPS:-1}"
WALL_LIMIT="${WALL_LIMIT:-1500}"
MEM_FLOOR_MIB="${MEM_FLOOR_MIB:-97280}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
SSH_TARGETS=(spark-0 198.51.100.88 198.51.100.59 198.51.100.68)
PEERS_CSV="198.51.100.80,198.51.100.88,198.51.100.59,198.51.100.68"
KERNEL_CUBIN_DIR="${KERNEL_CUBIN_DIR:-/home/user/sparkflash-bin/cubins}"
TOKEN_IDS="${TOKEN_IDS:-}"
# Teacher-forced reference token per dumped position (QA like-for-like vector). Optional: when set,
# the token argv carries --forward-reference-ids and every dump row gets its reference_id.
REFERENCE_IDS="${REFERENCE_IDS:-}"

usage() { sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
rank_evidence_dir() { printf '%s/%s/rank%s' "$EVIDENCE_ROOT" "$RUN_NAME" "$1"; }

# Single source of truth for the per-rank command line.
rank_argv() {
    local rank="$1" evidence
    evidence="$(rank_evidence_dir "$rank")"
    printf '%s --run-stage tp4-chain --checkpoint %s --tp-rank %s --tp-world-size 4 --evidence-dir %s --upload --loop --comm-peers %s --comm-base-port %s --comm-timeout 120 --comm-connect-timeout 120 --warmup-steps %s --decode-steps %s' \
        "$REMOTE_BIN" "$CHECKPOINT" "$rank" "$evidence" "$PEERS_CSV" "$BASE_PORT" "$WARMUP_STEPS" "$DECODE_STEPS"
}

mode_print_argv() {
    if [[ -z "${1:-}" ]]; then echo "usage: $0 --print-argv <rank 0..3>" >&2; return 2; fi
    rank_argv "$1"; echo
}

# The bind/walk argv plus the token stage's forward-only flags. --kernel-cubin-dir is the staged
# cubin directory and is deliberately NOT the older --cubin-dir (which invokes the world=1 hybrid
# diagnostic); --token-ids is the prompt prefix the step extends.
mode_print_token_argv() {
    local rank="${1:-}"
    if [[ -z "$rank" ]]; then echo "usage: $0 --print-argv-token <rank 0..3>" >&2; return 2; fi
    if [[ -z "$TOKEN_IDS" ]]; then
        echo "FAIL --print-argv-token: set TOKEN_IDS to the prompt prefix (comma separated)" >&2
        return 2
    fi
    printf '%s --forward --forward-preflight --kernel-cubin-dir %s --token-ids %s' \
        "$(rank_argv "$rank")" "$KERNEL_CUBIN_DIR" "$TOKEN_IDS"
    if [[ -n "$REFERENCE_IDS" ]]; then
        printf ' --forward-reference-ids %s' "$REFERENCE_IDS"
    fi
    echo
}

mode_rank_local_token() {
    local rank="${1:-}"
    if [[ -z "$rank" || "$rank" -ge 4 ]]; then echo "usage: $0 --rank-local-token <rank 0..3>" >&2; return 2; fi
    if [[ -z "$TOKEN_IDS" ]]; then
        echo "FAIL --rank-local-token: set TOKEN_IDS to the prompt prefix (comma separated)" >&2
        return 2
    fi
    local evidence; evidence="$(rank_evidence_dir "$rank")"
    mkdir -p "$evidence"
    echo "tp4-chain r$rank: evidence dir $evidence (token attempt)"
    exec "$BIN" --run-stage tp4-chain \
        --checkpoint "$CHECKPOINT" --tp-rank "$rank" --tp-world-size 4 \
        --evidence-dir "$evidence" --upload --loop \
        --comm-peers "$PEERS_CSV" --comm-base-port "$BASE_PORT" \
        --comm-timeout 120 --comm-connect-timeout 120 \
        --warmup-steps "$WARMUP_STEPS" --decode-steps "$DECODE_STEPS" \
        --forward --forward-preflight --kernel-cubin-dir "$KERNEL_CUBIN_DIR" \
        --token-ids "$TOKEN_IDS"
}

mode_rank_local() {
    local rank="${1:-}"
    if [[ -z "$rank" || "$rank" -ge 4 ]]; then echo "usage: $0 --rank-local <rank 0..3>" >&2; return 2; fi
    local evidence; evidence="$(rank_evidence_dir "$rank")"
    mkdir -p "$evidence"
    echo "tp4-chain r$rank: evidence dir $evidence"
    exec "$BIN" --run-stage tp4-chain \
        --checkpoint "$CHECKPOINT" --tp-rank "$rank" --tp-world-size 4 \
        --evidence-dir "$evidence" --upload --loop \
        --comm-peers "$PEERS_CSV" --comm-base-port "$BASE_PORT" \
        --comm-timeout 120 --comm-connect-timeout 120 \
        --warmup-steps "$WARMUP_STEPS" --decode-steps "$DECODE_STEPS"
}

remote() { local node="$1"; shift; ssh -n "${SSH_OPTS[@]}" "user@$node" "$@" </dev/null; }

mode_preflight() {
    local failures=0 rank node
    if [[ ! -x "$BIN" ]]; then
        echo "FAIL preflight: binary missing or not executable: $BIN (build: scripts/build-aarch64.sh --features=cuda)"
        return 1
    fi
    local bin_sha; bin_sha="$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "preflight: binary $BIN sha256 $bin_sha"
    for rank in 0 1 2 3; do
        node="${SSH_TARGETS[$rank]}"
        local checkpoint_ok mem_mib port_free
        checkpoint_ok="$(remote "$node" "test -f '$CHECKPOINT/config.json' && echo yes || echo no" 2>/dev/null || echo ssh-failed)"
        mem_mib="$(remote "$node" "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo" 2>/dev/null || echo 0)"
        port_free="$(remote "$node" "ss -ltn 2>/dev/null | grep -q ':$((BASE_PORT + rank)) ' && echo listening || echo free" 2>/dev/null || echo unknown)"
        printf 'preflight: rank %s node %-12s checkpoint=%s MemAvailable=%sMiB port=%s\n' "$rank" "$node" "$checkpoint_ok" "$mem_mib" "$port_free"
        [[ "$checkpoint_ok" == "yes" ]] || { echo "FAIL preflight: rank $rank missing checkpoint"; failures=$((failures + 1)); }
        [[ "${mem_mib:-0}" -ge "$MEM_FLOOR_MIB" ]] || { echo "FAIL preflight: rank $rank MemAvailable ${mem_mib}MiB < floor ${MEM_FLOOR_MIB}MiB"; failures=$((failures + 1)); }
        [[ "$port_free" != "listening" ]] || { echo "FAIL preflight: rank $rank port $((BASE_PORT + rank)) already listening"; failures=$((failures + 1)); }
    done
    [[ "$failures" -eq 0 ]] && echo "preflight: PASS"
    return "$failures"
}

mode_stage() {
    local rank node remote_dir bin_sha stage_sha
    bin_sha="$(sha256sum "$BIN" | cut -d' ' -f1)"
    for rank in 0 1 2 3; do
        node="${SSH_TARGETS[$rank]}"
        remote_dir="$(dirname "$REMOTE_BIN")"
        echo "stage: rank $rank -> $node:$REMOTE_BIN"
        remote "$node" "mkdir -p '$remote_dir'" || { echo "FAIL stage: rank $rank mkdir"; return 1; }
        scp -q -o BatchMode=yes -o ConnectTimeout=15 "$BIN" "user@$node:$REMOTE_BIN" || { echo "FAIL stage: rank $rank scp"; return 1; }
        remote "$node" "chmod +x '$REMOTE_BIN'" || { echo "FAIL stage: rank $rank chmod"; return 1; }
        stage_sha="$(remote "$node" "sha256sum '$REMOTE_BIN' | cut -d' ' -f1")"
        if [[ "$stage_sha" != "$bin_sha" ]]; then echo "FAIL stage: rank $rank sha mismatch staged=$stage_sha local=$bin_sha"; return 1; fi
        echo "stage: rank $rank ok sha $stage_sha"
    done
}

mode_launch() {
    local rank node evidence
    for rank in 0 1 2 3; do
        node="${SSH_TARGETS[$rank]}"
        evidence="$(rank_evidence_dir "$rank")"
        echo "launch: rank $rank on $node evidence $evidence"
        remote "$node" "mkdir -p '$evidence' && rm -f '$evidence/run.exit'" || { echo "FAIL launch: rank $rank prepare"; return 1; }
        remote "$node" "cd '$evidence' && nohup bash -c '$(rank_argv "$rank") > run.log 2>&1; echo \$? > run.exit' >/dev/null 2>&1 & echo launched" || { echo "FAIL launch: rank $rank"; return 1; }
    done
}

mode_wait() {
    local deadline=$((SECONDS + WALL_LIMIT)) rank node exit_code pending
    while :; do
        pending=0
        for rank in 0 1 2 3; do
            node="${SSH_TARGETS[$rank]}"
            exit_code="$(remote "$node" "cat '$(rank_evidence_dir "$rank")/run.exit' 2>/dev/null || echo running" 2>/dev/null || echo unreachable)"
            [[ "$exit_code" == "running" || "$exit_code" == "unreachable" ]] && pending=$((pending + 1))
        done
        [[ "$pending" -eq 0 ]] && return 0
        if [[ "$SECONDS" -ge "$deadline" ]]; then echo "FAIL wait: $pending rank(s) still running after ${WALL_LIMIT}s"; return 1; fi
        sleep 10
    done
}

mode_collect() {
    local rank node evidence out
    out="$EVIDENCE_ROOT/$RUN_NAME"
    for rank in 0 1 2 3; do
        node="${SSH_TARGETS[$rank]}"
        evidence="$(rank_evidence_dir "$rank")"
        mkdir -p "$out/rank$rank"
        echo "collect: rank $rank $node:$evidence -> $out/rank$rank"
        scp -q -o BatchMode=yes -o ConnectTimeout=15 -r "user@$node:$evidence/." "$out/rank$rank/" || { echo "FAIL collect: rank $rank"; return 1; }
    done
}

mode_verdict() {
    local out rank fails=0
    out="$EVIDENCE_ROOT/$RUN_NAME"
    for rank in 0 1 2 3; do
        local dir="$out/rank$rank" bind="$out/rank$rank/tp4-bind-r$rank.json" loop="$out/rank$rank/tp4-loop-r$rank.json" names="$out/rank$rank/tp4-bind-names-r$rank.txt" rc verdict
        rc="$(cat "$dir/run.exit" 2>/dev/null || echo missing)"
        if [[ "$rc" != "0" ]]; then echo "FAIL rank $rank: process exit $rc (see $dir/run.log)"; fails=$((fails + 1)); continue; fi
        if [[ ! -s "$bind" || ! -s "$loop" || ! -s "$names" ]]; then echo "FAIL rank $rank: missing bind/loop/names evidence"; fails=$((fails + 1)); continue; fi
        verdict="$(python3 - "$bind" "$loop" "$names" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); l = json.load(open(sys.argv[2]))
names = sum(1 for _ in open(sys.argv[3]))
checks = []
checks.append(("schema", b.get("schema") == "sparkflash-tp4-bind/1"))
checks.append(("summary.complete", b.get("summary", {}).get("complete") is True))
checks.append(("attention 208/208", b.get("attention", {}).get("tensors_bound") == 208 and b.get("attention", {}).get("tensors_expected") == 208))
moe = {row.get("class"): row for row in b.get("moe", {}).get("classes", [])}
for cls, count in {"routed_experts_gate_up": 61440, "routed_experts_down": 30720, "shared_experts": 240, "router": 120}.items():
    checks.append((cls, moe.get(cls, {}).get("bound") == count and moe.get(cls, {}).get("complete") is True))
checks.append(("census unclassified=0", b.get("census", {}).get("unclassified") == []))
checks.append(("census 95818", b.get("census", {}).get("resident_tensors") == 95818))
checks.append(("per-token weight H2D 0", b.get("moe", {}).get("weight_h2d_bytes_per_token") == 0))
checks.append(("names 185481", names == 185481))
checks.append(("forward=not-run", l.get("forward") == "not-run"))
checks.append(("tokens=0", l.get("tokens") == 0))
c = l.get("collectives", {})
checks.append(("collectives.performed", c.get("performed") is True))
checks.append(("80 reduces/step", c.get("calls_per_step_expected") == 80))
checks.append(("walk steps", len(c.get("steps", [])) == 3 and all(s.get("calls_per_step_ok") is True for s in c.get("steps", []))))
checks.append(("hybrid not executed", l.get("hybrid_sequence", {}).get("executed") is False))
bad = [n for n, ok in checks if not ok]
print("FAIL " + ", ".join(bad) if bad else "PASS bind complete, 208/208 attention, 92420 moe names, census 95818 resident, 80 reduces/step")
sys.exit(1 if bad else 0)
PY
)" || true
        if [[ "$verdict" == PASS* ]]; then echo "PASS rank $rank: ${verdict#PASS }"; else echo "FAIL rank $rank: $verdict"; fails=$((fails + 1)); fi
    done
    if [[ "$fails" -eq 0 ]]; then
        echo "VERDICT: PASS - four-rank arena boot + full-rank bind coverage + 80-reduce walk, no token claimed"
        return 0
    fi
    echo "VERDICT: FAIL - $fails rank(s); named names are in tp4-bind-r<R>.json missing/unexpected and run.log"
    return 1
}

main() {
    local mode="${1:-all}"
    case "$mode" in
        --print-argv) shift; mode_print_argv "${1:-}" ;;
        --print-argv-token) shift; mode_print_token_argv "${1:-}" ;;
        --rank-local-token) shift; mode_rank_local_token "${1:-}" ;;
        --rank-local) shift; mode_rank_local "${1:-}" ;;
        preflight) mode_preflight ;;
        stage) mode_stage ;;
        launch) mode_launch ;;
        wait) mode_wait ;;
        collect) mode_collect ;;
        verdict) mode_verdict ;;
        all)
            echo "window kit: run name '$RUN_NAME' evidence root $EVIDENCE_ROOT"
            mode_preflight || { echo "VERDICT: FAIL - preflight"; return 1; }
            mode_stage || { echo "VERDICT: FAIL - stage"; return 1; }
            mode_launch || { echo "VERDICT: FAIL - launch"; return 1; }
            echo "launch done; arena boot ~350 s/rank; polling exit codes every 10 s (limit ${WALL_LIMIT}s)"
            mode_wait || true
            mode_collect
            mode_verdict
            ;;
        -h|--help|help) usage ;;
        *) echo "unknown mode '$mode' (all|preflight|stage|launch|wait|collect|verdict|--print-argv|--rank-local)" >&2; return 2 ;;
    esac
}

main "$@"
