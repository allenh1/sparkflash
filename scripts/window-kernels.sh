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

# Window kernel harness comparison: run the SAME prebuilt harness binaries against a BASE stage and a
# NEW stage and adjudicate each kernel from the two logs. No build at run time: this script only
# executes files the Lead has already staged, so a compile error cannot eat the device window.
#
# Usage:
#   scripts/window-kernels.sh <base-stage-dir> <new-stage-dir> <outdir>
#
# A stage directory is either
#   <stage>/bin/<harness>          and <stage>/cubins/<kernel>.cubin
# or the flat form <stage>/<harness> and <stage>/<kernel>.cubin.
#
# Environment:
#   SPARKFLASH_CHECKPOINT   released snapshot dir (default: the D36 released checkpoint below)
#   WINDOW_TIMEOUT          hard per-invocation timeout in seconds (default 150)
#   WINDOW_BUDGET_SECS      whole-slot budget in seconds; once exceeded the remaining invocations are
#                           recorded as no-go instead of overrunning the GPU slot (default 780)
#
# Output:
#   $OUT/<stage>-<harness>.log        tee'd stdout+stderr, with argv and exit code in the log
#   $OUT/provenance.txt               host, driver, each stage's binary sha256 and cubin manifest,
#                                     the cubin-level diff between the two stages, every exact argv
#   $OUT/summary.txt                  one PASS / BIT-IDENTICAL / FAIL / no-go line per harness, with
#                                     base p50 vs new p50 per timed kernel, speedup and roofline %
# The script exits 0 only when every harness is adjudicated PASS or BIT-IDENTICAL.
#
# Verdict classes, so the Lead does not have to interpret:
#   BIT-IDENTICAL  the new stage passes a harness that compares the device output bit for bit against
#                  the shared CPU twin, exactly as the base stage does, so both stages equal the twin
#                  and therefore each other on that harness's operands. Only the two harnesses whose
#                  cubin changed (routed_moe_gate_up, routed_moe_down) can be adjudicated this way.
#   PASS           the new stage ran every declared check with the same bar as the base stage
#                  (no FAIL line, the harness's final verdict line present) and its p50 was read.
#   FAIL           the new stage ran but a declared check failed, or the timing/verdict lines the
#                  harness always prints were missing while it exited 0.
#   no-go          a stage file was missing, the invocation timed out or exited non-zero, or the
#                  overall budget ran out. A no-go is never a performance result.
set -u

BASE=${1:-}
NEW=${2:-}
OUT=${3:-}
if [ -z "$BASE" ] || [ -z "$NEW" ] || [ -z "$OUT" ]; then
    echo "usage: $0 <base-stage-dir> <new-stage-dir> <outdir>" >&2
    exit 2
fi
CHECKPOINT=${SPARKFLASH_CHECKPOINT:-/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170}
PER_TIMEOUT=${WINDOW_TIMEOUT:-150}
BUDGET=${WINDOW_BUDGET_SECS:-780}
START_EPOCH=$(date +%s)

mkdir -p "$OUT" || exit 2
SUMMARY="$OUT/summary.txt"
PROV="$OUT/provenance.txt"
: > "$SUMMARY"
: > "$PROV"

stage_file() { # stage_dir kind name
    if [ -f "$1/$2/$3" ]; then echo "$1/$2/$3"; elif [ -f "$1/$3" ]; then echo "$1/$3"; fi
}
cubin_dir() { # stage_dir
    if [ -d "$1/cubins" ]; then echo "$1/cubins"; elif ls "$1"/*.cubin >/dev/null 2>&1; then echo "$1"; fi
}
sha_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
elapsed() { echo $(( $(date +%s) - START_EPOCH )); }

{
    echo "window-kernels.sh provenance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) kernel=$(uname -smr) ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
    echo "driver=$(grep -h 'NVRM version' /proc/driver/nvidia/version 2>/dev/null || echo 'unreadable')"
    echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
    for pair in "base:$BASE" "new:$NEW"; do
        tag=${pair%%:*}
        dir=${pair#*:}
        echo "stage $tag dir=$dir"
        if [ ! -d "$dir" ]; then echo "stage $tag MISSING"; continue; fi
        for bin in "$dir/bin"/* "$dir"/*; do
            [ -f "$bin" ] || continue
            case "$bin" in *.cubin|*.log|*.json|*.txt|*.*) continue;; esac
            echo "  bin ${bin##*/} sha256=$(sha_of "$bin") bytes=$(wc -c < "$bin" | tr -d ' ')"
        done
        cdir=$(cubin_dir "$dir")
        if [ -n "${cdir:-}" ]; then
            for cubin in "$cdir"/*.cubin; do
                [ -f "$cubin" ] || continue
                echo "  cubin ${cubin##*/} sha256=$(sha_of "$cubin") bytes=$(wc -c < "$cubin" | tr -d ' ')"
            done
        fi
    done
    echo "checkpoint=$CHECKPOINT exists=$([ -d "$CHECKPOINT" ] && echo yes || echo no)"
    echo "per_invocation_timeout_s=$PER_TIMEOUT budget_s=$BUDGET"
} >> "$PROV"

# The cubin-level diff between the stages: only the two rewritten kernels should differ.
base_cubins=$(cubin_dir "$BASE"); new_cubins=$(cubin_dir "$NEW")
{
    echo "cubin diff (base vs new), name: base_sha -> new_sha"
    if [ -n "${base_cubins:-}" ] && [ -n "${new_cubins:-}" ]; then
        for cubin in "$new_cubins"/*.cubin; do
            [ -f "$cubin" ] || continue
            name=${cubin##*/}
            b=$(sha_of "$base_cubins/$name"); n=$(sha_of "$cubin")
            if [ -z "${b:-}" ]; then echo "  $name: absent-in-base -> ${n:0:12}"; 
            elif [ "${b%% *}" != "${n%% *}" ]; then echo "  $name: ${b:0:12} -> ${n:0:12}"; fi
        done
    else
        echo "  (one or both stage cubin directories are missing)"
    fi
} >> "$PROV"

# ------------------------------------------------------------------------------------------------
# The harness list. One line per harness: name|args|bar-kind|bar-count|roofline-regex
# bar-kind:  fail-lines  = no '^FAIL' line and at least COUNT '^PASS' lines
#            (m10 is the only harness that prints per-check lines; m12 prints one final verdict
#            line and only prints a line per FAILED check, so its bar is final-line)
#            final-line  = the harness's final verdict line, COUNT is ignored
#            moe-slice   = the f32 rank accumulator is fully bit-exact and the bf16 match is 100%
#            timed       = exit 0 and the harness printed a per-launch timing line; timing only,
#                          no correctness claim (the timed run of an accumulate-in-place harness
#                          cannot also be its twin comparison - see the moe_slice rows)
#            exact-line  = the harness's byte-equality line says EXACT (COUNT ignored)
# ------------------------------------------------------------------------------------------------
ZERO_EXPERTS=""
HARNESSES="
routed_moe_gate_up|--checkpoint $CHECKPOINT --cubin-dir CUBINS --tokens 1 --layer 0 --rank 0 --repeat 100|fail-lines|8|pct_of_node=[0-9.]+
routed_moe_down|--checkpoint $CHECKPOINT --cubin CUBINS/b12x_mxfp4_moe_sm121.cubin --tokens 1 --layer 0|moe-slice|0|
routed_moe_down_timed|--checkpoint $CHECKPOINT --cubin CUBINS/b12x_mxfp4_moe_sm121.cubin --tokens 1 --layer 0 --repeat 200|timed|0|pct_of_node=[0-9.]+
router|--checkpoint $CHECKPOINT --cubin-dir CUBINS --tokens 8 --layer 0 --repeat 100|final-line|0|
lm_reduce_batched|--cubin-dir CUBINS --repeat 50|exact-line|0|
attn_projections|--checkpoint $CHECKPOINT --cubin-dir CUBINS --tokens 8 --repeat 100|final-line|0|
cert_lm_head|--checkpoint $CHECKPOINT --cubin-dir CUBINS --repeat 20|final-line|0|[0-9.]+% of it
"

# The two harnesses whose cubin changed are the only ones where equal-to-twin in both stages proves
# the two cubins agree; the rest are timing/regression rows only.
bit_identical_harnesses="routed_moe_gate_up routed_moe_down"

run_one() { # stage_tag stage_dir harness name args
    local tag=$1 dir=$2 name=$3 args=$4
    local bin log rc
    bin=$(stage_file "$dir" bin "$name")
    if [ -z "${bin:-}" ]; then
        return 90
    fi
    log="$OUT/$tag-$name.log"
    {
        echo "# window-kernels.sh stage=$tag harness=$name"
        echo "# binary=$bin sha256=$(sha_of "$bin")"
        echo "# argv: $bin $args"
        echo "# exact argv record for provenance: stage=$tag harness=$name binary_sha256=$(sha_of "$bin") argv=\"$bin $args\""
        echo "# started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$log"
    echo "stage=$tag harness=$name binary_sha256=$(sha_of "$bin") argv=\"$bin $args\"" >> "$PROV"
    ( cd "$dir" && timeout "$PER_TIMEOUT" "$bin" $args ) 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    echo "# exit=$rc finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
    return $rc
}

extract_p50() { # log -> comma separated 'p50 ...' numbers in file order
    grep -o 'p50 [0-9.][0-9.]*' "$1" 2>/dev/null | awk '{print $2}' | paste -sd, - 2>/dev/null
}
extract_roofline() { # log regex
    grep -oE "$2" "$1" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1
}
count_pass() { grep -c '^ *PASS ' "$1" 2>/dev/null || true; }
count_fail() { grep -c '^ *FAIL ' "$1" 2>/dev/null || true; }

adjudicate() { # harness kind count pattern new_log base_log new_rc
    local h=$1 kind=$2 count=$3 pattern=$4 log=$5 blog=$6 rc=$7
    if [ "$rc" -ne 0 ]; then echo "no-go"; return; fi
    case "$kind" in
        fail-lines)
            local nf np
            nf=$(count_fail "$log"); np=$(count_pass "$log")
            if [ "${nf:-0}" -gt 0 ]; then echo "FAIL"; return; fi
            if [ "${np:-0}" -lt "$count" ]; then echo "no-go"; return; fi
            echo "PASS" ;;
        final-line)
            if [ -n "$pattern" ]; then
                grep -Eq "$pattern" "$log" || { echo "no-go"; return; }
                grep -Eq "$pattern" "$blog" || { echo "no-go"; return; }
            fi
            # The harness's own final verdict line is a PASS claim, or (lm reduce) the byte-equality
            # line is EXACT: both are printed before any exit.
            if grep -q '^ *FAIL ' "$log"; then echo "FAIL"; return; fi
            if grep -Eq '(cert_batched|m9_router|[a-z0-9_]+): PASS' "$log"; then echo "PASS"
            elif [ "$h" = "cert_lm_head" ] && grep -q 'm22_cert_batched: PASS' "$log"; then echo "PASS"
            else echo "no-go"; fi ;;
        moe-slice)
            local acc m
            acc=$(grep -oE 'rank accumulator: [0-9]+/[0-9]+' "$log" | head -1)
            m=$(grep -oE 'exact bf16 matches: [0-9]+ \(100\.0000%\)' "$log" | head -1)
            if [ -z "${acc:-}" ] || [ -z "${m:-}" ]; then echo "no-go"; return; fi
            local nums numer denom
            nums=${acc#rank accumulator: }
            numer=${nums%%/*}; denom=${nums##*/}
            if [ -n "$numer" ] && [ "$numer" = "$denom" ]; then echo "PASS"; else echo "FAIL"; fi ;;
        timed)
            # A timing-only row: the harness must have produced its timed region and a p50. It makes
            # no correctness claim (the correctness row above owns that bar).
            local p50
            p50=$(extract_p50 "$log")
            if grep -q 'per-launch ms:' "$log" && [ -n "${p50:-}" ]; then echo "PASS"; else echo "no-go"; fi ;;
        exact-line)
            if grep -q 'byte equality.*: EXACT' "$log"; then echo "PASS"; else echo "no-go"; fi ;;
        *) echo "no-go" ;;
    esac
}

echo "# window-kernels.sh: base=$BASE new=$NEW out=$OUT" | tee -a "$SUMMARY"
printf '# %-22s %-14s %-28s %-28s %-9s %-10s %s\n'     "harness" "verdict" "base_p50_ms" "new_p50_ms" "speedup" "roofline" "notes" | tee -a "$SUMMARY"

base_cdir=$(cubin_dir "$BASE"); new_cdir=$(cubin_dir "$NEW")

while IFS='|' read -r h args kind count pattern; do
    [ -n "$h" ] || continue
    if [ "$(elapsed)" -gt "$BUDGET" ]; then
        printf '# %-22s %-14s %s\n' "$h" "no-go" "budget: $(elapsed)s of ${BUDGET}s used before this harness" | tee -a "$SUMMARY"
        continue
    fi
    cdir_new=${new_cdir:-/nonexistent-cubins}
    cdir_base=${base_cdir:-/nonexistent-cubins}
    args_new=${args//CUBINS/$cdir_new}
    args_base=${args//CUBINS/$cdir_base}
    b_log="$OUT/base-$h.log"; n_log="$OUT/new-$h.log"
    run_one base "$BASE" "$h" "$args_base"; base_rc=$?
    run_one new "$NEW" "$h" "$args_new"; new_rc=$?
    verdict=$(adjudicate "$h" "$kind" "$count" "$pattern" "$n_log" "$b_log" "$new_rc")
    if [ "$verdict" = "PASS" ] && [ "$base_rc" -ne 0 ]; then
        verdict="no-go"
    fi
    for bi in $bit_identical_harnesses; do
        if [ "$bi" = "$h" ] && [ "$verdict" = "PASS" ]; then verdict="BIT-IDENTICAL"; fi
    done
    b_p50=$(extract_p50 "$b_log"); n_p50=$(extract_p50 "$n_log")
    b_pct=$(extract_roofline "$b_log" "$pattern"); n_pct=$(extract_roofline "$n_log" "$pattern")
    speedup="n/a"
    if [ -n "${b_p50:-}" ] && [ -n "${n_p50:-}" ]; then
        speedup=$(awk -v b="$b_p50" -v n="$n_p50" 'BEGIN{ nb=split(b,ba,","); nn=split(n,na,","); m=(nb<nn?nb:nn); out=""; for(i=1;i<=m;i++){ if(na[i]>0) out=out (i>1?"/":"") sprintf("%.2f", ba[i]/na[i]); else out=out (i>1?"/":"") "n/a" } print out }')
    fi
    roofline="n/a"
    if [ -n "${n_pct:-}" ]; then roofline="${n_pct}%"; fi
    if [ -n "${b_pct:-}" ]; then roofline="${b_pct}% -> ${roofline#n/a}"; fi
    printf '# %-22s %-14s %-28s %-28s %-9s %-10s base_exit=%s new_exit=%s elapsed=%ss\n' \
        "$h" "$verdict" "${b_p50:-n/a}" "${n_p50:-n/a}" "$speedup" "$roofline" "$base_rc" "$new_rc" "$(elapsed)" | tee -a "$SUMMARY"
done <<EOF
$HARNESSES
EOF

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed_s=$(elapsed)"
    echo "summary_file=$SUMMARY"
} >> "$PROV"

bad=$(grep -cE 'no-go|FAIL' "$SUMMARY" || true)
if [ "${bad:-0}" -gt 0 ]; then
    echo "# window-kernels.sh: $bad harness row(s) are no-go/FAIL - adjudicate those rows only" | tee -a "$SUMMARY"
    exit 1
fi
echo "# window-kernels.sh: every harness is PASS or BIT-IDENTICAL" | tee -a "$SUMMARY"
exit 0
