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

# The first-reference gate for one serving arm.
#
# The arm's served checks run first (scripts/served-gate.py all), then the first numerics reference
# is captured with scripts/numerics.py. The numerics tool is the registry's authority: it reads the
# server's numerics identity and writes the entry the registry expects, so this script does not
# duplicate the registry rules. An arm with an unwaived PartExec gap is refused there / by its
# registered entry, exactly as docs/engine/04-onboarding.md mechanism 6 says.
#
# This host has no GPU, so the default is --dry-run: the script prints the exact commands it would
# run and touches nothing. Pass --apply on the serving host to execute them.
#
# The generated conformance tests are not run by this script yet. As of this commit
# sparkflash_runtime::conformance::run_check is a stub: every generated #[ignore]d test panics with
# "needs <Arm>'s device fixture; run scripts/onboard-gate.sh <arm>". The script says so in its
# output instead of pretending they pass; wiring the runner to a fixture is the next half of
# phase 8.
#
# Device-test recipe on the CUDA host (this is what made the one real device test runnable):
#
#   # the build emits PTX, not every cubin; assemble what it produced into a scratch directory
#   for ptx in $(find target -name '*.ptx'); do
#       ptxas -arch=sm_121a "$ptx" -o "<scratch>/$(basename "${ptx%.ptx}").cubin"
#   done
#   export SPARKFLASH_KERNEL_DIR="<scratch>"
#   cargo test -p sparkflash-runtime -- --ignored
#
#   scripts/onboard-gate.sh <arm> [--url URL] [--port N] [--long FILE]
#                            [--reference FILE] [--model ORG/REPO] [--world N]
#                            [--evidence logprobs|text] [--strict]
#                            [--apply] [--force]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARM=""
URL="http://127.0.0.1:8000"
PORT="8000"
LONG="$ROOT/docs/engine/04-onboarding.md"
REFERENCE=""
MODEL=""
WORLD=""
EVIDENCE="logprobs"
STRICT=""
APPLY=0
FORCE=0

usage() {
    sed -n '16,56p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --url) URL="${2:?--url needs a value}"; shift 2 ;;
        --port) PORT="${2:?--port needs a value}"; shift 2 ;;
        --long) LONG="${2:?--long needs a path}"; shift 2 ;;
        --reference) REFERENCE="${2:?--reference needs a path}"; shift 2 ;;
        --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
        --world) WORLD="${2:?--world needs a value}"; shift 2 ;;
        --evidence) EVIDENCE="${2:?--evidence needs a value}"; shift 2 ;;
        --strict) STRICT="--strict"; shift ;;
        --apply) APPLY=1; shift ;;
        --dry-run) APPLY=0; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "onboard-gate: unknown option $1" >&2; usage >&2; exit 2 ;;
        *)
            if [ -n "$ARM" ]; then
                echo "onboard-gate: one arm, not two ($ARM and $1)" >&2
                exit 2
            fi
            ARM="$1"
            shift
            ;;
    esac
done

if [ -z "$ARM" ]; then
    usage >&2
    exit 2
fi

# The binary the gate reads ARMS through. A dry run on a host without a build still prints the
# commands; the arm is then its own slug unless --model is supplied.
BIN="${SPARKFLASH_BIN:-}"
if [ -z "$BIN" ]; then
    if [ -x "$ROOT/target/release/sparkflash" ]; then
        BIN="$ROOT/target/release/sparkflash"
    elif [ -x "$ROOT/target/debug/sparkflash" ]; then
        BIN="$ROOT/target/debug/sparkflash"
    fi
fi

if [ -z "$MODEL" ] && [ -n "$BIN" ] && [ -x "$BIN" ]; then
    row="$("$BIN" arms --list 2>/dev/null | awk -F'|' -v want="$ARM" '
        { for (i = 1; i <= NF; i++) { gsub(/^ +| +$/, "", $i) } if ($2 == want) print $3 "|" $7 }')"
    if [ -n "$row" ]; then
        MODEL="${row%%|*}"
        conversion="${row#*|}"
        if [ -z "$WORLD" ]; then
            world="$(printf '%s' "$conversion" | sed -n 's/^TP\([0-9][0-9]*\).*/\1/p')"
            WORLD="${world:-4}"
        fi
    fi
fi
MODEL="${MODEL:-$ARM}"
WORLD="${WORLD:-4}"

if [ -z "$REFERENCE" ]; then
    slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's#[^a-z0-9]\+#-#g; s#^-\+#\#; s#-\+$##')"
    REFERENCE="$ROOT/numerics/references/${slug}-tp${WORLD}-sm121a.json"
elif [ "${REFERENCE#/}" = "$REFERENCE" ]; then
    REFERENCE="$ROOT/$REFERENCE"
fi

if [ ! -f "$LONG" ]; then
    echo "onboard-gate: --long $LONG is not a file" >&2
    exit 2
fi
if [ "$EVIDENCE" != "logprobs" ] && [ "$EVIDENCE" != "text" ]; then
    echo "onboard-gate: --evidence is logprobs or text" >&2
    exit 2
fi

SERVED=(python3 "$ROOT/scripts/served-gate.py" all "$PORT" --long "$LONG" --chars 60000)
if [ -n "$STRICT" ]; then
    SERVED+=("$STRICT")
fi
NUMERICS=(python3 "$ROOT/scripts/numerics.py" capture --url "$URL" --out "$REFERENCE" --evidence "$EVIDENCE")

echo "onboard-gate: arm $ARM, model $MODEL, world $WORLD"
echo "onboard-gate: conformance runner NOT IMPLEMENTED: sparkflash_runtime::conformance::run_check is a stub, so the generated #[ignore] tests panic with \"needs <Arm>'s device fixture\"; this gate does not report them as passing."
echo "onboard-gate: device tests need SPARKFLASH_KERNEL_DIR pointed at a scratch cubin dir assembled with ptxas -arch=sm_121a from the build's .ptx (see --help)"
if [ -e "$REFERENCE" ] && [ "$FORCE" -ne 1 ]; then
    echo "onboard-gate: $REFERENCE exists; a first reference would be a recapture (pass --force)" >&2
    if [ "$APPLY" -eq 1 ]; then
        exit 1
    fi
fi
echo "onboard-gate: served checks:"
printf '  '; printf '%q ' "${SERVED[@]}"; printf '\n'
echo "onboard-gate: first numerics reference:"
printf '  '; printf '%q ' "${NUMERICS[@]}"; printf '\n'

if [ "$APPLY" -eq 0 ]; then
    echo
    echo "dry run; would run exactly:"
    printf '%q ' "${SERVED[@]}"
    printf '\n'
    printf '%q ' "${NUMERICS[@]}"
    printf '\n'
    exit 0
fi

echo
echo "+ "; printf '%q ' "${SERVED[@]}"; printf '\n'
"${SERVED[@]}"
echo "+ "; printf '%q ' "${NUMERICS[@]}"; printf '\n'
"${NUMERICS[@]}"
