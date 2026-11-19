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
# The arm's generated conformance checks run first, then its served checks
# (scripts/served-gate.py all), then the first numerics reference is captured with
# scripts/numerics.py. The numerics tool is the registry's authority: it reads the server's
# numerics identity and writes the entry the registry expects, so this script does not duplicate
# the registry rules. An arm with an unwaived PartExec gap is refused there / by its registered
# entry, exactly as docs/engine/04-onboarding.md mechanism 6 says.
#
# The conformance checks boot the arm's own fixture, not a served server:
#   dense     SPARKFLASH_CONFORMANCE_CHECKPOINT, or a synthetic checkpoint this script writes
#             with scripts/dense-synthetic-checkpoint.py from SPARKFLASH_TOKENIZER_DIR
#   GLM       the synthetic small checkpoint the arm writes
#   Qwen      the synthetic tiny checkpoint the arm writes
#   Inkling   the tiny source fixture the device tests share
# A check with no fixture on this host prints NEEDS-FIXTURE and fails the gate; it never passes
# silently. TP 1 = TP 2 = TP 4 prints SKIP with the reason when there is no group.
#
# This host has no GPU, so the default is --dry-run: the script prints the exact commands it would
# run and touches nothing. Pass --apply on the serving host to execute them.
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
#                            [--crate CRATE] [--fixture PATH]
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
CRATE=""
FIXTURE=""

usage() {
    sed -n '15,62p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
        --crate) CRATE="${2:?--crate needs a crate}"; shift 2 ;;
        --fixture) FIXTURE="${2:?--fixture needs a path}"; shift 2 ;;
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

# The arm's conformance crate. A new arm names its crate with --crate; the four landed arms are
# derived from their names so the gate cannot run another crate's checks by accident.
if [ -z "$CRATE" ]; then
    lower="$(printf '%s' "$ARM" | tr '[:upper:]' '[:lower:]')"
    case "$lower" in
        dense|acme-dense) CRATE="sparkflash-model-dense" ;;
        glm|glm5|glm5-next|glm-5.3-flash|glm53-flash|glm5_next)
            CRATE="sparkflash-model-glm5-next" ;;
        qwen|qwen4|qwen4-exp|qwen3.8-flash-next|qwen4_exp)
            CRATE="sparkflash-model-qwen4-exp" ;;
        inkling|inkling2|inkling-2-small|inkling2-small)
            CRATE="sparkflash-model-inkling2" ;;
        runtime) CRATE="sparkflash-runtime" ;;
        *)
            echo "onboard-gate: no conformance crate for arm \"$ARM\"; pass --crate" >&2
            exit 2
            ;;
    esac
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
        { for (i = 1; i <= NF; i++) { gsub(/^ +| +$/, "", $i) } if ($2 == want) print $3 }')"
    if [ -n "$row" ]; then
        MODEL="$row"
    fi
fi
MODEL="${MODEL:-$ARM}"
# The route's conversion default is not its serving width (Dory converts at TP1 and serves the
# four-rank group), so the first reference is taken at the group's world unless --world says
# otherwise.
WORLD="${WORLD:-4}"

if [ -z "$REFERENCE" ]; then
    slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's#[^a-z0-9]\+#-#g; s#^-\+##; s#-\+$##')"
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

# The dense arm's fixture is a checkpoint, not a type: write the one the script has always used
# when the caller did not name a path. The tokenizer only fixes the vocabulary; the checks read
# token ids, not words.
if [ -z "$FIXTURE" ] && [ "$CRATE" = "sparkflash-model-dense" ]; then
    FIXTURE="${SPARKFLASH_CONFORMANCE_CHECKPOINT:-$ROOT/target/conformance/dense-synthetic}"
fi
if [ -n "$FIXTURE" ]; then
    export SPARKFLASH_CONFORMANCE_CHECKPOINT="$FIXTURE"
fi

SERVED=(python3 "$ROOT/scripts/served-gate.py" all "$PORT" --long "$LONG" --chars 60000)
if [ -n "$STRICT" ]; then
    SERVED+=("$STRICT")
fi
NUMERICS=(python3 "$ROOT/scripts/numerics.py" capture --url "$URL" --out "$REFERENCE" --evidence "$EVIDENCE")
CONFORMANCE=(cargo test -p "$CRATE" --lib "conformance::" -- --ignored --nocapture --test-threads=1)

FIXTURE_CMD=()
if [ "$CRATE" = "sparkflash-model-dense" ] && [ ! -d "$FIXTURE" ]; then
    TOKENIZER="${SPARKFLASH_TOKENIZER_DIR:-$ROOT/reference}"
    FIXTURE_CMD=(python3 "$ROOT/scripts/dense-synthetic-checkpoint.py" "$TOKENIZER" "$FIXTURE"
        --layers 2 --hidden 1024 --intermediate 2816 --context 512)
fi

echo "onboard-gate: arm $ARM, model $MODEL, world $WORLD, crate $CRATE"
if [ -n "$FIXTURE" ]; then
    echo "onboard-gate: fixture $FIXTURE"
fi
echo "onboard-gate: device tests need SPARKFLASH_KERNEL_DIR pointed at a scratch cubin dir assembled with ptxas -arch=sm_121a from the build's .ptx (see --help)"
if [ -e "$REFERENCE" ] && [ "$FORCE" -ne 1 ]; then
    echo "onboard-gate: $REFERENCE exists; a first reference would be a recapture (pass --force)" >&2
fi
echo "onboard-gate: conformance checks:"
printf '  '; printf '%q ' "${CONFORMANCE[@]}"; printf '\n'
if [ "${#FIXTURE_CMD[@]}" -gt 0 ]; then
    echo "onboard-gate: writes the dense fixture first:"
    printf '  '; printf '%q ' "${FIXTURE_CMD[@]}"; printf '\n'
fi
echo "onboard-gate: served checks:"
printf '  '; printf '%q ' "${SERVED[@]}"; printf '\n'
echo "onboard-gate: first numerics reference:"
printf '  '; printf '%q ' "${NUMERICS[@]}"; printf '\n'

if [ "$APPLY" -eq 0 ]; then
    echo
    echo "dry run; would run exactly:"
    if [ "${#FIXTURE_CMD[@]}" -gt 0 ]; then
        printf '%q ' "${FIXTURE_CMD[@]}"; printf '\n'
    fi
    printf '%q ' "${CONFORMANCE[@]}"; printf '\n'
    printf '%q ' "${SERVED[@]}"; printf '\n'
    printf '%q ' "${NUMERICS[@]}"; printf '\n'
    exit 0
fi

if [ "${#FIXTURE_CMD[@]}" -gt 0 ]; then
    echo
    echo "+ "; printf '%q ' "${FIXTURE_CMD[@]}"; printf '\n'
    "${FIXTURE_CMD[@]}"
fi

echo
echo "+ "; printf '%q ' "${CONFORMANCE[@]}"; printf '\n'
conformance_log="$(mktemp)"
set +e
"${CONFORMANCE[@]}" >"$conformance_log" 2>&1
conformance_exit=$?
set -e
cat "$conformance_log"

echo
echo "onboard-gate: conformance result (crate $CRATE):"
report="$(grep -E 'conformance: .*: (PASS|SKIP|FAIL|NEEDS-FIXTURE)(: |$)' "$conformance_log" || true)"
if [ -z "$report" ]; then
    echo "  FAIL  the conformance target produced no check result (exit $conformance_exit); see the log above"
    conformance_exit=1
else
    printf '%s\n' "$report" | sed -e 's/^.*conformance: /conformance: /' -e 's/^conformance: //' -e 's/: PASS$/  PASS/' \
        -e 's/: SKIP: /  SKIP  /' -e 's/: FAIL: /  FAIL  /' \
        -e 's/: NEEDS-FIXTURE: /  NEEDS-FIXTURE  /' | awk '{ printf "  %s\n", $0 }'
    if printf '%s\n' "$report" | grep -qE ': (FAIL|NEEDS-FIXTURE)(: |$)'; then
        conformance_exit=1
    fi
fi
if [ "$conformance_exit" -ne 0 ]; then
    echo "onboard-gate: conformance checks failed or need a fixture" >&2
fi

echo
echo "+ "; printf '%q ' "${SERVED[@]}"; printf '\n'
"${SERVED[@]}"
echo "+ "; printf '%q ' "${NUMERICS[@]}"; printf '\n'
"${NUMERICS[@]}"

exit "$conformance_exit"
