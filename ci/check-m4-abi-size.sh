#!/bin/sh
# Size ratchet for the frozen Milestone-4 ABI module family (D26):
# crates/sparkflash-kernels/src/m4_abi.rs plus the pinned submodules under src/m4_abi/.
#
# Why: the module scans kernel source text for invariants, and its own contract says the scan is
# "explicitly NOT the proof" - the committed cubin baseline is - while the recorded review history
# reaches round 36 with a real false accept or false rejection in every round. D26 closes the class:
# a new kernel invariant goes to compiled-artifact inspection
# (crates/sparkflash-kernels/scripts/check-kernel-baseline.sh: cubin bytes, sha256 and PTX membar
# lines for every Milestone 4 kernel) or to a runtime device assertion in a harness, not to a new
# scanner here. Defects in the scanners already present are
# still fixed; the freeze is on length, not on correctness.
#
# The pin is a deliberate act: a commit that changes the family's total length changes
# PINNED_LINES in the same diff, and a reviewer reads that diff as the policy decision it is. Any
# length change is a failure, not a warning - growth would let a new scanner arrive as a side
# effect of an unrelated edit, and a shrink would let a scanner disappear without the reviewed pin
# diff. The total is summed over the whole family, so moving lines between files is still a change
# to the pin, and a new file under src/m4_abi/ is included in the sum automatically.
#
# The aggregate replaced the single-file 13,906 pin in the quarantine split (task-27): the test
# list before and after is byte-identical and the move is proven verbatim by
# ci/verify-m4-abi-split.py; docs/build/m4-abi-split.md records the composition and the split.
#
# This is the cheap, always-on half of the freeze. The expensive half - that the frozen scanners
# still agree with the kernels they read - is `cargo test -p sparkflash-kernels` in every pipeline.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC="$REPO_ROOT/crates/sparkflash-kernels/src"
ROOT_FILE="$SRC/m4_abi.rs"
SUB_DIR="$SRC/m4_abi"

# The family, as pinned by the split: the root file plus the submodules that were carved out of
# it. This list is checked exactly, so the sum cannot be moved to a different file unnoticed.
EXPECTED_FILES="m4_abi.rs m4_abi/acquire_guard.rs m4_abi/scanner.rs"

# Pre-split baseline (main 0639413, one file): 13,906 lines. The split adds only the glue that
# declares the submodules and re-exports the moved public entry points; the aggregate below is the
# pinned total after that move. Any later commit that changes the total changes this number in the
# same diff, cited to a Lead decision, and that diff is the review gate.
PINNED_LINES=13924

if [ ! -f "$ROOT_FILE" ]; then
    echo "error: $ROOT_FILE is missing; the frozen Milestone-4 ABI lint cannot be removed silently (D26)" >&2
    exit 1
fi
if [ ! -d "$SUB_DIR" ]; then
    echo "error: $SUB_DIR is missing; the frozen Milestone-4 ABI lint's submodules cannot be removed silently (D26)" >&2
    exit 1
fi

ACTUAL_FILES=$(cd "$SRC" && ls m4_abi.rs m4_abi/*.rs)
if [ "$(printf '%s\n' $ACTUAL_FILES | sort)" != "$(printf '%s\n' $EXPECTED_FILES | sort)" ]; then
    echo "error: the m4_abi family is [$ACTUAL_FILES], expected [$EXPECTED_FILES] (DECISIONS.md D26)." >&2
    echo "  Adding or removing a file in the family changes what the aggregate pin covers; do it in a" >&2
    echo "  commit that changes PINNED_LINES and this list in the same diff, cited to a Lead decision." >&2
    exit 1
fi

TOTAL=0
for rel in $EXPECTED_FILES; do
    file="$SRC/$rel"
    if [ ! -f "$file" ]; then
        echo "error: $file is missing (DECISIONS.md D26)" >&2
        exit 1
    fi
    lines=$(wc -l < "$file" | tr -d ' ')
    TOTAL=$((TOTAL + lines))
done

# Any change, not just growth, is a failure: a shrink is still an edit to the frozen family, and
# silently accepting it would let a scanner be deleted or "simplified" without the PINNED_LINES diff
# that D26 makes the review gate.
if [ "$TOTAL" -ne "$PINNED_LINES" ]; then
    if [ "$TOTAL" -gt "$PINNED_LINES" ]; then
        echo "error: the m4_abi family grew to $TOTAL lines, above the frozen pin of $PINNED_LINES (DECISIONS.md D26)." >&2
        echo "  New kernel invariants do NOT go into a source-text scanner: use" >&2
        echo "  crates/sparkflash-kernels/scripts/check-kernel-baseline.sh (cubin bytes, sha256 and PTX membar lines)" >&2
        echo "  or a runtime device assertion in a harness." >&2
    else
        echo "error: the m4_abi family shrank to $TOTAL lines, below the frozen pin of $PINNED_LINES (DECISIONS.md D26)." >&2
        echo "  A shrink is also an edit to the frozen family: lower PINNED_LINES here in the same commit so the diff is reviewed." >&2
    fi
    echo "  If this change is a fix to a scanner that already existed, set PINNED_LINES in this script" >&2
    echo "  in the same commit, cited to a Lead decision; that diff is the review gate." >&2
    exit 1
fi

echo "m4_abi freeze: the family is $TOTAL lines (root $(wc -l < "$ROOT_FILE" | tr -d ' '), scanner $(wc -l < "$SUB_DIR/scanner.rs" | tr -d ' '), acquire_guard $(wc -l < "$SUB_DIR/acquire_guard.rs" | tr -d ' ')), pin $PINNED_LINES (D26)"
