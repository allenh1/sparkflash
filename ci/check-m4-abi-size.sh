#!/bin/sh
# Size ratchet for crates/sparkflash-kernels/src/m4_abi.rs: the Milestone-4 ABI lint is FROZEN.
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
# The pin is a deliberate act: a commit that changes the file's length changes PINNED_LINES in the
# same diff, and a reviewer reads that diff as the policy decision it is. Any length change is a
# failure, not a warning - growth would let a new scanner arrive as a side effect of an unrelated
# edit, and a shrink would let a scanner disappear without the reviewed pin diff.
#
# This is the cheap, always-on half of the freeze. The expensive half - that the frozen scanners
# still agree with the kernels they read - is `cargo test -p sparkflash-kernels` in every pipeline.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FILE="$REPO_ROOT/crates/sparkflash-kernels/src/m4_abi.rs"

# Frozen 2026-10-18 (D26): 13,854 lines at main 2245a4e, 13,874 with the freeze notice, and
# 13,893 once the m7_layer0 split replaced the two include_str!s of the single example file with
# one concatenation constant. 13,906 after the task-11 follow-up hardening: the round-33 claim,
# the crate-relative baseline paths, the exact m7_layer0/main.rs witness, the four-token check over
# the concatenation, and this script failing on any length change rather than only on growth.
PINNED_LINES=13906

if [ ! -f "$FILE" ]; then
    echo "error: $FILE is missing; the frozen Milestone-4 ABI lint cannot be removed silently (D26)" >&2
    exit 1
fi

LINES=$(wc -l < "$FILE" | tr -d ' ')

# Any change, not just growth, is a failure: a shrink is still an edit to the frozen file, and
# silently accepting it would let a scanner be deleted or "simplified" without the PINNED_LINES diff
# that D26 makes the review gate.
if [ "$LINES" -ne "$PINNED_LINES" ]; then
    if [ "$LINES" -gt "$PINNED_LINES" ]; then
        echo "error: crates/sparkflash-kernels/src/m4_abi.rs grew to $LINES lines, above the frozen pin of $PINNED_LINES (DECISIONS.md D26)." >&2
        echo "  New kernel invariants do NOT go into a source-text scanner: use" >&2
        echo "  crates/sparkflash-kernels/scripts/check-kernel-baseline.sh (cubin bytes, sha256 and PTX membar lines)" >&2
        echo "  or a runtime device assertion in a harness." >&2
    else
        echo "error: crates/sparkflash-kernels/src/m4_abi.rs shrank to $LINES lines, below the frozen pin of $PINNED_LINES (DECISIONS.md D26)." >&2
        echo "  A shrink is also an edit to the frozen file: lower PINNED_LINES here in the same commit so the diff is reviewed." >&2
    fi
    echo "  If this change is a fix to a scanner that already existed, set PINNED_LINES in this script" >&2
    echo "  in the same commit, cited to a Lead decision; that diff is the review gate." >&2
    exit 1
fi

echo "m4_abi freeze: crates/sparkflash-kernels/src/m4_abi.rs is $LINES lines, pin $PINNED_LINES (D26)"
