#!/bin/sh
# Size ratchet for crates/sparkflash-kernels/src/m4_abi.rs: the Milestone-4 ABI lint is FROZEN.
#
# Why: the module scans kernel source text for invariants, and its own contract says the scan is
# "explicitly NOT the proof" - the committed cubin baseline is - while the recorded review history
# reaches round 24 with a real false accept or false rejection in every round. D26 closes the class:
# a new kernel invariant goes to compiled-artifact inspection (scripts/check-kernel-baseline.sh:
# cubin bytes, sha256 and PTX membar lines for every Milestone 4 kernel) or to a runtime device
# assertion in a harness, not to a new scanner here. Defects in the scanners already present are
# still fixed; the freeze is on growth, not on correctness.
#
# The pin is a deliberate act: a commit that changes the file's length changes PINNED_LINES in the
# same diff, and a reviewer reads that diff as the policy decision it is. Growth is a failure, not a
# warning, so a new scanner cannot arrive as a side effect of an unrelated edit.
#
# This is the cheap, always-on half of the freeze. The expensive half - that the frozen scanners
# still agree with the kernels they read - is `cargo test -p sparkflash-kernels` in every pipeline.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FILE="$REPO_ROOT/crates/sparkflash-kernels/src/m4_abi.rs"

# Frozen 2026-10-18 (D26): 13,854 lines at main 2245a4e, 13,874 with the freeze notice, and
# 13,893 once the m7_layer0 split replaced the two include_str!s of the single example file with
# one concatenation constant.
PINNED_LINES=13893

if [ ! -f "$FILE" ]; then
    echo "error: $FILE is missing; the frozen Milestone-4 ABI lint cannot be removed silently (D26)" >&2
    exit 1
fi

LINES=$(wc -l < "$FILE" | tr -d ' ')

if [ "$LINES" -gt "$PINNED_LINES" ]; then
    echo "error: crates/sparkflash-kernels/src/m4_abi.rs grew to $LINES lines, above the frozen pin of $PINNED_LINES (DECISIONS.md D26)." >&2
    echo "  New kernel invariants do NOT go into a source-text scanner: use scripts/check-kernel-baseline.sh" >&2
    echo "  (cubin bytes, sha256 and PTX membar lines) or a runtime device assertion in a harness." >&2
    echo "  If this growth is a fix to a scanner that already existed, raise PINNED_LINES in this script" >&2
    echo "  in the same commit, cited to a Lead decision; that diff is the review gate." >&2
    exit 1
fi

if [ "$LINES" -lt "$PINNED_LINES" ]; then
    echo "note: crates/sparkflash-kernels/src/m4_abi.rs is $LINES lines, below the pin of $PINNED_LINES; lower PINNED_LINES in the same commit" >&2
fi

echo "m4_abi freeze: crates/sparkflash-kernels/src/m4_abi.rs is $LINES lines, pin $PINNED_LINES (D26)"
