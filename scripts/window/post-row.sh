#!/usr/bin/env bash
# One post-row step: prove byte identity and record the one published number.
#
#   scripts/window/post-row.sh <window run dir> [commit]
#
# Called automatically at the end of every device row (the Lead rows wrapper invokes it) and usable
# by hand after a restore. It is intentionally the only acceptance step: identity-gate prints PASS/FAIL
# and speed-log appends the four-rank mean to docs/build/speed-log.csv. No separate document, no
# decision record, no hand re-run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: post-row.sh <run dir> [commit]}
COMMIT=${2:-$(git -C "$ROOT" rev-parse --short HEAD)}
bash "$ROOT/scripts/identity-gate.sh" --run "$RUN"
bash "$ROOT/scripts/speed-log.sh" append --run "$RUN" --commit "$COMMIT" --note "$(basename "$RUN")"
