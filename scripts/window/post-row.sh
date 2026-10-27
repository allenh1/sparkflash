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

# One post-row step: gate the row numerically and record the one published number.
#
#   scripts/window/post-row.sh <window run dir> [commit]
#
# Called automatically at the end of every device row (the Lead rows wrapper invokes it) and usable
# by hand after a restore. It is intentionally the only acceptance step: numerical-gate prints
# PASS/FAIL and speed-log appends the four-rank mean to docs/build/speed-log.csv. Byte identity is
# the strict mode, requested with SPARKFLASH_IDENTITY=1. No separate document, no decision record,
# no hand re-run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: post-row.sh <run dir> [commit]}
COMMIT=${2:-$(git -C "$ROOT" rev-parse --short HEAD)}
bash "$ROOT/scripts/numerical-gate.sh" --run "$RUN"
if [ "${SPARKFLASH_IDENTITY:-0}" = "1" ]; then
  bash "$ROOT/scripts/identity-gate.sh" --run "$RUN"
fi
bash "$ROOT/scripts/speed-log.sh" append --run "$RUN" --commit "$COMMIT" --note "$(basename "$RUN")"
