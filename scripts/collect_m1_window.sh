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

# Runs both post-window collectors on a copied four-rank evidence tree and records the artefacts.
# Usage: scripts/collect_m1_window.sh <evidence-dir> <out-dir>
# Expected on the frozen b69ebb2 m1-bind window: collector exit 3 (NOT-ATTEMPTED) and
# acceptance-runner exit 3 (shape PASS / pre_token PASS / overall NOT_ATTEMPTED). Any other exit is
# reported, never hidden.
set -uo pipefail
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "usage: collect_m1_window.sh <evidence-dir> <out-dir>" >&2
  exit 2
fi
EVIDENCE="$1"
OUT="$2"
# Tonight's frozen b69ebb2 evidence has 185,481 logical lines (208+208+19 attention names +
# 2x61440 + 2x30720 + 2x240 + 2x120 + 2x3); the earlier predicted 185,462 omitted the host-seam rows.
if [ "$#" -ge 3 ] && [ -n "$3" ]; then
  MANIFEST_LINES="$3"
else
  MANIFEST_LINES=185481
fi
mkdir -p "$OUT"

python3 scripts/d48_step_gate.py \
  --device-dir "$EVIDENCE" --kind device --expected-steps 3 \
  --out-json "$OUT/d48-step-gate.json" --out-md "$OUT/d48-step-gate.md"
GATE=$?

python3 scripts/accept_device_token.py \
  --device "$EVIDENCE" \
  --expect-attention-bound 208 \
  --expect-moe-bounds routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120 \
  --expect-manifest-lines "$MANIFEST_LINES" \
  --out-json "$OUT/accept-device-token.json" --out-md "$OUT/accept-device-token.md"
ACCEPT=$?

(
  cd "$EVIDENCE" || exit 1
  find . -type f | LC_ALL=C sort | while IFS= read -r file; do
    sha256sum "$file"
  done
) > "$OUT/evidence-sha256.txt"
(
  cd "$OUT" || exit 1
  for file in d48-step-gate.json d48-step-gate.md accept-device-token.json accept-device-token.md; do
    [ -f "$file" ] && sha256sum "$file"
  done
) > "$OUT/artefacts-sha256.txt"

echo "collector exit: $GATE (3 = NOT-ATTEMPTED expected)"
echo "acceptance exit: $ACCEPT (3 = NOT_ATTEMPTED expected)"
echo "artefacts: $OUT"
exit 0
