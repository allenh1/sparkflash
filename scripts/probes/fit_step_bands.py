#!/usr/bin/env python3
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

"""Fit the shared step's cost model from a traced run.

The engine prints one line a tick under SPARKFLASH_BATCH_TRACE=1:

  batch: tick 12 seqs 8 prefill None 0.0 ms (reads 0.0 ms) decode 40 rows 121.3 ms lanes ...

A pure decode tick is one shared step: its rows are the anchors plus the verified drafts, and its
milliseconds are the step's wall. This fits the model the share prices drafts with, in one
least-squares solve, so the bands and the per-sequence term come from the same data:

  ms = a0 + a1 * rows                      rows <= BAND_ROWS
     = b0 + b1 * rows                      rows >  BAND_ROWS
     + seq_ms * (seqs - 1)

Usage: fit_step_bands.py <rank0.log> [--band-rows 16]
"""
import argparse
import re
import sys

BAND = 16
LINE = re.compile(
    r"^batch: tick (?P<tick>\d+) seqs (?P<seqs>\d+) prefill (?P<prefill>None|Some\(.*?\)) "
    r"[\d.]+ ms \(reads [\d.]+ ms\) decode (?P<rows>\d+) rows (?P<ms>[\d.]+) ms"
)


def points(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = LINE.match(line.strip())
            if not hit or hit.group("prefill") != "None":
                continue
            rows = int(hit.group("rows"))
            if rows == 0:
                continue
            out.append((rows, int(hit.group("seqs")), float(hit.group("ms"))))
    return out


def solve(features, targets):
    n = len(features[0])
    a = [[sum(r[i] * r[j] for r in features) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * t for r, t in zip(features, targets)) for i in range(n)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        for k in range(i + 1, n):
            f = a[k][i] / a[i][i]
            for j in range(i, n):
                a[k][j] -= f * a[i][j]
            b[k] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--band-rows", type=int, default=BAND)
    args = parser.parse_args()
    band = args.band_rows
    pts = points(args.path)
    narrow = [p for p in pts if p[0] <= band]
    wide = [p for p in pts if p[0] > band]
    if len(narrow) < 8 or len(wide) < 8:
        print(f"not enough points: {len(narrow)} narrow, {len(wide)} wide")
        return 1
    # Features: the narrow band's own intercept and slope, the wide band's, and one shared
    # per-sequence term (both bands pay it).
    feats = []
    for rows, seqs, _ in pts:
        inside = 1.0 if rows <= band else 0.0
        feats.append([inside, inside * rows, 1.0 - inside, (1.0 - inside) * rows, float(seqs - 1)])
    a0, a1, b0, b1, seq_ms = solve(feats, [p[2] for p in pts])
    model = [sum(f[i] * c for i, c in enumerate((a0, a1, b0, b1, seq_ms))) for f in feats]
    resid = [t[2] - m for t, m in zip(pts, model)]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
    print(f"{len(pts)} pure decode ticks: {len(narrow)} narrow (<= {band} rows), {len(wide)} wide")
    print(f"  narrow  a0 {a0:6.2f} ms  a1 {a1:5.2f} ms a row")
    print(f"  wide    b0 {b0:6.2f} ms  b1 {b1:5.2f} ms a row")
    print(f"  per sequence {seq_ms:5.2f} ms, rms {rms:5.2f} ms")
    print()
    print("  profile value for SPARKFLASH_SPEC_BATCH_COST:")
    print(f"  {a0:.1f},{a1:.2f},{b0:.1f},{b1:.2f},{seq_ms:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
