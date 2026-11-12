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

"""Turn one step's routed experts into the numbers a row price needs.

Why this exists: issue #20 WS4 asks whether a verify row is worth what it costs, and both halves of
that are counts nobody has: the experts a step pays for, and the *marginal* experts each row brings -
a row that shares slots with rows already in the step costs almost nothing, a row that brings new ones
costs a matrix a layer. The alternative, guessing from the row count, is what the engine does today
(StepCost::row_ms, 4 ms a row, against a measured 1.88) and is the thing to be replaced.

Input: the rank-0 log lines a step writes under SPARKFLASH_MOE_IDS_DUMP=<layer>, one a row:

  sparkflash serve: moe_ids layer=20 p0=0 n=40 top_k=6 row=0 lane=3 seg_p0=0 depth=0 ids=[...]

Rows written by a build before the lane/depth fields existed have only the ids, and the depth section
then says so instead of inventing a grouping.

Usage: ws4_row_pricing.py <log-or-dump> [--expert-ms 0.97]

The default 0.97 ms an expert-matrix is docs/performance.md's c16 figure (79 ms of expert read over
81.5 experts a layer). The ms columns are therefore an estimate of what a step would save, not a
measurement.
"""

import argparse
import re
import sys

ROW = re.compile(
    r"moe_ids layer=(?P<layer>\d+) p0=(?P<p0>\d+) n=(?P<n>\d+) top_k=(?P<top_k>\d+) "
    r"row=(?P<row>\d+)(?: lane=(?P<lane>\S+) seg_p0=(?P<seg_p0>\S+) depth=(?P<depth>\S+))? "
    r"ids=\[(?P<ids>[0-9, ]*)\]"
)


def parse(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = ROW.search(line)
            if not hit:
                continue
            ids = [int(part) for part in hit.group("ids").split(",") if part.strip()]
            depth = hit.group("depth")
            lane = hit.group("lane")
            if lane is None or not lane.isdigit():
                lane = None
            if depth is None or not depth.isdigit():
                depth = None
            rows.append(
                {
                    "layer": int(hit.group("layer")),
                    "p0": int(hit.group("p0")),
                    "n": int(hit.group("n")),
                    "top_k": int(hit.group("top_k")),
                    "row": int(hit.group("row")),
                    "lane": int(lane) if lane is not None else None,
                    "depth": int(depth) if depth is not None else None,
                    "ids": ids,
                }
            )
    return rows


def first_step(rows):
    """One step's rows, from a log carrying up to MOE_IDS_DUMPS steps.

    A step prints its rows from row 0 to row n - 1, so a row 0 starts the next step. The table
    docs/performance.md records is one step; analysing 48 of them together would report the union
    over 48 steps and a per-row marginal that mixes steps.
    """
    current = []
    for row in rows:
        if row["row"] == 0 and current:
            break
        current.append(row)
    return current


import re

SEG = re.compile(r"moe_segs seg=(?P<index>\d+) lane=(?P<lane>\d+) p0=(?P<p0>\d+) n=(?P<n>\d+)(?: row0=(?P<row0>\d+))?")


def segments(path):
    """The row layout of the step whose ids the report reads.

    `moe_segs` is printed once by `upload_row_table` (the boot table) and once per dumped step,
    so a full log carries many sets, each starting at seg 0. The set that describes the first
    `moe_ids` line is the one printed immediately before it - not the file's first set, which is
    the boot table's and can have a different lane/depth shape.
    """
    current = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = SEG.search(line)
            if hit:
                seg = {k: int(v) for k, v in hit.groupdict().items() if v is not None}
                if seg.get("index") == 0:
                    current = []
                current.append(seg)
                continue
            if current and ROW.search(line):
                return current
    return current


def placement(segs):
    """row index -> (lane, depth), by walking the segments in the order they are emitted."""
    rows = []
    for seg in segs:
        for offset in range(seg["n"]):
            rows.append((seg["lane"], offset))
    return rows

def marginals(rows):
    """Experts each row adds to the step, in the order the step emits them."""
    seen = set()
    out = []
    for entry in rows:
        added = 0
        for expert in entry["ids"]:
            if expert not in seen:
                added += 1
        seen.update(entry["ids"])
        out.append(added)
    return out


def report(rows, expert_ms, path=None):
    if not rows:
        print("no moe_ids lines: was the run started with SPARKFLASH_MOE_IDS_DUMP=<layer>?")
        return 1
    n = len(rows)
    top_k = rows[0]["top_k"]
    slots = sum(len(r["ids"]) for r in rows)
    union = len({expert for r in rows for expert in r["ids"]})
    counts = {}
    for entry in rows:
        for expert in entry["ids"]:
            counts[expert] = counts.get(expert, 0) + 1
    singles = sum(1 for c in counts.values() if c == 1)
    marg = marginals(rows)

    print(f"layer {rows[0]['layer']} p0 {rows[0]['p0']}: {n} rows, top_k {top_k}, {slots} slots")
    print(f"  union {union} experts ({slots / union:.2f}x collapse), {union / n:.2f} a row")
    print(f"  {singles} of {union} experts are touched by exactly one row")
    print(f"  marginal experts, first row to last: {marg[0]}..{marg[-1]}, sum {sum(marg)} (= the union)")
    print(f"  the first half brings {sum(marg[: n // 2])} of them; the last ten bring {sum(marg[-10:])}")
    print()
    print("  rows kept | experts | a row | expert ms | bytes against all rows")
    best = None
    marks = {max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n}
    for keep in range(1, n + 1):
        experts = sum(marg[:keep])
        if keep in marks:
            print(f"  {keep:9} | {experts:7} | {experts / keep:5.2f} | {experts * expert_ms:9.1f} | "
                  f"{union / experts:.2f}x fewer")
        if best is None or experts / keep < best[1]:
            best = (keep, experts / keep)
    print()
    print(f"  cheapest experts-a-row: {best[0]} of {n} rows at {best[1]:.2f} a row")

    # The ids dump carries depth only when the row table was uploaded (KV cache off); in serving
    # the segment table is the source, and it is printed beside the ids by the same switch.
    placed = placement(segments(path))
    if all(r["depth"] is None for r in rows) and len(placed) >= len(rows):
        for entry, (lane, depth) in zip(rows, placed):
            entry["lane"] = lane
            entry["depth"] = depth
    if all(r["depth"] is None for r in rows):
        print("  no segment table and no depth fields: marginals cannot be grouped by draft")
        return 0
    print()
    if any(r["lane"] is not None for r in rows):
        print("  lane | rows | marginal experts | a row")
        by_lane = {}
        for entry, added in zip(rows, marg):
            if entry["lane"] is None:
                continue
            by_lane.setdefault(entry["lane"], []).append(added)
        for lane in sorted(by_lane):
            got = by_lane[lane]
            print(f"  {lane:4} | {len(got):4} | {sum(got):16} | {sum(got) / len(got):5.2f}")
        print()
    print("  depth | rows | marginal experts | a row")
    by_depth = {}
    for entry, added in zip(rows, marg):
        if entry["depth"] is None:
            continue
        by_depth.setdefault(entry["depth"], []).append(added)
    for depth in sorted(by_depth):
        got = by_depth[depth]
        print(f"  {depth:5} | {len(got):4} | {sum(got):16} | {sum(got) / len(got):5.2f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="a rank-0 log, or a file of its moe_ids lines")
    parser.add_argument("--expert-ms", type=float, default=0.97,
                        help="milliseconds one expert matrix costs a layer (default 0.97, c16)")
    args = parser.parse_args()
    rows = first_step(parse(args.path))
    return report(rows, args.expert_ms, args.path)


if __name__ == "__main__":
    sys.exit(main())
