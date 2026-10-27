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

"""DEBUGGER, not a gate (D55): the S1 device arm's attribution report - the FIRST differing layer.

Use this when `scripts/numerical-gate.sh` has already failed, or when a slice claims bit identity
and you need the layer that broke. It judges no row: the amended gate (D54/D55, `docs/RULES.md` §3)
is what decides a row, and it does so teacher-forced against the host arm's measured distance. This
report answers "where", never "pass/fail".

The device arm has no per-stage observation read in its product path, so a divergence inside it is
localised by a DEBUG row: `SPARKFLASH_DEVICE_DIFF=1` makes the four-rank loop run the host arm's
stages beside the device arm's and dump both arms' boundary rows
(`SPARKFLASH_DEVICE_DIFF_DUMP=<dir>`), one f32-LE file per boundary per layer per position:

    <dir>/p<position>-l<layer>-<boundary>-host.f32
    <dir>/p<position>-l<layer>-<boundary>-dev.f32

This script reads those rows and the declared classes in
`docs/build/reference/s1-device-bounds.json` and answers one question:

    what is the first (position, layer, boundary) whose difference leaves its class?

That number is the finding. The count of differing rows is NOT: a divergence at layer L makes every
boundary after it differ too, so "18 boundaries differ" is one bug plus its propagation.

Classes, as the bounds file declares them:

  bit_exact    every value's bits must agree; one differing value is a bug
  ulp1_f32     every differing value within 1 f32 ULP, and no value changes sign (a D51
               reassociated reduction)
  bf16_round   every differing value within 1 BF16 code (a boundary the host rounds to BF16 itself)
  composed     not an independent op; reported, never judged

Exit codes: 0 every judged boundary is inside its class; 1 a boundary left its class (the first one
is printed); 2 the inputs are missing or unreadable, so NOTHING was compared.

Usage:
    scripts/s1-first-diff.py --dump docs/archive/build/<tag>/diff-rows [--bounds FILE]
                             [--boundary normed,attn,...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_BOUNDS = "docs/build/reference/s1-device-bounds.json"


def fail_closed(why: str) -> "None":
    print(f"s1-first-diff: {why}", file=sys.stderr)
    raise SystemExit(2)


def read_row(path: str) -> List[float]:
    data = open(path, "rb").read()
    if len(data) % 4:
        fail_closed(f"{path} is {len(data)} B, not a whole number of f32 values")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ordered(bits: int) -> int:
    """A monotone integer order over f32 bit patterns (sign-magnitude to two's complement)."""
    return 0x80000000 - bits if bits & 0x80000000 else bits | 0x80000000


def ulp_distance(a: float, b: float) -> int:
    return abs(ordered(f32_bits(b)) - ordered(f32_bits(a)))


def bf16_code(value: float) -> int:
    """The BF16 code of a value that is already BF16-representable (RNE widening drops no bits)."""
    bits = f32_bits(value)
    return (bits >> 16) & 0xFFFF


def bf16_code_distance(a: float, b: float) -> int:
    left, right = bf16_code(a), bf16_code(b)
    if (left & 0x8000) != (right & 0x8000):
        # Opposite signs: the codes are not adjacent in magnitude, so score the pair as infinite
        # unless both are zero (in which case the sign bit is the whole difference).
        return 0 if (a == 0.0 and b == 0.0) else 1 << 30
    return abs(left - right)


def compare(host: List[float], dev: List[float], klass: str) -> Dict[str, object]:
    if len(host) != len(dev):
        fail_closed(f"row lengths differ: host {len(host)} dev {len(dev)}")
    differing = 0
    worst_abs = 0.0
    worst_rel = 0.0
    worst_ulp = 0
    worst_bf16 = 0
    sign_cross = 0
    first_index: Optional[int] = None
    for index, (left, right) in enumerate(zip(host, dev)):
        if f32_bits(left) == f32_bits(right):
            continue
        differing += 1
        if first_index is None:
            first_index = index
        if (left > 0.0) != (right > 0.0) and left != 0.0 and right != 0.0:
            sign_cross += 1
        delta = abs(left - right)
        worst_abs = max(worst_abs, delta)
        scale = max(abs(left), abs(right))
        if scale > 0.0:
            worst_rel = max(worst_rel, delta / scale)
        worst_ulp = max(worst_ulp, ulp_distance(left, right))
        worst_bf16 = max(worst_bf16, bf16_code_distance(left, right))

    if klass == "bit_exact":
        inside = differing == 0
        reason = "" if inside else f"{differing} values differ, the class allows none"
    elif klass == "ulp1_f32":
        inside = worst_ulp <= 1 and sign_cross == 0
        reason = (
            ""
            if inside
            else f"{differing} differ, max {worst_ulp} f32 ULP (class allows 1), {sign_cross} sign changes"
        )
    elif klass == "bf16_round":
        inside = worst_bf16 <= 1 and sign_cross == 0
        reason = (
            ""
            if inside
            else f"{differing} differ, max {worst_bf16} BF16 codes (class allows 1), {sign_cross} sign changes"
        )
    elif klass == "composed":
        inside = True
        reason = ""
    else:
        fail_closed(f"unknown class {klass!r} in the bounds file")
    return {
        "class": klass,
        "differing": differing,
        "values": len(host),
        "max_abs": worst_abs,
        "max_rel": worst_rel,
        "max_ulp": worst_ulp,
        "max_bf16_codes": worst_bf16,
        "sign_changes": sign_cross,
        "first_index": first_index,
        "inside_class": inside,
        "reason": reason,
    }


def discover(dump: str) -> Tuple[Dict[str, set], set]:
    positions: Dict[str, set] = {}
    layers: set = set()
    names = set()
    for entry in os.listdir(dump):
        if not entry.endswith("-host.f32"):
            continue
        stem = entry[: -len("-host.f32")]
        parts = stem.split("-")
        if len(parts) != 3 or not parts[0].startswith("p") or not parts[1].startswith("l"):
            continue
        position = int(parts[0][1:])
        layer = int(parts[1][1:])
        name = parts[2]
        names.add(name)
        positions.setdefault(name, set()).add(position)
        layers.add(layer)
    return positions, layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="the directory SPARKFLASH_DEVICE_DIFF_DUMP wrote")
    parser.add_argument("--bounds", default=DEFAULT_BOUNDS)
    parser.add_argument("--boundary", default="", help="comma list to judge (default: the bounds file)")
    parser.add_argument("--json", default="", help="write the full report here")
    args = parser.parse_args()

    if not os.path.isdir(args.dump):
        fail_closed(f"{args.dump} is not a directory")
    if not os.path.exists(args.bounds):
        fail_closed(f"no bounds file at {args.bounds}")
    bounds = json.load(open(args.bounds))
    declared = bounds["boundaries"]
    order = [entry["name"] for entry in declared]
    klass = {entry["name"]: entry["class"] for entry in declared}
    wanted = [name for name in order if not args.boundary or name in args.boundary.split(",")]

    positions, layers = discover(args.dump)
    if not layers:
        fail_closed(f"{args.dump} has no boundary rows: nothing was compared")
    if not positions:
        fail_closed(f"{args.dump} has no positions")

    report: Dict[str, object] = {"dump": args.dump, "bounds": args.bounds, "boundaries": {}}
    faults: List[Dict[str, object]] = []
    for name in order:
        if name not in wanted:
            continue
        if name not in positions:
            report["boundaries"][name] = {"class": klass[name], "missing": True}
            continue
        rows = []
        for position in sorted(positions[name]):
            for layer in sorted(layers):
                host_path = os.path.join(args.dump, f"p{position}-l{layer:02}-{name}-host.f32")
                dev_path = os.path.join(args.dump, f"p{position}-l{layer:02}-{name}-dev.f32")
                if not os.path.exists(host_path) or not os.path.exists(dev_path):
                    continue
                result = compare(read_row(host_path), read_row(dev_path), klass[name])
                result.update({"position": position, "layer": layer})
                rows.append(result)
                if not result["inside_class"] and klass[name] != "composed":
                    faults.append(dict(result, boundary=name, stage_index=order.index(name)))
        inside = [row for row in rows if row["inside_class"]]
        report["boundaries"][name] = {
            "class": klass[name],
            "rows": len(rows),
            "rows_inside_class": len(inside),
            "rows_outside_class": len(rows) - len(inside),
            "worst": max(rows, key=lambda row: row["max_ulp"]) if rows else None,
            "first_outside": next((row for row in rows if not row["inside_class"]), None),
        }

    print(f"dump: {args.dump}  ({len(layers)} layers, {len(set().union(*positions.values()))} positions)")
    print(f"{'boundary':10} {'class':11} {'rows':>5} {'out':>5} {'max|d|':>10} {'max rel':>10} {'max ULP':>8} {'BF16':>5} {'sign':>4}")
    for name in order:
        entry = report["boundaries"].get(name)
        if entry is None or entry.get("missing"):
            print(f"{name:10} {'MISSING':11}")
            continue
        worst = entry["worst"] or {}
        print(
            f"{name:10} {entry['class']:11} {entry['rows']:5d} {entry['rows_outside_class']:5d} "
            f"{worst.get('max_abs', 0.0):10.2e} {worst.get('max_rel', 0.0):10.2e} "
            f"{worst.get('max_ulp', 0):8d} {worst.get('max_bf16_codes', 0):5d} {worst.get('sign_changes', 0):4d}"
        )

    # Execution order is position-major: the first divergence in TIME is the smallest
    # (position, layer, stage order). A fault in a late boundary at an early position is still the
    # earlier fault, which is exactly the composition argument for reporting first_differing().
    first_fault = min(faults, key=lambda row: (row["position"], row["layer"], row["stage_index"])) if faults else None
    if first_fault is None:
        print("\nVERDICT: every judged boundary is inside its declared class.")
        verdict = "PASS"
    else:
        print(
            "\nVERDICT: first differing boundary -> "
            f"{first_fault['boundary']} (stage order) at position {first_fault['position']}, "
            f"layer {first_fault['layer']}: {first_fault['reason']}; "
            f"max|d|={first_fault['max_abs']:.3e} max_rel={first_fault['max_rel']:.2e}"
        )
        verdict = "FAIL"
    report["verdict"] = verdict
    report["first_fault"] = first_fault
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
