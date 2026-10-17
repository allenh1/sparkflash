#!/usr/bin/env python3
"""Regenerates and checks the independent golden for the Engram head fixtures.

WHY THIS EXISTS
    The Rust test in src/engram_head_ref.rs asserts the decoder against REAL_ROWS, eight rows
    of released checkpoint bytes. Those constants are evidence only because a second,
    independent implementation produced them: if the goldens were read out of decode_row
    itself, the test would be checking the Rust against the Rust. This script is that second
    implementation. It is stdlib-only and shares no code with the crate, so a change to
    decode_row cannot move the expectation along with it.

WHAT IS COMPARED, AND HOW EXACTLY
    * fnv1a64 over the 256 decoded f32 bit patterns, plus bits_at [0, 127, 255]: EXACT, bit
      for bit. These are the claims the Rust test makes bit-for-bit and the ones a decoder
      change breaks.
    * sum, sum_squares, min, max: independent f64 accumulations, compared with a stated
      relative tolerance (1e-5) because the Rust constants are printed to six significant
      digits. They catch a wrong scale or a slipped block, not a last-bit difference.
    E4M3: bias 7, 0x7F and 0xFF are NaN, subnormals are m * 2**-9 (quant.rs:109-128).
    E8M0: 2**(code - 127), 255 is NaN (quant.rs:135-142). One scale per 32 columns.

INPUTS
    golden_rows.txt next to this file: the eight rows as hex, extracted read-only from the
    released checkpoint by probe_rows.py. The checkpoint itself is not needed to run this.

HOW TO RUN
    python3 golden_decode.py
        Print the goldens as JSON.
    python3 golden_decode.py --write
        Rewrite golden_expect.json next to this file.
    python3 golden_decode.py --check-against ../../src/engram_head_ref.rs
        Compare the goldens against the REAL_ROWS constants in the Rust source. Exits 1 on
        any mismatch, exact for fnv1a64 and bits_at and within tolerance for the floats.

TO RE-DERIVE FROM THE RELEASED CHECKPOINT INSTEAD OF THE EMBEDDED BYTES
    python3 probe_rows.py --snapshot <snapshot directory> > golden_rows.txt
    which reads 2,112 bytes of row data and 2,672 bytes of headers and writes nothing, then
    run the check again.
"""

import argparse
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_NAME = "golden_rows.txt"
EXPECT_NAME = "golden_expect.json"
WEIGHT_BYTES = 256
SCALE_BYTES = 8
SCALE_BLOCK = 32
TOLERANCE = 1e-5

EXACT_FIELDS = ("fnv1a64", "bits_at")
FLOAT_FIELDS = ("sum", "sum_squares", "min", "max")


def e4m3(code):
    """F8_E4M3 byte to float: bias 7, 0x7F and 0xFF are NaN."""
    neg = code & 0x80 != 0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return float("nan")
    if exponent == 0:
        magnitude = mantissa * 2.0 ** -9
    else:
        magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -magnitude if neg else magnitude


def e8m0(code):
    """F8_E8M0 byte to float: 2**(code - 127), 255 is NaN."""
    if code == 255:
        return float("nan")
    return 2.0 ** (code - 127)


def fnv1a64(payload):
    digest = 0xCBF29CE484222325
    for byte in payload:
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


def read_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if parts and parts[0] == "ROW":
                rows.append((parts[1], int(parts[2]), parts[3], parts[4]))
    if not rows:
        raise SystemExit("no ROW lines in " + path)
    return rows


def decode(layer, row_id, weight_hex, scale_hex):
    weight = bytes.fromhex(weight_hex)
    scale = bytes.fromhex(scale_hex)
    if len(weight) != WEIGHT_BYTES or len(scale) != SCALE_BYTES:
        raise SystemExit("row %s/%d is not 256 + 8 bytes" % (layer, row_id))
    values = [e4m3(weight[i]) * e8m0(scale[i // SCALE_BLOCK]) for i in range(WEIGHT_BYTES)]
    bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    finite = [value for value in values if value == value]
    total = 0.0
    for value in values:
        total += value
    sum_squares = 0.0
    for value in values:
        sum_squares += value * value
    return {
        "layer": int(layer[1:]),
        "id": row_id,
        "sum": total,
        "sum_squares": sum_squares,
        "min": min(finite) if finite else float("nan"),
        "max": max(finite) if finite else float("nan"),
        "fnv1a64": "0x%016x" % fnv1a64(struct.pack("<256I", *bits)),
        "bits_at": ["0x%08x" % bits[0], "0x%08x" % bits[127], "0x%08x" % bits[255]],
    }


def goldens(rows_path):
    return [decode(*row) for row in read_rows(rows_path)]


def parse_rust(path):
    """Pull the REAL_ROWS entries out of the Rust source, without running Rust."""
    text = open(path).read()
    start = text.find("REAL_ROWS")
    if start < 0:
        raise SystemExit("no REAL_ROWS in " + path)
    body = text[start:]
    entries = []
    for block in body.split("RealRow {")[1:]:
        block = block.split("}")[0]
        # The struct definition and the doc comments also contain "RealRow {", and their
        # field lines read "pub layer: u32". Only an entry has a decimal layer and id.
        match = re.search(r"layer:\s*(\d+)", block)
        if not match:
            continue
        entry = {"layer": int(match.group(1))}
        for field in ("id",):
            match = re.search(field + r":\s*(\d+)", block)
            if not match:
                raise SystemExit("no " + field + " in a RealRow of " + path)
            entry[field] = int(match.group(1))
        for field in ("sum", "sum_squares", "min", "max"):
            match = re.search(field + r":\s*([-0-9.eE+]+)", block)
            if not match:
                raise SystemExit("no " + field + " in a RealRow of " + path)
            entry[field] = float(match.group(1))
        match = re.search(r"fnv1a64:\s*0x([0-9a-fA-F]+)", block)
        entry["fnv1a64"] = "0x%016x" % int(match.group(1), 16)
        match = re.search(r"bits_at:\s*\[(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+),\s*(0x[0-9a-fA-F]+)\]", block)
        entry["bits_at"] = [match.group(1), match.group(2), match.group(3)]
        entries.append(entry)
    return entries


def compare(expected, actual, rust_path):
    """Return a list of human-readable mismatches."""
    problems = []
    if len(expected) != len(actual):
        return ["%s has %d RealRows, golden_rows.txt has %d rows" % (rust_path, len(actual), len(expected))]
    for want, got in zip(expected, actual):
        where = "layer %d id %d" % (want["layer"], want["id"])
        if want["layer"] != got["layer"] or want["id"] != got["id"]:
            problems.append("%s: Rust has layer %d id %d, rows file has %s" % (where, got["layer"], got["id"], where))
            continue
        for field in EXACT_FIELDS:
            if want[field] != got[field]:
                problems.append("%s: %s is %s in Rust, %s here" % (where, field, got[field], want[field]))
        for field in FLOAT_FIELDS:
            mine = want[field]
            theirs = got[field]
            if mine != mine and theirs != theirs:
                continue
            if mine != mine or theirs != theirs:
                problems.append("%s: %s is NaN on one side only" % (where, field))
                continue
            scale = max(abs(mine), abs(theirs), 1e-30)
            if abs(mine - theirs) / scale > TOLERANCE:
                problems.append("%s: %s is %.17g in Rust, %.17g here (relative %.3g > %.0e)"
                                % (where, field, theirs, mine, abs(mine - theirs) / scale, TOLERANCE))
    return problems


def compare_recorded(recorded, actual):
    """Check golden_expect.json against what the decoder just produced.

    The recorded file is the checked-in copy of this script's output. It is not the source of
    truth - golden_rows.txt and this decoder are - but a stale recorded file is a trap for the
    next reader, so --check-against reports it.
    """
    problems = []
    if len(recorded) != len(actual):
        return ["%s has %d rows, golden_rows.txt has %d" % (EXPECT_NAME, len(recorded), len(actual))]
    for want, got in zip(recorded, actual):
        where = "layer %d id %d" % (want["layer"], want["id"])
        if want["layer"] != got["layer"] or want["id"] != got["id"]:
            problems.append("%s: %s and golden_rows.txt disagree on which row this is" % (where, EXPECT_NAME))
            continue
        for field in EXACT_FIELDS:
            if want[field] != got[field]:
                problems.append("%s: %s is %s in %s, %s here" % (where, field, want[field], EXPECT_NAME, got[field]))
        for field in FLOAT_FIELDS:
            mine = want[field]
            theirs = got[field]
            if mine != mine and theirs != theirs:
                continue
            if mine != mine or theirs != theirs:
                problems.append("%s: %s is NaN on one side only" % (where, field))
                continue
            scale = max(abs(mine), abs(theirs), 1e-30)
            if abs(mine - theirs) / scale > TOLERANCE:
                problems.append("%s: %s is %.17g in %s, %.17g here" % (where, field, theirs, EXPECT_NAME, mine))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=os.path.join(HERE, ROWS_NAME))
    parser.add_argument("--write", action="store_true", help="rewrite " + EXPECT_NAME)
    parser.add_argument("--check-against", metavar="RUST_FILE",
                        help="compare against the REAL_ROWS constants in this Rust file")
    args = parser.parse_args()

    result = goldens(args.rows)
    if args.write:
        with open(os.path.join(HERE, EXPECT_NAME), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("wrote " + os.path.join(HERE, EXPECT_NAME), file=sys.stderr)
    if args.check_against:
        problems = compare(result, parse_rust(args.check_against), args.check_against)
        recorded_path = os.path.join(HERE, EXPECT_NAME)
        if os.path.exists(recorded_path):
            with open(recorded_path) as handle:
                recorded = json.load(handle)
            problems += compare_recorded(recorded, result)
        else:
            print("note: " + EXPECT_NAME + " is missing, so only the Rust constants were checked",
                  file=sys.stderr)
        if problems:
            for problem in problems:
                print("MISMATCH " + problem, file=sys.stderr)
            return 1
        print("PASS: %d rows match %s, fnv1a64 and bits_at bit for bit, floats within %.0e"
              % (len(result), args.check_against, TOLERANCE))
        return 0
    if not args.write:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
