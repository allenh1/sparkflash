#!/usr/bin/env python3
"""Verify that the m7_layer0 split moved code without changing it.

The harness was one file, crates/sparkflash-kernels/examples/m7_layer0.rs (8,224 lines at main
2245a4e), and is now examples/m7_layer0/main.rs plus one module per stage. This script is the
proof that the move is verbatim: it rebuilds every original line range from the base commit and
compares it with the current files after removing only the lines the split added - the per-module
imports, the crate-root layout comment and mod/use glue, and the pub(crate) visibility the
cross-module items need - and after normalising the one thing rustfmt is allowed to change: when a
now-longer pub(crate) signature crosses max_width, it re-wraps it (stage 7; compared with
whitespace and commas ignored) and removes blank-line runs at the new file joins.

Usage: python3 ci/verify-m7-layer0-split.py [--base COMMIT]
Exit status 0 means the split is verbatim, 1 means it is not.
"""

import argparse
import pathlib
import re
import subprocess
import sys

BASE_COMMIT = "2245a4e"
EXAMPLE = "crates/sparkflash-kernels/examples/m7_layer0"
ORIGINAL = EXAMPLE + ".rs"

# Line ranges in the original file, 1-based inclusive, from the split manifest.
SLICES = {
    "stage0": (851, 1093),
    "stage1": (1094, 1527),
    "stage2": (1528, 2566),
    "stage3": (2567, 3311),
    "stage4": (3312, 3622),
    "stage5": (3623, 3915),
    "stage6": (3916, 4007),
    "stage8": (4008, 4331),
    "stage7": (4364, 6276),
    "stage9": (6277, 8219),
}
# The imports prepended to each module: the parent's items and the sibling stages whose items it
# uses. Anything else at the top of a file is part of the original range and is compared.
IMPORTS = {
    "stage0": ["use super::*;"],
    "stage1": ["use super::*;", "use crate::stage2::*;"],
    "stage2": ["use super::*;"],
    "stage3": ["use super::*;", "use crate::stage2::*;"],
    "stage4": ["use super::*;"],
    "stage5": ["use super::*;"],
    "stage6": ["use super::*;", "use crate::stage5::*;"],
    "stage8": ["use super::*;", "use crate::stage3::*;"],
    "stage7": ["use super::*;", "use crate::stage2::*;", "use crate::stage3::*;", "use crate::stage8::*;"],
    "stage9": ["use super::*;", "use crate::stage2::*;", "use crate::stage3::*;", "use crate::stage7::*;", "use crate::stage8::*;"],
}
# main.rs is the original crate root (lines 1-850) plus fn main (4332-4363); between them sit the
# layout comment and the mod/use glue. These are the only lines in main.rs that the split added.
GLUE = re.compile(r"^(mod stage\d+;|use crate::stage\d+::stage\d+;|// Layout:|// discipline,|// \(stage0\.rs|// doc comment live\.)")
# The old file ended in a five-line section banner with no code under it; the split dropped it.
DROPPED = (8220, 8224)
PATH_IN_ORIGINAL = '#[path = "../tests/support/b2_harness.rs"]'
PATH_IN_SPLIT = '#[path = "../../tests/support/b2_harness.rs"]'


def collapse(lines):
    out = []
    for line in lines:
        if line.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(line)
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def strip_visibility(lines):
    out = []
    for line in lines:
        if line.startswith("    pub(crate) "):
            out.append("    " + line[len("    pub(crate) "):])
        elif line.startswith("pub(crate) "):
            out.append(line[len("pub(crate) "):])
        else:
            out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_COMMIT)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    original = subprocess.run(
        ["git", "-C", str(root), "show", args.base + ":" + ORIGINAL],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines(keepends=True)
    split_dir = root / EXAMPLE
    ok = True

    for name, (start, end) in SLICES.items():
        path = split_dir / (name + ".rs")
        actual = path.read_text().splitlines(keepends=True)
        header = [line + "\n" for line in IMPORTS[name]]
        if actual[:len(header)] != header:
            print("FAIL", name, "header is not the expected import block")
            ok = False
            continue
        if not actual[len(header)].strip() == "":
            print("FAIL", name, "no blank line after the imports")
            ok = False
        body = strip_visibility(actual[len(header) + 1:])
        expected = original[start - 1:end]
        if collapse(expected) == collapse(body):
            print("OK   ", name, "verbatim", len(expected), "lines")
        elif name == "stage7":
            raw_expected = "".join("".join(expected).split())
            raw_body = "".join("".join(body).split())
            if raw_expected.replace(",", "") != raw_body.replace(",", ""):
                print("FAIL", name, "tokens differ")
                ok = False
            else:
                print("OK   ", name, "verbatim modulo rustfmt re-wrapping and",
                      raw_body.count(",") - raw_expected.count(","), "inserted trailing comma(s)")
        else:
            print("FAIL", name, "content differs")
            ok = False

    main_path = split_dir / "main.rs"
    text = main_path.read_text().replace(PATH_IN_SPLIT, PATH_IN_ORIGINAL)
    kept = [line for line in text.splitlines(keepends=True) if not GLUE.match(line)]
    expected_main = original[0:850] + original[4331:4363]
    if collapse(expected_main) == collapse(kept):
        print("OK   main.rs verbatim", len(expected_main), "lines (root 850 + fn main", len(original[4331:4363]), "lines, one trailing blank)")
    else:
        print("FAIL main.rs content differs")
        ok = False

    dropped = original[DROPPED[0] - 1:DROPPED[1]]
    if all(line.lstrip().startswith("//") for line in dropped):
        print("OK   dropped original lines", DROPPED[0], "to", DROPPED[1], "are comments only")
    else:
        print("FAIL dropped lines were not comments")
        ok = False
    present = "".join(
        path.read_text() for path in sorted(split_dir.glob("*.rs"))
    )
    for line in dropped:
        stripped = line.strip()
        # The banner's top and bottom rules are generic separators that also occur in the kept
        # comments; only its distinctive text can be checked for accidental resurrection.
        if stripped and not re.fullmatch(r"// -+", stripped) and stripped in present:
            print("FAIL a dropped banner line is still present:", stripped[:60])
            ok = False

    print("PASS split is verbatim" if ok else "FAIL split is not verbatim")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
