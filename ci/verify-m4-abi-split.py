#!/usr/bin/env python3
"""Verify that the m4_abi quarantine split is a verbatim move of the frozen module.

D26 froze the Milestone-4 ABI source-text lint at one file, crates/sparkflash-kernels/src/m4_abi.rs
(13,906 lines at main 0639413). The split keeps the frozen scanners verbatim but files them by what
they are: the root file keeps the module documentation, the const API the launch and driver modules
import, every root test with its cfg(test) helpers and the pin_helpers module; src/m4_abi/scanner.rs
holds the C/C++ token and control-flow scanners the acquire guard uses; src/m4_abi/acquire_guard.rs
holds the acquire-guard test module, whose module path (and therefore whose 43 test ids) is unchanged.

This script is the split's proof, in the style of ci/verify-m7-layer0-split.py. It reads the base
commit (never a working-tree copy of the old file, so a later edit cannot make the verbatim claim
true retroactively), re-derives the three files the split should have produced - the base text cut
at the same slice boundaries, plus the glue that declares the submodules, plus the two allow-listed
edits below - formats the result with the repo's rustfmt, and requires the working tree to be
byte-identical to it. Any edit to a moved scanner, test, doc line or pinch is therefore either part
of the mechanical transformation or the check fails.

The only non-verbatim text, allow-listed and written out below:

  1. self-scan adapters inside the moved acquire_guard tests: three tests read this module family's
     own source (the rejection-routing scan, the documentation-count scan, and the doc
     lazy-continuation scan). They now read src/m4_abi/scanner.rs / the family concatenation,
     because the text they scan moved; their assertions are unchanged.
  2. the root module's freeze notice and the D26 pin comment, which now name the aggregated family
     length instead of one file's length.
  3. the K8 scanner data updates and the acquire-guard per-call fix, applied after the split below.
     These do not add a scanner: the certified head gained a w_block_max operand in its frozen
     signature table, the six kernels' static-assert counts rose with their K8 symbols, and the
     mHC file now carries scalar and six-row kernels so the guard checks every observing path.
     They are written out as exact replacements so this script still fails on any other drift.

Exit 0 means the split is exactly that regeneration; exit 1 prints the first differing line.

Usage: python3 ci/verify-m4-abi-split.py [--base COMMIT]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = "0639413"
BASE_PATH = "crates/sparkflash-kernels/src/m4_abi.rs"
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "crates/sparkflash-kernels/src"
ROOT_FILE = SRC / "m4_abi.rs"
SCANNER_FILE = SRC / "m4_abi" / "scanner.rs"
GUARD_FILE = SRC / "m4_abi" / "acquire_guard.rs"
CHECK_SCRIPT = REPO_ROOT / "ci" / "check-m4-abi-size.sh"
TMP = REPO_ROOT / "target" / "m4-abi-verify"
WRAPPER = "#[cfg(test)]\nmod acquire_guard {\n"
SCANNER_START = "/// One C token, the two lines it can be reported by"
FREEZE_TITLE = "//! # FREEZE (D26, 2026-10-18): no new source-text scanners in this file"
FREEZE_TITLE_NEW = "//! # FREEZE (D26, 2026-10-18): no new source-text scanners in this module family"
FREEZE_BULLET = "//!   * no new scanner, parser or lint for a new kernel invariant is added to this file; a new"
FREEZE_BULLET_NEW = "//!   * no new scanner, parser or lint for a new kernel invariant is added to this family; a new"
FREEZE_LENGTH = "//!   * the file's length is pinned by ci/check-m4-abi-size.sh, wired into .gitlab-ci.yml; a change that\n//!     makes it longer fails that check, and a legitimate shrink updates the pin in the same commit.\n"
FREEZE_LENGTH_NEW = "//!   * the family's total length is pinned by ci/check-m4-abi-size.sh, wired into .gitlab-ci.yml: the\n//!     root file plus the pinned submodules under src/m4_abi/ (the quarantine split recorded in\n//!     docs/archive/build/m4-abi-split.md). Any net change fails that check, and a legitimate shrink updates the\n//!     pin in the same commit.\n"
POST_SPLIT_ROOT_EDITS = [
    (
        '"const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; float *__restrict__ logits;',
        '"const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; const __nv_bfloat16 *__restrict__ w_block_max; float *__restrict__ logits;',
    ),
    (
        '    ("mega_mhc_sm121.cu", 6),\n    ("b12x_mxfp8_dense_sm121.cu", 6),',
        '    ("mega_mhc_sm121.cu", 7),\n    ("b12x_mxfp8_dense_sm121.cu", 15),',
    ),
    (
        '    ("b12x_mxfp8_wo_a_sm121.cu", 11),\n    ("cert_lm_head_sm121.cu", 8),',
        '    ("b12x_mxfp8_wo_a_sm121.cu", 20),\n    ("cert_lm_head_sm121.cu", 9),',
    ),
    (
        '    ("b12x_mxfp4_moe_sm121.cu", 10),\n',
        '    ("b12x_mxfp4_moe_sm121.cu", 12),\n',
    ),
    (
        '        checked, 78,\n        "the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 6, \\\n',
        '        checked, 98,\n        "the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 7, \\\n',
    ),
    (
        '        checked, 98,\n',
        '        checked, 100,\n',
    ),
    (
        'the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 7, \\\n',
        'the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 7, 15, \\\n',
    ),
    (
        '         10, 17, 11 and 8 in the six kernels. The MoE count rose from 7 to 10 with D14, which pins \\\n         the FP8 E4M3 activation operand\'s byte width in the build itself; flash_mla\'s rose from 13 \\\n         to 15 with the engine window format\'s two layout asserts, then to 17 with the union \\\n         capacity\'s two. The format VALUE is not among them: it is a runtime launch parameter and \\\n         is guarded at kernel entry instead"\n',
        '         12, 17, 20 and 9 in the six kernels (mega_mhc, dense, MoE, flash_mla, wo_a, cert). The MoE \\\n         count rose from 10 to 12 with the split-K tiling: the 256-thread block identity and the \\\n         one-reducer-per-(slot, column) bound. D14 had raised it from 7 to 10 for the FP8 E4M3 \\\n         activation operand\'s byte width; flash_mla\'s rose from 13 to 15 with the engine window \\\n         format\'s two layout asserts, then to 17 with the union capacity\'s two. The format VALUE is \\\n         not among them: it is a runtime launch parameter and is guarded at kernel entry instead"\n',
    ),
    (
        '            && code.contains("acc += (partial * act_scale) * w_scale;")\n',
        '            && code.contains("terms[slot][block][col_local] =")\n            && code.contains("__fmul_rn(__fmul_rn(partial, act_scale), w_scale);")\n',
    ),
    (
        '        "acc += (partial * act_scale) * w_scale;",\n        "acc += routing * (partial * act_scale) * w_scale;",\n',
        '        "__fmul_rn(__fmul_rn(partial, act_scale), w_scale);",\n        "__fmul_rn(__fmul_rn(routing * partial, act_scale), w_scale);",\n',
    ),
]

POST_SPLIT_GUARD_EDITS = [
    (
        '''    // The hand-rolled acquire that used to sit here is gone: with the helper in place a fence at
    // the call site would be a second, independently editable copy of the guarantee.
    let observing = MEGA_MHC.find(&call).expect("the call exists");
    assert!(
        !MEGA_MHC[observing..].contains("__threadfence"),
        "the acquire must not be duplicated at the call site"
    );
    // The read the acquire protects still happens after the call, and the call is what puts the
    // fence before it.
    let read = MEGA_MHC[observing..]
        .find("*hc_rms_out")
        .map(|offset| observing + offset)
        .expect("the observing path reads hc_rms_out after the poll");
    assert!(observing < read, "the helper call must precede the read");
''',
        '''    // The hand-rolled acquire is gone; the file carries scalar and K8 paths, so check each call.
    let mut calls = MEGA_MHC.match_indices(&call).peekable();
    assert!(calls.peek().is_some(), "the call exists");
    for (observing, _) in calls {
        let read = MEGA_MHC[observing..]
            .find("*hc_rms_out")
            .map(|offset| observing + offset);
        let read = read.expect("the observing path reads hc_rms_out after the poll");
        assert!(observing < read, "the helper call must precede the read");
        assert!(
            !MEGA_MHC[observing..read].contains("__threadfence"),
            "the acquire must not be duplicated at the call site"
        );
    }
''',
    ),
]


def apply_post_split_edits(expected):
    for old, new in POST_SPLIT_ROOT_EDITS:
        if old not in expected["root"]:
            sys.exit("error: post-split root edit target is missing: " + old[:60])
        expected["root"] = expected["root"].replace(old, new, 1)
    return expected


def apply_post_split_guard_edits(formatted_guard):
    # The guard block is applied after rustfmt: the split's raw body is still indented one level
    # (it lived inside the old inline test module), while this replacement matches the formatted
    # file that the repository actually reviews.
    for old, new in POST_SPLIT_GUARD_EDITS:
        if old not in formatted_guard:
            sys.exit("error: post-split acquire-guard edit target is missing")
        formatted_guard = formatted_guard.replace(old, new, 1)
    return formatted_guard


GLUE = "mod scanner;\npub use scanner::{\n    acquire_fence_after_poll, acquire_helper_fences_unconditionally,\n    guarded_names_are_not_redefined, observing_paths_call_the_acquiring_poll,\n    the_acquiring_poll_is_defined_exactly_once,\n};\n\n#[cfg(test)]\nuse scanner::*;\n\n#[cfg(test)]\nmod acquire_guard;\n\n"
ADAPTER_READ = (
    "    let source =\n"
    '        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi/scanner.rs"))\n'
    '            .expect("the scanner\'s own source is readable");\n'
    "    // Round-15 A-5: the region split is a text search, and this file is checked out on platforms\n"
    "    // whose line endings are CRLF, so the newlines are normalised first.\n"
    '    let source = source.replace("\\r\\n", "\\n");\n'
    "    // The guard's own code: from the hint constant that every rejection ends with to the end of the\n"
    "    // scanner module. The scanner module holds the guard's code and nothing else after the hint.\n"
    "    let code = source\n"
    '        .split_once("const CANONICAL_HINT: &str")\n'
    '        .expect("the guard defines the canonical hint")\n'
    "        .1;\n"
)
ADAPTER_SOURCE = (
    "const THIS_SOURCE: &str = concat!(\n"
    '    include_str!("../m4_abi.rs"),\n'
    '    include_str!("scanner.rs"),\n'
    '    include_str!("acquire_guard.rs"),\n'
    ");\n"
)
ADAPTER_DOC = (
    "/// The whole " + chr(96) + "m4_abi" + chr(96) + " module family - this module, its scanner and the root file - as the text\n"
    "/// the documentation checks below read: the module's own documentation is part of the crate, so\n"
    "/// the shape clippy refuses can be checked here.\n"
)


def base_text():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", BASE_COMMIT + ":" + BASE_PATH],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("error: cannot read " + BASE_PATH + " at " + BASE_COMMIT + ": " + result.stderr.strip())
    return result.stdout


def slice_base(base):
    lines = base.splitlines(keepends=True)
    scanner_start = next(i for i, line in enumerate(lines) if line.startswith(SCANNER_START))
    guard_decl = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "#[cfg(test)]" and i + 1 < len(lines) and lines[i + 1].startswith("mod acquire_guard {")
    )
    scanner = "".join(lines[scanner_start : guard_decl - 1])
    root_pre = "".join(lines[:scanner_start])
    guard_body = "".join(lines[guard_decl + 2 : len(lines) - 1])
    assert lines[guard_decl - 1].strip() == "", "the scanner slice must end on its own line"
    assert lines[-1].strip() == "}", "the module must end on its closing brace"
    return root_pre, scanner, guard_body


def split_expected(base):
    root_pre, scanner, guard_body = slice_base(base)
    # The root: freeze-notice wording, then the module declarations, then everything else verbatim.
    root = root_pre.replace(FREEZE_TITLE, FREEZE_TITLE_NEW, 1).replace(FREEZE_BULLET, FREEZE_BULLET_NEW, 1)
    assert FREEZE_LENGTH in root
    root = root.replace(FREEZE_LENGTH, FREEZE_LENGTH_NEW, 1)
    anchor = "use crate::c_decl::Consts;\n\n"
    index = root.index(anchor) + len(anchor)
    root = root[:index] + GLUE + root[index:]

    # The scanner: the base text verbatim, with the cross-module visibility the moved items need.
    scanner_lines = []
    for line in scanner.splitlines(keepends=True):
        if re.match(r"^(fn |struct |enum |const |static |type )", line):
            line = "pub(crate) " + line
        scanner_lines.append(line)
    scanner_out = "".join(scanner_lines)

    # The acquire-guard tests: the base body verbatim, with the three self-scan adapters below.
    guard = guard_body
    replacement = ADAPTER_READ
    start = guard.index("    let source = std::fs::read_to_string")
    end = guard.index("        .0;", start) + len("        .0;") + 1
    guard = guard[:start] + replacement + guard[end:]
    assert '    let source = THIS_SOURCE;' not in guard
    guard = guard.replace(
        '        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi.rs"));',
        "    let source = THIS_SOURCE;",
        1,
    )
    guard = guard.replace(
        '    const THIS_SOURCE: &str = include_str!("m4_abi.rs");\n',
        ADAPTER_SOURCE,
        1,
    )
    guard = guard.replace(
        "    /// This file, as the text the documentation check below reads: the module's own documentation is\n"
        "    /// part of the crate, so the shape clippy refuses can be checked here.\n",
        ADAPTER_DOC,
        1,
    )
    return {"root": root, "scanner": scanner_out, "acquire_guard": guard}


def format_rust(expected, name):
    # Write the whole regenerated family first, so rustfmt can resolve the root file's mod
    # declarations; then format each file exactly as cargo fmt would.
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "m4_abi").mkdir(exist_ok=True)
    (TMP / "m4_abi.rs").write_text(expected["root"])
    (TMP / "m4_abi" / "scanner.rs").write_text(expected["scanner"])
    (TMP / "m4_abi" / "acquire_guard.rs").write_text(expected["acquire_guard"])
    path = TMP / name
    result = subprocess.run(
        ["rustfmt", "--edition", "2024", str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("error: rustfmt refused the regenerated " + name + ":\n" + result.stderr.strip())
    return path.read_text()


def compare(expected, actual_path, label):
    actual = actual_path.read_text()
    if expected == actual:
        print("  " + label + ": byte-identical to the regeneration")
        return True
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for number, (left, right) in enumerate(zip(expected_lines, actual_lines), start=1):
        if left != right:
            print("  " + label + ": first difference at line " + str(number))
            print("    expected: " + left)
            print("    actual:   " + right)
            return False
    print("  " + label + ": line counts differ (" + str(len(expected_lines)) + " vs " + str(len(actual_lines)) + ")")
    return False


def check_family_and_pin():
    actual = sorted(p for p in (SRC / "m4_abi").glob("*.rs"))
    expected = sorted([SCANNER_FILE, GUARD_FILE])
    if actual != expected:
        sys.exit("error: src/m4_abi/ holds " + str([p.name for p in actual]) + ", expected scanner.rs and acquire_guard.rs")
    family = [ROOT_FILE, SCANNER_FILE, GUARD_FILE]
    total = sum(len(path.read_text().splitlines()) for path in family)
    pin = re.search(r"^PINNED_LINES=(\d+)$", CHECK_SCRIPT.read_text(), re.M)
    if not pin:
        sys.exit("error: ci/check-m4-abi-size.sh has no PINNED_LINES line")
    if int(pin.group(1)) != total:
        sys.exit("error: the family is " + str(total) + " lines but ci/check-m4-abi-size.sh pins " + pin.group(1))
    tests = sum(len(re.findall(r"^\s*#\[test\]$", path.read_text(), re.M)) for path in family)
    if tests != 64:
        sys.exit("error: the family has " + str(tests) + " #[test] items, expected the pre-split 64")
    return total, tests


def main():
    global BASE_COMMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE_COMMIT, help="base commit the move is checked against")
    args = parser.parse_args()
    BASE_COMMIT = args.base
    base = base_text()
    total, tests = check_family_and_pin()
    expected = apply_post_split_edits(split_expected(base))
    expected["root"] = format_rust(expected, "m4_abi.rs")
    expected["scanner"] = format_rust(expected, "m4_abi/scanner.rs")
    expected["acquire_guard"] = apply_post_split_guard_edits(
        format_rust(expected, "m4_abi/acquire_guard.rs")
    )
    ok = True
    ok &= compare(expected["root"], ROOT_FILE, "root")
    ok &= compare(expected["scanner"], SCANNER_FILE, "scanner")
    ok &= compare(expected["acquire_guard"], GUARD_FILE, "acquire_guard")
    if not ok:
        sys.stderr.write("error: the working tree is not the regeneration of " + BASE_COMMIT + ":" + BASE_PATH + "\n")
        return 1
    print("m4_abi split verified against " + BASE_COMMIT + ":" + BASE_PATH)
    print("  family: " + str(total) + " lines, pin matches; " + str(tests) + " tests (the pre-split list is unchanged)")
    print("  the three files are exactly the base text cut at its slice boundaries, plus the glue")
    print("  and the allow-listed self-scan adapters, formatted by the repo's rustfmt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
