#!/usr/bin/env python3
"""One command for the next D41 capture window: plan check, capture, floors, flips, manifest.

Why this exists
---------------
D41 was refused twice: once against the wrong served root (the guard fired correctly), and once from
the two-class idle fallback because (a) busy_distractor was absent and (b) the idle_distractor class
had 19 rank-1 flips above the flat threshold. Fixing that next window needs one command that does not
let the window leave without the full evidence set:

    python3 scripts/d41_window.py capture  --base-url http://127.0.0.1:8000 --out-dir <dir>
    python3 scripts/d41_window.py analyse  --capture <capture.json> --out-dir <dir>

capture mode:
 1. validates the plan (dry-run, schema and frozen-plan sha256 check; no network for the dry-run),
 2. runs scripts/d41_recapture.py against the serving instance with a fresh --out name and its log,
 3. runs scripts/d41_floors.py with the strict default set (all three classes, never a widened
    --flat-prob) and --out-bars,
 4. runs scripts/d41_flip_investigation.py on the capture, then writes the window summary, the
    sha256 manifest and the decision/next-action list.

analyse mode runs steps 3-4 on an existing capture, so the post-window analysis is reproducible on
the host with no network. Both modes never modify the capture and never write a bars file unless the
floor tool itself produced one (status ok).

Exit codes: 0 = floors wrote a gate bars file (Lead review still required); 3 = requires review
(floors refused, flips found, or required classes missing - the evidence is recorded); 2 = refused or
invalid input (no capture and/or no capture JSON was written); 1 = unexpected tool failure.

The runner deliberately does not provide a way to widen --flat-prob; the 19-flip investigation is
the path, not a threshold change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import d41_floors as floors  # noqa: E402

SCHEMA = "sparkflash-d41-window/1"
RECAPTURE = os.path.join(SCRIPT_DIR, "d41_recapture.py")
FLOOR_TOOL = os.path.join(SCRIPT_DIR, "d41_floors.py")
FLIP_TOOL = os.path.join(SCRIPT_DIR, "d41_flip_investigation.py")
DEFAULT_PLAN = os.path.join(SCRIPT_DIR, "d41_capture_plan.json")
FROZEN_PLAN_SHA256 = "22746d98123a4ee67b33edf18c432e5fe067728076a1494b1ebd2514727353a6"
DEFAULT_EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"


class WindowError(Exception):
    pass


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise WindowError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise WindowError(f"{path}: not valid JSON: {error}") from error


def run(command: list, log_path: str) -> int:
    print("+ " + " ".join(command))
    with open(log_path, "w", encoding="utf-8") as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def plan_check(plan: str, expected_sha: str | None) -> dict:
    if not os.path.exists(plan):
        raise WindowError(f"--plan {plan}: no such file")
    actual = sha256_file(plan)
    if expected_sha is not None and actual != expected_sha:
        raise WindowError(
            f"--plan {plan}: sha256 {actual} does not match the expected frozen plan {expected_sha}; "
            "refusing to capture with a changed plan (pass --plan-sha256 explicitly only for a reviewed plan)"
        )
    document = read_json(plan)
    if document.get("schema") != "sparkflash-d41-capture-plan/1":
        raise WindowError(f"--plan {plan}: schema {document.get('schema')!r}, expected sparkflash-d41-capture-plan/1")
    blocks = document.get("blocks") or []
    classes = [block.get("class") for block in blocks]
    missing = [cls for cls in floors.CLASSES if cls not in classes]
    busy = [block for block in blocks if block.get("class") == "busy_distractor"]
    if missing:
        raise WindowError(
            f"--plan {plan}: missing required class(es) {missing}; the D41 floor needs repeat, idle_distractor and busy_distractor"
        )
    if not busy or not any(block.get("background") for block in busy):
        raise WindowError(f"--plan {plan}: busy_distractor block has no background generation; a controlled co-tenant is required")
    return {
        "path": os.path.abspath(plan),
        "sha256": actual,
        "blocks": [
            {
                "block_id": block.get("block_id"),
                "class": block.get("class"),
                "requests": len(block.get("requests") or []),
                "has_background": bool(block.get("background")),
            }
            for block in blocks
        ],
        "classes": classes,
        "all_required_classes": not missing,
    }


def dry_run(plan: str, base_url: str, log_path: str) -> int:
    command = [sys.executable, RECAPTURE, "--plan", plan, "--dry-run", "--base-url", base_url]
    return run(command, log_path)


def capture(plan: str, base_url: str, out_dir: str, capture_id: str, tag: str, expect_root: str, log_paths: dict) -> tuple:
    stamp = utc_now().replace("-", "").replace(":", "")
    out = os.path.join(out_dir, f"d41-capture-{stamp}.json")
    log = out + ".log"
    if os.path.exists(out):
        raise WindowError(f"{out}: refusing to overwrite an existing capture; wait one second or pick another --out-dir")
    command = [
        sys.executable,
        RECAPTURE,
        "--base-url", base_url,
        "--plan", plan,
        "--out", out,
        "--log", log,
        "--expect-root", expect_root,
        "--capture-id", capture_id,
        "--tag", tag,
        "--compact",
    ]
    code = run(command, log_paths["recapture"])
    return code, out, log


def run_floors(capture_path: str, out_dir: str, source_revision: str | None, flat_prob: float) -> dict:
    json_path = os.path.join(out_dir, "d41-floors.json")
    md_path = os.path.join(out_dir, "d41-floors.md")
    bars_path = os.path.join(out_dir, "d41-bars.json")
    log_path = os.path.join(out_dir, "d41-floors.log")
    command = [
        sys.executable,
        FLOOR_TOOL,
        "--capture", capture_path,
        "--out-json", json_path,
        "--out-md", md_path,
        "--out-bars", bars_path,
        "--flat-prob", str(flat_prob),
    ]
    if source_revision:
        command += ["--source-revision", source_revision]
    code = run(command, log_path)
    return {
        "exit": code,
        "log": log_path,
        "json": json_path if os.path.exists(json_path) else None,
        "md": md_path if os.path.exists(md_path) else None,
        "bars": bars_path if os.path.exists(bars_path) else None,
        "refusal": read_text(log_path).strip() if code != 0 else None,
        "document": read_json(json_path) if os.path.exists(json_path) else None,
    }


def run_flips(capture_path: str, out_dir: str, flat_prob: float) -> dict:
    json_path = os.path.join(out_dir, "d41-flips.json")
    md_path = os.path.join(out_dir, "d41-flips.md")
    log_path = os.path.join(out_dir, "d41-flips.log")
    command = [
        sys.executable,
        FLIP_TOOL,
        "--capture", capture_path,
        "--flat-prob", str(flat_prob),
        "--out-json", json_path,
        "--out-md", md_path,
    ]
    code = run(command, log_path)
    return {
        "exit": code,
        "log": log_path,
        "json": json_path if os.path.exists(json_path) else None,
        "md": md_path if os.path.exists(md_path) else None,
        "document": read_json(json_path) if os.path.exists(json_path) else None,
        "refusal": read_text(log_path).strip() if code not in (0, 3) else None,
    }


def compose(capture_path: str, plan_doc: dict | None, floors_run: dict, flips_run: dict, mode: str) -> dict:
    capture_doc = read_json(capture_path)
    capture_validation = None
    try:
        normalised = floors.validate_capture(capture_doc, capture_path)
        capture_validation = {
            "valid": True,
            "capture_id": normalised["capture_id"],
            "served_root": normalised["served_root"],
            "expect_root": normalised["expect_root"],
            "blocks": [
                {"block_id": block["block_id"], "class": block["class"], "pairs": len(block["pairs"])}
                for block in normalised["blocks"]
            ],
        }
    except floors.D41Error as error:
        capture_validation = {"valid": False, "error": str(error)}
    floors_doc = floors_run.get("document") or {}
    flips_doc = flips_run.get("document") or {}
    floors_ok = floors_run["exit"] == 0 and floors_run.get("bars")
    flips = (flips_doc.get("summary") or {}).get("flips_above_flat")
    if floors_ok:
        status = "OK-BARS-WRITTEN"
        next_actions = [
            "Lead reviews the bars file before compare_native_vs_vllm.py --bars consumes it",
            "record the bars file sha256 in the session evidence",
        ]
    elif flips_run["exit"] == 3 or flips:
        status = "REQUIRES_REVIEW"
        next_actions = [
            "review the flip table in d41-flips.md (per-pair prompts, positions, token ids, margins)",
            "choose a reviewed probe-design change or a reviewed close-margin decision; do not widen --flat-prob",
            "re-capture under the frozen plan discipline after the reviewed change",
        ]
    elif floors_run["exit"] == 2:
        status = "REQUIRES_REVIEW"
        next_actions = [
            "read d41-floors.log for the refusal reason",
            "capture the missing class(es) or resolve the named floor problem; do not widen --flat-prob",
        ]
    else:
        status = "TOOL-FAILURE"
        next_actions = ["inspect d41-floors.log and d41-flips.log; the window must not be reported as measured"]
    return {
        "schema": SCHEMA,
        "generated_utc": utc_now(),
        "mode": mode,
        "status": status,
        "capture": {
            "path": os.path.abspath(capture_path),
            "sha256": sha256_file(capture_path),
            "validation": capture_validation,
            "plan": plan_doc,
        },
        "rows": {
            "capture": {"verdict": "CAPTURED" if capture_validation and capture_validation.get("valid") else "INVALID", "path": os.path.abspath(capture_path)},
            "floors": {
                "verdict": "OK" if floors_ok else ("REFUSED" if floors_run["exit"] == 2 else "TOOL-FAILURE"),
                "exit": floors_run["exit"],
                "status": floors_doc.get("status"),
                "bars": os.path.abspath(floors_run["bars"]) if floors_run.get("bars") else None,
                "bars_sha256": sha256_file(floors_run["bars"]) if floors_run.get("bars") else None,
                "refusal": floors_run.get("refusal"),
            },
            "flip_investigation": {
                "verdict": "FLIPS-FOUND" if flips else ("NO-FLIPS" if flips_run["exit"] == 0 else "TOOL-FAILURE"),
                "exit": flips_run["exit"],
                "status": flips_doc.get("status"),
                "flips_above_flat": flips,
                "by_class": (flips_doc.get("summary") or {}).get("flips_above_flat_by_class"),
                "report": os.path.abspath(flips_run["json"]) if flips_run.get("json") else None,
            },
        },
        "next_actions": next_actions,
        "non_claims": [
            "a bars file is a proposal for Lead review, not an accepted gate",
            "no device or token-parity claim is made from a D41 capture: this is serving self-consistency evidence (D39)",
            "the capture JSON is the frozen input; this summary is derived evidence",
            "nothing in this window widens --flat-prob or edits docs/DECISIONS.md",
        ],
    }


def render_markdown(result: dict) -> str:
    lines = []
    lines.append("# D41 window record (" + result["generated_utc"] + ")")
    lines.append("")
    lines.append("Status: **" + result["status"] + "**  ")
    lines.append("Mode: " + result["mode"] + "  ")
    lines.append("Capture: " + result["capture"]["path"] + "  ")
    lines.append("Capture sha256: " + result["capture"]["sha256"])
    lines.append("")
    rows = result["rows"]
    lines.append("## Rows")
    lines.append("")
    lines.append("| row | verdict | detail |")
    lines.append("| :-- | :-- | :-- |")
    capture_row = rows["capture"]
    lines.append("| capture | " + capture_row["verdict"] + " | " + capture_row["path"] + " |")
    floor_row = rows["floors"]
    detail = "exit " + str(floor_row["exit"]) + "; status " + str(floor_row.get("status"))
    if floor_row.get("bars"):
        detail += "; bars " + str(floor_row["bars"])
    if floor_row.get("refusal"):
        detail += "; refusal: " + str(floor_row["refusal"]).splitlines()[-1]
    lines.append("| floors | " + floor_row["verdict"] + " | " + detail + " |")
    flip_row = rows["flip_investigation"]
    flip_detail = "exit " + str(flip_row["exit"]) + "; flips above flat " + str(flip_row.get("flips_above_flat"))
    if flip_row.get("by_class"):
        flip_detail += "; by class " + json.dumps(flip_row["by_class"])
    lines.append("| flip investigation | " + flip_row["verdict"] + " | " + flip_detail + " |")
    lines.append("")
    if result["capture"].get("validation"):
        validation = result["capture"]["validation"]
        lines.append("## Capture validation")
        lines.append("")
        if validation.get("valid"):
            lines.append("- capture_id: " + str(validation.get("capture_id")))
            lines.append("- served root: " + str(validation.get("served_root")))
            lines.append("- expect root: " + str(validation.get("expect_root")))
            for block in validation.get("blocks", []):
                lines.append("- block " + str(block["block_id"]) + ": class " + str(block["class"]) + ", pairs " + str(block["pairs"]))
        else:
            lines.append("- INVALID: " + str(validation.get("error")))
        lines.append("")
    if result.get("plan"):
        lines.append("## Plan")
        lines.append("")
        lines.append("- path: " + result["plan"]["path"])
        lines.append("- sha256: " + result["plan"]["sha256"])
        for block in result["plan"]["blocks"]:
            lines.append("- " + str(block["block_id"]) + ": class " + str(block["class"]) + ", requests " + str(block["requests"]) + ", background " + str(block["has_background"]))
        lines.append("")
    lines.append("## Next actions")
    lines.append("")
    for action in result["next_actions"]:
        lines.append("- " + action)
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    for claim in result["non_claims"]:
        lines.append("- " + claim)
    lines.append("")
    return chr(10).join(lines) + chr(10)


def write_manifest(out_dir: str, extra: list) -> None:
    rows = []
    skip = {"d41-window.json", "d41-window.md", "d41-window-sha256.txt"}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path) or name in skip:
            continue
        rows.append({"path": name, "sha256": sha256_file(path), "bytes": os.path.getsize(path)})
    for row in extra:
        if os.path.isfile(row["path"]):
            rows.append({"path": os.path.basename(row["path"]), "sha256": sha256_file(row["path"]), "bytes": os.path.getsize(row["path"])})
    with open(os.path.join(out_dir, "d41-window-sha256.txt"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["sha256"] + "  " + row["path"] + chr(10))


def analyse(capture_path: str, out_dir: str, source_revision: str | None, flat_prob: float, plan_doc: dict | None) -> int:
    if not os.path.exists(capture_path):
        raise WindowError(f"--capture {capture_path}: no such file")
    try:
        floors.validate_capture(read_json(capture_path), capture_path)
    except floors.D41Error as error:
        print("d41_window: invalid capture: " + str(error), file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)
    floors_run = run_floors(capture_path, out_dir, source_revision, flat_prob)
    flips_run = run_flips(capture_path, out_dir, flat_prob)
    result = compose(capture_path, plan_doc, floors_run, flips_run, "analyse")
    window_json = os.path.join(out_dir, "d41-window.json")
    window_md = os.path.join(out_dir, "d41-window.md")
    with open(window_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
        handle.write(chr(10))
    with open(window_md, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    write_manifest(out_dir, [{"path": window_json}, {"path": window_md}])
    print(render_markdown(result))
    print("wrote " + window_json + " and " + window_md)
    if result["status"] == "OK-BARS-WRITTEN":
        return 0
    if result["status"] == "REQUIRES_REVIEW":
        return 3
    return 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description="One-command D41 capture window: plan, capture, strict floors, flip investigation")
    parser.add_argument("mode", choices=("capture", "analyse"), help="capture a live instance, or analyse an existing capture offline")
    parser.add_argument("--capture", help="existing sparkflash-d41-capture/1 JSON (analyse mode)")
    parser.add_argument("--out-dir", required=True, help="artefact directory for this window (created if absent)")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="frozen capture plan (capture mode)")
    parser.add_argument("--plan-sha256", default=FROZEN_PLAN_SHA256, help="expected plan sha256; pass '' to skip the frozen-plan check only for a reviewed plan change")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-root", default=DEFAULT_EXPECT_ROOT)
    parser.add_argument("--capture-id", default="d41-window")
    parser.add_argument("--tag", default="d41")
    parser.add_argument("--source-revision")
    parser.add_argument("--flat-prob", type=float, default=0.05, help="floor flat threshold (default 0.05; widening it to pass a capture is not a measurement)")
    parser.add_argument("--dry-run", action="store_true", help="capture mode only: validate the plan and the frozen-plan sha, then stop before any HTTP request")
    parser.add_argument("--skip-dry-run", action="store_true", help="capture mode only: do not run the plan dry-run (the live capture still guards the root)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    try:
        if args.mode == "analyse":
            if not args.capture:
                raise WindowError("analyse mode needs --capture <capture.json>")
            if args.plan_sha256 == FROZEN_PLAN_SHA256 and os.path.exists(args.plan):
                plan_doc = plan_check(args.plan, args.plan_sha256)
            else:
                plan_doc = None
            return analyse(args.capture, out_dir, args.source_revision, args.flat_prob, plan_doc)
        if args.capture:
            raise WindowError("capture mode does not take --capture; use analyse mode for an existing capture")
        if os.path.exists(out_dir) and os.listdir(out_dir):
            raise WindowError(f"--out-dir {out_dir}: not empty; use a fresh directory per window so artefacts are never mixed")
        plan_doc = plan_check(args.plan, args.plan_sha256 or None)
        os.makedirs(out_dir, exist_ok=True)
        logs = {"dry_run": os.path.join(out_dir, "d41-plan-dry-run.log"), "recapture": os.path.join(out_dir, "d41-recapture.log")}
        if not args.skip_dry_run:
            code = dry_run(args.plan, args.base_url, logs["dry_run"])
            if code != 0:
                raise WindowError(f"plan dry-run failed (exit {code}); see {logs['dry_run']}")
        if args.dry_run:
            print("plan dry-run ok; stopping before any capture request (--dry-run)")
            print("plan sha256 " + plan_doc["sha256"])
            return 0
        code, capture_path, capture_log = capture(args.plan, args.base_url, out_dir, args.capture_id, args.tag, args.expect_root, logs)
        if code != 0:
            print("d41_window: capture refused (exit " + str(code) + "); the refusal log is " + capture_log, file=sys.stderr)
            return 2
        try:
            floors.validate_capture(read_json(capture_path), capture_path)
        except floors.D41Error as error:
            print("d41_window: capture written but invalid: " + str(error), file=sys.stderr)
            return 2
        return analyse(capture_path, out_dir, args.source_revision, args.flat_prob, plan_doc)
    except WindowError as error:
        print("d41_window: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
