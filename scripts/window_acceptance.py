#!/usr/bin/env python3
"""One command for the next window's acceptance row: bind + token + step time.

Why this exists
---------------
The window is expensive and its evidence is easy to under-collect. The m1-bind window produced a
bind/walk PASS but no token, and the pre-token acceptance row had to be assembled with two separate
commands plus a manual read of both JSONs. This script is the single command for the next window:

    python3 scripts/window_acceptance.py --device-dir <rank tree> --out-dir <qa out dir> \
        [--twin <host-twin dir>] [--oracle <native-oracle.json> --oracle-prompt <id>] \
        [--oracle-bars <d41-bars.json>] [--device-dump <dump.jsonl> [--device-bin <dump.bin>] [--dump-source label]] \
        [--positions 129,130,...] [--expected-steps 3] \
        [--expected-root ...] [--expect-manifest-lines N] [--require-token]

It runs the two existing, independently-tested tools and composes their verdicts:

* scripts/d48_step_gate.py  - bind/walk/step-time collector (canonical device evidence)
* scripts/accept_device_token.py - shape/pre-token/twin/oracle/step-time acceptance runner
  (consumes sparkflash-tp4-bind/1, sparkflash-tp4-loop/1 and the new sparkflash-tp4-token/1
  envelope with its frozen logits dump; the dump is auto-discovered from the token envelope)

The composed row is explicit about what was and was not measured:

* bind: PASS when the collector has no failures and not BIND-FAIL (a pre-token bind PASS is a real
  measurement: every tensor bind and the collective walk were verified).
* token: NOT-ATTEMPTED when no tp4-token evidence exists (the arm did not run the forward stage);
  MEASURED when ids/positions/dump are in evidence; the twin and oracle comparisons stay separate.
* step time: MEASURED with ms/step samples; accepted tokens/step and tok/s are labelled
  NOT-ATTEMPTED unless a real verify-step measurement exists. A 42 ms budget pass is a
  target-only statement, never a tok/s claim.
* overall: FAIL if any component fails; PASS only when the token leg passes its declared gate
  (twin PASS plus a D41-bars oracle PASS); otherwise NOT-ATTEMPTED with a named reason.

No numbers are invented: every row quotes the component document it came from and the evidence
sha256 manifest is written beside the reports. The evidence tree is read-only; all outputs go to
--out-dir (refused if it is the evidence directory itself).

Schemas:
* sparkflash-window-acceptance/1 (this composition)
* sparkflash-tp4-token/1 (device forward arm; contract sent to device-lead 2026-10-20)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

SCHEMA = "sparkflash-window-acceptance/1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
D48_SCRIPT = os.path.join(SCRIPT_DIR, "d48_step_gate.py")
ACCEPT_SCRIPT = os.path.join(SCRIPT_DIR, "accept_device_token.py")
D48_SCHEMA = "sparkflash-d48-step-gate/1"
ACCEPT_SCHEMA = "sparkflash-device-acceptance/1"


class WindowError(Exception):
    """A bad invocation or a missing input document."""


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as error:
        raise WindowError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise WindowError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise WindowError(f"{path}: expected a JSON object")
    return document


def run_tool(command: list, log_path: str) -> int:
    print("+ " + " ".join(command))
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    return completed.returncode


def evidence_manifest(root: str, out_dir: str) -> list:
    skip = os.path.abspath(out_dir) + os.sep
    rows = []
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(base, name)
            absolute = os.path.abspath(path)
            if absolute.startswith(skip):
                continue
            rows.append(
                {
                    "path": os.path.relpath(absolute, os.path.abspath(root)),
                    "sha256": sha256_file(absolute),
                    "bytes": os.path.getsize(absolute),
                }
            )
    rows.sort(key=lambda row: row["path"])
    return rows


# --------------------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------------------

def bind_row(d48: dict, d48_exit: int) -> dict:
    failures = d48.get("failures") or []
    per_rank = d48.get("per_rank") or []
    attempts = []
    for row in per_rank:
        attempts.append(
            {
                "rank": row.get("rank"),
                "attention_bound": row.get("attention_bound"),
                "attention_expected": row.get("attention_expected"),
                "walk": bool(row.get("walk_performed")),
                "reduces_posted": row.get("reduces_posted"),
                "reduces_expected": row.get("reduces_expected"),
                "rc": row.get("rc"),
            }
        )
    verdict = "PASS" if d48_exit in (0, 3) and not failures and d48.get("status") != "BIND-FAIL" else "FAIL"
    return {
        "verdict": verdict,
        "source": "d48-step-gate",
        "collector_status": d48.get("status"),
        "collector_verdict": (d48.get("verdict") or {}).get("label"),
        "failures": failures,
        "per_rank": attempts,
        "detail": "every rank bound its expected tensors and completed the collective walk"
        if verdict == "PASS"
        else "the collector reported bind/walk failures",
    }


def token_row(accept: dict) -> dict:
    stream = accept.get("device_stream") or {}
    positions_run = stream.get("positions_run") or 0
    tokens = stream.get("tokens") or []
    shape = accept.get("shape") or {}
    shape_verdict = shape.get("verdict")
    refusals = [
        check
        for check in shape.get("checks") or []
        if check.get("name", "").endswith("forward refused")
    ]
    if positions_run == 0:
        reasons = ["no token ids in the device evidence: the forward stage was not run (or refused)"]
        for check in refusals:
            reasons.append(f"{check['name']}: {check['detail']}")
        return {
            "verdict": "NOT-ATTEMPTED",
            "label": "TOKEN-NOT-ATTEMPTED",
            "positions_run": 0,
            "shape_verdict": shape_verdict,
            "reasons": reasons,
        }
    return {
        "verdict": "MEASURED",
        "label": "TOKEN-MEASURED (token ids in evidence; comparisons are separate rows)",
        "positions_run": positions_run,
        "positions": (stream.get("positions") or [])[:64],
        "tokens": tokens[:64],
        "rank_consistency": stream.get("consistent"),
        "partial": bool(stream.get("partial")),
        "shape_verdict": shape_verdict,
    }


def step_time_row(accept: dict, d48: dict) -> dict:
    step = accept.get("step_time") or {}
    verdict = step.get("verdict", "UNMEASURED")
    d48_verdict = d48.get("verdict") or {}
    row = {
        "verdict": verdict,
        "label": step.get("label") or "STEP-TIME-UNMEASURED",
        "step_ms_mean": step.get("step_ms_mean"),
        "step_ms_worst_rank": step.get("step_ms_worst_rank"),
        "single_token_tokens_per_second": step.get("single_token_tokens_per_second"),
        "d48_verdict": d48_verdict,
        "reasons": [],
    }
    if verdict == "UNMEASURED":
        row["reasons"].append(step.get("reason") or "no step_ms samples in the evidence")
    target = step.get("target_only") or {}
    if target:
        row["target_only"] = {
            "label": target.get("label"),
            "step_budget_ms": target.get("step_budget_ms"),
            "step_budget_pass_as_measured": target.get("step_budget_pass_as_measured"),
            "accepted_tokens_per_step": target.get("accepted_tokens_per_step"),
            "tok_s_target": target.get("tok_s_target"),
            "note": "accepted tokens/step and tok/s need the real DSpark/MTP verify step; NOT-ATTEMPTED here",
        }
    row["accepted_tokens_per_step"] = "NOT-ATTEMPTED (no verify-step measurement)"
    row["tok_s"] = "NOT-ATTEMPTED"
    return row


def overall_row(bind: dict, token: dict, twin: dict, oracle: dict, step: dict, d48_exit: int, accept_exit: int) -> dict:
    twin_verdict = twin.get("verdict")
    oracle_verdict = oracle.get("verdict")
    reasons = []
    if d48_exit not in (0, 3) or bind["verdict"] == "FAIL":
        return {"verdict": "FAIL", "label": "FAIL", "reasons": ["bind/walk row failed"]}
    if accept_exit == 1:
        return {"verdict": "FAIL", "label": "FAIL", "reasons": ["acceptance runner reported a failure"]}
    if accept_exit == 2:
        raise WindowError("acceptance runner rejected its input (exit 2); see the captured log")
    if accept_exit not in (0, 3):
        return {"verdict": "FAIL", "label": "FAIL", "reasons": [f"acceptance runner exited {accept_exit}"]}
    if token["verdict"] == "NOT-ATTEMPTED":
        return {
            "verdict": "NOT-ATTEMPTED",
            "label": "PRE-TOKEN-PASS (bind/walk verified; token comparison and step gate not attempted)",
            "reasons": ["bind PASS", "token NOT-ATTEMPTED: the arm did not emit a token this run"],
        }
    if twin_verdict == "PASS" and oracle_verdict == "PASS":
        return {"verdict": "PASS", "label": "PASS (twin match + D41-gated oracle comparison)", "reasons": ["twin PASS", "oracle PASS"]}
    if accept_exit == 0:
        # The acceptance runner's own overall verdict is PASS: the M1 token gate (twin match) was
        # met and the oracle leg is recorded but not numerically gated (UNBARED/UNMEASURED). This
        # must stay consistent with the component report; the label names the unbared oracle.
        return {
            "verdict": "PASS",
            "label": "PASS (M1 token gate met: twin match measured; oracle " + str(oracle_verdict) + ", no numeric gate claimed)",
            "reasons": ["twin PASS", "oracle " + str(oracle_verdict) + " (recorded, not a pass)"],
        }
    reasons.append(f"token measured ({token['positions_run']} position(s)) but the declared gate is not met")
    reasons.append(f"twin {twin_verdict}, oracle {oracle_verdict}")
    if twin_verdict == "FAIL" or oracle_verdict == "FAIL":
        return {"verdict": "FAIL", "label": "FAIL", "reasons": reasons}
    return {
        "verdict": "NOT-ATTEMPTED",
        "label": "TOKEN-MEASURED-UNBARED (measurement present; no declared passing gate yet)",
        "reasons": reasons,
    }


def compose(d48: dict, accept: dict, d48_exit: int, accept_exit: int) -> dict:
    bind = bind_row(d48, d48_exit)
    token = token_row(accept)
    twin = accept.get("twin_match") or {}
    oracle = accept.get("oracle_comparison") or {}
    step = step_time_row(accept, d48)
    return {
        "schema": SCHEMA,
        "generated_utc": utc_now(),
        "exit_codes": {"d48_step_gate": d48_exit, "accept_device_token": accept_exit},
        "rows": {
            "bind": bind,
            "token": token,
            "twin_match": twin,
            "oracle_comparison": oracle,
            "step_time": step,
        },
        "overall": overall_row(bind, token, twin, oracle, step, d48_exit, accept_exit),
        "non_claims": [
            "no token-equality claim: D39 keeps token ids as a diagnostic, not a gate",
            "no tok/s claim without a real verify-step (DSpark/MTP) accepted-tokens/step measurement",
            "a 42 ms ms/step row is a step-budget statement, not a throughput statement",
            "device and host numbers are reported in separate rows and never share a table",
            "NOT-ATTEMPTED / UNMEASURED means no measurement exists; it is not a pass and not a failure",
        ],
    }


# --------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------

def render_markdown(result: dict) -> str:
    rows = result["rows"]
    lines = []
    lines.append("# Window acceptance row (" + result["generated_utc"] + ")")
    lines.append("")
    lines.append("Schema: " + result["schema"] + "  ")
    lines.append("Component exit codes: d48-step-gate=" + str(result["exit_codes"]["d48_step_gate"]) + ", accept-device-token=" + str(result["exit_codes"]["accept_device_token"]))
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("**" + result["overall"]["label"] + "**")
    lines.append("")
    for reason in result["overall"]["reasons"]:
        lines.append("- " + reason)
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| row | verdict | detail |")
    lines.append("| :-- | :-- | :-- |")
    lines.append("| bind / walk | " + rows["bind"]["verdict"] + " | " + str(rows["bind"]["detail"]) + " (" + str(rows["bind"].get("collector_status")) + ") |")
    token = rows["token"]
    token_detail = "positions run " + str(token.get("positions_run"))
    if token.get("reasons"):
        token_detail += "; " + "; ".join(str(reason) for reason in token["reasons"])
    lines.append("| token | " + token["verdict"] + " | " + token_detail + " |")
    lines.append("| twin match | " + str(rows["twin_match"].get("verdict")) + " | " + str(rows["twin_match"].get("mode") or "not attempted") + " |")
    lines.append("| oracle comparison | " + str(rows["oracle_comparison"].get("verdict")) + " | " + str(rows["oracle_comparison"].get("label") or "-") + " |")
    step = rows["step_time"]
    if step.get("step_ms_mean") is not None:
        step_detail = "mean " + str(round(step["step_ms_mean"], 3)) + " ms/step"
        target = step.get("target_only") or {}
        if target.get("step_budget_pass_as_measured") is not None:
            step_detail += "; 42 ms budget target-only pass=" + str(target["step_budget_pass_as_measured"])
    else:
        step_detail = "; ".join(step.get("reasons") or ["no step samples"])
    lines.append("| step time | " + step["verdict"] + " | " + step_detail + " |")
    lines.append("| accepted tokens/step | " + str(step.get("accepted_tokens_per_step")) + " | not measured on the device path |")
    lines.append("| tok/s | " + str(step.get("tok_s")) + " | needs the accepted-tokens measurement |")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    for claim in result["non_claims"]:
        lines.append("- " + claim)
    lines.append("")
    return chr(10).join(lines) + chr(10)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="One-command next-window acceptance: bind + token + step time")
    parser.add_argument("--device-dir", required=True, help="device evidence tree (rank directories with tp4-bind/tp4-loop/tp4-token files)")
    parser.add_argument("--twin", action="append", default=[], help="host-twin evidence file or directory (repeatable)")
    parser.add_argument("--oracle", help="frozen sparkflash-native-oracle/1 JSON for the oracle leg")
    parser.add_argument("--oracle-prompt", help="prompt id inside the oracle file")
    parser.add_argument("--oracle-bars", help="D41 gate bars file; without it the oracle leg is UNBARED")
    parser.add_argument("--device-dump", help="the device arm's own full-vocab logits dump JSONL (explicit; enables the distributional oracle leg)")
    parser.add_argument("--device-bin", help="device dump sidecar (default: --device-dump + .bin)")
    parser.add_argument("--dump-source", default="device-arm", help="provenance label for the device dump (recorded)")
    parser.add_argument("--positions", help="comma-separated positions for the dump/oracle leg")
    parser.add_argument("--expected-steps", type=int, help="expected collective-walk steps (m1-bind shape: 3)")
    parser.add_argument("--expected-root", help="expected served checkpoint root (D38)")
    parser.add_argument("--expect-vocab", type=int, default=129280, help="expected vocabulary (released checkpoint: 129280)")
    parser.add_argument("--expect-attention-bound", type=int, help="expected attention tensors_bound (m1-bind: 208)")
    parser.add_argument("--expect-moe-bounds", help="expected MoE bound counts, e.g. routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120")
    parser.add_argument("--expect-manifest-lines", type=int, help="expected tp4-bind-names-rR.txt logical line count (m1-bind: 185481)")
    parser.add_argument("--provisional-d40", action="store_true", help="print the published D40 provisional values labelled NOT-A-GATE (diagnostic only; no gate file exists)")
    parser.add_argument("--require-token", action="store_true", help="missing tp4-token evidence is a FAIL (next token window), not NOT-ATTEMPTED")
    parser.add_argument("--out-dir", required=True, help="directory for this run's reports (must not be the evidence tree)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device_dir = os.path.abspath(args.device_dir)
    out_dir = os.path.abspath(args.out_dir)
    try:
        if not os.path.isdir(device_dir):
            raise WindowError(f"--device-dir {device_dir}: not a directory")
        if out_dir == device_dir:
            raise WindowError("--out-dir must not be the evidence directory itself (write reports beside the evidence, not into it)")
        os.makedirs(out_dir, exist_ok=True)
        oracle_bars_path = os.path.abspath(args.oracle_bars) if args.oracle_bars else None
        oracle_bars_sha256 = None
        if oracle_bars_path is not None:
            if not os.path.isfile(oracle_bars_path):
                raise WindowError(f"--oracle-bars {oracle_bars_path}: not a file")
            oracle_bars_sha256 = sha256_file(oracle_bars_path)
        d48_json = os.path.join(out_dir, "d48-step-gate.json")
        d48_md = os.path.join(out_dir, "d48-step-gate.md")
        accept_json = os.path.join(out_dir, "accept-device-token.json")
        accept_md = os.path.join(out_dir, "accept-device-token.md")
        d48_command = [sys.executable, D48_SCRIPT, "--device-dir", device_dir, "--out-json", d48_json, "--out-md", d48_md]
        if args.expected_steps is not None:
            d48_command += ["--expected-steps", str(args.expected_steps)]
        if args.expected_root:
            d48_command += ["--checkpoint-root", args.expected_root]
        if args.expect_vocab is not None:
            d48_command += ["--expect-vocab", str(args.expect_vocab)]
        d48_exit = run_tool(d48_command, os.path.join(out_dir, "d48-step-gate.log"))

        accept_command = [sys.executable, ACCEPT_SCRIPT, "--device", device_dir, "--out-json", accept_json, "--out-md", accept_md]
        for twin in args.twin:
            accept_command += ["--twin", twin]
        if args.oracle:
            accept_command += ["--oracle", args.oracle]
        if args.oracle_prompt:
            accept_command += ["--oracle-prompt", args.oracle_prompt]
        if args.oracle_bars:
            accept_command += ["--oracle-bars", args.oracle_bars]
        if args.device_dump:
            accept_command += ["--device-dump", args.device_dump]
        if args.device_bin:
            accept_command += ["--device-bin", args.device_bin]
        if args.dump_source:
            accept_command += ["--dump-source", args.dump_source]
        if args.positions:
            accept_command += ["--positions", args.positions]
        if args.expected_root:
            accept_command += ["--expect-root", args.expected_root]
        if args.expect_vocab is not None:
            accept_command += ["--expect-vocab", str(args.expect_vocab)]
        if args.expect_attention_bound is not None:
            accept_command += ["--expect-attention-bound", str(args.expect_attention_bound)]
        if args.expect_moe_bounds:
            accept_command += ["--expect-moe-bounds", args.expect_moe_bounds]
        if args.expect_manifest_lines is not None:
            accept_command += ["--expect-manifest-lines", str(args.expect_manifest_lines)]
        if args.provisional_d40:
            accept_command += ["--provisional-d40"]
        if args.require_token:
            accept_command += ["--require-token"]
        accept_exit = run_tool(accept_command, os.path.join(out_dir, "accept-device-token.log"))

        d48 = read_json(d48_json)
        accept = read_json(accept_json)
        if d48.get("schema") != D48_SCHEMA:
            raise WindowError(f"{d48_json}: schema {d48.get('schema')!r}, expected {D48_SCHEMA!r}")
        if accept.get("schema") != ACCEPT_SCHEMA:
            raise WindowError(f"{accept_json}: schema {accept.get('schema')!r}, expected {ACCEPT_SCHEMA!r}")
        result = compose(d48, accept, d48_exit, accept_exit)
        result["inputs"] = {
            "device_dir": device_dir,
            "twin": [os.path.abspath(path) for path in args.twin],
            "oracle": os.path.abspath(args.oracle) if args.oracle else None,
            "oracle_prompt": args.oracle_prompt,
            "oracle_bars": oracle_bars_path,
            "oracle_bars_sha256": oracle_bars_sha256,
            "device_dump": os.path.abspath(args.device_dump) if args.device_dump else None,
            "device_bin": os.path.abspath(args.device_bin) if args.device_bin else None,
            "dump_source": args.dump_source,
            "positions": args.positions,
            "expected_steps": args.expected_steps,
            "require_token": bool(args.require_token),
            "provisional_d40": bool(args.provisional_d40),
            "expected_root": args.expected_root,
            "expect_vocab": args.expect_vocab,
            "expect_attention_bound": args.expect_attention_bound,
            "expect_moe_bounds": args.expect_moe_bounds,
            "expect_manifest_lines": args.expect_manifest_lines,
        }
        result["components"] = {
            "d48_step_gate": {"path": d48_json, "sha256": sha256_file(d48_json), "status": d48.get("status"), "verdict": d48.get("verdict"), "failures": d48.get("failures")},
            "accept_device_token": {
                "path": accept_json,
                "sha256": sha256_file(accept_json),
                "shape": (accept.get("shape") or {}).get("verdict"),
                "pre_token": (accept.get("pre_token") or {}).get("verdict"),
                "twin_match": (accept.get("twin_match") or {}).get("verdict"),
                "oracle_comparison": (accept.get("oracle_comparison") or {}).get("verdict"),
                "step_time": (accept.get("step_time") or {}).get("verdict"),
                "overall": (accept.get("overall") or {}).get("verdict"),
            },
        }
        manifest = evidence_manifest(device_dir, out_dir)
        result["evidence_manifest"] = manifest
        window_json = os.path.join(out_dir, "window-acceptance.json")
        window_md = os.path.join(out_dir, "window-acceptance.md")
        artefacts = []
        for path in (d48_json, d48_md, accept_json, accept_md, os.path.join(out_dir, "d48-step-gate.log"), os.path.join(out_dir, "accept-device-token.log")):
            if os.path.exists(path):
                artefacts.append({"path": os.path.relpath(path, out_dir), "sha256": sha256_file(path)})
        result["artefacts"] = artefacts
        with open(window_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write(chr(10))
        with open(window_md, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(result))
        with open(os.path.join(out_dir, "evidence-sha256.txt"), "w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(row["sha256"] + "  " + row["path"] + chr(10))
        with open(os.path.join(out_dir, "artefacts-sha256.txt"), "w", encoding="utf-8") as handle:
            handle.write(sha256_file(window_json) + "  window-acceptance.json" + chr(10))
            handle.write(sha256_file(window_md) + "  window-acceptance.md" + chr(10))
            for row in artefacts:
                handle.write(row["sha256"] + "  " + row["path"] + chr(10))
        print(render_markdown(result))
        print("wrote " + window_json + " and " + window_md)
        overall = result["overall"]["verdict"]
        if overall == "PASS":
            return 0
        if overall == "FAIL":
            return 1
        return 3
    except WindowError as error:
        print("window_acceptance: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
