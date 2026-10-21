#!/usr/bin/env python3
"""D41 close-margin flip investigation: make the 19 refused flips reviewable.

Why this exists
---------------
The 2026-10-20 idle-only D41 capture was refused by scripts/d41_floors.py with 19 rank-1 flips
above the flat-position threshold (top-1 probability >= 0.05) in the idle_distractor class. A count
is not an investigation: the next capture window needs the per-flip evidence on one page, or the
same refusal will recur with no actionable record.

This tool reads a sparkflash-d41-capture/1 document, re-computes the pair metrics with the exact
same code path as the floor tool (d41_floors.py), and writes:

* every rank-1 flip above the flat threshold as a row: block, pair, position, both top-1 token ids
  and logprobs/probabilities, the top-1 margins (top1 - top2 logprob), and both prompt lengths;
* per-pair and per-class aggregates, including positions compared, top-1 agreement, flips above the
  flat threshold and flips below it;
* a verification field: the per-class flip counts recomputed through d41_floors._pair_metrics and
  _aggregate_class, so the investigation cannot silently disagree with the refusing tool;
* the decision options: a reviewed probe-design change, or a reviewed decision about close-margin
  positions. Widening --flat-prob to pass a capture is explicitly not an option.

Exit codes: 0 = no flip above the flat threshold in the distractor classes; 3 = flips found (the
capture needs review before any floor can be frozen); 2 = the capture is invalid or unreadable.

Host-only, offline: no GPU, no node, no network. The capture is read, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import d41_floors as floors  # noqa: E402  (same repo tool, same metrics)

SCHEMA = "sparkflash-d41-flip-investigation/1"
DISTRACTOR_CLASSES = ("idle_distractor", "busy_distractor")


class InvestigationError(Exception):
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
        raise InvestigationError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise InvestigationError(f"{path}: not valid JSON: {error}") from error


def top_two(record_position: dict) -> dict:
    ranked = record_position.get("ranked") or []
    if not ranked:
        return {}
    first_token, first_logprob = ranked[0]
    second_token, second_logprob = ranked[1] if len(ranked) > 1 else (None, None)
    return {
        "top1_token": first_token,
        "top1_logprob": first_logprob,
        "top1_prob": math.exp(first_logprob),
        "top2_token": second_token,
        "top2_logprob": second_logprob,
        "top2_prob": math.exp(second_logprob) if second_logprob is not None else None,
        "margin_nats": (first_logprob - second_logprob) if second_logprob is not None else None,
    }


def investigate(doc: dict, path: str, flat_prob: float, outlier_factor: float) -> dict:
    capture = floors.validate_capture(doc, path)
    classes_present = [cls for cls in floors.CLASSES if any(block["class"] == cls for block in capture["blocks"])]
    flips: list = []
    pair_rows_by_class: dict = {cls: [] for cls in DISTRACTOR_CLASSES}
    pair_aggregates: list = []
    for block in capture["blocks"]:
        if block["class"] not in DISTRACTOR_CLASSES:
            continue
        for left_id, right_id in block["pairs"]:
            left = block["records"][left_id]
            right = block["records"][right_id]
            where = f"{path}:{block['block_id']}:{left_id}<->{right_id}"
            metrics = floors._pair_metrics(left, right, where, flat_prob)
            pair_rows_by_class[block["class"]].append(metrics)
            pair_aggregates.append(
                {
                    "class": block["class"],
                    "block_id": block["block_id"],
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_prompt_len": left["prompt_len"],
                    "right_prompt_len": right["prompt_len"],
                    "positions_compared": metrics["positions"],
                    "position_first": metrics["position_first"],
                    "position_last": metrics["position_last"],
                    "top1_agreement": metrics["top1_agreement"],
                    "rank1_flips_above_flat": metrics["rank1_flips_above_flat"],
                    "rank1_flips_flat": metrics["rank1_flips_flat"],
                    "kl_mean": metrics["kl_mean"],
                    "abs_max": metrics["abs_max"],
                }
            )
            shared = sorted(set(left["positions"]).intersection(right["positions"]))
            for position in shared:
                left_pos = left["positions"][position]
                right_pos = right["positions"][position]
                top_left = top_two(left_pos)
                top_right = top_two(right_pos)
                same = top_left["top1_token"] == top_right["top1_token"]
                flat = min(top_left["top1_prob"], top_right["top1_prob"]) < flat_prob
                if same or flat:
                    continue
                flips.append(
                    {
                        "class": block["class"],
                        "block_id": block["block_id"],
                        "left_id": left_id,
                        "right_id": right_id,
                        "position": position,
                        "left_prompt_len": left["prompt_len"],
                        "right_prompt_len": right["prompt_len"],
                        "left": top_left,
                        "right": top_right,
                        "left_margin_nats": top_left["margin_nats"],
                        "right_margin_nats": top_right["margin_nats"],
                        "classification": "flip above flat threshold (both top-1 probabilities >= flat_prob)",
                    }
                )
    verified = {}
    for cls in DISTRACTOR_CLASSES:
        rows = pair_rows_by_class[cls]
        if not rows:
            verified[cls] = {"present": False}
            continue
        aggregate = floors._aggregate_class(rows, outlier_factor)
        expected = sum(row["rank1_flips_above_flat"] for row in rows)
        found = len([flip for flip in flips if flip["class"] == cls])
        verified[cls] = {
            "present": True,
            "pair_comparisons": len(rows),
            "flips_above_flat": aggregate["rank1"]["flips_above_flat"],
            "flips_flat": aggregate["rank1"]["flips_flat"],
            "investigation_flips": found,
            "flips_above_flat_total_check": expected,
            "agrees_with_investigation": found == expected == aggregate["rank1"]["flips_above_flat"],
        }
    total_flips = len(flips)
    per_class = {cls: len([flip for flip in flips if flip["class"] == cls]) for cls in DISTRACTOR_CLASSES}
    return {
        "schema": SCHEMA,
        "generated_utc": utc_now(),
        "status": "REQUIRES_REVIEW" if total_flips else "NO_FLIPS_ABOVE_FLAT",
        "capture": {"path": os.path.abspath(path), "sha256": sha256_file(path), "capture_id": capture["capture_id"]},
        "parameters": {
            "flat_prob": flat_prob,
            "outlier_factor": outlier_factor,
            "note": "flat_prob is read from the floor tool's default; this investigation never widens it",
        },
        "classes_present": classes_present,
        "summary": {
            "distractor_pairs": sum(len(rows) for rows in pair_rows_by_class.values()),
            "positions_compared": sum(row["positions_compared"] for row in pair_aggregates),
            "flips_above_flat": total_flips,
            "flips_above_flat_by_class": per_class,
            "classes_missing": [cls for cls in floors.CLASSES if cls not in classes_present],
        },
        "verified_against_floor_tool": verified,
        "pairs": pair_aggregates,
        "flips": flips,
        "decision_options": [
            "A reviewed probe-design change: adjust the close-margin prompt pairs so the compared positions are not near a top-1 boundary, then re-capture under the same frozen plan discipline.",
            "A reviewed decision about close-margin positions: exclude or separately classify such positions in the floor computation, with the 19-flip evidence attached.",
            "NOT acceptable: widening --flat-prob to pass the capture. It would hide the same flips and is not a measurement.",
        ],
        "non_claims": [
            "this tool does not freeze a floor and never writes a sparkflash-d41-bars/1 file",
            "a flip count is evidence about the served reference's self-consistency, not about device parity",
            "the capture remains the frozen input of scripts/d41_floors.py; this report is derived evidence",
        ],
    }


def render_markdown(doc: dict) -> str:
    lines = []
    lines.append("# D41 close-margin flip investigation (" + doc["generated_utc"] + ")")
    lines.append("")
    lines.append("Status: **" + doc["status"] + "**  ")
    lines.append("Capture: " + doc["capture"]["path"] + "  ")
    lines.append("Capture sha256: " + doc["capture"]["sha256"] + "  ")
    lines.append("flat_prob: " + str(doc["parameters"]["flat_prob"]) + " (not widened)")
    lines.append("")
    summary = doc["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append("- distractor pairs: " + str(summary["distractor_pairs"]))
    lines.append("- positions compared: " + str(summary["positions_compared"]))
    lines.append("- rank-1 flips above the flat threshold: **" + str(summary["flips_above_flat"]) + "**")
    for cls, count in summary["flips_above_flat_by_class"].items():
        lines.append("  - " + cls + ": " + str(count))
    if summary["classes_missing"]:
        lines.append("- classes missing: " + ", ".join(summary["classes_missing"]))
    lines.append("")
    lines.append("## Verified against the refusing tool")
    lines.append("")
    lines.append("| class | pairs | flips (floor tool) | flips (investigation) | agrees |")
    lines.append("| :-- | --: | --: | --: | :-- |")
    for cls, row in doc["verified_against_floor_tool"].items():
        if not row.get("present"):
            lines.append("| " + cls + " | 0 | - | - | class not present |")
            continue
        lines.append(
            "| " + cls + " | " + str(row["pair_comparisons"]) + " | " + str(row["flips_above_flat"]) + " | " + str(row["investigation_flips"]) + " | " + str(row["agrees_with_investigation"]) + " |"
        )
    lines.append("")
    if doc["flips"]:
        lines.append("## Flips")
        lines.append("")
        lines.append("| class | pair | position | left prompt | right prompt | left top-1 | right top-1 | left prob | right prob | left margin (nats) | right margin (nats) |")
        lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
        for flip in doc["flips"]:
            lines.append(
                "| " + flip["class"] + " | " + flip["left_id"] + " <-> " + flip["right_id"] + " | " + str(flip["position"])
                + " | " + str(flip["left_prompt_len"]) + " | " + str(flip["right_prompt_len"])
                + " | " + str(flip["left"]["top1_token"]) + " | " + str(flip["right"]["top1_token"])
                + " | " + str(round(flip["left"]["top1_prob"], 6)) + " | " + str(round(flip["right"]["top1_prob"], 6))
                + " | " + str(round(flip["left_margin_nats"], 6) if flip["left_margin_nats"] is not None else None)
                + " | " + str(round(flip["right_margin_nats"], 6) if flip["right_margin_nats"] is not None else None)
                + " |"
            )
        lines.append("")
    lines.append("## Decision options")
    lines.append("")
    for option in doc["decision_options"]:
        lines.append("- " + option)
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    for claim in doc["non_claims"]:
        lines.append("- " + claim)
    lines.append("")
    return chr(10).join(lines) + chr(10)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Investigate D41 close-margin rank-1 flips (offline, no GPU)")
    parser.add_argument("--capture", required=True, help="sparkflash-d41-capture/1 JSON capture")
    parser.add_argument("--flat-prob", type=float, default=floors.DEFAULT_FLAT_PROB if hasattr(floors, "DEFAULT_FLAT_PROB") else 0.05, help="flat-position threshold (default: the floor tool's 0.05; never widen to pass)")
    parser.add_argument("--outlier-factor", type=float, default=3.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_md = args.out_md or (os.path.splitext(args.out_json)[0] + ".md")
    try:
        doc = read_json(args.capture)
        report = investigate(doc, args.capture, args.flat_prob, args.outlier_factor)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
            handle.write(chr(10))
        with open(out_md, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(report))
        print(render_markdown(report))
        print("wrote " + args.out_json + " and " + out_md)
        if report["summary"]["flips_above_flat"]:
            return 3
        return 0
    except (InvestigationError, floors.D41Error) as error:
        print("d41_flip_investigation: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
