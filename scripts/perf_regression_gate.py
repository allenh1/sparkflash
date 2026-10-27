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

"""CI performance-regression gate over captured evidence: decode tok/s and TTFT rows.

WHAT THIS GATE IS
    It reads two rows out of committed measurement evidence and fails only when a NEW capture is
    slower than the declared baseline by more than a declared band. There is no GPU in CI and no
    server is contacted: the numbers are already on disk. The gate is therefore honest by
    construction - it cannot fake a measurement, it can only compare two recorded ones, and it
    SKIPS when no fresh, comparable, quiet capture exists.

    Inputs (both are docs/build JSON written by scripts/d38_vllm_recapture.py):

      * baseline: the committed deployment baseline (schema sparkflash-d38-vllm-baseline/1).
      * current : the newest evidence file in the evidence directory, discovered by name and
                  accepted only if it is strictly newer than the baseline, recent enough, from the
                  SAME served model root, captured with the SAME prompt/max_tokens/recipe, with at
                  least --min-samples ok samples and a QUIET load verdict.

    When no acceptable current file exists the gate prints the exact evidence directory, the exact
    glob it searched, the exact expected path, and the rejection reason for every candidate it
    found, then exits 3. Exit 3 is a SKIP: the CI job maps it to success with a loud SKIP line,
    because "no device evidence" is not the same as "the numbers pass".

DECLARED BANDS (declared here, not inferred from a run)
    --decode-band (default 0.10): fail when current decode tok/s p50 < baseline p50 * (1 - band).
    --ttft-band   (default 0.10): fail when current TTFT ms p90      > baseline p90 * (1 + band).
    Both bands are relative to the baseline capture's own p50/p90. A wider band is a weaker gate
    and must be a deliberate edit to this file's default or the CI invocation; nothing loosens it
    silently.

EXIT CODES
    0 PASS  - a fresh comparable quiet capture is within both bands.
    1 REGRESSION - a fresh comparable quiet capture is outside at least one band.
    2 INVALID - the baseline is missing/malformed, or the invocation is wrong. This is a broken
                gate, not a skip; CI fails loudly.
    3 SKIP  - no fresh, comparable, quiet current capture. Prints the exact searched paths.

No third-party packages; stdlib only. Tests: scripts/tests/test_perf_regression_gate.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

SCHEMA = "sparkflash-d38-vllm-baseline/1"
DEFAULT_BASELINE = os.path.join("docs", "build", "d38-vllm-baseline-20261021.json")
DEFAULT_EVIDENCE_DIR = os.path.join("docs", "build")
DEFAULT_EVIDENCE_GLOB = "d38-vllm-baseline-*.json"
DEFAULT_PROMPT = "b69-fixture-130"
DEFAULT_DECODE_BAND = 0.10
DEFAULT_TTFT_BAND = 0.10
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MAX_AGE_DAYS = 30.0

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_INVALID = 2
EXIT_SKIP = 3


class GateInvalid(Exception):
    """Raised for a broken gate input (baseline missing/malformed). Exit 2."""


def parse_utc(text: str) -> datetime:
    """Parse the evidence timestamp format. Raises ValueError on anything else."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def utc_stamp(moment: datetime) -> str:
    """Format a datetime in the evidence timestamp format."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: str) -> dict:
    """Load a JSON object, raising GateInvalid with a readable reason."""
    if not os.path.isfile(path):
        raise GateInvalid(f"evidence file does not exist: {os.path.abspath(path)}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise GateInvalid(f"cannot parse evidence file {os.path.abspath(path)}: {error}") from error
    if not isinstance(document, dict):
        raise GateInvalid(f"evidence file {os.path.abspath(path)} is not a JSON object")
    return document


def find_prompt_row(document: dict, prompt_id: str) -> dict:
    """Return the prompt row with this id, or raise GateInvalid naming what is available."""
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise GateInvalid("evidence has no prompts list")
    available = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        available.append(prompt.get("id"))
        if prompt.get("id") == prompt_id:
            return prompt
    raise GateInvalid(f"evidence has no prompt row {prompt_id!r}; available: {available}")


def require_number(value, label: str) -> float:
    """A row number must be a real finite number; a null is a broken input, never zero."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateInvalid(f"{label} is not a number: {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise GateInvalid(f"{label} is not finite: {value!r}")
    return float(value)


def extract_row(document: dict, prompt_id: str) -> dict:
    """Extract the gate inputs from one prompt row of one evidence document."""
    prompt = find_prompt_row(document, prompt_id)
    aggregate = prompt.get("aggregate")
    if not isinstance(aggregate, dict):
        raise GateInvalid(f"prompt row {prompt_id!r} has no aggregate")
    decode = aggregate.get("decode_tok_s")
    ttft = aggregate.get("ttft_ms")
    if not isinstance(decode, dict) or not isinstance(ttft, dict):
        raise GateInvalid(f"prompt row {prompt_id!r} has no decode_tok_s/ttft_ms aggregate")
    samples_ok = aggregate.get("samples_ok")
    if not isinstance(samples_ok, int):
        raise GateInvalid(f"prompt row {prompt_id!r} has no integer samples_ok")
    return {
        "prompt": prompt_id,
        "max_tokens": prompt.get("max_tokens"),
        "samples_ok": samples_ok,
        "decode_tok_s_p50": require_number(decode.get("p50"), f"{prompt_id} decode_tok_s.p50"),
        "decode_tok_s_p90": decode.get("p90"),
        "decode_tok_s_label": decode.get("label"),
        "ttft_ms_p50": ttft.get("p50"),
        "ttft_ms_p90": require_number(ttft.get("p90"), f"{prompt_id} ttft_ms.p90"),
        "ttft_ms_label": ttft.get("label"),
    }


def served_root_id(document: dict) -> str | None:
    """The served checkpoint root id recorded in the evidence (D38 precondition block)."""
    root = document.get("d38_root_precondition")
    if isinstance(root, dict) and isinstance(root.get("served_root_id"), str):
        return root["served_root_id"]
    return None


def load_verdict(document: dict) -> str | None:
    """The recorded load verdict (QUIET / UNDER-LOAD / MIXED / UNKNOWN)."""
    conditions = document.get("load_conditions")
    if isinstance(conditions, dict) and isinstance(conditions.get("verdict"), str):
        return conditions["verdict"]
    return None


def recipe_signature(document: dict, row: dict) -> dict:
    """The knobs that must match for two captures to be comparable at all."""
    recipe = document.get("recipe") if isinstance(document.get("recipe"), dict) else {}
    return {
        "prompt": row.get("prompt"),
        "max_tokens": row.get("max_tokens"),
        "temperature": recipe.get("temperature"),
        "top_p": recipe.get("top_p"),
        "seed": recipe.get("seed"),
        "stream": recipe.get("stream"),
        "ignore_eos": recipe.get("ignore_eos"),
    }


def candidate_rejection(
    document: dict,
    *,
    baseline_document: dict,
    baseline_row: dict,
    baseline_generated: datetime,
    now: datetime,
    max_age_days: float,
    min_samples: int,
    prompt_id: str,
) -> str | None:
    """Return None when this candidate is gate-grade comparable, else the exact reason."""
    schema = document.get("schema")
    if schema != SCHEMA:
        return f"schema is {schema!r}, expected {SCHEMA!r}"
    generated_text = document.get("generated_utc")
    if not isinstance(generated_text, str):
        return "no generated_utc timestamp"
    try:
        generated = parse_utc(generated_text)
    except ValueError:
        return f"generated_utc is not the evidence timestamp format: {generated_text!r}"
    if generated <= baseline_generated:
        return (
            f"not newer than the baseline ({generated_text} <= {utc_stamp(baseline_generated)}); "
            "the gate needs a NEW capture, not the baseline itself"
        )
    age_days = (now - generated).total_seconds() / 86400.0
    if age_days > max_age_days:
        return f"stale: {age_days:.1f} days old > --max-age-days {max_age_days}"
    if age_days < -0.05:
        return f"generated_utc is in the future relative to --now ({generated_text})"
    if served_root_id(document) != served_root_id(baseline_document):
        return (
            f"served root differs ({served_root_id(document)!r} != "
            f"{served_root_id(baseline_document)!r}); numbers from another checkpoint are not "
            "comparable"
        )
    verdict = load_verdict(document)
    if verdict != "QUIET":
        return (
            f"load verdict is {verdict!r}, not QUIET: an under-load capture is not comparable to "
            "a baseline capture and is not a single-stream number"
        )
    try:
        row = extract_row(document, prompt_id)
    except GateInvalid as error:
        return str(error)
    if row["samples_ok"] < min_samples:
        return f"only {row['samples_ok']} ok samples, need --min-samples {min_samples}"
    if recipe_signature(document, row) != recipe_signature(baseline_document, baseline_row):
        return (
            "recipe mismatch: {} != {}".format(
                recipe_signature(document, row), recipe_signature(baseline_document, baseline_row)
            )
        )
    return None


def discover_candidates(
    evidence_dir: str,
    evidence_glob: str,
    baseline_path: str,
    *,
    baseline_document: dict,
    baseline_row: dict,
    baseline_generated: datetime,
    now: datetime,
    max_age_days: float,
    min_samples: int,
    prompt_id: str,
) -> list[dict]:
    """Every evidence file matching the glob, with its acceptance decision (newest first)."""
    pattern = os.path.join(evidence_dir, evidence_glob)
    considered = []
    for path in sorted(glob.glob(pattern)):
        if os.path.abspath(path) == os.path.abspath(baseline_path):
            considered.append(
                {"path": os.path.abspath(path), "is_baseline": True, "accepted": False,
                 "reason": "this is the baseline file itself"}
            )
            continue
        record = {"path": os.path.abspath(path), "is_baseline": False}
        try:
            document = load_json(path)
        except GateInvalid as error:
            record.update({"accepted": False, "reason": str(error)})
            considered.append(record)
            continue
        reason = candidate_rejection(
            document,
            baseline_document=baseline_document,
            baseline_row=baseline_row,
            baseline_generated=baseline_generated,
            now=now,
            max_age_days=max_age_days,
            min_samples=min_samples,
            prompt_id=prompt_id,
        )
        record["accepted"] = reason is None
        record["reason"] = reason
        record["document"] = document
        record["generated_utc"] = document.get("generated_utc")
        considered.append(record)

    def sort_key(record: dict) -> str:
        text = record.get("generated_utc") or ""
        return text

    considered.sort(key=sort_key, reverse=True)
    return considered


def compare_rows(
    baseline_row: dict, current_row: dict, decode_band: float, ttft_band: float
) -> dict:
    """The declared-band comparison. Returns checks with PASS/REGRESSION per metric."""
    decode_floor = baseline_row["decode_tok_s_p50"] * (1.0 - decode_band)
    decode_ok = current_row["decode_tok_s_p50"] >= decode_floor
    ttft_ceiling = baseline_row["ttft_ms_p90"] * (1.0 + ttft_band)
    ttft_ok = current_row["ttft_ms_p90"] <= ttft_ceiling
    return {
        "decode_tok_s": {
            "baseline_p50": baseline_row["decode_tok_s_p50"],
            "current_p50": current_row["decode_tok_s_p50"],
            "floor": round(decode_floor, 3),
            "band": decode_band,
            "verdict": "PASS" if decode_ok else "REGRESSION",
            "label": "MEASURED-CLIENT decode tok/s p50 from each evidence file",
        },
        "ttft_ms": {
            "baseline_p90": baseline_row["ttft_ms_p90"],
            "current_p90": current_row["ttft_ms_p90"],
            "ceiling": round(ttft_ceiling, 3),
            "band": ttft_band,
            "verdict": "PASS" if ttft_ok else "REGRESSION",
            "label": "MEASURED-CLIENT TTFT ms p90 from each evidence file",
        },
        "verdict": "PASS" if (decode_ok and ttft_ok) else "REGRESSION",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI. Every band and input path is explicit."""
    parser = argparse.ArgumentParser(description="CI performance-regression gate over evidence JSON")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="committed baseline evidence")
    parser.add_argument(
        "--current",
        default=None,
        help="explicit current evidence; default = discover the newest comparable one",
    )
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--evidence-glob", default=DEFAULT_EVIDENCE_GLOB)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt row to gate")
    parser.add_argument("--decode-band", type=float, default=DEFAULT_DECODE_BAND)
    parser.add_argument("--ttft-band", type=float, default=DEFAULT_TTFT_BAND)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument(
        "--now", default=None, help="override now (ISO UTC) for deterministic tests"
    )
    parser.add_argument("--json-out", default=None, help="write the machine verdict to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline_path = os.path.abspath(args.baseline)
    now = parse_utc(args.now) if args.now else datetime.now(timezone.utc)

    print("perf-regression gate")
    print(f"  baseline evidence : {baseline_path}")
    print(f"  evidence dir      : {os.path.abspath(args.evidence_dir)}")
    print(f"  evidence glob     : {args.evidence_glob}")
    print(f"  prompt row        : {args.prompt}")

    try:
        baseline_document = load_json(baseline_path)
        if baseline_document.get("schema") != SCHEMA:
            raise GateInvalid(
                f"baseline schema is {baseline_document.get('schema')!r}, expected {SCHEMA!r}"
            )
        baseline_row = extract_row(baseline_document, args.prompt)
        generated_text = baseline_document.get("generated_utc")
        if not isinstance(generated_text, str):
            raise GateInvalid("baseline has no generated_utc timestamp")
        baseline_generated = parse_utc(generated_text)
    except (GateInvalid, ValueError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        print("perf-regression: FAIL (exit 2) - the gate input is broken, not skipped")
        return EXIT_INVALID

    print(
        "  baseline rows      : decode p50={} tok/s ({} ok samples), TTFT p90={} ms, "
        "max_tokens={}, generated={}".format(
            baseline_row["decode_tok_s_p50"],
            baseline_row["samples_ok"],
            baseline_row["ttft_ms_p90"],
            baseline_row["max_tokens"],
            utc_stamp(baseline_generated),
        )
    )
    if not baseline_document.get("gate_eligible"):
        print(
            "  baseline note      : gate_eligible=false (D38 deployment baseline, not a "
            "released-root parity row). It is still the declared reference; the comparison below "
            "treats it as the deployment's own bar."
        )

    if args.current:
        current_path = os.path.abspath(args.current)
        try:
            current_document = load_json(current_path)
        except GateInvalid as error:
            print(f"INVALID: explicit --current {error}", file=sys.stderr)
            print("perf-regression: FAIL (exit 2) - explicit current file is broken")
            return EXIT_INVALID
        reason = candidate_rejection(
            current_document,
            baseline_document=baseline_document,
            baseline_row=baseline_row,
            baseline_generated=baseline_generated,
            now=now,
            max_age_days=args.max_age_days,
            min_samples=args.min_samples,
            prompt_id=args.prompt,
        )
        if reason is not None:
            print(f"  current evidence  : {current_path}")
            print(f"  SKIP reason       : {reason}")
            print(
                "perf-regression: SKIP (exit 3) - the explicit current capture is not comparable "
                "gate-grade evidence; no number was compared and none was fabricated."
            )
            return EXIT_SKIP
        candidates = [
            {"path": current_path, "accepted": True, "document": current_document,
             "generated_utc": current_document.get("generated_utc")}
        ]
    else:
        candidates = discover_candidates(
            args.evidence_dir,
            args.evidence_glob,
            baseline_path,
            baseline_document=baseline_document,
            baseline_row=baseline_row,
            baseline_generated=baseline_generated,
            now=now,
            max_age_days=args.max_age_days,
            min_samples=args.min_samples,
            prompt_id=args.prompt,
        )

    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    if not accepted:
        expected = os.path.join(os.path.abspath(args.evidence_dir), args.evidence_glob)
        print(f"  current evidence  : NONE accepted")
        print("  searched          : " + expected)
        print(
            "  expected path     : a new capture file matching that glob, newer than {} and "
            "captured QUIET on served root {}".format(
                utc_stamp(baseline_generated), served_root_id(baseline_document)
            )
        )
        print(
            "  how to produce it : python3 scripts/d38_vllm_recapture.py --out-json "
            "<new path in docs/build/> --out-md <new path> --wait-idle-s <s> --idle-budget-s <s>"
        )
        if not candidates:
            print("  candidates checked: none matched the glob")
        for candidate in candidates:
            print(f"  candidate         : {candidate['path']}")
            print(f"    rejected        : {candidate.get('reason')}")
        print(
            "perf-regression: SKIP (exit 3) - no fresh, comparable, quiet evidence; nothing was "
            "compared and no number was fabricated."
        )
        return EXIT_SKIP

    chosen = accepted[0]
    current_path = chosen["path"]
    current_document = chosen["document"]
    current_row = extract_row(current_document, args.prompt)
    print(f"  current evidence  : {current_path}")
    print(
        "  current rows      : decode p50={} tok/s ({} ok samples), TTFT p90={} ms, "
        "max_tokens={}, generated={}".format(
            current_row["decode_tok_s_p50"],
            current_row["samples_ok"],
            current_row["ttft_ms_p90"],
            current_row["max_tokens"],
            current_document.get("generated_utc"),
        )
    )
    comparison = compare_rows(baseline_row, current_row, args.decode_band, args.ttft_band)
    print(
        "  decode tok/s      : {} vs baseline {} (floor {} = {}% band) -> {}".format(
            comparison["decode_tok_s"]["current_p50"],
            comparison["decode_tok_s"]["baseline_p50"],
            comparison["decode_tok_s"]["floor"],
            int(comparison["decode_tok_s"]["band"] * 100),
            comparison["decode_tok_s"]["verdict"],
        )
    )
    print(
        "  TTFT ms           : {} vs baseline {} (ceiling {} = {}% band) -> {}".format(
            comparison["ttft_ms"]["current_p90"],
            comparison["ttft_ms"]["baseline_p90"],
            comparison["ttft_ms"]["ceiling"],
            int(comparison["ttft_ms"]["band"] * 100),
            comparison["ttft_ms"]["verdict"],
        )
    )

    verdict = {
        "gate": "perf-regression",
        "baseline_path": baseline_path,
        "current_path": current_path,
        "prompt": args.prompt,
        "bands": {"decode_band": args.decode_band, "ttft_band": args.ttft_band},
        "comparison": comparison,
        "verdict": comparison["verdict"],
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(verdict, handle, indent=2)
            handle.write("\n")

    if comparison["verdict"] == "PASS":
        print("perf-regression: PASS (exit 0) - within the declared bands")
        return EXIT_PASS
    print(
        "perf-regression: FAIL (exit 1) - regression beyond the declared band; see the rows above"
    )
    return EXIT_REGRESSION


if __name__ == "__main__":
    sys.exit(main())
