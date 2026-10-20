#!/usr/bin/env python3
"""O3: compare a device full-vocabulary logits dump against the vLLM oracle.

DECISIONS D39 retires token equality: a correct native chain and a correct
oracle can generate different tokens, so the acceptance evidence is the
distribution each side assigns to the same prefix. This script joins

  * the device dump written by --dump-logits (one JSONL object per compared
    position plus a .bin sidecar of full-vocabulary little-endian f32 logits),
  * the oracle JSON written by scripts/oracle_native.py (id-keyed logprobs for
    the same positions, up to the recipe's --max-logprobs cap),

and reports, per position and in aggregate: top-1 agreement, top-k overlap,
max/mean |logprob delta| over the oracle's ids, KL(oracle || device) and the
symmetric JS divergence of the renormalised top-k distributions.

Bars are declared by the Lead in D40 (O1, task-13) and passed with --bars; this
script never invents a bar and never turns a token match into a pass. Without
--bars the run is reported as UNBARED: measured evidence, no verdict.

Position convention (see oracle_native.py): oracle positions are stored keyed
by device position p, so both sides join on the same integer. The dump's own
"position" is the device position from the hybrid_sequence token table.

The dump is checked for self-consistency before any metric: when a row carries
logits_bin it must name the dump's own FILE + ".bin" sidecar, the sidecar must be
exactly as long as the recorded blocks and the offsets must tile. prefix_match is
tri-state, and a row that cannot prove it is like-for-like (no usable
reference_id) is counted as a mismatch rather than silently matched. A run with any UNKNOWN or
MISMATCH row is NOT_LIKE_FOR_LIKE and exits 1 whatever the bars say: the metrics are recorded, but the
run is not a like-for-like comparison.

A --bars file is a gate input, not a hint: it must be a D41 bar file (schema
"sparkflash-d41-bars/1", status "ok", bar_class "idle_distractor"), and a demo-scale, provisional or
requires_review file is refused rather than applied. --expect-vocab is the operator's declaration of
the checkpoint's vocabulary (the released checkpoint is 129,280); a dump whose rows disagree is
refused. --positions, when given, must all exist in both the dump and the oracle: a list that would
silently compare fewer positions than it names is refused, not intersected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

DEFAULT_EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
TOP_K = (5, 20, 100, 1000)
ORACLE_SCHEMA = "sparkflash-native-oracle/1"
GATE_BARS_SCHEMA = "sparkflash-d41-bars/1"

#: Bars the harness knows how to evaluate. Threshold names carry their
#: direction: *_min is a floor, *_max is a ceiling. The Lead's D40 record may
#: use any subset; unknown names are an error so a typo cannot silently pass.
SUPPORTED_BARS = {
    "positions_min": ("positions", "min"),
    "top1_agreement_min": ("top1_agreement_rate", "min"),
    "top5_overlap_mean_min": ("top5_overlap_mean", "min"),
    "top20_overlap_mean_min": ("top20_overlap_mean", "min"),
    "top100_overlap_mean_min": ("top100_overlap_mean", "min"),
    "top1000_overlap_mean_min": ("top1000_overlap_mean", "min"),
    "max_abs_delta_max": ("max_abs_delta_max", "max"),
    "mean_abs_delta_max": ("mean_abs_delta_mean", "max"),
    "kl_mean_max": ("kl_mean", "max"),
    "kl_p95_max": ("kl_p95", "max"),
    "js_mean_max": ("js_mean", "max"),
}


class CompareError(RuntimeError):
    """Input that cannot be compared honestly."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise CompareError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CompareError(f"{path} is not JSON: {error}") from error


def read_dump(path: str) -> list[dict]:
    """Read the JSONL dump, validating the fields the comparison needs."""
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CompareError(f"{path}:{line_number}: invalid JSON: {error}") from error
                for key in ("position", "vocab", "logits_offset"):
                    if key not in row:
                        raise CompareError(f"{path}:{line_number}: missing {key!r}")
                if not isinstance(row["position"], int) or row["position"] < 0:
                    raise CompareError(f"{path}:{line_number}: bad position {row['position']!r}")
                if not isinstance(row["vocab"], int) or row["vocab"] <= 0:
                    raise CompareError(f"{path}:{line_number}: bad vocab {row['vocab']!r}")
                if not isinstance(row["logits_offset"], int) or row["logits_offset"] < 0:
                    raise CompareError(f"{path}:{line_number}: bad logits_offset")
                rows.append(row)
    except OSError as error:
        raise CompareError(f"cannot read {path}: {error}") from error
    if not rows:
        raise CompareError(f"{path}: dump has no positions")
    positions = [row["position"] for row in rows]
    if len(set(positions)) != len(positions):
        raise CompareError(f"{path}: duplicate positions in the dump")
    return rows


def read_logits(bin_path: str, offset: int, vocab: int) -> list[float]:
    try:
        with open(bin_path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read(4 * vocab)
    except OSError as error:
        raise CompareError(f"cannot read {bin_path}: {error}") from error
    if len(raw) != 4 * vocab:
        raise CompareError(
            f"{bin_path}: short read at offset {offset}: need {4 * vocab} bytes, got {len(raw)}"
        )
    return list(struct.unpack("<%df" % vocab, raw))


def check_dump_sidecar(rows: list[dict], dump_path: str, bin_path: str) -> None:
    """Refuse a dump whose rows and sidecar do not belong to each other.

    The portable derivation stays FILE + ".bin", so a copied dump is read next to itself.
    When the dump carries logits_bin (the writer always does) it must name that same
    sidecar, the file actually read must have that name, its length must be exactly the sum
    of the recorded blocks, and the offsets must tile in dump order - the frozen schema in
    docs/build/logits-dump.md. A dump moved next to another run's sidecar is refused instead
    of being compared silently.
    """
    expected_name = os.path.basename(dump_path) + ".bin"
    if os.path.basename(bin_path) != expected_name:
        raise CompareError(
            f"sidecar {bin_path!r} is not this dump's sidecar (expected basename {expected_name!r})"
        )
    total = 0
    for row in rows:
        recorded = row.get("logits_bin")
        if recorded is not None:
            if not isinstance(recorded, str) or not recorded:
                raise CompareError(f"position {row['position']}: logits_bin {recorded!r} is not a path")
            if os.path.basename(recorded) != expected_name:
                raise CompareError(
                    f"position {row['position']}: logits_bin {recorded!r} does not name this dump's "
                    f"sidecar {expected_name!r}"
                )
        if int(row["logits_offset"]) != total:
            raise CompareError(
                f"position {row['position']}: logits_offset {row['logits_offset']} does not tile "
                f"(expected {total})"
            )
        total += int(row["vocab"]) * 4
    try:
        size = os.path.getsize(bin_path)
    except OSError as error:
        raise CompareError(f"cannot stat {bin_path}: {error}") from error
    if size != total:
        raise CompareError(f"{bin_path}: {size} bytes for {total} bytes of recorded blocks")


def log_softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise CompareError("log_softmax of an empty vector")
    if any(value != value for value in logits):
        raise CompareError("log_softmax input contains NaN")
    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    total = 0.0
    for value in shifted:
        total += math.exp(value)
    if not total > 0.0 or not math.isfinite(total):
        raise CompareError("log_softmax normaliser is not finite")
    log_total = math.log(total)
    return [value - log_total for value in shifted]


def _ranked(logprobs: dict[int, float], limit: int | None = None) -> list[int]:
    ordered = sorted(logprobs.items(), key=lambda item: (-item[1], item[0]))
    ids = [token for token, _ in ordered]
    return ids if limit is None else ids[:limit]


def topk_overlap(left: list[int], right: list[int], k: int) -> float:
    """|A_k & B_k| / min(k, |A|, |B|); 0.0 when either list is empty."""
    if not left or not right:
        return 0.0
    denominator = max(1, min(k, len(left), len(right)))
    return len(set(left[:k]) & set(right[:k])) / float(denominator)


def _kl(p: list[float], q: list[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        total += pi * math.log(pi / qi)
    return total


def kl_divergence(p: list[float], q: list[float]) -> float:
    if len(p) != len(q) or not p:
        raise CompareError("KL needs two equal-length, non-empty distributions")
    return _kl(p, q)


def js_divergence(p: list[float], q: list[float]) -> float:
    if len(p) != len(q) or not p:
        raise CompareError("JS needs two equal-length, non-empty distributions")
    midpoint = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def _normalise(logprobs: list[float]) -> list[float]:
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    total = sum(weights)
    if not total > 0.0 or not math.isfinite(total):
        raise CompareError("cannot renormalise a distribution with no finite mass")
    return [weight / total for weight in weights]


def load_oracle(path: str) -> dict:
    oracle = load_json(path)
    if oracle.get("schema") != ORACLE_SCHEMA:
        raise CompareError(f"{path}: unknown oracle schema {oracle.get('schema')!r}")
    prompts = oracle.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise CompareError(f"{path}: oracle has no prompts")
    for prompt in prompts:
        if prompt.get("status") != "ok":
            continue
        positions = prompt.get("positions")
        if not isinstance(positions, list) or not positions:
            raise CompareError(f"{path}: prompt {prompt.get('id')!r} has no positions")
    return oracle


def select_oracle_prompt(oracle: dict, prompt_id: str | None) -> dict:
    prompts = [prompt for prompt in oracle["prompts"] if prompt.get("status") == "ok"]
    if prompt_id is not None:
        for prompt in prompts:
            if prompt.get("id") == prompt_id:
                return prompt
        raise CompareError(f"oracle has no ok prompt with id {prompt_id!r}")
    if len(prompts) != 1:
        raise CompareError(
            "oracle has several prompts; pass --oracle-prompt to choose one "
            f"(have {[prompt.get('id') for prompt in prompts]})"
        )
    return prompts[0]


def root_matches(root: str | None, expect: str | None) -> bool:
    if not expect:
        return True
    if not root:
        return False
    normalised = root.rstrip("/")
    return normalised == expect or normalised.endswith("/" + expect) or normalised.endswith(expect)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _prefix_match(row: dict, oracle_position: dict) -> bool | None:
    """Whether the dump proves this row is like-for-like with the oracle.

    True when the dumped reference token equals the oracle's teacher-forced target token,
    False when both are readable and disagree, and None when the row cannot prove it (no or
    unreadable reference_id, or no or unreadable oracle target_token). The aggregate counts
    None as a mismatch: a dump that cannot prove its prefix must never be scored as if it had.
    """
    reference = row.get("reference_id")
    target = oracle_position.get("target_token")
    if type(reference) is not int or type(target) is not int:
        return None
    return reference == target


def position_metrics(row: dict, logits: list[float], oracle_position: dict) -> dict:
    """One position: device logits (full vocab) against one oracle distribution."""
    vocab = int(row["vocab"])
    if len(logits) != vocab:
        raise CompareError(f"position {row['position']}: {len(logits)} logits for vocab {vocab}")
    if "logprobs" not in oracle_position:
        raise CompareError(f"position {row['position']}: oracle position has no logprobs")
    oracle_lp = {int(token): float(value) for token, value in oracle_position["logprobs"].items()}
    if not oracle_lp:
        raise CompareError(f"position {row['position']}: oracle distribution is empty")
    for token in oracle_lp:
        if not 0 <= token < vocab:
            raise CompareError(f"position {row['position']}: oracle token {token} outside vocab {vocab}")
    device_lp = log_softmax(logits)
    oracle_ids = _ranked(oracle_lp)
    device_ids = _ranked({token: device_lp[token] for token in range(vocab)}, limit=max(TOP_K))

    oracle_argmax = oracle_ids[0]
    device_argmax = max(range(vocab), key=lambda token: (device_lp[token], -token))
    deltas = [abs(device_lp[token] - oracle_lp[token]) for token in oracle_ids]
    p = _normalise([oracle_lp[token] for token in oracle_ids])
    q = _normalise([device_lp[token] for token in oracle_ids])
    oracle_mass = sum(math.exp(oracle_lp[token]) for token in oracle_ids)
    target = oracle_position.get("target_token")
    reference = row.get("reference_id")
    prefix_match = _prefix_match(row, oracle_position)
    return {
        "position": int(row["position"]),
        "oracle_argmax": oracle_argmax,
        "device_argmax": device_argmax,
        "target_token": target,
        "reference_id": reference,
        "prefix_match": prefix_match,
        "top1_agreement": oracle_argmax == device_argmax,
        "top5_overlap": topk_overlap(oracle_ids, device_ids, 5),
        "top20_overlap": topk_overlap(oracle_ids, device_ids, 20),
        "top100_overlap": topk_overlap(oracle_ids, device_ids, 100),
        "top1000_overlap": topk_overlap(oracle_ids, device_ids, 1000),
        "max_abs_delta": max(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "kl_oracle_to_device": kl_divergence(p, q),
        "js_divergence": js_divergence(p, q),
        "oracle_topk_mass": oracle_mass,
        "device_argmax_oracle_rank": (oracle_ids.index(device_argmax) + 1) if device_argmax in oracle_ids else None,
        "oracle_argmax_device_rank": (device_ids.index(oracle_argmax) + 1) if oracle_argmax in device_ids else None,
        "oracle_logprob_of_oracle_argmax": oracle_lp[oracle_argmax],
        "device_logprob_of_oracle_argmax": device_lp[oracle_argmax],
    }


def aggregate_metrics(positions: list[dict]) -> dict:
    if not positions:
        raise CompareError("no positions to aggregate")
    finite_kl = [row["kl_oracle_to_device"] for row in positions]
    aggregate = {
        "positions": len(positions),
        "position_first": positions[0]["position"],
        "position_last": positions[-1]["position"],
        "top1_agreement_rate": sum(1 for row in positions if row["top1_agreement"]) / len(positions),
        "prefix_mismatches": sum(1 for row in positions if row["prefix_match"] is not True),
        "prefix_unknown": sum(1 for row in positions if row["prefix_match"] is None),
        "max_abs_delta_max": max(row["max_abs_delta"] for row in positions),
        "max_abs_delta_mean": sum(row["max_abs_delta"] for row in positions) / len(positions),
        "max_abs_delta_median": _percentile([row["max_abs_delta"] for row in positions], 0.5),
        "max_abs_delta_p95": _percentile([row["max_abs_delta"] for row in positions], 0.95),
        "mean_abs_delta_mean": sum(row["mean_abs_delta"] for row in positions) / len(positions),
        "mean_abs_delta_median": _percentile([row["mean_abs_delta"] for row in positions], 0.5),
        "kl_mean": sum(finite_kl) / len(finite_kl),
        "kl_median": _percentile(finite_kl, 0.5),
        "kl_p95": _percentile(finite_kl, 0.95),
        "kl_max": max(finite_kl),
        "js_mean": sum(row["js_divergence"] for row in positions) / len(positions),
        "js_max": max(row["js_divergence"] for row in positions),
        "oracle_topk_mass_mean": sum(row["oracle_topk_mass"] for row in positions) / len(positions),
        "oracle_topk_mass_min": min(row["oracle_topk_mass"] for row in positions),
    }
    for k in TOP_K:
        values = [row[f"top{k}_overlap"] for row in positions]
        aggregate[f"top{k}_overlap_mean"] = sum(values) / len(values)
        aggregate[f"top{k}_overlap_median"] = _percentile(values, 0.5)
        aggregate[f"top{k}_overlap_min"] = min(values)
    return aggregate


def evaluate_bars(bars: dict, aggregate: dict) -> dict:
    declared = bars.get("bars") if isinstance(bars.get("bars"), dict) else bars
    if not isinstance(declared, dict) or not declared:
        raise CompareError("bars file must contain a non-empty 'bars' object")
    results = []
    for name, threshold in declared.items():
        if name not in SUPPORTED_BARS:
            raise CompareError(f"unknown bar {name!r}; known bars: {sorted(SUPPORTED_BARS)}")
        field, direction = SUPPORTED_BARS[name]
        value = aggregate[field]
        passed = value >= threshold if direction == "min" else value <= threshold
        results.append({"bar": name, "value": value, "threshold": threshold, "direction": direction, "pass": passed})
    return {"pass": all(result["pass"] for result in results), "bars": results}


def parse_positions(text: str) -> set[int]:
    """Parse --positions into a set of non-negative ints, refusing what a gate could miscount."""
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise CompareError("--positions names no positions")
    positions: list[int] = []
    for token in tokens:
        try:
            position = int(token)
        except ValueError as error:
            raise CompareError(f"--positions: {token!r} is not an integer") from error
        if position < 0:
            raise CompareError(f"--positions: {position} is negative")
        if position in positions:
            raise CompareError(f"--positions: duplicate position {position}")
        positions.append(position)
    return set(positions)


def check_gate_bars(bars: dict, path: str) -> None:
    """Refuse anything that is not a D41 gate bar file (D40/D41)."""
    if not isinstance(bars, dict):
        raise CompareError(f"{path}: bars file is not a JSON object")
    schema = bars.get("schema")
    if schema != GATE_BARS_SCHEMA:
        raise CompareError(
            f"{path}: not a D41 gate bars file (schema {schema!r}, expected {GATE_BARS_SCHEMA!r}); "
            "a demo-scale or hand-copied provisional bar file is not a gate"
        )
    status = bars.get("status")
    if status != "ok":
        raise CompareError(
            f"{path}: D41 bars status {status!r} is not 'ok'; a requires_review floor file must be "
            "reviewed and re-recorded before it can gate"
        )
    bar_class = bars.get("bar_class")
    if bar_class != "idle_distractor":
        raise CompareError(
            f"{path}: D41 bars were derived from bar_class {bar_class!r}, not 'idle_distractor'"
        )
    if not isinstance(bars.get("bars"), dict) or not bars["bars"]:
        raise CompareError(f"{path}: D41 bars file has no 'bars' object")


def render_markdown(result: dict) -> str:
    lines = [
        "# Native chain vs vLLM oracle (D39 distributional comparison)",
        "",
        f"- prompt: {result['prompt_id']!r}",
        f"- device dump: {result['dump_path']} (sha256 {result['dump_sha256'][:16]}...)",
        f"- oracle: {result['oracle_path']} (sha256 {result['oracle_sha256'][:16]}...)",
        f"- oracle served root: {result['oracle_served_root']}",
        f"- oracle response sha256: {result['oracle_response_sha256']}",
        f"- positions compared: {result['aggregate']['positions']} ({result['aggregate']['position_first']}..{result['aggregate']['position_last']})",
        f"- bars: {result['bars_path'] or 'none (UNBARED - measured metrics, no verdict)'}",
        f"- verdict: {result['verdict']}",
        "",
        "No token-equality claim is made (D39 rule 2).",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| :--- | ---: |",
    ]
    for key in (
        "positions",
        "top1_agreement_rate",
        "top5_overlap_mean",
        "top20_overlap_mean",
        "top100_overlap_mean",
        "top1000_overlap_mean",
        "max_abs_delta_max",
        "max_abs_delta_mean",
        "mean_abs_delta_mean",
        "kl_mean",
        "kl_p95",
        "js_mean",
        "oracle_topk_mass_mean",
        "prefix_mismatches",
        "prefix_unknown",
    ):
        value = result["aggregate"][key]
        lines.append(f"| {key} | {value:.6g} |" if isinstance(value, float) else f"| {key} | {value} |")
    if result["bars"] is not None:
        lines += ["", "## Bars", "", "| bar | value | threshold | pass |", "| :--- | ---: | ---: | :--- |"]
        for bar in result["bars"]["bars"]:
            lines.append(f"| {bar['bar']} | {bar['value']:.6g} | {bar['threshold']:.6g} | {bar['pass']} |")
    lines += ["", "## Per position", "",
              "| p | oracle argmax | device argmax | top1 | top5 | top20 | top100 | top1000 | max abs d | mean abs d | KL | JS | prefix |",
              "| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
    for row in result["positions"]:
        lines.append(
            "| {p} | {oa} | {da} | {t1} | {t5:.4f} | {t20:.4f} | {t100:.4f} | {t1000:.4f} | {m:.6g} | {a:.6g} | {kl:.6g} | {js:.6g} | {pm} |".format(
                p=row["position"],
                oa=row["oracle_argmax"],
                da=row["device_argmax"],
                t1="yes" if row["top1_agreement"] else "no",
                t5=row["top5_overlap"],
                t20=row["top20_overlap"],
                t100=row["top100_overlap"],
                t1000=row["top1000_overlap"],
                m=row["max_abs_delta"],
                a=row["mean_abs_delta"],
                kl=row["kl_oracle_to_device"],
                js=row["js_divergence"],
                pm=(
                    "ok" if row["prefix_match"] is True
                    else "UNKNOWN" if row["prefix_match"] is None
                    else "MISMATCH"
                ),
            )
        )
    lines += [
        "",
        "## Notes",
        "",
        "* KL is the truncation-renormalised KL over the oracle's id set; 'oracle_topk_mass_mean' reports",
        "  how much probability the oracle top-k list retains, so a KL over a heavily truncated list is",
        "  read in context. The direction is oracle || device; JS is symmetric and bounded.",
        "* 'prefix_match' is tri-state: 'ok' when the dumped reference token equals the oracle's",
        "  teacher-forced target token, 'MISMATCH' when both are readable and disagree, and 'UNKNOWN'",
        "  when the dump carries no usable reference_id (an older dump) or the oracle has no target",
        "  token. UNKNOWN counts in 'prefix_mismatches': the row is not proven like-for-like, so it is",
        "  never silently scored as a match.",
        "* 'NOT_LIKE_FOR_LIKE' means at least one compared row could not prove its prefix; the counts are",
        "  in 'prefix_mismatches' and 'prefix_unknown'. It overrides PASS and UNBARED and exits 1: the",
        "  numbers are recorded evidence, not a comparison verdict.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", required=True, help="device --dump-logits JSONL")
    parser.add_argument("--bin", help="logits sidecar (default: --dump + .bin)")
    parser.add_argument("--oracle", required=True, help="oracle JSON from scripts/oracle_native.py")
    parser.add_argument("--oracle-prompt", help="prompt id inside the oracle file (needed when it has several)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", help="Markdown evidence (default: --out-json with .md)")
    parser.add_argument("--bars", help="D41 gate bars JSON (sparkflash-d41-bars/1); without it the run is UNBARED")
    parser.add_argument("--expect-root", default=DEFAULT_EXPECT_ROOT)
    parser.add_argument("--positions", help="comma-separated device positions to compare (default: intersection)")
    parser.add_argument(
        "--expect-vocab",
        type=int,
        help="expected logits vocabulary (D40 released checkpoint: 129280); any row disagreeing is refused",
    )
    args = parser.parse_args(argv)
    if args.expect_root == "":
        args.expect_root = None
    bin_path = args.bin or (args.dump + ".bin")
    out_md = args.out_md or (os.path.splitext(args.out_json)[0] + ".md")
    try:
        oracle = load_oracle(args.oracle)
        if not root_matches(oracle.get("served", {}).get("root"), args.expect_root):
            raise CompareError(
                f"oracle served root {oracle.get('served', {}).get('root')!r} is not the released snapshot "
                f"{args.expect_root!r} (D38)"
            )
        prompt = select_oracle_prompt(oracle, args.oracle_prompt)
        oracle_positions = {int(row["position"]): row for row in prompt["positions"]}
        rows = read_dump(args.dump)
        check_dump_sidecar(rows, args.dump, bin_path)
        if args.expect_vocab is not None:
            for row in rows:
                if int(row["vocab"]) != args.expect_vocab:
                    raise CompareError(
                        f"position {row['position']}: vocab {row['vocab']} is not the expected "
                        f"{args.expect_vocab} (--expect-vocab, D40 released checkpoint)"
                    )
        wanted = parse_positions(args.positions) if args.positions else None
        if wanted is not None:
            dump_positions = {row["position"] for row in rows}
            missing_dump = sorted(wanted - dump_positions)
            if missing_dump:
                raise CompareError(
                    f"--positions names {missing_dump} but the dump has no such position(s) "
                    f"(dump positions {sorted(dump_positions)[:8]}...)"
                )
            missing_oracle = sorted(wanted - set(oracle_positions))
            if missing_oracle:
                raise CompareError(
                    f"--positions names {missing_oracle} but the oracle has no such position(s) "
                    f"(oracle positions {sorted(oracle_positions)[:8]}...)"
                )
        compared = (
            [row for row in rows if row["position"] in wanted]
            if wanted is not None
            else [row for row in rows if row["position"] in oracle_positions]
        )
        if not compared:
            raise CompareError(
                f"no overlapping positions: dump {sorted(row['position'] for row in rows)} "
                f"oracle {sorted(oracle_positions)[:8]}..."
            )
        compared.sort(key=lambda row: row["position"])
        results = []
        for row in compared:
            logits = read_logits(bin_path, int(row["logits_offset"]), int(row["vocab"]))
            results.append(position_metrics(row, logits, oracle_positions[row["position"]]))
        aggregate = aggregate_metrics(results)
        bars = load_json(args.bars) if args.bars else None
        if bars is not None:
            check_gate_bars(bars, args.bars)
        bar_result = evaluate_bars(bars, aggregate) if bars else None
        verdict = "UNBARED" if bar_result is None else ("PASS" if bar_result["pass"] else "FAIL")
        if aggregate["prefix_mismatches"] > 0:
            verdict = "NOT_LIKE_FOR_LIKE"
        result = {
            "schema": "sparkflash-native-comparison/1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_id": prompt.get("id"),
            "dump_path": os.path.abspath(args.dump),
            "dump_sha256": sha256_file(args.dump),
            "bin_path": os.path.abspath(bin_path),
            "bin_sha256": sha256_file(bin_path),
            "oracle_path": os.path.abspath(args.oracle),
            "oracle_sha256": sha256_file(args.oracle),
            "oracle_served_root": oracle.get("served", {}).get("root"),
            "oracle_response_sha256": prompt.get("response_sha256"),
            "bars_path": os.path.abspath(args.bars) if args.bars else None,
            "verdict": verdict,
            "bars": bar_result,
            "aggregate": aggregate,
            "positions": results,
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
        markdown = render_markdown(result)
        with open(out_md, "w", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.write("\n")
        print(markdown)
        print(f"wrote {args.out_json} and {out_md} (verdict {verdict})")
        return 0 if verdict in ("PASS", "UNBARED") else 1
    except CompareError as error:
        print(f"compare_native_vs_vllm: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
