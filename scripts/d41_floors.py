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

"""Task-41: compute the D41 floors from a validated D41 capture.

D40 froze the serving-parity metric set and left the numeric thresholds
provisional. D41 freezes them from the measured oracle self-consistency floor:
every bar must sit at least 3x the measured floor away from perfect. This
script is the reproducible half of that rule. It refuses to invent numbers:

  * a same-shape repeat pair must be bit-identical (KL 0, |delta| 0, top-k
    overlap 1.0, rank-1 same) or the whole run is refused;
  * a required distractor class (repeat / idle_distractor / busy_distractor)
    that the capture does not contain is named and the script exits non-zero;
  * a sample below --min-pairs / --min-position-comparisons is refused by
    name, not padded;
  * a single outlier cannot widen a median-based bar: central floors are
    medians over pair comparisons, tail floors are p90 over pairs, and every
    pair beyond --outlier-factor x the central floor is listed separately;
  * top-k set overlap is k = 20 / 100 / 1000. The old top-5 bar is dropped
    because the measured floor is below it (k >= 20 only).

Input is the offline capture schema of docs/archive/build/d41-floor-schema.md
("sparkflash-d41-capture/1"), produced by scripts/d41_recapture.py. Output is
one JSON document ("sparkflash-d41-floors/1"), one markdown table, and
optionally a D41 bar file ("sparkflash-d41-bars/1") that
scripts/compare_native_vs_vllm.py --bars consumes directly. All provenance
hashes (input captures, this script, the prompt suite, the served root) are in
the JSON and the markdown.

Exit codes: 0 floor computed; 2 refused (nothing written); 3 floor computed but
a bar has to be looser than the D40 provisional, so the D41 record must say so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

CAPTURE_SCHEMA = "sparkflash-d41-capture/1"
FLOOR_SCHEMA = "sparkflash-d41-floors/1"
BAR_SCHEMA = "sparkflash-d41-bars/1"

#: k >= 20 by D41. The previous top-5 set bar was below the measured floor and
#: is unmeasurable; it is not emitted at all.
TOP_K = (20, 100, 1000)

CLASSES = ("repeat", "idle_distractor", "busy_distractor")

PROVISIONAL_BARS = {
    "positions_min": 9,
    "top1_agreement_min": 0.95,
    "top20_overlap_mean_min": 0.90,
    "top100_overlap_mean_min": 0.80,
    "top1000_overlap_mean_min": 0.60,
    "max_abs_delta_max": 1.0,
    "mean_abs_delta_max": 0.10,
    "kl_mean_max": 0.02,
    "kl_p95_max": 0.10,
    "js_mean_max": 0.05,
}

DROPPED_BARS = {
    "top5_overlap_mean_min": (
        "dropped by D41: the measured top-5 floor is below the D40 provisional bar, "
        "so run-to-run noise alone fails it; top-k is k >= 20 (top-20/100/1000) only"
    )
}

#: Metrics where a larger number is worse and the bar is a ceiling.
CEILING_BARS = {
    "max_abs_delta_max": ("abs_max", "p90 over pairs"),
    "mean_abs_delta_max": ("abs_mean", "median over pairs"),
    "kl_mean_max": ("kl_mean", "median over pairs"),
    "kl_p95_max": ("kl_p95", "p90 over pairs"),
    "js_mean_max": ("js_mean", "median over pairs"),
}

#: Metrics where a smaller number is worse and the bar is a floor.
FLOOR_BARS = {
    "top1_agreement_min": ("top1_agreement", "aggregate rank-1 stable rate"),
    "top20_overlap_mean_min": ("top20_mean", "median over pairs"),
    "top100_overlap_mean_min": ("top100_mean", "median over pairs"),
    "top1000_overlap_mean_min": ("top1000_mean", "median over pairs"),
}

CENTRAL_PAIR_METRICS = (
    "kl_mean",
    "js_mean",
    "abs_mean",
    "top1_agreement",
) + tuple(f"top{k}_mean" for k in TOP_K)

TAIL_PAIR_METRICS = ("kl_p95", "abs_max", "kl_max", "abs_p95")


class D41Error(RuntimeError):
    """Input or floor computation that cannot be done honestly."""


# ---------------------------------------------------------------------------
# small helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise D41Error(f"{path}: cannot read capture: {error}") from error
    except json.JSONDecodeError as error:
        raise D41Error(f"{path}: not valid JSON: {error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D41Error(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_int(value, where: str, minimum: int | None = None) -> int:
    _require(_is_int(value), f"{where}: expected an integer, got {value!r}")
    if minimum is not None:
        _require(value >= minimum, f"{where}: expected >= {minimum}, got {value}")
    return value


def _require_hex(value, where: str) -> str:
    _require(isinstance(value, str), f"{where}: expected a sha256 string, got {value!r}")
    _require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{where}: not a lowercase sha256: {value!r}",
    )
    return value


def _require_string(value, where: str) -> str:
    _require(isinstance(value, str) and value, f"{where}: expected a non-empty string, got {value!r}")
    return value


def _parse_utc(value, where: str) -> datetime:
    _require_string(value, where)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise D41Error(f"{where}: not an RFC 3339 time: {value!r}") from error
    _require(parsed.tzinfo is not None, f"{where}: time has no timezone: {value!r}")
    return parsed


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise D41Error("percentile of an empty list")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _finite_logprob(value, where: str) -> float:
    _require(_is_number(value), f"{where}: logprob is not a number: {value!r}")
    result = float(value)
    _require(math.isfinite(result), f"{where}: logprob is not finite: {value!r}")
    return result


# ---------------------------------------------------------------------------
# capture validation


def _normalise_metrics(raw, where: str) -> dict:
    _require(isinstance(raw, dict), f"{where}: expected an object")
    running = _require_int(raw.get("num_requests_running"), f"{where}.num_requests_running", 0)
    waiting = _require_int(raw.get("num_requests_waiting"), f"{where}.num_requests_waiting", 0)
    raw_sha = _require_hex(raw.get("raw_sha256"), f"{where}.raw_sha256")
    return {
        "num_requests_running": running,
        "num_requests_waiting": waiting,
        "raw_sha256": raw_sha,
    }


def _normalise_idleness(raw, where: str) -> dict:
    _require(isinstance(raw, dict), f"{where}: idleness is missing or not an object")
    _require(
        raw.get("metrics_source") == "/metrics",
        f"{where}.metrics_source: expected '/metrics', got {raw.get('metrics_source')!r}",
    )
    before = _normalise_metrics(raw.get("metrics_before"), f"{where}.metrics_before")
    after = _normalise_metrics(raw.get("metrics_after"), f"{where}.metrics_after")
    during_raw = raw.get("metrics_during", [])
    _require(isinstance(during_raw, list), f"{where}.metrics_during: expected a list")
    during = [
        _normalise_metrics(item, f"{where}.metrics_during[{index}]")
        for index, item in enumerate(during_raw)
    ]
    observed = raw.get("observed_other_traffic")
    _require(isinstance(observed, bool), f"{where}.observed_other_traffic: expected a bool")
    peers = raw.get("external_peers", [])
    _require(
        isinstance(peers, list) and all(isinstance(peer, str) for peer in peers),
        f"{where}.external_peers: expected a list of strings",
    )
    verdict = raw.get("idle_verdict")
    _require(
        verdict in ("idle", "busy"),
        f"{where}.idle_verdict: {verdict!r} is not idle|busy (an unproven capture cannot be classified)",
    )
    if verdict == "idle":
        _require(not observed, f"{where}: idle_verdict is idle but observed_other_traffic is true")
        _require(not peers, f"{where}: idle_verdict is idle but external_peers is not empty")
        for phase, snapshot in (("metrics_before", before), ("metrics_after", after)):
            _require(
                snapshot["num_requests_running"] == 0 and snapshot["num_requests_waiting"] == 0,
                f"{where}: idle block has a non-zero {phase} baseline "
                f"(running={snapshot['num_requests_running']}, waiting={snapshot['num_requests_waiting']}); "
                "tag it busy instead of mixing classes",
            )
        for index, snapshot in enumerate(during):
            _require(
                snapshot["num_requests_running"] <= 1 and snapshot["num_requests_waiting"] == 0,
                f"{where}.metrics_during[{index}]: co-tenant traffic while the block was tagged idle "
                f"(running={snapshot['num_requests_running']}, waiting={snapshot['num_requests_waiting']}); "
                "tag it busy instead of mixing classes",
            )
    else:
        _require(observed, f"{where}: idle_verdict is busy but observed_other_traffic is not true")
        _require(
            bool(peers) or any(
                snap["num_requests_running"] > 1 or snap["num_requests_waiting"] > 0
                for snap in during
            ),
            f"{where}: idle_verdict is busy but no co-tenancy evidence (external_peers empty and no "
            "metrics_during snapshot with running>1 or waiting>0)",
        )
    return {
        "metrics_source": raw["metrics_source"],
        "metrics_before": before,
        "metrics_after": after,
        "metrics_during": during,
        "observed_other_traffic": observed,
        "external_peers": peers,
        "idle_verdict": verdict,
    }


def _normalise_record(raw, where: str) -> dict:
    _require(isinstance(raw, dict), f"{where}: expected an object")
    record_id = _require_string(raw.get("id"), f"{where}.id")
    prompt_len = _require_int(raw.get("prompt_len"), f"{where}.prompt_len", 2)
    batch_size = _require_int(raw.get("batch_size"), f"{where}.batch_size", 1)
    prompt_sha = _require_hex(raw.get("prompt_sha256"), f"{where}.prompt_sha256")
    request_sha = _require_hex(raw.get("request_sha256"), f"{where}.request_sha256")
    response_sha = _require_hex(raw.get("response_sha256"), f"{where}.response_sha256")
    positions_raw = raw.get("positions")
    _require(
        isinstance(positions_raw, list) and positions_raw,
        f"{where}.positions: expected a non-empty list",
    )
    positions: dict[int, dict] = {}
    for index, item in enumerate(positions_raw):
        item_where = f"{where}.positions[{index}]"
        _require(isinstance(item, dict), f"{item_where}: expected an object")
        position = _require_int(item.get("position"), f"{item_where}.position", 0)
        _require(
            position + 2 <= prompt_len,
            f"{item_where}.position: p+2 > prompt_len ({position} + 2 > {prompt_len})",
        )
        _require(position not in positions, f"{item_where}.position: duplicate position {position}")
        top = item.get("top")
        _require(isinstance(top, list) and top, f"{item_where}.top: expected a non-empty list")
        if "logprob_count" in item:
            count = _require_int(item["logprob_count"], f"{item_where}.logprob_count", 1)
            _require(count == len(top), f"{item_where}.logprob_count: {count} != len(top) {len(top)}")
        logprobs: dict[int, float] = {}
        for entry_index, entry in enumerate(top):
            entry_where = f"{item_where}.top[{entry_index}]"
            _require(
                isinstance(entry, (list, tuple)) and len(entry) == 2,
                f"{entry_where}: expected [token_id, logprob]",
            )
            token = _require_int(entry[0], f"{entry_where}.token", 0)
            _require(token not in logprobs, f"{entry_where}: duplicate token id {token}")
            logprobs[token] = _finite_logprob(entry[1], f"{entry_where}.logprob")
        ranked = sorted(logprobs.items(), key=lambda item: (-item[1], item[0]))
        positions[position] = {"logprobs": logprobs, "ranked": ranked}
    return {
        "id": record_id,
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "prompt_sha256": prompt_sha,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "repeat_of": raw.get("repeat_of"),
        "positions": positions,
    }


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _normalise_block(raw, where: str) -> dict:
    _require(isinstance(raw, dict), f"{where}: expected an object")
    block_id = _require_string(raw.get("block_id"), f"{where}.block_id")
    block_class = raw.get("class")
    _require(block_class in CLASSES, f"{where}.class: {block_class!r} is not one of {CLASSES}")
    started = _parse_utc(raw.get("started_utc"), f"{where}.started_utc")
    finished = _parse_utc(raw.get("finished_utc"), f"{where}.finished_utc")
    _require(finished >= started, f"{where}: finished_utc is before started_utc")
    prompt_len = raw.get("prompt_len")
    if prompt_len is not None:
        _require_int(prompt_len, f"{where}.prompt_len", 1)
    batch_size = _require_int(raw.get("batch_size"), f"{where}.batch_size", 1)
    idleness = _normalise_idleness(raw.get("idleness"), f"{where}.idleness")
    if block_class == "idle_distractor":
        _require(
            idleness["idle_verdict"] == "idle",
            f"{where}: idle_distractor block must be captured idle (idle_verdict is "
            f"{idleness['idle_verdict']!r}); a busy capture belongs to busy_distractor",
        )
    elif block_class == "busy_distractor":
        _require(
            idleness["idle_verdict"] == "busy",
            f"{where}: busy_distractor block must be captured busy (idle_verdict is "
            f"{idleness['idle_verdict']!r}); an idle capture belongs to idle_distractor",
        )

    requests_raw = raw.get("requests")
    _require(
        isinstance(requests_raw, list) and requests_raw,
        f"{where}.requests: expected a non-empty list",
    )
    records: dict[str, dict] = {}
    order: list[str] = []
    for index, item in enumerate(requests_raw):
        record = _normalise_record(item, f"{where}.requests[{index}]")
        _require(
            record["id"] not in records,
            f"{where}.requests[{index}].id: duplicate record id {record['id']!r}",
        )
        records[record["id"]] = record
        order.append(record["id"])

    pairs_raw = raw.get("pairs", [])
    _require(isinstance(pairs_raw, list), f"{where}.pairs: expected a list")
    declared_pairs: list[tuple[str, str]] = []
    seen = set()
    for index, item in enumerate(pairs_raw):
        _require(
            isinstance(item, (list, tuple)) and len(item) == 2,
            f"{where}.pairs[{index}]: expected [record_id, record_id]",
        )
        left = _require_string(item[0], f"{where}.pairs[{index}][0]")
        right = _require_string(item[1], f"{where}.pairs[{index}][1]")
        _require(left in records, f"{where}.pairs[{index}]: unknown record id {left!r}")
        _require(right in records, f"{where}.pairs[{index}]: unknown record id {right!r}")
        _require(left != right, f"{where}.pairs[{index}]: a record cannot be paired with itself")
        key = _pair_key(left, right)
        _require(key not in seen, f"{where}.pairs[{index}]: duplicate pair {key}")
        seen.add(key)
        declared_pairs.append((left, right))

    if block_class == "repeat":
        _require(
            idleness["idle_verdict"] == "idle",
            f"{where}: repeat class must be captured idle (verdict is {idleness['idle_verdict']!r})",
        )
        _require(
            prompt_len is not None,
            f"{where}.prompt_len: a repeat block has one uniform shape and must set prompt_len",
        )
        for record in records.values():
            _require(
                record["prompt_len"] == prompt_len,
                f"{where}: block prompt_len {prompt_len} != record {record['id']!r} prompt_len {record['prompt_len']}",
            )
        derived: list[tuple[str, str]] = []
        for left_id, right_id in zip(order, order[1:]):
            left = records[left_id]
            right = records[right_id]
            if (
                left["request_sha256"] == right["request_sha256"]
                and left["prompt_sha256"] == right["prompt_sha256"]
                and left["prompt_len"] == right["prompt_len"]
                and left["batch_size"] == right["batch_size"]
            ):
                derived.append((left_id, right_id))
        _require(
            derived,
            f"{where}: repeat class has no identical request-body pair "
            "(same request_sha256/prompt_sha256/prompt_len/batch_size)",
        )
        if declared_pairs:
            derived_keys = {_pair_key(a, b) for a, b in derived}
            for pair in declared_pairs:
                _require(
                    _pair_key(*pair) in derived_keys,
                    f"{where}: declared repeat pair {pair} is not an identical request body pair",
                )
        pairs = derived
    else:
        _require(
            declared_pairs,
            f"{where}: {block_class} block must declare its pairs",
        )
        for left_id, right_id in declared_pairs:
            left = records[left_id]
            right = records[right_id]
            _require(
                set(left["positions"]).intersection(right["positions"]),
                f"{where}: pair {left_id} <-> {right_id} shares no position",
            )
            if block_class == "idle_distractor":
                _require(
                    (left["prompt_len"], left["batch_size"]) != (right["prompt_len"], right["batch_size"]),
                    f"{where}: idle_distractor pair {left_id} <-> {right_id} has the same shape "
                    f"({left['prompt_len']}, {left['batch_size']}); same-shape belongs to the repeat class",
                )
        pairs = declared_pairs

    if prompt_len is not None:
        for record in records.values():
            _require(
                record["prompt_len"] == prompt_len,
                f"{where}.prompt_len: {prompt_len} is not the uniform record prompt_len "
                f"(record {record['id']!r} has {record['prompt_len']}); use null for mixed shapes",
            )

    return {
        "block_id": block_id,
        "class": block_class,
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "idleness": idleness,
        "records": records,
        "record_order": order,
        "pairs": pairs,
    }


def validate_capture(doc, path: str) -> dict:
    """Validate one capture against sparkflash-d41-capture/1 and normalise it."""
    where = path
    _require(isinstance(doc, dict), f"{where}: top level is not an object")
    _require(
        doc.get("schema") == CAPTURE_SCHEMA,
        f"{where}: schema is {doc.get('schema')!r}, expected {CAPTURE_SCHEMA!r}",
    )
    capture_id = _require_string(doc.get("capture_id"), f"{where}.capture_id")
    producer = doc.get("producer")
    _require(isinstance(producer, dict), f"{where}.producer: expected an object")
    _require_string(producer.get("script"), f"{where}.producer.script")
    _require_hex(producer.get("script_sha256"), f"{where}.producer.script_sha256")
    started = _parse_utc(doc.get("started_utc"), f"{where}.started_utc")
    finished = _parse_utc(doc.get("finished_utc"), f"{where}.finished_utc")
    _require(finished >= started, f"{where}: finished_utc is before started_utc")
    served = doc.get("served")
    _require(isinstance(served, dict), f"{where}.served: expected an object")
    _require_string(served.get("id"), f"{where}.served.id")
    served_root = _require_string(served.get("root"), f"{where}.served.root")
    expect_root = _require_string(doc.get("expect_root"), f"{where}.expect_root")
    _require(doc.get("root_verified") is True, f"{where}.root_verified: must be true (D38)")
    normalised_root = served_root.rstrip("/")
    _require(
        normalised_root == expect_root
        or normalised_root.endswith("/" + expect_root)
        or normalised_root.endswith(expect_root),
        f"{where}: served.root {served_root!r} does not match expect_root {expect_root!r}",
    )
    recipe = doc.get("recipe")
    _require(isinstance(recipe, dict), f"{where}.recipe: expected an object")
    prompt_logprobs = recipe.get("prompt_logprobs")
    _require(
        _is_int(prompt_logprobs) and prompt_logprobs >= max(TOP_K),
        f"{where}.recipe.prompt_logprobs: must be >= {max(TOP_K)} for top-{max(TOP_K)} overlap, "
        f"got {prompt_logprobs!r}",
    )
    suite = doc.get("suite")
    _require(isinstance(suite, dict), f"{where}.suite: expected an object")
    _require_string(suite.get("path"), f"{where}.suite.path")
    suite_sha = _require_hex(suite.get("sha256"), f"{where}.suite.sha256")
    _require_int(suite.get("prompt_count"), f"{where}.suite.prompt_count", 1)

    blocks_raw = doc.get("blocks")
    _require(isinstance(blocks_raw, list) and blocks_raw, f"{where}.blocks: expected a non-empty list")
    blocks: list[dict] = []
    seen_blocks = set()
    for index, raw in enumerate(blocks_raw):
        block = _normalise_block(raw, f"{where}.blocks[{index}]")
        _require(
            block["block_id"] not in seen_blocks,
            f"{where}.blocks[{index}].block_id: duplicate {block['block_id']!r}",
        )
        seen_blocks.add(block["block_id"])
        blocks.append(block)
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "capture_id": capture_id,
        "served_root": served_root,
        "expect_root": expect_root,
        "suite_sha256": suite_sha,
        "suite_path": suite["path"],
        "producer": producer,
        "started_utc": doc["started_utc"],
        "finished_utc": doc["finished_utc"],
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# comparison metrics


def _log_normalise(logprobs: list[float]) -> list[float]:
    """Return log-probabilities normalised over the compared support.

    Working in log space keeps KL/JS finite when one run's probability for a
    long-tail id underflows to zero; a floor computation must not turn an
    extreme but finite quotient into a crash or an invented number.
    """
    if not logprobs:
        raise D41Error("cannot normalise an empty distribution")
    peak = max(logprobs)
    weights = [math.exp(value - peak) for value in logprobs]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise D41Error("distribution has no usable probability mass")
    log_total = math.log(total) + peak
    return [value - log_total for value in logprobs]


def _kl(logp: list[float], logq: list[float]) -> float:
    total = 0.0
    for lp, lq in zip(logp, logq):
        pi = math.exp(lp)
        if pi > 0.0:
            total += pi * (lp - lq)
    return total


def _js(logp: list[float], logq: list[float]) -> float:
    total = 0.0
    for lp, lq in zip(logp, logq):
        pi = math.exp(lp)
        qi = math.exp(lq)
        mi = (pi + qi) / 2.0
        if pi > 0.0:
            total += 0.5 * pi * math.log(pi / mi)
        if qi > 0.0:
            total += 0.5 * qi * math.log(qi / mi)
    return total


def _topk_overlap(
    ranked_a: list[tuple[int, float]],
    ranked_b: list[tuple[int, float]],
    k: int,
    where: str,
) -> float:
    _require(k >= 20, f"{where}: top-k overlap uses k >= 20 (D41); got k={k}")
    _require(len(ranked_a) >= k, f"{where}: capture has {len(ranked_a)} entries for top-{k}; raise prompt_logprobs")
    _require(len(ranked_b) >= k, f"{where}: capture has {len(ranked_b)} entries for top-{k}; raise prompt_logprobs")
    left = {token for token, _ in ranked_a[:k]}
    right = {token for token, _ in ranked_b[:k]}
    return len(left.intersection(right)) / k


def _position_metrics(position: int, left: dict, right: dict, where: str, flat_prob: float) -> dict:
    left_lp = left["logprobs"]
    right_lp = right["logprobs"]
    shared = sorted(set(left_lp).intersection(right_lp))
    _require(
        len(shared) >= 2,
        f"{where}: position {position} has only {len(shared)} shared token ids; cannot compare",
    )
    logp = _log_normalise([left_lp[token] for token in shared])
    logq = _log_normalise([right_lp[token] for token in shared])
    kl_left_right = _kl(logp, logq)
    kl_right_left = _kl(logq, logp)
    deltas = [abs(left_lp[token] - right_lp[token]) for token in shared]
    ranked_left = left["ranked"]
    ranked_right = right["ranked"]
    top_left = ranked_left[0]
    top_right = ranked_right[0]
    same_top1 = top_left[0] == top_right[0]
    top1_prob_left = math.exp(top_left[1])
    top1_prob_right = math.exp(top_right[1])
    flat = min(top1_prob_left, top1_prob_right) < flat_prob
    return {
        "position": position,
        "shared_ids": len(shared),
        "kl": (kl_left_right + kl_right_left) / 2.0,
        "kl_left_right": kl_left_right,
        "kl_right_left": kl_right_left,
        "js": _js(logp, logq),
        "abs_mean": _mean(deltas),
        "abs_p50": _percentile(deltas, 0.50),
        "abs_p90": _percentile(deltas, 0.90),
        "abs_p95": _percentile(deltas, 0.95),
        "abs_max": max(deltas),
        "rank1_same": same_top1,
        "rank1_flat": (not same_top1) and flat,
        "top1_prob_min": min(top1_prob_left, top1_prob_right),
        **{
            f"top{k}_overlap": _topk_overlap(
                ranked_left, ranked_right, k, f"{where}:position {position}"
            )
            for k in TOP_K
        },
    }


def _pair_metrics(left: dict, right: dict, where: str, flat_prob: float) -> dict:
    shared_positions = sorted(set(left["positions"]).intersection(right["positions"]))
    _require(shared_positions, f"{where}: pair shares no position")
    rows = [
        _position_metrics(position, left["positions"][position], right["positions"][position], where, flat_prob)
        for position in shared_positions
    ]
    kls = [row["kl"] for row in rows]
    stable_positions = sum(1 for row in rows if row["rank1_same"] or row["rank1_flat"])
    flips_above_flat = sum(1 for row in rows if not row["rank1_same"] and not row["rank1_flat"])
    flips_flat = sum(1 for row in rows if row["rank1_flat"])
    metrics = {
        "positions": len(rows),
        "position_first": shared_positions[0],
        "position_last": shared_positions[-1],
        "kl_mean": _mean(kls),
        "kl_p95": _percentile(kls, 0.95),
        "kl_max": max(kls),
        "js_mean": _mean([row["js"] for row in rows]),
        "abs_mean": _mean([row["abs_mean"] for row in rows]),
        "abs_p90": _percentile([row["abs_p90"] for row in rows], 0.90),
        "abs_p95": _percentile([row["abs_p95"] for row in rows], 0.95),
        "abs_max": max(row["abs_max"] for row in rows),
        "top1_agreement": stable_positions / len(rows),
        "rank1_stable": stable_positions,
        "rank1_flips_above_flat": flips_above_flat,
        "rank1_flips_flat": flips_flat,
    }
    for k in TOP_K:
        metrics[f"top{k}_mean"] = _mean([row[f"top{k}_overlap"] for row in rows])
        metrics[f"top{k}_min"] = min(row[f"top{k}_overlap"] for row in rows)
    return metrics


def _assert_repeat_zero(metrics: dict, where: str) -> None:
    problems = []
    if metrics["kl_max"] != 0.0 or metrics["kl_mean"] != 0.0:
        problems.append(f"KL max={metrics['kl_max']!r} mean={metrics['kl_mean']!r} (expected exactly 0)")
    if metrics["abs_max"] != 0.0 or metrics["abs_mean"] != 0.0:
        problems.append(
            f"|delta| max={metrics['abs_max']!r} mean={metrics['abs_mean']!r} (expected exactly 0)"
        )
    if metrics["js_mean"] != 0.0:
        problems.append(f"JS={metrics['js_mean']!r} (expected exactly 0)")
    for k in TOP_K:
        if metrics[f"top{k}_min"] != 1.0 or metrics[f"top{k}_mean"] != 1.0:
            problems.append(
                f"top-{k} overlap min={metrics[f'top{k}_min']!r} mean={metrics[f'top{k}_mean']!r} "
                "(expected exactly 1.0)"
            )
    if metrics["rank1_flips_above_flat"]:
        problems.append(
            f"rank-1 flipped above the flat-position threshold at "
            f"{metrics['rank1_flips_above_flat']} position(s)"
        )
    if problems:
        raise D41Error(
            f"{where}: same-shape repeat is not bit-identical, so the noise floor cannot be zero: "
            + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# floor aggregation and bars


def _aggregate_class(pair_rows: list[dict], outlier_factor: float) -> dict:
    if not pair_rows:
        raise D41Error("cannot aggregate an empty class")
    keys = list(CENTRAL_PAIR_METRICS) + list(TAIL_PAIR_METRICS)
    metrics = {}
    for key in keys:
        values = [float(row[key]) for row in pair_rows]
        metrics[key] = {
            "median": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "max": max(values),
            "mean": _mean(values),
        }
    total_positions = sum(row["positions"] for row in pair_rows)
    stable_positions = sum(row["rank1_stable"] for row in pair_rows)
    flips_above_flat = sum(row["rank1_flips_above_flat"] for row in pair_rows)
    floor = {
        "kl_mean_max": metrics["kl_mean"]["median"],
        "kl_p95_max": metrics["kl_p95"]["p90"],
        "max_abs_delta_max": metrics["abs_max"]["p90"],
        "mean_abs_delta_max": metrics["abs_mean"]["median"],
        "js_mean_max": metrics["js_mean"]["median"],
        "top1_agreement_min": (stable_positions / total_positions) if total_positions else 0.0,
    }
    for k in TOP_K:
        floor[f"top{k}_overlap_mean_min"] = metrics[f"top{k}_mean"]["median"]

    outliers = _find_outliers(pair_rows, outlier_factor)
    return {
        "pairs": len(pair_rows),
        "position_comparisons": total_positions,
        "pair_metrics": pair_rows,
        "metrics": metrics,
        "floor": floor,
        "outliers": outliers,
        "rank1": {
            "stable_positions": stable_positions,
            "position_comparisons": total_positions,
            "flips_above_flat": flips_above_flat,
            "flips_flat": sum(row["rank1_flips_flat"] for row in pair_rows),
        },
    }


def _find_outliers(pair_rows: list[dict], factor: float) -> list[dict]:
    """List pair comparisons that a single-outlier check must show separately.

    The central floors use the median over pairs, not the maximum, so one bad
    pair cannot widen them. This function makes the bad pair visible rather
    than silent; it never changes a floor.
    """
    found: list[dict] = []
    fields = ["kl_mean", "js_mean", "abs_mean"] + [f"top{k}_mean" for k in TOP_K]
    for key in fields:
        values = [float(row[key]) for row in pair_rows]
        median = _percentile(values, 0.50)
        for row in pair_rows:
            value = float(row[key])
            if key.startswith("top") and key.endswith("_mean"):
                threshold = 1.0 - factor * (1.0 - median)
                if value < threshold:
                    found.append(
                        {
                            "pair": [row["left_id"], row["right_id"]],
                            "metric": key,
                            "value": value,
                            "median": median,
                            "threshold": threshold,
                            "kind": "below median floor",
                        }
                    )
            else:
                threshold = median * factor if median > 0.0 else 0.0
                if value > threshold:
                    found.append(
                        {
                            "pair": [row["left_id"], row["right_id"]],
                            "metric": key,
                            "value": value,
                            "median": median,
                            "threshold": threshold,
                            "kind": "above median ceiling",
                        }
                    )
    return found


def _derive_bars(class_result: dict, bar_class: str) -> dict:
    floor = class_result["floor"]
    bars: dict[str, dict] = {}
    review_reasons: list[str] = []

    bars["positions_min"] = {
        "provisional": PROVISIONAL_BARS["positions_min"],
        "floor": None,
        "rule": "provisional (not floor-derived: the floor cannot set the compared-position count)",
        "value": PROVISIONAL_BARS["positions_min"],
        "review": False,
        "review_reason": None,
    }

    for name, (metric, statistic) in CEILING_BARS.items():
        provisional = PROVISIONAL_BARS[name]
        floor_value = float(floor[name])
        value = max(provisional, 3.0 * floor_value)
        review = (3.0 * floor_value) > provisional
        if review:
            review_reasons.append(
                f"{name}: 3x floor ({3.0 * floor_value:.6g}) exceeds the D40 provisional {provisional:.6g}"
            )
        bars[name] = {
            "provisional": provisional,
            "floor": floor_value,
            "rule": f"value = max(provisional, 3 x {statistic}); ceiling",
            "value": value,
            "review": review,
            "review_reason": ("floor requires a looser bar than D40" if review else None),
        }

    for name, (metric, statistic) in FLOOR_BARS.items():
        provisional = PROVISIONAL_BARS[name]
        floor_value = float(floor[name])
        noise_implied = 1.0 - 3.0 * (1.0 - floor_value)
        value = min(provisional, max(0.0, noise_implied))
        review = noise_implied < provisional
        if review:
            review_reasons.append(
                f"{name}: the >=3x floor envelope only allows {noise_implied:.6g}, "
                f"below the D40 provisional {provisional:.6g}"
            )
        bars[name] = {
            "provisional": provisional,
            "floor": floor_value,
            "rule": f"value = min(provisional, 1 - 3 x (1 - {statistic})); floor",
            "value": value,
            "review": review,
            "review_reason": (
                "provisional is stricter than the measured floor envelope" if review else None
            ),
        }

    return {
        "bar_class": bar_class,
        "bars": bars,
        "gate_bars": {name: round(bars[name]["value"], 6) for name in bars},
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "dropped_bars": DROPPED_BARS,
        "busy_envelope_note": (
            "busy_distractor is the load-state envelope recorded alongside the floor; it is not the "
            "D41 gate class (task32-root-cause.md section 4.3)"
        ),
    }


# ---------------------------------------------------------------------------
# top-level computation


def compute(
    captures: list[dict],
    *,
    min_pairs: int = 3,
    min_position_comparisons: int = 12,
    required_classes: tuple[str, ...] = CLASSES,
    k_values: tuple[int, ...] = TOP_K,
    flat_prob: float = 0.05,
    outlier_factor: float = 3.0,
    bar_class: str = "idle_distractor",
    source_revision: str | None = None,
) -> dict:
    _require(captures, "no captures")
    _require(
        tuple(k_values) == tuple(TOP_K),
        f"k set must be {TOP_K} (top-5 dropped; k >= 20 only)",
    )
    _require(all(k >= 20 for k in k_values), "top-k overlap uses k >= 20")
    _require(len(set(required_classes)) == len(required_classes), "duplicate required class")
    for cls in required_classes:
        _require(cls in CLASSES, f"unknown required class {cls!r}; known: {CLASSES}")
    _require(bar_class in ("idle_distractor", "busy_distractor"), f"unknown bar class {bar_class!r}")
    _require(min_pairs >= 1, "min-pairs must be >= 1")
    _require(min_position_comparisons >= 1, "min-position-comparisons must be >= 1")

    roots = {capture["served_root"] for capture in captures}
    suites = {capture["suite_sha256"] for capture in captures}
    expect_roots = {capture["expect_root"] for capture in captures}
    _require(len(roots) == 1, f"captures disagree on served root: {sorted(roots)}")
    _require(len(suites) == 1, f"captures disagree on prompt-suite sha256: {sorted(suites)}")
    _require(len(expect_roots) == 1, f"captures disagree on expect_root: {sorted(expect_roots)}")

    found: dict[str, list[tuple[dict, dict]]] = {cls: [] for cls in CLASSES}
    for capture in captures:
        for block in capture["blocks"]:
            found[block["class"]].append((capture, block))

    missing = [cls for cls in required_classes if not found[cls]]
    _require(
        not missing,
        "missing required class(es): "
        + ", ".join(missing)
        + " (captured classes: "
        + (", ".join(cls for cls in CLASSES if found[cls]) or "none")
        + ")",
    )

    class_results: dict[str, dict] = {}
    for cls in CLASSES:
        if not found[cls]:
            class_results[cls] = {"present": False}
            continue
        pair_rows: list[dict] = []
        for capture, block in found[cls]:
            for left_id, right_id in block["pairs"]:
                left = block["records"][left_id]
                right = block["records"][right_id]
                where = f"{capture['path']}:{block['block_id']}:{left_id}<->{right_id}"
                metrics = _pair_metrics(left, right, where, flat_prob)
                if cls == "repeat":
                    _assert_repeat_zero(metrics, where)
                metrics["left_id"] = left_id
                metrics["right_id"] = right_id
                metrics["block_id"] = block["block_id"]
                metrics["capture_sha256"] = capture["sha256"]
                pair_rows.append(metrics)
        if cls in required_classes:
            _require(
                len(pair_rows) >= min_pairs,
                f"class {cls}: only {len(pair_rows)} pair comparisons, below min-pairs {min_pairs}; "
                "refusing to compute a floor from too few samples",
            )
            if cls != "repeat":
                total_positions = sum(row["positions"] for row in pair_rows)
                _require(
                    total_positions >= min_position_comparisons,
                    f"class {cls}: only {total_positions} position comparisons, below "
                    f"min-position-comparisons {min_position_comparisons}; refusing to compute a floor",
                )
        class_results[cls] = {
            "present": True,
            "blocks": len(found[cls]),
            **_aggregate_class(pair_rows, outlier_factor),
        }

    bar_result = class_results.get(bar_class)
    _require(
        bar_result is not None and bar_result.get("present"),
        f"bar class {bar_class!r} is not present in the capture",
    )
    if bar_result["rank1"]["flips_above_flat"]:
        raise D41Error(
            f"bar class {bar_class}: {bar_result['rank1']['flips_above_flat']} rank-1 flip(s) above the "
            f"flat-position threshold (top-1 probability >= {flat_prob}); that is not a noise floor"
        )
    bars = _derive_bars(bar_result, bar_class)

    report = {
        "schema": FLOOR_SCHEMA,
        "generated_utc": utc_now(),
        "status": "requires_review" if bars["review_required"] else "ok",
        "provenance": {
            "captures": [
                {
                    "path": capture["path"],
                    "sha256": capture["sha256"],
                    "capture_id": capture["capture_id"],
                    "blocks": len(capture["blocks"]),
                    "started_utc": capture["started_utc"],
                    "finished_utc": capture["finished_utc"],
                    "producer": capture["producer"],
                }
                for capture in captures
            ],
            "floor_script": {
                "path": os.path.abspath(__file__),
                "sha256": sha256_file(__file__),
            },
            "prompt_suite": {
                "sha256": next(iter(suites)),
                "path": captures[0]["suite_path"],
            },
            "served_root": next(iter(roots)),
            "expect_root": next(iter(expect_roots)),
            "source_revision": source_revision,
            "tool": {
                "argv": list(sys.argv),
                "python": platform.python_version(),
            },
        },
        "parameters": {
            "min_pairs": min_pairs,
            "min_position_comparisons": min_position_comparisons,
            "required_classes": list(required_classes),
            "k_values": list(k_values),
            "flat_prob": flat_prob,
            "outlier_factor": outlier_factor,
            "bar_class": bar_class,
        },
        "capture_validation": {
            "captures": [
                {
                    "path": capture["path"],
                    "sha256": capture["sha256"],
                    "capture_id": capture["capture_id"],
                    "blocks": [
                        {
                            "block_id": block["block_id"],
                            "class": block["class"],
                            "prompt_len": block["prompt_len"],
                            "batch_size": block["batch_size"],
                            "idle_verdict": block["idleness"]["idle_verdict"],
                            "observed_other_traffic": block["idleness"]["observed_other_traffic"],
                            "metrics_before": block["idleness"]["metrics_before"],
                            "metrics_after": block["idleness"]["metrics_after"],
                            "metrics_during": block["idleness"]["metrics_during"],
                            "records": len(block["records"]),
                            "pairs": len(block["pairs"]),
                        }
                        for block in capture["blocks"]
                    ],
                }
                for capture in captures
            ],
        },
        "classes": class_results,
        "bars": bars,
    }
    return report


# ---------------------------------------------------------------------------
# rendering and CLI


def render_markdown(report: dict) -> str:
    provenance = report["provenance"]
    bars = report["bars"]
    lines = [
        "# D41 floor computation",
        "",
        f"- status: **{report['status']}**",
        f"- bar class: {bars['bar_class']} (the D41 gate floor; busy is only the recorded envelope)",
        f"- generated: {report['generated_utc']}",
        "- k values: " + ", ".join(str(k) for k in report["parameters"]["k_values"]) + " (top-5 dropped: k >= 20 only)",
        "",
        "## Provenance",
        "",
        f"- served root: {provenance['served_root']}",
        f"- prompt suite sha256: {provenance['prompt_suite']['sha256']} ({provenance['prompt_suite']['path']})",
        f"- floor script sha256: {provenance['floor_script']['sha256']}",
        f"- source revision: {provenance.get('source_revision')}",
        "",
        "| capture | sha256 | id | blocks |",
        "|:--|:--|:--|:--|",
    ]
    for capture in provenance["captures"]:
        lines.append(
            f"| {capture['path']} | {capture['sha256']} | {capture['capture_id']} | {capture['blocks']} |"
        )
    lines += ["", "## Validation", "", "| class | blocks | pairs | position comparisons |", "|:--|--:|--:|--:|"]
    for cls in CLASSES:
        result = report["classes"][cls]
        if not result.get("present"):
            lines.append(f"| {cls} | 0 | 0 | 0 |")
        else:
            lines.append(
                f"| {cls} | {result['blocks']} | {result['pairs']} | {result['position_comparisons']} |"
            )
    lines += [
        "",
        "Idle blocks have metrics_before/metrics_after at 0 running and 0 waiting; a non-zero baseline "
        "is refused, not mixed into an idle class. Busy blocks carry co-tenancy evidence in metrics_during.",
        "",
        "## Floors",
        "",
        "| metric | floor statistic | repeat | idle_distractor | busy_distractor |",
        "|:--|:--|--:|--:|--:|",
    ]
    floor_rows = [
        ("kl_mean_max", "median pair KL"),
        ("kl_p95_max", "p90 pair KL p95"),
        ("max_abs_delta_max", "p90 pair max |delta|"),
        ("mean_abs_delta_max", "median pair mean |delta|"),
        ("js_mean_max", "median pair JS"),
        ("top1_agreement_min", "aggregate rank-1 stable rate"),
    ] + [(f"top{k}_overlap_mean_min", f"median pair top-{k} overlap mean") for k in TOP_K]
    for key, statistic in floor_rows:
        cells = []
        for cls in CLASSES:
            result = report["classes"][cls]
            cells.append(f"{result['floor'][key]:.6g}" if result.get("present") else "-")
        lines.append(f"| {key} | {statistic} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Central floors are medians over pair comparisons (one outlier cannot widen them); tail floors are "
        "p90 over pairs. busy_distractor is the load-state envelope, not the D41 gate floor.",
        "",
        "## Bars",
        "",
        "| bar | provisional | measured floor | rule | D41 value | review |",
        "|:--|--:|--:|:--|--:|:--|",
    ]
    for name, bar in bars["bars"].items():
        floor_text = "-" if bar["floor"] is None else f"{bar['floor']:.6g}"
        review_text = "REVIEW: " + bar["review_reason"] if bar["review"] else "ok"
        lines.append(
            f"| {name} | {bar['provisional']:.6g} | {floor_text} | {bar['rule']} | "
            f"{bar['value']:.6g} | {review_text} |"
        )
    if bars["review_required"]:
        lines += ["", "Review reasons:", ""] + [f"- {reason}" for reason in bars["review_reasons"]]
    lines += ["", "## Dropped bars", ""] + [
        f"- {name}: {reason}" for name, reason in bars["dropped_bars"].items()
    ]
    lines += ["", "## Outliers (reported, never used to widen a median floor)", ""]
    any_outliers = False
    for cls in CLASSES:
        result = report["classes"][cls]
        if not result.get("present") or not result["outliers"]:
            continue
        any_outliers = True
        for outlier in result["outliers"]:
            lines.append(
                f"- {cls}: pair {outlier['pair'][0]} <-> {outlier['pair'][1]}, "
                f"{outlier['metric']}={outlier['value']:.6g} "
                f"(median {outlier['median']:.6g}, threshold {outlier['threshold']:.6g})"
            )
    if not any_outliers:
        lines.append("- none")
    lines += [
        "",
        "## Refusals encoded in the computation",
        "",
        "- repeat class must be bit-identical (KL 0, |delta| 0, top-k 1.0, rank-1 same) or the run is refused;",
        "- a required class that is absent is named and the run exits non-zero;",
        "- pair and position minima are enforced by recount, never padded;",
        "- unknown/extra schema fields are ignored, but every consumed field is validated above.",
        "",
    ]
    return "\n".join(lines)


def _write_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(int(part) for part in raw.split(",") if part.strip()))
    except ValueError as error:
        raise D41Error(f"--k: not a comma-separated integer list: {raw!r}") from error
    _require(values, "--k: empty list")
    _require(all(value >= 20 for value in values), f"--k: top-k overlap uses k >= 20 (D41); got {values}")
    return values


def _parse_required(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    _require(values, "--require-classes: empty list")
    for value in values:
        _require(value in CLASSES, f"--require-classes: unknown class {value!r}; known: {CLASSES}")
    _require(len(set(values)) == len(values), "--require-classes: duplicate class")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the D41 floors from validated sparkflash-d41-capture/1 files (offline).",
    )
    parser.add_argument(
        "--capture",
        action="append",
        required=True,
        help="capture JSON (repeat for several; classes may be split across files)",
    )
    parser.add_argument("--out-json", required=True, help="floor JSON output")
    parser.add_argument("--out-md", required=True, help="floor markdown output")
    parser.add_argument(
        "--out-bars",
        help="optional D41 bar file (sparkflash-d41-bars/1) for compare_native_vs_vllm.py --bars",
    )
    parser.add_argument("--min-pairs", type=int, default=3, help="minimum pair comparisons per required class (default 3)")
    parser.add_argument(
        "--min-position-comparisons",
        type=int,
        default=12,
        help="minimum position comparisons per required distractor class (default 12)",
    )
    parser.add_argument(
        "--require-classes",
        default=",".join(CLASSES),
        help="comma-separated classes that must be present (default all three)",
    )
    parser.add_argument(
        "--k",
        default=",".join(str(k) for k in TOP_K),
        help="top-k set (D41: exactly 20,100,1000; k >= 20)",
    )
    parser.add_argument(
        "--flat-prob",
        type=float,
        default=0.05,
        help="rank-1 flips below this top-1 probability are flat, not failures",
    )
    parser.add_argument(
        "--outlier-factor",
        type=float,
        default=3.0,
        help="report pairs beyond this factor from the median",
    )
    parser.add_argument(
        "--bar-class",
        default="idle_distractor",
        choices=("idle_distractor", "busy_distractor"),
        help="class the bars are derived from (default idle_distractor, the D41 gate class)",
    )
    parser.add_argument(
        "--source-revision",
        help="optional source revision recorded in provenance (e.g. git sha)",
    )
    args = parser.parse_args(argv)

    try:
        k_values = _parse_k_values(args.k)
        required = _parse_required(args.require_classes)
        captures = [validate_capture(load_json(path), path) for path in args.capture]
        report = compute(
            captures,
            min_pairs=args.min_pairs,
            min_position_comparisons=args.min_position_comparisons,
            required_classes=required,
            k_values=k_values,
            flat_prob=args.flat_prob,
            outlier_factor=args.outlier_factor,
            bar_class=args.bar_class,
            source_revision=args.source_revision,
        )
    except D41Error as error:
        print(f"d41_floors: REFUSED: {error}", file=sys.stderr)
        return 2

    report_json = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_atomic(args.out_json, report_json)
    _write_atomic(args.out_md, render_markdown(report) + "\n")
    if args.out_bars:
        bars = {
            "schema": BAR_SCHEMA,
            "generated_utc": report["generated_utc"],
            "status": report["status"],
            "bar_class": report["bars"]["bar_class"],
            "provenance": {
                "floor_json": os.path.abspath(args.out_json),
                "floor_json_sha256": sha256_bytes(report_json.encode("utf-8")),
                "floor_script_sha256": report["provenance"]["floor_script"]["sha256"],
                "captures": report["provenance"]["captures"],
            },
            "bars": report["bars"]["gate_bars"],
        }
        _write_atomic(args.out_bars, json.dumps(bars, indent=2, sort_keys=True) + "\n")

    print(
        f"d41_floors: {report['status']} (bar class {report['bars']['bar_class']}); "
        f"wrote {args.out_json} and {args.out_md}"
        + (f" and {args.out_bars}" if args.out_bars else ""),
        file=sys.stderr,
    )
    return 3 if report["status"] == "requires_review" else 0


if __name__ == "__main__":
    sys.exit(main())
