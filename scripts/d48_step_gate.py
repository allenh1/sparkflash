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

"""D48 step-time collector for four-rank device evidence (window protocol, task-4).

Reads the per-rank evidence a four-rank run actually wrote and applies the D48 budget arithmetic
with every number attributed to its file. It writes exactly one *kind* per document (device or
host-twin) - never a summed row - and it never fills a missing measurement with a placeholder.

Verified window interface (2026-10-20, Lead correction against tp4_chain.rs / device_chain.rs):

  * the four-rank arm stage is --run-stage tp4-chain; tonight's runnable mode is the LOOP path
    (bind coverage + 80-call collective walk, token NOT emitted). There is no tp4-device stage.
  * the Lead's wrapper is /home/user/ev-session-20261020/run-fourrank.sh <stage-tag> <run-name>
    loop|raw, which preflights binary/cubin sha + MemAvailable + ports and collects
    4 x tp4-loop-rR.json + logs into /home/user/ev-session-20261020/window/<run-name>/rank{0..3}/.
  * the host twin is --run-stage tp4-host (only if the window has >= 6 min left); its numbers are
    written by a separate run of this collector with --kind host-twin.

Usage (device row):

    python3 scripts/d48_step_gate.py \
      --device-dir /home/user/ev-session-20261020/window/<run-name> \
      --out-json /home/user/ev-session-20261020/window/<run-name>/d48-step-gate.json \
      --out-md   /home/user/ev-session-20261020/window/<run-name>/d48-step-gate.md

The collector writes schema "sparkflash-d48-step-gate/1" and prints the row. Exit codes: 0
STEP-TIME-PASS, 1 STEP-TIME-FAIL / STEP-TIME-REFUSED, 3 NOT-ATTEMPTED (no step samples: the
loop-only arm), 2 bad input (nothing written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compare_native_vs_vllm as compare

OUTPUT_SCHEMA = "sparkflash-d48-step-gate/1"
# device-lead, 2026-10-20: these three multi-modal special-token embeddings are expected to be
# unclassified in the plan census; any other unclassified name is a defect.
EXPECTED_CENSUS_UNCLASSIFIED = {"image_end", "image_newline", "image_start"}
D48_STEP_BUDGET_MS = 42.0
D48_ACCEPTED_TOKENS_PER_STEP = 3.8
D48_TOK_S_TARGET = 90.0
D48_PARITY_BAND_TOK_S = (50.0, 60.0)


class GateError(RuntimeError):
    """Input that cannot be turned into a step row honestly."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as error:
        raise GateError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise GateError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(doc, dict):
        raise GateError(f"{path}: evidence must be a JSON object, got {type(doc).__name__}")
    return doc


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_float_list(value):
    if isinstance(value, list) and value and all(is_number(item) for item in value):
        return [float(item) for item in value]
    return None


def int_list(value):
    if isinstance(value, list) and value and all(is_int(item) and item >= 0 for item in value):
        return list(value)
    return None


def first_number(doc, keys):
    for key in keys:
        value = doc.get(key)
        if is_number(value):
            return float(value)
    return None


def collect_files(dirs, ignored: list) -> list:
    files: list = []
    for path in dirs:
        if os.path.isdir(path):
            for directory, _, names in os.walk(path):
                for name in sorted(names):
                    full = os.path.join(directory, name)
                    if name.endswith(".json"):
                        files.append(full)
                    elif name.endswith((".jsonl", ".bin")):
                        ignored.append(
                            {
                                "path": os.path.abspath(full),
                                "reason": "not a step-timing document: dumps are consumed by accept_device_token.py, never read here",
                            }
                        )
        else:
            files.append(path)
    unique: list = []
    seen: set = set()
    for path in files:
        absolute = os.path.abspath(path)
        if absolute in seen:
            continue
        seen.add(absolute)
        unique.append(absolute)
    return unique


def rank_of(path: str, doc: dict):
    for key in ("rank", "tp_rank"):
        value = doc.get(key)
        if is_int(value):
            return int(value)
    base = os.path.basename(path)
    match = None
    import re

    match = re.search(r"-r(\d+)\.json$", base)
    return int(match.group(1)) if match else None


def gate_bind_summary(doc: dict) -> dict:
    """The small projection of sparkflash-tp4-bind/1 the step collector reports and checks."""
    attention = doc.get("attention") if isinstance(doc.get("attention"), dict) else {}
    moe = doc.get("moe") if isinstance(doc.get("moe"), dict) else {}
    tail = doc.get("tail") if isinstance(doc.get("tail"), dict) else {}
    census = doc.get("census") if isinstance(doc.get("census"), dict) else {}
    summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
    attention_coverage = attention.get("coverage") if isinstance(attention.get("coverage"), dict) else {}
    classes = moe.get("classes") if isinstance(moe.get("classes"), list) else []
    class_rows = [
        {
            "class": row.get("class") if isinstance(row.get("class"), str) else None,
            "expected": row.get("expected") if is_int(row.get("expected")) else None,
            "bound": row.get("bound") if is_int(row.get("bound")) else None,
            "complete": row.get("complete") if isinstance(row.get("complete"), bool) else None,
        }
        for row in classes
        if isinstance(row, dict)
    ]
    attention_expected = attention.get("tensors_expected") if is_int(attention.get("tensors_expected")) else None
    attention_bound = attention.get("tensors_bound") if is_int(attention.get("tensors_bound")) else None
    attention_complete = attention_coverage.get("complete") if isinstance(attention_coverage.get("complete"), bool) else None
    if attention_complete is None and attention_expected is not None and attention_bound is not None and attention_expected > 0:
        attention_complete = attention_expected == attention_bound
    tail_coverage = tail.get("coverage") if isinstance(tail.get("coverage"), dict) else {}
    tail_complete = tail_coverage.get("complete") if isinstance(tail_coverage.get("complete"), bool) else None
    if tail_complete is None and is_int(tail_coverage.get("expected")) and is_int(tail_coverage.get("bound")):
        tail_complete = tail_coverage["expected"] == tail_coverage["bound"]
    unclassified = census.get("unclassified") if isinstance(census.get("unclassified"), list) else None
    return {
        "performed": doc.get("performed") is not False,
        "error": doc.get("error") if isinstance(doc.get("error"), str) else None,
        "attention": {
            "expected": attention_expected,
            "bound": attention_bound,
            "complete": attention_complete,
        },
        "moe": {
            "performed": bool(class_rows) and moe.get("performed") is not False,
            "classes": class_rows,
            "complete": (all(row["complete"] is True for row in class_rows) if class_rows else None),
            "weight_h2d_bytes_per_token": moe.get("weight_h2d_bytes_per_token") if is_int(moe.get("weight_h2d_bytes_per_token")) else None,
            "launch_incompatible": [
                row.get("class")
                for row in (moe.get("launch_compatibility") if isinstance(moe.get("launch_compatibility"), list) else [])
                if isinstance(row, dict) and row.get("launch_compatible") is False
            ],
        },
        "tail": {
            "performed": tail.get("performed") is not False,
            "complete": tail_complete,
            "pointers_nonzero": tail.get("pointers_nonzero") if isinstance(tail.get("pointers_nonzero"), bool) else None,
        },
        "census": {
            "performed": is_int(census.get("resident_tensors")),
            "resident_tensors": census.get("resident_tensors") if is_int(census.get("resident_tensors")) else None,
            "unclassified": unclassified,
        },
        "summary": {
            "device_bound_names": summary.get("device_bound_names") if is_int(summary.get("device_bound_names")) else None,
            "per_token_weight_h2d_bytes": summary.get("per_token_weight_h2d_bytes") if is_int(summary.get("per_token_weight_h2d_bytes")) else None,
            "complete": summary.get("complete") if isinstance(summary.get("complete"), bool) else None,
        },
    }


def extract_layer_rows(doc: dict) -> list:
    raw = doc.get("per_layer")
    if not isinstance(raw, list):
        return []
    rows: list = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        layer = item.get("layer")
        attention = item.get("attention") if isinstance(item.get("attention"), dict) else {}
        moe = item.get("moe") if isinstance(item.get("moe"), dict) else {}
        row = {
            "layer": layer if is_int(layer) else None,
            "attention": {key: value for key, value in attention.items() if is_number(value)},
            "moe": {key: value for key, value in moe.items() if is_number(value)},
            "total_us": float(item["total_us"]) if is_number(item.get("total_us")) else None,
        }
        rows.append(row)
    return rows


AUX_NAME = re.compile(r"^(timeline|preflight|wrapper|chain-timeline|window)[-_.].*\.json$", re.IGNORECASE)


def aux_by_name(path: str) -> bool:
    """Wrapper/timeline metadata JSON that may share a rank directory is not acceptance evidence."""
    return bool(AUX_NAME.match(os.path.basename(path)))


def refusal_evidence(directory: str, rank):
    """The labelled negative: tp4-token-rR.json says forward=refused/positions_run=0 with a named error.

    rc==2 is then the expected refusal exit code, not a bind/step failure; only that exact
    envelope shape is accepted. Any other non-zero rc stays a failure.
    """
    if rank is None:
        return None
    path = os.path.join(directory, "tp4-token-r%d.json" % rank)
    if not os.path.exists(path):
        return None
    try:
        doc = load_json(path)
    except Exception:
        return None
    if doc.get("schema") != "sparkflash-tp4-token/1":
        return None
    if doc.get("forward") != "refused":
        return None
    if doc.get("positions_run") != 0:
        return None
    error = doc.get("error")
    if not isinstance(error, str) or not error.strip():
        return None
    return {"error": error.strip(), "positions_run": 0, "path": path}


def extract_rank(path: str) -> dict:
    doc = load_json(path)
    kind = doc.get("schema") or doc.get("stage")
    entry = {
        "path": path,
        "sha256": sha256_file(path),
        "schema": doc.get("schema") if isinstance(doc.get("schema"), str) else None,
        "stage": doc.get("stage") if isinstance(doc.get("stage"), str) else None,
        "rank": rank_of(path, doc),
        "rc": int(doc["rc"]) if is_int(doc.get("rc")) else (int(doc["exit_code"]) if is_int(doc.get("exit_code")) else None),
        "rc_source": None,
        "tokens": [],
        "steps_run": None,
        "step_ms_samples": [],
        "first_token_ms": None,
        "boot_s": None,
        "layers_run": None,
        "reduces_posted": None,
        "reduces_expected": None,
        "weight_bytes_per_token": None,
        "attention_staged_bytes": None,
        "moe_staged_bytes": None,
        "accepted_tokens_per_step": None,
        "per_layer": extract_layer_rows(doc),
        "notes": [],
        "role": "step",
        "aux_reason": None,
        "refusal": None,
        "bind": None,
        "plan": doc.get("plan") if isinstance(doc.get("plan"), dict) else None,
    }

    schema_str = doc.get("schema") if isinstance(doc.get("schema"), str) else None
    if aux_by_name(path):
        entry["role"] = "aux"
        entry["aux_reason"] = "wrapper/timeline metadata filename, not an acceptance input"
        return entry
    if doc.get("stage") == "tp4-host":
        entry["role"] = "step"
    elif doc.get("stage") == "tp4-resident":
        entry["role"] = "aux"
        entry["aux_reason"] = "resident-stage evidence; it is not a bind/loop/step document"
        return entry
    elif schema_str is None:
        if isinstance(doc.get("schema"), int) or doc.get("stage") == "tp4-resident":
            entry["role"] = "aux"
            entry["aux_reason"] = "resident-stage evidence uses an integer schema version; it is not a bind/loop/step document"
            return entry
        if not (doc.get("tokens") or doc.get("generated_tokens")):
            entry["role"] = "aux"
            entry["aux_reason"] = "no string schema and no token block; not a bind/loop/step document"
            return entry
    elif schema_str == "sparkflash-tp4-bind/1":
        entry["role"] = "bind"
    elif schema_str not in ("sparkflash-tp4-loop/1", "sparkflash-tp4-chain/2", "sparkflash-tp4-host/1", "sparkflash-tp4-host/2"):
        entry["role"] = "aux"
        entry["aux_reason"] = f"schema {schema_str!r} is not a bind/loop/step document"
        return entry

    if doc.get("schema") == "sparkflash-tp4-bind/1":
        entry["role"] = "bind"
        entry["bind"] = gate_bind_summary(doc)
        attention = doc.get("attention") if isinstance(doc.get("attention"), dict) else {}
        entry["attention_bound"] = attention.get("tensors_bound") if is_int(attention.get("tensors_bound")) else None
        entry["attention_expected"] = attention.get("tensors_expected") if is_int(attention.get("tensors_expected")) else None
        entry["plan_only"] = doc.get("plan_only") if isinstance(doc.get("plan_only"), bool) else None
        return entry

    # tokens
    tokens_block = doc.get("tokens")
    if isinstance(tokens_block, dict):
        generated = int_list(tokens_block.get("generated"))
        if generated is None:
            generated = int_list(tokens_block.get("token_ids"))
        entry["tokens"] = generated or []
        if is_int(tokens_block.get("prompt_tokens")):
            entry["notes"].append(f"prompt length {tokens_block['prompt_tokens']} tokens")
    else:
        for key in ("generated_tokens", "token_ids"):
            found = int_list(doc.get(key))
            if found:
                entry["tokens"] = found
                break

    step = doc.get("step") if isinstance(doc.get("step"), dict) else {}
    samples = as_float_list(step.get("step_ms_samples"))
    if samples is None:
        samples = as_float_list(doc.get("step_ms"))
    if samples is None:
        timing = doc.get("timing") if isinstance(doc.get("timing"), dict) else {}
        samples = as_float_list(timing.get("step_ms")) or as_float_list(timing.get("step_ms_samples"))
    entry["step_ms_samples"] = samples or []
    entry["steps_run"] = None
    counted = step.get("counted_steps")
    if is_int(counted):
        entry["steps_run"] = int(counted)
    elif is_int(step.get("steps_run")):
        entry["steps_run"] = int(step["steps_run"])
    elif samples:
        entry["steps_run"] = len(samples)

    entry["first_token_ms"] = first_number(step, ("first_token_ms", "ttft_ms")) or first_number(
        doc, ("first_token_ms", "ttft_ms")
    )
    boot = doc.get("boot") if isinstance(doc.get("boot"), dict) else {}
    entry["boot_s"] = first_number(step, ("boot_s", "boot_seconds")) or first_number(doc, ("boot_s", "boot_seconds")) or first_number(boot, ("seconds", "boot_s"))
    if entry["boot_s"] is None:
        timing = doc.get("timing") if isinstance(doc.get("timing"), dict) else {}
        entry["boot_s"] = first_number(timing, ("boot_s", "boot_seconds"))

    collectives = doc.get("collectives") if isinstance(doc.get("collectives"), dict) else {}
    if is_int(step.get("layers_run")):
        entry["layers_run"] = int(step["layers_run"])
    elif is_int(collectives.get("layers")):
        entry["layers_run"] = int(collectives["layers"])
    elif entry["per_layer"]:
        entry["layers_run"] = len(entry["per_layer"])

    if is_int(step.get("reduces_posted")):
        entry["reduces_posted"] = int(step["reduces_posted"])
    if is_int(step.get("reduces_expected")):
        entry["reduces_expected"] = int(step["reduces_expected"])
    walk_steps = collectives.get("steps")
    if entry["reduces_posted"] is None and isinstance(walk_steps, list) and walk_steps:
        counts = [int(row["calls"]) for row in walk_steps if is_int(row.get("calls"))]
        if counts:
            entry["reduces_posted"] = sum(counts)
            expected = 2 * entry["layers_run"] * len(counts) if entry["layers_run"] else None
            entry["reduces_expected"] = expected
    if entry["weight_bytes_per_token"] is None and is_number(step.get("weight_bytes_per_token")):
        entry["weight_bytes_per_token"] = int(step["weight_bytes_per_token"])
    if entry["weight_bytes_per_token"] is None and is_number(doc.get("weight_bytes_per_token")):
        entry["weight_bytes_per_token"] = int(doc["weight_bytes_per_token"])
    for target, keys in (
        ("attention_staged_bytes", ("attention_staged_bytes", "attention_staging_bytes")),
        ("moe_staged_bytes", ("moe_staged_bytes", "moe_staging_bytes")),
    ):
        if entry[target] is None:
            for key in keys:
                if is_number(step.get(key)):
                    entry[target] = int(step[key])
                    break
                if is_number(doc.get(key)):
                    entry[target] = int(doc[key])
                    break
    if is_number(step.get("accepted_tokens_per_step")):
        entry["accepted_tokens_per_step"] = float(step["accepted_tokens_per_step"])

    attention = doc.get("attention") if isinstance(doc.get("attention"), dict) else {}
    entry["attention_bound"] = attention.get("tensors_bound") if is_int(attention.get("tensors_bound")) else None
    entry["attention_expected"] = attention.get("tensors_expected") if is_int(attention.get("tensors_expected")) else None
    entry["walk_performed"] = bool(collectives.get("performed")) if "performed" in collectives else None
    entry["plan_only"] = doc.get("plan_only") if isinstance(doc.get("plan_only"), bool) else None
    hybrid = doc.get("hybrid_sequence") if isinstance(doc.get("hybrid_sequence"), dict) else None
    if hybrid is not None:
        entry["hybrid_sequence"] = {
            "executed": bool(hybrid.get("executed")),
            "error": hybrid.get("error") if isinstance(hybrid.get("error"), str) else None,
            "note": hybrid.get("note") if isinstance(hybrid.get("note"), str) else None,
        }
    if isinstance(doc.get("bind"), dict):
        entry["bind"] = gate_bind_summary(doc["bind"])
    if entry["walk_performed"]:
        walk_ok = collectives.get("calls_per_step_expected")
        if is_int(walk_ok) and entry["reduces_posted"] is not None and entry["reduces_expected"] is not None:
            entry["notes"].append(
                f"collective walk: {entry['reduces_posted']} calls, 2 per layer per step expected {entry['reduces_expected']}"
            )

    if not entry["step_ms_samples"]:
        entry["notes"].append("no step_ms samples: this rank did not run/record a timed step")
    if entry["tokens"] and entry["steps_run"] is not None and len(entry["tokens"]) < entry["steps_run"]:
        entry["notes"].append(
            f"generated tokens ({len(entry['tokens'])}) < counted steps ({entry['steps_run']}): the token row is partial"
        )
    return entry


def summarise_samples(samples: list) -> dict:
    ordered = sorted(samples)
    if not ordered:
        return {"samples": [], "mean": None, "median": None, "min": None, "max": None, "p95": None}
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "samples": [round(value, 6) for value in samples],
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "p95": round(ordered[p95_index], 6),
    }


def layer_components_us(row: dict) -> float | None:
    total = 0.0
    seen = False
    for half in ("attention", "moe"):
        for key, value in row[half].items():
            if key.endswith("_us") and "staging" not in key:
                total += value
                seen = True
    return total if seen else None


def build_evidence(entries: list, args, bind_by_rank=None) -> dict:
    bind_by_rank = bind_by_rank or {}
    per_rank: list = []
    checks: list = []
    failures: list = []
    for entry in entries:
        samples = summarise_samples(entry["step_ms_samples"])
        layers = entry["per_layer"]
        components = [layer_components_us(row) for row in layers]
        measured_components = [value for value in components if value is not None]
        row = {
            "rank": entry["rank"],
            "rc": entry["rc"],
            "rc_source": entry["rc_source"],
            "refusal": entry.get("refusal"),
            "schema": entry["schema"],
            "stage": entry["stage"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "boot_s": entry["boot_s"],
            "first_token_ms": entry["first_token_ms"],
            "tokens": entry["tokens"][:16],
            "token_count": len(entry["tokens"]),
            "steps_run": entry["steps_run"],
            "step_ms": samples,
            "layers_run": entry["layers_run"],
            "reduces_posted": entry["reduces_posted"],
            "reduces_expected": entry["reduces_expected"],
            "weight_bytes_per_token": entry["weight_bytes_per_token"],
            "attention_staged_bytes": entry["attention_staged_bytes"],
            "moe_staged_bytes": entry["moe_staged_bytes"],
            "accepted_tokens_per_step": entry["accepted_tokens_per_step"],
            "per_layer_count": len(layers),
            "per_layer_components_us_sum": round(sum(measured_components), 6) if measured_components else None,
            "attention_bound": entry["attention_bound"],
            "attention_expected": entry["attention_expected"],
            "plan_only": entry["plan_only"],
            "plan": (
                {
                    "plan_hash": entry["plan"].get("plan_hash"),
                    "resident_tensors": entry["plan"].get("resident_tensors"),
                    "shards": entry["plan"].get("shards"),
                    "resident_bytes": entry["plan"].get("resident_bytes"),
                }
                if isinstance(entry.get("plan"), dict)
                else None
            ),
            "walk_performed": entry["walk_performed"],
            "hybrid_sequence": entry.get("hybrid_sequence"),
            "notes": entry["notes"],
        }
        bind_entry = bind_by_rank.get(entry["rank"])
        row["bind"] = bind_entry["bind"] if bind_entry else entry.get("bind")
        row["bind_path"] = bind_entry["path"] if bind_entry else None
        row["bind_sha256"] = bind_entry["sha256"] if bind_entry else None
        row["bind_names_manifest"] = (
            {"path": bind_entry.get("manifest_path"), "present": bind_entry.get("manifest_present")} if bind_entry else None
        )
        if layers:
            total_sum = sum(row["total_us"] for row in layers if row["total_us"] is not None)
            if any(row["total_us"] is not None for row in layers):
                step_wall_us = sum(entry["step_ms_samples"]) * 1000.0 / len(entry["step_ms_samples"])
                row["per_layer_reconcile"] = {
                    "layer_total_us_sum": round(total_sum, 6),
                    "step_wall_us_mean": round(step_wall_us, 6),
                    "residual_us": round(step_wall_us - total_sum, 6),
                }
        per_rank.append(row)

    # A rank that has the primary bind document but no loop document still gets a row, so the
    # missing loop evidence is visible instead of being dropped by an inner join.
    bound_ranks = {row["rank"] for row in per_rank}
    for rank in sorted(bind_by_rank):
        if rank in bound_ranks:
            continue
        entry = bind_by_rank[rank]
        per_rank.append(
            {
                "rank": rank,
                "rc": entry["rc"],
                "rc_source": entry["rc_source"],
                "schema": entry["schema"],
                "stage": entry["stage"],
                "path": entry["path"],
                "sha256": entry["sha256"],
                "boot_s": None,
                "first_token_ms": None,
                "tokens": [],
                "token_count": 0,
                "steps_run": None,
                "step_ms": summarise_samples([]),
                "layers_run": None,
                "reduces_posted": None,
                "reduces_expected": None,
                "weight_bytes_per_token": None,
                "attention_staged_bytes": None,
                "moe_staged_bytes": None,
                "accepted_tokens_per_step": None,
                "per_layer_count": 0,
                "per_layer_components_us_sum": None,
                "attention_bound": entry["attention_bound"],
                "attention_expected": entry["attention_expected"],
                "plan_only": entry["plan_only"],
                "walk_performed": None,
                "hybrid_sequence": None,
                "plan": None,
                "notes": ["bind-only evidence: no tp4-loop-rR.json next to it"],
                "bind": entry["bind"],
                "bind_path": entry["path"],
                "bind_sha256": entry["sha256"],
                "bind_names_manifest": {"path": entry.get("manifest_path"), "present": entry.get("manifest_present")},
            }
        )

    per_rank.sort(key=lambda row: (row["rank"] is None, row["rank"]))
    ranks = [row["rank"] for row in per_rank]
    step_ranks = [entry["rank"] for entry in entries]
    with_samples = [row for row in per_rank if row["step_ms"]["samples"]]
    means = [row["step_ms"]["mean"] for row in with_samples]
    aggregate = {
        "ranks": ranks,
        "ranks_with_step_samples": [row["rank"] for row in with_samples],
        "step_ms_mean_of_ranks": round(statistics.fmean(means), 6) if means else None,
        "step_ms_worst_rank": round(max(means), 6) if means else None,
        "step_ms_best_rank": round(min(means), 6) if means else None,
        "step_ms_median_all_ranks": round(statistics.median(means), 6) if means else None,
        "steps_counted_min": min((len(row["step_ms"]["samples"]) for row in with_samples), default=None),
        "steps_counted_max": max((len(row["step_ms"]["samples"]) for row in with_samples), default=None),
        "tokens_identical_all_ranks": None,
        "per_layer_time_sum_us": None,
        "step_wall_us": None,
        "unattributed_us": None,
    }
    token_streams = [row["tokens"] for row in per_rank if row["tokens"]]
    if token_streams:
        aggregate["tokens_identical_all_ranks"] = all(stream == token_streams[0] for stream in token_streams)
    reconcilable = [row for row in per_rank if row.get("per_layer_reconcile")]
    if reconcilable:
        aggregate["per_layer_time_sum_us"] = round(
            statistics.fmean(row["per_layer_reconcile"]["layer_total_us_sum"] for row in reconcilable), 6
        )
        aggregate["step_wall_us"] = round(reconcilable[0]["per_layer_reconcile"]["step_wall_us_mean"], 6)
        aggregate["unattributed_us"] = round(
            reconcilable[0]["per_layer_reconcile"]["residual_us"], 6
        )

    def check(name: str, ok, detail: str) -> None:
        status = "pass" if ok is True else ("fail" if ok is False else "info")
        checks.append({"name": name, "status": status, "detail": detail})
        if status == "fail":
            failures.append(name)

    expected_ranks = list(range(args.world_size))
    if args.world_size and ranks != expected_ranks:
        check("four ranks present", False, f"ranks {ranks} != expected {expected_ranks}")
    else:
        check("four ranks present", None if not ranks else True, f"ranks {ranks}")

    device_mode = args.kind == "device"

    if device_mode:
        plan_only_all = bool(entries) and all(entry["plan_only"] is True for entry in entries)
        loop_missing = [rank for rank in expected_ranks if rank not in step_ranks]
        if entries:
            check("loop evidence on every rank", not loop_missing, f"ranks {sorted(step_ranks)}" if not loop_missing else f"missing tp4-loop-rR.json for ranks {loop_missing}")
        bind_ranks = sorted(bind_by_rank)
        bind_missing = [rank for rank in expected_ranks if rank not in bind_ranks]
        if bind_ranks:
            check("primary bind evidence on every rank", not bind_missing, f"tp4-bind-rR.json for ranks {bind_ranks}" if not bind_missing else f"missing tp4-bind-rR.json for ranks {bind_missing}")
        elif plan_only_all:
            check("primary bind evidence on every rank", None, "plan-only run: the arm does not write tp4-bind-rR.json without --upload")
        else:
            check(
                "primary bind evidence on every rank",
                False,
                f"missing tp4-bind-rR.json for ranks {bind_missing}: the arm writes it before the walk by design, so this is a failed bind stage",
            )
        if bind_ranks:
            bind_docs = [bind_by_rank[rank] for rank in bind_ranks]
            performed = [entry for entry in bind_docs if entry["bind"]["performed"]]
            errors = [
                {"rank": entry["rank"], "error": entry["bind"]["error"][:120]}
                for entry in bind_docs
                if entry["bind"]["error"]
            ]
            check("bind errors", not errors, "no bind stage reported an error" if not errors else f"errors: {errors}")
            if performed:
                attention_bad = [entry["rank"] for entry in performed if entry["bind"]["attention"]["complete"] is not True]
                check(
                    "bind attention coverage complete",
                    not attention_bad,
                    f"{len(performed)} rank(s) complete" if not attention_bad else f"incomplete on ranks {attention_bad}",
                )
                moe_bad = [
                    entry["rank"]
                    for entry in performed
                    if entry["bind"]["moe"]["performed"] and entry["bind"]["moe"]["complete"] is not True
                ]
                check(
                    "bind MoE/shared/router coverage complete",
                    not moe_bad,
                    f"{len(performed)} rank(s) complete" if not moe_bad else f"incomplete on ranks {moe_bad}",
                )
                tail_bad = [
                    entry["rank"]
                    for entry in performed
                    if entry["bind"]["tail"]["performed"] and entry["bind"]["tail"]["complete"] is not True
                ]
                check("bind tail coverage complete", not tail_bad, f"{len(performed)} rank(s) complete" if not tail_bad else f"incomplete on ranks {tail_bad}")
                census_bad = [
                    {
                        "rank": entry["rank"],
                        "unexpected": [
                            name
                            for name in entry["bind"]["census"]["unclassified"]
                            if name not in EXPECTED_CENSUS_UNCLASSIFIED
                        ],
                    }
                    for entry in performed
                    if any(name not in EXPECTED_CENSUS_UNCLASSIFIED for name in entry["bind"]["census"]["unclassified"])
                ]
                check(
                    "bind census classified (expected unclassified names only)",
                    not census_bad,
                    "no unexpected unclassified shards (the expected multi-modal special-token embeddings are reported as info)"
                    if not census_bad
                    else f"unexpected unclassified names: {census_bad}",
                )
                expected_unclassified = [
                    entry["rank"] for entry in performed if entry["bind"]["census"]["unclassified"]
                ]
                if expected_unclassified:
                    check(
                        "bind census expected unclassified names",
                        None,
                        f"ranks {expected_unclassified} carry the three expected multi-modal special-token embeddings and nothing else",
                    )
                weight_bad = [
                    entry["rank"]
                    for entry in performed
                    if entry["bind"]["moe"]["weight_h2d_bytes_per_token"] not in (None, 0)
                    or entry["bind"]["summary"]["per_token_weight_h2d_bytes"] not in (None, 0)
                ]
                check("bind per-token weight H2D == 0", not weight_bad, "0 bytes/token on every bind" if not weight_bad else f"non-zero on ranks {weight_bad}")
                summary_bad = []
                summary_caveat = []
                for entry in performed:
                    bind = entry["bind"]
                    if bind["summary"]["complete"] is True:
                        continue
                    unclassified = sorted(bind["census"]["unclassified"])
                    clean_sets = (
                        not bind["error"]
                        and bind["attention"]["complete"] is True
                        and (bind["moe"]["complete"] is True or bind["moe"]["performed"] is False)
                        and (bind["tail"]["complete"] is True or bind["tail"]["performed"] is False)
                    )
                    if clean_sets and unclassified == sorted(EXPECTED_CENSUS_UNCLASSIFIED):
                        summary_caveat.append(entry["rank"])
                    else:
                        summary_bad.append(entry["rank"])
                check(
                    "bind summary complete (census caveat allowed)",
                    not summary_bad,
                    "summary.complete on every rank"
                    if not summary_bad and not summary_caveat
                    else (
                        f"summary.complete=false on ranks {summary_caveat} explained by the expected census names; bind sets clean"
                        if not summary_bad
                        else f"summary incomplete on ranks {summary_bad}"
                    ),
                )
                if summary_caveat:
                    check(
                        "bind census caveat",
                        None,
                        f"ranks {summary_caveat}: summary.complete=false by construction because census.unclassified == "
                        "[image_end, image_newline, image_start] (three resident multi-modal token embeddings no device class claims); "
                        "attention/MoE/tail sets complete, error null -> bind-set result clean",
                    )
                incompatible = [
                    {"rank": entry["rank"], "classes": entry["bind"]["moe"]["launch_incompatible"]}
                    for entry in performed
                    if entry["bind"]["moe"]["launch_incompatible"]
                ]
                check(
                    "bind launch compatibility",
                    None,
                    "all bound classes launch-compatible" if not incompatible else f"bound but not launch-compatible: {incompatible}",
                )
                manifest_missing = [entry["rank"] for entry in bind_docs if not entry.get("manifest_present")]
                check(
                    "bind names manifest present",
                    None,
                    "tp4-bind-names-rR.txt on every rank" if not manifest_missing else f"missing names manifest on ranks {manifest_missing}",
                )


    refused_ranks = [row["rank"] for row in per_rank if row.get("refusal")]
    bad_rc = [row["rank"] for row in per_rank if row["rc"] not in (None, 0) and not row.get("refusal")]
    if bad_rc:
        check("all ranks rc==0", False, f"non-zero rc on ranks {bad_rc}")
    else:
        check(
            "all ranks rc==0",
            None if all(row["rc"] is None for row in per_rank) else True,
            "rc 0 (or a labelled forward refusal)" if refused_ranks else "rc 0 (or not recorded)",
        )
    if refused_ranks:
        errors = {row["rank"]: (row.get("refusal") or {}).get("error", "") for row in per_rank if row.get("refusal")}
        check(
            "forward refusal (rc=2)",
            None,
            "ranks %s: forward=refused, positions_run=0, named error; rc=2 accepted as the labelled negative: %s"
            % (refused_ranks, "; ".join("r%d: %s" % (rank, text[:120]) for rank, text in errors.items())),
        )

    if not with_samples:
        check(
            "timed steps measured",
            None,
            "no rank wrote step_ms samples: the loop-only arm does not reach a token or a timed step",
        )
    else:
        missing = [row["rank"] for row in per_rank if not row["step_ms"]["samples"]]
        check("timed steps measured", not missing, f"counted steps on ranks {[row['rank'] for row in with_samples]}" + (f"; missing on {missing}" if missing else ""))
        if any(row["step_ms"]["max"] is not None and row["step_ms"]["max"] > args.step_budget_ms for row in with_samples):
            over = [
                {"rank": row["rank"], "max_ms": row["step_ms"]["max"]}
                for row in with_samples
                if row["step_ms"]["max"] is not None and row["step_ms"]["max"] > args.step_budget_ms
            ]
            check("every counted step within budget", False, f"steps over {args.step_budget_ms} ms: {over}")

    bind_rows = [row for row in per_rank if row["attention_bound"] is not None and row["attention_expected"] is not None]
    if bind_rows:
        plan_only = all(row["plan_only"] is True for row in bind_rows)
        wrong = [
            {"rank": row["rank"], "bound": row["attention_bound"], "expected": row["attention_expected"]}
            for row in bind_rows
            if row["attention_bound"] != row["attention_expected"]
        ]
        if plan_only and all(row["attention_bound"] == 0 for row in bind_rows):
            check("bind coverage bound==expected", None, "plan-only run: binding was not attempted")
        else:
            check("bind coverage bound==expected", not wrong, f"{len(bind_rows)} rank(s) bound" if not wrong else f"mismatches: {wrong}")

    reduce_rows = [row for row in per_rank if row["reduces_posted"] is not None and row["reduces_expected"] is not None]
    if reduce_rows:
        wrong = [
            {"rank": row["rank"], "posted": row["reduces_posted"], "expected": row["reduces_expected"]}
            for row in reduce_rows
            if row["reduces_posted"] != row["reduces_expected"]
        ]
        check("reduces posted == 2 per layer per step", not wrong, f"{len(reduce_rows)} rank(s) reconciled" if not wrong else f"mismatches: {wrong}")
    else:
        check("reduces posted == 2 per layer per step", None, "no reduce accounting in the evidence")

    weight_values = [row["weight_bytes_per_token"] for row in per_rank if row["weight_bytes_per_token"] is not None]
    if weight_values:
        check("weight_bytes_per_token == 0 (resident weights)", all(value == 0 for value in weight_values), f"values {weight_values}")
    else:
        check("weight_bytes_per_token == 0 (resident weights)", None, "not recorded")

    if args.require_per_layer and with_samples and args.kind == "device":
        with_layers = [row["rank"] for row in per_rank if row["per_layer_count"]]
        check("per-layer breakdown present", len(with_layers) == len(per_rank) and bool(per_rank), f"layers on ranks {with_layers} (needed on every rank)")
    if aggregate["tokens_identical_all_ranks"] is not None:
        check("tokens identical all ranks", aggregate["tokens_identical_all_ranks"], "generated token ids match across ranks")
    partial_row = False
    if args.expected_steps is not None:
        short = [
            {"rank": row["rank"], "steps": len(row["step_ms"]["samples"])}
            for row in with_samples
            if len(row["step_ms"]["samples"]) < args.expected_steps
        ]
        partial_row = bool(short)
        check(
            "expected counted steps recorded",
            None,
            f"partial row: {short} of {args.expected_steps} planned counted steps" if short else f"{args.expected_steps} counted steps per rank",
        )

    # ---------- budget arithmetic
    worst = aggregate["step_ms_worst_rank"]
    accepted = args.accepted_tokens_per_step
    if accepted is None and any(row["accepted_tokens_per_step"] is not None for row in per_rank):
        accepted = min(row["accepted_tokens_per_step"] for row in per_rank if row["accepted_tokens_per_step"] is not None)
        accepted_source = "arm step.accepted_tokens_per_step"
    else:
        accepted_source = args.accepted_tokens_source if accepted is not None else None
    tok_s = None
    if worst and accepted is not None:
        tok_s = accepted * 1000.0 / worst
    single = 1000.0 / worst if worst else None
    step_time_budget = None
    if worst is not None:
        if args.kind == "host-twin":
            step_time_budget = "NOT-APPLICABLE (host reference row: the 42 ms budget is a device target)"
        else:
            step_time_budget = "PASS" if worst <= args.step_budget_ms else "FAIL"
    tok_s_gate = "UNMEASURED" if tok_s is None else ("MET" if tok_s >= args.tok_s_target else "UNMET")
    if accepted is None:
        accepted_verdict = "NOT-WIRED (DSpark/MTP verify step not on the device path)"
    else:
        accepted_verdict = f"MEASURED (source: {accepted_source})"
    gate_grade = (
        step_time_budget == "PASS"
        and accepted is not None
        and not failures
    )
    if args.kind == "host-twin":
        label = "HOST-REFERENCE (separate kind: never placed in the device table)"
    elif gate_grade:
        label = "GATE-GRADE"
    elif step_time_budget in ("PASS", None):
        label = "STEP-TIME-ONLY"
    else:
        label = "STEP-TIME-FAIL"
    if failures and step_time_budget == "PASS":
        label = "STEP-TIME-REFUSED"

    budget = {
        "step_budget_ms": args.step_budget_ms,
        "accepted_tokens_per_step": accepted,
        "accepted_tokens_source": accepted_source,
        "tok_s_target": args.tok_s_target,
        "parity_band_tok_s": list(args.parity_band),
        "derived": {
            "ms_per_accepted_token_if_measured": round(worst / accepted, 6) if (worst and accepted) else None,
            "accepted_tokens_per_second_if_measured": round(tok_s, 6) if tok_s is not None else None,
            "single_token_tokens_per_second": round(single, 6) if single is not None else None,
        },
    }

    status = "NOT-ATTEMPTED"
    if worst is None and failures:
        status = "BIND-FAIL"
    if worst is not None:
        if args.kind == "host-twin":
            status = "HOST-TWIN-ROW"
        elif step_time_budget == "FAIL" or failures:
            status = "STEP-TIME-FAIL"
        else:
            status = "STEP-TIME-PASS"

    labels = [
        "device and host-twin rows never share a table; this document has exactly one kind",
        "single-token tok/s = 1000/step_ms is a lower bound, not the D48 verify-step target",
        "accepted tokens/step is null unless the arm's real verify step recorded it (D50 acceptance rule, no placeholders)",
        "a step budget PASS with no verify step is STEP-TIME-ONLY: the 90 tok/s gate stays unmet, not redefined",
        "a host-twin row is a reference measurement: it has its own kind and never shares the device table",
        "the progressive tp4-bind-rR.json is primary window evidence: it is written before the walk, so its absence is a failed bind stage, not an absent measurement",
        "per-layer sums are the arm's own timers; a residual is reported as unattributed, never split across layers",
    ]
    if partial_row:
        labels.insert(
            0,
            f"partial row: {args.expected_steps} counted steps were planned; the recorded step count is the honest denominator",
        )
    return {
        "per_rank": per_rank,
        "aggregate": aggregate,
        "checks": checks,
        "failures": failures,
        "budget": budget,
        "verdict": {
            "step_time_budget": step_time_budget or "UNMEASURED",
            "accepted_tokens_per_step": accepted_verdict,
            "tok_s_gate": tok_s_gate,
            "label": label,
        },
        "status": status,
        "labels": labels,
    }


def render_markdown(result: dict) -> str:
    lines = [
        f"# D48 step row ({result['kind']}) - schema {result['schema']}",
        "",
        f"- generated (UTC): {result['generated_utc']}",
        f"- status: **{result['status']}**",
        f"- verdict label: {result['verdict']['label']}",
        f"- step budget: worst-rank mean {result['aggregate']['step_ms_worst_rank']} ms vs {result['budget']['step_budget_ms']} ms -> {result['verdict']['step_time_budget']}",
        f"- accepted tokens/step: {result['budget']['accepted_tokens_per_step']} ({result['verdict']['accepted_tokens_per_step']})",
        f"- derived tok/s (only if accepted tokens measured): {result['budget']['derived']['accepted_tokens_per_second_if_measured']}",
        f"- single-token lower bound: {result['budget']['derived']['single_token_tokens_per_second']} tokens/s (not the gate metric)",
        f"- evidence: {', '.join(item['path'] for item in result['provenance']['evidence_files']) or 'none'}",
        "",
        "## Per-rank step rows",
        "",
        "| rank | steps | mean ms | worst ms | first_token_ms | tokens | reduces | weight B/token | layers |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |",
    ]
    for row in result["per_rank"]:
        reduces = (
            f"{row['reduces_posted']}/{row['reduces_expected']}"
            if row["reduces_posted"] is not None
            else "-"
        )
        lines.append(
            f"| {row['rank']} | {row['step_ms']['samples'] and len(row['step_ms']['samples']) or 0} | "
            f"{row['step_ms']['mean'] if row['step_ms']['mean'] is not None else '-'} | "
            f"{row['step_ms']['max'] if row['step_ms']['max'] is not None else '-'} | "
            f"{row['first_token_ms'] if row['first_token_ms'] is not None else '-'} | "
            f"{row['token_count']} | {reduces} | "
            f"{row['weight_bytes_per_token'] if row['weight_bytes_per_token'] is not None else '-'} | "
            f"{row['layers_run'] if row['layers_run'] is not None else '-'} |"
        )
    lines += ["", "Step samples per rank (counted decode steps, warm-up excluded by the arm):", ""]
    for row in result["per_rank"]:
        if row["step_ms"]["samples"]:
            lines.append(f"- rank {row['rank']}: {row['step_ms']['samples']} ms ({row['path']})")
        else:
            lines.append(f"- rank {row['rank']}: no timed step ({row['path']})")
    lines += ["", "## Checks", "", "| check | status | detail |", "| :--- | :--- | :--- |"]
    for check in result["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
    lines += [
        "",
        "## Budget arithmetic",
        "",
        "| quantity | value |",
        "| :--- | ---: |",
        f"| step_ms_worst_rank | {result['aggregate']['step_ms_worst_rank']} |",
        f"| step_ms_mean_of_ranks | {result['aggregate']['step_ms_mean_of_ranks']} |",
        f"| accepted_tokens_per_step | {result['budget']['accepted_tokens_per_step']} |",
        f"| accepted_tokens_per_second | {result['budget']['derived']['accepted_tokens_per_second_if_measured']} |",
        f"| single_token_tokens_per_second | {result['budget']['derived']['single_token_tokens_per_second']} |",
        "",
        "## Labels and non-claims",
        "",
    ]
    lines.extend(f"- {label}" for label in result["labels"])
    if result["failures"]:
        lines += ["", "**Refused cross-checks:** " + ", ".join(result["failures"])]
    return "\n".join(lines) + "\n"


def parse_args(argv):
    parser = argparse.ArgumentParser(description="D48 step-time collector for four-rank device evidence")
    parser.add_argument("--device-dir", action="append", default=[], help="run directory (rank*/...) or evidence files (repeatable)")
    parser.add_argument("--kind", choices=("device", "host-twin"), default="device", help="exactly one kind per document")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--expected-steps", type=int, help="planned counted steps (labels a short row as partial)")
    parser.add_argument("--step-budget-ms", type=float, default=D48_STEP_BUDGET_MS)
    parser.add_argument("--accepted-tokens-per-step", type=float, help="only when the arm's verify step measured it")
    parser.add_argument("--accepted-tokens-source", help="required with --accepted-tokens-per-step: which arm field/log line recorded it")
    parser.add_argument("--tok-s-target", type=float, default=D48_TOK_S_TARGET)
    parser.add_argument("--parity-band", type=float, nargs=2, default=list(D48_PARITY_BAND_TOK_S))
    parser.add_argument("--require-per-layer", action="store_true", default=True, help="require the per-layer breakdown on every rank (default on)")
    parser.add_argument("--no-require-per-layer", dest="require_per_layer", action="store_false")
    parser.add_argument("--tree", help="git sha of the integrated tree that produced the binaries")
    parser.add_argument("--binary-sha256")
    parser.add_argument("--cubins-manifest-sha256")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--expect-vocab", type=int, help="released vocabulary (129280)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_md = args.out_md or (os.path.splitext(args.out_json)[0] + ".md")
    try:
        if not args.device_dir:
            raise GateError("--device-dir is required (the window run directory with rank*/ evidence)")
        if args.accepted_tokens_per_step is not None and not args.accepted_tokens_source:
            raise GateError(
                "--accepted-tokens-per-step needs --accepted-tokens-source: only a real DSpark/MTP verify-step "
                "measurement may populate it, and the source must name the arm field/log line"
            )
        ignored: list = []
        files = collect_files(args.device_dir, ignored)
        entries = [extract_rank(path) for path in files]
        # The Lead's four-rank wrapper writes chain-rR.log / chain-rR.rc; our own older runs used
        # run.log / run.exit. Read the exit code beside the evidence under either name.
        for entry in entries:
            if entry["rc"] is not None:
                continue
            directory = os.path.dirname(entry["path"])
            names = ["run.exit"]
            if entry["rank"] is not None:
                names.insert(1, "chain-r%d.rc" % entry["rank"])
            names.append("chain-r0.rc")
            for name in names:
                rc_path = os.path.join(directory, name)
                if not os.path.exists(rc_path):
                    continue
                with open(rc_path, "r", encoding="utf-8", errors="replace") as handle:
                    rc_text = handle.read().strip()
                try:
                    entry["rc"] = int(rc_text)
                except ValueError:
                    entry["rc_source"] = "%s (unparseable %r)" % (name, rc_text)
                    break
                entry["rc_source"] = name
                break
        for entry in entries:
            if entry["rc"] not in (None, 0) and entry["rank"] is not None:
                entry["refusal"] = refusal_evidence(os.path.dirname(entry["path"]), entry["rank"])
        if not entries:
            raise GateError("no evidence JSON found in --device-dir")
        step_entries = [entry for entry in entries if entry["role"] == "step"]
        bind_entries = [entry for entry in entries if entry["role"] == "bind"]
        step_ranks = [entry["rank"] for entry in step_entries if entry["rank"] is not None]
        if len(step_ranks) != len(set(step_ranks)):
            raise GateError(f"duplicate step evidence: ranks {sorted(step_ranks)}")
        bind_by_rank: dict = {}
        for entry in bind_entries:
            if entry["rank"] is None:
                raise GateError(f"{entry['path']}: a bind document without a rank")
            if entry["rank"] in bind_by_rank:
                raise GateError(f"duplicate bind evidence for rank {entry['rank']}")
            entry["manifest_path"] = os.path.join(
                os.path.dirname(entry["path"]), f"tp4-bind-names-r{entry['rank']}.txt"
            )
            entry["manifest_present"] = os.path.exists(entry["manifest_path"])
            bind_by_rank[entry["rank"]] = entry
        for entry in entries:
            if entry["tokens"] and args.expect_vocab is not None:
                for token in entry["tokens"]:
                    if not 0 <= token < args.expect_vocab:
                        raise GateError(f"{entry['path']}: token {token} outside vocabulary {args.expect_vocab}")
        aux_entries = [entry for entry in entries if entry["role"] == "aux"]
        evidence = build_evidence(step_entries, args, bind_by_rank=bind_by_rank)
        if aux_entries:
            evidence["checks"].append(
                {
                    "name": "auxiliary evidence ignored",
                    "status": "info",
                    "detail": f"{len(aux_entries)} non-acceptance file(s) in the evidence directory: "
                    + "; ".join(
                        f"{os.path.basename(entry['path'])} ({entry['aux_reason']})" for entry in aux_entries
                    ),
                }
            )
        result = {
            "schema": OUTPUT_SCHEMA,
            "generated_utc": utc_now(),
            "kind": args.kind,
            "provenance": {
                "tree": args.tree,
                "binary_sha256": args.binary_sha256,
                "cubins_manifest_sha256": args.cubins_manifest_sha256,
                "checkpoint_root": args.checkpoint_root,
                "world_size": args.world_size,
                "expected_steps": args.expected_steps,
                "evidence_files": [
                    {"path": entry["path"], "sha256": entry["sha256"], "rank": entry["rank"], "schema": entry["schema"], "stage": entry["stage"], "role": entry["role"]}
                    for entry in entries
                ],
                "ignored_files": ignored,
                "clock": "host wall clock per rank; no GPU timestamps unless the rank recorded them",
            },
            **evidence,
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
        with open(out_md, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(result))
        print(render_markdown(result))
        print(f"wrote {args.out_json} and {out_md}")
        if result["status"] in ("STEP-TIME-PASS", "HOST-TWIN-ROW"):
            return 0
        if result["status"] in ("STEP-TIME-FAIL", "BIND-FAIL"):
            return 1
        return 3
    except (GateError, compare.CompareError) as error:
        print(f"d48_step_gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
