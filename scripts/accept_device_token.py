#!/usr/bin/env python3
"""D40/D41 device-token acceptance runner (M1 gate preparation).

Consumes what a four-rank device window actually emits and reports **separate** verdicts; it never
blends them into one pass and never treats a host artefact as device evidence.

Inputs (all explicit; nothing is auto-promoted):

  * --device (repeatable) - device evidence: a JSON file per rank or a directory of per-rank
    JSONs. Accepted layouts:
      - "sparkflash-device-token-evidence/1" (canonical, defined below),
      - "sparkflash-tp4-loop/*" / "sparkflash-tp4-chain/*" (the four-rank arm; reads
        tokens.generated / token_emitted / step / per_layer / collectives / attention),
      - a "tp4-host" evidence JSON (step_tokens/generated_tokens) - accepted as twin evidence, and
        as device evidence only when explicitly passed under --device and then labelled by schema.
  * --device-dump - the arm's own full-vocabulary dump (JSONL + sidecar, frozen
    docs/build/logits-dump.md schema) when it writes one. **Explicit only**: a directory scan never
    picks up a dump, so a world=1 hybrid-dump-logits.jsonl cannot be mistaken for the four-rank
    device dump. --dump-source records the provenance label.
  * --twin (repeatable) - host-twin evidence (same prompt and step count as the device run);
    token-id-only evidence is supported.
  * --oracle + --oracle-prompt - frozen "sparkflash-native-oracle/1" oracle; with a device dump the
    full D39 distributional comparison runs, otherwise a token-only oracle leg is reported (and is
    UNMEASURED when the device evidence has no explicit positions).
  * --oracle-bars - a real "sparkflash-d41-bars/1" gate file; without it the oracle leg is
    **UNBARED** (measured metrics, no verdict). No bars file exists today (see
    docs/build/d41-floor-verdict-20261020.md). --provisional-d40 prints the published D40
    provisional values as D40-PROVISIONAL-NOT-A-GATE diagnostics only.

Verdicts written to the JSON/MD (each independent):

  shape            - evidence parsed, token ids sane, dump/sidecar/vocab/positions rules, pinned
                     positions present (the frozen consumer's checks, reused).
  pre_token        - when no token exists: bind coverage bound==expected, collectives
                     2*layers*steps, canonical bf16 check, plan/rank consistency.
  twin_match       - device vs host-twin, per position when both sides carry explicit positions,
                     otherwise sequence-index alignment; PASS/FAIL/UNMEASURED/NOT_ATTEMPTED.
  oracle_comparison - PASS/FAIL with a D41 gate file, UNBARED without, NOT_LIKE_FOR_LIKE if the
                     dump cannot prove its prefix, UNMEASURED if not comparable.
  step_time        - reported only when the arm records step timings; the D48 42 ms /
                     3.8 accepted tok/step / 90 tok/s target is shown as a TARGET-ONLY comparison
                     (tonight's M1 gate is not the step budget).

Canonical device evidence ("sparkflash-device-token-evidence/1"), for a small sidecar the Lead can
write from window logs without touching the arm's evidence:

    {"schema": "sparkflash-device-token-evidence/1", "kind": "device",
     "rank": 0, "world_size": 4,
     "prompt_tokens": [11111, 201, 8197, 14, 270, 3417, 344, 28],
     "token_ids": [1192, 1623], "positions": [7, 8],
     "token_emitted": true, "first_token_ms": 12345.6, "step_ms": [12345.6],
     "logits_sha256": "...", "execution": "...", "notes": ["..."]}

Exit codes: 0 PASS, 1 FAIL, 3 UNMEASURED / NOT_ATTEMPTED, 2 bad input (nothing written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compare_native_vs_vllm as compare

CANONICAL_SCHEMA = "sparkflash-device-token-evidence/1"
# device-lead, 2026-10-20: the plan carries three multi-modal special-token embeddings that no device
# class claims. They are EXPECTED unclassified names, not a defect; anything else unclassified fails.
EXPECTED_CENSUS_UNCLASSIFIED = {"image_end", "image_newline", "image_start"}
OUTPUT_SCHEMA = "sparkflash-device-acceptance/1"
DEVICE_VOCAB_RELEASED = 129280
D40_POSITIONS_MIN = 9
FULL_POSITIONS = (129, 130, 131, 132, 133, 134, 135, 136, 137)

#: The published D40 provisional values (docs/DECISIONS.md D40, quoted as PROVISIONAL_BARS in
#: scripts/d41_floors.py). They are NOT a gate: D41 freezes the gate floor from a reviewed capture,
#: and no D41 bars file exists. Used only to print a labelled diagnostic when the operator asks for
#: it; never to turn a run into PASS/FAIL.
D40_PROVISIONAL_BARS = {
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

#: D48 target row (not tonight's M1 gate; docs/DECISIONS.md D48).
D48_STEP_BUDGET_MS = 42.0
D48_ACCEPTED_TOKENS_PER_STEP = 3.8
D48_TOK_S_TARGET = 90.0
D48_PARITY_TOK_S = (50.0, 60.0)

STEP_MS_TARGET_ONLY_LABEL = "D48-TARGET-ONLY (tonight's M1 gate is a token gate, not the step budget)"


class AcceptError(RuntimeError):
    """Input that cannot be accepted honestly."""


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
        raise AcceptError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise AcceptError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(doc, dict):
        raise AcceptError(f"{path}: evidence must be a JSON object, got {type(doc).__name__}")
    return doc


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_list(value):
    if isinstance(value, list) and all(_is_int(item) for item in value):
        return list(value)
    return None


def _is_float(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _float_list(value):
    if isinstance(value, list) and all(_is_float(item) for item in value):
        return [float(item) for item in value]
    return None


def _first_int_list(doc: dict, keys):
    for key in keys:
        value = doc.get(key)
        tokens = _int_list(value)
        if tokens is not None:
            return tokens, key
    return [], None


def _rank_from_name(path: str):
    match = re.search(r"-r(\d+)\.json$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _rank_from_doc(doc: dict):
    for key in ("rank", "tp_rank"):
        value = doc.get(key)
        if _is_int(value):
            return value
    return None


def _world_from_doc(doc: dict):
    for key in ("world", "world_size", "tp_world_size"):
        value = doc.get(key)
        if _is_int(value):
            return value
    return None


AUX_NAME = re.compile(
    r"^(timeline|preflight|wrapper|chain-timeline|window)[-_.].*\.json$|^(?:.*[-_])?adjudication(?:[-_.].*)?\.json$",
    re.IGNORECASE,
)


def _aux_by_name(path: str) -> bool:
    """Wrapper/timeline metadata JSON that may share a rank directory is not acceptance evidence."""
    return bool(AUX_NAME.match(os.path.basename(path)))


def _schema_kind(doc: dict) -> str:
    schema = doc.get("schema")
    if schema == CANONICAL_SCHEMA:
        return "canonical"
    if schema == "sparkflash-tp4-bind/1":
        return "tp4-bind"
    if schema == "sparkflash-tp4-token/1":
        return "tp4-token"
    if doc.get("stage") == "embedding-gathered":
        # device-lead's forward-stage evidence: prompt-token embeddings gathered, the first dense
        # attention launches recorded, token_claimed=false when the loop refuses before a token.
        # It is pre-token evidence, not an unknown layout.
        return "tp4-forward"
    if doc.get("stage") == "tp4-host" or doc.get("stage") == "tp4-loop":
        return "tp4-host"
    if doc.get("stage") == "tp4-resident":
        return "aux"
    if schema == "sparkflash-tp4-chain/1":
        return "aux"
    if isinstance(schema, str) and schema.startswith("sparkflash-tp4-resident"):
        return "aux"
    if isinstance(schema, str) and schema.startswith("sparkflash-tp4-") and schema != "sparkflash-tp4-bind/1":
        return "tp4-arm"
    if isinstance(schema, int):
        # The resident stage's evidence uses an integer schema version; it may share the evidence
        # directory with the loop run without becoming a device row.
        return "aux"
    if _int_list(doc.get("generated_tokens")) is not None:
        return "tp4-host"
    if _int_list(doc.get("token_ids")) is not None or _int_list(doc.get("tokens")) is not None:
        return "generic"
    if doc.get("stage") == "tp4-chain" or isinstance(doc.get("attention"), dict):
        return "tp4-arm"
    return "unknown"


def _collect_paths(paths, seen: list, ignored: list) -> list:
    files: list = []
    for path in paths:
        if os.path.isdir(path):
            # A window run directory holds rank0..rank3 subdirectories, so walk them; only JSON is
            # evidence, and JSONL dumps stay explicit-only (never promoted by a scan).
            for directory, _, names in os.walk(path):
                for name in sorted(names):
                    full = os.path.join(directory, name)
                    if name.endswith(".json"):
                        files.append(full)
                    elif name.endswith(".jsonl"):
                        ignored.append(
                            {
                                "path": os.path.abspath(full),
                                "reason": "dump files are explicit-only (pass --device-dump); a directory scan never promotes a JSONL to evidence",
                            }
                        )
        else:
            files.append(path)
    result: list = []
    for path in files:
        absolute = os.path.abspath(path)
        if absolute in seen:
            continue
        seen.append(absolute)
        result.append(absolute)
    return result


def bind_summary(doc: dict) -> dict:
    """Normalise a sparkflash-tp4-bind/1 document into the checks the runner applies."""
    attention = doc.get("attention") if isinstance(doc.get("attention"), dict) else {}
    moe = doc.get("moe") if isinstance(doc.get("moe"), dict) else {}
    tail = doc.get("tail") if isinstance(doc.get("tail"), dict) else {}
    census = doc.get("census") if isinstance(doc.get("census"), dict) else {}
    summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
    attention_expected = attention.get("tensors_expected") if _is_int(attention.get("tensors_expected")) else None
    attention_bound = attention.get("tensors_bound") if _is_int(attention.get("tensors_bound")) else None
    attention_coverage = attention.get("coverage") if isinstance(attention.get("coverage"), dict) else {}
    classes = moe.get("classes") if isinstance(moe.get("classes"), list) else []
    moe_rows = []
    for row in classes:
        if not isinstance(row, dict):
            continue
        moe_rows.append(
            {
                "class": row.get("class") if isinstance(row.get("class"), str) else None,
                "expected": row.get("expected") if _is_int(row.get("expected")) else None,
                "bound": row.get("bound") if _is_int(row.get("bound")) else None,
                "complete": row.get("complete") if isinstance(row.get("complete"), bool) else None,
                "missing": row.get("missing") if isinstance(row.get("missing"), list) else [],
                "unexpected": row.get("unexpected") if isinstance(row.get("unexpected"), list) else [],
            }
        )
    moe_expected = sum(row["expected"] for row in moe_rows if row["expected"] is not None)
    moe_bound = sum(row["bound"] for row in moe_rows if row["bound"] is not None)
    moe_complete = all(row["complete"] is True for row in moe_rows) if moe_rows else None
    compat = moe.get("launch_compatibility") if isinstance(moe.get("launch_compatibility"), list) else []
    incompatible = [
        {"class": row.get("class"), "reason": row.get("reason")}
        for row in compat
        if isinstance(row, dict) and row.get("launch_compatible") is False
    ]
    tail_expected = tail.get("tensors_expected") if _is_int(tail.get("tensors_expected")) else None
    tail_bound = tail.get("tensors_bound") if _is_int(tail.get("tensors_bound")) else None
    if tail_expected is None:
        tail_coverage = tail.get("coverage") if isinstance(tail.get("coverage"), dict) else {}
        if _is_int(tail_coverage.get("expected")):
            tail_expected = tail_coverage["expected"]
            tail_bound = tail_coverage.get("bound") if _is_int(tail_coverage.get("bound")) else None
    tail_complete = None
    tail_coverage = tail.get("coverage") if isinstance(tail.get("coverage"), dict) else {}
    if isinstance(tail_coverage.get("complete"), bool):
        tail_complete = tail_coverage["complete"]
    elif tail_expected is not None and tail_bound is not None and tail_expected > 0:
        tail_complete = tail_expected == tail_bound
    census_unclassified = census.get("unclassified") if isinstance(census.get("unclassified"), list) else None
    return {
        "performed": doc.get("performed") is not False,
        "error": doc.get("error") if isinstance(doc.get("error"), str) else None,
        "attention": {
            "expected": attention_expected,
            "bound": attention_bound,
            "complete": attention_coverage.get("complete") if isinstance(attention_coverage.get("complete"), bool) else (
                attention_expected == attention_bound if attention_expected is not None and attention_bound is not None and attention_expected > 0 else None
            ),
            "missing": attention_coverage.get("missing") if isinstance(attention_coverage.get("missing"), list) else [],
        },
        "moe": {
            "performed": moe.get("performed") is not False and bool(moe_rows),
            "expected": moe_expected if moe_rows else None,
            "bound": moe_bound if moe_rows else None,
            "complete": moe_complete,
            "classes": moe_rows,
            "weight_h2d_bytes_per_token": moe.get("weight_h2d_bytes_per_token") if _is_int(moe.get("weight_h2d_bytes_per_token")) else None,
            "launch_incompatible": incompatible,
        },
        "tail": {
            "performed": tail.get("performed") is not False,
            "expected": tail_expected,
            "bound": tail_bound,
            "complete": tail_complete,
            "pointers_nonzero": tail.get("pointers_nonzero") if isinstance(tail.get("pointers_nonzero"), bool) else None,
        },
        "census": {
            "performed": census.get("performed") is not False and isinstance(census.get("resident_tensors"), int),
            "resident_tensors": census.get("resident_tensors") if _is_int(census.get("resident_tensors")) else None,
            "unclassified": census_unclassified,
        },
        "summary": {
            "device_bound_names": summary.get("device_bound_names") if _is_int(summary.get("device_bound_names")) else None,
            "per_token_weight_h2d_bytes": summary.get("per_token_weight_h2d_bytes") if _is_int(summary.get("per_token_weight_h2d_bytes")) else None,
            "complete": summary.get("complete") if isinstance(summary.get("complete"), bool) else None,
        },
        "names_manifest": doc.get("names_manifest") if isinstance(doc.get("names_manifest"), str) else None,
    }


def extract_evidence(files: list, *, expect_vocab, side: str) -> list:
    """Normalise every evidence file into one small layout; unknown files are recorded, not dropped."""
    evidence: list = []
    for path in files:
        doc = load_json(path)
        kind = _schema_kind(doc)
        layout = {
            "path": path,
            "plan_only": doc.get("plan_only") if isinstance(doc.get("plan_only"), bool) else None,
            "sha256": sha256_file(path),
            "schema": doc.get("schema"),
            "kind": "aux" if kind == "unknown" and _aux_by_name(path) else kind,
            "aux_reason": "wrapper metadata filename, not an acceptance input" if kind == "unknown" and _aux_by_name(path) else None,
            "rank": _rank_from_doc(doc) if _rank_from_doc(doc) is not None else _rank_from_name(path),
            "world": _world_from_doc(doc),
            "token_emitted": doc.get("token_emitted") if isinstance(doc.get("token_emitted"), bool) else None,
            "tokens": [],
            "token_key": None,
            "positions": None,
            "start_position": None,
            "prompt_tokens": None,
            "prompt_len": None,
            "step_ms": [],
            "first_token_ms": None,
            "attention": doc.get("attention") if isinstance(doc.get("attention"), dict) else None,
            "bind": bind_summary(doc) if isinstance(doc.get("bind"), dict) else None,
            "plan": doc.get("plan") if isinstance(doc.get("plan"), dict) else None,
            "collectives": doc.get("collectives") if isinstance(doc.get("collectives"), dict) else None,
            "step": doc.get("step") if isinstance(doc.get("step"), dict) else None,
            "per_layer": doc.get("per_layer") if isinstance(doc.get("per_layer"), list) else None,
            "logits_sha256": doc.get("logits_sha256"),
            "forward": doc.get("forward") if isinstance(doc.get("forward"), str) else None,
            "forward_embed": None,
            "token_error": doc.get("error") if isinstance(doc.get("error"), str) else None,
            "positions_run": doc.get("positions_run") if _is_int(doc.get("positions_run")) else None,
            "logits_dump": doc.get("logits_dump") if isinstance(doc.get("logits_dump"), dict) else None,
            "accepted_tokens_per_step": (
                float(doc["accepted_tokens_per_step"])
                if isinstance(doc.get("accepted_tokens_per_step"), (int, float)) and not isinstance(doc.get("accepted_tokens_per_step"), bool)
                else None
            ),
            "reduce_calls_per_step": doc.get("reduce_calls_per_step") if _is_int(doc.get("reduce_calls_per_step")) else None,
            "notes": [],
        }

        tokens: list = []
        token_key = None
        if kind == "canonical":
            tokens = _int_list(doc.get("token_ids")) or []
            token_key = "token_ids"
            layout["positions"] = _int_list(doc.get("positions"))
            if _is_int(doc.get("start_position")):
                layout["start_position"] = int(doc["start_position"])
        elif kind == "tp4-token":
            token_block = doc.get("tokens") if isinstance(doc.get("tokens"), dict) else {}
            tokens = _int_list(token_block.get("ids")) or []
            token_key = "tokens.ids"
            positions = _int_list(token_block.get("positions"))
            start = token_block.get("start_position")
            if _is_int(start):
                layout["start_position"] = int(start)
                if positions is None:
                    positions = [int(start) + index for index in range(len(tokens))]
            if positions is not None and len(positions) == len(tokens):
                layout["positions"] = positions
            prompt = token_block.get("prompt_len")
            if _is_int(prompt):
                layout["prompt_len"] = int(prompt)
            if token_block.get("logits_sha256"):
                layout["logits_sha256"] = token_block["logits_sha256"]
            step_block = doc.get("step_ms") if isinstance(doc.get("step_ms"), dict) else {}
            samples = _float_list(step_block.get("samples")) if step_block else None
            if samples:
                layout["step_ms"] = samples
            wall = _float_list(doc.get("wall_ms"))
            if wall:
                layout["wall_ms"] = wall
                if not samples:
                    layout["step_ms"] = wall
            if doc.get("first_token_ms") is not None and _is_float(doc.get("first_token_ms")):
                layout["first_token_ms"] = float(doc["first_token_ms"])
        elif kind == "tp4-arm":
            token_block = doc.get("tokens")
            if isinstance(token_block, dict):
                generated = _int_list(token_block.get("generated"))
                if generated is None:
                    generated = _int_list(token_block.get("token_ids"))
                tokens = generated or []
                token_key = "tokens.generated"
                start = token_block.get("start_position")
                if _is_int(start):
                    layout["start_position"] = start
                    layout["positions"] = [start + index for index in range(len(tokens))]
                prompt = token_block.get("prompt_tokens")
                if _is_int(prompt):
                    layout["prompt_len"] = prompt
                elif isinstance(prompt, list):
                    layout["prompt_tokens"] = _int_list(prompt)
                    layout["prompt_len"] = len(prompt)
                if token_block.get("logits_sha256"):
                    layout["logits_sha256"] = token_block["logits_sha256"]
                if isinstance(token_block.get("token_emitted"), bool):
                    layout["token_emitted"] = token_block["token_emitted"]
            elif isinstance(token_block, list):
                tokens = _int_list(token_block) or []
                token_key = "tokens"
            if not tokens:
                for key in ("token_ids", "generated_tokens"):
                    candidate = _int_list(doc.get(key))
                    if candidate is not None:
                        tokens, token_key = candidate, key
                        break
            if layout["token_emitted"] is None:
                layout["token_emitted"] = bool(tokens)
        elif kind == "tp4-forward":
            # Pre-token forward-stage evidence: embeddings gathered and the first attention launches
            # ran, then the loop refused before a token id (token_claimed=false). It is consumed as
            # pre-token/report evidence only; it never becomes a token row.
            layout["token_emitted"] = False
            layout["forward_embed"] = {
                "stage": doc.get("stage"),
                "token_claimed": doc.get("token_claimed") if isinstance(doc.get("token_claimed"), bool) else None,
                "prompt_len": doc.get("prompt_len") if _is_int(doc.get("prompt_len")) else None,
                "decode_positions": doc.get("decode_positions") if _is_int(doc.get("decode_positions")) else None,
                "first_step_position": doc.get("first_step_position") if _is_int(doc.get("first_step_position")) else None,
                "reference_ids": _int_list(doc.get("reference_ids")) or [],
                "attention_launchable": doc.get("attention_launchable") if _is_int(doc.get("attention_launchable")) else None,
                "attention_blocked": doc.get("attention_blocked") if isinstance(doc.get("attention_blocked"), list) else [],
                "next_missing": doc.get("next_missing") if isinstance(doc.get("next_missing"), str) else None,
                "note": doc.get("note") if isinstance(doc.get("note"), str) else None,
            }
            token_key = "none (forward-embed evidence: no token ids, pre-token launch census)"
        elif kind == "tp4-bind":
            layout["bind"] = bind_summary(doc)
            layout["token_emitted"] = False
            token_key = "none (progressive bind evidence: forward=not-run, tokens=0)"
        elif kind == "tp4-host":
            tokens, token_key = _first_int_list(doc, ("generated_tokens", "token_ids", "step_tokens"))
            layout["notes"].append(
                "tp4-host layout: generated_tokens are the produced ids; step_tokens (per-step running sequence) are NOT used as token ids"
            )
            prompt = doc.get("prompt_tokens")
            if isinstance(prompt, list):
                layout["prompt_tokens"] = _int_list(prompt)
                layout["prompt_len"] = len(prompt)
            elif _is_int(prompt):
                layout["prompt_len"] = prompt
            timing = doc.get("timing")
            if isinstance(timing, dict):
                samples = timing.get("step_ms")
                if isinstance(samples, list) and all(isinstance(item, (int, float)) for item in samples):
                    layout["step_ms"] = [float(item) for item in samples]
                if isinstance(timing.get("mean_step_ms"), (int, float)):
                    layout["notes"].append(f"host twin mean_step_ms={timing['mean_step_ms']}")
            if layout["token_emitted"] is None:
                layout["token_emitted"] = bool(tokens)
        else:
            tokens, token_key = _first_int_list(doc, ("token_ids", "generated_tokens", "tokens", "generated"))
            if layout["token_emitted"] is None:
                layout["token_emitted"] = bool(tokens)

        step = layout["step"] or {}
        samples = step.get("step_ms_samples")
        if not layout["step_ms"] and isinstance(samples, list) and all(
            isinstance(item, (int, float)) for item in samples
        ):
            layout["step_ms"] = [float(item) for item in samples]
        if layout["first_token_ms"] is None and isinstance(step.get("first_token_ms"), (int, float)):
            layout["first_token_ms"] = float(step["first_token_ms"])
        if layout["first_token_ms"] is None and isinstance(doc.get("first_token_ms"), (int, float)):
            layout["first_token_ms"] = float(doc["first_token_ms"])

        layout["tokens"] = tokens
        layout["token_key"] = token_key
        if tokens and expect_vocab is not None:
            for token in tokens:
                if not 0 <= token < expect_vocab:
                    raise AcceptError(
                        f"{path}: token id {token} outside the expected vocabulary {expect_vocab}"
                    )
        if layout["kind"] == "unknown" and not tokens:
            layout["notes"].append("unrecognised evidence layout; no tokens and no pre-token fields read")
        evidence.append(layout)
    return evidence


def token_stream(evidence: list) -> dict:
    """One side's token stream: rank consistency + the token list (NO position invention)."""
    with_tokens = [entry for entry in evidence if entry["tokens"]]
    empty = [entry for entry in evidence if not entry["tokens"]]
    if not with_tokens:
        return {
            "tokens": [],
            "positions": None,
            "ranks": sorted(entry["rank"] for entry in evidence if entry["rank"] is not None),
            "consistent": None,
            "per_rank": {},
            "not_attempted": True,
            "reason": "no token ids in any evidence file",
        }
    by_rank: dict = {}
    for entry in with_tokens:
        by_rank[str(entry["rank"])] = entry["tokens"]
    sequences = list(by_rank.values())
    consistent = all(sequence == sequences[0] for sequence in sequences)
    positions = None
    positional = [entry for entry in with_tokens if entry["positions"] is not None]
    if positional and len({tuple(entry["positions"]) for entry in positional}) == 1:
        positions = list(positional[0]["positions"])
    return {
        "tokens": sequences[0],
        "positions": positions,
        "ranks": sorted(rank for rank in (entry["rank"] for entry in with_tokens) if rank is not None),
        "consistent": consistent,
        "per_rank": by_rank,
        "not_attempted": False,
        "reason": None,
        "empty_evidence": [entry["path"] for entry in empty],
    }


def compare_streams(device: dict, twin: dict, twin_positions_min: int) -> dict:
    base = {
        "device_positions_run": len(device["tokens"]),
        "twin_positions_run": len(twin["tokens"]),
        "mode": None,
        "per_position": [],
        "mismatches": 0,
        "positions_compared": 0,
        "device_rank_consistency": device["consistent"],
        "twin_rank_consistency": twin["consistent"],
    }
    if device["not_attempted"]:
        return {**base, "verdict": "NOT_ATTEMPTED", "reasons": ["device evidence carries no token ids (the arm did not emit a token)"]}
    if twin["not_attempted"]:
        return {**base, "verdict": "UNMEASURED", "reasons": ["no host-twin token evidence was supplied (--twin)"]}
    device_positions = device["positions"]
    twin_positions = twin["positions"]
    if device_positions is not None and twin_positions is not None:
        mode = "positions"
        twin_by_position = dict(zip(twin_positions, twin["tokens"]))
        pairs = [
            (position, token, twin_by_position[position])
            for position, token in zip(device_positions, device["tokens"])
            if position in twin_by_position
        ]
    else:
        mode = "index"
        pairs = [
            (index, token, twin["tokens"][index])
            for index, token in enumerate(device["tokens"])
            if index < len(twin["tokens"])
        ]
    if not pairs:
        return {**base, "mode": mode, "verdict": "UNMEASURED", "reasons": ["the two streams share no comparable position"]}
    rows = [
        {"position": position, "device": device_token, "twin": twin_token, "match": device_token == twin_token}
        for position, device_token, twin_token in pairs
    ]
    mismatches = sum(1 for row in rows if not row["match"])
    compared = len(rows)
    failures = []
    if device["consistent"] is False:
        failures.append("device ranks disagree on their token ids")
    if twin["consistent"] is False:
        failures.append("twin ranks disagree on their token ids")
    if mismatches:
        failures.append(f"{mismatches} of {compared} compared position(s) disagree")
    if compared < twin_positions_min:
        failures.append(f"only {compared} compared position(s), below min {twin_positions_min}")
    return {
        **base,
        "mode": mode,
        "per_position": rows,
        "mismatches": mismatches,
        "positions_compared": compared,
        "verdict": "PASS" if not failures else "FAIL",
        "reasons": failures,
    }


def dump_report(dump_path, bin_path, expect_vocab, expected_positions) -> dict:
    if not dump_path:
        return {"present": False, "checks": [], "rows": [], "positions": []}
    bin_path = bin_path or (dump_path + ".bin")
    checks: list = []
    rows = compare.read_dump(dump_path)
    compare.check_dump_sidecar(rows, dump_path, bin_path)
    if expect_vocab is not None:
        for row in rows:
            if int(row["vocab"]) != expect_vocab:
                raise compare.CompareError(
                    f"position {row['position']}: vocab {row['vocab']} is not the expected {expect_vocab}"
                )
    positions = sorted(row["position"] for row in rows)
    checks.append({"name": "dump/sidecar tile", "status": "ok", "detail": f"{len(rows)} rows, sidecar {bin_path}"})
    checks.append({"name": "vocab", "status": "ok", "detail": f"all rows vocab {expect_vocab}" if expect_vocab else "not declared"})
    if expected_positions:
        missing = sorted(set(expected_positions) - set(positions))
        checks.append(
            {
                "name": "pinned positions present",
                "status": "fail" if missing else "ok",
                "detail": f"missing {missing}" if missing else f"positions {positions[0]}..{positions[-1]}",
            }
        )
    return {
        "present": True,
        "path": os.path.abspath(dump_path),
        "sha256": sha256_file(dump_path),
        "bin_path": os.path.abspath(bin_path),
        "bin_sha256": sha256_file(bin_path),
        "rows": rows,
        "positions": positions,
        "checks": checks,
    }


def pre_token_checks(device: list) -> dict:
    """Bind-coverage and collective-walk checks, for a window that emits no token yet."""
    checks: list = []
    plan_only_bind: list = []
    for entry in device:
        rank = entry["rank"]
        attention = entry["attention"] or {}
        collectives = entry["collectives"] or {}
        bound = attention.get("tensors_bound", collectives.get("attention_tensors_bound"))
        expected = attention.get("tensors_expected", collectives.get("attention_tensors_expected"))
        plan_only = entry.get("plan_only") is True
        if entry["kind"] == "tp4-bind":
            pass  # the progressive bind file is checked as a whole below
        elif _is_int(bound) and _is_int(expected):
            if plan_only and bound == 0:
                plan_only_bind.append(rank)
                checks.append(
                    {
                        "name": f"r{rank}: bind coverage",
                        "status": "info",
                        "detail": f"{bound}/{expected} bound: plan-only run, binding was not attempted",
                    }
                )
            else:
                checks.append(
                    {
                        "name": f"r{rank}: bind coverage",
                        "status": "ok" if bound == expected and expected > 0 else "fail",
                        "detail": f"{bound}/{expected} device-served names bound",
                    }
                )
        elif entry["kind"] == "tp4-arm":
            checks.append({"name": f"r{rank}: bind coverage", "status": "info", "detail": "no bound/expected counts in evidence"})
        step = entry["step"] or {}
        layers = step.get("layers_run") if _is_int(step.get("layers_run")) else None
        if layers is None and _is_int(collectives.get("layers")):
            layers = int(collectives["layers"])
        if layers is None and isinstance(entry.get("per_layer"), list) and entry["per_layer"]:
            layers = len(entry["per_layer"])
        steps = None
        step_wall = step.get("step_ms_samples")
        if isinstance(step_wall, list) and step_wall:
            steps = len(step_wall)
        calls = None
        expected_calls = None
        walk_steps = collectives.get("steps")
        calls_per_step = collectives.get("calls_per_step_expected")
        if isinstance(walk_steps, list) and walk_steps:
            steps = len(walk_steps)
            calls = sum(int(row.get("calls", 0)) for row in walk_steps if _is_int(row.get("calls")))
            if _is_int(calls_per_step):
                expected_calls = int(calls_per_step) * steps
            elif layers is not None:
                expected_calls = 2 * layers * steps
            broken = [row.get("step") for row in walk_steps if row.get("calls_per_step_ok") is False]
            if broken:
                checks.append(
                    {"name": f"r{rank}: per-step call assertion", "status": "fail", "detail": f"steps {broken} did not assert 2 per layer"}
                )
        else:
            per_step = collectives.get("per_step_calls")
            if isinstance(per_step, list) and per_step:
                calls = sum(value for value in per_step if _is_int(value))
                steps = steps or len(per_step)
                if layers is not None:
                    expected_calls = 2 * layers * (steps or 1)
            elif _is_int(collectives.get("calls")):
                calls = int(collectives["calls"])
                if layers is not None:
                    expected_calls = 2 * layers * (steps or 1)
        if calls is not None and expected_calls is not None:
            checks.append(
                {
                    "name": f"r{rank}: collectives 2 per layer per step",
                    "status": "ok" if calls == expected_calls else "fail",
                    "detail": f"{calls} calls, expected {expected_calls} ({layers} layers x {steps} step(s) x 2)" if layers is not None else f"{calls} calls, expected {expected_calls}",
                }
            )
        elif entry["kind"] == "tp4-arm" and collectives and (calls is not None or collectives.get("performed")):
            checks.append(
                {
                    "name": f"r{rank}: collectives 2 per layer per step",
                    "status": "fail",
                    "detail": "calls present but no layer/step reconciliation fields: the 2-per-layer rule cannot be proven",
                }
            )
        rule = collectives.get("canonical_bf16_check")
        if rule is not None:
            if isinstance(rule, bool):
                ok = rule
                detail = f"canonical_bf16_check={rule}"
            elif isinstance(rule, dict):
                mismatches = rule.get("mismatches")
                ok = mismatches == 0 if _is_int(mismatches) else bool(rule.get("ok", False))
                detail = json.dumps(rule)[:200]
            else:
                ok = False
                detail = repr(rule)[:200]
            checks.append({"name": f"r{rank}: canonical bf16 reduce rule", "status": "ok" if ok else "fail", "detail": detail})
        elif isinstance(collectives.get("canonical_rule"), str):
            checks.append(
                {"name": f"r{rank}: canonical bf16 reduce rule", "status": "info", "detail": collectives["canonical_rule"][:200]}
            )
    # The progressive bind file (written before the walk) is the primary evidence: check it whole.
    bind_by_rank: dict = {}
    for entry in device:
        if entry["kind"] == "tp4-bind" and entry["bind"] is not None and entry["rank"] is not None:
            bind_by_rank.setdefault(entry["rank"], entry["bind"])
    for entry in device:
        if entry["bind"] is not None and entry["rank"] is not None and entry["rank"] not in bind_by_rank:
            bind_by_rank[entry["rank"]] = entry["bind"]
    bind_file_ranks = {
        entry["rank"] for entry in device if entry["kind"] == "tp4-bind" and entry["rank"] is not None and entry["bind"] is not None
    }
    for rank in sorted(bind_by_rank):
        bind = bind_by_rank[rank]
        prefix = f"r{rank}: bind"
        if rank not in bind_file_ranks:
            checks.append(
                {
                    "name": f"{prefix} source",
                    "status": "info",
                    "detail": "bind facts were read from the bind object embedded in tp4-loop-rR.json because no primary tp4-bind-rR.json was found; shape reports the missing primary file",
                }
            )
        if bind["performed"] is False:
            checks.append({"name": f"{prefix} performed", "status": "info", "detail": "bind not performed (plan-only evidence)"})
            continue
        if bind["error"]:
            checks.append({"name": f"{prefix} error", "status": "fail", "detail": bind["error"][:300]})
        attention = bind["attention"]
        if attention["complete"] is True:
            checks.append(
                {"name": f"{prefix} attention coverage", "status": "ok", "detail": f"{attention['bound']}/{attention['expected']} device-served names bound, missing 0"}
            )
        elif attention["complete"] is False:
            checks.append(
                {
                    "name": f"{prefix} attention coverage",
                    "status": "fail",
                    "detail": f"{attention['bound']}/{attention['expected']} bound, missing {len(attention['missing'])} ({attention['missing'][:8]})",
                }
            )
        moe = bind["moe"]
        if moe["performed"]:
            incomplete = [row["class"] for row in moe["classes"] if row["complete"] is False]
            checks.append(
                {
                    "name": f"{prefix} MoE/shared/router coverage",
                    "status": "ok" if moe["complete"] is True else "fail",
                    "detail": f"{moe['bound']}/{moe['expected']} names across {len(moe['classes'])} class(es)" + (f"; incomplete {incomplete}" if incomplete else ""),
                }
            )
            if moe["weight_h2d_bytes_per_token"] is not None:
                checks.append(
                    {
                        "name": f"{prefix} per-token weight H2D",
                        "status": "ok" if moe["weight_h2d_bytes_per_token"] == 0 else "fail",
                        "detail": f"{moe['weight_h2d_bytes_per_token']} bytes/token",
                    }
                )
            if moe["launch_incompatible"]:
                checks.append(
                    {
                        "name": f"{prefix} MoE launch compatibility",
                        "status": "info",
                        "detail": "; ".join(f"{row['class']}: {row['reason'][:160] if row['reason'] else 'incompatible'}" for row in moe["launch_incompatible"]),
                    }
                )
        elif bind["summary"]["complete"] is False:
            checks.append({"name": f"{prefix} MoE/shared/router coverage", "status": "fail", "detail": "MoE bind did not complete"})
        tail = bind["tail"]
        if tail["performed"]:
            checks.append(
                {
                    "name": f"{prefix} tail coverage",
                    "status": "ok" if tail["complete"] is True else ("fail" if tail["complete"] is False else "info"),
                    "detail": f"{tail['bound']}/{tail['expected']} tail names bound" if tail["expected"] is not None else "tail bind recorded",
                }
            )
            if tail["pointers_nonzero"] is False:
                checks.append({"name": f"{prefix} tail pointers", "status": "fail", "detail": "a tail pointer is null"})
        census = bind["census"]
        if census["performed"]:
            unclassified = census["unclassified"] or []
            unexpected = sorted(name for name in unclassified if name not in EXPECTED_CENSUS_UNCLASSIFIED)
            checks.append(
                {
                    "name": f"{prefix} census",
                    "status": "fail" if unexpected else "ok",
                    "detail": f"{census['resident_tensors']} resident shards; unclassified {len(unclassified)}"
                    + (f" = {sorted(unclassified)} (expected multi-modal special-token embeddings)" if not unexpected else f", unexpected {unexpected}"),
                }
            )
        summary = bind["summary"]
        if summary["complete"] is not None:
            unclassified = sorted(census["unclassified"] or []) if census["performed"] else []
            expected_census = census["performed"] and unclassified == sorted(EXPECTED_CENSUS_UNCLASSIFIED)
            clean_sets = (
                not bind["error"]
                and attention["complete"] is True
                and (moe["complete"] is True or moe["performed"] is False)
                and (tail["complete"] is True or tail["performed"] is False)
            )
            if summary["complete"]:
                status = "ok"
                detail = f"summary.complete=True, device_bound_names={summary['device_bound_names']}, per_token_weight_h2d_bytes={summary['per_token_weight_h2d_bytes']}"
            elif expected_census and clean_sets:
                status = "info"
                detail = (
                    "summary.complete=False by construction: census.unclassified == "
                    f"{unclassified} (three resident multi-modal token embeddings no device class claims), "
                    "while attention/MoE/tail sets are complete and error is null -> bind-set result clean, census caveat recorded"
                )
            else:
                status = "fail"
                detail = f"summary.complete=False, device_bound_names={summary['device_bound_names']}, per_token_weight_h2d_bytes={summary['per_token_weight_h2d_bytes']}"
            checks.append({"name": f"{prefix} complete", "status": status, "detail": detail})
            if status == "info":
                checks.append(
                    {
                        "name": f"{prefix} census caveat",
                        "status": "info",
                        "detail": f"census.unclassified == {unclassified} exactly; summary.complete=false is the bind_evidence census AND, not a bind-set defect",
                    }
                )

    loop_plan_only = {
        entry["rank"]: entry.get("plan_only")
        for entry in device
        if entry["kind"] == "tp4-arm" and isinstance(entry["schema"], str) and entry["schema"].startswith("sparkflash-tp4-loop/") and entry["rank"] is not None
    }
    missing_primary = [
        rank for rank, plan_only in sorted(loop_plan_only.items()) if rank not in bind_file_ranks and plan_only is not True
    ]
    if missing_primary:
        checks.append(
            {
                "name": "primary bind file present",
                "status": "fail",
                "detail": f"missing tp4-bind-rR.json for ranks {missing_primary}; the bind facts fall back to the embedded tp4-loop object, which is not the primary window evidence",
            }
        )

    plan_rows = {}
    for entry in device:
        plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else None
        if plan is not None and entry["rank"] is not None:
            plan_rows[entry["rank"]] = plan
    for rank in sorted(plan_rows):
        plan = plan_rows[rank]
        checks.append(
            {
                "name": f"r{rank}: plan identity",
                "status": "info",
                "detail": f"visible GPU/plan hash {plan.get('plan_hash')}, {plan.get('resident_tensors')} resident shards, {plan.get('resident_bytes')} resident bytes (from tp4-loop-rR.json)",
            }
        )

    failures = [check for check in checks if check["status"] == "fail"]
    attempted = any(check["status"] == "ok" for check in checks)
    bind_rows = [entry for entry in device if entry["attention"] and _is_int(entry["attention"].get("tensors_expected"))]
    if failures:
        verdict = "FAIL"
    elif not checks:
        verdict = "NO-EVIDENCE"
    elif plan_only_bind and len(plan_only_bind) == len(bind_rows):
        verdict = "NOT_ATTEMPTED"
    elif not attempted:
        verdict = "NOT_ATTEMPTED"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "checks": checks,
        "failures": failures,
        "bind_ranks": sorted(bind_by_rank),
    }


def oracle_distributional(args, dump: dict) -> dict:
    oracle = compare.load_oracle(args.oracle)
    if not compare.root_matches(oracle.get("served", {}).get("root"), args.expect_root):
        raise compare.CompareError(
            f"oracle served root {oracle.get('served', {}).get('root')!r} is not the released snapshot "
            f"{args.expect_root!r} (D38)"
        )
    prompt = compare.select_oracle_prompt(oracle, args.oracle_prompt)
    oracle_positions = {int(row["position"]): row for row in prompt["positions"]}
    rows = dump["rows"]
    wanted = compare.parse_positions(args.positions) if args.positions else None
    if wanted is not None:
        missing_dump = sorted(wanted - {row["position"] for row in rows})
        if missing_dump:
            raise compare.CompareError(f"--positions names {missing_dump} but the dump has no such position(s)")
        missing_oracle = sorted(wanted - set(oracle_positions))
        if missing_oracle:
            raise compare.CompareError(f"--positions names {missing_oracle} but the oracle has no such position(s)")
    compared = (
        [row for row in rows if row["position"] in wanted]
        if wanted is not None
        else [row for row in rows if row["position"] in oracle_positions]
    )
    if not compared:
        raise compare.CompareError("no overlapping positions between the device dump and the oracle")
    compared.sort(key=lambda row: row["position"])
    results = []
    for row in compared:
        logits = compare.read_logits(dump["bin_path"], int(row["logits_offset"]), int(row["vocab"]))
        results.append(compare.position_metrics(row, logits, oracle_positions[row["position"]]))
    aggregate = compare.aggregate_metrics(results)
    bars = compare.load_json(args.oracle_bars) if args.oracle_bars else None
    if bars is not None:
        compare.check_gate_bars(bars, args.oracle_bars)
    bar_result = compare.evaluate_bars(bars, aggregate) if bars else None
    verdict = "UNBARED" if bar_result is None else ("PASS" if bar_result["pass"] else "FAIL")
    if aggregate["prefix_mismatches"] > 0:
        verdict = "NOT_LIKE_FOR_LIKE"
    provisional = None
    if args.provisional_d40:
        provisional = {
            "label": "D40-PROVISIONAL-NOT-A-GATE",
            "evaluation": compare.evaluate_bars({"bars": D40_PROVISIONAL_BARS}, aggregate),
            "note": "published D40 provisional values only; no D41 gate file exists, so this is a diagnostic, not acceptance",
        }
    return {
        "verdict": verdict,
        "label": "FULL-DUMP-D39-METRICS",
        "mode": "distributional",
        "prompt_id": prompt.get("id"),
        "oracle_path": os.path.abspath(args.oracle),
        "oracle_sha256": sha256_file(args.oracle),
        "oracle_served_root": oracle.get("served", {}).get("root"),
        "oracle_response_sha256": prompt.get("response_sha256"),
        "positions": results,
        "aggregate": aggregate,
        "bars_path": os.path.abspath(args.oracle_bars) if args.oracle_bars else None,
        "bars": bar_result,
        "provisional_d40": provisional,
        "reasons": [],
    }


def oracle_token_only(args, device: dict) -> dict:
    oracle = compare.load_oracle(args.oracle)
    if not compare.root_matches(oracle.get("served", {}).get("root"), args.expect_root):
        raise compare.CompareError(
            f"oracle served root {oracle.get('served', {}).get('root')!r} is not the released snapshot "
            f"{args.expect_root!r} (D38)"
        )
    prompt = compare.select_oracle_prompt(oracle, args.oracle_prompt)
    oracle_positions = {int(row["position"]): row for row in prompt["positions"]}
    if device["positions"] is None:
        return {
            "verdict": "UNMEASURED",
            "label": "ORACLE-TOKEN-ONLY",
            "mode": "token-only",
            "reasons": [
                "device evidence has no explicit positions, so a token cannot be joined to an oracle position; nothing is guessed"
            ],
            "prompt_id": prompt.get("id"),
            "oracle_path": os.path.abspath(args.oracle),
            "oracle_sha256": sha256_file(args.oracle),
            "oracle_served_root": oracle.get("served", {}).get("root"),
        }
    rows = []
    for position, token in zip(device["positions"], device["tokens"]):
        oracle_position = oracle_positions.get(position)
        if oracle_position is None:
            continue
        ranked = compare._ranked(
            {int(key): float(value) for key, value in oracle_position["logprobs"].items()}
        )
        argmax = ranked[0] if ranked else None
        rows.append(
            {
                "position": position,
                "device_token": token,
                "oracle_argmax": argmax,
                "top1_hit": token == argmax,
                "oracle_target_token": oracle_position.get("target_token"),
                "target_hit": token == oracle_position.get("target_token"),
            }
        )
    if not rows:
        return {
            "verdict": "UNMEASURED",
            "label": "ORACLE-TOKEN-ONLY",
            "mode": "token-only",
            "reasons": ["device positions do not intersect the oracle positions"],
            "prompt_id": prompt.get("id"),
            "oracle_path": os.path.abspath(args.oracle),
            "oracle_sha256": sha256_file(args.oracle),
            "oracle_served_root": oracle.get("served", {}).get("root"),
        }
    hits = sum(1 for row in rows if row["top1_hit"])
    provisional = None
    if args.provisional_d40:
        provisional = {
            "label": "D40-PROVISIONAL-NOT-A-GATE",
            "top1_agreement_rate": hits / len(rows),
            "positions": len(rows),
            "note": "token-only top-1 hit rate against the oracle argmax; D40 also bars distributional metrics that need a logits dump, so this can never be a gate pass",
        }
    return {
        "verdict": "UNBARED",
        "label": "ORACLE-TOKEN-ONLY (no device logits dump; distributional bars cannot be evaluated)",
        "mode": "token-only",
        "prompt_id": prompt.get("id"),
        "oracle_path": os.path.abspath(args.oracle),
        "oracle_sha256": sha256_file(args.oracle),
        "oracle_served_root": oracle.get("served", {}).get("root"),
        "positions": rows,
        "positions_compared": len(rows),
        "top1_hit_rate": hits / len(rows),
        "provisional_d40": provisional,
        "reasons": ["no device logits dump: UNBARED by construction (D41 bars would still be required)"],
    }


def step_time_report(device_evidence: list) -> dict:
    samples: list = []
    per_rank = []
    first_token_ms = None
    for entry in device_evidence:
        if entry["step_ms"]:
            per_rank.append({"rank": entry["rank"], "step_ms": entry["step_ms"], "path": entry["path"]})
            samples.extend(entry["step_ms"])
        if first_token_ms is None and entry["first_token_ms"] is not None:
            first_token_ms = entry["first_token_ms"]
    if not samples:
        return {
            "verdict": "UNMEASURED",
            "label": "STEP-TIME-UNMEASURED",
            "reason": "no step_ms samples in the device evidence (the arm did not run/record a step)",
            "target_only": {
                "label": STEP_MS_TARGET_ONLY_LABEL,
                "step_budget_ms": D48_STEP_BUDGET_MS,
                "accepted_tokens_per_step": D48_ACCEPTED_TOKENS_PER_STEP,
                "tok_s_target": D48_TOK_S_TARGET,
                "parity_band_tok_s": list(D48_PARITY_TOK_S),
            },
        }
    mean = sum(samples) / len(samples)
    worst_rank = None
    for row in per_rank:
        row_mean = sum(row["step_ms"]) / len(row["step_ms"])
        row["mean_step_ms"] = row_mean
        if worst_rank is None or row_mean > worst_rank["mean_step_ms"]:
            worst_rank = row
    worst_mean = worst_rank["mean_step_ms"] if worst_rank else mean
    return {
        "verdict": "MEASURED",
        "label": STEP_MS_TARGET_ONLY_LABEL,
        "steps_samples": len(samples),
        "step_ms_mean": mean,
        "step_ms_worst_rank": worst_rank,
        "first_token_ms": first_token_ms,
        "per_rank": per_rank,
        "single_token_tokens_per_second": 1000.0 / worst_mean if worst_mean else None,
        "target_only": {
            "label": STEP_MS_TARGET_ONLY_LABEL,
            "step_budget_ms": D48_STEP_BUDGET_MS,
            "step_budget_pass_as_measured": worst_mean <= D48_STEP_BUDGET_MS,
            "accepted_tokens_per_step": D48_ACCEPTED_TOKENS_PER_STEP,
            "tok_s_target": D48_TOK_S_TARGET,
            "parity_band_tok_s": list(D48_PARITY_TOK_S),
            "note": "the accepted-tokens/step and 90 tok/s parts need the real DSpark/MTP verify step and are not measured here",
        },
    }


def parse_expected_moe_bounds(raw: str) -> dict:
    expected: dict = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise AcceptError(f"--expect-moe-bounds {part!r}: expected class=count")
        name, value = part.split("=", 1)
        try:
            expected[name.strip()] = int(value)
        except ValueError as error:
            raise AcceptError(f"--expect-moe-bounds {part!r}: {error}") from error
    if not expected:
        raise AcceptError("--expect-moe-bounds was empty")
    return expected


def count_lines(path: str) -> int:
    lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            lines += 1
    return lines


# device-lead's success path emits progress lines only, so the failure rule is deliberately
# narrow and case-sensitive: a line that starts with error/panicked/fatal, a Rust panic header, or
# the word mismatch. Broader error/conflated words are reported as suspicious but do not fail by
# themselves (run.exit and the JSON checks decide); this avoids flagging benign future wording.
LOG_DEFECT = re.compile(r"^(error|panicked|fatal)|thread .* panicked|mismatch")
LOG_SUSPICIOUS = re.compile(r"(?i)(error|fatal|panic)")


def scan_log(path: str, max_bytes: int = 8 << 20):
    defects: list = []
    suspicious: list = []
    read = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            read += len(line)
            if read > max_bytes:
                break
            if LOG_DEFECT.search(line):
                defects.append({"line": number, "text": line.strip()[:300]})
                if len(defects) >= 10:
                    break
            elif LOG_SUSPICIOUS.search(line) and len(suspicious) < 5:
                suspicious.append({"line": number, "text": line.strip()[:300]})
    return defects, suspicious


def _rank_of_rc_path(path: str):
    match = re.search(r"rank(\d+)", os.path.basename(os.path.dirname(path)) or "")
    if match:
        return int(match.group(1))
    match = re.search(r"-r(\d+)\.[^.]+$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _refusal_forward(directory: str, rank):
    """The labelled negative: tp4-token-rR.json says forward=refused/positions_run=0 with a named error."""
    if rank is None:
        return None
    path = os.path.join(directory, f"tp4-token-r{rank}.json")
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


def build_shape(evidence: list, twin: list, dump: dict, expected_positions, args=None) -> dict:
    checks: list = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add("device evidence files", "ok" if evidence else "fail", f"{len(evidence)} file(s)")
    unknown = [entry["path"] for entry in evidence if entry["kind"] == "unknown"]
    add("recognised layouts", "ok" if not unknown else "fail", f"unknown: {unknown}" if unknown else "all device evidence layouts recognised")
    aux = [entry["path"] for entry in evidence if entry["kind"] == "aux"]
    if aux:
        add(
            "auxiliary evidence ignored",
            "info",
            f"{len(aux)} non-acceptance file(s) in the evidence directory (resident-stage/summary documents): {aux}",
        )
    ranks = [entry["rank"] for entry in evidence if entry["rank"] is not None]
    # A plan-only preflight document (tp4-resident, world 1) may share rank0/ as an auxiliary file;
    # the run's world identity comes from acceptance evidence only.
    worlds = [
        entry["world"]
        for entry in evidence
        if entry["world"] is not None and entry["kind"] != "aux"
    ]
    # A window run legitimately carries two documents per rank (tp4-bind-rR.json and
    # tp4-loop-rR.json), so uniqueness is checked per evidence kind, not across the union.
    if ranks:
        per_kind: dict = {}
        for entry in evidence:
            if entry["rank"] is not None:
                per_kind.setdefault(entry["kind"], []).append(entry["rank"])
        duplicates = {kind: sorted(values) for kind, values in per_kind.items() if len(set(values)) != len(values)}
        if duplicates:
            add("unique ranks", "fail", f"duplicate rank in evidence: {duplicates}")
        else:
            add("unique ranks", "ok", f"ranks {sorted(set(ranks))}")
    # The progressive bind file is the primary window evidence: it is written before the walk by
    # design, so a loop document without it is a loud failure, not an absent measurement.
    loop_entries = [entry for entry in evidence if entry["kind"] == "tp4-arm" and isinstance(entry["schema"], str) and entry["schema"].startswith("sparkflash-tp4-loop/")]
    bind_files = {entry["rank"]: entry for entry in evidence if entry["kind"] == "tp4-bind" and entry["rank"] is not None}
    if loop_entries:
        plan_only_all = all(entry.get("plan_only") is True for entry in loop_entries)
        for entry in sorted(loop_entries, key=lambda item: (item["rank"] is None, item["rank"])):
            rank = entry["rank"]
            directory = os.path.dirname(entry["path"])
            bind_path = os.path.join(directory, f"tp4-bind-r{rank}.json")
            if rank in bind_files:
                add(f"r{rank}: primary bind evidence", "ok", bind_files[rank]["path"])
                manifest_path = os.path.join(directory, f"tp4-bind-names-r{rank}.txt")
                add(
                    f"r{rank}: bind names manifest",
                    "ok" if os.path.exists(manifest_path) else "info",
                    manifest_path if os.path.exists(manifest_path) else f"no names manifest at {manifest_path} (bind coverage still complete in the JSON)",
                )
            elif plan_only_all:
                add(f"r{rank}: primary bind evidence", "info", "plan-only run: the arm does not write tp4-bind-rR.json without --upload")
            else:
                add(
                    f"r{rank}: primary bind evidence",
                    "fail",
                    f"missing {bind_path}: the arm writes tp4-bind-rR.json before the walk by design, so its absence is a failed bind stage, not an absent measurement",
                )
    if worlds:
        add("world size", "ok" if len(set(worlds)) == 1 else "fail", f"world {sorted(set(worlds))}")
    token_entries = [entry for entry in evidence if entry["tokens"]]
    if token_entries:
        add("token ids present", "ok", f"{len(token_entries)} rank file(s) with tokens, key {token_entries[0]['token_key']}")
        tokens = token_entries[0]["tokens"]
        if all(_is_int(token) and token >= 0 for token in tokens):
            add("token range", "ok", f"{len(tokens)} token(s) >= 0")
        else:
            add("token range", "fail", "non-integer or negative token id")
    else:
        add("token ids present", "info", "no token ids (the arm did not emit a token)")
    token_envelopes = [entry for entry in evidence if entry["kind"] == "tp4-token"]
    if args is not None and getattr(args, "require_token", False) and not token_envelopes:
        add(
            "token evidence present",
            "fail",
            "no sparkflash-tp4-token/1 evidence in the tree; --require-token was set, so a missing token stage is a failure (the forward run did not publish a token JSON)",
        )
    if token_envelopes:
        ranks = sorted(entry["rank"] for entry in token_envelopes if entry["rank"] is not None)
        add(
            "token evidence on every rank",
            "ok" if len(set(ranks)) == len(ranks) and all(entry["rank"] is not None for entry in token_envelopes) else "fail",
            f"ranks {ranks}",
        )
        for entry in token_envelopes:
            prefix = f"r{entry['rank']}: token"
            forward = entry["forward"]
            if forward == "refused":
                add(f"{prefix} forward refused", "info", entry["token_error"] or "forward refused without an error string (report it as a finding)")
                continue
            if forward not in ("run", "not-run", None):
                add(f"{prefix} forward label", "fail", f"unknown forward value {forward!r}")
                continue
            if forward == "run" and not entry["tokens"]:
                add(f"{prefix} forward run without tokens", "fail", "forward:run but tokens.ids is empty")
                continue
            if not entry["tokens"]:
                add(f"{prefix} not run", "info", "forward not run (no token ids in this evidence file)")
                continue
            problems: list = []
            if entry["token_error"]:
                problems.append(f"error={entry['token_error']}")
            if entry["positions_run"] is not None and entry["positions_run"] != len(entry["tokens"]):
                problems.append(f"positions_run {entry['positions_run']} != tokens.ids {len(entry['tokens'])}")
            if entry["positions"] is not None and len(entry["positions"]) != len(entry["tokens"]):
                problems.append(f"tokens.positions length {len(entry['positions'])} != tokens.ids {len(entry['tokens'])}")
            if entry["positions"] is not None and entry["positions"] != sorted(entry["positions"]):
                problems.append("tokens.positions not strictly increasing")
            reference = entry["logits_dump"] if isinstance(entry["logits_dump"], dict) else None
            if reference is None:
                # Rank 0 is the primary dump owner. A replica rank may omit the dump entirely;
                # the contract only requires that the run published at least one dump (checked
                # once after this loop).
                if entry["rank"] in (0, None):
                    problems.append("forward:run without a logits_dump reference on the primary (rank 0)")
            elif dump["present"]:
                resolved = reference.get("path")
                base = os.path.dirname(entry["path"])
                if isinstance(resolved, str) and not os.path.isabs(resolved):
                    resolved = os.path.join(base, resolved)
                is_primary = isinstance(resolved, str) and os.path.abspath(resolved) == os.path.abspath(dump["path"])
                if is_primary:
                    if reference.get("sha256") and reference["sha256"] != dump["sha256"]:
                        problems.append(f"logits_dump.sha256 {reference.get('sha256')[:12]}... != dumped file {dump['sha256'][:12]}...")
                    if reference.get("bin_sha256") and reference["bin_sha256"] != dump["bin_sha256"]:
                        problems.append("logits_dump.bin_sha256 does not match the sidecar")
                    declared_rows = reference.get("rows")
                    if _is_int(declared_rows) and declared_rows != len(dump["rows"]):
                        problems.append(f"logits_dump.rows {declared_rows} != dumped rows {len(dump['rows'])}")
                    declared_positions = _int_list(reference.get("positions"))
                    if declared_positions is not None and declared_positions != dump["positions"]:
                        problems.append("logits_dump.positions does not match the dumped rows")
                elif reference.get("bin_sha256") and reference["bin_sha256"] != dump["bin_sha256"]:
                    # A replica rank's JSONL is allowed to differ only in its own logits_bin
                    # basename; the sidecar bytes must still be identical.
                    problems.append("replica logits_dump.bin_sha256 does not match the primary sidecar")
                if reference.get("vocab") is not None and args is not None and args.expect_vocab is not None and int(reference["vocab"]) != args.expect_vocab:
                    problems.append(f"logits_dump.vocab {reference['vocab']} != {args.expect_vocab}")
            add(f"{prefix} envelope", "ok" if not problems else "fail", "consistent" if not problems else "; ".join(problems))
        envelope_dumps = [
            entry["logits_dump"]
            for entry in token_envelopes
            if isinstance(entry["logits_dump"], dict)
        ]
        if not envelope_dumps and any(entry["forward"] == "run" and entry["tokens"] for entry in token_envelopes):
            add(
                "token logits dump present",
                "fail",
                "forward:run with token ids but no rank published a logits_dump reference; the distributional comparison cannot run",
            )
        elif envelope_dumps:
            add("token logits dump present", "ok", f"{len(envelope_dumps)} rank(s) reference a dump")
        if len(envelope_dumps) > 1:
            bin_shas = [reference["bin_sha256"] for reference in envelope_dumps if reference.get("bin_sha256")]
            if len(bin_shas) > 1:
                distinct = set(bin_shas)
                add(
                    "rank logits sidecars identical",
                    "ok" if len(distinct) == 1 else "fail",
                    f"{len(distinct)} distinct bin_sha256 across {len(bin_shas)} rank(s)",
                )
            else:
                jsonl_shas = [reference["sha256"] for reference in envelope_dumps if reference.get("sha256")]
                distinct = set(jsonl_shas)
                add(
                    "rank logits dumps identical",
                    "ok" if len(distinct) == 1 else "info",
                    f"{len(distinct)} distinct dump sha256 across {len(jsonl_shas)} rank(s) (no bin_sha256 to compare)",
                )
        token_ids_by_rank = {entry["rank"]: entry["tokens"] for entry in token_envelopes if entry["tokens"]}
        if len(token_ids_by_rank) > 1:
            sequences = list(token_ids_by_rank.values())
            add(
                "token ids rank-consistent",
                "ok" if all(sequence == sequences[0] for sequence in sequences) else "fail",
                f"{len(token_ids_by_rank)} rank(s) agree" if all(sequence == sequences[0] for sequence in sequences) else "ranks disagree on token ids",
            )
        step_samples = [sample for entry in token_envelopes for sample in (entry["step_ms"] or [])]
        if step_samples:
            add("token step timings", "ok", f"{len(step_samples)} sample(s) across token evidence")
    forward_entries = [entry for entry in evidence if entry["kind"] == "tp4-forward"]
    if forward_entries:
        rows = []
        problems = []
        for entry in sorted(forward_entries, key=lambda item: (item["rank"] is None, item["rank"])):
            info = entry.get("forward_embed") or {}
            rank = entry["rank"]
            rows.append(
                f"r{rank}: stage={info.get('stage')} token_claimed={info.get('token_claimed')} "
                f"prompt_len={info.get('prompt_len')} decode_positions={info.get('decode_positions')} "
                f"attention_launchable={info.get('attention_launchable')} "
                f"refs={len(info.get('reference_ids') or [])}"
            )
            if info.get("stage") != "embedding-gathered":
                problems.append(f"r{rank}: stage {info.get('stage')!r} is not embedding-gathered")
            if info.get("token_claimed") is True:
                problems.append(f"r{rank}: token_claimed true but the document carries no token ids")
        add(
            "forward-embed evidence",
            "ok" if not problems else "fail",
            "; ".join(rows) if not problems else "; ".join(problems),
        )
    bind_file_entries = [entry for entry in evidence if entry["kind"] == "tp4-bind" and entry["bind"] is not None]
    if args is not None and args.expect_attention_bound is not None and bind_file_entries:
        for entry in bind_file_entries:
            attention = entry["bind"]["attention"]
            bound, expected = attention["bound"], attention["expected"]
            ok = bound == args.expect_attention_bound and expected == args.expect_attention_bound and attention["complete"] is True
            add(
                f"r{entry['rank']}: attention bound == {args.expect_attention_bound}",
                "ok" if ok else "fail",
                f"{bound}/{expected} bound, complete={attention['complete']}",
            )
    if args is not None and args.expect_moe_bounds and bind_file_entries:
        for entry in bind_file_entries:
            rows = {row["class"]: row for row in entry["bind"]["moe"]["classes"]}
            problems = [
                {"class": name, "bound": rows[name]["bound"] if name in rows else None, "expected": size}
                for name, size in args.expect_moe_bounds.items()
                if name not in rows or rows[name]["bound"] != size or rows[name]["complete"] is not True
            ]
            add(
                f"r{entry['rank']}: MoE class bound counts",
                "ok" if not problems else "fail",
                "all classes match" if not problems else f"problems: {problems}",
            )
    if args is not None and args.expect_manifest_lines is not None and bind_file_entries:
        for entry in bind_file_entries:
            directory = os.path.dirname(entry["path"])
            manifest = os.path.join(directory, f"tp4-bind-names-r{entry['rank']}.txt")
            if os.path.exists(manifest):
                lines = count_lines(manifest)
                add(
                    f"r{entry['rank']}: bind names manifest lines == {args.expect_manifest_lines}",
                    "ok" if lines == args.expect_manifest_lines else "fail",
                    f"{lines} lines ({manifest})",
                )
            else:
                add(
                    f"r{entry['rank']}: bind names manifest",
                    "fail",
                    f"missing {manifest}: the arm writes it with the bind by design",
                )
    def first_named(directory: str, names: list):
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
        return None

    def first_pattern(directory: str, pattern: str):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return None
        for name in names:
            if re.fullmatch(pattern, name):
                return os.path.join(directory, name)
        return None

    explicit_logs = [path for path in getattr(args, "device_log", []) if path] if args is not None else []
    directories = sorted({os.path.dirname(entry["path"]) for entry in evidence if entry["path"]})
    log_rc: list = [(path, None) for path in explicit_logs]
    if args is not None and not getattr(args, "device_log", []):
        for directory in directories:
            # The Lead's wrapper names these chain-rR.log / chain-rR.rc; accept both conventions.
            log_path = first_named(directory, ["run.log"]) or first_pattern(directory, r"chain-r\d+\.log")
            rc_path = first_named(directory, ["run.exit"]) or first_pattern(directory, r"chain-r\d+\.rc")
            if log_path is not None or rc_path is not None:
                log_rc.append((log_path, rc_path))
    for log_path, rc_path in log_rc:
        if log_path and os.path.exists(log_path):
            defects, suspicious = scan_log(log_path)
            add(
                f"{os.path.basename(os.path.dirname(log_path)) or 'run'}: run log clean",
                "ok" if not defects else "fail",
                f"no defect lines in {os.path.basename(log_path)}" if not defects else f"{len(defects)} defect line(s) in {os.path.basename(log_path)}: {defects[:3]}",
            )
            if suspicious and not defects:
                add(
                    f"{os.path.basename(os.path.dirname(log_path)) or 'run'}: suspicious log wording",
                    "info",
                    f"reported, not failed (rank exit code and JSON checks decide): {suspicious[:3]}",
                )
        if rc_path is None and log_path:
            rc_path = first_named(os.path.dirname(log_path), ["run.exit"]) or first_pattern(os.path.dirname(log_path), r"chain-r\d+\.rc")
        if rc_path and os.path.exists(rc_path):
            with open(rc_path, "r", encoding="utf-8", errors="replace") as handle:
                rc_text = handle.read().strip()
            rc_rank = _rank_of_rc_path(rc_path)
            refusal = _refusal_forward(os.path.dirname(rc_path), rc_rank) if rc_text != "0" else None
            name = f"{os.path.basename(os.path.dirname(rc_path))}: rank exit code"
            if rc_text == "0":
                add(name, "ok", f"{os.path.basename(rc_path)}=0")
            elif rc_text == "2" and refusal is not None:
                add(
                    name,
                    "info",
                    f"{os.path.basename(rc_path)}=2 is the labelled forward refusal "
                    f"(forward=refused, positions_run=0): {refusal['error'][:160]}",
                )
            else:
                add(name, "fail", f"{os.path.basename(rc_path)}={rc_text!r}")
    if dump.get("present"):
        checks.extend(dump["checks"])
    else:
        add("device logits dump", "info", "no --device-dump: the oracle leg runs token-only or is UNMEASURED")
    failures = [check for check in checks if check["status"] == "fail"]
    return {"ok": not failures, "checks": checks, "failures": failures, "positions_run": len(token_entries[0]["tokens"]) if token_entries else 0}


def render_markdown(result: dict) -> str:
    lines = [
        "# Device-token acceptance (D40/D41) - separate verdicts, no blended pass",
        "",
        f"- generated (UTC): {result['generated_utc']}",
        f"- schema: {result['schema']}",
        f"- device evidence: {', '.join(item['path'] for item in result['inputs']['device_evidence']) or 'none'}",
        f"- twin evidence: {', '.join(item['path'] for item in result['inputs']['twin_evidence']) or 'none'}",
        f"- device dump: {result['inputs']['device_dump'] or 'none'}",
        f"- oracle: {result['inputs']['oracle'] or 'none'}",
        f"- oracle bars: {result['inputs']['oracle_bars'] or 'none (UNBARED)'}",
        "",
        "## Verdicts",
        "",
        "| leg | verdict | label / reason |",
        "| :--- | :--- | :--- |",
        f"| token shape | {result['shape']['verdict']} | {'ok' if result['shape']['ok'] else 'see failures'} |",
        f"| pre-token (bind + collectives) | {result['pre_token']['verdict']} | {len(result['pre_token']['checks'])} check(s) |",
        f"| twin match | {result['twin_match']['verdict']} | {result['twin_match'].get('mode') or '-'}; {'; '.join(result['twin_match']['reasons']) or 'no failures'} |",
        f"| oracle comparison | {result['oracle_comparison']['verdict']} | {result['oracle_comparison'].get('label') or '-'} |",
        f"| step time | {result['step_time']['verdict']} | {result['step_time'].get('label')} |",
        f"| **overall** | **{result['overall']['verdict']}** | {'; '.join(result['overall']['reasons']) or 'no failures'} |",
        "",
        "The overall verdict aggregates the legs; it does not turn an UNBARED oracle leg into a pass.",
        "",
    ]
    if result["shape"]["checks"]:
        lines += ["## Shape checks", "", "| check | status | detail |", "| :--- | :--- | :--- |"]
        for check in result["shape"]["checks"]:
            lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
        lines.append("")
    if result["pre_token"]["checks"]:
        lines += ["## Pre-token checks", "", "| check | status | detail |", "| :--- | :--- | :--- |"]
        for check in result["pre_token"]["checks"]:
            lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")
        lines.append("")
    twin = result["twin_match"]
    if twin.get("per_position"):
        lines += ["## Twin match (per position)", "", "| position | device | twin | match |", "| ---: | ---: | ---: | :--- |"]
        for row in twin["per_position"]:
            lines.append(f"| {row['position']} | {row['device']} | {row['twin']} | {'yes' if row['match'] else 'no'} |")
        lines.append("")
    oracle = result["oracle_comparison"]
    if oracle.get("aggregate"):
        aggregate = oracle["aggregate"]
        lines += [
            "## Oracle comparison aggregate (device dump vs released-snapshot oracle)",
            "",
            "| metric | value |",
            "| :--- | ---: |",
            f"| positions | {aggregate['positions']} ({aggregate['position_first']}..{aggregate['position_last']}) |",
            f"| top-1 agreement rate | {aggregate['top1_agreement_rate']} |",
            f"| top-20 overlap mean | {aggregate['top20_overlap_mean']} |",
            f"| KL mean / p95 | {aggregate['kl_mean']} / {aggregate['kl_p95']} |",
            f"| JS mean | {aggregate['js_mean']} |",
            f"| prefix mismatches / unknown | {aggregate['prefix_mismatches']} / {aggregate['prefix_unknown']} |",
            "",
        ]
    lines += [
        "## Non-claims",
        "",
        "- No token-equality claim against the reference/vLLM oracle (D39 rule 2).",
        "- No D41 gate file exists, so the oracle leg is UNBARED: it records metrics, not acceptance.",
        "- A host dump is never accepted as device evidence.",
        "- Step time (if shown) is D48-TARGET-ONLY here; tonight's M1 gate is a token gate.",
    ]
    return "\n".join(lines) + "\n"


def overall_verdict(result: dict) -> dict:
    reasons: list = []
    if result["shape"]["ok"] and result["pre_token"]["verdict"] == "FAIL":
        reasons.extend(
            f"pre-token: {failure['name']} - {failure['detail']}" for failure in result["pre_token"]["failures"]
        )
        return {"verdict": "FAIL", "reasons": reasons}
    if not result["shape"]["ok"]:
        reasons.extend(f"shape: {failure['name']} - {failure['detail']}" for failure in result["shape"]["failures"])
        return {"verdict": "FAIL", "reasons": reasons}
    twin_verdict = result["twin_match"]["verdict"]
    oracle_verdict = result["oracle_comparison"]["verdict"]
    if twin_verdict == "FAIL":
        reasons.append("twin match failed")
    if oracle_verdict == "FAIL":
        reasons.append("oracle comparison failed its declared bars")
    if reasons:
        return {"verdict": "FAIL", "reasons": reasons}
    if twin_verdict == "NOT_ATTEMPTED":
        return {
            "verdict": "NOT_ATTEMPTED",
            "reasons": [
                "no device token was emitted (or its shape is not in evidence); pre-token checks are reported above",
                "M1 token gate is not reached this run",
            ],
        }
    if twin_verdict == "PASS" and oracle_verdict in ("PASS", "UNBARED", "UNMEASURED", "NOT_LIKE_FOR_LIKE"):
        reasons.append("twin match passed")
        if oracle_verdict == "UNBARED":
            reasons.append("oracle leg UNBARED (no D41 gate file): metrics recorded, no numeric acceptance")
        if oracle_verdict == "UNMEASURED":
            reasons.append("oracle leg UNMEASURED (no comparable dump/positions)")
        if oracle_verdict == "NOT_LIKE_FOR_LIKE":
            reasons.append("oracle leg NOT_LIKE_FOR_LIKE: dump cannot prove its prefix - not acceptance evidence")
            return {"verdict": "FAIL", "reasons": reasons}
        return {"verdict": "PASS", "reasons": reasons}
    if twin_verdict == "UNMEASURED":
        return {"verdict": "UNMEASURED", "reasons": ["a device token exists but no twin comparison was measured", f"oracle leg: {oracle_verdict}"]}
    return {"verdict": "UNMEASURED", "reasons": [f"twin {twin_verdict}, oracle {oracle_verdict}"]}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="D40/D41 device-token acceptance: separate shape / pre-token / twin-match / oracle / step-time verdicts",
    )
    parser.add_argument("--device", action="append", default=[], help="device evidence JSON file or directory (repeatable)")
    parser.add_argument("--twin", action="append", default=[], help="host-twin evidence JSON file or directory (repeatable)")
    parser.add_argument("--device-dump", help="the device arm's own logits dump JSONL (explicit only)")
    parser.add_argument("--device-bin", help="device dump sidecar (default: --device-dump + .bin)")
    parser.add_argument("--dump-source", default="device-arm", help="provenance label for the device dump (recorded)")
    parser.add_argument("--oracle", help="frozen sparkflash-native-oracle/1 oracle JSON")
    parser.add_argument("--oracle-prompt", help="prompt id inside the oracle file")
    parser.add_argument("--oracle-bars", help="D41 gate bars file (sparkflash-d41-bars/1); without it the leg is UNBARED")
    parser.add_argument("--provisional-d40", action="store_true", help="print the published D40 provisional values labelled NOT-A-GATE")
    parser.add_argument("--positions", help="comma-separated positions to compare (dump/oracle leg and pinned-position check)")
    parser.add_argument("--expect-vocab", type=int, help="expected vocabulary (released checkpoint: 129280)")
    parser.add_argument("--expect-root", default=compare.DEFAULT_EXPECT_ROOT, help="oracle served root (D38)")
    parser.add_argument("--twin-positions-min", type=int, default=1, help="minimum compared positions for a twin PASS")
    parser.add_argument("--expect-attention-bound", type=int, help="expected attention tensors_bound (window: 208)")
    parser.add_argument("--expect-moe-bounds", help="expected MoE class bound counts, e.g. routed_experts_gate_up=61440,...")
    parser.add_argument("--expect-manifest-lines", type=int, help="expected tp4-bind-names-rR.txt logical line count (m1-bind window: 185481; wc -l prints 185480 because of no trailing newline)")
    parser.add_argument("--device-log", action="append", default=[], help="rank run.log to scan (default: run.log beside the evidence)")
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="require sparkflash-tp4-token/1 token evidence (a missing token file is then a FAIL, not NOT_ATTEMPTED)",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_md = args.out_md or (os.path.splitext(args.out_json)[0] + ".md")
    try:
        if not args.device:
            raise AcceptError("--device is required: pass the device evidence file(s) or directory")
        expected_positions = sorted(compare.parse_positions(args.positions)) if args.positions else None
        ignored: list = []
        seen: list = []
        device_files = _collect_paths(args.device, seen, ignored)
        twin_files = _collect_paths(args.twin, seen, ignored) if args.twin else []
        device_evidence = extract_evidence(device_files, expect_vocab=args.expect_vocab, side="device")
        twin_evidence = extract_evidence(twin_files, expect_vocab=args.expect_vocab, side="twin") if twin_files else []
        if not args.device_dump:
            # The next-window forward run names its own dump in the token envelope; rank 0 is the
            # primary dump owner. An explicit --device-dump always wins.
            candidates = [
                entry
                for entry in device_evidence
                if entry["kind"] == "tp4-token"
                and isinstance(entry["logits_dump"], dict)
                and isinstance(entry["logits_dump"].get("path"), str)
            ]
            if candidates:
                candidates.sort(key=lambda entry: (entry["rank"] is None, entry["rank"] if entry["rank"] is not None else 99))
                chosen = candidates[0]
                reference = chosen["logits_dump"]
                base = os.path.dirname(chosen["path"])
                dump_path = reference["path"] if os.path.isabs(reference["path"]) else os.path.join(base, reference["path"])
                bin_ref = reference.get("bin")
                bin_path = None
                if isinstance(bin_ref, str):
                    bin_path = bin_ref if os.path.isabs(bin_ref) else os.path.join(base, bin_ref)
                args.device_dump = dump_path
                args.device_bin = bin_path or (dump_path + ".bin")
                args.dump_source = f"tp4-token envelope rank {chosen['rank']}"
        dump = dump_report(args.device_dump, args.device_bin, args.expect_vocab, expected_positions)
        device_stream = token_stream(device_evidence)
        twin_stream = token_stream(twin_evidence)
        if args.expect_moe_bounds:
            args.expect_moe_bounds = parse_expected_moe_bounds(args.expect_moe_bounds)
        shape = build_shape(device_evidence, twin_evidence, dump, expected_positions, args)
        shape["verdict"] = "PASS" if shape["ok"] else "FAIL"
        pre_token = pre_token_checks(device_evidence)
        twin_match = compare_streams(device_stream, twin_stream, max(1, args.twin_positions_min))
        if args.oracle:
            if dump["present"]:
                oracle_comparison = oracle_distributional(args, dump)
            else:
                oracle_comparison = oracle_token_only(args, device_stream)
        else:
            oracle_comparison = {
                "verdict": "UNMEASURED",
                "label": "NO-ORACLE",
                "reasons": ["no --oracle supplied; nothing is claimed against the released-snapshot oracle"],
            }
        step_time = step_time_report(device_evidence)
        result = {
            "schema": OUTPUT_SCHEMA,
            "generated_utc": utc_now(),
            "inputs": {
                "device_evidence": [{"path": entry["path"], "sha256": entry["sha256"], "schema": entry["schema"], "layout": entry["kind"], "rank": entry["rank"]} for entry in device_evidence],
                "twin_evidence": [{"path": entry["path"], "sha256": entry["sha256"], "schema": entry["schema"], "layout": entry["kind"], "rank": entry["rank"]} for entry in twin_evidence],
                "device_dump": dump.get("path"),
                "device_dump_sha256": dump.get("sha256"),
                "dump_source": args.dump_source if dump.get("present") else None,
                "oracle": os.path.abspath(args.oracle) if args.oracle else None,
                "oracle_bars": os.path.abspath(args.oracle_bars) if args.oracle_bars else None,
                "provisional_d40": bool(args.provisional_d40),
                "positions": expected_positions,
                "expect_vocab": args.expect_vocab,
                "expect_root": args.expect_root,
                "ignored_files": ignored,
            },
            "shape": shape,
            "pre_token": pre_token,
            "device_stream": {"tokens": device_stream["tokens"][:64], "positions": device_stream["positions"][:64] if device_stream["positions"] else None, "ranks": device_stream["ranks"], "consistent": device_stream["consistent"], "positions_run": len(device_stream["tokens"]), "partial": len(device_stream["tokens"]) < D40_POSITIONS_MIN},
            "twin_stream": {"tokens": twin_stream["tokens"][:64], "positions": twin_stream["positions"][:64] if twin_stream["positions"] else None, "ranks": twin_stream["ranks"], "consistent": twin_stream["consistent"], "positions_run": len(twin_stream["tokens"])},
            "twin_match": twin_match,
            "oracle_comparison": oracle_comparison,
            "step_time": step_time,
        }
        result["overall"] = overall_verdict(result)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
        with open(out_md, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(result))
        print(render_markdown(result))
        print(f"wrote {args.out_json} and {out_md}")
        print(f"VERDICTS: shape={result['shape']['verdict']} pre_token={result['pre_token']['verdict']} twin={result['twin_match']['verdict']} oracle={result['oracle_comparison']['verdict']} step_time={result['step_time']['verdict']} overall={result['overall']['verdict']}")
        if result["overall"]["verdict"] == "PASS":
            return 0
        if result["overall"]["verdict"] == "FAIL":
            return 1
        return 3
    except (AcceptError, compare.CompareError) as error:
        print(f"accept_device_token: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
