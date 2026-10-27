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

"""Compare one engine f32 fixture position against one recorded oracle position.

The oracle harness (scripts/oracle_native.py) records, per prompt, the served
model's prompt_logprobs for every prompt token.  The engine fixture generator
(cargo run -p sparkflash-engine --example sequence_fixture) records, per run,
the f32 reference's logits for the last prefill position and any decode steps.
This script puts the two in the same coordinate system and reports the D39
metrics for one position.

Position convention
-------------------
prompt_logprobs[i] is the distribution for prompt token i given tokens
0..i-1.  Entry 0 is null, so oracle row "position = i - 1" is the distribution
the engine produces at prefill position i - 1, which predicts prompt token i.
A fixture whose prompt is the first N tokens of the oracle prompt therefore
compares at position N - 1, the fixture's last prefill position.

Why this exists
---------------
It makes the task-32 controls offline-reproducible without a server: the f32
fixture and the oracle JSON are both files, and this script is the only code
between them.  It refuses to compare when the fixture prompt is not a prefix of
the oracle prompt, so a mis-paired control fails loudly instead of scoring.

Usage
-----
    python3 scripts/compare_fixture_vs_oracle.py \
        --fixture f32-prompt8.json \
        --oracle native-oracle-suite.json \
        --prompt-id b69-fixture-130 \
        --out task32-control-p7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

TOP_K = (5, 20, 100, 1000)


def log_softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    normaliser = peak + math.log(sum(math.exp(value - peak) for value in logits))
    return [value - normaliser for value in logits]


def top_indices(logprobs: list[float], count: int) -> list[int]:
    return sorted(range(len(logprobs)), key=lambda token: (-logprobs[token], token))[:count]


def position_metrics(row: dict, logits: list[float], request: dict) -> dict:
    """The D39 comparison metrics for one position: two distributions, f32 vs served.

    row is the engine row (unused beyond the vocabulary check); logits are the
    engine's f32 logits; request is the oracle's recorded prompt-logprob entry.
    The oracle's distribution is its top-1000 map, so mean and max deltas are
    measured over the tokens the oracle returned, which the overlap and KL
    numbers also make explicit.
    """
    if row["vocab"] != len(logits):
        raise SystemExit(
            "vocabulary mismatch: fixture " + str(len(logits)) + " vs oracle row " + str(row["vocab"])
        )
    device = log_softmax(logits)
    oracle = {int(token): float(value) for token, value in request["logprobs"].items()}
    # A prompt-logprob entry carries NULL at entry 0 and the target's own value
    # can be present as the single extra entry; compare only finite values.
    oracle.pop(0, None)
    device_top = top_indices(device, max(TOP_K))
    oracle_order = [int(token) for token, _ in request["top"]]
    metrics = {
        "position": request["position"],
        "target_token": request["target_token"],
        "target_in_distribution": request["target_token"] in oracle,
        "device_argmax": device_top[0],
        "oracle_argmax": oracle_order[0],
        "device_top": [[token, device[token]] for token in device_top[:20]],
        "oracle_top": request["top"][:20],
        "top1_agreement": device_top[0] == oracle_order[0],
        "target_logprob_device": device[request["target_token"]],
        "target_logprob_oracle": oracle.get(request["target_token"]),
    }
    for k in TOP_K:
        metrics["top" + str(k) + "_overlap"] = len(set(oracle_order[:k]) & set(device_top[:k])) / k
    deltas = [abs(device[token] - value) for token, value in oracle.items() if token < len(device)]
    metrics["max_abs_delta"] = max(deltas) if deltas else None
    metrics["mean_abs_delta"] = sum(deltas) / len(deltas) if deltas else None
    metrics["compared_tokens"] = len(deltas)
    # KL(oracle || device) and Jensen-Shannon over the oracle's returned tokens.
    tokens = [token for token in oracle if token < len(device)]
    oracle_mass = sum(math.exp(oracle[token]) for token in tokens)
    device_mass = sum(math.exp(device[token]) for token in tokens)
    kl = 0.0
    js = 0.0
    for token in tokens:
        p = math.exp(oracle[token]) / oracle_mass
        q = math.exp(device[token]) / device_mass
        kl += p * math.log(p / q)
        m = 0.5 * (p + q)
        js += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)
    metrics["kl_oracle_to_device"] = kl
    metrics["js_divergence"] = js
    metrics["oracle_mass_on_returned"] = oracle_mass
    return metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fixture_position(fixture: dict, requested: int | None) -> tuple[int, dict]:
    steps = [step for step in fixture["steps"] if step.get("logits") is not None]
    if not steps:
        raise SystemExit("the fixture records no logits; regenerate with --logits prefill or all")
    if requested is None:
        return steps[0]["position"], steps[0]
    for step in steps:
        if step["position"] == requested:
            return requested, step
    raise SystemExit(
        "the fixture has no logits at position "
        + str(requested)
        + "; it records "
        + str([step["position"] for step in steps])
    )


def find_prompt(oracle: dict, prompt_id: str | None) -> dict:
    prompts = oracle["prompts"]
    if prompt_id is None:
        if len(prompts) != 1:
            raise SystemExit("--prompt-id is required when the oracle records more than one prompt")
        return prompts[0]
    matches = [prompt for prompt in prompts if prompt["id"] == prompt_id]
    if not matches:
        raise SystemExit(
            "oracle prompt "
            + repr(prompt_id)
            + " not found; it records "
            + str([prompt["id"] for prompt in prompts])
        )
    return matches[0]


def find_row(prompt: dict, position: int) -> dict:
    for row in prompt["positions"]:
        if row["position"] == position:
            return row
    raise SystemExit(
        "oracle prompt "
        + prompt["id"]
        + " has no position "
        + str(position)
        + " (it records "
        + str(len(prompt["positions"]))
        + " positions)"
    )


def check_prefix(
    prompt: dict,
    fixture: dict,
    oracle_ids: list[int],
    suite_sha256: str,
    complete: bool,
) -> dict:
    """Prove the fixture prompt is the oracle prompt's prefix; refuse otherwise.

    oracle_ids are the served prompt's ids in order: the selected suite prompt's
    tokens followed by its continuation tokens.  The fixture must agree token
    for token, and its last prefill position must predict exactly the next
    oracle prompt token.  Nothing is assumed from the filenames.
    """
    fixture_ids = [int(token) for token in fixture["prompt_tokens"]]
    if len(oracle_ids) < len(fixture_ids):
        raise SystemExit("the fixture prompt is longer than the oracle prompt")
    if oracle_ids[: len(fixture_ids)] != fixture_ids:
        first = next(
            i for i, (a, b) in enumerate(zip(oracle_ids, fixture_ids)) if a != b
        )
        raise SystemExit(
            "the fixture prompt is not a prefix of the oracle prompt for "
            + prompt["id"]
            + ": first difference at index "
            + str(first)
        )
    record = {
        "fixture_prompt_len": len(fixture_ids),
        "oracle_prompt_len": len(oracle_ids),
        "prefix_verified": True,
        "continuation_verified": complete,
        "suite_sha256": suite_sha256,
    }
    if complete:
        if len(oracle_ids) != prompt["prompt_len"]:
            raise SystemExit(
                "the prompt menu and the oracle disagree on "
                + prompt["id"]
                + " prompt length: menu "
                + str(len(oracle_ids))
                + " vs oracle "
                + str(prompt["prompt_len"])
            )
        if len(fixture_ids) >= len(oracle_ids):
            raise SystemExit("the fixture prompt does not leave an oracle target token")
        record["next_prompt_token"] = oracle_ids[len(fixture_ids)]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="sequence_fixture JSON output")
    parser.add_argument("--oracle", required=True, help="oracle JSON (sparkflash-native-oracle/1)")
    parser.add_argument("--prompt-id", help="oracle prompt id when the oracle has several")
    parser.add_argument(
        "--suite",
        required=True,
        help="oracle_prompt_suite.json; binds the oracle to the prompt ids it served",
    )
    parser.add_argument(
        "--continuation",
        help=(
            "JSON array of the tokens the oracle harness appended after the suite prompt "
            "(include the sentinel); omit to verify only the suite-prompt prefix"
        ),
    )
    parser.add_argument("--position", type=int, help="engine prefill position (default: last)")
    parser.add_argument("--out", help="write the comparison record here")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    oracle_path = Path(args.oracle)
    fixture = load_json(fixture_path)
    oracle = load_json(oracle_path)
    position, step = fixture_position(fixture, args.position)
    prompt = find_prompt(oracle, args.prompt_id)
    row = find_row(prompt, position)
    suite_path = Path(args.suite)
    suite_sha256 = sha256(suite_path)
    if suite_sha256 != oracle.get("suite", {}).get("sha256"):
        raise SystemExit(
            "the prompt menu "
            + str(suite_path)
            + " is not the one the oracle was run with (sha256 "
            + suite_sha256
            + " vs recorded "
            + str(oracle.get("suite", {}).get("sha256"))
            + ")"
        )
    suite = load_json(suite_path)
    menu = [entry for entry in suite["prompts"] if entry["id"] == prompt["id"]]
    if not menu:
        raise SystemExit("the prompt menu does not contain " + prompt["id"])
    continuation = json.loads(args.continuation) if args.continuation else None
    oracle_ids = [int(token) for token in menu[0]["tokens"]] + [
        int(token) for token in (continuation or [])
    ]
    prefix = check_prefix(prompt, fixture, oracle_ids, suite_sha256, continuation is not None)
    if continuation is not None and prefix["next_prompt_token"] != row["target_token"]:
        raise SystemExit(
            "oracle row "
            + str(position)
            + " predicts token "
            + str(row["target_token"])
            + ", but the fixture prompt's next token is "
            + str(prefix["next_prompt_token"])
            + "; the row and the prefix disagree"
        )

    request = {
        "position": row["position"],
        "target_token": row["target_token"],
        "logprob_count": row["logprob_count"],
        "logprobs": row["logprobs"],
        "top": row["top"],
    }
    device_row = {"position": position, "vocab": len(step["logits"]), "reference_id": None}
    metrics = position_metrics(device_row, [float(value) for value in step["logits"]], request)

    record = {
        "schema": "sparkflash-control-fixture-vs-oracle/1",
        "control": "short-context" if position < 8 else "prefill",
        "prompt_id": prompt["id"],
        "position": position,
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256(fixture_path),
            "generator": fixture.get("generator"),
            "config_label": fixture.get("config_label"),
            "command": fixture.get("provenance", {}).get("command"),
            "binary_sha256": fixture.get("provenance", {}).get("binary_sha256"),
        },
        "oracle": {
            "path": str(oracle_path),
            "sha256": sha256(oracle_path),
            "served_root": oracle.get("served", {}).get("root"),
            "generated_at": oracle.get("generated_at"),
            "root_verified": oracle.get("root_verified"),
        },
        "prefix": prefix,
        "engine_argmax": metrics["device_argmax"],
        "oracle_argmax": metrics["oracle_argmax"],
        "metrics": metrics,
    }

    print(
        "position "
        + str(position)
        + " (prompt "
        + prompt["id"]
        + ", "
        + fixture.get("config_label", "?")
        + " f32 vs served oracle)"
    )
    print(
        "  top1: f32 "
        + str(metrics["device_argmax"])
        + " / oracle "
        + str(metrics["oracle_argmax"])
        + ("  AGREE" if metrics["top1_agreement"] else "  DIFFER")
    )
    for key in ("top5_overlap", "top20_overlap", "top100_overlap", "top1000_overlap"):
        value = metrics.get(key)
        print("  " + key.ljust(16) + ("n/a" if value is None else format(value, ".4f")))
    for key in ("max_abs_delta", "mean_abs_delta", "kl_oracle_to_device", "js_divergence"):
        value = metrics.get(key)
        print("  " + key.ljust(20) + ("n/a" if value is None else format(value, ".6g")))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print("wrote " + str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
