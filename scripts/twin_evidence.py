#!/usr/bin/env python3
"""Build canonical --twin evidence from the pinned host-reference fixture sequence.

The D40/D41 M1 token gate compares the device's predicted token ids at the pinned
positions against a host-side token sequence (twf "twin").  The pinned sequence fixture
(sequence-130-released.json) records, for every compared position, the host reference
engine's next token in steps[].next_token.  This tool turns that host sequence into the
canonical token-evidence document the acceptance runner consumes under --twin, without
touching device evidence or inventing values.

Provenance label: host-reference-fixture.  It IS a host-side sequence (produced by the
reference engine from the released checkpoint), but it is NOT a second independent
four-rank run.  The independent option is the tp4-host twin run on the nodes; this
tool exists for the dev-host-only path and is labelled honestly in the twin row.

The output is one canonical file:

    {"schema": "sparkflash-device-token-evidence/1", "kind": "host-twin",
     "rank": 0, "world_size": 4,
     "prompt_tokens": [<130 ids>], "token_ids": [<9 ids>],
     "positions": [129, ..., 137], "start_position": 129,
     "token_emitted": true, ...}

Usage:
    python3 scripts/twin_evidence.py \
        --fixture /home/user/phase2-evidence/sequence-130-released.json \
        --out-dir /home/user/ev-session-<date>/qa/window/<run>-twin

Exit codes: 0 written, 2 bad input / refusing to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

CANONICAL_SCHEMA = "sparkflash-device-token-evidence/1"
DEFAULT_LABEL = "host-reference-fixture"


class TwinError(Exception):
    pass


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as error:
        raise TwinError(f"{path}: cannot read: {error}") from error
    except json.JSONDecodeError as error:
        raise TwinError(f"{path}: not JSON: {error}") from error
    if not isinstance(document, dict):
        raise TwinError(f"{path}: top level is {type(document).__name__}, not an object")
    return document


def _int_list(value, what: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TwinError(f"{what}: expected a non-empty list of token ids")
    tokens: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise TwinError(f"{what}: {item!r} is not a non-negative token id")
        tokens.append(item)
    return tokens


def plan_from_fixture(fixture: dict, prompt_id: str) -> dict:
    """Return the host reference token plan: positions + next tokens + prompt ids."""
    prompt = fixture.get("prompt_tokens")
    if prompt is None:
        prompt = fixture.get("tokens")
    prompt_tokens = _int_list(prompt, f"{prompt_id}: prompt_tokens")

    steps = fixture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TwinError(f"{prompt_id}: fixture has no steps[]")

    positions: list[int] = []
    tokens: list[int] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TwinError(f"{prompt_id}: steps[{index}] is not an object")
        position = step.get("position")
        token = step.get("next_token")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise TwinError(f"{prompt_id}: steps[{index}].position is not a non-negative integer")
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise TwinError(f"{prompt_id}: steps[{index}].next_token is not a non-negative token id")
        positions.append(position)
        tokens.append(token)

    if positions != sorted(positions):
        raise TwinError(f"{prompt_id}: steps[].position is not ascending: {positions}")
    if len(set(positions)) != len(positions):
        raise TwinError(f"{prompt_id}: duplicate step positions: {positions}")
    return {
        "prompt_id": prompt_id,
        "prompt_tokens": prompt_tokens,
        "positions": positions,
        "token_ids": tokens,
    }


def build_document(plan: dict, fixture_path: str, fixture_sha: str, label: str, world: int) -> dict:
    return {
        "schema": CANONICAL_SCHEMA,
        "kind": "host-twin",
        "rank": 0,
        "world_size": int(world),
        "prompt_tokens": list(plan["prompt_tokens"]),
        "token_ids": list(plan["token_ids"]),
        "positions": list(plan["positions"]),
        "start_position": int(plan["positions"][0]),
        "token_emitted": True,
        "first_token_ms": None,
        "step_ms": [],
        "logits_sha256": None,
        "execution": label,
        "notes": [
            "host reference-engine token sequence from " + fixture_path
            + " steps[].next_token (released checkpoint); positional join with the device rows",
            "this is the host-reference fixture sequence, NOT a second independent four-rank run;"
            " the independent twin is the tp4-host run on the nodes",
        ],
        "provenance": {
            "kind": label,
            "fixture": os.path.abspath(fixture_path),
            "fixture_sha256": fixture_sha,
            "source": "sequence fixture steps[].next_token (host reference engine, released checkpoint)",
            "prompt_id": plan["prompt_id"],
            "prompt_len": len(plan["prompt_tokens"]),
            "compared_positions": len(plan["positions"]),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build canonical --twin evidence from the pinned host-reference fixture sequence"
    )
    parser.add_argument("--fixture", required=True, help="sequence fixture JSON with prompt_tokens and steps[].next_token")
    parser.add_argument("--out-dir", required=True, help="fresh directory for twin-r0.json (never the evidence tree)")
    parser.add_argument("--prompt-id", default=None, help="oracle/device prompt id for provenance (default: fixture file stem)")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="execution label written into the twin document")
    parser.add_argument("--world", type=int, default=4, help="world size recorded in the twin document (informational)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing twin-r0.json")
    args = parser.parse_args(argv)

    try:
        fixture_sha = sha256_file(args.fixture)
        fixture = load_json(args.fixture)
        prompt_id = args.prompt_id or os.path.splitext(os.path.basename(args.fixture))[0]
        plan = plan_from_fixture(fixture, prompt_id)
        document = build_document(plan, args.fixture, fixture_sha, args.label, args.world)
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "twin-r0.json")
        if os.path.exists(out_path) and not args.force:
            raise TwinError(f"{out_path}: exists (pass --force to overwrite)")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write(chr(10))
    except TwinError as error:
        print(f"twin_evidence: {error}", file=sys.stderr)
        return 2

    print(
        "twin_evidence: wrote " + out_path
        + " kind=host-twin positions=" + str(plan["positions"][0]) + ".." + str(plan["positions"][-1])
        + " tokens=" + str(len(plan["token_ids"]))
        + " prompt_len=" + str(len(plan["prompt_tokens"]))
        + " fixture_sha256=" + fixture_sha[:16]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
